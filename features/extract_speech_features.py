from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch
import torchaudio
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel


MODEL_ALIASES: dict[str, str] = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "wav2vec2-large": "facebook/wav2vec2-large",
    "wav2vec2-base-960h": "facebook/wav2vec2-base-960h",
    "wav2vec2-large-960h": "facebook/wav2vec2-large-960h",
    "wav2vec2-large-960h-lv60-self": "facebook/wav2vec2-large-960h-lv60-self",
    "wav2vec2-large-xlsr-53": "facebook/wav2vec2-large-xlsr-53",
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "wavlm-large": "microsoft/wavlm-large",
    "hubert-base-ls960": "facebook/hubert-base-ls960",
    "hubert-large-ll60k": "facebook/hubert-large-ll60k",
    "hubert-xlarge-ll60k": "facebook/hubert-xlarge-ll60k",
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large-v2": "openai/whisper-large-v2",
    "whisper-large-v3": "openai/whisper-large-v3",
}

SUPPORTED_MODEL_TYPES = {"wav2vec2", "wavlm", "hubert", "whisper"}
DEFAULT_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class LoadedEncoder:
    checkpoint: str
    model_type: str
    encoder: torch.nn.Module
    feature_extractor: Any
    sampling_rate: int
    compute_dtype: torch.dtype
    device: torch.device


@dataclass
class AudioItem:
    filepath: Path
    output_path: Path
    waveform: torch.Tensor
    sampling_rate: int


def sanitize_name(value: str) -> str:
    value = value.strip().replace("/", "__")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_compute_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        if name in {"float16", "bfloat16"}:
            print(
                f"Warning: --compute-dtype={name} requested on CPU; using float32.",
                file=sys.stderr,
            )
        return torch.float32

    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    return DTYPE_MAP[name]


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
    floating_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            continue
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=floating_dtype, non_blocking=True)
        else:
            moved[key] = value.to(device=device, non_blocking=True)
    return moved


def tensor_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def load_encoder(
    model_name: str,
    device: torch.device,
    compute_dtype_name: str,
    attn_implementation: str | None,
    cache_dir: str | None,
) -> LoadedEncoder:
    checkpoint = MODEL_ALIASES.get(model_name, model_name)
    config = AutoConfig.from_pretrained(checkpoint, cache_dir=cache_dir)
    model_type = str(config.model_type).lower()

    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(
            f"Unsupported model_type={model_type!r} for {checkpoint!r}. "
            f"Supported types: {sorted(SUPPORTED_MODEL_TYPES)}"
        )

    compute_dtype = resolve_compute_dtype(compute_dtype_name, device)

    load_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation

    full_model = AutoModel.from_pretrained(checkpoint, **load_kwargs)

    if model_type == "whisper":
        encoder = full_model.get_encoder()
        del full_model
    else:
        encoder = full_model

    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        checkpoint,
        cache_dir=cache_dir,
    )
    sampling_rate = int(feature_extractor.sampling_rate)

    return LoadedEncoder(
        checkpoint=checkpoint,
        model_type=model_type,
        encoder=encoder,
        feature_extractor=feature_extractor,
        sampling_rate=sampling_rate,
        compute_dtype=compute_dtype,
        device=device,
    )


def read_audio(filepath: Path, target_sr: int) -> torch.Tensor:
    waveform, source_sr = torchaudio.load(str(filepath))

    if waveform.ndim != 2:
        raise ValueError(
            f"Expected torchaudio.load() to return [channels, samples], got {waveform.shape}"
        )

    waveform = waveform.mean(dim=0)

    if int(source_sr) != target_sr:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=int(source_sr),
            new_freq=target_sr,
        )

    waveform = waveform.to(dtype=torch.float32).contiguous()

    if waveform.numel() == 0:
        raise ValueError("Audio is empty after loading/resampling")
    if not torch.isfinite(waveform).all():
        raise ValueError("Audio contains NaN or Inf values")

    return waveform


def load_filelist(
    input_dir: Path,
    input_metadata: Path | None,
    column_name: str,
    extensions: Sequence[str],
) -> list[Path]:
    if input_metadata is None:
        normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
        files = [
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in normalized_exts
        ]
    else:
        suffix = input_metadata.suffix.lower()
        if suffix == ".parquet":
            frame = pd.read_parquet(input_metadata, columns=[column_name])
        elif suffix in {".csv", ".tsv"}:
            frame = pd.read_csv(
                input_metadata,
                sep="\t" if suffix == ".tsv" else ",",
                usecols=[column_name],
            )
        else:
            raise ValueError("Metadata must be .csv, .tsv, or .parquet")

        files = []
        for raw_value in frame[column_name].dropna().astype(str):
            path = Path(raw_value).expanduser()
            if not path.is_absolute():
                path = input_dir / path
            files.append(path)

    unique: dict[str, Path] = {}
    for path in files:
        absolute = Path(os.path.abspath(os.path.expanduser(str(path))))
        unique.setdefault(str(absolute), absolute)

    return sorted(unique.values(), key=lambda value: str(value))


def output_path_for(
    filepath: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    absolute_file = Path(os.path.abspath(str(filepath)))
    absolute_root = Path(os.path.abspath(str(input_dir)))

    try:
        relative = absolute_file.relative_to(absolute_root)
        return (output_dir / relative).with_suffix(".safetensors")
    except ValueError:
        digest = hashlib.sha1(str(absolute_file).encode("utf-8")).hexdigest()[:12]
        return output_dir / "_external" / f"{absolute_file.stem}__{digest}.safetensors"


def is_valid_safetensors(path: Path) -> bool:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            return bool(keys & {"hidden_states", "layer_000"})
    except Exception:
        return False


def extract_ssl_batch(
    waveforms: Sequence[torch.Tensor],
    loaded: LoadedEncoder,
) -> list[torch.Tensor]:
    """Extract Wav2Vec2/WavLM/HuBERT states with padding removed."""

    arrays = [waveform.numpy() for waveform in waveforms]
    encoded = loaded.feature_extractor(
        arrays,
        sampling_rate=loaded.sampling_rate,
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    original_sample_lengths = torch.tensor(
        [waveform.numel() for waveform in waveforms],
        dtype=torch.long,
    )

    model_inputs = move_batch_to_device(
        dict(encoded),
        device=loaded.device,
        floating_dtype=loaded.compute_dtype,
    )

    with torch.inference_mode():
        outputs = loaded.encoder(
            **model_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    length_device = next(loaded.encoder.parameters()).device
    valid_frames = loaded.encoder._get_feat_extract_output_lengths(  # type: ignore[attr-defined]
        original_sample_lengths.to(length_device)
    ).to("cpu")

    results: list[torch.Tensor] = []
    for sample_index, frame_count_tensor in enumerate(valid_frames):
        frame_count = int(frame_count_tensor.item())
        layers = [
            layer[sample_index, :frame_count, :].detach().to("cpu")
            for layer in hidden_states
        ]
        results.append(torch.stack(layers, dim=0).contiguous())

    del outputs, hidden_states, model_inputs
    return results


def whisper_max_samples(loaded: LoadedEncoder) -> int:
    extractor = loaded.feature_extractor
    if hasattr(extractor, "n_samples"):
        return int(extractor.n_samples)
    chunk_length = int(getattr(extractor, "chunk_length", 30))
    return chunk_length * loaded.sampling_rate


def split_whisper_audio(
    waveform: torch.Tensor,
    max_samples: int,
    sampling_rate: int,
    mode: str,
) -> list[torch.Tensor]:
    if waveform.numel() <= max_samples:
        return [waveform]

    if mode == "error":
        duration = waveform.numel() / float(sampling_rate)
        raise ValueError(
            f"Whisper input is {duration:.2f}s, exceeding its 30-second encoder window. "
            "Use --whisper-long-audio chunk or --whisper-long-audio truncate."
        )

    if mode == "truncate":
        return [waveform[:max_samples]]

    if mode == "chunk":
        return [
            waveform[start : start + max_samples]
            for start in range(0, waveform.numel(), max_samples)
        ]

    raise ValueError(f"Unknown Whisper long-audio mode: {mode}")


def extract_whisper_batch(
    waveforms: Sequence[torch.Tensor],
    loaded: LoadedEncoder,
    inference_batch_size: int,
    long_audio_mode: str,
    feature_device: str,
) -> tuple[list[torch.Tensor], list[int]]:

    max_samples = whisper_max_samples(loaded)

    expanded: list[tuple[int, torch.Tensor]] = []
    chunk_counts = [0 for _ in waveforms]
    for original_index, waveform in enumerate(waveforms):
        chunks = split_whisper_audio(
            waveform,
            max_samples=max_samples,
            sampling_rate=loaded.sampling_rate,
            mode=long_audio_mode,
        )
        chunk_counts[original_index] = len(chunks)
        expanded.extend((original_index, chunk) for chunk in chunks)

    accumulated: list[list[list[torch.Tensor]] | None] = [None for _ in waveforms]

    for chunk_batch in batched(expanded, inference_batch_size):
        owner_indices = [owner for owner, _ in chunk_batch]
        chunks = [chunk for _, chunk in chunk_batch]
        arrays = [chunk.numpy() for chunk in chunks]

        call_kwargs: dict[str, Any] = {
            "raw_speech": arrays,
            "sampling_rate": loaded.sampling_rate,
            "padding": "max_length",
            "max_length": max_samples,
            "truncation": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }
        if feature_device != "cpu":
            call_kwargs["device"] = str(loaded.device)

        encoded = loaded.feature_extractor(**call_kwargs)
        hop_length = int(getattr(loaded.feature_extractor, "hop_length", 160))
        valid_mel_frames = torch.tensor(
            [(chunk.numel() + hop_length - 1) // hop_length for chunk in chunks],
            dtype=torch.long,
        )

        input_features = encoded["input_features"].to(
            device=loaded.device,
            dtype=loaded.compute_dtype,
            non_blocking=True,
        )

        with torch.inference_mode():
            outputs = loaded.encoder(
                input_features=input_features,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Whisper encoder did not return hidden states")

        valid_encoder_frames = loaded.encoder._get_feat_extract_output_lengths(  # type: ignore[attr-defined]
            valid_mel_frames.to(loaded.device)
        ).to("cpu")

        for batch_index, owner_index in enumerate(owner_indices):
            frame_count = int(valid_encoder_frames[batch_index].item())

            if accumulated[owner_index] is None:
                accumulated[owner_index] = [[] for _ in hidden_states]

            owner_layers = accumulated[owner_index]
            assert owner_layers is not None

            for layer_index, layer in enumerate(hidden_states):
                owner_layers[layer_index].append(
                    layer[batch_index, :frame_count, :].detach().to("cpu")
                )

        del outputs, hidden_states, input_features, encoded

    results: list[torch.Tensor] = []
    for owner_layers in accumulated:
        if owner_layers is None:
            raise RuntimeError("Internal error: no Whisper chunks were extracted")
        concatenated_layers = [
            torch.cat(segments, dim=0) if len(segments) > 1 else segments[0]
            for segments in owner_layers
        ]
        results.append(torch.stack(concatenated_layers, dim=0).contiguous())

    return results, chunk_counts


def save_hidden_states(
    hidden_states: torch.Tensor,
    output_path: Path,
    metadata: dict[str, Any],
    save_dtype: torch.dtype,
    storage_layout: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    states = hidden_states.to(dtype=save_dtype).contiguous()

    if storage_layout == "stacked":
        tensors = {"hidden_states": states}
    elif storage_layout == "separate":
        tensors = {
            f"layer_{layer_index:03d}": states[layer_index].contiguous()
            for layer_index in range(states.shape[0])
        }
    else:
        raise ValueError(f"Unknown storage layout: {storage_layout}")

    string_metadata = {key: str(value) for key, value in metadata.items()}

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        save_file(tensors, str(temporary_path), metadata=string_metadata)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def process_batch(
    items: Sequence[AudioItem],
    loaded: LoadedEncoder,
    save_dtype: torch.dtype,
    storage_layout: str,
    whisper_long_audio: str,
    whisper_feature_device: str,
    inference_batch_size: int,
    manifest_path: Path,
) -> None:
    waveforms = [item.waveform for item in items]

    if loaded.model_type == "whisper":
        extracted, chunk_counts = extract_whisper_batch(
            waveforms=waveforms,
            loaded=loaded,
            inference_batch_size=inference_batch_size,
            long_audio_mode=whisper_long_audio,
            feature_device=whisper_feature_device,
        )
    else:
        extracted = extract_ssl_batch(waveforms, loaded)
        chunk_counts = [1] * len(items)

    for item, hidden_states, num_chunks in zip(items, extracted, chunk_counts):
        num_layers, valid_frames, hidden_size = hidden_states.shape
        metadata = {
            "source_path": str(item.filepath),
            "checkpoint": loaded.checkpoint,
            "model_type": loaded.model_type,
            "sampling_rate": item.sampling_rate,
            "num_audio_samples": item.waveform.numel(),
            "duration_seconds": f"{item.waveform.numel() / item.sampling_rate:.9f}",
            "num_hidden_state_tensors": num_layers,
            "valid_frames": valid_frames,
            "hidden_size": hidden_size,
            "saved_dtype": tensor_dtype_name(save_dtype),
            "storage_layout": storage_layout,
            "whisper_chunks": num_chunks,
            "layer_0_description": "encoder input/projection hidden state",
        }

        save_hidden_states(
            hidden_states=hidden_states,
            output_path=item.output_path,
            metadata=metadata,
            save_dtype=save_dtype,
            storage_layout=storage_layout,
        )

        append_jsonl(
            manifest_path,
            {
                "status": "saved",
                "input": str(item.filepath),
                "output": str(item.output_path),
                "shape": [num_layers, valid_frames, hidden_size],
                "dtype": tensor_dtype_name(save_dtype),
                "model_type": loaded.model_type,
                "checkpoint": loaded.checkpoint,
                "whisper_chunks": num_chunks,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract padding-free all-layer encoder features from Wav2Vec2, "
            "WavLM, HuBERT, or Whisper into .safetensors files."
        )
    )

    parser.add_argument("-b", "--base-dir", required=True, type=Path)
    parser.add_argument(
        "-i",
        "--input-dir-name",
        required=True,
        help="Input directory name inside --base-dir, or an absolute directory.",
    )
    parser.add_argument(
        "-o",
        "--output-dir-name",
        default="output_embeddings",
        help="Output directory name inside --base-dir, or an absolute directory.",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="hubert-large-ll60k",
        help="Alias from MODEL_ALIASES or any Hugging Face checkpoint ID.",
    )
    parser.add_argument(
        "-c",
        "--input-metadata",
        type=Path,
        help="Optional .csv, .tsv, or .parquet containing audio paths.",
    )
    parser.add_argument(
        "-col",
        "--column-name",
        default="filename",
        help="Audio-path column in --input-metadata.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Extensions used during recursive directory discovery.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of utterances/chunks per inference batch. Start with 1 for large models/all layers.",
    )
    parser.add_argument(
        "--compute-dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--save-dtype",
        choices=list(DTYPE_MAP),
        default="float16",
        help="On-disk feature dtype. float16 halves storage relative to float32.",
    )
    parser.add_argument(
        "--storage-layout",
        choices=["stacked", "separate"],
        default="stacked",
        help=(
            "stacked: one [layers, frames, dim] tensor; separate: one key per layer "
            "for convenient lazy layer loading."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
        help="Optional Transformers attention backend; support depends on the model/environment.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--whisper-long-audio",
        choices=["error", "truncate", "chunk"],
        default="error",
        help="How to handle Whisper inputs longer than its 30-second encoder window.",
    )
    parser.add_argument(
        "--whisper-feature-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for Whisper log-Mel/STFT preprocessing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs. Otherwise existing valid files are skipped.",
    )
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="Open existing .safetensors before deciding to skip it.",
    )
    parser.add_argument(
        "--no-model-subdir",
        action="store_true",
        help="Write directly into output-dir instead of output-dir/<checkpoint>/.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the deterministic file list across multiple independent jobs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index used with --num-shards.",
    )

    return parser.parse_args()


def resolve_inside_base(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    base_dir = args.base_dir.expanduser()
    input_dir = resolve_inside_base(base_dir, args.input_dir_name)
    output_dir = resolve_inside_base(base_dir, args.output_dir_name)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    if args.whisper_feature_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--whisper-feature-device cuda requires CUDA")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    loaded = load_encoder(
        model_name=args.model_name,
        device=device,
        compute_dtype_name=args.compute_dtype,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
    )

    if not args.no_model_subdir:
        output_dir = output_dir / sanitize_name(loaded.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_metadata = args.input_metadata
    if input_metadata is not None:
        input_metadata = input_metadata.expanduser()
        if not input_metadata.is_absolute():
            input_metadata = base_dir / input_metadata

    filelist = load_filelist(
        input_dir=input_dir,
        input_metadata=input_metadata,
        column_name=args.column_name,
        extensions=args.extensions,
    )
    filelist = filelist[args.shard_index :: args.num_shards]

    shard_tag = f"shard_{args.shard_index:05d}_of_{args.num_shards:05d}"
    manifest_path = output_dir / f"manifest_{shard_tag}.jsonl"
    error_path = output_dir / f"errors_{shard_tag}.jsonl"

    save_dtype = DTYPE_MAP[args.save_dtype]

    print(f"Checkpoint:        {loaded.checkpoint}")
    print(f"Model type:        {loaded.model_type}")
    print(f"Device:            {loaded.device}")
    print(f"Compute dtype:     {tensor_dtype_name(loaded.compute_dtype)}")
    print(f"Save dtype:        {args.save_dtype}")
    print(f"Storage layout:    {args.storage_layout}")
    print(f"Input directory:   {input_dir}")
    print(f"Output directory:  {output_dir}")
    print(f"Files in shard:    {len(filelist):,}")

    pending_items: list[AudioItem] = []
    progress = tqdm(filelist, desc="Extracting all-layer features", unit="file")

    def flush_pending() -> None:
        nonlocal pending_items
        if not pending_items:
            return

        current_items = pending_items
        pending_items = []

        def log_item_error(item: AudioItem, error: Exception) -> None:
            append_jsonl(
                error_path,
                {
                    "input": str(item.filepath),
                    "output": str(item.output_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

        try:
            process_batch(
                items=current_items,
                loaded=loaded,
                save_dtype=save_dtype,
                storage_layout=args.storage_layout,
                whisper_long_audio=args.whisper_long_audio,
                whisper_feature_device=args.whisper_feature_device,
                inference_batch_size=args.batch_size,
                manifest_path=manifest_path,
            )
            return
        except Exception as batch_error:
            if isinstance(batch_error, torch.cuda.OutOfMemoryError) and device.type == "cuda":
                torch.cuda.empty_cache()

            if len(current_items) == 1:
                log_item_error(current_items[0], batch_error)
                return

        for item in current_items:
            try:
                process_batch(
                    items=[item],
                    loaded=loaded,
                    save_dtype=save_dtype,
                    storage_layout=args.storage_layout,
                    whisper_long_audio=args.whisper_long_audio,
                    whisper_feature_device=args.whisper_feature_device,
                    inference_batch_size=1,
                    manifest_path=manifest_path,
                )
            except Exception as single_error:
                if isinstance(single_error, torch.cuda.OutOfMemoryError) and device.type == "cuda":
                    torch.cuda.empty_cache()
                log_item_error(item, single_error)

    for filepath in progress:
        output_path = output_path_for(filepath, input_dir, output_dir)

        if output_path.exists() and not args.overwrite:
            should_skip = (
                is_valid_safetensors(output_path)
                if args.validate_existing
                else True
            )
            if should_skip:
                continue

        try:
            if not filepath.is_file():
                raise FileNotFoundError(f"Audio file does not exist: {filepath}")

            waveform = read_audio(filepath, loaded.sampling_rate)
            pending_items.append(
                AudioItem(
                    filepath=filepath,
                    output_path=output_path,
                    waveform=waveform,
                    sampling_rate=loaded.sampling_rate,
                )
            )

            if len(pending_items) >= args.batch_size:
                flush_pending()

        except Exception as error:
            append_jsonl(
                error_path,
                {
                    "input": str(filepath),
                    "output": str(output_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )

    flush_pending()

    print(f"Done. Manifest: {manifest_path}")
    if error_path.exists():
        print(f"Errors:         {error_path}")


if __name__ == "__main__":
    main()
