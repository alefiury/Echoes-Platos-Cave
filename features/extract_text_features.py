from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


MODEL_ALIASES: dict[str, str] = {
    "bert-base-uncased": "google-bert/bert-base-uncased",
    "bert-large-uncased": "google-bert/bert-large-uncased",
    "bert-base-cased": "google-bert/bert-base-cased",
    "bert-large-cased": "google-bert/bert-large-cased",
    "bert-base-multilingual-cased": "google-bert/bert-base-multilingual-cased",
    "e5-small-v2": "intfloat/e5-small-v2",
    "e5-base-v2": "intfloat/e5-base-v2",
    "e5-large-v2": "intfloat/e5-large-v2",
    "gte-base-en-v1.5": "Alibaba-NLP/gte-base-en-v1.5",
    "gte-large-en-v1.5": "Alibaba-NLP/gte-large-en-v1.5",
    "gte-multilingual-base": "Alibaba-NLP/gte-multilingual-base",
    "qwen3-embedding-0.6b": "Qwen/Qwen3-Embedding-0.6B",
    "qwen3-embedding-4b": "Qwen/Qwen3-Embedding-4B",
    "qwen3-embedding-8b": "Qwen/Qwen3-Embedding-8B",
}

SUPPORTED_FAMILIES = {"bert", "e5", "gte", "qwen3_embedding"}
DEFAULT_EXTENSIONS = (".txt",)

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

DEFAULT_MAX_LENGTH: dict[str, int] = {
    "bert": 512,
    "e5": 512,
    "gte": 8192,
    "qwen3_embedding": 8192,
}


@dataclass
class LoadedEncoder:
    checkpoint: str
    family: str
    model_type: str
    model: torch.nn.Module
    tokenizer: Any
    max_length: int
    compute_dtype: torch.dtype
    input_device: torch.device
    trust_remote_code: bool


@dataclass
class TextItem:
    source: str
    source_id: str
    row_index: int | None
    raw_text: str
    output_path: Path


@dataclass
class ExtractedText:
    hidden_states: torch.Tensor
    input_ids: torch.Tensor
    special_tokens_mask: torch.Tensor | None
    token_type_ids: torch.Tensor | None
    valid_tokens: int
    num_layers: int
    hidden_size: int
    qwen_eos_present: bool | None


def sanitize_name(value: str, max_length: int = 120) -> str:
    value = value.strip().replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._") or "item"
    return value[:max_length]


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def tensor_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def resolve_inside_base(base_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base_dir / path


def resolve_compute_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        if name in {"float16", "bfloat16"}:
            print(
                f"Warning: --compute-dtype={name} requested without CUDA; "
                "using float32.",
                file=sys.stderr,
            )
        return torch.float32

    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    return DTYPE_MAP[name]


def infer_family(checkpoint: str, model_type: str, requested: str) -> str:
    if requested != "auto":
        return requested

    checkpoint_lower = checkpoint.lower()
    model_type_lower = model_type.lower()

    if "qwen3-embedding" in checkpoint_lower:
        return "qwen3_embedding"
    if re.search(r"(^|[/_-])e5([/_-]|$)", checkpoint_lower):
        return "e5"
    if "gte" in checkpoint_lower:
        return "gte"
    if model_type_lower in {"bert", "roberta", "xlm-roberta"}:
        return "bert"

    raise ValueError(
        f"Could not infer the embedding family for checkpoint={checkpoint!r}, "
        f"model_type={model_type!r}. Pass --model-family explicitly."
    )


def default_trust_remote_code(checkpoint: str, family_hint: str) -> bool:
    checkpoint_lower = checkpoint.lower()
    return family_hint == "gte" or checkpoint_lower.startswith("alibaba-nlp/gte")


def find_input_device(model: torch.nn.Module, fallback: torch.device) -> torch.device:
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is not None:
            device = embeddings.weight.device
            if device.type != "meta":
                return device
    except Exception:
        pass

    try:
        device = next(model.parameters()).device
        if device.type != "meta":
            return device
    except StopIteration:
        pass

    return fallback


def model_max_length(tokenizer: Any, config: Any) -> int | None:
    candidates: list[int] = []

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 10**9:
        candidates.append(tokenizer_limit)

    for name in (
        "max_position_embeddings",
        "max_sequence_length",
        "seq_length",
        "n_positions",
    ):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            candidates.append(value)

    return min(candidates) if candidates else None


def load_encoder(
    model_name: str,
    model_family: str,
    max_length: int | None,
    device: torch.device,
    device_map: str | None,
    compute_dtype_name: str,
    attn_implementation: str | None,
    cache_dir: str | None,
    trust_remote_code_arg: bool | None,
    revision: str | None,
) -> LoadedEncoder:
    checkpoint = MODEL_ALIASES.get(model_name.lower(), model_name)

    checkpoint_lower = checkpoint.lower()
    family_hint = model_family
    if family_hint == "auto":
        if "qwen3-embedding" in checkpoint_lower:
            family_hint = "qwen3_embedding"
        elif re.search(r"(^|[/_-])e5([/_-]|$)", checkpoint_lower):
            family_hint = "e5"
        elif "gte" in checkpoint_lower:
            family_hint = "gte"
        else:
            family_hint = "bert"

    trust_remote_code = (
        default_trust_remote_code(checkpoint, family_hint)
        if trust_remote_code_arg is None
        else trust_remote_code_arg
    )

    common_kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
    }
    if revision is not None:
        common_kwargs["revision"] = revision

    config = AutoConfig.from_pretrained(checkpoint, **common_kwargs)
    detected_model_type = str(getattr(config, "model_type", "unknown")).lower()
    family = infer_family(checkpoint, detected_model_type, model_family)

    compute_dtype = resolve_compute_dtype(compute_dtype_name, device)

    tokenizer_kwargs = dict(common_kwargs)
    tokenizer_kwargs["use_fast"] = True
    tokenizer_kwargs["padding_side"] = "left" if family == "qwen3_embedding" else "right"

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, **tokenizer_kwargs)
    except Exception:
        tokenizer_kwargs["use_fast"] = False
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, **tokenizer_kwargs)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token_id is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            raise ValueError(
                f"Tokenizer for {checkpoint!r} has no pad, EOS, or SEP token."
            )

    load_kwargs: dict[str, Any] = {
        **common_kwargs,
        "torch_dtype": compute_dtype,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    model = AutoModel.from_pretrained(checkpoint, **load_kwargs)

    if device_map is None:
        model.to(device)

    model.eval()
    model.requires_grad_(False)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    input_device = find_input_device(model, fallback=device)

    resolved_max_length = max_length or DEFAULT_MAX_LENGTH[family]

    hard_limit = model_max_length(tokenizer, config)
    if hard_limit is not None and resolved_max_length > hard_limit:
        print(
            f"Warning: requested max length {resolved_max_length:,} exceeds the "
            f"detected model/tokenizer limit {hard_limit:,}; using {hard_limit:,}.",
            file=sys.stderr,
        )
        resolved_max_length = hard_limit

    return LoadedEncoder(
        checkpoint=checkpoint,
        family=family,
        model_type=detected_model_type,
        model=model,
        tokenizer=tokenizer,
        max_length=resolved_max_length,
        compute_dtype=compute_dtype,
        input_device=input_device,
        trust_remote_code=trust_remote_code,
    )


def output_path_for_text_file(
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


def output_path_for_metadata_row(
    output_dir: Path,
    source_id: str,
) -> Path:
    safe_id = sanitize_name(source_id, max_length=220)
    return output_dir / f"{safe_id}.safetensors"


def read_metadata(path, text_column, id_column):
    columns = [text_column]
    if id_column is not None and id_column != text_column:
        columns.append(id_column)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path, columns=columns)
    elif suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(
            path,
            sep="\t" if suffix == ".tsv" else ",",
            usecols=columns,
        )
    else:
        raise ValueError("Metadata must be .csv, .tsv, or .parquet")
    return frame[columns]


def load_text_items(
    input_dir: Path,
    output_dir: Path,
    input_metadata: Path | None,
    text_column: str,
    id_column: str | None,
    extensions: Sequence[str],
    encoding: str,
) -> list[TextItem]:
    if input_metadata is None:
        normalized_exts = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
        }
        files = sorted(
            (
                path
                for path in input_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in normalized_exts
            ),
            key=lambda value: str(value),
        )

        items: list[TextItem] = []
        for filepath in files:
            text = filepath.read_text(encoding=encoding)
            items.append(
                TextItem(
                    source=str(filepath),
                    source_id=str(filepath),
                    row_index=None,
                    raw_text=text,
                    output_path=output_path_for_text_file(
                        filepath=filepath,
                        input_dir=input_dir,
                        output_dir=output_dir,
                    ),
                )
            )
        return items

    frame = read_metadata(
        path=input_metadata,
        text_column=text_column,
        id_column=id_column,
    )

    items = []
    for row_index, row in enumerate(frame.itertuples(index=False, name=None)):
        text_value = row[0]
        raw_text = "" if pd.isna(text_value) else str(text_value)

        if id_column is None:
            source_id = f"row_{row_index:012d}"
        else:
            id_position = 0 if id_column == text_column else 1
            id_value = row[id_position]
            if pd.isna(id_value) or not str(id_value).strip():
                raise ValueError(
                    f"Missing or empty value in --id-column={id_column!r} "
                    f"at metadata row {row_index}."
                )
            source_id = str(id_value).strip()

        items.append(
            TextItem(
                source=f"{input_metadata}#row={row_index}",
                source_id=source_id,
                row_index=row_index,
                raw_text=raw_text,
                output_path=output_path_for_metadata_row(
                    output_dir=output_dir,
                    source_id=source_id,
                ),
            )
        )

    seen_outputs: dict[Path, TextItem] = {}
    collisions: list[tuple[TextItem, TextItem]] = []
    for item in items:
        previous = seen_outputs.get(item.output_path)
        if previous is None:
            seen_outputs[item.output_path] = item
        else:
            collisions.append((previous, item))

    if collisions:
        examples = "\n".join(
            f"  {first.source_id!r} at row {first.row_index} and "
            f"{second.source_id!r} at row {second.row_index} -> "
            f"{first.output_path.name}"
            for first, second in collisions[:10]
        )
        extra = "" if len(collisions) <= 10 else f"\n  ... and {len(collisions) - 10} more"
        raise ValueError(
            "Duplicate --id-column values or filename collisions were found. "
            "Every metadata row must produce a unique output filename:\n"
            f"{examples}{extra}"
        )

    return items


def prepare_text(
    raw_text: str,
    family: str,
    text_role: str,
    instruction: str | None,
    text_prefix: str,
) -> str:
    text = f"{text_prefix}{raw_text}" if text_prefix else raw_text

    if family == "e5":
        stripped = text.lstrip()
        already_prefixed = stripped.startswith("query: ") or stripped.startswith("passage: ")
        if not already_prefixed:
            if text_role == "query":
                text = f"query: {text}"
            elif text_role == "document":
                text = f"passage: {text}"

    if family == "qwen3_embedding" and text_role == "query" and instruction:
        text = f"Instruct: {instruction}\nQuery:{text}"

    return text


def move_model_inputs(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    moved: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            continue
        moved[key] = value.to(device=device, non_blocking=True)
    return moved


def last_valid_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(
        attention_mask.shape[1],
        device=attention_mask.device,
    ).unsqueeze(0)
    return (positions * attention_mask.long()).max(dim=1).values


def tokenize_batch(
    texts: Sequence[str],
    loaded: LoadedEncoder,
    pad_to_multiple_of: int | None,
) -> dict[str, torch.Tensor]:
    encoded = loaded.tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=True,
        truncation=True,
        max_length=loaded.max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )

    return {key: value for key, value in dict(encoded).items() if torch.is_tensor(value)}


def extract_batch(
    texts: Sequence[str],
    loaded: LoadedEncoder,
    pad_to_multiple_of: int | None,
) -> list[ExtractedText]:
    encoded_cpu = tokenize_batch(
        texts=texts,
        loaded=loaded,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    if "attention_mask" not in encoded_cpu or "input_ids" not in encoded_cpu:
        raise RuntimeError("Tokenizer did not return input_ids and attention_mask")

    attention_mask_cpu = encoded_cpu["attention_mask"]
    input_ids_cpu = encoded_cpu["input_ids"]
    special_tokens_mask_cpu = encoded_cpu.get("special_tokens_mask")
    token_type_ids_cpu = encoded_cpu.get("token_type_ids")

    qwen_eos_flags: list[bool | None] = [None] * len(texts)
    if loaded.family == "qwen3_embedding" and loaded.tokenizer.eos_token_id is not None:
        eos_token_id = int(loaded.tokenizer.eos_token_id)
        last_positions = last_valid_positions(attention_mask_cpu)
        qwen_eos_flags = [
            bool(input_ids_cpu[index, int(position.item())].item() == eos_token_id)
            for index, position in enumerate(last_positions)
        ]
        if not all(qwen_eos_flags):
            print(
                "Warning: at least one Qwen3 input does not end in EOS. "
                "Check the tokenizer revision if this warning persists.",
                file=sys.stderr,
            )

    model_input_keys = {
        key: value
        for key, value in encoded_cpu.items()
        if key not in {"special_tokens_mask", "overflow_to_sample_mapping"}
    }
    model_inputs = move_model_inputs(model_input_keys, loaded.input_device)

    forward_kwargs: dict[str, Any] = {
        **model_inputs,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if loaded.family == "qwen3_embedding":
        forward_kwargs["use_cache"] = False

    with torch.inference_mode():
        outputs = loaded.model(**forward_kwargs)

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden states")

    results: list[ExtractedText] = []
    for sample_index in range(len(texts)):
        valid_mask_cpu = attention_mask_cpu[sample_index].bool()
        valid_tokens = int(valid_mask_cpu.sum().item())
        if valid_tokens < 1:
            raise RuntimeError("Tokenizer produced an input with zero valid tokens")

        layer_segments: list[torch.Tensor] = []
        for layer in hidden_states:
            mask_on_layer = valid_mask_cpu.to(layer.device)
            layer_segments.append(
                layer[sample_index, mask_on_layer, :].detach().to("cpu")
            )

        sample_hidden = torch.stack(layer_segments, dim=0).contiguous()

        results.append(
            ExtractedText(
                hidden_states=sample_hidden,
                input_ids=input_ids_cpu[sample_index, valid_mask_cpu].contiguous(),
                special_tokens_mask=(
                    special_tokens_mask_cpu[sample_index, valid_mask_cpu].contiguous()
                    if special_tokens_mask_cpu is not None
                    else None
                ),
                token_type_ids=(
                    token_type_ids_cpu[sample_index, valid_mask_cpu].contiguous()
                    if token_type_ids_cpu is not None
                    else None
                ),
                valid_tokens=valid_tokens,
                num_layers=len(hidden_states),
                hidden_size=int(hidden_states[-1].shape[-1]),
                qwen_eos_present=qwen_eos_flags[sample_index],
            )
        )

    del outputs, hidden_states, model_inputs
    return results


def is_valid_safetensors(path: Path) -> bool:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            return "hidden_states" in keys or "layer_000" in keys
    except Exception:
        return False


def save_extracted_text(
    extracted: ExtractedText,
    output_path: Path,
    metadata: dict[str, Any],
    save_dtype: torch.dtype,
    storage_layout: str,
    save_token_ids: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    states = extracted.hidden_states.to(dtype=save_dtype).contiguous()
    if storage_layout == "stacked":
        tensors: dict[str, torch.Tensor] = {"hidden_states": states}
    elif storage_layout == "separate":
        tensors = {
            f"layer_{layer_index:03d}": states[layer_index].contiguous()
            for layer_index in range(states.shape[0])
        }
    else:
        raise ValueError(f"Unknown storage layout: {storage_layout}")

    if save_token_ids:
        tensors["input_ids"] = extracted.input_ids.to(dtype=torch.int64).contiguous()
        if extracted.special_tokens_mask is not None:
            tensors["special_tokens_mask"] = extracted.special_tokens_mask.to(
                dtype=torch.int64
            ).contiguous()
        if extracted.token_type_ids is not None:
            tensors["token_type_ids"] = extracted.token_type_ids.to(
                dtype=torch.int64
            ).contiguous()

    string_metadata = {key: str(value) for key, value in metadata.items()}

    temporary_path = output_path.with_name(output_path.name + ".tmp.safetensors")
    try:
        save_file(tensors, str(temporary_path), metadata=string_metadata)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def process_batch(
    items: Sequence[TextItem],
    loaded: LoadedEncoder,
    text_role: str,
    instruction: str | None,
    text_prefix: str,
    pad_to_multiple_of: int | None,
    save_dtype: torch.dtype,
    storage_layout: str,
    save_token_ids: bool,
    text_preview_chars: int,
    manifest_path: Path,
) -> None:
    prepared_texts = [
        prepare_text(
            raw_text=item.raw_text,
            family=loaded.family,
            text_role=text_role,
            instruction=instruction,
            text_prefix=text_prefix,
        )
        for item in items
    ]

    extracted_items = extract_batch(
        texts=prepared_texts,
        loaded=loaded,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    for item, prepared_text, extracted in zip(items, prepared_texts, extracted_items):
        raw_hash = sha1_text(item.raw_text)
        prepared_hash = sha1_text(prepared_text)

        metadata = {
            "source": item.source,
            "source_id": item.source_id,
            "row_index": "" if item.row_index is None else item.row_index,
            "checkpoint": loaded.checkpoint,
            "model_family": loaded.family,
            "model_type": loaded.model_type,
            "text_role": text_role,
            "instruction": instruction or "",
            "text_prefix": text_prefix,
            "raw_text_sha1": raw_hash,
            "prepared_text_sha1": prepared_hash,
            "raw_text_characters": len(item.raw_text),
            "prepared_text_characters": len(prepared_text),
            "valid_tokens": extracted.valid_tokens,
            "num_hidden_state_tensors": extracted.num_layers,
            "hidden_size": extracted.hidden_size,
            "max_length": loaded.max_length,
            "saved_dtype": tensor_dtype_name(save_dtype),
            "storage_layout": storage_layout,
            "qwen_eos_present": extracted.qwen_eos_present,
            "layer_0_description": "token embedding output before transformer block 0",
        }
        if text_preview_chars > 0:
            metadata["raw_text_preview"] = item.raw_text[:text_preview_chars]

        save_extracted_text(
            extracted=extracted,
            output_path=item.output_path,
            metadata=metadata,
            save_dtype=save_dtype,
            storage_layout=storage_layout,
            save_token_ids=save_token_ids,
        )

        shape = list(extracted.hidden_states.shape)

        append_jsonl(
            manifest_path,
            {
                "status": "saved",
                "source": item.source,
                "source_id": item.source_id,
                "row_index": item.row_index,
                "output": str(item.output_path),
                "hidden_states_shape": shape,
                "valid_tokens": extracted.valid_tokens,
                "dtype": tensor_dtype_name(save_dtype),
                "model_family": loaded.family,
                "checkpoint": loaded.checkpoint,
                "raw_text_sha1": raw_hash,
                "prepared_text_sha1": prepared_hash,
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract padding-free token-level hidden states from every layer "
            "of BERT, E5-v2, GTE, or Qwen3-Embedding models "
            "into one .safetensors file per input text."
        )
    )

    parser.add_argument("-b", "--base-dir", required=True, type=Path)
    parser.add_argument(
        "-i",
        "--input-dir-name",
        default=".",
        help=(
            "Directory containing .txt files inside --base-dir, or an absolute "
            "directory. Used as the relative root when metadata is not supplied."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir-name",
        default="output_text_embeddings",
        help="Output directory inside --base-dir, or an absolute directory.",
    )
    parser.add_argument(
        "-m",
        "--model-name",
        default="e5-base-v2",
        help="Alias from MODEL_ALIASES or any Hugging Face checkpoint ID.",
    )
    parser.add_argument(
        "--model-family",
        choices=["auto", *sorted(SUPPORTED_FAMILIES)],
        default="auto",
        help="Override automatic model-family detection.",
    )
    parser.add_argument(
        "-c",
        "--input-metadata",
        type=Path,
        help="Optional .csv, .tsv, or .parquet containing one text per row.",
    )
    parser.add_argument(
        "-col",
        "--column-name",
        default="text",
        help="Text column in --input-metadata.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help=("Stable item-ID column. For metadata input, each output is named "
              "<id-column-value>.safetensors."),
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(DEFAULT_EXTENSIONS),
        help="Extensions used during recursive text-file discovery.",
    )
    parser.add_argument("--encoding", default="utf-8")
    parser.add_argument(
        "--text-role",
        choices=["query", "document", "raw"],
        default="query",
        help=(
            "E5: query adds 'query: ', document adds 'passage: '. "
            "Qwen3: query uses --instruction when supplied; document/raw do not."
        ),
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help=(
            "Optional Qwen3 query instruction. It is formatted as "
            "'Instruct: ...\\nQuery:<text>'."
        ),
    )
    parser.add_argument(
        "--text-prefix",
        default="",
        help="Optional literal prefix applied before model-specific formatting.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum tokenized length. Family-specific default when omitted.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of texts per inference batch. Default 1 for all-layer extraction.",
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=None,
        help="Optionally pad batch length to a multiple such as 8.",
    )
    parser.add_argument(
        "--save-token-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save valid input_ids and available token masks.",
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
        help="On-disk floating-point dtype.",
    )
    parser.add_argument(
        "--storage-layout",
        choices=["stacked", "separate"],
        default="stacked",
        help=(
            "stacked: hidden_states=[layers,tokens,dim]; separate: one tensor "
            "key per layer for lazy single-layer loading."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help=(
            "Optional Transformers/Accelerate device map, for example 'auto'. "
            "When set, --device is used only as a dtype/fallback hint."
        ),
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
        help="Optional Transformers attention backend.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Auto-enabled for Alibaba GTE checkpoints unless explicitly disabled.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
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
        help="Split the deterministic item list across independent jobs.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index used with --num-shards.",
    )
    parser.add_argument(
        "--text-preview-chars",
        type=int,
        default=0,
        help="Store at most this many raw text characters in safetensors metadata.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_length is not None and args.max_length < 1:
        raise ValueError("--max-length must be >= 1")
    if args.pad_to_multiple_of is not None and args.pad_to_multiple_of < 1:
        raise ValueError("--pad-to-multiple-of must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.text_preview_chars < 0:
        raise ValueError("--text-preview-chars must be >= 0")

    base_dir = args.base_dir.expanduser()
    input_dir = resolve_inside_base(base_dir, args.input_dir_name)
    output_dir = resolve_inside_base(base_dir, args.output_dir_name)

    input_metadata = args.input_metadata
    if input_metadata is not None:
        input_metadata = input_metadata.expanduser()
        if not input_metadata.is_absolute():
            input_metadata = base_dir / input_metadata

    device = torch.device(args.device)
    if args.device_map is None and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    if device.type == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    loaded = load_encoder(
        model_name=args.model_name,
        model_family=args.model_family,
        max_length=args.max_length,
        device=device,
        device_map=args.device_map,
        compute_dtype_name=args.compute_dtype,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
        trust_remote_code_arg=args.trust_remote_code,
        revision=args.revision,
    )

    if not args.no_model_subdir:
        output_dir = output_dir / sanitize_name(loaded.checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_text_items(
        input_dir=input_dir,
        output_dir=output_dir,
        input_metadata=input_metadata,
        text_column=args.column_name,
        id_column=args.id_column,
        extensions=args.extensions,
        encoding=args.encoding,
    )
    items = items[args.shard_index :: args.num_shards]

    shard_tag = f"shard_{args.shard_index:05d}_of_{args.num_shards:05d}"
    manifest_path = output_dir / f"manifest_{shard_tag}.jsonl"
    error_path = output_dir / f"errors_{shard_tag}.jsonl"

    save_dtype = DTYPE_MAP[args.save_dtype]

    print(f"Checkpoint:          {loaded.checkpoint}")
    print(f"Family:              {loaded.family}")
    print(f"Model type:          {loaded.model_type}")
    print(f"Input device:        {loaded.input_device}")
    print(f"Device map:          {args.device_map}")
    print(f"Compute dtype:       {tensor_dtype_name(loaded.compute_dtype)}")
    print(f"Save dtype:          {args.save_dtype}")
    print(f"Maximum length:      {loaded.max_length:,}")
    print(f"Storage layout:      {args.storage_layout}")
    print(f"Text role:           {args.text_role}")
    print(f"Input directory:     {input_dir}")
    print(f"Input metadata:      {input_metadata}")
    print(f"Output directory:    {output_dir}")
    print(f"Items in shard:      {len(items):,}")

    pending_items: list[TextItem] = []
    progress = tqdm(items, desc="Extracting token hidden states", unit="text")

    def log_item_error(item: TextItem, error: Exception) -> None:
        append_jsonl(
            error_path,
            {
                "source": item.source,
                "source_id": item.source_id,
                "row_index": item.row_index,
                "output": str(item.output_path),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )

    def flush_pending() -> None:
        nonlocal pending_items
        if not pending_items:
            return

        current_items = pending_items
        pending_items = []

        try:
            process_batch(
                items=current_items,
                loaded=loaded,
                text_role=args.text_role,
                instruction=args.instruction,
                text_prefix=args.text_prefix,
                pad_to_multiple_of=args.pad_to_multiple_of,
                save_dtype=save_dtype,
                storage_layout=args.storage_layout,
                save_token_ids=args.save_token_ids,
                text_preview_chars=args.text_preview_chars,
                manifest_path=manifest_path,
            )
            return
        except Exception as batch_error:
            if isinstance(batch_error, torch.cuda.OutOfMemoryError):
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if len(current_items) == 1:
                log_item_error(current_items[0], batch_error)
                return

        for item in current_items:
            try:
                process_batch(
                    items=[item],
                    loaded=loaded,
                    text_role=args.text_role,
                    instruction=args.instruction,
                    text_prefix=args.text_prefix,
                    pad_to_multiple_of=args.pad_to_multiple_of,
                    save_dtype=save_dtype,
                    storage_layout=args.storage_layout,
                    save_token_ids=args.save_token_ids,
                    text_preview_chars=args.text_preview_chars,
                    manifest_path=manifest_path,
                )
            except Exception as single_error:
                if isinstance(single_error, torch.cuda.OutOfMemoryError):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                log_item_error(item, single_error)

    for item in progress:
        if item.output_path.exists() and not args.overwrite:
            should_skip = (
                is_valid_safetensors(item.output_path)
                if args.validate_existing
                else True
            )
            if should_skip:
                continue

        if not item.raw_text.strip():
            log_item_error(item, ValueError("Text is empty or whitespace-only"))
            continue

        pending_items.append(item)
        if len(pending_items) >= args.batch_size:
            flush_pending()

    flush_pending()

    print(f"Done. Manifest: {manifest_path}")
    if error_path.exists():
        print(f"Errors:         {error_path}")


if __name__ == "__main__":
    main()
