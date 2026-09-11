#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Layer-wise speech-text alignment analysis with aggregation-aware permutation calibration."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import sys
import traceback
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from scipy.stats import kruskal, rankdata, spearmanr
except ImportError as error:
    raise ImportError(
        "scipy is required. Install dependencies with:\n"
        "pip install -U numpy pandas scipy matplotlib tqdm safetensors torch pyarrow"
    ) from error


PROJECT_ROOT = Path("data")
DEFAULT_TEXT_ROOT = PROJECT_ROOT / "Text-Features"
DEFAULT_SPEECH_ROOT = PROJECT_ROOT / "TTS-Speech-Features"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Alignment-Analysis"
DEFAULT_BEST_SAMPLES_CSV = PROJECT_ROOT / "best_available_emotion_samples.csv"

SUPPORTED_EXTENSIONS = {
    ".safetensors",
    ".pt",
    ".pth",
    ".npy",
    ".npz",
}

DEFAULT_EXCLUDE_KEY_REGEX = (
    r"(^|[._/\-])(?:"
    r"attention_mask|"
    r"special_tokens_mask|"
    r"input_ids|"
    r"input_values|"
    r"lengths?|"
    r"padding_mask|"
    r"token_type_ids|"
    r"position_ids|"
    r"overflow_to_sample_mapping|"
    r"valid_tokens|"
    r"num_layers|"
    r"hidden_size|"
    r"qwen_eos_present|"
    r"logits?|"
    r"loss|"
    r"labels?"
    r")($|[._/\-])"
)

DEFAULT_SENTENCE_ID_REGEX = r"(sick_sentence_\d+)"


def parse_csv_values(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None

    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def parse_int_csv(value: str) -> List[int]:
    values = []

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))

    if not values:
        raise ValueError("At least one integer value is required.")

    return values


def parse_float_csv(value: str) -> List[float]:
    values: List[float] = []

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))

    if not values:
        raise ValueError("At least one floating-point value is required.")

    return values


def safe_name(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "__", text)
    text = text.strip("._-")
    return text or "unnamed"


def stable_seed(*values: Any, base_seed: int = 42) -> int:
    text = "||".join(str(value) for value in values)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:8], "little") + base_seed) % (2**32 - 1)


def atomic_write_csv(dataframe: pd.DataFrame, output_path: Path, **kwargs: Any) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    dataframe.to_csv(temporary_path, index=False, **kwargs)
    temporary_path.replace(output_path)


def save_table(dataframe: pd.DataFrame, output_stem: Path) -> Path:
    """Save Parquet when available; otherwise save compressed CSV."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = output_stem.with_suffix(".parquet")

    try:
        dataframe.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception as error:
        warnings.warn(
            f"Could not write Parquet ({type(error).__name__}: {error}). "
            "Falling back to gzip-compressed CSV."
        )

    csv_path = output_stem.with_suffix(".csv.gz")
    dataframe.to_csv(csv_path, index=False, compression="gzip")
    return csv_path


def write_json(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    temporary_path.replace(output_path)


def layer_sort_key(layer_name: str) -> Tuple[int, int, str]:
    match = re.search(r"(?:^|_)layer_(\d+)(?:_|$)", layer_name)

    if match:
        return 0, int(match.group(1)), layer_name

    numbers = re.findall(r"\d+", layer_name)

    if numbers:
        return 1, int(numbers[-1]), layer_name

    return 2, 0, layer_name


def validate_directory(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist:\n{path}")

    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory:\n{path}")


def canonical_identifier(value: Any) -> str:
    """Normalize names for robust joins across CSV values and directories."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def normalize_speaker_value(value: Any) -> str:
    """Normalize actor/speaker identifiers such as 4, 04, and 4.0 to '4'."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def normalize_emotion_value(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def normalize_sentence_id_value(value: Any, sentence_id_regex: re.Pattern[str]) -> str:
    """Extract the canonical SICK sentence ID from a CSV value or path."""
    text = str(value).strip()
    match = sentence_id_regex.search(text)
    if match:
        return match.group(1) if match.groups() else match.group(0)
    return Path(text).stem


def first_existing_column(dataframe: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {canonical_identifier(column): column for column in dataframe.columns}
    for candidate in candidates:
        key = canonical_identifier(candidate)
        if key in normalized:
            return normalized[key]
    return None


def load_best_samples_index(
    path: Path,
    sentence_id_regex: re.Pattern[str],
    emotion_column: str,
    duplicate_policy: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "The curated sample CSV does not exist:\n"
            f"{path}\n\n"
            "Pass it with --best-samples-csv."
        )

    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"The curated sample CSV is empty: {path}")

    tts_column = first_existing_column(frame, ["tts_model", "model", "generator"])
    speaker_column = first_existing_column(frame, ["speaker_id", "speaker", "actor_id"])
    sentence_column = first_existing_column(
        frame,
        ["sample_id", "sentence_id", "filename", "filepath"],
    )
    selected_emotion_column = first_existing_column(
        frame,
        [emotion_column, "orig_emotion", "target_emotion", "emotion"],
    )

    missing = []
    if tts_column is None:
        missing.append("tts_model")
    if speaker_column is None:
        missing.append("speaker_id")
    if sentence_column is None:
        missing.append("sample_id/sentence_id/filename/filepath")
    if selected_emotion_column is None:
        missing.append(emotion_column)
    if missing:
        raise ValueError(
            "Could not identify required columns in best-samples CSV: "
            f"{missing}\nAvailable columns: {list(frame.columns)}"
        )

    curated = frame.copy()
    curated["selected_source_tts_model"] = curated[tts_column].astype(str).str.strip()
    curated["_tts_key"] = curated[tts_column].map(canonical_identifier)
    curated["speaker"] = curated[speaker_column].map(normalize_speaker_value)
    curated["emotion"] = curated[selected_emotion_column].map(normalize_emotion_value)
    curated["sentence_id"] = curated[sentence_column].map(
        lambda value: normalize_sentence_id_value(value, sentence_id_regex)
    )

    key_columns = ["speaker", "emotion", "sentence_id"]
    invalid = curated[
        curated["_tts_key"].eq("")
        | curated["speaker"].eq("")
        | curated["emotion"].eq("")
        | curated["sentence_id"].eq("")
    ]
    if not invalid.empty:
        raise ValueError(
            "Some curated rows have empty TTS, speaker, emotion, or sentence IDs.\n"
            + invalid.head(10).to_string(index=False)
        )

    duplicate_mask = curated.duplicated(key_columns, keep=False)
    if duplicate_mask.any():
        duplicates = curated.loc[duplicate_mask].copy()
        if duplicate_policy == "error":
            raise ValueError(
                "best_available_emotion_samples.csv contains multiple selected "
                "rows for the same sentence × speaker × emotion.\n"
                + duplicates[
                    key_columns + ["selected_source_tts_model"]
                ].head(30).to_string(index=False)
            )
        if duplicate_policy == "best_score":
            score_column = first_existing_column(
                curated,
                ["target_emotion_score", "selection_score", "emotion_score"],
            )
            if score_column is None:
                raise ValueError(
                    "--best-samples-duplicate-policy best_score requires a "
                    "target_emotion_score-like column."
                )
            curated[score_column] = pd.to_numeric(curated[score_column], errors="coerce")
            curated = curated.sort_values(score_column, ascending=False)
        curated = curated.drop_duplicates(key_columns, keep="first")

    protected = {
        "selected_source_tts_model",
        "_tts_key",
        "speaker",
        "emotion",
        "sentence_id",
    }
    rename_map = {
        column: f"selection_{column}"
        for column in curated.columns
        if column not in protected
    }
    curated = curated.rename(columns=rename_map)

    ordered = [
        "selected_source_tts_model",
        "_tts_key",
        "speaker",
        "emotion",
        "sentence_id",
    ]
    ordered += [column for column in curated.columns if column not in ordered]
    return curated[ordered].reset_index(drop=True)


def apply_curated_sample_selection(
    speech_manifest: pd.DataFrame,
    curated_index: pd.DataFrame,
    strict_coverage: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Join discovered feature files to the selected source TTS per condition."""
    manifest = speech_manifest.copy()
    manifest["_tts_key"] = manifest["source_tts_model"].map(canonical_identifier)
    manifest["speaker"] = manifest["speaker"].map(normalize_speaker_value)
    manifest["emotion"] = manifest["emotion"].map(normalize_emotion_value)

    join_columns = ["_tts_key", "speaker", "emotion", "sentence_id"]
    selected = manifest.merge(
        curated_index,
        on=join_columns,
        how="inner",
        validate="many_to_one",
    )

    if selected.empty:
        manifest_preview = manifest[
            ["source_tts_model", "speaker", "emotion", "sentence_id"]
        ].head(10)
        curated_preview = curated_index[
            ["selected_source_tts_model", "speaker", "emotion", "sentence_id"]
        ].head(10)
        raise RuntimeError(
            "No speech feature matched best_available_emotion_samples.csv.\n\n"
            "Discovered feature examples:\n"
            f"{manifest_preview.to_string(index=False)}\n\n"
            "Curated CSV examples:\n"
            f"{curated_preview.to_string(index=False)}"
        )

    selected["selection_source_matches_directory"] = (
        selected["_tts_key"]
        == selected["selected_source_tts_model"].map(canonical_identifier)
    )

    available_conditions = manifest[["speaker", "emotion"]].drop_duplicates()
    relevant_curated = curated_index.merge(
        available_conditions,
        on=["speaker", "emotion"],
        how="inner",
        validate="many_to_one",
    )
    curated_keys = relevant_curated[
        ["speaker", "emotion", "sentence_id"]
    ].drop_duplicates()
    coverage_rows = []
    for speech_encoder in sorted(manifest["speech_encoder"].astype(str).unique()):
        encoder_manifest = selected[
            selected["speech_encoder"].astype(str) == speech_encoder
        ]
        matched_keys = encoder_manifest[
            ["speaker", "emotion", "sentence_id"]
        ].drop_duplicates()
        coverage = curated_keys.merge(
            matched_keys.assign(matched=True),
            on=["speaker", "emotion", "sentence_id"],
            how="left",
        )
        coverage["matched"] = coverage["matched"].fillna(False).astype(bool)
        coverage["speech_encoder"] = speech_encoder
        coverage_rows.append(coverage)

    coverage_frame = pd.concat(coverage_rows, ignore_index=True)
    missing_count = int((~coverage_frame["matched"]).sum())
    if missing_count:
        message = (
            f"{missing_count:,} curated sentence conditions have no matching "
            "speech feature for at least one speech encoder."
        )
        if strict_coverage:
            missing_preview = coverage_frame[~coverage_frame["matched"]].head(30)
            raise RuntimeError(message + "\n" + missing_preview.to_string(index=False))
        warnings.warn(message)

    selected = selected.drop(columns=["_tts_key"])
    return selected.reset_index(drop=True), coverage_frame


def summarize_source_tts_provenance(manifest_group: pd.DataFrame) -> Dict[str, Any]:
    """Summarize the selected generator mix without making it an analysis factor."""
    per_sample = manifest_group[
        ["sentence_id", "source_tts_model"]
    ].drop_duplicates()
    counts = per_sample["source_tts_model"].value_counts()
    total = int(counts.sum())
    if total == 0:
        return {
            "source_tts_models_used": "",
            "number_of_source_tts_models": 0,
            "dominant_source_tts_model": "",
            "dominant_source_tts_fraction": np.nan,
            "source_tts_distribution": "{}",
        }
    probabilities = counts.to_numpy(dtype=np.float64) / total
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return {
        "source_tts_models_used": ";".join(sorted(counts.index.astype(str))),
        "number_of_source_tts_models": int(len(counts)),
        "dominant_source_tts_model": str(counts.index[0]),
        "dominant_source_tts_fraction": float(counts.iloc[0] / total),
        "source_tts_entropy_bits": entropy,
        "source_tts_distribution": json.dumps(
            {str(key): int(value) for key, value in counts.items()},
            sort_keys=True,
        ),
    }


def save_source_tts_audits(
    speech_manifest: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save provenance tables; these are audits, not convergence variables."""
    per_sample = speech_manifest[
        [
            "speech_encoder",
            "source_tts_model",
            "speaker",
            "emotion",
            "sentence_id",
        ]
    ].drop_duplicates()
    distribution = (
        per_sample.groupby(
            ["speech_encoder", "speaker", "emotion", "source_tts_model"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "number_of_selected_samples"})
    )
    total = distribution.groupby(
        ["speech_encoder", "speaker", "emotion"]
    )["number_of_selected_samples"].transform("sum")
    distribution["fraction_within_condition"] = (
        distribution["number_of_selected_samples"] / total
    )
    atomic_write_csv(
        distribution,
        output_dir / "manifests" / "source_tts_provenance_by_condition.csv",
    )

    by_emotion = (
        per_sample.groupby(
            ["speech_encoder", "emotion", "source_tts_model"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "number_of_selected_samples"})
    )
    atomic_write_csv(
        by_emotion,
        output_dir / "manifests" / "source_tts_provenance_by_emotion.csv",
    )


def flatten_feature_object(value: Any, prefix: str = "") -> Iterator[Tuple[str, Any]]:
    """Recursively flatten tensors/arrays in dictionaries and sequences."""
    try:
        import torch
        tensor_types = (torch.Tensor, np.ndarray)
    except ImportError:
        tensor_types = (np.ndarray,)

    if isinstance(value, tensor_types):
        yield prefix or "embedding", value
        return

    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten_feature_object(child, child_prefix)
        return

    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from flatten_feature_object(child, child_prefix)
        return


def tensor_to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().to(dtype=torch.float32, device="cpu").numpy()
    except ImportError:
        pass

    return np.asarray(value, dtype=np.float32)


def remove_leading_singleton_batch_dimensions(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)

    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]

    return array


def validate_pooled_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    """Return a clean one-dimensional float32 vector, or None if invalid."""
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)

    if vector.size < 2 or not np.isfinite(vector).all():
        return None

    return vector


def pool_sequence_matrix(
    matrix: np.ndarray,
    pooling: str,
    valid_position_mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(
            f"Expected a sequence matrix with shape [T, d], got {matrix.shape}."
        )

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return None

    if valid_position_mask is not None:
        valid_position_mask = np.asarray(valid_position_mask, dtype=bool).reshape(-1)

        if valid_position_mask.shape[0] != matrix.shape[0]:
            raise ValueError(
                "The valid-position mask length does not match the sequence "
                f"axis: {valid_position_mask.shape[0]} versus {matrix.shape[0]}."
            )

        if not valid_position_mask.any():
            raise ValueError("The pooling mask excludes every sequence position.")

    if pooling in {"first", "cls"}:
        vector = matrix[0]
    else:
        selected = (
            matrix[valid_position_mask]
            if valid_position_mask is not None
            else matrix
        )

        if pooling == "mean":
            vector = np.nanmean(selected, axis=0)
        elif pooling == "max":
            vector = np.nanmax(selected, axis=0)
        elif pooling == "flatten":
            vector = selected.reshape(-1)
        else:
            raise ValueError(f"Unsupported pooling mode: {pooling}")

    return validate_pooled_vector(vector)


def normalize_mask_array(value: Any) -> Optional[np.ndarray]:
    """Convert a possible mask tensor to a one-dimensional NumPy array."""
    try:
        array = tensor_to_numpy(value)
    except Exception:
        return None

    array = np.asarray(array)

    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != 1 or array.size == 0:
        return None

    return array.reshape(-1)


def find_companion_sequence_mask(
    raw_object: Any,
    sequence_length: int,
) -> Optional[np.ndarray]:
    if not isinstance(raw_object, Mapping):
        return None

    valid_mask = np.ones(sequence_length, dtype=bool)
    found_matching_mask = False

    for raw_key, value in flatten_feature_object(raw_object):
        normalized_key = re.sub(
            r"[^a-z0-9]+",
            "_",
            str(raw_key).strip().lower(),
        ).strip("_")

        mask = normalize_mask_array(value)

        if mask is None or mask.shape[0] != sequence_length:
            continue

        leaf_name = normalized_key.split("_")[-1] if normalized_key else ""
        del leaf_name

        if normalized_key.endswith("special_tokens_mask"):
            valid_mask &= mask == 0
            found_matching_mask = True
        elif normalized_key.endswith("padding_mask"):
            valid_mask &= mask == 0
            found_matching_mask = True
        elif normalized_key.endswith("attention_mask"):
            valid_mask &= mask.astype(bool)
            found_matching_mask = True
        elif normalized_key.endswith("valid_tokens"):
            valid_mask &= mask.astype(bool)
            found_matching_mask = True

    if not found_matching_mask:
        return None

    if not valid_mask.any():
        raise ValueError(
            "Companion masks exclude every position in a sequence of length "
            f"{sequence_length}."
        )

    return valid_mask


def layered_tensor_prefix(raw_key: str, source_path: Path) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", raw_key.lower()).strip("_")

    canonical_hidden_state_names = {
        "hidden_states",
        "hidden_state",
        "encoder_hidden_states",
        "encoder_hidden_state",
    }

    if normalized in canonical_hidden_state_names:
        return ""

    return canonical_layer_name(raw_key, source_path)


def pool_feature_tensor(
    array: np.ndarray,
    pooling: str,
    raw_key: str,
    source_path: Path,
    valid_position_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    array = remove_leading_singleton_batch_dimensions(array)

    if array.size == 0 or array.ndim == 0:
        return {}

    if array.ndim == 1:
        vector = validate_pooled_vector(array)
        if vector is None:
            return {}
        return {canonical_layer_name(raw_key, source_path): vector}

    if array.ndim == 2:
        vector = pool_sequence_matrix(
            array,
            pooling=pooling,
            valid_position_mask=valid_position_mask,
        )
        if vector is None:
            return {}
        return {canonical_layer_name(raw_key, source_path): vector}

    if array.ndim == 3:
        number_of_layers, sequence_length, _ = array.shape

        if (
            valid_position_mask is not None
            and valid_position_mask.shape[0] != sequence_length
        ):
            raise ValueError(
                "The companion mask length does not match the [L, T, d] "
                f"tensor: {valid_position_mask.shape[0]} versus T={sequence_length}."
            )

        prefix = layered_tensor_prefix(raw_key, source_path)
        vectors: Dict[str, np.ndarray] = {}

        for layer_index in range(number_of_layers):
            vector = pool_sequence_matrix(
                array[layer_index],
                pooling=pooling,
                valid_position_mask=valid_position_mask,
            )

            if vector is None:
                continue

            layer_token = f"layer_{layer_index:03d}"
            layer_name = (
                layer_token
                if not prefix
                else f"{prefix}__{layer_token}"
            )
            vectors[layer_name] = vector

        return vectors

    raise ValueError(
        "Unsupported representation shape. Expected [d], [T, d], or [L, T, d] "
        f"(optionally with leading singleton batch axes), got {array.shape} "
        f"for key '{raw_key}' in {source_path}."
    )


def canonical_layer_name(raw_key: str, source_path: Path) -> str:
    """Convert heterogeneous feature keys to sortable layer names."""
    combined = f"{raw_key} {source_path.stem}".lower()

    patterns = [
        r"(?:hidden_states?|hidden_state|encoder_layers?|encoder_layer|layers?|layer)[._/\-]?(\d+)",
        r"(?:block|transformer_block)[._/\-]?(\d+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, combined)
        if match:
            return f"layer_{int(match.group(1)):03d}"

    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", raw_key).strip("_").lower()

    if normalized in {"", "tensor", "array", "embedding", "features", "feature"}:
        source_normalized = re.sub(
            r"[^a-zA-Z0-9]+", "_", source_path.stem
        ).strip("_").lower()
        normalized = source_normalized or "embedding"

    return normalized


def load_feature_file(
    path: Path,
    pooling: str,
    include_key_regex: Optional[re.Pattern[str]],
    exclude_key_regex: Optional[re.Pattern[str]],
) -> Dict[str, np.ndarray]:
    """Load all usable layer vectors from one feature file."""
    suffix = path.suffix.lower()
    raw_object: Any

    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError(
                "safetensors is required to read .safetensors files. "
                "Install it with: pip install -U safetensors"
            ) from error

        raw_object = load_file(str(path), device="cpu")

    elif suffix in {".pt", ".pth"}:
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "torch is required to read .pt/.pth files. "
                "Install the PyTorch build suitable for your system."
            ) from error

        try:
            raw_object = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            raw_object = torch.load(path, map_location="cpu")

    elif suffix == ".npy":
        raw_object = np.load(path, allow_pickle=False)

    elif suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        raw_object = {key: archive[key] for key in archive.files}

    else:
        raise ValueError(f"Unsupported feature extension: {path}")

    vectors: Dict[str, np.ndarray] = {}

    for raw_key, tensor in flatten_feature_object(raw_object):
        if include_key_regex is not None and include_key_regex.search(raw_key) is None:
            continue

        if exclude_key_regex is not None and exclude_key_regex.search(raw_key) is not None:
            continue

        try:
            array = remove_leading_singleton_batch_dimensions(
                tensor_to_numpy(tensor)
            )

            valid_position_mask = None
            if array.ndim in {2, 3}:
                sequence_length = int(array.shape[-2])
                valid_position_mask = find_companion_sequence_mask(
                    raw_object,
                    sequence_length=sequence_length,
                )

            loaded_vectors = pool_feature_tensor(
                array=array,
                pooling=pooling,
                raw_key=raw_key,
                source_path=path,
                valid_position_mask=valid_position_mask,
            )
        except Exception as error:
            warnings.warn(
                f"Could not pool key '{raw_key}' in {path}: "
                f"{type(error).__name__}: {error}"
            )
            continue

        for layer_name, vector in loaded_vectors.items():
            final_layer_name = layer_name

            if final_layer_name in vectors:
                duplicate_index = 2
                candidate = f"{final_layer_name}__{duplicate_index}"

                while candidate in vectors:
                    duplicate_index += 1
                    candidate = f"{final_layer_name}__{duplicate_index}"

                final_layer_name = candidate

            vectors[final_layer_name] = vector

    if not vectors:
        raise ValueError(f"No usable feature tensors were found in {path}")

    return vectors


def extract_sentence_id(relative_path: Path, regex: re.Pattern[str]) -> Optional[str]:
    search_targets = [relative_path.stem, relative_path.as_posix()]

    for target in search_targets:
        match = regex.search(target)
        if match:
            return match.group(1) if match.groups() else match.group(0)

    return None


def discover_text_manifest(
    root: Path,
    sentence_id_regex: re.Pattern[str],
    extensions: Set[str],
) -> pd.DataFrame:
    records = []

    for path in tqdm(sorted(root.rglob("*")), desc="Discovering text features", unit="path"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        relative = path.relative_to(root)

        if len(relative.parts) < 2:
            warnings.warn(f"Skipping text feature outside a model directory: {path}")
            continue

        sentence_id = extract_sentence_id(relative, sentence_id_regex)

        if sentence_id is None:
            warnings.warn(f"Could not extract a sentence ID from text feature: {path}")
            continue

        records.append(
            {
                "modality": "text",
                "model": relative.parts[0],
                "sentence_id": sentence_id,
                "feature_path": str(path.resolve()),
                "relative_path": relative.as_posix(),
            }
        )

    manifest = pd.DataFrame(records)

    if manifest.empty:
        raise RuntimeError(
            "No text feature files were discovered. Check --text-root, "
            "--extensions, and --sentence-id-regex."
        )

    return manifest.sort_values(["model", "sentence_id", "feature_path"]).reset_index(drop=True)


def discover_speech_manifest(
    root: Path,
    sentence_id_regex: re.Pattern[str],
    extensions: Set[str],
) -> pd.DataFrame:
    records = []

    for path in tqdm(sorted(root.rglob("*")), desc="Discovering speech features", unit="path"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        relative = path.relative_to(root)

        if len(relative.parts) < 5:
            warnings.warn(
                "Skipping speech feature because the relative path does not "
                "contain speech_encoder/source_tts_model/speaker/emotion/file: "
                f"{relative}"
            )
            continue

        sentence_id = extract_sentence_id(relative, sentence_id_regex)

        if sentence_id is None:
            warnings.warn(f"Could not extract a sentence ID from speech feature: {path}")
            continue

        records.append(
            {
                "modality": "speech",
                "speech_encoder": relative.parts[0],
                "source_tts_model": relative.parts[1],
                "speaker": relative.parts[2],
                "emotion": relative.parts[3],
                "sentence_id": sentence_id,
                "feature_path": str(path.resolve()),
                "relative_path": relative.as_posix(),
            }
        )

    manifest = pd.DataFrame(records)

    if manifest.empty:
        raise RuntimeError(
            "No speech feature files were discovered. Check --speech-root, "
            "--extensions, --sentence-id-regex, and the directory layout."
        )

    return manifest.sort_values(
        [
            "speech_encoder",
            "source_tts_model",
            "speaker",
            "emotion",
            "sentence_id",
            "feature_path",
        ]
    ).reset_index(drop=True)


def apply_manifest_filters(
    text_manifest: pd.DataFrame,
    speech_manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if args.text_models:
        text_manifest = text_manifest[text_manifest["model"].isin(args.text_models)].copy()

    if args.speech_encoders:
        speech_manifest = speech_manifest[
            speech_manifest["speech_encoder"].isin(args.speech_encoders)
        ].copy()

    if args.source_tts_models:
        allowed = {canonical_identifier(value) for value in args.source_tts_models}
        speech_manifest = speech_manifest[
            speech_manifest["source_tts_model"].map(canonical_identifier).isin(allowed)
        ].copy()

    if args.speakers:
        allowed_speakers = {normalize_speaker_value(value) for value in args.speakers}
        speech_manifest = speech_manifest[
            speech_manifest["speaker"].map(normalize_speaker_value).isin(allowed_speakers)
        ].copy()

    if args.emotions:
        allowed_emotions = {normalize_emotion_value(value) for value in args.emotions}
        speech_manifest = speech_manifest[
            speech_manifest["emotion"].map(normalize_emotion_value).isin(allowed_emotions)
        ].copy()

    if text_manifest.empty:
        raise RuntimeError("No text files remain after applying model filters.")
    if speech_manifest.empty:
        raise RuntimeError("No speech files remain after applying condition filters.")

    return text_manifest.reset_index(drop=True), speech_manifest.reset_index(drop=True)


@dataclass
class LayerData:
    ids: np.ndarray
    matrix: np.ndarray


LayerCollection = Dict[str, LayerData]


def load_feature_group(
    manifest_group: pd.DataFrame,
    pooling: str,
    include_key_regex: Optional[re.Pattern[str]],
    exclude_key_regex: Optional[re.Pattern[str]],
    minimum_samples_per_layer: int,
    description: str,
) -> LayerCollection:
    """Load one model/condition group into layer-specific matrices."""
    vectors_by_layer: Dict[str, Dict[str, np.ndarray]] = defaultdict(dict)
    errors = []

    grouped = manifest_group.groupby("sentence_id", sort=True)

    for sentence_id, sample_rows in tqdm(
        grouped,
        total=manifest_group["sentence_id"].nunique(),
        desc=f"Loading {description}",
        unit="sample",
        leave=False,
    ):
        sample_vectors: Dict[str, np.ndarray] = {}

        for feature_path in sample_rows["feature_path"]:
            path = Path(feature_path)

            try:
                loaded_vectors = load_feature_file(
                    path=path,
                    pooling=pooling,
                    include_key_regex=include_key_regex,
                    exclude_key_regex=exclude_key_regex,
                )
            except Exception as error:
                errors.append(
                    {
                        "sentence_id": sentence_id,
                        "feature_path": str(path),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                continue

            for layer_name, vector in loaded_vectors.items():
                final_layer_name = layer_name

                if final_layer_name in sample_vectors:
                    file_specific = canonical_layer_name(path.stem, path)
                    final_layer_name = file_specific

                duplicate_index = 2
                base_name = final_layer_name

                while final_layer_name in sample_vectors:
                    final_layer_name = f"{base_name}__{duplicate_index}"
                    duplicate_index += 1

                sample_vectors[final_layer_name] = vector

        for layer_name, vector in sample_vectors.items():
            vectors_by_layer[layer_name][str(sentence_id)] = vector

    if errors:
        warnings.warn(
            f"{len(errors):,} feature files failed while loading {description}. "
            "The analysis will continue with successfully loaded files."
        )

    layers: LayerCollection = {}

    for layer_name in sorted(vectors_by_layer, key=layer_sort_key):
        sample_vectors = vectors_by_layer[layer_name]

        dimension_counts = Counter(vector.shape[0] for vector in sample_vectors.values())
        target_dimension, dimension_frequency = dimension_counts.most_common(1)[0]

        valid_items = [
            (sample_id, vector)
            for sample_id, vector in sample_vectors.items()
            if vector.shape[0] == target_dimension
        ]

        if len(valid_items) < minimum_samples_per_layer:
            continue

        valid_items.sort(key=lambda item: item[0])
        ids = np.asarray([item[0] for item in valid_items], dtype=object)
        matrix = np.stack([item[1] for item in valid_items]).astype(np.float32, copy=False)

        layers[layer_name] = LayerData(ids=ids, matrix=matrix)

        dropped = len(sample_vectors) - len(valid_items)
        if dropped:
            warnings.warn(
                f"Dropped {dropped} vectors from {description}/{layer_name} "
                f"because their dimensions differed from the modal width "
                f"{target_dimension}."
            )

    if not layers:
        error_preview = pd.DataFrame(errors).head(10).to_string(index=False) if errors else "None"
        raise RuntimeError(
            f"No usable layers were loaded for {description}.\n"
            f"Example loading errors:\n{error_preview}"
        )

    return layers


def subset_layer_collection(
    layers: LayerCollection,
    sample_ids: Sequence[str],
) -> Dict[str, np.ndarray]:
    sample_ids = [str(sample_id) for sample_id in sample_ids]
    result: Dict[str, np.ndarray] = {}

    for layer_name, layer_data in layers.items():
        index_by_id = {str(sample_id): index for index, sample_id in enumerate(layer_data.ids)}

        try:
            indices = [index_by_id[sample_id] for sample_id in sample_ids]
        except KeyError:
            continue

        result[layer_name] = layer_data.matrix[indices]

    return result


def common_ids_across_collections(*collections: LayerCollection) -> List[str]:
    id_sets: List[Set[str]] = []

    for collection in collections:
        for layer_data in collection.values():
            id_sets.append(set(str(value) for value in layer_data.ids))

    if not id_sets:
        return []

    common = set.intersection(*id_sets)
    return sorted(common)


def deterministic_subsample(
    sample_ids: Sequence[str],
    maximum_samples: int,
    seed: int,
) -> List[str]:
    sample_ids = list(sample_ids)

    if maximum_samples <= 0 or len(sample_ids) <= maximum_samples:
        return sample_ids

    generator = np.random.default_rng(seed)
    selected_indices = generator.choice(
        len(sample_ids),
        size=maximum_samples,
        replace=False,
    )

    return sorted(sample_ids[index] for index in selected_indices)


def l2_normalize(matrix: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, epsilon)
    return matrix / norms


def center_gram(gram: np.ndarray) -> np.ndarray:
    gram = np.asarray(gram, dtype=np.float64)
    row_mean = gram.mean(axis=1, keepdims=True)
    column_mean = gram.mean(axis=0, keepdims=True)
    grand_mean = gram.mean()
    return gram - row_mean - column_mean + grand_mean


def normalized_centered_gram_vector(gram: np.ndarray) -> np.ndarray:
    centered = center_gram(gram)
    vector = centered.reshape(-1)
    norm = np.linalg.norm(vector)

    if norm <= 1e-12:
        return np.zeros_like(vector)

    return vector / norm


def linear_gram_vector(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return normalized_centered_gram_vector(matrix @ matrix.T)


def pairwise_squared_distances(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    squared_norms = np.sum(matrix * matrix, axis=1, keepdims=True)
    distances = squared_norms + squared_norms.T - 2.0 * (matrix @ matrix.T)
    return np.maximum(distances, 0.0)


def preprocess_rbf_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)

    if mode == "l2":
        return l2_normalize(matrix)
    if mode == "center":
        return matrix - matrix.mean(axis=0, keepdims=True)
    if mode == "zscore":
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        scale = centered.std(axis=0, ddof=1, keepdims=True)
        return centered / np.maximum(scale, 1e-12)
    if mode == "none":
        return matrix

    raise ValueError(f"Unknown RBF preprocessing mode: {mode}")


def rbf_gram_vector(
    matrix: np.ndarray,
    sigma: Optional[float],
    preprocess: str,
) -> np.ndarray:
    processed = preprocess_rbf_matrix(matrix, mode=preprocess)
    distances = pairwise_squared_distances(processed)

    if sigma is None:
        upper = distances[np.triu_indices(distances.shape[0], k=1)]
        positive = upper[upper > 0]
        sigma_squared = float(np.median(positive)) if positive.size else 1.0
    else:
        if sigma <= 0:
            raise ValueError("RBF sigma must be positive.")
        sigma_squared = float(sigma) ** 2

    sigma_squared = max(sigma_squared, 1e-12)
    gram = np.exp(-distances / (2.0 * sigma_squared))
    return normalized_centered_gram_vector(gram)


def rsa_rank_vector(matrix: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(matrix)
    cosine_similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    cosine_distance = 1.0 - cosine_similarity
    upper = cosine_distance[np.triu_indices(cosine_distance.shape[0], k=1)]
    ranked = rankdata(upper, method="average").astype(np.float64)
    ranked -= ranked.mean()
    standard_deviation = ranked.std(ddof=1)

    if standard_deviation <= 1e-12:
        return np.zeros_like(ranked)

    return ranked / standard_deviation


def effective_k_value(k: int, number_of_samples: int) -> int:
    if number_of_samples <= 1:
        raise ValueError("At least two samples are needed for kNN analysis.")
    if k <= 0:
        raise ValueError("k must be positive.")
    return min(int(k), number_of_samples - 1)


def knn_indices(matrix: np.ndarray, k: int) -> np.ndarray:
    """Cosine kNN indices, excluding each sample from its own neighborhood."""
    matrix = l2_normalize(matrix)
    number_of_samples = matrix.shape[0]
    effective_k = effective_k_value(k, number_of_samples)
    similarities = matrix @ matrix.T
    np.fill_diagonal(similarities, -np.inf)

    indices = np.argpartition(
        -similarities,
        kth=effective_k - 1,
        axis=1,
    )[:, :effective_k]

    row_indices = np.arange(number_of_samples)[:, None]
    local_scores = similarities[row_indices, indices]
    order = np.argsort(-local_scores, axis=1, kind="stable")
    return indices[row_indices, order]


def knn_adjacency_vector_from_indices(
    indices: np.ndarray,
    number_of_samples: int,
) -> np.ndarray:
    effective_k = indices.shape[1]
    adjacency = np.zeros(
        (number_of_samples, number_of_samples),
        dtype=np.float32,
    )
    rows = np.repeat(np.arange(number_of_samples), effective_k)
    adjacency[rows, indices.reshape(-1)] = 1.0
    return adjacency.reshape(-1)


def knn_indicator_matrix_from_indices(
    indices: np.ndarray,
    number_of_samples: int,
) -> np.ndarray:
    matrix = np.zeros((number_of_samples, number_of_samples), dtype=np.float64)
    rows = np.arange(number_of_samples)[:, None]
    matrix[rows, indices] = 1.0
    return matrix


def mutual_knn_from_indices(
    first_indices: np.ndarray,
    second_indices: np.ndarray,
) -> float:
    if first_indices.shape != second_indices.shape:
        raise ValueError("The two kNN index arrays must have the same shape.")

    number_of_samples, effective_k = first_indices.shape
    first_adjacency = knn_adjacency_vector_from_indices(
        first_indices,
        number_of_samples,
    )
    second_adjacency = knn_adjacency_vector_from_indices(
        second_indices,
        number_of_samples,
    )
    return float(
        np.dot(first_adjacency, second_adjacency)
        / (number_of_samples * effective_k)
    )


def cycle_knn_from_indices(
    first_indices: np.ndarray,
    second_indices: np.ndarray,
) -> float:
    if first_indices.shape != second_indices.shape:
        raise ValueError("The two kNN index arrays must have the same shape.")

    number_of_samples = first_indices.shape[0]
    cycle_indices = first_indices[second_indices]
    targets = np.arange(number_of_samples)[:, None, None]
    recovered = np.any(cycle_indices == targets, axis=(1, 2))
    return float(recovered.mean())


def hsic_biased_numpy(first_gram: np.ndarray, second_gram: np.ndarray) -> float:
    first_centered = center_gram(first_gram)
    second_centered = center_gram(second_gram)
    return float(np.sum(first_centered * second_centered))


def hsic_unbiased_numpy(first_gram: np.ndarray, second_gram: np.ndarray) -> float:
    """Unbiased HSIC estimator used by the original PRH CKNNA code."""
    first = np.asarray(first_gram, dtype=np.float64).copy()
    second = np.asarray(second_gram, dtype=np.float64).copy()
    number_of_samples = first.shape[0]

    if number_of_samples < 4:
        return float("nan")

    np.fill_diagonal(first, 0.0)
    np.fill_diagonal(second, 0.0)

    term_one = np.sum(first * second.T)
    term_two = (
        np.sum(first) * np.sum(second)
        / ((number_of_samples - 1) * (number_of_samples - 2))
    )
    term_three = 2.0 * np.sum(first @ second) / (number_of_samples - 2)
    return float(
        (term_one + term_two - term_three)
        / (number_of_samples * (number_of_samples - 3))
    )


def cknna_from_prepared(
    first_gram: np.ndarray,
    second_gram: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    first_self_similarity: float,
    second_self_similarity: float,
    unbiased: bool,
    distance_agnostic: bool,
) -> float:
    shared_mask = first_mask * second_mask

    if distance_agnostic:
        numerator = float(shared_mask.sum())
    else:
        estimator = hsic_unbiased_numpy if unbiased else hsic_biased_numpy
        numerator = estimator(
            shared_mask * first_gram,
            shared_mask * second_gram,
        )

    denominator_product = first_self_similarity * second_self_similarity
    if not np.isfinite(denominator_product) or denominator_product <= 1e-12:
        return float("nan")

    score = numerator / math.sqrt(denominator_product)
    return float(np.clip(score, -1.0, 1.0))


def sigma_metric_name(sigma: float) -> str:
    token = format(float(sigma), ".12g")
    token = token.replace("-", "m").replace(".", "p")
    return f"rbf_cka_sigma_{token}"


def parse_sigma_metric_name(metric_name: str) -> float:
    prefix = "rbf_cka_sigma_"
    if not metric_name.startswith(prefix):
        raise ValueError(f"Not a fixed-sigma RBF CKA metric: {metric_name}")
    token = metric_name[len(prefix):].replace("m", "-").replace("p", ".")
    return float(token)


def prepare_layer_metric_objects(
    layer_matrices: Dict[str, np.ndarray],
    k_values: Sequence[int],
    cycle_k_values: Sequence[int],
    cknna_k_values: Sequence[int],
    rbf_sigmas: Sequence[float],
    rbf_preprocess: str,
    median_rbf_cka: bool,
    cknna_unbiased: bool,
    cknna_distance_agnostic: bool,
) -> Dict[str, Any]:
    layer_names = sorted(layer_matrices, key=layer_sort_key)
    all_k_values = sorted(
        set(int(k) for k in k_values)
        | set(int(k) for k in cycle_k_values)
        | set(int(k) for k in cknna_k_values)
    )

    prepared: Dict[str, Any] = {
        "layer_names": layer_names,
        "linear_vectors": np.stack(
            [linear_gram_vector(layer_matrices[layer]) for layer in layer_names]
        ),
        "rsa_vectors": np.stack(
            [rsa_rank_vector(layer_matrices[layer]) for layer in layer_names]
        ),
        "rbf_vectors": {},
        "knn_indices": {},
        "knn_adjacency": {},
        "normalized": {},
        "gram": {},
        "knn_masks": {},
        "cknna_self": {},
    }

    for sigma in rbf_sigmas:
        prepared["rbf_vectors"][sigma_metric_name(sigma)] = np.stack(
            [
                rbf_gram_vector(
                    layer_matrices[layer],
                    sigma=float(sigma),
                    preprocess=rbf_preprocess,
                )
                for layer in layer_names
            ]
        )

    if median_rbf_cka:
        prepared["rbf_vectors"]["rbf_cka_median"] = np.stack(
            [
                rbf_gram_vector(
                    layer_matrices[layer],
                    sigma=None,
                    preprocess=rbf_preprocess,
                )
                for layer in layer_names
            ]
        )

    for layer_name in layer_names:
        normalized = l2_normalize(layer_matrices[layer_name])
        prepared["normalized"][layer_name] = normalized
        prepared["gram"][layer_name] = normalized @ normalized.T

    for k in all_k_values:
        per_layer_indices: Dict[str, np.ndarray] = {}
        per_layer_adjacency: List[np.ndarray] = []
        per_layer_masks: Dict[str, np.ndarray] = {}

        for layer_name in layer_names:
            indices = knn_indices(layer_matrices[layer_name], k=k)
            per_layer_indices[layer_name] = indices
            per_layer_adjacency.append(
                knn_adjacency_vector_from_indices(
                    indices,
                    number_of_samples=indices.shape[0],
                )
            )
            per_layer_masks[layer_name] = knn_indicator_matrix_from_indices(
                indices,
                number_of_samples=indices.shape[0],
            )

        prepared["knn_indices"][k] = per_layer_indices
        prepared["knn_adjacency"][k] = np.stack(per_layer_adjacency)
        prepared["knn_masks"][k] = per_layer_masks

        if k in cknna_k_values:
            estimator = hsic_unbiased_numpy if cknna_unbiased else hsic_biased_numpy
            self_values: Dict[str, float] = {}

            for layer_name in layer_names:
                mask = per_layer_masks[layer_name]
                if cknna_distance_agnostic:
                    self_values[layer_name] = float(mask.sum())
                else:
                    masked_gram = mask * prepared["gram"][layer_name]
                    self_values[layer_name] = estimator(masked_gram, masked_gram)

            prepared["cknna_self"][k] = self_values

    return prepared


def compute_alignment_matrices(
    speech_matrices: Dict[str, np.ndarray],
    text_matrices: Dict[str, np.ndarray],
    k_values: Sequence[int],
    cycle_k_values: Sequence[int],
    cknna_k_values: Sequence[int],
    rbf_sigmas: Sequence[float],
    rbf_preprocess: str,
    median_rbf_cka: bool,
    cknna_unbiased: bool,
    cknna_distance_agnostic: bool,
) -> Dict[str, pd.DataFrame]:
    prepared_speech = prepare_layer_metric_objects(
        speech_matrices,
        k_values=k_values,
        cycle_k_values=cycle_k_values,
        cknna_k_values=cknna_k_values,
        rbf_sigmas=rbf_sigmas,
        rbf_preprocess=rbf_preprocess,
        median_rbf_cka=median_rbf_cka,
        cknna_unbiased=cknna_unbiased,
        cknna_distance_agnostic=cknna_distance_agnostic,
    )
    prepared_text = prepare_layer_metric_objects(
        text_matrices,
        k_values=k_values,
        cycle_k_values=cycle_k_values,
        cknna_k_values=cknna_k_values,
        rbf_sigmas=rbf_sigmas,
        rbf_preprocess=rbf_preprocess,
        median_rbf_cka=median_rbf_cka,
        cknna_unbiased=cknna_unbiased,
        cknna_distance_agnostic=cknna_distance_agnostic,
    )

    speech_layers = prepared_speech["layer_names"]
    text_layers = prepared_text["layer_names"]
    number_of_samples = next(iter(speech_matrices.values())).shape[0]
    results: Dict[str, pd.DataFrame] = {}

    linear_values = prepared_speech["linear_vectors"] @ prepared_text["linear_vectors"].T
    results["linear_cka"] = pd.DataFrame(
        np.clip(linear_values, -1.0, 1.0),
        index=speech_layers,
        columns=text_layers,
    )

    rsa_denominator = max(prepared_speech["rsa_vectors"].shape[1] - 1, 1)
    rsa_values = (
        prepared_speech["rsa_vectors"] @ prepared_text["rsa_vectors"].T
    ) / rsa_denominator
    results["rsa_spearman"] = pd.DataFrame(
        np.clip(rsa_values, -1.0, 1.0),
        index=speech_layers,
        columns=text_layers,
    )

    for metric_name, speech_vectors in prepared_speech["rbf_vectors"].items():
        text_vectors = prepared_text["rbf_vectors"][metric_name]
        values = speech_vectors @ text_vectors.T
        results[metric_name] = pd.DataFrame(
            np.clip(values, -1.0, 1.0),
            index=speech_layers,
            columns=text_layers,
        )

    for k in k_values:
        effective_k = effective_k_value(k, number_of_samples)
        speech_vectors = prepared_speech["knn_adjacency"][k]
        text_vectors = prepared_text["knn_adjacency"][k]
        values = (speech_vectors @ text_vectors.T) / (
            number_of_samples * effective_k
        )
        chance = effective_k / max(number_of_samples - 1, 1)
        adjusted = (values - chance) / max(1.0 - chance, 1e-12)

        metric_name = f"mutual_knn_k{k}"
        results[metric_name] = pd.DataFrame(
            values,
            index=speech_layers,
            columns=text_layers,
        )
        results[f"adjusted_{metric_name}"] = pd.DataFrame(
            adjusted,
            index=speech_layers,
            columns=text_layers,
        )

    for k in cycle_k_values:
        values = np.empty((len(speech_layers), len(text_layers)), dtype=np.float64)
        speech_indices = prepared_speech["knn_indices"][k]
        text_indices = prepared_text["knn_indices"][k]

        for speech_index, speech_layer in enumerate(speech_layers):
            for text_index, text_layer in enumerate(text_layers):
                values[speech_index, text_index] = cycle_knn_from_indices(
                    speech_indices[speech_layer],
                    text_indices[text_layer],
                )

        results[f"cycle_knn_k{k}"] = pd.DataFrame(
            values,
            index=speech_layers,
            columns=text_layers,
        )

    for k in cknna_k_values:
        if effective_k_value(k, number_of_samples) < 2:
            warnings.warn(f"Skipping CKNNA k={k}: CKNNA requires k >= 2.")
            continue

        values = np.empty((len(speech_layers), len(text_layers)), dtype=np.float64)

        for speech_index, speech_layer in enumerate(speech_layers):
            for text_index, text_layer in enumerate(text_layers):
                values[speech_index, text_index] = cknna_from_prepared(
                    first_gram=prepared_speech["gram"][speech_layer],
                    second_gram=prepared_text["gram"][text_layer],
                    first_mask=prepared_speech["knn_masks"][k][speech_layer],
                    second_mask=prepared_text["knn_masks"][k][text_layer],
                    first_self_similarity=prepared_speech["cknna_self"][k][speech_layer],
                    second_self_similarity=prepared_text["cknna_self"][k][text_layer],
                    unbiased=cknna_unbiased,
                    distance_agnostic=cknna_distance_agnostic,
                )

        results[f"cknna_k{k}"] = pd.DataFrame(
            values,
            index=speech_layers,
            columns=text_layers,
        )

    return results


def alignment_matrices_to_long(
    matrices: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    number_of_samples: int,
) -> pd.DataFrame:
    rows = []
    metric_names = list(matrices)
    reference = matrices[metric_names[0]]

    for speech_layer in reference.index:
        for text_layer in reference.columns:
            row = dict(metadata)
            row.update(
                {
                    "speech_layer": speech_layer,
                    "text_layer": text_layer,
                    "n_samples": number_of_samples,
                }
            )

            for metric_name, matrix in matrices.items():
                row[metric_name] = float(matrix.loc[speech_layer, text_layer])

            rows.append(row)

    return pd.DataFrame(rows)


def extract_top_layer_pairs(
    matrices: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    number_of_samples: int,
) -> pd.DataFrame:
    """Raw layer maxima. These are descriptive and not depth-calibrated."""
    rows = []

    for metric_name, matrix in matrices.items():
        values = matrix.to_numpy(dtype=np.float64)
        if not np.isfinite(values).any():
            continue
        flat_index = int(np.nanargmax(values))
        speech_index, text_index = np.unravel_index(flat_index, values.shape)

        row = dict(metadata)
        row.update(
            {
                "metric": metric_name,
                "speech_layer": matrix.index[speech_index],
                "text_layer": matrix.columns[text_index],
                "score": float(values[speech_index, text_index]),
                "n_samples": number_of_samples,
                "summary_status": "raw_descriptive_not_depth_calibrated",
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def best_text_layer_per_speech_layer(
    matrices: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
    number_of_samples: int,
) -> pd.DataFrame:
    rows = []

    for metric_name, matrix in matrices.items():
        for speech_layer in matrix.index:
            row_values = matrix.loc[speech_layer]
            if not np.isfinite(row_values.to_numpy(dtype=np.float64)).any():
                continue
            best_text_layer = str(row_values.idxmax())
            row = dict(metadata)
            row.update(
                {
                    "metric": metric_name,
                    "speech_layer": speech_layer,
                    "best_text_layer": best_text_layer,
                    "score": float(row_values.loc[best_text_layer]),
                    "n_samples": number_of_samples,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


class MatrixAccumulator:
    def __init__(self) -> None:
        self._sum: Dict[Tuple[str, ...], pd.DataFrame] = {}
        self._count: Dict[Tuple[str, ...], pd.DataFrame] = {}

    def add(self, key: Tuple[str, ...], matrix: pd.DataFrame) -> None:
        matrix = matrix.astype(np.float64)
        count_matrix = pd.DataFrame(
            np.isfinite(matrix.to_numpy()).astype(np.float64),
            index=matrix.index,
            columns=matrix.columns,
        )
        value_matrix = matrix.fillna(0.0)

        if key not in self._sum:
            self._sum[key] = value_matrix.copy()
            self._count[key] = count_matrix
            return

        all_rows = self._sum[key].index.union(value_matrix.index)
        all_columns = self._sum[key].columns.union(value_matrix.columns)

        self._sum[key] = (
            self._sum[key]
            .reindex(index=all_rows, columns=all_columns, fill_value=0.0)
            .add(
                value_matrix.reindex(index=all_rows, columns=all_columns, fill_value=0.0),
                fill_value=0.0,
            )
        )

        self._count[key] = (
            self._count[key]
            .reindex(index=all_rows, columns=all_columns, fill_value=0.0)
            .add(
                count_matrix.reindex(index=all_rows, columns=all_columns, fill_value=0.0),
                fill_value=0.0,
            )
        )

    def mean_items(self) -> Iterator[Tuple[Tuple[str, ...], pd.DataFrame]]:
        for key in sorted(self._sum):
            denominator = self._count[key].replace(0.0, np.nan)
            yield key, self._sum[key] / denominator


def save_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    value_label: str,
) -> None:
    import matplotlib.pyplot as plt

    matrix = matrix.reindex(
        index=sorted(matrix.index, key=layer_sort_key),
        columns=sorted(matrix.columns, key=layer_sort_key),
    )

    width = max(8.0, 0.36 * len(matrix.columns) + 3.0)
    height = max(6.0, 0.34 * len(matrix.index) + 2.5)

    figure, axes = plt.subplots(figsize=(width, height))
    image = axes.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest")
    colorbar = figure.colorbar(image, ax=axes)
    colorbar.set_label(value_label)

    axes.set_title(title)
    axes.set_xlabel("Text layer")
    axes.set_ylabel("Speech layer")
    axes.set_xticks(np.arange(len(matrix.columns)))
    axes.set_yticks(np.arange(len(matrix.index)))
    axes.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
    axes.set_yticklabels(matrix.index, fontsize=7)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def load_sick_pairs(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None

    if not path.exists():
        warnings.warn(
            f"SICK pair metadata was not found and semantic validation will be skipped:\n{path}"
        )
        return None

    dataframe = pd.read_csv(path)

    required = {
        "sentence_a_id",
        "sentence_b_id",
        "relatedness_score",
    }

    missing = required.difference(dataframe.columns)

    if missing:
        raise ValueError(
            f"The pairs CSV is missing required columns: {sorted(missing)}\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    dataframe["sentence_a_id"] = dataframe["sentence_a_id"].astype(str)
    dataframe["sentence_b_id"] = dataframe["sentence_b_id"].astype(str)
    dataframe["relatedness_score"] = pd.to_numeric(
        dataframe["relatedness_score"], errors="coerce"
    )
    dataframe = dataframe.dropna(subset=["relatedness_score"]).copy()

    if "similarity_group" not in dataframe.columns:
        dataframe["similarity_group"] = "unknown"

    return dataframe


def validate_relatedness_for_layers(
    layers: LayerCollection,
    pairs: pd.DataFrame,
    metadata: Dict[str, Any],
    minimum_pairs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    correlation_rows = []
    group_rows = []

    for layer_name, layer_data in layers.items():
        index_by_id = {
            str(sample_id): index
            for index, sample_id in enumerate(layer_data.ids)
        }

        valid_mask = pairs["sentence_a_id"].isin(index_by_id) & pairs[
            "sentence_b_id"
        ].isin(index_by_id)
        valid_pairs = pairs[valid_mask].copy()

        if len(valid_pairs) < minimum_pairs:
            continue

        indices_a = [index_by_id[value] for value in valid_pairs["sentence_a_id"]]
        indices_b = [index_by_id[value] for value in valid_pairs["sentence_b_id"]]

        vectors_a = l2_normalize(layer_data.matrix[indices_a].astype(np.float64))
        vectors_b = l2_normalize(layer_data.matrix[indices_b].astype(np.float64))
        cosine_similarities = np.sum(vectors_a * vectors_b, axis=1)
        human_scores = valid_pairs["relatedness_score"].to_numpy(dtype=np.float64)

        correlation, p_value = spearmanr(human_scores, cosine_similarities)

        row = dict(metadata)
        row.update(
            {
                "layer": layer_name,
                "n_pairs": len(valid_pairs),
                "spearman_relatedness": float(correlation),
                "spearman_p_value": float(p_value),
            }
        )
        correlation_rows.append(row)

        grouped_values = []

        for similarity_group, group_frame in valid_pairs.assign(
            embedding_cosine=cosine_similarities
        ).groupby("similarity_group", dropna=False):
            values = group_frame["embedding_cosine"].to_numpy(dtype=np.float64)
            grouped_values.append(values)

            group_row = dict(metadata)
            group_row.update(
                {
                    "layer": layer_name,
                    "similarity_group": str(similarity_group),
                    "n_pairs": len(values),
                    "mean_cosine": float(np.mean(values)),
                    "std_cosine": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
                    "median_cosine": float(np.median(values)),
                }
            )
            group_rows.append(group_row)

        nonempty = [values for values in grouped_values if len(values) > 0]

        if len(nonempty) >= 2:
            try:
                statistic, group_p_value = kruskal(*nonempty)
                correlation_rows[-1]["kruskal_statistic"] = float(statistic)
                correlation_rows[-1]["kruskal_p_value"] = float(group_p_value)
            except ValueError:
                correlation_rows[-1]["kruskal_statistic"] = np.nan
                correlation_rows[-1]["kruskal_p_value"] = np.nan

    return pd.DataFrame(correlation_rows), pd.DataFrame(group_rows)


def scalar_linear_cka(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    return float(np.dot(linear_gram_vector(matrix_a), linear_gram_vector(matrix_b)))


def scalar_rsa(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    vector_a = rsa_rank_vector(matrix_a)
    vector_b = rsa_rank_vector(matrix_b)
    denominator = max(len(vector_a) - 1, 1)
    return float(np.dot(vector_a, vector_b) / denominator)


def scalar_rbf_cka(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    sigma: Optional[float],
    preprocess: str,
) -> float:
    first = rbf_gram_vector(matrix_a, sigma=sigma, preprocess=preprocess)
    second = rbf_gram_vector(matrix_b, sigma=sigma, preprocess=preprocess)
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def scalar_mutual_knn(matrix_a: np.ndarray, matrix_b: np.ndarray, k: int) -> float:
    return mutual_knn_from_indices(knn_indices(matrix_a, k), knn_indices(matrix_b, k))


def scalar_cycle_knn(matrix_a: np.ndarray, matrix_b: np.ndarray, k: int) -> float:
    return cycle_knn_from_indices(knn_indices(matrix_a, k), knn_indices(matrix_b, k))


def scalar_cknna(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    k: int,
    unbiased: bool,
    distance_agnostic: bool,
) -> float:
    number_of_samples = matrix_a.shape[0]
    effective_k = effective_k_value(k, number_of_samples)
    if effective_k < 2:
        return float("nan")

    first_normalized = l2_normalize(matrix_a)
    second_normalized = l2_normalize(matrix_b)
    first_gram = first_normalized @ first_normalized.T
    second_gram = second_normalized @ second_normalized.T
    first_indices = knn_indices(first_normalized, effective_k)
    second_indices = knn_indices(second_normalized, effective_k)
    first_mask = knn_indicator_matrix_from_indices(first_indices, number_of_samples)
    second_mask = knn_indicator_matrix_from_indices(second_indices, number_of_samples)
    estimator = hsic_unbiased_numpy if unbiased else hsic_biased_numpy

    if distance_agnostic:
        first_self = float(first_mask.sum())
        second_self = float(second_mask.sum())
    else:
        first_self = estimator(first_mask * first_gram, first_mask * first_gram)
        second_self = estimator(second_mask * second_gram, second_mask * second_gram)

    return cknna_from_prepared(
        first_gram=first_gram,
        second_gram=second_gram,
        first_mask=first_mask,
        second_mask=second_mask,
        first_self_similarity=first_self,
        second_self_similarity=second_self,
        unbiased=unbiased,
        distance_agnostic=distance_agnostic,
    )


def metric_scalar_numpy(
    metric_name: str,
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    args: argparse.Namespace,
) -> float:
    if metric_name == "linear_cka":
        return scalar_linear_cka(matrix_a, matrix_b)
    if metric_name == "rsa_spearman":
        return scalar_rsa(matrix_a, matrix_b)
    if metric_name == "rbf_cka_median":
        return scalar_rbf_cka(
            matrix_a,
            matrix_b,
            sigma=None,
            preprocess=args.rbf_preprocess,
        )
    if metric_name.startswith("rbf_cka_sigma_"):
        return scalar_rbf_cka(
            matrix_a,
            matrix_b,
            sigma=parse_sigma_metric_name(metric_name),
            preprocess=args.rbf_preprocess,
        )
    if metric_name.startswith("mutual_knn_k"):
        k = int(metric_name.rsplit("k", maxsplit=1)[1])
        return scalar_mutual_knn(matrix_a, matrix_b, k=k)
    if metric_name.startswith("cycle_knn_k"):
        k = int(metric_name.rsplit("k", maxsplit=1)[1])
        return scalar_cycle_knn(matrix_a, matrix_b, k=k)
    if metric_name.startswith("cknna_k"):
        k = int(metric_name.rsplit("k", maxsplit=1)[1])
        return scalar_cknna(
            matrix_a,
            matrix_b,
            k=k,
            unbiased=args.cknna_unbiased,
            distance_agnostic=args.cknna_distance_agnostic,
        )
    raise ValueError(f"Unknown metric: {metric_name}")


def import_official_calibration():
    try:
        from calibrated_similarity import calibrate_layers
    except ImportError as error:
        raise ImportError(
            "Aggregation-aware calibration requires the official package. "
            "Install it with:\n\n"
            "    pip install calibrated-similarity\n\n"
            "If your cluster overrides the Python package index, use:\n\n"
            "    pip install --index-url https://pypi.org/simple calibrated-similarity\n"
        ) from error
    return calibrate_layers


def torch_center_gram(gram):
    return (
        gram
        - gram.mean(dim=1, keepdim=True)
        - gram.mean(dim=0, keepdim=True)
        + gram.mean()
    )


def torch_l2_normalize(matrix, epsilon: float = 1e-12):
    import torch
    return matrix / torch.linalg.vector_norm(
        matrix,
        dim=1,
        keepdim=True,
    ).clamp_min(epsilon)


def torch_linear_cka(first, second):
    import torch
    first_centered = first - first.mean(dim=0, keepdim=True)
    second_centered = second - second.mean(dim=0, keepdim=True)
    cross = first_centered.T @ second_centered
    first_self = first_centered.T @ first_centered
    second_self = second_centered.T @ second_centered
    numerator = torch.sum(cross * cross)
    denominator = torch.sqrt(
        torch.sum(first_self * first_self)
        * torch.sum(second_self * second_self)
    ).clamp_min(1e-12)
    return torch.clamp(numerator / denominator, min=-1.0, max=1.0)


def torch_preprocess_rbf(matrix, mode: str):
    if mode == "l2":
        return torch_l2_normalize(matrix)
    if mode == "center":
        return matrix - matrix.mean(dim=0, keepdim=True)
    if mode == "zscore":
        centered = matrix - matrix.mean(dim=0, keepdim=True)
        scale = centered.std(dim=0, unbiased=True, keepdim=True).clamp_min(1e-12)
        return centered / scale
    if mode == "none":
        return matrix
    raise ValueError(f"Unknown RBF preprocessing mode: {mode}")


def torch_rbf_cka(first, second, sigma: Optional[float], preprocess: str):
    import torch
    first = torch_preprocess_rbf(first, preprocess)
    second = torch_preprocess_rbf(second, preprocess)
    first_distances = torch.cdist(first, first).square()
    second_distances = torch.cdist(second, second).square()

    if sigma is None:
        n = first.shape[0]
        triangular = torch.triu_indices(n, n, offset=1, device=first.device)
        first_positive = first_distances[triangular[0], triangular[1]]
        second_positive = second_distances[triangular[0], triangular[1]]
        combined = torch.cat([first_positive, second_positive])
        combined = combined[combined > 0]
        sigma_squared = torch.median(combined) if combined.numel() else first.new_tensor(1.0)
    else:
        sigma_squared = first.new_tensor(float(sigma) ** 2)

    sigma_squared = sigma_squared.clamp_min(1e-12)
    first_gram = torch.exp(-first_distances / (2.0 * sigma_squared))
    second_gram = torch.exp(-second_distances / (2.0 * sigma_squared))
    first_centered = torch_center_gram(first_gram)
    second_centered = torch_center_gram(second_gram)
    denominator = (
        torch.linalg.vector_norm(first_centered)
        * torch.linalg.vector_norm(second_centered)
    ).clamp_min(1e-12)
    return torch.clamp(
        torch.sum(first_centered * second_centered) / denominator,
        min=-1.0,
        max=1.0,
    )


def torch_knn_indices(matrix, k: int):
    import torch
    normalized = torch_l2_normalize(matrix)
    n = normalized.shape[0]
    effective_k = min(int(k), n - 1)
    similarities = normalized @ normalized.T
    similarities.fill_diagonal_(-torch.inf)
    return torch.topk(similarities, k=effective_k, dim=1, largest=True).indices


def torch_mutual_knn(first, second, k: int):
    import torch
    first_indices = torch_knn_indices(first, k)
    second_indices = torch_knn_indices(second, k)
    n, effective_k = first_indices.shape
    rows = torch.arange(n, device=first.device).unsqueeze(1)
    first_mask = torch.zeros((n, n), device=first.device, dtype=first.dtype)
    second_mask = torch.zeros((n, n), device=first.device, dtype=first.dtype)
    first_mask[rows, first_indices] = 1.0
    second_mask[rows, second_indices] = 1.0
    return (first_mask * second_mask).sum(dim=1).div(effective_k).mean()


def torch_cycle_knn(first, second, k: int):
    import torch
    first_indices = torch_knn_indices(first, k)
    second_indices = torch_knn_indices(second, k)
    n = first_indices.shape[0]
    cycle_indices = first_indices[second_indices]
    targets = torch.arange(n, device=first.device).view(n, 1, 1)
    return (cycle_indices == targets).reshape(n, -1).any(dim=1).float().mean()


def torch_hsic_biased(first_gram, second_gram):
    first_centered = torch_center_gram(first_gram)
    second_centered = torch_center_gram(second_gram)
    return (first_centered * second_centered).sum()


def torch_hsic_unbiased(first_gram, second_gram):
    import torch
    first = first_gram.clone()
    second = second_gram.clone()
    n = first.shape[0]
    if n < 4:
        return first.new_tensor(float("nan"))
    first.fill_diagonal_(0.0)
    second.fill_diagonal_(0.0)
    value = (
        (first * second.T).sum()
        + first.sum() * second.sum() / ((n - 1) * (n - 2))
        - 2.0 * (first @ second).sum() / (n - 2)
    )
    return value / (n * (n - 3))


def torch_cknna(
    first,
    second,
    k: int,
    unbiased: bool,
    distance_agnostic: bool,
):
    import torch
    first = torch_l2_normalize(first)
    second = torch_l2_normalize(second)
    n = first.shape[0]
    effective_k = min(int(k), n - 1)
    if effective_k < 2:
        return first.new_tensor(float("nan"))

    first_gram = first @ first.T
    second_gram = second @ second.T
    first_indices = torch_knn_indices(first, effective_k)
    second_indices = torch_knn_indices(second, effective_k)
    rows = torch.arange(n, device=first.device).unsqueeze(1)
    first_mask = torch.zeros((n, n), device=first.device, dtype=first.dtype)
    second_mask = torch.zeros((n, n), device=first.device, dtype=first.dtype)
    first_mask[rows, first_indices] = 1.0
    second_mask[rows, second_indices] = 1.0
    shared_mask = first_mask * second_mask

    if distance_agnostic:
        numerator = shared_mask.sum()
        first_self = first_mask.sum()
        second_self = second_mask.sum()
    else:
        estimator = torch_hsic_unbiased if unbiased else torch_hsic_biased
        numerator = estimator(shared_mask * first_gram, shared_mask * second_gram)
        first_self = estimator(first_mask * first_gram, first_mask * first_gram)
        second_self = estimator(second_mask * second_gram, second_mask * second_gram)

    denominator = torch.sqrt((first_self * second_self).clamp_min(1e-12))
    return torch.clamp(numerator / denominator, min=-1.0, max=1.0)


def torch_rsa_spearman(first, second):
    """Torch rank correlation. Ties use deterministic ordinal ranks."""
    import torch

    def rank_vector(matrix):
        matrix = torch_l2_normalize(matrix)
        distances = 1.0 - (matrix @ matrix.T).clamp(-1.0, 1.0)
        indices = torch.triu_indices(
            distances.shape[0],
            distances.shape[0],
            offset=1,
            device=distances.device,
        )
        values = distances[indices[0], indices[1]]
        order = torch.argsort(values, stable=True)
        ranks = torch.empty_like(order, dtype=matrix.dtype)
        ranks[order] = torch.arange(
            len(values),
            device=matrix.device,
            dtype=matrix.dtype,
        )
        ranks = ranks - ranks.mean()
        return ranks / ranks.norm().clamp_min(1e-12)

    return torch.clamp((rank_vector(first) * rank_vector(second)).sum(), -1.0, 1.0)


def make_torch_similarity(metric_name: str, args: argparse.Namespace):
    if metric_name == "linear_cka":
        return torch_linear_cka
    if metric_name == "rsa_spearman":
        return torch_rsa_spearman
    if metric_name == "rbf_cka_median":
        return lambda first, second: torch_rbf_cka(
            first,
            second,
            sigma=None,
            preprocess=args.rbf_preprocess,
        )
    if metric_name.startswith("rbf_cka_sigma_"):
        sigma = parse_sigma_metric_name(metric_name)
        return lambda first, second: torch_rbf_cka(
            first,
            second,
            sigma=sigma,
            preprocess=args.rbf_preprocess,
        )
    if metric_name.startswith("mutual_knn_k"):
        k = int(metric_name.rsplit("k", maxsplit=1)[1])
        return lambda first, second: torch_mutual_knn(first, second, k=k)
    if metric_name.startswith("cycle_knn_k"):
        k = int(metric_name.rsplit("k", maxsplit=1)[1])
        return lambda first, second: torch_cycle_knn(first, second, k=k)
    if metric_name.startswith("cknna_k"):
        k = int(metric_name.rsplit("k", maxsplit=1)[1])
        return lambda first, second: torch_cknna(
            first,
            second,
            k=k,
            unbiased=args.cknna_unbiased,
            distance_agnostic=args.cknna_distance_agnostic,
        )
    raise ValueError(f"Calibration is not implemented for metric: {metric_name}")


def aggregation_aware_calibration(
    speech_matrices: Dict[str, np.ndarray],
    text_matrices: Dict[str, np.ndarray],
    observed_matrices: Dict[str, pd.DataFrame],
    metrics: Sequence[str],
    aggregators: Sequence[str],
    permutations: int,
    alpha: float,
    seed: int,
    device: str,
    metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Aggregation-aware permutation calibration (calibrated-similarity, Algorithm 2)."""
    import torch

    calibrate_layers = import_official_calibration()
    if args.calibration_num_threads is not None:
        torch.set_num_threads(args.calibration_num_threads)
    speech_layer_names = sorted(speech_matrices, key=layer_sort_key)
    text_layer_names = sorted(text_matrices, key=layer_sort_key)
    torch_device = torch.device(device)
    speech_layers = [
        torch.as_tensor(
            speech_matrices[name],
            dtype=torch.float32,
            device=torch_device,
        )
        for name in speech_layer_names
    ]
    text_layers = [
        torch.as_tensor(
            text_matrices[name],
            dtype=torch.float32,
            device=torch_device,
        )
        for name in text_layer_names
    ]

    rows = []

    for metric_name in metrics:
        if metric_name not in observed_matrices:
            warnings.warn(
                f"Skipping calibration metric {metric_name}: no raw matrix was computed."
            )
            continue

        similarity = make_torch_similarity(metric_name, args)
        raw_matrix = observed_matrices[metric_name]
        raw_values = raw_matrix.to_numpy(dtype=np.float64)
        if not np.isfinite(raw_values).any():
            continue

        flat_index = int(np.nanargmax(raw_values))
        best_speech_index, best_text_index = np.unravel_index(
            flat_index,
            raw_values.shape,
        )

        for aggregator in aggregators:
            if aggregator == "max":
                observed_aggregate = float(np.nanmax(raw_values))
            elif aggregator == "mean":
                observed_aggregate = float(np.nanmean(raw_values))
            else:
                raise ValueError(
                    "Only 'max' and 'mean' are supported by the CLI calibration interface."
                )

            generator = torch.Generator(device=torch_device)
            generator.manual_seed(
                stable_seed(metric_name, aggregator, base_seed=seed)
            )

            calibrated, p_value, tau = calibrate_layers(
                speech_layers,
                text_layers,
                similarity,
                agg=aggregator,
                K=permutations,
                alpha=alpha,
                smax=1.0,
                generator=generator,
            )

            row = dict(metadata)
            row.update(
                {
                    "metric": metric_name,
                    "aggregator": aggregator,
                    "observed_aggregate": observed_aggregate,
                    "calibrated_score": float(calibrated.detach().cpu()),
                    "permutation_p_value": float(p_value.detach().cpu()),
                    "critical_threshold_tau": float(tau.detach().cpu()),
                    "raw_best_speech_layer": speech_layer_names[best_speech_index],
                    "raw_best_text_layer": text_layer_names[best_text_index],
                    "raw_best_layer_score": float(
                        raw_values[best_speech_index, best_text_index]
                    ),
                    "n_samples": int(speech_layers[0].shape[0]),
                    "n_speech_layers": len(speech_layers),
                    "n_text_layers": len(text_layers),
                    "n_layer_pairs": len(speech_layers) * len(text_layers),
                    "n_permutations": permutations,
                    "alpha": alpha,
                    "calibration_method": (
                        "official_calibrated_similarity_aggregation_aware_algorithm_2"
                    ),
                    "calibration_device": str(torch_device),
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p_values_array = np.asarray(p_values, dtype=np.float64)
    q_values = np.full_like(p_values_array, np.nan)
    valid = np.isfinite(p_values_array)

    if not valid.any():
        return q_values

    valid_values = p_values_array[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    number = len(ranked)
    adjusted = ranked * number / np.arange(1, number + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    q_values[valid] = restored
    return q_values


def add_fdr_columns(
    calibration_frame: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    frame = calibration_frame.copy()
    frame["q_value_global"] = benjamini_hochberg(
        frame["permutation_p_value"].to_numpy(dtype=np.float64)
    )
    frame["significant_fdr_global"] = frame["q_value_global"] <= alpha

    frame["q_value_within_metric"] = np.nan
    for _, indices in frame.groupby(["metric", "aggregator"]).groups.items():
        indices = list(indices)
        frame.loc[indices, "q_value_within_metric"] = benjamini_hochberg(
            frame.loc[indices, "permutation_p_value"].to_numpy(dtype=np.float64)
        )

    frame["significant_fdr_within_metric"] = (
        frame["q_value_within_metric"] <= alpha
    )
    return frame


def build_aristotelian_topology_metric_summary(
    calibration_frame: pd.DataFrame,
    topology_k: int,
    local_rbf_sigma: float,
    alpha: float,
) -> pd.DataFrame:
    topology_metric = f"mutual_knn_k{topology_k}"
    distance_metric = sigma_metric_name(local_rbf_sigma)
    subset = calibration_frame[
        calibration_frame["metric"].isin([topology_metric, distance_metric])
    ].copy()

    if subset.empty:
        return pd.DataFrame()

    identifier_columns = [
        "speech_encoder",
        "speaker",
        "emotion",
        "text_model",
        "aggregator",
        "n_samples",
    ]
    value_columns = [
        "calibrated_score",
        "permutation_p_value",
        "q_value_within_metric",
        "significant_fdr_within_metric",
        "observed_aggregate",
        "critical_threshold_tau",
    ]
    pieces = []

    for metric_name, prefix in [
        (topology_metric, "topology"),
        (distance_metric, "local_distance"),
    ]:
        metric_frame = subset[subset["metric"] == metric_name][
            identifier_columns + value_columns
        ].copy()
        metric_frame = metric_frame.rename(
            columns={column: f"{prefix}_{column}" for column in value_columns}
        )
        pieces.append(metric_frame)

    summary = pieces[0].merge(
        pieces[1],
        on=identifier_columns,
        how="outer",
        validate="one_to_one",
    )
    topology_significant = summary[
        "topology_significant_fdr_within_metric"
    ].fillna(False)
    distance_significant = summary[
        "local_distance_significant_fdr_within_metric"
    ].fillna(False)

    summary["aristotelian_pattern"] = np.select(
        [
            topology_significant & ~distance_significant,
            topology_significant & distance_significant,
            ~topology_significant & distance_significant,
        ],
        [
            "topological_only_supports_aristotelian_view",
            "topological_and_metric_alignment",
            "metric_only",
        ],
        default="neither_significant",
    )
    summary["topology_metric"] = topology_metric
    summary["local_distance_metric"] = distance_metric
    summary["alpha"] = alpha
    return summary


def mutual_knn_layer_matrix_only(
    speech_matrices: Dict[str, np.ndarray],
    text_matrices: Dict[str, np.ndarray],
    k: int,
) -> pd.DataFrame:
    speech_layers = sorted(speech_matrices, key=layer_sort_key)
    text_layers = sorted(text_matrices, key=layer_sort_key)
    number_of_samples = next(iter(speech_matrices.values())).shape[0]
    effective_k = effective_k_value(k, number_of_samples)
    speech_vectors = np.stack(
        [
            knn_adjacency_vector_from_indices(
                knn_indices(speech_matrices[layer], effective_k),
                number_of_samples,
            )
            for layer in speech_layers
        ]
    )
    text_vectors = np.stack(
        [
            knn_adjacency_vector_from_indices(
                knn_indices(text_matrices[layer], effective_k),
                number_of_samples,
            )
            for layer in text_layers
        ]
    )
    values = (speech_vectors @ text_vectors.T) / (
        number_of_samples * effective_k
    )
    return pd.DataFrame(values, index=speech_layers, columns=text_layers)


def run_gallery_size_sweep_for_pair(
    speech_layers: LayerCollection,
    text_layers: LayerCollection,
    common_ids: Sequence[str],
    metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    maximum_available = len(common_ids)
    sizes = sorted(
        set(
            min(int(size), maximum_available)
            for size in args.gallery_sizes
            if int(size) >= args.minimum_samples
        )
    )

    if not sizes:
        return pd.DataFrame()

    full_speech = subset_layer_collection(speech_layers, common_ids)
    full_text = subset_layer_collection(text_layers, common_ids)

    reference_pairs: Dict[Tuple[str, str], Dict[str, Any]] = {}
    reference_schemes: List[Tuple[str, str, int]] = []
    for fixed_k in args.gallery_fixed_k_values:
        reference_schemes.append(("fixed", f"k={fixed_k}", int(fixed_k)))
    for divisor in args.gallery_proportional_divisors:
        full_k = max(1, int(round(maximum_available / divisor)))
        reference_schemes.append(
            ("proportional", f"k=round(n/{divisor})", full_k)
        )

    for scheme, scheme_label, reference_k in reference_schemes:
        key = (scheme, scheme_label)
        if key in reference_pairs:
            continue
        matrix = mutual_knn_layer_matrix_only(
            full_speech,
            full_text,
            k=reference_k,
        )
        values = matrix.to_numpy(dtype=np.float64)
        flat_index = int(np.nanargmax(values))
        speech_index, text_index = np.unravel_index(flat_index, values.shape)
        reference_pairs[key] = {
            "speech_layer": str(matrix.index[speech_index]),
            "text_layer": str(matrix.columns[text_index]),
            "reference_k": effective_k_value(reference_k, maximum_available),
            "reference_score": float(values[speech_index, text_index]),
        }

    for gallery_size in sizes:
        repeats = 1 if gallery_size == maximum_available else args.gallery_repeats

        for repeat in range(repeats):
            subset_ids = deterministic_subsample(
                common_ids,
                maximum_samples=gallery_size,
                seed=stable_seed(
                    "gallery",
                    gallery_size,
                    repeat,
                    base_seed=args.seed,
                ),
            )
            speech_matrices = subset_layer_collection(speech_layers, subset_ids)
            text_matrices = subset_layer_collection(text_layers, subset_ids)

            schemes: List[Tuple[str, str, int]] = []
            for fixed_k in args.gallery_fixed_k_values:
                schemes.append(("fixed", f"k={fixed_k}", int(fixed_k)))
            for divisor in args.gallery_proportional_divisors:
                proportional_k = max(1, int(round(gallery_size / divisor)))
                schemes.append(
                    (
                        "proportional",
                        f"k=round(n/{divisor})",
                        proportional_k,
                    )
                )

            seen = set()
            for scheme, scheme_label, requested_k in schemes:
                key = (scheme, scheme_label, requested_k)
                if key in seen:
                    continue
                seen.add(key)
                matrix = mutual_knn_layer_matrix_only(
                    speech_matrices,
                    text_matrices,
                    k=requested_k,
                )
                values = matrix.to_numpy(dtype=np.float64)
                flat_index = int(np.nanargmax(values))
                speech_index, text_index = np.unravel_index(flat_index, values.shape)
                effective_k = effective_k_value(requested_k, gallery_size)
                chance = effective_k / max(gallery_size - 1, 1)
                raw_max = float(values[speech_index, text_index])
                adjusted_max = (raw_max - chance) / max(1.0 - chance, 1e-12)

                reference = reference_pairs[(scheme, scheme_label)]
                reference_score = float(
                    matrix.loc[
                        reference["speech_layer"],
                        reference["text_layer"],
                    ]
                )
                adjusted_reference = (
                    (reference_score - chance) / max(1.0 - chance, 1e-12)
                )

                row = dict(metadata)
                row.update(
                    {
                        "gallery_size": gallery_size,
                        "repeat": repeat,
                        "k_scheme": scheme,
                        "k_scheme_label": scheme_label,
                        "requested_k": requested_k,
                        "effective_k": effective_k,
                        "chance_baseline": chance,
                        "raw_max_mutual_knn": raw_max,
                        "chance_adjusted_max_mutual_knn": adjusted_max,
                        "raw_mean_mutual_knn": float(np.nanmean(values)),
                        "best_speech_layer_at_this_size": matrix.index[speech_index],
                        "best_text_layer_at_this_size": matrix.columns[text_index],
                        "reference_speech_layer": reference["speech_layer"],
                        "reference_text_layer": reference["text_layer"],
                        "reference_full_gallery_k": reference["reference_k"],
                        "reference_full_gallery_score": reference["reference_score"],
                        "fixed_reference_layer_score": reference_score,
                        "chance_adjusted_fixed_reference_score": adjusted_reference,
                        "n_layer_pairs": values.size,
                        "sample_id_hash": hashlib.sha256(
                            "||".join(subset_ids).encode("utf-8")
                        ).hexdigest()[:16],
                        "recommended_gallery_curve": "fixed_reference_layer_score",
                        "summary_status": (
                            "raw_gallery_robustness; fixed reference layers avoid repeated layer selection"
                        ),
                    }
                )
                rows.append(row)

    return pd.DataFrame(rows)


def same_layer_invariance(
    first_layers: LayerCollection,
    second_layers: LayerCollection,
    k_values: Sequence[int],
    cycle_k_values: Sequence[int],
    maximum_samples: int,
    minimum_samples: int,
    seed: int,
    metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    common_layer_names = sorted(
        set(first_layers).intersection(second_layers),
        key=layer_sort_key,
    )
    rows = []

    for layer_name in common_layer_names:
        first = {layer_name: first_layers[layer_name]}
        second = {layer_name: second_layers[layer_name]}
        common_ids = common_ids_across_collections(first, second)
        common_ids = deterministic_subsample(common_ids, maximum_samples, seed)

        if len(common_ids) < minimum_samples:
            continue

        first_matrix = subset_layer_collection(first, common_ids)[layer_name]
        second_matrix = subset_layer_collection(second, common_ids)[layer_name]

        row = dict(metadata)
        row.update(
            {
                "layer": layer_name,
                "n_samples": len(common_ids),
                "linear_cka": scalar_linear_cka(first_matrix, second_matrix),
                "rsa_spearman": scalar_rsa(first_matrix, second_matrix),
            }
        )

        for k in k_values:
            row[f"mutual_knn_k{k}"] = scalar_mutual_knn(
                first_matrix,
                second_matrix,
                k=k,
            )

        for k in cycle_k_values:
            row[f"cycle_knn_k{k}"] = scalar_cycle_knn(
                first_matrix,
                second_matrix,
                k=k,
            )

        rows.append(row)

    return pd.DataFrame(rows)


def detail_output_path(
    output_dir: Path,
    speech_encoder: str,
    speaker: str,
    emotion: str,
    text_model: str,
) -> Path:
    return (
        output_dir
        / "layerwise_alignment"
        / safe_name(speech_encoder)
        / safe_name(speaker)
        / safe_name(emotion)
        / f"{safe_name(text_model)}.csv.gz"
    )


def run_cross_modal_analysis(
    text_manifest: pd.DataFrame,
    speech_manifest: pd.DataFrame,
    pairs: Optional[pd.DataFrame],
    args: argparse.Namespace,
) -> None:
    include_regex = re.compile(args.include_key_regex) if args.include_key_regex else None
    exclude_regex = re.compile(args.exclude_key_regex) if args.exclude_key_regex else None

    text_cache: Dict[str, LayerCollection] = {}
    text_relatedness_rows = []
    text_group_rows = []
    text_models = sorted(text_manifest["model"].unique())

    print("\n" + "=" * 80)
    print("LOADING TEXT MODELS")
    print("=" * 80)

    for text_model in text_models:
        model_manifest = text_manifest[text_manifest["model"] == text_model]
        layers = load_feature_group(
            manifest_group=model_manifest,
            pooling=args.pooling,
            include_key_regex=include_regex,
            exclude_key_regex=exclude_regex,
            minimum_samples_per_layer=args.minimum_samples,
            description=f"text/{text_model}",
        )
        text_cache[text_model] = layers

        print(
            f"Loaded text model {text_model}: "
            f"{len(layers)} layers, "
            f"{min(len(layer.ids) for layer in layers.values()):,}-"
            f"{max(len(layer.ids) for layer in layers.values()):,} samples/layer"
        )

        if pairs is not None:
            correlations, groups = validate_relatedness_for_layers(
                layers=layers,
                pairs=pairs,
                metadata={
                    "modality": "text",
                    "model": text_model,
                    "speech_encoder": None,
                    "speaker": None,
                    "emotion": None,
                },
                minimum_pairs=args.minimum_pairs,
            )
            text_relatedness_rows.append(correlations)
            text_group_rows.append(groups)

    speech_groups = list(
        speech_manifest.groupby(
            ["speech_encoder", "speaker", "emotion"],
            sort=True,
        )
    )

    top_rows = []
    per_speech_layer_rows = []
    calibration_rows = []
    gallery_rows = []
    speech_relatedness_rows = []
    speech_group_rows = []
    failures = []
    accumulator = MatrixAccumulator()

    print("\n" + "=" * 80)
    print("CROSS-MODAL ALIGNMENT")
    print("=" * 80)
    print(f"Speech conditions: {len(speech_groups):,}")
    print(f"Text models:       {len(text_models):,}")

    outer_progress = tqdm(speech_groups, desc="Speech conditions", unit="condition")

    for condition, condition_manifest in outer_progress:
        speech_encoder, speaker, emotion = [str(value) for value in condition]
        provenance = summarize_source_tts_provenance(condition_manifest)
        condition_label = f"{speech_encoder}/{speaker}/{emotion}"
        outer_progress.set_postfix_str(condition_label[-80:])

        try:
            speech_layers = load_feature_group(
                manifest_group=condition_manifest,
                pooling=args.pooling,
                include_key_regex=include_regex,
                exclude_key_regex=exclude_regex,
                minimum_samples_per_layer=args.minimum_samples,
                description=f"speech/{condition_label}",
            )
        except Exception as error:
            failures.append(
                {
                    "stage": "load_speech_group",
                    "speech_encoder": speech_encoder,
                    "speaker": speaker,
                    "emotion": emotion,
                    "text_model": None,
                    **provenance,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        if pairs is not None:
            correlations, groups = validate_relatedness_for_layers(
                layers=speech_layers,
                pairs=pairs,
                metadata={
                    "modality": "speech",
                    "model": speech_encoder,
                    "speech_encoder": speech_encoder,
                    "speaker": speaker,
                    "emotion": emotion,
                    **provenance,
                },
                minimum_pairs=args.minimum_pairs,
            )
            speech_relatedness_rows.append(correlations)
            speech_group_rows.append(groups)

        for text_model, text_layers in text_cache.items():
            metadata = {
                "speech_encoder": speech_encoder,
                "speaker": speaker,
                "emotion": emotion,
                "text_model": text_model,
                **provenance,
            }

            try:
                common_ids = common_ids_across_collections(speech_layers, text_layers)
                common_ids = deterministic_subsample(
                    common_ids,
                    maximum_samples=args.global_max_samples,
                    seed=stable_seed(
                        speech_encoder,
                        speaker,
                        emotion,
                        text_model,
                        base_seed=args.seed,
                    ),
                )

                if len(common_ids) < args.minimum_samples:
                    raise RuntimeError(
                        f"Only {len(common_ids)} sentence IDs are shared across all "
                        f"speech and text layers; minimum is {args.minimum_samples}."
                    )

                speech_matrices = subset_layer_collection(speech_layers, common_ids)
                text_matrices = subset_layer_collection(text_layers, common_ids)

                matrices = compute_alignment_matrices(
                    speech_matrices=speech_matrices,
                    text_matrices=text_matrices,
                    k_values=args.k_values,
                    cycle_k_values=args.cycle_k_values,
                    cknna_k_values=args.cknna_k_values,
                    rbf_sigmas=args.rbf_sigmas,
                    rbf_preprocess=args.rbf_preprocess,
                    median_rbf_cka=args.median_rbf_cka,
                    cknna_unbiased=args.cknna_unbiased,
                    cknna_distance_agnostic=args.cknna_distance_agnostic,
                )

                long_frame = alignment_matrices_to_long(
                    matrices=matrices,
                    metadata=metadata,
                    number_of_samples=len(common_ids),
                )
                detail_path = detail_output_path(
                    args.output_dir,
                    speech_encoder,
                    speaker,
                    emotion,
                    text_model,
                )
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                long_frame.to_csv(detail_path, index=False, compression="gzip")

                top_frame = extract_top_layer_pairs(
                    matrices=matrices,
                    metadata=metadata,
                    number_of_samples=len(common_ids),
                )
                if not top_frame.empty:
                    top_rows.append(top_frame)

                per_layer_frame = best_text_layer_per_speech_layer(
                    matrices=matrices,
                    metadata=metadata,
                    number_of_samples=len(common_ids),
                )
                if not per_layer_frame.empty:
                    per_speech_layer_rows.append(per_layer_frame)

                for metric_name, matrix in matrices.items():
                    accumulator.add(
                        (speech_encoder, text_model, metric_name),
                        matrix,
                    )

                if args.permutations > 0:
                    calibrated = aggregation_aware_calibration(
                        speech_matrices=speech_matrices,
                        text_matrices=text_matrices,
                        observed_matrices=matrices,
                        metrics=args.calibration_metrics,
                        aggregators=args.calibration_aggregators,
                        permutations=args.permutations,
                        alpha=args.alpha,
                        seed=stable_seed(
                            "calibration",
                            speech_encoder,
                                speaker,
                            emotion,
                            text_model,
                            base_seed=args.seed,
                        ),
                        device=args.calibration_device,
                        metadata=metadata,
                        args=args,
                    )
                    if not calibrated.empty:
                        calibration_rows.append(calibrated)

                if args.run_gallery_sweep:
                    gallery = run_gallery_size_sweep_for_pair(
                        speech_layers=speech_layers,
                        text_layers=text_layers,
                        common_ids=common_ids,
                        metadata=metadata,
                        args=args,
                    )
                    if not gallery.empty:
                        gallery_rows.append(gallery)

            except Exception as error:
                failures.append(
                    {
                        "stage": "cross_modal_analysis",
                        **metadata,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                continue

        if top_rows:
            atomic_write_csv(
                pd.concat(top_rows, ignore_index=True),
                args.output_dir / "summaries" / "raw_top_layer_pairs.csv",
            )
        if failures:
            atomic_write_csv(
                pd.DataFrame(failures),
                args.output_dir / "diagnostics" / "analysis_failures.csv",
            )

    if top_rows:
        top_frame = pd.concat(top_rows, ignore_index=True)
        atomic_write_csv(
            top_frame,
            args.output_dir / "summaries" / "raw_top_layer_pairs.csv",
        )

        group_columns = ["speech_encoder", "text_model", "metric"]
        condition_summary = (
            top_frame.groupby(group_columns, as_index=False)
            .agg(
                mean_raw_best_score=("score", "mean"),
                std_raw_best_score=("score", "std"),
                median_raw_best_score=("score", "median"),
                minimum_raw_best_score=("score", "min"),
                maximum_raw_best_score=("score", "max"),
                number_of_conditions=("score", "count"),
                mean_n_samples=("n_samples", "mean"),
            )
        )
        condition_summary["warning"] = (
            "raw maxima are descriptive; use aggregation-aware calibrated results for inference"
        )
        atomic_write_csv(
            condition_summary,
            args.output_dir / "summaries" / "raw_condition_summary.csv",
        )

        by_emotion = (
            top_frame.groupby(
                ["speech_encoder", "text_model", "metric", "emotion"],
                as_index=False,
            )
            .agg(
                mean_raw_best_score=("score", "mean"),
                std_raw_best_score=("score", "std"),
                number_of_speakers=("speaker", "nunique"),
            )
        )
        atomic_write_csv(
            by_emotion,
            args.output_dir / "summaries" / "raw_summary_by_emotion.csv",
        )

        by_speaker = (
            top_frame.groupby(
                ["speech_encoder", "text_model", "metric", "speaker"],
                as_index=False,
            )
            .agg(
                mean_raw_best_score=("score", "mean"),
                std_raw_best_score=("score", "std"),
                number_of_emotions=("emotion", "nunique"),
            )
        )
        atomic_write_csv(
            by_speaker,
            args.output_dir / "summaries" / "raw_summary_by_speaker.csv",
        )

    if per_speech_layer_rows:
        save_table(
            pd.concat(per_speech_layer_rows, ignore_index=True),
            args.output_dir / "summaries" / "raw_best_text_layer_per_speech_layer",
        )

    if calibration_rows:
        calibration_frame = add_fdr_columns(
            pd.concat(calibration_rows, ignore_index=True),
            alpha=args.alpha,
        )
        save_table(
            calibration_frame,
            args.output_dir / "calibration" / "aggregation_aware_calibration",
        )
        atomic_write_csv(
            calibration_frame,
            args.output_dir / "calibration" / "aggregation_aware_calibration.csv",
        )

        calibrated_summary = (
            calibration_frame.groupby(
                [
                    "speech_encoder",
                    "text_model",
                    "metric",
                    "aggregator",
                ],
                as_index=False,
            )
            .agg(
                mean_calibrated_score=("calibrated_score", "mean"),
                std_calibrated_score=("calibrated_score", "std"),
                median_calibrated_score=("calibrated_score", "median"),
                proportion_significant_fdr=(
                    "significant_fdr_within_metric",
                    "mean",
                ),
                number_of_conditions=("calibrated_score", "count"),
                mean_observed_aggregate=("observed_aggregate", "mean"),
                mean_critical_threshold=("critical_threshold_tau", "mean"),
            )
        )
        atomic_write_csv(
            calibrated_summary,
            args.output_dir / "summaries" / "calibrated_condition_summary.csv",
        )

        aristotelian_summary = build_aristotelian_topology_metric_summary(
            calibration_frame,
            topology_k=args.topology_k,
            local_rbf_sigma=args.local_rbf_sigma,
            alpha=args.alpha,
        )
        if not aristotelian_summary.empty:
            atomic_write_csv(
                aristotelian_summary,
                args.output_dir
                / "summaries"
                / "aristotelian_topology_vs_local_distance.csv",
            )

    if gallery_rows:
        gallery_frame = pd.concat(gallery_rows, ignore_index=True)
        save_table(
            gallery_frame,
            args.output_dir / "gallery_sweep" / "gallery_size_results",
        )
        gallery_summary = (
            gallery_frame.groupby(
                [
                    "speech_encoder",
                    "text_model",
                    "gallery_size",
                    "k_scheme",
                    "k_scheme_label",
                    "effective_k",
                ],
                as_index=False,
            )
            .agg(
                mean_fixed_reference_layer_score=(
                    "fixed_reference_layer_score",
                    "mean",
                ),
                std_fixed_reference_layer_score=(
                    "fixed_reference_layer_score",
                    "std",
                ),
                mean_adjusted_fixed_reference_score=(
                    "chance_adjusted_fixed_reference_score",
                    "mean",
                ),
                mean_raw_max_mutual_knn=("raw_max_mutual_knn", "mean"),
                std_raw_max_mutual_knn=("raw_max_mutual_knn", "std"),
                mean_adjusted_max_mutual_knn=(
                    "chance_adjusted_max_mutual_knn",
                    "mean",
                ),
                repeats=("repeat", "nunique"),
                number_of_conditions=("raw_max_mutual_knn", "count"),
            )
        )
        atomic_write_csv(
            gallery_summary,
            args.output_dir / "gallery_sweep" / "gallery_size_summary.csv",
        )

    relatedness_frames = [
        frame
        for frame in text_relatedness_rows + speech_relatedness_rows
        if frame is not None and not frame.empty
    ]
    group_frames = [
        frame
        for frame in text_group_rows + speech_group_rows
        if frame is not None and not frame.empty
    ]

    if relatedness_frames:
        relatedness_frame = pd.concat(relatedness_frames, ignore_index=True)
        save_table(
            relatedness_frame,
            args.output_dir / "semantic_validation" / "sick_relatedness_correlations",
        )
        best_relatedness = (
            relatedness_frame.sort_values("spearman_relatedness", ascending=False)
            .groupby(
                [
                    "modality",
                    "model",
                    "speech_encoder",
                    "speaker",
                    "emotion",
                ],
                dropna=False,
                as_index=False,
            )
            .head(1)
            .reset_index(drop=True)
        )
        atomic_write_csv(
            best_relatedness,
            args.output_dir / "semantic_validation" / "best_sick_layer_per_condition.csv",
        )

    if group_frames:
        save_table(
            pd.concat(group_frames, ignore_index=True),
            args.output_dir / "semantic_validation" / "sick_group_similarity_statistics",
        )

    if failures:
        atomic_write_csv(
            pd.DataFrame(failures),
            args.output_dir / "diagnostics" / "analysis_failures.csv",
        )

    aggregate_index_rows = []

    for key, mean_matrix in accumulator.mean_items():
        speech_encoder, text_model, metric_name = key
        aggregate_dir = (
            args.output_dir
            / "aggregated_alignment"
            / safe_name(speech_encoder)
            / safe_name(text_model)
        )
        matrix_path = aggregate_dir / f"{safe_name(metric_name)}.csv"
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        mean_matrix.to_csv(matrix_path, index=True)

        values = mean_matrix.to_numpy(dtype=np.float64)
        if not np.isfinite(values).any():
            continue
        flat_index = int(np.nanargmax(values))
        speech_index, text_index = np.unravel_index(flat_index, values.shape)
        aggregate_index_rows.append(
            {
                "speech_encoder": speech_encoder,
                "text_model": text_model,
                "metric": metric_name,
                "best_speech_layer": mean_matrix.index[speech_index],
                "best_text_layer": mean_matrix.columns[text_index],
                "best_mean_raw_score": float(values[speech_index, text_index]),
                "matrix_path": str(matrix_path),
                "warning": "raw aggregate matrix; not aggregation-aware calibrated",
            }
        )

        if not args.no_plots and metric_name in args.plot_metrics:
            plot_path = (
                args.output_dir
                / "plots"
                / "aggregate_heatmaps"
                / safe_name(metric_name)
                / f"{safe_name(speech_encoder)}__{safe_name(text_model)}.png"
            )
            save_heatmap(
                matrix=mean_matrix,
                output_path=plot_path,
                title=(
                    f"{metric_name}: {speech_encoder} vs {text_model}\n"
                    "Curated best-available samples; raw average over speakers and emotions"
                ),
                value_label=metric_name,
            )

    if aggregate_index_rows:
        atomic_write_csv(
            pd.DataFrame(aggregate_index_rows),
            args.output_dir / "summaries" / "raw_aggregate_matrix_index.csv",
        )


def load_speech_groups_for_encoder(
    speech_encoder_manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[Tuple[str, str], LayerCollection]:
    include_regex = re.compile(args.include_key_regex) if args.include_key_regex else None
    exclude_regex = re.compile(args.exclude_key_regex) if args.exclude_key_regex else None
    cache: Dict[Tuple[str, str], LayerCollection] = {}

    grouped = speech_encoder_manifest.groupby(["speaker", "emotion"], sort=True)

    for condition, manifest_group in tqdm(
        grouped,
        total=speech_encoder_manifest.groupby(["speaker", "emotion"]).ngroups,
        desc="Loading curated invariance groups",
        unit="condition",
    ):
        speaker, emotion = [str(value) for value in condition]
        cache[(speaker, emotion)] = load_feature_group(
            manifest_group=manifest_group,
            pooling=args.pooling,
            include_key_regex=include_regex,
            exclude_key_regex=exclude_regex,
            minimum_samples_per_layer=args.minimum_samples,
            description=f"invariance/{speaker}/{emotion}",
        )

    return cache


def run_speech_invariance_analysis(
    speech_manifest: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    all_rows = []

    for speech_encoder, encoder_manifest in speech_manifest.groupby(
        "speech_encoder", sort=True
    ):
        print(f"\nRunning curated speech invariance analysis for {speech_encoder}")
        cache = load_speech_groups_for_encoder(encoder_manifest, args)
        manifest_by_condition = {
            (str(speaker), str(emotion)): group
            for (speaker, emotion), group in encoder_manifest.groupby(
                ["speaker", "emotion"], sort=True
            )
        }

        by_speaker: Dict[str, List[str]] = defaultdict(list)
        for speaker, emotion in cache:
            by_speaker[speaker].append(emotion)

        for speaker, emotions in by_speaker.items():
            for emotion_a, emotion_b in itertools.combinations(sorted(set(emotions)), 2):
                provenance_a = summarize_source_tts_provenance(
                    manifest_by_condition[(speaker, emotion_a)]
                )
                provenance_b = summarize_source_tts_provenance(
                    manifest_by_condition[(speaker, emotion_b)]
                )
                frame = same_layer_invariance(
                    first_layers=cache[(speaker, emotion_a)],
                    second_layers=cache[(speaker, emotion_b)],
                    k_values=args.k_values,
                    cycle_k_values=args.cycle_k_values,
                    maximum_samples=args.global_max_samples,
                    minimum_samples=args.minimum_samples,
                    seed=stable_seed(
                        speech_encoder,
                        "emotion",
                        speaker,
                        emotion_a,
                        emotion_b,
                        base_seed=args.seed,
                    ),
                    args=args,
                    metadata={
                        "speech_encoder": speech_encoder,
                        "contrast_type": "emotion",
                        "speaker_a": speaker,
                        "speaker_b": speaker,
                        "emotion_a": emotion_a,
                        "emotion_b": emotion_b,
                        "source_tts_distribution_a": provenance_a[
                            "source_tts_distribution"
                        ],
                        "source_tts_distribution_b": provenance_b[
                            "source_tts_distribution"
                        ],
                    },
                )
                if not frame.empty:
                    all_rows.append(frame)

        by_emotion: Dict[str, List[str]] = defaultdict(list)
        for speaker, emotion in cache:
            by_emotion[emotion].append(speaker)

        for emotion, speakers in by_emotion.items():
            for speaker_a, speaker_b in itertools.combinations(sorted(set(speakers)), 2):
                provenance_a = summarize_source_tts_provenance(
                    manifest_by_condition[(speaker_a, emotion)]
                )
                provenance_b = summarize_source_tts_provenance(
                    manifest_by_condition[(speaker_b, emotion)]
                )
                frame = same_layer_invariance(
                    first_layers=cache[(speaker_a, emotion)],
                    second_layers=cache[(speaker_b, emotion)],
                    k_values=args.k_values,
                    cycle_k_values=args.cycle_k_values,
                    maximum_samples=args.global_max_samples,
                    minimum_samples=args.minimum_samples,
                    seed=stable_seed(
                        speech_encoder,
                        "speaker",
                        emotion,
                        speaker_a,
                        speaker_b,
                        base_seed=args.seed,
                    ),
                    args=args,
                    metadata={
                        "speech_encoder": speech_encoder,
                        "contrast_type": "speaker",
                        "speaker_a": speaker_a,
                        "speaker_b": speaker_b,
                        "emotion_a": emotion,
                        "emotion_b": emotion,
                        "source_tts_distribution_a": provenance_a[
                            "source_tts_distribution"
                        ],
                        "source_tts_distribution_b": provenance_b[
                            "source_tts_distribution"
                        ],
                    },
                )
                if not frame.empty:
                    all_rows.append(frame)

    if not all_rows:
        warnings.warn("No curated speech invariance comparisons could be computed.")
        return

    invariance = pd.concat(all_rows, ignore_index=True)
    save_table(
        invariance,
        args.output_dir / "speech_invariance" / "same_layer_invariance",
    )

    metric_columns = [
        column
        for column in invariance.columns
        if column in {"linear_cka", "rsa_spearman"}
        or column.startswith("mutual_knn_k")
        or column.startswith("cycle_knn_k")
    ]

    summary_rows = []
    for metric in metric_columns:
        summary = (
            invariance.groupby(
                ["speech_encoder", "contrast_type", "layer"],
                as_index=False,
            )[metric]
            .agg(["mean", "std", "median", "count"])
            .reset_index()
        )
        summary["metric"] = metric
        summary_rows.append(summary)

    save_table(
        pd.concat(summary_rows, ignore_index=True),
        args.output_dir / "speech_invariance" / "invariance_summary",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Layer-wise Platonic/Aristotelian alignment analysis between "
            "speech and text representations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    parser.add_argument("--speech-root", type=Path, default=DEFAULT_SPEECH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=None,
        help=(
            "Optional sick_selected_pairs.csv containing sentence_a_id, "
            "sentence_b_id, relatedness_score, and preferably similarity_group."
        ),
    )
    parser.add_argument(
        "--best-samples-csv",
        type=Path,
        default=DEFAULT_BEST_SAMPLES_CSV,
        help=(
            "best_available_emotion_samples.csv selecting one source TTS audio "
            "for every sentence × speaker × emotion."
        ),
    )
    parser.add_argument(
        "--best-samples-emotion-column",
        default="orig_emotion",
        help="Column interpreted as the target emotion in the curated CSV.",
    )
    parser.add_argument(
        "--best-samples-duplicate-policy",
        choices=["error", "first", "best_score"],
        default="error",
        help="How duplicate sentence × speaker × emotion selections are handled.",
    )
    parser.add_argument(
        "--strict-best-samples-coverage",
        action="store_true",
        help=(
            "Fail when a curated sample has no extracted feature for any speech "
            "encoder. Otherwise save the missing-coverage audit and continue."
        ),
    )

    parser.add_argument(
        "--sentence-id-regex",
        default=DEFAULT_SENTENCE_ID_REGEX,
        help="Regular expression used to extract the sentence ID from feature paths.",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(SUPPORTED_EXTENSIONS)),
        help="Comma-separated feature-file extensions.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "max", "first", "cls", "flatten"],
        default="mean",
        help=(
            "How each real layer's token/time axis T is reduced to one vector. "
            "For [L,T,d] tensors, L is always preserved. Mean pooling is "
            "recommended; flatten can create variable widths when T varies."
        ),
    )
    parser.add_argument(
        "--include-key-regex",
        default=None,
        help="Optional regex. Only tensor keys matching it are loaded.",
    )
    parser.add_argument(
        "--exclude-key-regex",
        default=DEFAULT_EXCLUDE_KEY_REGEX,
        help="Regex for non-representation tensors to skip.",
    )

    parser.add_argument("--text-models", type=parse_csv_values, default=None)
    parser.add_argument("--speech-encoders", type=parse_csv_values, default=None)
    parser.add_argument(
        "--source-tts-models",
        type=parse_csv_values,
        default=None,
        help=(
            "Optional provenance/debug filter applied before the curated join. "
            "Source TTS is never used as an analysis variable."
        ),
    )
    parser.add_argument("--speakers", type=parse_csv_values, default=None)
    parser.add_argument("--emotions", type=parse_csv_values, default=None)

    parser.add_argument(
        "--k-values",
        type=parse_int_csv,
        default=[10, 20, 50, 100],
        help="Neighborhood sizes for mutual-kNN.",
    )
    parser.add_argument(
        "--cycle-k-values",
        type=parse_int_csv,
        default=[10],
        help="Neighborhood sizes for cycle-kNN.",
    )
    parser.add_argument(
        "--cknna-k-values",
        type=parse_int_csv,
        default=[10],
        help="Neighborhood sizes for CKNNA. CKNNA is more expensive than mKNN.",
    )
    parser.add_argument(
        "--cknna-unbiased",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the original PRH unbiased-HSIC CKNNA formulation.",
    )
    parser.add_argument(
        "--cknna-distance-agnostic",
        action="store_true",
        help="Ignore Gram values inside shared neighborhoods. Mainly a diagnostic.",
    )
    parser.add_argument(
        "--rbf-sigmas",
        type=parse_float_csv,
        default=[0.1, 0.5, 2.0, 5.0],
        help="Fixed RBF bandwidths for locality analysis.",
    )
    parser.add_argument(
        "--rbf-preprocess",
        choices=["l2", "center", "zscore", "none"],
        default="l2",
        help="Common preprocessing applied before fixed-bandwidth RBF kernels.",
    )
    parser.add_argument(
        "--median-rbf-cka",
        action="store_true",
        help="Also compute a median-distance RBF CKA diagnostic.",
    )

    parser.add_argument(
        "--global-max-samples",
        type=int,
        default=600,
        help="Maximum shared sentences per comparison; 0 uses all.",
    )
    parser.add_argument("--minimum-samples", type=int, default=50)
    parser.add_argument("--minimum-pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help=(
            "Number of permutations for the aggregation-aware calibration "
            "calibration. Zero disables calibration."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--calibration-metrics",
        type=parse_csv_values,
        default=[
            "linear_cka",
            "mutual_knn_k10",
            "cycle_knn_k10",
            "cknna_k10",
            "rbf_cka_sigma_0p1",
        ],
        help="Metrics passed to official aggregation-aware calibration.",
    )
    parser.add_argument(
        "--calibration-aggregators",
        type=parse_csv_values,
        default=["max"],
        help="Layer-matrix aggregate(s): max and/or mean.",
    )
    parser.add_argument(
        "--calibration-device",
        default="cpu",
        help="Torch device used by calibrated-similarity, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--calibration-num-threads",
        type=int,
        default=1,
        help=(
            "Torch CPU threads during permutation calibration. Small matrix "
            "operations are often faster with one thread. Use None only by editing the script."
        ),
    )
    parser.add_argument(
        "--topology-k",
        type=int,
        default=10,
        help="mKNN k used in the Aristotelian topology-versus-distance summary.",
    )
    parser.add_argument(
        "--local-rbf-sigma",
        type=float,
        default=0.1,
        help="Small RBF sigma used as the exact-local-distance comparison.",
    )

    parser.add_argument(
        "--run-gallery-sweep",
        action="store_true",
        help="Run the gallery-size sensitivity analysis.",
    )
    parser.add_argument(
        "--gallery-sizes",
        type=parse_int_csv,
        default=[64, 128, 256, 512, 600],
    )
    parser.add_argument("--gallery-repeats", type=int, default=10)
    parser.add_argument(
        "--gallery-fixed-k-values",
        type=parse_int_csv,
        default=[10],
    )
    parser.add_argument(
        "--gallery-proportional-divisors",
        type=parse_int_csv,
        default=[100],
        help="For each divisor d, also evaluate k=round(n/d).",
    )

    parser.add_argument(
        "--run-speech-invariance",
        action="store_true",
        help="Compare curated emotion conditions at fixed speaker and speaker conditions at fixed emotion.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not create aggregate heatmaps.",
    )
    parser.add_argument(
        "--plot-metrics",
        type=parse_csv_values,
        default=[
            "linear_cka",
            "mutual_knn_k10",
            "cknna_k10",
            "rbf_cka_sigma_0p1",
        ],
        help="Raw aggregate metrics for which heatmaps are generated.",
    )

    return parser


def normalize_arguments(args: argparse.Namespace) -> argparse.Namespace:
    args.output_dir = args.output_dir.resolve()
    args.text_root = args.text_root.resolve()
    args.speech_root = args.speech_root.resolve()
    args.pairs_csv = args.pairs_csv.resolve() if args.pairs_csv is not None else None
    args.best_samples_csv = args.best_samples_csv.resolve()

    extensions = set()
    for extension in args.extensions.split(","):
        extension = extension.strip().lower()
        if not extension:
            continue
        extensions.add(extension if extension.startswith(".") else f".{extension}")
    args.extensions = extensions

    if args.permutations < 0:
        raise ValueError("--permutations cannot be negative.")
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.global_max_samples < 0:
        raise ValueError("--global-max-samples cannot be negative.")
    if args.minimum_samples < 4:
        raise ValueError("--minimum-samples must be at least 4.")
    if args.gallery_repeats <= 0:
        raise ValueError("--gallery-repeats must be positive.")
    if args.calibration_num_threads is not None and args.calibration_num_threads <= 0:
        raise ValueError("--calibration-num-threads must be positive.")

    for name in [
        "k_values",
        "cycle_k_values",
        "cknna_k_values",
        "gallery_sizes",
        "gallery_fixed_k_values",
        "gallery_proportional_divisors",
    ]:
        values = getattr(args, name)
        if any(value <= 0 for value in values):
            raise ValueError(f"Every value in --{name.replace('_', '-')} must be positive.")

    if any(value < 2 for value in args.cknna_k_values):
        raise ValueError("Every CKNNA k must be at least 2.")
    if any(value <= 0 for value in args.rbf_sigmas):
        raise ValueError("Every RBF sigma must be positive.")
    if args.topology_k not in args.k_values:
        warnings.warn(
            f"Adding topology k={args.topology_k} to --k-values so the summary can be built."
        )
        args.k_values = sorted(set(args.k_values + [args.topology_k]))
    if args.local_rbf_sigma not in args.rbf_sigmas:
        warnings.warn(
            f"Adding local RBF sigma={args.local_rbf_sigma} to --rbf-sigmas."
        )
        args.rbf_sigmas = sorted(set(args.rbf_sigmas + [args.local_rbf_sigma]))

    invalid_aggregators = set(args.calibration_aggregators) - {"max", "mean"}
    if invalid_aggregators:
        raise ValueError(
            f"Unsupported calibration aggregators: {sorted(invalid_aggregators)}"
        )

    if args.permutations > 0:
        import_official_calibration()

    return args


def main() -> None:
    parser = build_argument_parser()
    args = normalize_arguments(parser.parse_args())

    validate_directory(args.text_root, "Text feature root")
    validate_directory(args.speech_root, "Speech feature root")
    if not args.best_samples_csv.exists():
        raise FileNotFoundError(
            f"Best-available sample CSV does not exist:\n{args.best_samples_csv}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sentence_id_regex = re.compile(args.sentence_id_regex)

    print("=" * 80)
    print("CURATED SPEECH–TEXT PLATONIC / ARISTOTELIAN ALIGNMENT ANALYSIS")
    print("=" * 80)
    print(f"Text root:        {args.text_root}")
    print(f"Speech root:      {args.speech_root}")
    print(f"Best samples CSV: {args.best_samples_csv}")
    print(f"Pairs CSV:        {args.pairs_csv}")
    print(f"Output:           {args.output_dir}")
    print(f"Pooling over T:   {args.pooling}")
    print("Layer handling:   [L,T,d] is split along L before pooling")
    print(f"mKNN k values:    {args.k_values}")
    print(f"Cycle k:          {args.cycle_k_values}")
    print(f"CKNNA k:          {args.cknna_k_values}")
    print(f"RBF sigmas:       {args.rbf_sigmas}")
    print(f"Permutations:     {args.permutations}")
    print("Source TTS role:  provenance only; not an analysis factor")

    text_manifest = discover_text_manifest(
        root=args.text_root,
        sentence_id_regex=sentence_id_regex,
        extensions=args.extensions,
    )
    discovered_speech_manifest = discover_speech_manifest(
        root=args.speech_root,
        sentence_id_regex=sentence_id_regex,
        extensions=args.extensions,
    )

    text_manifest, discovered_speech_manifest = apply_manifest_filters(
        text_manifest=text_manifest,
        speech_manifest=discovered_speech_manifest,
        args=args,
    )

    curated_index = load_best_samples_index(
        path=args.best_samples_csv,
        sentence_id_regex=sentence_id_regex,
        emotion_column=args.best_samples_emotion_column,
        duplicate_policy=args.best_samples_duplicate_policy,
    )
    speech_manifest, coverage = apply_curated_sample_selection(
        speech_manifest=discovered_speech_manifest,
        curated_index=curated_index,
        strict_coverage=args.strict_best_samples_coverage,
    )

    manifests_dir = args.output_dir / "manifests"
    atomic_write_csv(text_manifest, manifests_dir / "text_manifest.csv")
    atomic_write_csv(
        discovered_speech_manifest,
        manifests_dir / "speech_manifest_before_curated_selection.csv",
    )
    atomic_write_csv(curated_index, manifests_dir / "best_samples_index_normalized.csv")
    atomic_write_csv(speech_manifest, manifests_dir / "speech_manifest_curated.csv")
    atomic_write_csv(coverage, manifests_dir / "curated_feature_coverage.csv")
    save_source_tts_audits(speech_manifest, args.output_dir)

    unique_curated_conditions = speech_manifest[
        ["speech_encoder", "speaker", "emotion", "sentence_id"]
    ].drop_duplicates()
    inventory_summary = {
        "text_models": sorted(text_manifest["model"].unique().tolist()),
        "speech_encoders": sorted(speech_manifest["speech_encoder"].unique().tolist()),
        "source_tts_models_used_as_provenance": sorted(
            speech_manifest["source_tts_model"].unique().tolist()
        ),
        "speakers": sorted(speech_manifest["speaker"].astype(str).unique().tolist()),
        "emotions": sorted(speech_manifest["emotion"].unique().tolist()),
        "number_of_curated_csv_rows": int(len(curated_index)),
        "number_of_text_feature_files": int(len(text_manifest)),
        "number_of_discovered_speech_feature_files": int(
            len(discovered_speech_manifest)
        ),
        "number_of_selected_speech_feature_files": int(len(speech_manifest)),
        "number_of_text_sentences": int(text_manifest["sentence_id"].nunique()),
        "number_of_selected_speech_sentences": int(
            speech_manifest["sentence_id"].nunique()
        ),
        "number_of_selected_sentence_conditions": int(
            len(unique_curated_conditions)
        ),
        "speech_conditions": int(
            speech_manifest.groupby(
                ["speech_encoder", "speaker", "emotion"]
            ).ngroups
        ),
        "tts_is_analysis_variable": False,
        "tts_is_provenance": True,
        "missing_curated_feature_conditions": int((~coverage["matched"]).sum()),
    }
    write_json(inventory_summary, manifests_dir / "inventory_summary.json")

    configuration = {
        key: (
            str(value)
            if isinstance(value, Path)
            else sorted(value)
            if isinstance(value, set)
            else value
        )
        for key, value in vars(args).items()
    }
    write_json(configuration, args.output_dir / "analysis_configuration.json")

    print("\nInventory:")
    print(json.dumps(inventory_summary, indent=2, ensure_ascii=False))

    pairs = load_sick_pairs(args.pairs_csv)

    run_cross_modal_analysis(
        text_manifest=text_manifest,
        speech_manifest=speech_manifest,
        pairs=pairs,
        args=args,
    )

    if args.run_speech_invariance:
        run_speech_invariance_analysis(
            speech_manifest=speech_manifest,
            args=args,
        )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}")
    print("\nMost useful outputs:")
    print("  manifests/speech_manifest_curated.csv")
    print("  manifests/source_tts_provenance_by_condition.csv")
    print("  calibration/aggregation_aware_calibration.csv")
    print("  summaries/calibrated_condition_summary.csv")
    print("  summaries/aristotelian_topology_vs_local_distance.csv")
    print("  summaries/raw_top_layer_pairs.csv")
    print("  speech_invariance/invariance_summary.*")
    print("  gallery_sweep/gallery_size_summary.csv")
    print("  aggregated_alignment/")
    print("  plots/aggregate_heatmaps/")
    print("  semantic_validation/sick_relatedness_correlations.*")
    print("  diagnostics/analysis_failures.csv")


if __name__ == "__main__":
    main()
