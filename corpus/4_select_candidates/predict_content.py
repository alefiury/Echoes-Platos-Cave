from __future__ import annotations

import os
from pathlib import Path

os.environ.pop("HF_TOKEN", None)
os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

cache_root = (
    os.environ.get("SLURM_TMPDIR")
    or os.environ.get("TMPDIR")
    or f"/tmp/{os.getuid()}"
)

numba_cache_dir = Path(cache_root) / "numba-cache"
numba_cache_dir.mkdir(parents=True, exist_ok=True)

os.environ["NUMBA_CACHE_DIR"] = str(numba_cache_dir)


import argparse
import gc
import glob
import unicodedata

import pandas as pd
import librosa
import torch
from jiwer import cer, wer
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoProcessor,
    AutoModelForMultimodalLM,
)

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("--tts-dir", required=True, help="Output directory of one TTS system, e.g. TTS-Outputs/IndexTTS2")
parser.add_argument("--sentences-csv", required=True, help="sick_unique_sentences.csv produced by corpus/1_select_sentences/filter_sick.py")
parser.add_argument("--parakeet-model", default="nvidia/parakeet-tdt-0.6b-v3")
parser.add_argument("--qwen-model", default="Qwen/Qwen3-ASR-1.7B-hf")
parser.add_argument("--whisper-model", default="openai/whisper-large-v3")
ARGS = parser.parse_args()

TEXT_METADATA_PATH = Path(ARGS.sentences_csv)
BASE_DIR = Path(ARGS.tts_dir)
OUTPUT_CSV = BASE_DIR / "asr_transcriptions.csv"

PARAKEET_MODEL_ID = ARGS.parakeet_model
QWEN_MODEL_ID = ARGS.qwen_model
WHISPER_MODEL_ID = ARGS.whisper_model


SAVE_EVERY = 10
LANGUAGE = "English"

ASR_TEXT_COLUMNS = (
    "parakeet_v3",
    "qwen3_asr",
    "whisper_v3",
)

if torch.cuda.is_available():
    PIPELINE_DEVICE = 0
    QWEN_DEVICE = "cuda:0"

    if torch.cuda.is_bf16_supported():
        TORCH_DTYPE = torch.bfloat16
    else:
        TORCH_DTYPE = torch.float16
else:
    PIPELINE_DEVICE = -1
    QWEN_DEVICE = "cpu"
    TORCH_DTYPE = torch.float32


def save_dataframe(df):
    """Save current transcription and error-rate results."""
    df.to_csv(OUTPUT_CSV, index=False)


def clear_memory():
    """Release CPU and GPU memory."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_empty(value):
    """Return True when a dataframe value contains no usable text."""
    if value is None or pd.isna(value):
        return True

    return str(value).strip() == ""


def normalize_text_for_metrics(text):
    text = unicodedata.normalize(
        "NFKC",
        str(text),
    ).lower()

    text = "".join(
        " " if unicodedata.category(character).startswith("P")
        else character
        for character in text
    )

    return " ".join(text.split())


def compute_error_rates(reference, hypothesis):
    """Compute normalized utterance-level WER and CER."""
    normalized_reference = normalize_text_for_metrics(reference)
    normalized_hypothesis = normalize_text_for_metrics(hypothesis)

    if not normalized_reference:
        return "", ""

    return (
        float(wer(normalized_reference, normalized_hypothesis)),
        float(cer(normalized_reference, normalized_hypothesis)),
    )


def update_row_metrics(df, index, transcription_column):
    """Calculate WER and CER for one transcription in one dataframe row."""
    reference = df.at[index, "original_text"]
    hypothesis = df.at[index, transcription_column]

    wer_column = f"{transcription_column}_wer"
    cer_column = f"{transcription_column}_cer"

    if is_empty(reference) or is_empty(hypothesis):
        df.at[index, wer_column] = pd.NA
        df.at[index, cer_column] = pd.NA
        return

    row_wer, row_cer = compute_error_rates(
        reference,
        hypothesis,
    )

    df.at[index, wer_column] = row_wer
    df.at[index, cer_column] = row_cer


def ensure_metric_columns(df):
    for transcription_column in ASR_TEXT_COLUMNS:
        for metric in ("wer", "cer"):
            metric_column = f"{transcription_column}_{metric}"

            if metric_column not in df.columns:
                df[metric_column] = pd.Series(
                    pd.NA,
                    index=df.index,
                    dtype="Float64",
                )
            else:
                df[metric_column] = pd.to_numeric(
                    df[metric_column],
                    errors="coerce",
                ).astype("Float64")


def backfill_missing_metrics(df):
    """Calculate metrics for existing transcriptions when resuming."""
    ensure_metric_columns(df)

    updated_rows = 0

    for index in df.index:
        for transcription_column in ASR_TEXT_COLUMNS:
            hypothesis = df.at[index, transcription_column]

            if is_empty(hypothesis):
                continue

            wer_column = f"{transcription_column}_wer"
            cer_column = f"{transcription_column}_cer"

            if (
                is_empty(df.at[index, wer_column])
                or is_empty(df.at[index, cer_column])
            ):
                update_row_metrics(
                    df,
                    index,
                    transcription_column,
                )
                updated_rows += 1

    if updated_rows:
        print(
            f"Calculated WER/CER for "
            f"{updated_rows:,} existing transcriptions."
        )


def build_dataframe():
    """Build the initial metadata dataframe."""
    text_df = pd.read_csv(
        TEXT_METADATA_PATH,
        dtype={"sentence_id": str},
    )

    sentence_id_to_text = dict(
        zip(
            text_df["sentence_id"],
            text_df["text"],
        )
    )

    filepaths = sorted(
        glob.glob(
            os.path.join(
                BASE_DIR,
                "**",
                "*.wav",
            ),
            recursive=True,
        )
    )

    print(f"Found {len(filepaths):,} WAV files in {BASE_DIR}")

    if not filepaths:
        raise FileNotFoundError(
            f"No WAV files found in {BASE_DIR}"
        )

    rows = []

    for filepath in filepaths:
        path = Path(filepath)
        filename = path.stem

        if filename not in sentence_id_to_text:
            raise KeyError(
                f"Filename {filename!r} does not have a corresponding "
                f"sentence_id in {TEXT_METADATA_PATH}"
            )

        rows.append(
            {
                "tts_model": BASE_DIR.name,
                "filepath": str(path),
                "filename": filename,
                "speaker_id": path.parents[1].name,
                "original_text": sentence_id_to_text[filename],

                "parakeet_v3": "",
                "parakeet_v3_wer": "",
                "parakeet_v3_cer": "",
                "parakeet_v3_error": "",

                "qwen3_asr": "",
                "qwen3_asr_wer": "",
                "qwen3_asr_cer": "",
                "qwen3_asr_language": "",
                "qwen3_asr_error": "",

                "whisper_v3": "",
                "whisper_v3_wer": "",
                "whisper_v3_cer": "",
                "whisper_v3_error": "",
            }
        )

    df = pd.DataFrame(rows)
    ensure_metric_columns(df)
    save_dataframe(df)

    return df


def load_dataframe():
    if OUTPUT_CSV.exists():
        print(f"Resuming from {OUTPUT_CSV}")

        df = pd.read_csv(
            OUTPUT_CSV,
            dtype=str,
            keep_default_na=False,
        )

        ensure_metric_columns(df)
        backfill_missing_metrics(df)
        save_dataframe(df)

        return df

    return build_dataframe()


def transcribe_parakeet(df):
    """Run NVIDIA Parakeet transcription."""
    print("\nLoading Parakeet...")

    model = pipeline(
        task="automatic-speech-recognition",
        model=PARAKEET_MODEL_ID,
        device=PIPELINE_DEVICE,
        torch_dtype=TORCH_DTYPE,
    )

    pending_indices = df.index[
        df["parakeet_v3"].fillna("").eq("")
    ].tolist()

    for count, index in enumerate(
        tqdm(
            pending_indices,
            desc="Parakeet",
        ),
        start=1,
    ):
        filepath = df.at[index, "filepath"]

        output = model(filepath)

        hypothesis = output["text"].strip()

        df.at[index, "parakeet_v3"] = hypothesis

        update_row_metrics(
            df,
            index,
            "parakeet_v3",
        )


        if count % SAVE_EVERY == 0:
            save_dataframe(df)

    save_dataframe(df)

    del model
    clear_memory()


def transcribe_qwen(df):
    """Run Qwen3-ASR transcription."""
    print("\nLoading Qwen3-ASR...")

    processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_ID
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        QWEN_MODEL_ID,
        dtype=TORCH_DTYPE,
        device_map="auto",
    ).eval()

    pending_indices = df.index[
        df["qwen3_asr"].fillna("").eq("")
    ].tolist()

    for count, index in enumerate(
        tqdm(
            pending_indices,
            desc="Qwen3-ASR",
        ),
        start=1,
    ):
        filepath = df.at[index, "filepath"]

        sampling_rate = processor.feature_extractor.sampling_rate

        audio, _ = librosa.load(
            filepath,
            sr=sampling_rate,
            mono=True,
            dtype="float32",
        )

        inputs = processor.apply_transcription_request(
            audio=audio,
            language="English",
        ).to(
            model.device,
            model.dtype,
        )

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated_ids = output_ids[
            :,
            inputs["input_ids"].shape[1]:,
        ]

        parsed = processor.decode(
            generated_ids,
            return_format="parsed",
        )[0]

        df.at[index, "qwen3_asr"] = (
            parsed["transcription"].strip()
        )

        df.at[index, "qwen3_asr_language"] = (
            parsed["language"] or ""
        )

        df.at[index, "qwen3_asr_error"] = ""


        update_row_metrics(
            df,
            index,
            "qwen3_asr",
        )

        if count % SAVE_EVERY == 0:
            save_dataframe(df)

    save_dataframe(df)

    del model
    del processor
    clear_memory()


def transcribe_whisper(df):
    """Run Whisper large-v3 transcription."""
    print("\nLoading Whisper large-v3...")

    model = pipeline(
        task="automatic-speech-recognition",
        model=WHISPER_MODEL_ID,
        device=PIPELINE_DEVICE,
        torch_dtype=TORCH_DTYPE,
    )

    pending_indices = df.index[
        df["whisper_v3"].fillna("").eq("")
    ].tolist()

    for count, index in enumerate(
        tqdm(
            pending_indices,
            desc="Whisper large-v3",
        ),
        start=1,
    ):
        filepath = df.at[index, "filepath"]

        output = model(
            filepath,
            generate_kwargs={
                "language": "english",
                "task": "transcribe",
            },
        )

        hypothesis = output["text"].strip()

        df.at[index, "whisper_v3"] = hypothesis
        df.at[index, "whisper_v3_error"] = ""

        update_row_metrics(
            df,
            index,
            "whisper_v3",
        )

        if count % SAVE_EVERY == 0:
            save_dataframe(df)

    save_dataframe(df)

    del model
    clear_memory()


def main():
    print(f"Device: {QWEN_DEVICE}")
    print(f"Dtype: {TORCH_DTYPE}")

    df = load_dataframe()

    transcribe_parakeet(df)
    transcribe_qwen(df)
    transcribe_whisper(df)

    backfill_missing_metrics(df)
    save_dataframe(df)

    print("\nTranscription and metric calculation completed.")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
