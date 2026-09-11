import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


WER_METRICS = [
    "parakeet_v3_wer",
    "qwen3_asr_wer",
    "whisper_v3_wer",
]

CER_METRICS = [
    "parakeet_v3_cer",
    "qwen3_asr_cer",
    "whisper_v3_cer",
]

ASR_METRICS = [
    "parakeet_v3_wer",
    "parakeet_v3_cer",
    "qwen3_asr_wer",
    "qwen3_asr_cer",
    "whisper_v3_wer",
    "whisper_v3_cer",
]

TTS_MODEL_PRIORITY = {
    "OmniVoice": 0,
    "VoxCPM2": 1,
    "IndexTTS2": 2,
    "Qwen3-TTS": 3,
}


def create_sample_id(row: pd.Series) -> str:
    """Create a normalized sample ID from filename or filepath."""
    filename = row.get("filename")

    if pd.notna(filename) and str(filename).strip():
        source = str(filename)
    else:
        source = os.path.basename(str(row.get("filepath", "")))

    return os.path.splitext(source)[0]


def load_metadata(
    base_dir: str,
    model_dir: str,
    model_name: str,
    emotion_metadata: str,
    asr_metadata: str,
    asr_metrics: list[str],
) -> pd.DataFrame:
    """Load and merge emotion and ASR metadata for one TTS system."""
    emotion_path = os.path.join(
        base_dir,
        model_dir,
        emotion_metadata,
    )

    asr_path = os.path.join(
        base_dir,
        model_dir,
        asr_metadata,
    )

    emotion_df = pd.read_csv(emotion_path)
    asr_df = pd.read_csv(asr_path)

    required_emotion_columns = {
        "filepath",
        "filename",
        "speaker_id",
        "orig_emotion",
        "pred_emotion",
    }

    required_asr_columns = {
        "filepath",
        *asr_metrics,
    }

    missing_emotion_columns = (
        required_emotion_columns - set(emotion_df.columns)
    )

    missing_asr_columns = (
        required_asr_columns - set(asr_df.columns)
    )

    if missing_emotion_columns:
        raise ValueError(
            f"{model_name} emotion metadata is missing columns: "
            f"{sorted(missing_emotion_columns)}"
        )

    if missing_asr_columns:
        raise ValueError(
            f"{model_name} ASR metadata is missing columns: "
            f"{sorted(missing_asr_columns)}"
        )

    emotion_df["tts_model"] = model_name

    emotion_df["correct"] = (
        emotion_df["pred_emotion"]
        == emotion_df["orig_emotion"]
    )

    merged_df = emotion_df.merge(
        asr_df[["filepath", *asr_metrics]],
        on="filepath",
        how="left",
        validate="many_to_one",
    )

    return merged_df


def print_original_emotion_results(
    model_dataframes: dict[str, pd.DataFrame],
) -> None:
    """Print emotion results before sample selection."""
    print("\n========== ORIGINAL EMOTION RESULTS ==========")

    for model_name, dataframe in model_dataframes.items():
        print(
            f"{model_name} Emotion Accuracy: "
            f"{dataframe['correct'].mean():.6f}"
        )

    for model_name, dataframe in model_dataframes.items():
        accuracy_per_emotion = (
            dataframe.groupby("orig_emotion")["correct"]
            .mean()
            .sort_index()
        )

        print(
            f"\n{model_name} Emotion Accuracy per Emotion:\n",
            accuracy_per_emotion,
        )


def print_original_asr_results(
    model_dataframes: dict[str, pd.DataFrame],
) -> None:
    """Print ASR results before sample selection."""
    print("\n========== ORIGINAL ASR RESULTS ==========")

    for model_name, dataframe in model_dataframes.items():
        wer_avg = (
            dataframe[WER_METRICS]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
            .mean()
        )

        cer_avg = (
            dataframe[CER_METRICS]
            .apply(pd.to_numeric, errors="coerce")
            .mean(axis=1)
            .mean()
        )

        print(
            f"\n{model_name} ASR:"
            f"\n  Parakeet-v3 WER: "
            f"{dataframe['parakeet_v3_wer'].mean():.6f}"
            f" | CER: "
            f"{dataframe['parakeet_v3_cer'].mean():.6f}"
            f"\n  Qwen3-ASR WER:   "
            f"{dataframe['qwen3_asr_wer'].mean():.6f}"
            f" | CER: "
            f"{dataframe['qwen3_asr_cer'].mean():.6f}"
            f"\n  Whisper-v3 WER:  "
            f"{dataframe['whisper_v3_wer'].mean():.6f}"
            f" | CER: "
            f"{dataframe['whisper_v3_cer'].mean():.6f}"
            f"\n  Average WER:     "
            f"{wer_avg:.6f}"
            f"\n  Average CER:     "
            f"{cer_avg:.6f}"
        )


def select_best_samples(
    combined_df: pd.DataFrame,
) -> pd.DataFrame:
    selected_df = combined_df.copy()

    selected_df["sample_id"] = selected_df.apply(
        create_sample_id,
        axis=1,
    )

    emotion_score_columns = set(selected_df.columns)

    def get_target_emotion_score(row: pd.Series) -> float:
        target_emotion = row["orig_emotion"]

        if target_emotion not in emotion_score_columns:
            return np.nan

        return pd.to_numeric(
            row[target_emotion],
            errors="coerce",
        )

    selected_df["target_emotion_score"] = selected_df.apply(
        get_target_emotion_score,
        axis=1,
    )

    wer_values = selected_df[WER_METRICS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    cer_values = selected_df[CER_METRICS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    selected_df["wer_avg"] = wer_values.mean(
        axis=1,
        skipna=True,
    )

    selected_df["cer_avg"] = cer_values.mean(
        axis=1,
        skipna=True,
    )

    selected_df["tts_priority"] = (
        selected_df["tts_model"]
        .map(TTS_MODEL_PRIORITY)
        .fillna(len(TTS_MODEL_PRIORITY))
        .astype(int)
    )

    sample_group_columns = [
        "speaker_id",
        "orig_emotion",
        "sample_id",
    ]

    expected_sample_groups = (
        selected_df[sample_group_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    selected_df = selected_df.sort_values(
        by=[
            *sample_group_columns,
            "correct",
            "target_emotion_score",
            "wer_avg",
            "cer_avg",
            "tts_priority",
            "tts_model",
            "filepath",
        ],
        ascending=[
            True,
            True,
            True,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
        ],
        na_position="last",
        kind="stable",
    )

    selected_df = selected_df.drop_duplicates(
        subset=sample_group_columns,
        keep="first",
    )

    selected_df = selected_df.sort_values(
        by=sample_group_columns,
        kind="stable",
    ).reset_index(drop=True)

    selected_sample_groups = (
        selected_df[sample_group_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if len(selected_sample_groups) != len(expected_sample_groups):
        raise RuntimeError(
            "Some sample groups were removed during selection. "
            f"Expected {len(expected_sample_groups)} groups, "
            f"but retained {len(selected_sample_groups)}."
        )

    selected_df = selected_df.drop(
        columns=["tts_priority"],
    )

    return selected_df


def reorder_columns(
    selected_df: pd.DataFrame,
) -> pd.DataFrame:
    """Place the most important output columns first."""
    preferred_columns = [
        "tts_model",
        "filepath",
        "filename",
        "sample_id",
        "speaker_id",
        "orig_emotion",
        "pred_emotion",
        "correct",
        "target_emotion_score",
        "wer_avg",
        "cer_avg",
        "angry",
        "disgusted",
        "fearful",
        "happy",
        "neutral",
        "other",
        "sad",
        "surprised",
        "unknown",
        "parakeet_v3_wer",
        "parakeet_v3_cer",
        "qwen3_asr_wer",
        "qwen3_asr_cer",
        "whisper_v3_wer",
        "whisper_v3_cer",
    ]

    existing_preferred_columns = [
        column
        for column in preferred_columns
        if column in selected_df.columns
    ]

    remaining_columns = [
        column
        for column in selected_df.columns
        if column not in existing_preferred_columns
    ]

    return selected_df[
        existing_preferred_columns + remaining_columns
    ]


def print_selected_results(
    selected_df: pd.DataFrame,
    asr_metrics: list[str],
    total_candidate_rows: int,
    total_sample_groups: int,
) -> None:
    """Print statistics for the final selected metadata."""
    print("\n========== SELECTED METADATA RESULTS ==========")

    print(f"\nNumber of candidate TTS rows: {total_candidate_rows}")
    print(f"Number of unique sample groups: {total_sample_groups}")
    print(f"Number of selected samples: {len(selected_df)}")

    selected_emotion_accuracy = selected_df["correct"].mean()

    print(
        f"\nSelected Emotion Accuracy: "
        f"{selected_emotion_accuracy:.6f}"
    )

    selected_accuracy_per_emotion = (
        selected_df.groupby("orig_emotion")["correct"]
        .mean()
        .sort_index()
    )

    print(
        "\nSelected Emotion Accuracy per Emotion:\n",
        selected_accuracy_per_emotion,
    )

    print("\nSelected Aggregated ASR Metrics:")

    print(
        f"  Average WER: "
        f"{selected_df['wer_avg'].mean():.6f}"
    )

    print(
        f"  Average CER: "
        f"{selected_df['cer_avg'].mean():.6f}"
    )

    print("\nSelected Individual ASR Metrics:")

    for metric in asr_metrics:
        print(
            f"  {metric}: "
            f"{selected_df[metric].mean():.6f}"
        )

    print("\nAggregated ASR Metrics by TTS Model:")

    aggregated_asr_by_model = (
        selected_df.groupby("tts_model")[
            ["wer_avg", "cer_avg"]
        ]
        .mean()
        .sort_index()
    )

    print(aggregated_asr_by_model)

    print("\nIndividual ASR Metrics by TTS Model:")

    selected_asr_by_model = (
        selected_df.groupby("tts_model")[asr_metrics]
        .mean()
        .sort_index()
    )

    print(selected_asr_by_model)

    print("\nAggregated ASR Metrics by Emotion:")

    aggregated_asr_by_emotion = (
        selected_df.groupby("orig_emotion")[
            ["wer_avg", "cer_avg"]
        ]
        .mean()
        .sort_index()
    )

    print(aggregated_asr_by_emotion)

    print("\nIndividual ASR Metrics by Emotion:")

    selected_asr_by_emotion = (
        selected_df.groupby("orig_emotion")[asr_metrics]
        .mean()
        .sort_index()
    )

    print(selected_asr_by_emotion)

    print("\nSelected samples per TTS model:")

    selected_per_model = (
        selected_df["tts_model"]
        .value_counts()
        .sort_index()
    )

    print(selected_per_model)

    print("\nSelected samples per emotion:")

    selected_per_emotion = (
        selected_df["orig_emotion"]
        .value_counts()
        .sort_index()
    )

    print(selected_per_emotion)

    print("\nCorrect and incorrect selected samples:")

    correctness_counts = (
        selected_df["correct"]
        .value_counts(dropna=False)
        .rename(
            index={
                True: "correct",
                False: "incorrect",
            }
        )
    )

    print(correctness_counts)

    incorrect_count = int(
        (~selected_df["correct"]).sum()
    )

    print(
        "\nIncorrect samples retained because no correct output "
        f"was available or ranked first: {incorrect_count}"
    )

    print(
        "\nFinal TTS tie-break priority: "
        "OmniVoice -> VoxCPM2 -> IndexTTS2 -> Qwen3-TTS"
    )


def save_confusion_matrix(
    selected_df: pd.DataFrame,
    base_dir: str,
) -> tuple[str, str]:
    required_columns = {
        "orig_emotion",
        "pred_emotion",
    }

    missing_columns = required_columns - set(selected_df.columns)

    if missing_columns:
        raise ValueError(
            "Cannot create the confusion matrix because these columns "
            f"are missing: {sorted(missing_columns)}"
        )

    confusion_df = selected_df[
        ["orig_emotion", "pred_emotion"]
    ].dropna()

    if confusion_df.empty:
        raise ValueError(
            "Cannot create the confusion matrix because there are no "
            "rows with both original and predicted emotions."
        )

    true_labels = confusion_df["orig_emotion"].astype(str)
    predicted_labels = confusion_df["pred_emotion"].astype(str)

    labels = sorted(
        set(true_labels).union(set(predicted_labels))
    )

    matrix = confusion_matrix(
        y_true=true_labels,
        y_pred=predicted_labels,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=pd.Index(
            labels,
            name="Original emotion",
        ),
        columns=pd.Index(
            labels,
            name="Predicted emotion",
        ),
    )

    matrix_csv_path = os.path.join(
        base_dir,
        "selected_metadata_confusion_matrix.csv",
    )

    matrix_png_path = os.path.join(
        base_dir,
        "selected_metadata_confusion_matrix.png",
    )

    matrix_df.to_csv(matrix_csv_path)

    figure_size = max(
        8.0,
        len(labels) * 1.2,
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_size,
            figure_size,
        )
    )

    sns.heatmap(
        matrix_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar=True,
        ax=axis,
    )

    axis.set_title(
        "Emotion Confusion Matrix — Selected Metadata",
        pad=16,
    )

    axis.set_xlabel("Predicted Emotion")
    axis.set_ylabel("Original Emotion")

    axis.tick_params(
        axis="x",
        rotation=45,
    )

    axis.tick_params(
        axis="y",
        rotation=0,
    )

    figure.tight_layout()

    figure.savefig(
        matrix_png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return matrix_csv_path, matrix_png_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-outputs-dir", required=True, help="Directory containing one sub-directory per TTS system (IndexTTS2, OmniVoice, Qwen3-TTS, VoxCPM2), each with emotion2vec_predictions.csv and asr_transcriptions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.tts_outputs_dir

    model_directories = {
        "IndexTTS2": "IndexTTS2",
        "OmniVoice": "OmniVoice",
        "Qwen3-TTS": "Qwen3-TTS",
        "VoxCPM2": "VoxCPM2",
    }

    emotion_metadata = "emotion2vec_predictions.csv"
    asr_metadata = "asr_transcriptions.csv"

    model_dataframes = {}

    for model_name, model_directory in model_directories.items():
        model_dataframes[model_name] = load_metadata(
            base_dir=base_dir,
            model_dir=model_directory,
            model_name=model_name,
            emotion_metadata=emotion_metadata,
            asr_metadata=asr_metadata,
            asr_metrics=ASR_METRICS,
        )

    print_original_emotion_results(model_dataframes)
    print_original_asr_results(model_dataframes)

    combined_df = pd.concat(
        model_dataframes.values(),
        ignore_index=True,
        sort=False,
    )

    combined_df["sample_id"] = combined_df.apply(
        create_sample_id,
        axis=1,
    )

    sample_group_columns = [
        "speaker_id",
        "orig_emotion",
        "sample_id",
    ]

    total_sample_groups = len(
        combined_df[
            sample_group_columns
        ].drop_duplicates()
    )

    selected_df = select_best_samples(combined_df)
    selected_df = reorder_columns(selected_df)

    print_selected_results(
        selected_df=selected_df,
        asr_metrics=ASR_METRICS,
        total_candidate_rows=len(combined_df),
        total_sample_groups=total_sample_groups,
    )

    print("\nBest available TTS system for each sample:")

    print_columns = [
        "tts_model",
        "filepath",
        "filename",
        "sample_id",
        "speaker_id",
        "orig_emotion",
        "pred_emotion",
        "correct",
        "target_emotion_score",
        "wer_avg",
        "cer_avg",
        "parakeet_v3_wer",
        "parakeet_v3_cer",
        "qwen3_asr_wer",
        "qwen3_asr_cer",
        "whisper_v3_wer",
        "whisper_v3_cer",
    ]

    existing_print_columns = [
        column
        for column in print_columns
        if column in selected_df.columns
    ]


    output_path = os.path.join(
        base_dir,
        "best_available_emotion_samples.csv",
    )

    selected_df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved selected rows to: {output_path}")

    confusion_csv_path, confusion_png_path = save_confusion_matrix(
        selected_df=selected_df,
        base_dir=base_dir,
    )

    print(
        "\nSaved confusion-matrix counts to: "
        f"{confusion_csv_path}"
    )

    print(
        "Saved Seaborn confusion-matrix image to: "
        f"{confusion_png_path}"
    )


if __name__ == "__main__":
    main()
