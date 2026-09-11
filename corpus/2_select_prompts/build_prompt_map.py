import argparse
import os
import glob
import random
from typing import Dict, Any
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from funasr import AutoModel

torch.backends.cudnn.enabled = False


modality_mapping = {
    "01": "full-AV",
    "02": "video-only",
    "03": "audio-only"
}

vocal_channel_mapping = {
    "01": "speech",
    "02": "song"
}

emotion_mapping = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

emotion_intensity_mapping = {
    "01": "normal",
    "02": "strong"
}

statement_mapping = {
    "01": "kids are talking by the door",
    "02": "dogs are sitting by the door"
}

repetition_mapping = {
    "01": "first",
    "02": "second"
}


def build_ravdess_metadata(ravdess_base_dir: str):
    filepaths = glob.glob(os.path.join(ravdess_base_dir, "**", "*.wav"), recursive=True)

    assert len(filepaths) > 0, f"No .wav files found in {ravdess_base_dir}"

    rows = []

    for filepath in tqdm(filepaths, desc="Processing RAVDESS files"):
        filename = os.path.basename(filepath)[:-4]
        parts = filename.split("-")
        modality = parts[0]
        vocal_channel = parts[1]
        emotion = parts[2]
        emotion_intensity = parts[3]
        statement = parts[4]
        repetition = parts[5]
        actor = int(parts[6])
        gender = "female" if actor % 2 == 0 else "male"

        rows.append({
            "filepath": filepath,
            "filename": filename,
            "modality": modality_mapping[modality],
            "vocal_channel": vocal_channel_mapping[vocal_channel],
            "emotion": emotion_mapping[emotion],
            "emotion_intensity": emotion_intensity_mapping[emotion_intensity],
            "statement": statement_mapping[statement],
            "repetition": repetition_mapping[repetition],
            "actor": actor,
            "gender": gender
        })

    df = pd.DataFrame(rows)

    return df


def run_emotion2vec(
    model: Any,
    wav_path: Path,
) -> Dict[str, Any]:
    rec_result = model.generate(
        str(wav_path),
        output_dir=None,
        granularity="utterance",
        extract_embedding=False,
        disable_pbar=True,
    )

    res = rec_result[0] if isinstance(rec_result, list) else rec_result

    scores = res["scores"]
    labels = res["labels"]

    emotion_id = max(range(len(scores)), key=lambda i: scores[i])
    emotion_label = labels[emotion_id].split("/")[-1]
    emotion_score = scores[emotion_id]

    res["pred_emotion_id"] = emotion_id
    res["pred_emotion"] = emotion_label
    res["pred_emotion_score"] = emotion_score

    return emotion_label, emotion_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ravdess-dir", required=True, help="Root of the RAVDESS speech recordings (Actor_XX folders)")
    parser.add_argument("--output-csv", default="filtered_ravdess_prompts_metadata.csv")
    parser.add_argument("--seed", type=int, default=84)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    ravdess_base_dir = args.ravdess_dir

    df = build_ravdess_metadata(ravdess_base_dir)

    model_id = "iic/emotion2vec_plus_large"
    model = AutoModel(
        model=model_id,
        hub="hf",
    )

    print(df)

    random.seed(seed)

    male_actors = df[df["gender"] == "male"]["actor"].unique()
    female_actors = df[df["gender"] == "female"]["actor"].unique()

    selected_male_actors = random.sample(list(male_actors), 3)
    selected_female_actors = random.sample(list(female_actors), 3)

    selected_actors = selected_male_actors + selected_female_actors

    filtered_df = df[df["actor"].isin(selected_actors)]

    print(f"Filtered metadata rows: {len(filtered_df):,}")

    filtered_df = filtered_df[(filtered_df["emotion"].isin(["neutral", "sad", "happy", "angry"]))]

    print(filtered_df["emotion"].value_counts())

    print(f"Filtered metadata rows after emotion filtering: {len(filtered_df):,}")

    print(filtered_df.groupby(["actor", "emotion"]).size().unstack(fill_value=0))

    filtered_df = filtered_df[filtered_df["statement"] == "dogs are sitting by the door"]

    print(f"Filtered metadata rows after statement filtering: {len(filtered_df):,}")

    print(filtered_df.groupby(["actor", "emotion"]).size().unstack(fill_value=0))

    new_rows = []

    for idx, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc="Running emotion2vec+"):
        wav_path = Path(row["filepath"])
        filename = row["filename"]
        orig_emotion = row["emotion"]

        emotion_label, emotion_score = run_emotion2vec(model, wav_path)

        if emotion_label==orig_emotion:
            new_rows.append({
                "filepath": wav_path,
                "filename": filename,
                "orig_emotion": orig_emotion,
                "pred_emotion": emotion_label,
                "pred_emotion_score": emotion_score,
                "modality": row["modality"],
                "vocal_channel": row["vocal_channel"],
                "emotion_intensity": row["emotion_intensity"],
                "statement": row["statement"],
                "repetition": row["repetition"],
                "actor": row["actor"],
                "gender": row["gender"]
            })

        print(f"File: {wav_path}, Original Emotion: {orig_emotion}, Predicted Emotion: {emotion_label}, Score: {emotion_score}")

    final_rows = []

    for actor in selected_actors:
        for emotion in ["neutral", "sad", "happy", "angry"]:
            samples = [row for row in new_rows if row["actor"] == actor and row["orig_emotion"] == emotion]

            if not samples:
                continue

            samples.sort(key=lambda x: (x["emotion_intensity"] == "strong", x["pred_emotion_score"]), reverse=True)

            final_rows.append(samples[0])

    final_df = pd.DataFrame(final_rows)

    print(f"Final filtered metadata rows: {len(final_df):,}")

    print(final_df)

    final_df.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
