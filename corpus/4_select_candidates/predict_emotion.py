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

    emotion_scores = {labels[i].split("/")[-1]: scores[i] for i in range(len(scores))}

    return emotion_label, emotion_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts-dir", required=True, help="Output directory of one TTS system, e.g. TTS-Outputs/IndexTTS2")
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = args.tts_dir
    tts_model = os.path.basename(base_dir)
    filepaths = glob.glob(os.path.join(base_dir, "**", "*.wav"), recursive=True)

    print(f"Searching for .wav files in {base_dir}...")

    assert len(filepaths) > 0, f"No .wav files found in {base_dir}"

    print(f"Found {len(filepaths):,} .wav files in {base_dir}")

    model_id = "iic/emotion2vec_plus_large"
    model = AutoModel(
        model=model_id,
        hub="hf",
    )

    rows = []

    for idx, filepath in tqdm(enumerate(filepaths), total=len(filepaths), desc="Running emotion2vec+"):
        filename = os.path.basename(filepath)
        orig_emotion = filepath.split("/")[-2]
        speaker_id = filepath.split("/")[-3]

        print(f"Processing file {idx + 1}/{len(filepaths)}: {filename} (Speaker: {speaker_id}, Original Emotion: {orig_emotion})")

        emotion_label, emotion_scores = run_emotion2vec(model, filepath)

        emotion_scores = {k.replace("<unk>", "unknown"): v for k, v in emotion_scores.items()}

        print(f"Predicted Emotion: {emotion_label}, Scores: {emotion_scores}")

        rows.append({
            "tts_model": tts_model,
            "filepath": filepath,
            "filename": filename,
            "speaker_id": speaker_id,
            "orig_emotion": orig_emotion,
            "pred_emotion": emotion_label,
            **emotion_scores
        })

    df = pd.DataFrame(rows)
    output_csv_path = os.path.join(base_dir, "emotion2vec_predictions.csv")
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


if __name__ == "__main__":
    main()
