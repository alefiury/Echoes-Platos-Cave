import argparse
import os
import random
from typing import Tuple

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import soundfile as sf

from voxcpm import VoxCPM

torch.backends.cudnn.enabled = False


def to_numpy_audio(wav):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()

    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 2 and wav.shape[0] <= 8 and wav.shape[0] < wav.shape[1]:
        wav = wav.T

    return wav


def infer_audio_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    return ext if ext else "wav"


def generate_nanovoxcpm(
    model: VoxCPM,
    text: str,
    ref_audio: str,
    ref_text: str,
) -> Tuple[torch.Tensor, int]:
    with open(ref_audio, "rb") as f:
        ref_audio_bytes = f.read()

    wav_format = infer_audio_format(ref_audio)

    ref_audio_latents = model.encode_latents(ref_audio_bytes, wav_format)
    chunks = []
    for chunk in model.generate(
        target_text=text,
        ref_audio_latents=ref_audio_latents,
        max_generate_length=600,
        temperature=0.7,
        cfg_value=2.0,
    ):
        chunks.append(chunk)

    if len(chunks) == 0:
        raise RuntimeError("VoxCPM2 generated no audio chunks.")

    wav = np.concatenate(chunks, axis=0).astype(np.float32)

    return wav


def generate_voxcpm2(
    model: VoxCPM,
    text: str,
    ref_audio: str,
    ref_text: str,
) -> Tuple[torch.Tensor, int]:
    wav = model.generate(
        text=text,
        prompt_wav_path=ref_audio,
        prompt_text=ref_text,
        reference_wav_path=ref_audio,
        cfg_value=2.0,
        inference_timesteps=10,
        min_len=2,
        max_len=600,
    )
    return wav


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-csv", required=True, help="sick_unique_sentences.csv produced by corpus/1_select_sentences/filter_sick.py")
    parser.add_argument("--prompts-csv", required=True, help="filtered_ravdess_prompts_metadata.csv produced by corpus/2_select_prompts/build_prompt_map.py")
    parser.add_argument("--output-dir", required=True, help="Directory that will receive <speaker>/<emotion>/<sentence_id>.wav")
    parser.add_argument("--model", default="openbmb/VoxCPM2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    output_dir = args.output_dir
    model_name = args.model
    prompts_map_filepath = args.prompts_csv
    metadata_filepath = args.sentences_csv

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tts = VoxCPM.from_pretrained(
        model_name,
        load_denoiser=False,
    )

    df = pd.read_csv(metadata_filepath)
    prompts_df = pd.read_csv(prompts_map_filepath)

    print(df)
    print(prompts_df)

    sr = int(tts.tts_model.sample_rate)
    print(f"[INFO] VoxCPM2 sample_rate: {sr}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        target_text = row["text"]
        sentence_id = row["sentence_id"]

        for jdx, prompt_row in prompts_df.iterrows():
            ref_audio_path = prompt_row["filepath"]
            ref_text = prompt_row["statement"]
            emotion = prompt_row["orig_emotion"]
            speaker = prompt_row["actor"]

            output_filepath = os.path.join(
                output_dir,
                str(speaker),
                emotion,
                f"{sentence_id}.wav"
            )

            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

            wav = generate_voxcpm2(
                model=tts,
                text=target_text,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
            )

            sf.write(output_filepath, to_numpy_audio(wav), sr)


if __name__ == "__main__":
    main()
