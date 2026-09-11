import argparse
import os
import random
from typing import Tuple

import torch
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from omnivoice import OmniVoice


torch.backends.cudnn.enabled = False


def to_numpy_audio(wav):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()

    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 2 and wav.shape[0] <= 8 and wav.shape[0] < wav.shape[1]:
        wav = wav.T

    return wav


def generate_omnivoice(
    model: OmniVoice,
    text: str,
    ref_audio: str,
    ref_text: str,
    output_path: str,
) -> Tuple[torch.Tensor, int]:
    wav = model.generate(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

    return wav[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-csv", required=True, help="sick_unique_sentences.csv produced by corpus/1_select_sentences/filter_sick.py")
    parser.add_argument("--prompts-csv", required=True, help="filtered_ravdess_prompts_metadata.csv produced by corpus/2_select_prompts/build_prompt_map.py")
    parser.add_argument("--output-dir", required=True, help="Directory that will receive <speaker>/<emotion>/<sentence_id>.wav")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
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

    tts = OmniVoice.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.float16
    )

    df = pd.read_csv(metadata_filepath)
    prompts_df = pd.read_csv(prompts_map_filepath)
    sr = 24000

    print(df)
    print(prompts_df)

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

            wav = generate_omnivoice(
                model=tts,
                text=target_text,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                output_path=output_filepath
            )

            sf.write(output_filepath, to_numpy_audio(wav), sr)


if __name__ == "__main__":
    main()
