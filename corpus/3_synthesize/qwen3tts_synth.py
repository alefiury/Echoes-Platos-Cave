#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import argparse
import os
import random
import re
import sys
import shutil
from typing import Tuple
import unicodedata
from typing import Callable, List, Optional

import jiwer
import torchaudio
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from qwen_tts import Qwen3TTSModel
from faster_qwen3_tts import FasterQwen3TTS

from nemo.utils import logging as nemo_logging
from nemo.collections.asr.models import EncDecMultiTaskModel

nemo_logging.setLevel("ERROR")
torch.backends.cudnn.enabled = False


def to_numpy_audio(wav):
    if isinstance(wav, torch.Tensor):
        wav = wav.detach().float().cpu().numpy()

    wav = np.asarray(wav, dtype=np.float32)

    if wav.ndim == 2 and wav.shape[0] <= 8 and wav.shape[0] < wav.shape[1]:
        wav = wav.T

    return wav


def generate_qwen_batch(
    model: FasterQwen3TTS,
    text: str,
    ref_audio: str,
    ref_text: str,
    language: str
) -> Tuple[torch.Tensor, int]:
    wavs = model.generate_voice_clone(
        text=text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        language=language,
        do_sample=True,
        temperature=0.9,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1.05,
    )

    wav, sr = wavs
    return wav, sr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences-csv", required=True, help="sick_unique_sentences.csv produced by corpus/1_select_sentences/filter_sick.py")
    parser.add_argument("--prompts-csv", required=True, help="filtered_ravdess_prompts_metadata.csv produced by corpus/2_select_prompts/build_prompt_map.py")
    parser.add_argument("--output-dir", required=True, help="Directory that will receive <speaker>/<emotion>/<sentence_id>.wav")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--language", default="English")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed = args.seed
    output_dir = args.output_dir
    model_name = args.model
    language = args.language
    prompts_map_filepath = args.prompts_csv
    metadata_filepath = args.sentences_csv

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tts = FasterQwen3TTS.from_pretrained(model_name)

    df = pd.read_csv(metadata_filepath)
    prompts_df = pd.read_csv(prompts_map_filepath)

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

            wav, sr = generate_qwen_batch(
                model=tts,
                text=target_text,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
                language=language,
            )

            sf.write(output_filepath, to_numpy_audio(wav), sr)


if __name__ == "__main__":
    main()
