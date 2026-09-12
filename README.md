# Echoes in Plato's Cave

**Measuring Global and Local Alignment Between Speech and Language Representations**
SALMA 2026, co-located with EMNLP 2026

[![Paper](https://img.shields.io/badge/paper-coming%20soon-lightgrey)](#citation)
[![Corpus](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-Echos--Platos--Cave-yellow)](https://huggingface.co/datasets/alefiury/Echos-Platos-Cave)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue)](LICENSE)

---

## Overview

Do speech encoders and text encoders converge on the same representational geometry? We test this on a
controlled corpus where **the same 600 sentences are rendered by 6 speakers x 4 emotions**, so lexical
content is held fixed while acoustic variation is manipulated. We compare **every layer pair** between
**9 speech encoders** and **10 text encoders**, and calibrate the strongest match with an
aggregation-aware permutation procedure.

**What we find**

- **Local neighborhood agreement survives calibration.** Mutual-kNN style metrics show reliable alignment,
  strongest between **late speech layers** and **early text layers**.
- **Global similarity does not.** CKA, RSA and RBF-CKA give no reliable evidence of a shared geometric space
  once the aggregation bias of "best layer pair" selection is accounted for.

**Resources**

| | |
|---|---|
| Paper | *link to be added* |
| Corpus | [`alefiury/Echos-Platos-Cave`](https://huggingface.co/datasets/alefiury/Echos-Platos-Cave) — 14,400 selected utterances plus all 57,600 candidates, with emotion and ASR scores |

## Contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start: use the released corpus](#quick-start-use-the-released-corpus)
- [Building the corpus from scratch](#building-the-corpus-from-scratch)
- [Feature extraction](#feature-extraction)
- [Analysis](#analysis)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Repository layout

```
corpus/                       Steps 1-4 build the corpus (already released on the Hub)
  1_select_sentences/           select the 600-sentence SICK subset
  2_select_prompts/             select 24 RAVDESS reference prompts (6 speakers x 4 emotions)
  3_synthesize/                 synthesize every sentence under every prompt with four TTS systems
  4_select_candidates/          score candidates (emotion2vec+, three ASR systems), keep one per cell
features/                     extract every-layer hidden states from speech and text encoders
analysis/                     alignment metrics, permutation calibration, auxiliary analyses
```

> [!TIP]
> To reproduce only the analysis, skip `corpus/` entirely: download the corpus from the Hub and start at
> [Feature extraction](#feature-extraction).

## Installation

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

`requirements.txt` covers **feature extraction and analysis**. The corpus-building scripts additionally
depend on the individual TTS and evaluation systems, each with its own installation procedure:

| Step | Package | Source |
|---|---|---|
| Prompt and candidate scoring | `funasr` (emotion2vec+ large) | [modelscope/FunASR](https://github.com/modelscope/FunASR) |
| ASR scoring | `nemo_toolkit[asr]` | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| | `transformers` | [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3) |
| Synthesis | `indextts` | [index-tts/index-tts](https://github.com/index-tts/index-tts) |
| | `omnivoice` | [k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice) |
| | `qwen_tts`, `faster_qwen3_tts` | [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) |
| | `voxcpm` | [OpenBMB/VoxCPM](https://github.com/OpenBMB/VoxCPM) |
| Calibration | `calibrated-similarity` | [mlbio-epfl/Aristotelian](https://github.com/mlbio-epfl/Aristotelian) |
| Prosody | `swift-f0`, `praat-parselmouth` | [SwiftF0](https://github.com/lars76/swift-f0/), [Praat-Parselmouth](https://github.com/YannickJadoul/Parselmouth) |

## Quick start: use the released corpus

```python
from datasets import load_dataset

ds    = load_dataset("alefiury/Echos-Platos-Cave", "filtered",       split="train")  # 14,400 selected utterances
cands = load_dataset("alefiury/Echos-Platos-Cave", "all_candidates", split="train")  # 57,600 candidates
```

The scripts below expect the audio **on disk** in the layout

```
TTS-Outputs/<tts_model>/<speaker_id>/<emotion>/<sentence_id>.wav
```

which is the `relative_path` column of the dataset. The two metadata tables used by the analysis —
`best_available_emotion_samples.csv` (the selection table) and `sick_selected_pairs.csv` — ship in the
dataset's `metadata/` folder.

Then continue with [Feature extraction](#feature-extraction) → [Analysis](#analysis).

## Building the corpus from scratch

<details>
<summary><b>Step 1 — Sentences</b>: select 600 SICK sentences across three relatedness bands</summary>

```bash
cd corpus/1_select_sentences
python plot_sick_distribution.py        # optional, inspects the relatedness distribution
python filter_sick.py                   # writes sick_filtered_subset/
```

`filter_sick.py` pools the SICK splits, removes pairs with fewer than 4 or more than 30 words, questions,
exclamations, URLs and HTML, splits the rest into **low** (<= 2.3), **medium** (3.0-3.7) and **high**
(>= 4.2) relatedness bands, and samples 100 sentence-disjoint pairs per band.

Outputs: `sick_unique_sentences.csv` (600 sentences) and `sick_selected_pairs.csv` (300 pairs).

</details>

<details>
<summary><b>Step 2 — Reference prompts</b>: pick 24 RAVDESS recordings (6 speakers x 4 emotions)</summary>

```bash
python corpus/2_select_prompts/build_prompt_map.py \
    --ravdess-dir /path/to/RAVDESS \
    --output-csv filtered_ravdess_prompts_metadata.csv
```

Randomly selects three male and three female actors, keeps neutral / happy / sad / angry recordings of the
statement *"Dogs are sitting by the door"*, discards recordings whose emotion2vec+ prediction disagrees with
the RAVDESS label, and retains one recording per speaker and emotion by intensity, then classifier confidence.

</details>

<details>
<summary><b>Step 3 — Synthesis</b>: 600 sentences x 24 prompts x 4 TTS systems</summary>

Each script renders all 600 sentences under all 24 prompts and writes
`<output-dir>/<speaker>/<emotion>/<sentence_id>.wav`.

```bash
python corpus/3_synthesize/indextts2_synth.py \
    --sentences-csv sick_unique_sentences.csv \
    --prompts-csv filtered_ravdess_prompts_metadata.csv \
    --output-dir TTS-Outputs/IndexTTS2 \
    --checkpoints-dir /path/to/indextts2/checkpoints

python corpus/3_synthesize/omnivoice_synth.py \
    --sentences-csv sick_unique_sentences.csv \
    --prompts-csv filtered_ravdess_prompts_metadata.csv \
    --output-dir TTS-Outputs/OmniVoice

python corpus/3_synthesize/qwen3tts_synth.py \
    --sentences-csv sick_unique_sentences.csv \
    --prompts-csv filtered_ravdess_prompts_metadata.csv \
    --output-dir TTS-Outputs/Qwen3-TTS

python corpus/3_synthesize/voxcpm2_synth.py \
    --sentences-csv sick_unique_sentences.csv \
    --prompts-csv filtered_ravdess_prompts_metadata.csv \
    --output-dir TTS-Outputs/VoxCPM2
```

</details>

<details>
<summary><b>Step 4 — Candidate selection</b>: score every candidate, keep one per (sentence, speaker, emotion)</summary>

```bash
for tts in IndexTTS2 OmniVoice Qwen3-TTS VoxCPM2; do
    python corpus/4_select_candidates/predict_emotion.py --tts-dir TTS-Outputs/$tts
    python corpus/4_select_candidates/predict_content.py --tts-dir TTS-Outputs/$tts --sentences-csv sick_unique_sentences.csv
done
python corpus/4_select_candidates/select_candidates.py --tts-outputs-dir TTS-Outputs
```

`predict_emotion.py` writes `emotion2vec_predictions.csv` and `predict_content.py` writes
`asr_transcriptions.csv` inside each TTS directory. `select_candidates.py` ranks the four candidates by
target-emotion agreement, target-emotion confidence, mean WER, mean CER, and a fixed tie-break order
(OmniVoice, VoxCPM2, IndexTTS2, Qwen3-TTS), then writes `TTS-Outputs/best_available_emotion_samples.csv`.

</details>

## Feature extraction

Both extractors save one `.safetensors` file per input with a `hidden_states` tensor of shape
`[layers, tokens_or_frames, dim]` — padding removed, layer 0 being the embedding or front-end output.
Pooling is deferred to the analysis script.

**Speech** — one run per encoder:

```bash
python features/extract_speech_features.py \
    -b /path/to/data -i TTS-Outputs -o TTS-Speech-Features -m hubert-large-ll60k
```

> Aliases: `wav2vec2-base`, `wav2vec2-large`, `hubert-large-ll60k`, `hubert-xlarge-ll60k`,
> `wavlm-base-plus`, `wavlm-large`, `whisper-small`, `whisper-medium`, `whisper-large-v3`

**Text** — one run per encoder:

```bash
python features/extract_text_features.py \
    -b /path/to/data -o Text-Features -m e5-base-v2 \
    -c sick_unique_sentences.csv -col text --id-column sentence_id
```

> Aliases: `bert-base-uncased`, `bert-large-uncased`, `e5-small-v2`, `e5-base-v2`, `e5-large-v2`,
> `gte-base-en-v1.5`, `gte-large-en-v1.5`, `qwen3-embedding-0.6b`, `qwen3-embedding-4b`, `qwen3-embedding-8b`

E5 inputs receive the `query:` prefix (`--text-role query`, the default); Qwen3-Embedding inputs are used
without an instruction. Validate the written files with:

```bash
python features/check_saved_features.py Text-Features
```

## Analysis

### Alignment and calibration

```bash
python analysis/analyze_alignment.py \
    --text-root   /path/to/data/Text-Features \
    --speech-root /path/to/data/TTS-Speech-Features \
    --best-samples-csv /path/to/data/best_available_emotion_samples.csv \
    --pairs-csv   sick_selected_pairs.csv \
    --output-dir  /path/to/data/Alignment-Analysis \
    --k-values 10,20,50,100 --cycle-k-values 10 --cknna-k-values 10 \
    --rbf-sigmas 0.1,0.5,2.0,5.0 --global-max-samples 600 \
    --permutations 200 \
    --calibration-metrics mutual_knn_k10,rbf_cka_sigma_0p1 \
    --calibration-aggregators max \
    --run-gallery-sweep --run-speech-invariance
```

Computes **mutual kNN, cycle kNN, CKNNA, linear CKA, RSA and RBF-CKA** for every speech-text layer pair in
every (speech encoder, text encoder, speaker, emotion) condition, applies the aggregation-aware permutation
calibration of Gröger et al. (2026) with Benjamini-Hochberg correction, and writes per-condition tables,
encoder rankings, layer heatmaps, gallery-size sweeps, within-speech invariance, and the SICK semantic
validation under `--output-dir`.

### Speaker and prosodic distances (Section 4.5)

```bash
python analysis/speaker_prosody_axes.py \
    --manifest /path/to/data/best_available_emotion_samples.csv \
    --audio-root /path/to/data/TTS-Outputs \
    --results-root /path/to/data/Alignment-Analysis \
    --out-dir /path/to/data/Prosody-Axes
```

## Acknowledgements

This work would not exist without the following openly released datasets, models and tools. We thank their
authors and maintainers.

### Source data

| Resource | Role in this work | Links |
|---|---|---|
| **SICK** — Sentences Involving Compositional Knowledge (Marelli et al., LREC 2014) | Source of the 600 sentences and 300 relatedness-annotated pairs | [Zenodo](https://zenodo.org/records/2787612) · [HF `RobZamp/sick`](https://huggingface.co/datasets/RobZamp/sick) · [paper](https://aclanthology.org/L14-1314/) |
| **RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song (Livingstone & Russo, *PLoS ONE* 2018) | Source of the 24 emotional reference prompts (6 speakers x 4 emotions) | [Zenodo](https://zenodo.org/records/1188976) · [paper](https://doi.org/10.1371/journal.pone.0196391) |

> [!NOTE]
> SICK is distributed under CC BY-NC-SA 3.0 and RAVDESS under CC BY-NC-SA 4.0 (commercial use of RAVDESS
> requires a separate license from its authors). Both are non-commercial — please respect these terms when
> using the derived corpus. See [License](#license).

### Speech synthesis

| System | Links |
|---|---|
| **IndexTTS2** | [GitHub](https://github.com/index-tts/index-tts) |
| **OmniVoice** | [GitHub](https://github.com/k2-fsa/OmniVoice) · [HF `k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) |
| **Qwen3-TTS** | [GitHub](https://github.com/QwenLM/Qwen3-TTS) · [HF `Qwen/Qwen3-TTS-12Hz-1.7B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) |
| **VoxCPM2** | [GitHub](https://github.com/OpenBMB/VoxCPM) · [HF `openbmb/VoxCPM2`](https://huggingface.co/openbmb/VoxCPM2) |

### Candidate scoring

| Model | Use | Links |
|---|---|---|
| **emotion2vec+ large** (Ma et al., ACL 2024) | Emotion verification of prompts and candidates | [GitHub](https://github.com/ddlBoJack/emotion2vec) · [HF](https://huggingface.co/emotion2vec/emotion2vec_plus_large) · [arXiv:2312.15185](https://arxiv.org/abs/2312.15185) · served via [FunASR](https://github.com/modelscope/FunASR) |
| **Parakeet TDT 0.6B v3** | ASR (WER/CER) | [HF](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) · [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) |
| **Qwen3-ASR 1.7B** | ASR (WER/CER) | [HF](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| **Whisper Large V3** (Radford et al., ICML 2023) | ASR (WER/CER) | [HF](https://huggingface.co/openai/whisper-large-v3) · [GitHub](https://github.com/openai/whisper) |

### Encoders under study

**Speech** — [wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-large) (Baevski et al., 2020),
[HuBERT](https://huggingface.co/facebook/hubert-large-ll60k) (Hsu et al., 2021),
[WavLM](https://huggingface.co/microsoft/wavlm-large) (Chen et al., 2022),
[Whisper](https://huggingface.co/openai/whisper-large-v3) (Radford et al., 2023).

**Text** — [BERT](https://huggingface.co/google-bert/bert-large-uncased) (Devlin et al., 2019),
[E5](https://huggingface.co/intfloat/e5-large-v2) (Wang et al., 2022),
[GTE v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5) (Li et al., 2023),
[Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-8B).

### Methods and tooling

- **Aggregation-aware permutation calibration** — Gröger et al. (2026),
  [`calibrated-similarity` / Aristotelian](https://github.com/mlbio-epfl/Aristotelian).
- **Alignment metrics** build on prior work on mutual kNN and CKNNA alignment
  ([Huh et al., 2024](https://arxiv.org/abs/2405.07987)) and CKA ([Kornblith et al., 2019](https://arxiv.org/abs/1905.00414)).
- **Infrastructure** — [PyTorch](https://pytorch.org), [Hugging Face Transformers / Datasets / Hub](https://huggingface.co),
  [safetensors](https://github.com/huggingface/safetensors), [jiwer](https://github.com/jitsi/jiwer),
  [SwiftF0](https://github.com/lars76/swift-f0), [Parselmouth / Praat](https://github.com/YannickJadoul/Parselmouth),
  NumPy, SciPy, scikit-learn, pandas, Matplotlib and seaborn.

## Citation

```bibtex
@inproceedings{ferreira2026echoes,
  title     = {Echoes in Plato's Cave: Measuring Global and Local Alignment Between Speech and Language Representations},
  author    = {Ferreira, Alef Iury Siqueira and Gris, Lucas Rafael Stefanel and de Oliveira, Frederico Santos and Galv{\~a}o Filho, Arlindo Rodrigues and Soares, Anderson da Silva},
  booktitle = {Proceedings of the Speech and Audio Language Models Workshop (SALMA)},
  year      = {2026}
}
```

## License

- **Code** — [MIT License](LICENSE).
- **Corpus** — derived from SICK and RAVDESS, released under **CC BY-NC 4.0**.
