#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Speaker (ECAPA2) and prosodic (SwiftF0 + Praat) distance axes for the synthesized corpus."""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd

GENDER = {4: "F", 6: "F", 8: "F", 11: "M", 15: "M", 21: "M"}
LOG = logging.getLogger("axes")

TARGET_SR = 16000


def resolve_path(row, audio_root):
    """Prefer the manifest's own filepath; otherwise rebuild from parts."""
    fp = str(row.get("filepath", "") or "")
    if fp and Path(fp).exists():
        return fp
    if audio_root:
        cand = (
            Path(audio_root)
            / str(row["tts_model"])
            / str(row["speaker_id"])
            / str(row["orig_emotion"])
            / f"{row['sample_id']}.wav"
        )
        if cand.exists():
            return str(cand)
    return fp or None


def load_audio_16k(wav_path):
    """Return mono float32 waveform at 16 kHz."""
    try:
        import torchaudio
        sig, sr = torchaudio.load(str(wav_path))
        if sig.shape[0] > 1:
            sig = sig.mean(0, keepdim=True)
        if sr != TARGET_SR:
            sig = torchaudio.functional.resample(sig, sr, TARGET_SR)
        return sig.squeeze(0).contiguous().numpy().astype(np.float32)
    except Exception as e:                                # pragma: no cover
        LOG.debug("torchaudio load failed (%s); using librosa for %s", e, wav_path)
        import librosa
        y, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
        return np.ascontiguousarray(y, dtype=np.float32)


def load_ecapa2(args, device):
    import torch

    path = args.ecapa2_model
    if not path:
        from huggingface_hub import hf_hub_download
        LOG.info("fetching ECAPA2 from HF repo %s (%s)", args.ecapa2_repo, args.ecapa2_file)
        path = hf_hub_download(
            repo_id=args.ecapa2_repo,
            filename=args.ecapa2_file,
            cache_dir=args.hf_cache or None,
        )
    LOG.info("loading ECAPA2 from %s on %s (fp16=%s)", path, device, args.fp16)
    model = torch.jit.load(path, map_location=device)
    if args.fp16 and str(device).startswith("cuda"):
        model = model.half()
    model.eval()
    return model


def ecapa2_embed(model, y, device, fp16, min_seconds=0.5):
    """y: mono float32 @16 kHz -> speaker embedding (float32, un-normalised)."""
    import torch

    need = int(min_seconds * TARGET_SR)
    if y.size < need:
        y = np.pad(y, (0, need - y.size))
    x = torch.from_numpy(y).unsqueeze(0).to(device)
    if fp16 and str(device).startswith("cuda"):
        x = x.half()
    with torch.no_grad():
        emb = model(x)
    return emb.squeeze().float().detach().cpu().numpy().astype(np.float32)


class SwiftF0Extractor:
    """SwiftF0 pitch + confidence over a 16 kHz numpy waveform."""

    def __init__(self, args):
        try:
            from swift_f0 import SwiftF0
        except ImportError as e:
            raise ImportError(
                "swift-f0 is not installed -- `pip install swift-f0` "
                "(https://github.com/lars76/swift-f0)"
            ) from e

        lo, hi = SwiftF0.MODEL_FMIN, SwiftF0.MODEL_FMAX
        self.fmin = float(np.clip(args.f0_min, lo, hi))
        self.fmax = float(np.clip(args.f0_max, lo, hi))
        if (self.fmin, self.fmax) != (float(args.f0_min), float(args.f0_max)):
            LOG.warning("clamped F0 range to SwiftF0's supported band "
                        "[%.3f, %.3f] Hz -> using %.1f-%.1f Hz",
                        lo, hi, self.fmin, self.fmax)

        self.threshold = float(args.swiftf0_confidence)
        self.median_filter = int(args.f0_median_filter)
        self.detector = SwiftF0(
            confidence_threshold=self.threshold,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        self.hopsize = SwiftF0.HOP_LENGTH / SwiftF0.TARGET_SAMPLE_RATE

        if args.swiftf0_threads != 1 or args.swiftf0_provider:
            self._rebuild_session(args)

        LOG.info("SwiftF0 ready (hop=%.4fs [%d samples @ %d Hz], range=%.1f-%.1f Hz, "
                 "voiced@confidence>%.2f, medfilt=%d, providers=%s)",
                 self.hopsize, SwiftF0.HOP_LENGTH, SwiftF0.TARGET_SAMPLE_RATE,
                 self.fmin, self.fmax, self.threshold, self.median_filter,
                 self.detector.pitch_session.get_providers())
        self._warmup()

    def _rebuild_session(self, args):
        import onnxruntime as ort
        import swift_f0.core as core_mod

        model_path = os.path.join(os.path.dirname(core_mod.__file__), "model.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(args.swiftf0_threads))
        opts.inter_op_num_threads = 1

        available = ort.get_available_providers()
        wanted = [p for p in (args.swiftf0_provider or "").split(",") if p]
        providers = [p for p in wanted if p in available]
        for p in wanted:
            if p not in available:
                LOG.warning("execution provider %r unavailable (have: %s)",
                            p, ", ".join(available))
        providers = providers or ["CPUExecutionProvider"]

        session = ort.InferenceSession(model_path, opts, providers=providers)
        self.detector.pitch_session = session
        self.detector.pitch_input_name = session.get_inputs()[0].name

    def f0(self, y):
        res = self.detector.detect_from_array(y, TARGET_SR)
        f0 = np.atleast_1d(np.asarray(res.pitch_hz, dtype=np.float64))
        conf = np.atleast_1d(np.asarray(res.confidence, dtype=np.float64))
        voiced = np.atleast_1d(np.asarray(res.voicing, dtype=bool))
        f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
        conf = np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0)

        n = min(f0.size, conf.size, voiced.size)
        f0, conf, voiced = f0[:n], conf[:n], voiced[:n]

        f0 = np.where(voiced, f0, 0.0)
        if self.median_filter > 1:
            f0 = _median_filter_voiced(f0, self.median_filter)
        return f0, conf

    def _warmup(self):
        t = np.arange(TARGET_SR, dtype=np.float32) / TARGET_SR
        probe = (0.1 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        f0, conf = self.f0(probe)
        voiced = f0[f0 > 0]
        LOG.info("SwiftF0 warm-up: %d frames, %.0f%% voiced, median f0=%.1f Hz "
                 "(probe is a 200 Hz tone), mean confidence=%.3f",
                 f0.size, 100.0 * (f0 > 0).mean() if f0.size else 0.0,
                 float(np.median(voiced)) if voiced.size else float("nan"),
                 float(np.mean(conf)) if conf.size else float("nan"))


def _median_filter_voiced(f0, k):
    if k < 3:
        return f0
    k |= 1
    out = f0.copy()
    voiced = f0 > 0
    if not voiced.any():
        return out
    padded = np.concatenate(([False], voiced, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    for start, stop in zip(edges[0::2], edges[1::2]):
        run = f0[start:stop]
        if run.size < k:
            continue
        half = k // 2
        ext = np.pad(run, (half, half), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(ext, k)
        out[start:stop] = np.median(win, axis=-1)
    return out


def f0_feats(f0, confidence, hop_s):
    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]
    voiced_fraction = float(voiced_mask.mean()) if f0.size else 0.0
    conf_mean = float(np.mean(confidence)) if confidence.size else 0.0
    conf_std = float(np.std(confidence)) if confidence.size else 0.0
    conf_voiced = float(np.mean(confidence[voiced_mask])) if voiced.size else 0.0

    base = dict(voiced_fraction=voiced_fraction, f0_conf_mean=conf_mean,
                f0_conf_std=conf_std, f0_conf_voiced_mean=conf_voiced)
    if voiced.size < 2:
        return dict(f0_mean=0.0, f0_median=0.0, f0_std=0.0, f0_std_st=0.0,
                    f0_range_st=0.0, f0_delta_st_per_s=0.0, voiced_seg_rate=0.0,
                    **base)

    med = float(np.median(voiced))
    st = 12.0 * np.log2(np.maximum(f0, 1e-6) / max(med, 1e-6))
    p05, p95 = np.percentile(voiced, [5, 95])

    d = np.diff(st)
    valid = voiced_mask[1:] & voiced_mask[:-1]
    delta = float(np.mean(np.abs(d[valid])) / hop_s) if valid.any() else 0.0

    edges = np.diff(voiced_mask.astype(np.int8))
    n_seg = int((edges == 1).sum()) + int(voiced_mask[0])
    dur = f0.size * hop_s
    return dict(
        f0_mean=float(np.mean(voiced)),
        f0_median=med,
        f0_std=float(np.std(voiced)),
        f0_std_st=float(np.std(st[voiced_mask])),
        f0_range_st=float(12 * np.log2(max(p95, 1e-6) / max(p05, 1e-6))),
        f0_delta_st_per_s=delta,
        voiced_seg_rate=float(n_seg / dur) if dur > 0 else 0.0,
        **base,
    )


def energy_feats(y, sr=TARGET_SR):
    try:
        return _energy_praat(y, sr)
    except Exception as e:                    # pragma: no cover - env dependent
        LOG.debug("parselmouth failed (%s); using librosa energy", e)
        return _energy_librosa(y, sr)


def _energy_praat(y, sr):
    import parselmouth

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    dur = snd.get_total_duration()
    intensity = snd.to_intensity(minimum_pitch=60, time_step=0.01)
    ivals = intensity.values[0]
    ivals = ivals[np.isfinite(ivals)]
    return dict(
        duration=float(dur),
        intensity_mean=float(np.mean(ivals)) if ivals.size else 0.0,
        intensity_std=float(np.std(ivals)) if ivals.size else 0.0,
        syllable_rate=_syllable_rate_praat(intensity, dur),
    )


def _syllable_rate_praat(intensity, duration, thresh_db=2.0, min_dist_s=0.10):
    vals = intensity.values[0].astype(float)
    ts = np.arange(vals.size) * intensity.get_time_step() + intensity.get_start_time()
    finite = np.isfinite(vals)
    vals, ts = vals[finite], ts[finite]
    if vals.size < 3 or duration <= 0:
        return 0.0
    thr = np.median(vals) + thresh_db
    peaks, last_t = 0, -1e9
    for i in range(1, vals.size - 1):
        if vals[i] > thr and vals[i] >= vals[i - 1] and vals[i] > vals[i + 1]:
            if ts[i] - last_t >= min_dist_s:
                peaks += 1
                last_t = ts[i]
    return float(peaks / duration)


def _energy_librosa(y, sr):
    import librosa

    dur = len(y) / sr if sr else 0.0
    rms = librosa.feature.rms(y=y)[0]
    try:
        ons = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        syll_rate = float(len(ons) / dur) if dur else 0.0
    except Exception:
        syll_rate = 0.0
    return dict(
        duration=float(dur),
        intensity_mean=float(20 * np.log10(np.mean(rms) + 1e-8)),
        intensity_std=float(np.std(20 * np.log10(rms + 1e-8))),
        syllable_rate=syll_rate,
    )


PROS = ["f0_mean", "f0_median", "f0_std", "f0_std_st", "f0_range_st",
        "f0_delta_st_per_s", "voiced_fraction", "voiced_seg_rate",
        "f0_conf_mean", "f0_conf_std", "f0_conf_voiced_mean",
        "intensity_mean", "intensity_std", "syllable_rate", "duration"]
ZPROS = [f"z_{f}" for f in PROS]


def step_extract(args, out_path):
    if out_path.exists() and not args.force:
        LOG.info("[1/4] extract: %s exists, skipping", out_path.name)
        return pd.read_parquet(out_path)

    man = pd.read_csv(args.manifest)
    keep = ["tts_model", "speaker_id", "orig_emotion", "sample_id", "filepath"]
    man = man[[c for c in keep if c in man.columns]].copy()
    if args.limit:
        man = man.head(args.limit)
    LOG.info("[1/4] extract: %d utterances", len(man))

    device = args.device
    spk_model = load_ecapa2(args, device)
    pitch = SwiftF0Extractor(args)

    rows, missing, emb_dim = [], 0, None
    for _, r in man.iterrows():
        path = resolve_path(r, args.audio_root)
        if not path or not Path(path).exists():
            missing += 1
            continue
        try:
            y = load_audio_16k(path)
            if y.size < int(0.05 * TARGET_SR):
                raise ValueError(f"clip too short ({y.size} samples)")
            emb = ecapa2_embed(spk_model, y, device, args.fp16)
            f0, conf = pitch.f0(y)
            pf = dict(**f0_feats(f0, conf, pitch.hopsize), **energy_feats(y))
        except Exception as e:
            LOG.warning("failed on %s: %s", path, e)
            missing += 1
            continue

        emb_dim = emb_dim or int(emb.size)
        rec = dict(
            tts_model=r["tts_model"], speaker=int(r["speaker_id"]),
            emotion=r["orig_emotion"], sample_id=r["sample_id"],
            gender=GENDER.get(int(r["speaker_id"]), "?"), **pf,
        )
        for j, v in enumerate(emb):
            rec[f"e{j:04d}"] = float(v)
        rows.append(rec)
        if (len(rows) % 500) == 0:
            LOG.info("  ...%d done (%d missing)", len(rows), missing)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    (out_path.parent / "feature_extraction_config.json").write_text(json.dumps(dict(
        speaker_encoder="ECAPA2",
        ecapa2_source=args.ecapa2_model or f"{args.ecapa2_repo}/{args.ecapa2_file}",
        embedding_dim=emb_dim,
        f0_estimator="SwiftF0",
        swiftf0_confidence_threshold=pitch.threshold,
        swiftf0_hopsize_s=pitch.hopsize,
        swiftf0_providers=pitch.detector.pitch_session.get_providers(),
        f0_min=pitch.fmin, f0_max=pitch.fmax,
        f0_median_filter=pitch.median_filter,
        sample_rate=TARGET_SR, fp16=bool(args.fp16),
        prosody_features=PROS,
    ), indent=2))
    LOG.info("[1/4] wrote %s (%d rows, %d missing, emb_dim=%s)",
             out_path.name, len(df), missing, emb_dim)
    return df


def _emb_cols(df):
    return [c for c in df.columns if c.startswith("e") and c[1:].isdigit()]


def _l2norm(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)


def step_speaker(df, out_dir, args):
    ecols = _emb_cols(df)
    if not ecols:
        LOG.error("no embedding columns in features table"); sys.exit(1)

    cen = df.groupby("speaker")[ecols].mean()
    cenn = _l2norm(cen.values)
    spk = list(cen.index)
    dmat = 1.0 - cenn @ cenn.T
    dm = pd.DataFrame(dmat, index=spk, columns=spk)
    dm.to_csv(out_dir / "speaker_distance_matrix.csv")

    centroids = cen.reset_index()
    centroids["gender"] = centroids.speaker.map(GENDER)
    centroids.to_parquet(out_dir / "speaker_embeddings.parquet", index=False)

    wrows = []
    for tts, g in df.groupby("tts_model"):
        c = g.groupby("speaker")[ecols].mean()
        cn = _l2norm(c.values); idx = list(c.index)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                wrows.append(dict(tts_model=tts, speaker_a=idx[a], speaker_b=idx[b],
                                  d_spk=float(1 - cn[a] @ cn[b])))
    pd.DataFrame(wrows).to_csv(out_dir / "speaker_distance_within_tts.csv", index=False)

    same = [dm.loc[a, b] for a in spk for b in spk
            if a < b and GENDER[a] == GENDER[b]]
    cross = [dm.loc[a, b] for a in spk for b in spk
             if a < b and GENDER[a] != GENDER[b]]
    LOG.info("[2/4] ECAPA2 gender check (%dd emb): same-gender d=%.3f  cross-gender d=%.3f "
             "(cross should be larger)", len(ecols), np.mean(same), np.mean(cross))
    return dm


def step_prosody(df, out_dir):
    feats = [f for f in PROS if f in df.columns]
    cond = df.groupby(["speaker", "emotion", "tts_model"])[feats].mean().reset_index()

    for f in feats:
        cond[f"z_{f}"] = cond.groupby("speaker")[f].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    zcols = [f"z_{f}" for f in feats]

    neutral = cond[cond.emotion == "neutral"].set_index(["speaker", "tts_model"])[zcols]
    dev = []
    for _, r in cond.iterrows():
        key = (r["speaker"], r["tts_model"])
        if key in neutral.index:
            base = np.asarray(neutral.loc[key].values, dtype=float)
            d = float(np.linalg.norm(r[zcols].values.astype(float) - base))
        else:
            d = np.nan
        dev.append(d)
    cond["d_pros_from_neutral"] = dev
    cond.to_parquet(out_dir / "condition_prosody.parquet", index=False)
    LOG.info("[3/4] prosody: %d (speaker,emotion,tts) conditions over %d SwiftF0/energy features",
             len(cond), len(feats))
    return cond


def _pearson(x, y):
    from scipy.stats import pearsonr
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), float("nan"), int(m.sum())
    r, p = pearsonr(np.asarray(x)[m], np.asarray(y)[m])
    return float(r), float(p), int(m.sum())


def step_correlate(dm, cond, out_dir, args):
    summary = {}
    inv_path = Path(args.results_root) / "speech_invariance" / "same_layer_invariance.parquet"

    if inv_path.exists():
        inv = pd.read_parquet(inv_path)
        sp = inv[(inv.contrast_type == "speaker") & (inv.layer != "layer_000")].copy()
        sp["speaker_a"] = sp.speaker_a.astype(int)
        sp["speaker_b"] = sp.speaker_b.astype(int)
        sp["d_spk"] = [dm.loc[a, b] for a, b in zip(sp.speaker_a, sp.speaker_b)]
        sp["same_gender"] = [GENDER[a] == GENDER[b] for a, b in zip(sp.speaker_a, sp.speaker_b)]
        sp.to_csv(out_dir / "speaker_distance_vs_invariance.csv", index=False)
        for met in ["mutual_knn_k10", "linear_cka", "rsa_spearman"]:
            if met in sp.columns:
                r, p, n = _pearson(sp["d_spk"].values, sp[met].values)
                summary[f"speaker_invariance/{met}_vs_dspk"] = dict(r=r, p=p, n=n)
                LOG.info("[4/4] A) d_spk vs %-15s self-sim: r=%+.3f p=%.3g (n=%d)",
                         met, r, p, n)
        LOG.info("     same/cross-gender mean d_spk: %.3f / %.3f",
                 sp[sp.same_gender].d_spk.mean(), sp[~sp.same_gender].d_spk.mean())
    else:
        LOG.warning("invariance file not found: %s", inv_path)

    if args.calibration and Path(args.calibration).exists():
        cal = pd.read_parquet(args.calibration)
        knn = cal[cal.metric == "mutual_knn_k10"]
        align = knn.groupby(["speaker", "emotion"]).calibrated_score.mean().reset_index()
        pc = cond.groupby(["speaker", "emotion"]).d_pros_from_neutral.mean().reset_index()
        mp = align.merge(pc, on=["speaker", "emotion"])
        mp.to_csv(out_dir / "prosody_vs_alignment.csv", index=False)
        r, p, n = _pearson(mp["d_pros_from_neutral"].values, mp["calibrated_score"].values)
        summary["prosody/dpros_vs_calibrated_mknn"] = dict(r=r, p=p, n=n)
        LOG.info("[4/4] B) d_pros(from neutral) vs calibrated mKNN: r=%+.3f p=%.3g (n=%d)",
                 r, p, n)

        if inv_path.exists():
            em = inv[(inv.contrast_type == "emotion") & (inv.layer != "layer_000")].copy()
            em["speaker_a"] = em.speaker_a.astype(int)
            zcols = [c for c in ZPROS if c in cond.columns]
            zc = cond.groupby(["speaker", "emotion"])[zcols].mean()

            def epair(row):
                try:
                    a = zc.loc[(row.speaker_a, row.emotion_a)].values.astype(float)
                    b = zc.loc[(row.speaker_a, row.emotion_b)].values.astype(float)
                    return float(np.linalg.norm(a - b))
                except Exception:
                    return np.nan

            if {"emotion_a", "emotion_b"}.issubset(em.columns):
                em["d_pros_pair"] = em.apply(epair, axis=1)
                r, p, n = _pearson(em["d_pros_pair"].values, em["mutual_knn_k10"].values)
                summary["emotion/dpros_pair_vs_selfsim"] = dict(r=r, p=p, n=n)
                LOG.info("[4/4] C) prosody-distance vs emotion self-sim: r=%+.3f p=%.3g (n=%d)",
                         r, p, n)
    else:
        LOG.warning("calibration parquet not given/found; skipping prosody-alignment corr")

    (out_dir / "correlations_summary.json").write_text(json.dumps(summary, indent=2))
    LOG.info("wrote correlations_summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="best_available_emotion_samples.csv")
    ap.add_argument("--audio-root", default=None,
                    help="root of TTS-Outputs (used only if manifest filepath is invalid)")
    ap.add_argument("--results-root", required=True,
                    help="Convergence-Analysis-Curated (for speech_invariance/)")
    ap.add_argument("--calibration", default=None,
                    help="aggregation_aware_calibration.parquet (for alignment corr)")
    ap.add_argument("--out-dir", default="axes_out")
    ap.add_argument("--device", default="cuda", help="cuda | cuda:N | cpu (ECAPA2 only)")
    ap.add_argument("--fp16", action="store_true",
                    help="run ECAPA2 in half precision (CUDA only)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N utterances")
    ap.add_argument("--force", action="store_true", help="recompute step 1")
    ap.add_argument("--steps", default="1,2,3,4",
                    help="comma list of steps to run (1 extract,2 speaker,3 prosody,4 correlate)")
    ap.add_argument("--ecapa2-model", default=None,
                    help="local ecapa2.pt (TorchScript); default = download from HF")
    ap.add_argument("--ecapa2-repo", default="Jenthe/ECAPA2")
    ap.add_argument("--ecapa2-file", default="ecapa2.pt")
    ap.add_argument("--hf-cache", default=None, help="HuggingFace cache dir")
    ap.add_argument("--swiftf0-confidence", type=float, default=0.9,
                    help="voicing confidence threshold (SwiftF0 default: 0.9)")
    ap.add_argument("--swiftf0-threads", type=int, default=1,
                    help="ONNX intra-op threads; SwiftF0 hardcodes 1")
    ap.add_argument("--swiftf0-provider", default=None,
                    help="comma-separated onnxruntime execution providers, e.g. "
                         "CUDAExecutionProvider (needs onnxruntime-gpu)")
    ap.add_argument("--f0-min", type=float, default=50.0,
                    help="clamped into SwiftF0's 46.875-2093.75 Hz band")
    ap.add_argument("--f0-max", type=float, default=1000.0,
                    help="clamped into SwiftF0's 46.875-2093.75 Hz band")
    ap.add_argument("--f0-median-filter", type=int, default=3,
                    help="odd-width median filter over voiced F0 runs; suppresses "
                         "isolated octave errors. 0 or 1 disables it")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                LOG.warning("cuda requested but unavailable; falling back to cpu")
                args.device, args.fp16 = "cpu", False
        except Exception:
            args.device, args.fp16 = "cpu", False

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    steps = set(args.steps.split(","))
    feat_path = out_dir / "utterance_features.parquet"

    df = None
    if "1" in steps:
        df = step_extract(args, feat_path)
    if df is None:
        if not feat_path.exists():
            LOG.error("need utterance_features.parquet; run step 1 first"); sys.exit(1)
        df = pd.read_parquet(feat_path)

    dm = cond = None
    if "2" in steps:
        dm = step_speaker(df, out_dir, args)
    elif (out_dir / "speaker_distance_matrix.csv").exists():
        dm = pd.read_csv(out_dir / "speaker_distance_matrix.csv", index_col=0)
        dm.columns = dm.columns.astype(int); dm.index = dm.index.astype(int)

    if "3" in steps:
        cond = step_prosody(df, out_dir)
    elif (out_dir / "condition_prosody.parquet").exists():
        cond = pd.read_parquet(out_dir / "condition_prosody.parquet")

    if "4" in steps:
        if dm is None or cond is None:
            LOG.error("step 4 needs steps 2 and 3 outputs present"); sys.exit(1)
        step_correlate(dm, cond, out_dir, args)

    LOG.info("done. outputs in %s", out_dir)


if __name__ == "__main__":
    main()
