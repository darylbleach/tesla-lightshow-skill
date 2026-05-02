#!/usr/bin/env python3
"""
Analyze a .wav file and emit a JSON timeline of musical features
aligned to 20 ms frames (50 FPS) for Tesla light-show generation.

Features extracted per frame:
    - rms             0..1   broadband loudness (perceptual)
    - bass            0..1   low-band energy (20-200 Hz)          kick/bass
    - low_mid         0..1   low-mid (200-800 Hz)                  bass/vox
    - mid             0..1   mid (800-3200 Hz)                     voice/lead
    - high            0..1   high (3200+ Hz)                       hats/air
    - onset           0..1   spectral flux (onset strength)
    - brightness      0..1   spectral centroid position
    - is_beat         0/1    boolean beat gate (tempo-aware)
    - is_strong_beat  0/1    downbeat gate (every Nth beat)
    - is_drop         0/1    loudness jump from a quiet section
    - tempo_bpm       float  estimated global tempo (same every frame)

The script depends only on stdlib + numpy. If numpy is unavailable,
it falls back to a pure-python implementation that is slower but
still viable for a few-minute track.

Usage:
    python3 analyze_audio.py path/to/song.wav out.json
"""
from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


FRAME_MS = 20  # must match FSEQ step_time


def load_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth == 2:
        fmt = "<" + "h" * (len(raw) // 2)
    elif sampwidth == 1:
        fmt = "<" + "b" * len(raw)
    elif sampwidth == 4:
        fmt = "<" + "i" * (len(raw) // 4)
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    samples = struct.unpack(fmt, raw)
    if HAVE_NUMPY:
        arr = np.asarray(samples, dtype=np.float32)
        if n_channels > 1:
            arr = arr.reshape(-1, n_channels).mean(axis=1)
        max_val = float(2 ** (8 * sampwidth - 1))
        arr /= max_val
        return arr, sr
    # pure-python fallback (slow)
    max_val = float(2 ** (8 * sampwidth - 1))
    if n_channels == 1:
        return [s / max_val for s in samples], sr
    mono = []
    for i in range(0, len(samples), n_channels):
        mono.append(sum(samples[i : i + n_channels]) / (n_channels * max_val))
    return mono, sr


def frame_windows(samples, sr, frame_ms=FRAME_MS):
    """Yield (frame_index, start_sample, end_sample, fft_size)."""
    hop = int(round(sr * frame_ms / 1000.0))
    # window size: ~4 frames' worth (80 ms) rounded to power of 2 for FFT
    win_target = hop * 4
    win = 1
    while win < win_target:
        win *= 2
    if HAVE_NUMPY:
        n = len(samples)
    else:
        n = len(samples)
    return hop, win, n


def analyze(samples, sr):
    hop, win, total = frame_windows(samples, sr)
    n_frames = total // hop
    if HAVE_NUMPY:
        return _analyze_np(samples, sr, hop, win, n_frames)
    return _analyze_py(samples, sr, hop, win, n_frames)


def _analyze_np(samples, sr, hop, win, n_frames):
    samples = np.asarray(samples, dtype=np.float32)
    # pad
    pad = np.zeros(win, dtype=np.float32)
    padded = np.concatenate([samples, pad])
    # hann window
    hann = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)

    def band_mask(lo, hi):
        return (freqs >= lo) & (freqs < hi)

    m_bass = band_mask(20, 200)
    m_lowmid = band_mask(200, 800)
    m_mid = band_mask(800, 3200)
    m_high = band_mask(3200, sr / 2)

    rms = np.zeros(n_frames, dtype=np.float32)
    bass = np.zeros(n_frames, dtype=np.float32)
    lowmid = np.zeros(n_frames, dtype=np.float32)
    mid = np.zeros(n_frames, dtype=np.float32)
    high = np.zeros(n_frames, dtype=np.float32)
    onset = np.zeros(n_frames, dtype=np.float32)
    centroid = np.zeros(n_frames, dtype=np.float32)

    prev_mag = None
    for i in range(n_frames):
        start = i * hop
        seg = padded[start : start + win] * hann
        mag = np.abs(np.fft.rfft(seg))
        rms[i] = float(np.sqrt(np.mean(seg * seg)))
        total = float(mag.sum()) + 1e-9
        bass[i] = float(mag[m_bass].sum()) / total
        lowmid[i] = float(mag[m_lowmid].sum()) / total
        mid[i] = float(mag[m_mid].sum()) / total
        high[i] = float(mag[m_high].sum()) / total
        centroid[i] = float((mag * freqs).sum() / total)
        if prev_mag is not None:
            diff = mag - prev_mag
            diff = np.where(diff > 0, diff, 0)
            onset[i] = float(diff.sum()) / total
        prev_mag = mag

    # normalize bands by full-band energy at each frame (already ratios),
    # but scale by RMS envelope so strong moments are actually strong
    rms_norm = rms / (float(np.percentile(rms, 99)) + 1e-9)
    rms_norm = np.clip(rms_norm, 0, 1)
    bass_abs = bass * rms_norm
    lowmid_abs = lowmid * rms_norm
    mid_abs = mid * rms_norm
    high_abs = high * rms_norm

    # normalize each band to its own 99th percentile
    def norm99(x):
        p = float(np.percentile(x, 99)) + 1e-9
        return np.clip(x / p, 0, 1)

    bass_n = norm99(bass_abs)
    lowmid_n = norm99(lowmid_abs)
    mid_n = norm99(mid_abs)
    high_n = norm99(high_abs)

    onset_n = norm99(onset)
    # brightness: map centroid in log space to 0..1 across [100 Hz, 8 kHz]
    centroid_clipped = np.clip(centroid, 100, 8000)
    brightness = (np.log(centroid_clipped) - math.log(100)) / (math.log(8000) - math.log(100))

    # Beat detection — onset autocorrelation for tempo, then beat tracking via
    # a simple dynamic programming (Ellis 2007 style, stripped down).
    tempo_bpm, beat_frames = estimate_beats_np(onset_n, hop, sr)

    # Drop detection: windowed RMS jump of >0.35 from a low baseline lasting 1+ s
    drop_frames = detect_drops_np(rms_norm)

    return {
        "sr": sr,
        "hop": hop,
        "frame_ms": FRAME_MS,
        "n_frames": n_frames,
        "tempo_bpm": tempo_bpm,
        "rms": rms_norm.tolist(),
        "bass": bass_n.tolist(),
        "low_mid": lowmid_n.tolist(),
        "mid": mid_n.tolist(),
        "high": high_n.tolist(),
        "onset": onset_n.tolist(),
        "brightness": brightness.tolist(),
        "beat_frames": beat_frames,
        "strong_beat_frames": beat_frames[::4] if beat_frames else [],
        "drop_frames": drop_frames,
    }


def estimate_beats_np(onset, hop, sr):
    """Return (tempo_bpm, list of beat frame indices)."""
    import numpy as np

    if len(onset) < 64:
        return 120.0, []
    # Smooth onset a bit for tempo estimation
    smooth = np.convolve(onset, np.ones(3) / 3.0, mode="same")
    # Autocorrelation in lag range 60..200 BPM
    frame_sr = sr / hop  # frames per second
    min_lag = int(frame_sr * 60.0 / 200.0)
    max_lag = int(frame_sr * 60.0 / 60.0)
    max_lag = min(max_lag, len(smooth) // 2)
    if max_lag <= min_lag:
        return 120.0, []
    ac = np.correlate(smooth, smooth, mode="full")
    ac = ac[len(ac) // 2 :]
    best = min_lag + int(np.argmax(ac[min_lag : max_lag + 1]))
    tempo_bpm = 60.0 * frame_sr / best

    # Greedy beat picking: every `best` frames, pick the local onset maximum
    # within a window of ±15% around the expected beat.
    beats = []
    # seed with the peak in the first 2 beats
    first_window = min(len(onset), 2 * best)
    t = int(np.argmax(smooth[:first_window]))
    beats.append(t)
    while True:
        next_center = beats[-1] + best
        if next_center >= len(onset):
            break
        tol = int(best * 0.15)
        lo = max(0, next_center - tol)
        hi = min(len(onset), next_center + tol + 1)
        t = lo + int(np.argmax(smooth[lo:hi]))
        beats.append(t)
    return float(tempo_bpm), [int(b) for b in beats]


def detect_drops_np(rms_norm):
    """Find timestamps (frame indices) where loudness jumps into sustained high energy."""
    import numpy as np

    n = len(rms_norm)
    if n < 200:
        return []
    # 1-second smoothed
    k = max(1, int(1000 / FRAME_MS))  # 50 frames = 1 s
    smooth = np.convolve(rms_norm, np.ones(k) / k, mode="same")
    drops = []
    last_drop = -9999
    lookback = int(3 * k)  # 3 s prior window
    for i in range(lookback, n - k):
        prior = smooth[i - lookback : i].mean()
        here = smooth[i : i + k].mean()
        if here - prior > 0.35 and here > 0.6 and i - last_drop > 4 * k:
            drops.append(i)
            last_drop = i
    return drops


# ---------- pure-python fallback (slower) --------------------------------

def _analyze_py(samples, sr, hop, win, n_frames):
    # Very rough fallback: only RMS + naive onset via RMS diff. No FFT bands.
    rms = []
    for i in range(n_frames):
        start = i * hop
        seg = samples[start : start + hop]
        if not seg:
            rms.append(0.0)
            continue
        rms.append(math.sqrt(sum(s * s for s in seg) / len(seg)))
    peak = max(rms) or 1e-9
    rms = [min(1.0, r / peak) for r in rms]
    onset = [0.0] + [max(0.0, rms[i] - rms[i - 1]) for i in range(1, n_frames)]
    # normalize onset
    op = max(onset) or 1e-9
    onset = [o / op for o in onset]
    # crude beat: every 500 ms
    period = int(500 / FRAME_MS)
    beats = list(range(period // 2, n_frames, period))
    return {
        "sr": sr,
        "hop": hop,
        "frame_ms": FRAME_MS,
        "n_frames": n_frames,
        "tempo_bpm": 120.0,
        "rms": rms,
        "bass": rms,  # no band split available
        "low_mid": rms,
        "mid": rms,
        "high": rms,
        "onset": onset,
        "brightness": [0.5] * n_frames,
        "beat_frames": beats,
        "strong_beat_frames": beats[::4],
        "drop_frames": [],
    }


def main():
    if len(sys.argv) != 3:
        print("usage: analyze_audio.py input.wav output.json", file=sys.stderr)
        sys.exit(2)
    wav = Path(sys.argv[1])
    out = Path(sys.argv[2])
    samples, sr = load_wav(wav)
    if sr != 44100:
        print(
            f"WARNING: sample rate is {sr} Hz — Tesla requires 44100 Hz. "
            "The generated .fseq will still play but may not sync.",
            file=sys.stderr,
        )
    analysis = analyze(samples, sr)
    out.write_text(json.dumps(analysis))
    dur = analysis["n_frames"] * FRAME_MS / 1000.0
    print(
        f"Analyzed {wav.name}: {dur:.1f}s, {analysis['n_frames']} frames, "
        f"tempo ≈ {analysis['tempo_bpm']:.1f} BPM, "
        f"{len(analysis['beat_frames'])} beats, "
        f"{len(analysis['drop_frames'])} drops"
    )


if __name__ == "__main__":
    main()
