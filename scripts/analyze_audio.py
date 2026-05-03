#!/usr/bin/env python3
"""
Analyze a .wav file and emit a JSON timeline of musical features
aligned to 20 ms frames (50 FPS) for Tesla light-show generation.

The analyzer keeps the stereo field all the way through the FFT
rather than downmixing to mono up front.  This preserves real
musical information (panning, stereo width, mid/side balance) that
the composer uses to choreograph left-vs-right lights in sync with
where things actually sit in the mix.

Features extracted per frame (20 ms, = 1 FSEQ step):

    Energy / loudness:
        rms             0..1   broadband loudness (perceptual)
        bass            0..1   20-200 Hz energy
        low_mid         0..1   200-800 Hz
        mid             0..1   800-3200 Hz
        high            0..1   3200 Hz+
        mel             32 bins 0..1, log-spaced between 20 Hz and Nyquist
        mid_energy      0..1   Mid  = (L+R)/2 — center of the mix
        side_energy     0..1   Side = (L-R)/2 — stereo/ambience
        stereo_width    0..1   0 = pure mono, 1 = fully decorrelated

    Stereo location (where the sound lives in the mix):
        pan_bass        -1..+1  -1 = hard left, 0 = center, +1 = hard right
        pan_mid         -1..+1
        pan_high        -1..+1
        pan_overall     -1..+1

    Percussive / harmonic split (HPSS):
        perc_rms        0..1   percussive component loudness (drums/hats)
        perc_onset      0..1   onset strength computed on percussive-only
        harm_rms        0..1   harmonic component loudness (chords/vox)

    Perceptual / structural:
        onset           0..1   spectral flux (broadband onset strength)
        brightness      0..1   spectral centroid (log-mapped)
        is_beat         via beat_frames[]
        is_strong_beat  via strong_beat_frames[] (every 4th beat)
        is_drop         via drop_frames[]
        tempo_bpm       float

The script depends on numpy (required) and optionally uses librosa
for higher-quality beat tracking / onset detection / CQT.  If
librosa is importable at runtime it's used automatically; otherwise
the pure-numpy path runs.

Usage:
    python3 analyze_audio.py input.wav output.json
"""
from __future__ import annotations

import json
import math
import struct
import sys
import wave
from pathlib import Path

import numpy as np

try:
    import librosa  # type: ignore
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False


FRAME_MS = 20  # must match FSEQ step_time


# ---------------------------------------------------------------------------
# Wav loading (stereo preserved)
# ---------------------------------------------------------------------------

def load_wav_stereo(path: Path):
    """Return (L, R, sr). For mono wavs both channels are equal."""
    with wave.open(str(path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        sr = w.getframerate()
        n_frames = w.getnframes()
        raw = w.readframes(n_frames)
    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 1:
        dtype = np.int8
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    max_val = float(2 ** (8 * sampwidth - 1))
    samples /= max_val
    if n_channels == 1:
        return samples, samples, sr
    # interleaved -> (N, C)
    samples = samples.reshape(-1, n_channels)
    # Use first two channels; ignore surround extras
    L = samples[:, 0]
    R = samples[:, 1] if samples.shape[1] > 1 else samples[:, 0]
    return L.copy(), R.copy(), sr


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------

def pick_window(sr: int, frame_ms: int = FRAME_MS):
    hop = int(round(sr * frame_ms / 1000.0))
    win_target = hop * 4
    win = 1
    while win < win_target:
        win *= 2
    return hop, win


def stft_stereo(L: np.ndarray, R: np.ndarray, sr: int):
    hop, win = pick_window(sr)
    pad = np.zeros(win, dtype=np.float32)
    Lp = np.concatenate([L, pad])
    Rp = np.concatenate([R, pad])
    n_frames = (len(L)) // hop
    hann = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr).astype(np.float32)
    magL = np.empty((n_frames, freqs.size), dtype=np.float32)
    magR = np.empty_like(magL)
    # RMS-time-domain per-channel for percussive/harmonic analysis too
    rms_L = np.empty(n_frames, dtype=np.float32)
    rms_R = np.empty(n_frames, dtype=np.float32)
    corr = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        sL = Lp[start : start + win]
        sR = Rp[start : start + win]
        wL = sL * hann
        wR = sR * hann
        magL[i] = np.abs(np.fft.rfft(wL))
        magR[i] = np.abs(np.fft.rfft(wR))
        rms_L[i] = float(np.sqrt(np.mean(sL * sL)))
        rms_R[i] = float(np.sqrt(np.mean(sR * sR)))
        # short-time Pearson correlation between L and R in this window
        lz = sL - sL.mean()
        rz = sR - sR.mean()
        denom = (np.linalg.norm(lz) * np.linalg.norm(rz)) + 1e-9
        corr[i] = float(np.dot(lz, rz) / denom)
    return magL, magR, freqs, rms_L, rms_R, corr, hop


# ---------------------------------------------------------------------------
# HPSS (median-filter-based percussive/harmonic separation)
# ---------------------------------------------------------------------------

def hpss_magnitude(mag: np.ndarray, kernel_time: int = 17, kernel_freq: int = 17):
    """Simple Fitzgerald-style HPSS on a magnitude spectrogram.

    Horizontal (time) median filter -> harmonic component.
    Vertical (frequency) median filter -> percussive component.
    Soft mask between them.

    Input mag shape (frames, bins). Output (perc, harm) same shape.
    """
    from scipy.ndimage import median_filter  # type: ignore
    # time axis is 0, frequency axis is 1
    harm = median_filter(mag, size=(kernel_time, 1), mode="reflect")
    perc = median_filter(mag, size=(1, kernel_freq), mode="reflect")
    # soft mask
    mask_p = perc / (perc + harm + 1e-9)
    mask_h = 1.0 - mask_p
    return mag * mask_p, mag * mask_h


def hpss_magnitude_numpy(mag: np.ndarray, kernel_time: int = 17, kernel_freq: int = 17):
    """Pure-numpy fallback HPSS when scipy isn't available.

    Uses a rolling-window median approximation (sort + middle element)
    — slower than scipy.ndimage but still O(N log K).
    """
    def roll_median(x, axis, k):
        if k <= 1:
            return x
        # pad
        pad_before = k // 2
        pad_after = k - 1 - pad_before
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (pad_before, pad_after)
        padded = np.pad(x, pad_width, mode="reflect")
        # sliding window view
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, window_shape=k, axis=axis)
        return np.median(windows, axis=-1)

    harm = roll_median(mag, axis=0, k=kernel_time)
    perc = roll_median(mag, axis=1, k=kernel_freq)
    mask_p = perc / (perc + harm + 1e-9)
    mask_h = 1.0 - mask_p
    return mag * mask_p, mag * mask_h


def run_hpss(mag: np.ndarray):
    try:
        return hpss_magnitude(mag)
    except Exception:
        return hpss_magnitude_numpy(mag)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def mel_filters(n_mels: int, n_fft_bins: int, sr: int):
    """A small hand-rolled mel filterbank.  Returns (n_mels, n_fft_bins)."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    low_mel = hz_to_mel(20.0)
    high_mel = hz_to_mel(sr / 2.0)
    mel_pts = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_edges = np.floor((n_fft_bins - 1) * 2 * hz_pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = bin_edges[m - 1], bin_edges[m], bin_edges[m + 1]
        for k in range(l, c):
            fb[m - 1, k] = (k - l) / max(1, (c - l))
        for k in range(c, r):
            fb[m - 1, k] = (r - k) / max(1, (r - c))
    return fb


def norm99(x: np.ndarray) -> np.ndarray:
    p = float(np.percentile(x, 99)) + 1e-9
    return np.clip(x / p, 0.0, 1.0)


def analyze(L: np.ndarray, R: np.ndarray, sr: int):
    magL, magR, freqs, rms_L, rms_R, corr, hop = stft_stereo(L, R, sr)
    n_frames = magL.shape[0]
    mag_sum = magL + magR + 1e-9
    mag_mono = 0.5 * (magL + magR)  # the "mid" spectrogram (L+R)/2 in magnitude
    mag_side = 0.5 * np.abs(magL - magR)

    # --- broadband RMS ------------------------------------------------------
    rms = 0.5 * (rms_L + rms_R)
    rms_n = norm99(rms)

    # --- stereo width ------------------------------------------------------
    # correlation ∈ [-1, 1]; width = 1 - |corr|
    width = np.clip(1.0 - np.abs(corr), 0.0, 1.0)

    # --- mid/side loudness --------------------------------------------------
    mid_energy = norm99(mag_mono.sum(axis=1) * rms)
    side_energy = norm99(mag_side.sum(axis=1) * rms)

    # --- fixed 4-band features (kept for backward compat) ------------------
    def band_mask(lo, hi):
        return (freqs >= lo) & (freqs < hi)

    bands = {
        "bass": band_mask(20, 200),
        "low_mid": band_mask(200, 800),
        "mid": band_mask(800, 3200),
        "high": band_mask(3200, sr / 2),
    }
    band_out = {}
    pan_out = {}
    for name, m in bands.items():
        eL = magL[:, m].sum(axis=1)
        eR = magR[:, m].sum(axis=1)
        total_band = eL + eR + 1e-9
        band_frac = total_band / mag_sum.sum(axis=1)  # ratio of total energy
        band_out[name] = norm99(band_frac * rms)
        pan_out[name] = (eR - eL) / total_band  # -1..+1

    pan_overall = (magR.sum(axis=1) - magL.sum(axis=1)) / mag_sum.sum(axis=1)

    # --- mel bands ----------------------------------------------------------
    n_mels = 32
    fb = mel_filters(n_mels, magL.shape[1], sr)
    mel_energy = (mag_mono @ fb.T) * rms[:, None]
    mel_norm = np.stack([norm99(mel_energy[:, i]) for i in range(n_mels)], axis=1)

    # --- HPSS on mono magnitude --------------------------------------------
    perc_mag, harm_mag = run_hpss(mag_mono)
    perc_rms = np.sqrt(np.mean(perc_mag * perc_mag, axis=1))
    harm_rms = np.sqrt(np.mean(harm_mag * harm_mag, axis=1))
    perc_rms = norm99(perc_rms)
    harm_rms = norm99(harm_rms)

    # percussive-only onset (spectral flux of perc component)
    perc_diff = np.diff(perc_mag, axis=0, prepend=perc_mag[:1])
    perc_diff = np.maximum(perc_diff, 0.0).sum(axis=1)
    perc_onset = norm99(perc_diff)

    # broadband onset (spectral flux of whole mono spectrogram)
    all_diff = np.diff(mag_mono, axis=0, prepend=mag_mono[:1])
    onset = norm99(np.maximum(all_diff, 0.0).sum(axis=1))

    # brightness (spectral centroid, log-mapped)
    centroid = (mag_mono * freqs[None, :]).sum(axis=1) / mag_mono.sum(axis=1, keepdims=False).clip(1e-9)
    centroid_clipped = np.clip(centroid, 100.0, 8000.0)
    brightness = (np.log(centroid_clipped) - math.log(100.0)) / (math.log(8000.0) - math.log(100.0))

    # --- beats / tempo -----------------------------------------------------
    tempo_bpm, beat_frames = beats_and_tempo(perc_onset, onset, hop, sr, L, R)

    # --- drops --------------------------------------------------------------
    drops = detect_drops(rms_n)

    result = {
        "sr": int(sr),
        "hop": int(hop),
        "frame_ms": FRAME_MS,
        "n_frames": int(n_frames),
        "analyzer": "librosa" if HAVE_LIBROSA else "numpy-stereo",
        "tempo_bpm": float(tempo_bpm),
        "rms": rms_n.tolist(),
        "bass": band_out["bass"].tolist(),
        "low_mid": band_out["low_mid"].tolist(),
        "mid": band_out["mid"].tolist(),
        "high": band_out["high"].tolist(),
        "mel": mel_norm.tolist(),
        "mid_energy": mid_energy.tolist(),
        "side_energy": side_energy.tolist(),
        "stereo_width": width.tolist(),
        "pan_bass": pan_out["bass"].tolist(),
        "pan_mid": pan_out["mid"].tolist(),
        "pan_high": pan_out["high"].tolist(),
        "pan_overall": pan_overall.tolist(),
        "perc_rms": perc_rms.tolist(),
        "perc_onset": perc_onset.tolist(),
        "harm_rms": harm_rms.tolist(),
        "onset": onset.tolist(),
        "brightness": brightness.tolist(),
        "beat_frames": [int(b) for b in beat_frames],
        "strong_beat_frames": [int(b) for b in beat_frames[::4]],
        "drop_frames": [int(d) for d in drops],
    }
    return result


# ---------------------------------------------------------------------------
# Beat + tempo
# ---------------------------------------------------------------------------

def beats_and_tempo(perc_onset, onset, hop, sr, L, R):
    """Prefer librosa if available; fall back to greedy autocorrelation."""
    if HAVE_LIBROSA:
        try:
            y = 0.5 * (L + R)
            # ensure we match our hop / frame grid
            tempo, beat_samples = librosa.beat.beat_track(
                y=y.astype(np.float32),
                sr=sr,
                hop_length=hop,
                units="frames",
            )
            tempo = float(np.atleast_1d(tempo)[0])
            return tempo, [int(b) for b in beat_samples]
        except Exception:
            pass  # fall through to the numpy implementation
    return _beats_numpy(perc_onset, hop, sr)


def _beats_numpy(onset, hop, sr):
    if len(onset) < 64:
        return 120.0, []
    smooth = np.convolve(onset, np.ones(3) / 3.0, mode="same")
    frame_sr = sr / hop
    min_lag = int(frame_sr * 60.0 / 200.0)
    max_lag = int(frame_sr * 60.0 / 60.0)
    max_lag = min(max_lag, len(smooth) // 2)
    if max_lag <= min_lag:
        return 120.0, []
    ac = np.correlate(smooth, smooth, mode="full")
    ac = ac[len(ac) // 2 :]
    best = min_lag + int(np.argmax(ac[min_lag : max_lag + 1]))
    tempo_bpm = 60.0 * frame_sr / best
    # greedy pick
    beats = []
    first_window = min(len(onset), 2 * best)
    t = int(np.argmax(smooth[:first_window]))
    beats.append(t)
    while True:
        nxt = beats[-1] + best
        if nxt >= len(onset):
            break
        tol = int(best * 0.15)
        lo = max(0, nxt - tol)
        hi = min(len(onset), nxt + tol + 1)
        t = lo + int(np.argmax(smooth[lo:hi]))
        beats.append(t)
    return float(tempo_bpm), [int(b) for b in beats]


def detect_drops(rms_norm):
    n = len(rms_norm)
    if n < 200:
        return []
    k = max(1, int(1000 / FRAME_MS))
    smooth = np.convolve(rms_norm, np.ones(k) / k, mode="same")
    drops = []
    last_drop = -9999
    lookback = int(3 * k)
    for i in range(lookback, n - k):
        prior = smooth[i - lookback : i].mean()
        here = smooth[i : i + k].mean()
        if here - prior > 0.35 and here > 0.6 and i - last_drop > 4 * k:
            drops.append(i)
            last_drop = i
    return drops


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) != 3:
        print("usage: analyze_audio.py input.wav output.json", file=sys.stderr)
        sys.exit(2)
    wav = Path(sys.argv[1])
    out = Path(sys.argv[2])
    L, R, sr = load_wav_stereo(wav)
    if sr != 44100:
        print(
            f"WARNING: sample rate is {sr} Hz — Tesla requires 44100 Hz. "
            "The generated .fseq will still play but may not sync.",
            file=sys.stderr,
        )
    analysis = analyze(L, R, sr)
    out.write_text(json.dumps(analysis))
    dur = analysis["n_frames"] * FRAME_MS / 1000.0
    stereo_mean = float(np.mean(analysis["stereo_width"]))
    print(
        f"Analyzed {wav.name}: {dur:.1f}s, {analysis['n_frames']} frames, "
        f"tempo ≈ {analysis['tempo_bpm']:.1f} BPM, "
        f"{len(analysis['beat_frames'])} beats, "
        f"{len(analysis['drop_frames'])} drops, "
        f"stereo_width≈{stereo_mean:.2f}, "
        f"analyzer={analysis['analyzer']}"
    )


if __name__ == "__main__":
    main()
