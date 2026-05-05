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

try:
    from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor  # type: ignore
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor  # type: ignore
    HAVE_MADMOM = True
except Exception:
    HAVE_MADMOM = False


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

    # --- beats / tempo / downbeats -----------------------------------------
    # Dual-tracker: run librosa and madmom beat trackers independently and
    # cross-check. Each detected beat gets a confidence score:
    #   2.0  both trackers agree within ±40 ms
    #   1.0  madmom-only beat (madmom usually wins on accuracy)
    #   0.7  librosa-only beat
    #   +0.5 bonus if coincides with a strong perc_onset spike
    # madmom also gives us real downbeats (not "every 4th beat" guesses).
    beat_data = dual_tracker_beats(L, R, sr, hop, n_frames, perc_onset)
    tempo_bpm = beat_data["tempo_bpm"]
    beat_frames = beat_data["beat_frames"]
    beat_confidence = beat_data["beat_confidence"]
    downbeat_frames = beat_data["downbeat_frames"]
    meter = beat_data["meter"]  # 3 or 4 for 3/4 or 4/4
    agreement_pct = beat_data["agreement_pct"]

    # --- drops --------------------------------------------------------------
    drops = detect_drops(rms_n)

    # --- chroma + key -------------------------------------------------------
    # Chroma is a 12-dim pitch-class profile per frame (C, C#, D, ..., B).
    # Used to drive interior RGB hue from actual harmony. Key/mode detection
    # gives us a whole-song or sliding-window tonic so hue mapping can be
    # key-relative instead of absolute.
    #
    # Both require librosa. If librosa is unavailable we fall back to
    # pseudo-chroma derived from our mel bands (less accurate but keeps
    # the feature usable).
    chroma, key_tonic, key_mode = chroma_and_key(L, R, sr, hop, n_frames, mag_mono, freqs)

    # --- structural segmentation -------------------------------------------
    # Replace the hardcoded 15/55/85% narrative split with real section
    # boundaries detected from the song's self-similarity matrix.
    # Section count adapts to song length: short songs get fewer segments.
    segments = detect_segments(mag_mono, chroma, n_frames, hop, sr, rms_n)

    # --- two-pass energy re-normalisation ----------------------------------
    # Pass 1 normalised against global 99th-percentile, which means a quiet
    # intro next to a loud chorus gets squashed. Re-normalise each section
    # against its own local 99th percentile so verse-level dynamics survive.
    rms_local = local_normalise(rms_n, segments)
    bass_local = local_normalise(band_out["bass"], segments)
    mid_local = local_normalise(band_out["mid"], segments)
    high_local = local_normalise(band_out["high"], segments)

    # Pick "strong" beats. If madmom gave us real downbeats, use those.
    # Otherwise fall back to high-confidence beats (≥ 1.5 = both trackers
    # agreed OR madmom + HPSS agreed). As a last resort: every Nth beat
    # where N = detected meter.
    if downbeat_frames:
        strong_frames = downbeat_frames
    elif beat_confidence:
        strong_frames = [
            b for b, c in zip(beat_frames, beat_confidence) if c >= 1.5
        ]
        if not strong_frames:
            strong_frames = beat_frames[::max(2, meter)]
    else:
        strong_frames = beat_frames[::max(2, meter)]

    analyzer_stack = []
    if HAVE_LIBROSA:
        analyzer_stack.append("librosa")
    if HAVE_MADMOM:
        analyzer_stack.append("madmom")
    if not analyzer_stack:
        analyzer_stack.append("numpy-stereo")

    result = {
        "sr": int(sr),
        "hop": int(hop),
        "frame_ms": FRAME_MS,
        "n_frames": int(n_frames),
        "analyzer": analyzer_stack[0],  # backward compat
        "analyzer_stack": analyzer_stack,
        "tempo_bpm": float(tempo_bpm),
        "meter": int(meter),
        "beat_agreement_pct": float(agreement_pct),
        "rms": rms_n.tolist(),
        "rms_local": rms_local.tolist(),
        "bass": band_out["bass"].tolist(),
        "bass_local": bass_local.tolist(),
        "low_mid": band_out["low_mid"].tolist(),
        "mid": band_out["mid"].tolist(),
        "mid_local": mid_local.tolist(),
        "high": band_out["high"].tolist(),
        "high_local": high_local.tolist(),
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
        "beat_confidence": [float(c) for c in beat_confidence],
        "strong_beat_frames": [int(b) for b in strong_frames],
        "downbeat_frames": [int(b) for b in downbeat_frames],
        "drop_frames": [int(d) for d in drops],
        "segments": segments,  # list of {"start", "end", "label", "energy"}
        "chroma": chroma.tolist(),
        "key_tonic": [int(t) for t in key_tonic],
        "key_mode": [int(m) for m in key_mode],
    }
    return result


def chroma_and_key(L, R, sr, hop, n_frames, mag_mono, freqs):
    """Compute a per-frame chroma vector (n_frames × 12) and sliding-window
    key detection (tonic 0..11, mode 0=minor / 1=major).

    Prefers librosa for both (chroma_cqt is robust and much cleaner than
    FFT-based chroma). Falls back to a simple FFT-based chroma if librosa
    is unavailable.
    """
    # 1) Chroma
    if HAVE_LIBROSA:
        try:
            y = 0.5 * (L + R)
            # chroma_cqt gives a 12 × T matrix; transpose to (T, 12).
            c = librosa.feature.chroma_cqt(
                y=y.astype(np.float32), sr=sr, hop_length=hop, n_chroma=12
            ).T
            # librosa may return a different T depending on centered framing;
            # align to n_frames by truncation/padding.
            if c.shape[0] > n_frames:
                c = c[:n_frames]
            elif c.shape[0] < n_frames:
                pad = np.zeros((n_frames - c.shape[0], 12), dtype=c.dtype)
                c = np.concatenate([c, pad], axis=0)
            chroma = c.astype(np.float32)
        except Exception:
            chroma = _fft_chroma(mag_mono, freqs)
    else:
        chroma = _fft_chroma(mag_mono, freqs)

    # Normalize each frame to sum-to-1 so chroma[t] is a probability-like
    # distribution over pitch classes.
    chroma = chroma / (chroma.sum(axis=1, keepdims=True) + 1e-9)

    # 2) Key + mode (Krumhansl-Schmuckler profiles, sliding window)
    key_tonic, key_mode = _key_detection_sliding(chroma, hop, sr)
    return chroma, key_tonic, key_mode


def _fft_chroma(mag_mono, freqs):
    """Pseudo-chroma from FFT magnitudes. Maps each positive frequency to
    its pitch class (note modulo 12) and sums magnitudes into 12 bins."""
    # Ignore sub-audible; any f < ~27 Hz is below musical range (A0 = 27.5)
    n_frames, n_bins = mag_mono.shape
    chroma = np.zeros((n_frames, 12), dtype=np.float32)
    # pitch class per bin: 12 * log2(f / 440) + 9 mod 12   (A = 9)
    valid = freqs > 27.0
    pc_all = np.zeros_like(freqs)
    pc_all[valid] = (12.0 * np.log2(freqs[valid] / 440.0) + 9.0) % 12.0
    pc_int = np.floor(pc_all).astype(int) % 12
    for pc in range(12):
        mask = valid & (pc_int == pc)
        if mask.any():
            chroma[:, pc] = mag_mono[:, mask].sum(axis=1)
    return chroma


# Krumhansl-Schmuckler major/minor profiles (standard reference values).
_KS_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
_KS_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)


def _key_detection_sliding(chroma: np.ndarray, hop: int, sr: int, window_s: float = 15.0):
    """Per-frame (tonic, mode) estimate via a ~15 s sliding window. For each
    window we correlate the averaged chroma against 24 key templates (12
    major + 12 minor rotations of the Krumhansl-Schmuckler profile) and pick
    the highest.

    Returns two arrays of length n_frames:
        tonic: 0..11 (0 = C, 1 = C#, ..., 11 = B)
        mode:  0 = minor, 1 = major
    """
    n_frames = chroma.shape[0]
    frames_per_sec = sr / hop
    win = int(window_s * frames_per_sec)
    win = max(win, 30)
    hop_win = max(1, win // 4)

    tonic = np.zeros(n_frames, dtype=np.int16)
    mode = np.zeros(n_frames, dtype=np.int16)
    # Precompute all 24 rotated templates.
    templates = np.zeros((24, 12), dtype=np.float32)
    for i in range(12):
        templates[i] = np.roll(_KS_MAJOR, i)       # major keys 0..11
        templates[12 + i] = np.roll(_KS_MINOR, i)  # minor keys 12..23
    # normalize
    templates = (templates - templates.mean(axis=1, keepdims=True))
    templates = templates / (np.linalg.norm(templates, axis=1, keepdims=True) + 1e-9)

    # Compute correlation for each window center, then interpolate per frame.
    centers = list(range(0, n_frames, hop_win))
    if centers[-1] != n_frames - 1:
        centers.append(n_frames - 1)
    key_choices = []
    for c in centers:
        lo = max(0, c - win // 2)
        hi = min(n_frames, c + win // 2)
        avg = chroma[lo:hi].mean(axis=0)
        avg = avg - avg.mean()
        norm = np.linalg.norm(avg) + 1e-9
        avg = avg / norm
        scores = templates @ avg  # length 24
        best = int(np.argmax(scores))
        key_choices.append(best)
    # Fill each frame with the nearest window center's choice.
    for i, c in enumerate(centers):
        lo = centers[i - 1] if i > 0 else 0
        hi = centers[i + 1] if i + 1 < len(centers) else n_frames
        mid = (lo + c) // 2 if i > 0 else 0
        end = (c + hi) // 2 if i + 1 < len(centers) else n_frames
        best = key_choices[i]
        t, m = best % 12, 1 if best < 12 else 0
        tonic[mid:end] = t
        mode[mid:end] = m
    return tonic, mode


# ---------------------------------------------------------------------------
# Dual-tracker beat detection (librosa + madmom cross-check)
# ---------------------------------------------------------------------------

def dual_tracker_beats(L, R, sr, hop, n_frames, perc_onset):
    """Run both beat trackers and fuse the results.

    Returns a dict with:
        beat_frames         merged beat positions (frame indices)
        beat_confidence     per-beat confidence 0.7..2.5
        downbeat_frames     madmom downbeats, empty if unavailable
        tempo_bpm           tempo estimate (prefers madmom)
        meter               3 or 4 (from madmom downbeats, default 4)
        agreement_pct       % of beats both trackers agreed on (0..100)
    """
    frame_sr = sr / hop  # frames per second
    tol_frames = int(0.040 * frame_sr)  # ±40 ms tolerance for agreement

    librosa_beats = []
    librosa_tempo = 120.0
    if HAVE_LIBROSA:
        try:
            y = 0.5 * (L + R)
            tempo, frames = librosa.beat.beat_track(
                y=y.astype(np.float32), sr=sr, hop_length=hop, units="frames"
            )
            librosa_tempo = float(np.atleast_1d(tempo)[0])
            librosa_beats = [int(b) for b in frames]
        except Exception:
            pass

    madmom_beats = []
    madmom_downbeats = []
    madmom_tempo = 0.0
    madmom_meter = 4
    if HAVE_MADMOM:
        try:
            mono = (0.5 * (L + R)).astype(np.float32)
            # madmom needs 44100 Hz mono float32
            if sr != 44100:
                # madmom can handle other rates but its pretrained models were
                # trained on 44.1 kHz, so degrade gracefully.
                raise RuntimeError("madmom expects 44100 Hz input")
            # RNN beat activation → DBN post-processing for final beat times.
            rnn_beat = RNNBeatProcessor()
            dbn_beat = DBNBeatTrackingProcessor(
                min_bpm=60.0, max_bpm=200.0, fps=100
            )
            act = rnn_beat(mono)
            beat_times = dbn_beat(act)
            madmom_beats = [int(round(t * frame_sr)) for t in beat_times]

            # Downbeats (same song, but separate RNN+DBN pipeline).
            rnn_db = RNNDownBeatProcessor()
            dbn_db = DBNDownBeatTrackingProcessor(
                beats_per_bar=[3, 4], fps=100
            )
            act_db = rnn_db(mono)
            db_out = dbn_db(act_db)
            # db_out is a Nx2 array: (time_s, beat_position_in_bar). Position
            # 1 = downbeat. Meter = number of unique positions per bar.
            if len(db_out):
                madmom_downbeats = [
                    int(round(t * frame_sr))
                    for t, pos in db_out
                    if int(round(pos)) == 1
                ]
                unique_positions = sorted({int(round(p)) for _, p in db_out})
                if unique_positions:
                    madmom_meter = max(unique_positions)

            # Tempo from DBN beats (median inter-beat interval).
            if len(beat_times) >= 2:
                intervals = np.diff(beat_times)
                if len(intervals) > 0:
                    madmom_tempo = float(60.0 / np.median(intervals))
        except Exception as e:
            # Don't crash the whole analysis if madmom has a bad day
            print(f"  (madmom failed: {e}; falling back to librosa only)",
                  file=sys.stderr)

    # Prefer madmom's tempo if we have it, else librosa.
    if madmom_tempo > 0:
        tempo_bpm = madmom_tempo
    elif librosa_beats:
        tempo_bpm = librosa_tempo
    else:
        tempo_bpm = 120.0
    meter = madmom_meter if madmom_beats else 4

    # Merge beats from both sources. Strategy:
    #   For each madmom beat: find closest librosa beat within ±tol.
    #     Found → confidence 2.0, average of the two positions
    #     Not found → confidence 1.0, madmom position
    #   Remaining unmatched librosa beats → confidence 0.7
    #   Bonus +0.5 if the merged position coincides with a perc_onset peak.
    merged = []
    used_librosa = set()

    def nearest(target, pool, used):
        best_i, best_d = -1, tol_frames + 1
        for i, p in enumerate(pool):
            if i in used:
                continue
            d = abs(p - target)
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    agreed = 0
    for mb in madmom_beats:
        i, d = nearest(mb, librosa_beats, used_librosa)
        if i >= 0 and d <= tol_frames:
            pos = (mb + librosa_beats[i]) // 2
            conf = 2.0
            used_librosa.add(i)
            agreed += 1
        else:
            pos = mb
            conf = 1.0
        merged.append((pos, conf))

    for i, lb in enumerate(librosa_beats):
        if i in used_librosa:
            continue
        merged.append((lb, 0.7))

    # If we have neither tracker's beats, fall back to numpy autocorrelation.
    if not merged:
        tempo_bpm, np_beats = _beats_numpy(perc_onset, hop, sr)
        merged = [(b, 0.7) for b in np_beats]

    # Sort by frame position and apply perc_onset bonus.
    merged.sort(key=lambda x: x[0])
    window = max(1, int(0.030 * frame_sr))  # ±30 ms window
    for idx, (pos, conf) in enumerate(merged):
        lo = max(0, pos - window)
        hi = min(n_frames, pos + window + 1)
        if hi > lo and np.max(perc_onset[lo:hi]) > 0.65:
            merged[idx] = (pos, conf + 0.5)

    beat_frames = [m[0] for m in merged if 0 <= m[0] < n_frames]
    beat_confidence = [m[1] for m in merged if 0 <= m[0] < n_frames]

    total = max(len(madmom_beats), len(librosa_beats), 1)
    agreement_pct = 100.0 * agreed / total

    return {
        "tempo_bpm": float(tempo_bpm),
        "meter": int(meter),
        "beat_frames": beat_frames,
        "beat_confidence": beat_confidence,
        "downbeat_frames": [b for b in madmom_downbeats if 0 <= b < n_frames],
        "agreement_pct": agreement_pct,
    }


# ---------------------------------------------------------------------------
# Structural segmentation (replace hardcoded 15/55/85% narrative arc)
# ---------------------------------------------------------------------------

def detect_segments(mag_mono, chroma, n_frames, hop, sr, rms_n):
    """Detect verse/chorus/bridge boundaries from the song's self-similarity.

    Uses librosa.segment.agglomerative on stacked chroma + MFCC features
    when librosa is available. Section count adapts to song length.

    Returns a list of dicts:
        [{"start": frame_int, "end": frame_int, "label": str,
          "energy": float, "index": int}, ...]

    Labels: "intro", "verse", "chorus", "bridge", "outro" inferred by
    energy pattern and position in the song. Fallback labels are
    "section_0", "section_1", ... if librosa isn't available.
    """
    duration_s = n_frames * FRAME_MS / 1000.0
    # Adaptive segment count
    if duration_s < 90:
        target = 4
    elif duration_s < 240:
        target = 6
    else:
        target = 8
    target = min(target, max(3, n_frames // 250))  # sanity bounds

    boundaries = None
    if HAVE_LIBROSA:
        try:
            # Build a compact feature stack: 12-dim chroma + 13-dim MFCC,
            # both smoothed. librosa does the heavy lifting.
            hop_l = int(hop)
            # Compute MFCC at our frame grid
            y_mfcc_frames = librosa.feature.mfcc(
                S=librosa.amplitude_to_db(mag_mono.T + 1e-9),
                n_mfcc=13,
            )
            feat = np.vstack([chroma.T, y_mfcc_frames])
            # agglomerative returns boundary frame indices in its own grid.
            # Since we built the features on our hop, the grid matches.
            boundaries = librosa.segment.agglomerative(feat, k=target + 1)
            boundaries = [int(b) for b in boundaries]
            # Ensure 0 and n_frames are endpoints
            if 0 not in boundaries:
                boundaries = [0] + boundaries
            if boundaries[-1] < n_frames:
                boundaries.append(n_frames)
            boundaries = sorted(set(boundaries))
        except Exception as e:
            print(f"  (segmentation failed: {e}; using uniform split)",
                  file=sys.stderr)
            boundaries = None

    if boundaries is None:
        # Uniform fallback split
        step = n_frames // target
        boundaries = list(range(0, n_frames, step)) + [n_frames]
        boundaries = sorted(set(boundaries))

    # Build segments with energy stats
    raw = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e <= s:
            continue
        energy = float(np.mean(rms_n[s:e]))
        raw.append({"start": s, "end": e, "energy": energy})

    # Label by heuristics: highest-energy repeated pattern = chorus,
    # first/last low-energy segments = intro/outro, else verse/bridge.
    if not raw:
        return []
    n_seg = len(raw)
    energies = [s["energy"] for s in raw]
    max_e = max(energies)
    min_e = min(energies)
    threshold_hi = min_e + 0.65 * (max_e - min_e)
    threshold_lo = min_e + 0.25 * (max_e - min_e)

    labels = [None] * n_seg

    # Intro: first 1-2 low-energy segments
    if raw[0]["energy"] < threshold_lo:
        labels[0] = "intro"
        if n_seg > 1 and raw[1]["energy"] < threshold_lo:
            labels[1] = "intro"

    # Outro: last 1-2 low/medium-energy segments
    if raw[-1]["energy"] < threshold_hi:
        labels[-1] = "outro"
        if n_seg > 1 and raw[-2]["energy"] < threshold_hi and labels[-2] is None:
            labels[-2] = "outro"

    # Choruses = high-energy segments
    for i, seg in enumerate(raw):
        if labels[i] is None and seg["energy"] >= threshold_hi:
            labels[i] = "chorus"

    # Bridges = a single medium-energy segment between choruses late in song
    for i in range(1, n_seg - 1):
        if labels[i] is None and labels[i - 1] == "chorus" and labels[i + 1] == "chorus":
            labels[i] = "bridge"

    # Remaining = verses
    for i in range(n_seg):
        if labels[i] is None:
            labels[i] = "verse"

    return [
        {
            "index": i,
            "start": seg["start"],
            "end": seg["end"],
            "energy": seg["energy"],
            "label": labels[i],
        }
        for i, seg in enumerate(raw)
    ]


def local_normalise(x, segments):
    """Re-normalise a feature array per segment against its own 99th
    percentile. Keeps dynamics visible inside a quiet verse even when a
    loud chorus dominates the global range."""
    if not segments:
        return np.asarray(x, dtype=np.float32)
    out = np.asarray(x, dtype=np.float32).copy()
    for seg in segments:
        s, e = seg["start"], seg["end"]
        if e <= s:
            continue
        chunk = out[s:e]
        p = float(np.percentile(chunk, 99)) + 1e-9
        out[s:e] = np.clip(chunk / p, 0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Beat + tempo (legacy single-tracker — kept for backward compat)
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
    stack = ",".join(analysis.get("analyzer_stack", [analysis.get("analyzer", "?")]))
    seg_summary = " + ".join(
        f"{s['label']}({(s['end']-s['start'])*FRAME_MS/1000:.0f}s)"
        for s in analysis.get("segments", [])
    ) or "no segments"
    print(
        f"Analyzed {wav.name}: {dur:.1f}s, {analysis['n_frames']} frames, "
        f"tempo ≈ {analysis['tempo_bpm']:.1f} BPM, meter {analysis.get('meter',4)}/4, "
        f"{len(analysis['beat_frames'])} beats "
        f"({analysis.get('beat_agreement_pct',0):.0f}% tracker agreement), "
        f"{len(analysis.get('downbeat_frames', []))} downbeats, "
        f"{len(analysis['drop_frames'])} drops, "
        f"stereo_width≈{stereo_mean:.2f}, "
        f"analyzers={stack}"
    )
    print(f"  structure: {seg_summary}")


if __name__ == "__main__":
    main()
