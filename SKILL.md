---
name: tesla-light-show
description: Analyze a .wav audio file and generate a Tesla custom light show (.fseq) that the Tesla Toybox Light Show app can play. Use this skill when the user wants to turn a song into a Tesla light show, create or design an .fseq for their Tesla, choreograph lights/closures to music, or asks about generating Tesla xLights sequences from music. Supports Model 3, Model S, Model X, Model Y, and Cybertruck with model-specific optimization (boolean vs ramping channels, Falcon Doors / Front Doors on Model X, Cybertruck light bars + full-brightness controls + 200-channel extended output).
---

# Tesla Light Show Generator

Turn any `.wav` file into a compelling, model-tailored Tesla custom light show. The output is a validator-clean FSEQ v2.0 uncompressed binary (`.fseq`) paired with the original audio, ready to put in a `LightShow/` folder on a USB drive.

## Inputs and model selection

The skill needs two things:

1. **A path to a `.wav` file** (must be 44.1 kHz — warn the user if it is not; output will still be written but may drift on the vehicle).
2. **The Tesla model.** If the user didn't state it in the prompt, ask which of these they have — **do not guess**:
   - `Model 3`
   - `Model S`
   - `Model X`
   - `Model Y`
   - `Cybertruck`

Accept natural variations ("cybertruck", "model s 2022", "3", "Y", "x", "plaid x", etc.) and map them to the five values.

## What the skill produces

For each song it produces two files in the user's current directory (or an `output/` subdir, ask if unclear):

- `<basename>.fseq` — the binary light-show sequence.
- `<basename>.wav` — a copy of the audio (Tesla requires a filename-matched audio file to live in the same directory).

The `.fseq` has:

- Magic bytes `PSEQ`, version 2.0, uncompressed.
- Frame rate 50 fps (step time 20 ms).
- **48 channels** for Model 3 / S / Y shows.
- **200 channels** for Cybertruck shows (activates front/rear light bars, offroad bar, interior RGB).
- Duration exactly matching the source audio, clipped to 4 h max.

The validator script in `light-show/validator.py` MUST accept the output (channel count 48 or 200, compression 0, frames × step ≤ 4 h).

## How to generate a show

Work in the project directory. The scripts live in this skill's `scripts/` folder.

### Step 1 — analyze the audio

```bash
python3 skill/scripts/analyze_audio.py <path/to/song.wav> /tmp/<basename>.json
```

This writes a JSON timeline sampled at 20 ms with:
- `rms`, `bass`, `low_mid`, `mid`, `high`, `onset`, `brightness` (0..1 arrays)
- `beat_frames` and `strong_beat_frames` (downbeats)
- `drop_frames` (loudness-jump moments — use for big reveals)
- `tempo_bpm`

### Step 2 — compose the show

```bash
python3 skill/scripts/compose_show.py --analysis /tmp/<basename>.json \
  --model {model_3|model_s|model_x|model_y|cybertruck} --out <basename>.fseq
```

Normalize the model argument to one of: `model_3`, `model_s`, `model_x`, `model_y`, `cybertruck`.

The composer binds audio features to model-appropriate channels (see `references/model_capabilities.md` and `references/channel_map.md`). It respects closure actuation limits and pre-opens liftgates/charge ports in time for dances at drops.

### Step 3 — copy the wav next to the fseq

Tesla requires the audio file next to the .fseq with the **same basename**. Either copy, symlink, or tell the user to do so — default: copy.

```bash
cp <path/to/song.wav> <basename>.wav
```

### Step 4 — validate

```bash
python3 light-show/validator.py <basename>.fseq
```

Expected: `Found <N> frames, step time of 20 ms for a total duration of <H:MM:SS.ffffff>.`
If the validator prints a warning or error, **fix it before declaring success** — a non-clean file will not play on the vehicle.

### Step 5 — report

Tell the user:
- Where the files are
- Duration, tempo, channel count
- A short written description of the choreography highlights (which lights fire on which musical features — they will appreciate the intent)
- How to install: put both files under a `LightShow/` folder on a FAT32/exFAT USB drive (not NTFS, no `TeslaCam/` folder), then Toybox → Light Show → Schedule Show.

## Channel binding rules (summary)

Full tables are in `references/channel_map.md` and `references/model_capabilities.md`. Quick reference for composition:

- **Kick / bass-heavy beats** → Main beams (outer + inner), front turn signals, brake lights, Cybertruck bed lights.
- **Snare / mid-heavy beats** → Signature, Channels 4-6, rear turn signals, tail lights.
- **Hi-hats / high-band onsets** → Side markers, side repeaters, fog (on non-CT), license plate — use sparsely for sparkle.
- **Sustained loudness (RMS)** → Interior RGB wash (CT only in 200-ch), Cybertruck light bar brightness envelope.
- **Drops** → Pre-open liftgate ~14 s before, **Falcon Doors ~25 s before and Front Doors ~25 s before on Model X**, charge port Dance through the chorus (rainbow), mirror wave 2 s before, full-front blast at the drop, light-bar curtain sweep on Cybertruck.
- **Brightness / spectral centroid** → Hue mapping for RGB (warm when dark, cool when bright).
- **Alternating Left/Right on soft beats** to build stereo motion.
- Use ramping variants (70/80/90%) on Model 3 / Y / Cybertruck where smooth fades read better than boolean snaps.

## Model-specific reminders

- **Model 3 / Model Y** — exploit the ramping on Front Turn, Signature, and Channels 4-6. Prefer ramp_pulse over instant pulses for anything lasting > 200 ms. Aux Park / Side Markers are OR'd together — don't stack them continuously or they won't appear to flash.
- **Model S** — Signature and Front Turn are boolean only; use crisp hits. Door Handles (4 independent) are a unique accent — pop them 1.2 s before a drop and close 0.5 s after for a cool reveal. 20-actuation budget is generous. No Falcon or Front Doors.
- **Model X** — Same boolean headlights as Model S (no ramping on Signature / Front Turn) but with **Falcon Doors** and **Front Doors** for the most dramatic choreography of any car. Falcon Doors open in ~20 s and close in ~8 s; Front Doors open in ~22 s and close in ~3 s — schedule Opens **~25 s before a drop** so doors are fully open for the Dance/Close beat. Only **6 actuations** each per show, use them for headline moments. Model X also has **Rear Fog (even in NA)** — an extra accent channel the other US cars don't have. No Door Handles (those are S-only).
- **Cybertruck** — Generate 200-channel output. Animate the front/rear light bars (curtain, chase, bars, full styles). Brake + Rear Turn are full-brightness controlled: drive them with the RMS envelope instead of 0/100 values. Bed Lights always ramp 500 ms regardless of request.
- **Cybertruck closures** — Liftgate is mapped to Powered Frunk; Aux Park is mapped to Frunk Light; Rear Turn Signals are disabled on the car (leave the xLights channels 0). The 48→200 mapping happens in the composer automatically.

## Constraints the output must satisfy

- Channel count: exactly 48 or 200 (`validator.py` rejects otherwise).
- Compression type byte: 0 (uncompressed).
- Max duration: 4 h. Longer audio should be truncated or the user should trim.
- Sample rate: 44.1 kHz. Warn if not.
- Frame interval: 20 ms (supported: 15–100 ms, but 20 ms is every example and recommended).
- Closure actuation limits per show (count only Open/Close/Dance):
  - Liftgate / Frunk ≤ 6, Mirrors ≤ 20, Charge Port ≤ 3, Windows ≤ 6, Door Handles (S) ≤ 20, Front Doors (X) ≤ 6, Falcon Doors (X) ≤ 6.
  - Total dance time ≤ ~30 s per show (thermal).
- Dance only works when a closure is already open — schedule an Open earlier and leave time per `references/channel_map.md` (Closure Movement Durations).
- Don't close windows during the show — music would be muffled.

## If something breaks

- **Validator rejects channel count** → ensure the writer uses 48 for M3/S/Y and 200 for Cybertruck. No other value is accepted.
- **Validator rejects compression** → header byte at offset 20 must be 0.
- **Validator rejects duration** → clip the audio or reduce frame count.
- **Show plays but a specific light is stuck on** → check the Aux Park / Side Marker OR'ing rules (a single channel staying on keeps the group lit).
- **Dance did nothing** → the closure wasn't open before Dance was commanded. Insert an Open at least (durations in closure table) before the Dance effect.

## Files in this skill

- `SKILL.md` — this file.
- `scripts/analyze_audio.py` — audio feature extractor (numpy preferred; pure-Python fallback).
- `scripts/fseq_writer.py` — minimal FSEQ v2.0 uncompressed writer + level/command constants.
- `scripts/compose_show.py` — binds audio features to Tesla channels per model and writes the .fseq.
- `references/channel_map.md` — complete 48/200 channel layout, brightness byte table, closure command bytes.
- `references/model_capabilities.md` — per-model capability matrix and production recipes.
- `references/fseq_format.md` — binary layout of the FSEQ v2.0 uncompressed header.
