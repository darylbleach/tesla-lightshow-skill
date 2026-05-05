# tesla-light-show

Generate Tesla custom light shows (`.fseq`) from any `.wav` audio file.

This is a [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) skill and a
standalone Python CLI. Point it at a song, tell it your Tesla model, and it produces a
validator-clean `.fseq` paired with the audio — drop both on a USB stick and you're
ready to play the show via Toybox → Light Show → Schedule Show.

Supported models:

- **Model 3** (pre-2024)
- **Model 3 Highland** (2024+ refresh — unlocks the interior accent RGB strip)
- **Model S** (2021+)
- **Model X** (2021+)
- **Model Y**
- **Cybertruck**

Highland shows are forward-compatible: they're 200-channel files that include
both the normal lighting and the interior RGB hue data. A Highland `.fseq`
plays on a pre-refresh Model 3 just fine — the firmware ignores the channels
the car doesn't have.

## What makes it different

Other show generators pick beats and flash lights. This one treats the show as a
three-act piece:

- **Narrative arc** — every show has an intro (quiet), build (ramping density),
  climax (full intensity + physical movement), and outro (calm fade). The climax
  always lands on the loudest sustained moment of the song, so the dramatic
  reveal actually lines up with the music.
- **Physical choreography at the climax** — liftgate pre-opens 14 s early,
  mirrors flap in a continuous wiper pattern, charge port does the rainbow
  Dance, and model-specific tricks fire (Model S door handles; Model X falcon
  + front doors).
- **Model-aware lighting** — uses ramping on cars that support it (Model 3/Y/CT),
  crisp boolean hits on cars that don't (S/X). Generates 200-channel output for
  Cybertruck so the light bars and interior RGB actually do something.
- **Stereo-aware audio analysis** — panning information drives which side of the
  car lights up (hi-hats panned right → right-side markers). Stereo width
  detection catches "the mix opens up" moments and fires a matching light wash.
- **HPSS percussive separation** — kick/snare beats are detected on a clean
  drums-only signal rather than inferred from EQ ratios, so non-percussive
  music (anthems, ballads) still gets good beat synchronisation.

## Requirements

- Python 3.10+
- `numpy` (required)
- `librosa` (optional but recommended — better beat tracking, especially on
  slow or non-percussive music)
- `scipy` (optional — speeds up HPSS; pure-numpy fallback included)

```bash
pip install numpy
pip install librosa  # optional, ~80 MB
```

All three are permissive / OSI-approved licences (BSD / ISC / MIT-equivalent) —
free for any use.

## Audio format

The analyzer reads `.wav` at 44.1 kHz. For anything else, convert with
[ffmpeg](https://ffmpeg.org/):

### Convert from mp3

```bash
ffmpeg -i song.mp3 -ar 44100 -ac 2 song.wav
```

### Convert a wav that's the wrong sample rate (48 kHz → 44.1 kHz)

```bash
ffmpeg -i source.wav -ar 44100 -ac 2 song.wav
```

### Convert from other formats (m4a, aac, flac, ogg, opus …)

Same pattern — ffmpeg auto-detects the input format:

```bash
ffmpeg -i song.flac  -ar 44100 -ac 2 song.wav
ffmpeg -i song.m4a   -ar 44100 -ac 2 song.wav
ffmpeg -i song.opus  -ar 44100 -ac 2 song.wav
```

Flag reference:

- `-ar 44100` → output sample rate 44.1 kHz (Tesla requirement)
- `-ac 2` → stereo output (our analyzer uses both channels for panning-aware light choreography)

## Install as a Claude Code skill

```bash
git clone https://github.com/<you>/tesla-light-show ~/.claude/skills/tesla-light-show
```

In any Claude Code session, just ask:

> Turn `my-song.wav` into a Tesla light show for my Model 3.

Claude will invoke the skill, analyse the audio, compose the show, and validate the
output. If you don't state your model, Claude will ask.

## Use as a standalone CLI

Two-step pipeline:

```bash
# Step 1 — analyse the audio (emits a JSON timeline)
python3 scripts/analyze_audio.py song.wav /tmp/song.json

# Step 2 — compose the show
python3 scripts/compose_show.py \
  --analysis /tmp/song.json \
  --model model_3 \
  --out song.fseq

# Step 3 — validate (optional but recommended)
python3 /path/to/light-show/validator.py song.fseq < /dev/null | head -1
# Success looks like: "Found <N> frames, step time of 20 ms for a total duration of ..."
```

Then copy `song.fseq` and `song.wav` into a `LightShow/` folder on a FAT32 / exFAT
USB stick (no NTFS, no `TeslaCam/` folder on the drive).

Model argument is one of: `model_3`, `model_3_highland`, `model_s`, `model_x`, `model_y`, `cybertruck`.

Pick `model_3_highland` if your car has the interior LED accent strip (all 2024+ Model 3s). The composer will drive the six RGB surfaces (center display, center accent, left/right front, left/right rear) with chroma-based colour — hue follows the song's harmony, saturation follows chord richness, brightness follows loudness, and the whole palette tilts toward warmer tones in minor keys.

## How it works

```
song.wav
   │
   ▼
┌──────────────────────┐       ┌────────────────────┐
│  analyze_audio.py    │       │  analysis.json     │
│                      │       │  (20 ms per frame) │
│  • stereo STFT       │──────▶│                    │
│  • mid/side + pan    │       │  rms bass mid high │
│  • HPSS (perc/harm)  │       │  pan_{bass,mid..}  │
│  • mel[32] bands     │       │  stereo_width      │
│  • librosa or numpy  │       │  perc_onset        │
│    beat tracking     │       │  beat_frames       │
└──────────────────────┘       │  tempo_bpm         │
                               │  drop_frames       │
                               └──────────┬─────────┘
                                          │
                                          ▼
┌────────────────────────────────────────────────────────┐
│  compose_show.py                                       │
│                                                        │
│  ┌──────────────────────────────────────┐              │
│  │ Section planner                      │              │
│  │  intro 0-15% / build 15-55% /        │              │
│  │  climax 55-85% / outro 85-100%       │              │
│  └──────────────────────────────────────┘              │
│                                                        │
│  Layers (stacked, written into the same FSEQ frames):  │
│   • Blinker layer  — yellow turn signals on beat subs  │
│   • Marker sparkle — pan-aware side marker triggers    │
│   • Stereo wash    — fog + aux park on mix "openings"  │
│   • Beat layer     — kick vs snare, model-appropriate  │
│   • Climax layer   — physical choreography (closures)  │
│   • Drop layer     — mini front-light blasts           │
│   • Interior RGB   — CT only, 200-ch extension         │
└────────────────────────────────────────────────────────┘
                                          │
                                          ▼
                                       song.fseq
                              (FSEQ v2.0 uncompressed,
                               48 or 200 channels)
```

See `references/` for the full Tesla channel map, per-model capability matrix,
and FSEQ binary format notes.

## File layout

```
SKILL.md                  — Claude Code skill entry point
README.md                 — this file
LICENSE                   — MIT
scripts/
  analyze_audio.py        — audio → features.json
  compose_show.py         — features → .fseq
  fseq_writer.py          — FSEQ v2.0 uncompressed writer
references/
  channel_map.md          — all 48/200 channels, per-model
  model_capabilities.md   — capability matrix + production recipes
  fseq_format.md          — binary header layout + validation rules
```

## Credits

Built on top of the Tesla Motors [light-show](https://github.com/teslamotors/light-show)
reference project (Tesla's official xLights guide, validator, and channel mapping).
This skill doesn't include or redistribute any Tesla code — just documents and uses
the published format.

## License

MIT — see [LICENSE](LICENSE).
