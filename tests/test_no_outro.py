"""The show has no outro: lights end with the music.

Run from the repo root:
    python3 -m unittest discover -s tests -v
(or `pytest tests`). Needs numpy; no audio analysis deps.
"""
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import compose_show  # noqa: E402
from compose_show import CH, Composer  # noqa: E402

MODELS = ["model_3", "model_3_highland", "model_s", "model_x", "model_y", "cybertruck"]

# Exterior light channels the old _outro_burst_layer blasted.
LIGHT_CHANNELS = [
    CH[k] for k in (
        "L_OUTER_BEAM", "R_OUTER_BEAM", "L_INNER_BEAM", "R_INNER_BEAM",
        "L_SIGNATURE", "R_SIGNATURE", "L_CH4", "R_CH4", "L_CH5", "R_CH5",
        "L_CH6", "R_CH6", "L_FRONT_TURN", "R_FRONT_TURN",
        "L_FRONT_FOG", "R_FRONT_FOG", "L_AUX_PARK", "R_AUX_PARK",
        "L_SIDE_MARKER", "R_SIDE_MARKER", "L_SIDE_REPEATER", "R_SIDE_REPEATER",
        "BRAKE", "L_TAIL", "R_TAIL", "REVERSE", "REAR_FOG", "LICENSE",
        "L_REAR_TURN", "R_REAR_TURN",
    )
]

N = 3000          # 60 s @ 20 ms
CLIMAX = (1000, 1800)
# Lights written by the last climax-section beat may trail it by a few
# frames (sparkle delay + hold); nothing past this is allowed.
SPILL_FRAMES = 25


def make_analysis(music_start=0, music_end=N, segments=True):
    """Loud, percussive song from start to finish — including a hot final
    quarter that used to trigger the outro burst."""
    def wave(period, lo=0.5, hi=1.0):
        return [lo + (hi - lo) * (0.5 + 0.5 * math.sin(2 * math.pi * f / period))
                for f in range(N)]

    beats = list(range(10, N, 25))
    a = {
        "n_frames": N,
        "tempo_bpm": 120.0,
        "beat_frames": beats,
        "strong_beat_frames": beats[::4],
        "drop_frames": [],
        # quiet-ish until frame 2000, then pinned hot (drum-roll finale)
        "rms": [r if f < 2000 else 1.0 for f, r in enumerate(wave(150, 0.2, 0.6))],
        "bass": wave(50),
        "low_mid": wave(70),
        "mid": wave(90),
        "high": wave(30, 0.7, 1.2),
        "onset": wave(25, 0.6, 1.2),
        "brightness": wave(400, 0.2, 0.8),
        "perc_onset": [p if f < 2000 else 1.0 for f, p in enumerate(wave(25, 0.1, 0.4))],
        "music_start": music_start,
        "music_end": music_end,
        "onset_frames": list(range(max(1, music_start), N - 1, 12)),
        "onset_strength": [0.9] * len(range(max(1, music_start), N - 1, 12)),
    }
    if segments:
        a["segments"] = [
            {"start": 0, "end": 300, "label": "intro", "energy": 0.3, "index": 0},
            {"start": 300, "end": CLIMAX[0], "label": "verse", "energy": 0.5, "index": 1},
            {"start": CLIMAX[0], "end": CLIMAX[1], "label": "chorus", "energy": 0.9, "index": 2},
            {"start": CLIMAX[1], "end": N, "label": "outro", "energy": 0.8, "index": 3},
        ]
    return a


def compose(analysis, model):
    c = Composer(analysis, model)
    c.compose()
    return c


def frame(c, f):
    return c.w.frames[f * c.channels:(f + 1) * c.channels]


def lit_lights(c, f):
    row = frame(c, f)
    return sum(1 for ch in LIGHT_CHANNELS if row[ch - 1] > 0)


class NoOutroTests(unittest.TestCase):
    def test_outro_burst_layer_removed(self):
        self.assertFalse(hasattr(Composer, "_outro_burst_layer"))

    def test_no_post_climax_section(self):
        c = compose(make_analysis(), "model_3")
        self.assertEqual(c._section(c.climax_end), "post")
        self.assertNotIn("outro", {c._section(f) for f in range(N)})

    def test_no_activity_after_music_end(self):
        for model in MODELS:
            for music_end in (2700, N):
                with self.subTest(model=model, music_end=music_end):
                    c = compose(make_analysis(music_start=100, music_end=music_end), model)
                    tail = c.w.frames[music_end * c.channels:]
                    self.assertFalse(any(tail), "bytes written after music_end")
                    last_lit = max(
                        (f for f in range(N) if any(frame(c, f))), default=-1
                    )
                    self.assertLess(last_lit, music_end)

    def test_no_activity_before_music_start(self):
        c = compose(make_analysis(music_start=100, music_end=2700), "model_3")
        self.assertFalse(any(c.w.frames[:100 * c.channels]))

    def test_fseq_length_tracks_audio(self):
        for music_end in (2700, N):
            c = compose(make_analysis(music_start=0, music_end=music_end), "model_3")
            self.assertEqual(c.w.frame_count, N)

    def test_lights_off_after_climax(self):
        # Hot final quarter + dense onsets: old code blasted every light here.
        for segments in (True, False):
            for model in MODELS:
                with self.subTest(model=model, segments=segments):
                    c = compose(make_analysis(segments=segments), model)
                    start = c.climax_end + SPILL_FRAMES
                    for f in range(start, N):
                        self.assertEqual(lit_lights(c, f), 0, f"lights on at frame {f}")

    def test_no_all_lights_blast_in_final_quarter(self):
        c = compose(make_analysis(music_start=100, music_end=2700), "model_3")
        for f in range(int(N * 0.75), 2700):
            self.assertLess(lit_lights(c, f), len(LIGHT_CHANNELS) // 2, f"blast at frame {f}")


if __name__ == "__main__":
    unittest.main()
