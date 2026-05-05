#!/usr/bin/env python3
"""
Compose a Tesla light show .fseq file from an analyzed audio timeline.

Inputs
------
    --analysis  JSON file from analyze_audio.py
    --model     one of: model_3, model_s, model_y, cybertruck
    --out       output .fseq path

Design goals
------------
Generate a genuinely musical, well-structured show:

* Uses model-appropriate channels (ramping vs boolean vs full-brightness).
* Binds specific musical features to specific lights:
    - Kick / bass     -> Main beams & front turn pulses, brake lights, bed lights
    - Snare / mids    -> Signature & Channel 4-6, rear turn signals, tail lights
    - Hats / highs    -> Side markers, fog, aux park (rare sparkles)
    - Loudness (RMS)  -> Interior RGB & light bar brightness (CT)
    - Onset strength  -> Strobe bursts and fast flickers
    - Beats           -> Predictable on-beat flashes for groove
    - Drops           -> Full-front "explosion" + closure choreography (liftgate, falcon doors, charge port dance)
    - Brightness      -> Color temperature / hue of RGB (dark = warm, bright = cool)
* Respects closure actuation limits and timing (open 14s before drop, etc.).
* Alternates left/right on every beat section to build stereo motion.
* Uses ramping duration on platforms that support it for smooth builds.
* For Cybertruck, animates the light bars with curtain/morph/sparkle patterns
  driven by the audio envelope.

Usage
-----
    python3 compose_show.py --analysis a.json --model model_3 --out out.fseq
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from fseq_writer import FseqWriter, LEVELS, CLOSURE, pct


# Tesla channel numbers (1-indexed) -----------------------------------------

CH = {
    "L_OUTER_BEAM": 1,
    "R_OUTER_BEAM": 2,
    "L_INNER_BEAM": 3,
    "R_INNER_BEAM": 4,
    "L_SIGNATURE": 5,
    "R_SIGNATURE": 6,
    "L_CH4": 7,
    "R_CH4": 8,
    "L_CH5": 9,
    "R_CH5": 10,
    "L_CH6": 11,
    "R_CH6": 12,
    "L_FRONT_TURN": 13,
    "R_FRONT_TURN": 14,
    "L_FRONT_FOG": 15,
    "R_FRONT_FOG": 16,
    "L_AUX_PARK": 17,
    "R_AUX_PARK": 18,
    "L_SIDE_MARKER": 19,
    "R_SIDE_MARKER": 20,
    "L_SIDE_REPEATER": 21,
    "R_SIDE_REPEATER": 22,
    "L_REAR_TURN": 23,
    "R_REAR_TURN": 24,
    "BRAKE": 25,
    "L_TAIL": 26,
    "R_TAIL": 27,
    "REVERSE": 28,
    "REAR_FOG": 29,
    "LICENSE": 30,
    "L_FALCON": 31,
    "R_FALCON": 32,
    "L_FRONT_DOOR": 33,
    "R_FRONT_DOOR": 34,
    "L_MIRROR": 35,
    "R_MIRROR": 36,
    "L_FRONT_WIN": 37,
    "L_REAR_WIN": 38,
    "R_FRONT_WIN": 39,
    "R_REAR_WIN": 40,
    "LIFTGATE": 41,
    "L_FRONT_HANDLE": 42,
    "L_REAR_HANDLE": 43,
    "R_FRONT_HANDLE": 44,
    "R_REAR_HANDLE": 45,
    "CHARGE_PORT": 46,
    # 200-channel extension (Cybertruck)
    "FRONT_BAR_START": 47,   # through 106 (60 LEDs)
    "REAR_BAR_START": 111,   # through 162 (52 LEDs)
    "OFFROAD_BAR_START": 167,  # through 172
    "SUSPENSION": 175,
    "CENTER_DISPLAY_R": 176,  # 176,177,178 = R,G,B
    "R_REAR_RGB_R": 179,
    "R_FRONT_RGB_R": 182,
    "CENTER_ACCENT_R": 185,
    "L_FRONT_RGB_R": 188,
    "L_REAR_RGB_R": 191,
}


def ms_to_frames(ms: float, step_ms: int = 20) -> int:
    return max(1, int(round(ms / step_ms)))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def hsv_to_rgb(h: float, s: float = 1.0, v: float = 1.0):
    """h in [0,1) -> (r,g,b) bytes."""
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return int(r * 255), int(g * 255), int(b * 255)


# ------------------------------------------------------------------

class Composer:
    STEP_MS = 20

    def __init__(self, analysis, model: str):
        self.a = analysis
        self.model = model
        self.n = analysis["n_frames"]
        self.tempo = analysis.get("tempo_bpm", 120.0)
        self.beats = analysis.get("beat_frames", [])
        self.strong = set(analysis.get("strong_beat_frames", []))
        self.drops = list(analysis.get("drop_frames", []))
        self.rms = analysis["rms"]
        self.bass = analysis["bass"]
        self.low_mid = analysis["low_mid"]
        self.mid = analysis["mid"]
        self.high = analysis["high"]
        self.onset = analysis["onset"]
        self.brightness = analysis["brightness"]
        # New stereo-aware features (added in the stereo analyzer pass).
        # Older analysis JSON files might not have them; fall back gracefully.
        n = analysis["n_frames"]
        self.pan_bass = analysis.get("pan_bass", [0.0] * n)
        self.pan_mid = analysis.get("pan_mid", [0.0] * n)
        self.pan_high = analysis.get("pan_high", [0.0] * n)
        self.pan_overall = analysis.get("pan_overall", [0.0] * n)
        self.stereo_width = analysis.get("stereo_width", [0.0] * n)
        self.side_energy = analysis.get("side_energy", [0.0] * n)
        self.perc_onset = analysis.get("perc_onset", analysis.get("onset", [0.0] * n))
        self.perc_rms = analysis.get("perc_rms", analysis.get("rms", [0.0] * n))
        self.harm_rms = analysis.get("harm_rms", analysis.get("rms", [0.0] * n))
        # Chroma + key (for chroma-driven interior RGB on Highland).
        self.chroma = analysis.get("chroma", None)  # list of 12-element lists or None
        self.key_tonic = analysis.get("key_tonic", [0] * n)
        self.key_mode = analysis.get("key_mode", [1] * n)  # default major
        # Per-frame tempo for songs that accelerate / decelerate (Stairway-
        # style). Falls back to a constant array if tempo_curve isn't
        # present in the analysis (backward compat).
        self.tempo_curve = analysis.get(
            "tempo_curve", [self.tempo] * n
        )
        self.tempo_change_points = analysis.get("tempo_change_points", [])
        # Music region (for trimming applause / silence from start / end)
        self.music_start = int(analysis.get("music_start", 0))
        self.music_end = int(analysis.get("music_end", n))
        # Grid-free onset peaks — every real percussive hit in the audio
        self.onset_frames = analysis.get("onset_frames", [])
        self.onset_strength = analysis.get("onset_strength", [])

        # Channel count per model:
        #   * Cybertruck: 200 (needs light bars + interior RGB).
        #   * Model 3 Highland (and any other trim with interior accent
        #     lights): also 200 so we can write channels 176-193 for
        #     interior RGB. Channels 47-175 stay zero (CT-specific hardware
        #     that Highland doesn't have — the firmware ignores them).
        #   * Everything else: 48.
        # A 200-channel file plays perfectly on a 48-channel car — the
        # firmware just ignores channels beyond what the hardware supports.
        if model == "cybertruck" or model == "model_3_highland":
            self.channels = 200
        else:
            self.channels = 48

        # Trims that share Model 3/Y's ramping lighting behaviour. Highland
        # is a Model 3 refresh — its exterior ramping capabilities are the
        # same as the base Model 3, so any per-model branch should treat
        # both identically.
        self._is_m3_or_my_family = model in ("model_3", "model_3_highland", "model_y")
        self.w = FseqWriter(self.channels, self.n, self.STEP_MS)

        # Closure budgets tracked as we place commands. Mirrors have a
        # 20-actuation budget per physical mirror, so they're tracked
        # separately as mirrors_left / mirrors_right (NOT a shared pool —
        # otherwise one mirror can run out while the other still has budget,
        # leaving them in mismatched states).
        self.closure_used = {
            "liftgate": 0,
            "mirrors_left": 0, "mirrors_right": 0,
            "charge_port": 0, "windows": 0,
            "door_handles": 0, "front_doors": 0, "falcon_doors": 0,
        }
        self.rng = random.Random(42)

    # ---- primitive effects ------------------------------------------------

    def pulse(self, ch: int, frame: int, hold_ms: int = 80, level: int = 255):
        end = frame + ms_to_frames(hold_ms, self.STEP_MS)
        self.w.set_range(frame, end, ch, level)

    def ramp_pulse(self, ch: int, frame: int, ramp: str, hold_ms: int = 400):
        level = {"500": 178, "1000": 204, "2000": 230, "instant": 255}[ramp]
        dur_map = {"500": 500, "1000": 1000, "2000": 2000, "instant": 80}
        # xlights semantics: effect duration must exceed ramp to reach 100%,
        # then blank makes it drop. We simulate that by setting the level byte
        # over (ramp+50ms, capped by hold_ms), then 0 afterwards.
        span_ms = max(hold_ms, dur_map[ramp] + 50)
        end = frame + ms_to_frames(span_ms, self.STEP_MS)
        self.w.set_range(frame, end, ch, level)

    def ramp_to_beat(self, ch: int, beat: int, ramp: str, tail_ms: int = 80,
                     max_ramp_frac_of_beat: float = 0.9):
        """Beat-aligned ramp pulse: the ramp PEAKS on the beat frame.

        A naive `ramp_pulse` starts the ramp byte AT the beat, so the LED
        reaches peak ramp_ms later — the eye locks onto the peak and the
        pulse reads as off-beat. Here we back-date the start by the ramp
        duration so the LED's swell completes exactly on the beat, then
        briefly hold peak for `tail_ms` before snapping off.

        Automatic tempo scaling: a ramp that would overlap with the
        previous beat is shrunk down. `max_ramp_frac_of_beat` is the
        most the ramp is allowed to span of the inter-beat gap
        (0.9 = 90% of a beat period). If even the shortest ramp
        ("500" = 500 ms) would blow past that budget, we fall back to an
        instant pulse, preserving the on-beat feel at any tempo.
        """
        ramp_ms_map = {"500": 500, "1000": 1000, "2000": 2000}
        level_map = {"500": 178, "1000": 204, "2000": 230}

        # Budget: how much of the upstream timeline can the ramp occupy?
        # Derived from the last preceding beat (so we don't overlap) or
        # the tempo-implied beat period if `self.beats` is empty.
        if getattr(self, "beats", None):
            prev = next((b for b in reversed(self.beats) if b < beat), None)
            gap_frames = (beat - prev) if prev is not None else beat
        else:
            gap_frames = ms_to_frames(60_000 / max(1.0, self.tempo), self.STEP_MS)
        gap_ms = max(1, gap_frames) * self.STEP_MS
        budget_ms = int(gap_ms * max_ramp_frac_of_beat)

        # Shrink ramp if the requested duration doesn't fit the budget.
        chosen = ramp
        for candidate in (ramp, "1000", "500"):
            if ramp_ms_map[candidate] <= budget_ms:
                chosen = candidate
                break
        else:
            # Nothing fits — fall back to an instant pulse ON the beat.
            end = beat + ms_to_frames(max(tail_ms, 60), self.STEP_MS)
            self.w.set_range(beat, end, ch, 255)
            return

        ramp_ms = ramp_ms_map[chosen]
        level = level_map[chosen]
        start = beat - ms_to_frames(ramp_ms, self.STEP_MS)
        # Also clamp to 0 so we never write before the show starts.
        if start < 0:
            # Not enough runway — fall back to instant pulse on-beat so the
            # show still attacks precisely on the first beat.
            end = beat + ms_to_frames(max(tail_ms, 60), self.STEP_MS)
            self.w.set_range(beat, end, ch, 255)
            return
        # Hold peak briefly past the beat for a clean "landing" then off.
        peak_end = beat + ms_to_frames(max(tail_ms, self.STEP_MS), self.STEP_MS)
        self.w.set_range(start, peak_end, ch, level)

    def local_bpm(self, frame: int) -> float:
        """Return the per-frame BPM estimate, clamped to a safe range.

        Songs like Stairway to Heaven accelerate from ~72 to ~98 BPM; this
        lets layers scale ramp durations to the *current* tempo rather than
        the whole-song median.
        """
        if 0 <= frame < len(self.tempo_curve):
            return max(40.0, min(220.0, float(self.tempo_curve[frame])))
        return max(40.0, min(220.0, float(self.tempo)))

    def tempo_ramp_choice(self, frame: int) -> str:
        """Pick a ramp duration class appropriate for the current tempo.

        Slow songs (< 70 BPM → beat > 857 ms) can sustain 2000 ms ramps
        without stepping on the next beat. Fast songs need tight 500 ms
        ramps. Between those, 1000 ms is the sweet spot.

        Returns one of the ramp keys accepted by ramp_pulse / ramp_to_beat:
        "500", "1000", or "2000".
        """
        bpm = self.local_bpm(frame)
        beat_ms = 60_000.0 / bpm
        # Use ≤ 65% of the beat so ramp finishes well before the next hit.
        budget = 0.65 * beat_ms
        if budget >= 2000:
            return "2000"
        if budget >= 1000:
            return "1000"
        return "500"

    def ramp_on_off(self, ch: int, frame: int, on_dur_ms: int, off_dur_ms: int, peak_hold_ms: int):
        """Ramp a channel up, hold, then ramp down (nice breathing effect)."""
        step = self.STEP_MS
        on_frames = ms_to_frames(on_dur_ms, step)
        hold_frames = ms_to_frames(peak_hold_ms, step)
        off_frames = ms_to_frames(off_dur_ms, step)
        # Choose the nearest ramp class
        on_level = 230 if on_dur_ms >= 1500 else (204 if on_dur_ms >= 750 else 178)
        off_level = 76 if off_dur_ms >= 1500 else (51 if off_dur_ms >= 750 else 26)
        a = frame
        b = a + on_frames
        c = b + hold_frames
        d = c + off_frames
        self.w.set_range(a, b, ch, on_level)
        self.w.set_range(b, c, ch, 255)
        self.w.set_range(c, d, ch, off_level)

    def closure(self, ch: int, frame: int, action: str, hold_ms: int = 400, budget_key: str | None = None, limit: int | None = None):
        if budget_key and limit is not None:
            if self.closure_used.get(budget_key, 0) >= limit:
                return False
            if action in ("open", "close", "dance"):
                self.closure_used[budget_key] = self.closure_used.get(budget_key, 0) + 1
        val = CLOSURE[action]
        end = frame + ms_to_frames(hold_ms, self.STEP_MS)
        self.w.set_range(frame, end, ch, val)
        return True

    # ---- higher-level patterns -------------------------------------------

    def all_front_flash(self, frame: int, hold_ms: int = 100):
        """On-beat flash of every front white channel on relevant models."""
        fronts = [
            CH["L_OUTER_BEAM"], CH["R_OUTER_BEAM"],
            CH["L_INNER_BEAM"], CH["R_INNER_BEAM"],
            CH["L_SIGNATURE"], CH["R_SIGNATURE"],
            CH["L_FRONT_TURN"], CH["R_FRONT_TURN"],
            CH["L_FRONT_FOG"], CH["R_FRONT_FOG"],
        ]
        for c in fronts:
            self.pulse(c, frame, hold_ms=hold_ms, level=255)

    def alternating_beam(self, frame: int, left: bool, hold_ms: int = 150):
        if left:
            self.pulse(CH["L_OUTER_BEAM"], frame, hold_ms=hold_ms)
            self.pulse(CH["L_INNER_BEAM"], frame, hold_ms=hold_ms)
        else:
            self.pulse(CH["R_OUTER_BEAM"], frame, hold_ms=hold_ms)
            self.pulse(CH["R_INNER_BEAM"], frame, hold_ms=hold_ms)

    def rear_beat(self, frame: int, hold_ms: int = 120):
        """Brake + tail flash on beats — with CT full-brightness gradient."""
        if self.model == "cybertruck":
            # Use envelope for nicer brake brightness
            end = frame + ms_to_frames(hold_ms, self.STEP_MS)
            for f in range(frame, min(self.n, end)):
                intensity = int(160 + 95 * self.rms[min(len(self.rms) - 1, f)])
                self.w.set(f, CH["BRAKE"], intensity)
        else:
            self.pulse(CH["BRAKE"], frame, hold_ms=hold_ms)
        self.pulse(CH["L_TAIL"], frame, hold_ms=hold_ms)
        self.pulse(CH["R_TAIL"], frame, hold_ms=hold_ms)

    def sparkle_high(self, frame: int):
        """Short flash of a random high-band channel (side markers / fog / license)."""
        pool = [
            CH["L_SIDE_MARKER"], CH["R_SIDE_MARKER"],
            CH["L_SIDE_REPEATER"], CH["R_SIDE_REPEATER"],
            CH["LICENSE"],
        ]
        if self.model != "cybertruck":
            pool += [CH["L_FRONT_FOG"], CH["R_FRONT_FOG"]]
        c = self.rng.choice(pool)
        self.pulse(c, frame, hold_ms=60, level=255)

    def rgb_wash(self, start_f: int, end_f: int, hue_fn, saturation: float = 1.0):
        """Wash all interior RGB (+ light bars on CT) with a hue function across time."""
        step = self.STEP_MS
        rgb_starts = [
            CH["CENTER_DISPLAY_R"], CH["R_REAR_RGB_R"], CH["R_FRONT_RGB_R"],
            CH["CENTER_ACCENT_R"], CH["L_FRONT_RGB_R"], CH["L_REAR_RGB_R"],
        ]
        for f in range(max(0, start_f), min(self.n, end_f)):
            frac = (f - start_f) / max(1, end_f - start_f)
            h = hue_fn(f, frac)
            v = 0.4 + 0.6 * self.rms[min(len(self.rms) - 1, f)]
            r, g, b = hsv_to_rgb(h % 1.0, saturation, min(1.0, v))
            for base in rgb_starts:
                if base + 2 <= self.channels:
                    self.w.set(f, base, r)
                    self.w.set(f, base + 1, g)
                    self.w.set(f, base + 2, b)

    def cybertruck_lightbar_sweep(self, start_f: int, end_f: int, style: str = "curtain"):
        """Animate the 60-LED front bar and 52-LED rear bar."""
        front_start = CH["FRONT_BAR_START"]
        front_count = 60
        rear_start = CH["REAR_BAR_START"]
        rear_count = 52
        for f in range(max(0, start_f), min(self.n, end_f)):
            frac = (f - start_f) / max(1, end_f - start_f)
            env = self.rms[min(len(self.rms) - 1, f)]
            peak_env = int(255 * env)
            if style == "curtain":
                edge = int(frac * front_count / 2)
                for i in range(front_count):
                    mid = front_count / 2
                    on = abs(i - mid) <= edge
                    self.w.set(f, front_start + i, peak_env if on else 0)
                edge_r = int(frac * rear_count / 2)
                for i in range(rear_count):
                    mid = rear_count / 2
                    on = abs(i - mid) <= edge_r
                    self.w.set(f, rear_start + i, peak_env if on else 0)
            elif style == "bars":
                # pulsing bands of 4 LEDs
                phase = (f * 0.25) % 4
                for i in range(front_count):
                    on = ((i + int(phase)) // 4) % 2 == 0
                    self.w.set(f, front_start + i, peak_env if on else 0)
                for i in range(rear_count):
                    on = ((i + int(phase)) // 4) % 2 == 0
                    self.w.set(f, rear_start + i, peak_env if on else 0)
            elif style == "chase":
                # single head chasing from left to right
                pos = int(frac * front_count) % front_count
                for i in range(front_count):
                    d = min(abs(i - pos), front_count - abs(i - pos))
                    val = max(0, peak_env - d * 40)
                    self.w.set(f, front_start + i, val)
                pos_r = int(frac * rear_count) % rear_count
                for i in range(rear_count):
                    d = abs(i - pos_r)
                    val = max(0, peak_env - d * 40)
                    self.w.set(f, rear_start + i, val)
            elif style == "full":
                for i in range(front_count):
                    self.w.set(f, front_start + i, peak_env)
                for i in range(rear_count):
                    self.w.set(f, rear_start + i, peak_env)

    # ---- composition -----------------------------------------------------

    def compose(self):
        self._plan_sections()
        self._baseline()
        self._blinker_layer()
        self._marker_sparkle_layer()
        self._stereo_wash_layer()
        self._beat_layer()
        self._onset_layer()
        self._outro_burst_layer()
        self._climax_choreography()
        self._drop_layer()
        self._interior_layer()
        # LAST: zero out everything before music_start and after music_end so
        # the car stays dark during applause / MC announcements / silent
        # lead-ins. Runs after all other layers so every other layer can be
        # written freely and this is the final authority on "no lights".
        self._mute_non_music_region()

    def _plan_sections(self):
        """Compute the narrative arc.

        Two modes:

        * **Music-structure mode** (preferred) — if the analyzer detected
          segments with labels (intro/verse/chorus/bridge/outro), we use
          them directly. The climax lives in the highest-energy chorus
          section; everything before it is intro/build; everything after
          is outro. This means the big moment lands on the song's actual
          peak, not a time-based proxy.

        * **Legacy fallback** — older analysis JSON without segments uses
          the original 15/55/85% split based on time.

        We also find the specific `climax_peak` frame (loudest sustained
        2 s) inside the climax region for the physical-movement timing.
        """
        import numpy as np
        n = self.n
        segments = self.a.get("segments") or []
        # Keep the raw list for the beat layer's per-segment tuning
        self.segments = segments

        if segments:
            # Pick the climax = highest-energy chorus (fallback: highest
            # energy overall). Surrounding segments become intro/build/outro.
            chorus_segs = [s for s in segments if s["label"] == "chorus"]
            pool = chorus_segs if chorus_segs else segments
            climax_seg = max(pool, key=lambda s: s["energy"])
            climax_start = climax_seg["start"]
            climax_end = climax_seg["end"]
            # Intro = up to the first non-intro segment
            first_non_intro = 0
            for i, s in enumerate(segments):
                if s["label"] != "intro":
                    first_non_intro = i
                    break
            self.intro_end = segments[first_non_intro]["start"] if first_non_intro else n // 10
            # Build = first non-intro segment → start of climax
            self.build_end = climax_start
            # Climax ends at end of climax segment
            self.climax_end = climax_end
        else:
            # Legacy fallback
            self.intro_end = int(n * 0.15)
            self.build_end = int(n * 0.55)
            self.climax_end = int(n * 0.85)

        # Tempo change points can also pull the intro/build boundary — a
        # sustained tempo shift is a structural cue that often aligns with
        # the end of an intro or the start of a build (e.g. Stairway's 72 →
        # 84 BPM transition when the drums enter). If there's a tempo shift
        # inside the intro region, respect it.
        for cp in self.tempo_change_points:
            if cp < self.intro_end and cp > n // 20:
                # Tempo shifted during the intro — move the intro end there
                self.intro_end = cp
                if self.build_end < cp:
                    self.build_end = min(n - 1, cp + n // 10)
                break

        # Guarantee sane ordering even on edge cases
        self.intro_end = max(0, min(self.intro_end, n - 1))
        self.build_end = max(self.intro_end + 1, min(self.build_end, n - 1))
        self.climax_end = max(self.build_end + 1, min(self.climax_end, n))

        # Find loudest sustained 2 s chunk inside the climax region
        win = ms_to_frames(2000, self.STEP_MS)
        rms = np.asarray(self.rms, dtype=np.float32)
        best = self.build_end + (self.climax_end - self.build_end) // 2
        if self.climax_end - self.build_end > win:
            kernel = np.ones(win, dtype=np.float32) / win
            smooth = np.convolve(rms, kernel, mode="same")
            seg = smooth[self.build_end : self.climax_end]
            best = self.build_end + int(np.argmax(seg))
        self.climax_peak = best

    def _section(self, f: int) -> str:
        if f < self.intro_end:
            return "intro"
        if f < self.build_end:
            return "build"
        if f < self.climax_end:
            return "climax"
        return "outro"

    def _section_intensity(self, f: int) -> float:
        """Multiplier 0..1 that scales how intense the beat layer is."""
        s = self._section(f)
        if s == "intro":
            # ramps 0.2 → 0.5
            return 0.2 + 0.3 * (f / max(1, self.intro_end))
        if s == "build":
            # ramps 0.5 → 0.9
            frac = (f - self.intro_end) / max(1, self.build_end - self.intro_end)
            return 0.5 + 0.4 * frac
        if s == "climax":
            return 1.0
        # outro — calm down 0.9 → 0.2
        frac = (f - self.climax_end) / max(1, self.n - self.climax_end)
        return 0.9 - 0.7 * frac

    def _baseline(self):
        """Intentionally empty — previous always-on inner-beam glow was washing
        out the show. Reference shows keep the car mostly dark with bright
        bursts; contrast is everything."""
        return

    def _stereo_wash_layer(self):
        """When the mix suddenly widens (side_energy / stereo_width spikes),
        fire a brief "wash" across both front fogs + both aux parks — this
        mimics the visual feel of a stereo opening moment (e.g. a reverb
        tail, a pad entering, strings spreading out)."""
        min_gap = ms_to_frames(4000, self.STEP_MS)
        last = -min_gap * 2
        import numpy as np
        side = np.asarray(self.side_energy, dtype=np.float32)
        width = np.asarray(self.stereo_width, dtype=np.float32)
        # ~3 s rolling baseline to detect *changes* (openings of the mix)
        k = ms_to_frames(3000, self.STEP_MS)
        if k >= len(side):
            return
        base_side = np.convolve(side, np.ones(k) / k, mode="same")
        base_width = np.convolve(width, np.ones(k) / k, mode="same")
        # Threshold: require a real jump both above baseline AND above an
        # absolute floor so we only fire on genuine "open" moments.
        for f in range(self.intro_end, self.climax_end):
            if f - last < min_gap:
                continue
            side_jump = side[f] - base_side[f] > 0.35 and side[f] > 0.6
            width_jump = width[f] - base_width[f] > 0.35 and width[f] > 0.5
            if not (side_jump or width_jump):
                continue
            hold = ms_to_frames(300, self.STEP_MS)
            if self.model != "cybertruck":
                for c in (CH["L_FRONT_FOG"], CH["R_FRONT_FOG"]):
                    self.w.set_range(f, f + hold, c, 255)
            for c in (CH["L_AUX_PARK"], CH["R_AUX_PARK"]):
                self.w.set_range(f, f + hold, c, 255)
            last = f

    def _blinker_layer(self):
        """Signature Tesla-show pattern: alternating L/R yellow turn signals
        running through musical subdivisions. This is the "yellow blinker"
        effect the user wanted more of.

        Strategy: between each pair of detected beats, fire a 2-pulse or
        4-pulse alternating blink pattern using the front turn signals
        (and rear turn if snare-heavy). Rate: build → climax only.
        """
        if len(self.beats) < 2:
            return
        for i in range(len(self.beats) - 1):
            b0 = self.beats[i]
            b1 = self.beats[i + 1]
            if b0 >= self.n or b1 >= self.n:
                break
            section = self._section(b0)
            if section == "intro":
                continue
            # Subdivision count per beat: intro 0, build 1 (on-beat only),
            # climax 3 (three blinks per beat). Previously build=2 and
            # climax=4 was too constant and washed out contrast.
            subdivs = {"build": 1, "climax": 2, "outro": 0}.get(section, 0)
            if subdivs == 0:
                continue
            beat_len = b1 - b0
            if beat_len < ms_to_frames(100, self.STEP_MS):
                continue
            sub_len = beat_len // subdivs
            # 22% of the subdivision is ON. Reference coffin-dance has turn
            # signals at ~9% duty cycle — this targets roughly that when
            # subdivs=1 (build) or higher when subdivs=2 (climax).
            on_hold = max(2, int(sub_len * 0.22))
            for s in range(subdivs):
                t = b0 + s * sub_len
                if t >= self.n:
                    break
                # Alternate across both beats and subdivisions so build mode
                # (1 blink per beat) still flips L/R between beats.
                use_left = ((i + s) % 2 == 0)
                front_ch = CH["L_FRONT_TURN"] if use_left else CH["R_FRONT_TURN"]
                rear_ch = CH["L_REAR_TURN"] if use_left else CH["R_REAR_TURN"]
                self.w.set_range(t, t + on_hold, front_ch, 255)
                # rear turn only in climax (busier = more drama)
                if section == "climax":
                    self.w.set_range(t, t + on_hold, rear_ch, 255)
                # Cybertruck has full-brightness rear turn — ramp it with envelope
                if self.model == "cybertruck" and section == "climax":
                    env = self.rms[min(len(self.rms) - 1, t)]
                    lvl = int(128 + 127 * env)
                    self.w.set_range(t, t + on_hold, rear_ch, lvl)

    def _marker_sparkle_layer(self):
        """Constant sparkle on side markers, side repeaters, and license
        plate whenever high-band energy spikes.

        With stereo-aware analysis, the side of the car that sparkles
        follows the actual stereo position of the high-frequency content
        in the mix (hi-hats panned right -> right-side sparkle, etc).
        """
        # Min gap raised from 320 ms → 500 ms and thresholds raised so the
        # sparkle layer leaves more silence between hits (reference has
        # markers on ~12% of frames, we were hitting 16%).
        min_gap = ms_to_frames(500, self.STEP_MS)
        last_fire = -min_gap * 2
        alt = 0
        for f in range(self.intro_end, self.climax_end):
            if f - last_fire < min_gap:
                continue
            section = self._section(f)
            if section == "build":
                thresh = 0.75
            elif section == "climax":
                thresh = 0.6
            else:
                thresh = 1.1
            if self.high[f] < thresh and self.onset[f] < thresh:
                continue
            hold_ms = 100 if section == "climax" else 70
            hold_frames = ms_to_frames(hold_ms, self.STEP_MS)
            left = [CH["L_SIDE_MARKER"], CH["L_SIDE_REPEATER"]]
            right = [CH["R_SIDE_MARKER"], CH["R_SIDE_REPEATER"]]
            # Stereo-aware side selection: if the high band is significantly
            # panned, fire that side; if near-center, fire both.
            pan = self.pan_high[f] if f < len(self.pan_high) else 0.0
            if pan < -0.15:
                target = left
                both = False
            elif pan > 0.15:
                target = right
                both = False
            else:
                target = left + right
                both = True
            for c in target:
                self.w.set_range(f, f + hold_frames, c, 255)
            # When the mix widens or we're in climax with both-side sparkle,
            # add license plate + aux park for extra density.
            width = self.stereo_width[f] if f < len(self.stereo_width) else 0.0
            if both and (section == "climax" or width > 0.5):
                self.w.set_range(f, f + hold_frames, CH["LICENSE"], 255)
                self.w.set_range(f, f + hold_frames, CH["L_AUX_PARK"], 255)
                self.w.set_range(f, f + hold_frames, CH["R_AUX_PARK"], 255)
            last_fire = f
            alt += 1

    def _beat_layer(self):
        """On-beat punches, modulated by the narrative arc.

        * intro: sparse; only ~every 4th beat, ramping not instant
        * build: every 2nd beat; mix of ramps and pulses
        * climax: every beat, full intensity, kick/snare differentiated
        * outro: every 4th beat, very soft ramps only
        """
        alt = 0
        for idx, beat in enumerate(self.beats):
            if beat >= self.n:
                break
            section = self._section(beat)
            strong = beat in self.strong
            bi = min(len(self.rms) - 1, beat)
            rms = self.rms[bi]
            bass_here = self.bass[bi]
            mid_here = self.mid[bi]
            perc_here = self.perc_onset[bi] if bi < len(self.perc_onset) else 0.0
            pan_b = self.pan_bass[bi] if bi < len(self.pan_bass) else 0.0

            # skip-rate per section
            if section == "intro" and (idx % 4) != 0:
                continue
            if section == "outro" and (idx % 4) != 0:
                continue
            if section == "build" and (idx % 2) != 0 and not strong:
                continue

            if section == "intro":
                # Beat-aligned swell: ramp PEAKS on the beat (not starts on it),
                # so the eye's perceived-onset lands on the downbeat.
                # Tempo-aware: slow songs (Stairway intro @ 72 BPM) can hold
                # a 2000 ms ramp; fast songs get tighter 500 ms ramps so
                # they don't smear into the next beat.
                intro_ramp = self.tempo_ramp_choice(beat)
                if alt % 2 == 0:
                    self.ramp_to_beat(CH["L_OUTER_BEAM"], beat, intro_ramp, tail_ms=80)
                else:
                    self.ramp_to_beat(CH["R_OUTER_BEAM"], beat, intro_ramp, tail_ms=80)
            elif section == "outro":
                # Long breathing swell that peaks on the beat, both sides.
                # Outro prefers the longest ramp that fits the current tempo.
                outro_ramp = self.tempo_ramp_choice(beat)
                # For the outro specifically, try one notch longer (even
                # breathier) if there's enough beat budget, since ramp_to_beat
                # already auto-shrinks if it can't fit.
                upgrade = {"500": "1000", "1000": "2000", "2000": "2000"}
                outro_ramp = upgrade[outro_ramp]
                self.ramp_to_beat(CH["L_OUTER_BEAM"], beat, outro_ramp, tail_ms=120)
                self.ramp_to_beat(CH["R_OUTER_BEAM"], beat, outro_ramp, tail_ms=120)
            else:
                # build / climax — full beat logic
                # Heavy = strong beat OR loud frame OR big percussive onset.
                # perc_onset is cleaner than RMS — it ignores sustained loudness
                # (pads, vocals) and spikes only on drum hits.
                heavy = (
                    strong
                    or rms > (0.5 if section == "build" else 0.4)
                    or perc_here > 0.55
                )
                # Kick vs snare: use percussive onset *and* bass/mid ratio.
                is_kick = (bass_here >= mid_here) or (perc_here > 0.6 and bass_here > 0.4)
                if heavy:
                    if is_kick:
                        # Kick: all fronts + both turn signals + rear
                        self.all_front_flash(
                            beat, hold_ms=80 if self._is_m3_or_my_family else 60
                        )
                        self.pulse(CH["L_FRONT_TURN"], beat, hold_ms=120)
                        self.pulse(CH["R_FRONT_TURN"], beat, hold_ms=120)
                        # Also fire all Ch4-6 simultaneously for density
                        for c in ("L_CH4", "R_CH4", "L_CH5", "R_CH5", "L_CH6", "R_CH6"):
                            self.pulse(CH[c], beat, hold_ms=80)
                        self.rear_beat(beat, hold_ms=120)
                        # License + reverse add rear density on climax kicks
                        if section == "climax":
                            self.pulse(CH["LICENSE"], beat, hold_ms=120)
                            self.pulse(CH["REVERSE"], beat, hold_ms=120)
                    else:
                        # Snare: signature + channels 4-6 + rear turn + fog
                        self.pulse(CH["L_SIGNATURE"], beat, hold_ms=100)
                        self.pulse(CH["R_SIGNATURE"], beat, hold_ms=100)
                        for c in ("L_CH4", "R_CH4", "L_CH5", "R_CH5", "L_CH6", "R_CH6"):
                            self.pulse(CH[c], beat, hold_ms=80)
                        self.pulse(CH["L_REAR_TURN"], beat, hold_ms=100)
                        self.pulse(CH["R_REAR_TURN"], beat, hold_ms=100)
                        # Fog adds visual weight on snare (not on CT — no fog)
                        if self.model != "cybertruck":
                            self.pulse(CH["L_FRONT_FOG"], beat, hold_ms=80)
                            self.pulse(CH["R_FRONT_FOG"], beat, hold_ms=80)
                        # Rear fog: Model X has it in NA; on 3/S/Y it only
                        # exists outside North America. Sending the command
                        # on a car that lacks it is harmless (the firmware
                        # just ignores the channel), so we fire it on every
                        # non-CT car and trust the hardware to filter.
                        if self.model != "cybertruck":
                            self.pulse(CH["REAR_FOG"], beat, hold_ms=100)
                else:
                    # Soft beat: short alternating OUTER beam hit (keep
                    # inner beams mostly dark so heavy beats pop). Only
                    # fire every other soft beat to preserve contrast.
                    if alt % 2 == 0:
                        # Instant pulse instead of a ramp so the outer beam
                        # contributes crisp short hits rather than long
                        # on-time between beats (was washing the car out).
                        if self._is_m3_or_my_family:
                            ch_use = CH["L_OUTER_BEAM"] if (alt // 2) % 2 == 0 else CH["R_OUTER_BEAM"]
                            self.pulse(ch_use, beat, hold_ms=60, level=255)
                        else:
                            self.alternating_beam(beat, left=((alt // 2) % 2 == 0), hold_ms=60)
                    if self.rng.random() < 0.25 and section != "intro":
                        self.sparkle_high(beat + ms_to_frames(120))
            alt += 1

        # High-band onset hats sparkle — only in build/climax
        for f in range(self.intro_end, self.climax_end):
            if self.high[f] > 0.7 and self.onset[f] > 0.5:
                if self.rng.random() < 0.15:
                    self.sparkle_high(f)

    def _climax_choreography(self):
        """Guaranteed physical-movement climax — fires once regardless of
        whether audio-analysis found a "drop". This is the trunk/mirror/
        door reveal that Tesla shows are known for.
        """
        d = self.climax_peak

        # Liftgate / Frunk opens ~14 s before the peak so it's fully open
        # when the climax hits (liftgate takes ~14 s to open).
        pre_open = d - ms_to_frames(14_000, self.STEP_MS)
        if pre_open < self.intro_end:
            # not enough runway — open as early as we can and shift peak
            pre_open = max(0, self.intro_end)
        self.closure(CH["LIFTGATE"], pre_open, "open", hold_ms=300,
                     budget_key="liftgate", limit=5)

        # Liftgate Dance: 2-3 short dance bursts clustered around the peak.
        # Dance only works when the liftgate is already open (14 s after
        # pre_open, matching how long it takes to fully open). Each dance
        # burst counts as one actuation against the 6-actuation budget,
        # so we do 3 bursts (+ the open + the close = 5 total, under 6).
        # Each burst is ~1.2 s long. Gap between bursts ~1.5 s.
        dance_first = d - ms_to_frames(1000, self.STEP_MS)
        if dance_first < pre_open + ms_to_frames(14_000, self.STEP_MS):
            # force it to land after the gate is physically open
            dance_first = pre_open + ms_to_frames(14_500, self.STEP_MS)
        burst_len = ms_to_frames(1200, self.STEP_MS)
        burst_gap = ms_to_frames(1500, self.STEP_MS)
        for i in range(3):
            t = dance_first + i * (burst_len + burst_gap)
            end = t + burst_len
            if end >= self.n:
                break
            # One Dance actuation = one contiguous run of byte=128 in the
            # stream, so we set the range to DANCE then explicitly IDLE after.
            self.w.set_range(t, end, CH["LIFTGATE"], CLOSURE["dance"])
            self.closure_used["liftgate"] = self.closure_used.get("liftgate", 0) + 1
            # idle between bursts so the next dance is a fresh actuation
            self.w.set_range(end, end + ms_to_frames(200, self.STEP_MS),
                             CH["LIFTGATE"], CLOSURE["idle"])

        # Charge port dance through a 10 s window starting at peak
        cp_pre = d - ms_to_frames(2500, self.STEP_MS)
        if cp_pre >= 0:
            self.closure(CH["CHARGE_PORT"], cp_pre, "open", hold_ms=300,
                         budget_key="charge_port", limit=2)
        dance_end = min(self.n, d + ms_to_frames(10_000, self.STEP_MS))
        self.w.set_range(d, dance_end, CH["CHARGE_PORT"], CLOSURE["dance"])

        # Mirror flap: reference-style continuous wiper pattern.
        #
        # Each mirror has its own 20-actuation thermal budget, counted
        # separately by the firmware. Each flap cycle consumes 2 actuations
        # per mirror (one Open + one Close), so 9 cycles = 18 actuations —
        # just inside the limit. Using separate budget keys per mirror
        # prevents the shared-budget bug that left one mirror open and the
        # other closed on long shows.
        #
        # CRITICAL: we must always end with both mirrors in the same state
        # (both closed = both folded in the factory default position). Any
        # path that bails out early must still emit the matching Close for
        # every Open it wrote. We do that by collecting planned moves, then
        # truncating the list so the final move on each mirror is a Close.
        flap_count = 9
        flap_period = ms_to_frames(2000, self.STEP_MS)
        mir_start = d - ms_to_frames(9000, self.STEP_MS)
        if mir_start < 0:
            mir_start = 0
        r_offset = ms_to_frames(100, self.STEP_MS)  # R trails L by 100 ms

        # Plan moves for both mirrors independently. Each list contains
        # (frame, action) tuples in chronological order.
        left_moves = []
        right_moves = []
        for i in range(flap_count):
            t = mir_start + i * flap_period
            if t + flap_period >= self.n:
                break
            left_moves.append((t, "open"))
            right_moves.append((t + r_offset, "open"))
            left_moves.append((t + flap_period // 2, "close"))
            right_moves.append((t + flap_period // 2 + r_offset, "close"))

        def write_mirror_sequence(moves, channel, budget_key):
            """Emit moves until the budget is hit, but guarantee the final
            emitted move is a Close so the mirror ends in the folded-back
            state. If we'd hit the limit mid-cycle, drop back to the last
            Close."""
            # trim to an even count so the last emitted move is a Close
            # (list is ordered open/close/open/close/...)
            trimmed = []
            for move in moves:
                # Check if adding this would exceed budget
                if self.closure_used.get(budget_key, 0) >= 19:
                    break
                trimmed.append(move)
                # Tentatively increment so the budget check above is accurate
                # for the next iteration.
                self.closure_used[budget_key] = self.closure_used.get(budget_key, 0) + 1
            # Ensure last action is "close". If it's "open" we dropped its
            # matching close — so remove that final open (and refund the
            # budget we tentatively spent on it).
            while trimmed and trimmed[-1][1] != "close":
                trimmed.pop()
                self.closure_used[budget_key] -= 1
            # Now actually write the FSEQ bytes. The budget was already
            # accounted for above, so bypass closure()'s internal counter
            # by passing budget_key=None.
            for frame, action in trimmed:
                self.closure(channel, frame, action, hold_ms=200)

        write_mirror_sequence(left_moves, CH["L_MIRROR"], "mirrors_left")
        write_mirror_sequence(right_moves, CH["R_MIRROR"], "mirrors_right")

        # Model S: door handles pop at the peak
        if self.model == "model_s":
            for c in (CH["L_FRONT_HANDLE"], CH["R_FRONT_HANDLE"],
                      CH["L_REAR_HANDLE"], CH["R_REAR_HANDLE"]):
                self.closure(c, d - ms_to_frames(1200), "open", hold_ms=200,
                             budget_key="door_handles", limit=18)
                self.closure(c, d + ms_to_frames(1500), "close", hold_ms=200,
                             budget_key="door_handles", limit=18)

        # Model X: full falcon + front-door reveal at the peak
        if self.model == "model_x":
            fd_open = d - ms_to_frames(25_000, self.STEP_MS)
            if fd_open < 0:
                fd_open = 0
            self.closure(CH["L_FALCON"], fd_open, "open",
                         hold_ms=200, budget_key="falcon_doors", limit=5)
            self.closure(CH["R_FALCON"], fd_open + ms_to_frames(200), "open",
                         hold_ms=200, budget_key="falcon_doors", limit=5)
            self.closure(CH["L_FRONT_DOOR"], fd_open, "open",
                         hold_ms=200, budget_key="front_doors", limit=5)
            self.closure(CH["R_FRONT_DOOR"], fd_open + ms_to_frames(200), "open",
                         hold_ms=200, budget_key="front_doors", limit=5)
            falcon_dance_end = min(self.n, d + ms_to_frames(6000, self.STEP_MS))
            self.w.set_range(d, falcon_dance_end, CH["L_FALCON"], CLOSURE["dance"])
            self.w.set_range(d, falcon_dance_end, CH["R_FALCON"], CLOSURE["dance"])
            close_doors = min(self.n - 1, d + ms_to_frames(7_000, self.STEP_MS))
            self.closure(CH["L_FRONT_DOOR"], close_doors, "close",
                         hold_ms=200, budget_key="front_doors", limit=5)
            self.closure(CH["R_FRONT_DOOR"], close_doors + ms_to_frames(200), "close",
                         hold_ms=200, budget_key="front_doors", limit=5)
            close_falcon = min(self.n - 1, falcon_dance_end + ms_to_frames(500, self.STEP_MS))
            self.closure(CH["L_FALCON"], close_falcon, "close",
                         hold_ms=200, budget_key="falcon_doors", limit=5)
            self.closure(CH["R_FALCON"], close_falcon + ms_to_frames(200), "close",
                         hold_ms=200, budget_key="falcon_doors", limit=5)

        # Big visual blast at the peak — all front lights full, 400 ms
        blast_end = d + ms_to_frames(400, self.STEP_MS)
        for f in range(d, min(self.n, blast_end)):
            for c in [CH["L_OUTER_BEAM"], CH["R_OUTER_BEAM"],
                      CH["L_INNER_BEAM"], CH["R_INNER_BEAM"],
                      CH["L_SIGNATURE"], CH["R_SIGNATURE"],
                      CH["L_CH4"], CH["R_CH4"], CH["L_CH5"], CH["R_CH5"],
                      CH["L_CH6"], CH["R_CH6"],
                      CH["L_FRONT_TURN"], CH["R_FRONT_TURN"],
                      CH["L_FRONT_FOG"], CH["R_FRONT_FOG"],
                      CH["BRAKE"], CH["L_TAIL"], CH["R_TAIL"], CH["REVERSE"]]:
                self.w.set(f, c, 255)

        # Cybertruck: dramatic light-bar sweep across the peak
        if self.model == "cybertruck":
            self.cybertruck_lightbar_sweep(
                d, d + ms_to_frames(6000, self.STEP_MS), style="curtain"
            )

        # Close the liftgate late in the climax so it's back down for outro.
        close_gate = min(self.n - 1, self.climax_end - ms_to_frames(4000, self.STEP_MS))
        self.closure(CH["LIFTGATE"], close_gate, "close", hold_ms=300,
                     budget_key="liftgate", limit=5)

    def _drop_layer(self):
        """Decorate additional audio-detected drops (if any) — but only as
        secondary moments, not a second climax. Keeps closures off these
        so the real climax stays the headline.
        """
        for d in self.drops:
            # skip if this drop coincides with the climax (within 3 s)
            if abs(d - self.climax_peak) < ms_to_frames(3000, self.STEP_MS):
                continue
            # Only decorate drops inside build/climax sections
            if d < self.intro_end or d > self.climax_end:
                continue
            # Mini-blast on the front lights (shorter than the climax blast)
            blast_end = d + ms_to_frames(250, self.STEP_MS)
            for f in range(d, min(self.n, blast_end)):
                for c in [CH["L_OUTER_BEAM"], CH["R_OUTER_BEAM"],
                          CH["L_INNER_BEAM"], CH["R_INNER_BEAM"],
                          CH["BRAKE"], CH["L_TAIL"], CH["R_TAIL"]]:
                    self.w.set(f, c, 255)
            if self.model == "cybertruck":
                self.cybertruck_lightbar_sweep(
                    d, d + ms_to_frames(2000, self.STEP_MS), style="chase"
                )

    def _onset_layer(self):
        """Grid-free: fire short pulses on every detected percussive onset.

        This is the "xLights way" — a human light designer watches the
        waveform and drops an effect on every kick, snare, or strum they
        see. Our onset_frames array contains exactly those hits: every
        local peak in perc_onset above a dynamic threshold, regardless of
        whether it fits a beat grid.

        Works brilliantly for songs with sparse / variable rhythm (ballads,
        live recordings, tempo-changing songs like Stairway to Heaven)
        where grid-based beat tracking breaks down.

        Intensity follows the section arc — intros get soft outer-beam
        sparkles, choruses get full-front hits, the outro goes wild if the
        song's outro is percussively active.
        """
        if not self.onset_frames:
            return

        # Avoid double-stimulating with the beat_layer: skip onsets that fall
        # within ±80 ms of an existing beat hit we already handled.
        beat_set = set(self.beats)
        beat_skip_window = ms_to_frames(80, self.STEP_MS)

        # Pre-compute set of nearby beat frames for fast lookup
        nearby_beat = set()
        for b in self.beats:
            for off in range(-beat_skip_window, beat_skip_window + 1):
                nearby_beat.add(b + off)

        for f, strength in zip(self.onset_frames, self.onset_strength):
            if f < self.music_start or f >= self.music_end:
                continue
            if f in nearby_beat:
                continue  # beat_layer already covered this one

            section = self._section(f)
            section_intensity = self._section_intensity(f)
            hit_strength = min(1.0, strength) * section_intensity

            if section == "intro":
                # very sparse, very soft — only the strongest onsets
                if hit_strength < 0.7:
                    continue
                ch = CH["L_OUTER_BEAM"] if (f // 50) % 2 == 0 else CH["R_OUTER_BEAM"]
                self.pulse(ch, f, hold_ms=40, level=180)
            elif section == "outro":
                # OUTRO ONSETS RUN FULL INTENSITY (handles outro drum rolls)
                if hit_strength < 0.35:
                    continue
                # Fire alternating front beams + signature + a side marker
                use_left = (f // 40) % 2 == 0
                if use_left:
                    self.pulse(CH["L_OUTER_BEAM"], f, hold_ms=80)
                    self.pulse(CH["L_INNER_BEAM"], f, hold_ms=80)
                    self.pulse(CH["L_SIGNATURE"], f, hold_ms=80)
                    self.pulse(CH["L_SIDE_MARKER"], f, hold_ms=80)
                else:
                    self.pulse(CH["R_OUTER_BEAM"], f, hold_ms=80)
                    self.pulse(CH["R_INNER_BEAM"], f, hold_ms=80)
                    self.pulse(CH["R_SIGNATURE"], f, hold_ms=80)
                    self.pulse(CH["R_SIDE_MARKER"], f, hold_ms=80)
            else:
                # build / climax — medium intensity, follow strength
                if hit_strength < 0.45:
                    continue
                # Pick a channel based on pan_bass so stereo hits fire the
                # correct side of the car.
                fi = min(len(self.pan_bass) - 1, f)
                pan = self.pan_bass[fi] if fi < len(self.pan_bass) else 0.0
                if pan < -0.2:
                    ch = CH["L_OUTER_BEAM"]
                elif pan > 0.2:
                    ch = CH["R_OUTER_BEAM"]
                else:
                    ch = CH["L_OUTER_BEAM"] if (f // 30) % 2 == 0 else CH["R_OUTER_BEAM"]
                self.pulse(ch, f, hold_ms=60, level=255)
                # Climax bonus: also hit signature on strong onsets
                if section == "climax" and hit_strength > 0.7:
                    side_sig = CH["L_SIGNATURE"] if ch == CH["L_OUTER_BEAM"] else CH["R_SIGNATURE"]
                    self.pulse(side_sig, f, hold_ms=60)

    def _outro_burst_layer(self):
        """Detect an energetic outro (drum roll, final blast, crescendo) and
        fire EVERY front and rear light simultaneously through it.

        Stairway's outro: after the quiet bridge, there's a final drum roll
        that deserves all-lights-blinking-like-crazy energy. Detect this by
        scanning the last 20% of the music region for a window where BOTH
        rms AND perc_onset density stay elevated for >= 2 seconds.

        If found, emit a high-intensity blast across every front and rear
        light for the duration of the burst. Works alongside the onset
        layer which fires individual hits within it.
        """
        import numpy as np
        music_n = self.music_end - self.music_start
        if music_n < ms_to_frames(1000, self.STEP_MS):
            return  # less than 1 s of music — nothing to do
        # scan last 25% of the music region so we catch the final build-up
        # even on long songs with a pre-outro quiet bridge (Stairway).
        scan_start = self.music_start + int(music_n * 0.75)
        scan_end = self.music_end

        rms = np.asarray(self.rms, dtype=np.float32)[scan_start:scan_end]
        # perc_onset per frame — need the array from analysis
        perc = np.asarray(self.a.get("perc_onset", self.rms), dtype=np.float32)
        perc = perc[scan_start:scan_end]
        if len(rms) < ms_to_frames(2000, self.STEP_MS):
            return

        # 1-second rolling averages
        k = max(1, ms_to_frames(1000, self.STEP_MS))
        kernel = np.ones(k, dtype=np.float32) / k
        rms_smooth = np.convolve(rms, kernel, mode="same")
        perc_smooth = np.convolve(perc, kernel, mode="same")

        # Thresholds: above the 60th percentile of the full song. Lower than
        # before so we catch the "build-up to the final note" even when it
        # hasn't yet reached the song's peak RMS. The strobe pattern itself
        # is dramatic so we err on the side of "fire it" whenever the outro
        # is clearly energetic.
        rms_thresh = float(np.percentile(self.rms, 60))
        perc_thresh = float(np.percentile(self.a.get("perc_onset", self.rms), 65))

        # Find contiguous windows where both are hot for >= 2 s. We take the
        # LAST qualifying burst, not the longest — because the classic
        # "outro drum roll" is the final high-energy chunk before the song
        # ends (Stairway's pattern: big chorus → quiet bridge → FINAL
        # CRESCENDO → song ends). Picking by "longest" would anchor on the
        # big chorus and miss the actual drum roll.
        hot = (rms_smooth > rms_thresh) & (perc_smooth > perc_thresh)
        hold = ms_to_frames(2000, self.STEP_MS)
        bursts = []
        current = None
        for i, h in enumerate(hot):
            if h:
                if current is None:
                    current = i
            else:
                if current is not None and i - current >= hold:
                    bursts.append((current, i))
                current = None
        if current is not None and len(hot) - current >= hold:
            bursts.append((current, len(hot)))

        if not bursts:
            return
        burst_start, burst_end = bursts[-1]  # the LAST burst wins

        # Translate back to absolute frames and write the blast.
        abs_start = scan_start + burst_start
        abs_end = scan_start + burst_end

        full_fronts = [
            CH["L_OUTER_BEAM"], CH["R_OUTER_BEAM"],
            CH["L_INNER_BEAM"], CH["R_INNER_BEAM"],
            CH["L_SIGNATURE"], CH["R_SIGNATURE"],
            CH["L_CH4"], CH["R_CH4"], CH["L_CH5"], CH["R_CH5"],
            CH["L_CH6"], CH["R_CH6"],
            CH["L_FRONT_TURN"], CH["R_FRONT_TURN"],
            CH["L_FRONT_FOG"], CH["R_FRONT_FOG"],
            CH["L_AUX_PARK"], CH["R_AUX_PARK"],
            CH["L_SIDE_MARKER"], CH["R_SIDE_MARKER"],
            CH["L_SIDE_REPEATER"], CH["R_SIDE_REPEATER"],
        ]
        full_rears = [
            CH["BRAKE"], CH["L_TAIL"], CH["R_TAIL"], CH["REVERSE"],
            CH["REAR_FOG"], CH["LICENSE"],
            CH["L_REAR_TURN"], CH["R_REAR_TURN"],
        ]
        # Strobe pattern inside the burst: all lights on during the "hot"
        # pulses, off between them. Use the onset signal itself as the gate
        # so the strobe *feels* the drum roll instead of a blind 10 Hz strobe.
        strobe_on_frames = ms_to_frames(60, self.STEP_MS)
        strobe_off_frames = ms_to_frames(60, self.STEP_MS)
        period = strobe_on_frames + strobe_off_frames
        for base in range(abs_start, min(abs_end, self.n), period):
            for f in range(base, min(base + strobe_on_frames, self.n)):
                for c in full_fronts + full_rears:
                    self.w.set(f, c, 255)

        # End of burst: hard blackout for 400 ms to emphasize the final note
        blackout_end = min(self.n, abs_end + ms_to_frames(400, self.STEP_MS))
        for f in range(abs_end, blackout_end):
            for c in full_fronts + full_rears:
                self.w.set(f, c, 0)

    def _mute_non_music_region(self):
        """Zero out every channel before music_start and after music_end.
        Keeps the car completely dark during applause / MC announcements.
        Runs LAST in the compose pipeline so it overrides every other layer.
        """
        if self.music_start <= 0 and self.music_end >= self.n:
            return  # nothing to trim
        if self.music_start > 0:
            for f in range(0, min(self.music_start, self.n)):
                for ci in range(1, self.channels + 1):
                    self.w.set(f, ci, 0)
        if self.music_end < self.n:
            for f in range(max(0, self.music_end), self.n):
                for ci in range(1, self.channels + 1):
                    self.w.set(f, ci, 0)

    def _interior_layer(self):
        """Interior RGB + (Cybertruck only) light-bar effects.

        Only 200-channel shows write RGB bytes. Two driving modes:

        * If chroma data is available (requires librosa during analysis),
          drive interior hue from the song's actual harmony — the tonic
          note maps to red, other pitch classes wrap around the color
          wheel, minor key shifts the whole palette toward warmer.
          Saturation follows HPSS harmonic richness; value follows RMS.
          This gives Model 3 Highland (and future RGB-equipped trims)
          a much more musical interior than a brightness-only wash.

        * Otherwise (older analysis JSON without chroma), fall back to
          the original brightness-driven hue wash.

        Cybertruck also gets its exterior light-bar baseline animation
        regardless of chroma availability.
        """
        if self.channels < 200:
            return

        # RGB surfaces live at channels 176..193 (six 3-byte groups).
        # They're the only RGB hardware we care about for interior light
        # on any current Tesla.
        has_chroma = self.chroma is not None and len(self.chroma) >= self.n

        if has_chroma:
            self._chroma_rgb_layer()
        else:
            # Legacy fallback: brightness-driven hue wash.
            beats_set = set(self.beats)

            def hue(f, frac):
                base = 0.66 * (1.0 - self.brightness[min(len(self.brightness) - 1, f)])
                base += 0.05 * math.sin(f * 0.002)
                if f in beats_set:
                    base += 0.12
                return base

            self.rgb_wash(0, self.n, hue)

        # Cybertruck-only: exterior light-bar baseline. Skip for Highland
        # since it has no light bars (channels 47-162 stay zero for it).
        if self.model == "cybertruck":
            front_start = CH["FRONT_BAR_START"]
            rear_start = CH["REAR_BAR_START"]
            for f in range(self.n):
                env = self.bass[f]
                lvl = int(clamp(env * 255, 0, 255))
                for i in range(10, 50):  # center 40 LEDs of front bar
                    if self.w.frames[f * self.channels + front_start - 1 + i] < lvl:
                        self.w.set(f, front_start + i, lvl)
                for i in range(8, 44):
                    if self.w.frames[f * self.channels + rear_start - 1 + i] < lvl:
                        self.w.set(f, rear_start + i, lvl)

    def _chroma_rgb_layer(self):
        """Drive the six interior RGB surfaces from audio harmony.

        Per frame:
          * Hue: dominant chroma bin, rotated so the current key's tonic
            is always red. Chord changes = visible color changes. Minor
            key shifts the whole hue ring by ~0.08 toward warmer.
          * Saturation: harmonic richness (HPSS harmonic RMS) with a 0.7
            floor so it stays visually intense. Drum breaks desaturate
            slightly toward white.
          * Value (brightness): RMS envelope with a 0.35 floor during
            harmonic content so the interior never fully blacks out
            mid-song.

        Each of the six surfaces gets a small per-surface hue offset
        (5-8% apart on the wheel) so colors spatial-gradient across the
        cabin instead of all being identical.
        """
        # (r_start_channel, hue_offset) for each of the six surfaces
        surfaces = [
            (CH["CENTER_DISPLAY_R"], 0.00),
            (CH["CENTER_ACCENT_R"], 0.03),
            (CH["L_FRONT_RGB_R"], -0.05),
            (CH["R_FRONT_RGB_R"], 0.05),
            (CH["L_REAR_RGB_R"], -0.08),
            (CH["R_REAR_RGB_R"], 0.08),
        ]
        chroma = self.chroma
        tonic = self.key_tonic
        mode = self.key_mode

        for f in range(self.n):
            cf = chroma[f]
            # Dominant pitch class in this frame
            dom = 0
            best = cf[0]
            for i in range(1, 12):
                if cf[i] > best:
                    best = cf[i]
                    dom = i
            # Rotate so tonic = 0 (red). Note: pitch classes go up in
            # semitones, but hue wraps smoothly — we map 12 pitch classes
            # uniformly around the hue wheel for simplicity.
            rel = (dom - tonic[f]) % 12
            base_hue = rel / 12.0
            if mode[f] == 0:  # minor — warm the palette
                base_hue = (base_hue + 0.08) % 1.0

            # Saturation: harmonic richness with a high floor for intensity.
            h_rms = self.harm_rms[f] if f < len(self.harm_rms) else 0.5
            sat = 0.7 + 0.3 * h_rms  # clamped to [0.7, 1.0]

            # Value: RMS envelope with a floor.
            rms = self.rms[f]
            if h_rms > 0.15:
                val = max(0.35, rms)
            else:
                val = rms  # pure drum breaks can fully dim

            for r_ch, offset in surfaces:
                if r_ch + 2 > self.channels:
                    continue
                h = (base_hue + offset) % 1.0
                r, g, b = hsv_to_rgb(h, sat, clamp(val, 0.0, 1.0))
                self.w.set(f, r_ch, r)
                self.w.set(f, r_ch + 1, g)
                self.w.set(f, r_ch + 2, b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True)
    p.add_argument(
        "--model",
        required=True,
        choices=[
            "model_3",
            "model_3_highland",  # 2024+ Model 3 refresh with interior accent lights
            "model_s",
            "model_x",
            "model_y",
            "cybertruck",
        ],
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    analysis = json.loads(Path(args.analysis).read_text())
    c = Composer(analysis, args.model)
    c.compose()
    c.w.save(args.out)
    dur = c.n * c.STEP_MS / 1000.0
    print(f"Wrote {args.out}: {c.channels} ch, {c.n} frames, {dur:.1f}s")


if __name__ == "__main__":
    main()
