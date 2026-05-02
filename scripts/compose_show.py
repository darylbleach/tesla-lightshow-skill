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

        self.channels = 200 if model == "cybertruck" else 48
        self.w = FseqWriter(self.channels, self.n, self.STEP_MS)

        # Closure budgets tracked as we place commands
        self.closure_used = {
            "liftgate": 0, "mirrors": 0, "charge_port": 0, "windows": 0,
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
        self._beat_layer()
        self._climax_choreography()
        self._drop_layer()
        self._interior_layer()

    def _plan_sections(self):
        """Compute the narrative arc: intro / build / climax / outro.

        Tesla shows follow a dramatic structure — start quiet, build light
        activity, culminate with physical movement (trunk/mirrors/doors),
        then wind down. We always schedule the physical-movement moment
        during the `climax` window, at the loudest sustained 2 s we can
        find in that window. This guarantees closures fire even for songs
        without a classic EDM drop.
        """
        import numpy as np  # analyze_audio already requires this
        n = self.n
        self.intro_end = int(n * 0.15)
        self.build_end = int(n * 0.55)
        self.climax_end = int(n * 0.85)
        # climax window: build_end .. climax_end
        # Find loudest sustained 2 s chunk inside it
        win = ms_to_frames(2000, self.STEP_MS)
        rms = np.asarray(self.rms, dtype=np.float32)
        best = self.build_end + (self.climax_end - self.build_end) // 2
        if self.climax_end - self.build_end > win:
            # simple boxcar
            kernel = np.ones(win, dtype=np.float32) / win
            smooth = np.convolve(rms, kernel, mode="same")
            seg = smooth[self.build_end : self.climax_end]
            best = self.build_end + int(np.argmax(seg))
        self.climax_peak = best  # single dramatic moment in the climax

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
        """Gentle breathing on main beams tied to bass envelope, scaled by section."""
        for f in range(self.n):
            b = self.bass[f]
            gate = self._section_intensity(f)
            if b * gate > 0.25:
                val = int(clamp(b * 255 * gate, 0, 255))
                self.w.set(f, CH["L_INNER_BEAM"], val)
                self.w.set(f, CH["R_INNER_BEAM"], val)

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
            rms = self.rms[min(len(self.rms) - 1, beat)]
            bass_here = self.bass[min(len(self.bass) - 1, beat)]
            mid_here = self.mid[min(len(self.mid) - 1, beat)]

            # skip-rate per section
            if section == "intro" and (idx % 4) != 0:
                continue
            if section == "outro" and (idx % 4) != 0:
                continue
            if section == "build" and (idx % 2) != 0 and not strong:
                continue

            if section == "intro":
                # soft, sparse ramp-pulse alternating sides
                if alt % 2 == 0:
                    self.ramp_pulse(CH["L_INNER_BEAM"], beat, "1000", hold_ms=600)
                else:
                    self.ramp_pulse(CH["R_INNER_BEAM"], beat, "1000", hold_ms=600)
            elif section == "outro":
                # fade the whole car down — long ramps
                self.ramp_pulse(CH["L_INNER_BEAM"], beat, "2000", hold_ms=1500)
                self.ramp_pulse(CH["R_INNER_BEAM"], beat, "2000", hold_ms=1500)
            else:
                # build / climax — full beat logic
                heavy = strong or rms > (0.5 if section == "build" else 0.4)
                if heavy:
                    if bass_here >= mid_here:
                        self.all_front_flash(
                            beat, hold_ms=80 if self.model in ("model_3", "model_y") else 60
                        )
                        self.pulse(CH["L_FRONT_TURN"], beat, hold_ms=100)
                        self.pulse(CH["R_FRONT_TURN"], beat, hold_ms=100)
                        self.rear_beat(beat, hold_ms=120)
                    else:
                        self.pulse(CH["L_SIGNATURE"], beat, hold_ms=100)
                        self.pulse(CH["R_SIGNATURE"], beat, hold_ms=100)
                        for c in ("L_CH4", "R_CH4", "L_CH5", "R_CH5"):
                            self.pulse(CH[c], beat, hold_ms=80)
                        self.pulse(CH["L_REAR_TURN"], beat, hold_ms=80)
                        self.pulse(CH["R_REAR_TURN"], beat, hold_ms=80)
                        if self.model == "model_x":
                            self.pulse(CH["REAR_FOG"], beat, hold_ms=80)
                else:
                    if self.model in ("model_3", "model_y"):
                        if alt % 2 == 0:
                            self.ramp_pulse(CH["L_OUTER_BEAM"], beat, "500", hold_ms=260)
                            self.ramp_pulse(CH["L_INNER_BEAM"], beat, "500", hold_ms=260)
                        else:
                            self.ramp_pulse(CH["R_OUTER_BEAM"], beat, "500", hold_ms=260)
                            self.ramp_pulse(CH["R_INNER_BEAM"], beat, "500", hold_ms=260)
                    else:
                        self.alternating_beam(beat, left=(alt % 2 == 0), hold_ms=100)
                    if self.rng.random() < 0.4 and section != "intro":
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

        # Liftgate / Frunk opens ~14 s before the peak
        pre_open = d - ms_to_frames(14_000, self.STEP_MS)
        if pre_open < self.intro_end:
            # not enough runway — open as early as we can and shift peak
            pre_open = max(0, self.intro_end)
        self.closure(CH["LIFTGATE"], pre_open, "open", hold_ms=300,
                     budget_key="liftgate", limit=5)

        # Charge port dance through a 10 s window starting at peak
        cp_pre = d - ms_to_frames(2500, self.STEP_MS)
        if cp_pre >= 0:
            self.closure(CH["CHARGE_PORT"], cp_pre, "open", hold_ms=300,
                         budget_key="charge_port", limit=2)
        dance_end = min(self.n, d + ms_to_frames(10_000, self.STEP_MS))
        self.w.set_range(d, dance_end, CH["CHARGE_PORT"], CLOSURE["dance"])

        # Mirror flap: 3× open/close alternating sides around the peak
        # (6 actuations total per mirror, well inside the 20 limit).
        # Start 1.5 s before peak, 600 ms between moves.
        mir_start = d - ms_to_frames(1500, self.STEP_MS)
        step = ms_to_frames(600, self.STEP_MS)
        for i in range(3):
            t = mir_start + i * 2 * step
            if t < 0:
                continue
            self.closure(CH["L_MIRROR"], t, "open", hold_ms=200,
                         budget_key="mirrors", limit=18)
            self.closure(CH["R_MIRROR"], t + step // 2, "open", hold_ms=200,
                         budget_key="mirrors", limit=18)
            self.closure(CH["L_MIRROR"], t + step, "close", hold_ms=200,
                         budget_key="mirrors", limit=18)
            self.closure(CH["R_MIRROR"], t + step + step // 2, "close", hold_ms=200,
                         budget_key="mirrors", limit=18)

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

    def _interior_layer(self):
        """Interior RGB wash + light-bar effects driven by audio envelope."""
        if self.channels < 200:
            # Only the center display is in 48-ch range — not present!
            # Actually RGB is channels 176+ which is only in 200-ch shows.
            # For 48-ch shows, we skip the interior RGB layer entirely.
            return

        # Overall hue drifts slowly with brightness (cool/warm mapping),
        # pulses forward on each beat.
        beats_set = set(self.beats)

        def hue(f, frac):
            base = 0.66 * (1.0 - self.brightness[min(len(self.brightness) - 1, f)])
            # slow drift
            base += 0.05 * math.sin(f * 0.002)
            # beat kick
            if f in beats_set:
                base += 0.12
            return base

        self.rgb_wash(0, self.n, hue)

        # Light bar baseline: low hum with kick synchronization
        front_start = CH["FRONT_BAR_START"]
        rear_start = CH["REAR_BAR_START"]
        for f in range(self.n):
            env = self.bass[f]
            lvl = int(clamp(env * 255, 0, 255))
            # Center segment always mirrors bass
            for i in range(10, 50):  # center 40 LEDs of front bar
                if self.w.frames[f * self.channels + front_start - 1 + i] < lvl:
                    self.w.set(f, front_start + i, lvl)
            for i in range(8, 44):
                if self.w.frames[f * self.channels + rear_start - 1 + i] < lvl:
                    self.w.set(f, rear_start + i, lvl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", required=True)
    p.add_argument("--model", required=True, choices=["model_3", "model_s", "model_x", "model_y", "cybertruck"])
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
