#!/usr/bin/env python3
"""
Minimal FSEQ v2.0 uncompressed writer for Tesla light shows.

Usage as a library:
    from fseq_writer import FseqWriter
    w = FseqWriter(channel_count=48, frame_count=N, step_time_ms=20)
    # w.frames is a bytearray of length N * channel_count
    w.set(frame_idx, channel_idx_1based, byte_value)
    w.save("out.fseq")

Validate produced files with `light-show/validator.py` (magic 'PSEQ',
channel count must be 48 or 200, compression byte must be 0).

The writer never emits variable headers — channel data starts at byte 53.
"""
from __future__ import annotations

import struct
from pathlib import Path


CHANNEL_DATA_OFFSET = 53


class FseqWriter:
    def __init__(self, channel_count: int, frame_count: int, step_time_ms: int = 20):
        if channel_count not in (48, 200):
            raise ValueError("channel_count must be 48 (all cars) or 200 (Cybertruck extended)")
        if not (15 <= step_time_ms <= 100):
            raise ValueError("step_time_ms must be between 15 and 100")
        if frame_count < 1:
            raise ValueError("frame_count must be >= 1")
        duration_s = frame_count * step_time_ms / 1000.0
        if duration_s > 4 * 60 * 60:
            raise ValueError(f"duration {duration_s:.1f}s exceeds 4 h limit")
        self.channel_count = channel_count
        self.frame_count = frame_count
        self.step_time_ms = step_time_ms
        self.frames = bytearray(channel_count * frame_count)

    def set(self, frame: int, channel_1based: int, value: int):
        if not (1 <= channel_1based <= self.channel_count):
            return
        if not (0 <= frame < self.frame_count):
            return
        v = max(0, min(255, int(value)))
        self.frames[frame * self.channel_count + (channel_1based - 1)] = v

    def set_range(self, start_frame: int, end_frame: int, channel_1based: int, value: int):
        for f in range(max(0, start_frame), min(self.frame_count, end_frame)):
            self.set(f, channel_1based, value)

    def set_rgb_range(
        self,
        start_frame: int,
        end_frame: int,
        r_channel_1based: int,
        r: int,
        g: int,
        b: int,
    ):
        for f in range(max(0, start_frame), min(self.frame_count, end_frame)):
            self.set(f, r_channel_1based, r)
            self.set(f, r_channel_1based + 1, g)
            self.set(f, r_channel_1based + 2, b)

    def save(self, path: str | Path):
        p = Path(path)
        with p.open("wb") as f:
            f.write(self._header())
            f.write(bytes(self.frames))

    def _header(self) -> bytes:
        hdr = bytearray(CHANNEL_DATA_OFFSET)
        hdr[0:4] = b"PSEQ"
        struct.pack_into("<H", hdr, 4, CHANNEL_DATA_OFFSET)   # channel_data_offset
        hdr[6] = 0  # minor version (use 0 to match validator "validated" range)
        hdr[7] = 2  # major version
        struct.pack_into("<H", hdr, 8, CHANNEL_DATA_OFFSET)   # header_extension_size
        struct.pack_into("<I", hdr, 10, self.channel_count)
        struct.pack_into("<I", hdr, 14, self.frame_count)
        hdr[18] = self.step_time_ms
        hdr[19] = 0           # flags
        # offset 20 = compression_type MUST be 0 for Tesla
        hdr[20] = 0
        hdr[21] = 0           # number_compressed_blocks
        hdr[22] = 0           # number_sparse_ranges
        hdr[23] = 0           # flags2
        # 24..31 reserved (unique id) — leave zero
        # 32..52 padding — zero
        return bytes(hdr)


# Byte values for discrete commands -----------------------------------------

# Ramping / boolean brightness levels (percent → byte)
LEVELS = {
    "off_instant": 0,       # 0%
    "off_500ms": 26,        # 10%
    "off_1000ms": 51,       # 20%
    "off_2000ms": 76,       # 30%
    "on_500ms": 178,        # 70%
    "on_1000ms": 204,       # 80%
    "on_2000ms": 230,       # 90%
    "on_instant": 255,      # 100%
}

# Closure command bytes (25% / 50% / 75% / 100%)
CLOSURE = {
    "idle": 0,
    "open": 64,     # 25%
    "dance": 128,   # 50%
    "close": 191,   # 75%
    "stop": 255,    # 100%
}


def pct(p: float) -> int:
    """Clamp 0..100 float/int percent and return 0..255 byte."""
    return max(0, min(255, round(max(0.0, min(100.0, p)) / 100.0 * 255)))
