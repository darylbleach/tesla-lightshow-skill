# FSEQ v2.0 Uncompressed Binary Format (Tesla)

Tesla accepts the **FPP v2.0 Sequence (FSEQ) uncompressed** format, the same format xLights produces with "File ▸ Preferences ▸ Sequences ▸ FSEQ Version = V2 Uncompressed".

Confirmed from live examples in `examples/` and from `light-show/validator.py`.

## Layout

```
+----------------+ offset 0
|  "PSEQ"        |  4 bytes magic
+----------------+
|  start offset  |  uint16 LE (where channel data begins — e.g. 53)
+----------------+
|  minor version |  uint8   (should be 0 or 2)
|  major version |  uint8   (2)
+----------------+
|  hdr ext size  |  uint16 LE (historically same as start offset)
+----------------+
|  channel count |  uint32 LE (must be 48 or 200 for Tesla)
+----------------+
|  frame count   |  uint32 LE
+----------------+
|  step time ms  |  uint8   (20 recommended)
+----------------+
|  flags         |  uint8   (0)
+----------------+
|  uid / extra   |  uint16 LE (0)
+----------------+
|  compression   |  uint8   (0 = uncompressed — REQUIRED for Tesla)
|  blk count hi  |  uint8   (0)
+----------------+
|  n sparse rngs |  uint8   (0)
+----------------+
|  flags2        |  uint8   (0)
+----------------+
|  unique id     |  uint64 LE (0 is fine)
+----------------+
|  variable hdrs |  (until start offset reached; can be all zeros)
+----------------+ offset = `start offset`
|  frame 0       |  `channel count` bytes
|  frame 1       |  `channel count` bytes
|  ...           |
|  frame N-1     |
+----------------+
```

### Exact fields observed in examples

Dumping `croatian-anthem.fseq` header bytes:

```
50 53 45 51   "PSEQ"
35 00         start_offset = 53
02 02         minor=2 major=2  (v2.2)
35 00         header_extension_size = 53 (duplicate of start offset)
2e 00 00 00   channel_count = 46 (for xLights-shipped default; we write 48)
69 1d 00 00   frame_count = 7529
14            step_time = 20 ms
00 00 00 00   flags etc.
00 00 00 00
14 00         ← ??
00 00 00 00   
... zeros until offset 53 (the variable header pad)
```

Note: the shipped examples write 46 channels. Because some firmware versions are lenient, 46-ch files still play, but **`validator.py` rejects anything other than 48 or 200** — so we target those.

### Writer-safe template

Here is the minimal valid header we emit (53-byte header, matching observed files exactly, no variable headers):

```
off  bytes                        meaning
0x00 50 53 45 51                  magic
0x04 35 00                        channel_data_offset = 0x35 (53)
0x06 00 02                        minor=0, major=2  (v2.0)
0x08 35 00                        header_extension_size = 53 (=data offset; no extra headers)
0x0A <u32> channel_count          48 or 200
0x0E <u32> frame_count            N
0x12 14                           step_time = 20 ms
0x13 00                           flags
0x14 00 00                        media offset / reserved
0x16 00                           compression = 0 (uncompressed)
0x17 00                           num_compressed_blocks = 0
0x18 00                           num_sparse_ranges = 0
0x19 00                           flags2 = 0
0x1A 00 00 00 00 00 00 00 00      64-bit unique ID (zero is valid)
0x22 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00   padding to 53 bytes
0x35 <channel data>               frame_count × channel_count bytes
```

Setting minor=0 keeps the validator silent (2.0 is the "validated" version). Minor=2 also works but triggers a soft warning.

## Writing tips

- No variable headers required — start channel data immediately at offset 53.
- Do **not** use compression_type=1 (zstd) or 2 (zlib) — Tesla rejects.
- Frame data is row-major: for frame `t`, write `channel_count` bytes in channel-1,2,3,… order.
- Pad unused channels (e.g. 47–48 in a 48-ch show, gaps in a 200-ch show) with zero.
- Total duration = `frame_count * step_time / 1000` seconds; keep under 4 × 3600.
- Step time 20 ms matches every shipped Tesla example and gives 50 fps.

## Validation

After writing, run:

```
python3 light-show/validator.py out.fseq
```

Expected output:
```
Found <N> frames, step time of 20 ms for a total duration of <H:MM:SS.ffffff>.
```

If the validator reports a different channel count or a compression error, the file will not play.
