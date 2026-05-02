# Tesla Model-Specific Capabilities

Source: Tesla Light Show xLights Guide (`light-show/README.md`) and the reference `xlights_rgbeffects.xml`.

## Summary Table

| Capability                     | Model 3           | Model S (2021+)        | Model X (2021+)        | Model Y            | Cybertruck                         |
|--------------------------------|-------------------|------------------------|------------------------|--------------------|-------------------------------------|
| Outer Main Beam ramping        | reflector only    | reflector only         | reflector only         | reflector only     | Yes                                 |
| Inner Main Beam ramping        | Yes               | Yes                    | Yes                    | Yes                | Yes                                 |
| Signature                      | Ramping           | Boolean                | Boolean                | Ramping            | —                                   |
| Channels 4-6 ramping           | Yes (merged into one output; Ch4 leads) | Yes (indep L/R; Ch4 sets ramp for all 3) | Yes (same as S) | Yes (merged; Ch4 leads) | — |
| Front Turn                     | Ramping           | Boolean                | Boolean                | Ramping            | Ramping                             |
| Front Side Markers             | Boolean           | Boolean                | Boolean                | Boolean            | Ramping                             |
| Front Fog                      | Yes (not on 3 SR+)| Yes                    | Yes                    | Yes                | —                                   |
| Aux Park                       | Yes (not on 3 SR+)| Yes (indep L/R shared w/ marker) | Yes (same as S) | Yes (shared all-4 OR) | Used as **Frunk Light**          |
| Side Marker / Aux Park coupling| L/R OR'd together | L/R independent, marker+park OR'd per side | same as S | all-4 OR'd together | — |
| Side Repeaters                 | Boolean           | Boolean                | Boolean                | Boolean            | Repurposed as Rear Side Markers     |
| Rear Turn                      | Boolean           | Boolean                | Boolean                | Boolean            | **Full Brightness Control**         |
| Brake Lights                   | Boolean           | Boolean                | Boolean                | Boolean            | **Full Brightness Control**         |
| Tail Lights                    | Boolean (pre-Oct 2020 merged L/R/plate) | Boolean | Boolean | Boolean | Used as Reverse Lights (L/R)   |
| Reverse Lights                 | Boolean           | Boolean                | Boolean                | Boolean            | Used as Bed Lights (500 ms fixed ramp) |
| Rear Fog                       | Only non-NA       | Only non-NA            | Yes (including NA)     | Only non-NA        | —                                   |
| License Plate                  | Boolean (merged with tails pre-Oct 2020) | Boolean | Boolean | Boolean | Boolean                   |
| Front Light Bar (60 LEDs)      | —                 | —                      | —                      | —                  | **Yes (full brightness per LED)**   |
| Rear Light Bar (52 LEDs on bed door) | —            | —                      | —                      | —                  | **Yes (full brightness per LED)**   |
| Offroad Light Bar (6 segs)     | —                 | —                      | —                      | —                  | Optional (full brightness per seg)  |
| Liftgate                       | Only power-liftgate trims | Yes            | Yes                    | Only power-liftgate trims | Yes (mapped from Liftgate → Frunk) |
| Mirrors                        | Yes               | Yes                    | Yes                    | Yes                | Yes                                 |
| Charge Port (dance=rainbow)    | Yes               | Yes                    | Yes                    | Yes                | Yes                                 |
| Windows                        | Yes               | Yes                    | Yes                    | Yes                | Yes                                 |
| Door Handles                   | —                 | Yes (4 indep)          | —                      | —                  | —                                   |
| Front Doors                    | —                 | —                      | Yes (20 s open)        | —                  | —                                   |
| Falcon Doors                   | —                 | —                      | Yes (20 s open)        | —                  | —                                   |
| Center Front Display RGB       | Yes               | Yes                    | Yes                    | Yes                | Yes                                 |
| Interior Accent RGB (5 segs)   | Only if equipped  | Only if equipped       | Only if equipped       | Only if equipped   | Yes                                 |

## Per-Model Production Recipes

### Model 3 / Model Y

Default headlight behavior: **fully ramping on every front light**. Lean into smooth fades and sweeping brightness curves — they look much more polished than boolean blinks on these cars.

Best creative channels:
- Outer/Inner Main Beam, Signature, Front Turn, Channels 4-6 → all smooth ramping
- Brake + Tail + Reverse → punchy boolean beats (back of car)
- Charge Port dance → rainbow accent
- Center Front Display RGB → full-color cabin wash (very visible from outside)

Constraints:
- Only 1 aux park / side marker group (OR'd across all 4). Use sparsely as a single "front accent" channel.
- Channel 4 is the ramp-duration leader for the merged ch4-6 output; the skill's writer does this automatically.
- Liftgate is only controllable on power-liftgate trims — treat as optional.

Recommend **48-channel** output for Model 3/Y.

### Model S

Signature and Front Turn are **boolean only** on Model S — so give them crisp on-beat blinks rather than fades. Ramping is still available for Main Beams and Ch 4-6.

Best creative channels:
- Inner/Outer Main Beam → ramping fades
- Channels 4-6 → tight boolean strobe patterns, with Ch 4 setting the global ramp duration per group
- Signature/Front Turn → hard on-beat hits
- Brake + Tail + Reverse → rear groove
- Door Handles (4 independent) can 'appear' before a drop (20 actuations = plenty for 4 handles)

Constraints:
- Aux Park + Side Marker are OR'd together per side. Leaving even one of the pair on prevents a visible flash.
- No Falcon/Front Doors (those are X-only). No Rear Fog in NA.

Recommend **48-channel** output for Model S.

### Model X

Boolean Signature/Front Turn like Model S, but the star of the Model X show is the **closure choreography**.

Best creative channels:
- Inner/Outer Main Beam → ramping fades
- Channels 4-6 → tight strobe patterns, Ch 4 leads
- Signature/Front Turn → hard on-beat hits
- Brake + Tail + Reverse + **Rear Fog (available even in NA)** → extra rear accent the other cars lack
- **Falcon Doors** → open ~25 s before a big drop, Dance through the first ~6 s of chorus, then Close
- **Front Doors** → open ~25 s before a big drop, Close ~6 s after (no Dance on front doors — use Open/Close)
- Charge Port dance → rainbow accent

Constraints:
- No Door Handles (those are S-only).
- Aux Park + Side Marker OR'd together per side like on Model S.
- Falcon Doors: 20 s open / 8 s close, limit 6 actuations, supports Dance.
- Front Doors: 22 s open / 3 s close, limit 6 actuations, **does not support Dance** — use Open + Close only.
- Moving Windows during Model X door movement can cause false pinch detections that stop the show — **avoid window movement while doors are opening or closing.**
- Plan the whole show around 1–2 major door moments; budget everything else around them.

Recommend **48-channel** output for Model X.

### Cybertruck

Cybertruck unlocks the most creative hardware:
- **Front Light Bar (60 LEDs)** with per-LED full brightness control → integrated xLights effects (Curtain, Bars, Marble, Morph, On) translate to striking sweeps. We emulate these effects in the writer.
- **Rear Light Bar (52 LEDs)** on the bed door → same effect potential, rear-facing.
- **Offroad Light Bar (6 segs)** → sharp accent bars if equipped.
- **Brake & Rear Turn full brightness** → use continuous intensity curves driven by the music envelope, not just on/off.
- Powered Frunk "liftgate" position → dramatic open on big drops (long 14 s open time, plan ahead).
- **Bed Lights** always ramp 500 ms no matter what; commands shorter than that just come out as half-lit pulses.

We must **generate a 200-channel show** to use the light bars and interior RGB. Put big sweeping light-bar effects on the drops and chorus; keep verses sparse so the contrast is felt.

### Universal rules

- Frame interval: **20 ms** (xLights FPS = 50). Any value 15–100 ms is legal but 20 ms matches every example and is recommended.
- Max duration: 4 hours.
- Sample rate: audio must be 44.1 kHz.
- File must be FSEQ v2.0 **uncompressed** (compression_type byte = 0).
- Channel count must be exactly 48 or 200 (the validator refuses anything else).
