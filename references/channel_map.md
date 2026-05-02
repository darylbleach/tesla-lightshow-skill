# Tesla FSEQ Channel Map

The Tesla light show firmware reads channel data byte-by-byte per frame. Channels are 1-indexed in xLights and in this document, but the file data is 0-indexed (channel 1 = byte offset 0 within a frame).

There are **two valid channel counts** (enforced by `validator.py`):

- **48** channels — classic show, covers all cars' lights + closures using OR-ing.
- **200** channels — extended show, enables Cybertruck light bars and all interior RGB segments.

**Any show can run on any car**, but extra channels beyond the car's hardware are ignored, and some channels are re-mapped (see Cybertruck Light Remapping below). To maximize impact per model, we generate 48-ch shows for Model 3/S/Y, and 200-ch shows for Cybertruck.

## Core 48-Channel Layout (1-indexed)

| Ch | Name                     | Type       | Ramp? |
|----|--------------------------|------------|-------|
| 1  | Left Outer Main Beam     | Headlight  | Yes (on reflector cars) |
| 2  | Right Outer Main Beam    | Headlight  | Yes (on reflector cars) |
| 3  | Left Inner Main Beam     | Headlight  | Yes |
| 4  | Right Inner Main Beam    | Headlight  | Yes |
| 5  | Left Signature           | Headlight  | Yes on 3/Y, Boolean on S/X |
| 6  | Right Signature          | Headlight  | Yes on 3/Y, Boolean on S/X |
| 7  | Left Channel 4           | Headlight subsegment | Yes (leader for 4-6) |
| 8  | Right Channel 4          | Headlight subsegment | Yes (leader for 4-6) |
| 9  | Left Channel 5           | Headlight subsegment | Yes |
| 10 | Right Channel 5          | Headlight subsegment | Yes |
| 11 | Left Channel 6           | Headlight subsegment | Yes |
| 12 | Right Channel 6          | Headlight subsegment | Yes |
| 13 | Left Front Turn          | Turn Signal | Yes on 3/Y & CT, Boolean on S/X |
| 14 | Right Front Turn         | Turn Signal | Yes on 3/Y & CT, Boolean on S/X |
| 15 | Left Front Fog           | Fog         | Boolean |
| 16 | Right Front Fog          | Fog         | Boolean |
| 17 | Left Aux Park            | Parking / CT Frunk Light | Boolean |
| 18 | Right Aux Park           | Parking / CT Frunk Light | Boolean |
| 19 | Left Side Marker         | Marker / CT Front Side Marker (ramping) | Partial |
| 20 | Right Side Marker        | Marker / CT Front Side Marker (ramping) | Partial |
| 21 | Left Side Repeater       | Side repeater / CT Left Rear Side Marker | Boolean |
| 22 | Right Side Repeater      | Side repeater / CT Right Rear Side Marker | Boolean |
| 23 | Left Rear Turn           | Rear turn (disabled on CT) | Boolean |
| 24 | Right Rear Turn          | Rear turn (disabled on CT) | Boolean |
| 25 | Brake Lights             | Brake / CT full-brightness | 3/Y/S/X Boolean; CT full brightness |
| 26 | Left Tail                | Tail / CT Left Reverse | Boolean |
| 27 | Right Tail               | Tail / CT Right Reverse | Boolean |
| 28 | Reverse Lights           | Reverse / CT Bed Lights | Boolean (CT bed always 500 ms ramp) |
| 29 | Rear Fog Lights          | Rear fog | Boolean (not installed in NA except X) |
| 30 | License Plate            | Plate light | Boolean |
| 31 | Left Falcon Door         | Closure (X only) | — |
| 32 | Right Falcon Door        | Closure (X only) | — |
| 33 | Left Front Door          | Closure (X only) | — |
| 34 | Right Front Door         | Closure (X only) | — |
| 35 | Left Mirror              | Closure (all) | — |
| 36 | Right Mirror             | Closure (all) | — |
| 37 | Left Front Window        | Closure (all) | — |
| 38 | Left Rear Window         | Closure (all) | — |
| 39 | Right Front Window       | Closure (all) | — |
| 40 | Right Rear Window        | Closure (all) | — |
| 41 | Liftgate / Frunk         | Closure (S/X/CT all; 3/Y only with power liftgate). CT = Powered Frunk | — |
| 42 | Left Front Door Handle   | Closure (S only) | — |
| 43 | Left Rear Door Handle    | Closure (S only) | — |
| 44 | Right Front Door Handle  | Closure (S only) | — |
| 45 | Right Rear Door Handle   | Closure (S only) | — |
| 46 | Charge Port              | Closure (all) | — |
| 47 | (reserved)               | Fill with 0 | — |
| 48 | (reserved)               | Fill with 0 | — |

Channels 47–48 do not map to any hardware. The file must still include them (48-byte frames) and the bytes MUST be 0 to keep the validator happy. Some existing shows use 46 channels which historical firmware tolerates, but **always write 48 for forward compatibility**.

## 200-Channel Extension (Cybertruck + RGB)

Channels 1–46 match the layout above. Channels 47–200 add Cybertruck-specific hardware:

| Range       | Count | Purpose                                                |
|-------------|-------|--------------------------------------------------------|
| 47–76       | 30    | Left Front Light Bar (30 LEDs, left segment)           |
| 77–106      | 30    | Right Front Light Bar (30 LEDs, right segment)         |
|  (the full 60-LED front bar spans 47–106; "left+center+right" are just grouping) |
| 111–136     | 26    | Left Rear Light Bar (26 LEDs)                          |
| 137–162     | 26    | Right Rear Light Bar (26 LEDs)                         |
|  (the full 52-LED rear bar spans 111–162; center = all 52 on bed door)           |
| 167–172     | 6     | Offroad Light Bar (6 forward+ditch segments)           |
| 175         | 1     | Suspension                                             |
| 176–178     | 3     | Center Front Display RGB (R,G,B triplet)               |
| 179–181     | 3     | Right Rear RGB (R,G,B)                                 |
| 182–184     | 3     | Right Front RGB (R,G,B)                                |
| 185–187     | 3     | Center Front Accent RGB (R,G,B)                        |
| 188–190     | 3     | Left Front RGB (R,G,B)                                 |
| 191–193     | 3     | Left Rear RGB (R,G,B)                                  |
| 194–200     | 7     | reserved / fill with 0                                 |

**Gaps (107–110 and 163–166 and 173–174) are reserved. Fill with 0.**

The front and rear light bar segment grouping is an xLights convenience. In the FSEQ bytes, the 60 front LEDs are just bytes at channel indexes 47..106 sequentially. The single-color light bars take one byte per LED (intensity 0–255, full brightness control).

## Cybertruck Light Remapping (automatic on vehicle)

When a 48-channel show plays on a Cybertruck, the firmware remaps:

- **Ch 26/27 (L/R Tail)** → Reverse Lights (L/R individual)
- **Ch 21/22 (Side Repeater)** → L/R Rear Side Markers
- **Ch 41 (Liftgate)** → Powered Frunk
- **Ch 17/18 (Aux Park)** → Frunk Light
- **Ch 28 (Reverse)** → Bed Lights
- **Ch 23/24 (Rear Turn)** → Disabled

This means when designing a 48-ch show that must look good on all cars including CT, keep tail/rear-turn effects synchronized with things that also make sense as reverse/side-marker flashes.

## Byte value ↔ percent brightness

Value curves are 0–255. The firmware uses `round(pct/100 * 255)` as the effective setpoint:

| Percent | Byte | Meaning (boolean channel)           | Meaning (ramping channel)                  |
|---------|------|--------------------------------------|--------------------------------------------|
| 0       | 0    | Off                                  | Off; Instant                               |
| 10      | 26   | Off                                  | Off; 500 ms                                |
| 20      | 51   | Off                                  | Off; 1000 ms                               |
| 30      | 76   | Off                                  | Off; 2000 ms                               |
| 40      | 102  | Off                                  | (unused by keyboard shortcut)              |
| 50      | 128  | **Boundary.** ≥128 = ON, else OFF on non-ramping channels | (unused) |
| 60      | 153  | On                                   | (unused)                                   |
| 70      | 178  | On                                   | On; 500 ms                                 |
| 80      | 204  | On                                   | On; 1000 ms                                |
| 90      | 230  | On                                   | On; 2000 ms                                |
| 100     | 255  | On                                   | On; Instant                                |

For Cybertruck **full-brightness-controlled channels** (front/offroad/rear light bars, brake, rear turn signals) any intermediate byte is a valid steady-state brightness.

For RGB channels, 0 = black, 255 = full color. Write R/G/B as three consecutive bytes.

## Closure commands

Closure channels (31–46 subset) use four discrete bytes:

| Byte (%)  | Movement  | Meaning                                                    |
|-----------|-----------|------------------------------------------------------------|
| 0 (0%)    | Idle      | Stop request, allow current Open/Close to finish           |
| 64 (25%)  | Open      | Start opening, stop when fully open                        |
| 128 (50%) | Dance     | Oscillate between 2 positions (rainbow on charge port)     |
| 191 (75%) | Close     | Start closing, stop when fully closed                      |
| 255 (100%)| Stop      | Immediate stop                                             |

The firmware looks for ~25/50/75/100% discrete levels; use the values above exactly.

### Closure command limits per show

| Closure             | Limit | Notes                                   |
|---------------------|-------|-----------------------------------------|
| Liftgate / Frunk    | 6     | 14 s to open, 4 s to close              |
| Mirrors             | 20    | 2 s                                     |
| Charge Port         | 3     | 2 s; auto-closes after 2 min            |
| Windows             | 6     | 4 s                                     |
| Door Handles (S)    | 20    | 2 s                                     |
| Front Doors (X)     | 6     | 22 s open, 3 s close                    |
| Falcon Doors (X)    | 6     | 20 s open, 8 s close                    |

Only Open/Close/Dance count toward limits. Dance only works when already open (windows are an exception). Total dance time per show ≤ ~30 s (thermal).
