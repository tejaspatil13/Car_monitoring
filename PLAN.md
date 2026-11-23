# Plan: track every car from gate-in to gate-out

**Goal:** for every car that enters the service center, know
- how many cars are inside right now,
- where each one is (yard / parking slot / workshop),
- how long it has been there, and how long the whole visit took.

**The hard part:** the number plate is only readable at a few points: the main gate, and
sometimes the WS exit. On the yard and parking cameras, plates are too small or turned away
(camera 3 parked cars read as junk at 0.3 confidence). So between cameras we need another way
to know "this is the same car".

---

## What the footage already tells us

- **Plates do link cameras.** On test3, `GJ27TD5691` and `GJ27CF9632` were read on WS Exit
  (15:51:38, 15:51:42) and again at the Main Gate (15:53:07, 15:53:46). So the plate is a
  reliable anchor at the gate and at WS exit.
- **Direction matters at the gate.** Those two cars left the workshop first, then crossed the
  gate line moving *towards* camera 1. So that direction is OUT. The gate line now counts
  ENTRY one way and EXIT the other (`dir` in `config/roi.json`, flip with `f` in the editor).
- **The test clips are not time-synced.** Camera 1 starts 15:53:05; cameras 2-4 start 15:50:50.
  For cross-camera testing, export all 4 channels for the **same** time window, ideally 30-60 min,
  so we see complete visits.

---

## Approach: plate as the anchor, car appearance + camera layout in between

We don't need to recognise a car anywhere in the world. We only need to pick it out from the
**cars currently inside** (usually 10-60). That is a much easier problem.

```
GATE IN ──► YARD ──► PARKING / WORKSHOP ──► WS EXIT ──► GATE OUT
 plate       appearance + timing              plate        plate
 (anchor)    (link to anchor)                 (re-check)   (close visit)
```

### Step 1 - Visit ID at the gate (have this now)
At the gate-in line: read the plate, save a car photo, and create a **visit**:
`visit_id, plate, entry_time, photos`. Every later sighting gets linked to a visit_id.

### Step 2 - Appearance fingerprint for each car
Each time a car is tracked on a camera, compute an **embedding** (a vector that describes how
the car looks), averaged over several good frames:
- Vehicle re-ID model, open source: **FastReID / OSNet trained on VeRi-776 / VehicleID**
  (made for "same car across cameras"). Alternative with no training: **DINOv2 or CLIP image
  embeddings**, which work surprisingly well for car colour, shape and model.
- Cheap extra attributes: colour, body type (hatchback / sedan / SUV), and **car make/model**.
  This also fills the "car name" column. Start with CLIP zero-shot on a list of common Indian
  models (WagonR, Swift, Dzire, Baleno, Ertiga, Creta, Nexon, Thar, ...), then fine-tune later.
- Plate colour (yellow = commercial / taxi) and partial plate reads (even 4 digits help).

### Step 3 - Camera layout + travel times
Write down which camera a car can go to next, and how long it usually takes:
```
Gate(1) -> Yard(2): 5-60 s      Yard(2) -> Parking(3): 10-120 s
Yard(2) -> Workshop: ...        WS Exit(4) -> Gate(1): 30-180 s
```
Learn the real times from the first synced recordings. When a car disappears from camera A's
exit area, only cars that could arrive at camera B in that time window are candidates.

### Step 4 - Matching ("handover")
When a new track appears on a camera, score every candidate visit:
`score = appearance similarity + timing fits the layout + attributes match (+ plate if read)`
and pick the best match (Hungarian assignment when several cars move at once).
- **Confident:** link to that visit_id and update its location ("C3-P2 since 15:51").
- **Not sure:** mark "unconfirmed"; the next plate read (WS exit / gate out) confirms it or
  corrects it.
- A plate read always wins over appearance.

### Step 5 - Close the visit at gate-out
Plate read at the gate-out line -> visit closed, `stay = exit - entry`. Any car still "inside"
at night is flagged for review.

### Step 6 - Human check in the dashboard
Show uncertain links with both photos side by side so staff can confirm/fix with one click.
Those corrections become training data for a better re-ID model on *our* cameras.

---

## Order of work

| # | Task | Result |
|---|------|--------|
| 1 | Record a **synced** 30-60 min clip of all 4 channels (same start time) | real test data |
| 2 | Draw final ROIs (gate in/out, staff, parking boxes, WS exit) with `roi_editor.py` | correct events |
| 3 | Label ground truth for ~20 cars: plate + which camera, when | way to measure accuracy |
| 4 | Add appearance embedding per track (DINOv2/CLIP first, then FastReID) | fingerprint per sighting |
| 5 | Add the camera layout + travel times, plus the matching step | car followed across cameras |
| 6 | Car make/model classifier -> "car name" column | richer dashboard |
| 7 | Dashboard: per-car timeline (gate -> yard -> P2 -> WS -> gate) + "needs review" list | operations view |
| 8 | Switch from files to live RTSP streams from the NVR | production |

## Things to decide / check
- Is the main gate the **only** way in and out? (If there is a second gate, it needs a camera
  with an in/out line too.)
- Which areas does WS Exit cover - workshop exit only, or also a road back to parking?
- Do staff cars use the same gate? (A staff list of plates would let us tag them at the gate.)
- Camera clocks: make sure the NVR keeps all channels' time in sync (NTP).
