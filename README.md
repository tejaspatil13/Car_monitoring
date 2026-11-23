# Car Monitoring - Service Center

Four CCTV cameras → one screen: every car that enters through the main gate is registered
by its number plate, then followed across the cameras (staff car, parking, exit) and shown
on a live dashboard with photos, model, colour and an activity timeline.

| Camera | Default role |
|--------|--------------|
| 1 Main Gate (6062 ch15) | ENTRY (towards camera) and EXIT (other way) line, plate read |
| 2 Yard (6065 ch2) | staff line, parking area |
| 3 Parking Line (6065 ch9) | parking area |
| 4 WS Exit (6065 ch12) | parking area, staff line |

Lines and parking areas are drawn by you with the mouse (`roi_editor.py`).

---

## 1. Install

Python 3.10 - 3.13.

```bash
git clone <repo-url> car_monitoring && cd car_monitoring
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
```

**NVIDIA GPU (Linux / Windows)** - install the CUDA builds first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip uninstall -y onnxruntime && pip install onnxruntime-gpu
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must say True
```

**Mac (Apple Silicon) / CPU**:
```bash
pip install -r requirements.txt
```

**Model weights** (too big for git, ~3 GB in total incl. CLIP):
```bash
python download_models.py
```

The device is picked automatically: NVIDIA GPU (`cuda`) → Apple GPU (`mps`) → CPU.
Force one with `--device cuda`.

---

## 2. Connect the live cameras

Edit `config.py` → `CAMERAS` → `"rtsp"` for each camera (NVR user, password, IP, channel).
Hikvision NVR format:
```
rtsp://USER:PASS@NVR_IP:554/Streaming/Channels/1501    # channel 15, main stream
rtsp://USER:PASS@NVR_IP:554/Streaming/Channels/1502    # channel 15, sub stream (lighter)
```
Test one stream quickly:
```bash
python -c "import cv2; c=cv2.VideoCapture('rtsp://USER:PASS@IP:554/Streaming/Channels/1501'); print(c.read()[0])"
```
`True` = OK. A USB webcam can be used as `"0"`.

Better (keeps the password out of git) - set them as environment variables instead:
```bash
export CAM1_RTSP="rtsp://USER:PASS@NVR_6062_IP:554/Streaming/Channels/1501"
export CAM2_RTSP="rtsp://USER:PASS@NVR_6065_IP:554/Streaming/Channels/201"
export CAM3_RTSP="rtsp://USER:PASS@NVR_6065_IP:554/Streaming/Channels/901"
export CAM4_RTSP="rtsp://USER:PASS@NVR_6065_IP:554/Streaming/Channels/1201"
```

Set `MODE = "live"` in `config.py`, or add `--live` to the commands below.

---

## 3. Draw lines and parking areas (once per camera)

```bash
python roi_editor.py --live          # snapshot of the live cameras
python roi_editor.py                 # or on the recordings
```
Everything with the mouse:
- **CAM 1-4** - choose camera. The slider takes a new frame / snapshot.
- **ENTRY / EXIT / STAFF / PARKING LINE** - click 2 points.
- **PARKING BOX** - click the corners, click the first corner again to close.
  All boxes of a camera together are ONE parking area.
- Drag a point to move it · click a line + **FLIP DIR** to count one direction only (arrow)
  · right-click a shape to delete · **SAVE** (red = unsaved).

Saved in `config/roi.json`.

---

## 4. Run

```bash
python run_four_camera.py --live                  # cameras + dashboard (right side)
python run_four_camera.py --live --layout below   # dashboard below the 4 cameras
python run_four_camera.py --live --no-video       # dashboard only
python run_four_camera.py --live --headless       # no window (server), results in output/
python run_four_camera.py                         # recorded files (recordings/test3)
```
Keys: **Q** quit · **SPACE** pause.

Default layout and other options are in `config.py`
(`DASHBOARD_POSITION`, `ENABLED_CAMERAS`, `SHOW_VIDEO`, `REID_ENABLED`, `IDENTIFY_ENABLED`, ...).

Output in `output/`:
- `vehicles.db` (SQLite) and `entries.csv` - the register
- `captures/` - car and plate photos
- `monitor.mp4` - recording of the screen (`--no-save` to switch off)

The register is cleared at every start; add `--keep-db` to keep it.

---

## 5. Improve car model names (optional)

The model name (WagonR, Ertiga, ...) comes from reference photos of real cars from these
cameras in `models/car_refs/<Model>/`. Add more after a day of running:
```bash
python label_cars.py        # click the model for each saved car photo
```
20-30 photos per model is enough. Unknown → shown as "?", never a wrong guess.

---

## How it works

| Step | What | Model |
|------|------|-------|
| Detect + track cars | per camera | YOLO11s + ByteTrack (Ultralytics) |
| Number plate | at gate lines / parking | open-image-models YOLOv9 plate detector + fast-plate-ocr `cct-xs-v2` + Indian format check, voting over frames |
| Follow a car without plate | other cameras | FastReID ResNet50-IBN (VERI-Wild) appearance fingerprint, matched only against cars inside |
| Car model + colour | at entry | CLIP ViT-L-14 (reference photos + colour words) |

Only cars that entered through the gate are followed; a plate read always overrides an
appearance match. Rules:

| Line / area | Effect |
|-------------|--------|
| entry | new row: SERVICE, car + plate photo, model, colour |
| exit | EXITED + stay time |
| staff | STAFF (not for service) |
| parking line / staying in a parking area (3 s) | PARKING |

## Files

| File | Purpose |
|------|---------|
| `run_four_camera.py` | main program |
| `config.py` | all settings, camera URLs |
| `config/roi.json` | your lines and parking areas |
| `roi_editor.py` | draw lines / areas |
| `label_cars.py` | add reference photos for car models |
| `download_models.py` | fetch model weights |
| `sources.py` | file / live RTSP reader (latest frame, auto-reconnect) |
| `plate_reader.py`, `reid.py`, `reid_matcher.py`, `car_identity.py` | recognition |
| `dashboard.py`, `registry.py`, `roi.py`, `device.py` | dashboard, database, geometry, device |
| `PLAN.md` | roadmap |

## Speed

Mac M-series: ~0.25x real time for 4 cameras at 10 analysed frames/s.
An NVIDIA GPU (RTX 3060 or better) should keep up with 4 live cameras. If not:
`STEP_SECONDS = 0.2` (5 frames/s) or the camera sub-stream in `config.py`.
