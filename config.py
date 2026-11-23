"""All settings in one place. Comment / change lines here."""

from pathlib import Path

BASE = Path(__file__).resolve().parent
VIDEO_DIR = BASE / "recordings" / "test3"
MODEL_PATH = BASE / "models" / "yolo11s.pt"
OUTPUT = BASE / "output"
CAPTURE_DIR = OUTPUT / "captures"
DB_PATH = OUTPUT / "vehicles.db"

# ---------------- source: recorded files or live cameras ----------------
# "file" = play recordings from VIDEO_DIR ("file" of each camera)
# "live" = connect to the cameras ("rtsp" of each camera); or run with --live
MODE = "file"

# ---------------- what to show ----------------
SHOW_VIDEO = True          # 4-camera video window   (or run with --no-video)
SHOW_DASHBOARD = True      # register/table window   (or run with --no-dashboard)
SAVE_VIDEOS = True         # record the screen to output/monitor.mp4
DASHBOARD_POSITION = "right"   # "right" of the 4 cameras, or "below" them (--layout below)

# Cameras to run - comment a line out to switch that camera off
ENABLED_CAMERAS = [
    1,   # Main Gate
    2,   # Yard
    3,   # Parking Line
    4,   # WS Exit
]

# "rtsp"  = live stream URL (NVR channel). Hikvision example:
#           rtsp://USER:PASS@NVR_IP:554/Streaming/Channels/1501   (channel 15, main stream)
#           use ...02 (e.g. 1502) for the lighter sub-stream if the GPU/network is slow.
#           A USB webcam can be given as "0", "1", ...
# "start" = CCTV clock time of the first frame of a RECORDING, so event times
#           match the camera OSD. Live mode always uses the real clock.
# "ignore" = burned-in camera text boxes (normalised) - plates under them
# are skipped because the text corrupts OCR.
CAMERAS = {
    1: {"name": "Main Gate", "device": "6062 ch15", "file": "1.mp4",
        "rtsp": "rtsp://USER:PASS@NVR_6062_IP:554/Streaming/Channels/1501",
        "start": "2026-11-20 15:53:05", "ignore": [(0.70, 0.86, 0.82, 0.93)]},
    2: {"name": "Yard", "device": "6065 ch2", "file": "2.mp4",
        "rtsp": "rtsp://USER:PASS@NVR_6065_IP:554/Streaming/Channels/201",
        "start": "2026-11-20 15:50:50", "ignore": [(0.72, 0.88, 0.87, 0.94)]},
    3: {"name": "Parking Line", "device": "6065 ch9", "file": "3.mp4",
        "rtsp": "rtsp://USER:PASS@NVR_6065_IP:554/Streaming/Channels/901",
        "start": "2026-11-20 15:50:50", "ignore": [(0.72, 0.88, 0.87, 0.94)]},
    4: {"name": "WS Exit", "device": "6065 ch12", "file": "4.mp4",
        "rtsp": "rtsp://USER:PASS@NVR_6065_IP:554/Streaming/Channels/1201",
        "start": "2026-11-20 15:50:50", "ignore": [(0.70, 0.85, 0.84, 0.92)]},
}

# ---------------- detection / timing ----------------
CAR_CLASSES = {2, 5, 7}        # COCO car, bus, truck
BIKE_CLASSES = {3}             # motorcycle: tracked, never registered
YOLO_IMGSZ = 640               # 960 = more accurate, needs a GPU
YOLO_CONF = 0.35
STEP_SECONDS = 0.10            # process 10 frames per video-second
IDLE_CAM_EVERY = 3             # cameras with no ROI: track every 3rd step
OCR_EVERY = 2                  # OCR a candidate car every N processed frames
OCR_NEAR_LINE = 0.40           # only OCR cars this close to a line / box
PENDING_TIMEOUT = 6.0          # seconds to wait for a plate after crossing
TRACK_EXPIRE = 3.0             # seconds unseen -> track lost
PLATE_COOLDOWN = 60.0          # same plate + same event ignored within this
MOVING_THRESH = 0.015          # normalised movement in ~1 s = moving

PARK_DWELL = 3.0               # car inside a parking box this long = parked
PARK_RELEASE = 4.0             # box empty this long = slot free again

# ---------------- re-identification (follow a car without its plate) ----------------
REID_ENABLED = True
REID_THRESH = 0.60             # look-similarity needed to say "same car" (0..1)
REID_MARGIN = 0.08             # best match must beat the 2nd best by this much
REID_THRESH_SINGLE = 0.72      # stricter when only ONE car is inside (no 2nd best)
REID_MIN_CROPS = 6             # crops collected from a track before matching it
REID_REQUIRE_AFTER_ENTRY = False   # True for synced/live cameras: a car can only be
                                   # seen after its gate entry time. (test3 clips are
                                   # NOT time-synced, so this is off for now.)

# ---------------- car identity (model + colour at gate entry) ----------------
IDENTIFY_ENABLED = True        # CLIP ViT-L-14: ~12 s to load, ~2.4 s per entered car

# ---------------- camera URLs from environment (keeps passwords out of git) ----------------
# export CAM1_RTSP="rtsp://user:pass@ip:554/Streaming/Channels/1501"   (CAM2_RTSP ... CAM4_RTSP)
import os as _os
for _cid, _cam in CAMERAS.items():
    _cam["rtsp"] = _os.environ.get(f"CAM{_cid}_RTSP", _cam.get("rtsp", ""))
