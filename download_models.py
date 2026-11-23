"""Download the model weights that are too big for git.

    python download_models.py

- models/yolo11s.pt                    car detector (Ultralytics)
- models/reid/veriwild_r50ibn.pt       car re-ID (FastReID VERI-Wild, slimmed to ~95 MB)
Downloaded automatically on first run (no action needed):
- plate detector + OCR (open-image-models / fast-plate-ocr)
- CLIP ViT-L-14 for car model/colour (open_clip, ~1.7 GB, Hugging Face cache)
"""

import shutil
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"
REID_URL = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth"


def yolo():
    dst = MODELS / "yolo11s.pt"
    if dst.exists():
        print("ok  ", dst)
        return
    from ultralytics import YOLO
    YOLO("yolo11s.pt")                       # downloads into the current folder
    shutil.move("yolo11s.pt", dst)
    print("got ", dst)


def reid():
    import reid as R
    if R.WEIGHTS.exists():
        print("ok  ", R.WEIGHTS)
        return
    R.SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not R.SOURCE.exists():
        print("downloading FastReID VERI-Wild weights (1 GB) ...")
        urllib.request.urlretrieve(REID_URL, R.SOURCE)
    R._slim_weights()
    R.SOURCE.unlink()                        # keep only the 95 MB inference file
    print("got ", R.WEIGHTS)


def clip():
    print("checking CLIP ViT-L-14 (first time downloads ~1.7 GB) ...")
    import open_clip
    open_clip.create_model_and_transforms("ViT-L-14", pretrained="datacomp_xl_s13b_b90k")
    print("ok   CLIP")


if __name__ == "__main__":
    MODELS.mkdir(exist_ok=True)
    yolo()
    reid()
    clip()
    print("All models ready.")
