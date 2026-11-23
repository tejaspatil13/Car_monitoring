"""Label car photos -> reference photos for car model recognition.

    python label_cars.py                      # photos saved by the monitor
    python label_cars.py --folder some/dir    # any folder of car photos

Shows each car photo; click the model name (mouse) to file it under
models/car_refs/<Model>/ . The more photos per model (20-30 is good, from
different angles), the better the model name on the dashboard.
SKIP = not a useful photo, BACK = previous photo, QUIT = close.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from car_identity import MARUTI, REF_DIR

BASE = Path(__file__).resolve().parent
WIN = "Label cars"
W, H = 1280, 760
IMG_H = 520
DONE_FILE = REF_DIR / ".labelled.txt"


class Labeller:
    def __init__(self, folder):
        done = set(DONE_FILE.read_text().split("\n")) if DONE_FILE.exists() else set()
        files = sorted(Path(folder).glob("*.jpg"))
        # car photos only (not plate photos), not yet labelled
        self.files = [f for f in files if "plate" not in f.name and str(f) not in done]
        self.i = 0
        self.buttons = []
        self.msg = f"{len(self.files)} photos to label"
        self.labels = MARUTI + ["Other brand"]
        self.history = []

    def counts(self):
        return {d.name: len(list(d.glob("*.jpg"))) for d in REF_DIR.iterdir()
                if d.is_dir()} if REF_DIR.exists() else {}

    def render(self):
        c = np.full((H, W, 3), 30, np.uint8)
        self.buttons = []
        if self.i >= len(self.files):
            cv2.putText(c, "All photos labelled. Click QUIT.", (40, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        else:
            img = cv2.imread(str(self.files[self.i]))
            if img is not None:
                s = min((W - 40) / img.shape[1], IMG_H / img.shape[0])
                img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
                x0 = (W - img.shape[1]) // 2
                c[10:10 + img.shape[0], x0:x0 + img.shape[1]] = img
            cv2.putText(c, f"{self.i + 1}/{len(self.files)}  {self.files[self.i].name}",
                        (10, IMG_H + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cnt = self.counts()
        x, y = 10, IMG_H + 45
        for lab in self.labels + ["SKIP", "BACK", "QUIT"]:
            n = cnt.get(lab, 0)
            text = f"{lab} ({n})" if lab in self.labels else lab
            w = max(110, cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0] + 20)
            if x + w > W - 10:
                x, y = 10, y + 44
            col = (60, 120, 60) if n >= 20 else (70, 70, 70)
            if lab in ("SKIP", "BACK", "QUIT"):
                col = (60, 60, 140)
            cv2.rectangle(c, (x, y), (x + w, y + 36), col, -1)
            cv2.putText(c, text, (x + 10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            self.buttons.append((x, y, w, 36, lab))
            x += w + 6
        cv2.putText(c, self.msg + "   (green = 20+ photos, enough)", (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 120), 1)
        return c

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for bx, by, bw, bh, lab in self.buttons:
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self.click(lab)
                return

    def click(self, lab):
        if lab == "QUIT":
            self.quit = True
            return
        if lab == "BACK":
            if self.history:
                self.i, dest = self.history.pop()
                if dest and Path(dest).exists():
                    Path(dest).unlink()
                self.msg = "Went back (previous label removed)"
            return
        if self.i >= len(self.files):
            return
        src = self.files[self.i]
        dest = None
        if lab != "SKIP":
            folder = REF_DIR / lab
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / src.name
            shutil.copy(src, dest)
            self.msg = f"Saved as {lab}"
        else:
            self.msg = "Skipped"
        with open(DONE_FILE, "a") as f:
            f.write(str(src) + "\n")
        self.history.append((self.i, str(dest) if dest else None))
        self.i += 1

    def run(self):
        self.quit = False
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W, H)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while not self.quit:
            cv2.imshow(WIN, self.render())
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        print("Reference photos:", self.counts())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=str(BASE / "output" / "captures"))
    Labeller(ap.parse_args().folder).run()
