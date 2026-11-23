"""Indian number-plate detection + OCR.

Pipeline (all open-source, ONNX, runs on CPU / Apple CoreML):
  1. Plate detector : open-image-models  `yolo-v9-s-608-license-plate-end2end`
  2. Plate OCR      : fast-plate-ocr     `cct-xs-v2-global-model`
     (tested on our footage: xs-v2 read GJ27TD5691 correctly where the larger
      s-v2 model failed; it is also the fastest)
  3. Indian-format correction (O<->0, I<->1, B<->8 ...) by character position
  4. Multi-frame voting per tracked vehicle (PlateVote)

The detector is run on the *vehicle crop*, not the full frame - plates on the
2560px gate camera are too small to be found when the whole frame is shrunk.
"""

import re
from collections import defaultdict

import cv2
import numpy as np

# ---------------- Indian plate format ----------------
# GJ27TD5691 (LLDDLLDDDD), GJ1A1234, MH12ABC1234, 22BH1234AB
INDIAN_PLATE = re.compile(
    r"^([A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$"
)

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "WB",
}

TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G",
             "4": "A", "7": "T", "3": "J"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "U": "0", "I": "1", "L": "1",
            "J": "3", "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4",
            "T": "7"}

# Position templates by length: L = letter, D = digit
TEMPLATES = {
    10: "LLDDLLDDDD",   # GJ27TD5691
    9: "LLDDLDDDD",     # GJ27A1234
    11: "LLDDLLLDDDD",  # DL01ABC1234
    8: "LLDDDDDD",      # old series GJ271234 (rare)
}


def fix_indian(text: str) -> str:
    """Correct OCR confusions using the Indian plate layout."""
    t = re.sub(r"[^A-Z0-9]", "", (text or "").upper())
    if len(t) >= 4 and t[2:4] == "BH":          # Bharat series
        return t
    tpl = TEMPLATES.get(len(t))
    if not tpl:
        return t
    out = []
    for ch, kind in zip(t, tpl):
        if kind == "L":
            out.append(TO_LETTER.get(ch, ch))
        else:
            out.append(TO_DIGIT.get(ch, ch))
    return "".join(out)


def is_valid(plate: str) -> bool:
    if not plate or not INDIAN_PLATE.fullmatch(plate):
        return False
    return plate[2:4] == "BH" or plate[:2] in STATE_CODES


def _overlap(a, b):
    """Intersection area / area(a) for boxes (x1, y1, x2, y2)."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


class PlateReader:
    def __init__(
        self,
        detector="yolo-v9-s-608-license-plate-end2end",
        ocr="cct-xs-v2-global-model",
        det_conf=0.3,
    ):
        from open_image_models import LicensePlateDetector
        from fast_plate_ocr import LicensePlateRecognizer

        # CoreML (Apple GPU/ANE) is ~10x faster than CPU (21 ms vs 236 ms).
        # On crops with no plate CoreML raises an internal error that the
        # library already catches and turns into "no plate" - just silence it.
        import logging
        import onnxruntime as ort
        ort.set_default_logger_severity(4)   # silence CoreML "no plate" noise (Mac)

        self.detector = LicensePlateDetector(
            detection_model=detector, conf_thresh=det_conf
        )
        logging.getLogger(
            "open_image_models.detection.core.yolo_v9.inference"
        ).setLevel(logging.ERROR)
        self.ocr = LicensePlateRecognizer(ocr)
        self.gray = self.ocr.config.image_color_mode == "grayscale"

    def _ocr(self, plate_bgr):
        img = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY) if self.gray else plate_bgr
        pred = self.ocr.run(img, return_confidence=True)
        p = pred[0] if isinstance(pred, list) else pred
        if hasattr(p, "plate"):
            text, probs = p.plate, p.char_probs
        else:  # older API: (texts, probs)
            text, probs = pred[0][0], pred[1][0]
        text = (text or "").replace("_", "")
        n = max(1, len(text))
        conf = float(np.mean(np.asarray(probs).ravel()[:n])) if probs is not None else 0.0
        return text, conf

    def read(self, bgr_crop, crop_origin=(0, 0), ignore_boxes=()):
        """
        bgr_crop     : vehicle crop (full resolution)
        crop_origin  : (x, y) of the crop inside the full frame
        ignore_boxes : full-frame boxes (camera OSD text) - plates under them
                       are skipped because the overlay corrupts the reading
        Returns list of (plate_text, confidence, plate_box_in_frame).
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return []
        ox, oy = crop_origin
        h, w = bgr_crop.shape[:2]
        results = []
        for det in self.detector.predict(bgr_crop):
            b = det.bounding_box
            fbox = (b.x1 + ox, b.y1 + oy, b.x2 + ox, b.y2 + oy)
            if any(_overlap(fbox, ib) > 0.05 for ib in ignore_boxes):
                continue
            pw, ph = b.x2 - b.x1, b.y2 - b.y1
            if pw < 18 or ph < 8:
                continue
            # The plate must belong to THIS car: its centre has to be in the
            # lower 55% of the vehicle box. (A car parked behind can show its
            # plate near the top of the crop.)
            if (b.y1 + b.y2) / 2 < h * 0.45:
                continue
            # small margin: the OCR model expects a little border
            x1 = max(0, int(b.x1 - pw * 0.05)); x2 = min(w, int(b.x2 + pw * 0.05))
            y1 = max(0, int(b.y1 - ph * 0.10)); y2 = min(h, int(b.y2 + ph * 0.10))
            plate = bgr_crop[y1:y2, x1:x2]
            if plate.size == 0:
                continue
            text, conf = self._ocr(plate)
            text = fix_indian(text)
            if len(text) < 6:
                continue
            final = conf * float(det.confidence) ** 0.3
            results.append((text, final, fbox))
        return results


def _similar(a, b):
    """True if two plate strings differ by at most one character."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    s, l = sorted((a, b), key=len)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


class PlateVote:
    """Aggregate OCR results across frames for one tracked vehicle."""

    def __init__(self):
        self.scores = defaultdict(float)
        self.count = defaultdict(int)
        self.best_conf = defaultdict(float)

    def add(self, text, conf):
        if not text or len(text) < 6:
            return
        valid = is_valid(text)
        weight = 1.0 if valid else 0.2
        self.scores[text] += float(conf) * weight
        self.count[text] += 1
        self.best_conf[text] = max(self.best_conf[text], float(conf))
        # near-identical readings support each other (GJ27TD5691 / GJ27TD569)
        for other in list(self.scores):
            if other != text and _similar(other, text):
                self.scores[other] += float(conf) * weight * 0.3

    def best(self):
        if not self.scores:
            return None, 0.0
        valid = [p for p in self.scores if is_valid(p)]
        pool = valid or list(self.scores)
        plate = max(pool, key=self.scores.get)
        total = sum(self.scores.values())
        share = self.scores[plate] / total if total else 0.0
        conf = round(min(1.0, share * self.best_conf[plate] + 0.0), 2)
        return plate, conf

    def confirmed(self):
        """A valid plate seen twice, or once with very high confidence."""
        plate, _ = self.best()
        if not plate or not is_valid(plate):
            return False
        return self.count[plate] >= 2 or self.best_conf[plate] >= 0.92

    def __len__(self):
        return sum(self.count.values())
