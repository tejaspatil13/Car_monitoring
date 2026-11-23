"""Dashboard window (no live video): vehicle table + parking slots.

Columns:  #  | MAIN GATE ENTRY (car photo, plate photo, number / car name / time)
             | STATUS
             | PARKING (car photo, plate photo, area / time)
             | EXIT    (car photo, plate photo, time / stay)
"""

import cv2
import numpy as np

W, H = 1370, 900
ROW_H = 96
SLOTS_H = 170
FONT = cv2.FONT_HERSHEY_SIMPLEX

BG = (24, 24, 24)
ROW_BG = (38, 38, 38)
GRID = (70, 70, 70)
TXT = (235, 235, 235)
DIM = (140, 140, 140)

STATUS_COLOR = {
    "SERVICE": (80, 200, 80),
    "STAFF": (255, 150, 40),
    "PARKING": (255, 0, 200),
    "EXITED": (120, 120, 255),
}

# (key, x, width)
COLS = [
    ("id", 10, 40),
    ("entry_photo", 50, 110), ("entry_plate_photo", 165, 110), ("entry_text", 280, 170),
    ("status", 455, 110),
    ("parking_photo", 570, 110), ("parking_plate_photo", 685, 110), ("parking_text", 800, 150),
    ("exit_photo", 955, 110), ("exit_plate_photo", 1070, 110), ("exit_text", 1185, 175),
]
GROUPS = [("MAIN GATE ENTRY", 50, 400), ("STATUS", 455, 110),
          ("PARKING", 570, 380), ("EXIT", 955, 405)]
SUB = {"entry_photo": "Car", "entry_plate_photo": "Plate", "entry_text": "Number / Name / Time",
       "status": "", "parking_photo": "Car", "parking_plate_photo": "Plate",
       "parking_text": "Area / Time", "exit_photo": "Car", "exit_plate_photo": "Plate",
       "exit_text": "Time / Stay", "id": "#"}


def fmt_stay(sec):
    if sec is None:
        return ""
    sec = int(abs(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


class Dashboard:
    def __init__(self):
        self._cache = {}

    def _img(self, path):
        if not path:
            return None
        if path not in self._cache:
            self._cache[path] = cv2.imread(path)
        return self._cache[path]

    @staticmethod
    def _fit(canvas, img, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 55, 55), -1)
        if img is None or img.size == 0:
            cv2.putText(canvas, "-", (x + w // 2 - 5, y + h // 2 + 5), FONT, 0.5, DIM, 1)
            return
        ih, iw = img.shape[:2]
        s = min(w / iw, h / ih)
        nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
        r = cv2.resize(img, (nw, nh))
        ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
        canvas[oy:oy + nh, ox:ox + nw] = r

    @staticmethod
    def _t(c, text, x, y, scale=0.5, color=TXT, thick=1):
        cv2.putText(c, str(text), (x, y), FONT, scale, color, thick, cv2.LINE_AA)

    def render(self, rows, slots, clock_text="", activity=None):
        activity = activity or {}
        c = np.full((H, W, 3), BG, np.uint8)

        # ---- header ----
        self._t(c, "SERVICE CENTER DASHBOARD", 14, 34, 0.85, TXT, 2)
        (cw, _), _ = cv2.getTextSize(clock_text, FONT, 0.5, 1)
        self._t(c, clock_text, W - cw - 14, 30, 0.5, DIM)
        counts = {k: sum(r["status"] == k for r in rows) for k in STATUS_COLOR}
        inside = sum(r["status"] != "EXITED" for r in rows)
        chips = [("Inside", inside, TXT)] + [(k.title(), v, STATUS_COLOR[k])
                                             for k, v in counts.items()]
        x = 14
        for name, v, col in chips:
            label = f"{name}: {v}"
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.55, 1)
            cv2.rectangle(c, (x, 46), (x + tw + 16, 70), (50, 50, 50), -1)
            cv2.rectangle(c, (x, 46), (x + 4, 70), col, -1)
            self._t(c, label, x + 10, 64, 0.55, TXT)
            x += tw + 26

        # ---- table headers ----
        y0 = 84
        for name, gx, gw in GROUPS:
            cv2.rectangle(c, (gx, y0), (gx + gw, y0 + 24), (60, 60, 60), -1)
            self._t(c, name, gx + 8, y0 + 17, 0.5, TXT, 1)
        for key, cx, cw in COLS:
            self._t(c, SUB[key], cx + 4, y0 + 44, 0.42, DIM)
        cv2.line(c, (10, y0 + 52), (W - 10, y0 + 52), GRID, 1)

        # ---- rows (newest first) ----
        y = y0 + 58
        max_rows = (H - SLOTS_H - y) // ROW_H
        shown = list(reversed(rows))[:max_rows]
        for r in shown:
            cv2.rectangle(c, (10, y), (W - 10, y + ROW_H - 6), ROW_BG, -1)
            ih = 60
            acts = activity.get(r["id"], [])
            if acts:
                short = {"ENTRY": "IN", "EXIT": "OUT", "STAFF": "STAFF", "PARKING": "PARKED",
                         "PARK_OUT": "LEFT SLOT"}
                txt = "  >  ".join(f"{a[0][11:16]} {short.get(a[2], a[2])} ({a[1]})"
                                   for a in acts[-5:])
                self._t(c, "Activity: " + txt, 50, y + ROW_H - 12, 0.42, (180, 200, 255))
            self._t(c, r["id"], 16, y + 40, 0.6, TXT, 2)
            # entry
            if r.get("entry_time"):
                self._fit(c, self._img(r.get("entry_photo")), 50, y + 5, 110, ih)
                self._fit(c, self._img(r.get("entry_plate_photo")), 165, y + 18, 110, 34)
                self._t(c, r["plate"], 284, y + 22, 0.6, TXT, 2)
                self._t(c, r.get("car_name") or "car name: -", 284, y + 42, 0.45,
                        (120, 230, 255) if r.get("car_name") else DIM)
                self._t(c, r["entry_time"][11:19], 284, y + 60, 0.45, TXT)
            else:
                self._t(c, r["plate"], 284, y + 22, 0.6, TXT, 2)
                self._t(c, "not seen at gate", 60, y + 40, 0.45, DIM)
            # status
            col = STATUS_COLOR.get(r["status"], DIM)
            cv2.rectangle(c, (460, y + 24), (558, y + 50), col, -1)
            self._t(c, r["status"], 466, y + 43, 0.5, (0, 0, 0), 2)
            # parking
            if r.get("parking_time"):
                self._fit(c, self._img(r.get("parking_photo")), 570, y + 5, 110, ih)
                self._fit(c, self._img(r.get("parking_plate_photo")), 685, y + 18, 110, 34)
                self._t(c, r.get("parking_slot") or "parking", 804, y + 28, 0.5, TXT, 1)
                self._t(c, r["parking_time"][11:19], 804, y + 52, 0.45, TXT)
            else:
                self._t(c, "-", 680, y + 40, 0.5, DIM)
            # exit
            if r.get("exit_time"):
                self._fit(c, self._img(r.get("exit_photo")), 955, y + 5, 110, ih)
                self._fit(c, self._img(r.get("exit_plate_photo")), 1070, y + 18, 110, 34)
                self._t(c, r["exit_time"][11:19], 1189, y + 28, 0.5, TXT)
                if r.get("stay_seconds") is not None:
                    self._t(c, "stay " + fmt_stay(r["stay_seconds"]), 1189, y + 52, 0.45, DIM)
            else:
                self._t(c, "-", 1065, y + 40, 0.5, DIM)
            y += ROW_H
        if not rows:
            self._t(c, "No vehicles yet", 20, y + 30, 0.6, DIM)
        if len(rows) > len(shown):
            self._t(c, f"+ {len(rows) - len(shown)} older rows in output/entries.csv",
                    20, H - SLOTS_H - 8, 0.45, DIM)

        # ---- PARKING: one combined area, every entered car with status PARKING ----
        ys = H - SLOTS_H + 6
        cv2.line(c, (10, ys - 4), (W - 10, ys - 4), GRID, 1)
        parked = [r for r in rows if r["status"] == "PARKING"]
        self._t(c, f"PARKING   ({len(parked)} cars parked now)", 14, ys + 18, 0.6, TXT, 2)
        if not parked:
            self._t(c, "No entered car is parked right now", 14, ys + 56, 0.5, DIM)
        x = 14
        for r in parked:
            if x + 260 > W:
                self._t(c, f"+{len(parked) - parked.index(r)} more", x, ys + 90, 0.5, DIM)
                break
            cv2.rectangle(c, (x, ys + 28), (x + 252, ys + SLOTS_H - 14), (45, 45, 45), -1)
            cv2.rectangle(c, (x, ys + 28), (x + 5, ys + SLOTS_H - 14), (255, 0, 200), -1)
            photo = self._img(r.get("parking_photo")) if r.get("parking_photo") else None
            if photo is None:
                photo = self._img(r.get("entry_photo"))
            self._fit(c, photo, x + 12, ys + 36, 100, 100)
            self._t(c, f"#{r['id']}", x + 120, ys + 52, 0.5, DIM)
            self._t(c, r["plate"], x + 120, ys + 76, 0.55, TXT, 2)
            self._t(c, f"since {(r.get('parking_time') or '')[11:19]}", x + 120, ys + 100, 0.45, DIM)
            x += 260
        return c
