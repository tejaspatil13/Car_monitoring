"""Regions of interest (lines + parking boxes) per camera.

Stored in config/roi.json, all coordinates normalised 0..1 so they work at
any resolution. Edit them with:   python roi_editor.py

Format:
{
  "1": {"lines": [{"type": "entry", "name": "Gate entry", "pts": [[x,y],[x,y]]}],
        "boxes": [{"name": "C3-P1", "pts": [[x,y],[x,y],[x,y],[x,y]]}]},
  ...
}
Line types:
  entry   - car crosses -> new SERVICE entry (plate read)
  exit    - car crosses -> entry marked EXITED (plate read), stay time computed
  staff   - car crosses -> car marked STAFF (not for service)
  parking - car crosses -> car marked PARKING
Boxes: parking slots; a car standing inside for a few seconds = slot occupied.

Each line has "dir": 0 = count both directions, +1 / -1 = count only cars
crossing towards the side the arrow points to (drawn in the editor and the
camera window). The same gate line can be an ENTRY (one way) and an EXIT
(the other way).
"""

import json
import os
from pathlib import Path

ROI_FILE = Path(os.environ.get("ROI_FILE") or
                Path(__file__).resolve().parent / "config" / "roi.json")

LINE_TYPES = ("entry", "exit", "staff", "parking")

# BGR colours used everywhere
COLORS = {
    "entry": (0, 165, 255),     # orange
    "exit": (60, 60, 255),      # red
    "staff": (255, 120, 0),     # blue
    "parking": (255, 0, 200),   # purple
    "box": (0, 230, 255),       # yellow
}

DEFAULT_ROI = {
    # Main gate: same line, two directions. Towards the camera = ENTRY,
    # the other way = EXIT. Redraw / flip with roi_editor.py if needed.
    "1": {"lines": [{"type": "entry", "name": "Gate in", "dir": 1,
                     "pts": [[0.17, 0.75], [0.64, 0.47]]},
                    {"type": "exit", "name": "Gate out", "dir": -1,
                     "pts": [[0.17, 0.75], [0.64, 0.47]]}], "boxes": []},
    "2": {"lines": [{"type": "staff", "name": "Staff line", "dir": 0,
                     "pts": [[0.476, 0.30], [0.442, 0.756]]}], "boxes": []},
    "3": {"lines": [], "boxes": []},
    "4": {"lines": [], "boxes": []},
}


def load_roi(path=ROI_FILE):
    path = Path(path)
    if not path.exists():
        save_roi(DEFAULT_ROI, path)
        return json.loads(json.dumps(DEFAULT_ROI))
    data = json.loads(path.read_text())
    for cam in data.values():
        cam.setdefault("lines", [])
        cam.setdefault("boxes", [])
    return data


def save_roi(data, path=ROI_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ---------------- geometry ----------------

def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(p1, p2, a, b):
    o1, o2 = _orient(p1, p2, a), _orient(p1, p2, b)
    o3, o4 = _orient(a, b, p1), _orient(a, b, p2)
    return o1 * o2 < 0 and o3 * o4 < 0


def side(p, a, b):
    """+1 / -1: which side of line a->b the point is on."""
    return 1 if _orient(a, b, p) > 0 else -1


def crossing_ok(prev, cur, line):
    """True if prev->cur crosses the line in the line's allowed direction."""
    a, b = line["pts"]
    if not segments_intersect(prev, cur, a, b):
        return False
    d = line.get("dir", 0)
    return d == 0 or side(cur, a, b) == d


def arrow_points(a, b, d, length=28):
    """Start/end pixel points of the direction arrow at the line midpoint."""
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    nx, ny = -(b[1] - a[1]), (b[0] - a[0])          # points to the + side
    n = (nx * nx + ny * ny) ** 0.5 or 1
    return (int(mx), int(my)), (int(mx + d * nx / n * length), int(my + d * ny / n * length))


def dist_to_segment(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy or 1e-9
    u = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L))
    return ((ax + u * dx - px) ** 2 + (ay + u * dy - py) ** 2) ** 0.5


def point_in_polygon(p, poly):
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
                inside = not inside
    return inside
