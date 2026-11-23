"""ROI editor - draw lines and parking boxes with the mouse.

    python roi_editor.py            # starts on camera 1
    python roi_editor.py --cam 3
    python roi_editor.py --live     # draw on a snapshot of the live camera

MOUSE
  Click a button in the top bar to pick a camera / tool / action.
  ENTRY / EXIT / STAFF / PARKING LINE : click 2 points on the image.
  PARKING BOX : click the corners; finish by clicking the FIRST corner again,
                double-clicking, right-clicking, or pressing ENTER.
  With no tool selected:
    drag a point          -> move that corner / line end
    click a line or box   -> select it (yellow); FLIP changes a selected line's direction
    right-click a shape   -> delete it
  Use the "frame" slider to find a good frame.

KEYS (optional)  1-4 camera | e x s p b tools | f flip | u undo | c clear | w save | q quit
"""

import argparse
import time

import cv2
import numpy as np

import config as CFG
from config import CAMERAS, VIDEO_DIR
from sources import open_capture
from roi import COLORS, ROI_FILE, arrow_points, dist_to_segment, load_roi, \
    point_in_polygon, save_roi

VIEW_W, VIEW_H = 1280, 720
BAR_H = 92                    # two rows of buttons
STATUS_H = 34
WIN = "ROI Editor"
FONT = cv2.FONT_HERSHEY_SIMPLEX
SNAP = 12                     # px: grab / close distance

TOOLS = {"entry": "ENTRY LINE", "exit": "EXIT LINE", "staff": "STAFF LINE",
         "parking": "PARKING LINE", "box": "PARKING BOX"}
KEYS = {ord("e"): "entry", ord("x"): "exit", ord("s"): "staff",
        ord("p"): "parking", ord("b"): "box"}


class Button:
    def __init__(self, x, y, w, h, label, action, color=(70, 70, 70)):
        self.rect = (x, y, w, h)
        self.label, self.action, self.color = label, action, color

    def hit(self, x, y):
        bx, by, bw, bh = self.rect
        return bx <= x <= bx + bw and by <= y <= by + bh


class Editor:
    def __init__(self, cam, live=False):
        self.live = live or CFG.MODE == "live"
        self.roi = load_roi()
        self.mode = None              # tool being drawn
        self.points = []              # in-progress points (image px)
        self.dirty = False
        self.quit_armed = False
        self.quit = False
        self.msg = "Pick a tool from the top bar"
        self.selected = None          # ("line"|"box", index)
        self.drag = None              # ("line"|"box", index, point index)
        self.hover = None
        self.mouse = (0, 0)
        self.last_click = (0.0, (0, 0))
        self.cap = None
        self.frame = None
        self.buttons = []
        self.history = []
        self.open_cam(cam)

    # ---------------- video ----------------
    def open_cam(self, cam):
        if self.cap:
            self.cap.release()
        self.cam = cam
        src = CAMERAS[cam]["rtsp"] if self.live else VIDEO_DIR / CAMERAS[cam]["file"]
        self.cap = open_capture(src)
        self.n_frames = 2 if self.live else max(2, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.mode, self.points, self.selected, self.drag = None, [], None, None
        try:
            cv2.setTrackbarMax("frame", WIN, self.n_frames - 1)
            cv2.setTrackbarPos("frame", WIN, 0)
        except cv2.error:
            pass
        self.seek(0)

    def seek(self, idx):
        if not self.live:               # live: slider just takes a fresh snapshot
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        if not ok and self.frame is None:
            f = np.zeros((VIEW_H, VIEW_W, 3), np.uint8)
            cv2.putText(f, "No video from this camera", (40, 360), FONT, 1.2, (0, 0, 255), 2)
            ok = True
        if ok:
            self.frame = cv2.resize(f, (VIEW_W, VIEW_H))

    # ---------------- data helpers ----------------
    @property
    def cam_roi(self):
        return self.roi.setdefault(str(self.cam), {"lines": [], "boxes": []})

    @staticmethod
    def to_px(p):
        return int(p[0] * VIEW_W), int(p[1] * VIEW_H)

    @staticmethod
    def to_norm(p):
        return [round(min(max(p[0], 0), VIEW_W) / VIEW_W, 4),
                round(min(max(p[1], 0), VIEW_H) / VIEW_H, 4)]

    def next_box_name(self):
        used = {b["name"] for c in self.roi.values() for b in c["boxes"]}
        i = 1
        while f"C{self.cam}-P{i}" in used:
            i += 1
        return f"C{self.cam}-P{i}"

    def set_tool(self, tool):
        self.mode, self.points, self.selected = tool, [], None
        if tool == "box":
            self.msg = "PARKING BOX: click corners, then click the first corner again (or right-click / ENTER)"
        else:
            self.msg = f"{TOOLS[tool]}: click 2 points on the image"

    def finish_shape(self):
        pts = [self.to_norm(p) for p in self.points]
        r = self.cam_roi
        if self.mode == "box":
            if len(pts) < 3:
                self.msg = "A box needs at least 3 corners - keep clicking"
                return False
            name = self.next_box_name()
            r["boxes"].append({"name": name, "pts": pts})
            self.selected = ("box", len(r["boxes"]) - 1)
            self.msg = f"Added parking box {name}. Press SAVE when done."
        else:
            n = sum(l["type"] == self.mode for l in r["lines"]) + 1
            r["lines"].append({"type": self.mode, "name": f"{self.mode} {n}",
                               "dir": 0, "pts": pts})
            self.selected = ("line", len(r["lines"]) - 1)
            self.msg = f"Added {TOOLS[self.mode]} (counts both ways). FLIP to set one direction."
        self.history.append(self.selected[0])
        self.dirty, self.points, self.mode = True, [], None
        return True


    # ---------------- actions ----------------
    def do(self, action):
        if action.startswith("cam"):
            self.open_cam(int(action[3:]))
            self.msg = f"Camera {self.cam}: {CAMERAS[self.cam]['name']}"
        elif action in TOOLS:
            self.set_tool(action)
        elif action == "done" and self.mode == "box":
            self.finish_shape()
        elif action == "cancel":
            self.mode, self.points = None, []
            self.msg = "Cancelled"
        elif action == "flip":
            self.flip()
        elif action == "delete":
            if self.selected:
                self.delete(*self.selected)
            else:
                self.msg = "Click a line or box first to select it"
        elif action == "undo":
            self.undo()
        elif action == "clear":
            self.roi[str(self.cam)] = {"lines": [], "boxes": []}
            self.selected, self.dirty = None, True
            self.msg = f"Cleared camera {self.cam}"
        elif action == "save":
            save_roi(self.roi)
            self.dirty, self.quit_armed = False, False
            self.msg = f"SAVED to {ROI_FILE.name}"
            print("Saved", ROI_FILE)
        elif action == "quit":
            if self.dirty and not self.quit_armed:
                self.quit_armed = True
                self.msg = "Not saved! Click SAVE, or click QUIT again to discard"
            else:
                self.quit = True

    def flip(self):
        lines = self.cam_roi["lines"]
        if self.selected and self.selected[0] == "line":
            l = lines[self.selected[1]]
        elif lines:
            l = lines[-1]
        else:
            self.msg = "No line to flip"
            return
        l["dir"] = {0: 1, 1: -1, -1: 0}[l.get("dir", 0)]
        self.dirty = True
        self.msg = f"{l['type'].upper()} line now counts: " + \
            ("both ways" if l["dir"] == 0 else "only cars moving in the arrow direction")

    def delete(self, kind, idx):
        lst = self.cam_roi["boxes" if kind == "box" else "lines"]
        if 0 <= idx < len(lst):
            item = lst.pop(idx)
            self.msg = f"Deleted {item.get('type', 'box')} {item['name']}"
            self.dirty, self.selected = True, None

    def undo(self):
        if self.points:
            self.points.pop()
            self.msg = "Removed last point"
            return
        r = self.cam_roi
        kind = self.history.pop() if self.history else ("box" if r["boxes"] else "line")
        lst = r["boxes"] if kind == "box" else r["lines"]
        if not lst:
            lst = r["lines"] or r["boxes"]
        if not lst:
            self.msg = "Nothing to undo"
            return
        item = lst.pop()
        self.msg = f"Undo: removed {item.get('type', 'box')} {item['name']}"
        self.dirty, self.selected = True, None

    # ---------------- hit testing ----------------
    def find_point(self, x, y):
        r = self.cam_roi
        for i, l in enumerate(r["lines"]):
            for j, p in enumerate(l["pts"]):
                px, py = self.to_px(p)
                if abs(px - x) <= SNAP and abs(py - y) <= SNAP:
                    return ("line", i, j)
        for i, b in enumerate(r["boxes"]):
            for j, p in enumerate(b["pts"]):
                px, py = self.to_px(p)
                if abs(px - x) <= SNAP and abs(py - y) <= SNAP:
                    return ("box", i, j)
        return None

    def find_shape(self, x, y):
        r = self.cam_roi
        n = (x / VIEW_W, y / VIEW_H)
        for i, l in enumerate(r["lines"]):
            a, b = self.to_px(l["pts"][0]), self.to_px(l["pts"][1])
            if dist_to_segment((x, y), a, b) <= SNAP:
                return ("line", i)
        for i, b in enumerate(r["boxes"]):
            if point_in_polygon(n, b["pts"]):
                return ("box", i)
        return None

    # ---------------- mouse ----------------
    def on_mouse(self, event, x, y, flags, param):
        # top bar
        if y < BAR_H:
            if event == cv2.EVENT_LBUTTONDOWN:
                for b in self.buttons:
                    if b.hit(x, y):
                        if b.action != "quit":
                            self.quit_armed = False
                        self.do(b.action)
                        break
            return
        iy = y - BAR_H
        if iy >= VIEW_H:
            return
        self.mouse = (x, iy)

        # drawing a shape
        if self.mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                now = time.time()
                double = now - self.last_click[0] < 0.35 and \
                    abs(self.last_click[1][0] - x) < 6 and abs(self.last_click[1][1] - iy) < 6
                self.last_click = (now, (x, iy))
                if self.mode == "box":
                    if double and len(self.points) >= 3:
                        self.finish_shape()
                        return
                    if len(self.points) >= 3:
                        fx, fy = self.points[0]
                        if abs(fx - x) <= SNAP and abs(fy - iy) <= SNAP:
                            self.finish_shape()
                            return
                    self.points.append((x, iy))
                    self.msg = f"PARKING BOX: {len(self.points)} corners - click first corner to close"
                else:
                    self.points.append((x, iy))
                    if len(self.points) == 2:
                        self.finish_shape()
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.mode == "box" and len(self.points) >= 3:
                    self.finish_shape()
                elif self.points:
                    self.points.pop()
            return

        # editing (no tool)
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self.find_point(x, iy)
            if hit:
                self.drag = hit
                self.selected = hit[:2]
            else:
                self.selected = self.find_shape(x, iy)
                if self.selected:
                    kind, i = self.selected
                    item = self.cam_roi["boxes" if kind == "box" else "lines"][i]
                    self.msg = f"Selected {item.get('type', 'box')} {item['name']} " \
                               "(drag corners, FLIP, DELETE or right-click to remove)"
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag:
                kind, i, j = self.drag
                lst = self.cam_roi["boxes" if kind == "box" else "lines"]
                lst[i]["pts"][j] = self.to_norm((x, iy))
                self.dirty = True
            else:
                self.hover = self.find_point(x, iy) or self.find_shape(x, iy)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drag:
                self.msg = "Point moved. Press SAVE when done."
            self.drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            shape = self.find_shape(x, iy)
            if shape:
                self.delete(*shape)

    # ---------------- drawing ----------------
    def draw_bar(self, canvas):
        cv2.rectangle(canvas, (0, 0), (VIEW_W, BAR_H), (25, 25, 25), -1)
        self.buttons = []
        x = 8
        for cid in (1, 2, 3, 4):
            self.buttons.append(Button(x, 6, 92, 36, f"CAM {cid}", f"cam{cid}",
                                       (150, 110, 40) if cid == self.cam else (70, 70, 70)))
            x += 98
        x += 12
        for tool, label in TOOLS.items():
            col = COLORS["box" if tool == "box" else tool]
            w = 132 if tool != "box" else 136
            self.buttons.append(Button(x, 6, w, 36, label, tool, col))
            x += w + 6
        x = 8
        row2 = [("FINISH BOX", "done"), ("CANCEL", "cancel"), ("FLIP DIR", "flip"),
                ("DELETE", "delete"), ("UNDO", "undo"), ("CLEAR CAM", "clear")]
        for label, act in row2:
            self.buttons.append(Button(x, 50, 118, 36, label, act))
            x += 124
        x = VIEW_W - 2 * 124
        self.buttons.append(Button(x, 50, 118, 36, "SAVE", "save",
                                   (40, 40, 200) if self.dirty else (40, 140, 40)))
        self.buttons.append(Button(x + 124, 50, 112, 36, "QUIT", "quit", (90, 90, 90)))

        for b in self.buttons:
            bx, by, bw, bh = b.rect
            active = (b.action == self.mode) or (b.action == f"cam{self.cam}")
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), b.color, -1)
            if active:
                cv2.rectangle(canvas, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2),
                              (255, 255, 255), 2)
            mx, my = self.mouse_raw
            if b.hit(mx, my):
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (230, 230, 230), 1)
            (tw, th), _ = cv2.getTextSize(b.label, FONT, 0.5, 1)
            cv2.putText(canvas, b.label, (bx + (bw - tw) // 2, by + (bh + th) // 2),
                        FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, b.label, (bx + (bw - tw) // 2, by + (bh + th) // 2),
                        FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    mouse_raw = (0, 0)

    def draw_image(self):
        img = self.frame.copy()
        r = self.cam_roi
        for i, b in enumerate(r["boxes"]):
            pts = np.array([self.to_px(p) for p in b["pts"]], np.int32)
            sel = self.selected == ("box", i)
            col = (0, 255, 255) if sel else COLORS["box"]
            ov = img.copy()
            cv2.fillPoly(ov, [pts], col)
            img = cv2.addWeighted(ov, 0.30 if sel else 0.2, img, 0.7 if sel else 0.8, 0)
            cv2.polylines(img, [pts], True, col, 3 if sel else 2)
            for p in pts:
                cv2.circle(img, tuple(int(v) for v in p), 6, col, -1)
            cx, cy = pts.mean(axis=0).astype(int)
            cv2.putText(img, "AREA", (cx - 25, cy), FONT, 0.6, (0, 0, 0), 4)
            cv2.putText(img, "AREA", (cx - 25, cy), FONT, 0.6, col, 2)
        for i, l in enumerate(r["lines"]):
            a, b = self.to_px(l["pts"][0]), self.to_px(l["pts"][1])
            sel = self.selected == ("line", i)
            c = (0, 255, 255) if sel else COLORS[l["type"]]
            cv2.line(img, a, b, c, 5 if sel else 4)
            cv2.circle(img, a, 7, c, -1)
            cv2.circle(img, b, 7, c, -1)
            d = l.get("dir", 0)
            for dd in ([d] if d else [1, -1]):
                p0, p1 = arrow_points(a, b, dd, 40)
                cv2.arrowedLine(img, p0, p1, c, 3, tipLength=0.35)
            txt = l["type"].upper() + (" (both ways)" if not d else " (arrow way)")
            cv2.putText(img, txt, (a[0] + 8, a[1] - 10 - 24 * i), FONT, 0.65, (0, 0, 0), 4)
            cv2.putText(img, txt, (a[0] + 8, a[1] - 10 - 24 * i), FONT, 0.65, c, 2)

        # shape being drawn, with a rubber-band line to the mouse
        if self.mode and self.points:
            c = COLORS["box" if self.mode == "box" else self.mode]
            for i, p in enumerate(self.points):
                cv2.circle(img, p, 6, c, -1)
                if i:
                    cv2.line(img, self.points[i - 1], p, c, 2)
            cv2.line(img, self.points[-1], self.mouse, c, 1)
            if self.mode == "box" and len(self.points) >= 3:
                cv2.circle(img, self.points[0], SNAP, (255, 255, 255), 2)
        elif self.mode:
            cv2.circle(img, self.mouse, 5, COLORS["box" if self.mode == "box" else self.mode], 1)

        # hover hint
        if not self.mode and self.hover and len(self.hover) == 3:
            kind, i, j = self.hover
            lst = self.cam_roi["boxes" if kind == "box" else "lines"]
            if i < len(lst) and j < len(lst[i]["pts"]):
                cv2.circle(img, self.to_px(lst[i]["pts"][j]), SNAP, (255, 255, 255), 2)
        return img

    def render(self):
        canvas = np.zeros((BAR_H + VIEW_H + STATUS_H, VIEW_W, 3), np.uint8)
        self.draw_bar(canvas)
        canvas[BAR_H:BAR_H + VIEW_H] = self.draw_image()
        y0 = BAR_H + VIEW_H
        cv2.rectangle(canvas, (0, y0), (VIEW_W, y0 + STATUS_H), (25, 25, 25), -1)
        cfg = CAMERAS[self.cam]
        state = "UNSAVED" if self.dirty else "saved"
        cv2.putText(canvas, f"CAM {self.cam} {cfg['name']}  |  {state}  |  {self.msg}",
                    (10, y0 + 23), FONT, 0.55,
                    (80, 80, 255) if self.dirty else (120, 220, 120), 1, cv2.LINE_AA)
        return canvas

    # ---------------- loop ----------------
    def _mouse(self, event, x, y, flags, param):
        self.mouse_raw = (x, y)
        self.on_mouse(event, x, y, flags, param)

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, VIEW_W, BAR_H + VIEW_H + STATUS_H)
        cv2.createTrackbar("frame", WIN, 0, self.n_frames - 1, self.seek)
        cv2.setMouseCallback(WIN, self._mouse)
        while not self.quit:
            cv2.imshow(WIN, self.render())
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k in (ord("1"), ord("2"), ord("3"), ord("4")) and int(chr(k)) in CAMERAS:
                self.do(f"cam{chr(k)}")
            elif k in KEYS:
                self.do(KEYS[k])
            elif k in (13, 10):
                self.do("done")
            elif k == 27:
                self.do("cancel")
            elif k == ord("f"):
                self.do("flip")
            elif k in (ord("u"), 8, 127):
                self.do("undo")
            elif k == ord("c"):
                self.do("clear")
            elif k == ord("w"):
                self.do("save")
            elif k == ord("q"):
                self.do("quit")
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=1)
    ap.add_argument("--live", action="store_true", help="snapshot from the live RTSP camera")
    a = ap.parse_args()
    Editor(a.cam, a.live).run()
