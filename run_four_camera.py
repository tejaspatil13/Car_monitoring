"""Four-camera car monitoring.

One window: 2x2 camera video (left) + dashboard table (right).
--no-video shows only the dashboard.

Lines & boxes come from config/roi.json  ->  edit with  python roi_editor.py
Settings (cameras on/off, windows on/off) are in config.py

Keys (any window):  Q quit | SPACE pause
Command line:  --no-video  --no-dashboard  --headless  --no-save  --fast  --keep-db
"""

import argparse
import time
from collections import deque
from datetime import datetime, timedelta

import cv2
import numpy as np
from ultralytics import YOLO

import config as C
from dashboard import Dashboard
from plate_reader import PlateReader, PlateVote, is_valid
from registry import Registry, plate_distance
from reid_matcher import ReIDMatcher
from sources import LiveSource, open_capture
from device import best_device
from roi import COLORS, arrow_points, crossing_ok, dist_to_segment, load_roi, point_in_polygon

TILE_W, TILE_H = 640, 360
WINDOW = "Car Monitoring"
VEHICLE_CLASSES = sorted(C.CAR_CLASSES | C.BIKE_CLASSES)


# ---------------- tracked vehicle ----------------

class Track:
    def __init__(self, tid, t):
        self.id = tid
        self.vote = PlateVote()
        self.hist = deque(maxlen=12)            # (t, x, y) bottom-centre, normalised
        self.car_score = self.bike_score = 0.0
        self.pending = {}                       # line index -> crossing time
        self.done_lines = set()
        self.result = ""
        self.last_seen = t
        self.n = 0
        self.box = None
        self.best_crop, self.best_area = None, 0
        self.plate_img, self.plate_conf = None, 0.0
        self.area_since = None      # entered the parking area at (video t)
        self.left_since = None
        self.parked = False
        self.park_checked = False
        self.entry_id = None
        self.crops = []                 # (area, crop) for the appearance fingerprint
        self.reid_tried = 0             # crops count at the last re-ID attempt
        self.match_how = ""             # "plate" or "look 0.72"

    @property
    def is_car(self):
        return self.car_score >= self.bike_score

    def moving(self):
        if len(self.hist) < 2:
            return True
        _, x0, y0 = self.hist[0]
        _, x1, y1 = self.hist[-1]
        return abs(x1 - x0) + abs(y1 - y0) > C.MOVING_THRESH


# ---------------- camera ----------------

class Camera:
    def __init__(self, cid, cfg, roi, device, live=False):
        self.id, self.cfg, self.name = cid, cfg, cfg["name"]
        self.lines = roi.get("lines", [])
        # all boxes of a camera together form ONE parking area (no slot names)
        self.areas = [[tuple(p) for p in b["pts"]] for b in roi.get("boxes", [])]
        self.area_name = "Parking"
        self.has_roi = bool(self.lines or self.areas)
        self.live = live
        if live:
            self.src = LiveSource(cfg["rtsp"], cfg["name"])
            print(f"  CAM {cid}: waiting for {cfg['name']} stream ...")
            if not self.src.wait_first_frame():
                raise RuntimeError(f"No video from {cfg['name']} ({cfg['rtsp']})")
            self.fps = 25.0
            self.h, self.w = self.src.frame.shape[:2]
            self.t0 = time.time()
            self.last_no = 0
        else:
            path = C.VIDEO_DIR / cfg["file"]
            self.cap = open_capture(path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ignore_px = [(a * self.w, b * self.h, c * self.w, d * self.h)
                          for a, b, c, d in cfg.get("ignore", [])]
        self.start = (datetime.strptime(cfg["start"], "%Y-%m-%d %H:%M:%S")
                      if cfg.get("start") and not live else None)
        self.model = YOLO(str(C.MODEL_PATH))
        self.device = device
        self.frame_idx, self.frame, self.finished = 0, None, False
        self.tile = np.zeros((TILE_H, TILE_W, 3), np.uint8)
        self.tracks = {}
        self.flash = {}                          # line index -> until (video t)

    @property
    def t(self):
        """Seconds since start (video time for files, wall time for live)."""
        return time.time() - self.t0 if self.live else self.frame_idx / self.fps

    def now(self):
        """CCTV time of the current frame (recordings) or wall clock (live)."""
        return self.start + timedelta(seconds=self.t) if self.start else datetime.now()

    def advance_to(self, t_target):
        if self.live:                       # always the newest frame
            no, frame = self.src.latest(self.last_no)
            if frame is None:
                return False
            self.last_no, self.frame = no, frame
            self.frame_idx += 1
            return True
        target = int(round(t_target * self.fps))
        if target <= self.frame_idx and self.frame is not None:
            return False
        while self.frame_idx < target - 1:
            if not self.cap.grab():
                self.finished = True
                return False
            self.frame_idx += 1
        ok, frame = self.cap.read()
        if not ok:
            self.finished = True
            return False
        self.frame_idx += 1
        self.frame = frame
        return True

    def detect(self):
        r = self.model.track(self.frame, persist=True, classes=VEHICLE_CLASSES,
                             conf=C.YOLO_CONF, imgsz=C.YOLO_IMGSZ,
                             tracker="bytetrack.yaml", device=self.device,
                             verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        return list(zip(r.boxes.xyxy.cpu().numpy().astype(int),
                        r.boxes.id.cpu().numpy().astype(int),
                        r.boxes.cls.cpu().numpy().astype(int)))


# ---------------- app ----------------

class App:
    def __init__(self, args):
        C.OUTPUT.mkdir(exist_ok=True)
        C.CAPTURE_DIR.mkdir(exist_ok=True)
        self.show_video = C.SHOW_VIDEO and not args.no_video
        self.show_dash = C.SHOW_DASHBOARD and not args.no_dashboard
        self.headless = args.headless
        self.save = C.SAVE_VIDEOS and not args.no_save
        self.realtime = not args.fast

        self.registry = Registry(C.DB_PATH, fresh=not args.keep_db)
        device = best_device(args.device)
        self.live = args.live or C.MODE == "live"
        self.layout = args.layout or C.DASHBOARD_POSITION
        roi = load_roi()
        print(f"Loading YOLO on {device} for cameras {C.ENABLED_CAMERAS} ...")
        print(f"Mode: {'LIVE cameras' if self.live else 'recorded files'}   device: {device}")
        self.cams = {cid: Camera(cid, C.CAMERAS[cid], roi.get(str(cid), {}), device, self.live)
                     for cid in C.ENABLED_CAMERAS}
        for cam in self.cams.values():
            desc = [l["type"] for l in cam.lines] + \
                ([f"parking area ({len(cam.areas)} boxes)"] if cam.areas else [])
            print(f"  CAM {cam.id} {cam.name:<13} ROI: {', '.join(desc) or '(none)'}")
        print("Loading plate reader ...")
        self.reader = PlateReader()
        self.matcher = ReIDMatcher() if C.REID_ENABLED else None
        self.identifier = None
        if C.IDENTIFY_ENABLED:
            from car_identity import CarIdentifier
            print("Loading car model/colour identifier (CLIP ViT-L-14) ...")
            self.identifier = CarIdentifier()
        if self.matcher:
            print("Loaded vehicle re-ID (FastReID VERI-Wild R50-IBN)")
        self.dashboard = Dashboard()
        self.recent = {}
        self.clock = 0.0
        self.paused = False
        self.banner = ("", 0.0, (255, 255, 255))
        self.photo_n = 0

    # ================= processing =================
    def process(self, cam):
        frame = cam.frame
        H, W = frame.shape[:2]
        t = cam.t
        seen = set()

        for (x1, y1, x2, y2), tid, cls in cam.detect():
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            tr = cam.tracks.get(tid) or cam.tracks.setdefault(tid, Track(tid, t))
            seen.add(tid)
            tr.last_seen, tr.box = t, (x1, y1, x2, y2)
            tr.n += 1
            area = (x2 - x1) * (y2 - y1)
            if cls in C.CAR_CLASSES:
                tr.car_score += area
            else:
                tr.bike_score += area
            pt = ((x1 + x2) / 2 / W, y2 / H)                 # ground point
            prev = tr.hist[-1] if tr.hist else None
            tr.hist.append((t, *pt))
            if area > tr.best_area:
                tr.best_area, tr.best_crop = area, frame[y1:y2, x1:x2].copy()
            # crops for the appearance fingerprint: whole car, not cut by the frame edge
            edge = x1 < 5 or y1 < 5 or x2 > W - 5 or y2 > H - 5
            if self.matcher and tr.n % 3 == 0 and not edge and x2 - x1 > 60:
                tr.crops.append((area, frame[y1:y2, x1:x2].copy()))
                if len(tr.crops) > 12:
                    tr.crops.remove(min(tr.crops, key=lambda z: z[0]))
            # follow entered cars on any camera by their look
            if (self.matcher and not tr.entry_id and tr.is_car and
                    len(tr.crops) >= C.REID_MIN_CROPS and
                    len(tr.crops) >= tr.reid_tried + C.REID_MIN_CROPS):
                self.try_reid(cam, tr)

            if not cam.has_roi or not tr.is_car:
                continue

            # ---- line crossings ----
            near = False
            for i, line in enumerate(cam.lines):
                a, b = line["pts"]
                if dist_to_segment(pt, a, b) < C.OCR_NEAR_LINE:
                    near = True
                if (i not in tr.pending and i not in tr.done_lines and prev is not None
                        and crossing_ok(prev[1:], pt, line)):
                    tr.pending[i] = t
                    cam.flash[i] = t + 1.0
                    self.log(cam, "LINE_CROSSING", detail=f"{line['type']} track={tid}")
                    print(f"[{cam.name}] track {tid} crossed {line['type'].upper()} line")

            # ---- parking box membership (lower-centre of the car) ----
            park_pt = ((x1 + x2) / 2 / W, (y1 + 0.8 * (y2 - y1)) / H)
            inside = any(point_in_polygon(park_pt, a) for a in cam.areas)
            if inside:
                tr.left_since = None
                if tr.area_since is None:
                    tr.area_since = t
            elif tr.parked:
                if tr.left_since is None:
                    tr.left_since = t
                elif t - tr.left_since >= C.PARK_RELEASE:
                    self.left_parking(cam, tr)
            else:
                tr.area_since = None
            area_needs_plate = inside and not tr.entry_id and not tr.park_checked

            # ---- plate OCR ----
            want = bool(tr.pending) or (near and tr.moving()) or \
                (area_needs_plate and tr.n % (C.OCR_EVERY * 3) == 0)
            if want and tr.n % C.OCR_EVERY == 0 and (x2 - x1) > 60:
                self.read_plate(cam, tr, frame, (x1, y1, x2, y2))

            # ---- parked: inside the area long enough ----
            if (inside and not tr.parked and not tr.park_checked
                    and t - tr.area_since >= C.PARK_DWELL):
                if tr.entry_id or tr.vote.confirmed():
                    self.mark_parked(cam, tr)
                elif t - tr.area_since >= C.PARK_DWELL + C.PENDING_TIMEOUT:
                    tr.park_checked = True      # plate never readable -> not followed

            # ---- finish pending crossings ----
            for i, t_cross in list(tr.pending.items()):
                if tr.vote.confirmed() or t - t_cross > C.PENDING_TIMEOUT:
                    self.finalize(cam, tr, i)

        for tid in list(cam.tracks):
            tr = cam.tracks[tid]
            if tid not in seen and t - tr.last_seen > C.TRACK_EXPIRE:
                if self.matcher and tr.entry_id and tr.match_how == "plate":
                    self.matcher.add_views(tr.entry_id, tr)   # full track = better view
                self.reidentify(tr)
                for i in list(tr.pending):
                    self.finalize(cam, tr, i)
                del cam.tracks[tid]

    def read_plate(self, cam, tr, frame, box):
        x1, y1, x2, y2 = box
        try:
            for text, conf, pbox in self.reader.read(frame[y1:y2, x1:x2], (x1, y1),
                                                     cam.ignore_px):
                tr.vote.add(text, conf)
                if is_valid(text) and conf > tr.plate_conf:
                    px1, py1, px2, py2 = map(int, pbox)
                    tr.plate_conf = conf
                    tr.plate_img = frame[py1:py2, px1:px2].copy()
        except Exception as e:
            print("OCR error:", e)

    # ================= parking area =================
    def mark_parked(self, cam, tr):
        """Car stayed in the parking area: mark PARKING if it entered via the gate."""
        tr.park_checked = True
        plate, conf = tr.vote.best()
        if tr.entry_id and (not plate or not is_valid(plate)):
            plate = self.registry.get(tr.entry_id)["plate"]
        car_photo = self.save_img(tr.best_crop, f"parking_{plate}_car")
        plate_photo = self.save_img(tr.plate_img, f"parking_{plate}_plate")
        eid = self.apply_event("parking", cam, tr, plate or "UNREAD", conf,
                               slot=cam.area_name, car_photo=car_photo,
                               plate_photo=plate_photo)
        tr.parked = eid is not None

    def left_parking(self, cam, tr):
        tr.parked, tr.area_since, tr.left_since = False, None, None
        tr.park_checked = False
        self.log(cam, "PARK_OUT", tr.entry_id, f"area={cam.area_name}")
        print(f"[{cam.name}] entry #{tr.entry_id} left the parking area")

    def parking_summary(self):
        """Per camera: entered cars currently parked in its area."""
        out = []
        for cam in self.cams.values():
            if not cam.areas:
                continue
            cars = [tr for tr in cam.tracks.values()
                    if tr.parked and cam.t - tr.last_seen < C.TRACK_EXPIRE]
            plates = []
            for tr in cars:
                row = self.registry.get(tr.entry_id) if tr.entry_id else None
                plates.append(row["plate"] if row else "?")
            out.append({"name": cam.area_name, "count": len(cars), "plates": plates})
        return out

    # ================= events =================
    def identify(self, eid, tr, again=False):
        """Car model + colour from the gate track -> car_name column."""
        if not self.identifier:
            return
        crops = [c for _, c in sorted(tr.crops, key=lambda z: -z[0])[:6]] or \
                ([tr.best_crop] if tr.best_crop is not None else [])
        if not crops:
            return
        r = self.identifier.identify(crops)
        name = f"{r['model']} · {r['colour']}" if r["model"] != "?" else f"? · {r['colour']}"
        self.registry.update(eid, car_name=name)
        print(f"    car #{eid}{' (full track)' if again else ''}: model={r['model']} "
              f"({r['model_conf']})  colour={r['colour']}")

    def reidentify(self, tr):
        """Gate track finished: if the model was '?', try again with all its views."""
        if not (self.identifier and tr.entry_id and getattr(tr, "gate_track", False)):
            return
        row = self.registry.get(tr.entry_id)
        if row and (row.get("car_name") or "?").startswith("?"):
            self.identify(tr.entry_id, tr, again=True)

    def try_reid(self, cam, tr):
        tr.reid_tried = len(tr.crops)
        active = [r for r in self.registry.entries() if r["status"] != "EXITED"]
        # a readable plate always wins: skip cars whose plate clearly differs
        plate, _ = tr.vote.best()
        if plate and is_valid(plate):
            active = [r for r in active if plate_distance(plate, r["plate"]) <= 2]
        if C.REID_REQUIRE_AFTER_ENTRY:
            now = cam.now().isoformat(sep=" ")
            active = [r for r in active if (r["entry_time"] or "") <= now]
        hit = self.matcher.match(tr, [r["id"] for r in active])
        if not hit:
            return
        eid, score = hit
        tr.entry_id, tr.match_how = eid, f"look {score:.2f}"
        row = self.registry.get(eid)
        self.log(cam, "SEEN", eid, f"by look {score:.2f} track={tr.id}")
        print(f"[{cam.name}] track {tr.id} looks like #{eid} {row['plate']} ({score:.2f})")

    def finalize(self, cam, tr, line_idx):
        """A line crossing is complete: plate confirmed, timed out, or car gone."""
        tr.pending.pop(line_idx, None)
        tr.done_lines.add(line_idx)
        line = cam.lines[line_idx]
        plate, conf = tr.vote.best()
        if not plate or not is_valid(plate):
            plate, conf = "UNREAD", 0.0
        car_photo = self.save_img(tr.best_crop, f"{line['type']}_{plate}_car")
        plate_photo = self.save_img(tr.plate_img, f"{line['type']}_{plate}_plate")
        self.apply_event(line["type"], cam, tr, plate, conf, slot=line["name"],
                         car_photo=car_photo, plate_photo=plate_photo)

    def apply_event(self, kind, cam, tr, plate, conf, slot="", car_photo="", plate_photo=""):
        """Write an entry / staff / parking / exit event to the register."""
        if plate != "UNREAD":
            key = (kind, plate)
            if key in self.recent and time.time() - self.recent[key] < C.PLATE_COOLDOWN:
                tr.result = plate
                return None
            self.recent[key] = time.time()
        when = cam.now()
        ts = when.isoformat(sep=" ", timespec="seconds")
        match = None
        if kind != "entry":
            match = self.registry.find_plate(plate)
            if match:
                tr.match_how = "plate"
            elif tr.entry_id:                       # followed by look
                row = self.registry.get(tr.entry_id)
                if plate != "UNREAD" and plate_distance(plate, row["plate"]) > 2:
                    # plate read now says it is a different car -> drop the look match
                    print(f"[{cam.name}] plate {plate} != #{row['id']} {row['plate']}: look match dropped")
                    self.log(cam, "LOOK_MATCH_REJECTED", None, f"plate={plate} was #{row['id']}")
                    tr.entry_id, tr.match_how = None, ""
                else:
                    match = row
        if kind != "entry" and not match:
            # Only cars that came in through the gate are followed.
            self.log(cam, f"IGNORED_{kind.upper()}", None, f"plate={plate} not entered")
            return None
        eid = None
        if match:
            tr.entry_id = match["id"]
            if self.matcher and tr.match_how == "plate":
                self.matcher.add_views(match["id"], tr)

        if kind == "entry":
            eid = self.registry.add_entry("SERVICE", plate, conf, entry_time=ts,
                                          entry_cam=cam.name, entry_photo=car_photo,
                                          entry_plate_photo=plate_photo)
            tr.entry_id, tr.match_how = eid, "plate"
            tr.gate_track = True
            self.identify(eid, tr)
            if self.matcher:
                self.matcher.add_views(eid, tr)
            msg, col = f"ENTRY #{eid} {plate}", (80, 200, 80)

        elif kind == "staff":
            if match:
                eid = match["id"]
                self.registry.update(eid, status="STAFF", staff_time=ts)
            else:
                eid = self.registry.add_entry("STAFF", plate, conf, staff_time=ts)
            msg, col = f"STAFF CAR #{eid} {plate}", (255, 150, 40)

        elif kind == "parking":
            fields = dict(status="PARKING", parking_time=ts, parking_slot="Parking",
                          parking_photo=car_photo, parking_plate_photo=plate_photo)
            if match:
                eid = match["id"]
                self.registry.update(eid, **fields)
            elif plate != "UNREAD":
                fields.pop("status")
                eid = self.registry.add_entry("PARKING", plate, conf, **fields)
            tr.parked = True
            msg, col = f"PARKED {plate}", (255, 0, 200)

        elif kind == "exit":
            fields = dict(status="EXITED", exit_time=ts, exit_photo=car_photo,
                          exit_plate_photo=plate_photo)
            if match:
                eid = match["id"]
                if match.get("entry_time"):
                    t_in = datetime.fromisoformat(match["entry_time"])
                    fields["stay_seconds"] = (when - t_in).total_seconds()
                self.registry.update(eid, **fields)
            else:
                fields.pop("status")
                eid = self.registry.add_entry("EXITED", plate, conf, **fields)
            msg, col = f"EXIT #{eid} {plate}", (120, 120, 255)
        else:
            return None

        self.log(cam, kind.upper(), eid, f"plate={plate} track={tr.id}")
        tr.result = msg
        self.banner = (msg, self.clock + 4.0, col)
        print(f"==> {msg}   ({cam.name}, {ts}, matched={'yes' if match else 'no'})")
        return eid

    def save_img(self, img, tag):
        if img is None or img.size == 0:
            return ""
        self.photo_n += 1
        path = C.CAPTURE_DIR / f"{self.photo_n:04d}_{tag}.jpg"
        cv2.imwrite(str(path), img)
        return str(path)

    def log(self, cam, event, eid=None, detail=""):
        self.registry.log(cam.now().isoformat(sep=" ", timespec="seconds"),
                          cam.name, event, eid, detail)

    # ================= drawing =================
    def draw_tile(self, cam):
        if cam.frame is None:
            return
        img = cv2.resize(cam.frame, (TILE_W, TILE_H))
        sx, sy = TILE_W / cam.w, TILE_H / cam.h

        def P(p):
            return int(p[0] * TILE_W), int(p[1] * TILE_H)

        if cam.areas:
            ov = img.copy()
            for a in cam.areas:
                cv2.fillPoly(ov, [np.array([P(p) for p in a], np.int32)], COLORS["box"])
            img = cv2.addWeighted(ov, 0.18, img, 0.82, 0)
            for a in cam.areas:
                cv2.polylines(img, [np.array([P(p) for p in a], np.int32)], True,
                              COLORS["box"], 1)
            n = sum(tr.parked for tr in cam.tracks.values())
            top = min((P(p) for a in cam.areas for p in a), key=lambda q: q[1])
            self._label(img, f"PARKING AREA  {n} parked", (top[0], max(40, top[1] - 4)),
                        COLORS["box"])

        for i, line in enumerate(cam.lines):
            col = (0, 255, 0) if cam.flash.get(i, -1) > cam.t else COLORS[line["type"]]
            a, b = P(line["pts"][0]), P(line["pts"][1])
            cv2.line(img, a, b, col, 3)
            d = line.get("dir", 0)
            for dd in ([d] if d else [1, -1]):
                p0, p1 = arrow_points(a, b, dd, 22)
                if d or dd == 1:
                    cv2.arrowedLine(img, p0, p1, COLORS[line["type"]], 2, tipLength=0.4)
            self._label(img, line["type"].upper(), (a[0] + 6, a[1] - 6 - 18 * i), col)

        for tr in cam.tracks.values():
            if tr.box is None or cam.t - tr.last_seen > 0.3:
                continue
            if not (tr.entry_id or tr.pending):
                continue            # only show cars being registered / followed
            x1, y1, x2, y2 = tr.box
            p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
            if tr.parked:
                color = (255, 0, 200)
            elif tr.result:
                color = (80, 220, 80)
            elif tr.pending:
                color = (0, 200, 255)
            else:
                color = (255, 255, 255) if tr.is_car else (150, 150, 150)
            cv2.rectangle(img, p1, p2, color, 2)
            plate, _ = tr.vote.best()
            if tr.entry_id:
                row = self.registry.get(tr.entry_id)
                text = f"#{tr.entry_id} {row['plate'] if row else ''}" + \
                    (f" ({tr.match_how})" if tr.match_how.startswith("look") else "")
            else:
                text = f"{tr.id}" + (f" {plate}" if plate and is_valid(plate) else "")
            text += "" if tr.is_car else " bike"
            self._label(img, text, (p1[0], max(14, p1[1] - 4)), color)
            sub = tr.result or ("reading plate..." if tr.pending else "")
            if sub:
                self._label(img, sub, (p1[0], min(TILE_H - 4, p2[1] + 16)), color)

        cv2.rectangle(img, (0, 0), (TILE_W, 24), (0, 0, 0), -1)
        state = " ENDED" if cam.finished else ""
        cv2.putText(img, f"CAM {cam.id} {cam.name} ({cam.cfg['device']}){state}", (8, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, cam.now().strftime("%H:%M:%S"), (TILE_W - 80, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.rectangle(img, (0, 0), (TILE_W - 1, TILE_H - 1), (60, 60, 60), 1)
        cam.tile = img

    @staticmethod
    def _label(img, text, org, color):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        x, y = org
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (0, 0, 0), -1)
        cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def compose_video(self):
        grid = np.zeros((TILE_H * 2, TILE_W * 2, 3), np.uint8)
        for cid in (1, 2, 3, 4):
            r, c = (cid - 1) // 2, (cid - 1) % 2
            if cid in self.cams:
                grid[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = self.cams[cid].tile
            else:
                cv2.putText(grid, f"CAM {cid} OFF", (c * TILE_W + 250, r * TILE_H + 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        text, until, col = self.banner
        if text and self.clock < until:
            cv2.rectangle(grid, (TILE_W - 220, TILE_H - 22), (TILE_W + 220, TILE_H + 22), col, -1)
            cv2.putText(grid, text, (TILE_W - 205, TILE_H + 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 0, 0), 2)
        return grid

    def compose_dashboard(self):
        clocks = " | ".join(f"C{c.id} {c.now():%H:%M:%S}" for c in self.cams.values())
        return self.dashboard.render(self.registry.entries(), [], clocks,
                                     self.registry.activity())

    # ================= main loop =================
    def compose_combined(self):
        """One screen: camera grid (left) + dashboard (right, same height)."""
        dash = self.compose_dashboard() if self.show_dash else None
        if not self.show_video:
            return dash
        grid = self.compose_video()
        if dash is None:
            return grid
        if self.layout == "below":
            w = grid.shape[1]
            dash = cv2.resize(dash, (w, int(dash.shape[0] * w / dash.shape[1])),
                              interpolation=cv2.INTER_AREA)
            return np.vstack([grid, dash])
        h = grid.shape[0]
        dash = cv2.resize(dash, (int(dash.shape[1] * h / dash.shape[0]), h),
                          interpolation=cv2.INTER_AREA)
        return np.hstack([grid, dash])

    # ================= main loop =================
    def run(self):
        writer = None
        windows = not self.headless and (self.show_video or self.show_dash)
        if windows:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        print("\nMONITORING STARTED  (Q quit, SPACE pause)\n")
        start, done, screen = time.time(), False, None
        while True:
            t0 = time.time()
            if not self.paused and not done:
                self.clock += C.STEP_SECONDS
                step = int(round(self.clock / C.STEP_SECONDS))
                for cam in self.cams.values():
                    if cam.finished:
                        continue
                    if cam.advance_to(self.clock):
                        if cam.has_roi or (self.show_video and step % C.IDLE_CAM_EVERY == 0):
                            self.process(cam)
                    elif cam.finished:
                        self.end_of_video(cam)
                    if self.show_video:
                        self.draw_tile(cam)
                screen = self.compose_combined()
                if self.save and screen is not None:
                    if writer is None:
                        writer = cv2.VideoWriter(str(C.OUTPUT / "monitor.mp4"),
                                                 cv2.VideoWriter_fourcc(*"mp4v"),
                                                 1.0 / C.STEP_SECONDS,
                                                 (screen.shape[1], screen.shape[0]))
                        if windows:
                            cv2.resizeWindow(WINDOW, min(1700, screen.shape[1]),
                                             int(screen.shape[0] * min(1700, screen.shape[1])
                                                 / screen.shape[1]))
                    writer.write(screen)
                if step % 100 == 0:
                    el = time.time() - start
                    if self.live:
                        print(f"  running {el:5.0f}s  ({step / el:.1f} analysed frames/s per camera,"
                              f" target {1 / C.STEP_SECONDS:.0f})")
                    else:
                        print(f"  video {self.clock:5.1f}s in {el:5.1f}s "
                              f"({self.clock / el:.2f}x real time)")
                if all(c.finished for c in self.cams.values()):
                    done = True
                    print("All videos finished." + (" Press Q to close." if windows else ""))
                    if not windows:
                        break
            if windows:
                if screen is not None:
                    cv2.imshow(WINDOW, screen)
                wait = C.STEP_SECONDS - (time.time() - t0) if self.realtime else 0
                key = cv2.waitKey(max(1, int(wait * 1000))) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    self.paused = not self.paused
            else:                                   # no window (headless)
                if self.paused:
                    break
                if self.realtime or self.live:      # pace the loop
                    time.sleep(max(0.0, C.STEP_SECONDS - (time.time() - t0)))
        if writer:
            writer.release()
            print("Saved video:", C.OUTPUT / "monitor.mp4")
        self.shutdown()

    def end_of_video(self, cam):
        for tr in cam.tracks.values():
            if self.matcher and tr.entry_id and tr.match_how == "plate":
                self.matcher.add_views(tr.entry_id, tr)
            self.reidentify(tr)
            for i in list(tr.pending):
                self.finalize(cam, tr, i)

    def shutdown(self):
        for cam in self.cams.values():
            (cam.src if cam.live else cam.cap).release()
        cv2.destroyAllWindows()
        self.registry.export_csv(C.OUTPUT / "entries.csv")
        print("\nFinal register:")
        for r in self.registry.entries():
            print(f"  #{r['id']:<3} {r['plate']:<12} {r['status']:<8} "
                  f"entry={r['entry_time'] or '-'}  parking={r['parking_slot'] or '-'}  "
                  f"exit={r['exit_time'] or '-'}")
        self.registry.close()
        print(f"Saved: {C.DB_PATH}  and  {C.OUTPUT / 'entries.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true", help="hide the camera window")
    ap.add_argument("--no-dashboard", action="store_true", help="hide the dashboard window")
    ap.add_argument("--headless", action="store_true", help="no windows at all")
    ap.add_argument("--no-save", action="store_true", help="don't write output videos")
    ap.add_argument("--fast", action="store_true", help="don't throttle to real time")
    ap.add_argument("--keep-db", action="store_true", help="keep entries from earlier runs")
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--live", action="store_true", help="use the live RTSP cameras (config.py)")
    ap.add_argument("--layout", choices=["right", "below"], help="dashboard right of or below the cameras")
    App(ap.parse_args()).run()


if __name__ == "__main__":
    main()
