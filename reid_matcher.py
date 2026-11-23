"""Follow entered cars across cameras by appearance.

Each entered car (gate entry) gets one or more "views": fingerprints from
tracks where its plate was read (gate front view, WS-exit rear view, ...).
A car seen on another camera without a readable plate is compared with the
views of the cars currently inside; if one is clearly the most similar, the
track is linked to that entry.

Tested on test3 (FastReID VERI-Wild, pooled feature, mean-centred):
true match ranked first in 7 of 10 cross-camera cases; true pairs scored
0.55-0.74, wrong pairs 99% below 0.63.
"""

from pathlib import Path

import numpy as np

import config as C
from reid import ReID

MEAN_FILE = Path(__file__).resolve().parent / "models" / "reid" / "mean_pool.npy"


class ReIDMatcher:
    def __init__(self):
        self.reid = ReID()
        self.reid.net.feat = "pool"          # tested better than the BN-neck output
        self.mu = np.load(MEAN_FILE) if MEAN_FILE.exists() else None
        self.views = {}                      # entry_id -> {view key: vector}

    def fingerprint(self, tr, max_crops=8):
        crops = [c for _, c in sorted(tr.crops, key=lambda z: -z[0])[:max_crops]]
        if not crops:
            return None
        v = self.reid.embed(crops)
        # running mean of all features: subtracting it removes what every car
        # shares and spreads the scores of different cars apart
        m = v.mean(0)
        self.mu = m if self.mu is None else 0.98 * self.mu + 0.02 * m
        x = v - self.mu
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        f = x.mean(0)
        return f / (np.linalg.norm(f) + 1e-9)

    def add_views(self, entry_id, tr):
        """Store this track's look as a view of entry_id (replaces same track)."""
        if len(tr.crops) < 3:
            return
        f = self.fingerprint(tr)
        if f is not None:
            self.views.setdefault(entry_id, {})[id(tr)] = f

    def match(self, tr, candidate_ids):
        """Return (entry_id, score) if tr clearly looks like one candidate."""
        cands = [e for e in candidate_ids if self.views.get(e)]
        if not cands:
            return None
        f = self.fingerprint(tr)
        if f is None:
            return None
        scores = sorted(((max(float(f @ v) for v in self.views[e].values()), e)
                         for e in cands), reverse=True)
        best, eid = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.0
        if len(scores) == 1:
            ok = best >= C.REID_THRESH_SINGLE
        else:
            ok = best >= C.REID_THRESH and best - second >= C.REID_MARGIN
        return (eid, best) if ok else None
