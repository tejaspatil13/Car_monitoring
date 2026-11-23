"""Car model + colour from car crops.

MODEL - two ways, best first:
  1. Reference photos (few-shot): models/car_refs/<Model>/*.jpg are real photos
     from OUR cameras. A car is compared with them (CLIP ViT-L-14 image
     features); if it clearly looks most like one model -> that model.
     Test on test3 (each car tested without its own photos): WagonR 5/5.
     Add photos with:  python label_cars.py   (more photos = better).
  2. CLIP text (zero-shot), Maruti list only - used only when very confident.
  Otherwise "?"  (never show a wrong guess).

COLOUR - CLIP over colour words. White and Silver are kept apart; most cars
  are white, so Silver needs to be clearly stronger than White.
  Test on 40 test3 cars: 36/40 correct (rule-based HSV: 33/40).
"""

from pathlib import Path

import cv2
import numpy as np
import torch

REF_DIR = Path(__file__).resolve().parent / "models" / "car_refs"

MARUTI = ["Alto", "Alto K10", "S-Presso", "Celerio", "WagonR", "Swift", "Dzire",
          "Baleno", "Ignis", "Brezza", "Fronx", "Ertiga", "XL6", "Ciaz",
          "Eeco", "Grand Vitara", "Jimny", "Invicto"]
# only very distinctive non-Maruti vehicles (look-alikes such as Innova were
# beating Ertiga, so they are left out - the service center is Maruti only)
OTHERS = ["Mahindra Thar", "Tata Ace mini truck", "Mahindra Bolero pickup"]

TEXT_MIN_CONF = 0.85     # zero-shot text: only when very sure
REF_MIN_SIM = 0.90       # reference photos: similarity needed ...
REF_MARGIN = 0.02        # ... and lead over the 2nd model
SILVER_BIAS = 1.5        # Silver only if p(silver) > 1.5 x p(white)

COLOUR_WORDS = ["white", "silver", "grey", "black", "blue", "red", "brown",
                "beige", "orange", "green", "yellow"]


class CarIdentifier:
    def __init__(self, device=None):
        import open_clip
        if device is None:
            from device import best_device
            device = best_device()
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="datacomp_xl_s13b_b90k")
        self.model.eval().to(device)
        self.tok = open_clip.get_tokenizer("ViT-L-14")

        self.names = MARUTI + OTHERS
        full = [f"Maruti Suzuki {m}" for m in MARUTI] + OTHERS
        self.text = self._text_set([[f"a photo of a {n}, a type of car.",
                                     f"a CCTV image of a {n} car."] for n in full])
        self.colour_text = self._text_set([[f"a photo of a {c} car."] for c in COLOUR_WORDS])
        self.load_refs()

    def _text_set(self, prompt_lists):
        with torch.no_grad():
            out = []
            for ps in prompt_lists:
                t = self.model.encode_text(self.tok(ps).to(self.device))
                t = (t / t.norm(dim=-1, keepdim=True)).mean(0)
                out.append(t / t.norm())
            return torch.stack(out)

    @torch.no_grad()
    def _features(self, crops):
        from PIL import Image
        feats = []
        for i in range(0, len(crops), 16):
            x = torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)))
                             for c in crops[i:i + 16]]).to(self.device)
            f = self.model.encode_image(x)
            feats.append(f / f.norm(dim=-1, keepdim=True))
        return torch.cat(feats)

    # ---------------- reference photos ----------------
    def load_refs(self):
        """Read models/car_refs/<Model>/*.jpg (call again after labelling)."""
        self.refs = {}
        if not REF_DIR.exists():
            return
        for d in sorted(p for p in REF_DIR.iterdir() if p.is_dir()):
            imgs = [cv2.imread(str(f)) for f in sorted(d.glob("*.jpg"))]
            imgs = [i for i in imgs if i is not None]
            if imgs:
                self.refs[d.name] = self._features(imgs)
        if self.refs:
            print("  car reference photos:",
                  ", ".join(f"{k} {len(v)}" for k, v in self.refs.items()))

    def _by_refs(self, feats):
        if not self.refs:
            return None, 0.0, []
        scores = []
        for model, R in self.refs.items():
            s = (feats @ R.T).max(1).values                 # best ref per query crop
            top = torch.sort(s).values[-3:]
            scores.append((float(top.mean()), model))
        scores.sort(reverse=True)
        best, model = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        ok = best >= REF_MIN_SIM and best - second >= REF_MARGIN
        return (model if ok else None), best, scores

    # ---------------- colour ----------------
    def colour(self, feats):
        """White and Silver separate. Silver only if it is the top colour AND
        clearly stronger than White (most cars here are white)."""
        p = (100.0 * feats @ self.colour_text.T).softmax(-1).mean(0).cpu().numpy()
        probs = dict(zip(COLOUR_WORDS, p))
        best = max(probs, key=probs.get)
        if best == "silver" and probs["silver"] < SILVER_BIAS * probs["white"]:
            best = "white"
        if best == "silver":
            return "Silver"
        if best == "white":
            return "White"
        return best.capitalize()

    # ---------------- both ----------------
    def identify(self, crops, max_crops=6):
        crops = [c for c in crops if c is not None and c.size][:max_crops]
        if not crops:
            return {"model": "?", "model_conf": 0.0, "colour": "?", "how": ""}
        feats = self._features(crops)

        model, ref_score, ref_scores = self._by_refs(feats)
        how, conf = "photos", ref_score
        if model is None:
            p = (100.0 * feats @ self.text.T).softmax(-1).mean(0).cpu().numpy()
            i = int(np.argmax(p))
            if p[i] >= TEXT_MIN_CONF:
                model, conf, how = self.names[i], float(p[i]), "text"
            else:
                model, conf, how = "?", float(p[i]), ""
        return {"model": model, "model_conf": round(conf, 2), "how": how,
                "colour": self.colour(feats),
                "ref_scores": [(m, round(s, 3)) for s, m in ref_scores[:3]]}
