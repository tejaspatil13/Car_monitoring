"""Vehicle re-identification (appearance fingerprint) with FastReID weights.

Model: FastReID "bagtricks" ResNet50-IBN-a trained on VERI-Wild
       (416k images of 40k vehicles from 174 surveillance cameras).
Output: a 2048-d L2-normalised vector per car crop. Same car on two cameras
        -> high cosine similarity (dot product), different cars -> low.

    from reid import ReID
    reid = ReID()
    vecs = reid.embed([crop1, crop2])      # BGR crops -> (N, 2048)
    sim = vecs @ vecs.T
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent
WEIGHTS = BASE / "models" / "reid" / "veriwild_r50ibn.pt"
SOURCE = BASE / "models" / "reid" / "veriwild_bot_R50-ibn.pth"
INPUT_SIZE = (256, 256)          # (h, w) used by FastReID for VERI-Wild


# ---------------- ResNet50-IBN-a (FastReID layout) ----------------

class IBN(nn.Module):
    """Half InstanceNorm + half BatchNorm (helps across cameras / lighting)."""

    def __init__(self, planes):
        super().__init__()
        self.half = planes // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        a, b = torch.split(x, [self.half, x.shape[1] - self.half], 1)
        return torch.cat([self.IN(a.contiguous()), self.BN(b.contiguous())], 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, ibn=False, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + idt)


class ResNet50IBN(nn.Module):
    def __init__(self, last_stride=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make(64, 3, 1, ibn=True)
        self.layer2 = self._make(128, 4, 2, ibn=True)
        self.layer3 = self._make(256, 6, 2, ibn=True)
        self.layer4 = self._make(512, 3, last_stride, ibn=False)

    def _make(self, planes, blocks, stride, ibn):
        down = None
        if stride != 1 or self.inplanes != planes * 4:
            down = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, 1, stride, bias=False),
                                 nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, ibn, down)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes, 1, ibn) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


class ReIDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet50IBN()
        self.bottleneck = nn.BatchNorm1d(2048)
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))

    feat = "bn"                               # "bn" (after BN neck) or "pool"

    def forward(self, x):                     # x: RGB float 0..255, (N,3,H,W)
        x = (x - self.pixel_mean) / self.pixel_std
        f = self.backbone(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1)
        if self.feat == "bn":
            f = self.bottleneck(f)
        return F.normalize(f, dim=1)


def _slim_weights():
    """Convert the 1 GB training checkpoint into a ~100 MB inference file."""
    ck = torch.load(SOURCE, map_location="cpu", weights_only=True)["model"]
    sd = {}
    for k, v in ck.items():
        if k.startswith("backbone.") or k in ("pixel_mean", "pixel_std"):
            sd[k] = v
        elif k.startswith("heads.bottleneck.0."):
            sd["bottleneck." + k.split(".", 3)[3]] = v
    torch.save(sd, WEIGHTS)
    return sd


class ReID:
    def __init__(self, device=None):
        if device is None:
            from device import best_device
            device = best_device()
        self.device = device
        sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True) \
            if WEIGHTS.exists() else _slim_weights()
        self.net = ReIDNet()
        missing, unexpected = self.net.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.endswith("num_batches_tracked")]
        if missing or unexpected:
            raise RuntimeError(f"ReID weights mismatch: missing={missing[:5]} "
                               f"unexpected={unexpected[:5]}")
        self.net.eval().to(device)

    @torch.no_grad()
    def embed(self, crops, batch=16):
        """BGR crops (any size) -> (N, 2048) float32 numpy, L2-normalised."""
        out = []
        h, w = INPUT_SIZE
        for i in range(0, len(crops), batch):
            imgs = [cv2.cvtColor(cv2.resize(c, (w, h), interpolation=cv2.INTER_CUBIC),
                                 cv2.COLOR_BGR2RGB) for c in crops[i:i + batch]]
            x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().to(self.device)
            out.append(self.net(x).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)
