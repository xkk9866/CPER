"""In-memory datasets with GPU-side augmentation for FERPlus (images) and RAVDESS / CREMA-D (videos)."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROC = f"{_data_root}/datasets/proc"

# label vocabularies (words are what the language reasoner sees and scores)
LABELS = {
    "ferplus": ["neutral", "happy", "surprise", "sad", "angry", "disgust", "fear", "contempt"],
    "ravdess": ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"],
    "cremad": ["angry", "disgust", "fear", "happy", "neutral", "sad"],
}
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


class FaceSet:
    """Holds uint8 faces in RAM. Images are treated as 1-frame clips so every sample is (T,H,W,3)."""

    def __init__(self, name, idx=None, meta=None):
        self.name = name
        self.labels = LABELS[name]
        if name == "ferplus":
            self.meta = pd.read_csv(f"{PROC}/ferplus_meta.csv")
            self.x = np.load(f"{PROC}/ferplus_img48.npy")
        else:
            self.meta = pd.read_csv(f"{PROC}/{name}_meta.csv")
            self.x = _load_cached(f"{PROC}/{name}_faces.npy")
        if meta is not None:
            self.meta = meta
        if idx is not None:
            self.meta = self.meta.iloc[idx].reset_index(drop=True)
        self.y = self.meta.label.values.astype(np.int64)
        self.video = name != "ferplus"
        self.neutral = self.labels.index("neutral")

    def __len__(self):
        return len(self.meta)

    def subset(self, mask):
        return _subset(self, mask)

    def use_features(self, feats):
        """Switch to cached encoder tokens: feats is (N,R,D) for images or (N,16,R,D) for videos."""
        self.x, self.cached = feats, True
        return self

    def fetch(self, rows, frames):
        """rows: array of positions in self.meta; frames: (B,T) frame ids -> uint8 tensor (B,T,H,W,3)."""
        src = self.meta.idx.values[rows]
        if getattr(self, "cached", False):
            if not self.video:
                return torch.from_numpy(np.asarray(self.x[src])[:, None])
            return torch.from_numpy(np.stack([self.x[s][f] for s, f in zip(src, frames)]))
        if not self.video:
            im = self.x[src]
            return torch.from_numpy(np.repeat(im[:, None, :, :, None], 3, -1))
        out = np.stack([self.x[s][f] for s, f in zip(src, frames)])
        return torch.from_numpy(out)


def _subset(ds, mask):
    new = object.__new__(FaceSet)
    new.__dict__.update(ds.__dict__)
    new.meta = ds.meta[mask].reset_index(drop=True)
    new.y = new.meta.label.values.astype(np.int64)
    return new


def sample_frames(B, T, n=16, train=True, offset=0):
    """Segment-based sampling: one frame per segment (random in train, fixed in test)."""
    seg = n // T
    base = torch.arange(T) * seg
    if train:
        return (base[None] + torch.randint(0, seg, (B, T))).numpy()
    return (base[None] + offset % seg).repeat(B, 1).numpy()


def gpu_augment(x, size, train):
    """x: uint8 (B,T,H,W,3) on GPU -> normalized float (B*T,3,size,size); same random transform per clip."""
    B, T = x.shape[:2]
    x = x.flatten(0, 1).permute(0, 3, 1, 2).float() / 255.0
    if train:
        ang = (torch.rand(B, device=x.device) - 0.5) * 2 * math.radians(10)
        sc = 1.0 / (0.85 + 0.25 * torch.rand(B, device=x.device))
        tx, ty = [(torch.rand(B, device=x.device) - 0.5) * 0.12 for _ in range(2)]
        flip = torch.where(torch.rand(B, device=x.device) < 0.5, -1.0, 1.0)
        cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
        theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
        theta = theta.repeat_interleave(T, 0)
        grid = F.affine_grid(theta, (B * T, 3, size, size), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
        # photometric jitter (per clip)
        br = (1 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.4).repeat_interleave(T, 0)
        ct = (1 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.4).repeat_interleave(T, 0)
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * ct + m) * br
        gray = torch.rand(B, device=x.device).repeat_interleave(T) < 0.2
        x[gray] = x[gray].mean(1, keepdim=True).expand(-1, 3, -1, -1)
        x = x.clamp(0, 1)
        # random erasing (per clip, same box over time)
        er = torch.rand(B, device=x.device) < 0.25
        if er.any():
            for b in torch.nonzero(er).flatten().tolist():
                h, w = int(size * (0.15 + 0.25 * np.random.rand())), int(size * (0.15 + 0.25 * np.random.rand()))
                i, j = np.random.randint(0, size - h), np.random.randint(0, size - w)
                x[b * T:(b + 1) * T, :, i:i + h, j:j + w] = torch.rand(3, 1, 1, device=x.device)
    else:
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


class Loader:
    """Class-balanced (optional) batch iterator over a FaceSet."""

    def __init__(self, ds, bs, T, train, balanced=False, offset=0):
        self.ds, self.bs, self.T, self.train, self.balanced, self.offset = ds, bs, (T if ds.video else 1), train, balanced, offset

    def epoch(self, n_batches=None):
        N = len(self.ds)
        if self.train and self.balanced:
            y = self.ds.y
            w = 1.0 / np.bincount(y)[y]
            order = np.random.choice(N, size=N, replace=True, p=w / w.sum())
        elif self.train:
            order = np.random.permutation(N)
        else:
            order = np.arange(N)
        nb = int(math.ceil(N / self.bs)) if not self.train else N // self.bs
        if n_batches:
            nb = min(nb, n_batches)
        for b in range(nb):
            rows = order[b * self.bs:(b + 1) * self.bs]
            fr = sample_frames(len(rows), self.T, train=self.train, offset=self.offset) if self.ds.video else np.zeros((len(rows), 1), int)
            yield self.ds.fetch(rows, fr), torch.from_numpy(self.ds.y[rows]), rows


_CACHE = {}


def _load_cached(path):
    """Load each face array once per process (fully into RAM)."""
    if path not in _CACHE:
        _CACHE[path] = np.load(path)
    return _CACHE[path]
