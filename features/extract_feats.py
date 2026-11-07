"""Frozen CLIP region tokens (CLS + 4x4 pooled patch grid) for every image / video frame.

Saves $EMO_ROOT/features/{model_tag}/{dataset}.npy with shape
  ferplus: (N, 17, D)       video: (N, 16, 17, D)      dtype float16
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel

PROC = f"{_data_root}/datasets/proc"
FEAT = f"{_data_root}/features"
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def to_input(x, size):
    """uint8 (B,H,W) or (B,H,W,3) -> normalised float (B,3,size,size)."""
    x = torch.from_numpy(np.ascontiguousarray(x)).cuda()
    if x.dim() == 3:
        x = x[..., None].expand(-1, -1, -1, 3)
    x = x.permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - MEAN.cuda()) / STD.cuda()


@torch.no_grad()
def encode(model, x, size):
    out = model(pixel_values=to_input(x, size).half())
    h = model.vision_model.post_layernorm(out.last_hidden_state)
    cls, patches = h[:, :1], h[:, 1:]
    g = int(patches.shape[1] ** 0.5)
    grid = patches.transpose(1, 2).reshape(h.shape[0], -1, g, g)
    grid = F.adaptive_avg_pool2d(grid.float(), 4).flatten(2).transpose(1, 2).half()
    return torch.cat([cls, grid], 1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{_data_root}/models/clip-vit-base-patch16")
    ap.add_argument("--which", default="ferplus,ravdess,cremad")
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    tag = a.model.split("/")[-1]
    os.makedirs(f"{FEAT}/{tag}", exist_ok=True)
    model = CLIPVisionModel.from_pretrained(a.model, torch_dtype=torch.float16).cuda().eval()
    size = model.config.image_size
    for name in a.which.split(","):
        src = np.load(f"{PROC}/ferplus_img48.npy" if name == "ferplus" else f"{PROC}/{name}_faces.npy", mmap_mode="r")
        shp = src.shape
        flat = src.reshape(-1, *shp[-3:]) if name != "ferplus" else src
        out = []
        for i in range(0, len(flat), a.bs):
            out.append(encode(model, flat[i:i + a.bs], size))
        out = np.concatenate(out)
        if name != "ferplus":
            out = out.reshape(shp[0], shp[1], *out.shape[1:])
        np.save(f"{FEAT}/{tag}/{name}.npy", out)
        print(name, out.shape, flush=True)


if __name__ == "__main__":
    main()
