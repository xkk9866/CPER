"""Expression-invariant identity codes from a face-recognition network (FaceNet, VGGFace2).

Saves $EMO_ROOT/features/facenet/{dataset}.npy : ferplus (N,512), video (N,16,512), float16, L2-normalised.
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import os
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

PROC = f"{_data_root}/datasets/proc"
OUT = f"{_data_root}/features/facenet"


@torch.no_grad()
def embed(net, x):
    x = torch.from_numpy(np.ascontiguousarray(x)).cuda()
    if x.dim() == 3:
        x = x[..., None].expand(-1, -1, -1, 3)
    x = x.permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
    x = (x - 127.5) / 128.0
    return F.normalize(net(x), dim=-1).half().cpu().numpy()


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    net = InceptionResnetV1(pretrained="vggface2").cuda().eval()
    for name in ["ferplus", "ravdess", "cremad"]:
        src = np.load(f"{PROC}/ferplus_img48.npy" if name == "ferplus" else f"{PROC}/{name}_faces.npy", mmap_mode="r")
        flat = src if name == "ferplus" else src.reshape(-1, *src.shape[-3:])
        out = np.concatenate([embed(net, flat[i:i + 512]) for i in range(0, len(flat), 512)])
        if name != "ferplus":
            out = out.reshape(src.shape[0], src.shape[1], -1)
        np.save(f"{OUT}/{name}.npy", out)
        print(name, out.shape, flush=True)
