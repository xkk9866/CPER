"""Clip-level frozen features for the backbone study: temporal mean of the CLS token of frozen CLIP ViT-B/16 and
ViT-L/14.  Output: $EMO_ROOT/features/clip/{data}_{b16|l14}.npy (N, 768 | 1024) fp32."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import sys, numpy as np
for d in sys.argv[1].split(","):
    for tag, m in [("b16", "clip-vit-base-patch16"), ("l14", "clip-vit-large-patch14")]:
        a = np.load(f"{_data_root}/features/{m}/{d}.npy", mmap_mode="r")
        np.save(f"{_data_root}/features/clip/{d}_{tag}.npy", np.asarray(a[:, :, 0], np.float32).mean(1))
        print(d, tag, a.shape, flush=True)
