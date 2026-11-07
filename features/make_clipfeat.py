"""Clip-level features for the mixed-effects study: temporal mean of the CLS token (frozen CLIP and the
stage-1 fine-tuned CLIP of each fold). Output: $EMO_ROOT/features/clip/{data}_{frozen|ft_f*}.npy (N,768) fp32."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import numpy as np, pandas as pd
OUT = f"{_data_root}/features/clip"
for d, nf in [("ravdess", 6), ("cremad", 5)]:
    a = np.load(f"{_data_root}/features/clip-vit-base-patch16/{d}.npy", mmap_mode="r")
    np.save(f"{OUT}/{d}_frozen.npy", np.asarray(a[:, :, 0], np.float32).mean(1))
    for f in range(nf):
        b = np.load(f"{_data_root}/features/ft/{d}_f{f}.npy", mmap_mode="r")
        np.save(f"{OUT}/{d}_ft_f{f}.npy", np.asarray(b[:, :, 0], np.float32).mean(1))
    print(d, flush=True)
