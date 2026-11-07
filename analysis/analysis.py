"""Mechanism analysis: how fine-tuning on few subjects changes subject structure of the token space.

For frozen CLIP tokens vs stage-1 fine-tuned tokens (fold 0, clip-mean CLS token, *test* subjects only):
  id_probe   : subject-ID linear probe accuracy (half of each test subject's clips -> other half)
  between/within : ratio of between-subject variance of subject means to within-subject residual variance
  expr_gain  : expression probe accuracy gain from within-subject centring (oracle-free, all clips of subject)
Writes paper/analysis.json
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import json, os
import numpy as np, pandas as pd, torch
from pilot import probe, metrics

PROC = f"{_data_root}/datasets/proc"
OUT = f"{_data_root}/reports/analysis.json"


def stats(X, meta, y, tr, te, C):
    rng = np.random.default_rng(0)
    Xt, at = X[te], meta.actor.values[te]
    half = np.zeros(len(Xt), bool)
    for a in np.unique(at):
        idx = np.where(at == a)[0]
        half[rng.choice(idx, len(idx) // 2, replace=False)] = True
    amap = {a: i for i, a in enumerate(np.unique(at))}
    ya = np.array([amap[a] for a in at], dtype=np.int64)
    p = probe(Xt[half], ya[half], Xt[~half], len(amap))
    id_acc = float(np.mean(p == ya[~half]) * 100)
    mus = np.stack([Xt[at == a].mean(0) for a in amap])
    within = np.mean([((Xt[at == a] - Xt[at == a].mean(0)) ** 2).sum(1).mean() for a in amap])
    between = ((mus - mus.mean(0)) ** 2).sum(1).mean()
    # expression probe: raw vs within-subject centred (centring uses each subject's own clips, no labels)
    Xc = X.copy()
    for a in np.unique(meta.actor.values):
        m = meta.actor.values == a
        Xc[m] -= X[m].mean(0)
    raw = metrics(probe(X[tr], y[tr], X[te], C), y[te], C)[0]
    cen = metrics(probe(Xc[tr], y[tr], Xc[te], C), y[te], C)[0]
    return dict(id_probe=id_acc, between_over_within=float(between / within), expr_raw=raw, expr_centred=cen, centring_gain=cen - raw)


if __name__ == "__main__":
    out = {}
    for d in ["ravdess", "cremad"]:
        meta = pd.read_csv(f"{PROC}/{d}_meta.csv"); y = meta.label.values; C = meta.label.nunique()
        res = {}
        for fold in sorted(meta.fold.unique()):
            tr, te = (meta.fold != fold).values, (meta.fold == fold).values
            for name, path in [("frozen", f"{_data_root}/features/clip-vit-base-patch16/{d}.npy"), ("finetuned", f"{_data_root}/features/ft/{d}_f{fold}.npy")]:
                X = np.load(path, mmap_mode="r")[:, :, 0].astype(np.float32).mean(1)
                X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
                res.setdefault(name, []).append(stats(X, meta, y, tr, te, C))
            print(d, fold, {k: {kk: round(vv, 2) for kk, vv in v[-1].items()} for k, v in res.items()}, flush=True)
        out[d] = {k: {kk: float(np.mean([r[kk] for r in v])) for kk in v[0]} for k, v in res.items()}
        print("==>", d, json.dumps(out[d]), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
