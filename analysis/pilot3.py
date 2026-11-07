"""Pilot 3: unlabeled same-person context as the source of the expression reference (frozen CLIP features).

For a query clip q of actor a, the context is K other clips of a (labels never used).
  raw        : probe on z_q
  ctxmean    : probe on z_q - mean(context)                (subject normalisation heuristic)
  icl        : transformer over [query; context] trained on episodes from training actors (learned relational inference)
Context compositions at test: 'bal' (random clips) and 'skew' (all context clips share one random emotion != query's).
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import json, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pilot import probe, metrics

FEAT, PROC = f"{_data_root}/features/clip-vit-base-patch16", f"{_data_root}/datasets/proc"
dev = "cuda"


def make_ctx(meta, rows, K, mode, rng):
    """For each query row, sample K context rows from the same actor (excluding itself)."""
    y, act = meta.label.values, meta.actor.values
    out = np.zeros((len(rows), K), int)
    for i, r in enumerate(rows):
        pool = np.where((act == act[r]) & (np.arange(len(meta)) != r))[0]
        if mode == "skew":
            emos = [e for e in np.unique(y[pool]) if e != y[r]]
            e = rng.choice(emos)
            pool = pool[y[pool] == e]
        out[i] = rng.choice(pool, K, replace=len(pool) < K)
    return out


class ICL(nn.Module):
    def __init__(self, d, C, h=256, rel=False, pure=False):
        super().__init__()
        self.rel, self.pure = rel, pure
        self.dinp = nn.Linear(2 * d, h)
        self.inp = nn.Linear(d, h)
        self.role = nn.Parameter(torch.randn(2, h) * 0.02)  # query / context role embeddings
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 4 * h, 0.1, batch_first=True, norm_first=True), 2)
        self.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, C))

    def forward(self, q, c):  # q (B,d)  c (B,K,d)
        if self.pure:  # only within-subject differences: query-context and the mean difference as a summary token
            dq = q[:, None] - c
            x = torch.cat([self.inp(dq.mean(1, keepdim=True)) + self.role[0], self.inp(dq) + self.role[1]], 1)
            return self.out(self.tf(x)[:, 0])
        ctx = self.dinp(torch.cat([q[:, None] - c, c], -1)) if self.rel else self.inp(c)
        x = torch.cat([self.inp(q)[:, None] + self.role[0], ctx + self.role[1]], 1)
        return self.out(self.tf(x)[:, 0])


def train_icl(Z, y, meta, tr_rows, K, C, rng, epochs=40, rel=False, pure=False):
    m = ICL(Z.shape[1], C, rel=rel, pure=pure).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 5e-4, weight_decay=0.05)
    Zt = torch.tensor(Z, device=dev); yt = torch.tensor(y, device=dev)
    self_pure = pure
    for ep in range(epochs):
        perm = rng.permutation(tr_rows)
        for i in range(0, len(perm), 128):
            r = perm[i:i + 128]
            k = rng.integers(1 if self_pure else 0, K + 1)  # variable context size
            mode = "skew" if rng.random() < 0.5 else "bal"
            c = make_ctx(meta, r, max(k, 1), mode, rng)
            cz = Zt[c] if k > 0 else Zt[c][:, :0]
            loss = F.cross_entropy(m(Zt[r], cz), yt[r], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def run(name, K=6):
    rng = np.random.default_rng(0)
    meta = pd.read_csv(f"{PROC}/{name}_meta.csv")
    Zf = np.load(f"{FEAT}/{name}.npy", mmap_mode="r")[:, :, 0].astype(np.float32).mean(1)
    y, C = meta.label.values, meta.label.nunique()
    res = {}
    for fold in sorted(meta.fold.unique()):
        tr, te = (meta.fold != fold).values, (meta.fold == fold).values
        Z = (Zf - Zf[tr].mean(0)) / (Zf[tr].std(0) + 1e-6)
        trr, ter = np.where(tr)[0], np.where(te)[0]
        p = probe(Z[tr], y[tr], Z[te], C); res.setdefault("raw", []).append(metrics(p, y[te], C))
        icl = train_icl(Z, y, meta, trr, K, C, rng)
        rsr = train_icl(Z, y, meta, trr, K, C, rng, rel=True)
        pure = train_icl(Z, y, meta, trr, K, C, rng, pure=True)
        for mode in ["bal", "skew"]:
            ctr, cte = make_ctx(meta, trr, K, "bal", rng), make_ctx(meta, ter, K, mode, rng)
            Xtr, Xte = Z[trr] - Z[ctr].mean(1), Z[ter] - Z[cte].mean(1)
            p = probe(Xtr, y[tr], Xte, C); res.setdefault(f"ctxmean_{mode}", []).append(metrics(p, y[te], C))
            with torch.no_grad():
                zt = torch.tensor(Z, device=dev)
                p = icl(zt[ter], zt[cte]).argmax(1).cpu().numpy()
            res.setdefault(f"icl_{mode}", []).append(metrics(p, y[te], C))
            with torch.no_grad():
                p = rsr(zt[ter], zt[cte]).argmax(1).cpu().numpy()
            res.setdefault(f"rsr_{mode}", []).append(metrics(p, y[te], C))
            with torch.no_grad():
                p = pure(zt[ter], zt[cte]).argmax(1).cpu().numpy()
            res.setdefault(f"pure_{mode}", []).append(metrics(p, y[te], C))
        with torch.no_grad():
            p = icl(zt[ter], zt[ter][:, None][:, :0]).argmax(1).cpu().numpy()
        res.setdefault("icl_K0", []).append(metrics(p, y[te], C))
        print(name, fold, {k: tuple(np.round(v[-1], 1)) for k, v in res.items()}, flush=True)
    return {k: tuple(np.round(np.mean(v, 0), 2)) for k, v in res.items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    for w in (sys.argv[1] if len(sys.argv) > 1 else "ravdess,cremad").split(","):
        print("==>", w, json.dumps(run(w)), flush=True)
