"""Pilot 4: neural random-effects inference (unrolled EM) over a subject reference set.

Generative model in a learned space h = W z:
    h = mu_s + f_y + eps,   mu_s ~ N(mu_0, 1/lam0),  eps ~ N(0, 1/lam_e)
Given the query and K unlabeled same-subject references, alternate
    E: r_c(k) = softmax_k( -lam_e/2 ||h_c - mu - f_k||^2 + b_k )
    M: mu = (lam0 mu_0 + lam_e sum_c (h_c - sum_k r_c(k) f_k)) / (lam0 + n lam_e)
All of (W, f, mu_0, lam0, lam_e, b) are learned end-to-end; K = 0 reduces to a prior-centred classifier.
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import json, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pilot import probe, metrics
from pilot3 import make_ctx

FEAT, PROC = f"{_data_root}/features/clip-vit-base-patch16", f"{_data_root}/datasets/proc"
dev = "cuda"


class RandomEffects(nn.Module):
    def __init__(self, D, C, d=256, iters=3):
        super().__init__()
        self.W = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, d))
        self.f = nn.Parameter(torch.randn(C, d) * 0.1)
        self.mu0 = nn.Parameter(torch.zeros(d))
        self.log_lam0 = nn.Parameter(torch.zeros(d))
        self.log_lame = nn.Parameter(torch.zeros(d))
        self.b = nn.Parameter(torch.zeros(C))
        self.iters = iters

    def loglik(self, h, mu):  # h (B,N,d), mu (B,d) -> (B,N,C)
        lam = self.log_lame.exp()
        r = h[:, :, None] - mu[:, None, None] - self.f[None, None]
        return -0.5 * (lam * r ** 2).sum(-1) + self.b

    def forward(self, q, c):
        h = self.W(torch.cat([q[:, None], c], 1))  # query is also an observation of the subject
        n = h.shape[1]
        lam0, lame = self.log_lam0.exp(), self.log_lame.exp()
        e = torch.zeros_like(h)
        for _ in range(self.iters):
            mu = (lam0 * self.mu0 + lame * (h - e).sum(1)) / (lam0 + n * lame)
            ll = self.loglik(h, mu)
            e = ll.softmax(-1) @ self.f
        return ll  # (B,N,C): logits for query (index 0) and every reference


def train_re(Z, y, meta, tr_rows, K, C, rng, epochs=40):
    m = RandomEffects(Z.shape[1], C).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.05)
    Zt, yt = torch.tensor(Z, device=dev), torch.tensor(y, device=dev)
    for ep in range(epochs):
        perm = rng.permutation(tr_rows)
        for i in range(0, len(perm), 128):
            r = perm[i:i + 128]
            k = rng.integers(0, K + 1)
            c = make_ctx(meta, r, max(k, 1), "skew" if rng.random() < 0.5 else "bal", rng)[:, :k]
            ll = m(Zt[r], Zt[c])
            lab = torch.cat([yt[r][:, None], yt[c]], 1)  # references are labelled at training time
            loss = F.cross_entropy(ll.flatten(0, 1), lab.flatten(), label_smoothing=0.1)
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
        m = train_re(Z, y, meta, trr, K, C, rng)
        zt = torch.tensor(Z, device=dev)
        for mode in ["bal", "skew"]:
            ctr, cte = make_ctx(meta, trr, K, "bal", rng), make_ctx(meta, ter, K, mode, rng)
            p = probe(Z[trr] - Z[ctr].mean(1), y[tr], Z[ter] - Z[cte].mean(1), C)
            res.setdefault(f"ctxmean_{mode}", []).append(metrics(p, y[te], C))
            with torch.no_grad():
                p = m(zt[ter], zt[cte])[:, 0].argmax(1).cpu().numpy()
            res.setdefault(f"re_{mode}", []).append(metrics(p, y[te], C))
        with torch.no_grad():
            for k in [0, 1, 3]:
                cte = make_ctx(meta, ter, max(k, 1), "bal", rng)[:, :k]
                p = m(zt[ter], zt[cte])[:, 0].argmax(1).cpu().numpy()
                res.setdefault(f"re_K{k}", []).append(metrics(p, y[te], C))
        print(name, fold, {k: tuple(np.round(v[-1], 1)) for k, v in res.items()}, flush=True)
    return {k: tuple(np.round(np.mean(v, 0), 2)) for k, v in res.items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    for w in (sys.argv[1] if len(sys.argv) > 1 else "ravdess,cremad").split(","):
        print("==>", w, json.dumps(run(w)), flush=True)
