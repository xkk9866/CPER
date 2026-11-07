"""Pilot 5: random-effects inference with empirical-Bayes variance components.

Variance components (between-subject tau^2, residual sigma^2, per dim) are running moment estimates
computed from labelled training subjects inside each batch (like BatchNorm statistics) instead of
being learned through the classification loss. Classifier temperature is a separate parameter.
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import json, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pilot import probe, metrics
from pilot3 import make_ctx

FEAT, PROC = f"{_data_root}/features/clip-vit-base-patch16", f"{_data_root}/datasets/proc"
dev = "cuda"


class REInfer(nn.Module):
    def __init__(self, D, C, d=256, iters=3, estep=True, eb=True):
        super().__init__()
        self.W = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, d))
        self.f = nn.Parameter(torch.randn(C, d) * 0.1)
        self.mu0 = nn.Parameter(torch.zeros(d))
        self.log_t = nn.Parameter(torch.zeros(d))  # classifier precision (temperature), separate
        self.b = nn.Parameter(torch.zeros(C))
        self.register_buffer("tau2", torch.ones(d))
        self.register_buffer("sig2", torch.ones(d))
        self.iters, self.estep, self.eb = iters, estep, eb

    def logits(self, h, mu):
        r = h[:, :, None] - mu[:, None, None] - self.f[None, None]
        return -0.5 * (self.log_t.exp() * r ** 2).sum(-1) + self.b

    @torch.no_grad()
    def update_moments(self, h, y):
        """h (B,N,d) all from one subject per row, y (B,N) labels: method-of-moments variance components."""
        res = h - self.f[y]                        # remove expression offsets
        m = res.mean(1)                             # per-subject means
        sig2 = ((res - m[:, None]) ** 2).sum(1).sum(0) / (res.shape[0] * (res.shape[1] - 1))
        tau2 = ((m - self.mu0) ** 2).mean(0) - sig2 / res.shape[1]
        self.sig2.lerp_(sig2, 0.05); self.tau2.lerp_(tau2.clamp_min(1e-4), 0.05)

    def forward(self, q, c):
        h = self.W(torch.cat([q[:, None], c], 1))
        n = h.shape[1]
        lam0, lame = (1 / self.tau2, 1 / self.sig2) if self.eb else (torch.ones_like(self.tau2), torch.ones_like(self.sig2))
        e = torch.zeros_like(h)
        for _ in range(self.iters if self.estep else 1):
            mu = (lam0 * self.mu0 + lame * (h - e).sum(1)) / (lam0 + n * lame)
            ll = self.logits(h, mu)
            if self.estep:
                e = ll.softmax(-1) @ self.f
        return ll, h


def train(Z, y, meta, tr_rows, K, C, rng, epochs=40, **kw):
    m = REInfer(Z.shape[1], C, **kw).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.05)
    Zt, yt = torch.tensor(Z, device=dev), torch.tensor(y, device=dev)
    for ep in range(epochs):
        perm = rng.permutation(tr_rows)
        for i in range(0, len(perm), 128):
            r = perm[i:i + 128]
            k = rng.integers(0, K + 1)
            c = make_ctx(meta, r, max(k, 1), "skew" if rng.random() < 0.5 else "bal", rng)
            lab = torch.cat([yt[r][:, None], yt[c]], 1)
            if m.eb:  # moments from a full (query + K refs) labelled subject sample
                m.update_moments(m.W(torch.cat([Zt[r][:, None], Zt[c]], 1)).detach(), lab)
            ll, _ = m(Zt[r], Zt[c][:, :k])
            loss = F.cross_entropy(ll.flatten(0, 1), lab[:, :k + 1].flatten(), label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def run(name, K=6, folds=None):
    rng = np.random.default_rng(0)
    meta = pd.read_csv(f"{PROC}/{name}_meta.csv")
    Zf = np.load(f"{FEAT}/{name}.npy", mmap_mode="r")[:, :, 0].astype(np.float32).mean(1)
    y, C = meta.label.values, meta.label.nunique()
    res = {}
    variants = {"re_eb": dict(), "re_eb_noE": dict(estep=False), "re_learned": dict(eb=False)}
    for fold in (folds or sorted(meta.fold.unique())):
        tr, te = (meta.fold != fold).values, (meta.fold == fold).values
        Z = (Zf - Zf[tr].mean(0)) / (Zf[tr].std(0) + 1e-6)
        trr, ter = np.where(tr)[0], np.where(te)[0]
        zt = torch.tensor(Z, device=dev)
        ctx = {mode: make_ctx(meta, ter, K, mode, rng) for mode in ["bal", "skew"]}
        ctr = make_ctx(meta, trr, K, "bal", rng)
        for mode in ["bal", "skew"]:
            p = probe(Z[trr] - Z[ctr].mean(1), y[tr], Z[ter] - Z[ctx[mode]].mean(1), C)
            res.setdefault(f"ctxmean_{mode}", []).append(metrics(p, y[te], C))
        for vn, kw in variants.items():
            m = train(Z, y, meta, trr, K, C, rng, **kw)
            with torch.no_grad():
                for mode in ["bal", "skew"]:
                    p = m(zt[ter], zt[ctx[mode]])[0][:, 0].argmax(1).cpu().numpy()
                    res.setdefault(f"{vn}_{mode}", []).append(metrics(p, y[te], C))
                p = m(zt[ter], zt[ctx["bal"]][:, :0])[0][:, 0].argmax(1).cpu().numpy()
                res.setdefault(f"{vn}_K0", []).append(metrics(p, y[te], C))
            if vn == "re_eb":
                print("  data weight K=6:", float((6 / m.sig2 / (1 / m.tau2 + 7 / m.sig2)).median()))
        print(name, fold, {k: v[-1][0].round(1) for k, v in res.items()}, flush=True)
    return {k: tuple(np.round(np.mean(v, 0), 2)) for k, v in res.items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    for w in (sys.argv[1] if len(sys.argv) > 1 else "ravdess,cremad").split(","):
        print("==>", w, json.dumps(run(w)), flush=True)
