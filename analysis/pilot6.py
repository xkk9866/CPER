"""Pilot 6: low-rank (PLDA-style) random-effects inference vs diagonal random effects (frozen CLIP features).

Model in a learned space h = W z:
    h = mu0 + U a + f_y + eps,   a ~ N(0, I_r),  eps ~ N(0, diag(1/lam))
Closed-form posterior of the subject factor a given N same-subject observations (query + K references):
    S = (I + N U^T L U)^-1,   a_hat = S U^T L sum_n (h_n - mu0 - e_n)
alternated with the E-step e_n = sum_k p(k | h_n - mu) f_k. A generative loss (group marginal
likelihood of same-subject residuals, PLDA) trains U to span shared within-subject variation.
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import json, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pilot import probe, metrics
from pilot3 import make_ctx

FEAT, PROC = f"{_data_root}/features/clip-vit-base-patch16", f"{_data_root}/datasets/proc"
dev = "cuda"


class LowRankRE(nn.Module):
    def __init__(self, D, C, d=256, rank=32, iters=3, gen=1.0, diag_only=False, pred=False):
        super().__init__()
        self.W = nn.Sequential(nn.Dropout(0.2), nn.Linear(D, d))
        self.f = nn.Parameter(torch.randn(C, d) * 0.1)
        self.mu0 = nn.Parameter(torch.zeros(d))
        self.U = nn.Parameter(torch.randn(d, rank) * 0.05)
        self.log_lam = nn.Parameter(torch.zeros(d))     # residual precision
        self.log_l0 = nn.Parameter(torch.zeros(d)); self.log_le = nn.Parameter(torch.zeros(d))  # diagonal RE
        self.log_t = nn.Parameter(torch.zeros(d)); self.b = nn.Parameter(torch.zeros(C))
        self.iters, self.gen, self.diag_only, self.rank, self.pred = iters, gen, diag_only, rank, pred

    def logits(self, h, mu):
        r = h[:, :, None] - mu[:, None, None] - self.f[None, None]
        return -0.5 * (self.log_t.exp() * r ** 2).sum(-1) + self.b

    def subject_effect(self, h, e):
        B, N, d = h.shape
        if self.diag_only:
            lam0, lame = self.log_l0.exp(), self.log_le.exp()
            return (lam0 * self.mu0 + lame * (h - e).sum(1)) / (lam0 + N * lame)
        lam = self.log_lam.exp()
        U = self.U
        S = torch.linalg.inv(torch.eye(self.rank, device=h.device) + N * (U.t() * lam) @ U)  # (r,r)
        a = ((h - e - self.mu0).sum(1) * lam) @ U @ S.t()                                     # (B,r)
        return self.mu0 + a @ U.t()

    def forward(self, q, c):
        if self.pred:
            return self.forward_pred(q, c)
        h = self.W(torch.cat([q[:, None], c], 1))
        e = torch.zeros_like(h)
        for _ in range(self.iters):
            mu = self.subject_effect(h, e)
            ll = self.logits(h, mu)
            e = ll.softmax(-1) @ self.f
        return ll, h

    def forward_pred(self, q, c):
        """Posterior predictive: subject effect inferred from references only (leave-one-out),
        its posterior variance widens the query's class likelihoods."""
        hq, hc = self.W(q), self.W(c)
        K = hc.shape[1]
        lam0, lame, lt = self.log_l0.exp(), self.log_le.exp(), self.log_t.exp()
        ec = torch.zeros_like(hc)
        mu = self.mu0.expand(len(hq), -1)
        llc = hc.new_zeros(len(hq), 0, self.f.shape[0])
        for _ in range(self.iters if K > 0 else 1):
            mu = (lam0 * self.mu0 + lame * (hc - ec).sum(1)) / (lam0 + K * lame)
            if K > 0:
                llc = self.logits(hc, mu)
                ec = llc.softmax(-1) @ self.f
        v = 1.0 / (lam0 + K * lame)                    # posterior variance of the subject effect
        prec = 1.0 / (1.0 / lt + v)                    # predictive precision
        r = hq[:, None] - mu[:, None] - self.f[None]
        llq = -0.5 * (prec * r ** 2).sum(-1) + self.b
        return torch.cat([llq[:, None], llc], 1), None

    def gen_loss(self, h, y):
        """PLDA group marginal NLL of residuals r_n = h_n - mu0 - f_{y_n} for one subject group (B groups of N)."""
        if self.diag_only:
            return h.new_zeros(())
        B, N, d = h.shape
        r = h - self.mu0 - self.f[y]
        lam = self.log_lam.exp(); U = self.U
        M = torch.eye(self.rank, device=h.device) + N * (U.t() * lam) @ U
        s = (r.sum(1) * lam) @ U                                 # (B,r)
        quad = (lam * r ** 2).sum((1, 2)) - (s @ torch.linalg.inv(M) * s).sum(1)
        logdet = torch.logdet(M) - N * torch.log(lam).sum()
        return 0.5 * (quad + logdet).mean() / (N * d)


def train(Z, y, meta, tr_rows, K, C, rng, epochs=40, **kw):
    m = LowRankRE(Z.shape[1], C, **kw).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.05)
    Zt, yt = torch.tensor(Z, device=dev), torch.tensor(y, device=dev)
    for ep in range(epochs):
        perm = rng.permutation(tr_rows)
        for i in range(0, len(perm), 128):
            r = perm[i:i + 128]
            k = rng.integers(0, K + 1)
            c = make_ctx(meta, r, max(k, 1), "skew" if rng.random() < 0.5 else "bal", rng)
            lab = torch.cat([yt[r][:, None], yt[c]], 1)
            ll, _ = m(Zt[r], Zt[c][:, :k])
            loss = F.cross_entropy(ll.flatten(0, 1), lab[:, :k + 1].flatten(), label_smoothing=0.1)
            if m.gen:
                hg = m.W(torch.cat([Zt[r][:, None], Zt[c]], 1))
                loss = loss + m.gen * m.gen_loss(hg, lab)
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def run(name, variants, K=6):
    rng = np.random.default_rng(0)
    meta = pd.read_csv(f"{PROC}/{name}_meta.csv")
    Zf = np.load(f"{FEAT}/{name}.npy", mmap_mode="r")[:, :, 0].astype(np.float32).mean(1)
    y, C = meta.label.values, meta.label.nunique()
    res = {}
    for fold in sorted(meta.fold.unique()):
        tr, te = (meta.fold != fold).values, (meta.fold == fold).values
        Z = (Zf - Zf[tr].mean(0)) / (Zf[tr].std(0) + 1e-6)
        trr, ter = np.where(tr)[0], np.where(te)[0]
        zt = torch.tensor(Z, device=dev)
        ctx = {m: make_ctx(meta, ter, K, m, np.random.default_rng(fold)) for m in ["bal", "skew"]}
        p = probe(Z[tr], y[tr], Z[te], C); res.setdefault("raw", []).append(metrics(p, y[te], C))
        for vn, kw in variants.items():
            m = train(Z, y, meta, trr, K, C, rng, **kw)
            with torch.no_grad():
                for k in [0, 1, 3]:
                    p = m(zt[ter], zt[ctx["bal"]][:, :k])[0][:, 0].argmax(1).cpu().numpy()
                    res.setdefault(f"{vn}_K{k}", []).append(metrics(p, y[te], C))
                for mode in ["bal", "skew"]:
                    p = m(zt[ter], zt[ctx[mode]])[0][:, 0].argmax(1).cpu().numpy()
                    res.setdefault(f"{vn}_{mode}", []).append(metrics(p, y[te], C))
        print(name, fold, {k: round(v[-1][0], 1) for k, v in res.items()}, flush=True)
    return {k: round(float(np.mean([x[0] for x in v])), 2) for k, v in res.items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    V = {"diag": dict(diag_only=True, gen=0.0), "pred": dict(diag_only=True, gen=0.0, pred=True)}
    for w in (sys.argv[1] if len(sys.argv) > 1 else "ravdess,cremad").split(","):
        print("==>", w, json.dumps(run(w, V)), flush=True)
