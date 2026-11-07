"""Pilot: is 'displacement from a counterfactual neutral' a better expression code than raw appearance?

Frozen CLIP CLS features. For each fold:
  raw      : linear probe on z
  mlp      : 2-layer MLP on z (capacity control)
  nsf      : linear probe on [z, e_s1..e_sK], e_s = z - D_neu(z, s)  (D_neu: denoiser trained on neutral faces only)
  allf     : same but denoiser trained on ALL faces (tests whether *neutral* anchoring matters)
  e_only   : linear probe on displacements only
  oracle   : z - mean(test actor's real neutral frames)   (upper bound, uses privileged reference)
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

FEAT = f"{_data_root}/features/clip-vit-base-patch16"
PROC = f"{_data_root}/datasets/proc"
dev = "cuda"


def metrics(pred, y, C):
    rec = [np.mean(pred[y == c] == c) for c in range(C) if np.any(y == c)]
    return 100 * np.mean(pred == y), 100 * np.mean(rec)


class Denoiser(nn.Module):
    """EDM-preconditioned MLP: D(x, s) = c_skip x + c_out F([c_in x, log s])."""
    def __init__(self, d, h=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 16, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))
        self.freq = nn.Parameter(torch.randn(8) * 2, requires_grad=False)

    def forward(self, x, s):
        s = s.view(-1, 1)
        c_skip, c_out, c_in = 1 / (s ** 2 + 1), s / (s ** 2 + 1).sqrt(), 1 / (s ** 2 + 1).sqrt()
        ls = torch.log(s) * self.freq
        emb = torch.cat([ls.sin(), ls.cos()], 1)
        return c_skip * x + c_out * self.net(torch.cat([c_in * x, emb], 1))


def train_denoiser(X, steps=3000, bs=512):
    D = Denoiser(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(D.parameters(), 1e-3, weight_decay=1e-4)
    X = torch.tensor(X, device=dev)
    for it in range(steps):
        x = X[torch.randint(len(X), (bs,), device=dev)]
        s = torch.exp(torch.randn(bs, device=dev) * 1.0 - 0.5)  # log-normal noise levels
        loss = (((D(x + s.view(-1, 1) * torch.randn_like(x), s) - x) ** 2).sum(1) * (s ** 2 + 1) / s ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return D.eval()


@torch.no_grad()
def displacements(D, X, sigmas):
    X = torch.tensor(X, device=dev)
    outs = [X - D(X, torch.full((len(X),), s, device=dev)) for s in sigmas]
    return torch.cat(outs, 1).cpu().numpy()


def probe(Xtr, ytr, Xte, C, hidden=0, epochs=300, wd=1e-3):
    Xtr, Xte = torch.tensor(Xtr, device=dev), torch.tensor(Xte, device=dev)
    ytr = torch.tensor(ytr, device=dev)
    layers = [nn.Linear(Xtr.shape[1], hidden), nn.GELU(), nn.Dropout(0.3), nn.Linear(hidden, C)] if hidden else [nn.Dropout(0.2), nn.Linear(Xtr.shape[1], C)]
    m = nn.Sequential(*layers).to(dev)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=wd)
    w = torch.bincount(ytr, minlength=C).float(); w = (w.sum() / (w + 1)) ; w = w / w.mean()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), 1024):
            b = perm[i:i + 1024]
            loss = F.cross_entropy(m(Xtr[b]), ytr[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m(Xte).argmax(1).cpu().numpy()


def run_video(name, sigmas, neutral_label):
    meta = pd.read_csv(f"{PROC}/{name}_meta.csv")
    F_ = np.load(f"{FEAT}/{name}.npy", mmap_mode="r")[:, :, 0].astype(np.float32)  # (N,16,D) CLS
    C = meta.label.nunique()
    res = {k: [] for k in ["raw", "mlp", "nsf", "allf", "e_only", "oracle"]}
    for fold in sorted(meta.fold.unique()):
        tr, te = (meta.fold != fold).values, (meta.fold == fold).values
        mu, sd = F_[tr].reshape(-1, F_.shape[-1]).mean(0), F_[tr].reshape(-1, F_.shape[-1]).std(0) + 1e-6
        Z = (F_ - mu) / sd
        y = meta.label.values
        clip = lambda A: A.mean(1)
        neu_tr = Z[tr & (y == neutral_label)].reshape(-1, Z.shape[-1])
        all_tr = Z[tr].reshape(-1, Z.shape[-1])
        Dn, Da = train_denoiser(neu_tr), train_denoiser(all_tr)
        flat = Z.reshape(-1, Z.shape[-1])
        En = displacements(Dn, flat, sigmas).reshape(Z.shape[0], Z.shape[1], -1)
        Ea = displacements(Da, flat, sigmas).reshape(Z.shape[0], Z.shape[1], -1)
        # oracle: subtract each actor's real neutral-clip mean (privileged)
        Zo = Z.copy()
        for a in meta.actor.unique():
            ia = (meta.actor == a).values
            Zo[ia] -= Z[ia & (y == neutral_label)].mean((0, 1))
        feats = {"raw": clip(Z), "mlp": clip(Z), "nsf": np.concatenate([clip(Z), clip(En)], 1),
                 "allf": np.concatenate([clip(Z), clip(Ea)], 1), "e_only": clip(En), "oracle": np.concatenate([clip(Z), clip(Zo)], 1)}
        for k, X in feats.items():
            p = probe(X[tr], y[tr], X[te], C, hidden=1024 if k == "mlp" else 0)
            res[k].append(metrics(p, y[te], C))
        print(name, "fold", fold, {k: tuple(np.round(v[-1], 2)) for k, v in res.items()}, flush=True)
    return {k: tuple(np.round(np.mean(v, 0), 2)) for k, v in res.items()}


def run_ferplus(sigmas):
    meta = pd.read_csv(f"{PROC}/ferplus_meta.csv")
    X = np.load(f"{FEAT}/ferplus.npy")[:, 0].astype(np.float32)
    tr, te = (meta.split == "train").values, (meta.split == "test").values
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Z = (X - mu) / sd
    y = meta.label.values
    Dn, Da = train_denoiser(Z[tr & (y == 0)]), train_denoiser(Z[tr])
    En, Ea = displacements(Dn, Z, sigmas), displacements(Da, Z, sigmas)
    feats = {"raw": Z, "mlp": Z, "nsf": np.concatenate([Z, En], 1), "allf": np.concatenate([Z, Ea], 1), "e_only": En}
    out = {}
    for k, F2 in feats.items():
        p = probe(F2[tr], y[tr], F2[te], 8, hidden=1024 if k == "mlp" else 0, epochs=60)
        out[k] = tuple(np.round(metrics(p, y[te], 8), 2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", default="0.5,1.0,2.0")
    ap.add_argument("--which", default="cremad,ravdess,ferplus")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    sig = [float(s) for s in a.sigmas.split(",")]
    allres = {}
    for w in a.which.split(","):
        if w == "ferplus":
            allres[w] = run_ferplus(sig)
        else:
            allres[w] = run_video(w, sig, neutral_label=4 if w == "cremad" else 0)
        print("==>", w, "(Acc/WAR, UAR)", allres[w], flush=True)
    print(json.dumps(allres))
