"""Pilot 2: identity-conditioned counterfactual neutral.

N(i): MLP mapping an expression-invariant identity code i (FaceNet) to the CLIP appearance of that
person's neutral face. Fit only on neutral-labelled faces (no pairs, no identity labels needed).
Expression evidence e = z - N(i).  Controls give the probe the identity code directly.
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import json, sys
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from pilot import probe, metrics

FEAT, IDF, PROC = f"{_data_root}/features/clip-vit-base-patch16", f"{_data_root}/features/facenet", f"{_data_root}/datasets/proc"
dev = "cuda"


def fit_neutral(I, Z, steps=4000, bs=512, wd=1e-3):
    net = nn.Sequential(nn.Linear(I.shape[1], 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, Z.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=wd)
    I, Z = torch.tensor(I, device=dev), torch.tensor(Z, device=dev)
    for it in range(steps):
        b = torch.randint(len(I), (bs,), device=dev)
        i = F.normalize(I[b] + 0.05 * torch.randn_like(I[b]), dim=-1)  # small jitter for smoothness
        loss = ((net(i) - Z[b]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()


@torch.no_grad()
def apply(net, I):
    return net(torch.tensor(I, device=dev)).cpu().numpy()


def video(name, neutral_label, extra_neutral=None):
    meta = pd.read_csv(f"{PROC}/{name}_meta.csv")
    Zf = np.load(f"{FEAT}/{name}.npy", mmap_mode="r")[:, :, 0].astype(np.float32)
    If = np.load(f"{IDF}/{name}.npy").astype(np.float32)
    y, C = meta.label.values, meta.label.nunique()
    keys = ["raw", "z+id(lin)", "z+id(mlp)", "e", "z+e", "oracle"]
    res, diag = {k: [] for k in keys}, []
    for fold in sorted(meta.fold.unique()):
        tr, te = (meta.fold != fold).values, (meta.fold == fold).values
        mu, sd = Zf[tr].reshape(-1, Zf.shape[-1]).mean(0), Zf[tr].reshape(-1, Zf.shape[-1]).std(0) + 1e-6
        Z = (Zf - mu) / sd
        Ic = F.normalize(torch.tensor(If.mean(1)), dim=-1).numpy()  # clip-level identity
        m = tr & (y == neutral_label)
        Itr, Ztr = np.repeat(Ic[m], 16, 0), Z[m].reshape(-1, Z.shape[-1])
        N = fit_neutral(Itr, Ztr)
        Nhat = apply(N, Ic)  # (clips, D)
        E = Z.mean(1) - Nhat
        Zo = Z.mean(1).copy()
        for a in meta.actor.unique():
            ia = (meta.actor == a).values
            Zo[ia] -= Z[ia & (y == neutral_label)].mean((0, 1))
        # diagnostic: distance of predicted neutral to the test actor's true neutral, vs population mean
        d_pred, d_pop = [], []
        for a in meta.actor[te].unique():
            ia = (meta.actor == a).values & te
            true = Z[ia & (y == neutral_label)].mean((0, 1))
            d_pred.append(np.linalg.norm(Nhat[ia].mean(0) - true)); d_pop.append(np.linalg.norm(Ztr.mean(0) - true))
        diag.append((np.mean(d_pred), np.mean(d_pop)))
        zc = Z.mean(1)
        feats = {"raw": zc, "z+id(lin)": np.concatenate([zc, Ic * 10], 1), "z+id(mlp)": np.concatenate([zc, Ic * 10], 1),
                 "e": E, "z+e": np.concatenate([zc, E], 1), "oracle": Zo}
        for k, X in feats.items():
            p = probe(X[tr], y[tr], X[te], C, hidden=1024 if "mlp" in k else 0)
            res[k].append(metrics(p, y[te], C))
        print(name, fold, {k: tuple(np.round(v[-1], 1)) for k, v in res.items()}, "dist pred/pop", np.round(diag[-1], 2), flush=True)
    return {k: tuple(np.round(np.mean(v, 0), 2)) for k, v in res.items()}, np.round(np.mean(diag, 0), 2).tolist()


def ferplus():
    meta = pd.read_csv(f"{PROC}/ferplus_meta.csv")
    X = np.load(f"{FEAT}/ferplus.npy")[:, 0].astype(np.float32)
    I = np.load(f"{IDF}/ferplus.npy").astype(np.float32)
    tr, te = (meta.split == "train").values, (meta.split == "test").values
    Z = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
    y = meta.label.values
    N = fit_neutral(I[tr & (y == 0)], Z[tr & (y == 0)])
    E = Z - apply(N, I)
    out = {}
    for k, F2 in {"raw": Z, "z+id(lin)": np.concatenate([Z, I * 10], 1), "z+id(mlp)": np.concatenate([Z, I * 10], 1), "e": E, "z+e": np.concatenate([Z, E], 1)}.items():
        p = probe(F2[tr], y[tr], F2[te], 8, hidden=1024 if "mlp" in k else 0, epochs=60)
        out[k] = tuple(np.round(metrics(p, y[te], 8), 2))
    return out


if __name__ == "__main__":
    torch.manual_seed(0); np.random.seed(0)
    which = sys.argv[1] if len(sys.argv) > 1 else "ravdess,cremad,ferplus"
    for w in which.split(","):
        r = ferplus() if w == "ferplus" else video(w, 0 if w == "ravdess" else 4)
        print("==>", w, json.dumps(r), flush=True)
