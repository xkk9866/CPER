"""Direct test of the additive subject x expression structure of a representation (test subjects of each fold).

Two-way decomposition of cell means  m_{s,c} = g + a_s + b_c + G_{s,c}  (G = subject x expression interaction).
Clips of every (subject, class) cell are split into halves A/B, and on standardised features (unit total variance):
  add_r2      held-out R^2 of the additive fit from A on the cell means of B
  inter_frac  ||G||^2 / ||M - g||^2 with G estimated as the cross-half product  <G^A, G^B>  (noise-unbiased)
  dir_cos     mean cosine between two subjects' expression displacements m_{s,c} - m_{s,.}  (consistency)
  subj_frac   between-subject share of variance ||a||^2 / ||M - g||^2 (noise-unbiased, cross-half)
  id_probe    subject identification, linear probe trained on one half of the sentences, tested on the others
usage: python additivity.py  -> paper/additivity.json   (spaces: frozen, ft, demix z if dumped)
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import json, os
import numpy as np, pandas as pd, torch, torch.nn.functional as F

PROC, FEAT, ZD = f"{_data_root}/datasets/proc", f"{_data_root}/features/clip", f"{_data_root}/runs/demix/z"
OUT = f"{_data_root}/reports/additivity.json"


def cellmeans(X, act, y, rng):
    S, Cs = np.unique(act), np.unique(y)
    A = np.zeros((len(S), len(Cs), X.shape[1])); B = np.zeros_like(A)
    for i, s in enumerate(S):
        for j, c in enumerate(Cs):
            idx = rng.permutation(np.where((act == s) & (y == c))[0])
            h = len(idx) // 2
            A[i, j], B[i, j] = X[idx[:h]].mean(0), X[idx[h:]].mean(0)
    return A, B


def decompose(M):
    g = M.mean((0, 1)); a = M.mean(1) - g; b = M.mean(0) - g
    return g, a, b, M - g - a[:, None] - b[None]


def id_probe(X, act, sen):
    S = np.unique(act); amap = {s: i for i, s in enumerate(S)}; ya = np.array([amap[s] for s in act], dtype=np.int64)
    us = np.unique(sen); trm = np.isin(sen, us[: max(1, len(us) // 2)])
    Xt, yt = torch.tensor(X[trm], dtype=torch.float32, device="cuda"), torch.tensor(ya[trm], device="cuda")
    W = torch.zeros(X.shape[1], len(S), device="cuda", requires_grad=True); bb = torch.zeros(len(S), device="cuda", requires_grad=True)
    opt = torch.optim.Adam([W, bb], 1e-2)
    for _ in range(300):
        loss = F.cross_entropy(Xt @ W + bb, yt) + 1e-3 * (W ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    p = (torch.tensor(X[~trm], dtype=torch.float32, device="cuda") @ W + bb).argmax(1).cpu().numpy()
    return float(np.mean(p == ya[~trm]) * 100)


def stats(X, act, y, sen, seed=0):
    X = (X - X.mean(0)) / np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean())  # unit total variance
    A, B = cellmeans(X, act, y, np.random.default_rng(seed))
    gA, aA, bA, GA = decompose(A); gB, aB, bB, GB = decompose(B)
    tot = ((A - gA) * (B - gB)).sum(-1).mean()  # cross-half: unbiased signal energies
    addA = gA + aA[:, None] + bA[None]
    add_r2 = 1 - ((B - addA) ** 2).sum(-1).mean() / ((B - B.mean((0, 1))) ** 2).sum(-1).mean()
    dA = A - A.mean(1, keepdims=True); dB = B - B.mean(1, keepdims=True)
    nA, nB = dA / np.linalg.norm(dA, axis=-1, keepdims=True), dB / np.linalg.norm(dB, axis=-1, keepdims=True)
    S = len(A)
    cos = np.mean([(nA[i] * nB[k]).sum(-1).mean() for i in range(S) for k in range(S) if i != k])
    return dict(add_r2=float(add_r2), inter_frac=float((GA * GB).sum(-1).mean() / tot),
                subj_frac=float((aA * aB).sum(-1).mean() / tot), expr_frac=float((bA * bB).sum(-1).mean() / tot),
                inter_over_task=float((GA * GB).sum(-1).mean() / ((GA * GB).sum(-1).mean() + (bA * bB).sum(-1).mean())),
                dir_cos=float(cos), id_probe=id_probe(X, act, sen))


if __name__ == "__main__":
    out = {}
    for d in ["ravdess", "cremad"]:
        meta = pd.read_csv(f"{PROC}/{d}_meta.csv")
        res = {}
        for fold in sorted(meta.fold.unique()):
            te = (meta.fold == fold).values
            spaces = {"frozen": np.load(f"{FEAT}/{d}_frozen.npy"), "finetuned": np.load(f"{FEAT}/{d}_ft_f{fold}.npy")}
            if os.path.exists(f"{ZD}/{d}_both_f{fold}.npy"):
                spaces["demix_z"] = np.load(f"{ZD}/{d}_both_f{fold}.npy")
            for k, X in spaces.items():
                res.setdefault(k, []).append(stats(X[te].astype(np.float64), meta.actor.values[te], meta.label.values[te], meta.sentence.values[te]))
        out[d] = {k: {**{m: float(np.mean([r[m] for r in v])) for m in v[0]}, **{m + "_sd": float(np.std([r[m] for r in v])) for m in v[0]}} for k, v in res.items()}
        print(d, json.dumps(out[d], indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
