"""Synthetic person-adaptation data with KNOWN subject offsets (controlled test of the theory and of every estimator).

Model of clip i of subject s with expression y (D-dimensional feature):
    x = mu_s + lambda_s * F_y + G_{s,y} + eps,      sum_c F_c = 0,  sum_c G_{s,c} = 0
    mu_s = B_F a_s + B_R r_s + N(0, s_mu^2 I)   subject offset: a_s ~ N(0, s_e^2) along the C-1 expression
                           directions B_F (it cannot be projected out without losing expression information, as in
                           frozen CLIP features), r_s ~ N(0, s_r^2) along r random identity directions B_R
    lambda_s ~ N(1, lam_sd^2)  subject expressivity (random slope); --person_scale: the whole within-person repertoire
                           (displacement, interaction and clip noise) scales with lambda_s, as observed for real
                           low-expressivity test people
    G_{s,c} ~ N(0, s_G^2 I)  subject x expression interaction (non-additive part)
    eps ~ N(0, I)          clip noise
The balanced subject mean is exactly mu_s, which cmn.py uses as the ground truth of the offset diagnostic.
Writes  proc/{name}_meta.csv, proc/{name}_truth.npz, features/clip/{name}_raw.npy   (run cmn.py with --feat raw).

usage:  python synth_data.py --name synth_add                      (additive, moderate interaction)
        python synth_data.py --name synth_inter --s_G 0.25 --lam_sd 0.3   (strong interaction / expressivity)
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, os
import numpy as np, pandas as pd

E = _data_root
PROC, FEATD = f"{E}/datasets/proc", f"{E}/features/clip"

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="synth_add")
ap.add_argument("--D", type=int, default=128)
ap.add_argument("--C", type=int, default=8)
ap.add_argument("--subjects", type=int, default=48)
ap.add_argument("--folds", type=int, default=6)
ap.add_argument("--per_class", type=int, default=10)
ap.add_argument("--s_mu", type=float, default=0.10)
ap.add_argument("--s_e", type=float, default=1.20)
ap.add_argument("--s_r", type=float, default=1.50)
ap.add_argument("--r", type=int, default=8)
ap.add_argument("--s_F", type=float, default=0.25)
ap.add_argument("--s_G", type=float, default=0.06)
ap.add_argument("--lam_sd", type=float, default=0.15)
ap.add_argument("--person_scale", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
assert a.name.startswith("synth"), "synthetic dataset names must start with 'synth'"

rng = np.random.default_rng(a.seed)
Fm = rng.normal(0, a.s_F, (a.C, a.D)); Fm -= Fm.mean(0)
subs = np.arange(a.subjects)
BF = np.linalg.qr(Fm.T)[0][:, : a.C - 1]  # expression directions
BR = np.linalg.qr(rng.normal(size=(a.D, a.r)))[0]
mu = (rng.normal(0, a.s_e, (a.subjects, a.C - 1)) @ BF.T + rng.normal(0, a.s_r, (a.subjects, a.r)) @ BR.T
      + rng.normal(0, a.s_mu, (a.subjects, a.D)))
lam = rng.normal(1, a.lam_sd, a.subjects).clip(0.2, None)
X, rows = [], []
for s in subs:
    G = rng.normal(0, a.s_G, (a.C, a.D)); G -= G.mean(0)
    for c in range(a.C):
        for k in range(a.per_class):
            if a.person_scale:
                X.append(mu[s] + lam[s] * (Fm[c] + G[c] + rng.normal(0, 1, a.D)))
            else:
                X.append(mu[s] + lam[s] * Fm[c] + G[c] + rng.normal(0, 1, a.D))
            rows.append(dict(clip=f"s{s}_c{c}_{k}", label=c, actor=int(s), sentence=k % 2, intensity=0,
                             fold=int(s % a.folds)))
os.makedirs(PROC, exist_ok=True)
os.makedirs(FEATD, exist_ok=True)
np.save(f"{FEATD}/{a.name}_raw.npy", np.asarray(X, np.float32))
pd.DataFrame(rows).to_csv(f"{PROC}/{a.name}_meta.csv", index=False)
np.savez(f"{PROC}/{a.name}_truth.npz", subjects=subs, mu=mu, lam=lam, F=Fm)
sig = dict(subject=a.D * a.s_mu ** 2 + (a.C - 1) * a.s_e ** 2 + a.r * a.s_r ** 2, expression=a.D * a.s_F ** 2 * (a.C - 1) / a.C, interaction=a.D * a.s_G ** 2 * (a.C - 1) / a.C)
tot = sum(sig.values())
print(a.name, len(rows), "clips;", {k: round(v / tot, 3) for k, v in sig.items()}, "(population signal shares)")
