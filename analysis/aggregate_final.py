"""All tables of the final study -> paper/final_tables.md
Accuracy: mean ± std over training seeds; each seed averages all folds and 3 evaluation-context draws.
Paired tests: per test person, accuracy averaged over the 3 draws and over seeds (the tabulated quantity); difference
COMPASS - baseline with a 95% bootstrap CI over persons and a two-sided sign-flip permutation test.
Deployment analysis: if each of the 6 references independently shares the query's expression with probability p, the
number j of such references is Binomial(6, p) and the expected accuracy is sum_j Binom(j; 6, p) acc(mix j)."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import glob, json, math, numpy as np, pandas as pd

R, PROC = f"{_data_root}/runs/final", f"{_data_root}/datasets/proc"
MIX = [f"mix{j}" for j in range(7)]
KC = ["K0", "K1", "K2", "K4", "K8", "K16"]
OURS = "compass_bmabnc"
BASE = ["source", "psn_rand", "bn", "tent", "sar", "delta", "shot", "t3a", "lame"]
ABL = [OURS, "compass_bmabn", "compass_em", "compass", "compass_nocomp"]
NAMES = {"source": "Source (no adaptation)", "psn_rand": "PSN (FG'15)", "bn": "BN-adapt (NeurIPS'20)", "tent": "TENT (ICLR'21)",
         "sar": "SAR (ICLR'23)", "delta": "DELTA (ICLR'23)", "shot": "SHOT (ICML'20)", "t3a": "T3A (NeurIPS'21)",
         "lame": "LAME (CVPR'22)", OURS: "COMPASS (ours)", "compass_bmabn": "two regimes only (rho in {0, 1})",
         "compass_em": "no regime inference (explaining away only)", "compass": "no regime inference, one step",
         "compass_nocomp": "no regime inference, no explaining away", "compass_oracle": "label-oracle composition"}
DATA = [("ravdess", "dual", "RAVDESS"), ("cremad", "dual", "CREMA-D"), ("mead", "l14", "MEAD"), ("enterface", "l14", "eNTERFACE'05"), ("meld", "l14", "MELD")]
BACKBONE = [("ravdess", "b16"), ("ravdess", "l14"), ("cremad", "b16"), ("cremad", "l14"), ("mead", "b16"), ("enterface", "b16"), ("meld", "b16")]


def tag(m, feat):
    return m + ("" if feat == "dual" else f"_{feat}")


def load(d, t):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{R}/{d}_{t}_s[0-9].json"))]


def prot(d):
    return ["K0", "rand", "xsent", "contam", "same", "other", "skew"] + (["past"] if d == "meld" else [])


def v(rs, k, met="acc"):
    return np.array([r["mean"][k][met] for r in rs])


def c(rs, k, met="acc"):
    x = v(rs, k, met); return f"{x.mean():.2f} ± {x.std():.2f}"


def mixavg(rs):
    return np.mean([v(rs, k) for k in MIX], 0)


def expected(rs, p):
    return sum(math.comb(6, j) * p ** j * (1 - p) ** (6 - j) * v(rs, f"mix{j}").mean() for j in range(7))


def subj(d, t, pr, P):
    meta = pd.read_csv(f"{PROC}/{d}_meta.csv")
    pi = [0] if pr == "K0" else [1 + 3 * P[1:].index(pr) + i for i in range(3)]
    acc = {}
    for p in sorted(glob.glob(f"{R}/pred/{d}_{t}_s[0-9]_f0.npy")):
        s = p.split("_s")[-1].split("_f")[0]
        for fold in sorted(meta.fold.unique()):
            te = np.where(meta.fold == fold)[0]; A = np.load(f"{R}/pred/{d}_{t}_s{s}_f{fold}.npy")
            ok = np.mean([A[i].astype(np.float32).argmax(1) == meta.label.values[te] for i in pi], 0)
            for a in np.unique(meta.actor.values[te]):
                acc.setdefault(a, []).append(ok[meta.actor.values[te] == a].mean() * 100)
    return {a: np.mean(x) for a, x in acc.items()}


def paired(d, ta, tb, pr, P, n=20000):
    a, b = subj(d, ta, pr, P), subj(d, tb, pr, P)
    dd = np.array([a[s] - b[s] for s in a]); rng = np.random.default_rng(0)
    p = (np.abs((dd[None] * rng.choice([-1, 1], (n, len(dd)))).mean(1)) >= abs(dd.mean())).mean()
    boot = dd[rng.integers(0, len(dd), (n, len(dd)))].mean(1)
    return dd.mean(), np.percentile(boot, [2.5, 97.5]), p


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


L = ["# Final study\n"]
for d, feat, nm in DATA:
    P = prot(d)
    H = "| Method | " + " | ".join(P) + " | Comp. avg | Worst (with refs) | ECE | NLL |\n|" + "---|" * (len(P) + 5)
    L += [f"\n## {nm}: main results ({feat})\n", H]
    for m in BASE + [OURS]:
        rs = load(d, tag(m, feat))
        if rs:
            worst = min(v(rs, k).mean() for k in P[1:])
            L.append(f"| {NAMES[m]} | " + " | ".join(c(rs, k) for k in P) + f" | {mixavg(rs).mean():.2f} ± {mixavg(rs).std():.2f} | {worst:.2f} | {v(rs,'rand','ece').mean():.2f} | {v(rs,'rand','nll').mean():.3f} |")
    L += [f"\n## {nm}: UAR\n", "| Method | " + " | ".join(P) + " |", "|" + "---|" * (len(P) + 1)]
    for m in BASE + [OURS]:
        rs = load(d, tag(m, feat))
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k, 'uar').mean():.2f}" for k in P) + " |")
    L += [f"\n## {nm}: composition curve\n", "| Method | " + " | ".join(f"j={j}" for j in range(7)) + " |", "|" + "---|" * 8]
    for m in BASE + [OURS]:
        rs = load(d, tag(m, feat))
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k).mean():.2f}" for k in MIX) + " |")
    L += [f"\n## {nm}: number of random references\n", "| Method | " + " | ".join(KC) + " |", "|" + "---|" * 7]
    for m in BASE + [OURS]:
        rs = load(d, tag(m, feat))
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k).mean():.2f}" for k in KC) + " |")
    C = pd.read_csv(f"{PROC}/{d}_meta.csv").label.nunique()
    ps = [1 / C, 0.25, 0.5, 0.75, 1.0]
    L += [f"\n## {nm}: expected accuracy when each reference shares the query's expression with probability p\n",
          "| Method | " + " | ".join(f"p={p:.3f}" for p in ps) + " | break-even p vs COMPASS |", "|" + "---|" * (len(ps) + 2)]
    ro = load(d, tag(OURS, feat))
    for m in BASE + [OURS]:
        rs = load(d, tag(m, feat))
        if rs and ro:
            grid = np.linspace(0, 1, 1001)
            diff = np.array([expected(ro, p) - expected(rs, p) for p in grid])
            if m == OURS:
                be = "reference"
            elif (diff >= 0).all():
                be = "COMPASS higher for all p"
            elif (diff < 0).all():
                be = "baseline higher for all p"
            else:
                be = f"COMPASS higher for p ≥ {grid[np.argmax(diff >= 0)]:.3f}" if diff[-1] >= 0 else f"COMPASS higher for p ≤ {grid[np.where(diff >= 0)[0].max()]:.3f}"
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{expected(rs, p):.2f}" for p in ps) + f" | {be} |")
    L += [f"\n## {nm}: ablations of COMPASS\n", H]
    for m in ABL:
        rs = load(d, tag(m, feat))
        if rs:
            worst = min(v(rs, k).mean() for k in P[1:])
            L.append(f"| {NAMES[m]} | " + " | ".join(c(rs, k) for k in P) + f" | {mixavg(rs).mean():.2f} ± {mixavg(rs).std():.2f} | {worst:.2f} | {v(rs,'rand','ece').mean():.2f} | {v(rs,'rand','nll').mean():.3f} |")
    orc = load(d, tag("compass_oracle", feat))
    if orc:
        L += [f"\n## {nm}: composition exploitation on random references\n", "| Method | rand | j=0 | j=1 | rand − oracle |", "|---|---|---|---|---|"]
        for m in ["bn", "tent", "delta", "shot", "compass_em", OURS, "compass_oracle"]:
            rs = load(d, tag(m, feat))
            if rs:
                L.append(f"| {NAMES[m]} | {v(rs,'rand').mean():.2f} | {v(rs,'mix0').mean():.2f} | {v(rs,'mix1').mean():.2f} | {v(rs,'rand').mean() - v(orc,'rand').mean():+.2f} |")
    tp = [k for k in P if k in ("rand", "contam", "same", "skew", "past")]
    L += [f"\n## {nm}: paired person-level tests, COMPASS − baseline: mean [95% CI] (p)\n", "| Baseline | " + " | ".join(tp) + " |", "|" + "---|" * (len(tp) + 1)]
    for b in BASE:
        cells = []
        for k in tp:
            dm, ci, p = paired(d, tag(OURS, feat), tag(b, feat), k, P); cells.append(f"{dm:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] ({fp(p)})")
        L.append(f"| {NAMES[b]} | " + " | ".join(cells) + " |")

L += ["\n## Backbone study (3 seeds): rand / same / composition average\n",
      "| Method | " + " | ".join(f"{d} {f}" for d, f in BACKBONE) + " |", "|" + "---|" * (len(BACKBONE) + 1)]
for m in BASE + [OURS]:
    cells = []
    for d, f in BACKBONE:
        rs = load(d, tag(m, f))
        cells.append(f"{v(rs,'rand').mean():.2f} / {v(rs,'same').mean():.2f} / {mixavg(rs).mean():.2f}" if rs else "n/a")
    L.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
open(f"{_data_root}/reports/final_tables.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
