"""Tables for the COMPASS study -> paper/compass_tables.md
Accuracy: mean ± std over 5 training seeds; each seed averages all folds and 3 evaluation-context draws.
Paired tests use the same quantity: per test subject, accuracy averaged over the 3 draws and over seeds; difference
COMPASS - baseline with a 95% bootstrap CI over subjects and a two-sided sign-flip permutation test."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import glob, json, numpy as np, pandas as pd

R, PROC = f"{_data_root}/runs/compass", f"{_data_root}/datasets/proc"
PROT = ["K0", "rand", "xsent", "contam", "same", "other", "neu"]
MIX = [f"mix{j}" for j in range(7)]
KC = ["K0", "K1", "K2", "K4", "K8", "K16"]
NAMES = {"source": "Source (no adaptation)", "psn_rand": "PSN (FG'15)", "bn": "BN-adapt (NeurIPS'20)", "tent": "TENT (ICLR'21)",
         "shot": "SHOT (ICML'20)", "t3a": "T3A (NeurIPS'21)", "lame": "LAME (CVPR'22)", "sar": "SAR (ICLR'23)",
         "compass_em": "COMPASS (ours)", "compass": "one inference step", "compass_nocomp": "no composition explaining-away",
         "compass_now": "no uncertainty weighting", "compass_a7": "Dirichlet composition prior, alpha = 7",
         "compass_oracle": "label-oracle composition"}
BASE = ["source", "psn_rand", "bn", "tent", "sar", "shot", "t3a", "lame"]
OURS = "compass_em"
ABL = ["compass_em", "compass", "compass_now", "compass_a7", "compass_nocomp"]


def load(d, t):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{R}/{d}_{t}_s[0-9].json"))]


def v(rs, k, met="acc"):
    return np.array([r["mean"][k][met] for r in rs])


def c(rs, k, met="acc"):
    x = v(rs, k, met); return f"{x.mean():.2f} ± {x.std():.2f}"


def mixavg(rs):
    x = np.mean([v(rs, k) for k in MIX], 0); return x


def subj(d, t, prot):
    meta = pd.read_csv(f"{PROC}/{d}_meta.csv")
    pi = [0] if prot == "K0" else [1 + 3 * PROT[1:].index(prot) + i for i in range(3)]
    acc = {}
    for p in sorted(glob.glob(f"{R}/pred/{d}_{t}_s[0-9]_f0.npy")):
        s = p.split("_s")[-1].split("_f")[0]
        for fold in sorted(meta.fold.unique()):
            te = np.where(meta.fold == fold)[0]; A = np.load(f"{R}/pred/{d}_{t}_s{s}_f{fold}.npy")
            ok = np.mean([A[i].astype(np.float32).argmax(1) == meta.label.values[te] for i in pi], 0)
            for a in np.unique(meta.actor.values[te]):
                acc.setdefault(a, []).append(ok[meta.actor.values[te] == a].mean() * 100)
    return {a: np.mean(x) for a, x in acc.items()}


def paired(d, ta, tb, prot, n=20000):
    a, b = subj(d, ta, prot), subj(d, tb, prot)
    dd = np.array([a[s] - b[s] for s in a]); rng = np.random.default_rng(0)
    p = (np.abs((dd[None] * rng.choice([-1, 1], (n, len(dd)))).mean(1)) >= abs(dd.mean())).mean()
    boot = dd[rng.integers(0, len(dd), (n, len(dd)))].mean(1)
    return dd.mean(), np.percentile(boot, [2.5, 97.5]), p


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main_row(rs):
    worst = min(v(rs, k).mean() for k in PROT)
    return " | ".join(c(rs, k) for k in PROT) + f" | {mixavg(rs).mean():.2f} ± {mixavg(rs).std():.2f} | {worst:.2f} | {v(rs, 'rand', 'ece').mean():.2f} | {v(rs, 'rand', 'nll').mean():.3f}"


H = "| Method | " + " | ".join(PROT) + " | Comp. avg | Worst | ECE | NLL |\n|" + "---|" * (len(PROT) + 5)
L = ["# COMPASS study\n"]
for d in ["ravdess", "cremad"]:
    L += [f"\n## {d}: main results\n", H]
    for m in BASE + [OURS]:
        rs = load(d, m)
        if rs:
            L.append(f"| {NAMES[m]} | " + main_row(rs) + " |")
    L += [f"\n## {d}: UAR\n", "| Method | " + " | ".join(PROT) + " |", "|" + "---|" * (len(PROT) + 1)]
    for m in BASE + [OURS]:
        rs = load(d, m)
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k, 'uar').mean():.2f}" for k in PROT) + " |")
    L += [f"\n## {d}: composition curve (j of 6 references share the query's expression)\n", "| Method | " + " | ".join(f"j={j}" for j in range(7)) + " |", "|" + "---|" * 8]
    for m in BASE + [OURS]:
        rs = load(d, m)
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k).mean():.2f}" for k in MIX) + " |")
    L += [f"\n## {d}: number of random references\n", "| Method | " + " | ".join(KC) + " |", "|" + "---|" * 7]
    for m in BASE + [OURS]:
        rs = load(d, m)
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k).mean():.2f}" for k in KC) + " |")
    L += [f"\n## {d}: ablations of COMPASS\n", H]
    for m in ABL:
        rs = load(d, m)
        if rs:
            L.append(f"| {NAMES[m]} | " + main_row(rs) + " |")
    L += [f"\n## {d}: composition exploitation on random references\n", "| Method | rand | j=0 | j=1 | rand − oracle |", "|---|---|---|---|---|"]
    orc = load(d, "compass_oracle")
    for m in ["bn", "tent", "shot", "compass_nocomp", "compass_oracle", OURS]:
        rs = load(d, m)
        if rs and orc:
            L.append(f"| {NAMES[m]} | {v(rs,'rand').mean():.2f} | {v(rs,'mix0').mean():.2f} | {v(rs,'mix1').mean():.2f} | {v(rs,'rand').mean() - v(orc,'rand').mean():+.2f} |")
    L += [f"\n## {d}: paired subject-level tests, COMPASS − baseline: mean [95% CI] (p)\n", "| Baseline | " + " | ".join(PROT[1:]) + " |", "|" + "---|" * len(PROT)]
    for b in BASE:
        cells = []
        for k in PROT[1:]:
            dm, ci, p = paired(d, OURS, b, k); cells.append(f"{dm:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] ({fp(p)})")
        L.append(f"| {NAMES[b]} | " + " | ".join(cells) + " |")
open(f"{_data_root}/reports/compass_tables.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
