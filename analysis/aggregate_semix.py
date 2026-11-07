"""Tables for the SEMIX study -> paper/semix_tables.md
Acc: mean ± std over 5 training seeds; each seed averages all folds and 3 evaluation-context draws.
Paired test: per test subject, accuracy of each seed's own predictions (not an ensemble), averaged over seeds,
DEMIX/SEMIX minus baseline; two-sided sign-flip permutation test over subjects."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import glob, json, numpy as np, pandas as pd

R, PROC = f"{_data_root}/runs/semix", f"{_data_root}/datasets/proc"
PROT = ["K0", "rand", "xsent", "contam", "same", "other"]
LF = ["rand", "xsent", "contam"]
MIX = [f"mix{j}" for j in range(7)]
KC = ["K0", "K1", "K2", "K4", "K8", "K16"]
NAMES = {"lp": "Linear probe", "mlp": "MLP", "ctr": "Centering + linear", "ctr_mlp": "Centering + MLP", "pctr": "Partial centering + MLP",
         "cat": "Raw ⊕ centred + MLP", "fuse": "Prediction fusion", "pl": "Pseudo-label correction", "settf": "Set transformer",
         "demix": "DEMIX (free E-step)", "semix": "SEMIX (ours)", "semix_free": "SEMIX, free prototypes", "semix_unemb": "SEMIX, LLM output-layer rows",
         "semix_05b": "SEMIX, 0.5B LLM", "semix_noem": "SEMIX, one M-step", "semix_nosubj": "SEMIX, no subject inference"}
BASE = ["lp", "mlp", "ctr", "ctr_mlp", "pctr", "cat", "fuse", "pl", "settf", "demix", "semix"]
ABL = ["semix", "semix_free", "semix_unemb", "semix_05b", "semix_noem", "semix_nosubj"]


def load(d, m):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{R}/{d}_{m}_s*.json"))]


def v(rs, k, met="acc"):
    return np.array([r["mean"][k][met] for r in rs])


def c(rs, k, met="acc"):
    x = v(rs, k, met); return f"{x.mean():.2f} ± {x.std():.2f}"


def subj(d, m, pi):
    meta = pd.read_csv(f"{PROC}/{d}_meta.csv"); seeds = sorted({p.split("_s")[-1].split("_f")[0] for p in glob.glob(f"{R}/pred/{d}_{m}_s*_f0.npy")})
    acc = {}
    for s in seeds:
        for fold in sorted(meta.fold.unique()):
            te = np.where(meta.fold == fold)[0]
            ok = np.load(f"{R}/pred/{d}_{m}_s{s}_f{fold}.npy")[pi].astype(np.float32).argmax(1) == meta.label.values[te]
            for a in np.unique(meta.actor.values[te]):
                acc.setdefault(a, []).append(ok[meta.actor.values[te] == a].mean() * 100)
    return {a: np.mean(x) for a, x in acc.items()}


def perm(a, b, n=20000):
    dd = np.array([a[s] - b[s] for s in a]); rng = np.random.default_rng(0)
    null = np.abs((dd[None] * rng.choice([-1, 1], (n, len(dd)))).mean(1))
    return dd.mean(), (null >= abs(dd.mean())).mean()


L = ["# SEMIX study\n", "Acc %, mean ± std over 5 training seeds (each: all folds × 3 evaluation-context draws). LF = mean of the three label-free context protocols.\n"]
for d in ["ravdess", "cremad"]:
    L += [f"\n## {d}: methods (all supervise every episode member)\n", "| method | " + " | ".join(PROT) + " | LF mean | worst | ECE rand | NLL rand |", "|" + "---|" * (len(PROT) + 5)]
    for m in BASE + ABL[1:]:
        rs = load(d, m)
        if rs:
            lf = np.mean([v(rs, k).mean() for k in LF]); worst = min(v(rs, k).mean() for k in PROT)
            L.append(f"| {NAMES[m]} | " + " | ".join(c(rs, k) for k in PROT) + f" | {lf:.2f} | {worst:.2f} | {v(rs,'rand','ece').mean():.2f} | {v(rs,'rand','nll').mean():.3f} |")
    L += [f"\n## {d}: composition curve (j of 6 references share the query's expression)\n", "| method | " + " | ".join(f"j={j}" for j in range(7)) + " |", "|" + "---|" * 8]
    for m in ["mlp", "ctr_mlp", "cat", "fuse", "pl", "demix", "semix"]:
        rs = load(d, m)
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k).mean():.2f}" for k in MIX) + " |")
    L += [f"\n## {d}: number of random references\n", "| method | " + " | ".join(KC) + " |", "|" + "---|" * 7]
    for m in ["ctr_mlp", "cat", "fuse", "settf", "semix"]:
        rs = load(d, m)
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k).mean():.2f}" for k in KC) + " |")
    L += [f"\n## {d}: supervision matching (query-only vs all-member supervision)\n", "| method | sup | K0 | rand | same | LF mean |", "|---|---|---|---|---|---|"]
    for m in ["mlp", "ctr_mlp", "settf", "semix"]:
        for suf, nm in [("", "all members"), ("_q", "query only")]:
            rs = load(d, m + suf)
            if rs:
                L.append(f"| {NAMES[m]} | {nm} | {c(rs,'K0')} | {c(rs,'rand')} | {c(rs,'same')} | {np.mean([v(rs,k).mean() for k in LF]):.2f} |")
    rs = load(d, "semix")
    if rs and "muerr_rand" in rs[0]["mean"]:
        L += [f"\n## {d}: subject-effect recovery (relative squared error vs label oracle, SEMIX space)\n", "| context | EM estimate | shrunk mean |", "|---|---|---|"]
        for k in ["rand", "same"]:
            L.append(f"| {k} | {v(rs, 'muerr_'+k, 'em').mean():.3f} | {v(rs, 'muerr_'+k, 'mean').mean():.3f} |")
    L += [f"\n## {d}: paired subject-level tests (per-seed predictions)\n", "| comparison | " + " | ".join(PROT) + " |", "|" + "---|" * 7]
    for a_, b_ in [("semix", "ctr_mlp"), ("semix", "mlp"), ("semix", "cat"), ("semix", "demix")]:
        cells = []
        for pi, k in enumerate(PROT):
            try:
                dm, p = perm(subj(d, a_, pi), subj(d, b_, pi)); cells.append(f"{dm:+.2f} (p={p:.3f})")
            except Exception:
                cells.append("n/a")
        L.append(f"| {NAMES[a_]} − {NAMES[b_]} | " + " | ".join(cells) + " |")
open(f"{_data_root}/reports/semix_tables.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
