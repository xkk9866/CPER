"""Tables for the DEMIX study: Acc mean +- std over 5 seeds (each seed = mean over subject-independent folds),
plus a paired subject-level comparison DEMIX vs centering (accuracy per test subject, seed-averaged predictions,
two-sided sign-flip permutation test).  -> paper/demix_tables.md"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import glob, json, numpy as np, pandas as pd

R, PROC = f"{_data_root}/runs/demix", f"{_data_root}/datasets/proc"
PROT = ["K0", "rand", "xsent", "contam", "same", "other"]
KC = ["K0", "K1", "K2", "K4", "K8", "K16"]
NAMES = {"lp": "Linear probe", "mlp": "MLP", "ctr": "Centering + linear", "ctr_mlp": "Centering + MLP", "settf": "Set transformer (ICL)",
         "demix": "DEMIX (ours)", "demix_noem": "DEMIX, one M-step (no explaining-away)", "demix_lin": "DEMIX, linear readout", "demix_nosubj": "DEMIX, no subject inference"}


def load(d, f, m):
    rs = [json.load(open(p)) for p in sorted(glob.glob(f"{R}/{d}_{f}_{m}_s*.json"))]
    return rs


def cell(rs, k, met="acc"):
    v = [r["mean"][k][met] for r in rs]
    return f"{np.mean(v):.2f} ± {np.std(v):.2f}"


def subject_acc(d, f, m, prot_i):
    meta = pd.read_csv(f"{PROC}/{d}_meta.csv"); out = {}
    for fold in sorted(meta.fold.unique()):
        te = np.where(meta.fold == fold)[0]
        P = np.mean([np.load(p)[prot_i] for p in glob.glob(f"{R}/pred/{d}_{f}_{m}_s*_f{fold}.npy")], 0)
        ok = P.argmax(1) == meta.label.values[te]
        for s in np.unique(meta.actor.values[te]):
            out[s] = ok[meta.actor.values[te] == s].mean() * 100
    return out


def perm_test(a, b, n=20000, seed=0):
    d = np.array([a[s] - b[s] for s in a]); rng = np.random.default_rng(seed)
    null = np.abs((d[None] * rng.choice([-1, 1], (n, len(d)))).mean(1))
    return d.mean(), float((null >= abs(d.mean())).mean())


lines = ["# DEMIX study (Acc %, mean ± std over 5 seeds; every seed averages all subject-independent folds)\n"]
for d in ["ravdess", "cremad"]:
    lines += [f"\n## {d}: protocols (K = 6 context clips except K0)\n", "| features | method | " + " | ".join(PROT) + " | mean ctx |", "|" + "---|" * (len(PROT) + 3)]
    for f, ms in [("both", list(NAMES)), ("frozen", ["lp", "ctr", "demix"]), ("ft", ["lp", "ctr", "demix"])]:
        for m in ms:
            rs = load(d, f, m)
            if not rs:
                continue
            mc = np.mean([np.mean([r["mean"][k]["acc"] for k in PROT[1:]]) for r in rs])
            lines.append(f"| {f} | {NAMES[m]} ({len(rs)}) | " + " | ".join(cell(rs, k) for k in PROT) + f" | {mc:.2f} |")
    lines += [f"\n## {d}: number of random context clips (features both)\n", "| method | " + " | ".join(KC) + " |", "|" + "---|" * (len(KC) + 1)]
    for m in ["lp", "ctr", "ctr_mlp", "settf", "demix", "demix_noem"]:
        rs = load(d, "both", m)
        if rs:
            lines.append(f"| {NAMES[m]} | " + " | ".join(cell(rs, k) for k in KC) + " |")
    lines += [f"\n## {d}: paired subject-level test, DEMIX − Centering+MLP (features both)\n", "| protocol | mean Δ (pts) | p |", "|---|---|---|"]
    for i, k in enumerate(PROT):
        try:
            dm, p = perm_test(subject_acc(d, "both", "demix", i), subject_acc(d, "both", "ctr_mlp", i))
            lines.append(f"| {k} | {dm:+.2f} | {p:.4f} |")
        except Exception as e:
            lines.append(f"| {k} | n/a | {e} |")
open(f"{_data_root}/reports/demix_tables.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
