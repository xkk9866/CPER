"""Tables for the CAME study -> paper/came_tables.md
Accuracy: mean ± std over 5 training seeds; each seed averages all folds and 3 evaluation-context draws.
Paired tests use exactly the same quantity: per test subject, accuracy averaged over the 3 draws (no ensembling of
predictions) and over seeds; difference A - B with a 95% bootstrap CI over subjects and a two-sided sign-flip test.
Equivalence (prototype study): 90% bootstrap CI inside ±1 point (TOST at alpha = 0.05)."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import glob, json, numpy as np, pandas as pd

R, PROC = f"{_data_root}/runs/came", f"{_data_root}/datasets/proc"
PROT = ["K0", "rand", "xsent", "contam", "same", "other", "neu"]
LF = ["rand", "xsent", "contam"]
MIX = [f"mix{j}" for j in range(7)]
KC = ["K0", "K1", "K2", "K4", "K8", "K16"]
NAMES = {"lp": "Linear probe", "mlp": "MLP", "ctr": "Centering + linear", "ctr_mlp": "Centering + MLP",
         "sstd": "Subject standardisation", "cbc": "Class-balanced centering", "pl": "Pseudo-label correction",
         "fuse": "Prediction fusion", "cat": "Raw ⊕ centred + MLP", "settf": "Set transformer", "orc": "Label-oracle centering",
         "came": "CAME (ours)", "came_nodir": "CAME, no composition latent", "came_noea": "CAME, no explaining away",
         "came_nosubj": "CAME, no subject inference", "came_llm": "LLM-1.5B, free adapter", "came_geo": "LLM-1.5B, geometry-locked",
         "came_georand": "Random vectors, geometry-locked", "came_geoshuf": "Shuffled LLM, geometry-locked",
         "came_onehot": "Simplex (one-hot), geometry-locked", "came_llm05": "LLM-0.5B, free adapter"}
MAIN_M = ["lp", "mlp", "ctr", "ctr_mlp", "sstd", "cbc", "pl", "fuse", "cat", "settf", "came"]


def load(d, t):
    return [json.load(open(p)) for p in sorted(glob.glob(f"{R}/{d}_{t}_s[0-9].json"))]


def v(rs, k, met="acc"):
    return np.array([r["mean"][k][met] for r in rs])


def c(rs, k, met="acc"):
    x = v(rs, k, met); return f"{x.mean():.2f} ± {x.std():.2f}"


def subj(d, t, prot):
    """per-subject accuracy, mean over the protocol's draws and over seeds"""
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
    return dd.mean(), np.percentile(boot, [2.5, 97.5]), np.percentile(boot, [5, 95]), p


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def row_main(rs):
    lf = np.mean([v(rs, k).mean() for k in LF]); worst = min(v(rs, k).mean() for k in PROT)
    return " | ".join(c(rs, k) for k in PROT) + f" | {lf:.2f} | {worst:.2f} | {v(rs, 'rand', 'ece').mean():.2f} | {v(rs, 'rand', 'nll').mean():.3f}"


H = "| Method | " + " | ".join(PROT) + " | LF | worst | ECE | NLL |\n|" + "---|" * (len(PROT) + 5)
L = ["# CAME study\n"]
for d in ["ravdess", "cremad"]:
    L += [f"\n## {d}: main (training composition Dir(1) for every method that uses references)\n", H]
    for m in MAIN_M + ["orc"]:
        rs = load(d, f"{m}_dir")
        if rs:
            L.append(f"| {NAMES[m]} | " + row_main(rs) + " |")
    L += [f"\n## {d}: training composition rand vs Dir(1)\n", "| Method | train | rand | same | other | LF | worst |", "|---|---|---|---|---|---|---|"]
    for m in ["ctr", "ctr_mlp", "pl", "cbc", "settf", "cat", "came"]:
        for comp in ["rand", "dir"]:
            rs = load(d, f"{m}_{comp}")
            if rs:
                L.append(f"| {NAMES[m]} | {comp} | {c(rs,'rand')} | {c(rs,'same')} | {c(rs,'other')} | {np.mean([v(rs,k).mean() for k in LF]):.2f} | {min(v(rs,k).mean() for k in PROT):.2f} |")
    L += [f"\n## {d}: composition curve\n", "| Method | " + " | ".join(f"j={j}" for j in range(7)) + " |", "|" + "---|" * 8]
    for m in ["mlp", "ctr", "ctr_mlp", "cbc", "pl", "settf", "came", "orc"]:
        rs = load(d, f"{m}_dir")
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k).mean():.2f}" for k in MIX) + " |")
    L += [f"\n## {d}: number of references\n", "| Method | " + " | ".join(KC) + " |", "|" + "---|" * 7]
    for m in ["ctr", "ctr_mlp", "pl", "cbc", "came"]:
        rs = load(d, f"{m}_dir")
        if rs:
            L.append(f"| {NAMES[m]} | " + " | ".join(f"{v(rs, k).mean():.2f}" for k in KC) + " |")
    L += [f"\n## {d}: CAME ablations\n", H]
    for m in ["came", "came_nodir", "came_noea", "came_nosubj"]:
        rs = load(d, f"{m}_dir")
        if rs:
            L.append(f"| {NAMES[m]} | " + row_main(rs) + " |")
    rs = load(d, "came_dir")
    if rs:
        L += [f"\n## {d}: subject-effect recovery (relative squared error; both estimators use the same references)\n",
              "| context | CAME estimate | shrunk reference mean |", "|---|---|---|"]
        for k in ["rand", "same"]:
            L.append(f"| {k} | {v(rs, 'muerr_'+k, 'em').mean():.3f} | {v(rs, 'muerr_'+k, 'mean').mean():.3f} |")
        L.append(f"\nlearned alpha: {v(rs, 'alpha', 'value').mean():.3f} ± {v(rs, 'alpha', 'value').std():.3f}")
    L += [f"\n## {d}: supervision (query only vs all members)\n", "| Method | sup | K0 | rand | same | LF |", "|---|---|---|---|---|---|"]
    for m in ["ctr_mlp", "settf", "came"]:
        for suf, nm in [("_dir", "all members"), ("_dir_q", "query only")]:
            rs = load(d, m + suf)
            if rs:
                L.append(f"| {NAMES[m]} | {nm} | {c(rs,'K0')} | {c(rs,'rand')} | {c(rs,'same')} | {np.mean([v(rs,k).mean() for k in LF]):.2f} |")
    L += [f"\n## {d}: prototypes (full data and 10% of the training clips)\n", "| Prototypes | data | K0 | rand | same | LF | worst |", "|---|---|---|---|---|---|---|"]
    for m in ["came", "came_llm", "came_llm05", "came_geo", "came_georand", "came_geoshuf", "came_onehot"]:
        for suf, nm in [("_dir", "100%"), ("_dir_fr0.1", "10%")]:
            rs = load(d, m + suf)
            if rs:
                L.append(f"| {NAMES[m] if m != 'came' else 'Free prototypes'} | {nm} | {c(rs,'K0')} | {c(rs,'rand')} | {c(rs,'same')} | {np.mean([v(rs,k).mean() for k in LF]):.2f} | {min(v(rs,k).mean() for k in PROT):.2f} |")
    L += [f"\n## {d}: low-data baselines (10% of the training clips)\n", "| Method | K0 | rand | same | LF | worst |", "|---|---|---|---|---|---|"]
    for m in ["mlp", "ctr", "pl", "came"]:
        rs = load(d, f"{m}_dir_fr0.1")
        if rs:
            L.append(f"| {NAMES[m]} | {c(rs,'K0')} | {c(rs,'rand')} | {c(rs,'same')} | {np.mean([v(rs,k).mean() for k in LF]):.2f} | {min(v(rs,k).mean() for k in PROT):.2f} |")
    L += [f"\n## {d}: paired subject-level tests, CAME − baseline: mean [95% CI] (p)\n", "| Baseline | " + " | ".join(PROT) + " |", "|" + "---|" * (len(PROT) + 1)]
    for b in ["mlp", "ctr", "ctr_mlp", "pl", "cbc", "settf"]:
        cells = []
        for k in PROT:
            try:
                dm, ci, _, p = paired(d, "came_dir", f"{b}_dir", k); cells.append(f"{dm:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}] ({fp(p)})")
            except Exception:
                cells.append("n/a")
        L.append(f"| {NAMES[b]} | " + " | ".join(cells) + " |")
    L += [f"\n## {d}: prototype equivalence vs free prototypes on rand (90% CI; equivalent if inside ±1)\n", "| Prototypes | data | diff | 90% CI | equivalent |", "|---|---|---|---|---|"]
    for m in ["came_llm", "came_geo", "came_georand", "came_geoshuf", "came_onehot"]:
        for suf, nm in [("_dir", "100%"), ("_dir_fr0.1", "10%")]:
            try:
                dm, _, ci9, _ = paired(d, m + suf, "came" + suf, "rand")
                L.append(f"| {NAMES[m]} | {nm} | {dm:+.2f} | [{ci9[0]:+.2f}, {ci9[1]:+.2f}] | {'yes' if ci9[0] > -1 and ci9[1] < 1 else 'no'} |")
            except Exception:
                pass
open(f"{_data_root}/reports/came_tables.md", "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
