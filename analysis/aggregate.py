"""Collect run JSONs into fold-averaged tables (markdown) -> paper/results_tables.md."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import glob, json, os
from collections import defaultdict
import numpy as np

RUNS = f"{_data_root}/runs"
OUT = f"{_data_root}/reports/results_tables.md"
NFOLD = {"ravdess": 6, "cremad": 5, "ferplus": 1}


def load():
    R = defaultdict(lambda: defaultdict(list))
    for f in glob.glob(f"{RUNS}/*/*.json"):
        tag = f.replace("\\", "/").split("/")[-2]
        if tag in ("smoke", "dbg", "zeroshot", "ft_base_old", "ft_base", "clip_ft") or tag.startswith("dev_"):
            continue
        r = json.load(open(f))
        if "std" not in r:
            continue
        R[tag][r["args"]["data"]].append(r)
    return R


def cell(runs, key, data):
    v = [r[key] for r in runs if key in r]
    if not v:
        return "–", None
    a, u = [x["acc"] for x in v], [x["uar"] for x in v]
    acc, uar = np.mean(a), np.mean(u)
    note = "" if len(v) == NFOLD[data] else f" ({len(v)}f)"
    sd = f" ±{np.std(a):.1f}" if len(v) > 1 else ""
    return f"{acc:.2f}{sd} / {uar:.2f}{note}", (acc, uar)


def main():
    R = load()
    lines = ["# Aggregated results (Acc/WAR / UAR, mean over folds)\n"]
    lines += ["\n## ferplus (static images, no subject IDs: standard protocol only)\n","| method | Acc | UAR |", "|---|---|---|"]
    for tag in sorted(R):
        if R[tag].get("ferplus"):
            lines.append(f"| {tag} | " + " | ".join(f"{cell(R[tag]['ferplus'],'std','ferplus')[1][i]:.2f}" for i in (0, 1)) + " |")
    for data in ["ravdess", "cremad"]:
        lines += [f"\n## {data}\n", "| method | std (K=0) | bal (K=6) | skew (K=6) | K=1 UAR | K=3 UAR |", "|---|---|---|---|---|---|"]
        for tag in sorted(R):
            runs = R[tag].get(data, [])
            if not runs:
                continue
            ks = [np.mean([r["Kcurve"][str(k)] if str(k) in r.get("Kcurve", {}) else r.get("Kcurve", {}).get(k, np.nan) for r in runs]) for k in [1, 3]] if "Kcurve" in runs[0] else [np.nan, np.nan]
            lines.append(f"| {tag} | {cell(runs,'std',data)[0]} | {cell(runs,'bal',data)[0]} | {cell(runs,'skew',data)[0]} | {ks[0]:.2f} | {ks[1]:.2f} |")
    for f in sorted(glob.glob(f"{RUNS}/zeroshot/*.json")):
        z = json.load(open(f))
        lines.append(f"\nzero-shot {os.path.basename(f)[:-5]}: {z['acc']:.2f} / {z['uar']:.2f}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
