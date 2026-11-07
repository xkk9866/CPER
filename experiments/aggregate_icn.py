"""Tables of the ICN study from the manifest of run_icn.py and the saved predictions
(-> paper/icn_tables.md, paper/icn_results.json).  Every number of the manuscript comes from these files.

Accuracy / UAR / macro-F1 are pooled over all held-out clips (every clip is held out exactly once), averaged over the
reference draws, then over training seeds (mean +- sample SD across seeds).  Identity-cluster bootstrap intervals and
sign-flip tests use the per-person correct counts of the seed-0 reference evaluation (paired draws).  Tuned baselines
choose their operating point per fold on held-out TRAINING people (cross-fit split, composition-average objective);
never on test data."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, json, pathlib, sys
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_icn as R  # noqa: E402
import icn as I  # noqa: E402  (sets the data paths of cmn)
import cmn as CM  # noqa: E402

PAPER = pathlib.Path(_data_root) / "reports"
DN = dict(ravdess="RAVDESS", cremad="CREMA-D", mead="MEAD", enterface="eNTERFACE'05", meld="MELD-AVT", shemo="ShEMO",
          emoz="Emozionalmente", thai="THAI-SER Zoom", thaix="THAI-SER Studio (frozen Zoom-trained model)", **{k: k for k in R.SYN})
MIX = [f"mix{j}" for j in range(7)]
FREE = ["rand", "skew", "dir02", "past"]
LAB = dict(source="Source", bn="BN-adapt (NeurIPS'20)", tent_grid="TENT (ICLR'21)", sar_grid="SAR (ICLR'23)",
           note="NOTE (NeurIPS'22), feature head", rotta_grid="RoTTA (CVPR'23), feature head", csn_grid="CSN (CoLLAs'24)",
           lame="LAME (CVPR'22)", t3a="T3A (NeurIPS'21)", srcbn_grid="Source/BN mixture (tuned)", bnprior_grid="BN + source prior (tuned)",
           laem_grid="LAEM (WACV'26), episodic feature head", dart="DART-style (TMLR'25)", cpn="CPN (round 10)",
           neucal="known-neutral references (SFDA-PFT setting)", cpn_oracle="CPN with the true reference labels (oracle)",
           icn="ICN (ours)", icn_noaug="- virtual-person augmentation", icn_noxfit="- cross-fitting (full classifier in training)",
           icn_iid="training compositions: natural only", icn_nocomp="- composition compensation", icn_noscale="- scale gate",
           icn_nosrc="- classifier evidence (geometry only)", icn_nogeo="- geometry (classifier evidence only)",
           icn_raw="appearance-aware tokens (raw features)", icn_free="free centre offset (no span(F) constraint)",
           icn_direct="no normalization bottleneck (set network predicts the label)")
TUNED = ["srcbn_grid", "bnprior_grid", "csn_grid", "tent_grid", "sar_grid", "laem_grid", "rotta_grid"]
BASE = [m for m in R.MAIN if m not in ("icn", "source")]  # every other main-table method, CPN included
OBJ = lambda v: np.mean([v[m] for m in MIX], 0)  # noqa: E731  validation objective of the tuned baselines
STREAM_LAB = dict(source="Source", bn_win="BN, causal window", cpn_win="CPN, causal window", icn_win="ICN, causal window",
                  tent_online="TENT-online", tent_online_reset="TENT-online (reset)", note_full="NOTE", note_full_reset="NOTE (reset)",
                  rotta_full="RoTTA", rotta_full_reset="RoTTA (reset)")
RNG = 20261107
LABSETS, NATSETS = R.LAB, list(R.NAT)
EP = ["s", "rho", "nu", "a_self", "comp", "err_icn", "err_bn", "err_src"]


def name(m):
    return LAB.get(m, m)


class Job:
    """predictions of one run: {readout: {protocol: (draws, N, C)}} ('' = the method, 'alt:x' = saved grid values,
    'tuned' = the validation-selected grid value per fold); EP: per-episode diagnostics {name: {protocol: (draws, N)}}"""

    def __init__(self, j, need_ep=False):
        st = R.stem(j)
        path = R.OUT / (st + ".json")
        self.ok = path.exists()
        if not self.ok:
            return
        r = json.loads(path.read_text())
        self.r, self.j = r, j
        meta = pd.read_csv(f"{CM.PROC}/{j['data']}_meta.csv")
        self.y, self.act, self.fold = meta.label.values, meta.actor.astype(str).values, meta.fold.values
        N, C = len(meta), meta.label.nunique()
        self.C = C
        order = r["pred_order"]
        groups = {}
        for i, pr in enumerate(order):
            groups.setdefault(pr.split("#")[0], []).append(i)
        names = [""] + [f"alt:{a}" for a in r.get("alt_names", [])]
        self.P = {nm: {pr: np.zeros((len(ix), N, C), np.float32) for pr, ix in groups.items()} for nm in names}
        self.EP = {}
        folds = sorted(int(f) for f in r["folds"])
        assert set(folds) == set(CM.test_folds(meta)), f"incomplete folds {path}"
        sel = {}
        for f in folds:
            rows = np.flatnonzero(meta.fold.values == f)
            base = R.OUT / "pred" / f"{st}_f{f}"
            a = np.load(f"{base}.npy")
            assert a.shape == (len(order), len(rows), C) and np.isfinite(a).all(), base
            assert a.min() >= -1e-6 and np.allclose(a.sum(-1), 1, atol=1e-3), f"invalid probabilities: {base}"
            for pr, ix in groups.items():
                self.P[""][pr][:, rows] = a[ix]
            if r.get("alt_names"):
                al = np.load(f"{base}_alt.npy").astype(np.float32)
                for k, an in enumerate(r["alt_names"]):
                    for pr, ix in groups.items():
                        self.P[f"alt:{an}"][pr][:, rows] = al[k][ix]
            if need_ep and r.get("ep_names"):
                z = np.load(f"{base}_ep.npz")
                for an in z.files:
                    if an not in EP:
                        continue
                    self.EP.setdefault(an, {pr: np.full((len(ix), N), np.nan, np.float32) for pr, ix in groups.items()})
                    for pr, ix in groups.items():
                        self.EP[an][pr][:, rows] = z[an][ix].astype(np.float32)
            v = r["folds"][str(f)].get("crossfit", {}).get("val", {}).get(j["method"])
            if v is not None:
                sel[f] = CM.grid_label(j["method"], CM.GRIDS[j["method"]][int(np.argmax(OBJ(v)))])
        if sel:
            self.P["tuned"] = {pr: np.zeros((len(ix), N, C), np.float32) for pr, ix in groups.items()}
            for f, an in sel.items():
                rows = np.flatnonzero(meta.fold.values == f)
                for pr in groups:
                    self.P["tuned"][pr][:, rows] = self.P[f"alt:{an}"][pr][:, rows]
        self.sel = sel

    def main_name(self):
        return "tuned" if "tuned" in self.P else ""


def metric(p, y, C, kind):
    pred = p.argmax(-1)
    if kind == "acc":
        return 100 * (pred == y).mean()
    cs = [c for c in range(C) if np.any(y == c)]
    if kind == "uar":
        return 100 * np.mean([(pred[y == c] == c).mean() for c in cs])
    return 100 * np.mean([2 * np.sum((pred == c) & (y == c)) / max(1, np.sum(pred == c) + np.sum(y == c)) for c in cs])


class Data:
    """all runs of one dataset for one seed and section: method -> Job"""

    def __init__(self, d, seed, section, methods=None, need_ep=False, feat=None):
        self.d, self.seed = d, seed
        self.J = {}
        for j in R.suites()[section]:
            if j["data"] == d and j["seed"] == seed and (methods is None or j["method"] in methods) and (feat is None or j["feat"] == feat):
                jb = Job(j, need_ep)
                if jb.ok:
                    self.J[j["method"]] = jb
        any_ = next(iter(self.J.values()), None)
        self.ok = any_ is not None
        if not self.ok:
            return
        self.y, self.act, self.C = any_.y, any_.act, any_.C
        self.subj, self.sid = np.unique(self.act, return_inverse=True)
        self.sizes = np.bincount(self.sid)
        self.protos = [p for p in ["K0"] + FREE + ["same", "neu6"] + MIX if p in any_.P[""]]

    def readout(self, m, nm=None):
        jb = self.J.get(m)
        if jb is None:
            return None
        return jb.P.get(jb.main_name() if nm is None else nm)

    def val(self, m, pr, kind="acc", nm=None, draws=None):
        P = self.readout(m, nm)
        if P is None:
            return None
        if pr == "cavg":
            return float(np.mean([self.val(m, x, kind, nm, draws) for x in MIX])) if all(x in P for x in MIX) else None
        if pr == "worst":
            ps = [x for x in FREE + ["same"] if x in P]
            return float(min(self.val(m, x, kind, nm, draws) for x in ps))
        if pr not in P:
            return None
        return float(np.mean([metric(q, self.y, self.C, kind) for q in P[pr][:draws]]))

    def corr(self, m, pr, nm=None, draws=None):
        P = self.readout(m, nm)
        prs = MIX if pr == "cavg" else [pr]
        c = np.mean([(P[x][:draws].argmax(-1) == self.y[None]).mean(0) for x in prs], 0)
        return np.bincount(self.sid, c, minlength=len(self.subj))

    def paired(self, m1, m2, pr, nm1=None, nm2=None, B=5000):
        """pooled accuracy difference m1 - m2: estimate, 95% identity-bootstrap CI, one-sided 95% lower bound, sign-flip p"""
        a, b = self.corr(m1, pr, nm1), self.corr(m2, pr, nm2)
        est = 100 * (a - b).sum() / self.sizes.sum()
        ix = np.random.default_rng(RNG).integers(len(self.subj), size=(B, len(self.subj)))
        bs = 100 * (a - b)[ix].sum(1) / self.sizes[ix].sum(1)
        return est, np.percentile(bs, [2.5, 97.5]), float(np.percentile(bs, 5)), signflip(a - b)

    def strongest(self, pr, kind="acc", pool=None):
        cand = [(self.val(m, pr, kind), m) for m in (pool or BASE) if self.val(m, pr, kind) is not None]
        return max(cand)[1] if cand else None

    def ep(self, m, key, pr):
        jb = self.J.get(m)
        if jb is None or key not in jb.EP or pr not in jb.EP[key]:
            return None
        return float(np.nanmean(jb.EP[key][pr]))


def signflip(dlt, B=20000):
    dlt = dlt[np.isfinite(dlt)]
    n = len(dlt)
    if n == 0:
        return float("nan")
    obs = abs(dlt.mean())
    if n <= 16:
        s = np.array(np.meshgrid(*[[-1, 1]] * n)).reshape(n, -1).T
    else:
        s = np.random.default_rng(7).choice([-1, 1], size=(B, n))
    hits = np.sum(np.abs((s * dlt).mean(1)) >= obs - 1e-12)
    return float(hits / len(s)) if n <= 16 else float((hits + 1) / (len(s) + 1))


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); out = np.empty(len(ps)); mx = 0
    for i, j in enumerate(order):
        mx = max(mx, min(1, (len(ps) - i) * ps[j])); out[j] = mx
    return out


def pooled_paired(Xs, m1, m2, pr, B=5000):
    rng = np.random.default_rng(RNG)
    est, bs = 0.0, np.zeros(B)
    for X in Xs:
        a, b = X.corr(m1, pr), X.corr(m2, pr)
        est += 100 * (a - b).sum() / X.sizes.sum() / len(Xs)
        ix = rng.integers(len(X.subj), size=(B, len(X.subj)))
        bs += 100 * (a - b)[ix].sum(1) / X.sizes[ix].sum(1) / len(Xs)
    return est, np.percentile(bs, [2.5, 97.5]), float(np.percentile(bs, 5))


def f2(x, dec=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{dec}f}"


def msd(vals, dec=2):
    v = [x for x in vals if x is not None]
    if not v:
        return "n/a"
    if len(v) == 1:
        return f2(v[0], dec)
    return f"{np.mean(v):.{dec}f} ± {np.std(v, ddof=1):.{dec}f}"


def table(head, rows):
    return "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"


def bold_max(cells, vals):
    """bold the cell with the largest mean"""
    v = [(-np.inf if x is None else x) for x in vals]
    if not v or max(v) == -np.inf:
        return cells
    k = int(np.argmax(v))
    out = list(cells); out[k] = f"**{out[k]}**"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="aggregate whatever is finished")
    a = ap.parse_args()
    PAPER.mkdir(exist_ok=True)
    out, res = ["# ICN: In-Context Normalization -- formal study\n"], {}
    man = json.loads((R.OUT / "manifest.json").read_text())
    out.append(f"Code `icn.py` {man['code']['icn.py']}, `cmn.py` {man['code']['cmn.py']}. Training seeds {R.SEEDS} for the main suite; "
               "other suites seed 0. Metrics are pooled over all held-out clips, averaged over reference draws, then mean ± sample SD over seeds.\n")

    # ------------------------------------------------------------------ main: lab datasets
    main_rows = {}
    D = {}  # (data, seed) -> Data
    for d in R.DATA:
        for s in R.SEEDS:
            X = Data(d, s, "main")
            if X.ok:
                D[(d, s)] = X
    missing = [(d, s, m) for d in R.DATA for s in R.SEEDS for m in R.MAIN if (d, s) not in D or m not in D[(d, s)].J]
    if missing and not a.partial:
        raise SystemExit(f"{len(missing)} main runs missing, e.g. {missing[:8]} (use --partial)")
    res["missing_main"] = [list(map(str, x)) for x in missing]
    out.append("## 1. Main comparison, laboratory face datasets (accuracy %, mean ± SD over three training seeds)\n")
    out.append("rand: random same-person references; skew: 4 clips of one random class + 2 random; dir02: Dirichlet(0.2) composition; "
               "same: all references share the query class; cavg: average over 0..6 same-class references; worst: minimum over rand/skew/dir02/same.\n")
    cols = ["K0", "rand", "skew", "dir02", "same", "cavg", "worst"]
    for d in LABSETS:
        Xs = [D[(d, s)] for s in R.SEEDS if (d, s) in D]
        if not Xs:
            continue
        out.append(f"### {DN[d]}\n")
        rows, means = [], {c: [] for c in cols}
        for m in R.MAIN + (["neucal"] if d in R.WITH_NEUTRAL else []):
            vals = {c: [X.val(m, c) for X in Xs if m in X.J] for c in cols}
            if not any(v for v in vals.values()):
                continue
            cells = [msd(vals[c]) for c in cols]
            rows.append([name(m)] + cells)
            for c in cols:
                means[c].append(np.mean([v for v in vals[c] if v is not None]) if any(v is not None for v in vals[c]) else None)
            main_rows[(d, m)] = {c: (np.mean(vals[c]) if vals[c] and all(v is not None for v in vals[c]) else None) for c in cols}
        # bold the column maxima
        for ci, c in enumerate(cols):
            k = int(np.nanargmax([(-np.inf if v is None else v) for v in means[c]]))
            rows[k][ci + 1] = f"**{rows[k][ci + 1]}**"
        out.append(table(["Method"] + cols, rows))
    # ------------------------------------------------------------------ main: natural / speech datasets
    out.append("## 2. Main comparison, conversational and speech datasets (mean ± SD over three training seeds)\n")
    out.append("rand references; past = the K most recent earlier clips of the same person where an order exists. Acc / UAR / macro-F1 in %.\n")
    for d in NATSETS:
        Xs = [D[(d, s)] for s in R.SEEDS if (d, s) in D]
        if not Xs:
            continue
        has_past = "past" in Xs[0].protos
        out.append(f"### {DN[d]}\n")
        head = ["Method", "rand Acc", "rand UAR", "rand F1"] + (["past Acc", "past UAR", "past F1"] if has_past else []) + ["same Acc", "cavg Acc", "K0 Acc"]
        rows, means = [], []
        for m in R.MAIN + (["neucal"] if d in R.WITH_NEUTRAL else []):
            cells, mv = [], []
            specs = [("rand", "acc"), ("rand", "uar"), ("rand", "f1")] + ([("past", "acc"), ("past", "uar"), ("past", "f1")] if has_past else []) + [("same", "acc"), ("cavg", "acc"), ("K0", "acc")]
            vals = {sp: [X.val(m, sp[0], sp[1]) for X in Xs if m in X.J] for sp in specs}
            if not any(v for v in vals.values()):
                continue
            for sp in specs:
                cells.append(msd(vals[sp])); mv.append(np.mean(vals[sp]) if vals[sp] and all(v is not None for v in vals[sp]) else None)
            rows.append([name(m)] + cells); means.append(mv)
            main_rows[(d, m)] = {f"{sp[0]}_{sp[1]}": mv[i] for i, sp in enumerate(specs)}
        for ci in range(len(head) - 1):
            col = [(-np.inf if mv[ci] is None else mv[ci]) for mv in means]
            k = int(np.argmax(col)); rows[k][ci + 1] = f"**{rows[k][ci + 1]}**"
        out.append(table(head, rows))
    res["main"] = {f"{d}|{m}": v for (d, m), v in main_rows.items()}

    # ------------------------------------------------------------------ composition curves
    out.append("## 3. Composition curves: accuracy (%) vs the number j of references sharing the query class (mean over seeds)\n")
    curves = {}
    for d in R.DATA:
        Xs = [D[(d, s)] for s in R.SEEDS if (d, s) in D]
        if not Xs:
            continue
        out.append(f"### {DN[d]}\n")
        rows = []
        for m in ["source", "bn", "cpn", "icn"] + [m for m in BASE if m not in ("bn", "cpn")]:
            vals = [[X.val(m, x) for X in Xs if m in X.J] for x in MIX]
            if not vals[0]:
                continue
            rows.append([name(m)] + [msd(v) for v in vals])
            curves[f"{d}|{m}"] = [float(np.mean(v)) for v in vals]
        out.append(table(["Method"] + [f"j={j}" for j in range(7)], rows))
    res["curves"] = curves

    # ------------------------------------------------------------------ K curve
    out.append("## 4. Accuracy (%) vs the number of references K (rand references, mean over seeds)\n")
    kc = {}
    rows = []
    for d in R.DATA:
        Xs = [D[(d, s)] for s in R.SEEDS if (d, s) in D]
        if not Xs:
            continue
        for m in ["bn", "cpn", "icn"]:
            vals = [[X.val(m, k) for X in Xs if m in X.J and k in X.J[m].P[""]] for k in ["K0", "K1", "K2", "K4", "rand", "K8"]]
            if not vals[1]:
                continue
            rows.append([DN[d], name(m)] + [msd(v) for v in vals])
            kc[f"{d}|{m}"] = [float(np.mean(v)) if v else None for v in vals]
    out.append(table(["Dataset", "Method", "K=0", "K=1", "K=2", "K=4", "K=6", "K=8"], rows))
    res["kcurve"] = kc

    # ------------------------------------------------------------------ statistics (seed 0)
    out.append("## 5. Paired comparisons (seed-0 reference evaluation; identity-cluster bootstrap, 95% CI; sign-flip p, Holm within dataset)\n")
    stats = {}
    rows = []
    for d in R.DATA:
        X = D.get((d, 0))
        if not X or "icn" not in X.J:
            continue
        comps = []
        sb = X.strongest("cavg")
        if sb: comps.append(("cavg", sb, f"ICN - strongest other ({name(sb)})"))
        if "bn" in X.J: comps.append(("rand", "bn", "ICN - BN-adapt"))
        if "source" in X.J: comps.append(("same", "source", "ICN - Source"))
        if "cpn" in X.J: comps += [("cavg", "cpn", "ICN - CPN"), ("rand", "cpn", "ICN - CPN")]
        ps = []
        for pr, m, lab in comps:
            est, ci, lo, p = X.paired("icn", m, pr)
            ps.append(p); stats[f"{d}|{pr}|{m}"] = dict(est=est, ci=list(map(float, ci)), lower=lo, p=p)
            rows.append([DN[d], pr, lab, f"{est:+.2f}", f"[{ci[0]:+.2f}, {ci[1]:+.2f}]", f"{lo:+.2f}", f"{p:.4f}", ""])
        adj = holm(ps)
        for k in range(len(ps)):
            rows[len(rows) - len(ps) + k][-1] = f"{adj[k]:.4f}"
            stats[f"{d}|{comps[k][0]}|{comps[k][1]}"]["p_holm"] = float(adj[k])
    out.append(table(["Dataset", "Protocol", "Comparison", "Difference", "95% CI", "One-sided 95% lower", "p", "Holm p"], rows))
    # pooled endpoints
    out.append("### Pooled endpoints (dataset-stratified identity bootstrap; non-inferiority margins as in the earlier study)\n")
    rows = []
    endpoints = [("lab rand vs BN-adapt", [D[(d, 0)] for d in LABSETS if (d, 0) in D], "bn", "rand", 1.5),
                 ("lab cavg vs CPN", [D[(d, 0)] for d in LABSETS if (d, 0) in D], "cpn", "cavg", None),
                 ("all real sets, same vs Source", [D[(d, 0)] for d in R.DATA if (d, 0) in D], "source", "same", 2.0),
                 ("natural + external sets, rand vs BN-adapt", [D[(d, 0)] for d in NATSETS if (d, 0) in D], "bn", "rand", 1.5),
                 ("natural + external sets, rand vs Source", [D[(d, 0)] for d in NATSETS if (d, 0) in D], "source", "rand", 1.0)]
    res["pooled"] = {}
    for lab, Xs, m, pr, margin in endpoints:
        Xs = [X for X in Xs if "icn" in X.J and m in X.J]
        if not Xs:
            continue
        est, ci, lo = pooled_paired(Xs, "icn", m, pr)
        ni = "n/a" if margin is None else ("yes" if lo > -margin else "no")
        rows.append([lab, len(Xs), f"{est:+.2f}", f"[{ci[0]:+.2f}, {ci[1]:+.2f}]", f"{lo:+.2f}", "n/a" if margin is None else margin, ni])
        res["pooled"][lab] = dict(est=est, ci=list(map(float, ci)), lower=lo, margin=margin, n=len(Xs))
    out.append(table(["Endpoint", "Datasets", "Difference", "95% CI", "Lower", "Margin", "NI"], rows))
    res["stats"] = stats

    # ------------------------------------------------------------------ ablation (seed 0, first draw, matched)
    out.append("## 6. Ablation of ICN (seed 0, first reference draw, matched episodes): accuracy (%)\n")
    abl = {}
    for pr in ["cavg", "rand", "same"]:
        out.append(f"### {pr}\n")
        rows = []
        full = {d: (D[(d, 0)].val("icn", pr, draws=1) if (d, 0) in D and "icn" in D[(d, 0)].J else None) for d in R.DATA}
        rows.append(["ICN (full)"] + [f2(full[d]) for d in R.DATA])
        A = {d: Data(d, 0, "ablation") for d in R.DATA}
        for m in R.ABL:
            vals = {d: (A[d].val(m, pr) if A[d].ok and m in A[d].J else None) for d in R.DATA}
            if not any(v is not None for v in vals.values()):
                continue
            rows.append([name(m)] + [(f"{vals[d]:.2f} ({vals[d] - full[d]:+.2f})" if vals[d] is not None and full[d] is not None else "n/a") for d in R.DATA])
            abl[f"{pr}|{m}"] = {d: vals[d] for d in R.DATA}
        abl[f"{pr}|icn"] = full
        out.append(table(["Variant"] + [DN[d] for d in R.DATA], rows))
    res["ablation"] = abl

    # ------------------------------------------------------------------ mechanism: gates and centre errors (seed 0)
    out.append("## 7. What the normalizer does: mean gates of the query clip and centre errors (seed 0, all draws)\n")
    out.append("s: centre trust; rho: scale trust; nu: composition-compensation trust; a_self: attention weight of the query in its own centre; "
               "comp: norm of the compensation. err_*: squared distance between the implied person centre and the person's true balanced "
               "mean, relative to the mean squared norm of the true centres (BN: the batch mean; Source: no centre).\n")
    mech = {}
    rows = []
    for d in R.DATA:
        X = Data(d, 0, "main", methods=["icn"], need_ep=True)
        if not X.ok:
            continue
        for pr in ["rand", "dir02", "same", "mix0", "mix3", "mix6"]:
            if pr not in X.protos:
                continue
            v = {k: X.ep("icn", k, pr) for k in EP}
            mech[f"{d}|{pr}"] = v
            rows.append([DN[d], pr] + [f2(v[k], 3) for k in EP])
    out.append(table(["Dataset", "Protocol"] + EP, rows))
    res["mechanism"] = mech

    # ------------------------------------------------------------------ synthetic
    out.append("## 8. Controlled synthetic variation (exact person centres): accuracy (%) and centre errors\n")
    syn = {}
    rows = []
    for dname, cfg in R.SYN.items():
        X = Data(dname, 0, "synth", need_ep=True)
        if not X.ok:
            continue
        for m in R.SYN_M:
            if m not in X.J:
                continue
            v = dict(rand=X.val(m, "rand"), same=X.val(m, "same"), cavg=X.val(m, "cavg"))
            e = {k: X.ep(m, k, "rand") for k in ("err_icn", "err_bn", "err_src")} if m == "icn" else {}
            syn[f"{dname}|{m}"] = {**v, **e}
            rows.append([dname, cfg["--s_G"], cfg["--s_e"], name(m), f2(v["rand"]), f2(v["same"]), f2(v["cavg"]),
                         f2(e.get("err_icn"), 3), f2(e.get("err_bn"), 3), f2(e.get("err_src"), 3)])
    out.append(table(["Regime", "G scale", "offset scale", "Method", "rand", "same", "cavg", "centre err ICN (rand)", "centre err BN", "centre err Source"], rows))
    res["synth"] = syn

    # ------------------------------------------------------------------ stream
    out.append("## 9. Person-blocked streams (seed 0): pooled UAR (%)\n")
    strm = {}
    rows = []
    keys = ["source", "bn_win", "cpn_win", "icn_win", "tent_online", "tent_online_reset", "note_full", "note_full_reset", "rotta_full", "rotta_full_reset"]
    for d in R.DATA:
        fn = R.OUT / f"stream_{d}_{R.FEAT[d]}_s0.json"
        if not fn.exists():
            continue
        r = json.loads(fn.read_text())
        strm[d] = {k: r["pooled"][k] for k in keys if k in r["pooled"]}
        rows.append([DN[d]] + [f2(r["pooled"][k]["uar"]) if k in r["pooled"] else "n/a" for k in keys])
    out.append(table(["Dataset"] + [STREAM_LAB[k] for k in keys], rows))
    res["stream"] = strm

    # ------------------------------------------------------------------ representation
    out.append("## 10. Alternative representation (CLIP-B/16 on the lab sets, face-only CLIP-L/14 on MELD; seed 0): accuracy (%)\n")
    rep = {}
    rows = []
    for d, f in R.REPR.items():
        X = Data(d, 0, "repr", feat=f)
        if not X.ok:
            continue
        for m in R.REPR_M:
            if m not in X.J:
                continue
            v = dict(rand=X.val(m, "rand"), same=X.val(m, "same"), cavg=X.val(m, "cavg"))
            rep[f"{d}|{m}"] = v
            rows.append([DN[d], f, name(m), f2(v["rand"]), f2(v["same"]), f2(v["cavg"])])
    out.append(table(["Dataset", "Features", "Method", "rand", "same", "cavg"], rows))
    res["repr"] = rep

    # ------------------------------------------------------------------ profile
    out.append("## 11. Prediction-head cost (RTX 4090 D, one episode online / batched; K references)\n")
    prof = {}
    rows = []
    for d, f in R.PROFILE.items():
        fn = R.OUT / f"profile_{d}_{f}_iid.json"
        if not fn.exists():
            continue
        r = json.loads(fn.read_text())
        prof[d] = r["rows"]
        for row in r["rows"]:
            rows.append([DN[d], row["K"], name(row["method"]), f2(row["online_ms"]), f2(row["batch_ms_per_episode"], 3), row["batch"], f2(row["peak_mb_per_batch"], 1), f2(row["acc"])])
        prof[d + "_icn_params"] = r.get("icn_params")
    out.append(table(["Dataset", "K", "Method", "Online ms", "Batch ms/episode", "Batch", "Peak MB", "rand acc"], rows))
    res["profile"] = prof

    (PAPER / "icn_tables.md").write_text("\n".join(out), encoding="utf-8")
    (PAPER / "icn_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("wrote", PAPER / "icn_tables.md")


if __name__ == "__main__":
    main()
