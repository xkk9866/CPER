"""Tables of the round-10 CPN study from the manifest of run_cpn.py and the saved predictions
(-> paper/cpn10_tables.md, paper/cpn10_results.json).  Every number of the manuscript must come from these files.

Accuracy / UAR / macro-F1 are pooled over all held-out clips (every clip is held out exactly once) and averaged over the
reference draws (one training seed).  Confidence intervals resample PEOPLE (identity-cluster bootstrap): accuracy
differences from per-person correct counts, UAR / macro-F1 differences from per-person confusion matrices (the macro
metrics are recomputed on every bootstrap population).  Paired accuracy tests flip the signs of per-person differences,
Holm-corrected within a table.  Tuned baselines choose their operating point per fold on held-out TRAINING people
(cross-fit split, composition-average objective fixed before the study); never on test data."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, json, pathlib, sys
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_cpn as R  # noqa: E402
import cmn as CM  # noqa: E402

PAPER = pathlib.Path(_data_root) / "reports"
DN = dict(ravdess="RAVDESS", cremad="CREMA-D", mead="MEAD", enterface="eNTERFACE'05", meld="MELD-AVT",
          shemo="ShEMO (diagnosed in round 7)", emoz="Emozionalmente (external, round 8)",
          thai="THAI-SER Zoom (session-isolated; first evaluated in round 9)",
          thaix="THAI-SER Studio, frozen Zoom-trained model (untouched, cross-condition)", **{k: k for k in R.SYN})
CLASSES = dict(shemo=["anger", "happiness", "neutral", "sadness", "surprise"],
               thai=["angry", "frustrated", "happy", "neutral", "sad"],
               thaix=["angry", "frustrated", "happy", "neutral", "sad"],
               emoz=["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
               meld=[f"c{k}" for k in range(7)])
MIX = [f"mix{j}" for j in range(7)]
FREE = ["rand", "skew", "dir02", "past"]
LAB = dict(source="Source", bn="BN-adapt (NeurIPS'20)", tent_grid="TENT (ICLR'21)", sar_grid="SAR (ICLR'23)",
           delta="DELTA (ICLR'23), episodic", note="NOTE (NeurIPS'22), feature head", rotta_grid="RoTTA (CVPR'23), feature head",
           csn_grid="CSN (CoLLAs'24)", lame="LAME (CVPR'22)", t3a="T3A (NeurIPS'21)", srcbn_grid="Source/BN mixture (tuned)",
           bnprior_grid="BN + source prior (tuned)", src_lab="Source + EB person label prior", cpn="CPN (ours)",
           laem_grid="LAEM (WACV'26), episodic feature head", dart="DART-style (TMLR'25)", laem="LAEM (10 steps)",
           neucal="known-neutral references (SFDA-PFT setting)", cpn_nopostcal="- calibrated composition posterior",
           cpn_std="- source-anchored transport (round-7 standardisation in evidence and read-outs)",
           cpn7="CPN round 7 (standardisation, uncalibrated)", cmn="CMN (round 6)",
           tent="TENT (10 steps)", sar="SAR (10 steps)", rotta="RoTTA (alpha 0.05)", csn="CSN (R 0.5)",
           cpn_p16="CPN, pruned M=16", cpn_p64="CPN, pruned M=64", cpn_p256="CPN, pruned M=256",
           pl_shrink="independent pseudo-labels -> conservative branch (compare with: conservative branch only)",
           cpn_nodens="- population density", cpn_isodens="density -> isotropic prototypes", cpn_noanchor="- set anchor",
           cpn_bnlik="network evidence -> one batch-mode normalisation", cpn_plainanchor="- identifiability weight of the anchor",
           cpn_indepG="- shared person x class interaction", cpn_noint="- person-scale integration",
           cpn_noequi="- scale-equivariant hypotheses", cpn_oracle="label oracle (true reference labels)")
ALT = {"alt:cond": "exact Bayes branch only (= exact joint posterior marginal)",
       "alt:eb": "conservative branch only (posterior-moment shrinkage)",
       "alt:bn": "batch-moment branch only (= BN-adapt, cross-fitted temperature)",
       "alt:src": "source branch only (= Source)",
       "alt:wg": "round-9 fixed weighting (w_E = g, no source branch, not fitted)",
       "alt:stack_nobn": "stacking without the batch-moment branch (round-10a design)",
       "alt:stack_nosrc": "stacking without the source branch",
       "alt:flat": "stacking without the identifiability covariate (constant weights)",
       "alt:gate_conc": "episode gate on composition concentration E[g] (same branches, data, objective)",
       "alt:gate_ent": "episode gate on Source entropy (same branches, data, objective)",
       "alt:gate_learn": "learned episode gate on 5 generic features (same branches, data, objective)",
       "alt:stack_map": "posterior -> MAP composition (same branches and combiner)",
       "alt:st:nocov": "- shrinkage in the conservative branch",
       "alt:st:r_std": "- transport in the conservative branch"}
COMBINERS = [("cpn", None, "identifiability stacking (ours)"), ("cpn", "alt:flat", "constant stacking (no g)"),
             ("cpn", "alt:gate_conc", "concentration gate E[g]"), ("cpn", "alt:gate_ent", "Source-entropy gate"),
             ("cpn", "alt:gate_learn", "learned gate (5 features)"), ("cpn", "alt:wg", "round-9 fixed w = g"),
             ("cpn", "alt:stack_nobn", "stacking without batch moments (round 10a)"),
             ("cpn", "alt:stack_nosrc", "stacking without Source"), ("cpn", "alt:cond", "exact Bayes only"),
             ("cpn", "alt:eb", "conservative only"), ("cpn", "alt:bn", "batch moments only"),
             ("bn", None, "BN-adapt"), ("source", None, "Source")]
KCTX = 6  # references per episode (cmn.py --nctx default)
TUNED = ["srcbn_grid", "bnprior_grid", "csn_grid", "tent_grid", "sar_grid", "laem_grid", "rotta_grid"]
BASE = [m for m in R.MAIN if m not in ("cpn", "source")]
STREAM_LAB = dict(source="Source", bn_win="BN, causal window K=6", cpn_win="CPN, causal window K=6",
                  tent_online="TENT-online (continual)", tent_online_reset="TENT-online (reset per person)",
                  note_full="NOTE feature-head (continual)", note_full_reset="NOTE feature-head (reset per person)",
                  rotta_full="RoTTA feature-head (continual)", rotta_full_reset="RoTTA feature-head (reset per person)")
OBJ = lambda v: np.mean([v[m] for m in MIX], 0)  # noqa: E731  held-out composition average (pre-registered)
SHA = None
RNG = 20261106


def name(m):
    return LAB.get(m, m)


class Job:
    """predictions of one run: {readout: {protocol: (draws, N, C)}} ('' = the method, 'alt:x' saved read-outs or grid
    values, 'tuned' = the validation-selected grid value per fold); EP: per-episode diagnostics"""

    def __init__(self, j):
        st = R.stem(j)
        path = R.OUT / (st + ".json")
        self.ok = path.exists()
        if not self.ok:
            return
        r = json.loads(path.read_text())
        assert r["code_sha256"] == SHA, f"version mismatch {path}"
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
        assert set(folds) == set(CM.test_folds(meta)) or R.SMOKE, f"incomplete folds {path}"
        sel = {}
        self.xfit = {}
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
                assert np.isfinite(al).all() and al.min() >= -1e-6 and np.allclose(al.sum(-1), 1, atol=1e-3), f"invalid alternative probabilities: {base}"
                for k, an in enumerate(r["alt_names"]):
                    for pr, ix in groups.items():
                        self.P[f"alt:{an}"][pr][:, rows] = al[k][ix]
            if r.get("ep_names"):
                z = np.load(f"{base}_ep.npz")
                for an in z.files:
                    self.EP.setdefault(an, {pr: np.full((len(ix), N), np.nan, np.float32) for pr, ix in groups.items()})
                    for pr, ix in groups.items():
                        self.EP[an][pr][:, rows] = z[an][ix]
            xf = r["folds"][str(f)].get("crossfit", {})
            self.xfit[f] = {k: v for k, v in xf.items() if k not in ("val", "cert")}
            v = xf.get("val", {}).get(j["method"])
            if v is not None:
                sel[f] = CM.grid_label(j["method"], CM.GRIDS[j["method"]][int(np.argmax(OBJ(v)))])
        self.rows = np.flatnonzero(np.isin(meta.fold.values, folds))
        if sel:
            self.P["tuned"] = {pr: np.zeros((len(ix), N, C), np.float32) for pr, ix in groups.items()}
            for f, an in sel.items():
                rows = np.flatnonzero(meta.fold.values == f)
                for pr in groups:
                    self.P["tuned"][pr][:, rows] = self.P[f"alt:{an}"][pr][:, rows]
        self.sel = sel
        if len(self.rows) < N:  # smoke test: keep the evaluated folds only
            self.y, self.act, self.fold = self.y[self.rows], self.act[self.rows], self.fold[self.rows]
            for nm in self.P:
                for pr in self.P[nm]:
                    self.P[nm][pr] = self.P[nm][pr][:, self.rows]
            for an in self.EP:
                for pr in self.EP[an]:
                    self.EP[an][pr] = self.EP[an][pr][:, self.rows]

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


def conf_metric(cm, kind):
    """acc / UAR / macro-F1 from confusion matrices (..., C, C) (rows: true class)"""
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    row, col = cm.sum(-1), cm.sum(-2)
    if kind == "acc":
        return 100 * tp.sum(-1) / cm.sum((-2, -1))
    ok = row > 0
    if kind == "uar":
        return 100 * np.where(ok, tp / np.maximum(row, 1e-12), 0).sum(-1) / ok.sum(-1)
    return 100 * np.where(ok, 2 * tp / np.maximum(row + col, 1e-12), 0).sum(-1) / ok.sum(-1)


class Data:
    def __init__(self, d, jobs, section=("main", "ablation", "synth")):
        self.d = d
        self.J = {}
        for j in jobs:
            if j["data"] == d and j["section"] in section and j["method"] not in ("profile", "stream"):
                jb = Job(j)
                if jb.ok:
                    self.J[j["method"]] = jb
        any_ = next(iter(self.J.values()), None)
        if any_ is None:
            self.subj = None; return
        self.y, self.act, self.C = any_.y, any_.act, any_.C
        self.subj, self.sid = np.unique(self.act, return_inverse=True)
        self.sizes = np.bincount(self.sid)
        self.protos = [p for p in FREE + ["same"] + MIX if p in any_.P[""]]

    def readout(self, m, nm=None):
        jb = self.J.get(m)
        if jb is None:
            return None
        nm = jb.main_name() if nm is None else nm
        return jb.P.get(nm)

    def has(self, m, pr, nm=None):
        P = self.readout(m, nm)
        return P is not None and (pr in P or (pr == "cavg" and all(x in P for x in MIX)))

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

    def corr(self, m, pr, nm=None):
        """per-person correct counts averaged over draws (cavg: averaged over the curve)"""
        P = self.readout(m, nm)
        prs = MIX if pr == "cavg" else [pr]
        c = np.mean([(P[x].argmax(-1) == self.y[None]).mean(0) for x in prs], 0)
        return np.bincount(self.sid, c, minlength=len(self.subj))

    def pconf(self, m, pr, nm=None):
        """Per-person confusion matrices for each draw (draws, people, C, C)."""
        P = self.readout(m, nm)
        prs = MIX if pr == "cavg" else [pr]
        draws = []
        for x in prs:
            for q in P[x]:
                out = np.zeros((len(self.subj), self.C, self.C))
                np.add.at(out, (self.sid, self.y, q.argmax(-1)), 1.0)
                draws.append(out)
        return np.stack(draws)

    def paired(self, m1, m2, pr, nm1=None, nm2=None, B=5000):
        """difference of pooled accuracy m1 - m2: estimate, 95% identity-bootstrap CI, one-sided 95% lower, sign-flip p"""
        a, b = self.corr(m1, pr, nm1), self.corr(m2, pr, nm2)
        est = 100 * (a - b).sum() / self.sizes.sum()
        ix = np.random.default_rng(RNG).integers(len(self.subj), size=(B, len(self.subj)))
        bs = 100 * (a - b)[ix].sum(1) / self.sizes[ix].sum(1)
        return est, np.percentile(bs, [2.5, 97.5]), np.percentile(bs, 5), signflip(a - b)

    def paired_macro(self, m1, m2, pr, kind, nm1=None, nm2=None, B=2000):
        """difference of a pooled macro metric (UAR / F1): estimate, 95% identity-bootstrap CI, one-sided lower bound,
        bootstrap two-sided p"""
        c1, c2 = self.pconf(m1, pr, nm1), self.pconf(m2, pr, nm2)
        assert len(c1) == len(c2), "paired macro metrics require matched draws"
        est = float(np.mean(conf_metric(c1.sum(1), kind) - conf_metric(c2.sum(1), kind)))
        assert np.isclose(est, self.val(m1,pr,kind,nm1)-self.val(m2,pr,kind,nm2)), "macro estimand mismatch"
        w = np.random.default_rng(RNG).multinomial(len(self.subj), np.full(len(self.subj), 1 / len(self.subj)), size=B)
        P_, C2 = c1.shape[1], self.C * self.C
        bs = np.zeros(B)
        for a_,b_ in zip(c1,c2):
            bs += conf_metric((w @ a_.reshape(P_, C2)).reshape(B, self.C, self.C), kind) - \
                  conf_metric((w @ b_.reshape(P_, C2)).reshape(B, self.C, self.C), kind)
        bs /= len(c1)
        p = float(min(1.0, 2 * min((np.count_nonzero(bs <= 0)+1)/(B+1), (np.count_nonzero(bs >= 0)+1)/(B+1))))
        return est, np.percentile(bs, [2.5, 97.5]), np.percentile(bs, 5), p

    def strongest(self, pr, kind="acc"):
        """strongest baseline (held-out tuned where applicable) on a protocol, chosen ON THE TEST SET (conservative)"""
        cand = [(self.val(m, pr, kind), m) for m in BASE if self.has(m, pr)]
        cand = [c for c in cand if c[0] is not None]
        return max(cand)[1] if cand else None


def pooled_paired(Xs, m1, m2, pr, B=5000):
    """mean over datasets of the pooled-accuracy difference m1 - m2 on protocol pr: estimate, 95% CI and one-sided 95%
    lower bound of a dataset-stratified identity bootstrap (people resampled within each dataset)"""
    rng = np.random.default_rng(RNG)
    est, bs = 0.0, np.zeros(B)
    for X in Xs:
        a, b = X.corr(m1, pr), X.corr(m2, pr)
        est += 100 * (a - b).sum() / X.sizes.sum() / len(Xs)
        ix = rng.integers(len(X.subj), size=(B, len(X.subj)))
        bs += 100 * (a - b)[ix].sum(1) / X.sizes[ix].sum(1) / len(Xs)
    return est, np.percentile(bs, [2.5, 97.5]), np.percentile(bs, 5)


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


def auroc(score, target):
    """Mann-Whitney AUROC of a score for a binary target (nan-safe)"""
    ok = np.isfinite(score) & np.isfinite(target)
    s, t = score[ok], target[ok].astype(bool)
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = pd.Series(s).rank().values
    return float((r[t].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra, rb = pd.Series(a[ok]).rank().values, pd.Series(b[ok]).rank().values
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def f2(x, dec=2):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{dec}f}"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def bold_best(vals):
    ok = [v for v in vals if v is not None]
    best = max(ok) if ok else None
    return [("**" + f2(v) + "**") if v is not None and best is not None and abs(v - best) < 1e-9 else f2(v) for v in vals]


def ci(est, c):
    return f"{est:+.2f} [{c[0]:+.2f}, {c[1]:+.2f}]"


def stream_tables(datas, out, res):
    rows = []
    for d in datas:
        f = R.OUT / f"stream_{d}_{R.FEAT[d]}_s0.json"
        if not f.exists():
            continue
        s = json.loads(f.read_text())
        assert s["code_sha256"] == SHA, f"version mismatch {f}"
        meta = pd.read_csv(f"{CM.PROC}/{d}_meta.csv")
        C = meta.label.nunique()
        assert {int(k) for k in s["folds"]} == set(CM.test_folds(meta)), f"incomplete stream folds: {d}"
        for fold, rec in s["folds"].items():
            order=np.asarray(rec["order"])
            assert np.array_equal(np.sort(order), np.flatnonzero(meta.fold.values==int(fold))), f"stream order coverage: {d}/{fold}"
            pred=np.load(R.OUT/"pred"/f"stream_{d}_{R.FEAT[d]}_s0_f{fold}.npy",mmap_mode="r")
            assert pred.shape==(len(s["methods"]),len(order),C) and np.isfinite(pred).all(), f"stream predictions: {d}/{fold}"
            assert pred.min()>=-1e-6 and np.allclose(pred.sum(-1),1,atol=1e-3), f"stream probabilities: {d}/{fold}"
        res.setdefault("stream", {})[d] = s["pooled"]
        keys = s["methods"]
        best = {k: max(s["pooled"][m][k] for m in keys if m != "cpn_win") for k in ("acc", "uar", "f1")}
        for m in keys:
            v = s["pooled"][m]
            rows.append([DN[d], STREAM_LAB.get(m, m)] + [f2(v[k]) for k in ("acc", "uar", "f1")]
                        + ([" / ".join(f"{v[k] - best[k]:+.2f}" for k in ("acc", "uar", "f1"))] if m == "cpn_win" else ["—"]))
    if rows:
        out += ["## Table 7. Person-blocked test stream: implementations on the shared frozen feature head",
                "TENT/RoTTA process mini-batches of 8 consecutive clips; NOTE predicts individual clips. Implementations carry "
                "their state over the stream (continually, or reset at person boundaries), i.e. they accumulate the whole "
                "history; BN / CPN see only the 6 most recent EARLIER clips of the person.  Last column: CPN minus the best "
                "other stream entry (acc / UAR / F1). NOTE uses PBRS and vector BN without spatial instance statistics; RoTTA uses feature dropout/noise in place of image augmentation. These are feature-head adaptations, not unchanged official image implementations."] + \
            table(["Dataset", "Method", "acc", "UAR", "macro-F1", "CPN − best other"], rows) + [""]


def main():
    global SHA
    ap = argparse.ArgumentParser(); ap.add_argument("--partial", action="store_true"); a = ap.parse_args()
    man = json.loads((R.OUT / "manifest.json").read_text())
    SHA = man["code_sha256"]
    cur = CM.CODE_SHA
    assert SHA == cur or a.partial, f"manifest {SHA} != current cmn.py {cur}"
    jobs = man["jobs"]
    missing = [R.stem(j) for j in jobs if not (R.OUT / (R.stem(j) + ".json")).exists()]
    if missing and not a.partial:
        raise SystemExit(f"{len(missing)} formal jobs missing: {missing[:8]}")
    D = {d: Data(d, [j for j in jobs if j["feat"] == R.FEAT.get(d, "raw")]) for d in R.DATA + list(R.SYN)}
    D = {d: v for d, v in D.items() if v.subj is not None}
    out, res = ["# CPN round-10 experiments", f"Code SHA256 prefix `{SHA}`; one training seed; main protocols averaged over "
                "3 paired reference draws (ablation, representation and synthetic: 1).  Pooled over all held-out clips; cavg = "
                "composition curve j = 0..6 (stress test, equal weights); worst = minimum over the label-free protocols and "
                "same. Tuned baselines: operating point on held-out training people. THAI-SER Zoom and Emozionalmente "
                "were evaluated previously. Studio is the new cross-condition test, with a single frozen Zoom source "
                "model and shared calibration across target session partitions.", ""], {}
    real = [d for d in R.DATA if d in D]
    labs = [d for d in R.LAB if d in D]
    nat = [d for d in R.NAT if d in D]
    # ------------------------------------------------------------------ 1. main table (accuracy)
    out.append("## Table 1. Main comparison (accuracy, %)")
    for d in real:
        X = D[d]
        cols = [p for p in FREE if p in X.protos] + ["same", "cavg", "worst"]
        rows, vals = [], {c: [] for c in cols}
        ms = [m for m in R.MAIN if m in X.J]
        for m in ms:
            for c in cols:
                vals[c].append(X.val(m, c))
        B_ = {c: bold_best(vals[c]) for c in cols}
        for i, m in enumerate(ms):
            rows.append([name(m)] + [B_[c][i] for c in cols])
            res.setdefault(d, {})[m] = {c: vals[c][i] for c in cols}
        out += [f"**{DN[d]}**"] + table(["Method"] + cols, rows) + [""]
    for grp, ds in (("four lab datasets", labs), ("natural / external datasets", nat)):
        if not ds:
            continue
        cols = ["rand", "skew", "dir02", "same", "cavg", "worst"]
        rows = []
        for m in R.MAIN:
            v = [np.mean([res[d][m][c] for d in ds]) if all(m in res.get(d, {}) for d in ds) else None for c in cols]
            rows.append((m, v))
        B_ = list(zip(*[bold_best([r[1][k] for r in rows]) for k in range(len(cols))]))
        out += [f"**Average over the {grp}**"] + table(["Method"] + cols, [[name(m)] + list(B_[i]) for i, (m, _) in enumerate(rows)]) + [""]
    out.append("**Laboratory composition curves for Figure 1 (matched first reference draw)**")
    for d in labs:
        X=D[d]
        out += [DN[d]] + table(["Method"]+MIX,
            [[name(m)]+[f2(X.val(m,p,draws=1)) for p in MIX] for m in ["source","bn","csn_grid","cpn"] if m in X.J]) + [""]
    # ------------------------------------------------------------------ 2. natural data: acc / UAR / F1 with CIs
    if nat:
        out.append("## Table 2. Natural and external data: accuracy / UAR / macro-F1; CPN minus Source and minus the strongest baseline")
        for d in nat:
            X = D[d]
            prs = [p for p in ["past", "rand", "skew", "dir02"] if p in X.protos]
            rows = []
            for m in [m for m in R.MAIN if m in X.J]:
                rows.append([name(m)] + [" / ".join(f2(X.val(m, p, k)) for k in ("acc", "uar", "f1")) for p in prs])
            out += [f"**{DN[d]}** (acc / UAR / macro-F1)"] + table(["Method"] + prs, rows) + [""]
            rows = []
            for p in prs:
                for k in ("acc", "uar", "f1"):
                    sb = X.strongest(p, k)
                    if "cpn" not in X.J:
                        continue
                    for ref in ["source", sb]:
                        if ref is None or ref not in X.J:
                            continue
                        if k == "acc":
                            est, c, lo, pv = X.paired("cpn", ref, p)
                        else:
                            est, c, lo, pv = X.paired_macro("cpn", ref, p, k)
                        rows.append([p, k, name(ref), f2(X.val(ref, p, k)), f2(X.val("cpn", p, k)), ci(est, c), f"{lo:+.2f}", f2(pv, 4)])
                        res.setdefault("natural", {}).setdefault(d, {})[f"{p}|{k}|{ref}"] = dict(delta=est, ci=list(c), lower=lo, p=pv)
            out += [f"CPN minus Source and minus the strongest baseline (chosen per protocol and metric ON THE TEST SET; intervals "
                    f"do not adjust for that selection), {DN[d]}"] + \
                table(["protocol", "metric", "reference", "reference value", "CPN", "Δ [95% CI, people]", "one-sided 95% lower", "p"], rows) + [""]
    # ------------------------------------------------------------------ 3. ablation
    out.append("## Table 3. Ablation, replacement controls and read-outs (one component changed; each variant re-fits its own "
               "posterior weights and stacking combiner on the same held-out training people; matched first reference draw)")
    variants = [("cpn", "")] + [("cpn", k) for k in ALT] + [(m, "") for m in R.ABL]
    rows = []
    ds_lab = [d for d in labs if "cpn" in D[d].J]
    ds_nat = [d for d in nat if "cpn" in D[d].J]
    out.append("All datasets (matched first reference draw): " + ", ".join(DN[d] for d in ds_lab + ds_nat) + ".")
    for m, nm in variants:
        okl = [d for d in ds_lab if D[d].has(m, "rand", nm or None)]
        okn = [d for d in ds_nat if D[d].has(m, "rand", nm or None)]
        if len(okl) < len(ds_lab) or len(okn) < len(ds_nat):
            continue
        vl = {c: np.mean([D[d].val(m, c, nm=nm or None, draws=1) for d in okl]) if okl else None for c in ["rand", "same", "cavg"]}
        vn = {k: np.mean([D[d].val(m, "rand", k, nm=nm or None, draws=1) for d in okn]) if okn else None for k in ["acc", "uar", "f1"]}
        rows.append([("CPN (full)" if (m, nm) == ("cpn", "") else ALT.get(nm, name(m)))]
                    + [f2(vl[c]) for c in ["rand", "same", "cavg"]] + [f2(vn[k]) for k in ["acc", "uar", "f1"]])
        res.setdefault("ablation", {})[f"{m}{nm}"] = dict(lab=vl, nat=vn)
    if rows:
        full = res["ablation"].get("cpn")
        for r_ in rows[1:]:
            pass
        out += table(["Variant", "lab rand acc", "lab same acc", "lab cavg acc", "natural rand acc", "natural rand UAR",
                      "natural rand F1"], rows) + [""]
        rows = []
        for m, nm in variants:
            if not all(D[d].has(m, "rand", nm or None) for d in ds_lab + ds_nat):
                continue
            rows.append([("CPN (full)" if (m, nm) == ("cpn", "") else ALT.get(nm, name(m)))]
                        + [" / ".join(f2(D[d].val(m, p, nm=nm or None, draws=1)) for p in ["rand", "same", "cavg"]) for d in ds_lab]
                        + [" / ".join([f2(D[d].val(m, "rand", "acc", nm=nm or None, draws=1)), f2(D[d].val(m, "rand", "uar", nm=nm or None, draws=1))]
                                          + [f2(D[d].val(m, p, nm=nm or None, draws=1)) for p in ["same", "cavg"]]) for d in ds_nat])
        out += ["Per dataset (lab: rand / same / cavg accuracy; speech: rand accuracy / rand UAR / same accuracy / cavg accuracy)"] + \
            table(["Variant"] + [DN[d] for d in ds_lab + ds_nat], rows) + [""]
    # ------------------------------------------------------------------ 4. safety / non-inferiority
    out.append("## Table 4. Safety: non-inferiority to Source (pre-specified margins: hard protocol 2 points accuracy; "
               "natural random references 1 point UAR)")
    rows = []
    for d in real:
        X = D[d]
        if "cpn" not in X.J:
            continue
        est, c, lo, pv = X.paired("cpn", "source", "same")
        rows.append([DN[d], "same (all references share the query's class)", "acc", f"{est:+.2f}", f"{lo:+.2f}", "yes" if lo > -2 else "no"])
        res.setdefault("safety", {}).setdefault(d, {})["same"] = dict(delta=est, lower=lo)
        if d in nat:
            for p in [x for x in ["rand", "past"] if x in X.protos]:
                est, c, lo, pv = X.paired_macro("cpn", "source", p, "uar")
                rows.append([DN[d], p, "UAR", f"{est:+.2f}", f"{lo:+.2f}", "yes" if lo > -1 else "no"])
                res["safety"][d][p] = dict(delta=est, lower=lo)
    out += table(["Dataset", "protocol", "metric", "CPN − Source", "one-sided 95% lower bound", "non-inferior"], rows) + [""]
    rows = []
    for d in real:
        X = D[d]
        if "cpn" not in X.J or "bn" not in X.J:
            continue
        est, c, lo, pv = X.paired("cpn", "bn", "rand")
        rows.append([DN[d], ci(est, c), f"{lo:+.2f}", "yes" if lo > -1.5 else "no"])
        res.setdefault("rand_vs_bn", {})[d] = dict(delta=est, ci=list(c), lower=lo)
    out += ["Random references: CPN minus BN-adapt (accuracy; non-inferiority margin 1.5 points)"] + \
        table(["Dataset", "Δ [95% CI]", "one-sided 95% lower", "non-inferior"], rows) + [""]
    rows = []
    for lab_, ds_, m2, pr, mg in (("lab sets, random references, vs BN-adapt", [d for d in R.LAB if d in real], "bn", "rand", 1.5),
                                  ("all real sets, hard protocol, vs Source", real, "source", "same", 2.0),
                                  ("natural + external sets, random references, vs BN-adapt", [d for d in real if d in nat], "bn", "rand", 1.5)):
        ds_ = [d for d in ds_ if "cpn" in D[d].J and m2 in D[d].J and D[d].has("cpn", pr)]
        if not ds_:
            continue
        est, c, lo = pooled_paired([D[d] for d in ds_], "cpn", m2, pr)
        rows.append([lab_, len(ds_), ci(est, c), f"{lo:+.2f}", f"{mg:g}", "yes" if lo > -mg else "no"])
        res.setdefault("pooled_endpoints", {})[lab_] = dict(datasets=ds_, delta=est, ci=list(c), lower=lo, margin=mg)
    out += ["Pre-specified pooled endpoints (round 10b): mean over datasets of the CPN − reference accuracy difference "
            "(equal dataset weights), dataset-stratified identity bootstrap"] + \
        table(["Endpoint", "#datasets", "Δ [95% CI]", "one-sided 95% lower", "margin", "non-inferior"], rows) + [""]
    rows = []
    for d in real:
        meta = pd.read_csv(f"{CM.PROC}/{d}_meta.csv")
        ev = np.isin(meta.fold.values, CM.test_folds(meta))
        cell = meta.groupby(["actor", "label"]).label.transform("size").values - 1  # other clips, same person and class
        sh = float(np.mean(cell[ev] < KCTX)) * 100
        rows.append([DN[d], f"{sh:.1f}%", f"{np.median(cell[ev]):.0f}"])
        res.setdefault("same_repeats", {})[d] = sh
    out += [f"Hard-protocol reference sets: share of queries with fewer than K = {KCTX} other clips of the same person and "
            "class (their references repeat clips; identical for every method), and the median number of such clips"] + \
        table(["Dataset", "queries with repeated references", "median same-class clips"], rows) + [""]
    # ------------------------------------------------------------------ 5. uncertainty calibration and failure prediction
    out.append("## Table 5. Is the posterior's uncertainty trustworthy?  Calibration of the predicted centre error, the centre "
               "error of the three read-outs, and prediction of adaptation failure (test people; all reference protocols pooled)")
    rows, rows2, rows3 = [], [], []
    diag = [("err_pred", "predicted centre error tr Σ_err (ours)", 1), ("shrink", "EB shrinkage (ours)", 1),
            ("H_N", "composition-posterior entropy", 1), ("pl_var", "independent pseudo-label variance", 1),
            ("ent_src", "Source entropy of the query", 1), ("maxp", "1 − max probability", -1)]

    def score(jb, k_, p, shape):
        v = jb.EP[k_][p] if k_ in jb.EP else np.full(shape, np.nan)
        return (1 - v) if k_ == "maxp" else v

    for d in real + [s_ for s_ in R.SYN if s_ in D]:
        X = D[d]
        jb = X.J.get("cpn")
        if jb is None or "z2" not in jb.EP:
            continue
        prs = [p for p in jb.EP["z2"] if p != "K0" and p in X.readout("cpn")]
        cat = lambda k_: np.concatenate([jb.EP[k_][p].ravel() for p in prs]).astype(float)  # noqa: E731
        z2, er, ep_ = cat("z2"), cat("err_real"), cat("err_pred")
        k = X.C - 1
        valid_z2 = z2[np.isfinite(z2)]
        cov = float(np.mean(valid_z2 <= CM.chi2_90(k))) if len(valid_z2) else float("nan")
        xs = list(jb.xfit.values())
        byp = {}
        for grp_, gp_ in (("label-free", [p for p in prs if p in FREE]), ("same", [p for p in prs if p == "same"])):
            if gp_:
                zz = np.concatenate([jb.EP["z2"][p].ravel() for p in gp_]).astype(float)
                zz = zz[np.isfinite(zz)]
                byp[grp_] = (float(zz.mean() / k), float(np.mean(zz <= CM.chi2_90(k)))) if len(zz) else (np.nan, np.nan)
        rows.append([DN[d], f2(np.nanmean(z2) / k), f2(cov)] +
                    [f"{f2(byp[g_][0])} / {f2(byp[g_][1])}" if g_ in byp else "—" for g_ in ("label-free", "same")] +
                    [f2(spearman(ep_, er), 3)])
        rec = res.setdefault("calibration", {}).setdefault(d, {})
        rec.update(mean_z2=float(np.nanmean(z2) / k), cover90=cov, by_protocol=byp, spearman=spearman(ep_, er))
        if "err_src" in jb.EP:  # centre error of: full adaptation (posterior mean), no adaptation, CPN; per protocol group
            grp = {"rand / skew / dir02 / past": [p for p in prs if p in FREE], "same": [p for p in prs if p == "same"],
                   "composition curve": [p for p in prs if p in MIX]}
            cells = []
            for gname, gp in grp.items():
                if not gp:
                    cells.append("—"); continue
                g = {k_: np.concatenate([jb.EP[k_][p].ravel() for p in gp]).astype(float) for k_ in ("err_real", "err_src", "err_cpn")}
                rec.setdefault("centre_error", {})[gname] = {k_: float(np.nanmean(v)) for k_,v in g.items()}
                cells.append(" / ".join(f2(np.nanmean(g[k_]), 2) for k_ in ("err_real", "err_src", "err_cpn")))
            rows3.append([DN[d]] + cells)
            harm = (cat("err_real") > cat("err_src")).astype(float)  # adapting moved the centre AWAY from the truth
            harm[~np.isfinite(cat("err_real"))] = np.nan
            shape = jb.EP["z2"][prs[0]].shape
            au = {k_: auroc(np.concatenate([score(jb, k_, p, shape).ravel() for p in prs]).astype(float), harm) for k_, _, _ in diag}
            sp = {k_: spearman(np.concatenate([score(jb, k_, p, shape).ravel() for p in prs]).astype(float), er) for k_, _, _ in diag}
            rows2.append([DN[d], f"{100 * np.nanmean(harm):.1f}%"] + [f"{f2(au[k_], 3)} ({f2(sp[k_], 2)})" for k_, _, _ in diag])
            rec.update(auroc_centre_harm=au, spearman_error=sp)
    out += table(["Dataset", "mean z²/(C−1) (1 = calibrated)", "90% ellipsoid coverage (0.90)",
                  "label-free z²/(C−1) / coverage", "same z²/(C−1) / coverage", "Spearman(predicted, realised error)"], rows)
    out += ["Realised error: the posterior-mean centre minus the person's balanced centre estimated from all their labelled clips "
            "(synthetic data: the exact centre).  The model's own error covariance, without calibration factors (round 10); "
            "an analysis of the conservative branch, not a component claim.", ""]
    if rows3:
        out += ["Mean squared centre error in span(F): full adaptation (posterior mean) / no adaptation (source centre) / CPN "
                "(empirical diagnostic; the conditional scalar risk bound does not automatically apply to fitted beta or data-dependent weights)"] +             table(["Dataset", "label-free protocols", "same", "composition curve"], rows3) + [""]
    if rows2:
        out += ["Which diagnostic tells that adaptation is unsafe?  Target: full adaptation moves the centre AWAY from the truth "
                "(its error exceeds the source centre's).  AUROC (Spearman with the realised error in brackets)"] +             table(["Dataset", "harm rate"] + [t for _, t, _ in diag], rows2) + [""]
    # ------------------------------------------------------------------ 5b. calibration of the composition posterior
    rows = []
    for d in real + [s_ for s_ in R.SYN if s_ in D]:
        X = D[d]
        cells = []
        for m in ("cpn_nopostcal", "cpn"):
            jb = X.J.get(m)
            if jb is None or "lp_true" not in jb.EP:
                cells.append("—"); continue
            v = {g_: np.concatenate([jb.EP["lp_true"][p][:1].ravel() for p in gp_]).astype(float)
                 for g_, gp_ in (("label-free", [p for p in jb.EP["lp_true"] if p in FREE]), ("same", ["same"] if "same" in jb.EP["lp_true"] else []))
                 if gp_}
            cells.append(" / ".join(f2(np.nanmean(x)) for x in v.values()))
            res.setdefault("posterior", {}).setdefault(d, {})[m] = {g_: float(np.nanmean(x)) for g_, x in v.items()}
        jb = X.J.get("cpn")
        pw = ""
        if jb is not None:
            ws = [x.get("post_w") for x in jb.xfit.values() if x.get("post_w")]
            if ws:
                pw = "/".join(f2(np.mean([w[i_] for w in ws])) for i_ in range(3))
                ho = [x.get("posterior_calibration", {}) for x in jb.xfit.values()]
                hb = np.mean([h_["post_ll_pooled_before"] for h_ in ho if h_]); ha = np.mean([h_["post_ll_pooled_after"] for h_ in ho if h_])
                pw += f" (held-out {hb:.2f} -> {ha:.2f})"
        rows.append([DN[d]] + cells + [pw or "—"])
    out += ["## Table 5b. Calibration of the composition posterior: mean log posterior probability of the TRUE batch "
            "composition on test people (label-free / same; higher is better), uncalibrated vs calibrated posterior, and the "
            "held-out conditional-maximum-likelihood weights (t_net / t_dens / t_anc)"] + \
        table(["Dataset", "uncalibrated posterior", "calibrated posterior (ours)", "weights (held-out pooled log-lik before -> after)"], rows) + [""]
    # ------------------------------------------------------------------ 5c. the known-neutral setting (SFDA-PFT)
    rows = []
    for d in real:
        X = D[d]
        if "neu6" not in X.protos and not any("neu6" in (X.readout(m) or {}) for m in X.J):
            continue
        meta = pd.read_csv(f"{CM.PROC}/{d}_meta.csv")
        nl = CM.neutral(d)
        if nl is None:
            continue
        ev = np.isin(meta.fold.values, CM.test_folds(meta))
        has_n = meta.groupby("actor").label.transform(lambda s_: (s_ == nl).sum()).values
        keep = ((has_n - (meta.label.values == nl)) >= 1)[ev]  # another neutral clip of the person exists
        for m in ("source", "bn", "neucal", "cpn", "cpn_oracle"):
            P = X.readout(m)
            if P is None or "neu6" not in P:
                continue
            q = P["neu6"][:, keep]
            yk = X.y[keep]
            accs = [metric(q_, yk, X.C, "acc") for q_ in q]; uars = [metric(q_, yk, X.C, "uar") for q_ in q]
            rows.append([DN[d], name(m), f2(np.mean(accs)), f2(np.mean(uars))])
            res.setdefault("neu6", {}).setdefault(d, {})[m] = dict(acc=float(np.mean(accs)), uar=float(np.mean(uars)))
    if rows:
        out += ["## Table 5c. The known-neutral reference setting of SFDA-PFT (all 6 references are the person's neutral clips): "
                "'known-neutral references' is TOLD that they are neutral (the information of SFDA-PFT), CPN is not, the label "
                "oracle is told all reference labels.  People without another neutral clip are excluded."] + \
            table(["Dataset", "Method", "acc", "UAR"], rows) + [""]
    # ------------------------------------------------------------------ 5d. combiners of the SAME three branches
    rows = []
    for d in real:
        X = D[d]
        if "cpn" not in X.J:
            continue
        for m, nm, lab in COMBINERS:
            if not X.has(m, "rand", nm):
                continue
            cells = [f2(X.val(m, "rand", nm=nm, draws=1)), f2(X.val(m, "same", nm=nm, draws=1)), f2(X.val(m, "cavg", nm=nm, draws=1))]
            if d in nat:
                cells.append(f2(X.val(m, "rand", "uar", nm=nm, draws=1)))
                cells.append(f2(X.val(m, "past", "uar", nm=nm, draws=1)) if X.has(m, "past", nm) else "—")
            else:
                cells += ["—", "—"]
            rows.append([DN[d], lab] + cells)
            res.setdefault("combiners", {}).setdefault(d, {})[lab] = cells
    out += ["## Table 5d. How the four predictive distributions are combined: identifiability stacking vs constant "
            "stacking, episode gates (concentration, entropy, learned) and fixed weighting -- identical branches, identical "
            "held-out people and objective (matched first draw)"] + \
        table(["Dataset", "combiner", "rand acc", "same acc", "cavg acc", "rand UAR", "past UAR"], rows) + [""]
    rows = []
    for d in real:
        jb = D[d].J.get("cpn")
        if jb is None:
            continue
        cur = [x.get("stacking", {}).get("trust_curve") for x in jb.xfit.values()]
        cur = [c for c in cur if c]
        if not cur:
            continue
        W = np.mean([np.array(c["w"]) for c in cur], 0)  # (11, 4): w_B, w_E, w_M, w_S over g = 0, 0.1, .., 1
        gg = cur[0]["g"]
        res.setdefault("trust_curve", {})[d] = dict(g=gg, w=W.tolist())
        rows.append([DN[d]] + [" / ".join(f"{v:.2f}" for v in W[i]) for i in (0, 2, 5, 8, 10)])
    if rows:
        out += ["## Table 5e. The learned trust curve (mean over folds): vote shares (exact Bayes / conservative / batch "
                "moments / Source) "
                "of a composition hypothesis as a function of its identifiability g"] + \
            table(["Dataset", "g = 0", "g = 0.2", "g = 0.5", "g = 0.8", "g = 1"], rows) + [""]
    rows = []
    for d in real:
        X = D[d]
        if "cpn" not in X.J:
            continue
        for p in [x for x in ["rand", "past", "same"] if X.has("cpn", x)]:
            a_, b_ = X.corr("cpn", p), X.corr("source", p)
            dl = 100 * (a_ - b_) / X.sizes
            rows.append([DN[d], p, f2(np.percentile(dl, 10)), f2(np.median(dl)), f"{100 * np.mean(dl < -5):.1f}%",
                         f"{100 * np.mean(dl > 5):.1f}%", len(dl)])
            res.setdefault("per_person", {}).setdefault(d, {})[p] = dict(p10=float(np.percentile(dl, 10)), median=float(np.median(dl)),
                                                                     worse5=float(np.mean(dl < -5)), better5=float(np.mean(dl > 5)))
    out += ["## Table 5f. Per-person accuracy change CPN − Source (worst-identity view): 10th percentile, median, share of "
            "people losing / gaining more than 5 points"] + \
        table(["Dataset", "protocol", "10th percentile", "median", "people < −5", "people > +5", "#people"], rows) + [""]
    # ------------------------------------------------------------------ 6. per-class recall on natural data
    out.append("## Table 6. Per-class recall on dialogue/speech data (random references; matched first draw)")
    for d in nat:
        X = D[d]
        cls = CLASSES.get(d, [f"c{k}" for k in range(X.C)])
        rows = []
        for m, nm, lab in [("source", None, "Source"), ("bn", None, "BN-adapt"), ("cpn7", None, "CPN round 7"),
                           ("cpn", "alt:wg", "round-9 weighting"), ("cpn_std", None, "CPN - transport"),
                           ("cpn", None, "CPN (ours)")]:
            P = X.readout(m, nm)
            if P is None or "rand" not in P:
                continue
            pred = P["rand"][0].argmax(-1)
            rec = [100 * (pred[X.y == c] == c).mean() for c in range(X.C)]
            share = [100 * (pred == c).mean() for c in range(X.C)]
            res.setdefault("recall", {}).setdefault(d, {})[m + (":" + nm if nm else "")] = dict(classes=cls, recall=rec, share=share, uar=float(np.mean(rec)), draws=1)
            rows.append([lab] + [f"{r:.1f} ({s:.0f}%)" for r, s in zip(rec, share)] + [f2(np.mean(rec))])
        cnt = np.bincount(X.y, minlength=X.C)
        out += [f"**{DN[d]}** recall (share of predictions); class frequency " + ", ".join(f"{c}: {100 * n / cnt.sum():.0f}%" for c, n in zip(cls, cnt))] + \
            table(["Method"] + cls + ["UAR"], rows) + [""]
    # ------------------------------------------------------------------ 7. stream
    stream_tables(real, out, res)
    # ------------------------------------------------------------------ 8. efficiency
    rows = []
    for d, f in R.PROFILE.items():
        pf = R.OUT / f"profile_{d}_{f}_iid.json"
        if not pf.exists():
            continue
        p = json.loads(pf.read_text())
        assert p["code_sha256"] == SHA, "timing code version mismatch"
        res.setdefault("profile", {})[d] = p["rows"]
        for r in p["rows"]:
            rows.append([DN[d], p["C"], r["K"], name(r["method"]) if r["method"] in LAB else r["method"], f2(r["online_ms"]),
                         f2(r["batch_ms_per_episode"], 3), f"{r['peak_mb_per_batch']:.0f}", r["hypotheses"] or "—", f2(r.get("acc"))])
    if rows:
        out += ["## Table 8. Accuracy, latency and memory vs number of references K, classes C and pruning level M "
                "(one fold; rand references; excludes feature extraction and offline training/calibration; "
                "cpn_pM = two-stage inference with the exact evidence for the M retained compositions; stage one still scores the full lattice)"] + \
            table(["Dataset", "C", "K", "method", "online ms / episode", "batched ms / episode", "peak MB", "stage-two compositions", "acc"], rows) + [""]
    # ------------------------------------------------------------------ 9. second representation
    rows = []
    for d, f in R.REPR.items():
        X = Data(d, [j for j in jobs if j["section"] == "repr" and j["feat"] == f], section=("repr",))
        if X.subj is None:
            continue
        for m in R.REPR_M:
            if m not in X.J:
                continue
            if d in nat:
                rows.append([f"{DN[d]} [{f}]", name(m), f2(X.val(m, "rand")), f2(X.val(m, "rand", "uar")), f2(X.val(m, "same")), f2(X.val(m, "cavg"))])
            else:
                rows.append([f"{DN[d]} [{f}]", name(m), f2(X.val(m, "rand")), "—", f2(X.val(m, "same")), f2(X.val(m, "cavg"))])
            res.setdefault("repr", {}).setdefault(d, {})[m] = {pr: X.val(m, pr) for pr in ["rand", "same", "cavg"]}
    if rows:
        out += ["## Table 9. A second representation (CLIP ViT-B/16 for the lab sets, face-only CLIP-L/14 for MELD; 1 draw)"] + \
            table(["Dataset [features]", "Method", "rand acc", "rand UAR", "same acc", "cavg acc"], rows) + [""]
    # ------------------------------------------------------------------ 10. synthetic data with exact centres
    syn = [d for d in R.SYN if d in D]
    if syn:
        out.append("## Table 10. Synthetic data with exact person centres (one factor at a time; rand / dir02 / same / cavg accuracy)")
        rows = []
        for d in syn:
            X = D[d]
            rows.append([d, ", ".join(f"{k}={v}" for k, v in R.SYN[d].items())] +
                        [" / ".join(f2(X.val(m, p), 1) for p in ["rand", "dir02", "same", "cavg"]) if m in X.J else "—" for m in R.SYN_M])
        out += table(["data", "regime"] + [name(m) for m in R.SYN_M], rows) + [""]
    # ------------------------------------------------------------------ 11. paired tests vs every baseline
    out.append("## Table 11. CPN minus each baseline (accuracy; identity bootstrap 95% CI; sign-flip p, Holm within dataset)")
    for d in real:
        X = D[d]
        if "cpn" not in X.J:
            continue
        rows, ps = [], []
        for m in BASE:
            if m not in X.J:
                continue
            for p in [x for x in ["rand", "skew", "dir02", "past", "same", "cavg"] if X.has(m, x) and X.has("cpn", x)]:
                est, c, lo, pv = X.paired("cpn", m, p)
                rows.append([name(m), p, f"{est:+.2f}", f"[{c[0]:+.2f}, {c[1]:+.2f}]", pv]); ps.append(pv)
        hp = holm(ps) if ps else []
        out += [f"**{DN[d]}**"] + table(["baseline", "protocol", "Δ", "95% CI", "Holm p"], [r[:4] + [f2(h, 4)] for r, h in zip(rows, hp)]) + [""]
    # ------------------------------------------------------------------ cross-fit constants
    rows = []
    for d in real:
        jb = D[d].J.get("cpn")
        if jb is None:
            continue
        xs = list(jb.xfit.values())
        g = lambda k: f2(np.nanmean([float(x.get(k, np.nan)) for x in xs]))  # noqa: E731
        al = [x.get("alpha_p") for x in xs]
        pw = [x.get("post_w") for x in xs if x.get("post_w")]
        sp = [x.get("stack_params", {}).get("stack", {}).get("theta") for x in xs]
        sp = [v for v in sp if v]
        rows.append([DN[d], "/".join(f2(np.mean([w[i_] for w in pw])) for i_ in range(3)) if pw else "—",
                     "/".join(f2(np.mean([v[i_] for v in sp])) for i_ in range(len(sp[0]))) if sp else "—",
                     g("kappa"), g("tau_tp"), g("tau_src"), g("tau_bn"), g("aux_heldout_source_acc"), g("aux_heldout_bn_acc"),
                     g("aux_heldout_truecomp_tp_acc"),
                     f"{sum(any(x.get('posterior_calibration', {}).get('at_edge', [])) for x in xs)}/{len(xs)}"])
    out += ["## Table 12. Cross-fitted constants per dataset (held-out training people; mean over folds)"] + \
        table(["Dataset", "posterior weights t_net / t_dens / t_anc", "stacking a_E / b_E / a_M / b_M / a_S / b_S", "κ",
               "τ (transport)", "τ (source)", "τ (batch moments)", "held-out Source acc", "held-out BN acc",
               "held-out true-composition transport acc", "folds with a posterior weight at the search edge"], rows) + [""]
    PAPER.mkdir(exist_ok=True)
    res["_meta"] = dict(code_sha256=SHA, missing=missing, jobs=len(jobs), manifest=man)
    res["_summary"] = {}
    for d, X in D.items():
        dest = res["_summary"].setdefault(d, {})
        for m, jb in X.J.items():
            nms = [jb.main_name()] + ([k for k in ALT if k in jb.P] if m == "cpn" else [])
            for nm in nms:
                key = m if nm == jb.main_name() else m + ":" + nm
                dest[key] = {pr: {kind: X.val(m, pr, kind, nm) for kind in ["acc", "uar", "f1"]}
                             for pr in list(jb.P[nm]) + (["cavg"] if all(x in jb.P[nm] for x in MIX) else [])}
                dest[key]["draw0"] = {pr: {kind: X.val(m, pr, kind, nm, draws=1) for kind in ["acc", "uar", "f1"]}
                                      for pr in list(jb.P[nm]) + (["cavg"] if all(x in jb.P[nm] for x in MIX) else [])}
    res["_tables"] = "\n".join(out)
    (PAPER / "cpn10_tables.md").write_text("\n".join(out), encoding="utf-8")
    (PAPER / "cpn10_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print("wrote", PAPER / "cpn10_tables.md")


if __name__ == "__main__":
    main()
