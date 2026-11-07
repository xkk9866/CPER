"""Round-10 main-paper study of CPN (manifest-driven, restartable, version-checked, local GPU, single training seed).

    python run_cpn.py --list                                         # job counts, manifest, jobs_cpn.txt
    python run_cpn.py --preflight                                    # inputs, folds, selftest, GPU, disk (~1 min)
    python run_cpn.py --suites main,ablation,synth,stream,repr --workers 5 --mem-frac 0.12
    python run_cpn.py --suites profile --workers 1                   # timing: ALONE on the GPU

Suites (frozen encoders, all subject folds, training seed 0; reference draws are paired across methods):
  main      9 datasets x 14 methods (+ the known-neutral SFDA-PFT-style reference 'neucal' where a neutral class exists).
            Lab: RAVDESS, CREMA-D, MEAD, eNTERFACE'05 (CLIP-L/14 face).  Development corpora: MELD, ShEMO.  External:
            Emozionalmente (first evaluated in round 8), THAI-SER Zoom with SESSION-isolated folds (first evaluated in
            round 9), and THAI-SER Studio under a FROZEN model trained on the Zoom recordings only (cross-condition;
            downloaded in round 10, never evaluated before the formal run).  Protocols: rand, skew, dir02, (past), same,
            neu6, composition curve j = 0..6; 3 reference draws.
  ablation  7 variants on all 9 datasets, 1 draw: calibrated posterior off, independent pseudo-labels, density, anchor,
            batch-mode evidence, transport off, label oracle; every CPN variant re-fits its own posterior weights and
            stacking combiner on the cached auxiliary network.  The branch / combiner controls (each of the four branches
            alone, the round-9 fixed weighting, stacking without batch moments / without source / without the
            identifiability covariate, episode gates on concentration / Source entropy / five learned features, MAP
            composition) come from the main CPN run itself.
  synth     exact person centres: 5 regimes (base, interaction low/high, identity offset low/high); Source, BN, CPN, label oracle.
  stream    person-blocked test stream, every dataset.
  repr      CLIP-B/16 for the four lab sets, face-only CLIP-L/14 for MELD; 6 methods.
  profile   latency / memory / accuracy vs K and pruning level on RAVDESS (C=8), CREMA-D (C=6), ShEMO (C=5); run alone.
Outputs: $CMN_OUT (default $EMO_ROOT/runs/cpn10_frozen): json, pred/, logs/, manifest.json, status.json."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, concurrent.futures, hashlib, json, os, pathlib, subprocess, sys, threading, time

ROOT = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(os.environ.get("CMN_OUT", f"{_data_root}/runs/cpn10_frozen"))
PROC = pathlib.Path(f"{_data_root}/datasets/proc")
FEATD = pathlib.Path(f"{_data_root}/features/clip")
LAB = ["ravdess", "cremad", "mead", "enterface"]
NAT = {"meld": "avt", "shemo": "wavlm", "emoz": "wavlm", "thai": "wavlm", "thaix": "wavlm"}
FEAT = {**{d: "l14" for d in LAB}, **NAT}
DATA = LAB + list(NAT)
EXTERNAL = ["emoz", "thai", "thaix"]  # first evaluated in round 8 / round 9 / never (cross-condition, frozen model)
WITH_NEUTRAL = [d for d in DATA if d != "enterface"]
MAIN = ["source", "bn", "tent_grid", "sar_grid", "laem_grid", "note", "rotta_grid", "csn_grid", "dart", "lame", "t3a",
        "srcbn_grid", "bnprior_grid", "cpn"]
ABL = ["cpn_nopostcal", "pl_shrink", "cpn_nodens", "cpn_noanchor", "cpn_bnlik", "cpn_std", "cpn_oracle"]
REPR = {**{d: "b16" for d in LAB}, "meld": "l14"}
REPR_M = ["source", "bn", "rotta_grid", "csn_grid", "srcbn_grid", "cpn"]
SYN_BASE = {"--s_G": "0.12", "--lam_sd": "0.15", "--s_F": "0.25", "--s_e": "1.2"}
SYN_FACT = {"G": ("--s_G", "0.03", "0.25"), "E": ("--s_e", "0.6", "2.0")}
PROFILE = {"ravdess": "l14", "cremad": "l14", "shemo": "wavlm"}
PROFILE_METHODS = ["bn", "tent", "laem", "rotta", "dart", "cpn", "cpn_p64", "cpn_p256"]


def syn_sets():
    out = {"synth_o_base": dict(SYN_BASE)}
    for k, (arg, lo, hi) in SYN_FACT.items():
        for tag, v in (("lo", lo), ("hi", hi)):
            out[f"synth_o_{k}{tag}"] = {**SYN_BASE, arg: v}
    return out


SYN = syn_sets()
SYN_M = ["source", "bn", "cpn", "cpn_oracle"]


def J(section, data, method, nctx, suite, feat=None):
    return dict(section=section, data=data, method=method, seed=0, feat=feat or FEAT.get(data, "raw"), net="iid",
                nctx=nctx, suite=suite)


def suites():
    return {"main": [J("main", d, m, 3, "lite") for m in MAIN for d in DATA]
                    + [J("main", d, "neucal", 3, "lite") for d in WITH_NEUTRAL],
            "ablation": [J("ablation", d, m, 1, "lite") for m in ABL for d in DATA],
            "synth": [J("synth", d, m, 1, "lite", "raw") for d in SYN for m in SYN_M],
            "stream": [J("stream", d, "stream", 1, "lite") for d in DATA],
            "repr": [J("repr", d, m, 1, "lite", f) for d, f in REPR.items() for m in REPR_M],
            "profile": [dict(J("profile", d, "profile", 1, "lite", f), profile_k=[1, 2, 4, 6, 8],
                             profile_methods=PROFILE_METHODS) for d, f in PROFILE.items()]}


SMOKE = ["--folds", "0", "--steps", "300", "--nctx", "1"] if os.environ.get("CMN_SMOKE") else []  # pipeline test only


def argv(j):
    if j["method"] == "profile":
        return (["--data", j["data"], "--profile", "--feat", j["feat"], "--net", j["net"], "--profile-k"]
                + list(map(str, j["profile_k"])) + ["--profile-methods"] + j["profile_methods"] + SMOKE[2:4])
    if j["method"] == "stream":
        return ["--data", j["data"], "--stream", "--feat", j["feat"], "--net", j["net"], "--seed", str(j["seed"])] + SMOKE
    a = ["--data", j["data"], "--method", j["method"], "--seed", str(j["seed"]), "--feat", j["feat"], "--net", j["net"],
         "--suite", j["suite"], "--nctx", str(j["nctx"])]
    return a + SMOKE


_CM, _LOCK = None, threading.Lock()


def stem(j):
    """output stem of a job, from cmn.py's own tag rule (loaded once, thread-safe)"""
    global _CM
    if j["method"] == "profile":
        return f"profile_{j['data']}_{j['feat']}_{j['net']}"
    if j["method"] == "stream":
        return f"stream_{j['data']}_{j['feat']}_s{j['seed']}"
    with _LOCK:
        if _CM is None:
            import importlib.util
            spec = importlib.util.spec_from_file_location("cmn_tag", ROOT / "cmn.py")
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            _CM = m
    return f"{j['data']}_{_CM.get_args(argv(j)).tag}_s{j['seed']}"


def ensure_synth(names):
    for name in names:
        if not (PROC / f"{name}_meta.csv").exists():
            extra = [x for kv in SYN[name].items() for x in kv] + ["--person_scale"]
            subprocess.run([sys.executable, str(ROOT / "synth_data.py"), "--name", name] + extra, check=True, cwd=ROOT)


def check_data(jobs):
    miss = sorted({f"{FEATD / (j['data'] + '_' + j['feat'] + '.npy')}" for j in jobs if j["section"] != "synth"
                   and not (FEATD / f"{j['data']}_{j['feat']}.npy").exists()}
                  | {str(PROC / f"{j['data']}_meta.csv") for j in jobs if j["section"] != "synth"
                     and not (PROC / f"{j['data']}_meta.csv").exists()})
    if miss:
        raise SystemExit("missing inputs (run prep_natural.py --meld / --shemo / --emoz / --thai / --thaix first):\n  " + "\n  ".join(miss))


def preflight(jobs, a):
    """everything a long run can fail on, checked in about a minute BEFORE any job starts: inputs, row counts, folds
    (complete, person-disjoint, session-isolated where sessions exist, training-only pool disjoint from the test
    people), classes per fold, the cmn.py selftest, GPU and disk"""
    import numpy as np, pandas as pd, shutil
    errs, notes = [], []
    for d in sorted({j["data"] for j in jobs if j["section"] != "synth"}):
        feats = sorted({j["feat"] for j in jobs if j["data"] == d})
        mp = PROC / f"{d}_meta.csv"
        if not mp.exists():
            errs.append(f"{d}: missing {mp}"); continue
        meta = pd.read_csv(mp)
        for ft in feats:
            fp = FEATD / f"{d}_{ft}.npy"
            if not fp.exists():
                errs.append(f"{d}: missing {fp}"); continue
            n = np.load(fp, mmap_mode="r").shape[0]
            if n != len(meta):
                errs.append(f"{d}: {fp.name} has {n} rows, meta has {len(meta)}")
        tf = sorted(int(f) for f in meta.fold.unique() if f >= 0)
        if len(tf) < 2:
            errs.append(f"{d}: fewer than two test folds {tf}")
        af = meta.groupby("actor").fold.nunique()
        if (af > 1).any():
            errs.append(f"{d}: {int((af > 1).sum())} people in more than one fold")
        if "session" in meta:
            sf = meta[meta.fold >= 0].groupby("session").fold.nunique()
            if (sf > 1).any():
                errs.append(f"{d}: {int((sf > 1).sum())} recording sessions shared by test folds")
        if (meta.fold < 0).any():
            pool, test = set(meta.actor[meta.fold < 0]), set(meta.actor[meta.fold >= 0])
            if pool & test:
                errs.append(f"{d}: {len(pool & test)} training-pool people also in the test folds")
            notes.append(f"{d}: cross-condition (training-only pool of {len(pool)} people, {len(test)} test people)")
        C = meta.label.nunique()
        for f in tf:
            te = meta[meta.fold == f]
            if te.label.nunique() < C:
                notes.append(f"{d}: fold {f} lacks {C - te.label.nunique()} class(es)")
            if (te.groupby("actor").size() < 2).any():
                errs.append(f"{d}: fold {f} has people with a single clip (no reference set)")
        notes.append(f"{d}: {len(meta)} clips, {meta.actor.nunique()} people, {C} classes, test folds {tf}")
    r = subprocess.run([sys.executable, str(ROOT / "cmn.py"), "--selftest"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or "selftest passed" not in r.stdout:
        errs.append("cmn.py --selftest failed:\n" + (r.stdout + r.stderr)[-3000:])
    # GPU check in a child process: the runner itself must not hold a CUDA context during the run
    code = ("import torch,json;f,t=torch.cuda.mem_get_info() if torch.cuda.is_available() else (0,0);"
            "print(json.dumps(dict(ok=torch.cuda.is_available(),free=f,tot=t,"
            "name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')))")
    g = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        gi = json.loads(g.stdout.strip().splitlines()[-1])
        if not gi["ok"]:
            errs.append("no CUDA device")
        else:
            need = a.workers * a.mem_frac * gi["tot"]
            notes.append(f"GPU {gi['name']}: {gi['free'] / 2 ** 30:.1f} of {gi['tot'] / 2 ** 30:.1f} GB free, "
                         f"{a.workers} workers x {a.mem_frac} need {need / 2 ** 30:.1f} GB")
            if gi["free"] < need:
                errs.append("not enough free GPU memory for workers x mem-frac (close other GPU jobs)")
    except Exception as e:  # noqa: BLE001
        errs.append(f"torch / CUDA check failed: {e} {g.stderr[-500:]}")
    OUT.mkdir(parents=True, exist_ok=True)
    fr = shutil.disk_usage(OUT).free / 2 ** 30
    notes.append(f"free disk at {OUT}: {fr:.0f} GB")
    if fr < 30:
        errs.append(f"less than 30 GB free at {OUT}")
    for x in notes:
        print("  ", x)
    if errs:
        raise SystemExit("PREFLIGHT FAILED:\n  " + "\n  ".join(errs))
    print("preflight passed", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="main,ablation,synth,stream,repr")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--mem-frac", type=float, default=0.12)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--preflight", action="store_true", help="only the pre-run checks")
    ap.add_argument("--no-preflight", action="store_true", help="skip the automatic pre-run checks")
    ap.add_argument("--retries", type=int, default=1, help="re-runs of a failed job (transient GPU errors)")
    ap.add_argument("--only-data", default="", help="comma-separated subset of datasets (partial runs / tests)")
    ap.add_argument("--only-methods", default="", help="comma-separated subset of methods (partial runs / tests)")
    a = ap.parse_args()
    if a.workers < 1 or not 0 < a.mem_frac <= 0.9 or a.workers * a.mem_frac > 0.9:
        ap.error("workers must be positive and workers * mem-frac must be <= 0.9")
    S = suites()
    sel = [x for x in a.suites.split(",") if x]
    jobs = [j for x in sel for j in S[x]]
    if a.only_data:
        jobs = [j for j in jobs if j["data"] in a.only_data.split(",")]
    if a.only_methods:
        jobs = [j for j in jobs if j["method"] in a.only_methods.split(",")]
    sha = hashlib.sha256((ROOT / "cmn.py").read_bytes()).hexdigest()[:16]
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "logs").mkdir(exist_ok=True)
    man_fn = OUT / "manifest.json"
    man = json.loads(man_fn.read_text()) if man_fn.exists() else {}
    done_ = [q for q in OUT.glob("*.json") if q.name not in ("manifest.json", "status.json")]
    if man.get("code_sha256") not in (None, sha) and done_:  # a manifest without results (e.g. --list) is refreshed
        raise SystemExit(f"manifest was written by cmn.py {man['code_sha256']}, current code is {sha}: use a new CMN_OUT "
                         "or delete the old results (never mix code versions)")
    allj = [j for x in S for j in S[x]]
    man = dict(code_sha256=sha, seeds=[0], context_draws={"main": 3, "ablation": 1, "synth": 1, "repr": 1}, steps=3000,
               suites={k: len(v) for k, v in S.items()}, jobs=allj, synth=SYN, external=EXTERNAL,
               created=man.get("created", time.strftime("%Y-%m-%d %H:%M:%S")))
    man_fn.write_text(json.dumps(man, indent=1), encoding="utf-8")
    with open(OUT / "jobs_cpn.txt", "w", newline="\n") as fo:
        fo.write("\n".join(" ".join(argv(j)) for j in allj) + "\n")
    print({k: len(v) for k, v in S.items()}, "selected:", len(jobs), "code", sha, flush=True)
    if a.list:
        return
    check_data(jobs)
    if a.preflight or not a.no_preflight:
        preflight(jobs, a)
        if a.preflight:
            return
    ensure_synth(sorted({j["data"] for j in jobs if j["section"] == "synth"}))
    env = os.environ.copy()
    env.update(CMN_OUT=str(OUT), MEM_FRAC=str(a.mem_frac), CMN_CPU_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONIOENCODING="utf-8", CUDA_VISIBLE_DEVICES="0", TEMP=f"{_data_root}/tmp", TMP=f"{_data_root}/tmp",
               HF_HOME=f"{_data_root}/cache/hf")
    os.makedirs(f"{_data_root}/tmp", exist_ok=True)

    def run(j):
        st = stem(j)
        path = OUT / (st + ".json")
        if j["method"] != "profile" and path.exists():
            if json.loads(path.read_text()).get("code_sha256") != sha:
                raise RuntimeError(f"version mismatch: {path}")
            return st, "cached"
        for attempt in range(a.retries + 1):
            print("START" if attempt == 0 else f"RETRY{attempt}", st, flush=True)
            t0 = time.time()
            with (OUT / "logs" / (st + (".log" if attempt == 0 else f".retry{attempt}.log"))).open("w", encoding="utf-8") as log:
                r = subprocess.run([sys.executable, "-u", str(ROOT / "cmn.py")] + argv(j), cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
            ok = r.returncode == 0 and (j["method"] == "profile" or path.exists())
            if ok:
                return st, "done" if attempt == 0 else f"done(retry{attempt})"
            print(f"FAILED attempt {attempt + 1} after {(time.time() - t0) / 60:.1f} min: {st}", flush=True)
            time.sleep(20)
        return st, "FAILED"

    status_fn = OUT / "status.json"
    status = json.loads(status_fn.read_text()) if status_fn.exists() else {}
    prof = [j for j in jobs if j["method"] == "profile"]
    rest = [j for j in jobs if j["method"] != "profile"]
    # first one job per (dataset, feature): the shared network and its cross-fit are trained once and cached
    first, seen = [], set()
    for j in rest:
        if j["method"] == "cpn" and (j["data"], j["feat"]) not in seen:
            first.append(j); seen.add((j["data"], j["feat"]))
    later = [j for j in rest if j not in first]
    # Release each dataset's dependent jobs after its own complete cache warm-up,
    # without waiting for every other dataset. All work stays within the same pool.
    key = lambda j: (j["data"], j["feat"])
    waiting = {key(j): [] for j in first}
    independent = []
    for j in later:
        if key(j) in waiting:
            waiting[key(j)].append(j)
        else:
            independent.append(j)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        fut = {pool.submit(run, j): (j, True) for j in first}
        fut.update({pool.submit(run, j): (j, False) for j in independent})
        while fut:
            done, _ = concurrent.futures.wait(fut, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                j, warm = fut.pop(f)
                st, state = f.result(); status[st] = state; completed += 1
                status_fn.write_text(json.dumps(status, indent=1))
                print(f"{completed}/{len(rest)} {state} {st}", flush=True)
                if warm:  # released also after a failed warm-up: the dependents then fit their own cache (logged)
                    fut.update({pool.submit(run, x): (x, False) for x in waiting.pop(key(j))})
    for j in prof:  # timing runs: sequential, nothing else on the GPU
        st, state = run(j); status[st] = state; status_fn.write_text(json.dumps(status, indent=1))
        print("profile", state, st, flush=True)
    failed = [k for k, v in status.items() if v == "FAILED"]
    missing = [stem(j) for j in rest if stem(j) not in status]
    if missing:
        raise SystemExit(f"{len(missing)} jobs never ran: {missing[:10]}")
    if failed:
        raise SystemExit(f"{len(failed)} failed: {failed[:10]} ... see {OUT / 'logs'}")
    print("ALL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
