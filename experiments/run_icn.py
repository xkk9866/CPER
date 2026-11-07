"""Formal study of ICN: job manifest + long-lived GPU workers that claim jobs through lock files (NAS-safe, restartable).

    python run_icn.py --list                                   # job counts and manifest
    python run_icn.py --launch --workers 6 --mem-frac 0.08     # start 6 worker processes (GPU 0), returns at once
    python run_icn.py --worker --mem-frac 0.08                 # one worker (what --launch starts)
    python run_icn.py --status                                 # done / failed / running / pending
    python run_icn.py --profile                                # the 3 timing jobs, one process, nothing else on the GPU

Suites (all people-disjoint folds, reference draws paired across methods and seeds; everything on frozen encoders):
  main      9 datasets x 15 methods (+ neucal where a neutral class exists), training seeds 0/1/2, 3 reference draws,
            protocols rand / skew / dir02 / (past) / same / (neu6) / composition curve j = 0..6 / accuracy vs K.
  ablation  10 ICN variants x 9 datasets, seed 0, 1 draw.
  synth     5 generative regimes with exact person centres: source, bn, cpn, cpn_oracle, icn.
  stream    person-blocked streams, 9 datasets (window BN / CPN / ICN, TENT-online, NOTE, RoTTA; continual and reset).
  repr      CLIP-B/16 on the four lab sets, face-only CLIP-L/14 on MELD; 7 methods.
  profile   latency / memory / accuracy vs K on RAVDESS (C=8), CREMA-D (C=6), ShEMO (C=5); run alone.
Each worker imports torch once and runs jobs in-process (the shared workstation makes process start-up expensive).
Outputs: $CMN_OUT (default $EMO_ROOT/runs/icn): json, pred/, logs/, locks/, manifest.json."""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, hashlib, json, os, pathlib, subprocess, sys, threading, time, traceback

ROOT = pathlib.Path(__file__).resolve().parent
E = _data_root
OUT = pathlib.Path(os.environ.get("CMN_OUT", f"{E}/runs/icn"))
PROC, FEATD = pathlib.Path(f"{E}/datasets/proc"), pathlib.Path(f"{E}/features/clip")
LAB = ["ravdess", "cremad", "mead", "enterface"]
NAT = {"meld": "avt", "shemo": "wavlm", "emoz": "wavlm", "thai": "wavlm", "thaix": "wavlm"}
FEAT = {**{d: "l14" for d in LAB}, **NAT}
DATA = LAB + list(NAT)
WITH_NEUTRAL = [d for d in DATA if d != "enterface"]
MAIN = ["source", "icn", "bn", "tent_grid", "sar_grid", "laem_grid", "note", "rotta_grid", "csn_grid", "dart", "lame",
        "t3a", "srcbn_grid", "bnprior_grid", "cpn"]  # 'source' first (trains the shared classifier); CPN last (slowest)
ABL = ["icn_noaug", "icn_noxfit", "icn_iid", "icn_dir", "icn_nocomp", "icn_noscale", "icn_nosrc", "icn_nogeo", "icn_raw", "icn_free", "icn_direct"]
REPR = {**{d: "b16" for d in LAB}, "meld": "l14"}
REPR_M = ["source", "icn", "cpn", "bn", "rotta_grid", "csn_grid", "srcbn_grid"]
SYN_BASE = {"--s_G": "0.12", "--lam_sd": "0.15", "--s_F": "0.25", "--s_e": "1.2"}
SYN_FACT = {"G": ("--s_G", "0.03", "0.25"), "E": ("--s_e", "0.6", "2.0")}
SYN_M = ["source", "icn", "cpn", "bn", "cpn_oracle"]
PROFILE = {"ravdess": "l14", "cremad": "l14", "shemo": "wavlm"}
PROFILE_METHODS = ["bn", "tent", "laem", "rotta", "dart", "cpn", "icn"]
SEEDS = [0, 1, 2]


def syn_sets():
    out = {"synth_o_base": dict(SYN_BASE)}
    for k, (arg, lo, hi) in SYN_FACT.items():
        for tag, v in (("lo", lo), ("hi", hi)):
            out[f"synth_o_{k}{tag}"] = {**SYN_BASE, arg: v}
    return out


SYN = syn_sets()


def J(section, data, method, seed, nctx, suite, feat=None):
    return dict(section=section, data=data, method=method, seed=seed, feat=feat or FEAT.get(data, "raw"), net="iid",
                nctx=nctx, suite=suite)


def suites():
    main = []
    for s in SEEDS:  # order: source (shared classifier), ICN, the baselines, CPN last; CPN without the K = 8 lattice
        for m in MAIN:
            for d in DATA:
                main.append(J("main", d, m, s, 3, "lite" if m == "cpn" else "paper"))
        main += [J("main", d, "neucal", s, 3, "paper") for d in WITH_NEUTRAL]
    return {"main": main,
            "ablation": [J("ablation", d, m, 0, 1, "lite") for m in ABL for d in DATA],
            "synth": [J("synth", d, m, 0, 1, "lite", "raw") for d in SYN for m in SYN_M],
            "stream": [J("stream", d, "stream", 0, 1, "lite") for d in DATA],
            "repr": [J("repr", d, m, 0, 1, "lite", f) for d, f in REPR.items() for m in REPR_M],
            "profile": [dict(J("profile", d, "profile", 0, 1, "lite", f), profile_k=[1, 2, 4, 6, 8], profile_methods=PROFILE_METHODS)
                        for d, f in PROFILE.items()]}


def argv(j):
    if j["method"] == "profile":
        return (["--data", j["data"], "--profile", "--feat", j["feat"], "--net", j["net"], "--profile-k"]
                + list(map(str, j["profile_k"])) + ["--profile-methods"] + j["profile_methods"])
    if j["method"] == "stream":
        return ["--data", j["data"], "--stream", "--feat", j["feat"], "--net", j["net"], "--seed", str(j["seed"])]
    return ["--data", j["data"], "--method", j["method"], "--seed", str(j["seed"]), "--feat", j["feat"], "--net", j["net"],
            "--suite", j["suite"], "--nctx", str(j["nctx"])]


def stem(j):
    if j["method"] == "profile":
        return f"profile_{j['data']}_{j['feat']}_{j['net']}"
    if j["method"] == "stream":
        return f"stream_{j['data']}_{j['feat']}_s{j['seed']}"
    tag = (j["method"] + ("" if j["feat"] == "l14" else f"_{j['feat']}") + ("" if j["suite"] == "full" else f"_{j['suite']}")
           + ("" if j["nctx"] == 3 else f"_c{j['nctx']}"))
    return f"{j['data']}_{tag}_s{j['seed']}"


def ensure_synth(names):
    for name in names:
        if not (PROC / f"{name}_meta.csv").exists():
            extra = [x for kv in SYN[name].items() for x in kv] + ["--person_scale"]
            subprocess.run([sys.executable, str(ROOT / "synth_data.py"), "--name", name] + extra, check=True, cwd=ROOT)


def all_jobs(a):
    S = suites()
    sel = [x for x in a.suites.split(",") if x]
    jobs = [j for x in sel for j in S[x]]
    if a.only_data:
        jobs = [j for j in jobs if j["data"] in a.only_data.split(",")]
    if a.only_methods:
        jobs = [j for j in jobs if j["method"] in a.only_methods.split(",")]
    if a.seeds:
        jobs = [j for j in jobs if j["seed"] in [int(s) for s in a.seeds.split(",")]]
    # priority: everything of training seed 0 (main, then the diagnostic suites) before the seed-1/2 repeats of main
    rank = {x: i for i, x in enumerate(sel)}
    jobs.sort(key=lambda j: (j["seed"] > 0, rank[j["section"]], j["seed"]))
    return S, jobs


def pid_alive(pid):
    try:
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code)); ctypes.windll.kernel32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    except Exception:  # noqa: BLE001
        return True


def clean_locks():
    """remove the locks of jobs whose worker process no longer exists (killed workers) and that have no result"""
    n = 0
    for lock in (OUT / "locks").glob("*.lock"):
        if (OUT / (lock.stem + ".json")).exists():
            continue
        try:
            pid = int(lock.read_text().split()[0])
        except Exception:  # noqa: BLE001
            pid = -1
        if not pid_alive(pid):
            lock.unlink(); n += 1
            print("released", lock.stem)
    print("released", n, "stale locks")


def data_ready(j):
    if j["section"] == "synth":
        return True
    return (PROC / f"{j['data']}_meta.csv").exists() and (FEATD / f"{j['data']}_{j['feat']}.npy").exists()


def source_job(j):
    """the 'source' job that trains the shared classifier of this (data, feat, seed)"""
    if j["section"] in ("main", "ablation", "stream"):
        return J("main", j["data"], "source", j["seed"] if j["section"] == "main" else 0, 3, "paper")
    if j["section"] == "repr":
        return J("repr", j["data"], "source", 0, 1, "lite", j["feat"])
    if j["section"] == "synth":
        return J("synth", j["data"], "source", 0, 1, "lite", "raw")
    return None


def deps_ready(j):
    """a job starts only after the shared classifier of its (data, feat, seed) exists (one worker trains it; the
    others reuse the cache instead of training duplicates)"""
    if j["method"] in ("source", "profile"):
        return True
    s = source_job(j)
    return s is None or (OUT / (stem(s) + ".json")).exists()


class ThreadLog:
    """stdout/stderr proxy that writes each thread's output to that thread's current job log"""

    def __init__(self, fallback):
        self.fallback, self.local = fallback, threading.local()

    def set(self, f):
        self.local.f = f

    def write(self, s):
        f = getattr(self.local, "f", None) or self.fallback
        f.write(s)

    def flush(self):
        f = getattr(self.local, "f", None) or self.fallback
        f.flush()


def job_loop(a, jobs, tlog, wid):
    """claim and run jobs until none is left (one thread; several threads share one process = one CUDA context, which
    matters on a GPU that other processes time-slice)"""
    import torch, icn
    done_here, attempts = 0, {}
    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    while True:
        progressed = False
        for j in jobs:
            st = stem(j)
            if (OUT / (st + ".json")).exists() or not data_ready(j) or not deps_ready(j) or attempts.get(st, 0) >= 2:
                continue
            lock = OUT / "locks" / (st + ".lock")
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')} {wid}\n".encode()); os.close(fd)
            except FileExistsError:
                continue
            attempts[st] = attempts.get(st, 0) + 1
            progressed = True
            t0 = time.time()
            log = open(OUT / "logs" / (st + ".log"), "a", encoding="utf-8")
            tlog.set(log)
            try:
                print("START", st, time.strftime("%Y-%m-%d %H:%M:%S"), "worker", wid, flush=True)
                if stream is not None:
                    with torch.cuda.stream(stream):
                        icn.run_job(argv(j))
                else:
                    icn.run_job(argv(j))
                state = "done"
            except Exception:  # noqa: BLE001
                traceback.print_exc(file=log)
                state = "FAILED"
                (OUT / "logs" / (st + ".failed")).write_text(traceback.format_exc(), encoding="utf-8")
            finally:
                tlog.set(None)
                log.close()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            done_here += 1
            print(f"[{wid}] {state} {st} {(time.time() - t0) / 60:.1f} min", flush=True)
            if state == "FAILED":
                try:
                    lock.unlink()  # a failed job can be retried by a later pass / another worker
                except OSError:
                    pass
        if not progressed:
            pending = [stem(j) for j in jobs if not (OUT / (stem(j) + ".json")).exists() and not (OUT / "locks" / (stem(j) + ".lock")).exists()]
            if not pending:
                break
            time.sleep(60)  # data not ready yet (features still being extracted): wait and rescan
    print(f"[{wid}] finished after {done_here} jobs", flush=True)


def worker(a):
    """one process (one CUDA context), `--threads` job threads; one torch import per worker"""
    import torch
    sys.path.insert(0, str(ROOT))
    import icn  # noqa: F401
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    S, jobs = all_jobs(a)
    jobs = [j for j in jobs if j["method"] != "profile"]
    ensure_synth(sorted({j["data"] for j in jobs if j["section"] == "synth"}))
    (OUT / "locks").mkdir(parents=True, exist_ok=True); (OUT / "logs").mkdir(exist_ok=True)
    tlog = ThreadLog(sys.stdout)
    sys.stdout = sys.stderr = tlog
    ths = [threading.Thread(target=job_loop, args=(a, jobs, tlog, f"{os.getpid()}.{t}"), daemon=True) for t in range(a.threads)]
    for t in ths:
        t.start(); time.sleep(2)
    for t in ths:
        t.join()
    print(f"[{os.getpid()}] worker finished", flush=True)


def status(a):
    S, jobs = all_jobs(a)
    cnt = {"done": 0, "failed": 0, "running": 0, "pending": 0, "waiting_data": 0}
    rows = []
    for j in jobs:
        st = stem(j)
        if (OUT / (st + ".json")).exists():
            s = "done"
        elif (OUT / "logs" / (st + ".failed")).exists() and not (OUT / "locks" / (st + ".lock")).exists():
            s = "failed"
        elif (OUT / "locks" / (st + ".lock")).exists():
            s = "running"
        elif not data_ready(j):
            s = "waiting_data"
        else:
            s = "pending"
        cnt[s] += 1; rows.append((s, st))
    print(json.dumps(cnt))
    for s, st in rows:
        if s in ("failed", "running"):
            print(s, st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", default="main,ablation,synth,stream,repr")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mem-frac", type=float, default=0.08)
    ap.add_argument("--threads", type=int, default=1, help="job threads per worker process (one CUDA context)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--only-data", default="")
    ap.add_argument("--only-methods", default="")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--clean-locks", action="store_true", help="release the locks of killed workers")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.clean_locks:
        clean_locks(); return
    if a.worker:
        worker(a); return
    if a.status:
        status(a); return
    S, jobs = all_jobs(a)
    sha = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest()[:16] for f in ("icn.py", "cmn.py")}
    man = dict(code=sha, seeds=SEEDS, context_draws={"main": 3, "ablation": 1, "synth": 1, "stream": 1, "repr": 1},
               suites={k: len(v) for k, v in S.items()}, jobs=[dict(j, stem=stem(j)) for x in S for j in S[x]], synth=SYN,
               created=time.strftime("%Y-%m-%d %H:%M:%S"))
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    print({k: len(v) for k, v in S.items()}, "selected:", len(jobs), "code", sha, flush=True)
    if a.list:
        return
    if a.profile:
        sys.path.insert(0, str(ROOT))
        import torch, icn
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.5)
        for j in S["profile"]:
            icn.run_job(argv(j))
        return
    if a.launch:
        env = os.environ.copy()
        env.update(CMN_OUT=str(OUT), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", CMN_CPU_THREADS="2", PYTHONIOENCODING="utf-8",
                   CUDA_VISIBLE_DEVICES=os.environ.get("ICN_GPU", "0"))  # GPU 0 unless the author assigns another card
        (OUT / "logs").mkdir(exist_ok=True)
        pids = []
        for w in range(a.workers):
            log = open(OUT / "logs" / f"worker_{w}_{time.strftime('%H%M%S')}.log", "a", encoding="utf-8")
            args = [sys.executable, "-u", str(ROOT / "run_icn.py"), "--worker", "--mem-frac", str(a.mem_frac), "--suites", a.suites, "--threads", str(a.threads)]
            if a.only_data: args += ["--only-data", a.only_data]
            if a.only_methods: args += ["--only-methods", a.only_methods]
            if a.seeds: args += ["--seeds", a.seeds]
            p = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            pids.append(p.pid); time.sleep(3)
        (OUT / "workers.json").write_text(json.dumps(dict(pids=pids, started=time.strftime("%Y-%m-%d %H:%M:%S"))))
        print("workers", pids)


if __name__ == "__main__":
    main()
