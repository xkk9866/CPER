"""Extend the frozen round-10 accuracy study to seeds 0, 1, 2 on the local GPU.

Seed 0 stays in its original directory. Only scheduling and reporting are new;
cmn.py, training settings, folds, features and paired reference draws are unchanged.
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_cpn as R

BASE = Path(f'{_data_root}/runs/cpn10_frozen')
OUT = Path(f'{_data_root}/runs/cpn10_multiseed')
EXEC_ROOT = OUT / 'code_snapshot'
PROJECT = Path(_data_root)
SECTIONS = ['main', 'ablation', 'synth', 'stream', 'repr']



def write_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf8')
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--mem-frac', type=float, default=.12)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--preflight', action='store_true')
    a = ap.parse_args()
    assert 0 < a.workers * a.mem_frac <= .9 and a.workers > 0
    assert not os.environ.get('CMN_SMOKE'), 'CMN_SMOKE must be unset for formal experiments'
    sha = hashlib.sha256((R.ROOT / 'cmn.py').read_bytes()).hexdigest()[:16]
    old = json.loads((BASE / 'manifest.json').read_text())
    assert old['code_sha256'] == sha
    template = R.suites()
    jobs = [dict(j, seed=s) for sec in SECTIONS for s in [1, 2] for j in template[sec]]
    reused = [j for sec in SECTIONS for j in template[sec]]
    for j in reused:
        f = BASE / (R.stem(j) + '.json')
        assert f.exists() and json.loads(f.read_text())['code_sha256'] == sha, f
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    (OUT / 'code_snapshot').mkdir(exist_ok=True)
    manifest = dict(code_sha256=sha, seeds=[0, 1, 2], new_seeds=[1, 2],
                    seed0_directory=str(BASE), new_directory=str(OUT), steps=3000,
                    context_draws=old['context_draws'], jobs=jobs, reused_jobs=reused,
                    sections={sec: 2 * len(template[sec]) for sec in SECTIONS},
                    timing='Existing seed-0 exclusive-GPU profiles retained; no profiling repeated.',
                    seed_semantics='Independent source, auxiliary training, calibration and baseline selection; fixed outer folds and matched reference draws.',
                    metric_semantics='Pool held-out clips, average reference draws within each seed, then mean and sample SD across three seeds; no prediction ensemble.',
                    created=time.strftime('%Y-%m-%d %H:%M:%S'))
    write_json(OUT / 'manifest.json', manifest)
    for name in ['cmn.py', 'run_cpn.py', 'synth_data.py', 'aggregate_cpn.py', 'aggregate_cpn_multiseed.py', Path(__file__).name]:
        source, target = R.ROOT/name, EXEC_ROOT/name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    # Every training child reads its script and working directory from local E:.
    # A disconnected network share must not interrupt the GPU task queue.
    R.ROOT = EXEC_ROOT
    print(f'Frozen code {sha}; reuse {len(reused)} seed-0 jobs; NEW {len(jobs)} jobs: {manifest["sections"]}', flush=True)
    if a.list:
        return
    R.OUT = OUT
    R.check_data(jobs)
    R.preflight(jobs, a)
    if a.preflight:
        return
    env = os.environ.copy()
    env.update(CMN_OUT=str(OUT), MEM_FRAC=str(a.mem_frac), CMN_CPU_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONIOENCODING='utf-8',
               CUDA_VISIBLE_DEVICES='0', TEMP=f'{_data_root}/tmp', TMP=f'{_data_root}/tmp',
               HF_HOME=f'{_data_root}/cache/hf', CPN_PROJECT_ROOT=str(PROJECT))
    env.pop('CMN_SMOKE', None)
    status_path = OUT / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    start = time.time()

    def run(j):
        stem = R.stem(j)
        path = OUT / (stem + '.json')
        if path.exists():
            r = json.loads(path.read_text())
            assert r['code_sha256'] == sha and r['args']['seed'] == j['seed'], path
            assert r['args']['steps'] == 3000 and r['args'].get('folds') is None, path
            return stem, 'cached'
        for attempt in range(2):
            print('START' if attempt == 0 else 'RETRY', stem, flush=True)
            log = OUT / 'logs' / (stem + ('' if attempt == 0 else '.retry1') + '.log')
            with log.open('a', encoding='utf8') as fp:
                try:
                    rc = subprocess.run([sys.executable, '-u', str(EXEC_ROOT / 'cmn.py')] + R.argv(j),
                                        cwd=EXEC_ROOT, env=env, stdout=fp, stderr=subprocess.STDOUT).returncode
                except OSError as error:
                    print(f'Child launch error: {error}', file=fp, flush=True)
                    rc = -1
            if rc == 0 and path.exists():
                r = json.loads(path.read_text())
                assert r['code_sha256'] == sha and r['args']['seed'] == j['seed'], path
                return stem, 'done' if attempt == 0 else 'done(retry1)'
            print('FAILED ATTEMPT', stem, rc, flush=True)
        return stem, 'FAILED'

    for sec in SECTIONS:
        selected = [j for j in jobs if j['section'] == sec]
        key = lambda j: (j['data'], j['feat'], j['seed'])
        warm = [j for j in selected if j['method'] == 'cpn']
        waiting = {key(j): [] for j in warm}
        ready = []
        for j in selected:
            if j in warm:
                continue
            (waiting[key(j)] if key(j) in waiting else ready).append(j)
        print('SUITE', sec, len(selected), flush=True)
        with cf.ThreadPoolExecutor(max_workers=a.workers) as pool:
            pending = {pool.submit(run, j): (j, True) for j in warm}
            pending.update({pool.submit(run, j): (j, False) for j in ready})
            while pending:
                done, _ = cf.wait(pending, timeout=60, return_when=cf.FIRST_COMPLETED)
                if not done:
                    print('HEARTBEAT', sec, f'{sum(v != "FAILED" for v in status.values())}/{len(jobs)} complete', flush=True)
                for f in done:
                    j, is_warm = pending.pop(f)
                    st, state = f.result()
                    status[st] = state
                    write_json(status_path, status)
                    print(f'{len(status)}/{len(jobs)} {state} {st}', flush=True)
                    if is_warm:
                        pending.update({pool.submit(run, q): (q, False) for q in waiting.pop(key(j))})
        bad = [R.stem(j) for j in selected if status.get(R.stem(j)) not in ['done', 'cached', 'done(retry1)']]
        if bad:
            raise RuntimeError(f'Suite {sec} has failed jobs: {bad}')
        print('SUITE COMPLETE', sec, flush=True)
    write_json(OUT / 'execution_complete.json', dict(jobs=len(jobs), seconds=time.time()-start,
                completed=time.strftime('%Y-%m-%d %H:%M:%S'), code_sha256=sha))
    print('ALL 512 NEW ACCURACY JOBS COMPLETE; AGGREGATING', flush=True)
    subprocess.run([sys.executable, '-u', str(EXEC_ROOT / 'aggregate_cpn_multiseed.py')], cwd=EXEC_ROOT, env=env, check=True)


if __name__ == '__main__':
    main()
