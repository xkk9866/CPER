"""Report three independent training seeds, never treating reference draws as seeds.

Uses the frozen aggregator's pooled metrics and training-only baseline selection.
Original seed-0 tables are preserved. New tables expose every seed and sample SD.
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import aggregate_cpn as A
import run_cpn as R

OUT = Path(f'{_data_root}/runs/cpn10_multiseed')
BASE = Path(f'{_data_root}/runs/cpn10_frozen')
PROJECT = Path(_data_root)
PAPER = PROJECT / 'reports'
SEEDS = [0, 1, 2]
METRICS = ['acc', 'uar', 'f1']


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1), encoding='utf8')
    tmp.replace(path)


def describe(values):
    a = np.asarray(values, dtype=float)
    assert a.shape == (3,) and np.isfinite(a).all()
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)), seeds=SEEDS, values=a.tolist())


def cell(v):
    return f'{v["mean"]:.2f} ± {v["sd"]:.2f}'


def extract(j, sha):
    R.OUT = BASE if j['seed'] == 0 else OUT
    stem = R.stem(j)
    path = R.OUT / (stem + '.json')
    raw_bytes = path.read_bytes()
    fingerprint = hashlib.sha256(raw_bytes).hexdigest()
    cache = OUT / 'metrics' / (stem + '.json')
    if cache.exists():
        r = json.loads(cache.read_text())
        if r.get('input_sha256') == fingerprint:
            return r['metrics']
    raw = json.loads(raw_bytes)
    assert raw['code_sha256'] == sha and raw['args']['seed'] == j['seed'], path
    assert raw['args']['steps'] == 3000 and raw['args'].get('folds') is None, path
    A.SHA = sha
    if j['section'] == 'stream':
        # The original routine pools all held-out predictions, rather than fold means.
        import pandas as pd
        meta = pd.read_csv(f'{A.CM.PROC}/{j["data"]}_meta.csv')
        assert set(map(int, raw['folds'])) == set(A.CM.test_folds(meta))
        labels, arrays = [], []
        for f, value in raw['folds'].items():
            q = np.asarray(value['order'], dtype=int)
            a = np.load(R.OUT / 'pred' / f'{stem}_f{f}.npy')
            assert a.shape == (len(raw['methods']), len(q), meta.label.nunique())
            assert np.isfinite(a).all() and a.min() >= -1e-6 and np.allclose(a.sum(-1), 1, atol=1e-3)
            labels.append(meta.label.values[q]); arrays.append(a)
        y = np.concatenate(labels); arr = np.concatenate(arrays, axis=1)
        result = {}
        for i, m in enumerate(raw['methods']):
            v = {k: A.metric(arr[i], y, meta.label.nunique(), k) for k in METRICS}
            for k in METRICS:
                assert np.isclose(v[k], raw['pooled'][m][k], atol=1e-5), (stem, m, k)
            result[m] = {'stream': v}
    else:
        X = A.Data(j['data'], [j], section=(j['section'],))
        jb = X.J[j['method']]
        names = [jb.main_name()]
        if j['method'] == 'cpn':
            names += [x for x in A.ALT if x in jb.P]
        result = {}
        for nm in names:
            name = j['method'] if nm == jb.main_name() else j['method'] + ':' + nm
            protocols = list(jb.P[nm])
            if all(q in protocols for q in A.MIX):
                protocols += ['cavg']
            if 'rand' in protocols:
                protocols += ['worst']
            v = {pr: {k: X.val(j['method'], pr, k, nm) for k in METRICS} for pr in protocols}
            v['draw0'] = {pr: {k: X.val(j['method'], pr, k, nm, draws=1) for k in METRICS}
                          for pr in protocols if pr != 'worst'}
            result[name] = v
        del X, jb
    save(cache, dict(input_sha256=fingerprint, metrics=result))
    gc.collect()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate-seed0', action='store_true', help='Check all seed-0 main metrics against the frozen published summary; no manuscript edits.')
    ap.add_argument('--main-only', action='store_true', help='Publish the complete three-seed main suite while other suites continue.')
    a = ap.parse_args()
    man = json.loads((OUT / 'manifest.json').read_text())
    sha = man['code_sha256']
    assert sha == A.CM.CODE_SHA
    (OUT / 'metrics').mkdir(exist_ok=True)
    jobs = man['reused_jobs'] + man['jobs']
    if a.main_only:
        jobs = [j for j in jobs if j['section'] == 'main']
    if a.validate_seed0:
        jobs = [j for j in jobs if j['section'] == 'main' and j['seed'] == 0]
    missing = [R.stem(j) for j in jobs if not ((BASE if j['seed'] == 0 else OUT) / (R.stem(j)+'.json')).exists()]
    if missing:
        raise SystemExit(f'{len(missing)} results pending; no complete tables or manuscript will be written. First: {missing[:5]}')
    frozen = json.loads((PAPER / 'cpn10_results.json').read_text())['_summary']
    records = {}
    checked = 0
    for i,j in enumerate(jobs):
        result = extract(j, sha)
        if j['seed'] == 0 and j['section'] == 'main':
            for m, scores in result.items():
                for pr, values in scores.items():
                    if pr in ['draw0', 'worst']:
                        continue
                    for k, v in values.items():
                        assert np.isclose(v, frozen[j['data']][m][pr][k], atol=1e-7), (j['data'],m,pr,k,v)
                        checked += 1
        for m,v in result.items():
            records.setdefault(j['section'], {}).setdefault(j['data'], {}).setdefault(m,{})[str(j['seed'])] = v
        print(f'METRICS {i+1}/{len(jobs)} {R.stem(j)}', flush=True)
    if a.validate_seed0:
        save(OUT/'seed0_metric_validation.json', dict(jobs=len(jobs), checked_values=checked, matched=True, code_sha256=sha))
        print(f'PASS: {checked} pooled values match the frozen seed-0 summary.', flush=True)
        return
    summary = {}
    for sec, datasets in records.items():
        for d, methods in datasets.items():
            for m, seeds in methods.items():
                assert sorted(map(int,seeds)) == SEEDS, (sec,d,m,seeds.keys())
                dest = summary.setdefault(sec,{}).setdefault(d,{}).setdefault(m,{})
                protocols = set(seeds['0']) - {'draw0'}
                assert all(set(v)-{'draw0'} == protocols for v in seeds.values())
                for pr in sorted(protocols):
                    dest[pr] = {k: describe([seeds[str(s)][pr][k] for s in SEEDS]) for k in METRICS}
    contrasts = {}
    for d in R.DATA:
        mm = summary['main'][d]
        cc = contrasts[d] = {}
        for pr,m,k in [('rand','bn','acc'),('same','source','acc'),('cavg','bn','acc'),('rand','source','uar')]:
            vals = np.asarray(mm['cpn'][pr][k]['values']) - np.asarray(mm[m][pr][k]['values'])
            cc[f'{pr}:{k}:cpn-{m}'] = describe(vals)
        # Ablations use the matched first draw of the complete model.
        for m, seeds in records.get('ablation', {}).get(d, {}).items():
            full = [records['main'][d]['cpn'][str(s)]['draw0']['cavg']['acc'] for s in SEEDS]
            ab = [seeds[str(s)]['cavg']['acc'] for s in SEEDS]
            cc['cavg:acc:cpn-'+m] = describe(np.asarray(full)-np.asarray(ab))
    new_count=sum(j['seed'] != 0 for j in jobs)
    result = dict(_meta=dict(code_sha256=sha,seeds=SEEDS,jobs=len(jobs),new_jobs=new_count,reused_jobs=len(jobs)-new_count,
                            seed0_metric_values_checked=checked,complete=True,scope='main' if a.main_only else 'all_accuracy',manifest=man,
                            sd='Sample SD across training seeds (ddof=1), after pooling clips and averaging reference draws; no ensemble.'),
                  summary=summary,per_seed=records,paired_seed_differences=contrasts)
    lines=['# CPN: three-training-seed evaluation',
           f'Frozen algorithm `{sha}`. Seeds 0, 1, 2. Scope: {result["_meta"]["scope"]}. All {len(jobs)} jobs in this scope complete ({len(jobs)-new_count} reused + {new_count} new).',
           'Values are mean ± sample standard deviation across independently trained runs. Main episodes average three paired reference draws within each training seed. Other suites retain one draw. No prediction ensembling. Outer folds, features, training budget and hyperparameters are fixed.',
           'Seed-level variability and identity-bootstrap confidence intervals measure different quantities. Original identity-bootstrap intervals remain conditional on seed 0; this report does not relabel standard deviations as confidence intervals.']
    for sec,datasets in summary.items():
        lines += ['\n## '+sec]
        for d, methods in datasets.items():
            lines += ['\n### '+A.DN.get(d,d)]
            protocols = ['stream'] if sec=='stream' else [x for x in ['rand','skew','dir02','past','same','cavg','worst'] if x in next(iter(methods.values()))]
            for k in METRICS:
                lines += ['\n'+k+' (%)', '| Method | '+' | '.join(protocols)+' |','|---|'+'---|'*len(protocols)]
                for m,scores in methods.items():
                    lines += ['| '+A.LAB.get(m,A.ALT.get(m.replace('cpn:',''),m))+' | '+' | '.join(cell(scores[pr][k]) for pr in protocols)+' |']
    lines += ['\n## Paired seed-level differences','Differences pair the same seed, folds and reference draws. Ablation contrasts use CPN draw 0. Positive means CPN is higher.','| Dataset | Comparison | Mean ± SD | seed 0 | seed 1 | seed 2 |','|---|---|---|---|---|---|']
    for d,cs in contrasts.items():
        for key,v in cs.items():
            lines += ['| '+d+' | '+key+' | '+cell(v)+' | '+' | '.join(f'{x:+.2f}' for x in v['values'])+' |']
    prefix='cpn_multiseed_main' if a.main_only else 'cpn_multiseed'
    save(OUT/(prefix+'_results.json'), result)
    save(PAPER/(prefix+'_results.json'), result)
    (PAPER/(prefix+'_tables.md')).write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('COMPLETE:',prefix,flush=True)


if __name__ == '__main__':
    main()
