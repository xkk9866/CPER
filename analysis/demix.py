"""DEMIX: a deep mixed-effects network for subject-relative facial expression recognition.

One generative model drives representation, subject inference and classification:
    z_i = phi(x_i),     z_i = mu_s + f_{y_i} + eps_i,   mu_s ~ N(0, diag tau^2),  eps_i ~ N(0, diag sigma^2)
For an unlabeled set of clips of one subject (query + K context clips, K >= 0) the latent subject effect and
the latent expressions of *all* members are inferred jointly by unrolled EM; the query prediction is its
E-step posterior under the posterior-predictive likelihood (variance sigma^2 + Var[mu]). Training maximises
the same posterior on labeled training episodes, so phi is learned to make the mixed-effects model true.

Baselines share the inputs, episodes, optimiser and budget:
  lp      linear classifier on x                         mlp     2-layer MLP on x
  ctr     x - mean(context) -> linear  (K=0: x - 0)       ctr_mlp same with MLP
  settf   set transformer over [query; context] (generic in-context learner)
Ablations of DEMIX: demix_noem (one E-step from the population prior), demix_mean (mu = mean of members,
no explaining-away), demix_novar (plug-in mu, no posterior variance), demix_lin (phi linear).

usage: python demix.py --data ravdess --feat frozen|ft|both --method demix --seed 0
writes $EMO_ROOT/runs/demix/{data}_{feat}_{method}_s{seed}.json  (all folds, all protocols)
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

PROC, FEAT, OUT = f"{_data_root}/datasets/proc", f"{_data_root}/features/clip", f"{_data_root}/runs/demix"
dev = "cuda"
SHARED6 = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
LAB = {"ravdess": ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"],
       "cremad": ["angry", "disgust", "fear", "happy", "neutral", "sad"]}


def load(data, feat, fold):
    parts = []
    if feat in ("frozen", "both"):
        parts.append(np.load(f"{FEAT}/{data}_frozen.npy"))
    if feat in ("ft", "both"):
        parts.append(np.load(f"{FEAT}/{data}_ft_f{fold}.npy"))
    return np.concatenate(parts, 1)


# ----------------------------------------------------------------------------------------------- models
class Encoder(nn.Module):
    def __init__(self, D, d, linear=False, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(p), nn.Linear(D, d)) if linear else \
            nn.Sequential(nn.Dropout(p), nn.Linear(D, 512), nn.GELU(), nn.Dropout(p), nn.Linear(512, d))

    def forward(self, x):
        return self.net(x)


class DEMIX(nn.Module):
    """Linear additivity-preserving lift -> joint EM over (subject effect, member expressions) -> nonlinear readout.

    lift      z = W x  (linear: maps an additive subject + expression model to an additive one, discards nothing
              the readout could later need; every nonlinearity sits *after* subject correction)
    E-step    r_i = softmax(g(z_i - m))           g: readout on the subject-relative code
    M-step    m = sum_i (z_i - r_i F) / (n + lambda)  F: learned class displacements, lambda = sigma^2/tau^2 per dim
    The fixed point is a MAP estimate of the random subject effect with every member's expression explained away;
    with no context (n = 1) the query alone is (shrunk) evidence about its own subject.
    """

    def __init__(self, D, C, d=256, iters=5, variant="full"):
        super().__init__()
        self.lift = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, d))
        self.g = nn.Linear(d, C) if variant == "lin" else nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, C))
        self.F = nn.Parameter(torch.randn(C, d) * 0.02)
        self.log_lam = nn.Parameter(torch.zeros(d))
        self.iters, self.variant = iters, variant

    def forward(self, x):
        z = self.lift(x)
        B, n, d = z.shape
        lam = self.log_lam.exp()
        if self.variant == "nosubj":
            return self.g(z)
        e = torch.zeros_like(z)  # E-step 0: uniform label prior, F has (learned) zero-mean expectation
        for _ in range(1 if self.variant == "noem" else self.iters):
            m = (z - e).sum(1) / (n + lam)
            lg = self.g(z - m[:, None])
            e = lg.softmax(-1) @ (self.F - self.F.mean(0))
        return lg


class Plain(nn.Module):
    """lp / mlp / ctr / ctr_mlp: classify the (optionally context-centred) query; context members get the same map."""

    def __init__(self, D, C, kind):
        super().__init__()
        self.kind = kind
        self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, C)) if kind in ("lp", "ctr") else \
            nn.Sequential(nn.Dropout(0.1), nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, C))

    def forward(self, x):
        if self.kind.startswith("ctr") and x.shape[1] > 1:
            x = x - x[:, 1:].mean(1, keepdim=True)  # centre every member by the mean of the context clips
        return self.net(x)


class SetTF(nn.Module):
    """Generic in-context learner: transformer over the set, query/context role embedding."""

    def __init__(self, D, C, d=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, d))
        self.role = nn.Parameter(torch.randn(2, d) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, C))

    def forward(self, x):
        h = self.inp(x)
        role = torch.zeros(x.shape[1], dtype=torch.long, device=x.device); role[1:] = 1
        return self.out(self.tf(h + self.role[role]))


def build(method, D, C):
    if method.startswith("demix"):
        return DEMIX(D, C, variant={"demix": "full", "demix_noem": "noem", "demix_lin": "lin", "demix_nosubj": "nosubj"}[method])
    if method == "settf":
        return SetTF(D, C)
    return Plain(D, C, method)


# --------------------------------------------------------------------------------------------- episodes
def contexts(meta, rows, K, mode, rng, pool_rows=None):
    """Context indices (len(rows),K) of the *same subject*, never the query itself.
    rand   : uniform over the subject's other clips (label-free; main protocol)
    xsent  : only clips with a different sentence/statement than the query (label-free, cross-content)
    contam : rand, but the first third of the slots come from other subjects (label-free, contaminated)
    same   : all of the query's own class (diagnostic, label-controlled)
    other  : all of one class != query's (diagnostic, label-controlled; = earlier 'skew')
    """
    act, y, sen = meta.actor.values, meta.label.values, meta.sentence.values
    allr = np.arange(len(meta)) if pool_rows is None else pool_rows
    out = np.zeros((len(rows), K), int)
    for i, r in enumerate(rows):
        pool = allr[(act[allr] == act[r]) & (allr != r)]
        if mode == "xsent":
            pool = pool[sen[pool] != sen[r]]
        elif mode == "same":
            pool = pool[y[pool] == y[r]]
        elif mode == "other":
            e = rng.choice([c for c in np.unique(y[pool]) if c != y[r]])
            pool = pool[y[pool] == e]
        out[i] = rng.choice(pool, K, replace=len(pool) < K)
        if mode == "contam":
            n_bad = K // 3
            others = allr[act[allr] != act[r]]
            out[i, :n_bad] = rng.choice(others, n_bad, replace=False)
    return out


def train(model, X, meta, rows, C, a, rng):
    y = torch.tensor(meta.label.values, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    episodic = a.method not in ("lp", "mlp")
    for it in range(a.steps):
        q = rng.choice(rows, a.bs)
        K = int(rng.integers(a.kmin, a.kmax + 1)) if episodic else 0
        idx = np.concatenate([q[:, None], contexts(meta, q, K, "rand", rng, rows)], 1) if K else q[:, None]
        idx = torch.from_numpy(idx).to(dev)
        lg = model(X[idx])  # (B,n,C)
        if isinstance(model, DEMIX):  # every labeled member of the training episode supervises the posterior
            loss = F.cross_entropy(lg.flatten(0, 1), y[idx].flatten(), label_smoothing=0.1)
        else:
            loss = F.cross_entropy(lg[:, 0], y[idx[:, 0]], label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def predict(model, X, ctx_idx, rows):
    model.eval()
    q = torch.from_numpy(rows).to(dev)[:, None]
    idx = q if ctx_idx is None or ctx_idx.shape[1] == 0 else torch.cat([q, torch.from_numpy(ctx_idx).to(dev)], 1)
    out = torch.cat([model(X[idx[i:i + 512]])[:, 0].softmax(-1) for i in range(0, len(idx), 512)])
    model.train()
    return out.cpu().numpy()


def metrics(p, y, C):
    pred = p.argmax(1)
    return dict(acc=float(np.mean(pred == y) * 100), uar=float(np.mean([np.mean(pred[y == c] == c) for c in range(C) if np.any(y == c)]) * 100))


PROTOCOLS = ["rand", "xsent", "contam", "same", "other"]
KCURVE = [1, 2, 4, 8, 16]


def run(a):
    meta = pd.read_csv(f"{PROC}/{a.data}_meta.csv")
    C = meta.label.nunique()
    res = dict(args=vars(a), folds={})
    for fold in sorted(meta.fold.unique()):
        torch.manual_seed(a.seed * 100 + fold); rng = np.random.default_rng(a.seed * 100 + fold)
        X = load(a.data, a.feat, fold)
        tr, te = np.where(meta.fold != fold)[0], np.where(meta.fold == fold)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        X = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
        model = train(build(a.method, X.shape[1], C).to(dev), X, meta, tr, C, a, rng)
        yt = meta.label.values[te]
        erng = np.random.default_rng(1000 + fold)  # identical contexts for every method / seed
        r = {"K0": metrics(predict(model, X, None, te), yt, C)}
        P = {"K0": predict(model, X, None, te)}
        for mode in PROTOCOLS:
            c = contexts(meta, te, a.kctx, mode, erng, te)
            P[mode] = predict(model, X, c, te); r[mode] = metrics(P[mode], yt, C)
        for k in KCURVE:
            r[f"K{k}"] = metrics(predict(model, X, contexts(meta, te, k, "rand", erng, te), te), yt, C)
        res["folds"][int(fold)] = r
        np.save(f"{OUT}/pred/{a.data}_{a.feat}_{a.method}{a.suffix}_s{a.seed}_f{fold}.npy", np.stack([P[k] for k in ["K0"] + PROTOCOLS]))
        if a.dump_z and isinstance(model, DEMIX):
            model.eval()
            with torch.no_grad():
                np.save(f"{OUT}/z/{a.data}_{a.feat}_f{fold}.npy", model.lift(X).cpu().numpy())
        print(fold, {k: round(v["acc"], 2) for k, v in r.items()}, flush=True)
    keys = list(res["folds"][0].keys()) if 0 in res["folds"] else list(next(iter(res["folds"].values())).keys())
    res["mean"] = {k: {m: float(np.mean([f[k][m] for f in res["folds"].values()])) for m in ["acc", "uar"]} for k in keys}
    return res


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ravdess")
    ap.add_argument("--feat", default="both", choices=["frozen", "ft", "both"])
    ap.add_argument("--method", default="demix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--kmin", type=int, default=0)
    ap.add_argument("--kctx", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--dump_z", type=int, default=0)
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = get_args()
    for d in ["pred", "z"]:
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    t0 = time.time()
    res = run(a)
    res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(f"{OUT}/{a.data}_{a.feat}_{a.method}{a.suffix}_s{a.seed}.json", "w"))
    print("RESULT", a.data, a.feat, a.method, a.seed, json.dumps({k: round(v["acc"], 2) for k, v in res["mean"].items()}), flush=True)
