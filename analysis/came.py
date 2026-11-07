"""CAME: composition-aware mixed-effects inference for subject-relative facial expression recognition.

Generative model of one subject's reference set R = {z_1..z_K} (z = W x, a linear lift of the clip feature):
    theta_R ~ Dir(alpha 1)                  composition of the reference set (a latent of the set, not of the person)
    y_k | theta_R ~ Cat(theta_R)
    mu_s ~ N(0, diag(sigma^2 / lambda))     subject effect, population prior
    z_k = mu_s + f_{y_k} + eps_k,  eps_k ~ N(0, sigma^2 I),  sum_c f_c = 0
The query is an independent draw  y_q ~ Cat(pi),  z_q = mu_s + f_{y_q} + eps_q:  the composition of the references
carries no information about the query label, so the model cannot profit from label-controlled protocols.

Inference is mean-field variational Bayes  q(mu) q(theta) prod_k q(y_k)  (coordinate ascent, each step increases the ELBO):
    q(y_k):   log r_kc = E[log theta_c] - ||z_k - m - f_c||^2 / (2 sigma^2) + const      (E[||mu - m||^2] cancels in c)
    q(theta): Dir(alpha + sum_k r_k)
    q(mu):    N(m, diag(v)),  m = sum_k (z_k - r_k F) / (K + lambda),  v = sigma^2 / (K + lambda)
Query:      p(y_q = c | z_q, q) ∝ pi_c N(z_q ; m + f_c, diag(sigma^2 + v))   (mu integrated out under q)
The population prior on mu breaks the shift symmetry z -> z + f_c - f_c' that makes single-class sets ambiguous for any
shift-equivariant estimator (e.g. centering), and q(theta) pools the references' evidence at the level of the set.

Training composition (--comp): rand = references drawn uniformly from the subject's clips (as at test time);
dir = composition drawn from Dir(1) (uniform on the simplex) independently of the query, then references of those classes.
Both are applied identically to every method that uses references.

usage: python came.py --data ravdess --method came --seed 0 --comp dir   -> $EMO_ROOT/runs/came/*.json
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

PROC, FEAT, LLM, OUT = f"{_data_root}/datasets/proc", f"{_data_root}/features/clip", f"{_data_root}/features/llm", f"{_data_root}/runs/came"
dev = "cuda"
LAB = {"ravdess": ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"],
       "cremad": ["angry", "disgust", "fear", "happy", "neutral", "sad"]}
NEUTRAL = {"ravdess": 0, "cremad": 4}


def load(data, fold):
    return np.concatenate([np.load(f"{FEAT}/{data}_frozen.npy"), np.load(f"{FEAT}/{data}_ft_f{fold}.npy")], 1)


def proto_source(data, src, seed):
    """class vectors T (C x D_t) before centering: LLM states / LLM output-layer rows / random / shuffled LLM / one-hot"""
    C = len(LAB[data])
    if src == "onehot":
        return np.eye(C)
    if src == "rand":
        return np.random.default_rng(seed).normal(size=(C, 1536))
    z = np.load(f"{LLM}/{'Qwen2.5-0.5B-Instruct' if src == 'llm05' else 'Qwen2.5-1.5B-Instruct'}.npz")
    words = list(z["words"])
    T = z["unemb" if src == "unemb" else "state"][[words.index(w) for w in LAB[data]]].astype(np.float64)
    if src == "shuf":  # language vectors assigned to the wrong classes (a fixed derangement)
        T = np.roll(T, 1, 0)
    return T


def prototypes(data, src, seed):
    T = proto_source(data, src, seed)
    T = T - T.mean(0)
    return torch.tensor(T / np.linalg.norm(T, axis=1).mean(), dtype=torch.float32)


def mlp(D, C, h=512):
    return nn.Sequential(nn.Dropout(0.1), nn.Linear(D, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, C))


def ctx_mean(x):
    return x[:, 1:].mean(1, keepdim=True) if x.shape[1] > 1 else torch.zeros_like(x[:, :1])


# ------------------------------------------------------------------------------------------------ CAME
class CAME(nn.Module):
    """proto: None (free prototypes) | (T, 'free') LLM vectors through a free linear adapter |
    (T, 'geo') geometry-locked: f = s * G Q^T with G the Gram-preserving coordinates of T and Q orthonormal"""

    def __init__(self, D, C, proto=None, d=256, iters=10, variant="full"):
        super().__init__()
        self.joint = variant.startswith("joint")
        deep = variant.endswith("deep")
        self.lift = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, 512), nn.GELU(), nn.Linear(512, d)) if deep else nn.Sequential(nn.Dropout(0.1), nn.Linear(D, d))
        variant = {"jointdeep": "full", "joint": "full", "deep": "full"}.get(variant, variant)
        self.mode = "free" if proto is None else proto[1]
        if proto is None:
            self.f_free = nn.Parameter(torch.randn(C, d) * 0.1)
        elif self.mode == "free":
            self.register_buffer("T", proto[0]); self.A = nn.Linear(proto[0].shape[1], d, bias=False)
        else:
            U, S, _ = torch.linalg.svd(proto[0], full_matrices=False)
            self.register_buffer("G", (U * S)[:, : C - 1])  # C x (C-1), G G^T = T T^T
            self.Q = nn.utils.parametrizations.orthogonal(nn.Linear(C - 1, d, bias=False))
            self.log_scale = nn.Parameter(torch.zeros(()))
        self.log_s2 = nn.Parameter(torch.zeros(()))
        self.log_lam = nn.Parameter(torch.zeros(d))
        self.log_alpha = nn.Parameter(torch.zeros(()))
        self.prior = nn.Parameter(torch.zeros(C))
        self.iters, self.variant, self.C = iters, variant, C

    @property
    def f(self):
        if self.mode == "geo":
            return self.log_scale.exp() * self.Q(self.G)
        f = self.f_free if hasattr(self, "f_free") else self.A(self.T)
        return f - f.mean(0)

    def sqdist(self, z, m, var):  # sum_d (z - m - f_c)^2 / var  -> (B, n, C)
        r = z[:, :, None] - m[:, None, None] - self.f[None, None]
        return (r ** 2 / var[:, None, None]).sum(-1)

    def infer(self, x):
        z = self.lift(x)
        B, n, d = z.shape
        s2, lam, logpi = self.log_s2.exp(), self.log_lam.exp(), self.prior.log_softmax(0)
        if self.joint:
            return self.infer_joint(z)
        zr, K = z[:, 1:], n - 1
        m = torch.zeros(B, d, device=z.device)
        logr = logpi.expand(B, K, self.C)
        if self.variant != "nosubj" and K:
            if self.variant == "noea":  # no explaining away: shrunk reference mean
                m = zr.sum(1) / (K + lam)
            else:
                elog = logpi.expand(B, self.C)
                for _ in range(self.iters):
                    logr = (elog[:, None] - 0.5 * self.sqdist(zr, m, s2.expand(B, d))).log_softmax(-1)
                    r = logr.exp()
                    if self.variant == "full":  # q(theta) = Dir(alpha + n_c)
                        a = self.log_alpha.exp() + r.sum(1)
                        elog = torch.digamma(a) - torch.digamma(a.sum(-1, keepdim=True))
                    m = (zr - r @ self.f).sum(1) / (K + lam)
        v = s2 / (K + lam) if self.variant != "nosubj" else torch.zeros_like(lam)
        lq = logpi - 0.5 * self.sqdist(z[:, :1], m, (s2 + v).expand(B, d))  # (B,1,C)
        return torch.cat([lq, logr], 1) if K else lq, m, z

    def infer_joint(self, z):
        """exact-model VB over the whole set: the query's label is a latent with prior pi, the references' with
        q(theta); mu is inferred from all members, and the query prediction is its own q(y_q)"""
        B, n, d = z.shape
        s2, lam, logpi = self.log_s2.exp(), self.log_lam.exp(), self.prior.log_softmax(0)
        m = torch.zeros(B, d, device=z.device)
        elog = logpi.expand(B, self.C)
        for _ in range(self.iters if n > 1 else 1):
            pri = torch.cat([logpi.expand(B, 1, self.C), elog[:, None].expand(B, n - 1, self.C)], 1)
            logr = (pri - 0.5 * self.sqdist(z, m, s2.expand(B, d))).log_softmax(-1)
            r = logr.exp()
            if n > 1:
                a = self.log_alpha.exp() + r[:, 1:].sum(1)
                elog = torch.digamma(a) - torch.digamma(a.sum(-1, keepdim=True))
            m = (z - r @ self.f).sum(1) / (n + lam)
        if n == 1:  # no references: marginal likelihood under the population prior on mu
            logr = (logpi - 0.5 * self.sqdist(z, m, (s2 + s2 / lam).expand(B, d))).log_softmax(-1)
        return logr, m, z

    def forward(self, x, yr=None):
        return self.infer(x)[0]


# --------------------------------------------------------------------------------------------- baselines
class Plain(nn.Module):
    """lp / mlp: no context.  ctr / ctr_mlp: x - mean(R).  sstd: (x - mean(R)) / std(R) (subject statistics, as in
    test-time BN adaptation).  cbc: class-balanced centering, references re-weighted by the inverse of their predicted
    class mass (as in DELTA's dynamic online re-weighting).  pl: pseudo-label correction.  fuse: prediction fusion.
    cat: [x ; x - mean(R)].  orc: centering corrected with the references' TRUE labels (label-oracle upper reference)."""

    def __init__(self, D, C, kind):
        super().__init__()
        self.kind = kind
        lin = kind in ("lp", "ctr", "sstd", "cbc", "orc")
        self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, C)) if lin else mlp(2 * D if kind == "cat" else D, C)
        if kind == "fuse":
            self.raw = mlp(D, C)
        if kind in ("pl", "cbc"):
            self.p0 = mlp(D, C)
        if kind in ("pl", "orc"):
            self.M = nn.Parameter(torch.zeros(C, D))

    def forward(self, x, yr=None):
        k = self.kind
        if k in ("lp", "mlp"):
            return self.net(x)
        if x.shape[1] == 1 and k not in ("fuse", "cat"):  # no references: the subject estimate is the population mean 0
            return self.p0(x) if k == "pl" else self.net(x)
        if k in ("ctr", "ctr_mlp"):
            return self.net(x - ctx_mean(x))
        if k == "sstd":
            K = x.shape[1] - 1  # reference variance shrunk towards the population variance 1 (features are standardised)
            s2 = (K * x[:, 1:].var(1, keepdim=True, unbiased=False) + 1) / (K + 1)
            return self.net((x - ctx_mean(x)) / s2.sqrt())
        if k == "cat":
            return self.net(torch.cat([x, x - ctx_mean(x)], -1))
        if k == "fuse":
            p = 0.5 * self.raw(x).softmax(-1) + 0.5 * self.net(x - ctx_mean(x)).softmax(-1)
            return p.clamp_min(1e-8).log()
        if k == "cbc":
            p = self.p0(x[:, 1:]).softmax(-1)
            w = (p / p.sum(1, keepdim=True)).sum(-1, keepdim=True)  # sum_c p_kc / n_c
            return self.net(x - (w * x[:, 1:]).sum(1, keepdim=True) / w.sum(1, keepdim=True))
        if k == "pl":
            res = x[:, 1:] - self.p0(x[:, 1:]).softmax(-1) @ self.M
            return self.net(x - res.mean(1, keepdim=True))
        if k == "orc":
            return self.net(x - (x[:, 1:] - self.M[yr]).mean(1, keepdim=True))
        raise ValueError(k)

    def aux(self, x, y):
        """pl / cbc: the no-context head p0 is supervised; pl / orc: M are the class means of subject-centred features"""
        loss = 0
        if self.kind in ("pl", "cbc"):
            loss = loss + F.cross_entropy(self.p0(x).flatten(0, 1), y.flatten())
        if self.kind in ("pl", "orc") and x.shape[1] > 1:
            loss = loss + ((x - ctx_mean(x) - self.M[y]) ** 2).mean()
        return loss


class SetTF(nn.Module):
    def __init__(self, D, C, d=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, d))
        self.role = nn.Parameter(torch.randn(2, d) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, C))

    def forward(self, x, yr=None):
        role = torch.ones(x.shape[1], dtype=torch.long, device=x.device); role[0] = 0
        return self.out(self.tf(self.inp(x) + self.role[role]))


PROTO = {"came_llm": ("llm", "free"), "came_geo": ("llm", "geo"), "came_georand": ("rand", "geo"),
         "came_geoshuf": ("shuf", "geo"), "came_onehot": ("onehot", "geo"), "came_llm05": ("llm05", "free")}


def build(method, D, C, data, seed):
    if method.startswith("came"):
        variant = {"came_nodir": "nodir", "came_noea": "noea", "came_nosubj": "nosubj", "came_joint": "joint",
                   "came_deep": "deep", "came_jointdeep": "jointdeep"}.get(method, "full")
        p = PROTO.get(method)
        return CAME(D, C, None if p is None else (prototypes(data, p[0], seed), p[1]), variant=variant)
    if method == "settf":
        return SetTF(D, C)
    return Plain(D, C, method)


# --------------------------------------------------------------------------------------------- episodes
def contexts(meta, rows, K, mode, rng, pool_rows, j=0):
    """label-free: rand / xsent / contam.  label-controlled diagnostics: same / other / neu / mix(j) / dir (training)."""
    act, y, sen = meta.actor.values, meta.label.values, meta.sentence.values
    C = meta.label.nunique()
    out = np.zeros((len(rows), K), int)
    for i, r in enumerate(rows):
        pool = pool_rows[(act[pool_rows] == act[r]) & (pool_rows != r)]
        if mode == "dir":  # composition ~ Dir(1), independent of the query label
            n = rng.multinomial(K, rng.dirichlet(np.ones(C)))
            out[i] = np.concatenate([rng.choice(pool[y[pool] == c], n[c], replace=True) for c in range(C) if n[c]])
            continue
        if mode == "past":  # the K most recent earlier clips of the same person (natural temporal composition)
            od = meta["order"].values
            pp = pool[od[pool] < od[r]]
            pp = pp[np.argsort(od[pp])][-K:] if len(pp) else rng.choice(pool, K, replace=len(pool) < K)
            out[i] = pp if len(pp) == K else np.concatenate([pp, rng.choice(pp, K - len(pp), replace=True)])
            continue
        if mode == "skew":  # dominant expression chosen uniformly at random, independently of the query: 4 + 2 random
            cs = np.unique(y[pool]); pd_ = pool[y[pool] == rng.choice(cs)]
            out[i] = np.concatenate([rng.choice(pd_, 4, replace=len(pd_) < 4), rng.choice(pool, K - 4, replace=False)])
            continue
        if mode == "neu":  # neutral-dominated: 4 neutral references, the rest random
            pn = pool[y[pool] == NEUTRAL[meta.attrs["data"]]]
            out[i] = np.concatenate([rng.choice(pn, 4, replace=len(pn) < 4), rng.choice(pool, K - 4, replace=False)])
            continue
        if mode == "xsent":
            pool = pool[sen[pool] != sen[r]]
        if mode == "same":
            pool = pool[y[pool] == y[r]]
        if mode == "other":
            e = rng.choice([c for c in np.unique(y[pool]) if c != y[r]])
            pool = pool[y[pool] == e]
        if mode == "mix":
            ps, po = pool[y[pool] == y[r]], pool[y[pool] != y[r]]
            out[i] = np.concatenate([rng.choice(ps, j, replace=len(ps) < j), rng.choice(po, K - j, replace=len(po) < K - j)])
            continue
        out[i] = rng.choice(pool, K, replace=len(pool) < K)
        if mode == "contam":
            others = pool_rows[act[pool_rows] != act[r]]
            out[i, :K // 3] = rng.choice(others, K // 3, replace=False)
    return out


def train(model, X, meta, rows, a, rng):
    y = torch.tensor(meta.label.values, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    for it in range(a.steps):
        q = rng.choice(rows, a.bs)
        K = int(rng.integers(0, a.kmax + 1))
        idx = np.concatenate([q[:, None], contexts(meta, q, K, a.comp, rng, rows)], 1) if K else q[:, None]
        idx = torch.from_numpy(idx).to(dev)
        xb, yb = X[idx], y[idx]
        if a.method in ("lp", "mlp"):
            xb, yb = xb.reshape(-1, 1, xb.shape[-1]), yb.reshape(-1, 1)
        lg = model(xb, yb[:, 1:])
        if a.sup == "all":
            loss = F.cross_entropy(lg.flatten(0, 1), yb.flatten(), label_smoothing=0.1)
        else:
            loss = F.cross_entropy(lg[:, 0], yb[:, 0], label_smoothing=0.1)
        if isinstance(model, Plain):
            loss = loss + model.aux(xb, yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def predict(model, X, ctx, rows, y):
    model.eval()
    q = torch.from_numpy(rows).to(dev)[:, None]
    idx = q if ctx is None else torch.cat([q, torch.from_numpy(ctx).to(dev)], 1)
    yy = y[idx[:, 1:]]
    out = torch.cat([model(X[idx[i:i + 512]], yy[i:i + 512])[:, 0].float().softmax(-1) for i in range(0, len(idx), 512)])
    model.train()
    return out.cpu().numpy()


@torch.no_grad()
def subject_error(model, X, ctx, rows, meta):
    """relative squared error of m against the label oracle mu*_s = mean_i (z_i - f_{y_i}) over all of the subject's test
    clips, for the model's estimate and for the shrunk mean of the SAME references (query excluded in both)"""
    model.eval()
    idx = torch.cat([torch.from_numpy(rows).to(dev)[:, None], torch.from_numpy(ctx).to(dev)], 1)
    _, m, z = model.infer(X[idx])
    res = model.lift(X[torch.from_numpy(rows).to(dev)]) - model.f[torch.tensor(meta.label.values[rows], device=dev)]
    act = meta.actor.values[rows]
    mu = torch.stack([res[torch.from_numpy(act == act[i]).to(dev)].mean(0) for i in range(len(rows))])
    mean_est = z[:, 1:].sum(1) / (z.shape[1] - 1 + model.log_lam.exp())
    den = ((mu - mu.mean(0)) ** 2).sum(1).mean()
    model.train()
    return float(((m - mu) ** 2).sum(1).mean() / den), float(((mean_est - mu) ** 2).sum(1).mean() / den)


def metrics(p, y, C):
    pred = p.argmax(1)
    conf = p.max(1); bins = np.linspace(0, 1, 16); ece = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            ece += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
    return dict(acc=float(np.mean(pred == y) * 100), uar=float(np.mean([np.mean(pred[y == c] == c) for c in range(C) if np.any(y == c)]) * 100),
                nll=float(-np.log(p[np.arange(len(y)), y] + 1e-8).mean()), ece=float(ece * 100))


MAIN = ["rand", "xsent", "contam", "same", "other", "neu"]
NCTX = 3


def run(a):
    meta = pd.read_csv(f"{PROC}/{a.data}_meta.csv"); meta.attrs["data"] = a.data
    C = meta.label.nunique()
    res = dict(args=vars(a), folds={})
    for fold in sorted(meta.fold.unique()):
        torch.manual_seed(a.seed * 100 + fold); rng = np.random.default_rng(a.seed * 100 + fold)
        X = load(a.data, fold)
        tr, te = np.where(meta.fold != fold)[0], np.where(meta.fold == fold)[0]
        if a.frac < 1:  # low-data: a stratified fraction of the training clips
            tr = np.sort(np.concatenate([g.sample(n=max(1, round(a.frac * len(g))), random_state=a.seed).index.values
                                         for _, g in meta.iloc[tr].groupby(["actor", "label"])]))
        X = torch.tensor((X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6), dtype=torch.float32, device=dev)
        yall = torch.tensor(meta.label.values, device=dev)
        model = train(build(a.method, X.shape[1], C, a.data, a.seed).to(dev), X, meta, tr, a, rng)
        yt = meta.label.values[te]
        P = [predict(model, X, None, te, yall)]
        r = {"K0": metrics(P[0], yt, C)}
        for mode in MAIN + [f"mix{j}" for j in range(0, a.kctx + 1)]:
            ps = []
            for c_seed in range(NCTX):
                erng = np.random.default_rng(10000 * c_seed + 1000 + fold)  # shared by all methods / seeds
                md, j = ("mix", int(mode[3:])) if mode.startswith("mix") else (mode, 0)
                ps.append(predict(model, X, contexts(meta, te, a.kctx, md, erng, te, j), te, yall))
            if mode in MAIN:
                P += ps
            ms = [metrics(p, yt, C) for p in ps]
            r[mode] = {k: float(np.mean([m[k] for m in ms])) for k in ["acc", "uar", "nll", "ece"]}
            r[mode]["acc_ctxsd"] = float(np.std([m["acc"] for m in ms]))
        for k in [1, 2, 4, 8, 16]:
            r[f"K{k}"] = metrics(predict(model, X, contexts(meta, te, k, "rand", np.random.default_rng(1000 + fold), te), te, yall), yt, C)
        if isinstance(model, CAME) and model.variant != "nosubj":
            for mode in ["rand", "same"]:
                c = contexts(meta, te, a.kctx, mode, np.random.default_rng(1000 + fold), te)
                r[f"muerr_{mode}"] = dict(zip(["em", "mean"], subject_error(model, X, c, te, meta)))
            r["alpha"] = {"value": float(model.log_alpha.exp())}
        res["folds"][int(fold)] = r
        # predictions per draw: [K0, rand x3, xsent x3, contam x3, same x3, other x3, neu x3]
        np.save(f"{OUT}/pred/{a.data}_{a.tag}_s{a.seed}_f{fold}.npy", np.stack(P).astype(np.float16))
        print(fold, {k: round(v["acc"], 2) for k, v in r.items() if "acc" in v}, flush=True)
    f0 = next(iter(res["folds"].values()))
    res["mean"] = {k: {m: float(np.mean([f[k][m] for f in res["folds"].values()])) for m in f0[k]} for k in f0}
    return res


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ravdess")
    ap.add_argument("--method", default="came")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comp", default="dir", choices=["rand", "dir"])
    ap.add_argument("--sup", default="all", choices=["all", "query"])
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--kctx", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    a = ap.parse_args(argv)
    a.tag = f"{a.method}_{a.comp}" + ("_q" if a.sup == "query" else "") + (f"_fr{a.frac:g}" if a.frac < 1 else "")
    return a


if __name__ == "__main__":
    a = get_args()
    os.makedirs(f"{OUT}/pred", exist_ok=True)
    t0 = time.time(); res = run(a); res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(f"{OUT}/{a.data}_{a.tag}_s{a.seed}.json", "w"))
    print("RESULT", a.data, a.tag, a.seed, json.dumps({k: round(v["acc"], 2) for k, v in res["mean"].items() if "acc" in v}), flush=True)
