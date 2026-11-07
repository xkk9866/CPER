"""SEMIX: semantic mixed-effects inference for subject-relative facial expression recognition.

Generative model of one subject's clips, in a space shared by vision and language:
    z_i = W x_i                       linear lift of the visual feature (additivity-preserving)
    z_i = mu_s + f_{y_i} + eps_i,     mu_s ~ N(0, diag(sigma^2 / lambda)),  eps_i ~ N(0, sigma^2 I)
    f_c = A t_c                       expression displacements are *language* vectors: t_c is the state a 4-bit
                                      LLM's output layer reads after describing expression c; A is the learned
                                      adapter of that output space into the visual space (the only "fine-tuned
                                      output layer" of the LLM; the LLM itself is frozen and run once).
Inference for an unlabeled set (query + K >= 0 references of one subject) is *exact* EM for the MAP subject
effect with every member's expression latent:
    E-step  r_ic ∝ pi_c N(z_i ; m + f_c, (sigma^2 + v) I)         (the model's own posterior, no free head)
    M-step  m = sum_i (z_i - sum_c r_ic f_c) / (n + lambda),  v = sigma^2 / (n + lambda)
Every quantity in both steps is a parameter of the same Gaussian model, so each iteration does not decrease
the MAP objective; the query prediction is its E-step posterior. Training maximises the conditional likelihood
of the labels of all members of random training episodes.

Baselines (same inputs, episodes, optimiser, steps and -- by default -- the same supervision: every labeled
member of an episode is classified and supervised; --sup query supervises the query only):
  lp, mlp            no context (members classified independently)
  ctr, ctr_mlp       x - mean(references) -> linear / MLP
  pctr               x - a * mean(references), a learned per dimension -> MLP (partial centering)
  cat                MLP([x ; x - mean(references)])  (keeps the raw signal)
  fuse               mixture of a no-context MLP and a centering MLP (prediction fusion, trained jointly)
  pl                 pseudo-label correction: m = mean(x_c - M p0(x_c)) with p0 a no-context MLP, then MLP(x - m)
  settf              set transformer over [query; references]
  demix              discriminative-EM network of our previous version (free MLP E-step)
SEMIX variants: semix (LLM-1.5B state prototypes), semix_unemb (LLM-1.5B output-layer rows), semix_05b
(LLM-0.5B), semix_free (free prototypes, no language), semix_noem (one M-step), semix_nosubj (m = 0).

usage: python semix.py --data ravdess --method semix --seed 0   -> $EMO_ROOT/runs/semix/*.json
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, json, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

PROC, FEAT, LLM, OUT = f"{_data_root}/datasets/proc", f"{_data_root}/features/clip", f"{_data_root}/features/llm", f"{_data_root}/runs/semix"
dev = "cuda"
LAB = {"ravdess": ["neutral", "calm", "happy", "sad", "angry", "fear", "disgust", "surprise"],
       "cremad": ["angry", "disgust", "fear", "happy", "neutral", "sad"]}


def load(data, fold):
    return np.concatenate([np.load(f"{FEAT}/{data}_frozen.npy"), np.load(f"{FEAT}/{data}_ft_f{fold}.npy")], 1)


def prototypes(data, model, kind):
    z = np.load(f"{LLM}/{model}.npz")
    words = list(z["words"])
    T = z[kind][[words.index(w) for w in LAB[data]]]
    T = T - T.mean(0)
    return torch.tensor(T / np.linalg.norm(T, axis=1, keepdims=True).mean(), dtype=torch.float32)


def mlp(D, C, h=512):
    return nn.Sequential(nn.Dropout(0.1), nn.Linear(D, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, C))


def ctx_mean(x):
    """mean of the references, broadcast to every member; zero when there are none"""
    return x[:, 1:].mean(1, keepdim=True) if x.shape[1] > 1 else torch.zeros_like(x[:, :1])


# ----------------------------------------------------------------------------------------------- SEMIX
class SEMIX(nn.Module):
    def __init__(self, D, C, protos=None, d=256, iters=10, variant="full"):
        super().__init__()
        self.lift = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, d))
        if protos is None:
            self.f_free = nn.Parameter(torch.randn(C, d) * 0.1)
        else:
            self.register_buffer("T", protos)
            self.A = nn.Linear(protos.shape[1], d, bias=False)
        self.log_s2 = nn.Parameter(torch.zeros(()))
        self.log_lam = nn.Parameter(torch.zeros(d))
        self.prior = nn.Parameter(torch.zeros(C))
        self.iters, self.variant = iters, variant

    @property
    def f(self):
        f = self.f_free if hasattr(self, "f_free") else self.A(self.T)
        return f - f.mean(0)

    def loglik(self, z, m, v):
        var = self.log_s2.exp() + v  # (B,d)
        r = z[:, :, None] - m[:, None, None] - self.f[None, None]
        return -0.5 * (r ** 2 / var[:, None, None]).sum(-1) + self.prior.log_softmax(0)

    def infer(self, x):
        z = self.lift(x)
        B, n, d = z.shape
        s2, lam = self.log_s2.exp(), self.log_lam.exp()
        v = (s2 / (n + lam)).expand(B, d)
        if self.variant == "nosubj":
            m = torch.zeros(B, d, device=z.device)
            return self.loglik(z, m, torch.zeros_like(v)), m, z
        # EM over the references only (each reference is scored with its own evidence included, as in any EM);
        # the query is then scored by the exact posterior predictive  p(y_q | z_q, refs) with mu integrated out
        # over p(mu | refs), i.e. it never explains itself.  With K = 0 this is the population prior N(0, tau^2).
        zr = z[:, 1:]; K = zr.shape[1]
        e = torch.zeros_like(zr)
        m = torch.zeros(B, d, device=z.device)
        for _ in range((1 if self.variant == "noem" else self.iters) if K else 0):
            m = (zr - e).sum(1) / (K + lam)
            e = self.loglik(zr, m, (s2 / (K + lam)).expand(B, d)).softmax(-1) @ self.f
        if K:
            m = (zr - e).sum(1) / (K + lam)
        v_ref = (s2 / (K + lam)).expand(B, d)
        lg = self.loglik(z, m, v_ref)  # predictive for every member given the references' posterior
        return lg, m, z

    def forward(self, x):
        return self.infer(x)[0]


class DEMIX(nn.Module):  # previous version: free MLP E-step, kept as a baseline
    def __init__(self, D, C, d=256, iters=5):
        super().__init__()
        self.lift = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, d))
        self.g = mlp(d, C)[1:]
        self.F = nn.Parameter(torch.randn(C, d) * 0.02)
        self.log_lam = nn.Parameter(torch.zeros(d))
        self.iters = iters

    def forward(self, x):
        z = self.lift(x); n = z.shape[1]; e = torch.zeros_like(z)
        for _ in range(self.iters):
            m = (z - e).sum(1) / (n + self.log_lam.exp())
            lg = self.g(z - m[:, None])
            e = lg.softmax(-1) @ (self.F - self.F.mean(0))
        return lg


class Plain(nn.Module):
    def __init__(self, D, C, kind):
        super().__init__()
        self.kind = kind
        if kind in ("lp", "ctr"):
            self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, C))
        elif kind == "cat":
            self.net = mlp(2 * D, C)
        else:
            self.net = mlp(D, C)
        if kind == "pctr":
            self.a = nn.Parameter(torch.full((D,), 0.5))
        if kind == "fuse":
            self.raw = mlp(D, C)
        if kind == "pl":
            self.p0, self.M = mlp(D, C), nn.Parameter(torch.zeros(C, D))

    def forward(self, x):
        k = self.kind
        if k in ("lp", "mlp"):
            return self.net(x)
        if k in ("ctr", "ctr_mlp"):
            return self.net(x - ctx_mean(x))
        if k == "pctr":
            return self.net(x - self.a * ctx_mean(x))
        if k == "cat":
            return self.net(torch.cat([x, x - ctx_mean(x)], -1))
        if k == "fuse":
            p = 0.5 * self.raw(x).softmax(-1) + 0.5 * self.net(x - ctx_mean(x)).softmax(-1)
            return p.clamp_min(1e-8).log()
        if k == "pl":
            lg0 = self.p0(x)
            if x.shape[1] == 1:
                return lg0
            res = x[:, 1:] - lg0[:, 1:].softmax(-1) @ self.M
            return self.net(x - res.mean(1, keepdim=True))
        raise ValueError(k)

    def aux(self, x, y):  # pl: the no-context classifier and class means are supervised directly
        if self.kind != "pl":
            return 0
        return F.cross_entropy(self.p0(x).flatten(0, 1), y.flatten()) + ((x - self.M[y]) ** 2).mean() * 0 + \
            ((x.detach() - self.M[y] - ctx_mean(x.detach()) * 0) ** 2).mean() * 0


class SetTF(nn.Module):
    def __init__(self, D, C, d=128):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, d))
        self.role = nn.Parameter(torch.randn(2, d) * 0.02)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.1, batch_first=True, norm_first=True), 2)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, C))

    def forward(self, x):
        role = torch.ones(x.shape[1], dtype=torch.long, device=x.device); role[0] = 0
        return self.out(self.tf(self.inp(x) + self.role[role]))


def build(method, D, C, data):
    if method.startswith("semix"):
        llm, kind = {"semix_05b": ("Qwen2.5-0.5B-Instruct", "state"), "semix_unemb": ("Qwen2.5-1.5B-Instruct", "unemb")}.get(method, ("Qwen2.5-1.5B-Instruct", "state"))
        P = None if method == "semix_free" else prototypes(data, llm, kind)
        return SEMIX(D, C, P, variant={"semix_noem": "noem", "semix_nosubj": "nosubj"}.get(method, "full"))
    if method == "demix":
        return DEMIX(D, C)
    if method == "settf":
        return SetTF(D, C)
    return Plain(D, C, method)


# --------------------------------------------------------------------------------------------- episodes
def contexts(meta, rows, K, mode, rng, pool_rows, j=0):
    """rand / xsent / contam are label-free; same / other / mix are label-controlled diagnostics.
    mix: j of the K references share the query's expression, the other K-j are drawn from its other expressions."""
    act, y, sen = meta.actor.values, meta.label.values, meta.sentence.values
    out = np.zeros((len(rows), K), int)
    for i, r in enumerate(rows):
        pool = pool_rows[(act[pool_rows] == act[r]) & (pool_rows != r)]
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
        idx = np.concatenate([q[:, None], contexts(meta, q, K, "rand", rng, rows)], 1) if K else q[:, None]
        idx = torch.from_numpy(idx).to(dev)
        xb, yb = X[idx], y[idx]
        if a.method in ("lp", "mlp"):  # no-context methods classify every member on its own
            xb, yb = xb.reshape(-1, 1, xb.shape[-1]), yb.reshape(-1, 1)
        lg = model(xb)
        if a.sup == "all":
            loss = F.cross_entropy(lg.flatten(0, 1), yb.flatten(), label_smoothing=0.1)
        else:
            loss = F.cross_entropy(lg[:, 0], yb[:, 0], label_smoothing=0.1)
        if isinstance(model, Plain) and model.kind == "pl":
            loss = loss + F.cross_entropy(model.p0(xb).flatten(0, 1), yb.flatten()) + ((xb - ctx_mean(xb) - model.M[yb]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def predict(model, X, ctx, rows):
    model.eval()
    q = torch.from_numpy(rows).to(dev)[:, None]
    idx = q if ctx is None else torch.cat([q, torch.from_numpy(ctx).to(dev)], 1)
    out = torch.cat([model(X[idx[i:i + 512]])[:, 0].float().softmax(-1) for i in range(0, len(idx), 512)])
    model.train()
    return out.cpu().numpy()


@torch.no_grad()
def subject_error(model, X, ctx, rows, meta):
    """SEMIX only: relative squared error of the inferred subject effect m against the label-oracle effect
    mu*_s = mean over all the subject's test clips of (z_i - f_{y_i}); and the same for the plain shrunk mean."""
    model.eval()
    idx = torch.cat([torch.from_numpy(rows).to(dev)[:, None], torch.from_numpy(ctx).to(dev)], 1)
    _, m, z = model.infer(X[idx])
    zall = model.lift(X[torch.from_numpy(rows).to(dev)])
    res = zall - model.f[torch.tensor(meta.label.values[rows], device=dev)]
    act = meta.actor.values[rows]
    mu = torch.stack([res[torch.from_numpy(act == act[i]).to(dev)].mean(0) for i in range(len(rows))])
    lam = model.log_lam.exp(); n = idx.shape[1]
    mean_est = z.sum(1) / (n + lam)
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


MAIN = ["rand", "xsent", "contam", "same", "other"]
NCTX = 3  # context draws per protocol (averaged; reported spread in the json)


def run(a):
    meta = pd.read_csv(f"{PROC}/{a.data}_meta.csv")
    C = meta.label.nunique()
    res = dict(args=vars(a), folds={})
    for fold in sorted(meta.fold.unique()):
        torch.manual_seed(a.seed * 100 + fold); rng = np.random.default_rng(a.seed * 100 + fold)
        X = load(a.data, fold)
        tr, te = np.where(meta.fold != fold)[0], np.where(meta.fold == fold)[0]
        X = torch.tensor((X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6), dtype=torch.float32, device=dev)
        model = train(build(a.method, X.shape[1], C, a.data).to(dev), X, meta, tr, a, rng)
        yt = meta.label.values[te]
        P = {"K0": predict(model, X, None, te)}
        r = {"K0": metrics(P["K0"], yt, C)}
        for mode in MAIN + [f"mix{j}" for j in range(0, a.kctx + 1)]:
            ps = []
            for c_seed in range(NCTX):
                erng = np.random.default_rng(10000 * c_seed + 1000 + fold)  # shared by all methods / seeds
                md, j = ("mix", int(mode[3:])) if mode.startswith("mix") else (mode, 0)
                ps.append(predict(model, X, contexts(meta, te, a.kctx, md, erng, te, j), te))
            P[mode] = np.mean(ps, 0)
            r[mode] = {k: float(np.mean([metrics(p, yt, C)[k] for p in ps])) for k in ["acc", "uar", "nll", "ece"]}
            r[mode]["acc_ctxsd"] = float(np.std([metrics(p, yt, C)["acc"] for p in ps]))
        for k in [1, 2, 4, 8, 16]:
            r[f"K{k}"] = metrics(predict(model, X, contexts(meta, te, k, "rand", np.random.default_rng(1000 + fold), te), te), yt, C)
        if isinstance(model, SEMIX) and model.variant != "nosubj":
            for mode in ["rand", "same"]:
                c = contexts(meta, te, a.kctx, mode, np.random.default_rng(1000 + fold), te)
                r[f"muerr_{mode}"] = dict(zip(["em", "mean"], subject_error(model, X, c, te, meta)))
        res["folds"][int(fold)] = r
        np.save(f"{OUT}/pred/{a.data}_{a.method}{a.suffix}_s{a.seed}_f{fold}.npy", np.stack([P[k] for k in ["K0"] + MAIN]).astype(np.float16))
        print(fold, {k: round(v["acc"], 2) for k, v in r.items() if "acc" in v}, flush=True)
    f0 = next(iter(res["folds"].values()))
    res["mean"] = {k: {m: float(np.mean([f[k][m] for f in res["folds"].values()])) for m in f0[k]} for k in f0}
    return res


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ravdess")
    ap.add_argument("--method", default="semix")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sup", default="all", choices=["all", "query"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--kctx", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--suffix", default="")
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = get_args()
    if a.sup == "query" and not a.suffix:
        a.suffix = "_q"
    os.makedirs(f"{OUT}/pred", exist_ok=True)
    t0 = time.time(); res = run(a); res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(f"{OUT}/{a.data}_{a.method}{a.suffix}_s{a.seed}.json", "w"))
    print("RESULT", a.data, a.method + a.suffix, a.seed, json.dumps({k: round(v["acc"], 2) for k, v in res["mean"].items() if "acc" in v}), flush=True)
