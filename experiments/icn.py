"""ICN: In-Context Normalization -- a neural normalizer that reads a person's unlabeled clips and emits the transform.

Problem.  A recognizer sees a query clip and K unlabeled clips of the same person.  Personal normalization (BN-adapt and
its descendants) subtracts the reference statistics as if they described the person, but the statistics also contain
the emotion composition of the references: a set of six angry clips centres anger away.  Hand-derived fixes (CPN)
enumerate compositions under a Gaussian model and need calibration grids, quadrature and fitted combiners.

Method.  ICN replaces the hand-derived inference by a learned one.  A small set network f_phi reads the episode
U = {x_0 (query), x_1..x_K} (frozen features) together with the shared classifier's own evidence L_i = log p_src(. | x_i)
and emits, for every clip, ONE normalization transform
        x~_i = (x_i - s_i (c_i + beta_i)) * v_i^(-rho_i / 2),
        c_i = sum_k a_ik x_k        (attention centre: which clips to subtract),
        v_i = sum_k a_ik (x_k - c_i)^2  (its scatter),
        beta_i = b_i^T F, 1^T b_i = 0  (composition compensation in the emotion-displacement subspace span(F)),
        s_i, rho_i in [0, 1]           (how much of the centre / scale to trust),
and the UNCHANGED source classifier predicts from x~_i.  The family contains the source model (s = rho = 0), BN-adapt
(a uniform, s = rho = 1, b = 0) and composition-corrected centring (b = -theta); the network chooses per episode.
It is trained by maximum likelihood on episodes of the training people under a broad composition prior (the same
Dirichlet hyper-prior that CPN marginalises), with two provisions that make the learned rule a rule for NEW people:
  * cross-fitting: the training people are split into R groups; the evidence L_i of a group's episodes comes from an
    auxiliary classifier trained WITHOUT that group, so the normalizer always sees the classifier's behaviour on
    people the classifier has not seen (the deployment condition);
  * orbit augmentation: every training episode is moved by a random person transform (offset ~ the between-person
    covariance of the training people, log-expressivity ~ their spread), i.e. episodes are drawn from a kernel-density
    estimate of the person prior rather than from the 20-90 training identities only.
Prediction is one forward pass: no gradients, no grids, no calibration, no combiner.  The population minimizer of the
episode log loss is the posterior predictive p(y | U) under the sampled hierarchical prior (amortized Bayesian
inference), and since BN-adapt and Source are members of the family the optimum is no worse than either in expected log
score on episodes of the training prior.

Everything else (source network, protocols, metrics, baselines, CPN, outputs) is cmn.py's, imported and reused so that
every method shares the same classifier and the same reference draws.

usage:  python icn.py --data ravdess --method icn --seed 0 [--feat l14]   -> $CMN_OUT/{data}_{tag}_s{seed}.json
        python icn.py --data meld --feat avt --stream                      -> $CMN_OUT/stream_{data}_{feat}_s{seed}.json
        python icn.py --data ravdess --profile                             -> $CMN_OUT/profile_{data}_{feat}_iid.json
        python icn.py --selftest
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, hashlib, json, math, os, sys, threading, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

E = _data_root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmn  # noqa: E402

cmn.PROC = f"{E}/datasets/proc"
cmn.FEATD = f"{E}/features/clip"
cmn.CACHE = f"{E}/cache/cmn"
cmn.OUT = os.environ.get("CMN_OUT", f"{E}/runs/icn")
dev = cmn.dev
ICN_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]

# ------------------------------------------------------------------------------------------------ variants
ICN_BASE = dict(aug=True, xfit=True, comp=True, scale=True, src=True, geo=True, sub="rel", raw=False, free=False,
                direct=False, iid_frac=0.25, layers=2, d=128, heads=4, steps=3000, episodes=256, temp=True, resid=0.5, val=False)
ICN_VARIANTS = {
    "icn": {},
    "icn_noaug": dict(aug=False),                 # no virtual-person augmentation: episodes of the training identities only
    "icn_noxfit": dict(xfit=False),               # evidence and loss from the full classifier (seen people): no cross-fitting
    "icn_iid": dict(iid_frac=1.0),                # training compositions = the person's natural composition only
    "icn_dir": dict(iid_frac=0.0),                # training compositions = Dirichlet hyper-prior only
    "icn_nocomp": dict(comp=False),               # no composition compensation (centre + scale gates only)
    "icn_noscale": dict(scale=False),             # no scale gate
    "icn_nosrc": dict(src=False),                 # no classifier evidence in the input (geometry only)
    "icn_nogeo": dict(geo=False),                 # no geometry in the input (classifier evidence only)
    "icn_raw": dict(raw=True, d=256),             # appearance-aware: raw features as tokens (memorizes the training people)
    "icn_free": dict(free=True),                  # free centre offset in R^D instead of the evidence composition
    "icn_direct": dict(direct=True),              # no normalization bottleneck: the set network predicts the label itself
}
ICN_EP = ["s", "rho", "a_self", "comp", "err_icn", "err_bn", "err_src"]  # per-episode diagnostics (query clip)


def icn_cfg(method):
    return {**ICN_BASE, **ICN_VARIANTS[method]}


class RelLayer(nn.Module):
    """pre-LN transformer layer whose attention logits carry a learned function of the pairwise clip distances"""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(0.1)

    def forward(self, h, bias):
        B, n, d = h.shape
        q, k, v = self.qkv(self.ln1(h)).view(B, n, 3, self.h, self.dk).unbind(2)
        att = torch.einsum("bihd,bjhd->bhij", q, k) / math.sqrt(self.dk) + bias
        att = self.drop(att.softmax(-1))
        h = h + self.o(torch.einsum("bhij,bjhd->bihd", att, v).reshape(B, n, d))
        return h + self.drop(self.ff(self.ln2(h)))


class ICN(nn.Module):
    """set network -> normalization transform of every clip of an episode (see module docstring).

    The network is appearance-blind: a clip enters as the classifier's evidence L_i, its coordinates in the emotion-
    displacement subspace (absolute, z_i = Q^T x_i, and relative to the set, Q^T (x_i - xbar)) and two scalar sizes;
    clips interact through their pairwise squared distances (a learned kernel enters every attention map).  The full
    feature vectors appear only inside the transform (attention centre, scatter, F-compensation), so the learned rule
    cannot memorize the appearance of the training people (ablation icn_raw does and over-fits them)."""

    def __init__(self, D, C, Fx, cfg):
        super().__init__()
        self.cfg, self.C, self.D = cfg, C, D
        d, H = cfg["d"], cfg["heads"]
        self.register_buffer("Fx", Fx)  # (C, D) emotion displacements in the standardized feature space, centred
        self.register_buffer("Q", torch.linalg.qr(Fx.T)[0][:, : C - 1])  # orthonormal basis of span(F)
        nsub = {"none": 0, "rel": 1, "both": 2}[cfg["sub"]] * (C - 1)
        nin = (2 * D if cfg["raw"] else 0) + (C if cfg["src"] else 0) + ((nsub + 3) if cfg["geo"] else 0)
        self.inp = nn.Linear(max(nin, 1), d)
        self.kern = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, H + 1))  # pairwise-distance kernel: H heads + centring attention
        self.layers = nn.ModuleList([RelLayer(d, H) for _ in range(cfg["layers"])])
        self.norm = nn.LayerNorm(d)
        self.q, self.k = nn.Linear(d, d), nn.Linear(d, d)
        self.gate = nn.Linear(d, 3)                                  # s (centre trust), rho (scale trust), nu (composition trust)
        self.log_tau = nn.Parameter(torch.zeros(()))                 # temperature of the evidence in the composition estimate
        self.ebias = nn.Parameter(torch.zeros(C))                    # class recalibration of the evidence
        self.comp = nn.Linear(d, D) if cfg["free"] else None         # ablation: free centre offset written by the network
        self.out = nn.Linear(d, C) if cfg["direct"] else None
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.kern[-1].weight); nn.init.zeros_(self.kern[-1].bias)
        if self.comp is not None:
            nn.init.zeros_(self.comp.weight); nn.init.zeros_(self.comp.bias)

    def tokens(self, x, L):
        cfg = self.cfg
        B, n, D = x.shape
        xbar = x.mean(1, keepdim=True)
        feats = []
        if cfg["raw"]:
            feats += [x, x - xbar]
        if cfg["src"]:
            feats.append(L)
        if cfg["geo"]:
            if cfg["sub"] in ("rel", "both"):
                feats.append((x - xbar) @ self.Q)  # position relative to the set, in the emotion subspace
            if cfg["sub"] == "both":
                feats.append(x @ self.Q)           # absolute position in the emotion subspace
            r = (x - xbar).pow(2).mean(-1, keepdim=True)  # squared distance to the set mean, per feature dimension
            m = x.pow(2).mean(-1, keepdim=True)            # squared distance to the population centre
            feats += [r, m, torch.full_like(r, math.log(n))]
        if not feats:
            feats = [torch.zeros(B, n, 1, device=x.device)]
        return torch.cat(feats, -1)

    def forward(self, x, L):
        """x (B, n, D) standardized features, L (B, n, C) classifier log-probabilities -> x~ (B, n, D), diagnostics.
        Centre of clip i:  mu_i = c_i - nu_i theta_i^T F,  c_i = sum_j a_ij x_j (which clips to subtract),
        theta_i = sum_{j != i} a_ij q_j / sum_{j != i} a_ij  (the composition of what was subtracted, from the tempered
        classifier evidence q_j of the OTHER clips: the network cannot write a clip's own class into its transform)."""
        B, n, D = x.shape
        cfg = self.cfg
        e = torch.cdist(x, x).pow(2) / D  # (B, n, n) squared distance per feature dimension
        kb = self.kern(torch.stack([e, torch.log(e + 1e-3)], -1))  # (B, n, n, H + 1)
        bias = kb[..., :-1].permute(0, 3, 1, 2)
        h = self.inp(self.tokens(x, L))
        for layer in self.layers:
            h = layer(h, bias)
        h = self.norm(h)
        if cfg["direct"]:
            return None, dict(logits=self.out(h))
        a = ((self.q(h) @ self.k(h).transpose(1, 2)) / math.sqrt(h.shape[-1]) + kb[..., -1]).softmax(-1)  # (B, n, n)
        c = a @ x
        v = (a @ x.pow(2) - c.pow(2)).clamp_min(1e-4)
        g = self.gate(h).sigmoid()
        s, rho, nu = g[..., :1], g[..., 1:2], g[..., 2:]
        if not cfg["scale"]:
            rho = torch.zeros_like(rho)
        q = ((L if cfg["src"] else torch.zeros_like(L)) + self.ebias) / self.log_tau.exp()
        q = q.softmax(-1)                                                      # (B, n, C) tempered evidence
        a_loo = a * (1 - torch.eye(n, device=x.device))
        theta = (a_loo @ q) / a_loo.sum(-1, keepdim=True).clamp_min(1e-6)       # (B, n, C) composition of the others
        if cfg["free"]:
            beta = self.comp(h)
        elif cfg["comp"]:
            beta = -nu * (theta @ self.Fx)
        else:
            beta = torch.zeros_like(c)
        mu = c + beta
        xt = (x - s * mu) * torch.exp(-0.5 * rho * torch.log(v))
        return xt, dict(s=s[..., 0], rho=rho[..., 0], nu=nu[..., 0], a_self=torch.diagonal(a, dim1=1, dim2=2), mu=s * mu,
                        beta=beta, theta=theta, comp=beta.norm(dim=-1))


# ------------------------------------------------------------------------------------------------ fitting
def person_table(meta, rows, X):
    """balanced person means (one per training person) and the centred emotion displacements F in the standardized
    feature space (closed form, training rows only)"""
    act, y = meta.actor.values[rows], meta.label.values[rows]
    C = meta.label.nunique()
    subs = np.unique(act); sid = np.searchsorted(subs, act)
    D = X.shape[1]
    M = torch.zeros(len(subs), C, D, device=dev); cell = torch.from_numpy(sid * C + y).to(dev)
    cnt = torch.bincount(cell, minlength=len(subs) * C).float()
    M.view(-1, D).index_add_(0, cell, X[torch.from_numpy(rows).to(dev)])
    M = M / cnt.view(len(subs), C, 1).clamp_min(1)
    has = (cnt.view(len(subs), C) > 0).float()
    b = (M * has[..., None]).sum(1) / has.sum(1, keepdim=True)
    Fx = ((M - b[:, None]) * has[..., None]).sum(0) / has.sum(0)[:, None]
    Fx = Fx - Fx.mean(0)
    # log-expressivity spread of the training people: lam_s^2 = mean squared residual of person s / pooled
    r2 = (X[torch.from_numpy(rows).to(dev)] - b[torch.from_numpy(sid).to(dev)] - Fx[torch.from_numpy(y).to(dev)]).pow(2).mean(1)
    per = torch.zeros(len(subs), device=dev).index_add_(0, torch.from_numpy(sid).to(dev), r2) / torch.bincount(torch.from_numpy(sid).to(dev), minlength=len(subs)).float()
    lam_sd = float((0.5 * per.log()).std()) if len(subs) > 1 else 0.0
    # person x class interactions G_sc = M_sc - b_s - F_c of every observed cell: the empirical law of the interaction
    # (ZG^T eps, eps ~ N(0, I) has their second-moment matrix); cell-mean noise is left in (slightly conservative)
    Gsc = (M - b[:, None] - Fx[None]) * has[..., None]  # (S, C, D), zero for unobserved cells
    G = Gsc[has.bool()]
    ZG = G / math.sqrt(max(1, len(G)))
    # clip residuals around their cell mean: the empirical law of the within-cell noise (Ze^T eta has its second moment)
    res = X[torch.from_numpy(rows).to(dev)] - M.view(-1, D)[cell]
    Ze = res / math.sqrt(len(res))
    return dict(b=b, subs=subs, Fx=Fx, lam_sd=lam_sd, ZG=ZG, Gsc=Gsc, M=M, Ze=Ze)


def aux_groups(meta, rows, R, seed):
    """R person-disjoint groups of the training rows (deterministic given the seed)"""
    act = meta.actor.values[rows]
    subs = np.unique(act)
    R = int(min(R, len(subs)))
    perm = np.random.default_rng(seed).permutation(len(subs))
    g_of_sub = {subs[p]: i % R for i, p in enumerate(perm)}
    return np.array([g_of_sub[s_] for s_ in act]), R


def aux_heads(a, X, meta, tr, fold, C, R=4):
    """cross-fitting classifiers: for group r, a source network trained on the training people outside r (cached)"""
    key = f"{cmn.model_key(a, fold)}.icnaux{R}.pt"
    grp, R = aux_groups(meta, tr, R, a.seed * 100 + fold + 11)
    heads = []
    if os.path.exists(key):
        saved = torch.load(key, map_location=dev, weights_only=False)
        for sd in saved:
            h = cmn.Source(X.shape[1], C).to(dev); h.load_state_dict(sd); h.eval().requires_grad_(False); heads.append(h)
        return heads, grp
    for r in range(R):
        torch.manual_seed(a.seed * 1000 + fold * 10 + r)
        rng = np.random.default_rng(a.seed * 1000 + fold * 10 + r)
        h = cmn.train_source(cmn.Source(X.shape[1], C).to(dev), X, meta, tr[grp != r], a, rng)
        h.requires_grad_(False); heads.append(h)
    tmp = key + f".{os.getpid()}.tmp"
    torch.save([h.state_dict() for h in heads], tmp); os.replace(tmp, key)
    return heads, grp


def route(heads, grp_ep, x):
    """logits of every clip of every episode, each episode scored by the head that did not see its person (cross-fit)"""
    out = torch.empty(*x.shape[:2], heads[0].cls.out_features, device=x.device)
    for r, h in enumerate(heads):
        m = grp_ep == r
        if m.any():
            out[m] = h(x[m].flatten(0, 1)).view(int(m.sum()), x.shape[1], -1)
    return out


@torch.no_grad()
def evidence(heads, grp_ep, x):
    """classifier log-probabilities of every clip (cross-fitted)"""
    return route(heads, grp_ep, x).log_softmax(-1)


def virtual_person(x, ep, y, tab, sub_row, Z, cfg, g):
    """hierarchical orbit augmentation: the episode's clips are re-expressed by a NEW virtual person drawn from the
    hierarchical model fitted to the training people -- offset ~ between-person law, expressivity ~ its spread, fresh
    person x class interactions ~ their empirical law, and a variance-preserving resample of the clip residuals -- while
    the real within-cell structure of the clips is kept:  x' = b_s + delta + lam (F_y + G~_y + sqrt(1-r) e + sqrt(r) eps)"""
    E_, n, D = x.shape
    C = tab["M"].shape[1]
    s_ = sub_row[ep]
    cell = tab["M"][s_, y[ep]]                                         # (E, n, D) the clip's cell mean (person, class)
    e = x - cell                                                       # real residual
    if cfg["resid"] > 0:
        eps = torch.randn(E_, n, tab["Ze"].shape[0], device=x.device, generator=g) @ tab["Ze"]
        e = math.sqrt(1 - cfg["resid"]) * e + math.sqrt(cfg["resid"]) * eps
    Gt = torch.randn(E_, C, tab["ZG"].shape[0], device=x.device, generator=g) @ tab["ZG"]  # (E, C, D) new interactions
    Gi = torch.gather(Gt, 1, y[ep][..., None].expand(-1, -1, D))
    mu_s = tab["b"][s_[:, 0]][:, None]
    delta = (torch.randn(E_, Z.shape[0], device=x.device, generator=g) @ Z)[:, None]
    lam = torch.exp(torch.randn(E_, 1, 1, device=x.device, generator=g) * tab["lam_sd"])
    return mu_s + delta + lam * (tab["Fx"][y[ep]] + Gi + e)


def train_icn(a, S, X, meta, tr, fold, C, cfg, log=print):
    """maximum-likelihood training of the normalizer on cross-fitted, orbit-augmented episodes of the training people"""
    t0 = time.time()
    tab = person_table(meta, tr, X)
    D = X.shape[1]
    model = ICN(D, C, tab["Fx"], cfg).to(dev)
    if cfg["xfit"]:
        heads, grp = aux_heads(a, X, meta, tr, fold, C)
    else:
        heads, grp = [S], np.zeros(len(tr), int)
    grp_row = torch.zeros(len(meta), dtype=torch.long, device=dev)
    grp_row[torch.from_numpy(tr).to(dev)] = torch.from_numpy(grp).to(dev)
    sub_row = torch.zeros(len(meta), dtype=torch.long, device=dev)
    sub_row[torch.from_numpy(tr).to(dev)] = torch.from_numpy(np.searchsorted(tab["subs"], meta.actor.values[tr])).to(dev)
    Z = (tab["b"] - tab["b"].mean(0)) / math.sqrt(max(1, len(tab["b"]) - 1))  # delta = Z^T eps ~ N(0, between-person cov)
    y = torch.tensor(meta.label.values, device=dev)
    R = len(heads)
    vgrp = R - 1 if (cfg["val"] and cfg["xfit"] and R > 1) else None  # validation people: the last cross-fit group
    fit_rows = tr if vgrp is None else tr[grp != vgrp]
    smp = cmn.EpisodeSampler(meta, fit_rows, C, a.seed * 100 + fold + 7)
    g = torch.Generator(device=dev).manual_seed(a.seed * 100 + fold + 8)
    rng = np.random.default_rng(a.seed * 100 + fold + 9)
    log_T = nn.Parameter(torch.zeros((), device=dev))  # free temperature of the training loss (calibration is not the normalizer's job)
    params = list(model.parameters()) + ([log_T] if cfg["temp"] else [])
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=cfg["steps"], pct_start=0.05)
    if vgrp is not None:  # fixed validation episodes of the held-out people, scored by the head that never saw them
        vs = cmn.EpisodeSampler(meta, tr[grp == vgrp], C, a.seed * 100 + fold + 13)
        vep = torch.cat([vs(96, a.kctx + 1, iid_frac=0.5) for _ in range(5)])
        vL = evidence(heads, grp_row[vep[:, 0]], X[vep])
    best, best_it, best_state = float("inf"), -1, None

    def logits_of(x, L, grp_ep):
        """the loss is cross-fitted too: the normalized clips of an episode are classified by the head that never saw
        its person (the full classifier S would already know the training people and would need no normalization)"""
        xt, d = model(x, L)
        return (d["logits"] if cfg["direct"] else route(heads, grp_ep, xt)), d

    model.train()
    for it in range(cfg["steps"]):
        n = int(rng.integers(2, a.kmax + 2))
        ep = smp(cfg["episodes"], n, iid_frac=cfg["iid_frac"])  # (E, n) rows of one training person each
        x = X[ep]
        if cfg["aug"]:
            x = virtual_person(x, ep, y, tab, sub_row, Z, cfg, g)
        ge = grp_row[ep[:, 0]]
        L = evidence(heads, ge, x)
        lg, d = logits_of(x, L, ge)
        loss = F.cross_entropy((lg / log_T.exp()).flatten(0, 1), y[ep].flatten(), label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if vgrp is not None and (it % 100 == 99 or it == cfg["steps"] - 1):
            model.eval()
            with torch.no_grad():
                vl = F.cross_entropy(logits_of(X[vep], vL, grp_row[vep[:, 0]])[0].flatten(0, 1), y[vep].flatten()).item()
            model.train()
            if vl < best:
                best, best_it, best_state = vl, it, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if it % 500 == 0 or it == cfg["steps"] - 1:
            log(f"  icn step {it} n={n} loss {loss.item():.3f} T={log_T.exp().item():.2f} s={d.get('s', torch.zeros(1)).mean().item():.2f} "
                f"rho={d.get('rho', torch.zeros(1)).mean().item():.2f} val={best:.3f}@{best_it} ({(time.time() - t0) / 60:.1f} min)")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    model.minutes, model.best_it, model.best_val = (time.time() - t0) / 60, best_it, best
    return model


def icn_key(a, fold, method):
    d = f"{cmn.CACHE}/icn/{ICN_SHA}"
    os.makedirs(d, exist_ok=True)
    return f"{d}/{a.data}_{a.feat}_{a.net}_s{a.seed}_f{fold}_t{a.steps}_{method}.pt"


class build_lock:
    """file mutex that serializes the construction of cached artefacts across workers / threads, so that every job of
    a fold uses the identical cached model (the first caller builds, the others find the cache).  A lock older than
    `timeout` seconds is treated as left behind by a dead builder."""

    def __init__(self, key, timeout=7200):
        self.lock, self.timeout = key + ".mutex", timeout

    def __enter__(self):
        while True:
            try:
                fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
                return self
            except FileExistsError:
                try:
                    if time.time() - os.path.getmtime(self.lock) > self.timeout:
                        os.remove(self.lock); continue
                except OSError:
                    pass
                time.sleep(5)

    def __exit__(self, *exc):
        try:
            os.remove(self.lock)
        except OSError:
            pass


def get_icn(a, S, X, meta, tr, fold, C, method=None):
    method = method or a.method
    cfg = icn_cfg(method)
    key = icn_key(a, fold, method)
    tab = person_table(meta, tr, X)
    model = ICN(X.shape[1], C, tab["Fx"], cfg).to(dev)
    with build_lock(key):
        if os.path.exists(key):
            model.load_state_dict(torch.load(key, map_location=dev)); model.eval().requires_grad_(False)
            return model
        model = train_icn(a, S, X, meta, tr, fold, C, cfg)
        tmp = key + f".{os.getpid()}.{threading.get_ident()}.tmp"
        torch.save(model.state_dict(), tmp); os.replace(tmp, key)
    return model


# ------------------------------------------------------------------------------------------------ hooks into cmn.py
_tta0, _get_model0 = cmn.tta, cmn.get_model


def tta(S, x, method, C, yr=None, extras=False):
    if not method.startswith("icn"):
        return _tta0(S, x, method, C, yr, extras)
    B, n, _ = x.shape
    with torch.no_grad():
        if n == 1:
            return cmn.p_source(S, x), {}
        L = S(x.flatten(0, 1)).log_softmax(-1).view(B, n, C)
        net = S.icn[method]
        xt, d = net(x, L)
        if net.cfg["direct"]:
            return d["logits"][:, 0].softmax(-1), {}
        p = S(xt[:, 0]).softmax(-1)
        out = {k: d[k][:, 0] for k in ("s", "rho", "a_self", "comp")}
        if hasattr(S, "bx_true"):  # diagnostic only: centre error of the implied person centre vs the true balanced mean
            bt = S.bx_true[S.idx_rows]
            scale = S.bx_spread
            out["err_icn"] = (d["mu"][:, 0] - bt).pow(2).sum(-1) / scale
            out["err_bn"] = (x.mean(1) - bt).pow(2).sum(-1) / scale
            out["err_src"] = bt.pow(2).sum(-1) / scale
        out["Etheta"] = torch.full((B, C), 1.0 / C, device=x.device)
        return p, out


def get_model(a, X, meta, tr, te, fold, rng, C):
    mfold = 0 if (meta.fold.values < 0).any() else fold
    with build_lock(cmn.model_key(a, mfold)):  # the shared classifier, its cross-fit and DART module: one builder at a time
        S = _get_model0(a, X, meta, tr, te, fold, rng, C)
    if a.method.startswith("icn") or getattr(a, "stream", False) or getattr(a, "profile", False):
        S.icn = {}
        ifold = 0 if (meta.fold.values < 0).any() else fold  # a fixed training pool: one normalizer for every target fold
        for m in ([a.method] if a.method.startswith("icn") else ["icn"]):
            S.icn[m] = get_icn(a, S, X, meta, tr, ifold, C, m)
        with torch.no_grad():  # true balanced means of the TEST people in the standardized feature space (diagnostics)
            tab_te = person_table(meta, te, X)
            S.bx_true = torch.zeros(len(meta), X.shape[1], device=dev)
            S.bx_true[torch.from_numpy(te).to(dev)] = tab_te["b"][torch.from_numpy(np.searchsorted(tab_te["subs"], meta.actor.values[te])).to(dev)]
            truth = f"{cmn.PROC}/{a.data}_truth.npz"
            if os.path.exists(truth):
                z = np.load(truth)
                S.bx_true[torch.from_numpy(te).to(dev)] = torch.tensor((z["mu"][np.searchsorted(z["subjects"], meta.actor.values[te])] - X.xm) / X.xs, dtype=torch.float32, device=dev)
            S.bx_spread = float(S.bx_true[torch.from_numpy(te).to(dev)].pow(2).sum(1).mean().clamp_min(1e-6))
    return S


_predict0 = cmn.predict
cmn.tta, cmn.get_model = tta, get_model
cmn.EP_KEYS = list(dict.fromkeys(list(cmn.EP_KEYS) + ICN_EP))  # ordered, no duplicates
cmn.METHODS = list(cmn.METHODS) + list(ICN_VARIANTS)
_chunk0 = cmn.chunk_size
cmn.chunk_size = lambda method, n, C: (512 if method.startswith("icn") else _chunk0(method, n, C))


def _predict_rows_aware(a, model, X, ctx, rows, C, Y, diag=None, store=None):
    """cmn.predict evaluates chunks of `ch` episodes in order: expose the rows of each chunk through model.idx_rows"""
    rows = np.asarray(rows)
    if ctx is None or not a.method.startswith("icn"):
        return _predict0(a, model, X, ctx, rows, C, Y, diag, store)
    ch = cmn.chunk_size(a.method, ctx.shape[1] + 1, C)
    ps, stores = [], []
    for i in range(0, len(rows), ch):
        model.idx_rows = torch.from_numpy(rows[i:i + ch]).to(dev)
        st_ = {} if store is not None else None
        ps.append(_predict0(a, model, X, ctx[i:i + ch], rows[i:i + ch], C, Y, diag if i == 0 else None, st_))
        if st_ is not None:
            stores.append(st_)
    p = np.concatenate(ps)
    if store is not None:
        store["alt"] = {k: np.concatenate([s_["alt"][k] for s_ in stores]) for k in stores[0].get("alt", {})}
        store["ep"] = {k: np.concatenate([s_["ep"][k] for s_ in stores]) for k in stores[0].get("ep", {})}
        store["ab"] = {}
    return p


cmn.predict = _predict_rows_aware


# ------------------------------------------------------------------------------------------------ stream / profile
def run_stream(a):
    """cmn.run_stream with the ICN window method added (strictly causal references, Source at a cold start)"""
    meta = pd.read_csv(f"{cmn.PROC}/{a.data}_meta.csv"); meta.attrs["data"] = a.data
    C = meta.label.nunique()
    Y = torch.tensor(meta.label.values, device=dev)
    keys = ["source", "bn_win", "cpn_win", "icn_win"] + [f"{m}{r}" for m in ("tent_online", "note_full", "rotta_full") for r in ("", "_reset")]
    res = dict(args=vars(a), code_sha256=cmn.CODE_SHA, icn_sha256=ICN_SHA, methods=keys, folds={})
    allP, allY = {k: [] for k in keys}, []
    for fold in cmn.test_folds(meta):
        torch.manual_seed(a.seed * 100 + fold)
        X, tr, te = cmn.load_fold(a, meta, fold)
        a.method = "cpn"
        model = get_model(a, X, meta, tr, te, fold, np.random.default_rng(a.seed * 100 + fold), C)
        order = cmn.stream_order(meta, te, 5000 + fold)
        act = meta.actor.values
        y = meta.label.values[order]
        P = {}
        for k in ("source", "bn_win", "cpn_win", "icn_win"):
            a_ = argparse.Namespace(**vars(a)); a_.method = {"icn_win": "icn"}.get(k, k)
            P[k] = cmn.stream_window(a_, model, X, meta, order, {"icn_win": "icn"}.get(k, k), C, Y)
        for reset in (False, True):
            sfx = "_reset" if reset else ""
            P["tent_online" + sfx] = cmn.stream_tent(model, X, order, act, reset, C).cpu().numpy()
            P["note_full" + sfx] = cmn.stream_note(model, X, order, act, reset, C, a.seed * 100 + fold).cpu().numpy()
            P["rotta_full" + sfx] = cmn.stream_rotta(model, X, order, act, reset, C, a.seed * 100 + fold).cpu().numpy()
        res["folds"][int(fold)] = {k: cmn.metrics(P[k], y, C) for k in keys}
        res["folds"][int(fold)]["order"] = order.tolist()
        np.save(f"{cmn.OUT}/pred/stream_{a.data}_{a.feat}_s{a.seed}_f{fold}.npy", np.stack([P[k] for k in keys]).astype(np.float32))
        for k in keys:
            allP[k].append(P[k])
        allY.append(y)
        print(fold, {k: round(v["acc"], 2) for k, v in res["folds"][int(fold)].items() if isinstance(v, dict)}, flush=True)
    yy = np.concatenate(allY)
    res["pooled"] = {k: cmn.metrics(np.concatenate(allP[k]), yy, C) for k in keys}
    fn = f"{cmn.OUT}/stream_{a.data}_{a.feat}_s{a.seed}.json"
    with open(fn + ".tmp", "w") as fo:
        json.dump(res, fo)
    os.replace(fn + ".tmp", fn)
    print("STREAM", a.data, json.dumps({k: round(v["acc"], 2) for k, v in res["pooled"].items()}), flush=True)


def profile(a):
    """latency / memory / accuracy vs K of the test-time methods incl. ICN (one fold; run alone on the GPU)"""
    meta = pd.read_csv(f"{cmn.PROC}/{a.data}_meta.csv"); meta.attrs["data"] = a.data
    C = meta.label.nunique()
    fold = cmn.test_folds(meta)[0]
    X, tr, te = cmn.load_fold(a, meta, fold)
    a.method = "cpn"
    model = get_model(a, X, meta, tr, te, fold, np.random.default_rng(a.seed * 100 + fold), C)
    Y = torch.tensor(meta.label.values, device=dev)
    res = dict(data=a.data, feat=a.feat, net=a.net, device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
               code_sha256=cmn.CODE_SHA, icn_sha256=ICN_SHA, C=C, rows=[],
               icn_params=int(sum(p.numel() for p in model.icn["icn"].parameters())))
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    for K in a.profile_k:
        ctx = cmn.contexts(meta, te, K, "rand", np.random.default_rng(1000 + fold), te)
        idx = torch.cat([torch.from_numpy(te).to(dev)[:, None], torch.from_numpy(ctx).to(dev)], 1)
        for m in a.profile_methods:
            ch = min(cmn.chunk_size(m, K + 1, C), len(idx))
            xb1, xb = X[idx[:1]], X[idx[:ch]]
            model.idx_rows = torch.from_numpy(te[:ch]).to(dev)
            tta(model, xb1, m, C, Y[idx[:1, 1:]]); sync()
            ts = []
            for _ in range(30):
                model.idx_rows = torch.from_numpy(te[:1]).to(dev)
                t = time.perf_counter(); tta(model, xb1, m, C, Y[idx[:1, 1:]]); sync(); ts.append(time.perf_counter() - t)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated() if dev == "cuda" else 0
            model.idx_rows = torch.from_numpy(te[:ch]).to(dev)
            t = time.perf_counter()
            for _ in range(3):
                tta(model, xb, m, C, Y[idx[:ch, 1:]])
            sync(); tb = (time.perf_counter() - t) / 3
            peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20 if dev == "cuda" else float("nan")
            H = math.comb(K + C, C - 1) if m.startswith(("cmn", "cpn")) else None
            a_ = argparse.Namespace(**vars(a)); a_.method = m
            pr = cmn.predict(a_, model, X, ctx, te, C, Y)
            res["rows"].append(dict(K=K, method=m, online_ms=1000 * float(np.median(ts)), batch=ch,
                                    batch_ms_per_episode=1000 * tb / ch, peak_mb_per_batch=peak, hypotheses=H,
                                    acc=float((pr.argmax(1) == meta.label.values[te]).mean() * 100)))
            print(K, m, round(res["rows"][-1]["online_ms"], 2), "ms online", round(res["rows"][-1]["batch_ms_per_episode"], 3), "ms/episode batched", round(peak, 1), "MB", flush=True)
    fn = f"{cmn.OUT}/profile_{a.data}_{a.feat}_{a.net}.json"
    with open(fn + ".tmp", "w") as fo: json.dump(res, fo, indent=1)
    os.replace(fn + ".tmp", fn)


# ------------------------------------------------------------------------------------------------ self-test
def selftest():
    """the transform family contains Source and BN-adapt exactly; permutation equivariance; cross-fit groups are
    person-disjoint; the orbit augmentation has the between-person covariance"""
    torch.manual_seed(0)
    D, C, B, n = 12, 4, 3, 5
    Fx = torch.randn(C, D); Fx -= Fx.mean(0)
    net = ICN(D, C, Fx, icn_cfg("icn")).eval()
    x = torch.randn(B, n, D); L = torch.randn(B, n, C).log_softmax(-1)
    with torch.no_grad():
        # Source member: s = rho = 0  (gate logits -> -inf)
        net.gate.bias.fill_(-50.0)
        xt, _ = net(x, L)
        assert torch.allclose(xt, x, atol=1e-5), "source member"
        # BN member: uniform attention (q, k = 0), s = rho = 1, nu = 0
        net.gate.bias.copy_(torch.tensor([50.0, 50.0, -50.0])); net.q.weight.zero_(); net.q.bias.zero_(); net.k.weight.zero_(); net.k.bias.zero_()
        xt, d = net(x, L)
        bn = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, unbiased=False, keepdim=True).clamp_min(1e-4))
        assert torch.allclose(xt, bn, atol=1e-4), "BN member"
        # permutation equivariance
        net.gate.bias.zero_(); nn.init.normal_(net.q.weight, std=0.1); nn.init.normal_(net.k.weight, std=0.1)
        perm = torch.randperm(n)
        xt1, _ = net(x, L); xt2, _ = net(x[:, perm], L[:, perm])
        assert torch.allclose(xt1[:, perm], xt2, atol=1e-5), "equivariance"
        # compensation lies in span(F)
        _, d = net(x, L)
        Q = torch.linalg.qr(Fx.T)[0][:, : C - 1]
        beta = d["beta"]
        assert beta.norm() > 1e-3 and torch.allclose(beta @ Q @ Q.T, beta, atol=1e-4), "compensation outside span(F)"
    print("transform family: Source and BN-adapt members exact; permutation-equivariant; compensation in span(F)")
    meta = pd.DataFrame(dict(actor=np.repeat(np.arange(9), 4), label=np.tile(np.arange(4), 9)))
    rows = np.arange(36)
    grp, R = aux_groups(meta, rows, 4, 0)
    assert R == 4 and all(len(set(grp[meta.actor.values == s_])) == 1 for s_ in range(9)), "groups not person-disjoint"
    print("cross-fit groups: person-disjoint")
    b = torch.randn(7, D)
    Z = (b - b.mean(0)) / math.sqrt(6)
    cov = Z.T @ Z
    assert torch.allclose(cov, torch.cov(b.T), atol=1e-5), "orbit covariance"
    print("orbit augmentation: covariance of the sampled offsets = between-person covariance")
    print("selftest passed")


# ------------------------------------------------------------------------------------------------ main
def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ravdess")
    ap.add_argument("--method", default="icn", choices=cmn.METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat", default="l14", choices=["l14", "b16", "dual", "raw", "avt", "wavlm"])
    ap.add_argument("--net", default="iid", choices=cmn.NETS)
    ap.add_argument("--suite", default="lite", choices=["full", "lite", "paper"])
    ap.add_argument("--nctx", type=int, default=cmn.NCTX)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--kctx", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--mem_frac", type=float, default=float(os.environ.get("MEM_FRAC", "0.1")))
    ap.add_argument("--folds", default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--profile-k", type=int, nargs="+", default=cmn.PROFILE_K)
    ap.add_argument("--profile-methods", nargs="+", default=["bn", "tent", "laem", "rotta", "dart", "cpn", "icn"])
    a = ap.parse_args(argv)
    a.tag = (a.method + ("" if a.feat == "l14" else f"_{a.feat}") + ("" if a.net == "iid" else f"_{a.net}")
             + ("" if a.suite == "full" else f"_{a.suite}") + ("" if a.nctx == cmn.NCTX else f"_c{a.nctx}")
             + ("" if a.folds is None and a.steps == 3000 else "_smoke"))
    return a


def run_job(argv):
    """one job (episodic run / stream / profile) in the current process; returns the result file"""
    a = get_args(argv)
    os.makedirs(f"{cmn.OUT}/pred", exist_ok=True)
    if a.profile:
        profile(a); return f"{cmn.OUT}/profile_{a.data}_{a.feat}_{a.net}.json"
    if a.stream:
        run_stream(a); return f"{cmn.OUT}/stream_{a.data}_{a.feat}_s{a.seed}.json"
    t0 = time.time(); res = cmn.run(a); res["minutes"] = (time.time() - t0) / 60
    res["icn_sha256"] = ICN_SHA
    if a.method.startswith("icn"):
        res["icn_cfg"] = icn_cfg(a.method)
    fn = f"{cmn.OUT}/{a.data}_{a.tag}_s{a.seed}.json"
    with open(fn + ".tmp", "w") as fo: json.dump(res, fo, allow_nan=False)
    os.replace(fn + ".tmp", fn)
    print("RESULT", a.data, a.tag, a.seed, json.dumps({k: round(v["acc"], 2) for k, v in res["mean"].items()}), flush=True)
    return fn


if __name__ == "__main__":
    a = get_args()
    if a.selftest:
        selftest(); raise SystemExit
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    run_job(sys.argv[1:])
