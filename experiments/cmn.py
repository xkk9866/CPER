"""CPN: Composition-Posterior Normalization for person-adaptive emotion recognition (round 10; CPN7 = round 7).

Problem.  A recognizer is adapted to one person with K unlabeled reference clips.  Batch-statistics methods (BN-adapt,
TENT, SAR, LAEM, NOTE, RoTTA, CSN, DART, ...) estimate the person's statistics as if the references covered the person's
expressions evenly; when they do not, the statistics absorb the expression and recognition collapses, and when all
references share one class the person's centre is not even identified -- it is confounded with the class (Prop. 1).

Method (round 10): ONE hierarchical model of a person's clips (label composition x person centre x expressivity) and
ONE prediction rule.
  1. Composition posterior p(N | U): exact count dynamic program over every label assignment of the batch.  Its three
     evidence families -- the network under every hypothesis' transport, the population density of the location-
     normalised clips, the identifiability-weighted set anchor -- are weighted by CONDITIONAL MAXIMUM LIKELIHOOD of the
     true compositions of held-out people (calibrate_posterior; the KL projection of the true composition posterior
     onto this log-linear family).
  2. The posterior defines FOUR predictive distributions of the query label -- the four classical estimators of the
     person's normalisation: the exact Bayes prediction GIVEN each composition N, p(y | N, U) (hypothesis-conditional;
     its average over p(N | U) is the exact joint posterior marginal); the conservative prediction at the posterior-
     mean centre shrunk by the posterior uncertainty, p_EB (empirical Bayes); the batch's own moments, p_M (the
     composition-agnostic moment estimator = the plug-in under the balanced composition = BN-adapt); and the source
     model, p_S (no adaptation).
  3. Identifiability stacking:
        p(y | U) = sum_N p(N | U) sum_k w_k(g_N) p_k(y | N, U),   g(theta) = (C |theta|^2 - 1)/(C - 1),
        (w_B, w_E, w_M, w_S)(g) = softmax(0, a_E + b_E g, a_M + b_M g, a_S + b_S g).
     Prop. 1: a composition hypothesis identifies the person's centre less, the larger g; every hypothesis therefore
     splits its vote among the four estimators as a function of its own identifiability.  The six numbers are the
     class-balanced log-score optimum on held-out training people (fit_stacking).  The closure of the family contains
     every estimator alone -- in particular the two reference methods, Source (w_S = 1) and BN-adapt (w_M = 1) -- and
     every constant mixture; by the oracle inequality of proper-score stacking, the fitted combiner is on new people of
     the same population no worse in expected balanced log score than any member of the family (each reference method
     included), up to the estimation error of six parameters on ~1.4k held-out episodes.  The non-inferiority tests
     of the paper (vs Source on the hard protocol, vs BN on random references) are the empirical check of this claim
     in accuracy.  The branch weights are chosen by the data under a proper scoring rule, not postulated.
  4. Source-anchored transport: every read-out moves the person's centre / per-unit spread to the population's and
     uses the SOURCE head, so the source model is exactly the zero-information limit (Prop. 11).

  Person model, in the population-standardised units u of the source's BN layer:
      u_i = mu_s + lam_s (F_{y_i} + G_{s, y_i} + e_i)
  (offset, expressivity, population displacement, person x class interaction shared by the clips of one class, noise).

Cross-fitted constants (auxiliary network on 3/4 of the training people, held-out 1/4): temperatures (person-mode,
source, batch-moment), expressivity prior, kappa, posterior weights (t_net, t_dens, t_anc), the stacking combiner
(a_E, b_E, a_M, b_M, a_S, b_S) and its control combiners, and the validation accuracies of every grid value of the
tuned baselines.  An ablation variant (VARIANT_REFIT) re-fits its own posterior weights and combiner on the same cached
auxiliary network and held-out people: one component changed, everything else by the same procedure.

Every "cpn" run also saves, from the identical posterior: each branch alone (cond = exact joint marginal, eb, bn,
src), the round-9 fixed weighting (wg), the control combiners fitted on the same held-out people with the same
objective (stack_nobn without the batch-moment branch = the round-10a design, stack_nosrc without the source model,
flat without the identifiability covariate, episode-level gates on concentration E[g], on Source entropy, and a learned
gate on five generic features), the stacked predictor at the MAP composition, and the conservative branch without
shrinkage / transport inside the stacked predictor; plus per-episode diagnostics.

Baselines (feature-level re-implementations, same source network, same episodes): source, bn (NeurIPS'20), tent_grid
(ICLR'21), sar_grid (ICLR'23), laem_grid (WACV'26), note (NeurIPS'22), rotta_grid (CVPR'23), csn_grid (CoLLAs'24), dart
(NeurIPS'24), lame (CVPR'22), t3a (NeurIPS'21), srcbn_grid / bnprior_grid (tuned simple controls), neucal (the known-
neutral setting of SFDA-PFT), pl_shrink (independent pseudo-label composition, identical read-out); and, in the person-
blocked STREAM protocol (--stream), feature-head TENT-online, NOTE (PBRS) and RoTTA (CSTU, robust BN, EMA teacher).

usage:  python cmn.py --data ravdess --method cpn --seed 0 [--feat l14|avt|wavlm]  -> $CMN_OUT/{data}_{tag}_s{seed}.json
        python cmn.py --data meld --feat avt --stream                              -> $CMN_OUT/stream_{data}_{feat}_s0.json
        python cmn.py --selftest      (exactness of the count DP; Prop. 11; stacking nests its endpoints)
        the paper suite: run_cpn.py; tables: aggregate_cpn.py
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, hashlib, itertools, json, math, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(int(os.environ.get('CMN_CPU_THREADS', '1')))

PROC = f"{_data_root}/datasets/proc"
FEATD = f"{_data_root}/features/clip"
CACHE = f"{_data_root}/cache/cmn"
OUT = os.environ.get("CMN_OUT", f"{_data_root}/runs/cpn10")
dev = "cuda" if torch.cuda.is_available() else "cpu"
NEUTRAL = {"ravdess": 0, "cremad": 4, "mead": 0, "meld": 0, "shemo": 2, "emoz": 4, "thai": 3, "thaix": 3}  # eNTERFACE'05 has no neutral class


def neutral(data):
    return 0 if data.startswith("synth") else NEUTRAL.get(data)
CODE_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
INF = float("inf")


def load(data, fold, feat):
    """l14 / b16: temporal mean of the CLS token of frozen CLIP ViT-L/14 / ViT-B/16 (main representation: l14);
    avt: MELD face (l14) + speech (frozen WavLM-Large, layer-averaged) + transcript (frozen MPNet) -- prep_natural.py;
    wavlm: speech only (ShEMO); dual: frozen B/16 + per-fold fine-tuned B/16; raw: synthetic features"""
    if feat == "dual":
        return np.concatenate([np.load(f"{FEATD}/{data}_frozen.npy"), np.load(f"{FEATD}/{data}_ft_f{fold}.npy")], 1)
    return np.load(f"{FEATD}/{data}_{feat}.npy")


# ============================================================================ exact composition enumeration
def compositions(n, C):
    """all count vectors of n items over C classes, (S, C) int64 (stars and bars)"""
    out = []
    for bars in itertools.combinations(range(n + C - 1), C - 1):
        b = (-1,) + bars + (n + C - 1,)
        out.append([b[i + 1] - b[i] - 1 for i in range(C)])
    return np.array(out, dtype=np.int64).reshape(-1, C)


_SUB = {}


def sublattice(n, C):
    """For every batch composition N_h (|N_h| = n), every sub-composition M <= N_h with |M| <= n - 1, flattened.
    par[i, c] = index of M_i - e_c (same hypothesis), q[h, c] = index of N_h - e_c.  The count DP over the K = n - 1
    references on this lattice gives, for every hypothesis h and query class c, the exact sum over reference label
    assignments with counts N_h - e_c."""
    key_ = (n, C)
    if key_ in _SUB:
        return _SUB[key_]
    os.makedirs(CACHE, exist_ok=True)
    fn = f"{CACHE}/sublattice_n{n}_C{C}.npz"
    if os.path.exists(fn):
        z = np.load(fn)
        H, hyp, vec, par, q = z["H"], z["hyp"], z["vec"], z["par"], z["q"]
    else:
        H = compositions(n, C)
        hyp, vec = [], []
        for h, N in enumerate(H):
            g = np.stack(np.meshgrid(*[np.arange(v + 1) for v in N], indexing="ij"), -1).reshape(-1, C)
            g = g[g.sum(1) <= n - 1]
            hyp.append(np.full(len(g), h)); vec.append(g)
        hyp, vec = np.concatenate(hyp), np.concatenate(vec)
        base = n + 1
        pw = base ** np.arange(C, dtype=np.int64)
        key = hyp.astype(np.int64) * base ** C + vec @ pw
        order = np.argsort(key); ks = key[order]
        tot = len(vec)
        par = np.full((tot, C), tot, dtype=np.int64)
        for c in range(C):
            ok = vec[:, c] > 0
            par[ok, c] = order[np.searchsorted(ks, key[ok] - pw[c])]
        q = np.full((len(H), C), tot, dtype=np.int64)
        qk = np.arange(len(H), dtype=np.int64)[:, None] * base ** C + (H @ pw)[:, None] - pw[None]
        ok = H > 0
        q[ok] = order[np.searchsorted(ks, qk[ok])]
        tmp = f"{fn}.{os.getpid()}.tmp.npz"
        np.savez(tmp, H=H, hyp=hyp, vec=vec, par=par, q=q)
        os.replace(tmp, fn)  # atomic: parallel workers never read a partial file
    lvl = vec.sum(1)
    out = dict(H=torch.tensor(H, dtype=torch.float32, device=dev), hyp=torch.tensor(hyp, device=dev),
               par=torch.tensor(par, device=dev), q=torch.tensor(q, device=dev), tot=len(vec),
               levels=[torch.tensor(np.where(lvl == k)[0], device=dev) for k in range(n)])
    _SUB[key_] = out
    return out


def count_dp(SL, unary):
    """unary: (B, Hh, n, C) log-potentials of every clip under every hypothesis (clip 0 = query, ignored here).
    returns (B, Hh, C): log sum over label assignments of the references 1..n-1 with counts N_h - e_c of
    prod_k exp(unary[b, h, k, y_k])  (exact; -inf when N_h,c = 0)"""
    B = unary.shape[0]
    la = torch.full((B, SL["tot"] + 1), -INF, device=unary.device)
    la[:, SL["levels"][0]] = 0.0
    for k in range(1, len(SL["levels"])):
        idx = SL["levels"][k]
        un = unary[:, SL["hyp"][idx], k]  # reference k under the hypothesis of each state
        la[:, idx] = torch.logsumexp(la[:, SL["par"][idx]] + un, -1)
    return la[:, SL["q"]]


ALPHAS = [2.0 ** e for e in range(-3, 8)]  # log-uniform hyperprior on the Dirichlet concentration of the references


def log_dmseq(M, pi, prior, each=False):
    """log prior probability of one reference label SEQUENCE with counts M (..., C); each=True: one value per alpha of
    the hyperprior grid (len(ALPHAS), ...)"""
    if prior == "mult":
        return (M * pi.log()).sum(-1)
    K = M.sum(-1)
    al = torch.tensor(ALPHAS if prior == "dm" else [float(len(pi))], device=M.device)
    al = al.view(-1, *([1] * M.dim()))
    ap = al * pi
    lg = torch.lgamma(al[..., 0]) - torch.lgamma(al[..., 0] + K) + (torch.lgamma(ap + M) - torch.lgamma(ap)).sum(-1)
    return lg if each else torch.logsumexp(lg, 0) - math.log(al.shape[0])


# ============================================================================ source network
class Source(nn.Module):
    """netF (bottleneck): Linear -> BatchNorm -> GELU;  netC: Linear.  Trained on i.i.d. labelled clips."""

    def __init__(self, D, C, h=512):
        super().__init__()
        self.drop, self.f1, self.bn, self.cls = nn.Dropout(0.1), nn.Linear(D, h), nn.BatchNorm1d(h), nn.Linear(h, C)

    def forward(self, x):
        return self.cls(self.drop(F.gelu(self.bn(self.f1(self.drop(x))))))


def pop_units(S, x):
    """population-standardised pre-activation units u of the BN layer"""
    return (x @ S.f1.weight.T + S.f1.bias - S.bn.running_mean) / torch.sqrt(S.bn.running_var + S.bn.eps)


def head(S, un):
    """network output from person-normalised units (person-mode affine when the network has one)"""
    g, b = (S.pg, S.pb) if hasattr(S, "pg") else (S.bn.weight, S.bn.bias)
    return F.gelu(un * g + b) @ S.cls.weight.T + S.cls.bias


def head_pop(S, un):
    """network output from population-standardised units (the source / population mode)"""
    return F.gelu(un * S.bn.weight + S.bn.bias) @ S.cls.weight.T + S.cls.bias


def train_source(model, X, meta, rows, a, rng):
    y = torch.tensor(meta.label.values, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    for it in range(a.steps):
        idx = torch.from_numpy(rng.choice(rows, a.bs * 5)).to(dev)
        loss = F.cross_entropy(model(X[idx]), y[idx], label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model.eval()


def train_episodic(model, X, meta, rows, a, rng, C, kind):
    """control network 'epibn': the i.i.d. loss of train_source + a BatchNorm-over-the-episode loss on person episodes
    (n = 2..kmax+1 clips of one training subject, compositions from the CMN prior).  It uses the same information as
    CMN's statistics (identities of the training people) to train the network for test-time batch statistics."""
    y = torch.tensor(meta.label.values, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    smp = EpisodeSampler(meta, rows, C, int(rng.integers(1 << 30)))
    for it in range(a.steps):
        idx = torch.from_numpy(rng.choice(rows, a.bs * 5)).to(dev)
        loss = F.cross_entropy(model(X[idx]), y[idx], label_smoothing=0.1)
        n = int(rng.integers(2, a.kmax + 2)); ep = smp(max(1, a.bs * 5 // n), n)
        h = model.drop(X[ep]) @ model.f1.weight.T + model.f1.bias
        act = F.gelu((h - h.mean(1, keepdim=True)) / torch.sqrt(h.var(1, unbiased=False, keepdim=True) + model.bn.eps)
                     * model.bn.weight + model.bn.bias)
        loss = loss + F.cross_entropy(model.cls(model.drop(act)).flatten(0, 1), y[ep].flatten(), label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model.eval()


NETS = ["iid", "epibn"]


def make_network(kind, X, meta, rows, a, rng, C):
    """train the network of the given kind on the labelled rows and attach its closed-form population statistics"""
    if kind == "iid":
        model = train_source(Source(X.shape[1], C).to(dev), X, meta, rows, a, rng)
    else:
        model = train_episodic(Source(X.shape[1], C).to(dev), X, meta, rows, a, rng, C, kind)
    model.st = subject_stats(model, X, meta, rows, C)
    return model


# ============================================================================ closed-form population statistics
@torch.no_grad()
def subject_stats(S, X, meta, rows, C):
    """statistics of labelled subjects in the population-standardised units u of network S:
    b      balanced subject means (one per subject)                 Fc   expression displacements (C, H), centred
    sig2   within-subject residual variance per unit (= sin2 + sg2)  Q    orthonormal basis of span(F), (H, C-1)
    sin2   within-cell (clip) variance per unit                      sg2  subject x expression interaction variance per unit
    f      displacements in the subspace (C, C-1)                    Sin  within-cell residual covariance (subspace)
    SG     subject x expression interaction covariance (subspace, noise-corrected)
    T      between-subject covariance of balanced means (subspace)   m0   their mean      pi  class frequencies"""
    u = pop_units(S, X[rows])
    act, y = meta.actor.values[rows], meta.label.values[rows]
    subs = np.unique(act)
    M = torch.zeros(len(subs), C, u.shape[1], device=dev); has = torch.zeros(len(subs), C, device=dev)
    sid = np.searchsorted(subs, act)
    cell = torch.from_numpy(sid * C + y).to(dev)
    cnt = torch.bincount(cell, minlength=len(subs) * C).float()
    M.view(-1, u.shape[1]).index_add_(0, cell, u)
    M = M / cnt.view(len(subs), C, 1).clamp_min(1)
    has = (cnt.view(len(subs), C) > 0).float()
    b = (M * has[..., None]).sum(1) / has.sum(1, keepdim=True)
    Fc = ((M - b[:, None]) * has[..., None]).sum(0) / has.sum(0)[:, None]
    Fc = Fc - Fc.mean(0)
    yt = torch.from_numpy(y).to(dev); st = torch.from_numpy(sid).to(dev)
    res = u - b[st] - Fc[yt]
    Q = torch.linalg.qr(Fc.T)[0][:, : C - 1]
    rin = (u - M.view(-1, u.shape[1])[cell]) @ Q
    corr = cnt[cell] / (cnt[cell] - 1).clamp_min(1)  # unbiased within-cell covariance
    Sin = (rin * corr[:, None]).T @ rin / len(rin)
    G = ((M - b[:, None] - Fc[None]) @ Q)[has.bool()]
    ncell = cnt.view(len(subs), C)[has.bool()]
    SG = G.T @ G / len(G) - Sin * (1 / ncell).mean()  # remove the sampling noise of the cell means
    ev, V = torch.linalg.eigh(SG)
    SG = V @ torch.diag(ev.clamp_min(1e-3 * ev.max().clamp_min(1e-6))) @ V.T
    bz = b @ Q
    T = torch.cov(bz.T) if len(subs) > 1 else torch.eye(C - 1, device=dev)
    pi = torch.bincount(yt, minlength=C).float(); pi = pi / pi.sum()
    # per-unit variance components: clips of one person and one class share the interaction G_sc
    r_in = u - M.view(-1, u.shape[1])[cell]
    sin2 = (r_in ** 2 * corr[:, None]).mean(0)
    Gu = (M - b[:, None] - Fc[None])[has.bool()]  # (cells, H); sums to ~0 over the classes of a subject
    sg2 = (Gu ** 2).mean(0) * C / (C - 1) - sin2 * (1 / ncell).mean()
    sg2 = sg2.clamp_min(0.05 * sin2)
    return dict(b=b, subs=subs, Fc=Fc, sig2=res.var(0), sin2=sin2, sg2=sg2, Q=Q, f=Fc @ Q, Sin=Sin, SG=SG, T=T,
                m0=bz.mean(0), pi=pi, bbar=b.mean(0))


def bbar(st):
    """population balanced centre in the units u (mean of the training subjects' balanced means, all units).  In the
    population-standardised units the frequency-weighted mean is 0, so bbar = -pi^T F: a person centre removed to 0
    instead of to bbar shifts every clip by +pi^T F, i.e. towards the frequent classes."""
    return st["bbar"] if "bbar" in st else st["b"].mean(0)


def transport(st, u, mu, sc):
    """Source-anchored transport (round 8): a person's clips are moved into the source network's coordinates -- the
    person's centre mu to the population's balanced centre, the person's expressivity sc to the population's (= 1):
        u_tilde = bbar + (u - mu) / sc,   prediction = source head (population BatchNorm affine) of u_tilde.
    With no information about the person (mu = bbar, sc = 1) this IS the source model (Prop. 11, zero-information
    limit); the round-7 read-out standardised the units to the person's own repertoire variance and centred them at 0,
    which differs from the source model even with no adaptation (the frequent-class shift of bbar above and a
    re-scaling), and is what collapsed the rare classes of the natural data sets."""
    return bbar(st) + (u - mu) / sc


@torch.no_grad()
def fit_temperature(lg, y, offset=None, hi=20.0):
    """1-D maximum-likelihood temperature (golden-section search on log tau); optional additive log-prior offset"""
    lo, hi = math.log(0.05), math.log(hi)
    g = (math.sqrt(5) - 1) / 2
    off = 0.0 if offset is None else offset
    f = lambda t: F.cross_entropy(lg / math.exp(t) + off, y).item()
    a_, b_ = lo, hi
    c_, d_ = b_ - g * (b_ - a_), a_ + g * (b_ - a_)
    for _ in range(60):
        if f(c_) < f(d_):
            b_ = d_
        else:
            a_ = c_
        c_, d_ = b_ - g * (b_ - a_), a_ + g * (b_ - a_)
    return math.exp((a_ + b_) / 2)


class EpisodeSampler:
    """person episodes (n clips of one labelled training subject) on the GPU.  Compositions follow the CMN prior:
    with prob. 1/2 random clips of the person (natural composition), otherwise class proportions ~ Dirichlet(alpha pi)
    with alpha drawn from the same log-uniform hyperprior ALPHAS that CMN marginalises at test time."""

    def __init__(self, meta, rows, C, seed):
        act, y = meta.actor.values[rows], meta.label.values[rows]
        subs = np.unique(act); sid = np.searchsorted(subs, act)
        cnt = np.zeros((len(subs), C), int)
        mx = max(1, np.bincount(sid * C + y, minlength=len(subs) * C).max())
        tab = np.zeros((len(subs), C, mx), int)
        for r, s_, c in zip(rows, sid, y):
            tab[s_, c, cnt[s_, c]] = r; cnt[s_, c] += 1
        self.tab, self.cnt = torch.tensor(tab, device=dev), torch.tensor(cnt, device=dev)
        pi = np.bincount(meta.label.values[rows], minlength=C).astype(float)
        self.pi = torch.tensor(pi / pi.sum(), dtype=torch.float32, device=dev)
        self.al = torch.tensor(ALPHAS, device=dev)
        self.g = torch.Generator(device=dev).manual_seed(int(seed))

    def __call__(self, E, n, iid_frac=0.5):
        g = self.g
        s = torch.randint(0, self.cnt.shape[0], (E,), device=dev, generator=g)
        cnt = self.cnt[s].float(); has = (cnt > 0).float()
        al = self.al[torch.randint(0, len(self.al), (E,), device=dev, generator=g)]
        w = torch._standard_gamma((al[:, None] * self.pi[None]).clamp_min(1e-3)) * has + 1e-12 * has
        w = torch.where((torch.rand(E, device=dev, generator=g) < iid_frac)[:, None], cnt, w)
        lab = torch.multinomial(w / w.sum(1, keepdim=True), n, replacement=True, generator=g)
        k = (torch.rand(E, n, device=dev, generator=g) * self.cnt[s[:, None], lab]).long()
        return self.tab[s[:, None], lab, k]


@torch.no_grad()
def log_expressivity(S, X, meta, rows, st):
    """log lam_s of every subject in rows relative to the statistics st: lam_s^2 = sum_h var_s(u_h) / sum_h V_h(theta_s)"""
    u = pop_units(S, X[rows]); act, y = meta.actor.values[rows], meta.label.values[rows]
    C = st["pi"].numel(); out = []
    for s_ in np.unique(act):
        m = act == s_
        th = torch.tensor(np.bincount(y[m], minlength=C) / m.sum(), dtype=torch.float32, device=dev)
        Vh = batch_var(th[None], st, int(m.sum()))[0]
        s2 = u[torch.from_numpy(np.where(m)[0]).to(dev)].var(0, unbiased=False)
        out.append(0.5 * math.log(float(hypothesis_scale(s2[None], Vh[None], st, True))))
    return np.array(out)


ALPHA_P = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 1e12]  # grid of the person-label concentration
#   (1e12 = no coupling: the query ~ pi; a finite number keeps the result files strict JSON)


def fit_alpha_person(meta, rows, C, K, seed, nq=4000):
    """Empirical-Bayes concentration alpha_p of the PERSON-LEVEL label distributions, from the labels of the training
    people only (no network): a person's label distribution phi ~ Dirichlet(alpha_p pi), the query and the K random
    references of that person ~ phi, so that the references predict the query by the Dirichlet-multinomial predictive
        p(y_q = c | reference counts M) = (alpha_p pi_c + M_c) / (alpha_p + K).
    alpha_p maximises its leave-query-out log-likelihood over random K-reference episodes of the training people.
    alpha_p = inf (no coupling, the query ~ pi) whenever a person's other clips say nothing about the query's label --
    every acted corpus, where each actor records every emotion equally often."""
    rng = np.random.default_rng(seed)
    q = rows if len(rows) <= nq else rng.choice(rows, nq, replace=False)
    ctx = contexts(meta, q, K, "rand", rng, rows)
    lab = meta.label.values
    M = np.stack([np.bincount(lab[c], minlength=C) for c in ctx]).astype(float)
    pi = np.bincount(lab[rows], minlength=C) / len(rows)
    yq = lab[q]
    ll = [np.log(((al * pi[None] + M) / (al + K))[np.arange(len(q)), yq]).mean() for al in ALPHA_P]
    return float(ALPHA_P[int(np.argmax(ll))]), [float(v) for v in ll]


def chi2_90(k):
    """90% quantile of chi^2 with k degrees of freedom"""
    from scipy.stats import chi2
    return float(chi2.ppf(0.9, k))


STACK_BRANCHES = ["cond", "eb", "bn", "src"]  # branch 0 (exact Bayes given N) is the reference of the trust logits
STACK_KINDS = ["stack", "stack_nobn", "stack_nosrc", "flat", "gate_conc", "gate_ent", "gate_learn"]
GATE_FEATS = {"gate_conc": ["Eg"], "gate_ent": ["H_src"], "gate_learn": ["H_src", "unc_src", "pl_var", "Eg", "H_N"]}
# which of the three non-reference branches (eb, bn, src) a combiner may use
STACK_MASK = {"stack": (True, True, True), "stack_nobn": (True, False, True), "stack_nosrc": (True, True, False)}


def stack_inputs(pN, pc, th, C, p_eb, p_bn, p_src, pl_var, H_N):
    """sufficient statistics of the four predictive distributions for every combiner: the posterior mass P_v and the
    conditional Bayes votes S_v = sum_{N: g(N) = v} p(N | U) p(y | N, U) grouped by identifiability level v (a set of
    n clips has only a handful of distinct g values), the conservative, batch-moment and source predictions, and the
    episode features of the gate controls"""
    g = idf_weight(th, C)
    gv, lev = torch.unique(torch.round(g * 1e4) / 1e4, return_inverse=True)
    oh = F.one_hot(lev, len(gv)).to(pN.dtype)                      # (Hh, V)
    Pv = pN @ oh                                                    # (B, V)
    Sv = torch.einsum("bh,bhc,hv->bvc", pN, pc, oh)                 # (B, V, C)
    Eg = Pv @ gv
    H_src = -(p_src * p_src.clamp_min(1e-12).log()).sum(-1)
    feats = dict(Eg=Eg, H_src=H_src, unc_src=1 - p_src.max(-1).values, pl_var=pl_var, H_N=H_N)
    return dict(Sv=Sv, Pv=Pv, gv=gv, p_eb=p_eb, p_bn=p_bn, p_src=p_src, Eg=Eg, feats=feats)


def trust_logits(theta, gv, kind="stack"):
    """(V, 3) logits of the conservative, batch-moment and source branches relative to the exact Bayes branch, linear in
    the identifiability g of the hypothesis; a branch the combiner does not use gets -inf"""
    out, j = [], 0
    for on in STACK_MASK[kind]:
        if on:
            out.append(theta[j] + theta[j + 1] * gv); j += 2
        else:
            out.append(torch.full_like(gv, -INF))
    return torch.stack(out, -1)


def stack_combine(SI, kind, par):
    """Identifiability-stacked prediction and the control combiners of the same four distributions:
      stack        p = sum_N p(N|U) sum_k w_k(g_N) p_k(y | N, U),   k in {B: exact Bayes given N, E: conservative
                   posterior-moment shrinkage, M: batch moments (BN-adapt), S: source},
                   (w_B, w_E, w_M, w_S)(g) = softmax(0, a_E + b_E g, a_M + b_M g, a_S + b_S g):
                   the four classical estimators of the person's normalisation -- hypothesis-conditional, empirical
                   Bayes, the composition-agnostic moment estimator (the batch's own moments, i.e. the plug-in under the
                   balanced composition) and none (the population's) -- and every composition hypothesis splits its vote
                   among them as a function of its identifiability g = (C|theta|^2 - 1)/(C - 1) (Prop. 1); the six
                   numbers are fitted by the class-balanced log score of held-out training people (fit_stacking)
      stack_nobn   without the batch-moment branch (the round-10a design);  stack_nosrc: without the source branch
      flat         constant weights (no identifiability covariate)
      wg           round 9: w_E = g, w_B = 1 - g (fixed, not fitted)
      gate_*       episode-level softmax gates on generic features (concentration E[g], Source entropy, five learned
                   features) instead of per-hypothesis identifiability; same branches, held-out people and objective"""
    Sv, Pv, gv = SI["Sv"], SI["Pv"], SI["gv"]
    Pb = torch.stack([SI["p_eb"], SI["p_bn"], SI["p_src"]], 1)    # (B, 3, C)
    if kind == "wg":
        q = (Sv * (1 - gv)[None, :, None]).sum(1) + (Pv @ gv)[:, None] * SI["p_eb"]
    elif kind in STACK_MASK or kind == "flat":
        th_ = torch.as_tensor(par["theta"], dtype=Sv.dtype, device=Sv.device)
        lg = th_[None, :3].expand(len(gv), 3) if kind == "flat" else trust_logits(th_, gv, kind)
        w = torch.cat([torch.zeros_like(gv)[:, None], lg], -1).softmax(-1)   # (V, 4)
        q = (Sv * w[None, :, :1]).sum(1) + torch.einsum("bv,vk,bkc->bc", Pv, w[:, 1:], Pb)
    else:
        th_ = torch.as_tensor(par["theta"], dtype=Sv.dtype, device=Sv.device)
        fe = torch.stack([SI["feats"][k] for k in GATE_FEATS[kind]], -1)
        fe = (fe - torch.as_tensor(par["mu"], dtype=fe.dtype, device=fe.device)) / torch.as_tensor(par["sd"], dtype=fe.dtype, device=fe.device)
        Th = th_.view(3, 1 + fe.shape[1])
        lg = Th[:, 0][None] + fe @ Th[:, 1:].T                                # (B, 3)
        w = torch.cat([torch.zeros_like(lg[:, :1]), lg], -1).softmax(-1)      # (B, 4)
        q = w[:, :1] * Sv.sum(1) + (w[:, 1:, None] * Pb).sum(1)
    return q / q.sum(-1, keepdim=True).clamp_min(1e-30)


def n_stack_params(kind):
    if kind in STACK_MASK:
        return 2 * sum(STACK_MASK[kind])
    if kind == "flat":
        return 3
    return 3 * (1 + len(GATE_FEATS[kind]))


def fit_stack_params(SI, y, C, kind, ridge=1e-3):
    """class-balanced log score of held-out people (the proper scoring rule of the balanced risk that UAR measures),
    L-BFGS over the few combiner parameters; the ridge only keeps the optimisation well-posed when a branch is never
    useful (separable data)"""
    par = {}
    if kind.startswith("gate"):
        fe = torch.stack([SI["feats"][k] for k in GATE_FEATS[kind]], -1)
        par["mu"] = fe.mean(0).tolist(); par["sd"] = fe.std(0).clamp_min(1e-6).tolist()
    cnt = torch.bincount(y, minlength=C).float()
    wy = (1.0 / (C * cnt.clamp_min(1) / cnt.sum()))[y]
    theta = torch.zeros(n_stack_params(kind), device=y.device, dtype=SI["Sv"].dtype, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=300, line_search_fn="strong_wolfe")
    ar = torch.arange(len(y), device=y.device)

    def closure():
        opt.zero_grad()
        q = stack_combine(SI, kind, dict(par, theta=theta))
        loss = -(wy * torch.log(q[ar, y] + 1e-8)).sum() / wy.sum() + ridge * (theta ** 2).sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    th_ = theta.detach()
    if not torch.isfinite(th_).all():  # numerical failure: the constant combination (never silently a NaN predictor)
        th_ = torch.zeros_like(th_)
    par["theta"] = th_.tolist()
    with torch.no_grad():
        q = stack_combine(SI, kind, par)
        par["heldout_bal_logscore"] = float((wy * torch.log(q[ar, y] + 1e-8)).sum() / wy.sum())
    return par


def heldout_episodes(Sx, X, meta, rows, a, C, seed, nq, cfg, key):
    """the cross-fit episodes of held-out training people (rand, dir02 and same, the protocols of the posterior
    calibration): one pass of the CPN computation per chunk, returning what `key` asks for"""
    rng = np.random.default_rng(seed + 66)
    q = rows if len(rows) <= nq else np.sort(rng.choice(rows, nq, replace=False))
    Yl = torch.tensor(meta.label.values, device=dev)
    for proto in ("rand", "dir02", "same"):
        ctx = contexts(meta, q, a.kctx, proto, np.random.default_rng(seed + 5), rows)
        idx = torch.cat([torch.from_numpy(q).to(dev)[:, None], torch.from_numpy(ctx).to(dev)], 1)
        ch = chunk_size("cpn", idx.shape[1], C)
        for i in range(0, len(idx), ch):
            ii = idx[i:i + ch]
            _, D = cmn(Sx, X[ii], cfg, yr=Yl[ii[:, 1:]] if cfg.get("oracle") else None)
            yield proto, D[key], Yl[ii[:, 0]]


@torch.no_grad()
def fit_stacking(Sx, X, meta, rows, a, C, seed, nq=480, method="cpn"):
    """Cross-fit of the identifiability-stacked combiner and of the control combiners, on held-out training people
    (rand, dir02 and same episodes pooled, the protocols of the posterior calibration), for the CPN variant `method`
    (an ablation variant gets its own combiner: one component changed, everything else re-fitted by the same
    procedure).  Also stores the learned trust curve w(g) and the held-out balanced log score of every branch alone."""
    cfg = dict(cmn_cfg(method), return_stack=True)
    ch_, ys = [], []
    for _, SI_, y_ in heldout_episodes(Sx, X, meta, rows, a, C, seed, nq, cfg, "stack_in"):
        ch_.append(SI_); ys.append(y_)
    gv = ch_[0]["gv"]
    assert all(torch.equal(c["gv"], gv) for c in ch_), "identifiability levels must not depend on the batch"
    SI = dict(Sv=torch.cat([c["Sv"] for c in ch_]), Pv=torch.cat([c["Pv"] for c in ch_]), gv=gv,
              p_eb=torch.cat([c["p_eb"] for c in ch_]), p_bn=torch.cat([c["p_bn"] for c in ch_]),
              p_src=torch.cat([c["p_src"] for c in ch_]), Eg=torch.cat([c["Eg"] for c in ch_]),
              feats={k: torch.cat([c["feats"][k] for c in ch_]) for k in ch_[0]["feats"]})
    y = torch.cat(ys)
    out = {}
    with torch.enable_grad():
        for kind in STACK_KINDS:
            out[kind] = fit_stack_params(SI, y, C, kind)
    cnt = torch.bincount(y, minlength=C).float(); wy = (1.0 / (C * cnt.clamp_min(1) / cnt.sum()))[y]
    ar = torch.arange(len(y), device=y.device)
    score = lambda q_: float((wy * torch.log(q_[ar, y] + 1e-8)).sum() / wy.sum())  # noqa: E731
    diag = dict(cond=score(SI["Sv"].sum(1)), eb=score(SI["p_eb"]), bn=score(SI["p_bn"]), src=score(SI["p_src"]),
                wg=score(stack_combine(SI, "wg", None)), stack=out["stack"]["heldout_bal_logscore"])
    gg = torch.linspace(0, 1, 11)
    lg = trust_logits(torch.tensor(out["stack"]["theta"]), gg)
    curve = torch.cat([torch.zeros_like(gg)[:, None], lg], -1).softmax(-1)  # (11, 4): w_B, w_E, w_M, w_S
    return dict(stack_params=out, heldout_bal_logscore=diag, trust_curve=dict(g=gg.tolist(), w=curve.tolist(),
                branches=STACK_BRANCHES), n_episodes=int(len(y)), method=method)


# log-spaced axes of the conditional-ML search of the posterior weights; the search is EXTENDED automatically while the
# optimum lies on an outer edge (round 10b: in round 9 the anchor weight sat at the old edge 4 on every speech fold and
# the evidence weights at 0.5 on MEAD), so the reported weights are an interior optimum, not an artefact of the range
POST_T = [0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0]   # weights of the network evidence / the density
POST_A = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]    # weight of the set anchor (0 = no anchor: a natural edge)
POST_EXT = 3                                      # at most this many extensions per axis


@torch.no_grad()
def posterior_cll(Sx, X, meta, q, rows, a, C, seed, cfg, T_net, T_dens, T_anc):
    """mean held-out log p(N_true | U) of every weight triple of the grid T_net x T_dens x T_anc, per protocol"""
    lab = meta.label.values
    grid = [(tn, td, ta) for tn in T_net for td in T_dens for ta in T_anc]
    tot = {g: {p_: 0.0 for p_ in ("rand", "dir02", "same")} for g in grid}
    cnt = {p_: 0 for p_ in ("rand", "dir02", "same")}
    for proto in ("rand", "dir02", "same"):
        ctx = contexts(meta, q, a.kctx, proto, np.random.default_rng(seed + 5), rows)
        idx_np = np.concatenate([q[:, None], ctx], 1)
        idx = torch.from_numpy(idx_np).to(dev)
        n = idx.shape[1]
        SL = sublattice(n, C)
        Hnp = SL["H"].long().cpu().numpy()
        where = {tuple(r): i for i, r in enumerate(Hnp)}
        htrue = torch.tensor([where[tuple(np.bincount(lab[r], minlength=C))] for r in idx_np], device=dev)
        ch = chunk_size("cpn", n, C)
        for i in range(0, len(idx), ch):
            _, D = cmn(Sx, X[idx[i:i + ch]], cfg)
            hb = htrue[i:i + ch]; ar = torch.arange(len(hb), device=dev)
            for tn in T_net:
                base = count_dp(SL, tn * D["unary"]) + tn * D["qterm"] + D["Tprior"]
                for td in T_dens:
                    bd = base + td * D["dens"][..., None]
                    for ta in T_anc:
                        lN = (bd + ta * D["anc"][..., None]).logsumexp(2)
                        tot[(tn, td, ta)][proto] += float((lN[ar, hb] - lN.logsumexp(1)).sum())
            cnt[proto] += len(hb)
    return {g: {p_: tot[g][p_] / max(cnt[p_], 1) for p_ in cnt} for g in grid}


@torch.no_grad()
def calibrate_posterior(Sx, X, meta, rows, a, C, seed, nq=480, method="cpn"):
    """Calibrated composition posterior (round 9).  The composition log-posterior is a log-linear model of three
    evidence families -- network evidence of every clip under every hypothesis' transport (inside the count DP),
    population density of the location-normalised clips, identifiability-weighted set anchor -- plus the composition
    prior.  Their weights (t_net, t_dens, t_anc) maximise the conditional log-likelihood log p(N_true | U) of the TRUE
    compositions of held-out training people (the auxiliary network has not seen them), pooled over rand, dir02 and
    same episodes: the exact posterior of round 8 was under-confident on mixed sets and over-confident on single-class
    sets, because a hypothesis' own transport makes its evidence partly self-confirming.  The search runs on log-spaced
    axes that are extended (x2 / /2) while the optimum lies on an outer edge (round 10b), so the weights are an interior
    CML optimum.  Returns the weights, the searched axes and the held-out log-likelihood of the true composition before
    (weights 1) and after, per protocol."""
    rng = np.random.default_rng(seed + 55)
    q = rows if len(rows) <= nq else np.sort(rng.choice(rows, nq, replace=False))
    cfg = dict(cmn_cfg(method), post_w=(1.0, 1.0, 1.0), return_terms=True)
    # a family the variant does not have (no density / no set anchor) has no weight to fit: its axis is {1}
    has_dens = bool(cfg["dens"]) and cfg["lik"] in ("hyb", "bnmode")
    has_anc = cfg["anchor"] in ("net", "idf")
    axes = [list(POST_T), list(POST_T) if has_dens else [1.0], list(POST_A) if has_anc else [1.0]]
    mean, ext = {}, [0, 0, 0]
    todo = [list(ax) for ax in axes]
    while True:
        # only the weight triples not evaluated yet (an extension adds one value on one axis)
        new = posterior_cll(Sx, X, meta, q, rows, a, C, seed, cfg, *todo) if all(todo) else {}
        mean.update(new)
        grid = [(tn, td, ta) for tn in axes[0] for td in axes[1] for ta in axes[2]]
        pooled = {g: float(np.mean(list(mean[g].values()))) for g in grid}
        best = max(pooled, key=pooled.get)
        grow = None
        for k in range(3):
            ax, v = axes[k], best[k]
            if len(ax) == 1:
                continue
            if ext[k] < POST_EXT and v == ax[-1]:
                grow = (k, ax[-1] * 2.0, True); break
            if ext[k] < POST_EXT and v == ax[0] and ax[0] > 0:
                grow = (k, ax[0] / 2.0, False); break
        if grow is None:
            break
        k, v, up = grow
        ext[k] += 1
        axes[k] = axes[k] + [v] if up else [v] + axes[k]
        todo = [list(ax) for ax in axes]; todo[k] = [v]
    edge = [bool(len(axes[k]) > 1 and best[k] in (axes[k][0], axes[k][-1]) and not (k == 2 and best[k] == 0.0))
            for k in range(3)]
    return dict(post_w=list(best), post_ll_before=mean[(1.0, 1.0, 1.0)], post_ll_after=mean[best],
                post_ll_pooled_before=float(np.mean(list(mean[(1.0, 1.0, 1.0)].values()))), post_ll_pooled_after=pooled[best],
                axes=axes, at_edge=edge, extensions=ext)


def crossfit(X, meta, tr, a, seed, fold, C, kind="iid", light=False, aux_path=None):
    """Auxiliary network (same architecture and training as the main one) on 3/4 of the training subjects; on the held-out
    1/4 (people it has not seen), unit-free calibration constants that transfer to the main network:
      kappa         tr Cov(held-out balanced means) / tr Cov(fit-subject balanced means) in span(F): the prior spread of
                    a new person's centre relative to the fitted people's (the network absorbs the people it was fitted
                    on, so the fitted spread underestimates a new person's)
      tau, tau_tp   ML temperatures of the person-mode classifier at the TRUE composition hypothesis, on n-clip episodes
                    of held-out people, n = kctx + 1 (tau: round-7 standardisation; tau_tp: round-8 transport)
      lam_m, lam_s  mean and spread of the log-expressivity of held-out people relative to the fitted statistics
                    (the population prior on the person scale of scale-equivariant hypotheses)
      alpha_p       EB concentration of the person-level label distributions (labels only, all training people;
                    appendix only)
      post_w        weights of the evidence families of the composition posterior (conditional ML of the TRUE
                    compositions of held-out people: calibrate_posterior)
      tau_bn        ML temperature of the batch-moment (BN-mode) prediction on the same held-out episodes
      stack_params  the identifiability-stacking combiner and the control combiners (class-balanced log score of
                    held-out people: fit_stacking)
    aux_path: the auxiliary network, its constants and held-out rows are saved there for the ablation re-fits.
    light=True: only the constants (no grids of the baselines) -- design pilots."""
    subs = np.unique(meta.actor.values[tr])
    rng = np.random.default_rng(seed * 100 + fold + 7)
    perm = rng.permutation(subs)
    nho = min(max(2, int(round(0.25 * len(subs)))), len(subs) - 2)
    ho = perm[:nho]
    r_ho, r_in = tr[np.isin(meta.actor.values[tr], ho)], tr[~np.isin(meta.actor.values[tr], ho)]
    torch.manual_seed(seed * 100 + fold + 7)
    Sx = make_network(kind, X, meta, r_in, a, rng, C)
    s_in, s_ho = Sx.st, subject_stats(Sx, X, meta, r_ho, C)
    with torch.no_grad():
        bh = s_ho["b"] @ s_in["Q"]
        kappa = float(torch.trace(torch.cov(bh.T)) / torch.trace(s_in["T"]))
        Pp = torch.eye(s_in["Q"].shape[0], device=dev) - s_in["Q"] @ s_in["Q"].T  # orthogonal complement of span(F)
        kappa_perp = float((s_ho["b"] @ Pp).var(0).sum() / (s_in["b"] @ Pp).var(0).sum())
        le_ho, le_in = log_expressivity(Sx, X, meta, r_ho, s_in), log_expressivity(Sx, X, meta, r_in, s_in)
        n = a.kctx + 1
        smp = EpisodeSampler(meta, r_ho, C, seed * 100 + fold + 11)
        ep = smp(4096, n, iid_frac=1.0)  # random clips of held-out people
        y = torch.tensor(meta.label.values, device=dev)
        u = pop_units(Sx, X[ep]); th = F.one_hot(y[ep], C).float().mean(1)
        lg = head(Sx, person_norm(u, th, s_in, n, 1.0, "equi", Sx.bn.eps)).flatten(0, 1)
        yh = y[ep].flatten()
        tau = fit_temperature(lg, yh)
        acc = float((lg.argmax(-1) == yh).float().mean() * 100)
        mu_t, _ = person_centre(u, th, s_in, n)
        sc_t = torch.sqrt(person_scale(u, th, s_in, n, 1.0, "equi") / target_var(s_in))  # per-unit expressivity
        lgt = head_pop(Sx, transport(s_in, u, mu_t[:, None], sc_t[:, None])).flatten(0, 1)
        tau_tp = fit_temperature(lgt, yh)
        acc_tp = float((lgt.argmax(-1) == yh).float().mean() * 100)
        ubn = pop_units(Sx, X[ep]); hb = ubn  # BN on the same episodes (reference for the oracle normalisation)
        lgb = head(Sx, (hb - hb.mean(1, keepdim=True)) / torch.sqrt(hb.var(1, unbiased=False, keepdim=True) + Sx.bn.eps)).flatten(0, 1)
        acc_bn = float((lgb.argmax(-1) == yh).float().mean() * 100)
        tau_bn = fit_temperature(lgb, yh)
        lgs = head_pop(Sx, u).flatten(0, 1)
        acc_src = float((lgs.argmax(-1) == yh).float().mean() * 100)
        tau_src = fit_temperature(lgs, yh)
        lam_m, lam_s = float(le_ho.mean() - le_in.mean()), (float(le_ho.std(ddof=1)) if len(le_ho) > 1 else 0.25)
        alpha_p, ll_p = fit_alpha_person(meta, tr, C, a.kctx, seed * 100 + fold + 13)
        out = dict(kappa=kappa, kappa_perp=kappa_perp, tau=tau, tau_tp=tau_tp, tau_src=tau_src, tau_bn=tau_bn, lam_m=lam_m, lam_s=lam_s, lam_s_fit=float(le_in.std(ddof=1)),
                   alpha_p=alpha_p, alpha_p_loglik=ll_p,
                   aux_heldout_truecomp_acc=acc, aux_heldout_truecomp_tp_acc=acc_tp, aux_heldout_bn_acc=acc_bn,
                   aux_heldout_source_acc=acc_src, n_heldout=int(nho), n_fit=int(len(subs) - nho))
    Sx.xfit = {k: out[k] for k in XFIT_BASE}
    if aux_path:  # the auxiliary network and its held-out people: the ablation variants re-fit on exactly these
        tmp = aux_path + f".{os.getpid()}.tmp"
        torch.save(dict(weights=Sx.state_dict(), stats=Sx.st, xfit=dict(Sx.xfit), r_ho=r_ho), tmp)
        os.replace(tmp, aux_path)
    pc = calibrate_posterior(Sx, X, meta, r_ho, a, C, seed * 100 + fold)
    out.update(post_w=pc["post_w"], posterior_calibration=pc)
    Sx.xfit["post_w"] = pc["post_w"]
    st_ = fit_stacking(Sx, X, meta, r_ho, a, C, seed * 100 + fold)
    out.update(stack_params=st_["stack_params"], stacking=st_)
    Sx.xfit["stack_params"] = st_["stack_params"]
    if light:
        return out
    attach_csn(Sx, X, meta, r_in, C)
    out["val"] = validate_grids(Sx, X, meta, r_ho, a, C, seed * 100 + fold)
    out["cert"] = certify(Sx, X, meta, r_ho, a, C, seed * 100 + fold)
    return out


XFIT_BASE = ("kappa", "kappa_perp", "tau", "tau_tp", "tau_src", "tau_bn", "lam_m", "lam_s", "alpha_p")
# ablation variants whose posterior calibration and stacking combiner are RE-FITTED (one component changed, everything
# else by the same procedure); the pruned variants approximate the full model and keep its constants
VARIANT_REFIT = {"cpn_nopostcal", "cpn_nodens", "cpn_noanchor", "cpn_bnlik", "cpn_std", "cpn_oracle",
                 "cpn_plainanchor", "cpn_indepG", "cpn_noint", "cpn_noequi"}


def refit_variant(method, aux_path, X, meta, a, fold, C):
    """posterior weights (if the variant calibrates its posterior; the label oracle keeps the full model's, its
    posterior is restricted by the labels) and stacking combiner of an ablation variant, on the cached auxiliary
    network and held-out people of the cross-fit"""
    sv = torch.load(aux_path, map_location=dev, weights_only=False)
    Sx = Source(X.shape[1], C).to(dev)
    Sx.load_state_dict(sv["weights"]); Sx.st = sv["stats"]; Sx.eval().requires_grad_(False)
    Sx.xfit = dict(sv["xfit"])
    r_ho = np.asarray(sv["r_ho"])
    cfg = cmn_cfg(method)
    out = dict(method=method)
    if cfg.get("post_cal") and not cfg.get("oracle"):
        pc = calibrate_posterior(Sx, X, meta, r_ho, a, C, a.seed * 100 + fold, method=method)
        out.update(post_w=pc["post_w"], posterior_calibration=pc)
        Sx.xfit["post_w"] = pc["post_w"]
    st_ = fit_stacking(Sx, X, meta, r_ho, a, C, a.seed * 100 + fold, method=method)
    out.update(stack_params=st_["stack_params"], stacking=st_)
    return out


CERT_PROTOS = ["same", "rand", "dir02"]


@torch.no_grad()
def certify(Sx, X, meta, rows, a, C, seed, nq=480):
    """Pre-deployment certificate (computed before any test person is seen): CPN and Source on held-out TRAINING
    people with the auxiliary network, under the pre-specified hard protocol (all references share the query's class)
    and the label-free protocols.  Saves per-clip correctness and person ids; the aggregator turns them into the
    identity-bootstrap lower bound of CPN - Source and reports it next to the test-time non-inferiority result: the
    certificate predicts, before deployment, on which population the safety claim can be made."""
    rng = np.random.default_rng(seed + 321)
    q = rows if len(rows) <= nq else np.sort(rng.choice(rows, nq, replace=False))
    Y = torch.tensor(meta.label.values, device=dev)
    out = dict(actor=[str(v) for v in meta.actor.values[q]])
    for proto in CERT_PROTOS:
        ctx = contexts(meta, q, a.kctx, proto, np.random.default_rng(seed + 9), rows)
        idx = torch.cat([torch.from_numpy(q).to(dev)[:, None], torch.from_numpy(ctx).to(dev)], 1)
        res = {"cpn": [], "source": []}
        ch = chunk_size("cpn", idx.shape[1], C)
        for i in range(0, len(idx), ch):
            xb, yq = X[idx[i:i + ch]], Y[idx[i:i + ch, 0]]
            res["cpn"] += (cmn(Sx, xb, cmn_cfg("cpn"))[0].argmax(1) == yq).int().tolist()
            res["source"] += (p_source(Sx, xb).argmax(1) == yq).int().tolist()
        out[proto] = res
    return out


VAL_PROTOS = ["rand", "skew", "dir02"] + [f"mix{j}" for j in range(7)]


def validate_grids(Sx, X, meta, rows, a, C, seed, nq=600):
    """accuracy of every grid value of the simple baselines on held-out TRAINING people (auxiliary network): the
    validation data from which the aggregator selects their operating points.  {method: {protocol: [acc]}}"""
    rng = np.random.default_rng(seed + 123)
    q = rows if len(rows) <= nq else np.sort(rng.choice(rows, nq, replace=False))
    Y = torch.tensor(meta.label.values, device=dev)
    out = {m: {} for m in GRIDS}
    for proto in VAL_PROTOS:
        md, j = ("mix", int(proto[3:])) if proto.startswith("mix") else (proto, 0)
        ctx = contexts(meta, q, a.kctx, md, np.random.default_rng(seed + 7 + j), rows, j)
        idx = torch.cat([torch.from_numpy(q).to(dev)[:, None], torch.from_numpy(ctx).to(dev)], 1)
        for m in GRIDS:
            acc = {}
            for i in range(0, len(idx), 256):
                _, d = tta(Sx, X[idx[i:i + 256]], m, C)
                for k, v in d["alt"].items():
                    acc[k] = acc.get(k, 0) + float((v.argmax(1) == Y[idx[i:i + 256, 0]]).sum())
            out[m][proto] = [100 * acc[grid_label(m, v)] / len(idx) for v in GRIDS[m]]
    return out


@torch.no_grad()
def attach_csn(model, X, meta, rows, C):
    """per-class channel statistics distances of CSN (training rows of the network)"""
    h = X[rows] @ model.f1.weight.T + model.f1.bias
    ytr = torch.tensor(meta.label.values[rows], device=dev)
    ms, sds = model.bn.running_mean, torch.sqrt(model.bn.running_var + model.bn.eps)
    model.csn_score = torch.stack([(h[ytr == c].mean(0) - ms) ** 2 + (h[ytr == c].std(0) - sds) ** 2 for c in range(C)])


# ============================================================================ CMN
CMN_DEFAULT = dict(dens=True, anchor="net", student=True, kappa=True, temp=True, lik="hyb", prior="dm", pred="joint",
                   scale="equi", nu=1.0, oracle=False, shared_G=True, dens_scale="int", dens_cov="full", shrink_k=1.0,
                   norm="std", err="comp", lab=False, perp="full", cov=True, prune=None, rho=None)
CPN7 = dict(anchor="idf", pred="shrink")  # round 7: identifiability-weighted anchor + posterior-moment EB read-out
# round 8 (the paper's method): per-unit Source-anchored transport (evidence and read-out), calibrated centre-error
# covariance, empirical-Bayes shrinkage of the UNFAMILIAR part of the person deviation
# round 10 (the paper's method): calibrated composition posterior, its four predictive distributions (exact Bayes per
# composition, posterior-moment shrinkage, batch moments, source) combined by identifiability-stacking; the shrinkage branch uses the
# model's own (uncalibrated) error covariance and no adaptation-strength scalar -- round-9 components that the
# ablations showed to be ineffective (fourth error component, four-factor calibration, beta) are removed
CPN = dict(CPN7, norm="tpunit", err="full", rho=(1.0, 1.0, 1.0, 0.0), prior_c=None, lab=False, post_cal=True, pred="stack")
# one component of the conservative branch changed inside the stacked predictor (saved by every "cpn" run)
READOUT_ALTS = {
    "nocov": dict(cov=False),                           # no EB shrinkage in the conservative branch
    "r_std": dict(norm="std"),                          # round-7 standardisation instead of the transport
}


CMN_VARIANTS = {  # method -> overrides of CMN_DEFAULT
    "cmn": {},                                           # round-6 method
    "cpn7": dict(CPN7),                                  # round-7 method
    "cpn": dict(CPN),                                    # round-10 method (the paper's)
    # ---- posterior-level ablations (one component changed; posterior weights and combiner re-fitted: VARIANT_REFIT)
    "cpn_nopostcal": dict(CPN, post_cal=False),          # - calibrated composition posterior
    "cpn_nodens": dict(CPN, dens=False),                 # - population density
    "cpn_noanchor": dict(CPN, anchor="none"),            # - set anchor
    "cpn_bnlik": dict(CPN, lik="bnmode"),                # network evidence under ONE batch-mode normalisation
    "cpn_std": dict(CPN, norm="std"),                    # - source-anchored transport (evidence and read-outs)
    "cpn_oracle": dict(CPN, oracle=True),                # label oracle: the references' true labels
    # ---- model details (appendix)
    "cpn_plainanchor": dict(CPN, anchor="net"), "cpn_indepG": dict(CPN, shared_G=False),
    "cpn_noint": dict(CPN, dens_scale="one"), "cpn_noequi": dict(CPN, scale="bal"),
    # ---- two-stage pruned inference (efficiency only)
    "cpn_p64": dict(CPN, prune=64), "cpn_p256": dict(CPN, prune=256),
}
CMN_ALTS = ["top1", "cond", "eb", "bn", "src", "wg", "stack", "stack_map"] + STACK_KINDS[1:] + ["st:" + k for k in READOUT_ALTS]


def batch_var(th, st, n, Fc=None):
    """expected batch variance per unit (at unit expressivity) of n clips with EXACT composition counts theta (.., C):
        V(theta) = Var_theta F + (1 - |theta|^2) sigma_G^2 + (n-1)/n sigma_in^2
    (class displacements, the person x class interaction shared by the clips of one class, clip noise)"""
    Fc = st["Fc"] if Fc is None else Fc
    m = th @ Fc
    return th @ Fc ** 2 - m ** 2 + (1 - (th ** 2).sum(-1, keepdim=True)) * st["sg2"] + (n - 1) / n * st["sin2"]


def target_var(st, Fc=None):
    """variance per unit of a person's balanced repertoire at unit expressivity: the scale of the normalised units"""
    Fc = st["Fc"] if Fc is None else Fc
    C = Fc.shape[0]
    return (Fc ** 2).mean(0) + (1 - 1 / C) * st["sg2"] + st["sin2"]


def hypothesis_scale(s2, V, st, equi):
    """person expressivity implied by composition hypotheses (scale-equivariant hypotheses).
    A person's repertoire is u = mu_s + lam_s (F_y + G_sy + e), lam_s = 1 for the training population.  Under N the
    expected batch variance is lam^2 V(theta); its inverse-noise-weighted moment estimate is
        lam^2(N) = sum_h w_h s_h^2 / sum_h w_h V_h(theta),   w_h = 1 / (sigma_in,h^2 + sigma_G,h^2).
    With lam fixed to 1 (scale='bal', the previous CMN) a low-expressivity person is over-corrected by theta^T F and
    squeezed to the population scale.  s2: (B, H), V: (Hh, H) -> (B, Hh)"""
    if not equi:
        return torch.ones(s2.shape[0], V.shape[0], device=s2.device)
    w = 1.0 / (st["sin2"] + st["sg2"])
    return (s2 * w).sum(-1, keepdim=True) / (V * w).sum(-1)[None]


def scale_grid(S, device):
    """quadrature of the population prior on a new person's log-expressivity, N(lam_m, lam_s^2), cross-fitted on
    held-out people (13 points over +-3 sd)"""
    m, s = S.xfit.get("lam_m", 0.0), max(S.xfit.get("lam_s", 0.25), 0.02)
    tg = torch.linspace(-3, 3, 13, device=device)
    return torch.exp(m + s * tg), (-0.5 * tg ** 2).log_softmax(0)


def person_norm(u, th, st, n, nu, scale, eps, Fc=None):
    """normalised units of the clips u (B, n, H) of one person under ONE composition theta (B, C) per episode.
    Identical to the per-hypothesis normalisation inside cmn()."""
    Fc = st["Fc"] if Fc is None else Fc
    ub, s2 = u.mean(1), u.var(1, unbiased=False)
    m = th @ Fc
    V = batch_var(th, st, n, Fc)
    if scale == "equi":
        w = 1.0 / (st["sin2"] + st["sg2"])
        lam2 = (s2 * w).sum(-1, keepdim=True) / (V * w).sum(-1, keepdim=True)
        lu = (n * s2 + nu * lam2 * V) / ((n + nu) * V)  # per unit, shrunk to the pooled lam^2
        mu, var = ub - lam2.sqrt() * m, lu * target_var(st, Fc)
    elif scale == "bal":
        mu = ub - m
        var = (n * (s2 - (th @ Fc ** 2 - m ** 2)).clamp_min(0) + nu * st["sig2"]) / (n + nu) + (Fc ** 2).mean(0)
    elif scale == "pop":
        mu, var = ub - m, (st["sig2"] + (Fc ** 2).mean(0)).expand_as(ub)
    else:
        mu, var = ub - m, s2
    return (u - mu[:, None]) / torch.sqrt(var[:, None] + eps)


def person_scale(u, th, st, n, nu, scale, Fc=None):
    """per-unit variance of the person normalisation under ONE composition theta (B, C) (as person_norm)"""
    Fc = st["Fc"] if Fc is None else Fc
    s2 = u.var(1, unbiased=False)
    m = th @ Fc
    V = batch_var(th, st, n, Fc)
    if scale == "equi":
        w = 1.0 / (st["sin2"] + st["sg2"])
        lam2 = (s2 * w).sum(-1, keepdim=True) / (V * w).sum(-1, keepdim=True)
        return (n * s2 + nu * lam2 * V) / ((n + nu) * V) * target_var(st, Fc)
    if scale == "bal":
        return (n * (s2 - (th @ Fc ** 2 - m ** 2)).clamp_min(0) + nu * st["sig2"]) / (n + nu) + (Fc ** 2).mean(0)
    if scale == "pop":
        return (st["sig2"] + (Fc ** 2).mean(0)).expand_as(s2)
    return s2


def person_centre(u, th, st, n, Fc=None):
    """centre and expressivity of the person under ONE composition theta (B, C) per episode (scale-equivariant
    hypothesis): mu = mean(u) - lam theta^T F, lam^2 = inverse-noise-weighted moment estimate (hypothesis_scale)"""
    Fc = st["Fc"] if Fc is None else Fc
    ub, s2 = u.mean(1), u.var(1, unbiased=False)
    V = batch_var(th, st, n, Fc)
    w = 1.0 / (st["sin2"] + st["sg2"])
    lam = ((s2 * w).sum(-1, keepdim=True) / (V * w).sum(-1, keepdim=True)).sqrt()
    return ub - lam * (th @ Fc), lam


def idf_weight(th, C):
    """g(theta) = (C |theta|^2 - 1) / (C - 1) = Var(sum_c theta_c G_c) / v_G: the share of the person x class
    interaction G (which sums to zero over a person's classes) that the mean of a set with composition theta keeps --
    0 for a balanced set, 1 for a single-class set.  It weights the set anchor (round 7) AND is the interaction part of
    the sampling error of the centre (round 8)."""
    return ((C * (th ** 2).sum(-1) - 1) / (C - 1)).clamp(0, 1)


def post_weights(S, cfg):
    """(t_net, t_dens, t_anc): weights of the network-evidence, population-density and set-anchor families of the
    composition log-posterior.  cfg['post_w'] explicit; cfg['post_cal'] True: the held-out conditional-maximum-
    likelihood weights of the cross-fit (calibrate_posterior); otherwise 1 (the uncalibrated posterior)."""
    if cfg.get("post_w") is not None:
        return tuple(cfg["post_w"])
    if cfg.get("post_cal") and hasattr(S, "xfit") and "post_w" in S.xfit:
        return tuple(S.xfit["post_w"])
    return 1.0, 1.0, 1.0


def calib_factors(S, cfg):
    """(rho_c, rho_G, rho_e, rho_N): calibration of the four variance components of the centre error -- composition
    uncertainty, person x class interaction, clip noise, non-identified class offset -- fitted on held-out people
    (round 9).  cfg['rho']: explicit tuple (2: round-8 composition / sampling, 3: without the non-identifiability
    component, 4: all); cfg['cal'] in ('two', 'three'): the round-8 / three-component calibration (ablations)."""
    r = cfg.get("rho")
    if r is not None:
        r = tuple(r)
        return (r[0], r[1], r[1], 0.0) if len(r) == 2 else (r + (0.0,) if len(r) == 3 else r)
    x = S.xfit
    if cfg.get("cal") == "two":
        r2 = x.get("rho2_s", x.get("rho_s", 1.0))
        return x.get("rho2_c", x.get("rho_c", 1.0)), r2, r2, 0.0
    if cfg.get("cal") == "three":
        return x.get("rho3_c", 1.0), x.get("rho3_G", 1.0), x.get("rho3_e", 1.0), 0.0
    return x.get("rho_c", 1.0), x.get("rho_G", x.get("rho_s", 1.0)), x.get("rho_e", x.get("rho_s", 1.0)), x.get("rho_N", 0.0)


def readout_tau(S, cfg):
    """cross-fitted temperature of the person-mode classifier of the given normalisation"""
    if not cfg["temp"]:
        return 1.0
    return S.xfit.get("tau_tp", S.xfit["tau"]) if cfg.get("norm", "std") in ("tp", "tpunit", "tpinfl") else S.xfit["tau"]


def cpn_readout(S, u, Mo, cfg, kap, Fc=None):
    """The CONSERVATIVE branch of the composition posterior: one normalisation at the posterior-mean centre, shrunk in
    span(F) towards the population centre by its posterior uncertainty.
    Mo: posterior moments  Elt = E[lam theta], Clt = Cov(lam theta), El = E[lam], El2 = E[lam^2],
        El2g = E[lam^2 g(theta)], Et = E[theta], pq = posterior of the query label (B, C).
    1) centre-error covariance in span(F), the model's own (round 10: rho = 1, no calibration factors):
           Sigma_err = f^T Clt f                  (the composition is uncertain)
                     + El2g SG + El2 Sin / n      (the clips are a finite, interaction-sharing sample)
       (rounds 8-9 multiplied the components by factors fitted on held-out people; the ablations showed no stable gain)
    2) empirical-Bayes centre with the cross-fitted prior N(m0, kappa T) of a new person's centre (optionally scaled
       by an adaptation strength beta: prior_c='heldout', round 9 only):
           mu = mbar + Q (W - I)(Q^T mbar - m0),   W = beta kappa T (kappa T + Sigma_err)^-1,   mbar = mean(u) - Elt^T F
       beta = 0: the source centre in span(F); beta = 1: the EB centre.  prior_c=None: beta = 1.  prior_c='excess'
       (appendix): beta replaced by (kappa - 1)_+ / kappa, "remove only the deviation the network has not seen" -- wrong
       whenever kappa ~ 1 although the source network does NOT tolerate the deviation (synthetic data).
       optionally (perp='eb') also outside span(F), per unit, with the cross-fitted kappa_perp.
    3) Source-anchored transport (norm='tpunit'): source head of bbar + (u_q - mu) * sqrt(v_pop / v_person), per unit
       (v_person: the person's within-person variance at the posterior-mean composition, v_pop: the population's); the
       source model is the zero-information limit (Prop. 11).  norm='tp': one scalar expressivity E[lam];
       norm='std': the round-7 standardisation to unit within-person variance, centred at 0.
    4) (lab='soft' / True, appendix only) person-level label prior: the query's label exchangeable with the
       references' labels under the EB concentration alpha_p of the training people's label distributions.
    returns probabilities (B, C), centre (B, H), shrinkage 1 - tr(W)/(C-1) (B,), diagnostics dict"""
    st = S.st; n = u.shape[1]
    Fc = st["Fc"] if Fc is None else Fc
    C = Fc.shape[0]
    Q = st["Q"]; f = Fc @ Q
    mbar = u.mean(1) - Mo["Elt"] @ Fc
    msub = mbar @ Q
    Sp = torch.einsum("cd,bce,ef->bdf", f, Mo["Clt"], f)
    SsG = Mo["El2g"][:, None, None] * st["SG"][None]          # person x class interaction kept by the set mean
    SsE = Mo["El2"][:, None, None] * st["Sin"][None] / n      # clip noise of n clips
    pi = st["pi"]
    Phi = f.T @ (torch.diag(pi) - pi[:, None] * pi[None]) @ f  # spread of the class displacements (population)
    SsN = Mo["El2g"][:, None, None] * Phi[None]               # non-identified class offset of the set (Prop. 1)
    Ss = SsG + SsE
    I = torch.eye(C - 1, device=u.device)
    if cfg.get("err", "comp") == "full":
        rc, rg, re_, rn = calib_factors(S, cfg)
        Serr = rc * Sp + rg * SsG + re_ * SsE + rn * SsN
    elif cfg.get("err") == "fixed":  # control: the calibrated error averaged over held-out episodes, the same for all
        Serr = torch.tensor(S.xfit["Serr_mean"], device=u.device, dtype=Sp.dtype)[None].expand_as(Sp)
    else:  # round 7: composition-posterior covariance only
        Serr = Sp
    excess = cfg.get("prior_c") == "excess"
    if cfg.get("cov", True):
        mq = msub - st["m0"]
        if cfg.get("tail"):  # heavy-tailed prior: scale mixture kappa * 2^j, j in [-2, 2], posterior weights per person
            ks = kap * 2.0 ** torch.linspace(-2, 2, 9, device=u.device)
            Vk = ks[:, None, None, None] * st["T"][None, None] + Serr[None] + 1e-6 * I  # (k, B, d, d)
            Vi = torch.linalg.inv(Vk)
            ll = -0.5 * torch.einsum("bd,kbde,be->kb", mq, Vi, mq) - 0.5 * torch.logdet(Vk)
            wk = ll.softmax(0)  # (k, B)
            fam = ((ks - 1.0).clamp_min(0) / ks) if excess else torch.ones_like(ks)
            Wk = fam[:, None, None, None] * (ks[:, None, None, None] * st["T"][None, None]) @ Vi
            W = torch.einsum("kb,kbde->bde", wk, Wk)
        else:
            kT = st["T"] * kap * cfg.get("shrink_k", 1.0)
            W = kT[None] @ torch.linalg.inv(kT[None] + Serr + 1e-6 * I)
            if excess:  # remove only the UNFAMILIAR part of the deviation (kappa-based variant, appendix)
                W = W * (max(kap - 1.0, 0.0) / kap)
        if cfg.get("prior_c") == "heldout":  # held-out maximum-likelihood adaptation strength (Prop. 12)
            W = W * (cfg["beta"] if cfg.get("beta") is not None else S.xfit.get("beta", 1.0))
        if cfg.get("gate") == "pl":  # control: simple uncertainty gate -- full EB centre if the independent
            # pseudo-label variance is below a held-out threshold, otherwise the source centre
            thr = cfg["gate_thr"] if cfg.get("gate_thr") is not None else S.xfit.get("gate_thr", INF)
            W = W * (Mo["pl_var"] <= thr).float()[:, None, None]
        mu = mbar + (torch.einsum("bde,be->bd", W, mq) - mq) @ Q.T
        shr = 1 - torch.diagonal(W, dim1=1, dim2=2).mean(-1)
    else:
        mu, shr = mbar, torch.zeros(u.shape[0], device=u.device)
    if cfg.get("perp", "full") == "eb":  # orthogonal complement of span(F): per-unit EB towards bbar
        bp = st["b"] - (st["b"] @ Q) @ Q.T
        kq = S.xfit.get("kappa_perp", kap)
        tpp = bp.var(0) * kq
        sp = Mo["El2g"][:, None] * st["sg2"] + Mo["El2"][:, None] * st["sin2"] / n
        w = tpp / (tpp + sp)
        if excess:
            w = w * (max(kq - 1.0, 0.0) / kq)
        P_ = torch.eye(Q.shape[0], device=u.device) - Q @ Q.T
        mp, bpp = mu @ P_, bbar(st) @ P_
        mu = mu - mp + bpp + w * (mp - bpp)
    nrm = cfg.get("norm", "std")
    if nrm == "tp":
        def norm_(v):
            return transport(st, v, mu[:, None] if v.dim() == 3 else mu,
                             Mo["El"][:, None, None] if v.dim() == 3 else Mo["El"][:, None])
    else:
        var = person_scale(u, Mo["Et"], st, n, cfg["nu"], cfg["scale"], Fc)
        if nrm == "tpunit":
            sc = torch.sqrt((var + S.bn.eps) / target_var(st, Fc))

            def norm_(v):
                return transport(st, v, mu[:, None] if v.dim() == 3 else mu, sc[:, None] if v.dim() == 3 else sc)
        else:
            def norm_(v):
                return (v - (mu[:, None] if v.dim() == 3 else mu)) / torch.sqrt((var[:, None] if v.dim() == 3 else var) + S.bn.eps)
    hd = head_pop if nrm in ("tp", "tpunit") else head
    tau_r = readout_tau(S, cfg)
    p = (hd(S, norm_(u[:, 0])) / tau_r).softmax(-1)
    al = S.xfit.get("alpha_p", INF)
    if cfg.get("lab") and al < 1e9:
        pi = st["pi"]
        if cfg["lab"] == "soft":  # references' labels from the same normalised classifier (independent soft labels)
            EM = (hd(S, norm_(u[:, 1:])) / tau_r).softmax(-1).sum(1)
        else:
            EM = (n * Mo["Et"] - Mo["pq"]).clamp_min(0)
        p = p * (al * pi + EM) / ((al + n - 1) * pi)
        p = p / p.sum(-1, keepdim=True)
    return p, mu, shr, dict(m_sub=msub, Sp=Sp, Ss=Ss, SsG=SsG, SsE=SsE, SsN=SsN, Serr=Serr)


def eb_normalise(S, u, Elt, Clt, Et, kap, cfg, tau, nu, Fc=None, cov=True):
    """Composition-posterior normalisation of the query (clip 0) with empirical-Bayes shrinkage of the person centre.
    The composition hypotheses differ in the centre only inside span(F): mu(N) = mean(u) - lam(N) theta_N^T F.  From the
    posterior moments Elt = E[lam theta] (B, C), Clt = Cov(lam theta) (B, C, C):
        mbar = mean(u) - Elt^T F,   Sp = f^T Clt f   (posterior mean / covariance of the centre in span(F))
    and the cross-fitted population prior of a new person's centre N(m0, kappa T) (precision weighting, Prop. 8):
        mu = mbar + Q (W - I)(Q^T mbar - m0),   W = k kappa T (k kappa T + Sp)^-1.
    W = I when the posterior identifies the composition (BatchNorm-like composition-aware statistics); W -> 0 along the
    directions in which the posterior leaves the centre confounded with the class (the source model's centre).
    Scale: person statistics of the posterior-mean composition Et.  cov=False: no shrinkage (posterior mean only).
    returns query probabilities (B, C), the centre (B, H), the shrinkage 1 - tr(W)/(C-1) (B,)"""
    st = S.st; n = u.shape[1]
    Fc = st["Fc"] if Fc is None else Fc
    Q = st["Q"]; f = Fc @ Q
    mbar = u.mean(1) - Elt @ Fc
    var = person_scale(u, Et, st, n, nu, cfg["scale"], Fc)
    if cov:
        Sp = torch.einsum("cd,bce,ef->bdf", f, Clt, f)
        kT = st["T"] * kap * cfg.get("shrink_k", 1.0)
        I = torch.eye(Q.shape[1], device=u.device)
        W = kT[None] @ torch.linalg.inv(kT[None] + Sp + 1e-6 * I)
        mq = mbar @ Q - st["m0"]
        mu = mbar + (torch.einsum("bde,be->bd", W, mq) - mq) @ Q.T
        shr = 1 - torch.diagonal(W, dim1=1, dim2=2).mean(-1)
    else:
        mu, shr = mbar, torch.zeros(u.shape[0], device=u.device)
    un = (u[:, 0] - mu) / torch.sqrt(var + S.bn.eps)
    return (head(S, un) / tau).softmax(-1), mu, shr


@torch.no_grad()
def cmn(S, x, cfg, yr=None, extras=False):
    """x: (B, n, D) episodes (index 0 = query).  returns query probabilities (B, C) and diagnostics.
    extras=True also returns, from the SAME scores, the controls that replace the marginalization by a point estimate
    (MAP composition, top-k compositions, plug-in posterior-mean composition) and the per-episode uncertainty."""
    st, B, n = S.st, x.shape[0], x.shape[1]
    C = st["pi"].numel()
    if n == 1:
        p = head_pop(S, pop_units(S, x)).softmax(-1)[:, 0]
        return p, dict(Etheta=torch.full((B, C), 1.0 / C, device=x.device), mu=torch.zeros(B, S.bn.weight.numel(), device=x.device))
    SL = sublattice(n, C)
    Hn = SL["H"]; Hh = Hn.shape[0]; th = Hn / n  # every composition hypothesis of the batch
    ev_cfg = dict(cfg, norm=cfg.get("ev_norm") or cfg.get("norm", "std"))
    tp = ev_cfg["norm"] in ("tp", "tpunit")  # normalisation of the network evidence (default: that of the read-out)
    tau = readout_tau(S, ev_cfg)
    kap = S.xfit["kappa"] if cfg["kappa"] else 1.0
    u = pop_units(S, x)
    ub, s2 = u.mean(1), u.var(1, unbiased=False)
    Fc, Q, f, pi, sig2 = st["Fc"], st["Q"], st["f"], st["pi"], st["sig2"]
    cs = cfg.get("cov_scale") or 1.0        # robustness study: misspecified density covariance
    Sin_, SG_ = st["Sin"] * cs, st["SG"] * cs
    Sig = Sin_ + SG_
    if cfg.get("proto_scale") is not None:  # robustness study: misspecified prototype magnitude
        Fc, f = Fc * cfg["proto_scale"], f * cfg["proto_scale"]
    equi = cfg["scale"] == "equi"
    mshift = th @ Fc  # (Hh, H): theta^T F
    V = batch_var(th, st, n, Fc)  # (Hh, H) expected batch variance under each hypothesis (unit expressivity)
    lam2 = hypothesis_scale(s2, V, st, equi)
    lam = lam2.sqrt()  # (B, Hh)
    VF = (Fc ** 2).mean(0)
    tv = target_var(st, Fc)
    nu = cfg["nu"]

    def evidence(sel):
        """network log posterior (B, m, n, C) of every clip under the hypotheses sel: a slice of the shared hypothesis
        list, or (B, m) per-episode hypothesis indices (pruned inference)"""
        if isinstance(sel, slice):
            lam_s, ms, V_s, th_s = lam[:, sel], mshift[sel][None], V[sel][None], th[sel][None]
        else:
            lam_s, ms, V_s, th_s = torch.gather(lam, 1, sel), mshift[sel], V[sel], th[sel]
        if equi:
            mu = ub[:, None] - lam_s[..., None] * ms
            if ev_cfg["norm"] == "tp":  # scalar expressivity
                return (head_pop(S, transport(st, u[:, None], mu[:, :, None], lam_s[..., None, None])) / tau).log_softmax(-1)
            lu = (n * s2[:, None] + nu * lam_s[..., None] ** 2 * V_s) / ((n + nu) * V_s)
            if ev_cfg["norm"] == "tpunit":  # per-unit expressivity: the person's within-person variance -> the population's
                return (head_pop(S, transport(st, u[:, None], mu[:, :, None], lu.sqrt()[:, :, None])) / tau).log_softmax(-1)
            var = lu * tv
        else:
            mu = (ub[:, None] - ms).expand(B, -1, -1)
            if tp:
                return (head_pop(S, transport(st, u[:, None], mu[:, :, None], 1.0)) / tau).log_softmax(-1)
            if cfg["scale"] == "bal":
                varF = th_s @ Fc ** 2 - ms ** 2
                var = (n * (s2[:, None] - varF).clamp_min(0) + nu * sig2) / (n + nu) + VF
            elif cfg["scale"] == "pop":
                var = (sig2 + VF).expand_as(mu)
            else:
                var = s2[:, None].expand_as(mu)
        un = (u[:, None] - mu[:, :, None]) / torch.sqrt(var[:, :, None] + S.bn.eps)
        return (head(S, un) / tau).log_softmax(-1)

    # population Gaussian-mixture density of the location-normalised clips in the expression subspace.  The person
    # scale of the density is integrated out exactly on a grid under the cross-fitted population prior (label-marginal
    # density: its scale must not depend on N, otherwise a hypothesis with more classes "explains" a tight set by a
    # smaller scale); the network side uses the moment scale lam(N) of the hypothesis (normalisation only).
    # Clips of one person and one class share the interaction G_sc: around its class centre a clip of a set with
    # composition theta varies by Sigma_in + k(theta) Sigma_G, k = (1 - |theta|^2) n / (n - 1), which reproduces the
    # expected within-set scatter exactly (k = 0 for a single-class set, k = 1 when all labels differ).
    z = u @ Q
    d = z - z.mean(1, keepdim=True)  # (B, n, C-1)
    if cfg.get("shared_G", True):
        kG = (1 - (th ** 2).sum(-1)) * n / (n - 1)  # (Hh,)
        Sig_h = Sin_[None] + kG[:, None, None] * SG_[None]
    else:
        Sig_h = Sig[None].expand(Hh, -1, -1)
    if cfg.get("dens_cov", "full") == "diag":   # replacement control: diagonal Gaussian
        Sig_h = torch.diag_embed(torch.diagonal(Sig_h, dim1=-2, dim2=-1))
    elif cfg.get("dens_cov", "full") == "iso":  # replacement control: isotropic prototype density
        Sig_h = torch.diagonal(Sig_h, dim1=-2, dim2=-1).mean(-1)[:, None, None] * torch.eye(C - 1, device=x.device)
    P = torch.linalg.inv(Sig_h)  # (Hh, C-1, C-1)
    ldS = torch.logdet(Sig_h)    # (Hh,)
    g = (th @ f)[:, None] - f[None]  # (Hh, C, C-1): hypothesis location relative to every class centre
    A = torch.einsum("bnd,hde,bne->bhn", d, P, d)
    Bt = torch.einsum("bnd,hde,hce->bhnc", d, P, g)
    Cc = torch.einsum("hcd,hde,hce->hc", g, P, g)
    if cfg.get("dens_scale", "int") == "int":
        lgrid, lw = scale_grid(S, x.device)
    else:
        lgrid, lw = torch.ones(1, device=x.device), torch.zeros(1, device=x.device)
    dens = []
    for lk, wk in zip(lgrid, lw):
        qd_k = A[..., None] / lk ** 2 + 2 * Bt / lk + Cc[None, :, None]
        dens.append(wk + (pi.log() - 0.5 * qd_k).logsumexp(-1).sum(-1) - (n - 1) * (C - 1) * lk.log())
    dens = torch.stack(dens).logsumexp(0) - 0.5 * (n - 1) * ldS[None]  # (B, Hh)
    qd = A[..., None] + 2 * Bt + Cc[None, :, None]  # unit scale (generative-only ablation)
    NRc = Hn[:, None, :] - torch.eye(C, device=x.device)[None]  # reference counts N_h - e_c
    Tprior = log_dmseq(NRc.clamp_min(0), pi, cfg["prior"])[None].expand(B, -1, -1)  # composition prior (B, Hh, C)
    # calibrated composition posterior (round 9): the network-evidence, density and anchor families enter with weights
    # (t_net, t_dens, t_anc) fitted by conditional maximum likelihood of the TRUE compositions of held-out people
    t_net, t_dens, t_anc = post_weights(S, cfg)
    dens_t = dens if (cfg["dens"] and cfg["lik"] in ("hyb", "bnmode")) else torch.zeros_like(dens)
    T = Tprior + t_dens * dens_t[..., None]
    anc = torch.zeros(B, Hh, device=x.device)
    lgm = head_pop(S, pop_units(S, x.mean(1)))  # population-mode network posterior of the batch-mean clip
    if cfg["anchor"] == "offset":  # prior on the implied subject offset (expression subspace)
        m = (ub @ Q)[:, None] - lam[..., None] * (th @ f)[None] - st["m0"]  # (B, Hh, C-1)
        Sw = Sig / n
        ks = kap * (torch.logspace(-1, 1, 9, device=x.device) if cfg["student"] else torch.ones(1, device=x.device))
        Vs = st["T"][None] * ks[:, None, None] + Sw[None]
        qo = torch.einsum("bhd,kde,bhe->bhk", m, torch.linalg.inv(Vs), m)
        T = T + ((-0.5 * qo - 0.5 * torch.logdet(Vs)).logsumexp(-1) - math.log(len(ks)))[..., None]
    elif cfg["anchor"] in ("net", "idf"):  # counted once, log-linear in theta: single class -> log p(mean clip|c)+const
        anc = (lgm.log_softmax(-1) - pi.log()) @ th.T  # (B, Hh)
        if cfg["anchor"] == "idf":  # identifiability weight g(theta): share of the interaction the set mean keeps
            anc = anc * idf_weight(th, C)[None]
        T = T + t_anc * anc[..., None]
    # ---- network evidence of every clip under every composition hypothesis
    prune = cfg.get("prune")
    if cfg["lik"] == "bnmode":  # one common (batch-mode) normalisation for every hypothesis: no self-fulfilling evidence
        logp = (head(S, (u - ub[:, None]) / torch.sqrt(s2[:, None] + S.bn.eps)) / tau).log_softmax(-1)[:, None].expand(B, Hh, n, C)
    elif prune and prune < Hh:
        # two-stage (pruned) inference: stage 1 ranks every composition with the cheap batch-mode evidence (one network
        # pass; the count DP and the composition terms are cheap), stage 2 computes the hypothesis-specific evidence only
        # for the M most probable compositions of each episode; the others get probability 0.  M = Hh is exact.
        lp0 = (head(S, (u - ub[:, None]) / torch.sqrt(s2[:, None] + S.bn.eps)) / tau).log_softmax(-1)  # (B, n, C)
        L0 = count_dp(SL, (lp0 - pi.log())[:, None].expand(B, Hh, n, C)) + lp0[:, None, 0] + T
        sel = L0.logsumexp(2).topk(int(prune), 1).indices  # (B, M)
        logp = torch.full((B, Hh, n, C), -INF, device=x.device)
        for j0 in range(0, sel.shape[1], 256):
            sj = sel[:, j0:j0 + 256]
            logp.scatter_(1, sj[..., None, None].expand(-1, -1, n, C), evidence(sj))
    else:
        logp = torch.empty(B, Hh, n, C, device=x.device)
        for i0 in range(0, Hh, 256):
            logp[:, i0:i0 + 256] = evidence(slice(i0, i0 + 256))
    if cfg["lik"] in ("hyb", "bnmode"):
        unary = logp - pi.log()  # p_net(y|.) / pi_y
        qterm = logp[:, :, 0]
    else:  # generative-only ablation: Gaussian class-conditionals in the subspace, no network evidence
        unary = -0.5 * qd
        qterm = pi.log() - 0.5 * qd[:, :, 0]
    if cfg["oracle"] and yr is not None:  # the references' true labels (same computation otherwise)
        oh = F.one_hot(yr, C).bool()[:, None]  # (B, 1, K, C)
        # Conditioning on labels restricts support; it must retain their observed feature likelihood.
        unary = unary.clone(); unary[:, :, 1:] = unary[:, :, 1:].masked_fill(~oh, -INF)
    if cfg.get("return_terms"):  # posterior calibration: the separate evidence families (held-out people only)
        return None, dict(unary=unary, qterm=qterm, Tprior=Tprior, dens=dens_t, anc=anc, SL=SL)
    coef = count_dp(SL, t_net * unary)  # (B, Hh, C): exact sum over reference assignments with counts N_h - e_c
    L = coef + t_net * qterm + T
    lN = L.logsumexp(2)
    pN = lN.softmax(1)  # posterior over the batch composition
    pq = L.flatten(1).softmax(1).view(B, Hh, C).sum(1)  # exact posterior marginal of the query label
    if cfg["pred"] == "joint":
        p = pq
    elif cfg["pred"] == "mix":
        p = (pN[..., None] * logp[:, :, 0].exp()).sum(1)
    elif cfg["pred"].startswith("top"):  # point-estimate controls on the same scores
        p = topk_predict(L, lN, int(cfg["pred"][3:]))
    elif cfg["pred"] == "plugin":
        p = (head(S, person_norm(u, pN @ th, st, n, nu, cfg["scale"], S.bn.eps, Fc)[:, 0]) / readout_tau(S, dict(cfg, norm="std"))).softmax(-1)
    elif cfg["pred"] not in ("shrink", "stack"):
        raise ValueError(cfg["pred"])
    Et = pN @ th
    lt = lam[..., None] * th[None]  # (B, Hh, C): lam(N) theta_N, the composition part of the centre hypotheses
    Elt = torch.einsum("bh,bhc->bc", pN, lt)
    D = dict(Etheta=Et, mu=ub - Elt @ Fc, lam=(pN * lam).sum(1), H_N=-(pN * pN.clamp_min(1e-30).log()).sum(1))
    if cfg["pred"] in ("shrink", "stack") or extras:
        Clt = torch.einsum("bh,bhc,bhd->bcd", pN, lt, lt) - Elt[:, :, None] * Elt[:, None, :]
        gth = idf_weight(th, C)[None]
        lgb = head(S, (u - ub[:, None]) / torch.sqrt(s2[:, None] + S.bn.eps))  # batch-moment (BN-mode) logits
        qb = lgb.softmax(-1)  # BN soft labels
        Cpl = (torch.diag_embed(qb) - qb[..., :, None] * qb[..., None, :]).sum(1) / n ** 2
        pl_var = torch.einsum("cd,bce,ed->b", f, Cpl, f)  # independent pseudo-label variance of the centre
        Mo = dict(Elt=Elt, Clt=Clt, El=(pN * lam).sum(1), El2=(pN * lam2).sum(1), El2g=(pN * lam2 * gth).sum(1),
                  Et=Et, pq=pq, pl_var=pl_var)
        p_sh, mu_sh, shr, dg = cpn_readout(S, u, Mo, cfg, kap, Fc)  # the conservative branch
        D.update(shrink=shr, m_sub=dg["m_sub"], Sp=dg["Sp"], Ss=dg["Ss"], Serr=dg["Serr"])
        # ---- the four predictive distributions (round 10): exact Bayes given each composition, the conservative
        # posterior-moment shrinkage, the batch's own moments (the composition-agnostic estimator) and the source
        xf = S.xfit if hasattr(S, "xfit") else {}
        pc = torch.nan_to_num(L.softmax(-1), nan=0.0)                 # p(y_q | N_h, U): exact Bayes given N_h
        p_bn = (lgb[:, 0] / xf.get("tau_bn", 1.0)).softmax(-1)
        p_src = (head_pop(S, u[:, 0]) / xf.get("tau_src", 1.0)).softmax(-1)
        H_N = -(pN * pN.clamp_min(1e-30).log()).sum(1)
        SI = stack_inputs(pN, pc, th, C, p_sh, p_bn, p_src, pl_var, H_N)
        D["Eg"] = SI["Eg"]
        if cfg.get("return_stack"):
            D["stack_in"] = SI
        par = xf.get("stack_params", {})
        p_st = stack_combine(SI, "stack", par["stack"]) if par.get("stack") else stack_combine(SI, "wg", None)
        if cfg["pred"] == "shrink":
            p = p_sh; D["mu"] = mu_sh
        elif cfg["pred"] == "stack":
            p = p_st; D["mu"] = mu_sh
    if extras:
        D["logpN"] = pN.clamp_min(1e-30).log()
        D["anchor_p"] = lgm.softmax(-1)
        D["ent_src"] = ent(head_pop(S, u[:, 0]))  # simple failure diagnostics (uncertainty table)
        D["maxp"] = p.max(-1).values
        D["pl_var"] = Mo["pl_var"]
        alt = {"top1": topk_predict(L, lN, 1)}
        # the four branches alone and the round-9 fixed weighting (w = g on the conservative branch)
        alt["cond"] = SI["Sv"].sum(1)        # = the exact joint posterior marginal of the query label
        alt["eb"] = p_sh
        alt["bn"] = p_bn                     # = BN-adapt (same argmax; cross-fitted temperature)
        alt["src"] = p_src
        alt["wg"] = stack_combine(SI, "wg", None)
        # other combiners of the SAME four branches, fitted on the SAME held-out people with the SAME objective
        for kind in STACK_KINDS[1:]:
            if par.get(kind):
                alt[kind] = stack_combine(SI, kind, par[kind])
        # one component of the conservative branch changed inside the full stacked predictor
        for nm, over in READOUT_ALTS.items():
            pe_ = cpn_readout(S, u, Mo, dict(cfg, **over), kap, Fc)[0]
            alt["st:" + nm] = stack_combine(dict(SI, p_eb=pe_), "stack", par["stack"]) if par.get("stack") else pe_
        # posterior -> its MAP composition (a point estimate), identical branches and combiner
        hs = pN.argmax(1)
        lam_m = lam.gather(1, hs[:, None])[:, 0]
        th_m = th[hs]
        Mo_map = dict(Elt=lam_m[:, None] * th_m, Clt=torch.zeros_like(Clt), El=lam_m, El2=lam_m ** 2,
                      El2g=lam_m ** 2 * idf_weight(th_m, C), Et=th_m,
                      pq=L[torch.arange(B, device=x.device), hs].softmax(-1), pl_var=Mo["pl_var"])
        pe_map = cpn_readout(S, u, Mo_map, cfg, kap, Fc)[0]
        pN_map = F.one_hot(hs, Hh).float()
        SI_map = stack_inputs(pN_map, pc, th, C, pe_map, p_bn, p_src, pl_var, torch.zeros_like(H_N))
        alt["stack_map"] = stack_combine(SI_map, "stack", par["stack"]) if par.get("stack") else stack_combine(SI_map, "wg", None)
        alt["stack"] = p_st
        D["alt"] = alt
    return p, D


TOPK = [1, 4, 16, 64, 256]


def topk_predict(L, lN, k):
    """keep only the k most probable batch compositions (k = 1: MAP composition), same scores, renormalised"""
    B, Hh, C = L.shape
    idx = lN.topk(min(k, Hh), 1).indices
    Lk = torch.gather(L, 1, idx[..., None].expand(-1, -1, C))
    return Lk.flatten(1).softmax(1).view(B, -1, C).sum(1)


def cmn_cfg(method):
    cfg = dict(CMN_DEFAULT); cfg.update(CMN_VARIANTS[method]); return cfg


# ============================================================================ baselines
def fwd(S, P, x, mode):
    """functional forward with per-episode parameters P; mode: 'eval' running stats | 'batch' episode stats"""
    h = torch.einsum("bnd,bdh->bnh", x, P["W1"]) + P["b1"][:, None] if P["W1"].dim() == 3 else x @ P["W1"] + P["b1"]
    if mode == "eval":
        mu, var = P.get("rm", S.bn.running_mean), P.get("rv", S.bn.running_var)
        mu, var = (mu[:, None], var[:, None]) if mu.dim() == 2 else (mu, var)
    else:
        mu, var = h.mean(1, keepdim=True), h.var(1, unbiased=False, keepdim=True)
    f = F.gelu((h - mu) / torch.sqrt(var + S.bn.eps) * P["g"][:, None] + P["be"][:, None])
    return f @ S.cls.weight.T + S.cls.bias, f, (h.mean(1), h.var(1, unbiased=True) if h.shape[1] > 1 else h.new_ones(h.shape[0], h.shape[2]))


def ent(lg):
    return -(lg.softmax(-1) * lg.log_softmax(-1)).sum(-1)


def episode_params(S, B, full=False):
    P = {"g": S.bn.weight.detach().expand(B, -1).clone(), "be": S.bn.bias.detach().expand(B, -1).clone()}
    if full:
        P["W1"] = S.f1.weight.detach().T.expand(B, -1, -1).clone(); P["b1"] = S.f1.bias.detach().expand(B, -1).clone()
        P["rm"] = S.bn.running_mean.expand(B, -1).clone(); P["rv"] = S.bn.running_var.expand(B, -1).clone()
    else:
        P["W1"], P["b1"] = S.f1.weight.detach().T, S.f1.bias.detach()
    return P


STEPS, FILTER_K, BN_PRIOR_N, CSN_R = 10, 20, 16, 0.5
# grids of the simple baselines: one run saves every grid value as an alternative prediction; the operating point is
# selected on held-out TRAINING people (cross-fit split, auxiliary network) by the aggregator, and the whole test-set
# accuracy-robustness frontier is reported as well
GRIDS = {"srcbn_grid": [i / 10 for i in range(11)],                     # weight of BN in the Source/BN mixture
         "bnprior_grid": [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1e9],    # pseudo-samples of the source statistics
         "csn_grid": [i / 10 for i in range(11)],                       # fraction of source-kept channels
         "tent_grid": [(lr, k) for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2) for k in (1, 3, 10)],
         "laem_grid": [(lr, k) for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2) for k in (1, 3, 10)],
         "sar_grid": [(m, k) for m in (0.25, 1.0, 4.0, 16.0, 64.0) for k in (1, 3, 10)],
         "rotta_grid": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]}                # RBN momentum (official 0.05)
GRID_DEFAULT = {"srcbn_grid": 0.5, "bnprior_grid": 16, "csn_grid": 0.5, "tent_grid": (1e-3, 10), "sar_grid": (1.0, 10),
                "laem_grid": (1e-3, 10),
                "rotta_grid": 0.05}


def grid_label(method, v):
    return f"{method[:-5]}={v[0]:g},{v[1]}" if isinstance(v, tuple) else f"{method[:-5]}={v:g}"


def p_bn(S, x):
    return fwd(S, episode_params(S, x.shape[0]), x, "batch")[0][:, 0].softmax(-1)


def p_source(S, x):
    return fwd(S, episode_params(S, x.shape[0]), x, "eval")[0][:, 0].softmax(-1)


def p_bnprior(S, x, Np):
    """Schneider et al. 2020: source statistics as a prior worth Np pseudo-samples"""
    n = x.shape[1]
    h = x @ S.f1.weight.T + S.f1.bias
    w = n / (n + Np)
    mu = w * h.mean(1, keepdim=True) + (1 - w) * S.bn.running_mean
    var = w * h.var(1, unbiased=False, keepdim=True) + (1 - w) * S.bn.running_var
    return head_pop(S, (h - mu) / torch.sqrt(var + S.bn.eps))[:, 0].softmax(-1)


def p_csn(S, x, R, C):
    """Vianna et al. (CoLLAs'24): keep the source statistics on the label-sensitive channels (fraction R)"""
    B = x.shape[0]
    h = x @ S.f1.weight.T + S.f1.bias
    mt, sdt = h.mean(1), torch.sqrt(h.var(1, unbiased=False) + S.bn.eps)
    ms, sds = S.bn.running_mean, torch.sqrt(S.bn.running_var + S.bn.eps)
    prior = torch.full((B, C), 1.0 / C, device=x.device)
    k = int(round(R * h.shape[-1]))
    for _ in range(2):  # uniform class prior, then the prior from the batch's pseudo-labels
        score = prior @ S.csn_score  # (B, H)
        keep = torch.ones_like(score)
        if k > 0:
            keep = keep.scatter_(1, score.topk(k, 1).indices, 0.0)  # 1 = adapt
        mu, sd = keep * mt + (1 - keep) * ms, keep * sdt + (1 - keep) * sds
        lg = head_pop(S, (h - mu[:, None]) / sd[:, None])
        prior = lg.softmax(-1).mean(1)
    return lg[:, 0].softmax(-1)


def _balanced_stats(h, lab, C, conf=None):
    """class-balanced batch statistics: every predicted class gets the same total weight (optionally confidence-
    weighted inside a class), the weights of a batch sum to 1.  h (B, n, H), lab (B, n)"""
    oh = F.one_hot(lab, C).float()  # (B, n, C)
    w = oh if conf is None else oh * conf[..., None]
    w = w / w.sum(1, keepdim=True).clamp_min(1e-8)  # within-class weights
    present = (oh.sum(1) > 0).float()  # (B, C)
    w = (w * (present / present.sum(-1, keepdim=True))[:, None]).sum(-1)  # (B, n)
    mu = (w[..., None] * h).sum(1, keepdim=True)
    var = (w[..., None] * (h - mu) ** 2).sum(1, keepdim=True)
    return mu, var


def p_note(S, x, C):
    """NOTE (Gong et al., NeurIPS'22) on a vector bottleneck: normalisation statistics of a prediction-balanced memory
    (PBRS keeps the memory class-balanced by the model's own predictions; here the episode's clips weighted so that every
    BN-predicted class has equal weight).  IABN's instance statistics need spatial positions; on a feature vector they
    reduce to the batch statistics."""
    h = x @ S.f1.weight.T + S.f1.bias
    lab = fwd(S, episode_params(S, x.shape[0]), x, "batch")[0].argmax(-1)
    mu, var = _balanced_stats(h, lab, C)
    return head_pop(S, (h - mu) / torch.sqrt(var + S.bn.eps))[:, 0].softmax(-1)


def p_rotta(S, x, C, alpha):
    """RoTTA (Yuan et al., CVPR'23) robust BN on one episode: statistics = (1 - alpha) source + alpha * statistics of a
    category-balanced memory (CSTU: class-balanced by the teacher's -- source model's -- predictions, confident clips
    weighted more).  The teacher-student and timeliness terms of the stream version have no role in one episode;
    alpha (official 0.05 for long streams) is a grid tuned on held-out training people."""
    h = x @ S.f1.weight.T + S.f1.bias
    lg = fwd(S, episode_params(S, x.shape[0]), x, "eval")[0]
    conf = torch.exp(-ent(lg) / math.log(C))
    mu_m, var_m = _balanced_stats(h, lg.argmax(-1), C, conf)
    ms, vs = S.bn.running_mean, S.bn.running_var
    mu = (1 - alpha) * ms + alpha * mu_m
    var = (1 - alpha) * vs + alpha * var_m + alpha * (1 - alpha) * (mu_m - ms) ** 2
    return head_pop(S, (h - mu) / torch.sqrt(var + S.bn.eps))[:, 0].softmax(-1)


def pl_shrink(S, x):
    """simple-composition control of CPN (same slot, same information, same read-out): the composition and its
    uncertainty come from the soft pseudo-labels q_i of the BN-adapted network treated as INDEPENDENT categorical labels,
        E[theta] = mean_i q_i,  Cov(theta) = sum_i (diag q_i - q_i q_i^T) / n^2,
    followed by the identical round-8 read-out (transport, calibrated centre-error covariance, EB shrinkage, person
    label prior).  Only the composition posterior is replaced."""
    n = x.shape[1]; st = S.st; C = st["pi"].numel()
    u = pop_units(S, x)
    q = fwd(S, episode_params(S, x.shape[0]), x, "batch")[0].softmax(-1)
    th = q.mean(1)
    _, lam = person_centre(u, th, st, n)  # (B, 1)
    Ct = (torch.diag_embed(q) - q[..., :, None] * q[..., None, :]).sum(1) / n ** 2
    l2 = lam[:, 0] ** 2
    Mo = dict(Elt=lam * th, Clt=l2[:, None, None] * Ct, El=lam[:, 0], El2=l2, El2g=l2 * idf_weight(th, C), Et=th,
              pq=q[:, 0])
    return cpn_readout(S, u, Mo, cmn_cfg("cpn"), S.xfit.get("kappa", 1.0))[0]


def src_lab(S, x):
    """same-information control of the person-level label prior: the SOURCE model's calibrated posterior of the query
    times the Dirichlet-multinomial predictive of the references' Source soft labels, with the identical EB
    concentration alpha_p (EM-style label-shift correction, Saerens et al. 2002, per person, with a Dirichlet prior)"""
    st = S.st; n = x.shape[1]; pi = st["pi"]
    p = (head_pop(S, pop_units(S, x)) / S.xfit.get("tau_src", 1.0)).softmax(-1)
    al = S.xfit.get("alpha_p", INF)
    if not al < 1e9:
        return p[:, 0]
    q = p[:, 0] * (al * pi + p[:, 1:].sum(1)) / ((al + n - 1) * pi)
    return q / q.sum(-1, keepdim=True)


def _upd(P, g0, b0):
    """relative size of the affine update of one episode"""
    num = ((P["g"] - g0) ** 2).sum(-1) + ((P["be"] - b0) ** 2).sum(-1)
    return (num / ((g0 ** 2).sum(-1) + (b0 ** 2).sum(-1))).sqrt()


def run_laem(S, x, lr=1e-3, steps=STEPS, record=None):
    """LAEM feature-head episodic adaptation of the authors' TENT_LA_APX implementation:
    entropy of z + tau log pi_t; square-root hard-prediction counts and EMA prior.
    Official conf.py defaults: tau=1, epsilon=1e-12, alpha=0.1, warmup=0.
    Return adjusted pre-update logits as in forward_and_adapt; reset each episode.
    https://github.com/thilinawee/latta/blob/main/classification/methods/experiments/tent_la_apx.py"""
    B = x.shape[0]
    P = episode_params(S, B)
    prm = [P["g"].requires_grad_(), P["be"].requires_grad_()]
    opt = torch.optim.Adam(prm, lr=lr, betas=(0.9, 0.999))
    outs, prior = {}, None
    for t in range(1, steps + 1):
        with torch.enable_grad():
            lg = fwd(S, P, x, "batch")[0]
            with torch.no_grad():
                counts = F.one_hot(lg.argmax(-1), lg.shape[-1]).sum(1).float().clamp_min(1e-12)
                batch_prior = counts.sqrt()
                batch_prior = batch_prior / batch_prior.sum(-1, keepdim=True)
                prior = batch_prior if prior is None else 0.1 * prior + 0.9 * batch_prior
                prior = prior.clamp_min(1e-12)
                prior = prior / prior.sum(-1, keepdim=True)
            adjusted = lg + prior[:, None].log()
            loss = ent(adjusted).mean(1).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if record is not None and t in record:
            with torch.no_grad():
                outs[t] = adjusted[:, 0].detach().softmax(-1)
    with torch.no_grad():
        return adjusted[:, 0].detach().softmax(-1), {}, outs


def run_tent(S, x, lr=1e-3, steps=STEPS, record=None):
    """official TENT: Adam, BN in batch mode, affine parameters only, entropy of the batch; episode-wise reset.
    record (list of step counts) returns the prediction after each listed step (grid of the step budget)."""
    B = x.shape[0]
    P = episode_params(S, B); g0, b0 = P["g"].clone(), P["be"].clone()
    prm = [P["g"].requires_grad_(), P["be"].requires_grad_()]
    opt = torch.optim.Adam(prm, lr=lr, betas=(0.9, 0.999))
    with torch.no_grad():
        e0 = ent(fwd(S, P, x, "batch")[0]).mean(1)
    outs = {}
    for t in range(1, steps + 1):
        with torch.enable_grad():
            loss = ent(fwd(S, P, x, "batch")[0]).mean(1).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        if record is not None and t in record:
            with torch.no_grad():
                outs[t] = fwd(S, P, x, "batch")[0][:, 0].softmax(-1)
    with torch.no_grad():
        lg = fwd(S, P, x, "batch")[0]
        diag = dict(upd_rel=_upd(P, g0, b0), ent_drop=e0 - ent(lg).mean(1))
        return lg[:, 0].softmax(-1), diag, outs


def run_sar(S, x, C, lr_mult=1.0, steps=STEPS, record=None):
    """official SAR: SAM(SGD, momentum 0.9, rho 0.05), lr (0.00025/64)*bs*2 for bs < 32, reliable-entropy filter
    E0 = 0.4 ln C, episode-wise reset.  Diagnostics: relative update size, fraction of samples passing the filter,
    number of steps with any update."""
    B, n = x.shape[0], x.shape[1]
    P = episode_params(S, B); g0, b0 = P["g"].clone(), P["be"].clone()
    prm = [P["g"].requires_grad_(), P["be"].requires_grad_()]
    lr, rho, e0 = lr_mult * 0.00025 / 64 * n * 2, 0.05, 0.4 * math.log(C)
    opt = torch.optim.SGD(prm, lr=lr, momentum=0.9)
    frac, active, outs = 0.0, 0.0, {}
    for t in range(1, steps + 1):
        with torch.enable_grad():
            e = ent(fwd(S, P, x, "batch")[0]); w1 = (e < e0).float()
            frac = frac + w1.mean(1); active = active + (w1.sum(1) > 0).float()
            ((e * w1).sum(1) / w1.sum(1).clamp_min(1)).sum().backward()
            gn = torch.sqrt(sum((p.grad ** 2).sum(-1) for p in prm)) + 1e-12
            eps = [rho * p.grad / gn[:, None] for p in prm]
            with torch.no_grad():
                for p, ep in zip(prm, eps): p.add_(ep)
            opt.zero_grad()
            e2 = ent(fwd(S, P, x, "batch")[0]); w2 = w1 * (e2 < e0).float()
            ((e2 * w2).sum(1) / w2.sum(1).clamp_min(1)).sum().backward()
            with torch.no_grad():
                for p, ep in zip(prm, eps): p.sub_(ep)
            opt.step(); opt.zero_grad()
        if record is not None and t in record:
            with torch.no_grad():
                outs[t] = fwd(S, P, x, "batch")[0][:, 0].softmax(-1)
    with torch.no_grad():
        diag = dict(upd_rel=_upd(P, g0, b0), eff_frac=frac / steps, eff_steps=active)
        return fwd(S, P, x, "batch")[0][:, 0].softmax(-1), diag, outs


def pl_predict(S, x, iters):
    """composition point estimate from soft pseudo-labels, then ONE normalisation of the same family as CMN
    (scale-equivariant person statistics): iters=1 uses the BN-adapted soft labels, iters=10 iterates to a fixed point.
    The 'simple composition estimate' control of the marginalisation."""
    n = x.shape[1]
    tau = S.xfit.get("tau", 1.0) if hasattr(S, "xfit") else 1.0
    u = pop_units(S, x)
    p = fwd(S, episode_params(S, x.shape[0]), x, "batch")[0].softmax(-1)
    for _ in range(iters):
        lg = head(S, person_norm(u, p.mean(1), S.st, n, 1.0, "equi", S.bn.eps)) / tau
        p = lg.softmax(-1)
    return p[:, 0]


class DARTNet(nn.Module):
    """DART-style prediction refinement (Jo et al., 2024): maps the batch's average BN-adapted prediction to an affine
    correction of the logits (identity at initialisation); trained on labelled training episodes whose label
    distributions are Dirichlet-shifted"""

    def __init__(self, C, h=128):
        super().__init__()
        self.C = C
        self.net = nn.Sequential(nn.Linear(C, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, C * C + C))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, lg):
        o = self.net(lg.softmax(-1).mean(1)); C = self.C
        W = torch.eye(C, device=lg.device) + o[:, : C * C].view(-1, C, C)
        return torch.einsum("bnc,bdc->bnd", lg, W) + o[:, None, C * C:]


def train_dart(S, X, meta, rows, a, rng, C, steps=2000):
    y = torch.tensor(meta.label.values, device=dev)
    g = DARTNet(C).to(dev)
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    smp = EpisodeSampler(meta, rows, C, int(rng.integers(1 << 30)))
    for it in range(steps):
        n = int(rng.integers(2, a.kmax + 2)); ep = smp(max(1, a.bs * 5 // n), n)
        with torch.no_grad():
            lg = fwd(S, episode_params(S, ep.shape[0]), X[ep], "batch")[0]
        loss = F.cross_entropy(g(lg).flatten(0, 1), y[ep].flatten())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return g.eval().requires_grad_(False)


NEEDS_XFIT = set(CMN_VARIANTS) | {"pl_bn", "pl_iter", "pl_shrink", "src_lab", "neucal"} | set(GRIDS)


def tta(S, x, method, C, yr=None, extras=False):
    """x: (B, n, D) episodes; returns query probabilities (B, C) and diagnostics (alternative predictions 'alt',
    per-episode diagnostics)"""
    B, n, _ = x.shape
    if method in CMN_VARIANTS:
        return cmn(S, x, cmn_cfg(method), yr, extras)
    if n == 1:  # a single test clip gives no batch to adapt to: every batch method returns the source model
        with torch.no_grad():
            p = p_source(S, x)
        return p, (dict(alt={grid_label(method, v): p for v in GRIDS[method]}) if method in GRIDS else {})
    if method in ("tent", "sar", "tent_grid", "sar_grid", "laem", "laem_grid"):
        if method in ("tent", "sar", "laem"):
            p, d, _ = run_tent(S, x) if method == "tent" else (run_laem(S, x) if method == "laem" else run_sar(S, x, C))
            return p, d
        alt = {}
        for lr in sorted({v[0] for v in GRIDS[method]}):
            ks = sorted(v[1] for v in GRIDS[method] if v[0] == lr)
            if method == "sar_grid":
                _, _, outs = run_sar(S, x, C, lr, max(ks), set(ks))
            else:
                _, _, outs = (run_tent if method == "tent_grid" else run_laem)(S, x, lr, max(ks), set(ks))
            for k in ks:
                alt[grid_label(method, (lr, k))] = outs[k]
        return alt[grid_label(method, GRID_DEFAULT[method])], dict(alt=alt)
    with torch.no_grad():
        if method in ("source", "bn", "srcbn", "srcbn_grid"):
            p_src = p_source(S, x)
            if method == "source":
                return p_src, {}
            pb = p_bn(S, x)
            u = pop_units(S, x)
            d = dict(mu=u.mean(1), Etheta=torch.full((B, C), 1.0 / C, device=x.device))
            if method == "bn":
                return pb, d
            if method == "srcbn":
                return 0.5 * (p_src + pb), d
            d["alt"] = {grid_label(method, w): w * pb + (1 - w) * p_src for w in GRIDS[method]}
            return 0.5 * (p_src + pb), d
        if method == "bnprior":
            return p_bnprior(S, x, BN_PRIOR_N), {}
        if method == "bnprior_grid":
            alt = {grid_label(method, v): p_bnprior(S, x, v) for v in GRIDS[method]}
            return alt[grid_label(method, BN_PRIOR_N)], dict(alt=alt)
        if method == "csn":
            return p_csn(S, x, CSN_R, C), {}
        if method == "csn_grid":
            alt = {grid_label(method, v): p_csn(S, x, v, C) for v in GRIDS[method]}
            return alt[grid_label(method, CSN_R)], dict(alt=alt)
        if method == "note":
            return p_note(S, x, C), {}
        if method == "rotta":
            return p_rotta(S, x, C, GRID_DEFAULT["rotta_grid"]), {}
        if method == "rotta_grid":
            alt = {grid_label(method, v): p_rotta(S, x, C, v) for v in GRIDS[method]}
            return alt[grid_label(method, GRID_DEFAULT[method])], dict(alt=alt)
        if method == "src_lab":
            return src_lab(S, x), {}
        if method == "pl_shrink":
            return pl_shrink(S, x), {}
        if method == "pl_bn":
            return pl_predict(S, x, 1), {}
        if method == "pl_iter":
            return pl_predict(S, x, 10), {}
        if method == "dart":
            return S.dart(fwd(S, episode_params(S, B), x, "batch")[0])[:, 0].softmax(-1), {}
        if method == "neucal":  # SFDA-PFT setting: the references are KNOWN to be neutral (personalised translation
            # with the person's neutral clips): centre and per-unit expressivity of the neutral references, the same
            # source-anchored transport as CPN -> the information-matched upper reference for CPN on 'neu6'
            u = pop_units(S, x)
            st = S.st; ur = u[:, 1:]; k = ur.shape[1]
            th = F.one_hot(torch.full((B,), S.neutral, device=x.device), C).float()
            mu, _ = person_centre(ur, th, st, k)
            sc = torch.sqrt((person_scale(ur, th, st, k, 1.0, "equi") + S.bn.eps) / target_var(st)) if k > 1 else 1.0
            lg = head_pop(S, transport(st, u[:, 0], mu, sc)) / S.xfit.get("tau_tp", 1.0) if hasattr(S, "xfit") else \
                head_pop(S, transport(st, u[:, 0], mu, sc))
            return lg.softmax(-1), {}
    if method == "delta":  # official (entw+delta): TBR (prior 0.95), entropy weighting, DOT 0.95, SGD 2.5e-4
        P = episode_params(S, B); prm = [P["g"].requires_grad_(), P["be"].requires_grad_()]
        opt = torch.optim.SGD(prm, lr=0.00025, momentum=0.9)
        qhat = torch.full((B, C), 1.0 / C, device=x.device)
        for _ in range(STEPS):
            with torch.enable_grad():
                lg = fwd(S, P, x, "batch")[0]
                p = lg.softmax(-1); e = ent(lg); pls = p.argmax(-1)
                ew = torch.exp(math.log(C) * 0.5 - e.detach()); use = (ew > 1).float()
                cw = 1 / qhat; cw = cw / cw.sum(-1, keepdim=True)
                sw = torch.gather(cw, 1, pls); sw = sw / sw.sum(1, keepdim=True) * n
                ((e * ew * sw * use).sum(1) / use.sum(1).clamp_min(1)).sum().backward()
                opt.step(); opt.zero_grad()
                qhat = 0.95 * qhat + 0.05 * p.detach().mean(1)
        with torch.no_grad():
            return fwd(S, P, x, "batch")[0][:, 0].softmax(-1), {}
    if method == "shot":  # official: SGD(0.01, momentum 0.9, nesterov, wd 1e-3) on the bottleneck, classifier frozen,
        P = episode_params(S, B, full=True)  # loss = ent - div + 0.3 CE(cluster pseudo-labels), 15 iterations
        prm = [P[k].requires_grad_() for k in ["W1", "b1", "g", "be"]]
        opt = torch.optim.SGD(prm, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-3)
        for _ in range(15):
            with torch.no_grad():
                lg0, f0, _ = fwd(S, P, x, "eval")
                fe = torch.cat([f0, torch.ones_like(f0[..., :1])], -1); fe = fe / fe.norm(dim=-1, keepdim=True)
                aff = lg0.softmax(-1); pred = aff.argmax(-1)
                for _ in range(2):
                    cen = aff.transpose(1, 2) @ fe / (1e-8 + aff.sum(1)[..., None])
                    cen = cen / cen.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    present = F.one_hot(pred, C).sum(1) > 0
                    pred = (1 - fe @ cen.transpose(1, 2)).masked_fill(~present[:, None], 1e9).argmin(-1)
                    aff = F.one_hot(pred, C).float()
            with torch.enable_grad():
                lg, _, (bm, bv) = fwd(S, P, x, "batch")
                p = lg.softmax(-1)
                loss = 0.3 * F.cross_entropy(lg.flatten(0, 1), pred.flatten(), reduction="none").view(B, n).mean(1)
                msm = p.mean(1)
                loss = loss + ent(lg).mean(1) + (msm * torch.log(msm + 1e-5)).sum(-1)
                opt.zero_grad(); loss.sum().backward(); opt.step()
            with torch.no_grad():
                P["rm"].mul_(0.9).add_(0.1 * bm.detach()); P["rv"].mul_(0.9).add_(0.1 * bv.detach())
        with torch.no_grad():
            return fwd(S, P, x, "eval")[0][:, 0].softmax(-1), {}
    with torch.no_grad():
        if method == "compass_em":  # our earlier method: anchored explaining-away, 10-iteration point estimate
            lg0 = fwd(S, episode_params(S, B), x, "eval")[0]
            u = pop_units(S, x)
            Fc, sig2, tau2 = S.st["Fc"], S.st["sig2"], S.st["b"].var(0).clamp_min(1e-4)
            a = lg0.softmax(-1)
            for _ in range(10):
                e = u - a @ Fc
                varF = a @ Fc ** 2 - (a @ Fc) ** 2
                w = 1 / (sig2 + varF)
                mu = (w * e).sum(1, keepdim=True) / (w.sum(1, keepdim=True) + 1 / tau2)
                wn = w / w.sum(1, keepdim=True)
                s2 = (wn * ((e - mu) ** 2 - varF)).sum(1, keepdim=True).clamp_min(1e-3) + (Fc ** 2).mean(0)
                s2 = (s2 * (n - 1) + sig2 + (Fc ** 2).mean(0)) / n
                lg = head(S, (u - mu) / torch.sqrt(s2))
                a = lg.softmax(-1)
            return lg[:, 0].softmax(-1), dict(mu=mu[:, 0])
        if method == "t3a":  # official: supports initialised with classifier weights, entropy filter filter_K
            Wc = S.cls.weight; wl = Wc @ Wc.T + S.cls.bias
            _, f, _ = fwd(S, episode_params(S, B), x, "eval")
            lg = f @ Wc.T + S.cls.bias
            sup = torch.cat([Wc.expand(B, -1, -1), f], 1)
            lab = torch.cat([wl.argmax(-1).expand(B, -1), lg.argmax(-1)], 1)
            e = torch.cat([ent(wl).expand(B, -1), ent(lg)], 1)
            keep = torch.zeros_like(e, dtype=torch.bool)
            for c in range(C):
                ec = e.masked_fill(lab != c, INF)
                idx = ec.topk(min(FILTER_K, ec.shape[1]), largest=False).indices
                keep |= torch.zeros_like(keep).scatter_(1, idx, torch.gather(lab == c, 1, idx))
            sup = F.normalize(sup, dim=-1) * keep[..., None]
            Wt = F.normalize(sup.transpose(1, 2) @ F.one_hot(lab, C).float(), dim=1)
            return (f[:, :1] @ Wt)[:, 0].softmax(-1), {}
        if method == "lame":  # official: rbf affinity, knn 5, sigma = mean k-th neighbour distance, 100 iterations
            lg, f, _ = fwd(S, episode_params(S, B), x, "eval")
            unary = -torch.log(lg.softmax(-1) + 1e-10)
            fe = F.normalize(f, dim=-1)
            dist = torch.cdist(fe, fe)
            sig = dist.topk(min(5, n), dim=-1, largest=False).values[..., -1].mean(1)[:, None, None]
            Wk = torch.exp(-dist ** 2 / (2 * sig ** 2))
            Y = (-unary).softmax(-1)
            for _ in range(100):
                Y = (-unary + Wk @ Y).softmax(-1)
            return Y[:, 0], {}
    raise ValueError(method)


class PSN(nn.Module):
    """person-specific normalisation (Baltrusaitis et al., FG'15): the median of the references is subtracted; its own
    linear classifier is trained on median-normalised episodes of the training subjects (the one baseline that does
    not use the shared source network)"""

    def __init__(self, D, C):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, C))

    def forward(self, x):
        med = x[:, 1:].median(1, keepdim=True).values if x.shape[1] > 1 else torch.zeros_like(x[:, :1])
        return self.net(x - med)


def train_psn(X, meta, rows, a, rng, C):
    model = PSN(X.shape[1], C).to(dev)
    y = torch.tensor(meta.label.values, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    for it in range(a.steps):
        q = rng.choice(rows, a.bs)
        K = int(rng.integers(0, a.kmax + 1))
        idx = np.concatenate([q[:, None], contexts(meta, q, K, "rand", rng, rows)], 1) if K else q[:, None]
        idx = torch.from_numpy(idx).to(dev)
        loss = F.cross_entropy(model(X[idx])[:, 0], y[idx[:, 0]], label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model.eval()


# ============================================================================ protocols
def contexts(meta, rows, K, mode, rng, pool_rows, j=0):
    """K references of the query's person (never the query itself).
    label-free:        rand, xsent (other sentence), contam (1/3 other people), past (K most recent earlier clips),
                       skew (4 clips of one class chosen uniformly at random + 2 random), dir{a} (composition
                       ~ Dirichlet(a), independent of the query; dir02 = 0.2, dir1 = 1)
    label-controlled:  same, other, neu6 (all neutral: SFDA-PFT calibration setting), mix (j share the query's class)"""
    act, y, sen = meta.actor.values, meta.label.values, meta.sentence.values
    out = np.zeros((len(rows), K), int)
    for i, r in enumerate(rows):
        pool = pool_rows[(act[pool_rows] == act[r]) & (pool_rows != r)]
        if mode.startswith("dir"):
            alpha = float("0." + mode[4:]) if mode[3] == "0" else float(mode[3:])
            cs = np.unique(y[pool])
            cnt = rng.multinomial(K, rng.dirichlet(np.full(len(cs), alpha)))
            out[i] = np.concatenate([rng.choice(pool[y[pool] == c], k, replace=True) for c, k in zip(cs, cnt) if k])
            continue
        if mode == "past":
            od = meta["order"].values
            pp = pool[od[pool] < od[r]]
            pp = pp[np.argsort(od[pp])][-K:]
            # An all-query sentinel means cold start, evaluated by Source (never by an adapting batch).
            out[i] = pp if len(pp) == K else np.full(K, r)
            continue
        if mode == "skew":
            cs = np.unique(y[pool]); pd_ = pool[y[pool] == rng.choice(cs)]
            out[i] = np.concatenate([rng.choice(pd_, 4, replace=len(pd_) < 4), rng.choice(pool, K - 4, replace=len(pool) < K - 4)])
            continue
        if mode == "neu6":
            pn = pool[y[pool] == neutral(meta.attrs["data"])]
            # a person without neutral clips has no neu6 episode: cold-start sentinel (excluded by the aggregator)
            out[i] = rng.choice(pn, K, replace=len(pn) < K) if len(pn) else np.full(K, r)
            continue
        if mode == "mix":
            ps, po = pool[y[pool] == y[r]], pool[y[pool] != y[r]]
            out[i] = np.concatenate([rng.choice(ps, j, replace=len(ps) < j), rng.choice(po, K - j, replace=len(po) < K - j)])
            continue
        if mode == "xsent":
            pool = pool[sen[pool] != sen[r]]
        if mode == "same":
            pool = pool[y[pool] == y[r]]
        if mode == "other":
            pool = pool[y[pool] == rng.choice([c for c in np.unique(y[pool]) if c != y[r]])]
        out[i] = rng.choice(pool, K, replace=len(pool) < K)
        if mode == "contam":
            others = pool_rows[act[pool_rows] != act[r]]
            out[i, : K // 3] = rng.choice(others, K // 3, replace=False)
    return out


def protocols(data, meta, suite="full"):
    """free: compositions not chosen with the query's label (rand, contam: 1/3 other people, skew: 4 + 2 random,
    dir02 / dir1: Dirichlet compositions independent of the query, past: natural history, MELD).  ctrl: label-controlled
    diagnostics (same; neu6 = all-neutral references, the condition-matched SFDA-PFT setting)."""
    past = ["past"] if "order" in meta else []
    if suite in ("lite", "paper"):  # neu6: all references neutral (the known-neutral setting of SFDA-PFT)
        return ["rand", "skew", "dir02"] + past, ["same"] + (["neu6"] if neutral(data) is not None else [])
    return ["rand", "skew", "dir02", "dir1", "contam"] + past, ["same"] + (["neu6"] if neutral(data) is not None else [])


def metrics(p, y, C):
    pred = p.argmax(1)
    conf = p.max(1); bins = np.linspace(0, 1, 16); ece = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (conf > lo) & (conf <= hi)
        if s.any():
            ece += s.mean() * abs((pred[s] == y[s]).mean() - conf[s].mean())
    f1 = [2 * np.sum((pred == c) & (y == c)) / max(1, np.sum(pred == c) + np.sum(y == c)) for c in range(C) if np.any(y == c)]
    return dict(acc=float(np.mean(pred == y) * 100), uar=float(np.mean([np.mean(pred[y == c] == c) for c in range(C) if np.any(y == c)]) * 100),
                f1=float(np.mean(f1) * 100), nll=float(-np.log(p[np.arange(len(y)), y] + 1e-8).mean()), ece=float(ece * 100))


def chunk_size(method, n, C):
    if method in CMN_VARIANTS:
        H = math.comb(n + C - 1, C - 1)
        return int(max(1, min(64, 32 * 3432 * 7 // (H * n))))
    return 64 if method == "shot" else 512


EP_KEYS = ["H_N", "lam", "anchor_p", "shrink", "upd_rel", "ent_drop", "eff_frac", "eff_steps",
           "err_real", "err_pred", "z2", "err_src", "err_cpn", "ent_src", "maxp", "pl_var",  # round 8: uncertainty
           "lp_true", "Eg"]  # round 9: posterior probability of the true composition, non-identified share


def eb_combine(ab):
    """deployment-level empirical Bayes of the composition hyperprior (CMN-EB): the posterior over the Dirichlet
    concentration alpha given ALL unlabelled reference sets of the evaluation stream, p(alpha | U_1..U_E) prop. to
    prod_e p(U_e | alpha) (uniform hyperprior on the grid), then p(y_e | U_1..U_E) = sum_alpha p(alpha | .) p(y_e | U_e,
    alpha).  Transductive across people (it pools their unlabelled reference sets), never uses a label.
    returns (predictions (E, C), posterior mean log2 alpha)"""
    w = torch.from_numpy(ab["alpha_ll"]).double().sum(0).softmax(0).numpy()
    return (w[None, :, None] * ab["alpha_pred"]).sum(1).astype(np.float32), float((w * np.log2(ALPHAS)).sum())


def predict(a, model, X, ctx, rows, C, Y, diag=None, store=None):
    """query probabilities (len(rows), C).  diag: summary diagnostics; store: receives 'alt' {name: (N, C)} (the
    alternative read-outs / grid values of the same run) and 'ep' {name: (N,)} (per-episode diagnostics)"""
    if ctx is not None and np.any(np.all(ctx == rows[:, None], axis=1)):  # cold start (past): Source, never future clips
        cold = np.all(ctx == rows[:, None], axis=1)
        p = np.empty((len(rows), C), dtype=np.float32)
        sc, sw = {}, {}
        p[cold] = predict(a, model, X, None, rows[cold], C, Y, None, sc)
        if (~cold).any():
            p[~cold] = predict(a, model, X, ctx[~cold], rows[~cold], C, Y, diag, sw)
        if diag is not None: diag['cold_start_fraction'] = float(cold.mean())
        if store is not None:
            store["alt"], store["ep"] = {}, {}
            for k in set(sc.get("alt", {})) | set(sw.get("alt", {})):
                v = p.copy()
                if k in sc.get("alt", {}): v[cold] = sc["alt"][k]
                if k in sw.get("alt", {}): v[~cold] = sw["alt"][k]
                store["alt"][k] = v
            for k in sw.get("ep", {}):
                v = np.full(len(rows), np.nan, np.float32); v[~cold] = sw["ep"][k]; store["ep"][k] = v
            if sw.get("ab"):  # cold-start episodes carry no evidence about alpha
                store["ab"] = dict(alpha_ll=np.zeros((len(rows), len(ALPHAS)), np.float32),
                                   alpha_pred=np.repeat(p[:, None], len(ALPHAS), 1))
                store["ab"]["alpha_ll"][~cold] = sw["ab"]["alpha_ll"]; store["ab"]["alpha_pred"][~cold] = sw["ab"]["alpha_pred"]
        return p
    q = torch.from_numpy(rows).to(dev)[:, None]
    idx = q if ctx is None else torch.cat([q, torch.from_numpy(ctx).to(dev)], 1)
    out, D, ALT, EP, AB = [], {"Etheta": [], "mu": []}, {}, {}, {}
    ch = chunk_size(a.method, idx.shape[1], C)
    for i in range(0, len(idx), ch):
        xb = X[idx[i:i + ch]]
        if a.method == "psn":
            with torch.no_grad():
                p, d = model(xb)[:, 0].float().softmax(-1), {}
        else:
            p, d = tta(model, xb, a.method, C, Y[idx[i:i + ch, 1:]], extras=store is not None)
            if "logpN" in d:  # DIAGNOSTIC ONLY: log posterior probability of the true batch composition
                nb = idx.shape[1]
                where = getattr(model, "_hwhere", {}).get(nb)
                if where is None:
                    where = {tuple(r): k_ for k_, r in enumerate(sublattice(nb, C)["H"].long().cpu().numpy())}
                    model._hwhere = {**getattr(model, "_hwhere", {}), nb: where}
                lab_ = Y[idx[i:i + ch]].cpu().numpy()
                hidx = torch.tensor([where[tuple(np.bincount(r, minlength=C))] for r in lab_], device=d["logpN"].device)
                d["lp_true"] = d["logpN"].gather(1, hidx[:, None])[:, 0]
            if "m_sub" in d and hasattr(model, "b_true"):  # DIAGNOSTIC ONLY: realised centre error vs its prediction
                with torch.no_grad():
                    bt = model.b_true[idx[i:i + ch, 0]] @ model.st["Q"]
                    e = d["m_sub"] - bt
                    d["err_real"], d["err_pred"] = (e ** 2).sum(-1), torch.diagonal(d["Serr"], dim1=1, dim2=2).sum(-1)
                    d["z2"] = torch.einsum("bd,bde,be->b", e, torch.linalg.inv(d["Serr"] + 1e-6 * torch.eye(C - 1, device=e.device)), e)
                    d["err_src"] = ((model.st["m0"] - bt) ** 2).sum(-1)       # no adaptation (source centre)
                    d["err_cpn"] = ((d["mu"] @ model.st["Q"] - bt) ** 2).sum(-1)  # the CPN centre
        out.append(p.float())
        for k in D:
            if k in d: D[k].append(d[k].float())
        for k, v in d.get("alt", {}).items():
            ALT.setdefault(k, []).append(v.float())
        for k in EP_KEYS:
            if d.get(k) is not None:
                EP.setdefault(k, []).append((d[k].argmax(-1) if d[k].dim() == 2 else d[k]).float())
        for k in ("alpha_ll", "alpha_pred"):
            if k in d: AB.setdefault(k, []).append(d[k].float())
    P = torch.cat(out).cpu().numpy()
    if store is not None:
        store["alt"] = {k: torch.cat(v).cpu().numpy() for k, v in ALT.items()}
        store["ep"] = {k: torch.cat(v).cpu().numpy() for k, v in EP.items()}
        store["ab"] = {k: torch.cat(v).cpu().numpy() for k, v in AB.items()}
    if ctx is not None and D["Etheta"]:
        th_true = F.one_hot(Y[idx], C).float().mean(1)
        l1 = (torch.cat(D["Etheta"]) - th_true).abs().sum(1)
        if store is not None:
            store["ep"]["comp_l1"] = l1.cpu().numpy()
        if diag is not None:
            diag["comp_l1"] = float(l1.mean())
    if diag is not None and ctx is not None and D["mu"]:  # subject-offset recovery
        Qm = model.st["Q"]
        mu = torch.cat(D["mu"])
        err = ((mu - model.b_true[torch.from_numpy(rows).to(dev)]) @ Qm).pow(2).sum(1).mean()
        diag["offset_relerr"] = float(err / model.b_spread)
    return P


def additivity_u(u, act, y, C, rng):
    """two-way decomposition of the cell means M_sc = g + a_s + b_c + G_sc in the BN units u used by CMN (and in the
    expression subspace), with cross-half inner products so that clip noise does not bias the shares"""
    S_ = np.unique(act)
    A = np.zeros((len(S_), C, u.shape[1])); Bh = np.zeros_like(A)
    for i, s in enumerate(S_):
        for c in range(C):
            ix = rng.permutation(np.where((act == s) & (y == c))[0])
            if len(ix) < 2:
                return None
            h = len(ix) // 2
            A[i, c], Bh[i, c] = u[ix[:h]].mean(0), u[ix[h:]].mean(0)

    def dec(M):
        g = M.mean((0, 1)); a_ = M.mean(1) - g; b_ = M.mean(0) - g
        return a_, b_, M - g - a_[:, None] - b_[None], M - g

    aA, bA, GA, TA = dec(A); aB, bB, GB, TB = dec(Bh)
    tot = (TA * TB).sum()
    return dict(subject=float((aA * aB).sum() * C / tot), expression=float((bA * bB).sum() * len(S_) / tot),
                interaction=float((GA * GB).sum() / tot))


# ============================================================================ run
def model_key(a, fold):
    cache_dir = f'{CACHE}/models/{CODE_SHA}'
    os.makedirs(cache_dir, exist_ok=True)
    return f'{cache_dir}/{a.data}_{a.feat}_{a.net}_s{a.seed}_f{fold}_t{a.steps}_b{a.bs}_lr{a.lr}_wd{a.wd}_k{a.kmax}'


def get_model(a, X, meta, tr, te, fold, rng, C):
    """the source network of this (data, feature, network kind, seed, fold), identical for every test-time method
    (paired comparisons), its closed-form statistics, and the cached cross-fit calibration / DART module"""
    if (meta.fold.values < 0).any():
        # A fixed source-only pool defines one frozen model, irrespective of how
        # target sessions are partitioned for evaluation. Reuse its calibration
        # and variant fits too; target labels only enter the diagnostics below.
        fold = 0
        torch.manual_seed(a.seed * 100)
        rng = np.random.default_rng(a.seed * 100)
    ck = model_key(a, fold)
    model = Source(X.shape[1], C).to(dev)
    if os.path.exists(ck + '.pt'):
        saved = torch.load(ck + '.pt', map_location=dev, weights_only=False)
        model.load_state_dict(saved['weights']); model.st = saved['stats']; model.eval()
    else:
        model = make_network(a.net, X, meta, tr, a, rng, C)
        tmp = ck + f'.{os.getpid()}.tmp'
        torch.save(dict(weights=model.state_dict(), stats=model.st), tmp)
        os.replace(tmp, ck + '.pt')
    model.requires_grad_(False)
    model.neutral = neutral(a.data) or 0
    attach_csn(model, X, meta, tr, C)
    if a.method in NEEDS_XFIT or a.profile:
        if os.path.exists(ck + '.crossfit.json'):
            with open(ck + '.crossfit.json') as fi: model.xfit = json.load(fi)
        else:
            model.xfit = crossfit(X, meta, tr, a, a.seed, fold, C, a.net, aux_path=ck + '.aux.pt')
            tmp = ck + f'.{os.getpid()}.json.tmp'
            with open(tmp, 'w') as fo: json.dump(model.xfit, fo)
            os.replace(tmp, ck + '.crossfit.json')
        if a.method in VARIANT_REFIT and not a.profile:
            vp = ck + f'.xfit_{a.method}.json'
            if os.path.exists(vp):
                with open(vp) as fi: v = json.load(fi)
            else:
                if not os.path.exists(ck + '.aux.pt'):  # e.g. a cross-fit cached without it: recompute it once
                    xf = crossfit(X, meta, tr, a, a.seed, fold, C, a.net, light=True, aux_path=ck + '.aux.pt')
                    model.xfit.setdefault("tau_bn", xf["tau_bn"])
                v = refit_variant(a.method, ck + '.aux.pt', X, meta, a, fold, C)
                tmp = ck + f'.{os.getpid()}.{a.method}.json.tmp'
                with open(tmp, 'w') as fo: json.dump(v, fo)
                os.replace(tmp, vp)
            model.xfit = dict(model.xfit, stack_params=v["stack_params"], variant_refit=v)
            if "post_w" in v:
                model.xfit["post_w"] = v["post_w"]
    if a.method == "dart" or a.profile:
        if os.path.exists(ck + '.dart.pt'):
            model.dart = DARTNet(C).to(dev); model.dart.load_state_dict(torch.load(ck + '.dart.pt', map_location=dev))
            model.dart.eval().requires_grad_(False)
        else:
            torch.manual_seed(a.seed * 100 + fold + 3)
            model.dart = train_dart(model, X, meta, tr, a, np.random.default_rng(a.seed * 100 + fold + 3), C)
            tmp = ck + f'.{os.getpid()}.dart.tmp'
            torch.save(model.dart.state_dict(), tmp); os.replace(tmp, ck + '.dart.pt')
    cfg = cmn_cfg(a.method) if a.method in CMN_VARIANTS else {}
    if cfg.get("stats_frac"):  # robustness: population statistics from a random fraction of the training people
        subs = np.unique(meta.actor.values[tr])
        keep = np.random.default_rng(a.seed * 100 + fold + 5).choice(subs, max(2, int(round(cfg["stats_frac"] * len(subs)))), replace=False)
        model.st = subject_stats(model, X, meta, tr[np.isin(meta.actor.values[tr], keep)], C)
    with torch.no_grad():
        st_te = subject_stats(model, X, meta, te, C)  # diagnostics only (offset recovery): never used to predict
        model.b_true = torch.zeros(len(meta), model.bn.weight.numel(), device=dev)
        model.b_true[torch.from_numpy(te).to(dev)] = st_te["b"][torch.from_numpy(np.searchsorted(st_te["subs"], meta.actor.values[te])).to(dev)]
        truth = f"{PROC}/{a.data}_truth.npz"  # synthetic data: the exact subject offsets are known
        if os.path.exists(truth):
            z = np.load(truth)
            mu_x = torch.tensor((z["mu"][np.searchsorted(z["subjects"], meta.actor.values[te])] - X.xm) / X.xs, dtype=torch.float32, device=dev)
            model.b_true[torch.from_numpy(te).to(dev)] = pop_units(model, mu_x)
        bt = model.b_true[torch.from_numpy(te).to(dev)]
        model.b_spread = float((bt @ model.st["Q"] - model.st["m0"]).pow(2).sum(1).mean())
    return model


def test_folds(meta):
    """evaluation folds (a negative fold marks the training-only pool of the cross-condition protocol)"""
    return [int(f) for f in sorted(meta.fold.unique()) if f >= 0]


def load_fold(a, meta, fold):
    X = load(a.data, fold, a.feat)
    if (meta.fold.values < 0).any():
        # cross-condition protocol (round 10): rows with fold -1 form the ONLY training pool (one recording
        # condition); the test folds are people of another condition the source network never saw (frozen model)
        tr, te = np.where(meta.fold.values < 0)[0], np.where(meta.fold.values == fold)[0]
    else:
        tr, te = np.where(meta.fold != fold)[0], np.where(meta.fold == fold)[0]
    xm, xs = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xt = torch.tensor((X - xm) / xs, dtype=torch.float32, device=dev)
    Xt.xm, Xt.xs = xm, xs
    return Xt, tr, te


def run(a):
    meta = pd.read_csv(f"{PROC}/{a.data}_meta.csv"); meta.attrs["data"] = a.data
    C = meta.label.nunique()
    free, ctrl = protocols(a.data, meta, a.suite)
    nctx = a.nctx
    if a.method == "neucal" and neutral(a.data) is None:
        raise ValueError("neucal needs a neutral class")
    mixes = [f"mix{j}" for j in range(a.kctx + 1)]
    kcurve = [] if a.suite == "lite" else KCURVE  # "paper": lite protocols + accuracy vs K
    order = ["K0"] + [f"{m}#{c}" for m in free + ctrl + mixes for c in range(nctx)] + [f"K{k}#0" for k in kcurve]
    res = dict(args=vars(a), code_sha256=CODE_SHA, protocols_free=free, protocols_ctrl=ctrl, pred_order=order, folds={})
    Y = torch.tensor(meta.label.values, device=dev)
    folds = test_folds(meta) if a.folds is None else [int(v) for v in a.folds.split(",")]
    stem = f"{OUT}/pred/{a.data}_{a.tag}_s{a.seed}"
    for fold in folds:
        t0 = time.time()
        torch.manual_seed(a.seed * 100 + fold); rng = np.random.default_rng(a.seed * 100 + fold)
        X, tr, te = load_fold(a, meta, fold)
        r = {}
        if a.method == "psn":
            model = train_psn(X, meta, tr, a, rng, C)
        else:
            model = get_model(a, X, meta, tr, te, fold, rng, C)
            if hasattr(model, "xfit"):
                r["crossfit"] = model.xfit
            r["stats"] = dict(trT=float(torch.trace(model.st["T"])), trSin=float(torch.trace(model.st["Sin"])),
                              trSG=float(torch.trace(model.st["SG"])), f2=float((model.st["f"] ** 2).sum(1).mean()))
            if a.method == "source":
                with torch.no_grad():
                    ut = pop_units(model, X[te]).cpu().numpy()
                r["additivity_u"] = additivity_u(ut, meta.actor.values[te], meta.label.values[te], C, np.random.default_rng(fold))
                r["additivity_sub"] = additivity_u(ut @ model.st["Q"].cpu().numpy(), meta.actor.values[te], meta.label.values[te], C, np.random.default_rng(fold))
        yt = meta.label.values[te]
        st0 = {}
        P, ALT, EP = [predict(a, model, X, None, te, C, Y, None, st0)], [st0.get("alt", {})], [st0.get("ep", {})]
        r["K0"] = metrics(P[0], yt, C)
        saved = {}
        for mode in free + ctrl + mixes:
            if mode == f'mix{a.kctx}' and 'same' in saved:  # identical sampling seed and pool: the same protocol
                r[mode] = dict(r['same']); p_, al_, ep_ = saved['same']
                P += p_; ALT += al_; EP += ep_
                continue
            ps, als, eps_, dg = [], [], [], {}
            for c_seed in range(nctx):
                erng = np.random.default_rng(10000 * c_seed + 1000 + fold)  # identical for every method and seed
                md, j = ("mix", int(mode[3:])) if mode.startswith("mix") else (mode, 0)
                stor = {}
                ps.append(predict(a, model, X, contexts(meta, te, a.kctx, md, erng, te, j), te, C, Y, dg if c_seed == 0 else None, stor))
                if stor.get("ab"):
                    stor["alt"]["eb"], la = eb_combine(stor["ab"])
                    dg.setdefault("eb_log2alpha", []).append(la)
                als.append(stor.get("alt", {})); eps_.append(stor.get("ep", {}))
            P += ps; ALT += als; EP += eps_
            saved[mode] = (ps, als, eps_)
            ms = [metrics(p, yt, C) for p in ps]
            r[mode] = {k: float(np.mean([m[k] for m in ms])) for k in ["acc", "uar", "f1", "nll", "ece"]}
            r[mode]["acc_ctxsd"] = float(np.std([m["acc"] for m in ms]))
            if "eb_log2alpha" in dg:
                dg["eb_log2alpha"] = float(np.mean(dg["eb_log2alpha"]))
            r[mode].update(dg)
            for k in ("upd_rel", "ent_drop", "eff_frac", "eff_steps", "H_N", "lam", "shrink"):
                v = [e[k] for e in eps_ if k in e]
                if v: r[mode][k] = float(np.nanmean(np.concatenate(v)))
        for k in kcurve:
            stor = {}
            P.append(predict(a, model, X, contexts(meta, te, k, "rand", np.random.default_rng(1000 + fold), te), te, C, Y, None, stor))
            if stor.get("ab"):
                stor["alt"]["eb"] = eb_combine(stor["ab"])[0]
            ALT.append(stor.get("alt", {})); EP.append(stor.get("ep", {}))
            r[f"K{k}"] = metrics(P[-1], yt, C)
        r["minutes"] = (time.time() - t0) / 60
        res["folds"][int(fold)] = r
        np.save(f"{stem}_f{fold}.npy", np.stack(P).astype(np.float32))
        names = sorted(set().union(*[set(x) for x in ALT]))
        if names:  # missing read-outs (e.g. K = 0) fall back to the main prediction
            res["alt_names"] = names
            np.save(f"{stem}_f{fold}_alt.npy", np.stack([np.stack([al.get(nm, p) for al, p in zip(ALT, P)]) for nm in names]).astype(np.float32))
        enames = sorted(set().union(*[set(x) for x in EP]))
        if enames:
            res["ep_names"] = enames
            np.savez_compressed(f"{stem}_f{fold}_ep.npz", **{nm: np.stack([e.get(nm, np.full(len(te), np.nan, np.float32)) for e in EP]).astype(np.float16) for nm in enames})
        print(fold, {k: round(v["acc"], 2) for k, v in r.items() if isinstance(v, dict) and "acc" in v}, flush=True)
    keys = [k for k, v in next(iter(res["folds"].values())).items() if isinstance(v, dict) and "acc" in v]
    res["mean"] = {k: {m: float(np.mean([f[k][m] for f in res["folds"].values() if m in f[k]]))
                       for m in next(iter(res["folds"].values()))[k] if isinstance(next(iter(res["folds"].values()))[k][m], float)} for k in keys}
    return res


# ============================================================================ streaming protocol (round 8)
# The episodic protocols give every method the same K references.  Stream methods (TENT-online, NOTE, RoTTA) are
# designed for long, temporally correlated test streams; here they run COMPLETE (all their components, official
# hyper-parameters, no tuning) on a person-blocked test stream -- people one after the other, a person's clips in their
# natural order (MELD) or a random order -- either continually (never reset; their native setting) or reset at every
# person boundary (the person identity that CPN uses).  They process mini-batches of 8 consecutive clips (about the
# 7-clip episode of CPN; a batch of 64 would hand them a person's entire repertoire at once, including clips later than
# the one predicted) and carry their state (statistics, memory, teacher) over the stream, i.e. they accumulate the
# person's whole history; the window methods (BN, CPN) see only the K = 6 most recent EARLIER clips of the person.
STREAM_METHODS = ["source", "bn_win", "cpn_win", "tent_online", "note_full", "rotta_full"]
STREAM_BS, STREAM_MEM = 8, 64  # stream mini-batch, memory size / update interval (official NOTE / RoTTA: 64)


def stream_order(meta, te, seed):
    """person-blocked test stream: people in random order; within a person the natural order (MELD: 'order') or random"""
    rng = np.random.default_rng(seed)
    act = meta.actor.values[te]
    out = []
    for p_ in rng.permutation(np.unique(act)):
        r = te[act == p_]
        out.append(r[np.argsort(meta["order"].values[r], kind="stable")] if "order" in meta else rng.permutation(r))
    return np.concatenate(out)


def _bn_logits(S, h, mu, var, g, be):
    return F.gelu((h - mu) / torch.sqrt(var + S.bn.eps) * g + be) @ S.cls.weight.T + S.cls.bias


def _h(S, x):
    return x @ S.f1.weight.T + S.f1.bias


def _blocks(order, act, reset, bs):
    """stream batches: consecutive chunks of at most bs clips; with reset, never across a person boundary.
    yields (indices into order, new_person_flag)"""
    i, prev = 0, None
    while i < len(order):
        j = min(i + bs, len(order))
        if reset:
            same = np.where(act[order[i:j]] != act[order[i]])[0]
            j = i + (same[0] if len(same) else j - i)
        yield np.arange(i, j), act[order[i]] != prev
        prev = act[order[i]]; i = j


def stream_tent(S, X, order, act, reset, C):
    """TENT online (Wang et al., ICLR'21): BN in batch mode, one Adam(1e-3) entropy step on the BN affine per stream
    batch; the prediction of a batch is the forward pass before its update (official forward_and_adapt)"""
    out = torch.zeros(len(order), C, device=dev)
    g = be = opt = None
    for b, new in _blocks(order, act, reset, STREAM_BS):
        if g is None or (reset and new):
            g, be = S.bn.weight.detach().clone().requires_grad_(), S.bn.bias.detach().clone().requires_grad_()
            opt = torch.optim.Adam([g, be], lr=1e-3, betas=(0.9, 0.999))
        h = _h(S, X[torch.from_numpy(order[b]).to(dev)])
        with torch.enable_grad():
            mu, var = h.mean(0), h.var(0, unbiased=False)
            lg = _bn_logits(S, h, mu, var, g, be)
            loss = ent(lg).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        out[torch.from_numpy(b).to(dev)] = lg.detach().softmax(-1)
    return out


class _PBRS:
    """NOTE's prediction-balanced reservoir sampling memory (official algorithm)"""

    def __init__(self, cap, C, rng):
        self.cap, self.C, self.rng = cap, C, rng
        self.data = [[] for _ in range(C)]; self.counter = [0] * C

    def occ(self):
        return [len(d) for d in self.data]

    def add(self, h, c):
        self.counter[c] += 1
        ok = True
        if sum(self.occ()) >= self.cap:
            o = self.occ(); mx = max(o); largest = [k for k in range(self.C) if o[k] == mx]
            if c not in largest:
                lc = int(self.rng.choice(largest)); self.data[lc].pop(int(self.rng.integers(len(self.data[lc]))))
            elif self.rng.uniform() <= o[c] / self.counter[c]:
                self.data[c].pop(int(self.rng.integers(len(self.data[c]))))
            else:
                ok = False
        if ok:
            self.data[c].append(h)

    def get(self):
        return torch.stack([h for d in self.data for h in d])


def stream_note(S, X, order, act, reset, C, seed):
    """NOTE (Gong et al., NeurIPS'22): online single-clip inference with test-time BN statistics; a prediction-balanced
    reservoir memory (PBRS, 64); every 64 clips the BN statistics are updated from the memory (momentum 0.01) and the
    BN affine takes one Adam(1e-4) entropy step on the memory.  IABN's instance statistics need spatial positions; on
    the vector bottleneck IABN reduces to BN."""
    out = torch.zeros(len(order), C, device=dev)
    rng = np.random.default_rng(seed)
    mu = var = g = be = opt = mem = None; cnt = 0
    prev = None
    for i, r in enumerate(order):
        if mu is None or (reset and act[r] != prev):
            mu, var = S.bn.running_mean.clone(), S.bn.running_var.clone()
            g, be = S.bn.weight.detach().clone().requires_grad_(), S.bn.bias.detach().clone().requires_grad_()
            opt = torch.optim.Adam([g, be], lr=1e-4); mem = _PBRS(STREAM_MEM, C, rng); cnt = 0
        prev = act[r]
        h = _h(S, X[r][None])
        with torch.no_grad():
            lg = _bn_logits(S, h, mu, var, g, be)
        out[i] = lg[0].softmax(-1)
        mem.add(h[0], int(lg.argmax()))
        cnt += 1
        if cnt % STREAM_MEM == 0:
            H = mem.get()
            bm, bv = H.mean(0), H.var(0, unbiased=False)
            with torch.enable_grad():
                loss = ent(_bn_logits(S, H, bm, bv, g, be)).mean()
                opt.zero_grad(); loss.backward(); opt.step()
            mu, var = 0.99 * mu + 0.01 * bm, 0.99 * var + 0.01 * bv
    return out


class _CSTU:
    """RoTTA's category-balanced sampling with timeliness and uncertainty (official algorithm)"""

    def __init__(self, cap, C):
        self.cap, self.C, self.per = cap, C, cap / C
        self.data = [[] for _ in range(C)]  # items: [h, uncertainty, age]

    def score(self, age, unc):
        return 1 / (1 + math.exp(-age / self.cap)) + unc / math.log(self.C)

    def occ(self):
        return [len(d) for d in self.data]

    def _remove_from(self, classes, base):
        best = None
        for c in classes:
            for k, it in enumerate(self.data[c]):
                sc = self.score(it[2], it[1])
                if best is None or sc >= best[0]:
                    best = (sc, c, k)
        if best is None:
            return True
        if best[0] > base:
            self.data[best[1]].pop(best[2]); return True
        return False

    def add(self, h, c, unc):
        base = self.score(0, unc)
        o = self.occ()
        if o[c] < self.per:
            ok = True if sum(o) < self.cap else self._remove_from([k for k in range(self.C) if o[k] == max(o)], base)
        else:
            ok = self._remove_from([c], base)
        if ok:
            self.data[c].append([h, unc, 0])
        for d in self.data:
            for it in d:
                it[2] += 1

    def get(self):
        items = [it for d in self.data for it in d]
        return torch.stack([it[0] for it in items]), torch.tensor([it[2] / self.cap for it in items], device=dev)


def stream_rotta(S, X, order, act, reset, C, seed):
    """RoTTA (Yuan et al., CVPR'23), complete: robust BN (statistics = EMA with alpha 0.05 of the memory's statistics),
    category-balanced memory with timeliness and uncertainty (CSTU, 64), EMA teacher (nu 0.001) predicting the stream
    batch, student updated every 64 clips on the memory with a timeliness-weighted teacher-student loss, Adam(1e-3) on the
    robust-BN affine.  Strong augmentation at the feature level: input dropout 0.2 + Gaussian noise 0.1 (the image
    augmentations of the paper have no feature-vector counterpart)."""
    out = torch.zeros(len(order), C, device=dev)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    st = None; cnt = 0
    for b, new in _blocks(order, act, reset, STREAM_BS):
        if st is None or (reset and new):
            st = dict(mu_t=S.bn.running_mean.clone(), var_t=S.bn.running_var.clone(),
                      mu_s=S.bn.running_mean.clone(), var_s=S.bn.running_var.clone(),
                      g_t=S.bn.weight.detach().clone(), be_t=S.bn.bias.detach().clone(),
                      g_s=S.bn.weight.detach().clone().requires_grad_(), be_s=S.bn.bias.detach().clone().requires_grad_())
            opt = torch.optim.Adam([st["g_s"], st["be_s"]], lr=1e-3, betas=(0.9, 0.999))
            mem = _CSTU(STREAM_MEM, C); cnt = 0
        xb = X[torch.from_numpy(order[b]).to(dev)]
        h = _h(S, xb)
        with torch.no_grad():
            lg = _bn_logits(S, h, st["mu_t"], st["var_t"], st["g_t"], st["be_t"])  # teacher, eval mode
            pr = lg.softmax(-1)
            unc = -(pr * torch.log(pr + 1e-6)).sum(-1)
        out[torch.from_numpy(b).to(dev)] = pr
        for k in range(len(b)):
            mem.add(xb[k], int(pr[k].argmax()), float(unc[k]))
            cnt += 1
            if cnt % STREAM_MEM == 0:
                xm, age = mem.get()
                w = torch.exp(-age) / (1 + torch.exp(-age))
                xs = F.dropout(xm, 0.2) + 0.1 * torch.randn(xm.shape, device=dev, generator=gen)
                with torch.no_grad():  # teacher in train mode: robust BN statistics updated on the weak memory
                    ht = _h(S, xm)
                    st["mu_t"] = 0.95 * st["mu_t"] + 0.05 * ht.mean(0)
                    st["var_t"] = 0.95 * st["var_t"] + 0.05 * ht.var(0, unbiased=False)
                    pt = _bn_logits(S, ht, st["mu_t"], st["var_t"], st["g_t"], st["be_t"]).softmax(-1)
                hs = _h(S, xs)
                st["mu_s"] = (0.95 * st["mu_s"] + 0.05 * hs.mean(0)).detach()
                st["var_s"] = (0.95 * st["var_s"] + 0.05 * hs.var(0, unbiased=False)).detach()
                with torch.enable_grad():
                    ls = _bn_logits(S, hs, st["mu_s"], st["var_s"], st["g_s"], st["be_s"]).log_softmax(-1)
                    loss = (-(pt * ls).sum(-1) * w).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    st["g_t"] = 0.999 * st["g_t"] + 0.001 * st["g_s"].detach()
                    st["be_t"] = 0.999 * st["be_t"] + 0.001 * st["be_s"].detach()
    return out


def stream_window(a, S, X, meta, order, method, C, Y):
    """window methods: the query's K most recent EARLIER clips of the same person are its references (strictly
    causal); the first clip of a person is predicted by the source model"""
    act = meta.actor.values[order]
    pos = np.zeros(len(order), int)
    for i in range(1, len(order)):
        pos[i] = pos[i - 1] + 1 if act[i] == act[i - 1] else 0
    P = np.zeros((len(order), C), np.float32)
    m = {"bn_win": "bn", "cpn_win": "cpn"}.get(method, method)
    a_ = argparse.Namespace(**vars(a)); a_.method = m
    for k in range(0, a.kctx + 1):
        sel = np.where(np.minimum(pos, a.kctx) == k)[0]
        if not len(sel):
            continue
        if k == 0:
            a_.method = "source"
            P[sel] = predict(a_, S, X, None, order[sel], C, Y)
            a_.method = m
            continue
        ctx = np.stack([order[i - k:i] for i in sel])
        P[sel] = predict(a_, S, X, ctx, order[sel], C, Y)
    return P


def run_stream(a):
    """person-blocked stream on every fold -> $CMN_OUT/stream_{data}_{feat}_s{seed}.json (+ predictions)"""
    meta = pd.read_csv(f"{PROC}/{a.data}_meta.csv"); meta.attrs["data"] = a.data
    C = meta.label.nunique()
    Y = torch.tensor(meta.label.values, device=dev)
    keys = ["source", "bn_win", "cpn_win"] + [f"{m}{r}" for m in ("tent_online", "note_full", "rotta_full") for r in ("", "_reset")]
    res = dict(args=vars(a), code_sha256=CODE_SHA, methods=keys, folds={})
    allP, allY = {k: [] for k in keys}, []
    for fold in test_folds(meta):
        torch.manual_seed(a.seed * 100 + fold)
        X, tr, te = load_fold(a, meta, fold)
        a.method = "cpn"
        model = get_model(a, X, meta, tr, te, fold, np.random.default_rng(a.seed * 100 + fold), C)
        order = stream_order(meta, te, 5000 + fold)
        act = meta.actor.values
        y = meta.label.values[order]
        P = {}
        for k in ("source", "bn_win", "cpn_win"):
            P[k] = stream_window(a, model, X, meta, order, k, C, Y)
        for reset in (False, True):
            sfx = "_reset" if reset else ""
            P["tent_online" + sfx] = stream_tent(model, X, order, act, reset, C).cpu().numpy()
            P["note_full" + sfx] = stream_note(model, X, order, act, reset, C, a.seed * 100 + fold).cpu().numpy()
            P["rotta_full" + sfx] = stream_rotta(model, X, order, act, reset, C, a.seed * 100 + fold).cpu().numpy()
        res["folds"][int(fold)] = {k: metrics(P[k], y, C) for k in keys}
        res["folds"][int(fold)]["order"] = order.tolist()
        np.save(f"{OUT}/pred/stream_{a.data}_{a.feat}_s{a.seed}_f{fold}.npy", np.stack([P[k] for k in keys]).astype(np.float32))
        for k in keys:
            allP[k].append(P[k])
        allY.append(y)
        print(fold, {k: round(v["acc"], 2) for k, v in res["folds"][int(fold)].items() if isinstance(v, dict)}, flush=True)
    yy = np.concatenate(allY)
    res["pooled"] = {k: metrics(np.concatenate(allP[k]), yy, C) for k in keys}
    fn = f"{OUT}/stream_{a.data}_{a.feat}_s{a.seed}.json"
    with open(fn + ".tmp", "w") as fo:
        json.dump(res, fo)
    os.replace(fn + ".tmp", fn)
    print("STREAM", a.data, json.dumps({k: round(v["acc"], 2) for k, v in res["pooled"].items()}), flush=True)


PROFILE_K = [1, 2, 4, 6, 8]
PROFILE_METHODS = ["bn", "tent", "sar", "delta", "note", "rotta", "csn", "lame", "t3a", "src_lab", "pl_shrink", "cmn",
                   "cpn7", "cpn", "cpn_p16", "cpn_p64", "cpn_p256"]


def profile(a):
    """cost of every test-time method on one fold of one dataset: online latency of one episode (median of 30), batched
    time per episode, peak GPU memory, for K references in PROFILE_K -> $CMN_OUT/profile_{data}_{feat}_{net}.json"""
    meta = pd.read_csv(f"{PROC}/{a.data}_meta.csv"); meta.attrs["data"] = a.data
    C = meta.label.nunique()
    fold = test_folds(meta)[0]
    X, tr, te = load_fold(a, meta, fold)
    a.method = "cpn"
    model = get_model(a, X, meta, tr, te, fold, np.random.default_rng(a.seed * 100 + fold), C)
    Y = torch.tensor(meta.label.values, device=dev)
    res = dict(data=a.data, feat=a.feat, net=a.net, device=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu",
               code_sha256=CODE_SHA, C=C, rows=[])
    sync = (lambda: torch.cuda.synchronize()) if dev == "cuda" else (lambda: None)
    for K in a.profile_k:
        ctx = contexts(meta, te, K, "rand", np.random.default_rng(1000 + fold), te)
        idx = torch.cat([torch.from_numpy(te).to(dev)[:, None], torch.from_numpy(ctx).to(dev)], 1)
        for m in a.profile_methods:
            ch = min(chunk_size(m, K + 1, C), len(idx))
            xb1, xb = X[idx[:1]], X[idx[:ch]]
            tta(model, xb1, m, C, Y[idx[:1, 1:]]); sync()
            ts = []
            for _ in range(30):
                t = time.perf_counter(); tta(model, xb1, m, C, Y[idx[:1, 1:]]); sync(); ts.append(time.perf_counter() - t)
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated() if dev == "cuda" else 0
            t = time.perf_counter()
            for _ in range(3):
                tta(model, xb, m, C, Y[idx[:ch, 1:]])
            sync(); tb = (time.perf_counter() - t) / 3
            peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 20 if dev == "cuda" else float("nan")
            H = math.comb(K + C, C - 1) if m.startswith(("cmn", "cpn")) else None
            if H is not None and CMN_VARIANTS.get(m, {}).get("prune"):
                H = min(H, CMN_VARIANTS[m]["prune"])
            a_ = argparse.Namespace(**vars(a)); a_.method = m  # accuracy of the same configuration on the whole fold
            pr = predict(a_, model, X, ctx, te, C, Y)
            res["rows"].append(dict(K=K, method=m, online_ms=1000 * float(np.median(ts)), batch=ch,
                                    batch_ms_per_episode=1000 * tb / ch, peak_mb_per_batch=peak, hypotheses=H,
                                    acc=float((pr.argmax(1) == meta.label.values[te]).mean() * 100)))
            print(K, m, round(res["rows"][-1]["online_ms"], 2), "ms online", round(res["rows"][-1]["batch_ms_per_episode"], 3), "ms/episode batched", round(peak, 1), "MB", flush=True)
    fn = f"{OUT}/profile_{a.data}_{a.feat}_{a.net}.json"
    with open(fn + ".tmp", "w") as fo: json.dump(res, fo, indent=1)
    os.replace(fn + ".tmp", fn)


NCTX, KCURVE = 3, [1, 2, 4, 8]
METHODS = ["source", "psn", "bn", "bnprior", "srcbn", "csn", "tent", "sar", "delta", "shot", "t3a", "lame", "neucal",
           "compass_em", "dart", "pl_bn", "pl_iter", "pl_shrink", "note", "rotta", "src_lab", "laem"] + list(GRIDS) + list(CMN_VARIANTS)


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ravdess")
    ap.add_argument("--method", default="cmn", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--feat", default="l14", choices=["l14", "b16", "dual", "raw", "avt", "wavlm"])
    ap.add_argument("--net", default="iid", choices=NETS, help="source network: i.i.d. training (default) or the "
                    "episodic-BN control network")
    ap.add_argument("--suite", default="full", choices=["full", "lite", "paper"], help="full: all protocols, K-curve; "
                    "lite: rand / skew / dir02 / past / same + composition curve; paper: lite + accuracy vs K")
    ap.add_argument("--nctx", type=int, default=NCTX, help="independent reference draws per query and protocol")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--kctx", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--mem_frac", type=float, default=float(os.environ.get("MEM_FRAC", "0.15")))
    ap.add_argument("--folds", default=None, help="comma-separated subset of folds (smoke tests only)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--build_cache", action="store_true", help="precompute the exact composition lattices once")
    ap.add_argument("--profile", action="store_true", help="latency / memory / K-scaling of all test-time methods")
    ap.add_argument("--stream", action="store_true", help="person-blocked streaming protocol (complete stream methods)")
    ap.add_argument("--profile-k", type=int, nargs="+", default=PROFILE_K)
    ap.add_argument("--profile-methods", nargs="+", choices=METHODS, default=PROFILE_METHODS)
    a = ap.parse_args(argv)
    a.tag = (a.method + ("" if a.feat == "l14" else f"_{a.feat}") + ("" if a.net == "iid" else f"_{a.net}")
             + ("" if a.suite == "full" else f"_{a.suite}") + ("" if a.nctx == NCTX else f"_c{a.nctx}")
             + ("" if a.folds is None and a.steps == 3000 else "_smoke"))
    return a


def selftest():
    """exactness of the count DP against brute-force enumeration of all label assignments"""
    g = torch.Generator().manual_seed(0)
    for n, C in [(3, 3), (4, 3), (5, 4), (4, 6)]:
        SL = sublattice(n, C)
        H = SL["H"].long().cpu().numpy(); Hh = len(H)
        un = torch.randn(2, Hh, n, C, generator=g).to(dev)
        coef = count_dp(SL, un).cpu()
        ref = torch.full((2, Hh, C), -INF)
        for h in range(Hh):
            for ys in itertools.product(range(C), repeat=n - 1):
                cnt = np.bincount(ys, minlength=C)
                for c in range(C):
                    tgt = H[h].copy(); tgt[c] -= 1
                    if tgt[c] >= 0 and (cnt == tgt).all():
                        v = sum(un[:, h, k + 1, y_].cpu() for k, y_ in enumerate(ys))
                        ref[:, h, c] = torch.logaddexp(ref[:, h, c], v)
        err = (torch.nan_to_num(coef, neginf=0) - torch.nan_to_num(ref, neginf=0)).abs().max().item()
        assert ((coef == -INF) == (ref == -INF)).all(), "support mismatch"
        print(f"count DP n={n} C={C}: {Hh} hypotheses, max |error| = {err:.2e}")
        assert err < 1e-4
    # Prop. 8 is conditional on zero-mean, uncorrelated centre and estimation errors, with variances a and b.
    # A composition posterior covariance is not automatically the frequentist estimation-error variance b.
    # This scalar algebra check is not a classification-safety or real-data calibration guarantee.
    for a_, b_ in [(1.0, 0.1), (1.0, 1.0), (0.1, 1.0), (5.0, 0.3)]:
        for r in [0.5, 0.8, 1.0, 1.5, 2.0]:
            w = r * a_ / (r * a_ + b_); mse = (1 - w) ** 2 * a_ + w ** 2 * b_
            assert mse <= min(a_, b_) + 1e-12, (a_, b_, r, mse)
    print("scalar shrinkage model: MSE <= min(a,b) for prior scale x[1/2,2], assuming uncorrelated unbiased errors")
    # Prop. 11: the source-anchored transport with no information about the person (mu = bbar, scale 1) IS the source model
    torch.manual_seed(0)
    Sx = Source(16, 4).eval()
    with torch.no_grad():
        Sx.bn.running_mean.normal_(); Sx.bn.running_var.uniform_(0.5, 2.0); Sx.bn.weight.normal_(); Sx.bn.bias.normal_()
        x = torch.randn(9, 16)
        u = pop_units(Sx, x)
        st = dict(b=torch.randn(5, u.shape[1]))
        lg_src, lg_tp = Sx(x), head_pop(Sx, transport(st, u, bbar(st), 1.0))
    assert torch.allclose(lg_src, lg_tp, atol=1e-5), (lg_src - lg_tp).abs().max()
    print("Prop. 11: transport at the zero-information point reproduces the source model exactly")
    # Prop. 12: E[d_excess | mbar] = (kappa - 1) T (kappa T + Sigma)^-1 (mbar - bbar)  (Monte-Carlo regression check)
    g = np.random.default_rng(0)
    for kap_, t_, sg in [(1.27, 1.0, 0.8), (20.0, 1.0, 0.3), (3.0, 2.0, 2.0)]:
        d_fit = g.normal(0, np.sqrt(t_), 400000); d_exc = g.normal(0, np.sqrt((kap_ - 1) * t_), 400000)
        obs = d_fit + d_exc + g.normal(0, np.sqrt(sg), 400000)
        slope = np.cov(d_exc, obs)[0, 1] / obs.var()
        assert abs(slope - (kap_ - 1) * t_ / (kap_ * t_ + sg)) < 0.01, (kap_, slope)
    print("appendix: posterior mean of the unfamiliar part of the person deviation verified")
    # stacking: a proper distribution, and the family contains pure Bayes / pure conservative / pure source
    g_ = torch.Generator().manual_seed(1)
    Bq, Hq, Cq = 5, 12, 4
    thq = torch.tensor(compositions(3, Cq)[:Hq], dtype=torch.float32) / 3
    pNq = torch.rand(Bq, Hq, generator=g_).softmax(-1)
    pcq = torch.rand(Bq, Hq, Cq, generator=g_).softmax(-1)
    peq = torch.rand(Bq, Cq, generator=g_).softmax(-1); psq = torch.rand(Bq, Cq, generator=g_).softmax(-1)
    pbq = torch.rand(Bq, Cq, generator=g_).softmax(-1)
    SIq = stack_inputs(pNq, pcq, thq, Cq, peq, pbq, psq, torch.rand(Bq), torch.rand(Bq))
    joint = (pNq[..., None] * pcq).sum(1)
    lo, hi = -60.0, 60.0
    for theta, ref in (([lo, 0, lo, 0, lo, 0], joint), ([hi, 0, lo, 0, lo, 0], peq), ([lo, 0, hi, 0, lo, 0], pbq),
                       ([lo, 0, lo, 0, hi, 0], psq)):
        q_ = stack_combine(SIq, "stack", dict(theta=theta))
        assert torch.allclose(q_, ref, atol=1e-5) and torch.allclose(q_.sum(-1), torch.ones(Bq), atol=1e-6)
    for kind in STACK_KINDS[1:]:  # every control combiner is a distribution at random parameters
        par_ = dict(theta=torch.randn(n_stack_params(kind), generator=g_).tolist())
        if kind.startswith("gate"):
            par_.update(mu=[0.0] * len(GATE_FEATS[kind]), sd=[1.0] * len(GATE_FEATS[kind]))
        q_ = stack_combine(SIq, kind, par_)
        assert torch.isfinite(q_).all() and torch.allclose(q_.sum(-1), torch.ones(Bq), atol=1e-5), kind
    print("stacking: proper distribution; nests the exact Bayes, conservative, batch-moment and source predictors")
    print("selftest passed")


if __name__ == "__main__":
    a = get_args()
    if a.selftest:
        selftest(); raise SystemExit
    if a.build_cache:
        for C_ in (6, 7, 8):
            for k in sorted(set([0] + KCURVE + PROFILE_K + [a.kctx])):
                t = time.time(); SL = sublattice(k + 1, C_) if k else None
                if SL: print(f"lattice n={k + 1} C={C_}: {SL['H'].shape[0]} compositions, {SL['tot']} sub-compositions ({time.time() - t:.1f}s)")
        raise SystemExit
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.mem_frac)  # WDDM: exceeding memory pages instead of failing
    os.makedirs(f"{OUT}/pred", exist_ok=True)
    if a.profile:
        profile(a); raise SystemExit
    if a.stream:
        run_stream(a); raise SystemExit
    t0 = time.time(); res = run(a); res["minutes"] = (time.time() - t0) / 60
    fn = f"{OUT}/{a.data}_{a.tag}_s{a.seed}.json"
    with open(fn + '.tmp', 'w') as fo: json.dump(res, fo, allow_nan=False)
    os.replace(fn + '.tmp', fn)
    print("RESULT", a.data, a.tag, a.seed, json.dumps({k: round(v["acc"], 2) for k, v in res["mean"].items()}), flush=True)
