"""COMPASS: composition-uncertainty-propagating subject inference for subject-relative FER, and published baselines.

Model (one subject, learned embedding z = phi(x)):
    mu ~ N(0, diag(t2)),   z_i = mu + F_{y_i} + eps_i,   eps_i ~ N(0, s2 I),   y_i ~ Cat(pi)
Inference from K unlabeled references R of the query's subject:
  1. Anchor. Each reference's expression posterior is computed from its own appearance with mu marginalised under the
     population prior,  a_k = p(y_k | z_k) ∝ pi_c N(z_k ; F_c, s2 + t2).  This quantity does not change when the whole
     set is shifted by a subject offset, so it is not subject to the shift ambiguity that defeats shift-equivariant
     estimators such as centering (Proposition 1 of the paper).
  2. Subject posterior. Given labels, mu | R, y is Gaussian; with y_k ~ a_k independently, its exact first two moments are
        m = v / s2 * sum_k (z_k - a_k F),   v = (1/t2 + K/s2)^-1
        Cov = diag(v) + (v/s2)^2 * F^T A F,   A = sum_k (diag(a_k) - a_k a_k^T)        (law of total variance)
     The second term is the composition uncertainty: it lies in the span of the expression displacements and is large
     exactly when the references' expressions are ambiguous.
  3. Query. p(y_q = c | z_q, R) ∝ pi_c N(z_q ; m + F_c, diag(s2 + v) + (v/s2)^2 F^T A F)   (Woodbury, rank <= C - 1).
     With K = 0, or when every reference is ambiguous along an expression direction, this reduces to the population
     (appearance-only) classifier along that direction.
Training: cross-entropy on the query predictive and on every reference's anchor posterior, over episodes whose reference
composition is drawn from Dir(1) independently of the query label (composition-randomised training).

Published baselines, run on the same features, folds, references and evaluation contexts (each test episode = query +
its K references, processed as one test batch, model reset after every episode):
  psn      person-specific normalisation, median subtraction (Baltrusaitis et al., FG 2015)
  source   source model (MLP with BatchNorm, trained i.i.d.), no adaptation
  bn       test-time BN statistics (Schneider et al., NeurIPS 2020)
  tent     entropy minimisation of BN affine parameters (Wang et al., ICLR 2021), Adam lr 1e-3
  shot     information maximisation + clustering pseudo-labels on the feature extractor (Liang et al., ICML 2020)
  t3a      test-time template adjuster (Iwasawa & Matsuo, NeurIPS 2021)
  lame     Laplacian adjusted maximum-likelihood estimation of the outputs (Boudiaf et al., CVPR 2022)
  sar      sharpness-aware reliable entropy minimisation (Niu et al., ICLR 2023)
usage: python compass.py --data ravdess --method compass --seed 0  -> $EMO_ROOT/runs/compass/*.json
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, json, math, os, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from came import PROC, LAB, contexts, metrics, prototypes

FEATD = f"{_data_root}/features/clip"


def load(data, fold, feat="dual"):
    """dual: frozen CLIP-B/16 + per-fold fine-tuned CLIP-B/16 (RAVDESS, CREMA-D); b16 / l14: frozen CLIP-B/16 / -L/14"""
    if feat == "dual":
        return np.concatenate([np.load(f"{FEATD}/{data}_frozen.npy"), np.load(f"{FEATD}/{data}_ft_f{fold}.npy")], 1)
    return np.load(f"{FEATD}/{data}_{feat}.npy")

OUT = os.environ.get("COMPASS_OUT", f"{_data_root}/runs/compass")
dev = "cuda"


# --------------------------------------------------------------------------------------------- COMPASS
class COMPASS(nn.Module):
    def __init__(self, D, C, d=256, variant="full", proto=None):
        super().__init__()
        lin = variant == "lin"
        self.phi = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, d)) if lin else \
            nn.Sequential(nn.Dropout(0.1), nn.Linear(D, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, d))
        if proto is None:
            self.F_free = nn.Parameter(torch.randn(C, d) * 0.1)
        else:  # geometry-locked language prototypes: F = s G Q^T, G G^T = T T^T, Q orthonormal
            U, S, _ = torch.linalg.svd(proto, full_matrices=False)
            self.register_buffer("G", (U * S)[:, : C - 1])
            self.Q = nn.utils.parametrizations.orthogonal(nn.Linear(C - 1, d, bias=False))
            self.log_scale = nn.Parameter(torch.zeros(()))
        self.log_s2 = nn.Parameter(torch.zeros(()))
        self.log_t2 = nn.Parameter(torch.zeros(d))
        self.prior = nn.Parameter(torch.zeros(C))
        self.variant, self.C = variant, C

    @property
    def Fm(self):
        f = self.log_scale.exp() * self.Q(self.G) if hasattr(self, "G") else self.F_free
        return f - f.mean(0)

    def anchor(self, z):  # p(y | z) with mu marginalised under the population prior
        var = self.log_s2.exp() + self.log_t2.exp()
        d2 = ((z[..., None, :] - self.Fm) ** 2 / var).sum(-1)
        return self.prior.log_softmax(0) - 0.5 * d2

    def forward(self, x, yr=None):
        z = self.phi(x)
        B, n, d = z.shape
        la = self.anchor(z).log_softmax(-1)  # (B, n, C)
        K = n - 1
        if K == 0 or self.variant == "nosubj":
            return la
        s2, t2, Fm = self.log_s2.exp(), self.log_t2.exp(), self.Fm
        zr, a = z[:, 1:], la[:, 1:].exp()
        if self.variant == "vb":  # mean-field refinement of the reference labels (iterated, self-reinforcing)
            for _ in range(10):
                v = 1 / (1 / t2 + K / s2)
                m = v / s2 * (zr - a @ Fm).sum(1)
                a = (self.prior.log_softmax(0) - 0.5 * ((zr[..., None, :] - m[:, None, None] - Fm) ** 2 / s2).sum(-1)).softmax(-1)
        v = 1 / (1 / t2 + K / s2)  # (d,)
        m = v / s2 * (zr - a @ Fm).sum(1)  # (B, d)
        r = z[:, :1, None, :] - m[:, None, None] - Fm  # (B, 1, C, d)
        D = s2 + v
        quad = (r ** 2 / D).sum(-1)  # (B, 1, C)
        if self.variant != "noprop":
            A = torch.diag_embed(a.sum(1)) - a.transpose(1, 2) @ a  # (B, C, C), PSD
            ev, U = torch.linalg.eigh(A)
            Ah = U * ev.clamp_min(0).sqrt()[:, None]  # A = Ah Ah^T
            L = (v / s2)[None, :, None] * (Fm.T[None] @ Ah)  # (B, d, C): composition covariance = L L^T
            u = torch.einsum("bqcd,bdk->bqck", r / D, L)  # L^T D^-1 r
            Mi = torch.eye(self.C, device=z.device) + L.transpose(1, 2) @ (L / D[None, :, None])
            quad = quad - (u * torch.linalg.solve(Mi[:, None], u.transpose(-1, -2)).transpose(-1, -2)).sum(-1)
        lq = self.prior.log_softmax(0) - 0.5 * quad
        return torch.cat([lq, la[:, 1:]], 1)


class CASS(nn.Module):
    """Composition-Aware Subject Statistics network:  Linear -> CASS-Norm -> GELU -> Linear.
    CASS-Norm normalises the hidden units of every member of an episode by the statistics of the query's SUBJECT, which
    it infers from the references while explaining away the expressions they contain:
      u_k  = (h_k - m_pop) / s_pop                               population-standardised hidden units
      a_k  = p_pop(y | x_k)                                       the network's own appearance-only posterior
                                                                  (the same layers with population statistics)
      e_k  = u_k - a_k F                                          composition-explained residual (F: class displacements)
      w_k  = 1 / (sigma_w^2 + tr Var_{a_k}(F) / H)                inverse-variance weight: ambiguous references count less
      mu   = sum_k w_k e_k / (sum_k w_k + lambda_m)               subject mean, shrunk to the population
      s^2  = (sum_k w_k (e_k - mu)^2 + lambda_v) / (sum_k w_k + lambda_v)    subject scale, shrunk to 1
      out  = gamma * (u - mu) / s + beta
    With no references (or lambda -> inf) it reduces to ordinary BatchNorm with population statistics."""

    def __init__(self, D, C, H=512, variant="full"):
        super().__init__()
        self.drop, self.f1, self.cls = nn.Dropout(0.1), nn.Linear(D, H), nn.Linear(H, C)
        self.gamma, self.beta = nn.Parameter(torch.ones(H)), nn.Parameter(torch.zeros(H))
        self.register_buffer("rm", torch.zeros(H)); self.register_buffer("rv", torch.ones(H))
        self.Fd = nn.Parameter(torch.zeros(C, H))
        self.log_sw, self.log_lm, self.log_lv = nn.Parameter(torch.zeros(())), nn.Parameter(torch.zeros(H)), nn.Parameter(torch.zeros(()))
        self.variant = variant

    def head(self, u):
        return self.cls(self.drop(F.gelu(self.gamma * u + self.beta)))

    def forward(self, x, yr=None, both=False):
        h = self.f1(self.drop(x))
        if self.training:  # population statistics over all members of all episodes in the batch (as BatchNorm)
            m, v = h.flatten(0, 1).mean(0), h.flatten(0, 1).var(0, unbiased=False)
            with torch.no_grad():
                self.rm.mul_(0.9).add_(0.1 * m); self.rv.mul_(0.9).add_(0.1 * v)
        else:
            m, v = self.rm, self.rv
        u = (h - m) / torch.sqrt(v + 1e-5)
        lp = self.head(u)  # population (appearance-only) path
        K = x.shape[1] - 1
        if K == 0:
            return (lp, lp) if both else lp
        joint = self.variant.endswith("joint")  # subject statistics from every unlabeled clip, query included
        ur = u if joint else u[:, 1:]
        if self.variant.startswith("nocomp"):  # no composition explaining-away: plain shrunk subject statistics
            e, w = ur, torch.ones_like(ur[..., :1])
        else:
            a = (lp if joint else lp[:, 1:]).softmax(-1)
            Fd = self.Fd - self.Fd.mean(0)
            e = ur - a @ Fd
            if self.variant.startswith("now"):
                w = torch.ones_like(ur[..., :1])
            else:
                varF = (a @ Fd ** 2 - (a @ Fd) ** 2).mean(-1, keepdim=True)
                w = 1 / (self.log_sw.exp() + varF)
        mu = (w * e).sum(1, keepdim=True) / (w.sum(1, keepdim=True) + self.log_lm.exp())
        lv = self.log_lv.exp()
        s2 = ((w * (e - mu) ** 2).sum(1, keepdim=True) + lv) / (w.sum(1, keepdim=True) + lv)
        ls = self.head((u - mu) / torch.sqrt(s2))
        return (ls, lp) if both else ls


class PSN(nn.Module):  # person-specific normalisation: median of the person's references subtracted
    def __init__(self, D, C):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, C))

    def forward(self, x, yr=None):
        med = x[:, 1:].median(1, keepdim=True).values if x.shape[1] > 1 else torch.zeros_like(x[:, :1])
        return self.net(x - med)


# --------------------------------------------------------------------------------------------- source + TTA
class Source(nn.Module):
    """netF (bottleneck): Linear -> BatchNorm -> GELU;  netC: Linear.  Trained on i.i.d. labeled clips."""

    def __init__(self, D, C, h=512):
        super().__init__()
        self.drop, self.f1, self.bn, self.cls = nn.Dropout(0.1), nn.Linear(D, h), nn.BatchNorm1d(h), nn.Linear(h, C)

    def forward(self, x):
        f = F.gelu(self.bn(self.f1(self.drop(x))))
        return self.cls(self.drop(f))


def fwd(S, P, x, mode):
    """functional forward with per-episode parameters P (each (B, ...)); mode: 'eval' running stats | 'batch' episode stats"""
    h = torch.einsum("bnd,bdh->bnh", x, P["W1"]) + P["b1"][:, None] if P["W1"].dim() == 3 else x @ P["W1"] + P["b1"]
    if mode == "eval":
        mu, var = P.get("rm", S.bn.running_mean), P.get("rv", S.bn.running_var)
        mu, var = (mu[:, None], var[:, None]) if mu.dim() == 2 else (mu, var)
    elif mode == "refs":
        mu, var = h[:, 1:].mean(1, keepdim=True), h[:, 1:].var(1, unbiased=False, keepdim=True)
    else:
        mu, var = h.mean(1, keepdim=True), h.var(1, unbiased=False, keepdim=True)
    f = F.gelu((h - mu) / torch.sqrt(var + S.bn.eps) * P["g"][:, None] + P["be"][:, None])
    return f @ S.cls.weight.T + S.cls.bias, f, (h.mean(1), h.var(1, unbiased=True))


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


def tta(S, x, method, C, yr=None):
    """x: (B, n, D) episodes; returns query probabilities (B, C)"""
    B, n, _ = x.shape
    if n == 1:  # no references: a single test sample gives no batch to adapt to, every method returns the source model
        method = "source"
    if method in ("source", "bn", "bnref"):
        with torch.no_grad():
            return fwd(S, episode_params(S, B), x, {"source": "eval", "bn": "batch", "bnref": "refs"}[method])[0][:, 0].softmax(-1)
    if method == "tent":  # official: Adam(lr 1e-3, beta 0.9), BN in batch mode, affine parameters only
        P = episode_params(S, B); prm = [P["g"].requires_grad_(), P["be"].requires_grad_()]
        opt = torch.optim.Adam(prm, lr=1e-3, betas=(0.9, 0.999))
        for _ in range(STEPS):
            with torch.enable_grad():
                loss = ent(fwd(S, P, x, "batch")[0]).mean(1).sum()
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            return fwd(S, P, x, "batch")[0][:, 0].softmax(-1)
    if method == "sar":  # official: SAM(SGD, momentum 0.9, rho 0.05), lr (0.00025/64)*bs*2 for bs < 32, E0 = 0.4 ln C
        P = episode_params(S, B); prm = [P["g"].requires_grad_(), P["be"].requires_grad_()]
        lr, rho, e0 = 0.00025 / 64 * n * 2, 0.05, 0.4 * math.log(C)
        opt = torch.optim.SGD(prm, lr=lr, momentum=0.9)
        for _ in range(STEPS):
            with torch.enable_grad():
                e = ent(fwd(S, P, x, "batch")[0]); w1 = (e < e0).float()
                ((e * w1).sum(1) / w1.sum(1).clamp_min(1)).sum().backward()
                gn = torch.sqrt(sum((p.grad ** 2).sum(-1) for p in prm)) + 1e-12  # per-episode gradient norm
                eps = [rho * p.grad / gn[:, None] for p in prm]
                with torch.no_grad():
                    for p, ep in zip(prm, eps): p.add_(ep)
                opt.zero_grad()
                e2 = ent(fwd(S, P, x, "batch")[0]); w2 = w1 * (e2 < e0).float()
                ((e2 * w2).sum(1) / w2.sum(1).clamp_min(1)).sum().backward()
                with torch.no_grad():
                    for p, ep in zip(prm, eps): p.sub_(ep)
                opt.step(); opt.zero_grad()
        with torch.no_grad():
            return fwd(S, P, x, "batch")[0][:, 0].softmax(-1)
    if method == "delta":  # official (entw+delta): TBR (prior 0.95, initialised from the first batch), entropy
        P = episode_params(S, B); prm = [P["g"].requires_grad_(), P["be"].requires_grad_()]  # weighting, DOT 0.95,
        opt = torch.optim.SGD(prm, lr=0.00025, momentum=0.9)  # SGD lr 0.00025 momentum 0.9
        qhat = torch.full((B, C), 1.0 / C, device=x.device)
        for _ in range(STEPS):
            with torch.enable_grad():  # TBR forward value equals the batch statistics of a fixed batch
                lg = fwd(S, P, x, "batch")[0]
                p = lg.softmax(-1); e = ent(lg); pls = p.argmax(-1)
                ew = torch.exp(math.log(C) * 0.5 - e.detach()); use = (ew > 1).float()
                cw = 1 / qhat; cw = cw / cw.sum(-1, keepdim=True)
                sw = torch.gather(cw, 1, pls); sw = sw / sw.sum(1, keepdim=True) * n
                w = ew * sw * use
                ((e * w).sum(1) / use.sum(1).clamp_min(1)).sum().backward()
                opt.step(); opt.zero_grad()
                qhat = 0.95 * qhat + 0.05 * p.detach().mean(1)
        with torch.no_grad():
            return fwd(S, P, x, "batch")[0][:, 0].softmax(-1)
    if method == "shot":  # official: SGD(lr 0.01, momentum 0.9, nesterov, wd 1e-3) on the bottleneck, classifier frozen,
        P = episode_params(S, B, full=True)  # loss = ent - div + 0.3 * CE(cluster pseudo-labels), 15 iterations
        prm = [P[k].requires_grad_() for k in ["W1", "b1", "g", "be"]]
        opt = torch.optim.SGD(prm, lr=0.01, momentum=0.9, nesterov=True, weight_decay=1e-3)
        for _ in range(15):
            with torch.no_grad():  # pseudo-labels: 2 rounds of weighted-centroid cosine clustering (obtain_label)
                lg0, f0, _ = fwd(S, P, x, "eval")
                fe = torch.cat([f0, torch.ones_like(f0[..., :1])], -1); fe = fe / fe.norm(dim=-1, keepdim=True)
                aff = lg0.softmax(-1); pred = aff.argmax(-1)
                for _ in range(2):
                    cen = aff.transpose(1, 2) @ fe / (1e-8 + aff.sum(1)[..., None])
                    cen = cen / cen.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    present = F.one_hot(pred, C).sum(1) > 0
                    dist = 1 - fe @ cen.transpose(1, 2)
                    pred = dist.masked_fill(~present[:, None], 1e9).argmin(-1)
                    aff = F.one_hot(pred, C).float()
            with torch.enable_grad():
                lg, _, (bm, bv) = fwd(S, P, x, "batch")
                p = lg.softmax(-1)
                loss = 0.3 * F.cross_entropy(lg.flatten(0, 1), pred.flatten(), reduction="none").view(B, n).mean(1)
                msm = p.mean(1)
                loss = loss + ent(lg).mean(1) + (msm * torch.log(msm + 1e-5)).sum(-1)
                opt.zero_grad(); loss.sum().backward(); opt.step()
            with torch.no_grad():  # BatchNorm running statistics follow the target batches (momentum 0.1)
                P["rm"].mul_(0.9).add_(0.1 * bm.detach()); P["rv"].mul_(0.9).add_(0.1 * bv.detach())
        with torch.no_grad():
            return fwd(S, P, x, "eval")[0][:, 0].softmax(-1)
    if method.startswith("compass_bma"):  # ours: Bayesian model averaging over the composition regime of the set
        with torch.no_grad():
            p_div = tta(S, x, "compass_nocomp" if method.startswith("compass_bmanc") else "bn", C)  # diverse repertoire
            p_con = tta(S, x, "compass_em", C)  # composition explained away
            h = x @ S.f1.weight.T + S.f1.bias
            u = (h - S.bn.running_mean) / torch.sqrt(S.bn.running_var + S.bn.eps)
            r = u - u.mean(1, keepdim=True)  # centred set: invariant to the subject offset
            z = r @ S.Qf  # coordinates in the expression subspace span(F), (B, n, C-1)
            rhos = torch.linspace(0, 1, 11, device=x.device) if method.endswith("c") else torch.tensor([1.0, 0.0], device=x.device)
            ll = []
            for rho in rhos:  # Gaussian evidence of the centred set, covariance Sigma_w + rho * Sigma_F
                Sig = S.Sig_con + rho * (S.Sig_div - S.Sig_con)
                Li = torch.linalg.cholesky(Sig)
                sol = torch.cholesky_solve(z.transpose(1, 2), Li)  # (B, C-1, n)
                quad = (z.transpose(1, 2) * sol).sum((1, 2))
                ll.append(-0.5 * (quad * n / (n - 1) + (n - 1) * 2 * torch.log(torch.diagonal(Li)).sum()))
            post = torch.stack(ll, 1).softmax(1)  # uniform prior over the diversity rho
            pi_div = (post * rhos).sum(1, keepdim=True)  # posterior mean diversity
            return pi_div * p_div + (1 - pi_div) * p_con
    if method.startswith("compass"):  # ours: composition-aware subject statistics in the source's BatchNorm layer
        with torch.no_grad():
            lg0, _, _ = fwd(S, episode_params(S, B), x, "eval")
            h = x @ S.f1.weight.T + S.f1.bias
            u = (h - S.bn.running_mean) / torch.sqrt(S.bn.running_var + S.bn.eps)  # population-standardised units
            Fc, sig2, tau2 = S.Fc, S.sig2, S.tau2
            a = lg0.softmax(-1)
            if method == "compass_oracle":  # references' TRUE labels (the query's label is still inferred)
                a = torch.cat([a[:, :1], F.one_hot(yr, C).float()], 1)
            alpha = float(method.split("_a")[1]) if "_a" in method else 0.0  # Dirichlet composition prior (pseudo-counts)
            rho = alpha / (alpha + n)
            for it in range(10 if method.startswith("compass_em") else 1):
                if method == "compass_nocomp":  # no explaining away (shrunk subject statistics only)
                    e, w = u, (1 / sig2).expand_as(u)
                else:
                    a = (1 - rho) * a + rho / a.shape[-1]  # posterior-mean composition under Dir(alpha/C)
                    e = u - a @ Fc  # composition-explained residuals
                    varF = a @ Fc ** 2 - (a @ Fc) ** 2  # label-uncertainty variance of the explained displacement
                    w = 1 / (sig2 + varF) if method != "compass_now" else (1 / sig2).expand_as(u)
                mu = (w * e).sum(1, keepdim=True) / (w.sum(1, keepdim=True) + 1 / tau2)  # Gaussian posterior mean
                wn = w / w.sum(1, keepdim=True)
                if method == "compass_nocomp":
                    s2 = (wn * (e - mu) ** 2).sum(1, keepdim=True)
                else:  # within-expression variance (label-uncertainty part removed) + balanced-repertoire variance
                    s2 = (wn * ((e - mu) ** 2 - varF)).sum(1, keepdim=True).clamp_min(1e-3) + (Fc ** 2).mean(0)
                s2 = (s2 * (n - 1) + sig2 + (Fc ** 2).mean(0)) / n  # one prior pseudo-observation of the population value
                un = (u - mu) / torch.sqrt(s2)
                lg = F.gelu(un * S.bn.weight + S.bn.bias) @ S.cls.weight.T + S.cls.bias
                a = lg.softmax(-1) if method != "compass_oracle" else torch.cat([lg[:, :1].softmax(-1), a[:, 1:]], 1)
            return lg[:, 0].softmax(-1)
    if method == "t3a":  # official: supports initialised with classifier weights, entropy filter filter_K per class
        with torch.no_grad():
            Wc = S.cls.weight; wl = Wc @ Wc.T + S.cls.bias
            _, f, _ = fwd(S, episode_params(S, B), x, "eval")
            lg = f @ Wc.T + S.cls.bias
            sup = torch.cat([Wc.expand(B, -1, -1), f], 1)
            lab = torch.cat([wl.argmax(-1).expand(B, -1), lg.argmax(-1)], 1)
            e = torch.cat([ent(wl).expand(B, -1), ent(lg)], 1)
            keep = torch.zeros_like(e, dtype=torch.bool)
            for c in range(C):  # lowest-entropy filter_K supports of each pseudo-class
                ec = e.masked_fill(lab != c, float("inf"))
                k = min(FILTER_K, ec.shape[1])
                idx = ec.topk(k, largest=False).indices
                keep |= torch.zeros_like(keep).scatter_(1, idx, torch.gather(lab == c, 1, idx))
            sup = F.normalize(sup, dim=-1) * keep[..., None]
            Wt = F.normalize(sup.transpose(1, 2) @ F.one_hot(lab, C).float(), dim=1)  # (B, h, C)
            return (f[:, :1] @ Wt)[:, 0].softmax(-1)
    if method == "lame":  # official: rbf affinity, knn 5, sigma = mean k-th neighbour distance, lambda 1, 100 steps
        with torch.no_grad():
            lg, f, _ = fwd(S, episode_params(S, B), x, "eval")
            unary = -torch.log(lg.softmax(-1) + 1e-10)
            fe = F.normalize(f, dim=-1)
            dist = torch.cdist(fe, fe)
            kth = dist.topk(min(5, n), dim=-1, largest=False).values[..., -1]
            sig = kth.mean(1)[:, None, None]
            Wk = torch.exp(-dist ** 2 / (2 * sig ** 2))
            Y = (-unary).softmax(-1)
            for _ in range(100):
                Y = (-unary + Wk @ Y).softmax(-1)
            return Y[:, 0]
    raise ValueError(method)


STEPS, FILTER_K = 10, 20
TTA = ["compass_bmabn", "compass_bmabnc", "compass_bmancc", "compass_oracle", "compass_a7", "compass_a21", "compass_a63", "compass_em_a7", "compass_em_a21", "compass", "compass_em", "compass_nocomp", "compass_now", "source", "bnref", "bn", "delta", "tent", "shot", "t3a", "lame", "sar"]


# --------------------------------------------------------------------------------------------- training
def train_episodic(model, X, meta, rows, a, rng):
    y = torch.tensor(meta.label.values, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    for it in range(a.steps):
        q = rng.choice(rows, a.bs)
        K = int(rng.integers(0, a.kmax + 1))
        idx = np.concatenate([q[:, None], contexts(meta, q, K, a.comp, rng, rows)], 1) if K else q[:, None]
        idx = torch.from_numpy(idx).to(dev)
        if isinstance(model, CASS):
            ls, lp = model(X[idx], both=True)
            loss = F.cross_entropy(ls.flatten(0, 1), y[idx].flatten(), label_smoothing=0.1) +                 F.cross_entropy(lp.flatten(0, 1), y[idx].flatten(), label_smoothing=0.1)
        else:
            lg = model(X[idx])
            loss = F.cross_entropy(lg.flatten(0, 1), y[idx].flatten(), label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model


@torch.no_grad()
def class_statistics(S, X, meta, rows):
    """closed-form statistics of the training subjects in the source's population-standardised BN units:
    Fc (expression displacements within subject), sig2 (within-subject residual variance), tau2 (between-subject
    variance of balanced subject means)"""
    u = (S.f1(X[rows]) - S.bn.running_mean) / torch.sqrt(S.bn.running_var + S.bn.eps)
    act, y = meta.actor.values[rows], meta.label.values[rows]
    C = meta.label.nunique(); subs = np.unique(act)
    M = torch.zeros(len(subs), C, u.shape[1], device=u.device); has = torch.zeros(len(subs), C, device=u.device)
    for i, s_ in enumerate(subs):
        for c in range(C):
            msk = torch.from_numpy((act == s_) & (y == c)).to(u.device)
            if msk.any():
                M[i, c] = u[msk].mean(0); has[i, c] = 1
    b = (M * has[..., None]).sum(1) / has.sum(1, keepdim=True)  # balanced subject means
    D = (M - b[:, None]) * has[..., None]
    Fc = D.sum(0) / has.sum(0)[:, None]; Fc = Fc - Fc.mean(0)
    sidx = torch.from_numpy(np.searchsorted(subs, act)).to(u.device)
    res = u - b[sidx] - Fc[torch.from_numpy(y).to(u.device)]
    S.Fc, S.sig2, S.tau2 = Fc, res.var(0), b.var(0).clamp_min(1e-4)
    # expression subspace and the two composition regimes, in its coordinates
    Q, _ = torch.linalg.qr(Fc.T)  # (H, C) -> keep C-1 directions (Fc is centred, rank C-1)
    S.Qf = Q[:, : C - 1]
    zr = res @ S.Qf  # within-subject, within-expression residuals in the expression subspace (full covariance)
    noise = zr.T @ zr / len(zr)
    SigF = S.Qf.T @ (Fc.T @ Fc / C) @ S.Qf
    S.Sig_div, S.Sig_con = noise + SigF, noise
    return S


def train_source(model, X, meta, rows, a, rng):
    y = torch.tensor(meta.label.values, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    for it in range(a.steps):  # same number of labeled clip presentations as the episodic methods (64 x 5 on average)
        idx = torch.from_numpy(rng.choice(rows, a.bs * 5)).to(dev)
        loss = F.cross_entropy(model(X[idx]), y[idx], label_smoothing=0.1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return model.eval()


def predict(a, model, X, ctx, rows, C):
    q = torch.from_numpy(rows).to(dev)[:, None]
    idx = q if ctx is None else torch.cat([q, torch.from_numpy(ctx).to(dev)], 1)
    out = []
    ch = 64 if a.method == "shot" else 512
    for i in range(0, len(idx), ch):
        xb = X[idx[i:i + ch]]
        if a.method in TTA:
            out.append(tta(model, xb, a.method, C, YALL[idx[i:i + ch, 1:]]).float())
        else:
            with torch.no_grad():
                out.append(model(xb)[:, 0].float().softmax(-1))
    return torch.cat(out).cpu().numpy()


MAIN = ["rand", "xsent", "contam", "same", "other", "skew"]
NCTX = 3


def run(a):
    meta = pd.read_csv(f"{PROC}/{a.data}_meta.csv"); meta.attrs["data"] = a.data
    C = meta.label.nunique()
    global MAIN
    MAIN = ["rand", "xsent", "contam", "same", "other", "skew"] + (["past"] if "order" in meta else [])
    res = dict(args=vars(a), folds={})
    for fold in sorted(meta.fold.unique()):
        torch.manual_seed(a.seed * 100 + fold); rng = np.random.default_rng(a.seed * 100 + fold)
        X = load(a.data, fold, a.feat)
        tr, te = np.where(meta.fold != fold)[0], np.where(meta.fold == fold)[0]
        X = torch.tensor((X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6), dtype=torch.float32, device=dev)
        D = X.shape[1]
        if a.method in TTA:
            model = class_statistics(train_source(Source(D, C).to(dev), X, meta, tr, a, rng), X, meta, tr)
        elif a.method.startswith("cass"):
            var = {"cass_nocomp": "nocomp", "cass_now": "now", "cass_joint": "joint", "cass_nocomp_joint": "nocomp_joint",
                   "cass_now_joint": "now_joint"}.get(a.method, "full")
            model = train_episodic(CASS(D, C, variant=var).to(dev), X, meta, tr, a, rng).eval()
        elif a.method == "psn":
            model = train_episodic(PSN(D, C).to(dev), X, meta, tr, a, rng).eval()
        else:
            var = {"compass_noprop": "noprop", "compass_vb": "vb", "compass_nosubj": "nosubj", "compass_lin": "lin"}.get(a.method, "full")
            pr = prototypes(a.data, "llm", a.seed) if a.method == "compass_llm" else None
            model = train_episodic(COMPASS(D, C, variant=var, proto=pr).to(dev), X, meta, tr, a, rng).eval()
        global YALL
        YALL = torch.tensor(meta.label.values, device=dev)
        yt = meta.label.values[te]
        P = [predict(a, model, X, None, te, C)]
        r = {"K0": metrics(P[0], yt, C)}
        for mode in MAIN + [f"mix{j}" for j in range(0, a.kctx + 1)]:
            ps = []
            for c_seed in range(NCTX):
                erng = np.random.default_rng(10000 * c_seed + 1000 + fold)  # identical for every method and seed
                md, j = ("mix", int(mode[3:])) if mode.startswith("mix") else (mode, 0)
                ps.append(predict(a, model, X, contexts(meta, te, a.kctx, md, erng, te, j), te, C))
            if mode in MAIN:
                P += ps
            ms = [metrics(p, yt, C) for p in ps]
            r[mode] = {k: float(np.mean([m[k] for m in ms])) for k in ["acc", "uar", "nll", "ece"]}
            r[mode]["acc_ctxsd"] = float(np.std([m["acc"] for m in ms]))
        for k in [1, 2, 4, 8, 16]:
            r[f"K{k}"] = metrics(predict(a, model, X, contexts(meta, te, k, "rand", np.random.default_rng(1000 + fold), te), te, C), yt, C)
        res["folds"][int(fold)] = r
        np.save(f"{OUT}/pred/{a.data}_{a.tag}_s{a.seed}_f{fold}.npy", np.stack(P).astype(np.float16))
        print(fold, {k: round(v["acc"], 2) for k, v in r.items() if "acc" in v}, flush=True)
    f0 = next(iter(res["folds"].values()))
    res["mean"] = {k: {m: float(np.mean([f[k][m] for f in res["folds"].values()])) for m in f0[k]} for k in f0}
    return res


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ravdess")
    ap.add_argument("--method", default="compass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--comp", default="dir", choices=["rand", "dir"])
    ap.add_argument("--feat", default="dual", choices=["dual", "b16", "l14"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--kctx", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    a = ap.parse_args(argv)
    a.tag = (a.method if a.method in TTA else f"{a.method}_{a.comp}") + ("" if a.feat == "dual" else f"_{a.feat}")
    return a


if __name__ == "__main__":
    a = get_args()
    os.makedirs(f"{OUT}/pred", exist_ok=True)
    t0 = time.time(); res = run(a); res["minutes"] = (time.time() - t0) / 60
    json.dump(res, open(f"{OUT}/{a.data}_{a.tag}_s{a.seed}.json", "w"))
    print("RESULT", a.data, a.tag, a.seed, json.dumps({k: round(v["acc"], 2) for k, v in res["mean"].items() if "acc" in v}), flush=True)
