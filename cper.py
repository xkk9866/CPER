"""Composition-posterior normalization for personal feature vectors."""
import itertools
import math
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

INF = float("inf")

def compositions(n, C):
    """all count vectors of n items over C classes, (S, C) int64 (stars and bars)"""
    out = []
    for bars in itertools.combinations(range(n + C - 1), C - 1):
        b = (-1,) + bars + (n + C - 1,)
        out.append([b[i + 1] - b[i] - 1 for i in range(C)])
    return np.array(out, dtype=np.int64).reshape(-1, C)


@lru_cache(maxsize=4)
def sublattice(n, C, dev):
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
    lvl = vec.sum(1)
    out = dict(H=torch.tensor(H, dtype=torch.float32, device=dev), hyp=torch.tensor(hyp, device=dev),
               par=torch.tensor(par, device=dev), q=torch.tensor(q, device=dev), tot=len(vec),
               levels=[torch.tensor(np.where(lvl == k)[0], device=dev) for k in range(n)])
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


@torch.no_grad()
def population_stats(S, X, labels, subjects):
    """Estimate population statistics from labeled float32 source embeddings."""
    dev = X.device
    C = S.cls.out_features
    act = np.asarray(subjects)
    y = np.asarray(labels, dtype=np.int64)
    if len(X) != len(y) or len(X) != len(act) or not np.array_equal(np.unique(y), np.arange(C)):
        raise ValueError("Provide matching rows and labels covering classes 0 through C-1.")
    u = pop_units(S, X)
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
    T = torch.cov(bz.T).reshape(C - 1, C - 1) if len(subs) > 1 else torch.eye(C - 1, device=dev)
    pi = torch.bincount(yt, minlength=C).float(); pi = pi / pi.sum()
    # per-unit variance components: clips of one person and one class share the interaction G_sc
    r_in = u - M.view(-1, u.shape[1])[cell]
    sin2 = (r_in ** 2 * corr[:, None]).mean(0)
    Gu = (M - b[:, None] - Fc[None])[has.bool()]  # (cells, H); sums to ~0 over the classes of a subject
    sg2 = (Gu ** 2).mean(0) * C / (C - 1) - sin2 * (1 / ncell).mean()
    sg2 = sg2.clamp_min(0.05 * sin2)
    return dict(b=b, subs=subs, Fc=Fc, sig2=res.var(0), sin2=sin2, sg2=sg2, Q=Q, f=Fc @ Q, Sin=Sin, SG=SG, T=T,
                m0=bz.mean(0), pi=pi, bbar=b.mean(0))
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


@torch.no_grad()
def predict(S, st, x, theta, post_weights=(1.,1.,1.), kappa=1., tau=1., tau_bn=1., tau_src=1., log_scale=(0.,0.25)):
    """Return query class probabilities for float32 episodes shaped (batch, clips, features)."""
    B,n,_ = x.shape
    C = S.cls.out_features
    if n == 1:
        return head_pop(S,pop_units(S,x[:,0])).softmax(-1)
    SL = sublattice(n,C,x.device)
    Hn = SL["H"]
    th = Hn / n
    Hh = len(th)
    u = pop_units(S,x)
    ub,s2 = u.mean(1),u.var(1,unbiased=False)
    Fc,Q,f,pi = (st[k] for k in ("Fc","Q","f","pi"))
    V = batch_var(th,st,n).clamp_min(1e-12)
    w = (st["sin2"]+st["sg2"]).clamp_min(1e-12).reciprocal()
    lam2 = (s2*w).sum(-1,keepdim=True)/(V*w).sum(-1)
    lam = lam2.sqrt()
    mu = ub[:,None]-lam[...,None]*(th@Fc)
    scale = ((n*s2[:,None]/V+lam2[...,None])/(n+1)).clamp_min(1e-12).sqrt()
    units = st["bbar"]+(u[:,None]-mu[:,:,None])/scale[:,:,None]
    logp = (head_pop(S,units)/tau).log_softmax(-1)
    Sin_,SG_ = st["Sin"],st["SG"]
    z = u @ Q
    d = z - z.mean(1, keepdim=True)  # (B, n, C-1)
    kG = (1 - (th ** 2).sum(-1)) * n / (n - 1)  # (Hh,)
    Sig_h = Sin_[None] + kG[:, None, None] * SG_[None]

    Sig_h = Sig_h + 1e-8 * torch.eye(C-1,device=x.device)
    P = torch.linalg.inv(Sig_h)  # (Hh, C-1, C-1)
    ldS = torch.logdet(Sig_h)    # (Hh,)
    g = (th @ f)[:, None] - f[None]  # (Hh, C, C-1): hypothesis location relative to every class centre
    A = torch.einsum("bnd,hde,bne->bhn", d, P, d)
    Bt = torch.einsum("bnd,hde,hce->bhnc", d, P, g)
    Cc = torch.einsum("hcd,hde,hce->hc", g, P, g)

    grid = torch.linspace(-3,3,13,device=x.device)
    lgrid = (log_scale[0]+max(log_scale[1],0.02)*grid).exp()
    lw = (-0.5*grid.square()).log_softmax(0)
    dens = []
    for lk, wk in zip(lgrid, lw):
        qd_k = A[..., None] / lk ** 2 + 2 * Bt / lk + Cc[None, :, None]
        dens.append(wk + (pi.log() - 0.5 * qd_k).logsumexp(-1).sum(-1) - (n - 1) * (C - 1) * lk.log())
    dens = torch.stack(dens).logsumexp(0) - 0.5 * (n - 1) * ldS[None]  # (B, Hh)

    counts = Hn[:,None]-torch.eye(C,device=x.device)[None]
    prior = log_dmseq(counts.clamp_min(0),pi,"dm")
    g = ((C*th.square().sum(-1)-1)/(C-1)).clamp(0,1)
    anchor = ((head_pop(S,u.mean(1)).log_softmax(-1)-pi.log())@th.T)*g
    a,b,c = post_weights
    L = count_dp(SL,a*(logp-pi.log()))+a*logp[:,:,0]+prior+b*dens[...,None]+c*anchor[...,None]
    mass = L.logsumexp(-1).softmax(-1)
    conditional = L.softmax(-1).nan_to_num()
    lt = lam[...,None]*th
    elt = (mass[...,None]*lt).sum(1)
    cov = torch.einsum("bh,bhc,bhd->bcd",mass,lt,lt)-elt[:,:,None]*elt[:,None,:]
    centre = ub-elt@Fc
    error = f.T@cov@f+(mass*lam2*g).sum(1)[:,None,None]*SG_+(mass*lam2).sum(1)[:,None,None]*Sin_/n
    prior_cov = kappa*st["T"]
    delta = centre@Q-st["m0"]
    shrink = prior_cov@torch.linalg.inv(prior_cov+error+1e-6*torch.eye(C-1,device=x.device))
    centre = centre+((shrink@delta.unsqueeze(-1)).squeeze(-1)-delta)@Q.T
    v = batch_var(mass@th,st,n).clamp_min(1e-12)
    amp = (s2*w).sum(-1,keepdim=True)/(v*w).sum(-1,keepdim=True)
    sc = ((n*s2/v+amp)/(n+1)+S.bn.eps/target_var(st).clamp_min(1e-12)).sqrt()
    eb = (head_pop(S,st["bbar"]+(u[:,0]-centre)/sc)/tau).softmax(-1)
    bn = (head(S,(u[:,0]-ub)/(s2+S.bn.eps).sqrt())/tau_bn).softmax(-1)
    src = (head_pop(S,u[:,0])/tau_src).softmax(-1)
    coeff = torch.as_tensor(theta,device=x.device,dtype=x.dtype).reshape(3,2)
    g = torch.round(g*1e4)/1e4
    logits = coeff[:,0]+g[:,None]*coeff[:,1]
    weights = torch.cat((torch.zeros_like(g[:,None]),logits),1).softmax(-1)
    return ((mass*weights[:,0])[...,None]*conditional).sum(1)+(torch.stack((eb,bn,src),1)*(mass@weights[:,1:])[...,None]).sum(1)
