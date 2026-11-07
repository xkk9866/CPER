"""Train / evaluate REIN (or a baseline variant) on one dataset split.

Each training sample is a subject episode: a query clip + K unlabeled clips of the same subject
(K ~ U{0..ctx}, composition balanced or skewed). Context clips are encoded without gradient.
Evaluation reports three protocols from the same model:
  std   : K = 0 (standard subject-independent protocol, no extra test data)
  bal   : K = ctx_test random other test clips of the same subject (unlabeled)
  skew  : K = ctx_test clips of the same subject that all share one emotion != query's
"""
import os as _os
_data_root = _os.path.abspath(_os.path.expanduser(_os.environ.get('EMO_ROOT', '~/.cper')))

import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn.functional as F

from data import FaceSet, Loader, gpu_augment, sample_frames, LABELS
from model import ExpressionModel

RUNS = f"{_data_root}/runs"


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cremad")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--tag", default="dbg")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--re", default="learned", choices=["eb", "learned", "mean", "none", "icl"])
    ap.add_argument("--estep", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=4, help="max context clips during training (0 = no episodes)")
    ap.add_argument("--ctx_test", type=int, default=6)
    ap.add_argument("--Tc", type=int, default=4, help="frames per context clip")
    ap.add_argument("--aux", type=float, default=1.0, help="weight of the random-effects likelihood loss")
    ap.add_argument("--head", default="tf", choices=["tf", "mil", "cliptext"])
    ap.add_argument("--tent", type=int, default=0, help="also evaluate with per-subject TENT adaptation")
    ap.add_argument("--tent_lr", type=float, default=1e-3)
    ap.add_argument("--tent_steps", type=int, default=3)
    ap.add_argument("--reasoner", default="llm", choices=["llm", "tiny"])
    ap.add_argument("--llm_path", default=f"{_data_root}/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--M", type=int, default=4)
    ap.add_argument("--train_blocks", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--bs", type=int, default=0)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr_enc", type=float, default=2e-5)
    ap.add_argument("--lr_llm", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--cached", default="", help="stage 2: dir of cached fine-tuned tokens ('' = end-to-end)")
    ap.add_argument("--cached2", default="", help="second (frozen) token stream concatenated along the token axis")
    ap.add_argument("--R", type=int, default=0, help="number of region tokens (0 = 1 + grid^2)")
    ap.add_argument("--dump_feats", type=int, default=0, help="stage 1: save encoder tokens for all clips")
    ap.add_argument("--mem_frac", type=float, default=0.45)
    return ap.parse_args(argv)


DEFAULTS = {"ferplus": dict(epochs=10, bs=64), "cremad": dict(epochs=10, bs=16), "ravdess": dict(epochs=25, bs=16)}


def splits(a):
    ds = FaceSet(a.data)
    if a.cached:
        p = f"{a.cached}/{a.data}_f{a.fold}.npy"
        p = p if os.path.exists(p) else f"{a.cached}/{a.data}.npy"  # frozen features are shared by all folds
        F1 = np.load(p, mmap_mode="r" if a.data == "cremad" else None)
        if a.cached2:  # dual stream: [fine-tuned tokens ; frozen tokens]
            F2 = np.load(f"{a.cached2}/{a.data}.npy", mmap_mode="r")
            F1 = np.concatenate([np.asarray(F1), np.asarray(F2)], axis=-2)
            a.R = F1.shape[-2]
        ds.use_features(F1)
    m = ds.meta
    if a.data == "ferplus":
        return ds.subset((m.split == "train").values), ds.subset((m.split == "test").values), ds.subset((m.split == "val").values)
    return ds.subset((m.fold != a.fold).values), ds.subset((m.fold == a.fold).values), None


def metrics(prob, y, C):
    pred = prob.argmax(1)
    rec = [float(np.mean(pred[y == c] == c)) * 100 if np.any(y == c) else float("nan") for c in range(C)]
    return dict(acc=float(np.mean(pred == y) * 100), uar=float(np.nanmean(rec)), recall=rec)


def prep(x, ds, train):
    """Pixels -> augmented CLIP input; cached tokens -> (N,R,D) float with light token noise."""
    if getattr(ds, "cached", False):
        x = x.cuda(non_blocking=True).float().flatten(0, 1)
        if train:
            x = x + 0.05 * x.std() * torch.randn_like(x)
            keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > 0.1).float()
            x = x * keep
        return x
    return gpu_augment(x.cuda(non_blocking=True), 224, train)


def make_ctx(ds, rows, K, mode, rng):
    y, act = ds.y, ds.meta.actor.values
    out = np.zeros((len(rows), K), int)
    for i, r in enumerate(rows):
        pool = np.where((act == act[r]) & (np.arange(len(ds)) != r))[0]
        if mode == "skew":
            e = rng.choice([e for e in np.unique(y[pool]) if e != y[r]])
            pool = pool[y[pool] == e]
        out[i] = rng.choice(pool, K, replace=len(pool) < K)
    return out


@torch.no_grad()
def encode_clips(model, ds, rows, Tc, train):
    """Clip-mean region tokens (N,R*d) for reference clips, no gradient."""
    was = model.training
    model.eval()
    fr = sample_frames(len(rows), Tc, train=train)
    x = ds.fetch(rows, fr).cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        Z = model.features(prep(x, ds, False)).view(len(rows), Tc, -1)
    model.train(was)
    return Z.float().mean(1)


@torch.no_grad()
def evaluate(model, ds, a, K, mode, seed=123):
    model.eval()
    rng = np.random.default_rng(seed)
    C = len(ds.labels)
    H = None
    if K > 0:  # precompute reference embeddings for every test clip once
        H = torch.cat([encode_clips(model, ds, np.arange(i, min(i + 128, len(ds))), a.Tc, False) for i in range(0, len(ds), 128)])
        ctx = make_ctx(ds, np.arange(len(ds)), K, mode, rng)
    probs = 0
    offsets = [0, 1] if ds.video else [0]
    for off in offsets:
        P = []
        for x, y, rows in Loader(ds, 32, a.T, train=False, offset=off).epoch():
            B, T = x.shape[:2]
            hc = H[torch.from_numpy(ctx[rows])] if K > 0 else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _, _ = model(prep(x, ds, False), B, T, ds.labels, ds.video, hc)
            P.append(logits.float().softmax(1).cpu())
        probs = probs + torch.cat(P).numpy()
    model.train()
    return probs / len(offsets)


def evaluate_tent(model, ds, a):
    """TENT: per test subject, adapt LayerNorm affine params by entropy minimisation on that subject's
    unlabeled clips, predict them, then restore the source weights."""
    params = [p for n, p in model.named_parameters() if p.requires_grad and ("norm" in n or ".ln" in n or "layer_norm" in n)]
    src = [p.detach().clone() for p in params]
    print(f"TENT adapting {sum(p.numel() for p in params)} norm params", flush=True)
    pred = np.zeros((len(ds), len(ds.labels)), np.float32)
    for actor in np.unique(ds.meta.actor.values):
        rows = np.where(ds.meta.actor.values == actor)[0]
        sub = ds.subset(ds.meta.actor.values == actor)
        opt = torch.optim.Adam(params, lr=a.tent_lr)
        model.train()
        for _ in range(a.tent_steps):
            for x, _, _ in Loader(sub, 16, a.T, train=True).epoch():
                B, T = x.shape[:2]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg, _, _ = model(prep(x, ds, False), B, T, ds.labels, ds.video, None)
                p = lg.float().softmax(1)
                ent = -(p * p.clamp_min(1e-8).log()).sum(1).mean()
                opt.zero_grad(set_to_none=True); ent.backward(); opt.step()
        pred[rows] = evaluate(model, sub, a, 0, "bal")
        with torch.no_grad():
            for p, s0 in zip(params, src):
                p.copy_(s0)
    return pred


@torch.no_grad()
def dump_features(model, a):
    """Region tokens (pre-projection, 768-d) of the fine-tuned encoder for every clip / image of the dataset."""
    model.eval()
    ds = FaceSet(a.data)
    out = []
    for i in range(0, len(ds), 16 if ds.video else 256):
        rows = np.arange(i, min(i + (16 if ds.video else 256), len(ds)))
        fr = np.tile(np.arange(16), (len(rows), 1)) if ds.video else np.zeros((len(rows), 1), int)
        x = ds.fetch(rows, fr)
        B, T = x.shape[:2]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.enc(gpu_augment(x.cuda(), 224, False))
        h = h.float().view(B, T, *h.shape[1:]).half().cpu().numpy()
        out.append(h if ds.video else h[:, 0])
    d = f"{_data_root}/features/ft"
    os.makedirs(d, exist_ok=True)
    arr = np.concatenate(out)
    # rows of FaceSet(a.data).meta are in idx order 0..N-1, so index by meta.idx later works directly
    full = np.zeros((int(ds.meta.idx.max()) + 1,) + arr.shape[1:], np.float16)
    full[ds.meta.idx.values] = arr
    np.save(f"{d}/{a.data}_f{a.fold}.npy", full)


def main(argv=None):
    a = get_args(argv)
    for k, v in DEFAULTS[a.data].items():
        if not getattr(a, k):
            setattr(a, k, v)
    if a.data == "ferplus":
        a.ctx = 0
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    if a.cached:
        a.train_blocks = 0
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(f"{RUNS}/{a.tag}", exist_ok=True)
    out_file = f"{RUNS}/{a.tag}/{a.data}_f{a.fold}_s{a.seed}.json"

    tr, te, va = splits(a)
    labels, C = tr.labels, len(tr.labels)
    model = ExpressionModel(a, vocab=sorted({w for v in LABELS.values() for w in v}), n_classes=C).cuda()
    enc_p = [p for p in model.enc.parameters() if p.requires_grad] if model.enc is not None else []
    llm_p = [p for p in model.reasoner.llm.parameters() if p.requires_grad] if a.reasoner == "llm" else []
    taken = {id(p) for p in enc_p + llm_p}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in taken]
    groups = [dict(params=enc_p, lr=a.lr_enc), dict(params=rest, lr=a.lr)] + ([dict(params=llm_p, lr=a.lr_llm, weight_decay=0.0)] if llm_p else [])
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    base = [g["lr"] for g in opt.param_groups]
    allp = [p for g in groups for p in g["params"]]

    spe = len(tr) // a.bs if not a.max_batches else min(a.max_batches, len(tr) // a.bs)
    total, warm = spe * a.epochs, max(1, int(0.05 * spe * a.epochs))
    log, step, t0, best = [], 0, time.time(), (-1, None, -1)
    for ep in range(a.epochs):
        tl, n = 0, 0
        for x, y, rows in Loader(tr, a.bs, a.T, train=True).epoch(n_batches=a.max_batches or None):
            B, T = x.shape[:2]
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, step / total)))
            K = int(rng.integers(0, a.ctx + 1)) if a.ctx else 0
            hc, yc = None, None
            if K > 0:
                c = make_ctx(tr, rows, K, "skew" if rng.random() < 0.5 else "bal", rng)
                hc = encode_clips(model, tr, c.flatten(), a.Tc, True).view(B, K, -1)
                yc = torch.from_numpy(tr.y[c]).cuda()
            y = y.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, ll, hq = model(prep(x, tr, True), B, T, labels, tr.video, hc)
                lab = y[:, None] if yc is None else torch.cat([y[:, None], yc], 1)
                loss = F.cross_entropy(logits.float(), y, label_smoothing=a.ls)
                if a.re not in ("none", "icl"):
                    loss = loss + a.aux * F.cross_entropy(ll.flatten(0, 1).float(), lab.flatten(), label_smoothing=a.ls)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(allp, 1.0); opt.step()
            if a.re == "eb" and K > 0:
                model.re.update_moments(torch.cat([hq.detach().float()[:, None], hc], 1), lab)
            tl += loss.item(); n += 1; step += 1
        msg = dict(ep=ep, loss=tl / n, min=round((time.time() - t0) / 60, 2))
        if va is not None:
            pv = evaluate(model, va, a, 0, "bal")
            msg["val"] = metrics(pv, va.y, C)["acc"]
            if msg["val"] > best[0]:
                best = (msg["val"], metrics(evaluate(model, te, a, 0, "bal"), te.y, C), ep)
        log.append(msg); print(json.dumps(msg), flush=True)

    res = dict(args=vars(a), log=log, n_test=len(te))
    if va is not None:
        res["std"] = best[1]; res["best_epoch"] = best[2]
    else:
        res["std"] = metrics(evaluate(model, te, a, 0, "bal"), te.y, C)
        if a.data != "ferplus":
            for mode in ["bal", "skew"]:
                res[mode] = metrics(evaluate(model, te, a, a.ctx_test, mode), te.y, C)
            if a.tent:
                res["tent"] = metrics(evaluate_tent(model, te, a), te.y, C)
            res["Kcurve"] = {k: metrics(evaluate(model, te, a, k, "bal"), te.y, C)["uar"] for k in [1, 3]}
    if a.dump_feats:
        dump_features(model, a)
    json.dump(res, open(out_file, "w"))
    s = " ".join(f"{k}={res[k]['acc']:.2f}/{res[k]['uar']:.2f}" for k in ["std", "bal", "skew", "tent"] if k in res)
    print(f"RESULT {a.tag} {a.data} f{a.fold} {s}", flush=True)
    return res


if __name__ == "__main__":
    main()
