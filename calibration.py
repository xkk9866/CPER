"""Estimate posterior weights, branch temperatures and stacking coefficients."""

import itertools
import math

import numpy as np
import torch
from torch.nn import functional as F

from cper import _predict, batch_var, count_dp, head_pop, pop_units, population_stats


def temperature(logits, labels):
    lo, hi = math.log(0.05), math.log(20.0)
    for _ in range(40):
        a, b = (2 * lo + hi) / 3, (lo + 2 * hi) / 3
        if F.cross_entropy(logits / math.exp(a), labels) < F.cross_entropy(
            logits / math.exp(b), labels
        ):
            hi = b
        else:
            lo = a
    return math.exp((lo + hi) / 2)


def episodes(rows, labels, subjects, k, rng):
    for q in rng.choice(rows, min(len(rows), 480), replace=False):
        pool = rows[(subjects[rows] == subjects[q]) & (rows != q)]
        same = pool[labels[pool] == labels[q]]
        if not len(pool):
            continue
        yield [q, *rng.choice(pool, k, replace=len(pool) < k)]
        classes = np.unique(labels[pool])
        proportions = rng.dirichlet(np.full(len(classes), 0.2))
        refs = [
            rng.choice(pool[labels[pool] == c])
            for c in rng.choice(classes, k, p=proportions)
        ]
        yield [q, *refs]
        if len(same):
            yield [q, *rng.choice(same, k, replace=len(same) < k)]


def fit_weights(parts):
    axes = [[0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0] for _ in range(2)] + [
        [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    ]
    scores, extensions = {}, [0, 0, 0]
    while True:
        grid = [key for key in itertools.product(*axes) if key not in scores]
        for key in grid:
            scores[key] = 0.0
        for sl, unary, query, prior, density, anchor, target in parts:
            for a, group in itertools.groupby(grid, key=lambda v: v[0]):
                base = count_dp(sl, a * unary) + a * query + prior
                for key in group:
                    joint = (
                        base + key[1] * density[..., None] + key[2] * anchor[..., None]
                    )
                    scores[key] += (
                        joint.logsumexp(-1)
                        .log_softmax(-1)
                        .gather(1, target[:, None])
                        .sum()
                        .item()
                    )
        best = max(scores, key=scores.get)
        for i, axis in enumerate(axes):
            if extensions[i] < 3 and best[i] == axis[-1]:
                axis.append(2 * axis[-1])
            elif extensions[i] < 3 and best[i] == axis[0] and axis[0] > 0:
                axis.insert(0, axis[0] / 2)
            else:
                continue
            extensions[i] += 1
            break
        else:
            return best


@torch.no_grad()
def calibrate(model, st, x, labels, subjects, valid, k, rng):
    rows = np.flatnonzero(valid)
    vstats = population_stats(model, x[valid], labels[valid], subjects[valid])
    held = vstats["b"] @ st["Q"]
    kappa = (held.var(0).sum() / st["T"].trace().clamp_min(1e-9)).item()
    classes = model.cls.out_features
    idx = np.asarray(list(episodes(rows, labels, subjects, k, rng)))
    if not len(idx):
        raise ValueError("Held-out people need repeated observations.")
    y = torch.as_tensor(labels, device=x.device, dtype=torch.long)
    logs = []
    for selected in (~valid, valid):
        values = []
        for person in np.unique(subjects[selected]):
            mask = selected & (subjects == person)
            th = F.one_hot(y[mask], classes).float().mean(0)
            v = batch_var(th, st, int(mask.sum()))
            w = (st["sin2"] + st["sg2"]).clamp_min(1e-12).reciprocal()
            observed = pop_units(model, x[mask]).var(0, unbiased=False)
            values.append(
                0.5
                * (
                    (observed * w).sum().clamp_min(1e-12)
                    / (v * w).sum().clamp_min(1e-12)
                )
                .log()
                .item()
            )
        logs.append(np.asarray(values))
    log_scale = (
        float(logs[1].mean() - logs[0].mean()),
        max(0.02, float(logs[1].std(ddof=1))),
    )
    # Fit temperatures on random reference sets with known compositions.
    draw = []
    for person in rng.choice(np.unique(subjects[valid]), 4096):
        draw.append(rng.choice(rows[subjects[rows] == person], k + 1))
    logits = [[], [], []]
    for group in np.array_split(np.asarray(draw), 64):
        u = pop_units(model, x[group])
        th = F.one_hot(y[group], classes).float().mean(1)
        v = batch_var(th, st, k + 1).clamp_min(1e-12)
        obs = u.var(1, unbiased=False)
        w = (st["sin2"] + st["sg2"]).clamp_min(1e-12).reciprocal()
        amp = (obs * w).sum(-1, keepdim=True) / (v * w).sum(-1, keepdim=True)
        centre = u.mean(1) - amp.sqrt() * (th @ st["Fc"])
        scale = (((k + 1) * obs / v + amp) / (k + 2)).clamp_min(1e-12).sqrt()
        for buf, units in zip(
            logits,
            (
                st["bbar"] + (u - centre[:, None]) / scale[:, None],
                (u - u.mean(1, keepdim=True)) / (obs[:, None] + model.bn.eps).sqrt(),
                u,
            ),
        ):
            buf.append(head_pop(model, units).flatten(0, 1))
    temps = [
        temperature(torch.cat(buf), y[np.asarray(draw)].flatten()) for buf in logits
    ]
    # Bound the number of simultaneous hypothesis-normalized embeddings.
    hypotheses = math.comb(k + classes, classes - 1)
    batch = max(1, min(64, 8000000 // (hypotheses * (k + 1) * model.cls.in_features)))
    groups = [idx[i : i + batch] for i in range(0, len(idx), batch)]

    def outputs(group, weights, which):
        return _predict(
            model,
            st,
            x[group],
            [0.0] * 6,
            weights,
            kappa,
            *temps,
            log_scale,
            output=which,
        )

    parts = []
    for group in groups:
        evidence = outputs(group, (1.0, 1.0, 1.0), "evidence")
        counts = F.one_hot(y[group], classes).sum(1)
        target = (counts[:, None] == evidence[0]["H"]).all(-1).long().argmax(-1)
        parts.append((*evidence, target))
    weights = fit_weights(parts)
    del parts
    # Pool posterior votes at repeated concentration levels before optimization.
    pooled = []
    for group in groups:
        mass, conditional, branches, g = outputs(group, weights, "branches")
        levels, inv = torch.unique(g, return_inverse=True)
        bins = F.one_hot(inv, len(levels)).float()
        pooled.append(
            (
                mass @ bins,
                torch.einsum("bh,bhc,hv->bvc", mass, conditional, bins),
                branches,
            )
        )
    mass, votes, branches = [torch.cat(items) for items in zip(*pooled)]
    targets = y[idx[:, 0]]
    freq = targets.bincount(minlength=classes).clamp_min(1)
    sample_weights = 1.0 / freq[targets]
    with torch.enable_grad():
        theta = torch.zeros(3, 2, device=x.device, requires_grad=True)
        opt = torch.optim.LBFGS([theta], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            lg = theta[:, 0] + levels[:, None] * theta[:, 1]
            w = torch.cat((torch.zeros_like(levels[:, None]), lg), 1).softmax(1)
            p = (votes * w[None, :, :1]).sum(1) + (
                (mass @ w[:, 1:])[:, :, None] * branches
            ).sum(1)
            loss = (
                -(
                    sample_weights
                    * p.gather(1, targets[:, None]).flatten().clamp_min(1e-8).log()
                ).sum()
                / sample_weights.sum()
                + 0.001 * theta.square().sum()
            )
            loss.backward()
            return loss

        opt.step(closure)
    return dict(
        theta=theta.detach().flatten().tolist(),
        post_weights=weights,
        kappa=kappa,
        tau=temps[0],
        tau_bn=temps[1],
        tau_src=temps[2],
        log_scale=log_scale,
    )
