"""Fit a source classifier and its CPN calibration from labeled embeddings."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cper import Source, population_stats
from calibration import calibrate


def train_head(x, labels, classes, hidden, steps, rng):
    model = Source(x.shape[1], classes, hidden).to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 0.001, total_steps=steps, pct_start=0.05
    )
    for _ in range(steps):
        idx = rng.integers(len(x), size=320)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x[idx]), labels[idx], label_smoothing=0.1)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model.eval()


def fit(
    features,
    labels,
    subjects,
    *,
    references=6,
    steps=3000,
    hidden=512,
    seed=0,
    device="cpu",
):
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    labels, subjects = np.asarray(labels), np.asarray(subjects)
    if (
        x.ndim != 2
        or labels.shape != (len(x),)
        or subjects.shape != labels.shape
        or not torch.isfinite(x).all()
    ):
        raise ValueError(
            "Expected finite 2D features with matching label and subject vectors."
        )
    classes = len(np.unique(labels))
    if (
        classes < 2
        or not np.array_equal(np.unique(labels), np.arange(classes))
        or hidden < classes - 1
    ):
        raise ValueError(
            "Use contiguous labels starting at 0 and hidden >= classes - 1."
        )
    if references < 1 or steps < 21 or len(np.unique(subjects)) < 4:
        raise ValueError(
            "Use at least one reference, 21 training steps and four people."
        )
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    people = rng.permutation(np.unique(subjects))
    valid = np.isin(subjects, people[: max(2, round(len(people) / 4))])
    mean, scale = x.mean(0), x.std(0).clamp_min(1e-6)
    aux_x = (x - x[~valid].mean(0)) / x[~valid].std(0).clamp_min(1e-6)
    x = (x - mean) / scale
    torch.manual_seed(seed)
    aux = train_head(aux_x[~valid], y[~valid], classes, hidden, steps, rng)
    st = population_stats(aux, aux_x[~valid], labels[~valid], subjects[~valid])
    calibration = calibrate(aux, st, aux_x, labels, subjects, valid, references, rng)
    model = train_head(x, y, classes, hidden, steps, rng)
    st = population_stats(model, x, labels, subjects)
    st.pop("subs")
    return dict(
        weights=model.state_dict(),
        stats=st,
        calibration=calibration,
        shape=(x.shape[1], classes, hidden),
        mean=mean,
        scale=scale,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--references", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = vars(parser.parse_args())
    checkpoint = args.pop("checkpoint")
    with np.load(args.pop("input"), allow_pickle=False) as data:
        state = fit(data["features"], data["labels"], data["subjects"], **args)
    torch.save(state, checkpoint)


if __name__ == "__main__":
    main()
