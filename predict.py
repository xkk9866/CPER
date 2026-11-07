"""Predict class probabilities for query-first embedding sets."""

import argparse

import numpy as np
import torch

from cper import Source, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("episodes")
    parser.add_argument("output")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=True)
    model = Source(*state["shape"]).to(args.device).eval()
    model.load_state_dict(state["weights"])
    episodes = np.load(args.episodes, allow_pickle=False, mmap_mode="r")
    if (
        episodes.ndim != 3
        or episodes.shape[0] == 0
        or episodes.shape[1] == 0
        or episodes.shape[2] != state["shape"][0]
    ):
        parser.error("Expected a nonempty (batch, clips, embedding dimension) array.")
    result = []
    for start in range(0, len(episodes), args.batch_size):
        x = torch.tensor(
            episodes[start : start + args.batch_size],
            device=args.device,
            dtype=torch.float32,
        )
        x = (x - state["mean"]) / state["scale"]
        result.append(
            predict(model, state["stats"], x, **state["calibration"]).cpu().numpy()
        )
    np.save(args.output, np.concatenate(result))


if __name__ == "__main__":
    main()
