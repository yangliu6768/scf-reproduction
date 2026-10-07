"""Source aggregation and calibrated participation inference."""

import math

import numpy as np


def _mean(values):
    ordered = np.sort(values)
    if ordered[0] == ordered[-1]:
        return float(ordered[0])
    return math.fsum(ordered) / len(ordered)


def raw_scores(values, offsets):
    """Return Mean, Max, Min, and lower-tail-trimmed RaMIA scores."""
    scores = np.empty((len(offsets) - 1, 4), dtype=np.float64)
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        ordered = np.sort(values[start:stop])
        keep = max(1, 3 * len(ordered) // 4)
        scores[i] = (
            _mean(ordered),
            ordered[-1],
            ordered[0],
            np.mean(ordered[::-1][:keep]),
        )
    return scores


def reference_tail(values, offsets, reference_indices):
    """Evaluate the smoothed, source-balanced inclusive upper tail."""
    bags = [values[offsets[i]:offsets[i + 1]] for i in reference_indices]
    count = sum(map(len, bags))
    denominator = math.lcm(*map(len, bags))
    total_weight = denominator * len(bags)
    if (count + 1) * total_weight >= 2**53:
        raise ValueError("Reference weights exceed exact numerical range.")
    reference = np.concatenate(bags)
    weights = np.concatenate([
        np.full(len(bag), denominator // len(bag), dtype=np.int64)
        for bag in bags
    ])
    order = np.argsort(reference, kind="stable")
    prefix = np.r_[0, np.cumsum(weights[order])]
    below = prefix[np.searchsorted(reference[order], values, side="left")]
    return (total_weight + count * (total_weight - below)) / (
        (count + 1) * total_weight
    )


def fisher_scores(q, offsets, alphas):
    """Average capped log-tail evidence; alpha=0 retains smoothing."""
    scores = np.empty((len(offsets) - 1, len(alphas)), dtype=np.float64)
    for j, alpha in enumerate(alphas):
        evidence = -np.log(np.maximum(q, alpha))
        for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            scores[i, j] = _mean(evidence[start:stop])
    return scores


def auc_credits(participant_scores, null_scores):
    """Return twice the total pair credit for each method, with half-credit ties."""
    credits = np.empty(participant_scores.shape[1], dtype=np.int64)
    for j in range(participant_scores.shape[1]):
        positive = np.sort(participant_scores[:, j])
        below = np.searchsorted(positive, null_scores[:, j], side="left")
        at_most = np.searchsorted(positive, null_scores[:, j], side="right")
        credits[j] = np.sum(2 * len(positive) - below - at_most, dtype=np.int64)
    return credits


def conformal_ranks(calibration_scores, target_scores):
    """Return 1 + the number of calibration scores at least as large as each target."""
    ranks = np.empty(target_scores.shape, dtype=np.int64)
    for j in range(target_scores.shape[1]):
        calibration = np.sort(calibration_scores[:, j])
        ranks[:, j] = len(calibration) + 1 - np.searchsorted(
            calibration, target_scores[:, j], side="left"
        )
    return ranks
