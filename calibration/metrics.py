"""Calibration metrics.

confidence  array of predicted probabilities that the answer is correct (0 to 1)
correct     array of 0/1 outcomes
"""
import numpy as np


def _arrays(confidence, correct):
    p = np.asarray(confidence, dtype=float)
    y = np.asarray(correct, dtype=float)
    if p.shape != y.shape or p.ndim != 1:
        raise ValueError("confidence and correct must be 1-D arrays of equal length")
    return p, y


def brier_score(confidence, correct):
    """Mean squared error of the confidence. 0 is perfect. Always saying 50% scores 0.25."""
    p, y = _arrays(confidence, correct)
    return float(np.mean((p - y) ** 2))


def reliability_table(confidence, correct, n_bins=10):
    """Equal-width bins on [0, 1]. Returns one row per non-empty bin with the
    bin range, count, mean confidence, and accuracy."""
    p, y = _arrays(confidence, correct)
    edges = np.linspace(0, 1, n_bins + 1)
    # Right edge included in the last bin, so a confidence of 1.0 is counted
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin_low": edges[b], "bin_high": edges[b + 1],
                         "count": int(mask.sum()),
                         "mean_confidence": float(p[mask].mean()),
                         "accuracy": float(y[mask].mean())})
    return rows


def expected_calibration_error(confidence, correct, n_bins=10):
    """Average gap between confidence and accuracy across bins, weighted by
    how many questions fall in each bin. 0 is perfect."""
    p, _ = _arrays(confidence, correct)
    rows = reliability_table(confidence, correct, n_bins)
    return float(sum(r["count"] * abs(r["mean_confidence"] - r["accuracy"])
                     for r in rows) / len(p))


def auroc(confidence, correct):
    """Probability that a randomly chosen correct answer got higher confidence
    than a randomly chosen wrong one (ties count half). 0.5 means confidence
    does not separate right from wrong answers. Undefined if all outcomes match."""
    p, y = _arrays(confidence, correct)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def summarize(confidence, correct, n_bins=10):
    p, y = _arrays(confidence, correct)
    return {
        "n": len(p),
        "accuracy": float(y.mean()),
        "mean_confidence": float(p.mean()),
        "overconfidence": float(p.mean() - y.mean()),
        "brier": brier_score(p, y),
        "ece": expected_calibration_error(p, y, n_bins),
        "auroc": auroc(p, y),
    }


def bootstrap_ci(confidence, correct, n_boot=2000, level=0.95, seed=0, n_bins=10):
    """Percentile bootstrap intervals for every metric in summarize(),
    resampling questions with replacement."""
    p, y = _arrays(confidence, correct)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in summarize(p, y, n_bins) if k != "n"}
    for _ in range(n_boot):
        i = rng.integers(0, len(p), len(p))
        s = summarize(p[i], y[i], n_bins)
        for k in draws:
            draws[k].append(s[k])
    alpha = (1 - level) / 2
    out = {}
    for k, v in draws.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        # AUROC is undefined in a resample with no wrong (or no right) answers
        out[k] = ((float(np.quantile(v, alpha)), float(np.quantile(v, 1 - alpha)))
                  if len(v) else (float("nan"), float("nan")))
    return out
