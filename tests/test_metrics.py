import math

import numpy as np
import pytest

from calibration.metrics import (auroc, bootstrap_ci, brier_score,
                                 expected_calibration_error, reliability_table,
                                 summarize)


def test_brier_known_values():
    assert brier_score([1, 0], [1, 0]) == 0
    assert brier_score([0.5, 0.5], [1, 0]) == 0.25
    assert brier_score([0.9, 0.2], [1, 0]) == pytest.approx((0.01 + 0.04) / 2)


def test_ece_perfectly_calibrated_is_zero():
    # 10 answers at 0.7 confidence, 7 correct
    p = [0.7] * 10
    y = [1] * 7 + [0] * 3
    assert expected_calibration_error(p, y) == pytest.approx(0)


def test_ece_known_gap():
    # Always 90% confident, right half the time: gap of 0.4
    p = [0.9] * 10
    y = [1, 0] * 5
    assert expected_calibration_error(p, y) == pytest.approx(0.4)


def test_ece_weights_bins_by_count():
    # 8 answers at 0.95 all correct (gap 0.05), 2 at 0.15 all correct (gap 0.85)
    p = [0.95] * 8 + [0.15] * 2
    y = [1] * 10
    assert expected_calibration_error(p, y) == pytest.approx((8 * 0.05 + 2 * 0.85) / 10)


def test_confidence_of_one_lands_in_top_bin():
    rows = reliability_table([1.0, 0.0], [1, 0], n_bins=10)
    assert rows[-1]["bin_high"] == 1.0 and rows[-1]["count"] == 1
    assert rows[0]["bin_low"] == 0.0 and rows[0]["count"] == 1


def test_auroc():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert math.isnan(auroc([0.5, 0.6], [1, 1]))


def test_auroc_matches_mann_whitney():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = np.round(rng.random(200) * 0.5 + 0.4 * y, 1)          # coarse values create ties
    u = scipy_stats.mannwhitneyu(p[y == 1], p[y == 0]).statistic
    assert auroc(p, y) == pytest.approx(u / ((y == 1).sum() * (y == 0).sum()))


def test_summary_and_bootstrap():
    rng = np.random.default_rng(0)
    p = rng.random(100)
    y = (rng.random(100) < p).astype(int)
    s = summarize(p, y)
    ci = bootstrap_ci(p, y, n_boot=200)
    for k in ("accuracy", "brier", "ece"):
        assert ci[k][0] <= s[k] <= ci[k][1]
