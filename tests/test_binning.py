"""Tests for the shared interval → bin resampling.

``mean`` and ``sum`` are computed from a cumulative integral rather than by
walking record-by-bin pairs, so they are checked against a deliberately
naive reference that is obviously correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from ggtracks._binning import bin_intervals


def reference(starts, ends, values, lo, hi, bins, summary):
    """Brute force: every record against every bin."""
    edges = np.linspace(lo, hi, bins + 1)
    weighted = np.zeros(bins)
    covered = np.zeros(bins)
    extreme = np.full(bins, np.nan)
    for s, e, v in zip(starts, ends, values):
        for b in range(bins):
            overlap = min(float(e), edges[b + 1]) - max(float(s), edges[b])
            if overlap <= 0:
                continue
            weighted[b] += v * overlap
            covered[b] += overlap
            extreme[b] = v if np.isnan(extreme[b]) else (
                max(extreme[b], v) if summary == "max" else min(extreme[b], v)
            )
    if summary == "sum":
        return weighted
    if summary == "mean":
        return np.divide(weighted, covered, out=np.zeros(bins), where=covered > 1e-12)
    return np.nan_to_num(extreme, nan=0.0)


CASES = {
    "contiguous": (np.arange(0, 1000, 10), np.arange(10, 1010, 10)),
    "gapped": (np.arange(0, 1000, 20), np.arange(0, 1000, 20) + 7),
    "spanning many bins": (np.array([0, 300, 700]), np.array([250, 650, 1000])),
    "single record": (np.array([120]), np.array([880])),
    "overlapping": (np.array([0, 100, 150]), np.array([400, 600, 900])),
}


@pytest.mark.parametrize("name", list(CASES))
@pytest.mark.parametrize("summary", ["mean", "sum", "max", "min"])
@pytest.mark.parametrize("bins", [1, 7, 37, 250])
def test_matches_the_naive_reference(name, summary, bins):
    starts, ends = CASES[name]
    values = np.linspace(1, 100, len(starts))
    _lo, _hi, got = bin_intervals(starts, ends, values, 0.0, 1000.0, bins, summary)
    want = reference(starts, ends, values, 0.0, 1000.0, bins, summary)
    assert np.allclose(got, want, rtol=1e-9, atol=1e-9)


def test_bin_edges_tile_the_range():
    lo, hi = bin_intervals(np.array([0]), np.array([10]), np.array([1.0]),
                           0.0, 100.0, 4, "mean")[:2]
    assert np.allclose(lo, [0, 25, 50, 75])
    assert np.allclose(hi, [25, 50, 75, 100])


def test_integral_is_independent_of_bin_count():
    starts = np.arange(0, 1000, 10)
    ends = starts + 10
    values = np.random.default_rng(0).random(len(starts)) * 50
    exact = float((values * 10).sum())
    for bins in (1, 3, 64, 997):
        _lo, _hi, out = bin_intervals(starts, ends, values, 0.0, 1000.0, bins, "sum")
        assert out.sum() == pytest.approx(exact, rel=1e-9)


def test_records_outside_the_range_are_ignored():
    starts = np.array([-500, 100, 5000])
    ends = np.array([-100, 200, 6000])
    values = np.array([99.0, 1.0, 99.0])
    _lo, _hi, out = bin_intervals(starts, ends, values, 0.0, 1000.0, 10, "sum")
    assert out.sum() == pytest.approx(100.0)


def test_a_record_straddling_the_edge_contributes_only_its_inside_part():
    _lo, _hi, out = bin_intervals(
        np.array([900]), np.array([1500]), np.array([2.0]), 0.0, 1000.0, 1, "sum"
    )
    assert out[0] == pytest.approx(200.0)  # 100 bp inside x 2.0


def test_empty_input_gives_empty_bins():
    empty = np.array([], dtype=float)
    _lo, _hi, out = bin_intervals(empty, empty, empty, 0.0, 100.0, 5, "mean")
    assert out.shape == (5,)
    assert np.all(out == 0)


def test_untouched_bins_are_zero_not_nan():
    _lo, _hi, out = bin_intervals(
        np.array([0]), np.array([10]), np.array([5.0]), 0.0, 100.0, 10, "max"
    )
    assert np.isfinite(out).all()
    assert out[0] == pytest.approx(5.0) and out[5] == 0.0


def test_mean_divides_by_covered_length_not_bin_width():
    """Half a bin covered at value 10 reads 10, not 5 — uncovered stretches
    are missing data, not zeros."""
    _lo, _hi, out = bin_intervals(
        np.array([0]), np.array([50]), np.array([10.0]), 0.0, 100.0, 1, "mean"
    )
    assert out[0] == pytest.approx(10.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"summary": "median"}, "summary must be"),
        ({"bins": 0}, "bins must be"),
        ({"lo": 100.0, "hi": 100.0}, "hi must exceed lo"),
    ],
)
def test_validation(kwargs, match):
    args = dict(starts=np.array([0]), ends=np.array([10]), values=np.array([1.0]),
                lo=0.0, hi=100.0, bins=4, summary="mean")
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        bin_intervals(**args)


def test_large_input_stays_fast():
    """The cumulative-integral path is what keeps a whole-chromosome query
    from taking seconds; a regression to per-record looping shows up here."""
    import time

    n = 200_000
    starts = np.arange(n, dtype=float) * 50
    ends = starts + 50
    values = np.random.default_rng(0).random(n)
    t0 = time.perf_counter()
    bin_intervals(starts, ends, values, float(starts[0]), float(ends[-1]), 500, "mean")
    assert time.perf_counter() - t0 < 1.0
