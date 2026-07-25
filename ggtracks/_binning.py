"""Shared interval → bin resampling.

Both the bigWig reader (resampling a wide region at read time) and
:class:`~ggtracks.StatBinCoverage` (resampling at render time) need the same
piece of arithmetic, so it lives here once rather than in two dialects.

The rule: a record is distributed across every bin it overlaps, weighted by
the **length of that overlap**. A value spanning two bins therefore
contributes to both in proportion, and the integral of the signal is
preserved exactly under any binning.
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

__all__ = ["Summary", "bin_intervals"]

Summary = Literal["mean", "max", "min", "sum"]

_SUMMARIES = ("mean", "max", "min", "sum")


def _integrate(
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-bin ``(∫ value dx, ∫ covered dx)``.

    The records describe a piecewise-constant function whose cumulative
    integral is piecewise *linear*, with a knot wherever a record starts or
    ends. Sampling that cumulative integral at the bin edges and taking
    differences therefore gives each bin's exact share in one pass, instead
    of walking record-by-bin pairs.
    """
    x = np.concatenate([starts, ends])
    step_value = np.concatenate([values, -values])
    step_cover = np.concatenate([np.ones_like(values), -np.ones_like(values)])

    order = np.argsort(x, kind="stable")
    x = x[order]
    widths = np.diff(x)

    cumulative_value = np.concatenate(
        [[0.0], np.cumsum(np.cumsum(step_value[order])[:-1] * widths)]
    )
    cumulative_cover = np.concatenate(
        [[0.0], np.cumsum(np.cumsum(step_cover[order])[:-1] * widths)]
    )
    return (
        np.diff(np.interp(edges, x, cumulative_value)),
        np.diff(np.interp(edges, x, cumulative_cover)),
    )


def _extremes(
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    edges: np.ndarray,
    lo: float,
    hi: float,
    bins: int,
    largest: bool,
) -> np.ndarray:
    """Per-bin max or min.

    Unlike an integral these do not decompose into a cumulative sum, so this
    walks the records. It is the slower path and the reason ``"mean"`` is the
    default.
    """
    width = (hi - lo) / bins
    out = np.full(bins, np.nan, dtype=np.float64)
    pick = np.fmax if largest else np.fmin

    for s, e, v in zip(starts, ends, values):
        s = max(float(s), lo)
        e = min(float(e), hi)
        if e <= s:
            continue
        first = min(bins - 1, max(0, int((s - lo) / width)))
        last = min(bins - 1, max(0, int(np.ceil((e - lo) / width)) - 1))
        out[first : last + 1] = pick(out[first : last + 1], v)
    return np.nan_to_num(out, nan=0.0)


def bin_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    lo: float,
    hi: float,
    bins: int,
    summary: Summary = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample interval-valued records onto *bins* equal bins of ``[lo, hi)``.

    Parameters
    ----------
    starts, ends, values
        Interval-valued records. Intervals are half-open and need not be
        contiguous; anything falling outside ``[lo, hi)`` is ignored.
    lo, hi
        The range to bin over.
    bins
        Number of equal-width bins.
    summary
        ``"mean"`` — coverage-weighted average, i.e. the integral over the
        bin divided by the length actually covered (uncovered stretches do
        not drag the value toward zero).
        ``"sum"`` — the integral itself.
        ``"max"`` / ``"min"`` — the extreme value touching the bin.

    Returns
    -------
    tuple of ndarray
        ``(bin_starts, bin_ends, values)``, all of length *bins*. Bins that
        no record touches come back as ``0.0``.

    Raises
    ------
    ValueError
        Unknown *summary*, non-positive *bins*, or ``hi <= lo``.
    """
    if summary not in _SUMMARIES:
        raise ValueError(
            f"bin_intervals: summary must be one of {_SUMMARIES} (got {summary!r})."
        )
    if bins < 1:
        raise ValueError(f"bin_intervals: bins must be >= 1 (got {bins}).")
    if hi <= lo:
        raise ValueError(f"bin_intervals: hi must exceed lo (got {lo}-{hi}).")

    edges = np.linspace(lo, hi, bins + 1)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    if starts.size == 0:
        return edges[:-1], edges[1:], np.zeros(bins, dtype=np.float64)

    if summary in ("max", "min"):
        out = _extremes(starts, ends, values, edges, lo, hi, bins, summary == "max")
        return edges[:-1], edges[1:], out

    weighted, covered = _integrate(starts, ends, values, edges)
    if summary == "sum":
        return edges[:-1], edges[1:], weighted
    mean = np.divide(weighted, covered, out=np.zeros(bins), where=covered > 1e-12)
    return edges[:-1], edges[1:], mean
