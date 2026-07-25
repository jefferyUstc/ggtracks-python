"""Tests for the signal-track geom and its binning stat.

The contract worth defending is the *step*: interval-valued data must render
flat across each interval and vertical at its edges, and must break — not
slope — across a gap. Everything else here guards the pieces that make a
coverage track line up with the gene model under it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.plot import ggplot_build
import ggtracks as ggt
from ggtracks.geom_coverage import _step_outline


# adjacent pair, then a gap, then a lone interval
COV = pd.DataFrame(
    {"xstart": [100, 150, 400], "xend": [150, 200, 500], "value": [5.0, 9.0, 3.0]}
)


def _built(layer, data=COV, extra=None):
    p = gg.ggplot(data, gg.aes(xstart="xstart", xend="xend", y="value")) + layer
    if extra is not None:
        p = p + extra
    return ggplot_build(p).data[0]


# --------------------------------------------------------------------------
# step outline
# --------------------------------------------------------------------------


def test_outline_is_flat_then_vertical():
    xs, ys = _step_outline(
        np.array([100.0, 150.0]), np.array([150.0, 200.0]),
        np.array([5.0, 9.0]), np.zeros(2),
    )
    pts = list(zip(xs, ys))
    # the shared edge appears twice, at both heights — a vertical riser
    assert (150.0, 5.0) in pts and (150.0, 9.0) in pts


def test_outline_drops_to_baseline_across_a_gap():
    """A gap is empty, not bridged: the profile must touch the baseline."""
    xs, ys = _step_outline(
        np.array([100.0, 400.0]), np.array([150.0, 500.0]),
        np.array([5.0, 3.0]), np.zeros(2),
    )
    pts = list(zip(xs, ys))
    assert (150.0, 0.0) in pts and (400.0, 0.0) in pts
    # and nothing is drawn at a height in between while crossing the gap
    inside = [y for x, y in pts if 150.0 < x < 400.0]
    assert not inside


def test_outline_respects_a_nonzero_baseline():
    xs, ys = _step_outline(
        np.array([100.0]), np.array([150.0]), np.array([5.0]), np.array([2.0])
    )
    assert ys[0] == 2.0 and ys[-1] == 2.0


def test_outline_sorts_unordered_input():
    xs, _ = _step_outline(
        np.array([400.0, 100.0]), np.array([500.0, 150.0]),
        np.array([3.0, 5.0]), np.zeros(2),
    )
    assert xs[0] == 100.0


# --------------------------------------------------------------------------
# geom
# --------------------------------------------------------------------------


def test_baseline_is_added_and_trains_the_y_axis():
    d = _built(ggt.geom_coverage())
    assert "ymin" in d.columns
    assert (d["ymin"] == 0.0).all()


def test_explicit_ymin_is_kept():
    data = COV.assign(base=1.0)
    p = (
        gg.ggplot(data, gg.aes(xstart="xstart", xend="xend", y="value", ymin="base"))
        + ggt.geom_coverage()
    )
    assert (ggplot_build(p).data[0]["ymin"] == 1.0).all()


@pytest.mark.parametrize("style", ["area", "step", "bar"])
def test_styles_render(style, tmp_path):
    p = (
        gg.ggplot(COV, gg.aes(xstart="xstart", xend="xend", y="value"))
        + ggt.geom_coverage(style=style)
        + ggt.theme_tracks()
    )
    out = tmp_path / f"{style}.png"
    gg.ggsave(str(out), p, width=3, height=1.5, dpi=72)
    assert out.stat().st_size > 0


def _chain(exc):
    """Flatten an exception and everything it was raised from."""
    out = []
    while exc is not None:
        out.append(exc)
        exc = exc.__cause__ or exc.__context__
    return out


def test_bad_style_fails_loud(tmp_path):
    """Style is checked at draw time, so the error surfaces on render."""
    p = (
        gg.ggplot(COV, gg.aes(xstart="xstart", xend="xend", y="value"))
        + ggt.geom_coverage(style="sideways")
    )
    with pytest.raises(Exception) as ei:
        gg.ggsave(str(tmp_path / "bad.png"), p, width=2, height=1, dpi=36)
    assert any("style" in str(e) for e in _chain(ei.value))


def test_default_aes_keep_the_fill_translucent_and_the_outline_opaque():
    d = _built(ggt.geom_coverage())
    assert d["alpha"].iloc[0] == pytest.approx(0.8)
    assert d["fill"].iloc[0] == d["colour"].iloc[0]
    assert d["linewidth"].iloc[0] == pytest.approx(0.3)


def test_groups_are_outlined_separately():
    data = pd.concat([COV.assign(g="a"), COV.assign(g="b", value=COV.value * 2)])
    p = (
        gg.ggplot(data, gg.aes(xstart="xstart", xend="xend", y="value", fill="g"))
        + ggt.geom_coverage()
    )
    assert ggplot_build(p).data[0]["group"].nunique() == 2


# --------------------------------------------------------------------------
# genomic axis integration
# --------------------------------------------------------------------------


def test_coverage_is_compressed_by_the_genomic_scale():
    """xstart/xend must ride the same transform as geom_range, or the signal
    would drift away from the gene model beneath it."""
    mapper = ggt.GenomicMapper.from_intervals(
        [(100, 200), (400, 500)], intron_scale=0.1, intron_min=20
    )
    d = _built(ggt.geom_coverage(), extra=ggt.scale_x_genomic(mapper))
    assert d["xstart"].max() == pytest.approx(
        mapper.to_display(400.0), abs=1e-6
    ) or d["xend"].max() == pytest.approx(mapper.to_display(500.0), abs=1e-6)
    # the compressed intron must not occupy its full genomic width
    assert d["xend"].max() < 400


def test_coverage_and_range_share_edges_under_compression():
    mapper = ggt.GenomicMapper.from_intervals([(100, 200), (400, 500)], intron_scale=0.1)
    exons = pd.DataFrame({"xstart": [100, 400], "xend": [200, 500], "y": [1.0, 1.0]})
    p = (
        gg.ggplot()
        + ggt.geom_range(gg.aes(xstart="xstart", xend="xend", y="y"), data=exons)
        + ggt.geom_coverage(
            gg.aes(xstart="xstart", xend="xend", y="value"), data=COV, inherit_aes=False
        )
        + ggt.scale_x_genomic(mapper)
    )
    built = ggplot_build(p)
    rng = built.data[0]
    assert rng["xmin"].min() == pytest.approx(mapper.to_display(100.0), abs=1e-6)


# --------------------------------------------------------------------------
# StatBinCoverage
# --------------------------------------------------------------------------


def test_binning_returns_the_requested_number_of_bins():
    d = _built(ggt.geom_coverage(stat=ggt.StatBinCoverage, bins=8))
    assert len(d) == 8


def test_sum_binning_preserves_the_integral():
    """``sum`` is the integral, so it must not depend on the bin count."""
    exact = float(((COV.xend - COV.xstart) * COV.value).sum())
    for bins in (3, 17, 64):
        d = _built(ggt.geom_coverage(stat=ggt.StatBinCoverage, bins=bins, summary="sum"))
        assert float(d.y.sum()) == pytest.approx(exact, rel=1e-9)


def test_mean_averages_over_covered_length_not_bin_width():
    """A half-empty bin reports the value that *was* measured rather than
    half of it — uncovered stretches are missing data, not zeros. (Same
    convention the bigWig summary statistics use.)"""
    data = pd.DataFrame({"xstart": [0], "xend": [50], "value": [10.0]})
    # single bin spanning twice the covered length
    d = _built(
        ggt.geom_coverage(stat=ggt.StatBinCoverage, bins=1),
        data=pd.concat([data, pd.DataFrame({"xstart": [100], "xend": [100], "value": [10.0]})]),
    )
    assert float(d.y.iloc[0]) == pytest.approx(10.0)


def test_mean_binning_is_stable_when_coverage_is_gapless():
    gapless = pd.DataFrame(
        {"xstart": [0, 100, 200], "xend": [100, 200, 300], "value": [1.0, 5.0, 3.0]}
    )
    exact = float(((gapless.xend - gapless.xstart) * gapless.value).sum())
    for bins in (3, 12, 60):
        d = _built(ggt.geom_coverage(stat=ggt.StatBinCoverage, bins=bins), data=gapless)
        got = float(((d.xend - d.xstart) * d.y).sum())
        assert got == pytest.approx(exact, rel=1e-9)


def test_bins_are_uniform_in_display_space():
    """Binning genomic coordinates would lavish bins on a compressed intron;
    binning what the scale produced keeps the columns even on screen."""
    mapper = ggt.GenomicMapper.from_intervals(
        [(100, 200), (400, 500)], intron_scale=0.05, intron_min=20
    )
    d = _built(
        ggt.geom_coverage(stat=ggt.StatBinCoverage, bins=10),
        extra=ggt.scale_x_genomic(mapper),
    )
    widths = (d["xend"] - d["xstart"]).to_numpy()
    assert np.allclose(widths, widths[0])


@pytest.mark.parametrize("summary", ["mean", "max", "min", "sum"])
def test_summaries_run(summary):
    d = _built(ggt.geom_coverage(stat=ggt.StatBinCoverage, bins=5, summary=summary))
    assert len(d) == 5
    assert np.isfinite(d["y"]).all()


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"bins": 0}, "positive int"),
        ({"bins": 2.5}, "positive int"),
        ({"summary": "median"}, "summary must be"),
    ],
)
def test_stat_validation_raises_rather_than_blanking_the_panel(kwargs, match):
    """Validation lives in setup_params on purpose: the build swallows
    exceptions thrown inside compute_group and hands back an empty frame,
    which would reach the user as a blank panel instead of an error.

    The build decorates the failure with its own RuntimeError, so the real
    cause is checked through the exception chain.
    """
    p = (
        gg.ggplot(COV, gg.aes(xstart="xstart", xend="xend", y="value"))
        + ggt.geom_coverage(stat=ggt.StatBinCoverage, **kwargs)
    )
    with pytest.raises(Exception) as ei:
        ggplot_build(p)
    chain = _chain(ei.value)
    assert any(isinstance(e, ValueError) and match in str(e) for e in chain)


def test_stat_bin_coverage_constructor():
    p = gg.ggplot(COV, gg.aes(xstart="xstart", xend="xend", y="value")) + \
        ggt.stat_bin_coverage(bins=6)
    assert len(ggplot_build(p).data[0]) == 6
