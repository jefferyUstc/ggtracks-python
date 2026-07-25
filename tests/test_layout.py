"""Tests for the grid-unit layout baseline.

The contract: panel sizes may be relative *or* absolute grid units, and the
figure height is **measured** off the gtable rather than assumed. Measuring
matters because the non-panel overhead ("chrome" — axes, title, strips,
margins) grows with ``base_size``; a hard-coded constant is right at one
typography and clips the figure at another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.plot import ggplot_build
from grid_py import Unit
import ggtracks as ggt
from ggtracks._render import natural_height
from ggtracks.plot_tracks import _panel_rows


MAPPER = ggt.GenomicMapper.from_intervals([(1000, 1100), (3000, 3100)])
GM = pd.DataFrame({
    "xstart": [1000, 3000], "xend": [1100, 3100], "y": [1.0, 1.0],
    "feature": ["CDS", "CDS"], "track": "gene model",
})


def _tracks(height):
    return [ggt.Track(
        "gene model",
        [ggt.geom_range(gg.aes(xstart="xstart", xend="xend", y="y"), data=GM)],
        height=height,
    )]


# --------------------------------------------------------------------------
# natural_height
# --------------------------------------------------------------------------


def test_relative_panels_measure_chrome_only():
    """A plot whose panels are relative has no absolute panel height, so the
    measurement is exactly the fixed overhead."""
    p = ggt.plot_tracks(_tracks(1.0), MAPPER, show=False)
    assert 0.2 < natural_height(p) < 1.5


def test_chrome_grows_with_base_size():
    """The bug a hard-coded constant hides: overhead is not constant."""
    small = natural_height(ggt.plot_tracks(_tracks(1.0), MAPPER, show=False, base_size=8))
    large = natural_height(ggt.plot_tracks(_tracks(1.0), MAPPER, show=False, base_size=20))
    assert large > small * 1.5


def test_absolute_panels_are_included_in_the_measurement():
    rel = natural_height(ggt.plot_tracks(_tracks(1.0), MAPPER, show=False))
    absolute = natural_height(
        ggt.plot_tracks(_tracks(Unit(2, "inches")), MAPPER, show=False)
    )
    assert absolute == pytest.approx(rel + 2.0, abs=1e-6)


# --------------------------------------------------------------------------
# _panel_rows
# --------------------------------------------------------------------------


def test_all_relative_stays_a_plain_list():
    rows, inches = _panel_rows([1.0, 0.5])
    assert rows == [1.0, 0.5]
    assert inches == pytest.approx(1.5)


def test_any_unit_promotes_everything_to_units():
    from grid_py import is_unit

    rows, inches = _panel_rows([1.0, Unit(0.5, "inches")])
    assert is_unit(rows)
    # only the relative entry counts toward the inch allowance
    assert inches == pytest.approx(1.0)


def test_all_absolute_needs_no_inch_allowance():
    rows, inches = _panel_rows([Unit(1, "cm"), Unit(2, "cm")])
    assert inches == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Track.height
# --------------------------------------------------------------------------


def test_float_height_is_coerced_but_unit_is_preserved():
    from grid_py import is_unit

    assert ggt.Track("t", [], height=2).height == 2.0
    assert is_unit(ggt.Track("t", [], height=Unit(3, "lines")).height)


def test_figure_height_tracks_absolute_panel_size():
    """Doubling an absolute panel adds exactly that much figure height."""
    one = ggt.plot_tracks(_tracks(Unit(1, "inches")), MAPPER, show=False)
    two = ggt.plot_tracks(_tracks(Unit(2, "inches")), MAPPER, show=False)
    assert two.fig_height == pytest.approx(one.fig_height + 1.0, abs=1e-6)


def test_relative_height_still_counts_as_inches():
    """Backward compatibility: a bare number keeps its historic meaning."""
    one = ggt.plot_tracks(_tracks(1.0), MAPPER, show=False)
    three = ggt.plot_tracks(_tracks(3.0), MAPPER, show=False)
    assert three.fig_height == pytest.approx(one.fig_height + 2.0, abs=1e-6)


def test_mixed_relative_and_absolute_renders(tmp_path):
    tracks = [
        ggt.Track("gene model",
                  [ggt.geom_range(gg.aes(xstart="xstart", xend="xend", y="y"), data=GM)],
                  height=Unit(1.2, "cm")),
        ggt.Track("blank",
                  [gg.geom_blank(gg.aes(x="xstart", y="y"),
                                 data=GM.assign(track="blank"))],
                  height=1.0),
    ]
    out = tmp_path / "mixed.png"
    ggt.plot_tracks(tracks, MAPPER, show=False, save=str(out))
    assert out.stat().st_size > 0


# --------------------------------------------------------------------------
# finalize_gg
# --------------------------------------------------------------------------


def test_finalize_height_none_measures():
    p = ggt.plot_tracks(_tracks(Unit(1.5, "inches")), MAPPER, show=False)
    measured = natural_height(p)
    from ggtracks import finalize_gg

    finalize_gg(p, show=False, height=None)
    assert p.fig_height == pytest.approx(measured, abs=1e-6)


def test_finalize_explicit_height_wins():
    from ggtracks import finalize_gg

    p = ggt.plot_tracks(_tracks(1.0), MAPPER, show=False)
    finalize_gg(p, show=False, height=7.25)
    assert p.fig_height == pytest.approx(7.25)


# --------------------------------------------------------------------------
# plot_tracks must not write through to the caller's layers
# --------------------------------------------------------------------------


def _two_tracks():
    def cov(name):
        return pd.DataFrame(
            {"xstart": [1000, 3000], "xend": [1100, 3100],
             "value": [5.0, 3.0], "track": name}
        )
    return [
        ggt.Track(n, [ggt.geom_coverage(
            gg.aes(xstart="xstart", xend="xend", y="value"), data=cov(n))])
        for n in ("A", "B")
    ]


def _row_order(plot):
    layout = ggplot_build(plot).layout.layout
    return list(layout.sort_values("ROW")["track"].astype(str))


def test_two_figures_from_one_track_list_keep_their_own_order():
    """Ordering the facet key rewrites layer data. Writing that through to
    the caller's layers would make the first figure re-render with the
    second's order — silently, since both share the layer objects."""
    tracks = _two_tracks()
    first = ggt.plot_tracks(tracks, MAPPER, show=False, track_order=["A", "B"])
    second = ggt.plot_tracks(tracks, MAPPER, show=False, track_order=["B", "A"])
    assert _row_order(first) == ["A", "B"]
    assert _row_order(second) == ["B", "A"]


def test_the_callers_layers_are_left_alone():
    tracks = _two_tracks()
    before = tracks[0].layers[0].data
    ggt.plot_tracks(tracks, MAPPER, show=False)
    assert tracks[0].layers[0].data is before
    assert before["track"].dtype == object


def test_track_has_a_useful_repr():
    text = repr(ggt.Track("coverage", [], height=2.0, y_limits=(0.0, 10.0)))
    assert "coverage" in text and "y_limits" in text
