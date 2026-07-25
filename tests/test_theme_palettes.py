"""Tests for the visual baseline: ``theme_tracks`` knobs and signal ramps.

Themes are hard to assert on pixel-for-pixel, so these tests pin the
*contract* instead: the knobs exist, they are reversible, the ramp
interpolates rather than recycles, and a plot carrying the theme still
renders.
"""

from __future__ import annotations

import pandas as pd
import pytest

import ggplot2_py as gg
import ggtracks as ggt


DF = pd.DataFrame({"x": [1, 2, 3], "y": [1.0, 3.0, 2.0]})


def _render(theme, tmp_path, tag):
    p = gg.ggplot(DF, gg.aes(x="x", y="y")) + gg.geom_point() + theme
    out = tmp_path / f"{tag}.png"
    gg.ggsave(str(out), p, width=3, height=2, dpi=72)
    return out


# --------------------------------------------------------------------------
# theme_tracks
# --------------------------------------------------------------------------


def test_default_theme_renders(tmp_path):
    out = _render(ggt.theme_tracks(), tmp_path, "default")
    assert out.stat().st_size > 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"border_colour": None},
        {"border_colour": "grey20"},
        {"major_grid": True},
        {"minor_grid": True},
        {"panel_spacing": gg.unit(6, "pt")},
        {"base_size": 14},
    ],
)
def test_every_knob_still_renders(kwargs, tmp_path):
    """Each knob is independently exercisable and reversible."""
    out = _render(ggt.theme_tracks(**kwargs), tmp_path, "knob")
    assert out.stat().st_size > 0


def _pt(unit):
    """Resolve a grid Unit to points (``valueOnly`` yields an array)."""
    import numpy as np
    from grid_py import convert_height

    return float(np.asarray(convert_height(unit, "points", valueOnly=True)).ravel()[0])


def test_panel_spacing_scales_with_base_size():
    """Spacing is derived from base_size, not a fixed measure."""
    small = _pt(ggt.theme_tracks(base_size=8)["panel.spacing"])
    large = _pt(ggt.theme_tracks(base_size=16)["panel.spacing"])
    assert large == pytest.approx(2 * small)


def test_explicit_panel_spacing_wins_over_base_size():
    th = ggt.theme_tracks(base_size=16, panel_spacing=gg.unit(3, "pt"))
    assert _pt(th["panel.spacing"]) == pytest.approx(3.0)


def test_border_colour_none_blanks_the_border():
    blank = ggt.theme_tracks(border_colour=None)["panel.border"]
    assert isinstance(blank, type(gg.element_blank()))


def test_default_border_is_light():
    """grey80 default rather than theme_bw's grey20 — the point of the change."""
    assert ggt.PANEL_BORDER_COLOUR.lower() == "#cccccc"
    assert str(ggt.theme_tracks()["panel.border"].colour).lower() == "#cccccc"


def test_grids_are_off_by_default():
    th = ggt.theme_tracks()
    blank = type(gg.element_blank())
    assert isinstance(th["panel.grid.major"], blank)
    assert isinstance(th["panel.grid.minor"], blank)


# --------------------------------------------------------------------------
# signal_palette
# --------------------------------------------------------------------------


def test_endpoints_by_default():
    assert ggt.signal_palette("grey") == list(ggt.SIGNAL_PALETTES["grey"])


def test_single_shade_is_the_dark_end():
    """A lone coverage track should be prominent, not faint."""
    assert ggt.signal_palette("grey", n=1) == [ggt.SIGNAL_PALETTES["grey"][1]]


def test_two_shades_run_light_to_dark():
    low, high = ggt.signal_palette("grey", n=2)
    assert low.lower() == ggt.SIGNAL_PALETTES["grey"][0].lower()
    assert high.lower() == ggt.SIGNAL_PALETTES["grey"][1].lower()


def test_interpolates_rather_than_recycles():
    """Unlike a qualitative palette, a ramp has no length limit and never
    repeats a colour."""
    cols = ggt.signal_palette("blue", n=9)
    assert len(cols) == 9
    assert len(set(c.lower() for c in cols)) == 9


def test_monotonic_darkening():
    def lum(hex_colour):
        h = hex_colour.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    lums = [lum(c) for c in ggt.signal_palette("grey", n=6)]
    assert all(a > b for a, b in zip(lums, lums[1:]))


def test_unknown_ramp_fails_loud():
    with pytest.raises(KeyError, match="unknown ramp"):
        ggt.signal_palette("nope")


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_bad_n_fails_loud(bad):
    with pytest.raises(ValueError, match="positive int"):
        ggt.signal_palette("grey", n=bad)


def test_qualitative_and_sequential_are_distinct_families():
    """Guards the documented split: categories vs intensity."""
    assert set(ggt.TRACK_PALETTES) & set(ggt.SIGNAL_PALETTES) == set()
