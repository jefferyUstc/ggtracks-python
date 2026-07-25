"""Tests for the cross-track highlight band."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.plot import ggplot_build
import ggtracks as ggt


MAPPER = ggt.GenomicMapper.from_intervals([(100, 200), (400, 500)])


def _cov(track):
    return pd.DataFrame(
        {"xstart": [100, 400], "xend": [200, 500], "value": [5.0, 3.0], "track": track}
    )


def _tracks(names=("a", "b")):
    return [
        ggt.Track(
            n,
            [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"), data=_cov(n))],
        )
        for n in names
    ]


def test_band_spans_the_full_panel_height():
    layer = ggt.geom_highlight(xstart=150, xend=190)
    p = gg.ggplot() + layer
    d = ggplot_build(p).data[0]
    assert np.isneginf(d["ymin"]).all() and np.isposinf(d["ymax"]).all()


def test_band_does_not_widen_the_y_scale():
    """Infinite bounds must fill the panel, not stretch it."""
    plain = ggt.plot_tracks(_tracks(("a",)), MAPPER, show=False)
    banded = ggt.plot_tracks(
        _tracks(("a",)), MAPPER, show=False,
        background=[ggt.geom_highlight(xstart=150, xend=190)],
    )
    a = ggplot_build(plain).layout.panel_params[0]["y_range"]
    b = ggplot_build(banded).layout.panel_params[0]["y_range"]
    assert np.allclose(np.asarray(a, float), np.asarray(b, float))


def test_band_reaches_every_track():
    p = ggt.plot_tracks(
        _tracks(), MAPPER, show=False,
        background=[ggt.geom_highlight(xstart=150, xend=190)],
    )
    built = ggplot_build(p)
    assert sorted(built.data[0]["PANEL"].unique()) == [1, 2]


def test_a_track_column_confines_the_band():
    """Carrying the facet variable opts out of the broadcast."""
    band = pd.DataFrame({"xstart": [150], "xend": [190], "track": ["b"]})
    p = ggt.plot_tracks(
        _tracks(), MAPPER, show=False, background=[ggt.geom_highlight(band)]
    )
    assert sorted(ggplot_build(p).data[0]["PANEL"].unique()) == [2]


def test_track_layers_still_do_not_leak():
    """The broadcast must not weaken per-track containment."""
    p = ggt.plot_tracks(
        _tracks(), MAPPER, show=False,
        background=[ggt.geom_highlight(xstart=150, xend=190)],
    )
    built = ggplot_build(p)
    assert sorted(built.data[1]["PANEL"].unique()) == [1]
    assert sorted(built.data[2]["PANEL"].unique()) == [2]


def test_background_layers_are_drawn_first():
    p = ggt.plot_tracks(
        _tracks(), MAPPER, show=False,
        background=[ggt.geom_highlight(xstart=150, xend=190)],
    )
    assert isinstance(p.layers[0].geom, gg.geom_rect().geom.__class__)


def test_multiple_regions_from_a_frame():
    band = pd.DataFrame({"xstart": [110, 420], "xend": [140, 460]})
    d = ggplot_build(gg.ggplot() + ggt.geom_highlight(band)).data[0]
    assert len(d) == 2


def test_extra_aesthetics_can_be_mapped():
    band = pd.DataFrame(
        {"xstart": [110, 420], "xend": [140, 460], "reason": ["peak", "exon"]}
    )
    layer = ggt.geom_highlight(band, gg.aes(fill="reason"))
    d = ggplot_build(gg.ggplot() + layer).data[0]
    assert d["fill"].nunique() == 2


def test_band_is_compressed_by_the_genomic_scale():
    mapper = ggt.GenomicMapper.from_intervals(
        [(100, 200), (400, 500)], intron_scale=0.1, intron_min=20
    )
    p = gg.ggplot() + ggt.geom_highlight(xstart=100, xend=500) + ggt.scale_x_genomic(mapper)
    d = ggplot_build(p).data[0]
    assert d["xmax"].iloc[0] == pytest.approx(mapper.to_display(500.0), abs=1e-6)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "not both, not neither"),
        ({"data": pd.DataFrame({"a": [1]}), "xstart": 1, "xend": 2}, "not both"),
        ({"data": pd.DataFrame({"a": [1]})}, "missing column"),
    ],
)
def test_argument_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ggt.geom_highlight(**kwargs)


def test_renders(tmp_path):
    p = ggt.plot_tracks(
        _tracks(), MAPPER, show=False,
        background=[ggt.geom_highlight(xstart=150, xend=190)],
    )
    out = tmp_path / "hl.png"
    gg.ggsave(str(out), p, width=4, height=p.fig_height, dpi=72)
    assert out.stat().st_size > 0
