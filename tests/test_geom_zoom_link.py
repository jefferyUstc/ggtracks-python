"""Tests for the focus + context connector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.plot import ggplot_build
import ggtracks as ggt


MAPPER = ggt.GenomicMapper.from_intervals([(1000, 5000)], collapse_introns=False)
COV = pd.DataFrame(
    {
        "xstart": np.arange(1000, 5000, 100),
        "xend": np.arange(1100, 5100, 100),
        "value": np.linspace(1, 10, 40),
        "track": "overview",
    }
)


def _figure(**link_kwargs):
    tracks = [
        ggt.Track(
            "overview",
            [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"), data=COV)],
        ),
        ggt.Track(
            "zoom",
            [ggt.geom_zoom_link(xstart=2600, xend=3000, track="zoom", **link_kwargs)],
            height=0.35,
            y_breaks=[0.0],
            y_labels=[""],
        ),
    ]
    return ggt.plot_tracks(tracks, MAPPER)


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({}, "not both, not neither"),
        ({"xstart": 1}, "give both xstart and xend"),
        (
            {"data": pd.DataFrame({"xstart": [1], "xend": [2]}), "xstart": 1, "xend": 2},
            "not both",
        ),
    ],
)
def test_argument_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ggt.geom_zoom_link(**kwargs)


def test_track_shortcut_only_applies_to_the_generated_frame():
    frame = pd.DataFrame({"xstart": [1], "xend": [2]})
    with pytest.raises(ValueError, match="put the column in `data`"):
        ggt.geom_zoom_link(data=frame, track="zoom")


def test_colours_must_be_two_endpoints():
    tracks = [
        ggt.Track(
            "zoom",
            [ggt.geom_zoom_link(xstart=2600, xend=3000, track="zoom",
                                colours=("#fff", "#000", "#111"))],
        )
    ]
    with pytest.raises(Exception) as ei:
        # plot_tracks measures the figure, which draws it.
        ggt.plot_tracks(tracks, MAPPER)
    chain, exc = [], ei.value
    while exc is not None:
        chain.append(str(exc))
        exc = exc.__cause__ or exc.__context__
    assert any("two endpoints" in c for c in chain)


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def test_link_stays_in_its_own_track():
    """Without the facet column the band would be draped over the data
    tracks as well."""
    built = ggplot_build(_figure())
    link = built.data[1]
    assert sorted(link["PANEL"].unique()) == [2]


def test_forgetting_the_track_column_fails_loud():
    """Without it the band is repeated into every panel instead of sitting in
    its own row — visibly wrong, and easy to do by accident."""
    tracks = [
        ggt.Track(
            "overview",
            [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"), data=COV)],
        ),
        ggt.Track("zoom", [ggt.geom_zoom_link(xstart=2600, xend=3000)], height=0.3),
    ]
    with pytest.raises(ValueError, match="drawn on every panel"):
        ggt.plot_tracks(tracks, MAPPER)


def test_a_trackless_layer_still_broadcasts_from_the_background():
    """The broadcast itself is intact — it is just not how a Track works."""
    tracks = [
        ggt.Track(
            "overview",
            [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"), data=COV)],
        ),
        ggt.Track(
            "second",
            [
                ggt.geom_coverage(
                    gg.aes(xstart="xstart", xend="xend", y="value"),
                    data=COV.assign(track="second"),
                )
            ],
        ),
    ]
    p = ggt.plot_tracks(
        tracks, MAPPER,
        background=[ggt.geom_zoom_link(xstart=2600, xend=3000)],
    )
    assert len(set(ggplot_build(p).data[0]["PANEL"].unique())) == 2


def test_region_rides_the_shared_genomic_scale():
    """The taper lands on the region because the connector shares the
    overview's x scale, not because anything was measured by hand."""
    built = ggplot_build(_figure())
    link = built.data[1]
    assert link["xstart"].iloc[0] == pytest.approx(MAPPER.to_display(2600), abs=1e-6)
    assert link["xend"].iloc[0] == pytest.approx(MAPPER.to_display(3000), abs=1e-6)


def test_a_y_column_is_supplied_so_the_panel_exists():
    built = ggplot_build(_figure())
    assert "y" in built.data[1].columns


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("flip", [False, True])
def test_renders_both_orientations(flip, tmp_path):
    out = tmp_path / f"zoom_{flip}.png"
    p = _figure(flip=flip)
    gg.ggsave(str(out), p, width=5, height=p.fig_height, dpi=72)
    assert out.stat().st_size > 0


def test_renders_at_very_different_aspect_ratios(tmp_path):
    """Panel-relative geometry is what keeps the taper attached to the
    region when the figure is resized."""
    p = _figure()
    for w in (3.0, 12.0):
        out = tmp_path / f"aspect_{w}.png"
        gg.ggsave(str(out), p, width=w, height=p.fig_height, dpi=72)
        assert out.stat().st_size > 0


def test_default_gradient_runs_pale_to_deep():
    from ggtracks.geom_zoom_link import geom_zoom_link
    import inspect

    default = inspect.signature(geom_zoom_link).parameters["colours"].default
    assert tuple(default) == ("#E4E8EC", ggt.FEATURE_COLOURS["intron"])


def test_works_alongside_a_highlight(tmp_path):
    p = ggt.plot_tracks(
        [
            ggt.Track(
                "overview",
                [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"), data=COV)],
            ),
            ggt.Track(
                "zoom",
                [ggt.geom_zoom_link(xstart=2600, xend=3000, track="zoom")],
                height=0.3,
                y_breaks=[0.0],
                y_labels=[""],
            ),
        ],
        MAPPER,
        background=[ggt.geom_highlight(xstart=2600, xend=3000, fill="#E74C3C", alpha=0.15)],
    )
    out = tmp_path / "combo.png"
    gg.ggsave(str(out), p, width=5, height=p.fig_height, dpi=72)
    assert out.stat().st_size > 0
