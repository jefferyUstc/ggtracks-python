"""Tests for multi-locus figures: one coordinate system per facet column."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.plot import ggplot_build
import ggtracks as ggt


ALPHA = ggt.GenomicMapper.from_intervals(
    [(1000, 1100), (3000, 3100)], intron_mode="clamp", target_gap_width=50
)
BETA = ggt.GenomicMapper.from_intervals(
    [(90_000, 90_200), (95_000, 95_300)], intron_mode="clamp", target_gap_width=50
)
MAPPERS = {"Alpha": ALPHA, "Beta": BETA}


def _data(locus_col=True):
    df = pd.DataFrame(
        {
            "xstart": [1000, 3000, 90_000, 95_000],
            "xend": [1100, 3100, 90_200, 95_300],
            "value": [5.0, 9.0, 3.0, 7.0],
            "track": "coverage",
            "locus": ["Alpha", "Alpha", "Beta", "Beta"],
        }
    )
    return df if locus_col else df.drop(columns="locus")


def _tracks(locus_col=True):
    return [
        ggt.Track(
            "coverage",
            [
                ggt.geom_coverage(
                    gg.aes(xstart="xstart", xend="xend", y="value"),
                    data=_data(locus_col),
                )
            ],
        )
    ]


# --------------------------------------------------------------------------
# argument handling
# --------------------------------------------------------------------------


def test_exactly_one_mapper_argument_is_required():
    with pytest.raises(ValueError, match="exactly one"):
        ggt.plot_tracks(_tracks())
    with pytest.raises(ValueError, match="exactly one"):
        ggt.plot_tracks(_tracks(), ALPHA, mappers=MAPPERS)


def test_a_dict_in_the_positional_slot_points_at_mappers():
    with pytest.raises(TypeError, match="mappers="):
        ggt.plot_tracks(_tracks(), MAPPERS)


def test_empty_mappers_is_rejected():
    with pytest.raises(ValueError, match="mappers is empty"):
        ggt.plot_tracks(_tracks(), mappers={})


def test_genomic_xlim_is_single_locus_only():
    with pytest.raises(ValueError, match="single locus"):
        ggt.plot_tracks(
            _tracks(), mappers=MAPPERS, genomic_xlim=(1000, 2000))


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def test_grid_is_tracks_by_loci():
    tracks = _tracks() + [
        ggt.Track(
            "gene model",
            [
                ggt.geom_range(
                    gg.aes(xstart="xstart", xend="xend", y="y"),
                    data=_data().assign(track="gene model", y=1.0),
                )
            ],
        )
    ]
    built = ggplot_build(ggt.plot_tracks(tracks, mappers=MAPPERS))
    assert len(built.layout.panel_params) == 4


def test_column_order_follows_the_mapping_order():
    built = ggplot_build(ggt.plot_tracks(_tracks(), mappers=MAPPERS))
    layout = built.layout.layout
    ordered = layout.sort_values("COL")["locus"].astype(str).tolist()
    assert ordered == ["Alpha", "Beta"]


def test_each_column_gets_its_own_coordinate_system():
    """The whole point: two loci that share no genomic range still plot."""
    built = ggplot_build(ggt.plot_tracks(_tracks(), mappers=MAPPERS))
    spans = [
        float(np.ptp(np.asarray(pp["x_range"], dtype=float)))
        for pp in built.layout.panel_params
    ]
    assert len(set(np.round(spans, 3))) == 2
    # each span tracks its own mapper's display extent
    assert max(spans) > min(spans)


def test_tick_labels_are_each_locus_own_genomic_coordinates():
    built = ggplot_build(ggt.plot_tracks(_tracks(), mappers=MAPPERS))
    labels = [pp.get("x_labels") for pp in built.layout.panel_params]
    joined = [" ".join(l or []) for l in labels]
    assert any("kb" in j for j in joined)
    # Alpha is around 1-3 kb, Beta around 90-95 kb: no shared label text
    assert joined[0] != joined[1]


def test_the_global_x_scale_stays_untransformed():
    """A transformed global scale would compress the data a second time,
    on top of the per-panel transform."""
    sc = ggt.base_x_scale()
    assert "xstart" in sc.aesthetics
    values = np.array([1000.0, 3000.0])
    assert np.allclose(np.asarray(sc.transform(values), dtype=float), values)


def test_compression_is_applied_per_panel():
    """Each column's data is compressed by *its own* mapper — faceting has
    consumed ``locus`` into ``PANEL`` by this point."""
    built = ggplot_build(ggt.plot_tracks(_tracks(), mappers=MAPPERS))
    d = built.data[0]
    reach = d.groupby("PANEL", observed=True)["xend"].max().sort_index().to_numpy()
    assert reach[0] == pytest.approx(ALPHA.display_extent[1], abs=1e-6)
    assert reach[1] == pytest.approx(BETA.display_extent[1], abs=1e-6)


# --------------------------------------------------------------------------
# facet-key validation
# --------------------------------------------------------------------------


def test_unknown_locus_fails_loud():
    """Unmatched facet values are dropped silently by the build; that would
    delete data from the figure without a word."""
    bad = _data()
    bad.loc[0, "locus"] = "Gamma"
    tracks = [
        ggt.Track(
            "coverage",
            [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"), data=bad)],
        )
    ]
    with pytest.raises(ValueError, match="locus value.*Gamma"):
        ggt.plot_tracks(tracks, mappers=MAPPERS)


def test_unknown_track_fails_loud():
    bad = _data()
    bad["track"] = "typo"
    tracks = [
        ggt.Track(
            "coverage",
            [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"), data=bad)],
        )
    ]
    with pytest.raises(ValueError, match="track value.*typo"):
        ggt.plot_tracks(tracks, mappers=MAPPERS)


def test_a_layer_without_locus_spans_every_column():
    p = ggt.plot_tracks(
        _tracks(),
        mappers=MAPPERS,
        background=[ggt.geom_highlight(xstart=1000, xend=1100)],
    )
    built = ggplot_build(p)
    assert len(set(built.data[0]["PANEL"].unique())) == 2


# --------------------------------------------------------------------------
# single-locus path is untouched
# --------------------------------------------------------------------------


def test_single_locus_still_works_positionally():
    p = ggt.plot_tracks(_tracks(locus_col=False), ALPHA)
    assert len(ggplot_build(p).layout.panel_params) == 1


def test_renders(tmp_path):
    p = ggt.plot_tracks(_tracks(), mappers=MAPPERS, title="two loci")
    out = tmp_path / "multi.png"
    gg.ggsave(str(out), p, width=6, height=p.fig_height, dpi=72)
    assert out.stat().st_size > 0
