"""Tests for the ggtracks track grammar (geoms + stats).

Covers the genomic coordinate transform, GeomRange compression
(both-side vs the mapper), to_intron / StatIntron, GeomIntron strand
arrows, and the half-range orientation param.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.plot import ggplot_build
from ggtracks import GenomicMapper
import ggtracks as G


EXONS = pd.DataFrame({
    "xstart": [1000, 3000, 5000],
    "xend": [1100, 3100, 5100],
    "tx": ["T1", "T1", "T1"],
    "feature": ["exon", "exon", "exon"],
})
MAPPER = GenomicMapper.from_intervals(
    [(1000, 1100), (3000, 3100), (5000, 5100)], intron_scale=0.1, intron_min=20
)


def test_to_intron_handcomputed():
    introns = G.to_intron(EXONS, group_var="tx")
    assert list(zip(introns["xstart"], introns["xend"])) == [(1100, 3000), (3100, 5000)]


def test_to_intron_drops_adjacent_exons():
    ex = pd.DataFrame({"xstart": [10, 21], "xend": [20, 30], "tx": ["a", "a"]})
    assert G.to_intron(ex, group_var="tx").empty


def test_stat_intron_matches_to_intron():
    p = (gg.ggplot(EXONS, gg.aes(xstart="xstart", xend="xend", y="tx"))
         + G.geom_intron(stat=G.StatIntron))
    d = ggplot_build(p).data[0]
    got = sorted(zip(np.round(d["x"]).astype(int), np.round(d["xend"]).astype(int)))
    assert got == [(1100, 3000), (3100, 5000)]


def test_geom_range_compressed_bothside():
    p = (gg.ggplot(EXONS, gg.aes(xstart="xstart", xend="xend", y="tx"))
         + G.geom_range()
         + G.scale_x_genomic(MAPPER))
    d = ggplot_build(p).data[0]
    exp_min = np.sort(MAPPER.to_display_array(EXONS["xstart"].values))
    exp_max = np.sort(MAPPER.to_display_array(EXONS["xend"].values))
    assert np.allclose(np.sort(d["xmin"].values), exp_min, atol=1e-6)
    assert np.allclose(np.sort(d["xmax"].values), exp_max, atol=1e-6)
    w = np.sort(d["xmax"].values) - np.sort(d["xmin"].values)
    assert np.allclose(w, 100.0, atol=1e-6)


def test_geom_range_height_and_orientation():
    p = gg.ggplot(EXONS, gg.aes(xstart="xstart", xend="xend", y="tx")) + G.geom_range()
    d = ggplot_build(p).data[0]
    assert np.allclose(d["ymax"].values - d["ymin"].values, 0.5)
    p2 = (gg.ggplot(EXONS, gg.aes(xstart="xstart", xend="xend", y="tx"))
          + G.geom_range(orientation="top"))
    d2 = ggplot_build(p2).data[0]
    assert np.allclose(d2["ymax"].values - d2["ymin"].values, 0.25)


def test_geom_range_bad_orientation_fails_loud():
    p = (gg.ggplot(EXONS, gg.aes(xstart="xstart", xend="xend", y="tx"))
         + G.geom_range(orientation="sideways"))
    with pytest.raises(Exception) as ei:
        ggplot_build(p)
    msg = str(ei.value) + str(getattr(ei.value, "__cause__", ""))
    assert "orientation" in msg


def test_geom_intron_renders_both_strands(tmp_path):
    introns = G.to_intron(EXONS, group_var="tx")
    introns_minus = introns.assign(strand="-")
    introns_plus = introns.assign(strand="+")
    for tag, idf in (("plus", introns_plus), ("minus", introns_minus)):
        p = (gg.ggplot(EXONS, gg.aes(xstart="xstart", xend="xend", y="tx"))
             + G.geom_range()
             + G.geom_intron(data=idf, mapping=gg.aes(xstart="xstart", xend="xend",
                                                      y="tx", strand="strand"),
                             inherit_aes=False)
             + G.scale_x_genomic(MAPPER))
        gg.ggsave(str(tmp_path / f"{tag}.png"), p, width=6, height=2, dpi=80)
    assert (tmp_path / "plus.png").stat().st_size > 0
    assert (tmp_path / "minus.png").stat().st_size > 0


def test_geom_intron_chevron_style(tmp_path):
    introns = G.to_intron(EXONS, group_var="tx")
    p = (gg.ggplot(EXONS, gg.aes(xstart="xstart", xend="xend", y="tx"))
         + G.geom_range()
         + G.geom_intron(data=introns, mapping=gg.aes(xstart="xstart", xend="xend", y="tx"),
                         inherit_aes=False, style="chevron")
         + G.scale_x_genomic(MAPPER))
    gg.ggsave(str(tmp_path / "chev.png"), p, width=6, height=2, dpi=80)
    assert (tmp_path / "chev.png").stat().st_size > 0


def test_coord_genomic_xlim_in_display_space():
    comps = G.coord_genomic(MAPPER, genomic_xlim=(3000, 5100))
    assert len(comps) == 2
    from ggplot2_py.coord import CoordCartesian
    coord = comps[1]
    assert isinstance(coord, CoordCartesian)


def test_track_palettes():
    assert len(G.track_palettes("stallion")) == 20
    assert G.track_palettes("zissou", n=3) == ["#3B9AB2", "#78B7C5", "#EBCC2A"]
    with pytest.raises(KeyError):
        G.track_palettes("nope")


def test_plot_tracks_no_facet_leak():
    """Each track's layer must render only in its own panel (regression for
    the draw_layer per-panel null-padding fix)."""
    import ggplot2_py as gg
    cov = pd.DataFrame({"x": MAPPER.to_genomic_array(np.linspace(*MAPPER.display_extent, 20)),
                        "depth": np.r_[np.linspace(1, 30, 10), np.linspace(30, 1, 10)],
                        "track": "coverage"})
    gm = pd.DataFrame({"xstart": [1000, 3000, 5000], "xend": [1100, 3100, 5100],
                       "y": [1.0, 1.0, 1.0], "feature": ["CDS"] * 3, "track": "gene model"})
    tracks = [
        G.Track("coverage", [gg.geom_area(gg.aes(x="x", y="depth"), data=cov, fill="black")],
                height=1.0),
        G.Track("gene model",
                [G.geom_range(gg.aes(xstart="xstart", xend="xend", y="y", fill="feature"),
                              data=gm, height=0.5)],
                height=0.5, new_scale="fill"),
    ]
    p = G.plot_tracks(tracks, MAPPER, track_order=["coverage", "gene model"], show=False)
    built = ggplot_build(p)
    panel_of = {}
    for d in built.data:
        if d is not None and "PANEL" in d.columns and len(d):
            panel_of[id(d)] = set(d["PANEL"].unique())
    panels_per_layer = sorted(panel_of.values(), key=lambda s: min(s))
    assert panels_per_layer[0] == {1} and panels_per_layer[1] == {2}
