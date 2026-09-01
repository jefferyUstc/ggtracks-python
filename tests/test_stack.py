"""Tests for vstack_gg — the vertical composition path.

``plot_tracks`` composes rows that share one x scale; these tests pin
the other case: rows with deliberately different x domains (overview vs
zoom, chromosome vs locus), which facets cannot express.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
import ggtracks as ggt

EXONS = pd.DataFrame({"xstart": [1000, 4000], "xend": [1600, 4400], "y": 1.0})
COV = pd.DataFrame({"xstart": np.arange(1000, 4400, 200)})
COV["xend"] = COV.xstart + 200
COV["y"] = np.linspace(1.0, 30.0, len(COV))


def gene_track():
    return ggt.Track(
        "gene",
        [ggt.geom_range(gg.aes(xstart="xstart", xend="xend", y="y"),
                        data=EXONS.assign(track="gene"))],
        height=0.4, y_breaks=[1.0], y_labels=[""],
    )


def cov_track():
    return ggt.Track(
        "cov",
        [ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="y"),
                           data=COV.assign(track="cov"))],
    )


@pytest.fixture(scope="module")
def mapper():
    return ggt.GenomicMapper.from_intervals(zip(EXONS.xstart, EXONS.xend))


def test_stack_carries_the_summed_display_size(mapper):
    p1 = ggt.plot_tracks([cov_track(), gene_track()], mapper)
    p2 = ggt.plot_tracks([gene_track()], mapper)
    stack = ggt.vstack_gg([p1, p2])
    assert stack.fig_height == pytest.approx(p1.fig_height + p2.fig_height)
    assert stack.fig_width == pytest.approx(max(p1.fig_width, p2.fig_width))
    assert hasattr(stack, "to_gtable") and hasattr(stack, "_repr_png_")


def test_zoom_figure_saves_as_one_artifact(mapper, tmp_path):
    """Overview + zoom link + magnified detail — one file, panels aligned."""
    link = ggt.Track(
        "zoom",
        [ggt.geom_zoom_link(xstart=3800, xend=4600, track="zoom")],
        height=0.3, y_breaks=[0.0], y_labels=[""],
    )
    overview = ggt.plot_tracks([cov_track(), gene_track(), link], mapper)
    detail = ggt.plot_tracks([cov_track(), gene_track()], mapper,
                             genomic_xlim=(3800, 4600))
    out = tmp_path / "zoom.png"
    ggt.vstack_gg([overview, detail], save=out, dpi=72)
    assert out.stat().st_size > 0


def test_mixed_plain_and_faceted_rows_align(mapper, tmp_path):
    """An ideogram strip (plain ggplot, whole-chromosome x) above a faceted
    track figure — the row layout facets cannot express."""
    bands = pd.DataFrame({
        "xstart": [1, 4_000_000, 8_000_000],
        "xend": [4_000_000, 8_000_000, 12_000_000],
        "y": 1.0,
        "stain": ["gneg", "acen", "gpos50"],
    })
    ctx = (gg.ggplot()
           + ggt.geom_ideogram(gg.aes(xstart="xstart", xend="xend", y="y",
                                      stain="stain", fill="stain"),
                               data=bands)
           + ggt.scale_fill_giemsa() + ggt.theme_tracks()
           + gg.labs(x="", y=""))
    main = ggt.plot_tracks([cov_track(), gene_track()], mapper)
    out = tmp_path / "ctx.png"
    stack = ggt.vstack_gg([ctx, main], heights=[0.6, main.fig_height],
                          save=out, dpi=72)
    assert out.stat().st_size > 0
    assert stack.fig_height == pytest.approx(0.6 + main.fig_height)


def test_empty_stack_fails_loud():
    with pytest.raises(ValueError, match="no plots"):
        ggt.vstack_gg([])


def test_height_count_must_match(mapper):
    p = ggt.plot_tracks([gene_track()], mapper)
    with pytest.raises(ValueError, match="heights for"):
        ggt.vstack_gg([p], heights=[1.0, 2.0])


def test_heights_must_be_positive(mapper):
    p = ggt.plot_tracks([gene_track()], mapper)
    with pytest.raises(ValueError, match="positive"):
        ggt.vstack_gg([p], heights=[0.0])


def test_default_heights_need_measured_plots(mapper):
    class Bare:
        pass

    p = ggt.plot_tracks([gene_track()], mapper)
    with pytest.raises(ValueError, match="fig_height"):
        ggt.vstack_gg([p, Bare()])
