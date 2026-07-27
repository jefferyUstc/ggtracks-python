"""Tests for the chromosome ideogram and its Giemsa fill scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.plot import ggplot_build
import ggtracks as ggt
from ggtracks.geom_ideogram import GeomIdeogram


BANDS = pd.DataFrame(
    {
        "xstart": [1, 1000, 2000, 2500, 3000],
        "xend": [1000, 2000, 2500, 3000, 4000],
        "name": ["p12", "p11", "p11.1", "q11", "q12"],
        "stain": ["gneg", "gpos50", "acen", "acen", "gpos100"],
        "y": 1.0,
    }
)


def _layer(data=BANDS, **kwargs):
    return ggt.geom_ideogram(
        gg.aes(xstart="xstart", xend="xend", y="y", stain="stain", fill="stain"),
        data=data,
        **kwargs,
    )


def _plot(data=BANDS, **kwargs):
    return gg.ggplot() + _layer(data, **kwargs) + ggt.scale_fill_giemsa()


# --------------------------------------------------------------------------
# stain palette
# --------------------------------------------------------------------------


def test_palette_covers_the_standard_stains():
    expected = {
        "gneg", "gpos25", "gpos33", "gpos50",
        "gpos66", "gpos75", "gpos100", "gvar", "acen", "stalk",
    }
    assert expected <= set(ggt.GIEMSA_COLOURS)


def test_density_ramp_darkens_monotonically():
    def lum(h):
        h = h.lstrip("#")
        return sum(int(h[i : i + 2], 16) for i in (0, 2, 4))

    ramp = ["gneg", "gpos25", "gpos33", "gpos50", "gpos66", "gpos75", "gpos100"]
    lums = [lum(ggt.GIEMSA_COLOURS[s]) for s in ramp]
    assert all(a >= b for a, b in zip(lums, lums[1:]))


def test_centromere_is_the_only_saturated_colour():
    """The accent must stay unique, or it stops drawing the eye."""
    assert ggt.GIEMSA_COLOURS["acen"] == "#8B0000"


def test_scale_is_usable_and_quiet_by_default():
    sc = ggt.scale_fill_giemsa()
    assert sc is not None
    built = ggplot_build(_plot())
    assert len(built.data[0]) == len(BANDS)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def test_unknown_stain_fails_loud():
    """Colouring an unknown stain grey would misreport the karyotype."""
    bad = BANDS.assign(stain=["gneg", "gpos50", "acen", "acen", "mystery"])
    with pytest.raises(Exception) as ei:
        ggplot_build(_plot(bad))
    chain, exc = [], ei.value
    while exc is not None:
        chain.append(str(exc))
        exc = exc.__cause__ or exc.__context__
    assert any("unrecognised Giemsa stain" in c for c in chain)


def test_bands_become_rectangles():
    d = ggplot_build(_plot()).data[0]
    assert {"xmin", "xmax", "ymin", "ymax"} <= set(d.columns)
    assert (d["xmax"] > d["xmin"]).all()


def test_height_centres_the_band_on_y():
    d = ggplot_build(_plot(height=0.4)).data[0]
    assert np.allclose(d["ymax"] - d["ymin"], 0.4)
    assert np.allclose((d["ymax"] + d["ymin"]) / 2, 1.0)


def test_centromere_makes_two_triangles_that_meet():
    """The waist: the lower band tapers right, the upper one left."""
    prepared = GeomIdeogram().setup_data(BANDS.copy(), {})
    acen = prepared[prepared["stain"] == "acen"]
    tri = GeomIdeogram._centromere(acen)
    assert tri["group"].nunique() == 2
    assert len(tri) == 6  # three vertices each
    first, second = (g for _, g in tri.groupby("group", sort=True))
    # they meet at the shared boundary, at mid height
    assert first["x"].max() == pytest.approx(second["x"].min())
    apex_first = first.loc[first["x"].idxmax(), "y"]
    apex_second = second.loc[second["x"].idxmin(), "y"]
    assert apex_first == pytest.approx(1.0) == pytest.approx(apex_second)


def test_a_chromosome_without_a_centromere_still_draws(tmp_path):
    plain = BANDS.assign(stain=["gneg", "gpos50", "gpos25", "gneg", "gpos100"])
    out = tmp_path / "noacen.png"
    gg.ggsave(str(out), _plot(plain), width=4, height=1, dpi=72)
    assert out.stat().st_size > 0


@pytest.mark.parametrize("outline", ["grey30", None])
def test_outline_is_optional(outline, tmp_path):
    out = tmp_path / f"outline_{outline}.png"
    gg.ggsave(str(out), _plot(outline=outline), width=4, height=1, dpi=72)
    assert out.stat().st_size > 0


def test_renders_with_theme(tmp_path):
    p = _plot() + ggt.theme_tracks()
    out = tmp_path / "ideo.png"
    gg.ggsave(str(out), p, width=5, height=1.1, dpi=72)
    assert out.stat().st_size > 0


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------


def test_ideogram_joins_a_multi_locus_figure(tmp_path):
    """An ideogram spans a whole chromosome, so it can only share a figure
    with a compressed gene view when each column carries its own scale."""
    chrom = ggt.GenomicMapper.from_intervals([(1, 4000)], collapse_introns=False)
    bands = BANDS.assign(track="ideogram", locus="chr1")
    tracks = [
        ggt.Track(
            "ideogram",
            [_layer(bands)],
            height=0.4,
            y_breaks=[1.0],
            y_labels=[""],
        )
    ]
    p = ggt.plot_tracks(tracks, mappers={"chr1": chrom})
    p = p + ggt.scale_fill_giemsa()
    out = tmp_path / "multi_ideo.png"
    gg.ggsave(str(out), p, width=5, height=p.fig_height, dpi=72)
    assert out.stat().st_size > 0
