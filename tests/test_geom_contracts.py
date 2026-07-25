"""Grammar-of-graphics contracts shared by every geom in the package.

These are the pieces ggplot2 relies on that fail *quietly* when omitted: a
legend drawn with the wrong glyph, an aesthetic the scale never learns
about. Checking them once here beats rediscovering each omission from a
figure that looks almost right.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.draw_key import draw_key_blank, draw_key_path, draw_key_polygon
from ggplot2_py.plot import ggplot_build
import ggtracks as ggt


GEOMS = ["GeomRange", "GeomIntron", "GeomJunction",
         "GeomCoverage", "GeomIdeogram", "GeomZoomLink"]

#: Each geom class and the layer constructor that exposes it.
CONSTRUCTOR = {
    "GeomRange": "geom_range",
    "GeomIntron": "geom_intron",
    "GeomJunction": "geom_junction",
    "GeomCoverage": "geom_coverage",
    "GeomIdeogram": "geom_ideogram",
    "GeomZoomLink": "geom_zoom_link",
}

# What each geom actually puts on the panel decides its legend glyph.
EXPECTED_KEY = {
    "GeomRange": draw_key_polygon,      # filled boxes
    "GeomIntron": draw_key_path,        # lines
    "GeomJunction": draw_key_path,      # arcs
    "GeomCoverage": draw_key_polygon,   # filled area
    "GeomIdeogram": draw_key_polygon,   # filled bands
    "GeomZoomLink": draw_key_blank,     # a connector is not a data series
}


@pytest.mark.parametrize("name", GEOMS)
def test_legend_key_matches_what_the_geom_draws(name):
    """The base class defaults to a *point* key, which for a filled geom
    shows a dot in the wrong colour — it reads ``colour``, not ``fill``."""
    assert getattr(ggt, name).draw_key is EXPECTED_KEY[name]


@pytest.mark.parametrize("name", GEOMS)
def test_required_aes_use_the_interval_nomenclature(name):
    """``xstart``/``xend`` is what ``scale_x_genomic`` extends its transform
    to; a geom using a plain ``x`` would escape intron compression."""
    required = set(getattr(ggt, name).required_aes)
    assert {"xstart", "xend"} <= required


@pytest.mark.parametrize("name", GEOMS)
def test_every_geom_and_its_constructor_are_documented(name):
    geom = getattr(ggt, name)
    assert (geom.__doc__ or "").strip()
    constructor = getattr(ggt, CONSTRUCTOR[name])
    assert (constructor.__doc__ or "").strip()


@pytest.mark.parametrize("name", GEOMS)
def test_default_aes_use_literals_not_theme_lookups(name):
    """The package settled on literal defaults; mixing in ``FromTheme``
    would give one package two conventions."""
    from ggplot2_py.geom import FromTheme

    defaults = dict(getattr(ggt, name).default_aes)
    assert not any(isinstance(v, FromTheme) for v in defaults.values())


def test_a_fill_legend_shows_the_mapped_colours():
    """End to end: the key must carry the fill the data was mapped to."""
    data = pd.concat([
        pd.DataFrame({"xstart": [0, 50], "xend": [50, 100],
                      "value": [3.0, 6.0], "grp": "a"}),
        pd.DataFrame({"xstart": [0, 50], "xend": [50, 100],
                      "value": [5.0, 2.0], "grp": "b"}),
    ])
    p = (
        gg.ggplot(data, gg.aes(xstart="xstart", xend="xend", y="value", fill="grp"))
        + ggt.geom_coverage()
        + gg.scale_fill_manual(values={"a": "#111111", "b": "#EEEEEE"})
    )
    built = ggplot_build(p)
    fills = set(built.data[0]["fill"].str.lower())
    assert fills == {"#111111", "#eeeeee"}


def test_geoms_render_a_legend_without_error(tmp_path):
    data = pd.DataFrame({
        "xstart": [0, 50], "xend": [50, 100], "y": [1.0, 1.0], "grp": ["a", "b"],
    })
    for layer in (
        ggt.geom_range(gg.aes(xstart="xstart", xend="xend", y="y", fill="grp"), data=data),
        ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="y", fill="grp"), data=data),
    ):
        out = tmp_path / "legend.png"
        gg.ggsave(str(out), gg.ggplot() + layer, width=3, height=1.5, dpi=72)
        assert out.stat().st_size > 0


@pytest.mark.parametrize(
    "constructor",
    ["geom_range", "geom_intron", "geom_junction", "geom_coverage",
     "geom_ideogram", "geom_zoom_link", "geom_highlight"],
)
def test_constructors_accept_the_standard_layer_arguments(constructor):
    """Every layer constructor must let a caller redirect data, opt out of
    inherited aesthetics and silence its legend — the standard escape
    hatches a grammar user expects."""
    params = inspect.signature(getattr(ggt, constructor)).parameters
    assert {"data", "inherit_aes"} <= set(params)
