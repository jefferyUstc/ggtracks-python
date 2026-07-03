"""ggtracks — grammar-of-graphics genomic tracks for the ggplot2-python ecosystem.

A small set of *real* ggplot2-python ``Geom``/``Stat`` citizens (modeled on
R's ``ggtranscript`` / ``gggenes``) plus a genomic coordinate model with
intron compression, all composable with faceting, scales and themes.

Inputs are plain ``pandas`` DataFrames in **genomic coordinates** plus a
:class:`GenomicMapper`. The package has **no coupling to any data
container** (no AnnData / mudata / lrdata), so any genomics tool that can
produce feature DataFrames can use it.

Public surface
--------------
* coordinate model — :class:`GenomicMapper`
* ggplot scale / coord — :func:`scale_x_genomic`, :func:`coord_genomic`,
  :func:`genomic_transform`
* geoms / stats — :func:`geom_range`, :func:`geom_intron`,
  :func:`geom_junction`, :func:`to_intron`, :class:`StatPileup` /
  :func:`pack_rows`
* track composition — :class:`Track`, :func:`plot_tracks`
* style — :func:`theme_tracks`, :func:`track_palettes`
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _version

from .mapper import GenomicMapper
from .scale import coord_genomic, genomic_transform, scale_x_genomic
from .palettes import TRACK_PALETTES, track_palettes
from .theme import PUB_BASE_SIZE, theme_tracks
from .geom_range import GeomRange, geom_range
from .geom_intron import GeomIntron, StatIntron, geom_intron, to_intron
from .geom_junction import GeomJunction, geom_junction
from .stat_pileup import StatPileup, pack_rows
from .plot_tracks import Track, plot_tracks
from ._render import finalize_gg

try:
    __version__ = _version("ggtracks-python")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "GenomicMapper",
    "scale_x_genomic",
    "coord_genomic",
    "genomic_transform",
    "track_palettes",
    "TRACK_PALETTES",
    "theme_tracks",
    "PUB_BASE_SIZE",
    "GeomRange",
    "geom_range",
    "GeomIntron",
    "StatIntron",
    "geom_intron",
    "to_intron",
    "GeomJunction",
    "geom_junction",
    "StatPileup",
    "pack_rows",
    "Track",
    "plot_tracks",
]
