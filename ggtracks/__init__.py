"""ggtracks — grammar-of-graphics genomic tracks for the ggplot2-python ecosystem.

A small set of *real* ggplot2-python ``Geom``/``Stat`` citizens (modeled on
R's ``ggtranscript`` / ``gggenes``) plus a genomic coordinate model with
intron compression, all composable with faceting, scales and themes.

Inputs are plain ``pandas`` DataFrames in **genomic coordinates** plus a
:class:`GenomicMapper` — either built by hand or read straight out of an
annotation and a signal file by :mod:`ggtracks.io`. The package has **no
coupling to any data container** (no AnnData / mudata / lrdata); the
readers parse *file formats*, which is what keeps that true.

Public surface
--------------
* readers — :mod:`ggtracks.io`: :func:`read_annotations`, :class:`BigWig` /
  :func:`read_bigwig`, :class:`BedGraph` / :func:`read_bedgraph`,
  :func:`read_cytoband`
* coordinate model — :class:`GenomicMapper`
* ggplot scale / coord — :func:`scale_x_genomic`, :func:`coord_genomic`,
  :func:`genomic_transform`, :func:`base_x_scale`, :func:`signal_limits`
* geoms / stats — :func:`geom_range`, :func:`geom_intron`,
  :func:`geom_junction`, :func:`geom_coverage` / :class:`StatBinCoverage`,
  :func:`geom_ideogram` / :func:`scale_fill_giemsa`,
  :func:`geom_highlight`, :func:`geom_zoom_link`, :func:`to_intron`,
  :class:`StatPileup` / :func:`pack_rows`
* track composition — :class:`Track`, :func:`plot_tracks`,
  :func:`natural_height`
* transcript helpers — :func:`rank_transcripts`,
  :func:`collapse_transcripts`
* style — :func:`theme_tracks`, :func:`track_palettes`,
  :func:`signal_palette`
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError, version as _version

from . import io
from .io import (
    BedGraph,
    BigWig,
    read_annotations,
    read_bedgraph,
    read_bigwig,
    read_cytoband,
)
from .mapper import GenomicMapper
from .scale import (
    base_x_scale,
    coord_genomic,
    genomic_transform,
    scale_x_genomic,
    signal_limits,
)
from .palettes import (
    SIGNAL_PALETTES,
    TRACK_PALETTES,
    signal_palette,
    track_palettes,
)
from .theme import PANEL_BORDER_COLOUR, PUB_BASE_SIZE, theme_tracks
from .geom_range import GeomRange, geom_range
from .geom_intron import GeomIntron, StatIntron, geom_intron, to_intron
from .geom_junction import GeomJunction, geom_junction
from .geom_highlight import geom_highlight
from .geom_zoom_link import GeomZoomLink, geom_zoom_link
from .geom_ideogram import (
    GIEMSA_COLOURS,
    GeomIdeogram,
    geom_ideogram,
    scale_fill_giemsa,
)
from .geom_coverage import (
    GeomCoverage,
    StatBinCoverage,
    geom_coverage,
    stat_bin_coverage,
)
from .transcripts import collapse_transcripts, rank_transcripts
from .stat_pileup import StatPileup, pack_rows
from .plot_tracks import Track, plot_tracks
from ._render import finalize_gg, natural_height

try:
    __version__ = _version("ggtracks-python")
except _PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "io",
    "read_annotations",
    "BigWig",
    "read_bigwig",
    "BedGraph",
    "read_bedgraph",
    "read_cytoband",
    "GenomicMapper",
    "scale_x_genomic",
    "coord_genomic",
    "genomic_transform",
    "base_x_scale",
    "signal_limits",
    "track_palettes",
    "TRACK_PALETTES",
    "signal_palette",
    "SIGNAL_PALETTES",
    "theme_tracks",
    "PUB_BASE_SIZE",
    "PANEL_BORDER_COLOUR",
    "GeomRange",
    "geom_range",
    "GeomIntron",
    "StatIntron",
    "geom_intron",
    "to_intron",
    "GeomJunction",
    "geom_junction",
    "geom_highlight",
    "GeomZoomLink",
    "geom_zoom_link",
    "GeomIdeogram",
    "geom_ideogram",
    "scale_fill_giemsa",
    "GIEMSA_COLOURS",
    "GeomCoverage",
    "geom_coverage",
    "StatBinCoverage",
    "stat_bin_coverage",
    "rank_transcripts",
    "collapse_transcripts",
    "StatPileup",
    "pack_rows",
    "Track",
    "plot_tracks",
    "finalize_gg",
    "natural_height",
]
