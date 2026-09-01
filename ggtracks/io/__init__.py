"""File readers that turn genomics formats into ggtracks-ready DataFrames.

Without this layer the package's own quick start asks for a hand-built
DataFrame of exons — which nobody has. What people have is an annotation
file and a signal file, so these readers close that gap and make ggtracks
usable on its own, not only behind a toolkit that already owns a
transcript catalog.

Everything returns a plain :class:`pandas.DataFrame` in the column contract
the geoms expect. **No data container is involved** — these parse *file
formats*, which is what keeps the package free of any dependency on
AnnData / MuData / lrdata and friends.

Coordinate contract
-------------------
Every frame is **1-based, half-open** ``[xstart, xend)``.

Genomics formats disagree: GTF and GFF3 are 1-based *inclusive*, while
BED, bedGraph, bigWig and cytoband files are 0-based *half-open*. Left
alone, that skew misaligns a coverage track against the gene model drawn
underneath it by one base. The translation happens here, at the file
boundary, and nowhere else — so downstream every interval means the same
thing and ``xend - xstart`` is always the length.

Readers
-------
* :func:`read_annotations` (plus :func:`read_gtf`, :func:`read_gff3`) —
  gene models.
* :func:`read_bed` — BED / narrowPeak / broadPeak interval features
  (peak calls, generic regions).
* :class:`BigWig` / :func:`read_bigwig` — indexed signal, with zoom-level
  support for wide regions.
* :class:`BedGraph` / :func:`read_bedgraph` — unindexed signal.
* :func:`read_cytoband` — chromosome bands for the ideogram.
* :func:`resolve_chrom` / :func:`normalize_chrom` — ``chr1`` vs ``1``.
"""

from __future__ import annotations

from ._bed import read_bed
from ._bedgraph import BedGraph, read_bedgraph
from ._bigwig import BigWig, read_bigwig
from ._chrom import (
    ChromStyle,
    detect_chrom_style,
    normalize_chrom,
    resolve_chrom,
)
from ._cytoband import CYTOBAND_COLUMNS, read_cytoband
from ._gtf import FEATURE_COLUMNS, read_annotations, read_gff3, read_gtf

__all__ = [
    "read_annotations",
    "read_gtf",
    "read_gff3",
    "FEATURE_COLUMNS",
    "read_bed",
    "BigWig",
    "read_bigwig",
    "BedGraph",
    "read_bedgraph",
    "read_cytoband",
    "CYTOBAND_COLUMNS",
    "normalize_chrom",
    "detect_chrom_style",
    "resolve_chrom",
    "ChromStyle",
]
