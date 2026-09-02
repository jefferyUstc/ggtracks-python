"""Locus preparation — from a tidy feature table to plot-ready pieces.

Every figure over one locus starts the same way: take the locus's
features, build a :class:`~ggtracks.GenomicMapper` over the union of its
exons, and note the genomic range a signal query needs. The tutorials
wrote that as a private helper in every notebook; :class:`Locus` is the
same preparation promoted into the library. :func:`gene_model_layers` is
its companion: the standard collapsed gene-model layers (exon boxes,
taller CDS boxes, introns with strand arrows), ready to drop into a
:class:`~ggtracks.Track` or a bare ggplot.
"""

from __future__ import annotations

from .palettes import FEATURE_COLOURS

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import pandas as pd

from .mapper import GenomicMapper, _merge_intervals

__all__ = ["Locus", "gene_model_layers"]


@dataclass(frozen=True)
class Locus:
    """One locus, ready to plot: features, mapper, and genomic range.

    Build with :meth:`from_features`. Fields:

    ``features``
        The tidy feature table the locus was built from.
    ``mapper``
        Intron-compressing coordinate model over the exonic union.
    ``chrom``, ``start``, ``end``
        Genomic extent, 1-based half-open — the exon extent widened by
        the ``flank`` it was built with.
    """

    features: pd.DataFrame
    mapper: GenomicMapper
    chrom: str
    start: int
    end: int

    @property
    def region(self) -> Tuple[str, int, int]:
        """``(chrom, start, end)``, ready to unpack into a signal reader::

            cov = read_bigwig("signal.bw", *locus.region, bins=300)
        """
        return (self.chrom, self.start, self.end)

    @classmethod
    def from_features(
        cls,
        features: pd.DataFrame,
        *,
        flank: int = 0,
        **mapper_options: Any,
    ) -> "Locus":
        """Build a :class:`Locus` from a tidy feature table.

        Parameters
        ----------
        features
            Table with ``chrom``, ``xstart``, ``xend`` and ``feature``
            columns covering a single chromosome — typically
            ``read_annotations(path, genes=["Actb"])`` output. Filter
            first when the table holds more than the locus you mean.
        flank
            Extra bp of context on each side (clamped at position 1).
            Flanks enter the mapper as *uncompressed* spans: they are
            genomic context, not introns, and compressing them would
            misstate promoter and terminator regions.
        **mapper_options
            Forwarded to :meth:`GenomicMapper.from_intervals` —
            ``intron_mode``, ``target_gap_width``, ``intron_scale``,
            ``intron_min``, ``exon_scale``, ``collapse_introns``.

        Raises
        ------
        ValueError
            Missing columns, several chromosomes, no exon rows, or a
            negative flank.
        """
        missing = [c for c in ("chrom", "xstart", "xend", "feature")
                   if c not in features.columns]
        if missing:
            raise ValueError(
                f"Locus.from_features: features is missing column(s) "
                f"{missing!r} (have {list(features.columns)!r})."
            )
        if flank < 0:
            raise ValueError(
                f"Locus.from_features: flank must be >= 0 (got {flank!r})."
            )
        chroms = sorted({str(c) for c in features["chrom"].unique()})
        if len(chroms) != 1:
            raise ValueError(
                f"Locus.from_features: features span chromosomes {chroms!r}; "
                "a locus lives on one — filter the table first."
            )
        exons = features[features["feature"] == "exon"]
        if exons.empty:
            raise ValueError(
                "Locus.from_features: features has no exon rows to build "
                "the mapper from."
            )
        exon_lo = int(exons["xstart"].min())
        exon_hi = int(exons["xend"].max())
        start = max(1, exon_lo - int(flank))
        end = exon_hi + int(flank)
        intervals = list(zip(exons["xstart"], exons["xend"]))
        if start < exon_lo:
            intervals.append((start, exon_lo))
        if end > exon_hi:
            intervals.append((exon_hi, end))
        mapper = GenomicMapper.from_intervals(intervals, **mapper_options)
        return cls(features=features, mapper=mapper,
                   chrom=chroms[0], start=start, end=end)


def gene_model_layers(
    features: pd.DataFrame,
    *,
    y: float = 1.0,
    track: Optional[str] = None,
    exon_fill: str = FEATURE_COLOURS["exon"],
    cds_fill: str = FEATURE_COLOURS["cds"],
    exon_height: float = 0.3,
    cds_height: float = 0.55,
) -> List[Any]:
    """The standard collapsed gene model, as a list of ggplot layers.

    Exon boxes at ``exon_height`` (which is also what UTRs and
    non-coding transcripts show as, since exons cover them), CDS boxes
    overlaid taller at ``cds_height``, and introns — the gaps in the
    merged exon union — drawn with a central strand arrow when the
    features agree on a strand. Everything sits on one row at ``y``; for
    an isoform-per-row view, lay out rows with
    :class:`~ggtracks.StatPileup` / :func:`~ggtracks.pack_rows` instead.

    Parameters
    ----------
    features
        Tidy feature table (``feature``, ``xstart``, ``xend``;
        ``strand`` drives the intron arrows).
    y
        Baseline row the model sits on.
    track
        When given, stamped as a ``track`` column on every layer's data
        so the layers drop straight into a
        :func:`~ggtracks.plot_tracks` :class:`~ggtracks.Track`.
    exon_fill, cds_fill, exon_height, cds_height
        Box styling.

    Returns
    -------
    list
        ggplot layers; empty pieces (no CDS, no introns) are omitted.
    """
    import ggplot2_py as gg

    from .geom_intron import geom_intron, to_intron
    from .geom_range import geom_range

    exons = features[features["feature"] == "exon"]
    if exons.empty:
        raise ValueError("gene_model_layers: features has no exon rows.")
    cds = features[features["feature"] == "CDS"]

    def prep(frame: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out = frame[cols].copy()
        out["y"] = float(y)
        if track is not None:
            out["track"] = track
        return out

    layers: List[Any] = [
        geom_range(gg.aes(xstart="xstart", xend="xend", y="y"),
                   data=prep(exons, ["xstart", "xend"]),
                   height=exon_height, fill=exon_fill, inherit_aes=False),
    ]
    if not cds.empty:
        layers.append(
            geom_range(gg.aes(xstart="xstart", xend="xend", y="y"),
                       data=prep(cds, ["xstart", "xend"]),
                       height=cds_height, fill=cds_fill, inherit_aes=False)
        )

    merged = pd.DataFrame(
        _merge_intervals(list(zip(exons["xstart"], exons["xend"]))),
        columns=["xstart", "xend"],
    )
    introns = to_intron(merged)
    if not introns.empty:
        strands: set = set()
        if "strand" in features.columns:
            strands = {s for s in features["strand"].dropna().unique()
                       if s in ("+", "-")}
        mapping = dict(xstart="xstart", xend="xend", y="y")
        cols = ["xstart", "xend"]
        if len(strands) == 1:
            introns = introns.assign(strand=next(iter(strands)))
            mapping["strand"] = "strand"
            cols.append("strand")
        layers.append(
            geom_intron(gg.aes(**mapping), data=prep(introns, cols),
                        inherit_aes=False,
                        arrow="default" if len(strands) == 1 else None)
        )
    return layers
