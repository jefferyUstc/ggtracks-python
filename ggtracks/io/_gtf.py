"""GTF / GFF3 → tidy feature table.

The output is a plain ``pandas`` DataFrame in the coordinate contract the
geoms expect, so a user with nothing but an annotation file can go straight
to :func:`~ggtracks.geom_range` without hand-building frames.

**Coordinates.** GTF and GFF3 are 1-based *inclusive* ``[start, end]``;
ggtracks intervals are half-open ``[xstart, xend)``. The conversion
(``xend = end + 1``, preserving both position and length) happens here and
**only** here — it is the single boundary at which foreign coordinate
conventions are translated.

**Columns.** ``chrom``, ``xstart``, ``xend``, ``strand``, ``feature``,
``gene_id``, ``gene_name``, ``tx_id``.

**Feature vocabulary.** The two formats spell the same things differently
(``mRNA`` vs ``transcript``, ``five_prime_UTR`` vs ``five_prime_utr``), so
feature names are normalised to one vocabulary: ``gene``, ``transcript``,
``exon``, ``CDS``, ``five_prime_utr``, ``three_prime_utr``.

One difference is a real disagreement between the formats rather than a
spelling, and is passed through unchanged: **GTF excludes the stop codon
from its ``CDS`` records** (carrying it as a separate ``stop_codon`` feature,
which is not part of the vocabulary above) while **GFF3 includes it**. The
same transcript therefore has a CDS three bases shorter when read from GTF —
invisible at any realistic zoom, but worth knowing before comparing the two.
"""

from __future__ import annotations

import gzip
import os
import re
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple

import pandas as pd

from ._chrom import ChromStyle, normalize_chrom

__all__ = ["read_gtf", "read_gff3", "read_annotations", "FEATURE_COLUMNS"]

#: Column order of every frame returned by this module.
FEATURE_COLUMNS = (
    "chrom",
    "xstart",
    "xend",
    "strand",
    "feature",
    "gene_id",
    "gene_name",
    "tx_id",
)

#: Format spellings → the one vocabulary ggtracks uses.
_FEATURE_ALIASES = {
    "gene": "gene",
    "transcript": "transcript",
    "mrna": "transcript",
    "exon": "exon",
    "cds": "CDS",
    "five_prime_utr": "five_prime_utr",
    "5utr": "five_prime_utr",
    "three_prime_utr": "three_prime_utr",
    "3utr": "three_prime_utr",
    "utr": "utr",
}

_GTF_ATTR = re.compile(r'(\S+)\s+"([^"]*)"')


def _open(path: str):
    """Open plain or gzipped text."""
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(
        path, "rt", encoding="utf-8"
    )


def _canonical_feature(raw: str) -> Optional[str]:
    return _FEATURE_ALIASES.get(raw.lower())


def _overlaps(start: int, end: int, region: Tuple[int, int]) -> bool:
    return max(start, region[0]) < min(end, region[1])


def _finish(rows: list, chrom_style: Optional[ChromStyle]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=list(FEATURE_COLUMNS))
    if not df.empty:
        df["xstart"] = df["xstart"].astype("int64")
        df["xend"] = df["xend"].astype("int64")
        if chrom_style is not None:
            df["chrom"] = [normalize_chrom(c, chrom_style) for c in df["chrom"]]
    return df.reset_index(drop=True)


def _parse_region(region: Any) -> Optional[Tuple[str, int, int]]:
    """``"chr7:1000-2000"`` or ``("chr7", 1000, 2000)`` → a normalised triple."""
    if region is None:
        return None
    if isinstance(region, str):
        m = re.fullmatch(r"\s*([^:]+):([\d,_]+)-([\d,_]+)\s*", region)
        if not m:
            raise ValueError(
                f"read_annotations: region {region!r} is not 'chrom:start-end'."
            )
        chrom, start, end = m.group(1), m.group(2), m.group(3)
        start_i = int(start.replace(",", "").replace("_", ""))
        end_i = int(end.replace(",", "").replace("_", ""))
    else:
        chrom, start_i, end_i = region
        start_i, end_i = int(start_i), int(end_i)
    if end_i <= start_i:
        raise ValueError(
            f"read_annotations: region end must exceed start (got {start_i}-{end_i})."
        )
    return str(chrom), start_i, end_i


def _region_chrom_matches(chrom: str, want: str) -> bool:
    return chrom == want or normalize_chrom(chrom, "ensembl") == normalize_chrom(
        want, "ensembl"
    )


def read_gtf(
    path: str,
    *,
    genes: Optional[Iterable[str]] = None,
    region: Any = None,
    chrom_style: Optional[ChromStyle] = None,
) -> pd.DataFrame:
    """Parse a GTF file into the tidy feature table. See module docstring."""
    wanted = set(genes) if genes is not None else None
    reg = _parse_region(region)
    rows: list = []

    with _open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = _canonical_feature(parts[2])
            if feature is None:
                continue
            chrom = parts[0]
            start, end = int(parts[3]), int(parts[4])
            if reg is not None:
                if not _region_chrom_matches(chrom, reg[0]):
                    continue
                if not _overlaps(start, end + 1, (reg[1], reg[2])):
                    continue
            attrs = dict(_GTF_ATTR.findall(parts[8]))
            gene_id = attrs.get("gene_id", "")
            gene_name = attrs.get("gene_name", "")
            if wanted is not None and not (gene_id in wanted or gene_name in wanted):
                continue
            rows.append(
                (
                    chrom,
                    start,
                    end + 1,
                    parts[6],
                    feature,
                    gene_id,
                    gene_name,
                    attrs.get("transcript_id", ""),
                )
            )
    return _finish(rows, chrom_style)


def _gff3_attrs(field: str) -> dict:
    out = {}
    for chunk in field.split(";"):
        chunk = chunk.strip()
        if "=" in chunk:
            key, value = chunk.split("=", 1)
            out[key] = value
    return out


def _strip_prefix(value: str) -> str:
    """GFF3 ids are often namespaced (``transcript:ENST…``)."""
    return value.split(":", 1)[1] if ":" in value else value


def read_gff3(
    path: str,
    *,
    genes: Optional[Iterable[str]] = None,
    region: Any = None,
    chrom_style: Optional[ChromStyle] = None,
) -> pd.DataFrame:
    """Parse a GFF3 file into the tidy feature table.

    GFF3 links features by ``Parent`` rather than repeating ids on every
    line, so this reads the file twice: once to resolve genes (and apply the
    *genes* / *region* filter), once to collect their transcripts and the
    children of those transcripts.
    """
    wanted = set(genes) if genes is not None else None
    reg = _parse_region(region)

    gene_of: dict = {}  # gene record id -> (gene_id, gene_name)
    rows: list = []

    with _open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2].lower() != "gene":
                continue
            chrom = parts[0]
            start, end = int(parts[3]), int(parts[4])
            if reg is not None:
                if not _region_chrom_matches(chrom, reg[0]):
                    continue
                if not _overlaps(start, end + 1, (reg[1], reg[2])):
                    continue
            attrs = _gff3_attrs(parts[8])
            record_id = attrs.get("ID", "")
            gene_id = _strip_prefix(record_id)
            gene_name = attrs.get("Name", "")
            if wanted is not None and not (gene_id in wanted or gene_name in wanted):
                continue
            gene_of[record_id] = (gene_id, gene_name)
            rows.append(
                (chrom, start, end + 1, parts[6], "gene", gene_id, gene_name, "")
            )

    if not gene_of:
        return _finish(rows, chrom_style)

    tx_of: dict = {}  # transcript record id -> (gene_id, gene_name, tx_id)
    pending: list = []  # children seen before their transcript

    with _open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = _canonical_feature(parts[2])
            if feature is None or feature == "gene":
                continue
            attrs = _gff3_attrs(parts[8])
            parent = attrs.get("Parent", "")
            if not parent:
                continue
            chrom = parts[0]
            start, end = int(parts[3]), int(parts[4])
            strand = parts[6]

            if feature == "transcript":
                if parent not in gene_of:
                    continue
                gene_id, gene_name = gene_of[parent]
                record_id = attrs.get("ID", "")
                tx_id = _strip_prefix(record_id)
                tx_of[record_id] = (gene_id, gene_name, tx_id)
                rows.append(
                    (chrom, start, end + 1, strand, "transcript", gene_id, gene_name, tx_id)
                )
            else:
                pending.append((chrom, start, end + 1, strand, feature, parent))

    for chrom, xstart, xend, strand, feature, parent in pending:
        owner = tx_of.get(parent)
        if owner is None:
            continue
        gene_id, gene_name, tx_id = owner
        rows.append((chrom, xstart, xend, strand, feature, gene_id, gene_name, tx_id))

    return _finish(rows, chrom_style)


def read_annotations(
    path: str,
    *,
    genes: Optional[Iterable[str]] = None,
    region: Any = None,
    chrom_style: Optional[ChromStyle] = None,
    format: Optional[Literal["gtf", "gff3"]] = None,
) -> pd.DataFrame:
    """Read a GTF or GFF3 annotation into the tidy feature table.

    Parameters
    ----------
    path
        Annotation file; ``.gz`` is handled transparently.
    genes
        Keep only these genes, matched against **either** ``gene_id`` or
        ``gene_name``. ``None`` keeps everything.
    region
        Keep only features overlapping this region, given as
        ``"chrom:start-end"`` or ``(chrom, start, end)``. Combines with
        *genes* as a conjunction.
    chrom_style
        Rewrite chromosome names to ``"ucsc"`` (``chr1``) or ``"ensembl"``
        (``1``). ``None`` leaves them as found.
    format
        Force ``"gtf"`` or ``"gff3"``. ``None`` (default) infers from the
        file name.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`FEATURE_COLUMNS`, coordinates half-open.

    Examples
    --------
    >>> exons = read_annotations("genes.gtf.gz", genes=["Actb"])   # doctest: +SKIP
    >>> exons = exons[exons["feature"] == "exon"]                  # doctest: +SKIP
    """
    if format is None:
        base = os.path.basename(str(path)).lower()
        format = "gff3" if ".gff" in base else "gtf"
    if format == "gtf":
        return read_gtf(path, genes=genes, region=region, chrom_style=chrom_style)
    if format == "gff3":
        return read_gff3(path, genes=genes, region=region, chrom_style=chrom_style)
    raise ValueError(
        f"read_annotations: format must be 'gtf' or 'gff3' (got {format!r})."
    )
