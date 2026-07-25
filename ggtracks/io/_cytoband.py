"""Cytoband → tidy band table (for :func:`~ggtracks.geom_ideogram`).

The ``cytoBandIdeo`` layout is BED-like: ``chrom start end name stain``,
where *stain* is the Giemsa class the band is drawn with (``gneg``,
``gpos25``…``gpos100``, ``acen`` for the centromere, ``gvar``, ``stalk``).

**Coordinates.** The file is 0-based half-open; the frame returned here is
1-based half-open, matching the rest of :mod:`ggtracks.io`.
"""

from __future__ import annotations

import gzip
from typing import Optional

import pandas as pd

from ._chrom import ChromStyle, normalize_chrom, resolve_chrom

__all__ = ["read_cytoband", "CYTOBAND_COLUMNS"]

#: Column order of the frame returned by :func:`read_cytoband`.
CYTOBAND_COLUMNS = ("chrom", "xstart", "xend", "name", "stain")


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(
        path, "rt", encoding="utf-8"
    )


def read_cytoband(
    path: str,
    *,
    chrom: Optional[str] = None,
    chrom_style: Optional[ChromStyle] = None,
) -> pd.DataFrame:
    """Read a cytoband file into a tidy band table.

    Parameters
    ----------
    path
        ``cytoBandIdeo``-style file; ``.gz`` is handled transparently.
    chrom
        Keep only this chromosome (a ``chr`` prefix mismatch is reconciled).
        ``None`` keeps every chromosome.
    chrom_style
        Rewrite chromosome names to ``"ucsc"`` or ``"ensembl"``.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`CYTOBAND_COLUMNS`, coordinates 1-based half-open.
    """
    rows = []
    with _open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                raise ValueError(
                    f"read_cytoband({path!r}): line {lineno} has {len(parts)} "
                    "fields, expected 5 (chrom start end name stain)."
                )
            rows.append(
                (parts[0], int(parts[1]) + 1, int(parts[2]) + 1, parts[3], parts[4])
            )

    df = pd.DataFrame(rows, columns=list(CYTOBAND_COLUMNS))
    if chrom is not None and not df.empty:
        key = resolve_chrom(chrom, df["chrom"].unique())
        df = df[df["chrom"] == key]
    if chrom_style is not None and not df.empty:
        df = df.assign(chrom=[normalize_chrom(c, chrom_style) for c in df["chrom"]])
    return df.reset_index(drop=True)
