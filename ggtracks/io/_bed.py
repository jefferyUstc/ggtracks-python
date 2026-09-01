"""BED family → tidy interval tables (BED, narrowPeak, broadPeak).

Peak calls are the most common companion to a coverage + gene-model
figure, and they arrive as BED-family files. This reader emits a plain
DataFrame in the ggtracks column contract, ready for
:func:`~ggtracks.geom_range` or :func:`~ggtracks.geom_highlight`.

**Coordinates.** BED is 0-based half-open; the frames returned here are
1-based half-open, matching the rest of :mod:`ggtracks.io`. The
narrowPeak ``peak`` field is a coordinate too — a 0-based offset from
the interval start, ``-1`` for "not called" — so it is translated to an
absolute 1-based position (``NaN`` when not called). The statistical
fields (``signal_value``, ``p_value``, ``q_value``) pass through
untouched: ENCODE writes ``-1`` there for "not assigned", and
reinterpreting a statistic is the caller's call, not a coordinate
translation.

**Columns.** Plain BED keeps the standard fields it finds, up to six:
``chrom, xstart, xend`` plus ``name``, ``score``, ``strand`` when the
file carries them (a ``.`` score becomes ``NaN``). The field count is
fixed by the first data line; shorter lines later fail loudly. Fields
beyond the sixth — the BED12 block structure — are ignored, not
expanded. narrowPeak adds ``signal_value, p_value, q_value, peak``;
broadPeak the same minus ``peak``.
"""

from __future__ import annotations

import gzip
import math
import os
from typing import Any, Optional

import pandas as pd

from ._chrom import ChromStyle, normalize_chrom
from ._gtf import _parse_region, _region_chrom_matches

__all__ = ["read_bed"]

_BED_COLUMNS = ("chrom", "xstart", "xend", "name", "score", "strand")
_FORMATS = {"bed": 3, "broadpeak": 9, "narrowpeak": 10}


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(
        path, "rt", encoding="utf-8"
    )


def _parse_score(token: str) -> float:
    return math.nan if token == "." else float(token)


def read_bed(
    path: str,
    *,
    region: Any = None,
    chrom_style: Optional[ChromStyle] = None,
    format: Optional[str] = None,
) -> pd.DataFrame:
    """Read a BED / narrowPeak / broadPeak file into a tidy interval table.

    Parameters
    ----------
    path
        BED-family file; ``.gz`` is handled transparently. ``track``,
        ``browser`` and ``#`` lines are skipped.
    region
        Keep only intervals overlapping this region (unclipped), given
        as ``"chrom:start-end"`` or ``(chrom, start, end)`` in 1-based
        half-open coordinates. A ``chr`` prefix mismatch is tolerated.
    chrom_style
        Rewrite chromosome names to ``"ucsc"`` (``chr1``) or
        ``"ensembl"`` (``1``). ``None`` leaves them as found.
    format
        ``"bed"``, ``"narrowPeak"`` or ``"broadPeak"``
        (case-insensitive). ``None`` (default) infers from the file name
        and falls back to ``"bed"``.

    Returns
    -------
    pandas.DataFrame
        See the module docstring for the column contract; coordinates
        are 1-based half-open.

    Examples
    --------
    >>> peaks = read_bed("m6A.narrowPeak.gz",
    ...                  region="chr7:11,000,000-11,020,000")  # doctest: +SKIP
    """
    if format is None:
        base = os.path.basename(str(path)).lower()
        if ".narrowpeak" in base:
            format = "narrowpeak"
        elif ".broadpeak" in base:
            format = "broadpeak"
        else:
            format = "bed"
    fmt = format.lower()
    if fmt not in _FORMATS:
        raise ValueError(
            f"read_bed: format must be 'bed', 'narrowPeak' or 'broadPeak' "
            f"(got {format!r})."
        )
    min_fields = _FORMATS[fmt]
    reg = _parse_region(region)

    rows: list = []
    n_keep: Optional[int] = None
    with _open(str(path)) as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if n_keep is None:
                n_keep = min(len(parts), 6) if fmt == "bed" else 6
            needed = max(min_fields, n_keep)
            if len(parts) < needed:
                raise ValueError(
                    f"read_bed({path!r}): line {lineno} has {len(parts)} "
                    f"field(s), expected at least {needed} for {format!r}."
                )
            chrom = parts[0]
            xstart = int(parts[1]) + 1
            xend = int(parts[2]) + 1
            if reg is not None:
                if not _region_chrom_matches(chrom, reg[0]):
                    continue
                if not (xstart < reg[2] and reg[1] < xend):
                    continue
            row: list = [chrom, xstart, xend]
            if n_keep >= 4:
                row.append(parts[3])
            if n_keep >= 5:
                row.append(_parse_score(parts[4]))
            if n_keep >= 6:
                row.append(parts[5])
            if fmt != "bed":
                row += [float(parts[6]), float(parts[7]), float(parts[8])]
                if fmt == "narrowpeak":
                    offset = int(parts[9])
                    row.append(math.nan if offset < 0 else float(xstart + offset))
            rows.append(row)

    if n_keep is None:
        n_keep = 3 if fmt == "bed" else 6
    columns = list(_BED_COLUMNS[:n_keep])
    if fmt != "bed":
        columns += ["signal_value", "p_value", "q_value"]
        if fmt == "narrowpeak":
            columns.append("peak")

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        df["xstart"] = df["xstart"].astype("int64")
        df["xend"] = df["xend"].astype("int64")
        if chrom_style is not None:
            df["chrom"] = [normalize_chrom(c, chrom_style) for c in df["chrom"]]
    return df.reset_index(drop=True)
