"""bedGraph → tidy interval table.

bedGraph is the uncompressed, unindexed cousin of bigWig: four columns,
``chrom start end value``. Without an index there is nothing to seek to, so
the file is read once into arrays and queried in memory — re-scanning the
file per query is what makes naive readers quadratic when several regions
are plotted.

**Coordinates.** bedGraph is 0-based half-open; the frames returned here are
1-based half-open, matching the rest of :mod:`ggtracks.io`.
"""

from __future__ import annotations

import gzip
from typing import Optional

import numpy as np
import pandas as pd

from ._chrom import resolve_chrom

__all__ = ["BedGraph", "read_bedgraph"]


def _open(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(
        path, "rt", encoding="utf-8"
    )


class BedGraph:
    """In-memory bedGraph, indexed by chromosome on load."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        per_chrom: dict = {}
        with _open(self.path) as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(
                        f"BedGraph({self.path!r}): line {lineno} has "
                        f"{len(parts)} fields, expected at least 4 "
                        "(chrom start end value)."
                    )
                chrom = parts[0]
                bucket = per_chrom.setdefault(chrom, ([], [], []))
                bucket[0].append(int(parts[1]))
                bucket[1].append(int(parts[2]))
                bucket[2].append(float(parts[3]))

        self._data: dict = {}
        for chrom, (starts, ends, values) in per_chrom.items():
            s = np.asarray(starts, dtype=np.int64)
            e = np.asarray(ends, dtype=np.int64)
            v = np.asarray(values, dtype=np.float64)
            order = np.argsort(s, kind="stable")
            self._data[chrom] = (s[order], e[order], v[order])

    def __repr__(self) -> str:
        n = sum(len(s) for s, _e, _v in self._data.values())
        return (
            f"<BedGraph {self.path!r} chroms={len(self._data)} intervals={n}>"
        )

    @property
    def chroms(self) -> dict:
        """``{chromosome: highest end coordinate seen}``."""
        return {c: int(e.max()) if e.size else 0 for c, (_s, e, _v) in self._data.items()}

    def to_frame(self) -> pd.DataFrame:
        """The whole file as ``chrom, xstart, xend, value`` (1-based half-open)."""
        frames = [
            pd.DataFrame({"chrom": c, "xstart": s + 1, "xend": e + 1, "value": v})
            for c, (s, e, v) in self._data.items()
        ]
        if not frames:
            return pd.DataFrame(columns=["chrom", "xstart", "xend", "value"])
        return pd.concat(frames, ignore_index=True)

    def query(self, chrom: str, start: int, end: int) -> pd.DataFrame:
        """Intervals overlapping ``[start, end)`` (1-based), clipped to it."""
        start, end = int(start), int(end)
        if start < 1:
            raise ValueError(
                f"BedGraph.query: start is 1-based and must be >= 1 (got {start})."
            )
        if end <= start:
            raise ValueError(
                f"BedGraph.query: end must exceed start (got {start}-{end})."
            )
        key = resolve_chrom(chrom, self._data)
        s, e, v = self._data[key]
        lo, hi = start - 1, end - 1
        keep = (e > lo) & (s < hi)
        return pd.DataFrame(
            {
                "xstart": np.clip(s[keep], lo, hi) + 1,
                "xend": np.clip(e[keep], lo, hi) + 1,
                "value": v[keep],
            }
        )


def read_bedgraph(
    path: str,
    chrom: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
) -> pd.DataFrame:
    """Read a bedGraph, optionally restricted to one region.

    With no region the whole file is returned as ``chrom, xstart, xend,
    value``; with one, the frame matches :meth:`BigWig.query` so the two
    signal sources are interchangeable downstream.
    """
    bg = BedGraph(path)
    if chrom is None:
        return bg.to_frame()
    if start is None or end is None:
        raise ValueError(
            "read_bedgraph: give both start and end when a chrom is supplied."
        )
    return bg.query(chrom, start, end)
