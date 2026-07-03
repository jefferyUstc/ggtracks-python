"""``StatPileup`` — assign each read a vertical row for pileup plots.

Operates per panel (so each facet, e.g. a cell type, packs independently)
and per ``group`` (one group = one read; all of a read's blocks share a
row). Two modes:

* ``"stack"`` — one row per read, ordered by 5′ start (the dense IGV
  "squished" look where every read has its own line).
* ``"pack"`` — greedy interval partition so non-overlapping reads share a
  row (the compact IGV "collapsed" look); this is the disjoint-ranges
  algorithm (cf. ggh4x ``position_disjoint_ranges``).

The computed row is written to ``y`` so it pairs directly with
:class:`~ggtracks.GeomRange` (blocks) and
:class:`~ggtracks.GeomIntron` (connectors).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd

from ggplot2_py.stat import Stat

__all__ = ["StatPileup", "pack_rows"]


def pack_rows(lo: np.ndarray, hi: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Greedy interval partition → a row index per interval.

    Intervals are assigned (in ascending ``lo`` order) to the lowest row
    whose currently-occupied right edge does not overlap; a new row opens
    only when none is free. Returns rows in the *original* input order.
    """
    n = len(lo)
    order = np.argsort(lo, kind="stable")
    row_last_hi: List[float] = []
    rows = np.empty(n, dtype=float)
    for idx in order:
        placed = False
        for r, last in enumerate(row_last_hi):
            if lo[idx] > last:
                rows[idx] = r * spacing
                row_last_hi[r] = hi[idx]
                placed = True
                break
        if not placed:
            rows[idx] = len(row_last_hi) * spacing
            row_last_hi.append(hi[idx])
    return rows


class StatPileup(Stat):
    """Assign reads to rows for a pileup; writes ``y`` = row."""

    required_aes: List[str] = ["xstart", "xend"]
    dropped_aes: List[str] = []

    def compute_panel(
        self,
        data: pd.DataFrame,
        scales: Any,
        mode: str = "stack",
        spacing: float = 1.0,
        **params: Any,
    ) -> pd.DataFrame:
        spacing = float(spacing)
        if mode not in ("stack", "pack"):
            raise ValueError(
                f"StatPileup: mode must be 'stack' or 'pack' (got {mode!r})."
            )
        data = data.copy()
        if "group" not in data.columns:
            data["group"] = 0

        ext = (
            data.groupby("group", sort=False)
            .agg(lo=("xstart", "min"), hi=("xend", "max"))
            .reset_index()
        )
        ext = ext.sort_values("lo", kind="stable").reset_index(drop=True)
        if mode == "stack":
            row = np.arange(len(ext), dtype=float) * spacing
        else:
            row = pack_rows(ext["lo"].to_numpy(), ext["hi"].to_numpy(), spacing)
        row_map = dict(zip(ext["group"], row))
        data["y"] = data["group"].map(row_map).astype(float)
        return data
