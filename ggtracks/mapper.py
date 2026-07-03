"""Piecewise-linear genomic → display coordinate mapping with intron compression.

Every transcript-axis plot needs a shared mapper that compresses
intronic spans — without it, plotting a gene of >50 kb leaves the exons
as sub-pixel slivers.

The mapper is built from a sequence of ``(start, end, kind)`` spans
where ``kind ∈ {"exon", "intron"}``. The "exonic union" approach
(union of all transcripts' exons in the gene) gives a stable display
even when transcripts have different exon boundaries — junctions of
one isoform may fall *inside* an exon of another. The intron spans
that survive are the parts truly *no* transcript exons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np


Kind = Literal["exon", "intron"]


@dataclass(slots=True)
class GenomicSpan:
    """One piece of the piecewise mapping."""

    genomic_start: int
    genomic_end: int
    kind: Kind
    display_start: float
    display_end: float

    @property
    def genomic_length(self) -> int:
        return self.genomic_end - self.genomic_start

    @property
    def display_length(self) -> float:
        return self.display_end - self.display_start


class GenomicMapper:
    """Map genomic ↔ display coordinates with optional intron compression.

    Parameters
    ----------
    spans
        Ordered list of ``(genomic_start, genomic_end, kind)`` triples,
        each a **half-open interval** — ``genomic_end`` is exclusive, so a
        span's length is ``end - start``. The coordinate origin is the
        caller's (0- or 1-based both work): the mapper only computes
        *relative* layout, and ``to_genomic`` returns positions in the same
        frame you supplied. Spans must be in strict increasing order and
        must not overlap.
    intron_mode
        How intron spans are compressed (only spans of kind ``"intron"``
        with genomic length ``>= intron_min`` are affected):

        * ``"scale"`` (default) — *multiplicative*: display length =
          ``length * intron_scale`` (the historic behaviour).
        * ``"clamp"`` — *absolute*: display length =
          ``min(length, target_gap_width)`` (ggtranscript
          ``shorten_gaps``-style; never *enlarges* a gap).
    intron_scale
        Multiplier applied to intron length when ``intron_mode="scale"``.
        ``0.15`` = compress to 15% of original (default).
    target_gap_width
        The display width every compressible intron is clamped to when
        ``intron_mode="clamp"`` (in display units, which equal genomic bp
        for unscaled exons). Default ``100``.
    intron_min
        Introns shorter than this (in genomic bp) are NOT compressed.
    exon_scale
        Multiplier applied to exon length. Default 1.0.
    collapse_introns
        When False, disables compression entirely (identity map).

    Raises
    ------
    ValueError
        Spans are empty, not strictly ordered, overlap, or contain a
        non-positive length, or scale parameters are negative.
    """

    def __init__(
        self,
        spans: Sequence[tuple[int, int, Kind]],
        *,
        intron_mode: Literal["scale", "clamp"] = "scale",
        intron_scale: float = 0.15,
        target_gap_width: int = 100,
        intron_min: int = 20,
        exon_scale: float | Literal["none"] = 1.0,
        collapse_introns: bool = True,
    ) -> None:
        if not spans:
            raise ValueError(
                "GenomicMapper: spans is empty. Pass at least one "
                "(start, end, kind) triple."
            )
        if intron_mode not in ("scale", "clamp"):
            raise ValueError(
                f"GenomicMapper: intron_mode must be 'scale' or 'clamp' "
                f"(got {intron_mode!r})."
            )
        if exon_scale == "none":
            exon_scale_f = 1.0
        else:
            exon_scale_f = float(exon_scale)
        if exon_scale_f <= 0:
            raise ValueError(
                f"GenomicMapper: exon_scale must be > 0 (got {exon_scale!r})."
            )
        if intron_scale < 0:
            raise ValueError(
                f"GenomicMapper: intron_scale must be ≥ 0 (got {intron_scale!r})."
            )
        if target_gap_width < 0:
            raise ValueError(
                f"GenomicMapper: target_gap_width must be ≥ 0 "
                f"(got {target_gap_width!r})."
            )
        if intron_min < 0:
            raise ValueError(
                f"GenomicMapper: intron_min must be ≥ 0 (got {intron_min!r})."
            )

        self._intron_mode = intron_mode
        self._intron_scale = intron_scale
        self._target_gap_width = int(target_gap_width)
        self._intron_min = intron_min
        self._exon_scale = exon_scale_f
        self._collapse_introns = collapse_introns

        compiled: list[GenomicSpan] = []
        cursor = 0.0
        prev_end: int | None = None
        for i, (gs, ge, kind) in enumerate(spans):
            if ge <= gs:
                raise ValueError(
                    f"GenomicMapper: span #{i} has non-positive length "
                    f"({gs!r}, {ge!r}); end must be > start."
                )
            if kind not in ("exon", "intron"):
                raise ValueError(
                    f"GenomicMapper: span #{i} kind={kind!r}; "
                    f"expected 'exon' or 'intron'."
                )
            if prev_end is not None and gs < prev_end:
                raise ValueError(
                    f"GenomicMapper: span #{i} starts at {gs} but the "
                    f"previous span ended at {prev_end}. Spans must be "
                    f"strictly ordered without overlap."
                )
            length = ge - gs
            if kind == "exon":
                disp_len = length * self._exon_scale
            else:
                if not self._collapse_introns or length < self._intron_min:
                    disp_len = float(length)
                elif self._intron_mode == "clamp":
                    disp_len = float(min(length, self._target_gap_width))
                else:
                    disp_len = length * self._intron_scale
            compiled.append(
                GenomicSpan(
                    genomic_start=int(gs),
                    genomic_end=int(ge),
                    kind=kind,
                    display_start=cursor,
                    display_end=cursor + disp_len,
                )
            )
            cursor += disp_len
            prev_end = ge

        self._spans: tuple[GenomicSpan, ...] = tuple(compiled)
        self._g_starts = np.asarray(
            [s.genomic_start for s in compiled], dtype=np.int64
        )
        self._g_ends = np.asarray(
            [s.genomic_end for s in compiled], dtype=np.int64
        )
        self._d_starts = np.asarray(
            [s.display_start for s in compiled], dtype=np.float64
        )
        self._d_ends = np.asarray(
            [s.display_end for s in compiled], dtype=np.float64
        )

    @property
    def spans(self) -> tuple[GenomicSpan, ...]:
        return self._spans

    @property
    def display_extent(self) -> tuple[float, float]:
        return (0.0, float(self._d_ends[-1]))

    @property
    def genomic_extent(self) -> tuple[int, int]:
        return (int(self._g_starts[0]), int(self._g_ends[-1]))

    @property
    def intron_mode(self) -> str:
        return self._intron_mode

    @property
    def intron_scale(self) -> float:
        return self._intron_scale

    @property
    def target_gap_width(self) -> int:
        return self._target_gap_width

    @property
    def exon_scale(self) -> float:
        return self._exon_scale

    def to_display(self, pos: int | float) -> float:
        """Map a single genomic position to its display coordinate.

        Positions outside the mapper's extent are clamped to the nearest
        boundary.
        """
        return float(
            self.to_display_array(np.asarray([pos], dtype=np.float64))[0]
        )

    def to_display_array(self, positions: np.ndarray) -> np.ndarray:
        """Vectorized genomic → display."""
        pos = np.asarray(positions, dtype=np.float64)
        lo = float(self._g_starts[0])
        hi = float(self._g_ends[-1])
        pos_clamped = np.clip(pos, lo, hi)
        idx = np.searchsorted(self._g_starts, pos_clamped, side="right") - 1
        idx = np.clip(idx, 0, len(self._g_starts) - 1)
        g_start = self._g_starts[idx]
        g_end = self._g_ends[idx]
        d_start = self._d_starts[idx]
        d_end = self._d_ends[idx]
        gl = (g_end - g_start).astype(np.float64)
        gl[gl == 0] = 1.0
        frac = (pos_clamped - g_start) / gl
        return d_start + frac * (d_end - d_start)

    def to_genomic(self, display: float) -> float:
        return float(
            self.to_genomic_array(np.asarray([display], dtype=np.float64))[0]
        )

    def to_genomic_array(self, display: np.ndarray) -> np.ndarray:
        d = np.asarray(display, dtype=np.float64)
        d_min = self._d_starts[0]
        d_max = self._d_ends[-1]
        d_clamped = np.clip(d, d_min, d_max)
        idx = np.searchsorted(self._d_starts, d_clamped, side="right") - 1
        idx = np.clip(idx, 0, len(self._d_starts) - 1)
        d_start = self._d_starts[idx]
        d_end = self._d_ends[idx]
        g_start = self._g_starts[idx].astype(np.float64)
        g_end = self._g_ends[idx].astype(np.float64)
        dl = d_end - d_start
        dl[dl == 0] = 1.0
        frac = (d_clamped - d_start) / dl
        return g_start + frac * (g_end - g_start)

    @classmethod
    def from_intervals(
        cls,
        exons: Iterable[tuple[int, int]],
        *,
        intron_mode: Literal["scale", "clamp"] = "scale",
        intron_scale: float = 0.15,
        target_gap_width: int = 100,
        intron_min: int = 20,
        exon_scale: float | Literal["none"] = 1.0,
        collapse_introns: bool = True,
    ) -> "GenomicMapper":
        """Build a mapper from a list of EXON intervals.

        Intronic spans are filled in automatically between adjacent
        merged exons. Overlapping or adjacent exons are merged into a
        single span.

        Parameters
        ----------
        exons
            Iterable of ``(start, end)`` tuples — **half-open**
            ``[start, end)``. The origin is the caller's (0- or 1-based both
            work); the mapper only does relative layout, so ``[100, 201)``
            and ``[99, 200)`` produce the same display.
        """
        merged = _merge_intervals(list(exons))
        if not merged:
            raise ValueError(
                "GenomicMapper.from_intervals: no exons supplied."
            )
        spans: list[tuple[int, int, Kind]] = []
        prev_end: int | None = None
        for gs, ge in merged:
            if prev_end is not None and gs > prev_end:
                spans.append((prev_end, gs, "intron"))
            spans.append((gs, ge, "exon"))
            prev_end = ge
        return cls(
            spans,
            intron_mode=intron_mode,
            intron_scale=intron_scale,
            target_gap_width=target_gap_width,
            intron_min=intron_min,
            exon_scale=exon_scale,
            collapse_introns=collapse_introns,
        )

    def tick_positions(self, n: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """``(display_positions, genomic_labels)`` for ``n`` evenly spaced ticks."""
        d_min, d_max = self.display_extent
        disp = np.linspace(d_min, d_max, n)
        gen = self.to_genomic_array(disp)
        return disp, np.round(gen).astype(np.int64)

    def __repr__(self) -> str:
        n_exon = sum(1 for s in self._spans if s.kind == "exon")
        n_intron = sum(1 for s in self._spans if s.kind == "intron")
        g0, g1 = self.genomic_extent
        d0, d1 = self.display_extent
        if self._intron_mode == "clamp":
            comp = f"clamp(target_gap_width={self._target_gap_width})"
        else:
            comp = f"scale(intron_scale={self._intron_scale})"
        return (
            f"<GenomicMapper genomic={g0}-{g1} ({g1-g0:,} bp) → "
            f"display=0-{d1:.0f} ({n_exon} exons, {n_intron} introns, "
            f"{comp})>"
        )


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Sort and merge overlapping/adjacent half-open intervals."""
    if not intervals:
        return []
    sorted_iv = sorted((int(a), int(b)) for a, b in intervals if b > a)
    merged: list[tuple[int, int]] = [sorted_iv[0]]
    for s, e in sorted_iv[1:]:
        prev_s, prev_e = merged[-1]
        if s <= prev_e:
            merged[-1] = (prev_s, max(prev_e, e))
        else:
            merged.append((s, e))
    return merged
