"""Pure-Python bigWig reader — no compiled dependency.

A plotting library should be installable without a C toolchain, so the
format is parsed here with :mod:`struct` and :mod:`zlib` alone.

The implementation follows the ``bbiFile`` layout: a 64-byte header, a
B+ tree of chromosome names, an R-tree ("cirTree") index over the data
blocks, and — the part that makes browsers fast — a stack of pre-computed
**zoom levels**, each a coarser summary of the whole file.

Three properties are load-bearing and easy to get wrong:

* **The header is 64 bytes.** Stopping short leaves every zoom header
  misaligned, which is silent because a misparsed zoom header is only
  noticed if you actually use it.
* **``uncompressBufSize == 0`` means the blocks are stored uncompressed.**
  Inflating unconditionally fails on such files, and swallowing that
  failure turns a corrupt read into an empty plot.
* **The R-tree is for descending, not for walking.** Materialising every
  leaf and scanning it linearly makes each query cost the whole index.

Nothing here degrades quietly: a bad magic number, a truncated node or a
failed inflate raises.

**Coordinates.** bigWig stores 0-based half-open intervals; annotations are
1-based. Rather than leave that one-base skew for the caller to trip over
when a coverage track is drawn under a gene model, this module speaks the
same 1-based half-open dialect as :mod:`ggtracks.io` as a whole — query
bounds are 1-based, and returned intervals are too.
"""

from __future__ import annotations

import struct
import zlib
from typing import Any, Iterable, List, Literal, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._binning import bin_intervals
from ._chrom import resolve_chrom

__all__ = ["BigWig", "read_bigwig"]

_BIGWIG_MAGIC = 0x888FFC26
_BIGBED_MAGIC = 0x8789F2EB
_BPT_MAGIC = 0x78CA8C91
_RTREE_MAGIC = 0x2468ACE0

_HEADER_SIZE = 64
_ZOOM_HEADER_SIZE = 24
_RTREE_HEADER_SIZE = 48
_SECTION_HEADER_SIZE = 24
_ZOOM_RECORD_SIZE = 32

Summary = Literal["mean", "max", "min", "sum"]


class _Zoom(NamedTuple):
    reduction: int
    data_offset: int
    index_offset: int


class _Block(NamedTuple):
    offset: int
    size: int


class BigWig:
    """Random-access reader for a bigWig file.

    Parameters
    ----------
    path
        Path to a ``.bw`` / ``.bigWig`` file.

    Raises
    ------
    ValueError
        The file is not a bigWig (including the common mix-up of handing it
        a bigBed), or its internal structure is corrupt.

    Examples
    --------
    >>> with BigWig("signal.bw") as bw:                      # doctest: +SKIP
    ...     df = bw.query("chr7", 10_900_000, 11_000_000, bins=500)
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._fh = open(self.path, "rb")
        self._order = self._read_header()
        self._chroms: dict = {}
        self._chrom_ids: dict = {}
        self._read_chrom_tree()

    # -- construction ------------------------------------------------------

    def _read(self, offset: int, size: int) -> bytes:
        self._fh.seek(offset)
        raw = self._fh.read(size)
        if len(raw) != size:
            raise ValueError(
                f"BigWig({self.path!r}): truncated read of {size} bytes at "
                f"offset {offset} (got {len(raw)})."
            )
        return raw

    def _read_header(self) -> str:
        raw = self._read(0, _HEADER_SIZE)
        magic_le = struct.unpack_from("<I", raw, 0)[0]
        magic_be = struct.unpack_from(">I", raw, 0)[0]
        if magic_le == _BIGWIG_MAGIC:
            order = "<"
        elif magic_be == _BIGWIG_MAGIC:
            order = ">"
        elif _BIGBED_MAGIC in (magic_le, magic_be):
            raise ValueError(
                f"BigWig({self.path!r}): this is a bigBed file, not a bigWig."
            )
        else:
            raise ValueError(
                f"BigWig({self.path!r}): not a bigWig file "
                f"(magic 0x{magic_le:08X})."
            )

        fields = struct.unpack_from(order + "IHHQQQHHQQIQ", raw, 0)
        (
            _magic,
            self.version,
            n_zoom,
            self._chrom_tree_offset,
            self._full_data_offset,
            self._full_index_offset,
            self.field_count,
            self.defined_field_count,
            self._autosql_offset,
            self._total_summary_offset,
            self._uncompress_buf_size,
            _reserved,
        ) = fields

        zooms: List[_Zoom] = []
        for i in range(n_zoom):
            chunk = self._read(_HEADER_SIZE + i * _ZOOM_HEADER_SIZE, _ZOOM_HEADER_SIZE)
            reduction, _res, data_off, index_off = struct.unpack(order + "IIQQ", chunk)
            zooms.append(_Zoom(reduction, data_off, index_off))
        # Coarsest last, so a search for "the finest level at least as coarse
        # as the target" can stop at the first hit.
        self._zooms = tuple(sorted(zooms, key=lambda z: z.reduction))
        return order

    def _read_chrom_tree(self) -> None:
        order = self._order
        head = self._read(self._chrom_tree_offset, 32)
        magic, _block_size, key_size, val_size, _count, _res = struct.unpack(
            order + "IIIIQQ", head
        )
        if magic != _BPT_MAGIC:
            raise ValueError(
                f"BigWig({self.path!r}): bad chromosome B+ tree magic "
                f"0x{magic:08X}."
            )
        if val_size != 8:
            raise ValueError(
                f"BigWig({self.path!r}): unexpected B+ tree value size {val_size} "
                "(expected 8)."
            )

        stack = [self._chrom_tree_offset + 32]
        while stack:
            node_offset = stack.pop()
            is_leaf, _res, count = struct.unpack(
                order + "BBH", self._read(node_offset, 4)
            )
            item_size = key_size + (val_size if is_leaf else 8)
            body = self._read(node_offset + 4, count * item_size)
            for i in range(count):
                base = i * item_size
                key = body[base : base + key_size].rstrip(b"\x00").decode()
                if is_leaf:
                    chrom_id, chrom_size = struct.unpack_from(
                        order + "II", body, base + key_size
                    )
                    self._chroms[key] = chrom_size
                    self._chrom_ids[key] = chrom_id
                else:
                    (child,) = struct.unpack_from(order + "Q", body, base + key_size)
                    stack.append(child)

    # -- public surface ----------------------------------------------------

    @property
    def chroms(self) -> dict:
        """``{chromosome: length}`` as recorded in the file."""
        return dict(self._chroms)

    @property
    def zoom_levels(self) -> Tuple[int, ...]:
        """Reduction levels available, finest first."""
        return tuple(z.reduction for z in self._zooms)

    def close(self) -> None:
        """Release the file handle. Also done by leaving a ``with`` block."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "BigWig":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"<BigWig {self.path!r} version={self.version} "
            f"chroms={len(self._chroms)} zooms={len(self._zooms)}>"
        )

    # -- index descent -----------------------------------------------------

    def _overlapping_blocks(
        self, index_offset: int, chrom_id: int, start: int, end: int
    ) -> List[_Block]:
        """Descend the R-tree, visiting only nodes that can overlap the query."""
        order = self._order
        head = self._read(index_offset, _RTREE_HEADER_SIZE)
        magic = struct.unpack_from(order + "I", head, 0)[0]
        if magic != _RTREE_MAGIC:
            raise ValueError(
                f"BigWig({self.path!r}): bad R-tree magic 0x{magic:08X} at "
                f"offset {index_offset}."
            )

        def touches(s_ix: int, s_base: int, e_ix: int, e_base: int) -> bool:
            if (e_ix, e_base) <= (chrom_id, start):
                return False
            if (s_ix, s_base) >= (chrom_id, end):
                return False
            return True

        blocks: List[_Block] = []
        stack = [index_offset + _RTREE_HEADER_SIZE]
        while stack:
            node_offset = stack.pop()
            is_leaf, _res, count = struct.unpack(
                order + "BBH", self._read(node_offset, 4)
            )
            item_size = 32 if is_leaf else 24
            body = self._read(node_offset + 4, count * item_size)
            for i in range(count):
                base = i * item_size
                s_ix, s_base, e_ix, e_base = struct.unpack_from(
                    order + "IIII", body, base
                )
                if not touches(s_ix, s_base, e_ix, e_base):
                    continue
                if is_leaf:
                    data_offset, data_size = struct.unpack_from(
                        order + "QQ", body, base + 16
                    )
                    blocks.append(_Block(data_offset, data_size))
                else:
                    (child,) = struct.unpack_from(order + "Q", body, base + 16)
                    stack.append(child)
        return blocks

    def _block_bytes(self, block: _Block) -> bytes:
        raw = self._read(block.offset, block.size)
        # A zero buffer size is the format's way of saying "stored raw".
        return zlib.decompress(raw) if self._uncompress_buf_size else raw

    # -- decoding ----------------------------------------------------------

    def _decode_sections(
        self, blocks: Iterable[_Block], chrom_id: int, start: int, end: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full-resolution intervals overlapping the query."""
        order = self._order
        starts: List[np.ndarray] = []
        ends: List[np.ndarray] = []
        values: List[np.ndarray] = []

        for block in blocks:
            raw = self._block_bytes(block)
            pos = 0
            while pos + _SECTION_HEADER_SIZE <= len(raw):
                (
                    sec_chrom,
                    sec_start,
                    _sec_end,
                    step,
                    span,
                    sec_type,
                    _res,
                    item_count,
                ) = struct.unpack_from(order + "IIIIIBBH", raw, pos)
                pos += _SECTION_HEADER_SIZE

                if sec_type == 1:  # bedGraph: start, end, value
                    need = item_count * 12
                    arr = np.frombuffer(raw, dtype=np.dtype(
                        [("s", order + "u4"), ("e", order + "u4"), ("v", order + "f4")]
                    ), count=item_count, offset=pos)
                    s = arr["s"].astype(np.int64)
                    e = arr["e"].astype(np.int64)
                    v = arr["v"].astype(np.float64)
                elif sec_type == 2:  # variableStep: start, value
                    need = item_count * 8
                    arr = np.frombuffer(raw, dtype=np.dtype(
                        [("s", order + "u4"), ("v", order + "f4")]
                    ), count=item_count, offset=pos)
                    s = arr["s"].astype(np.int64)
                    e = s + span
                    v = arr["v"].astype(np.float64)
                elif sec_type == 3:  # fixedStep: value only
                    need = item_count * 4
                    v = np.frombuffer(
                        raw, dtype=np.dtype(order + "f4"), count=item_count, offset=pos
                    ).astype(np.float64)
                    s = sec_start + np.arange(item_count, dtype=np.int64) * step
                    e = s + span
                else:
                    raise ValueError(
                        f"BigWig({self.path!r}): unknown section type {sec_type}."
                    )
                pos += need

                if sec_chrom != chrom_id:
                    continue
                keep = (e > start) & (s < end)
                if keep.any():
                    starts.append(s[keep])
                    ends.append(e[keep])
                    values.append(v[keep])

        if not starts:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, np.empty(0, dtype=np.float64)
        return (
            np.concatenate(starts),
            np.concatenate(ends),
            np.concatenate(values),
        )

    def _decode_zoom(
        self, blocks: Iterable[_Block], chrom_id: int, start: int, end: int, summary: Summary
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pre-summarised records from a zoom level, reduced to *summary*."""
        order = self._order
        dtype = np.dtype(
            [
                ("chrom", order + "u4"),
                ("start", order + "u4"),
                ("end", order + "u4"),
                ("valid", order + "u4"),
                ("min", order + "f4"),
                ("max", order + "f4"),
                ("sum", order + "f4"),
                ("sumsq", order + "f4"),
            ]
        )
        starts: List[np.ndarray] = []
        ends: List[np.ndarray] = []
        values: List[np.ndarray] = []

        for block in blocks:
            raw = self._block_bytes(block)
            n = len(raw) // _ZOOM_RECORD_SIZE
            if n == 0:
                continue
            arr = np.frombuffer(raw, dtype=dtype, count=n)
            keep = (
                (arr["chrom"] == chrom_id)
                & (arr["end"].astype(np.int64) > start)
                & (arr["start"].astype(np.int64) < end)
            )
            if not keep.any():
                continue
            rec = arr[keep]
            valid = rec["valid"].astype(np.float64)
            if summary == "mean":
                v = np.divide(
                    rec["sum"].astype(np.float64),
                    valid,
                    out=np.zeros(len(rec)),
                    where=valid > 0,
                )
            elif summary == "max":
                v = rec["max"].astype(np.float64)
            elif summary == "min":
                v = rec["min"].astype(np.float64)
            else:  # sum
                # ``sumData`` is already an integral over the record's bases,
                # whereas full-resolution records carry a per-base value. Hand
                # back a density so that both paths mean the same thing once
                # ``_rebin`` multiplies by the overlap length.
                span = (rec["end"].astype(np.float64) - rec["start"].astype(np.float64))
                v = np.divide(
                    rec["sum"].astype(np.float64),
                    span,
                    out=np.zeros(len(rec)),
                    where=span > 0,
                )
            starts.append(rec["start"].astype(np.int64))
            ends.append(rec["end"].astype(np.int64))
            values.append(v)

        if not starts:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, np.empty(0, dtype=np.float64)
        return np.concatenate(starts), np.concatenate(ends), np.concatenate(values)

    # -- query -------------------------------------------------------------

    def _pick_zoom(self, span: int, bins: int) -> Optional[_Zoom]:
        """Finest zoom level that is still coarse enough for *bins* columns.

        Reading a level finer than the requested resolution just decompresses
        data that is about to be averaged away; reading one coarser than the
        bin width would visibly under-resolve the plot. Returns ``None`` when
        every level is too coarse, in which case full resolution is used.
        """
        target = span / float(bins)
        chosen = None
        for z in self._zooms:
            if z.reduction <= target:
                chosen = z
            else:
                break
        return chosen

    def query(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        bins: Optional[int] = None,
        summary: Summary = "mean",
    ) -> pd.DataFrame:
        """Read signal over ``[start, end)`` as a tidy interval frame.

        Parameters
        ----------
        chrom
            Chromosome name; a ``chr`` prefix mismatch is reconciled
            automatically (and reported if it cannot be).
        start, end
            Half-open genomic interval, **1-based** — the same frame
            :func:`~ggtracks.io.read_annotations` returns, so a coverage
            track and a gene model line up without a manual shift.
        bins
            ``None`` (default) returns the file's own intervals at full
            resolution — nothing is averaged behind your back. An integer
            resamples the region into that many equal bins, drawing on a
            pre-computed zoom level when one is coarse enough to serve.
        summary
            How values are combined within a bin: ``"mean"`` (coverage
            weighted), ``"max"``, ``"min"`` or ``"sum"``. Ignored when
            *bins* is ``None``.

        Returns
        -------
        pandas.DataFrame
            Columns ``xstart``, ``xend``, ``value`` — half-open, sorted, and
            ready for :func:`~ggtracks.geom_coverage`.
        """
        if summary not in ("mean", "max", "min", "sum"):
            raise ValueError(
                f"BigWig.query: summary must be 'mean', 'max', 'min' or 'sum' "
                f"(got {summary!r})."
            )
        start, end = int(start), int(end)
        if start < 1:
            raise ValueError(
                f"BigWig.query: start is 1-based and must be >= 1 (got {start})."
            )
        if end <= start:
            raise ValueError(
                f"BigWig.query: end must exceed start (got {start}-{end})."
            )
        if bins is not None and (not isinstance(bins, int) or isinstance(bins, bool) or bins < 1):
            raise ValueError(f"BigWig.query: bins must be a positive int (got {bins!r}).")

        key = resolve_chrom(chrom, self._chroms)
        chrom_id = self._chrom_ids[key]

        # The file itself is 0-based; drop into that frame for the lookup and
        # lift the results back on the way out.
        lo, hi = start - 1, end - 1

        zoom = self._pick_zoom(hi - lo, bins) if bins is not None else None
        if zoom is None:
            blocks = self._overlapping_blocks(self._full_index_offset, chrom_id, lo, hi)
            s, e, v = self._decode_sections(blocks, chrom_id, lo, hi)
        else:
            blocks = self._overlapping_blocks(zoom.index_offset, chrom_id, lo, hi)
            s, e, v = self._decode_zoom(blocks, chrom_id, lo, hi, summary)

        if s.size:
            sort = np.argsort(s, kind="stable")
            s, e, v = s[sort], e[sort], v[sort]

        if bins is None:
            return pd.DataFrame(
                {
                    "xstart": np.clip(s, lo, hi) + 1,
                    "xend": np.clip(e, lo, hi) + 1,
                    "value": v,
                }
            )
        binned = _rebin(s, e, v, lo, hi, bins, summary)
        binned["xstart"] += 1
        binned["xend"] += 1
        return binned


def _rebin(
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
    bins: int,
    summary: Summary,
) -> pd.DataFrame:
    """Resample records onto *bins* equal genomic bins (see :mod:`ggtracks._binning`)."""
    lo, hi, value = bin_intervals(starts, ends, values, start, end, bins, summary)
    return pd.DataFrame({"xstart": lo, "xend": hi, "value": value})


def read_bigwig(
    path: str,
    chrom: str,
    start: int,
    end: int,
    *,
    bins: Optional[int] = None,
    summary: Summary = "mean",
) -> pd.DataFrame:
    """One-shot :class:`BigWig` query — open, read, close.

    Convenient for a single track; keep a :class:`BigWig` open instead when
    querying the same file repeatedly, so the chromosome index is parsed once.
    """
    with BigWig(path) as bw:
        return bw.query(chrom, start, end, bins=bins, summary=summary)
