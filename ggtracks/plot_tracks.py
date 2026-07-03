"""``plot_tracks`` — compose stacked genome-browser tracks into one ggplot.

Each track is a list of already-constructed ggplot layers whose data
carries a ``track`` column (the facet-row key). Tracks are stacked with
``facet_grid2(rows="track", scales="free_y")`` over a shared
intron-compressed genomic x (``scale_x_genomic``), with per-track panel
heights via ``force_panelsizes`` and an independent fill/colour scale per
track via ``new_scale_*`` — so a gene-model track's feature legend and a
coverage track's cluster legend coexist without clashing.

All tracks must use a **numeric y** (e.g. transcript rank, depth, arc
height) so the single shared y aesthetic is type-consistent across the
free-y facets.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import ggplot2_py as gg
import ggh4x

from .mapper import GenomicMapper
from .scale import scale_x_genomic
from ._render import finalize_gg
from .theme import theme_tracks, PUB_BASE_SIZE

__all__ = ["Track", "plot_tracks"]


class Track:
    """One browser track: a name, its ggplot layers, a panel height, and the
    aesthetic (``"fill"``/``"colour"``/``None``) to start fresh before it."""

    __slots__ = ("name", "layers", "height", "new_scale")

    def __init__(
        self,
        name: str,
        layers: Sequence[Any],
        *,
        height: float = 1.0,
        new_scale: Optional[str] = None,
    ) -> None:
        self.name = name
        self.layers = list(layers)
        self.height = float(height)
        self.new_scale = new_scale


def plot_tracks(
    tracks: Sequence[Track],
    mapper: GenomicMapper,
    *,
    track_order: Optional[Sequence[str]] = None,
    genomic_xlim: Optional[Tuple[float, float]] = None,
    title: str = "",
    base_size: float = PUB_BASE_SIZE,
    show: bool = True,
    save: Any = None,
):
    """Stack *tracks* into one faceted browser ggplot over a shared genomic x.

    Parameters
    ----------
    tracks
        Ordered tracks (top → bottom). Each track's layer data must carry
        a ``track`` column equal to the track's ``name`` and a numeric
        ``y``.
    mapper
        Shared :class:`GenomicMapper` (one per region → all tracks aligned).
    track_order
        Explicit top→bottom facet order; defaults to the order of *tracks*.
    genomic_xlim
        Optional ``(start, end)`` genomic clip (display-space via mapper).
    """
    if not tracks:
        raise ValueError("plot_tracks: no tracks to plot.")
    order = list(track_order) if track_order is not None else [t.name for t in tracks]

    import pandas as _pd
    for tr in tracks:
        for lyr in tr.layers:
            data = getattr(lyr, "data", None)
            if isinstance(data, _pd.DataFrame) and "track" in data.columns:
                d = data.copy()
                d["track"] = _pd.Categorical(d["track"], categories=order, ordered=True)
                lyr.data = d

    import ggnewscale

    p = gg.ggplot()
    for i, tr in enumerate(tracks):
        if tr.new_scale == "fill":
            p = p + ggnewscale.new_scale_fill()
        elif tr.new_scale == "colour":
            p = p + ggnewscale.new_scale_colour()
        for lyr in tr.layers:
            p = p + lyr

    p = p + ggh4x.facet_grid2(rows="track", scales="free_y",
                              labeller=gg.label_wrap_gen(width=13))
    heights = [next(t.height for t in tracks if t.name == name) for name in order]
    p = p + ggh4x.force_panelsizes(rows=heights)

    comps = [scale_x_genomic(mapper)]
    if genomic_xlim is not None:
        lo, hi = genomic_xlim
        comps.append(gg.coord_cartesian(
            xlim=(float(mapper.to_display(lo)), float(mapper.to_display(hi)))
        ))
    for c in comps:
        p = p + c
    p = (
        p
        + gg.labs(title=title, x="Genomic position", y="")
        + theme_tracks(base_size)
    )
    return finalize_gg(p, show=show, save=save, height=sum(heights) + 0.5)
