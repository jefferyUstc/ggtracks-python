"""Genomic coordinate system for track plots — intron compression.

Built on this package's :class:`~ggtracks.mapper.GenomicMapper` plus
ggplot2-python's scale-transform machinery. Data is authored in **true
genomic coordinates**; the transform compresses introns at *render* time
(the same mechanism a log scale uses) and the x axis shows
genomic-valued tick labels via the inverse map. No display coordinates
are ever baked into the data.

``scale_x_genomic`` is the engine: a continuous x position scale whose
transform is the piecewise-linear genomic→display map. Because
``xstart``/``xend`` (and the standard ``xmin``/``xmax``/``x``) are
registered x aesthetics, every range/intron/junction geom is compressed
consistently and stays aligned across stacked tracks that share one
mapper. ``coord_genomic`` is the user-facing convenience that adds the
scale and, optionally, a genomic region clip.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np
import ggplot2_py as gg
from scales import new_transform, label_number, cut_si, extended_breaks

from .mapper import GenomicMapper

__all__ = ["genomic_transform", "scale_x_genomic", "coord_genomic"]


def genomic_transform(
    mapper: GenomicMapper,
    *,
    name: str = "genomic",
    labels: Optional[Callable] = None,
    n_breaks: int = 5,
):
    """A :class:`scales.Transform` for genomic→display coordinates.

    Forward = ``mapper.to_display_array``; inverse = ``to_genomic_array``.
    Breaks are generated in genomic space (so they land on round genomic
    positions) and formatted with SI bp units (``120 b`` / ``4 kb`` /
    ``32 Mb``) unless *labels* overrides the formatter.
    """
    g0, g1 = mapper.genomic_extent

    def _fwd(x: Any) -> np.ndarray:
        return mapper.to_display_array(np.asarray(x, dtype=float))

    def _inv(x: Any) -> np.ndarray:
        return mapper.to_genomic_array(np.asarray(x, dtype=float))

    def _breaks(limits: Sequence[float], n: int = n_breaks) -> np.ndarray:
        lo, hi = float(min(limits)), float(max(limits))
        if lo == hi:
            return np.asarray([lo])
        cand = np.asarray(extended_breaks(n=n)((lo, hi)), dtype=float)
        cand = cand[(cand >= lo) & (cand <= hi)]
        if cand.size <= 1:
            return cand
        disp = mapper.to_display_array(cand)
        span = abs(float(mapper.to_display(hi)) - float(mapper.to_display(lo))) or 1.0
        min_sep = 0.06 * span
        keep = [0]
        for i in range(1, cand.size):
            if abs(disp[i] - disp[keep[-1]]) >= min_sep:
                keep.append(i)
        return cand[keep]

    fmt = labels if labels is not None else label_number(scale_cut=cut_si("b"))
    return new_transform(
        name,
        transform=_fwd,
        inverse=_inv,
        breaks=_breaks,
        format=fmt,
        domain=(float(g0), float(g1)),
    )


def scale_x_genomic(
    mapper: GenomicMapper,
    *,
    name: Any = None,
    labels: Optional[Callable] = None,
    n_breaks: int = 5,
    expand: Any = None,
    **kwargs: Any,
):
    """Continuous x scale that compresses introns via *mapper*.

    Drop-in for ``scale_x_continuous``; everything else (limits, guides,
    secondary axes) composes normally. Data x/xstart/xend/xmin/xmax stay
    genomic and are transformed to the compressed display space at
    render, with genomic tick labels.

    The transform must reach the ggtranscript-style ``xstart``/``xend``
    aesthetics (not just ``x``/``xmin``/``xmax``). Rather than registering
    ``xstart`` in the *global* position-aesthetic list (which would make
    every default scale deviate from R), we extend only **this scale
    instance**'s aesthetics — so ``geom_range``/``geom_intron`` get
    compressed while the package's default scales stay R-faithful.
    """
    sc = gg.scale_x_continuous(
        transform=genomic_transform(mapper, labels=labels, n_breaks=n_breaks),
        name=name,
        expand=expand,
        **kwargs,
    )
    if "xstart" not in sc.aesthetics:
        sc.aesthetics = list(sc.aesthetics) + ["xstart"]
    return sc


def coord_genomic(
    mapper: GenomicMapper,
    *,
    genomic_xlim: Optional[Tuple[float, float]] = None,
    expand: Any = None,
    labels: Optional[Callable] = None,
    n_breaks: int = 5,
    **kwargs: Any,
) -> list:
    """Add genomic intron-compression to a plot (the user-facing entry).

    Returns a list of ggplot components: ``scale_x_genomic(mapper)`` and,
    when *genomic_xlim* is given, a ``coord_cartesian`` that clips the
    view to that genomic interval (converted to display space through the
    mapper). Add it to a plot like any component::

        p + coord_genomic(mapper, genomic_xlim=(31_659_500, 31_660_000))

    The list form keeps the genomic axis labels (from the scale) while
    letting the user zoom in genomic terms.
    """
    comps: list = [
        scale_x_genomic(
            mapper, labels=labels, n_breaks=n_breaks, expand=expand, **kwargs
        )
    ]
    if genomic_xlim is not None:
        lo, hi = genomic_xlim
        dlo = float(mapper.to_display(lo))
        dhi = float(mapper.to_display(hi))
        comps.append(gg.coord_cartesian(xlim=(dlo, dhi)))
    return comps
