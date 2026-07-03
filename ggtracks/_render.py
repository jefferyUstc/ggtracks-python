"""Shared finalize hook for ggplot-based track figures.

Follows the scanpy-style ``show=`` / ``save=`` convention: a ``ggplot``
always renders itself in notebooks, so :func:`finalize_gg` returns the plot
unconditionally and only adds the ``save=`` side effect. ``show=False`` is
kept for API symmetry with imperative plotting backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import ggplot2_py as gg

__all__ = ["finalize_gg"]


def finalize_gg(
    plot: Any,
    *,
    show: bool = True,
    save: Union[str, Path, None] = None,
    width: float = 4.8,
    height: float = 3.0,
    dpi: int = 300,
    display_dpi: int = 150,
) -> Any:
    """Set the figure size, apply ``save=``, and return the ``ggplot``.

    A ``ggplot`` is the composable, inspectable artifact we want callers to
    keep — so it is always returned. The key job here is **WYSIWYG sizing**:
    a ``GGPlot`` carries ``fig_width``/``fig_height``/``fig_dpi`` that drive
    its notebook rendering (``_repr_png_``), but those default to a fixed
    7×5 in regardless of content, and ``ggsave`` has its own 7×7 default —
    so what you see and what you save diverge. We pin the plot's display
    size to the content-appropriate ``width``×``height`` (compact default
    4.8 in wide, so a figure drops into a panel at 1:1 — 8-pt text stays
    8-pt, no rescaling) and save at exactly the same
    inches, so the notebook figure and the saved file are dimensionally
    identical (only the raster resolution differs: ``display_dpi`` on screen
    vs print ``dpi`` on disk; physical layout — fonts in pt, lines in mm —
    is invariant to dpi).
    """
    plot.fig_width = float(width)
    plot.fig_height = float(height)
    plot.fig_dpi = int(display_dpi)
    if save is not None:
        gg.ggsave(str(save), plot, width=float(width), height=float(height), dpi=dpi)
    return plot
