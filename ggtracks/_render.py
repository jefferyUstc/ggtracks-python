"""Shared finalize hook for ggplot-based track figures.

Follows the scanpy-style ``show=`` / ``save=`` convention: a ``ggplot``
always renders itself in notebooks, so :func:`finalize_gg` returns the plot
unconditionally and only adds the ``save=`` side effect.  ``show=False`` is
kept for API symmetry with imperative plotting backends.

Sizing is deliberately *measured* rather than guessed — see
:func:`natural_height`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import ggplot2_py as gg

__all__ = ["finalize_gg", "natural_height"]


def natural_height(plot: Any) -> float:
    """Absolute height of *plot* in inches, ignoring relative panels.

    Builds the plot's gtable and sums its row heights.  Panels sized in
    ``"null"`` (relative) units contribute **zero**, because a relative unit
    has no absolute value until it is resolved against a device; everything
    else — axes, titles, facet strips, legends, margins — contributes its
    true measured size.

    So for a figure whose panels are relative, this returns exactly the
    *chrome*: the fixed overhead a caller must add on top of its panel
    allowance.  That number is not a constant — it grows with ``base_size``
    (at base 8 it is ≈0.48 in, at base 20 ≈0.91 in), which is why hard-coding
    it silently truncates large-typography figures.

    When panels are given absolute sizes (a ``grid_py.Unit`` of inches, cm,
    lines, …) their size is included too, and the result is the figure's
    complete natural height.
    """
    from ggplot2_py.plot_render import ggplotGrob
    from gtable_py import gtable_height
    from grid_py import convert_height

    gt = ggplotGrob(plot)
    inches = convert_height(gtable_height(gt), "inches", valueOnly=True)
    return float(np.asarray(inches).ravel().sum())


def finalize_gg(
    plot: Any,
    *,
    show: bool = True,
    save: Union[str, Path, None] = None,
    width: float = 4.8,
    height: Optional[float] = 3.0,
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

    Parameters
    ----------
    height
        Figure height in inches. Pass ``None`` to **measure** it from the
        plot's own gtable via :func:`natural_height` instead of supplying a
        number — appropriate when every panel carries an absolute size.
    """
    if height is None:
        height = natural_height(plot)
    plot.fig_width = float(width)
    plot.fig_height = float(height)
    plot.fig_dpi = int(display_dpi)
    if save is not None:
        gg.ggsave(str(save), plot, width=float(width), height=float(height), dpi=dpi)
    return plot
