"""``geom_highlight`` — a region band drawn across every track.

Calling out a region ("this exon", "this peak", "the variant") means a
vertical band that runs through the whole stack, so the eye can follow it
from the coverage down to the gene model.

Two pieces of existing grammar do all the work, so this is a thin wrapper
rather than a new geom:

* ``ymin=-inf`` / ``ymax=inf`` make a rectangle span whatever height its
  panel happens to have, without widening the y scale;
* a layer whose data **omits the faceting variable** is repeated into every
  panel — standard ggplot2 behaviour. So a highlight frame that carries no
  ``track`` column lands on all tracks automatically, while one that does
  carry a ``track`` value is confined to that track.
"""

from __future__ import annotations

from .palettes import FEATURE_COLOURS

from typing import Any, Optional

import numpy as np
import pandas as pd

import ggplot2_py as gg

__all__ = ["geom_highlight"]


def geom_highlight(
    data: Any = None,
    mapping: Any = None,
    *,
    xstart: Optional[float] = None,
    xend: Optional[float] = None,
    fill: str = FEATURE_COLOURS["highlight"],
    alpha: float = 0.2,
    colour: Any = None,
    inherit_aes: bool = False,
    **kwargs: Any,
):
    """Shade one or more genomic regions across the full panel height.

    Parameters
    ----------
    data
        Frame with ``xstart`` / ``xend`` columns (half-open, same frame as
        every other ggtracks layer). Include a ``track`` column to confine
        the band to that track; omit it and the band spans them all.
    mapping
        Extra aesthetics, e.g. ``aes(fill="reason")`` to colour several
        regions differently. Position aesthetics are supplied internally.
    xstart, xend
        A single region, as an alternative to passing *data*.
    fill, alpha, colour
        Band appearance. The default is a translucent amber with no border,
        which tints the tracks underneath rather than hiding them.

    Returns
    -------
    A ggplot layer.

    Raises
    ------
    ValueError
        Neither (or both) of *data* and the *xstart*/*xend* pair supplied,
        or *data* is missing the coordinate columns.

    Examples
    --------
    >>> p + geom_highlight(xstart=31_659_500, xend=31_660_000)   # doctest: +SKIP
    """
    if (data is None) == (xstart is None and xend is None):
        raise ValueError(
            "geom_highlight: give either data with xstart/xend columns, or "
            "the xstart/xend arguments — not both, not neither."
        )

    if data is None:
        if xstart is None or xend is None:
            raise ValueError("geom_highlight: give both xstart and xend.")
        data = pd.DataFrame({"xstart": [xstart], "xend": [xend]})

    missing = [c for c in ("xstart", "xend") if c not in data.columns]
    if missing:
        raise ValueError(
            f"geom_highlight: data is missing column(s) {missing!r}; "
            f"have {list(data.columns)!r}."
        )

    frame = data.copy()
    frame["xmin"] = frame["xstart"]
    frame["xmax"] = frame["xend"]
    frame["ymin"] = -np.inf
    frame["ymax"] = np.inf

    # Mapping is dict-like; the position aesthetics are supplied here and win
    # over anything the caller passed, since they address generated columns.
    merged = dict(mapping) if mapping is not None else {}
    merged.update(xmin="xmin", xmax="xmax", ymin="ymin", ymax="ymax")
    position = gg.aes(**merged)

    # A fixed value would override the mapped channel, so only defaults for
    # channels the caller did *not* map are passed through.
    defaults = {
        "fill": fill,
        "alpha": alpha,
        "colour": "none" if colour is None else colour,
    }
    params = {k: v for k, v in defaults.items() if k not in merged}
    params.update(kwargs)

    return gg.geom_rect(
        position, data=frame, inherit_aes=inherit_aes, **params
    )
