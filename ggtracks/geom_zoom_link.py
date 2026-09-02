"""``geom_zoom_link`` — the tapered connector of a focus + context figure.

An overview and a magnified detail view only read as one figure if
something states the relationship between them. The convention is a
trapezoid: narrow where the region sits in the overview, widening to the
full width of the detail panel, shaded so the eye follows the taper.

The band is drawn from the panel's own coordinates. Its top edge comes from
``xstart``/``xend`` put through the panel's x scale — so if the connector
shares a scale with the overview above it (as tracks in one
:func:`~ggtracks.plot_tracks` figure do), the taper lands on the region
automatically, with no measuring by hand. Its bottom edge spans the panel.

The shading is a real ``linear_gradient`` fill, not a stack of slightly
different polygons faked to look like one, so it stays smooth at any output
size and any dpi. Nothing here is expressed in figure coordinates, which is
what would otherwise drift the moment the figure is resized.
"""

from __future__ import annotations

from .palettes import FEATURE_COLOURS

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ggplot2_py.aes import Mapping
from ggplot2_py.draw_key import draw_key_blank
from ggplot2_py.geom import Geom, _coord_transform, _ggname
from grid_py import Gpar, linear_gradient, null_grob, polygon_grob

__all__ = ["GeomZoomLink", "geom_zoom_link"]


class GeomZoomLink(Geom):
    """Tapered connector between an overview and a detail view.

    Parameters (as layer params)
    ----------------------------
    colours : (str, str)
        Gradient endpoints, **narrow edge first**: pale where the band
        leaves the overview, deepening as it opens out toward the detail
        view.
    flip : bool
        ``False`` (default) tapers upward — the detail view is **below**
        the overview. ``True`` tapers downward, for the reverse layout.
    """

    required_aes: Tuple[str, ...] = ("xstart", "xend")
    non_missing_aes: Tuple[str, ...] = ()
    default_aes: Mapping = Mapping(
        colour="none",
        linewidth=0.2,
        linetype=1,
        alpha=0.6,
    )
    # A connector is not a data series; a legend entry for it is noise.
    draw_key = draw_key_blank

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        # The band fills its panel vertically, but the panel still needs a y
        # scale to exist at all.
        if "y" not in data.columns:
            data["y"] = 0.0
        return data

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        colours: Sequence[str] = ("#E4E8EC", FEATURE_COLOURS["intron"]),
        flip: bool = False,
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        if data is None or data.empty:
            return null_grob()
        if len(colours) != 2:
            raise ValueError(
                f"GeomZoomLink: colours must be two endpoints (got {colours!r})."
            )

        # coord transform puts positions in the panel's 0–1 space, so the
        # taper is expressed relative to the panel and survives resizing.
        frame = data.copy()
        frame["xmin"] = frame["xstart"]
        frame["xmax"] = frame["xend"]
        coords = _coord_transform(coord, frame, panel_params)

        narrow_left = float(np.min(coords["xmin"].to_numpy(dtype=float)))
        narrow_right = float(np.max(coords["xmax"].to_numpy(dtype=float)))
        alpha = data["alpha"].iloc[0] if "alpha" in data.columns else 0.6
        alpha = 1.0 if alpha is None else float(alpha)

        narrow_y, wide_y = (0.0, 1.0) if flip else (1.0, 0.0)
        xs = np.array([narrow_left, narrow_right, 1.0, 0.0])
        ys = np.array([narrow_y, narrow_y, wide_y, wide_y])

        gradient = linear_gradient(
            colours=[colours[0], colours[1]],
            x1=0.0, y1=narrow_y, x2=0.0, y2=wide_y,
            default_units="npc",
        )
        outline = data["colour"].iloc[0] if "colour" in data.columns else "none"
        return _ggname(
            "geom_zoom_link",
            polygon_grob(
                x=xs,
                y=ys,
                default_units="npc",
                gp=Gpar(
                    fill=gradient,
                    col=None if outline in (None, "none") else outline,
                    alpha=alpha,
                ),
            ),
        )


def geom_zoom_link(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    *,
    xstart: Optional[float] = None,
    xend: Optional[float] = None,
    track: Optional[str] = None,
    colours: Sequence[str] = ("#E4E8EC", FEATURE_COLOURS["intron"]),
    flip: bool = False,
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = False,
    **kwargs: Any,
) -> Any:
    """Connector layer for a focus + context pair.

    Parameters
    ----------
    xstart, xend
        The magnified region, in genomic coordinates. Supplying these builds
        the one-row frame for you, as an alternative to *data* / *mapping*.
    track
        Name of the :class:`~ggtracks.Track` this connector belongs to. Set
        it whenever the layer goes into a :func:`~ggtracks.plot_tracks`
        figure: a frame without the facet column is repeated into *every*
        panel, which would drape the band across the data tracks too.
    colours
        Gradient endpoints, narrow edge first.
    flip
        Taper downward instead of upward, when the detail view sits above.

    Examples
    --------
    Put it in a thin track of the overview figure, then place the detail
    figure beneath::

        link = Track("zoom", [geom_zoom_link(xstart=31_659_500,
                                             xend=31_660_000, track="zoom")],
                     height=0.3, y_breaks=[0.0], y_labels=[""])
    """
    from ggplot2_py.layer import layer

    if (data is None) == (xstart is None and xend is None):
        raise ValueError(
            "geom_zoom_link: give either data with xstart/xend, or the "
            "xstart/xend arguments — not both, not neither."
        )
    if data is None:
        if xstart is None or xend is None:
            raise ValueError("geom_zoom_link: give both xstart and xend.")
        data = pd.DataFrame({"xstart": [xstart], "xend": [xend]})
        if track is not None:
            data["track"] = track
        if mapping is None:
            import ggplot2_py as gg

            mapping = gg.aes(xstart="xstart", xend="xend")
    elif track is not None:
        raise ValueError(
            "geom_zoom_link: track= builds the frame for you; put the column "
            "in `data` instead when supplying your own."
        )

    return layer(
        geom=GeomZoomLink,
        stat=stat,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "colours": tuple(colours),
            "flip": flip,
            "na_rm": na_rm,
            **kwargs,
        },
    )
