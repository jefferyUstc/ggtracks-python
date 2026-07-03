"""``GeomRange`` — range-based genomic features (exon / CDS / UTR / read blocks).

Faithful port of ggtranscript ``GeomRange`` (``geom_range.R``): a
``GeomTile`` whose aesthetics use genetic nomenclature ``xstart``/``xend``
(plus ``y`` for the row). The half-range behaviour of ggtranscript's
separate ``GeomHalfRange`` is folded in here as an ``orientation`` param
(``"top"``/``"bottom"``) rather than a second geom — one geom covers
both, with the orientation simply setting the rectangle's ``vjust``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ggplot2_py.aes import Mapping
from ggplot2_py.geom import (
    GeomTile,
    _coord_transform,
    _fill_alpha,
    _ggname,
    scales_alpha,
    PT,
)
from grid_py import rect_grob, Gpar, null_grob

__all__ = ["GeomRange", "geom_range"]


class GeomRange(GeomTile):
    """Tiles for range features; aes ``xstart``, ``xend``, ``y``.

    Parameters (as layer params)
    ----------------------------
    height : float
        Box height in data units (default ``0.5``; ``0.25`` when an
        ``orientation`` is given, matching ggtranscript ``geom_half_range``).
    orientation : {"top", "bottom"} or None
        When set, draw only half the range above (``"top"``) or below
        (``"bottom"``) the baseline ``y`` — useful for comparing two
        transcripts or freeing space for junction arcs.
    vjust : float or None
        Explicit vertical justification override (advanced).
    """

    required_aes: Tuple[str, ...] = ("xstart", "xend", "y")
    non_missing_aes: Tuple[str, ...] = ()
    default_aes: Mapping = Mapping(
        fill="grey",
        colour="black",
        linewidth=0.25,
        linetype=1,
        alpha=None,
        height=None,
    )

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        orientation = params.get("orientation")
        if orientation is not None and orientation not in ("top", "bottom"):
            raise ValueError(
                f"GeomRange: orientation must be 'top' or 'bottom', "
                f"got {orientation!r}."
            )
        if "height" in data.columns and data["height"].notna().any():
            h = np.asarray(data["height"], dtype=float)
        else:
            default_h = 0.25 if orientation else 0.5
            h = float(params.get("height") or default_h)
        data["xmin"] = data["xstart"]
        data["xmax"] = data["xend"]
        data["ymin"] = data["y"] - np.asarray(h) / 2.0
        data["ymax"] = data["y"] + np.asarray(h) / 2.0
        if "height" in data.columns:
            data = data.drop(columns=["height"])
        return data

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        vjust: Optional[float] = None,
        orientation: Optional[str] = None,
        height: Any = None,
        lineend: str = "butt",
        linejoin: str = "mitre",
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        if data is None or data.empty:
            return null_grob()
        if orientation is not None:
            if orientation not in ("top", "bottom"):
                raise ValueError(
                    f"GeomRange: orientation must be 'top' or 'bottom', "
                    f"got {orientation!r}."
                )
            if vjust is None:
                vjust = 1.5 if orientation == "bottom" else 0.5

        coords = _coord_transform(coord, data, panel_params)
        return _ggname(
            "geom_range",
            rect_grob(
                x=coords["xmin"].values,
                y=coords["ymax"].values,
                width=coords["xmax"].values - coords["xmin"].values,
                height=coords["ymax"].values - coords["ymin"].values,
                default_units="native",
                just=("left", "top"),
                vjust=vjust,
                gp=Gpar(
                    col=coords["colour"].values if "colour" in coords.columns else None,
                    fill=_fill_alpha(
                        coords["fill"].values if "fill" in coords.columns else "grey",
                        coords["alpha"].values if "alpha" in coords.columns else None,
                    ),
                    lwd=(
                        coords["linewidth"].values * PT
                        if "linewidth" in coords.columns
                        else 0.25 * PT
                    ),
                    lty=coords["linetype"].values if "linetype" in coords.columns else 1,
                    linejoin=linejoin,
                    lineend=lineend,
                ),
            ),
        )


def geom_range(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    *,
    height: Any = None,
    orientation: Optional[str] = None,
    vjust: Optional[float] = None,
    linejoin: str = "mitre",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Layer of range features (exon / CDS / UTR / read blocks).

    Maps ``xstart``, ``xend``, ``y`` (and optionally ``fill``). See
    :class:`GeomRange`.
    """
    from ggplot2_py.layer import layer

    return layer(
        geom=GeomRange,
        stat=stat,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "height": height,
            "orientation": orientation,
            "vjust": vjust,
            "linejoin": linejoin,
            "na_rm": na_rm,
            **kwargs,
        },
    )
