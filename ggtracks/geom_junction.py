"""``GeomJunction`` — sashimi / splice-junction arcs (and loop arcs).

Modeled on ggtranscript ``GeomJunction`` (``geom_junction.R``): a curve
per junction whose shape comes from grid's X-spline control points
(``grid:::calcControlPoints``, ported as
:func:`grid_py._curve._calc_control_points`) with ``curvature=-0.5``,
``angle=90``, ``ncp=15``. The endpoints are re-appended so the arc meets
the intron line.

Extends ggtranscript with the two conventions long-read sashimi plots
need, selectable by parameter (principle 1 — no hardcoding):

* arc **height** is either ``"fixed"`` (all arcs reach ``arc_height_max``;
  thickness then encodes count via the standard ``linewidth`` aesthetic,
  the ggtranscript convention) or ``"count"`` (apex ∝ a per-junction
  ``count`` aesthetic, the BioSeqUtils convention);
* ``style="line"`` draws the arc as a polyline; ``style="ribbon"`` draws
  a filled band whose thickness grows with count.

``orientation`` places arcs ``"top"`` / ``"bottom"`` / ``"alternating"``
relative to the baseline ``y``.
"""

from __future__ import annotations

from .palettes import FEATURE_COLOURS

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ggplot2_py.aes import Mapping
from ggplot2_py.geom import GeomPath, GeomPolygon, _coord_transform, _ggname
from ggplot2_py.geom import GeomLine
from grid_py import null_grob, grob_tree
from grid_py._curve import _calc_control_points

__all__ = ["GeomJunction", "geom_junction"]


def _arc_points(
    x: float, xend: float, y: float,
    *, angle: float, curvature: float, ncp: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Control-point arc between (x,y) and (xend,y), endpoints re-appended."""
    cpx, cpy = _calc_control_points(
        np.array([x], float), np.array([y], float),
        np.array([xend], float), np.array([y], float),
        curvature=curvature, angle=angle, ncp=ncp,
    )
    xs = np.concatenate([[x], cpx.ravel(), [xend]])
    ys = np.concatenate([[y], cpy.ravel(), [y]])
    return xs, ys


class GeomJunction(GeomLine):
    """Junction arcs; aes ``xstart``, ``xend``, ``y`` (optional ``count``)."""

    required_aes: Tuple[str, ...] = ("xstart", "xend", "y")
    optional_aes: Tuple[str, ...] = ("count",)
    non_missing_aes: Tuple[str, ...] = ()
    default_aes: Mapping = Mapping(
        colour=FEATURE_COLOURS["junction"],
        linewidth=0.5,
        linetype=1,
        alpha=None,
        fill=FEATURE_COLOURS["junction"],
    )

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy().reset_index(drop=True)
        if "group" not in data.columns:
            data["group"] = np.arange(len(data))
        elif data["group"].duplicated().any():
            data["group"] = [f"{g}-{i}" for i, g in enumerate(data["group"])]
        data["x"] = data["xstart"]
        return data.drop(columns=["xstart"])

    def _curve_frame(
        self, data: pd.DataFrame, *, angle: float, curvature: float, ncp: int,
        orientation: str, arc_height_max: float, height_by: str,
        offset_scale: float = 1.0,
    ) -> pd.DataFrame:
        if height_by not in ("fixed", "count"):
            raise ValueError(
                f"GeomJunction: height_by must be 'fixed' or 'count' (got {height_by!r})."
            )
        if orientation not in ("top", "bottom", "alternating"):
            raise ValueError(
                f"GeomJunction: orientation must be 'top'/'bottom'/'alternating' "
                f"(got {orientation!r})."
            )
        if height_by == "count":
            if "count" not in data.columns:
                raise ValueError(
                    "GeomJunction: height_by='count' needs a `count` aesthetic."
                )
            wmax = float(max(np.nanmax(data["count"].to_numpy()), 1e-12))

        data = data.copy()
        data["_jidx"] = data.groupby("y").cumcount()
        carry = [c for c in data.columns if c not in ("x", "xend", "y", "_jidx")]

        frames: List[pd.DataFrame] = []
        for _, r in data.iterrows():
            xs, ys = _arc_points(
                float(r["x"]), float(r["xend"]), float(r["y"]),
                angle=angle, curvature=curvature, ncp=ncp,
            )
            offset = ys - float(r["y"])
            apex_raw = float(np.max(np.abs(offset))) or 1.0
            desired = arc_height_max * (
                float(r["count"]) / wmax if height_by == "count" else 1.0
            )
            if orientation == "top":
                sign = 1.0
            elif orientation == "bottom":
                sign = -1.0
            else:
                sign = 1.0 if (int(r["_jidx"]) % 2 == 1) else -1.0
            y_new = float(r["y"]) + sign * np.abs(offset) / apex_raw * desired * offset_scale
            seg = pd.DataFrame({"x": xs, "y": y_new})
            for c in carry:
                seg[c] = r[c]
            frames.append(seg)
        return pd.concat(frames, ignore_index=True) if frames else data.iloc[0:0]

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        orientation: str = "alternating",
        arc_height_max: float = 1.0,
        height_by: str = "fixed",
        angle: float = 90.0,
        curvature: float = -0.5,
        ncp: int = 15,
        style: str = "line",
        band_width: float = 0.5,
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        if data is None or data.empty:
            return null_grob()
        if style not in ("line", "ribbon"):
            raise ValueError(
                f"GeomJunction: style must be 'line' or 'ribbon' (got {style!r})."
            )

        if style == "line":
            curves = self._curve_frame(
                data, angle=angle, curvature=curvature, ncp=ncp,
                orientation=orientation, arc_height_max=arc_height_max,
                height_by=height_by,
            )
            return _ggname(
                "geom_junction",
                GeomPath.draw_panel(GeomPath(), curves, panel_params, coord),
            )

        if height_by == "count" and "count" in data.columns:
            wmax = float(max(np.nanmax(data["count"].to_numpy()), 1e-12))
            inner_scale = 1.0 - band_width * (data["count"].to_numpy() / wmax)
        else:
            inner_scale = np.full(len(data), 1.0 - band_width)
        data = data.copy()
        data["_inner_scale"] = inner_scale

        outer = self._curve_frame(
            data, angle=angle, curvature=curvature, ncp=ncp,
            orientation=orientation, arc_height_max=arc_height_max,
            height_by=height_by, offset_scale=1.0,
        )
        children = []
        for gid, g_out in outer.groupby("group", sort=False):
            isc = float(g_out["_inner_scale"].iloc[0])
            g_in = self._curve_frame(
                data[data["group"] == gid], angle=angle, curvature=curvature, ncp=ncp,
                orientation=orientation, arc_height_max=arc_height_max,
                height_by=height_by, offset_scale=isc,
            )
            poly = pd.concat([g_out, g_in.iloc[::-1]], ignore_index=True)
            poly["group"] = gid
            if "fill" not in poly.columns:
                poly["fill"] = FEATURE_COLOURS["junction"]
            children.append(
                GeomPolygon.draw_panel(GeomPolygon(), poly, panel_params, coord)
            )
        if not children:
            return null_grob()
        return _ggname("geom_junction", grob_tree(*children))


def geom_junction(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: Any = "identity",
    position: str = "identity",
    *,
    orientation: str = "alternating",
    arc_height_max: float = 1.0,
    height_by: str = "fixed",
    angle: float = 90.0,
    curvature: float = -0.5,
    ncp: int = 15,
    style: str = "line",
    band_width: float = 0.5,
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Junction-arc layer; aes ``xstart``, ``xend``, ``y`` (optional ``count``).

    See :class:`GeomJunction` for height/style/orientation options.
    """
    from ggplot2_py.layer import layer

    return layer(
        geom=GeomJunction,
        stat=stat,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "orientation": orientation,
            "arc_height_max": arc_height_max,
            "height_by": height_by,
            "angle": angle,
            "curvature": curvature,
            "ncp": ncp,
            "style": style,
            "band_width": band_width,
            "na_rm": na_rm,
            **kwargs,
        },
    )
