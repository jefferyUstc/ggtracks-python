"""``GeomCoverage`` / ``StatBinCoverage`` — signal tracks over intervals.

Coverage, bigWig signal and bedGraph are **interval-valued**: each record
says "over ``[xstart, xend)`` the value is *v*". The faithful rendering is
therefore a **step** — flat across each interval, vertical at its edges —
not a line drawn between interval midpoints. Interpolating between midpoints
invents a slope the data never claimed and, at coarse resolution, visibly
rounds off peaks and fills gaps that should be empty.

Using the ``xstart``/``xend`` nomenclature (rather than a plain ``x``) also
means the layer is compressed by :func:`~ggtracks.scale_x_genomic` exactly
like :func:`~ggtracks.geom_range` and :func:`~ggtracks.geom_intron`, so a
coverage track stays aligned with the gene model beneath it.

:class:`StatBinCoverage` resamples to a fixed number of bins. It bins the
data it is handed, which by then is already in the scale's **display**
space — so bins are of equal width *on screen*. Binning in genomic space
instead would over-sample whatever the mapper compressed, spending most of
the bins on introns that occupy a sliver of the axis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ggplot2_py.aes import Mapping
from ggplot2_py.draw_key import draw_key_polygon
from ggplot2_py.geom import Geom, GeomPath, GeomPolygon, GeomRect, _ggname
from ggplot2_py.stat import Stat
from grid_py import null_grob

from ._binning import Summary, bin_intervals

__all__ = ["GeomCoverage", "geom_coverage", "StatBinCoverage", "stat_bin_coverage"]

_STYLES = ("area", "step", "bar")


def _step_outline(
    xstart: np.ndarray, xend: np.ndarray, y: np.ndarray, ymin: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Trace one group's records as a closed step profile.

    Walks the intervals in order, rising to each value and dropping back to
    the baseline wherever the records leave a gap, so unmeasured stretches
    read as empty instead of being bridged by a slope.
    """
    order = np.argsort(xstart, kind="stable")
    xstart, xend, y, ymin = xstart[order], xend[order], y[order], ymin[order]

    xs: List[float] = []
    ys: List[float] = []
    prev_end: Optional[float] = None
    for s, e, height, base in zip(xstart, xend, y, ymin):
        if prev_end is None:
            xs.append(s)
            ys.append(base)
        elif s > prev_end:
            xs.append(prev_end)
            ys.append(base)
            xs.append(s)
            ys.append(base)
        xs.append(s)
        ys.append(height)
        xs.append(e)
        ys.append(height)
        prev_end = e
    if prev_end is not None:
        xs.append(prev_end)
        ys.append(ymin[-1])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


class GeomCoverage(Geom):
    """Step-rendered signal track; aes ``xstart``, ``xend``, ``y``.

    Parameters (as layer params)
    ----------------------------
    style : {"area", "step", "bar"}
        ``"area"`` (default) fills under the step profile; ``"step"`` draws
        the profile as a line only; ``"bar"`` draws each interval as its own
        rectangle, which reads better when records are sparse.

    Notes
    -----
    The default aesthetics give a translucent fill under an opaque hairline
    of the same colour: the peak edges stay crisp while the mass of the
    track stays light enough to stack several without them fighting.
    ``alpha`` applies to the fill only, which is the standard behaviour for
    polygon-like geoms.
    """

    required_aes: Tuple[str, ...] = ("xstart", "xend", "y")
    optional_aes: Tuple[str, ...] = ("ymin",)
    non_missing_aes: Tuple[str, ...] = ()
    default_aes: Mapping = Mapping(
        fill="grey35",
        colour="grey35",
        linewidth=0.3,
        linetype=1,
        alpha=0.8,
    )
    # The legend key must show what the geom draws: a filled patch, not the
    # base class's point — which would also ignore ``fill`` entirely.
    draw_key = draw_key_polygon

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        if "ymin" not in data.columns:
            data["ymin"] = 0.0
        data["ymin"] = data["ymin"].fillna(0.0)
        return data

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        style: str = "area",
        lineend: str = "butt",
        linejoin: str = "round",
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        if data is None or data.empty:
            return null_grob()
        if style not in _STYLES:
            raise ValueError(
                f"GeomCoverage: style must be one of {_STYLES} (got {style!r})."
            )

        if style == "bar":
            rects = data.copy()
            rects["xmin"] = rects["xstart"]
            rects["xmax"] = rects["xend"]
            rects["ymax"] = rects["y"]
            return _ggname(
                "geom_coverage",
                GeomRect.draw_panel(GeomRect(), rects, panel_params, coord),
            )

        carried = [
            c
            for c in data.columns
            if c not in ("xstart", "xend", "x", "y", "ymin", "ymax")
        ]
        frames: List[pd.DataFrame] = []
        groups = data.groupby("group", sort=False) if "group" in data.columns else [(0, data)]
        for _key, part in groups:
            xs, ys = _step_outline(
                part["xstart"].to_numpy(dtype=float),
                part["xend"].to_numpy(dtype=float),
                part["y"].to_numpy(dtype=float),
                part["ymin"].to_numpy(dtype=float),
            )
            if xs.size == 0:
                continue
            seg = pd.DataFrame({"x": xs, "y": ys})
            first = part.iloc[0]
            for col in carried:
                seg[col] = first[col]
            frames.append(seg)

        if not frames:
            return null_grob()
        outline = pd.concat(frames, ignore_index=True)

        if style == "area":
            return _ggname(
                "geom_coverage",
                GeomPolygon.draw_panel(GeomPolygon(), outline, panel_params, coord),
            )
        return _ggname(
            "geom_coverage",
            GeomPath.draw_panel(GeomPath(), outline, panel_params, coord),
        )


def geom_coverage(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: Any = "identity",
    position: str = "identity",
    *,
    style: str = "area",
    lineend: str = "butt",
    linejoin: str = "round",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Signal track layer; aes ``xstart``, ``xend``, ``y`` (optional ``ymin``).

    Pass interval-valued signal directly (``stat="identity"``, e.g. straight
    from :meth:`ggtracks.BigWig.query`), or resample it in-pipeline with
    ``stat=StatBinCoverage``. See :class:`GeomCoverage`.
    """
    from ggplot2_py.layer import layer

    return layer(
        geom=GeomCoverage,
        stat=stat,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "style": style,
            "lineend": lineend,
            "linejoin": linejoin,
            "na_rm": na_rm,
            **kwargs,
        },
    )


class StatBinCoverage(Stat):
    """Resample interval-valued signal onto a fixed number of bins.

    Bins are uniform in the data this stat receives — which the build
    pipeline has already put through the position scale — so they are
    uniform *on screen* even when the x axis compresses introns.
    """

    required_aes: List[str] = ["xstart", "xend", "y"]
    dropped_aes: List[str] = []

    def setup_params(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        # Validation belongs here: exceptions raised in ``compute_group`` are
        # caught by the build and downgraded to a warning plus an empty
        # frame, which surfaces to the user as a blank panel rather than an
        # error.
        bins = params.get("bins", 200)
        if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
            raise ValueError(
                f"StatBinCoverage: bins must be a positive int (got {bins!r})."
            )
        summary = params.get("summary", "mean")
        if summary not in ("mean", "max", "min", "sum"):
            raise ValueError(
                f"StatBinCoverage: summary must be 'mean', 'max', 'min' or "
                f"'sum' (got {summary!r})."
            )
        return params

    def compute_group(
        self,
        data: pd.DataFrame,
        scales: Any,
        bins: int = 200,
        summary: Summary = "mean",
        **params: Any,
    ) -> pd.DataFrame:
        if data.empty:
            return data
        lo = float(np.min(data["xstart"]))
        hi = float(np.max(data["xend"]))
        if hi <= lo:
            return data.iloc[0:0]
        starts, ends, values = bin_intervals(
            data["xstart"].to_numpy(dtype=float),
            data["xend"].to_numpy(dtype=float),
            data["y"].to_numpy(dtype=float),
            lo,
            hi,
            bins,
            summary,
        )
        return pd.DataFrame({"xstart": starts, "xend": ends, "y": values})


def stat_bin_coverage(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    geom: Any = GeomCoverage,
    position: str = "identity",
    *,
    bins: int = 200,
    summary: Summary = "mean",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Binned signal layer — :class:`StatBinCoverage` under :class:`GeomCoverage`."""
    from ggplot2_py.layer import layer

    return layer(
        geom=geom,
        stat=StatBinCoverage,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={"bins": bins, "summary": summary, "na_rm": na_rm, **kwargs},
    )
