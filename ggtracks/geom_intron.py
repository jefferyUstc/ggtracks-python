"""Introns — ``to_intron`` / ``StatIntron`` + ``GeomIntron``.

Faithful port of ggtranscript's intron tooling (``to_intron.R``,
``geom_intron.R``):

* :func:`to_intron` derives intron ranges from a transcript's exons
  (``intron = (prev_exon_end, this_exon_start)`` per group, dropping the
  leading NA and directly-adjacent exons).
* :class:`StatIntron` is the same computation as a real ``Stat`` so a
  single ``geom_intron(stat=StatIntron)`` can run over exon data.
* :class:`GeomIntron` is ``GeomSegment`` with ``xstart``/``xend``
  nomenclature plus a central strand arrow (``"+"``/``"-"``) indicating
  the direction of transcription. A ``style="chevron"`` option draws the
  IGV-style peaked connector instead of a straight line + arrow.
"""

from __future__ import annotations

from .palettes import FEATURE_COLOURS

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ggplot2_py.aes import Mapping
from ggplot2_py.geom import GeomSegment, _ggname
from ggplot2_py.stat import Stat
from grid_py import grob_tree, null_grob, arrow as _arrow, Unit

__all__ = ["to_intron", "StatIntron", "GeomIntron", "geom_intron"]


def _default_intron_arrow():
    return _arrow(ends="last", length=Unit(0.055, "inches"), angle=25)


def _introns_from_exons(d: pd.DataFrame) -> pd.DataFrame:
    """``exon → intron`` for one group: intron = (lag(xend), xstart).

    Sorts by ``xstart``/``xend``, takes ``intron_start = previous exon
    xend`` and ``intron_end = this exon xstart``, drops the leading NA
    and any directly-adjacent pair (|gap| == 1). Carries the per-row
    aesthetics of the *downstream* exon (matching ggtranscript).
    """
    if len(d) < 2:
        return d.iloc[0:0].copy()
    d = d.sort_values(["xstart", "xend"], kind="stable").reset_index(drop=True)
    out = d.copy()
    out["xstart"] = d["xend"].shift(1)
    out["xend"] = d["xstart"]
    out = out.iloc[1:]
    keep = (out["xend"] - out["xstart"]).abs() != 1
    return out[keep].reset_index(drop=True)


def to_intron(
    exons: pd.DataFrame,
    group_var: Optional[str] = None,
    *,
    start: str = "xstart",
    end: str = "xend",
) -> pd.DataFrame:
    """Derive intron ranges from exon ranges (ggtranscript ``to_intron``).

    Parameters
    ----------
    exons
        Exon table with start/end columns (named ``xstart``/``xend`` by
        default; override via *start*/*end* to accept ``start``/``end``).
    group_var
        Column grouping exons into transcripts. ``None`` treats all
        exons as one group.
    """
    df = exons.rename(columns={start: "xstart", end: "xend"}).copy()
    if group_var is None:
        return _introns_from_exons(df)
    parts = [
        _introns_from_exons(g) for _, g in df.groupby(group_var, sort=False)
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


class StatIntron(Stat):
    """Compute introns from exons per group (the ``to_intron`` Stat)."""

    required_aes: List[str] = ["xstart", "xend"]
    dropped_aes: List[str] = []

    def compute_group(self, data: pd.DataFrame, scales: Any, **params: Any) -> pd.DataFrame:
        return _introns_from_exons(data)


def _check_strand(strand: Any) -> None:
    vals = pd.unique(pd.Series(strand).dropna())
    bad = [v for v in vals if v not in ("+", "-")]
    if bad or pd.isna(pd.Series(strand)).any():
        raise ValueError(
            f"GeomIntron: strand values must each be '+' or '-' (got {list(vals)!r})."
        )


class GeomIntron(GeomSegment):
    """Intron lines with central strand arrows; aes ``xstart``, ``xend``, ``y``."""

    required_aes: Tuple[str, ...] = ("xstart", "xend", "y")
    non_missing_aes: Tuple[str, ...] = ()
    default_aes: Mapping = Mapping(
        colour=FEATURE_COLOURS["intron"],
        linewidth=0.5,
        linetype=1,
        alpha=None,
        strand="+",
    )

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        data["x"] = data["xstart"]
        data["yend"] = data["y"]
        return data.drop(columns=["xstart"])

    @staticmethod
    def _strand_arrow_grob(
        seg: GeomSegment,
        target: str,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        arrow_obj: Any,
        arrow_min: float,
        lineend: str,
        linejoin: str,
        na_rm: bool,
        arrow_density: int = 1,
    ) -> Any:
        match = (data["strand"] == target) if "strand" in data.columns else (target == "+")
        ab_min = (data["x"] - data["xend"]).abs() > arrow_min
        ad = data[match & ab_min].copy()
        if ad.empty:
            return null_grob()

        # Each arrow is the head of a segment ending where the arrow sits.
        # With density 1 that segment runs from the intron's start to its
        # midpoint — the long-standing single centred arrow.
        left = ad["x"].to_numpy(dtype=float)
        right = ad["xend"].to_numpy(dtype=float)
        span = right - left
        pieces = []
        for i in range(arrow_density):
            part = ad.copy()
            begin = i / arrow_density
            point = (i + 0.5) / arrow_density
            if target == "+":
                part["x"] = left + span * begin
                part["xend"] = left + span * point
            else:
                part["x"] = right - span * begin
                part["xend"] = right - span * point
            pieces.append(part)
        ad = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0]

        return seg.draw_panel(
            ad, panel_params, coord, arrow=arrow_obj,
            lineend=lineend, linejoin=linejoin, na_rm=na_rm,
        )

    def _chevron_grob(
        self,
        seg: GeomSegment,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        chevron_height: float,
        lineend: str,
        linejoin: str,
        na_rm: bool,
    ) -> Any:
        mid = (data["x"] + data["xend"]) / 2.0
        peak = data["y"] + chevron_height
        other = [c for c in data.columns if c not in ("x", "y", "xend", "yend")]
        left = data[other].copy()
        left["x"], left["y"], left["xend"], left["yend"] = data["x"], data["y"], mid, peak
        right = data[other].copy()
        right["x"], right["y"], right["xend"], right["yend"] = mid, peak, data["xend"], data["y"]
        chev = pd.concat([left, right], ignore_index=True)
        return seg.draw_panel(
            chev, panel_params, coord, arrow=None,
            lineend=lineend, linejoin=linejoin, na_rm=na_rm,
        )

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        arrow: Any = "default",
        arrow_fill: Any = None,
        lineend: str = "butt",
        linejoin: str = "round",
        na_rm: bool = False,
        arrow_min_intron_length: float = 0,
        arrow_density: int = 1,
        style: str = "line",
        chevron_height: float = 0.25,
        **params: Any,
    ) -> Any:
        if data is None or data.empty:
            return null_grob()
        if arrow_min_intron_length < 0:
            raise ValueError(
                "GeomIntron: arrow_min_intron_length must be >= 0 "
                f"(got {arrow_min_intron_length!r})."
            )
        if not isinstance(arrow_density, int) or isinstance(arrow_density, bool) \
                or arrow_density < 1:
            raise ValueError(
                f"GeomIntron: arrow_density must be a positive int "
                f"(got {arrow_density!r})."
            )
        seg = GeomSegment()

        if style == "chevron":
            return _ggname(
                "geom_intron",
                self._chevron_grob(
                    seg, data, panel_params, coord, chevron_height,
                    lineend, linejoin, na_rm,
                ),
            )
        if style != "line":
            raise ValueError(
                f"GeomIntron: style must be 'line' or 'chevron' (got {style!r})."
            )

        arrow_obj = _default_intron_arrow() if arrow == "default" else arrow

        intron_grob = seg.draw_panel(
            data, panel_params, coord, arrow=None,
            lineend=lineend, linejoin=linejoin, na_rm=na_rm,
        )
        if arrow_obj is None:
            return _ggname("geom_intron", intron_grob)

        if "strand" in data.columns:
            _check_strand(data["strand"])
        plus_grob = self._strand_arrow_grob(
            seg, "+", data, panel_params, coord, arrow_obj,
            arrow_min_intron_length, lineend, linejoin, na_rm, arrow_density,
        )
        minus_grob = self._strand_arrow_grob(
            seg, "-", data, panel_params, coord, arrow_obj,
            arrow_min_intron_length, lineend, linejoin, na_rm, arrow_density,
        )
        return _ggname(
            "geom_intron", grob_tree(intron_grob, plus_grob, minus_grob)
        )


def geom_intron(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: Any = "identity",
    position: str = "identity",
    *,
    arrow: Any = "default",
    arrow_fill: Any = None,
    lineend: str = "butt",
    linejoin: str = "round",
    arrow_min_intron_length: float = 0,
    arrow_density: int = 1,
    style: str = "line",
    chevron_height: float = 0.25,
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Intron layer with strand arrows; aes ``xstart``, ``xend``, ``y``.

    Pass intron data directly (``stat="identity"``, e.g. from
    :func:`to_intron`) or derive introns from exon data in-pipeline with
    ``stat=StatIntron``. See :class:`GeomIntron`.
    """
    from ggplot2_py.layer import layer

    return layer(
        geom=GeomIntron,
        stat=stat,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "arrow": arrow,
            "arrow_fill": arrow_fill,
            "lineend": lineend,
            "linejoin": linejoin,
            "arrow_min_intron_length": arrow_min_intron_length,
            "arrow_density": arrow_density,
            "style": style,
            "chevron_height": chevron_height,
            "na_rm": na_rm,
            **kwargs,
        },
    )
