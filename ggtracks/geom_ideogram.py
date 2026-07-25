"""``geom_ideogram`` — the chromosome, banded, with its centromere.

A browser view of a gene shows a few kilobases; the ideogram answers the
question that view cannot ("where on the chromosome *is* this?") by drawing
the whole chromosome with its Giemsa bands, so a reader can place the region
at a glance.

Two things make it read as a chromosome rather than a bar chart:

* bands are filled by **Giemsa stain**, on the long-established greyscale
  where darker means more condensed chromatin (:func:`scale_fill_giemsa`);
* the **centromere** (``acen``) is drawn as a pair of triangles meeting
  point-to-point, giving the familiar waist.

The x here is a whole chromosome — a different coordinate domain from the
compressed gene region of the other tracks — so an ideogram row belongs to a
figure built with ``plot_tracks(..., mappers=...)``, where each column
carries its own scale.

To mark the region under view, add a
:func:`~ggtracks.geom_highlight` over the same panel: in an otherwise
greyscale figure a single saturated band is what the eye finds first.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import ggplot2_py as gg
from ggplot2_py.aes import Mapping
from ggplot2_py.draw_key import draw_key_polygon
from ggplot2_py.geom import Geom, GeomPolygon, GeomRect, _ggname
from grid_py import grob_tree, null_grob

__all__ = ["GIEMSA_COLOURS", "scale_fill_giemsa", "GeomIdeogram", "geom_ideogram"]

#: Giemsa stain → fill, following the UCSC/IGV convention: a greyscale ramp
#: for ``gpos*`` density, dark red for the centromere and blue for stalks.
GIEMSA_COLOURS: Dict[str, str] = {
    "gneg": "#FFFFFF",
    "gpos25": "#D9D9D9",
    "gpos33": "#BFBFBF",
    "gpos50": "#808080",
    "gpos66": "#575757",
    "gpos75": "#404040",
    "gpos100": "#000000",
    "gvar": "#000000",
    "acen": "#8B0000",
    "stalk": "#19A7CE",
}


def scale_fill_giemsa(**kwargs: Any):
    """Fill scale for cytoband stains (:data:`GIEMSA_COLOURS`).

    Pair with :func:`geom_ideogram`, mapping the stain column to both
    ``stain`` (which decides a band's *shape*) and ``fill`` (its colour)::

        geom_ideogram(aes(xstart="xstart", xend="xend", y="y",
                          stain="stain", fill="stain"), data=bands)
        + scale_fill_giemsa()
    """
    kwargs.setdefault("values", dict(GIEMSA_COLOURS))
    kwargs.setdefault("guide", "none")
    return gg.scale_fill_manual(**kwargs)


class GeomIdeogram(Geom):
    """Cytogenetic bands; aes ``xstart``, ``xend``, ``y`` (+ ``stain``).

    Parameters (as layer params)
    ----------------------------
    height : float
        Band height in data units (default ``0.6``).
    outline : str or None
        Colour of the border drawn around the whole chromosome. ``None``
        omits it.
    """

    required_aes: Tuple[str, ...] = ("xstart", "xend", "y")
    non_missing_aes: Tuple[str, ...] = ()
    default_aes: Mapping = Mapping(
        fill="grey80",
        colour="none",
        linewidth=0.2,
        linetype=1,
        alpha=None,
        stain="gneg",
    )
    draw_key = draw_key_polygon

    def setup_data(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        data = data.copy()
        if "stain" in data.columns:
            unknown = sorted(
                {str(s) for s in pd.unique(data["stain"].dropna())}
                - set(GIEMSA_COLOURS)
            )
            if unknown:
                raise ValueError(
                    f"GeomIdeogram: unrecognised Giemsa stain(s) {unknown!r}. "
                    f"Known stains: {sorted(GIEMSA_COLOURS)}. Colouring an "
                    "unknown stain grey would misreport the karyotype."
                )
        height = float(params.get("height") or 0.6)
        data["xmin"] = data["xstart"]
        data["xmax"] = data["xend"]
        data["ymin"] = data["y"] - height / 2.0
        data["ymax"] = data["y"] + height / 2.0
        return data

    @staticmethod
    def _centromere(acen: pd.DataFrame) -> pd.DataFrame:
        """Turn the ``acen`` bands into the two triangles of the waist.

        The lower band tapers to a point on its right, the upper one on its
        left, so together they pinch at the centromere.
        """
        acen = acen.sort_values("xmin", kind="stable").reset_index(drop=True)
        carried = [
            c
            for c in acen.columns
            if c not in ("x", "y", "xstart", "xend", "xmin", "xmax", "ymin", "ymax")
        ]
        pieces: List[pd.DataFrame] = []
        half = len(acen) / 2.0
        for i, row in acen.iterrows():
            mid = (row["ymin"] + row["ymax"]) / 2.0
            if i < half:  # tapers rightward
                xs = [row["xmin"], row["xmin"], row["xmax"]]
                ys = [row["ymin"], row["ymax"], mid]
            else:  # tapers leftward
                xs = [row["xmax"], row["xmax"], row["xmin"]]
                ys = [row["ymin"], row["ymax"], mid]
            piece = pd.DataFrame({"x": xs, "y": ys})
            for col in carried:
                piece[col] = row[col]
            piece["group"] = f"acen-{i}"
            pieces.append(piece)
        return pd.concat(pieces, ignore_index=True)

    def draw_panel(
        self,
        data: pd.DataFrame,
        panel_params: Any,
        coord: Any,
        height: Any = None,
        outline: Optional[str] = "grey30",
        na_rm: bool = False,
        **params: Any,
    ) -> Any:
        if data is None or data.empty:
            return null_grob()

        is_acen = (
            data["stain"].astype(str) == "acen"
            if "stain" in data.columns
            else pd.Series(False, index=data.index)
        )
        children = []

        bands = data[~is_acen]
        if not bands.empty:
            children.append(GeomRect.draw_panel(GeomRect(), bands, panel_params, coord))

        acen = data[is_acen]
        if not acen.empty:
            children.append(
                GeomPolygon.draw_panel(
                    GeomPolygon(), self._centromere(acen), panel_params, coord
                )
            )

        if outline is not None:
            frame = pd.DataFrame(
                {
                    "xmin": [data["xmin"].min()],
                    "xmax": [data["xmax"].max()],
                    "ymin": [data["ymin"].min()],
                    "ymax": [data["ymax"].max()],
                    "fill": ["none"],
                    "colour": [outline],
                    "linewidth": [float(data["linewidth"].iloc[0])],
                    "linetype": [data["linetype"].iloc[0]],
                    "alpha": [None],
                }
            )
            children.append(GeomRect.draw_panel(GeomRect(), frame, panel_params, coord))

        if not children:
            return null_grob()
        return _ggname("geom_ideogram", grob_tree(*children))


def geom_ideogram(
    mapping: Optional[Mapping] = None,
    data: Any = None,
    stat: str = "identity",
    position: str = "identity",
    *,
    height: Any = None,
    outline: Optional[str] = "grey30",
    na_rm: bool = False,
    show_legend: Any = None,
    inherit_aes: bool = True,
    **kwargs: Any,
) -> Any:
    """Chromosome ideogram layer; aes ``xstart``, ``xend``, ``y``, ``stain``.

    Feed it :func:`~ggtracks.read_cytoband` output and pair with
    :func:`scale_fill_giemsa`. See :class:`GeomIdeogram`.
    """
    from ggplot2_py.layer import layer

    return layer(
        geom=GeomIdeogram,
        stat=stat,
        data=data,
        mapping=mapping,
        position=position,
        show_legend=show_legend,
        inherit_aes=inherit_aes,
        params={
            "height": height,
            "outline": outline,
            "na_rm": na_rm,
            **kwargs,
        },
    )
