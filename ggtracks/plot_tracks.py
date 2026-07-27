"""``plot_tracks`` — compose stacked genome-browser tracks into one ggplot.

Each track is a list of already-constructed ggplot layers whose data
carries a ``track`` column (the facet-row key). Tracks are stacked with
``facet_grid2(rows="track", scales="free_y")`` over a shared
intron-compressed genomic x (``scale_x_genomic``), with per-track panel
heights via ``force_panelsizes`` and an independent fill/colour scale per
track via ``new_scale_*`` — so a gene-model track's feature legend and a
coverage track's cluster legend coexist without clashing.

All tracks must use a **numeric y** (e.g. transcript rank, depth, arc
height) so the single shared y aesthetic is type-consistent across the
free-y facets.
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Sequence, Tuple

import ggplot2_py as gg
import ggh4x
from grid_py import Unit, is_unit, unit_c

from .mapper import GenomicMapper
from .scale import base_x_scale, scale_x_genomic
from ._render import finalize_gg, natural_height
from .theme import theme_tracks, PUB_BASE_SIZE

__all__ = ["Track", "plot_tracks"]


class Track:
    """One browser track: a name, its ggplot layers, a panel height, and the
    aesthetic (``"fill"``/``"colour"``/``None``) to start fresh before it.

    Parameters
    ----------
    height
        Panel height. A plain number is a **relative** share (grid ``"null"``
        units) that also counts as one inch of figure height, which is the
        historic behaviour. A ``grid_py.Unit`` instead pins the panel to an
        **absolute** size — ``unit(1.5, "cm")``, ``unit(4, "lines")``, … — and
        the figure grows to fit it exactly.

        Note that grid resolves ``"lines"`` against the *device* font (12 pt),
        not against ``theme_tracks(base_size=)``; for panels that track the
        theme's typography, derive inches from ``base_size`` yourself.
    y_limits
        ``(low, high)`` for this panel's y axis. Tracks default to free y,
        which is right for a browser but **wrong for a comparison**: an IP
        track drawn to its own maximum next to an input track drawn to its
        own looks equally tall whatever the ratio between them. To compare,
        give the tracks the same limits — :func:`~ggtracks.signal_limits`
        computes a robust pair from the pooled values.
    y_breaks, y_labels
        Explicit y tick positions for this panel, and optionally the text to
        print at them. A gene-model row's ``y`` is a row index, so numeric
        ticks on it are noise; ``y_breaks=[1.0], y_labels=[""]`` reduces the
        axis to a single unlabelled mark.

        (An empty ``y_breaks`` list does *not* remove the axis: the
        underlying ``scale_y_continuous`` then prints the panel-relative
        positions instead of nothing, so it is rejected here rather than
        silently drawing meaningless numbers.)
    range_label
        Draw the panel's range as a ``[low-high]`` badge in the corner, the
        genome-browser convention that buys back the horizontal space an
        axis would spend. Requires *y_limits* — the badge states the axis,
        so there must be an axis to state.
    """

    __slots__ = (
        "name",
        "layers",
        "height",
        "new_scale",
        "y_limits",
        "y_breaks",
        "y_labels",
        "range_label",
    )

    def __init__(
        self,
        name: str,
        layers: Sequence[Any],
        *,
        height: Any = 1.0,
        new_scale: Optional[str] = None,
        y_limits: Optional[Tuple[float, float]] = None,
        y_breaks: Optional[Sequence[float]] = None,
        y_labels: Optional[Sequence[str]] = None,
        range_label: bool = False,
    ) -> None:
        if range_label and y_limits is None:
            raise ValueError(
                f"Track({name!r}): range_label needs y_limits — the badge "
                "reports the axis range. Compute one with signal_limits()."
            )
        if y_breaks is not None and len(list(y_breaks)) == 0:
            raise ValueError(
                f"Track({name!r}): an empty y_breaks does not remove the "
                "axis — the scale falls back to printing panel-relative "
                'positions. Use y_breaks=[<pos>], y_labels=[""] instead.'
            )
        if y_labels is not None:
            if y_breaks is None:
                raise ValueError(
                    f"Track({name!r}): y_labels needs matching y_breaks."
                )
            if len(list(y_labels)) != len(list(y_breaks)):
                raise ValueError(
                    f"Track({name!r}): y_labels has {len(list(y_labels))} "
                    f"entries but y_breaks has {len(list(y_breaks))}."
                )
        if y_limits is not None:
            lo, hi = float(y_limits[0]), float(y_limits[1])
            if hi <= lo:
                raise ValueError(
                    f"Track({name!r}): y_limits high must exceed low "
                    f"(got {y_limits!r})."
                )
            y_limits = (lo, hi)
        self.name = name
        self.layers = list(layers)
        self.height = height if is_unit(height) else float(height)
        self.new_scale = new_scale
        self.y_limits = y_limits
        self.y_breaks = None if y_breaks is None else list(y_breaks)
        self.y_labels = None if y_labels is None else list(y_labels)
        self.range_label = bool(range_label)

    def __repr__(self) -> str:
        bits = [f"{self.name!r}", f"layers={len(self.layers)}", f"height={self.height!r}"]
        if self.y_limits is not None:
            bits.append(f"y_limits={self.y_limits!r}")
        return f"<Track {' '.join(bits)}>"


def _format_range(lo: float, hi: float) -> str:
    """``[0-1479]`` — a compact statement of the axis, not a precise one.

    The badge exists to save the space an axis would take, so it rounds:
    coverage ceilings land on values like 232.895 and printing every digit
    of that would defeat the purpose.
    """
    def fmt(v: float) -> str:
        v = float(v)
        if v.is_integer():
            return f"{v:.0f}"
        return f"{v:.0f}" if abs(v) >= 10 else f"{v:.3g}"

    return f"[{fmt(lo)}-{fmt(hi)}]"


def _y_scales(tracks_by_name: dict, order: Sequence[str]) -> Optional[list]:
    """Per-panel y scales, or ``None`` when every track is content with free y."""
    scales: list = []
    used = False
    for name in order:
        track = tracks_by_name[name]
        if track.y_limits is None and track.y_breaks is None:
            scales.append(None)
            continue
        kwargs: dict = {}
        if track.y_limits is not None:
            kwargs["limits"] = track.y_limits
        if track.y_breaks is not None:
            kwargs["breaks"] = track.y_breaks
        if track.y_labels is not None:
            kwargs["labels"] = track.y_labels
        scales.append(gg.scale_y_continuous(**kwargs))
        used = True
    return scales if used else None


def _panel_rows(heights: Sequence[Any]) -> Tuple[Any, float]:
    """``heights`` → (``force_panelsizes`` argument, relative inch allowance).

    Plain numbers stay relative (``"null"``) and are also counted as inches
    of figure height; ``Unit`` entries are absolute and are measured off the
    gtable instead, so they contribute nothing here.
    """
    if not any(is_unit(h) for h in heights):
        return list(heights), float(sum(heights))
    rows = unit_c(*(h if is_unit(h) else Unit(h, "null") for h in heights))
    relative = float(sum(h for h in heights if not is_unit(h)))
    return rows, relative


def _as_facet_column(values: Any, categories: Sequence[str], field: str) -> Any:
    """Order a facet key, refusing values that have no panel to land in.

    The build drops rows whose facet value matches no panel *silently*
    (``dropna(subset=["PANEL"])``), so a typo would quietly delete data from
    the figure. Catching it here turns that into an error.
    """
    import pandas as _pd

    known = set(categories)
    seen = {v for v in _pd.unique(_pd.Series(values).dropna())}
    unknown = sorted(str(v) for v in seen - known)
    if unknown:
        raise ValueError(
            f"plot_tracks: {field} value(s) {unknown!r} match no panel "
            f"(expected one of {list(categories)!r}). Rows carrying them "
            "would be dropped from the figure without warning."
        )
    return _pd.Categorical(values, categories=list(categories), ordered=True)


def plot_tracks(
    tracks: Sequence[Track],
    mapper: Optional[GenomicMapper] = None,
    *,
    mappers: Optional[dict] = None,
    track_order: Optional[Sequence[str]] = None,
    genomic_xlim: Optional[Tuple[float, float]] = None,
    background: Optional[Sequence[Any]] = None,
    n_breaks: int = 5,
    title: str = "",
    base_size: float = PUB_BASE_SIZE,
    save: Any = None,
):
    """Stack *tracks* into one faceted browser ggplot.

    Tracks become facet **rows**. With *mappers*, loci become facet
    **columns**, each with its own genomic coordinate system — so several
    genes can be compared side by side without pretending they share an
    axis.

    Parameters
    ----------
    tracks
        Ordered tracks (top → bottom). Each track's layer data must carry
        a ``track`` column equal to the track's ``name`` and a numeric
        ``y``.
    mapper
        Shared :class:`GenomicMapper` for a single locus — every track is
        aligned to it.
    mappers
        ``{locus name: GenomicMapper}`` for a multi-locus figure. The
        mapping's order *is* the column order, and layer data selects its
        column through a ``locus`` column. A layer that omits ``locus``
        appears in every column, which is what a highlight or a reference
        line wants.

        Give exactly one of *mapper* and *mappers*.
    track_order
        Explicit top→bottom facet order; defaults to the order of *tracks*.
    genomic_xlim
        Optional ``(start, end)`` genomic clip. Single-locus only — with
        several loci, scope each :class:`GenomicMapper` to the region you
        want instead.
    background
        Layers drawn **before** every track, i.e. underneath the data —
        where a :func:`~ggtracks.geom_highlight` band belongs. Layers added
        to the returned plot with ``+`` land on top instead.
    n_breaks
        Target number of x tick labels per panel. Genomic labels are wide
        ("142.904 Mb"), so narrow panels — several loci side by side, in
        particular — need fewer of them than the default.

    Notes
    -----
    Per-track y behaviour (limits, ticks, range badge) is declared on the
    :class:`Track` itself; see its ``y_limits`` / ``y_breaks`` /
    ``range_label`` arguments.

    Examples
    --------
    >>> plot_tracks(tracks, mappers={"Actb": m1, "Myc": m2})   # doctest: +SKIP
    """
    if not tracks:
        raise ValueError("plot_tracks: no tracks to plot.")
    if isinstance(mapper, dict):
        raise TypeError(
            "plot_tracks: a mapping of loci goes to the keyword argument "
            "`mappers=`; the positional `mapper` takes a single GenomicMapper."
        )
    if (mapper is None) == (mappers is None):
        raise ValueError(
            "plot_tracks: give exactly one of mapper= (one locus) or "
            "mappers= (several loci)."
        )
    if mappers is not None:
        if not mappers:
            raise ValueError("plot_tracks: mappers is empty.")
        if genomic_xlim is not None:
            raise ValueError(
                "plot_tracks: genomic_xlim applies to a single locus. With "
                "mappers=, build each GenomicMapper over the region you want."
            )
    loci = list(mappers) if mappers is not None else None

    order = list(track_order) if track_order is not None else [t.name for t in tracks]
    by_name = {t.name: t for t in tracks}
    missing = [n for n in order if n not in by_name]
    if missing:
        raise ValueError(
            f"plot_tracks: track_order names {missing!r} have no matching "
            f"Track (have {sorted(by_name)!r})."
        )

    import pandas as _pd
    import warnings as _warnings

    # First pass: find out which tracks the data actually names, and catch
    # layers that name none. A facet row exists only where data puts one.
    seen_tracks: set = set()
    broadcasting: list = []
    for tr in tracks:
        for lyr in tr.layers:
            data = getattr(lyr, "data", None)
            if not isinstance(data, _pd.DataFrame):
                continue
            if "track" in data.columns:
                seen_tracks.update(str(v) for v in _pd.unique(data["track"].dropna()))
            elif len(data):
                broadcasting.append(tr.name)

    if broadcasting:
        # Without the facet column a layer is repeated into *every* panel, so
        # it would be draped over the other tracks rather than sitting in its
        # own. That is a mistake inside a Track; `background=` is the
        # supported way to span the stack deliberately.
        raise ValueError(
            f"plot_tracks: layer(s) in track(s) {sorted(set(broadcasting))!r} "
            "carry no `track` column, so they would be drawn on every panel "
            "instead of their own. Add a track column to that layer's data, "
            "or pass the layer as background= to span the stack on purpose."
        )

    # Checked before the empty-track handling below, so that a mistyped value
    # is reported as the mismatch it is rather than as "that track has no
    # data" — the two look identical from the far side.
    unknown = sorted(seen_tracks - {str(n) for n in order})
    if unknown:
        raise ValueError(
            f"plot_tracks: track value(s) {unknown!r} in the layer data match "
            f"no Track (have {[str(n) for n in order]!r}). Rows carrying them "
            "would be dropped from the figure without warning."
        )

    # A track with nothing to draw (no layers, or layers whose data is empty)
    # cannot get a panel. Dropping it here keeps `heights` aligned with the
    # panels that remain — otherwise force_panelsizes shifts every height
    # onto the wrong row.
    absent = [n for n in order if str(n) not in seen_tracks]
    if absent:
        _warnings.warn(
            f"plot_tracks: no data for track(s) {absent!r}; they are omitted "
            "from the figure.",
            stacklevel=2,
        )
        order = [n for n in order if str(n) in seen_tracks]
    if not order:
        raise ValueError("plot_tracks: no track has any data to draw.")

    # Ordering the facet keys means rewriting each layer's data — onto a
    # *copy* of the layer. Writing through to the caller's layer would make
    # the returned plot depend on what happens to those Track objects
    # afterwards: build two figures from one Track list and the first would
    # silently re-render with the second's facet order.
    prepared: List[List[Any]] = []
    for tr in tracks:
        layers: List[Any] = []
        for lyr in tr.layers:
            data = getattr(lyr, "data", None)
            if isinstance(data, _pd.DataFrame):
                d = data.copy()
                touched = False
                if "track" in d.columns:
                    d["track"] = _as_facet_column(d["track"], order, "track")
                    touched = True
                if loci is not None and "locus" in d.columns:
                    d["locus"] = _as_facet_column(d["locus"], loci, "locus")
                    touched = True
                if touched:
                    lyr = copy.copy(lyr)
                    lyr.data = d
            layers.append(lyr)
        prepared.append(layers)

    import ggnewscale

    p = gg.ggplot()
    for layer in background or ():
        p = p + layer
    for tr, layers in zip(tracks, prepared):
        if tr.new_scale == "fill":
            p = p + ggnewscale.new_scale_fill()
        elif tr.new_scale == "colour":
            p = p + ggnewscale.new_scale_colour()
        for lyr in layers:
            p = p + lyr
        if tr.range_label and tr.name in order:
            badge = _pd.DataFrame({
                "xpos": [0.99],
                "ypos": [0.95],
                "label": [_format_range(*tr.y_limits)],
                "track": _pd.Categorical([tr.name], categories=order, ordered=True),
            })
            p = p + gg.geom_abs_text(
                gg.aes(xpos="xpos", ypos="ypos", label="label"),
                data=badge,
                inherit_aes=False,
                hjust=1.0,
                vjust=1.0,
                size=base_size * 0.32,
            )

    if loci is None:
        p = p + ggh4x.facet_grid2(rows="track", scales="free_y",
                                  labeller=gg.label_wrap_gen(width=13))
    else:
        p = p + ggh4x.facet_grid2(rows="track", cols="locus", scales="free",
                                  labeller=gg.label_wrap_gen(width=13))
    heights = [next(t.height for t in tracks if t.name == name) for name in order]
    rows, relative_inches = _panel_rows(heights)
    p = p + ggh4x.force_panelsizes(rows=rows)

    y_scales = _y_scales(by_name, order)
    if loci is None:
        if y_scales is not None:
            p = p + ggh4x.facetted_pos_scales(y=y_scales)
        p = p + scale_x_genomic(mapper, n_breaks=n_breaks)
        if genomic_xlim is not None:
            lo, hi = genomic_xlim
            p = p + gg.coord_cartesian(
                xlim=(float(mapper.to_display(lo)), float(mapper.to_display(hi)))
            )
    else:
        # The genomic transform now lives on the per-panel scales, so the
        # global one must stay untransformed — it exists only so the build
        # creates a panel scale list at all.
        p = p + base_x_scale()
        # facetted_pos_scales indexes by SCALE_X / SCALE_Y id, and under a
        # free grid those run per column and per row respectively — so the x
        # list is one scale per locus, in the order `mappers` was given.
        p = p + ggh4x.facetted_pos_scales(
            x=[scale_x_genomic(mappers[name], n_breaks=n_breaks) for name in loci],
            y=y_scales,
        )
    p = (
        p
        + gg.labs(title=title, x="Genomic position", y="")
        + theme_tracks(base_size)
    )
    # Figure height = everything grid can measure (axes, title, strips,
    # margins, and any absolutely-sized panels) + the inch allowance for
    # panels left relative. Measuring the chrome rather than assuming a
    # constant is what keeps large `base_size` figures from being clipped.
    return finalize_gg(
        p, save=save, height=natural_height(p) + relative_inches
    )
