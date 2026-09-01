"""Vertical composition — stack several ggplots into one aligned figure.

``plot_tracks`` composes tracks that share one x scale. Some figures
need rows that deliberately do not share it — a whole-chromosome
ideogram strip above a gene panel, or an overview locus above a zoomed
detail view. Facets cannot express that (in a grid, x scales run per
*column*), so such a figure is several plots, stacked.

:func:`vstack_gg` stacks through patchwork-python, which aligns panel
edges across rows even when the plots differ in structure (a faceted
track stack over a plain ggplot, with or without legends). What
ggtracks adds is its sizing convention: each plot keeps the height it
was measured at (``fig_height``, which :func:`~ggtracks.plot_tracks`
sets from its own gtable), so stacking never squashes a panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import ggplot2_py as gg
import patchwork

__all__ = ["vstack_gg"]


class _SizedPatchwork(patchwork.Patchwork):
    """Patchwork that renders in notebooks at its own pinned size.

    ``patchwork.Patchwork._repr_png_`` draws at a fixed 7 x 5 in; here
    the stack carries ``fig_width`` / ``fig_height`` / ``fig_dpi`` the
    way a finalized ggplot does, so display and ``save=`` stay WYSIWYG.
    """

    fig_width: float = 7.0
    fig_height: float = 5.0
    fig_dpi: int = 150

    def _repr_png_(self):
        from grid_py import get_state, grid_draw, grid_newpage

        grid_newpage(width=self.fig_width, height=self.fig_height,
                     dpi=float(self.fig_dpi))
        grid_draw(self.to_gtable())
        renderer = get_state().get_renderer()
        return renderer.to_png_bytes() if renderer is not None else None


def vstack_gg(
    plots: Sequence[Any],
    *,
    heights: Optional[Sequence[float]] = None,
    width: Optional[float] = None,
    save: Union[str, Path, None] = None,
    dpi: int = 300,
    display_dpi: int = 150,
):
    """Stack *plots* top-to-bottom into one panel-aligned figure.

    Parameters
    ----------
    plots
        The plots, top first — anything patchwork composes: ggplots,
        :func:`~ggtracks.plot_tracks` output, other patchworks.
    heights
        Row heights in inches, one per plot. Defaults to each plot's own
        ``fig_height`` (measured by :func:`~ggtracks.finalize_gg` /
        ``plot_tracks``). The figure's total height is their sum either
        way, so every plot renders at exactly the height it was sized
        for; rendering at a different total height scales the rows
        proportionally.
    width
        Figure width in inches. Defaults to the widest plot's
        ``fig_width``.
    save
        Optional output path; saved at ``width`` x ``sum(heights)``
        inches at *dpi*, like :func:`~ggtracks.finalize_gg`.
    dpi, display_dpi
        Raster resolution on disk and on screen.

    Returns
    -------
    patchwork.Patchwork
        The composed figure — display it, ``gg.ggsave`` it, or keep
        composing with patchwork operators and annotations.

    Examples
    --------
    Focus + context: the same ``Track`` list rendered twice (safe —
    ``plot_tracks`` copies layers), plus a connector track::

        link = Track("zoom", [geom_zoom_link(xstart=lo, xend=hi,
                                             track="zoom")],
                     height=0.3, y_breaks=[0.0], y_labels=[""])
        overview = plot_tracks(tracks + [link], mapper)
        detail = plot_tracks(tracks, mapper, genomic_xlim=(lo, hi))
        vstack_gg([overview, detail], save="zoom.png")

    Chromosome context above a browser figure::

        bands = read_cytoband("mm10_cytoband.txt.gz", chrom="chr7")
        ctx = (gg.ggplot()
               + geom_ideogram(gg.aes(xstart="xstart", xend="xend", y="y",
                                      stain="stain", fill="stain"),
                               data=bands.assign(y=1.0))
               + geom_highlight(xstart=locus.start, xend=locus.end)
               + scale_fill_giemsa() + theme_tracks()
               + gg.labs(x="", y=""))
        vstack_gg([ctx, main], heights=[0.6, main.fig_height])
    """
    plots = list(plots)
    if not plots:
        raise ValueError("vstack_gg: no plots to stack.")
    if heights is None:
        unsized = [i for i, p in enumerate(plots)
                   if getattr(p, "fig_height", None) is None]
        if unsized:
            raise ValueError(
                f"vstack_gg: plot(s) {unsized!r} carry no fig_height to "
                "default from — pass heights= explicitly."
            )
        heights = [float(p.fig_height) for p in plots]
    else:
        heights = [float(h) for h in heights]
        if len(heights) != len(plots):
            raise ValueError(
                f"vstack_gg: {len(heights)} heights for {len(plots)} plots."
            )
    if any(h <= 0 for h in heights):
        raise ValueError(
            f"vstack_gg: heights must be positive (got {heights!r})."
        )
    if width is None:
        known = [w for w in (getattr(p, "fig_width", None) for p in plots)
                 if w is not None]
        width = max(known) if known else 7.0
    total = float(sum(heights))

    composed = patchwork.wrap_plots(plots, ncol=1, heights=list(heights))
    stack = _SizedPatchwork(composed.plot, composed.patches)
    stack.fig_width = float(width)
    stack.fig_height = total
    stack.fig_dpi = int(display_dpi)
    if save is not None:
        gg.ggsave(str(save), stack, width=float(width), height=total, dpi=dpi)
    return stack
