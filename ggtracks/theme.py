"""Publication theme for the Biobabel track grammar.

:func:`theme_tracks` is ``theme_bw`` at a small publication ``base_size``
(body text and axis titles render at that size, in points), but with the
ggplot2 ``rel(0.8)`` small-text cascade **lifted**: in dense genomics
panels the tick / facet-strip / legend labels are load-bearing (a facet
strip maps each panel to a cell type; tick labels carry genomic position
and magnitude), so they are pinned to ~0.9–0.95× base rather than 0.8×.
At the default ``base_size=8`` that keeps the smallest text ~7.2pt instead
of ~5.6pt — inside the 7–8pt publication band — while body/axis titles
sit at 8pt and the title at ~9.6pt. The minor grid is dropped to declutter.

All grammar ``pl.*`` track functions take a ``base_size`` argument that is
forwarded here, so a single number tunes every figure's typography.
"""

from __future__ import annotations

import ggplot2_py as gg

__all__ = ["PUB_BASE_SIZE", "theme_tracks"]

#: Default publication base font size (points). Body text and axis titles
#: render at this size; tick / strip / legend labels at ~0.9–0.95× (lifted
#: from ggplot2's default 0.8× cascade — see module docstring).
PUB_BASE_SIZE: float = 8.0


def theme_tracks(base_size: float = PUB_BASE_SIZE, *, minor_grid: bool = False):
    """``theme_bw(base_size)`` with the small-text cascade lifted for print.

    Parameters
    ----------
    base_size
        Body / axis-title font size in points. Tick, facet-strip and legend
        labels follow at ~0.9–0.95× (not the ggplot2 default 0.8×).
    minor_grid
        Keep the minor grid lines (default ``False`` — dropped to declutter
        dense multi-panel track figures).
    """
    return gg.theme_bw(base_size=base_size) + gg.theme(
        axis_text=gg.element_text(size=gg.rel(0.9)),
        strip_text=gg.element_text(size=gg.rel(0.95)),
        strip_text_y=gg.element_text(size=gg.rel(0.95), angle=0),
        legend_text=gg.element_text(size=gg.rel(0.9)),
        panel_grid_minor=(gg.element_line() if minor_grid else gg.element_blank()),
        plot_margin=gg.margin(t=10, r=5, b=4, l=4),
    )
