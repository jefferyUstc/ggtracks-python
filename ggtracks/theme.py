"""Publication theme for the Biobabel track grammar.

:func:`theme_tracks` is ``theme_bw`` at a small publication ``base_size``
(body text and axis titles render at that size, in points), but with the
ggplot2 ``rel(0.8)`` small-text cascade **lifted**: in dense genomics
panels the tick / facet-strip / legend labels are load-bearing (a facet
strip maps each panel to a cell type; tick labels carry genomic position
and magnitude), so they are pinned to ~0.9–0.95× base rather than 0.8×.
At the default ``base_size=8`` that keeps the smallest text ~7.2pt instead
of ~5.6pt — inside the 7–8pt publication band — while body/axis titles
sit at 8pt and the title at ~9.6pt.

Three further departures from plain ``theme_bw``, all aimed at the dense
stacked-track look of a genome browser:

* **The panel border is lightened.** ``theme_bw`` draws it at
  ``col_mix(ink, paper, 0.2)`` — i.e. grey20, a near-black. With one
  border per track the browser view accumulates ``4 × n_tracks`` dark
  lines that compete with the data. The default here is grey80, which
  still delimits each panel but recedes.
* **The major grid is dropped** (as well as the minor grid). Genomic
  tracks are read against the shared x axis and each other, not against
  a grid.
* **Panel spacing is tightened** to a quarter of ``base_size``. Stacked
  tracks should read as *one* figure; ggplot2's default half-line gap
  makes them read as several. Being derived from ``base_size`` it
  rescales with the typography instead of being a fixed measure.
* **Chrome recedes, data stays dark.** Facet strips are a pale cool
  wash with no outline instead of ``theme_bw``'s grey85 box in a grey20
  frame, and tick marks, tick labels and titles use a muted ink rather
  than pure black, so the strip that names a track is read *after* the
  track itself.

All grammar ``pl.*`` track functions take a ``base_size`` argument that is
forwarded here, so a single number tunes every figure's typography *and*
its spacing.

Palettes are adaptive from https://github.com/omicverse/omicverse.
"""

from __future__ import annotations

from typing import Any, Optional

import ggplot2_py as gg

__all__ = ["PUB_BASE_SIZE", "PANEL_BORDER_COLOUR", "theme_tracks"]

#: Facet-strip wash; the pale cool grey behind every track name.
STRIP_FILL: str = "#EEF1F4"
#: Ink for titles and strip labels, and the lighter ink for tick labels.
INK: str = "#2B2B2B"
INK_MUTED: str = "#5A5A5A"
#: Tick-mark colour.
TICK_COLOUR: str = "#A0A0A0"

#: Default publication base font size (points). Body text and axis titles
#: render at this size; tick / strip / legend labels at ~0.9–0.95× (lifted
#: from ggplot2's default 0.8× cascade — see module docstring).
PUB_BASE_SIZE: float = 8.0

#: Default panel-border colour (grey80). Lighter than ``theme_bw``'s
#: grey20 so that stacked tracks are delimited without the borders
#: out-weighting the data.
PANEL_BORDER_COLOUR: str = "#CCCCCC"


def theme_tracks(
    base_size: float = PUB_BASE_SIZE,
    *,
    minor_grid: bool = False,
    major_grid: bool = False,
    border_colour: Optional[str] = PANEL_BORDER_COLOUR,
    panel_spacing: Any = None,
):
    """``theme_bw(base_size)`` tuned for stacked genomic tracks.

    Parameters
    ----------
    base_size
        Body / axis-title font size in points. Tick, facet-strip and legend
        labels follow at ~0.9–0.95× (not the ggplot2 default 0.8×), and the
        default panel spacing is derived from it.
    minor_grid, major_grid
        Keep the minor / major grid lines. Both default to ``False``:
        genomic tracks are read against the shared x axis and against each
        other, and a grid behind every panel only adds clutter.
    border_colour
        Panel-border colour. Defaults to :data:`PANEL_BORDER_COLOUR`
        (grey80). Pass ``None`` to drop the border entirely, or any colour
        string to override (``"grey20"`` restores the ``theme_bw`` look).
    panel_spacing
        Gap between panels, as a ``grid_py.Unit`` (e.g.
        ``ggplot2_py.unit(2, "pt")``). ``None`` (default) uses
        ``base_size / 4`` points, so the gap tracks the typography.

    Returns
    -------
    ggplot2_py.Theme
    """
    if panel_spacing is None:
        panel_spacing = gg.unit(base_size / 4.0, "pt")

    border = (
        gg.element_blank()
        if border_colour is None
        else gg.element_rect(colour=border_colour, fill=None)
    )

    return gg.theme_bw(base_size=base_size) + gg.theme(
        axis_text=gg.element_text(size=gg.rel(0.9), colour=INK_MUTED),
        axis_ticks=gg.element_line(colour=TICK_COLOUR),
        strip_background=gg.element_rect(fill=STRIP_FILL, colour="none"),
        strip_text=gg.element_text(size=gg.rel(0.95), colour=INK),
        strip_text_y=gg.element_text(size=gg.rel(0.95), angle=0, colour=INK),
        legend_text=gg.element_text(size=gg.rel(0.9), colour=INK_MUTED),
        plot_title=gg.element_text(colour=INK),
        panel_border=border,
        panel_grid_major=(gg.element_line() if major_grid else gg.element_blank()),
        panel_grid_minor=(gg.element_line() if minor_grid else gg.element_blank()),
        panel_spacing=panel_spacing,
        plot_margin=gg.margin(t=10, r=5, b=4, l=4),
    )
