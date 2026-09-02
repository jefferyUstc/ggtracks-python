"""Colour palettes for track plots — three jobs, three vocabularies.

**Feature roles** (:data:`FEATURE_COLOURS`) — what a gene-model figure is
*made of*: coding blocks, non-coding blocks, intron connectors, a lone
signal track, junction arcs, a highlight band, read rows, and one accent.
Every geom's default colour comes from here, so an unstyled figure already
reads as one system, and a caller who needs a role names it instead of
repeating a hex literal.

**Qualitative** (:data:`TRACK_PALETTES` / :func:`track_palettes`) — for
**categories**: read clusters, cell types, feature classes. The default,
``"ggtracks"``, is the house palette, ordered so that every *adjacent*
pair of slots stays apart under simulated colour-vision deficiency (see
the entry's note); the ArchR / Kelly / Wes-Anderson families are kept as
alternatives. Exposed as a simple getter so track geoms can drive
``scale_fill_manual`` / ``scale_colour_manual`` with a coherent set rather
than ad-hoc hex literals.

**Sequential** (:data:`SIGNAL_PALETTES` / :func:`signal_palette`) — light→dark
ramps for **signal tracks** (coverage, pileup depth, IP vs input). Genome
browsers encode intensity with *lightness*, not hue: a darker track reads as
"more signal" without consulting a legend, survives greyscale printing, and
several stacked tracks do not fight each other for attention. Reaching for a
20-colour qualitative palette to paint four coverage tracks is the common
mistake this family exists to prevent.

The two families differ in how they answer for more colours than they hold:
a qualitative palette **recycles** (with a warning) because its entries are
arbitrary and unordered; a sequential ramp **interpolates**, because it is
defined by its endpoints and any number of steps between them is meaningful.
"""

from __future__ import annotations

from typing import Optional

__all__ = [
    "FEATURE_COLOURS",
    "TRACK_PALETTES",
    "track_palettes",
    "SIGNAL_PALETTES",
    "signal_palette",
]

#: The colour of each thing a track figure is made of. Blues carry the gene
#: model (coding dark, non-coding light — the same hue family, so they read
#: as one gene), a slate neutral carries connectors, magenta carries
#: junction arcs so they never merge with the model, gold is the region
#: band, muted aqua the read rows, and one red is reserved for emphasis.
#: The hues share a family with the ``"ggtracks"`` qualitative palette, so
#: a figure sits next to a categorical panel without a visible seam.
FEATURE_COLOURS: dict[str, str] = {
    "cds": "#1F577B",
    "exon": "#78C2ED",
    "intron": "#4C5A66",
    "signal": "#279AD7",
    "junction": "#E069A6",
    "highlight": "#FCBC10",
    "read": "#9DC3C3",
    "accent": "#CB3E35",
}

#: Qualitative palettes. ``"ggtracks"`` is the default: the house hues,
#: pruned to the slots that hold enough lightness and chroma to do identity
#: work and ordered so every adjacent pair clears a simulated-CVD distance
#: of 9 (12.7 across the first eight) and a normal-vision distance of 15
#: (OKLab ΔE ×100 under simulated protanopia / deuteranopia, checked with a
#: palette validator against a white surface). Four slots
#: (orange, leaf, olive, coral) sit below 3:1 contrast on white and rely on
#: the strip label or legend that every track figure carries.
TRACK_PALETTES: dict[str, list[str]] = {
    "ggtracks": [
        "#279AD7", "#D48F3E", "#E069A6", "#368650", "#5E4D9A",
        "#01A0A7", "#CB3E35", "#941456", "#A56BA7", "#7CBB5F",
        "#D3396D", "#B6B812", "#EF7B77", "#5860A7", "#6E944A",
    ],
    "stallion": [
        "#D51F26", "#272E6A", "#208A42", "#89288F", "#F47D2B",
        "#FEE500", "#8A9FD1", "#C06CAB", "#E6C2DC", "#90D5E4",
        "#89C75F", "#F37B7D", "#9983BD", "#D24B27", "#3BBCA8",
        "#6E4B9E", "#0C727C", "#7E1416", "#D8A767", "#3D3D3D",
    ],
    "calm": [
        "#7DD06F", "#844081", "#688EC1", "#C17E73", "#484125",
        "#6CD3A7", "#597873", "#7B6FD0", "#CF4A31", "#D0CD47",
        "#722A2D", "#CBC594", "#D19EC4", "#5A7E36", "#D4477D",
        "#403552", "#76D73C", "#96CED5", "#CE54D1", "#C48736",
    ],
    "kelly": [
        "#FFB300", "#803E75", "#FF6800", "#A6BDD7", "#C10020",
        "#CEA262", "#817066", "#007D34", "#F6768E", "#00538A",
        "#FF7A5C", "#53377A", "#FF8E00", "#B32851", "#F4C800",
        "#7F180D", "#93AA00", "#593315", "#F13A13", "#232C16",
    ],
    "bear": [
        "#faa818", "#41a30d", "#fbdf72", "#367d7d", "#d33502",
        "#6ebcbc", "#37526d", "#916848", "#f5b390", "#342739",
        "#bed678", "#a6d9ee", "#0d74b6", "#60824f", "#725ca5", "#e0598b",
    ],
    "ironMan": [
        "#371377", "#7700FF", "#9E0142", "#FF0080", "#DC494C",
        "#F88D51", "#FAD510", "#FFFF5F", "#88CFA4", "#238B45",
        "#02401B", "#0AD7D3", "#046C9A", "#A2A475", "#595959",
    ],
    "circus": [
        "#D52126", "#88CCEE", "#FEE52C", "#117733", "#CC61B0",
        "#99C945", "#2F8AC4", "#332288", "#E68316", "#661101",
        "#F97B72", "#DDCC77", "#11A579", "#89288F", "#E73F74",
    ],
    "paired": [
        "#A6CDE2", "#1E78B4", "#74C476", "#34A047", "#F59899", "#E11E26",
        "#FCBF6E", "#F47E1F", "#CAB2D6", "#6A3E98", "#FAF39B", "#B15928",
    ],
    "grove": [
        "#1a1334", "#01545a", "#017351", "#03c383", "#aad962",
        "#fbbf45", "#ef6a32", "#ed0345", "#a12a5e", "#710162", "#3B9AB2",
    ],
    "summerNight": [
        "#2a7185", "#a64027", "#fbdf72", "#60824f", "#9cdff0",
        "#022336", "#725ca5",
    ],
    "zissou": ["#3B9AB2", "#78B7C5", "#EBCC2A", "#E1AF00", "#F21A00"],
    "darjeeling": ["#FF0000", "#00A08A", "#F2AD00", "#F98400", "#5BBCD6"],
    "rushmore": ["#E1BD6D", "#EABE94", "#0B775E", "#35274A", "#F2300F"],
    "captain": ["#7F7F7F", "#A1CDE1", "#12477C", "#EC9274", "#67001E"],
}


def track_palettes(name: str = "ggtracks", n: Optional[int] = None) -> list[str]:
    """Return the hex colours of a named palette.

    Parameters
    ----------
    name
        One of :data:`TRACK_PALETTES` (default ``"ggtracks"``, the
        CVD-ordered house set; ``"stallion"`` is ArchR's 20 colours).
    n
        If given, return exactly *n* colours — truncating, or recycling
        with a warning when the palette is shorter than requested.

    Raises
    ------
    KeyError
        Unknown palette name (fail loud rather than silently defaulting).
    """
    if name not in TRACK_PALETTES:
        raise KeyError(
            f"track_palettes: unknown palette {name!r}; choose from "
            f"{sorted(TRACK_PALETTES)}."
        )
    cols = list(TRACK_PALETTES[name])
    if n is None:
        return cols
    if n <= len(cols):
        return cols[:n]
    import warnings

    warnings.warn(
        f"track_palettes({name!r}, n={n}): palette has only {len(cols)} "
        f"colours; recycling to fill {n}.",
        stacklevel=2,
    )
    reps = (n + len(cols) - 1) // len(cols)
    return (cols * reps)[:n]


#: Sequential light→dark ramps for signal tracks, as ``(light, dark)``
#: endpoints. ``"grey"`` is the genome-browser default (see
#: :func:`signal_palette`); the hued ramps run from a pale tint to the dark
#: member of the matching hue family, for when several signal groups must
#: also be told apart.
SIGNAL_PALETTES: dict[str, tuple[str, str]] = {
    "grey": ("#D0D0D0", "#1A1A1A"),
    "blue": ("#C9E4F6", "#1F577B"),
    "red": ("#F0C3C3", "#5A1713"),
    "green": ("#CDE5D2", "#2D5C33"),
    "purple": ("#D8C9DC", "#823D86"),
    "orange": ("#F3DEB6", "#745228"),
}


def signal_palette(name: str = "grey", n: Optional[int] = None) -> list[str]:
    """Return *n* colours along a sequential light→dark signal ramp.

    Parameters
    ----------
    name
        One of :data:`SIGNAL_PALETTES` (default ``"grey"``, the IGV-style
        greyscale).
    n
        How many shades to return, evenly spaced along the ramp.
        ``None`` returns the two endpoints — the ramp's own definition.

        ``n=1`` is a deliberate special case: it returns the **dark** end,
        not the light one. A lone coverage track should be prominent, and
        an evenly-spaced sample of size one is otherwise ambiguous.

    Returns
    -------
    list of str
        Hex colours, lightest first (so the last entry is always the
        darkest — a natural fit for ``control → treatment`` orderings).

    Raises
    ------
    KeyError
        Unknown ramp name (fail loud rather than silently defaulting).
    ValueError
        ``n`` is not a positive integer.

    Examples
    --------
    >>> signal_palette("grey", n=2)      # input (light) vs IP (dark)
    ['#d0d0d0', '#1a1a1a']
    """
    if name not in SIGNAL_PALETTES:
        raise KeyError(
            f"signal_palette: unknown ramp {name!r}; choose from "
            f"{sorted(SIGNAL_PALETTES)}."
        )
    low, high = SIGNAL_PALETTES[name]
    if n is None:
        return [low, high]
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError(f"signal_palette: n must be a positive int (got {n!r}).")
    if n == 1:
        return [high]

    from scales import seq_gradient_pal

    ramp = seq_gradient_pal(low, high)
    return list(ramp([i / (n - 1) for i in range(n)]))
