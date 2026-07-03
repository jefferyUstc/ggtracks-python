"""Curated qualitative colour palettes for track plots.

Faithful values from jjAnno's ``useMyCol`` (the ArchR/Kelly/Wes-Anderson
families commonly used for genomics tracks). Exposed as a simple getter
so track geoms can drive ``scale_fill_manual``/``scale_colour_manual``
with a coherent, colour-blind-aware set rather than ad-hoc hex literals.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["TRACK_PALETTES", "track_palettes"]

TRACK_PALETTES: dict[str, list[str]] = {
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


def track_palettes(name: str = "stallion", n: Optional[int] = None) -> list[str]:
    """Return the hex colours of a named palette.

    Parameters
    ----------
    name
        One of :data:`TRACK_PALETTES` (default ``"stallion"``, ArchR's
        20-colour set).
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
