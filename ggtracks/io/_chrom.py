"""Chromosome-name reconciliation.

Annotation and signal files routinely disagree on whether chromosomes carry
the ``chr`` prefix (UCSC ``chr1`` vs Ensembl ``1``). Mixing the two silently
yields an empty plot, so the reconciliation is done explicitly and loudly
here rather than being guessed at each call site.
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional

__all__ = ["ChromStyle", "normalize_chrom", "detect_chrom_style", "resolve_chrom"]

#: Naming style for chromosomes: ``"ucsc"`` is ``chr1``, ``"ensembl"`` is ``1``.
ChromStyle = Literal["ucsc", "ensembl"]


def normalize_chrom(name: str, style: Optional[ChromStyle]) -> str:
    """Return *name* rewritten in *style*.

    ``"ucsc"`` prefixes with ``chr``; ``"ensembl"`` strips it. ``style=None``
    returns *name* unchanged, so callers can thread an optional style through
    without branching.
    """
    if style is None:
        return name
    if style == "ucsc":
        return name if name.startswith("chr") else f"chr{name}"
    if style == "ensembl":
        return name[3:] if name.startswith("chr") else name
    raise ValueError(
        f"normalize_chrom: style must be 'ucsc' or 'ensembl' (got {style!r})."
    )


def detect_chrom_style(names: Iterable[str]) -> ChromStyle:
    """Infer the prevailing style of a collection of chromosome names."""
    names = list(names)
    if not names:
        raise ValueError("detect_chrom_style: no chromosome names supplied.")
    n_chr = sum(1 for n in names if n.startswith("chr"))
    return "ucsc" if n_chr * 2 >= len(names) else "ensembl"


def resolve_chrom(name: str, available: Iterable[str]) -> str:
    """Match *name* against *available*, tolerating a ``chr`` prefix mismatch.

    Raises
    ------
    KeyError
        No match under either naming style. The message lists what *is*
        present, because "no data" from a silent miss is the single most
        common and most confusing failure in track plotting.
    """
    available = list(available)
    present = set(available)
    for candidate in (name, normalize_chrom(name, "ucsc"), normalize_chrom(name, "ensembl")):
        if candidate in present:
            return candidate
    shown = sorted(available)[:12]
    more = "" if len(available) <= 12 else f" (+{len(available) - 12} more)"
    raise KeyError(
        f"resolve_chrom: chromosome {name!r} not found. "
        f"Available: {shown}{more}."
    )
