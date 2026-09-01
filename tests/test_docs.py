"""The docs must cover the public surface — and only real names.

The API reference is curated by hand (a flat, 51-name surface groups
better by role than by module), which is exactly how reference docs rot.
These tests pin both directions: every ``__all__`` name has a mkdocstrings
directive, and every directive points at something that exists.
"""

from __future__ import annotations

import pathlib
import re

import ggtracks

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
DIRECTIVE = re.compile(r"^::: (ggtracks[\w.]*)$", re.MULTILINE)


def directives() -> list[str]:
    out: list[str] = []
    for md in sorted((DOCS / "reference").glob("*.md")):
        out += DIRECTIVE.findall(md.read_text())
    return out


def test_every_public_name_is_documented():
    found = directives()
    assert found, "no mkdocstrings directives found under docs/reference/"
    documented = {d.rsplit(".", 1)[-1] for d in found}
    missing = [n for n in ggtracks.__all__ if n not in documented]
    assert not missing, (
        f"public names missing from docs/reference: {missing!r}"
    )


def test_every_directive_resolves():
    """A typo in a directive would fail `mkdocs build --strict`; catching it
    here keeps the docs gate inside the test suite."""
    for identifier in directives():
        obj = ggtracks
        for part in identifier.split(".")[1:]:
            obj = getattr(obj, part)
