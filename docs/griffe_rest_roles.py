"""Griffe extension: render Sphinx reST cross-reference roles as plain Markdown.

Kept byte-identical in behaviour to the ``lrdata`` / ``scLR`` copies so the
three sites render cross-references the same way.

``ggtracks`` docstrings carry Sphinx roles throughout —
``:func:`~ggtracks.geom_range```, ``:class:`GenomicMapper```,
``:mod:`ggtracks.io``` and so on. MkDocs / mkdocstrings does not understand
reST roles, so without this they would render literally (e.g. the text
``:class:`Foo``` would appear verbatim).

This extension rewrites every ``:role:`target``` occurrence in a docstring into
inline code, using the short (last dotted component) name when the target is
``~``-prefixed or fully qualified — matching how Sphinx displays such links.
It is intentionally non-linking: it only cleans up presentation, which is safe
regardless of whether the target resolves.
"""

from __future__ import annotations

import re
from typing import Any

from griffe import Extension, Object

# :role:`target` , :py:role:`target` , :role:`Title <target>` , :role:`~mod.Name`
_ROLE = re.compile(r":[a-zA-Z][a-zA-Z:]*:`~?(?P<body>[^`]+?)`")


def _short(body: str) -> str:
    # ":role:`Display <mod.Name>`" -> Sphinx shows "Display"
    display = body.split("<", 1)[0].strip() or body.strip()
    # "~package.module.Name" / "package.module.Name" -> "Name"
    return display.rsplit(".", 1)[-1]


def _rewrite(text: str) -> str:
    return _ROLE.sub(lambda m: f"`{_short(m.group('body'))}`", text)


class ReSTRoles(Extension):
    """Strip Sphinx reST roles down to inline code in every docstring."""

    def on_object(self, *, obj: Object, **kwargs: Any) -> None:  # noqa: D102
        if obj.docstring and obj.docstring.value:
            obj.docstring.value = _rewrite(obj.docstring.value)
