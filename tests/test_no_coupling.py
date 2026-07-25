"""Static guarantee that ggtracks stays container-agnostic.

The whole point of the package is that it draws from plain DataFrames and does
not depend on any single-cell / genomics *data container*. This scans the
source with ``ast`` so even a lazy (function-local) import is caught — a
stronger check than the runtime ``sys.modules`` probe in ``test_smoke.py``.
"""

from __future__ import annotations

import ast
import pathlib

import ggtracks

FORBIDDEN = {"lrdata", "sclr", "mudata", "anndata", "scanpy"}


def _source_files():
    # rglob, not glob: subpackages (e.g. ``ggtracks.io``) must be scanned too,
    # otherwise a container import could hide one directory down.
    pkg_dir = pathlib.Path(ggtracks.__file__).parent
    return sorted(pkg_dir.rglob("*.py"))


def test_no_container_imports_in_source():
    offenders = []
    for f in _source_files():
        tree = ast.parse(f.read_text(), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".")[0] in FORBIDDEN:
                    offenders.append(f"{f.name}: imports {name}")
    assert not offenders, (
        "ggtracks must not import a data container:\n" + "\n".join(offenders)
    )
