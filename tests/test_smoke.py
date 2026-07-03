"""Import / wiring smoke test for the ggtracks public surface.

Confirms the package loads all modules (catches import-rewrite / relocation
breakage) and that the coordinate model round-trips. Geom-level rendering is
covered by ``test_geoms.py``.
"""

from __future__ import annotations

import ggtracks as ggt


def test_public_surface_present():
    for name in ggt.__all__:
        assert hasattr(ggt, name), f"missing public name: {name}"


def test_no_container_coupling():
    import sys
    import ggtracks  # noqa: F401
    assert "lrdata" not in sys.modules
    assert "sclr" not in sys.modules


def test_mapper_roundtrip():
    m = ggt.GenomicMapper.from_intervals(
        [(1000, 1200), (1500, 1700), (2000, 2300)],
        intron_mode="clamp", target_gap_width=100,
    )
    d = m.to_display(1100.0)
    g = m.to_genomic(d)
    assert abs(g - 1100.0) < 1e-6
    lo, hi = m.genomic_extent
    assert lo == 1000 and hi == 2300
