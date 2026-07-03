"""Tests for :class:`ggtracks.GenomicMapper`.

Covers both intron-compression modes:

* ``scale`` — multiplicative (the historic behaviour), and
* ``clamp`` — absolute ``min(length, target_gap_width)``, the
  ggtranscript ``shorten_gaps``-style mode added for the track grammar.
"""

from __future__ import annotations

import numpy as np
import pytest

from ggtracks import GenomicMapper

SPANS = [(0, 100, "exon"), (100, 1100, "intron"), (1100, 1200, "exon")]


def test_scale_mode_is_default_and_unchanged():
    m = GenomicMapper(SPANS, intron_scale=0.15, intron_min=20)
    assert m.intron_mode == "scale"
    assert m.to_display(1100) == pytest.approx(250.0)
    assert m.display_extent == (0.0, 350.0)


def test_clamp_mode_preserves_exon_widths_and_clamps_gap():
    m = GenomicMapper(SPANS, intron_mode="clamp", target_gap_width=50, intron_min=20)
    assert m.to_display(0) == pytest.approx(0.0)
    assert m.to_display(100) == pytest.approx(100.0)
    assert m.to_display(1100) == pytest.approx(150.0)
    assert m.to_display(1200) == pytest.approx(250.0)
    assert m.display_extent == (0.0, 250.0)
    assert m.target_gap_width == 50


def test_clamp_only_reduces_never_enlarges():
    spans = [(0, 100, "exon"), (100, 130, "intron"), (130, 230, "exon")]
    m = GenomicMapper(spans, intron_mode="clamp", target_gap_width=50, intron_min=20)
    assert m.to_display(130) == pytest.approx(130.0)


def test_clamp_respects_intron_min():
    spans = [(0, 100, "exon"), (100, 115, "intron"), (115, 215, "exon")]
    m = GenomicMapper(spans, intron_mode="clamp", target_gap_width=5, intron_min=20)
    assert m.to_display(115) == pytest.approx(115.0)


def test_roundtrip_on_exonic_positions_both_modes():
    exonic = np.array([0, 50, 99, 1100, 1150, 1199], dtype=float)
    for mode, kw in (("scale", {}), ("clamp", {"target_gap_width": 40})):
        m = GenomicMapper(SPANS, intron_mode=mode, intron_min=20, **kw)
        back = m.to_genomic_array(m.to_display_array(exonic))
        assert np.allclose(back, exonic, atol=1e-6), (mode, back)


def test_from_intervals_threads_clamp_params():
    m = GenomicMapper.from_intervals(
        [(0, 100), (1100, 1200)],
        intron_mode="clamp",
        target_gap_width=25,
        intron_min=20,
    )
    assert m.intron_mode == "clamp"
    assert m.target_gap_width == 25
    assert m.to_display(1100) == pytest.approx(125.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"intron_mode": "bogus"},
        {"target_gap_width": -1},
    ],
)
def test_invalid_params_fail_loud(kwargs):
    with pytest.raises(ValueError):
        GenomicMapper(SPANS, **kwargs)


class TestGenomicMapper:
    """Coordinate math, compression and validation (ported from scLR's
    mapper unit tests when GenomicMapper moved into this package)."""

    def test_from_intervals_basic(self):
        m = GenomicMapper.from_intervals(
            [(1000, 1200), (3000, 3500), (5000, 5300)],
            intron_scale=0.1,
        )
        assert m.to_display(1000) == 0.0
        assert m.to_display(1200) == 200.0
        assert m.to_display(3000) == pytest.approx(380.0)
        assert m.to_display(5300) == pytest.approx(1330.0)

    def test_round_trip(self):
        m = GenomicMapper.from_intervals(
            [(1000, 1200), (3000, 3500), (5000, 5300)],
            intron_scale=0.1,
        )
        for p in (1000, 1100, 1200, 3100, 3499, 5000, 5300):
            d = m.to_display(p)
            back = m.to_genomic(d)
            assert back == pytest.approx(p)

    def test_intron_min_preserves_short_introns(self):
        m = GenomicMapper.from_intervals(
            [(1000, 1100), (1110, 1200)],
            intron_scale=0.1, intron_min=20,
        )
        assert m.to_display(1200) == pytest.approx(200.0)

    def test_no_compression(self):
        m = GenomicMapper.from_intervals(
            [(1000, 1200), (3000, 3500)], collapse_introns=False,
        )
        assert m.to_display(3500) == pytest.approx(2500.0)

    def test_validation_overlapping_spans_raises(self):
        with pytest.raises(ValueError, match="overlap"):
            GenomicMapper([
                (100, 200, "exon"),
                (150, 300, "exon"),
            ])

    def test_validation_unknown_kind(self):
        with pytest.raises(ValueError, match="kind"):
            GenomicMapper([(100, 200, "unknown")])  # type: ignore[arg-type]

    def test_validation_zero_length(self):
        with pytest.raises(ValueError, match="non-positive"):
            GenomicMapper([(100, 100, "exon")])

    def test_tick_positions(self):
        m = GenomicMapper.from_intervals([(1000, 2000)])
        disp, gen = m.tick_positions(n=3)
        assert len(disp) == 3
        assert int(gen[0]) == 1000
        assert int(gen[-1]) == 2000
