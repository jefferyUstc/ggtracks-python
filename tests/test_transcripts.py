"""Tests for the transcript-table helpers and repeated strand arrows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggplot2_py as gg
from ggplot2_py.plot import ggplot_build
import ggtracks as ggt


def _row(feature, xstart, xend, tx, gene="G1"):
    return {
        "chrom": "7", "xstart": xstart, "xend": xend, "strand": "+",
        "feature": feature, "gene_id": gene, "gene_name": "Alpha", "tx_id": tx,
    }


# T1: coding, both UTRs, 3 exons.  T2: coding, no UTRs, 2 exons.
# T3: non-coding, 4 exons (most exons, but should rank last as non-coding).
FEATURES = pd.DataFrame(
    [
        _row("exon", 100, 200, "T1"), _row("exon", 300, 400, "T1"),
        _row("exon", 500, 600, "T1"),
        _row("CDS", 150, 350, "T1"),
        _row("five_prime_utr", 100, 150, "T1"),
        _row("three_prime_utr", 500, 600, "T1"),

        _row("exon", 100, 200, "T2"), _row("exon", 500, 900, "T2"),
        _row("CDS", 120, 180, "T2"),

        _row("exon", 100, 150, "T3"), _row("exon", 200, 250, "T3"),
        _row("exon", 300, 350, "T3"), _row("exon", 400, 450, "T3"),
    ]
)


# --------------------------------------------------------------------------
# rank_transcripts
# --------------------------------------------------------------------------


def test_canonical_puts_coding_before_noncoding():
    ranked = ggt.rank_transcripts(FEATURES)
    assert ranked[-1] == "T3"


def test_canonical_prefers_full_utr_annotation():
    """T1 and T2 are both coding; T1 is annotated at both ends."""
    ranked = ggt.rank_transcripts(FEATURES)
    assert ranked.index("T1") < ranked.index("T2")


def test_length_ranks_by_exonic_span():
    ranked = ggt.rank_transcripts(FEATURES, by="length")
    assert ranked[0] == "T2"  # 100 + 400 = 500 bp, the longest


def test_name_is_a_stable_arbitrary_order():
    assert ggt.rank_transcripts(FEATURES, by="name") == ["T1", "T2", "T3"]


def test_unknown_ordering_fails_loud():
    with pytest.raises(ValueError, match="canonical.*length.*name"):
        ggt.rank_transcripts(FEATURES, by="best")


def test_missing_tx_column_fails_loud():
    with pytest.raises(ValueError, match="no tx_id column"):
        ggt.rank_transcripts(FEATURES.drop(columns="tx_id"))


def test_no_transcripts_fails_loud():
    empty = FEATURES[FEATURES["feature"] == "gene"]
    with pytest.raises(ValueError, match="no transcripts"):
        ggt.rank_transcripts(empty)


# --------------------------------------------------------------------------
# collapse_transcripts
# --------------------------------------------------------------------------


def test_overlapping_and_touching_exons_are_unioned():
    """Three transcripts contribute nine exon records; overlapping *and*
    abutting ones fuse, so the gene view is three blocks."""
    collapsed = ggt.collapse_transcripts(FEATURES)
    exons = collapsed[collapsed["feature"] == "exon"]
    got = sorted(zip(exons["xstart"], exons["xend"]))
    assert got == [(100, 250), (300, 450), (500, 900)]


def test_collapse_drops_the_transcript_identity():
    collapsed = ggt.collapse_transcripts(FEATURES)
    assert "tx_id" not in collapsed.columns


def test_collapse_keeps_feature_classes_separate():
    collapsed = ggt.collapse_transcripts(FEATURES)
    assert {"exon", "CDS", "five_prime_utr", "three_prime_utr"} == set(
        collapsed["feature"]
    )


def test_collapse_never_widens_the_gene():
    collapsed = ggt.collapse_transcripts(FEATURES)
    assert collapsed["xstart"].min() == FEATURES["xstart"].min()
    assert collapsed["xend"].max() == FEATURES["xend"].max()


def test_collapse_by_gene_name_also_works():
    collapsed = ggt.collapse_transcripts(FEATURES, group_by="gene_name")
    assert set(collapsed["gene_name"]) == {"Alpha"}


def test_collapse_unknown_group_fails_loud():
    with pytest.raises(ValueError, match="no 'locus' column"):
        ggt.collapse_transcripts(FEATURES, group_by="locus")


def test_collapse_without_usable_features_fails_loud():
    genes_only = pd.DataFrame([_row("gene", 100, 900, "")])
    with pytest.raises(ValueError, match="none of"):
        ggt.collapse_transcripts(genes_only)


def test_collapsed_model_plots(tmp_path):
    collapsed = ggt.collapse_transcripts(FEATURES)
    exons = collapsed[collapsed["feature"] == "exon"].assign(y=1.0)
    p = (
        gg.ggplot()
        + ggt.geom_range(gg.aes(xstart="xstart", xend="xend", y="y"), data=exons)
        + ggt.theme_tracks()
    )
    out = tmp_path / "collapsed.png"
    gg.ggsave(str(out), p, width=3, height=1, dpi=72)
    assert out.stat().st_size > 0


# --------------------------------------------------------------------------
# repeated strand arrows
# --------------------------------------------------------------------------


INTRONS = pd.DataFrame(
    {"xstart": [200, 400], "xend": [300, 500], "y": [1.0, 1.0], "strand": ["+", "-"]}
)


def _intron_plot(**kwargs):
    return (
        gg.ggplot()
        + ggt.geom_intron(
            gg.aes(xstart="xstart", xend="xend", y="y", strand="strand"),
            data=INTRONS,
            **kwargs,
        )
    )


def test_default_density_is_one_arrow():
    from ggtracks.geom_intron import geom_intron
    import inspect

    assert inspect.signature(geom_intron).parameters["arrow_density"].default == 1


@pytest.mark.parametrize("density", [1, 3, 8])
def test_density_renders(density, tmp_path):
    out = tmp_path / f"arrows_{density}.png"
    gg.ggsave(str(out), _intron_plot(arrow_density=density), width=3, height=1, dpi=72)
    assert out.stat().st_size > 0


@pytest.mark.parametrize("bad", [0, -1, 2.5, True])
def test_bad_density_fails_loud(bad, tmp_path):
    with pytest.raises(Exception) as ei:
        gg.ggsave(
            str(tmp_path / "bad.png"), _intron_plot(arrow_density=bad),
            width=2, height=1, dpi=36,
        )
    chain, exc = [], ei.value
    while exc is not None:
        chain.append(str(exc))
        exc = exc.__cause__ or exc.__context__
    assert any("arrow_density" in c for c in chain)


def test_arrow_segments_scale_with_density():
    """Density n lays n arrowheads along each intron instead of one."""
    from ggtracks.geom_intron import GeomIntron
    from ggplot2_py.geom import GeomSegment

    prepared = pd.DataFrame(
        {"x": [200.0], "xend": [300.0], "y": [1.0], "yend": [1.0], "strand": ["+"]}
    )
    captured = {}

    class _Spy(GeomSegment):
        def draw_panel(self, data, *a, **kw):
            captured["n"] = len(data)
            return None

    GeomIntron._strand_arrow_grob(
        _Spy(), "+", prepared, None, None, object(), 0, "butt", "round", False, 4
    )
    assert captured["n"] == 4
