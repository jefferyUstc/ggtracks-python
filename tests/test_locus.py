"""Tests for the locus-preparation helpers (Locus, gene_model_layers)."""

from __future__ import annotations

import pandas as pd
import pytest

import ggtracks as ggt
from ggtracks import Locus, gene_model_layers


def make_features():
    # Post-read_annotations frame: 1-based half-open. Exons [100,200) and
    # [400,501); CDS inside exon 1; the intron gap is [200,400).
    return pd.DataFrame({
        "chrom": "7",
        "xstart": [100, 100, 400, 150, 100, 400],
        "xend": [501, 200, 501, 200, 150, 501],
        "strand": "+",
        "feature": ["transcript", "exon", "exon", "CDS",
                    "five_prime_utr", "three_prime_utr"],
        "gene_id": "G1",
        "gene_name": "Alpha",
        "tx_id": "T1",
    })


# --------------------------------------------------------------------------
# Locus
# --------------------------------------------------------------------------


def test_from_features_builds_the_exon_union_mapper():
    loc = Locus.from_features(make_features())
    assert loc.chrom == "7"
    assert (loc.start, loc.end) == (100, 501)
    assert loc.region == ("7", 100, 501)
    assert loc.mapper.genomic_extent == (100, 501)
    assert [s.kind for s in loc.mapper.spans] == ["exon", "intron", "exon"]


def test_flank_extends_the_range_uncompressed():
    loc = Locus.from_features(make_features(), flank=50, intron_scale=0.15)
    assert (loc.start, loc.end) == (50, 551)
    m = loc.mapper
    assert m.to_display(100) - m.to_display(50) == pytest.approx(50)
    assert m.to_display(551) - m.to_display(501) == pytest.approx(50)
    # the real intron still compresses
    assert m.to_display(400) - m.to_display(200) == pytest.approx(200 * 0.15)


def test_flank_clamps_at_position_one():
    assert Locus.from_features(make_features(), flank=200).start == 1


def test_mapper_options_are_forwarded():
    loc = Locus.from_features(make_features(), collapse_introns=False)
    lo, hi = loc.mapper.display_extent
    assert hi - lo == pytest.approx(501 - 100)


def test_missing_columns_fail_loud():
    with pytest.raises(ValueError, match="missing column"):
        Locus.from_features(pd.DataFrame({"xstart": [1], "xend": [2]}))


def test_multi_chromosome_frames_are_refused():
    f = make_features()
    f.loc[f.index[-1], "chrom"] = "8"
    with pytest.raises(ValueError, match="lives on one"):
        Locus.from_features(f)


def test_needs_exons():
    with pytest.raises(ValueError, match="no exon rows"):
        Locus.from_features(make_features().query("feature != 'exon'"))


def test_negative_flank_is_refused():
    with pytest.raises(ValueError, match="flank"):
        Locus.from_features(make_features(), flank=-1)


# --------------------------------------------------------------------------
# gene_model_layers
# --------------------------------------------------------------------------


def test_standard_model_is_exon_cds_intron():
    layers = gene_model_layers(make_features(), track="gene")
    assert len(layers) == 3
    for lyr in layers:
        assert set(lyr.data["track"]) == {"gene"}
        assert (lyr.data["y"] == 1.0).all()


def test_introns_come_from_the_merged_exon_union():
    introns = gene_model_layers(make_features())[-1].data
    assert introns[["xstart", "xend"]].values.tolist() == [[200, 400]]
    assert set(introns["strand"]) == {"+"}


def test_no_cds_no_cds_layer():
    layers = gene_model_layers(make_features().query("feature != 'CDS'"))
    assert len(layers) == 2


def test_single_exon_has_no_intron_layer():
    f = make_features()
    f = f[~((f.feature == "exon") & (f.xstart == 400))]
    assert len(gene_model_layers(f)) == 2  # exon + CDS


def test_mixed_strands_drop_the_arrow_claim():
    f = make_features()
    f.loc[f.index[-1], "strand"] = "-"
    introns = gene_model_layers(f)[-1].data
    assert "strand" not in introns.columns


def test_requires_exons():
    with pytest.raises(ValueError, match="no exon rows"):
        gene_model_layers(make_features().query("feature != 'exon'"))


def test_layers_drop_into_plot_tracks():
    loc = Locus.from_features(make_features())
    p = ggt.plot_tracks(
        [ggt.Track("gene", gene_model_layers(loc.features, track="gene"),
                   height=0.4, y_breaks=[1.0], y_labels=[""])],
        loc.mapper,
    )
    assert p.fig_height > 0
