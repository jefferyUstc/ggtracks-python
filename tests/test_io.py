"""Tests for the file readers.

Text formats are exercised against fixtures written by the tests themselves,
so the assertions can be exact and the suite stays portable. bigWig needs a
real binary file; those tests skip when one is not available rather than
weakening what they check.
"""

from __future__ import annotations

import gzip
import os
import pathlib

import numpy as np
import pandas as pd
import pytest

import ggtracks as ggt
from ggtracks.io import (
    BedGraph,
    BigWig,
    detect_chrom_style,
    normalize_chrom,
    read_annotations,
    read_bed,
    read_bedgraph,
    read_cytoband,
    resolve_chrom,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

# One gene, one transcript, two exons, a CDS and both UTRs.
# GTF/GFF3 are 1-based inclusive, so exon 1 spans bases 100..199 = 100 bp.
GTF = """\
#!genome-build test
7\ttest\tgene\t100\t500\t.\t+\t.\tgene_id "G1"; gene_name "Alpha";
7\ttest\ttranscript\t100\t500\t.\t+\t.\tgene_id "G1"; gene_name "Alpha"; transcript_id "T1";
7\ttest\texon\t100\t199\t.\t+\t.\tgene_id "G1"; gene_name "Alpha"; transcript_id "T1";
7\ttest\texon\t400\t500\t.\t+\t.\tgene_id "G1"; gene_name "Alpha"; transcript_id "T1";
7\ttest\tCDS\t150\t199\t.\t+\t0\tgene_id "G1"; gene_name "Alpha"; transcript_id "T1";
7\ttest\tfive_prime_utr\t100\t149\t.\t+\t.\tgene_id "G1"; gene_name "Alpha"; transcript_id "T1";
7\ttest\tthree_prime_utr\t400\t500\t.\t+\t.\tgene_id "G1"; gene_name "Alpha"; transcript_id "T1";
7\ttest\tgene\t9000\t9500\t.\t-\t.\tgene_id "G2"; gene_name "Beta";
7\ttest\texon\t9000\t9500\t.\t-\t.\tgene_id "G2"; gene_name "Beta"; transcript_id "T2";
"""

GFF3 = """\
##gff-version 3
7\ttest\tgene\t100\t500\t.\t+\t.\tID=gene:G1;Name=Alpha
7\ttest\tmRNA\t100\t500\t.\t+\t.\tID=transcript:T1;Parent=gene:G1;Name=T1
7\ttest\texon\t100\t199\t.\t+\t.\tParent=transcript:T1
7\ttest\texon\t400\t500\t.\t+\t.\tParent=transcript:T1
7\ttest\tCDS\t150\t199\t.\t+\t0\tParent=transcript:T1
7\ttest\tfive_prime_UTR\t100\t149\t.\t+\t.\tParent=transcript:T1
7\ttest\tthree_prime_UTR\t400\t500\t.\t+\t.\tParent=transcript:T1
7\ttest\tgene\t9000\t9500\t.\t-\t.\tID=gene:G2;Name=Beta
7\ttest\tmRNA\t9000\t9500\t.\t-\t.\tID=transcript:T2;Parent=gene:G2;Name=T2
7\ttest\texon\t9000\t9500\t.\t-\t.\tParent=transcript:T2
"""

# BED-family: 0-based half-open.
BEDGRAPH = "chr7\t99\t199\t1.0\nchr7\t199\t299\t3.0\nchr7\t399\t499\t2.0\n"
CYTOBAND = "chr7\t0\t1000\tp11\tgneg\nchr7\t1000\t1500\tp10\tacen\nchr7\t1500\t3000\tq11\tgpos50\n"

BED6 = (
    "track name=peaks\n"
    "chr7\t99\t199\tpeak1\t50\t+\n"
    "chr7\t399\t499\tpeak2\t.\t-\n"
)
BED3 = "chr7\t99\t199\nchr7\t399\t499\n"
NARROWPEAK = (
    "chr7\t99\t199\tp1\t100\t.\t5.5\t3.2\t2.9\t40\n"
    "chr7\t399\t499\tp2\t200\t.\t8.1\t4.0\t3.5\t-1\n"
)
BROADPEAK = (
    "chr7\t99\t199\tp1\t100\t.\t5.5\t3.2\t2.9\n"
    "chr7\t399\t499\tp2\t200\t.\t8.1\t4.0\t3.5\n"
)

# GFF3 with SO gene/transcript types beyond the coding pair: a lncRNA
# under an ncRNA_gene and a pseudogenic transcript under a pseudogene.
GFF3_NCRNA = """\
##gff-version 3
7\ttest\tgene\t100\t500\t.\t+\t.\tID=gene:G1;Name=Alpha
7\ttest\tmRNA\t100\t500\t.\t+\t.\tID=transcript:T1;Parent=gene:G1;Name=T1
7\ttest\texon\t100\t199\t.\t+\t.\tParent=transcript:T1
7\ttest\texon\t400\t500\t.\t+\t.\tParent=transcript:T1
7\ttest\tncRNA_gene\t2000\t2400\t.\t-\t.\tID=gene:G3;Name=Mhrt
7\ttest\tlnc_RNA\t2000\t2400\t.\t-\t.\tID=transcript:T3;Parent=gene:G3;Name=T3
7\ttest\texon\t2000\t2100\t.\t-\t.\tParent=transcript:T3
7\ttest\texon\t2300\t2400\t.\t-\t.\tParent=transcript:T3
7\ttest\tpseudogene\t3000\t3200\t.\t+\t.\tID=gene:G4;Name=Ps1
7\ttest\tpseudogenic_transcript\t3000\t3200\t.\t+\t.\tID=transcript:T4;Parent=gene:G4
7\ttest\texon\t3000\t3200\t.\t+\t.\tParent=transcript:T4
"""


@pytest.fixture
def gtf(tmp_path):
    p = tmp_path / "a.gtf"
    p.write_text(GTF)
    return str(p)


@pytest.fixture
def gff3(tmp_path):
    p = tmp_path / "a.gff3"
    p.write_text(GFF3)
    return str(p)


@pytest.fixture
def bedgraph(tmp_path):
    p = tmp_path / "a.bedgraph"
    p.write_text(BEDGRAPH)
    return str(p)


@pytest.fixture
def cytoband(tmp_path):
    p = tmp_path / "cyto.txt"
    p.write_text(CYTOBAND)
    return str(p)


# --------------------------------------------------------------------------
# chromosome names
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,style,expected",
    [("1", "ucsc", "chr1"), ("chr1", "ucsc", "chr1"),
     ("chr1", "ensembl", "1"), ("1", "ensembl", "1"), ("1", None, "1")],
)
def test_normalize_chrom(name, style, expected):
    assert normalize_chrom(name, style) == expected


def test_normalize_chrom_rejects_unknown_style():
    with pytest.raises(ValueError, match="ucsc"):
        normalize_chrom("1", "hg38")


def test_detect_chrom_style():
    assert detect_chrom_style(["chr1", "chr2", "chrX"]) == "ucsc"
    assert detect_chrom_style(["1", "2", "X"]) == "ensembl"


def test_resolve_chrom_bridges_the_prefix_gap():
    assert resolve_chrom("7", ["chr7", "chr8"]) == "chr7"
    assert resolve_chrom("chr7", ["7", "8"]) == "7"
    assert resolve_chrom("chr7", ["chr7"]) == "chr7"


def test_resolve_chrom_lists_what_is_present():
    """A silent miss here is the most confusing failure in track plotting."""
    with pytest.raises(KeyError) as ei:
        resolve_chrom("chrZ", ["chr1", "chr2"])
    assert "chr1" in str(ei.value)


# --------------------------------------------------------------------------
# annotations
# --------------------------------------------------------------------------


def test_gtf_coordinates_are_half_open_and_length_preserving(gtf):
    df = read_annotations(gtf, genes=["Alpha"])
    exon = df[(df.feature == "exon") & (df.xstart == 100)].iloc[0]
    # GTF said 100..199 inclusive = 100 bp
    assert exon.xend == 200
    assert exon.xend - exon.xstart == 100


def test_gtf_and_gff3_agree(gtf, gff3):
    a = read_annotations(gtf, genes=["Alpha"]).sort_values(
        ["feature", "xstart"]).reset_index(drop=True)
    b = read_annotations(gff3, genes=["Alpha"]).sort_values(
        ["feature", "xstart"]).reset_index(drop=True)
    cols = ["chrom", "xstart", "xend", "strand", "feature", "gene_name", "tx_id"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_columns_match_the_contract(gtf):
    from ggtracks.io import FEATURE_COLUMNS

    assert tuple(read_annotations(gtf).columns) == FEATURE_COLUMNS


@pytest.mark.parametrize("fixture", ["gtf", "gff3"])
def test_gene_filter_accepts_name_or_id(fixture, request):
    path = request.getfixturevalue(fixture)
    by_name = read_annotations(path, genes=["Alpha"])
    by_id = read_annotations(path, genes=["G1"])
    assert set(by_name.gene_name) == {"Alpha"}
    assert set(by_id.gene_name) == {"Alpha"}
    assert len(by_name) == len(by_id)


@pytest.mark.parametrize("fixture", ["gtf", "gff3"])
def test_region_filter(fixture, request):
    path = request.getfixturevalue(fixture)
    df = read_annotations(path, region="7:8000-10000")
    assert set(df.gene_name) == {"Beta"}
    tup = read_annotations(path, region=("7", 8000, 10000))
    assert len(tup) == len(df)


def test_region_string_must_be_well_formed(gtf):
    with pytest.raises(ValueError, match="chrom:start-end"):
        read_annotations(gtf, region="7:8000")


def test_region_end_must_exceed_start(gtf):
    with pytest.raises(ValueError, match="end must exceed start"):
        read_annotations(gtf, region="7:100-100")


def test_chrom_style_rewrites_names(gtf):
    df = read_annotations(gtf, chrom_style="ucsc")
    assert set(df.chrom) == {"chr7"}


def test_format_is_inferred_but_overridable(tmp_path):
    # A GFF3 payload behind a .txt name still parses when told what it is.
    p = tmp_path / "ann.txt"
    p.write_text(GFF3)
    assert read_annotations(str(p), format="gff3").shape[0] > 0


def test_bad_format_fails_loud(gtf):
    with pytest.raises(ValueError, match="gtf.*gff3"):
        read_annotations(gtf, format="bed")


def test_gzip_is_transparent(tmp_path):
    p = tmp_path / "a.gtf.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(GTF)
    assert len(read_annotations(str(p), genes=["Alpha"])) > 0


def test_feature_vocabulary_is_normalised(gff3):
    """GFF3 spells them mRNA / five_prime_UTR; the output must not."""
    features = set(read_annotations(gff3).feature)
    assert "transcript" in features and "mRNA" not in features
    assert "five_prime_utr" in features and "five_prime_UTR" not in features


# --------------------------------------------------------------------------
# GFF3 beyond the coding vocabulary
# --------------------------------------------------------------------------


@pytest.fixture
def gff3_ncrna(tmp_path):
    p = tmp_path / "nc.gff3"
    p.write_text(GFF3_NCRNA)
    return str(p)


def test_gff3_noncoding_transcripts_survive(gff3_ncrna):
    """lnc_RNA under ncRNA_gene is a transcript like any other; silently
    dropping it (the literal-name matching bug) loses whole loci."""
    df = read_annotations(gff3_ncrna)
    mhrt = df[df.gene_name == "Mhrt"]
    assert set(mhrt.feature) == {"gene", "transcript", "exon"}
    assert len(mhrt[mhrt.feature == "exon"]) == 2
    assert set(mhrt[mhrt.feature == "transcript"].tx_id) == {"T3"}
    exon = mhrt[(mhrt.feature == "exon") & (mhrt.xstart == 2000)].iloc[0]
    assert exon.xend == 2101  # 1-based inclusive 2000..2100 -> half-open


def test_gff3_pseudogenes_are_gene_level(gff3_ncrna):
    df = read_annotations(gff3_ncrna)
    ps = df[df.gene_name == "Ps1"]
    assert "gene" in set(ps.feature)
    assert set(ps[ps.feature == "transcript"].tx_id) == {"T4"}


def test_gff3_gene_filter_matches_noncoding_genes(gff3_ncrna):
    df = read_annotations(gff3_ncrna, genes=["Mhrt"])
    assert set(df.gene_name) == {"Mhrt"}
    assert "transcript" in set(df.feature)


def test_gff3_so_terms_are_normalised_to_the_vocabulary(gff3_ncrna):
    features = set(read_annotations(gff3_ncrna).feature)
    assert "lnc_RNA" not in features and "ncRNA_gene" not in features
    assert features <= {"gene", "transcript", "exon", "CDS",
                        "five_prime_utr", "three_prime_utr", "utr"}


# --------------------------------------------------------------------------
# BED / narrowPeak / broadPeak
# --------------------------------------------------------------------------


@pytest.fixture
def bed6(tmp_path):
    p = tmp_path / "peaks.bed"
    p.write_text(BED6)
    return str(p)


@pytest.fixture
def narrowpeak(tmp_path):
    p = tmp_path / "peaks.narrowPeak"
    p.write_text(NARROWPEAK)
    return str(p)


def test_bed_is_lifted_to_one_based(bed6):
    df = read_bed(bed6)
    assert df.iloc[0].xstart == 100  # BED 99 -> 1-based 100
    assert df.iloc[0].xend == 200
    assert df.iloc[0].xend - df.iloc[0].xstart == 100


def test_bed_keeps_the_fields_it_finds(bed6, tmp_path):
    assert list(read_bed(bed6).columns) == [
        "chrom", "xstart", "xend", "name", "score", "strand"]
    p3 = tmp_path / "b3.bed"
    p3.write_text(BED3)
    assert list(read_bed(str(p3)).columns) == ["chrom", "xstart", "xend"]


def test_bed_dot_score_is_nan(bed6):
    df = read_bed(bed6)
    assert df.score.iloc[0] == 50.0
    assert np.isnan(df.score.iloc[1])


def test_narrowpeak_columns_and_peak_position(narrowpeak):
    df = read_bed(narrowpeak)
    assert list(df.columns) == [
        "chrom", "xstart", "xend", "name", "score", "strand",
        "signal_value", "p_value", "q_value", "peak"]
    # peak offset 40 from BED start 99 -> absolute 1-based 140
    assert df.peak.iloc[0] == 140.0
    assert np.isnan(df.peak.iloc[1])  # -1 means "not called"
    assert df.signal_value.iloc[0] == 5.5


def test_broadpeak_has_no_peak_column(tmp_path):
    p = tmp_path / "b.broadPeak"
    p.write_text(BROADPEAK)
    df = read_bed(str(p))
    assert "peak" not in df.columns
    assert list(df.columns[-3:]) == ["signal_value", "p_value", "q_value"]


def test_bed_format_is_inferred_but_overridable(narrowpeak, tmp_path):
    assert "signal_value" in read_bed(narrowpeak).columns  # from extension
    p = tmp_path / "renamed.bed"
    p.write_text(NARROWPEAK)
    assert "signal_value" in read_bed(str(p), format="narrowPeak").columns
    assert "signal_value" not in read_bed(str(p)).columns  # first 6 kept


def test_bed_region_filter_tolerates_chr_prefix(bed6):
    df = read_bed(bed6, region="7:50-250")
    assert len(df) == 1
    assert df.iloc[0]["name"] == "peak1"
    tup = read_bed(bed6, region=("7", 50, 250))
    assert len(tup) == 1


def test_bed_chrom_style_rewrites_names(bed6):
    assert set(read_bed(bed6, chrom_style="ensembl").chrom) == {"7"}


def test_bed_gzip_is_transparent(tmp_path):
    p = tmp_path / "peaks.bed.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(BED6)
    assert len(read_bed(str(p))) == 2


def test_bed_rejects_short_lines(tmp_path):
    p = tmp_path / "bad.bed"
    p.write_text("chr7\t99\n")
    with pytest.raises(ValueError, match="expected at least 3"):
        read_bed(str(p))


def test_narrowpeak_rejects_underfilled_lines(tmp_path):
    p = tmp_path / "bad.narrowPeak"
    p.write_text(BROADPEAK)  # 9 fields, narrowPeak needs 10
    with pytest.raises(ValueError, match="expected at least 10"):
        read_bed(str(p))


def test_bed_bad_format_fails_loud(bed6):
    with pytest.raises(ValueError, match="narrowPeak"):
        read_bed(bed6, format="gtf")


def test_bed_empty_file_keeps_the_contract(tmp_path):
    p = tmp_path / "empty.bed"
    p.write_text("# nothing here\n")
    df = read_bed(str(p))
    assert df.empty
    assert list(df.columns) == ["chrom", "xstart", "xend"]


# --------------------------------------------------------------------------
# bedGraph
# --------------------------------------------------------------------------


def test_bedgraph_is_lifted_to_one_based(bedgraph):
    """BED starts at 0, annotations at 1; the readers must agree."""
    df = read_bedgraph(bedgraph, "chr7", 100, 300)
    assert df.iloc[0].xstart == 100  # BED 99 -> 1-based 100
    assert df.iloc[0].xend == 200


def test_bedgraph_clips_to_the_query(bedgraph):
    df = read_bedgraph(bedgraph, "chr7", 150, 250)
    assert df.xstart.min() == 150
    assert df.xend.max() == 250


def test_bedgraph_whole_file(bedgraph):
    df = read_bedgraph(bedgraph)
    assert list(df.columns) == ["chrom", "xstart", "xend", "value"]
    assert len(df) == 3


def test_bedgraph_region_needs_both_bounds(bedgraph):
    with pytest.raises(ValueError, match="both start and end"):
        read_bedgraph(bedgraph, "chr7")


def test_bedgraph_rejects_short_lines(tmp_path):
    p = tmp_path / "bad.bedgraph"
    p.write_text("chr7\t0\t100\n")
    with pytest.raises(ValueError, match="expected at least 4"):
        BedGraph(str(p))


def test_bedgraph_chroms(bedgraph):
    assert BedGraph(bedgraph).chroms == {"chr7": 499}


# --------------------------------------------------------------------------
# cytoband
# --------------------------------------------------------------------------


def test_cytoband_columns_and_offset(cytoband):
    from ggtracks.io import CYTOBAND_COLUMNS

    df = read_cytoband(cytoband)
    assert tuple(df.columns) == CYTOBAND_COLUMNS
    assert df.iloc[0].xstart == 1  # BED 0 -> 1-based 1
    assert set(df.stain) == {"gneg", "acen", "gpos50"}


def test_cytoband_chrom_filter(cytoband):
    assert len(read_cytoband(cytoband, chrom="7")) == 3


def test_cytoband_rejects_short_lines(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("chr7\t0\t100\tp11\n")
    with pytest.raises(ValueError, match="expected 5"):
        read_cytoband(str(p))


# --------------------------------------------------------------------------
# bigWig  (needs a real binary; skipped when absent)
# --------------------------------------------------------------------------

_DEMO = pathlib.Path(
    os.environ.get(
        "GGTRACKS_TEST_BIGWIG",
        "/home/groups/xiaojie/nianping/projects/scLong/trackPy/demo/testdata/"
        "GSM5746911_MS_2cell_Input_rep2.bigWig",
    )
)
needs_bigwig = pytest.mark.skipif(
    not _DEMO.exists(), reason="no bigWig fixture (set GGTRACKS_TEST_BIGWIG)"
)


@pytest.fixture
def bw():
    with BigWig(str(_DEMO)) as handle:
        yield handle


@needs_bigwig
def test_header_and_index_parse(bw):
    assert bw.version >= 3
    assert len(bw.chroms) > 0
    # Zoom headers land at the right offset only if the header is read as the
    # full 64 bytes; a short read yields nonsense reduction levels.
    assert len(bw.zoom_levels) > 0
    assert all(0 < r < 10 ** 9 for r in bw.zoom_levels)
    assert list(bw.zoom_levels) == sorted(bw.zoom_levels)


@needs_bigwig
def test_full_resolution_query(bw):
    df = bw.query("chr7", 10_897_740, 10_908_050)
    assert list(df.columns) == ["xstart", "xend", "value"]
    assert len(df) > 0
    assert df.xstart.min() >= 10_897_740
    assert df.xend.max() <= 10_908_050
    assert (df.xend > df.xstart).all()


@needs_bigwig
def test_chr_prefix_is_reconciled(bw):
    a = bw.query("chr7", 10_900_000, 10_901_000)
    b = bw.query("7", 10_900_000, 10_901_000)
    pd.testing.assert_frame_equal(a, b)


@needs_bigwig
def test_binning_preserves_the_weighted_total(bw):
    """The integral of the signal must not depend on how it is binned."""
    lo, hi = 10_897_740, 10_908_050
    full = bw.query("chr7", lo, hi)
    exact = (full.value * (full.xend - full.xstart)).sum()
    for bins in (37, 200, 1000):
        b = bw.query("chr7", lo, hi, bins=bins)
        assert len(b) == bins
        got = (b.value * (b.xend - b.xstart)).sum()
        assert got == pytest.approx(exact, rel=1e-6)


@needs_bigwig
def test_zoom_level_is_used_for_wide_regions(bw):
    """A wide, coarsely binned request should come off a summary level."""
    lo, hi = 10_000_000, 15_000_000
    assert bw._pick_zoom(hi - lo, 200) is not None
    assert bw._pick_zoom(2000, 1000) is None  # too fine for any level


@needs_bigwig
def test_zoom_and_full_resolution_agree_on_the_integral(bw):
    lo, hi = 10_000_000, 15_000_000
    full = bw.query("chr7", lo, hi)
    exact = (full.value * (full.xend - full.xstart)).sum()
    zoomed = bw.query("chr7", lo, hi, bins=200, summary="sum")
    assert zoomed.value.sum() == pytest.approx(exact, rel=1e-3)


@needs_bigwig
@pytest.mark.parametrize("summary", ["mean", "max", "min", "sum"])
def test_summaries_are_ordered_sensibly(bw, summary):
    lo, hi = 10_897_740, 10_908_050
    df = bw.query("chr7", lo, hi, bins=20, summary=summary)
    assert len(df) == 20
    assert np.isfinite(df.value).all()


@needs_bigwig
def test_max_dominates_mean_dominates_min(bw):
    lo, hi = 10_897_740, 10_908_050
    kw = dict(bins=20)
    mx = bw.query("chr7", lo, hi, summary="max", **kw).value
    mn = bw.query("chr7", lo, hi, summary="min", **kw).value
    me = bw.query("chr7", lo, hi, summary="mean", **kw).value
    assert (mx >= me - 1e-9).all()
    assert (me >= mn - 1e-9).all()


@needs_bigwig
def test_unknown_chromosome_fails_loud(bw):
    with pytest.raises(KeyError, match="not found"):
        bw.query("chrNope", 1, 100)


@needs_bigwig
@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"start": 0, "end": 100}, "1-based"),
        ({"start": 100, "end": 100}, "end must exceed start"),
        ({"start": 1, "end": 100, "bins": 0}, "positive int"),
        ({"start": 1, "end": 100, "summary": "median"}, "summary must be"),
    ],
)
def test_query_argument_validation(bw, kwargs, match):
    with pytest.raises(ValueError, match=match):
        bw.query("chr7", **kwargs)


def test_non_bigwig_fails_loud(tmp_path):
    p = tmp_path / "not.bw"
    p.write_bytes(b"\x00" * 128)
    with pytest.raises(ValueError, match="not a bigWig"):
        BigWig(str(p))


def test_truncated_file_fails_loud(tmp_path):
    p = tmp_path / "short.bw"
    p.write_bytes(b"\x26\xfc\x8f\x88")  # correct magic, nothing else
    with pytest.raises(ValueError, match="truncated"):
        BigWig(str(p))


# --------------------------------------------------------------------------
# package wiring
# --------------------------------------------------------------------------


def test_io_is_reachable_from_the_top_level(gtf):
    assert ggt.read_annotations is read_annotations
    assert len(ggt.read_annotations(gtf, genes=["Alpha"])) > 0
