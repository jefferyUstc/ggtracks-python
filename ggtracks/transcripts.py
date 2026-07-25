"""Transcript-table helpers: which isoform first, and the collapsed view.

Two questions come up for every gene model and neither is a plotting
question, which is why they are plain functions over the tidy feature table
rather than geoms:

* **Which transcript should be on top?** Row order carries meaning — the
  reader takes the first row as the representative isoform — so
  :func:`rank_transcripts` encodes a defensible ordering instead of leaving
  it to whatever the annotation file happened to list first.
* **One row per transcript, or one row for the gene?** An isoform view
  answers "what splicing is there"; a collapsed view answers "where is this
  gene". :func:`collapse_transcripts` produces the second from the first.
"""

from __future__ import annotations

from typing import Iterable, List, Literal, Optional, Sequence

import pandas as pd

from .mapper import _merge_intervals

__all__ = ["rank_transcripts", "collapse_transcripts"]

RankBy = Literal["canonical", "length", "name"]


def _tx_stats(features: pd.DataFrame) -> pd.DataFrame:
    """Per-transcript summary used by the ranking heuristics."""
    exons = features[features["feature"] == "exon"]
    cds = features[features["feature"] == "CDS"]
    utr5 = features[features["feature"] == "five_prime_utr"]
    utr3 = features[features["feature"] == "three_prime_utr"]

    length = (exons["xend"] - exons["xstart"]).groupby(exons["tx_id"]).sum()
    n_exons = exons.groupby("tx_id").size()
    cds_len = (cds["xend"] - cds["xstart"]).groupby(cds["tx_id"]).sum()

    stats = pd.DataFrame({"exon_length": length, "n_exons": n_exons})
    stats["cds_length"] = cds_len.reindex(stats.index).fillna(0)
    stats["coding"] = stats["cds_length"] > 0
    stats["both_utrs"] = stats.index.isin(set(utr5["tx_id"])) & stats.index.isin(
        set(utr3["tx_id"])
    )
    return stats


def rank_transcripts(
    features: pd.DataFrame,
    *,
    by: RankBy = "canonical",
) -> List[str]:
    """Order a gene's transcripts, most representative first.

    Parameters
    ----------
    features
        Tidy feature table (:func:`~ggtracks.read_annotations` output, or
        anything with ``feature``, ``tx_id``, ``xstart``, ``xend``).
    by
        ``"canonical"`` (default) — the isoform a reader would call the
        gene's main form: coding before non-coding, then annotated at both
        ends (both UTRs present), then more exons, then a longer CDS. It is
        a heuristic, not a database lookup; when an annotation carries a
        real canonical flag, prefer that.

        ``"length"`` — longest total exonic length first, which keeps the
        widest model on top.

        ``"name"`` — transcript id, when a stable arbitrary order is what
        you actually want.

    Returns
    -------
    list of str
        Transcript ids, best first.

    Raises
    ------
    ValueError
        Unknown *by*, or the table has no transcripts.
    """
    if by not in ("canonical", "length", "name"):
        raise ValueError(
            f"rank_transcripts: by must be 'canonical', 'length' or 'name' "
            f"(got {by!r})."
        )
    if "tx_id" not in features.columns:
        raise ValueError("rank_transcripts: features has no tx_id column.")

    stats = _tx_stats(features)
    stats = stats[stats.index.astype(str) != ""]
    if stats.empty:
        raise ValueError("rank_transcripts: no transcripts with exons found.")

    if by == "name":
        return sorted(stats.index.astype(str))
    if by == "length":
        ordered = stats.sort_values(
            ["exon_length", "n_exons"], ascending=[False, False], kind="stable"
        )
        return list(ordered.index.astype(str))

    ordered = stats.sort_values(
        ["coding", "both_utrs", "n_exons", "cds_length"],
        ascending=[False, False, False, False],
        kind="stable",
    )
    return list(ordered.index.astype(str))


def collapse_transcripts(
    features: pd.DataFrame,
    *,
    group_by: str = "gene_id",
    features_kept: Sequence[str] = ("exon", "CDS", "five_prime_utr", "three_prime_utr"),
) -> pd.DataFrame:
    """Merge a gene's transcripts into one representative model.

    Overlapping intervals of the same feature type are unioned, so the
    result answers "where does this gene have coding sequence" without the
    isoform-by-isoform detail — the gene view, as opposed to the isoform
    view.

    Parameters
    ----------
    features
        Tidy feature table.
    group_by
        Column identifying the gene (``"gene_id"`` by default;
        ``"gene_name"`` also works).
    features_kept
        Which feature types to collapse. Types outside this list are
        dropped, since ``gene`` and ``transcript`` spans are already
        summaries and would double up with the merged exons.

    Returns
    -------
    pandas.DataFrame
        Same columns as the input minus ``tx_id`` (which no longer applies),
        with one row per merged interval.
    """
    if group_by not in features.columns:
        raise ValueError(
            f"collapse_transcripts: no {group_by!r} column "
            f"(have {list(features.columns)!r})."
        )
    kept = features[features["feature"].isin(list(features_kept))]
    if kept.empty:
        raise ValueError(
            f"collapse_transcripts: none of {list(features_kept)!r} present in "
            f"the table (have {sorted(features['feature'].unique())!r})."
        )

    rows = []
    for (gene, feature), part in kept.groupby([group_by, "feature"], sort=False):
        merged = _merge_intervals(list(zip(part["xstart"], part["xend"])))
        template = part.iloc[0]
        for start, end in merged:
            row = {
                c: template[c]
                for c in part.columns
                if c not in ("xstart", "xend", "tx_id", "feature", group_by)
            }
            row[group_by] = gene
            row["feature"] = feature
            row["xstart"] = start
            row["xend"] = end
            rows.append(row)

    out = pd.DataFrame(rows)
    ordered = [c for c in features.columns if c in out.columns]
    extra = [c for c in out.columns if c not in ordered]
    return out[ordered + extra].sort_values(
        [group_by, "feature", "xstart"], kind="stable"
    ).reset_index(drop=True)
