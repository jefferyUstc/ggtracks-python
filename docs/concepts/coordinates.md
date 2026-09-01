# One coordinate dialect

Every DataFrame a ggtracks geom consumes speaks the same dialect:
intervals are **1-based, half-open** — `[xstart, xend)`, so an interval's
length is always `xend - xstart`. An exon covering bases 100..200 of a
chromosome (101 bases, counting from 1) is the row `xstart=100, xend=201`.

Genomics formats disagree with each other, and with this dialect:

| Source | Native convention | Translation applied by `ggtracks.io` |
|---|---|---|
| GTF / GFF3 (`read_annotations`) | 1-based **inclusive** `[start, end]` | `xend = end + 1` (position and length both preserved) |
| BED / narrowPeak / broadPeak (`read_bed`) | 0-based half-open | `+1` to **both** ends |
| bedGraph (`read_bedgraph`) | 0-based half-open | `+1` to both ends |
| bigWig (`BigWig.query`) | 0-based half-open | query bounds taken 1-based; results lifted `+1` |
| cytoband (`read_cytoband`) | 0-based half-open | `+1` to both ends |

The point of doing this **at the file boundary and nowhere else** is a
one-base bug that is otherwise almost undetectable: mix a 1-based gene
model with 0-based signal on one axis and the coverage track sits one base
off the exons beneath it — invisible at any realistic zoom, and wrong in
every figure.

!!! warning "The trap: everything still renders"
    `GenomicMapper` computes only *relative* layout and reports positions
    back in whatever frame you fed it. A frame built with the wrong origin
    (for example applying the BED `start - 1` conversion to data that is
    already 1-based) still renders — the axis labels just read one base
    low, and that layer sits one base off every other layer in the figure.
    If you build frames by hand, pick the 1-based half-open dialect and
    stay consistent within a figure.

## Points are coordinates too

The narrowPeak `peak` field is a 0-based *offset* from the interval start
(`-1` for "not called"). `read_bed` translates it like any other
coordinate — to an absolute 1-based position, `NaN` when absent. The
statistical fields (`signal_value`, `p_value`, `q_value`) pass through
untouched: ENCODE writes `-1` there for "not assigned", and reinterpreting
a statistic is the caller's decision, not a coordinate translation.

## `chr1` vs `1`

Annotation and signal files routinely disagree on the `chr` prefix, and
the failure mode is an *empty plot*, not an error. The readers reconcile
the two styles at query time (`BigWig.query("7", ...)` finds `chr7`), and
say what *is* present when they cannot:

```python
from ggtracks.io import resolve_chrom, normalize_chrom, detect_chrom_style

resolve_chrom("7", ["chr7", "chr8"])   # -> "chr7"
resolve_chrom("chrZ", ["chr1"])        # KeyError listing what exists
normalize_chrom("7", "ucsc")           # -> "chr7"
```

Every reader also takes `chrom_style="ucsc" | "ensembl"` to rewrite names
on the way in, so a whole figure can be forced into one style.
