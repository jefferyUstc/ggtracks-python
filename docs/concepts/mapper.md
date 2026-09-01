# The coordinate model

A >50 kb gene drawn at genomic scale leaves its exons as sub-pixel
slivers — the introns take all the room. Every transcript-axis plot
therefore needs a map that **compresses intronic spans**, and every layer
of the figure must go through the *same* map or the layers drift apart.
That map is `GenomicMapper`.

## Building a mapper

A mapper is a piecewise-linear genomic ↔ display transform, built from
ordered `(start, end, kind)` spans (`kind` is `"exon"` or `"intron"`) that
must **tile** the region — no overlap, no hole. A hole would give two
genomic positions the same display coordinate, so it raises instead.

You rarely build spans by hand. `GenomicMapper.from_intervals(exons)`
merges overlapping exons and fills the gaps as introns:

```python
import ggtracks as ggt

ann = ggt.read_annotations("genes.gtf.gz", genes=["Actb"])
exons = ann[ann.feature == "exon"]
mapper = ggt.GenomicMapper.from_intervals(
    zip(exons.xstart, exons.xend),
    intron_mode="clamp", target_gap_width=150,
)
```

This is the **exonic union** idiom: the union of *all* transcripts' exons.
Isoforms disagree on exon boundaries — a junction of one may fall inside
an exon of another — and the union keeps the display stable: the intron
spans that survive are the parts *no* transcript exon covers.

`Locus.from_features` wraps this whole step (plus the genomic range a
signal query needs) into one call — see
[Composing figures](composition.md).

## Compression modes

| Option | Behaviour |
|---|---|
| `intron_mode="scale"` (default) | multiplicative: display length = `length × intron_scale` (default 0.15) |
| `intron_mode="clamp"` | absolute: display length = `min(length, target_gap_width)` — the ggtranscript `shorten_gaps` behaviour; never *enlarges* a gap |
| `intron_min` (default 20 bp) | introns shorter than this are left alone |
| `collapse_introns=False` | identity map — no compression at all |

## Compression is a scale transform, not a data edit

Data stays in true genomic coordinates. `scale_x_genomic(mapper)` wraps
the mapper as the x scale's *transform* — the same mechanism a log scale
uses — so compression happens at render time and the axis shows
genomic-valued tick labels through the inverse map (SI-formatted: `4 kb`,
`32 Mb`). Tick candidates are generated in genomic space, so they land on
round genomic positions, then thinned in display space so compressed
regions don't pile labels on top of each other.

```python
p = (gg.ggplot()
     + ggt.geom_range(gg.aes(xstart="xstart", xend="xend", y="y"), data=...)
     + ggt.scale_x_genomic(mapper)
     + ggt.theme_tracks())
```

One subtlety makes this compose cleanly: the track geoms use
`xstart`/`xend` aesthetics rather than a plain `x`, and `scale_x_genomic`
registers `xstart` on **its own instance** rather than in ggplot2-python's
global position-aesthetic list. Every range/intron/junction/coverage layer
gets compressed consistently, while the package's default scales stay
faithful to R.

`coord_genomic(mapper, genomic_xlim=(lo, hi))` is the user-facing
convenience: the scale plus an optional clip expressed in *genomic*
coordinates. `base_x_scale()` exists only for multi-locus figures, where
the transform lives on per-panel scales instead (see
[Composing figures](composition.md)).

!!! tip "Common errors"
    - `GenomicMapper: span #i … leaves a hole / overlaps` — hand-built
      spans that don't tile. Use `from_intervals`, which builds a valid
      tiling for you.
    - A layer sits one base off the others — a frame missed the
      inclusive → half-open conversion. See
      [One coordinate dialect](coordinates.md).
