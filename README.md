<img src="assets/logo.svg" align="right" width="128" height="128" alt="ggtracks logo"/>

# ggtracks

**Grammar-of-graphics genomic tracks for the [ggplot2-python](https://github.com/Bio-Babel/ggplot2-python) ecosystem**

`ggtracks` gives you real ggplot2-python `Geom`/`Stat` citizens for genomic tracks
(exon/feature ranges, introns with strand chevrons, splice-junction arcs, read
pileups, coverage signal, chromosome ideograms), readers for the file formats
those tracks come in, a **genomic coordinate model with intron compression**
(`GenomicMapper`), and a track-stacking composer (`plot_tracks`) that aligns any
number of tracks — over one locus or several side by side.


## Install

```bash
pip install ggtracks-python
```

## Documentation

Concepts (coordinate contract, mapper, composition) and the full API
reference are an MkDocs site:

```bash
pip install -e ".[docs]"
mkdocs serve        # or: mkdocs build --strict
```

## Quick start

From an annotation file and a signal file — no hand-built frames:

```python
import ggplot2_py as gg
import ggtracks as ggt

# 1. gene model straight out of a GTF (or GFF3; .gz is fine)
ann   = ggt.read_annotations("genes.gtf.gz", genes=["Actb"])
exons = ann[ann["feature"] == "exon"]

# 2. a coordinate model that compresses the introns at render time
mapper = ggt.GenomicMapper.from_intervals(
    zip(exons["xstart"], exons["xend"]),
    intron_mode="clamp", target_gap_width=100,
)

# 3. coverage over the same region, binned off the file's own zoom levels
chrom = exons["chrom"].iloc[0]
lo, hi = int(exons["xstart"].min()), int(exons["xend"].max())
cov = ggt.read_bigwig("signal.bw", chrom, lo, hi, bins=400)

# 4. stack them on one shared, intron-compressed genomic axis
ggt.plot_tracks(
    [
        ggt.Track("coverage", [
            ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"),
                              data=cov.assign(track="coverage")),
        ]),
        ggt.Track("gene model", [
            ggt.geom_range(gg.aes(xstart="xstart", xend="xend", y="y", fill="feature"),
                           data=exons.assign(y=1.0, track="gene model")),
        ], height=0.4, y_breaks=[1.0], y_labels=[""]),
    ],
    mapper,
    title="Actb",
)
```

Everything is a normal `ggplot`, so keep adding to it: `+ scale_fill_manual(...)`,
`+ labs(...)`, `+ theme(...)`.

## What's in the box

**Readers** (`ggtracks.io`, no compiled dependencies) — `read_annotations` /
`read_gtf` / `read_gff3`, `BigWig` / `read_bigwig` (pure-Python, uses the file's
zoom levels for wide regions), `BedGraph` / `read_bedgraph`, `read_cytoband`,
and `resolve_chrom` for the perennial `chr1` vs `1` mismatch.

**Geoms and stats** — `geom_range` (exons, CDS, read blocks), `geom_intron`
(strand arrows or IGV chevrons), `geom_junction` (sashimi arcs), `geom_coverage`
+ `StatBinCoverage` (interval signal, drawn as steps), `StatPileup` (read
stacking), `geom_ideogram` + `scale_fill_giemsa` (chromosome bands),
`geom_highlight` (a region band across every track), `geom_zoom_link`
(focus + context connector).

**Coordinates and scales** — `GenomicMapper`, `scale_x_genomic`,
`coord_genomic`, `signal_limits` (robust y limits for spiky coverage).

**Composition** — `Track` + `plot_tracks`, `theme_tracks`, `track_palettes`
(categories) and `signal_palette` (intensity).

**Transcript helpers** — `rank_transcripts` (which isoform goes on top),
`collapse_transcripts` (gene view instead of isoform view), `to_intron`.

## Several loci at once

Give `plot_tracks` a mapping instead of a single mapper and each locus becomes a
column with its own coordinate system — genes that share no genomic range still
share a figure:

```python
ggt.plot_tracks(tracks, mappers={"Actb": m1, "Myc": m2})
```

Layer data selects its column through a `locus` column; a layer without one
(a highlight, a reference line) spans them all.

## Data contract

Geoms consume genomic-coordinate DataFrames (`xstart`/`xend`, a numeric `y`, and
geom-specific columns — see each geom's docstring). For `plot_tracks`, every
layer's data carries a `track` column (the facet-row key) and a numeric `y`.
Genomic→display intron compression is applied by the shared `GenomicMapper` via
`scale_x_genomic`.

**Coordinates.** Intervals are **half-open** — `[xstart, xend)`, so length is
`xend - xstart`. The readers all emit **1-based** half-open coordinates:
annotations are natively 1-based inclusive and the BED family 0-based half-open,
and reconciling that at the file boundary is what keeps a coverage track from
sitting one base off the gene model beneath it. If you build frames yourself the
origin is yours — `GenomicMapper` only computes *relative* layout and reports
tick labels back in whatever frame you fed it. Just stay consistent within a
figure.

## License

MIT
