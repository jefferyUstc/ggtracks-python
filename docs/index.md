# ggtracks

**Grammar-of-graphics genomic tracks for the
[ggplot2-python](https://github.com/Bio-Babel/ggplot2-python) ecosystem.**

`ggtracks` gives you real ggplot2-python `Geom`/`Stat` citizens for genomic
tracks — exon/feature ranges, introns with strand arrows, splice-junction
arcs, read pileups, coverage signal, chromosome ideograms — plus readers for
the file formats those tracks come in, a genomic coordinate model with
intron compression (`GenomicMapper`), and composers that align any number of
tracks: `plot_tracks` stacks tracks over one shared axis, `vstack_gg` stacks
whole figures whose axes deliberately differ (overview + zoom, chromosome
context + locus).

Everything is a normal `ggplot`, so `+ scale_fill_manual(...)`, `+ labs(...)`
and `+ theme(...)` keep working. The package has **no coupling to any data
container** — no AnnData, MuData or friends; inputs are plain `pandas`
DataFrames, and the readers parse *file formats*.

## Install

```bash
pip install ggtracks-python
```

For working on the docs:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Quick start

From an annotation file and a signal file — no hand-built frames:

```python
import ggplot2_py as gg
import ggtracks as ggt

# 1. gene model straight out of a GTF (or GFF3; .gz is fine)
ann = ggt.read_annotations("genes.gtf.gz", genes=["Actb"])

# 2. mapper + genomic range for the locus, in one step
locus = ggt.Locus.from_features(ann, flank=300,
                                intron_mode="clamp", target_gap_width=150)

# 3. coverage over the same region, binned off the file's own zoom levels
cov = ggt.read_bigwig("signal.bw", *locus.region, bins=400)

# 4. stack them on one shared, intron-compressed genomic axis
ggt.plot_tracks(
    [
        ggt.Track("coverage", [
            ggt.geom_coverage(gg.aes(xstart="xstart", xend="xend", y="value"),
                              data=cov.assign(track="coverage")),
        ]),
        ggt.Track("gene model", ggt.gene_model_layers(ann, track="gene model"),
                  height=0.4, y_breaks=[1.0], y_labels=[""]),
    ],
    locus.mapper,
    title="Actb",
)
```

## What's in the box

| Piece | Names |
|---|---|
| Readers ([reference](reference/io.md)) | `read_annotations`, `read_bed` (BED / narrowPeak / broadPeak), `BigWig` / `read_bigwig` (pure Python, zoom-level aware), `BedGraph` / `read_bedgraph`, `read_cytoband`, `resolve_chrom` |
| Coordinate model ([concepts](concepts/mapper.md)) | `GenomicMapper`, `scale_x_genomic`, `coord_genomic`, `signal_limits` |
| Geoms & stats ([reference](reference/geoms.md)) | `geom_range`, `geom_intron`, `geom_junction`, `geom_coverage` + `StatBinCoverage`, `geom_ideogram` + `scale_fill_giemsa`, `geom_highlight`, `geom_zoom_link`, `StatPileup` / `pack_rows` |
| Composition ([concepts](concepts/composition.md)) | `Track` + `plot_tracks`, `vstack_gg`, `finalize_gg` / `natural_height` |
| Helpers | `Locus`, `gene_model_layers`, `rank_transcripts`, `collapse_transcripts`, `to_intron` |
| Style | `theme_tracks`, `track_palettes` (categories), `signal_palette` (intensity) |

## Where to go

- **[One coordinate dialect](concepts/coordinates.md)** — the 1-based
  half-open contract, and why translation happens at the file boundary and
  nowhere else. Read this before building any frame by hand.
- **[The coordinate model](concepts/mapper.md)** — how intron compression
  works as a render-time scale transform, and the exonic-union idiom.
- **[Composing figures](concepts/composition.md)** — the `Track` data
  contract, stacking, multi-locus figures, and the zoom / ideogram-context
  recipes built on `vstack_gg`.
- **[API reference](reference/io.md)** — every public export, grouped by
  role.

Three executed notebooks (quickstart, every geom, composition) live in the
companion `ggtracks-python-tutorials` repository, worked on real m6A RIP-seq
data.

## Design rules

1. **No data container.** A test walks the source with `ast` and fails on
   any AnnData / MuData / scanpy import — the readers exist so this stays
   liveable.
2. **One coordinate dialect.** Every frame is 1-based half-open; foreign
   conventions are translated in `ggtracks.io` and nowhere else.
3. **Fail loud.** Mistyped facet keys, holes in a mapper tiling, unknown
   palettes and stains, truncated bigWig blocks — all raise with a message
   that says what to do, rather than degrading into an empty or subtly
   wrong figure.

## License

MIT
