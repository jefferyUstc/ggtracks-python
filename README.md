<img src="assets/logo.svg" align="right" width="128" height="128" alt="ggtracks logo"/>

# ggtracks

**Grammar-of-Graphics genomic tracks for the [ggplot2-python](https://github.com/Bio-Babel/ggplot2-python) ecosystem**

`ggtracks` gives you real ggplot2-python `Geom`/`Stat` citizens for genomic tracks
(exon/feature ranges, introns with strand chevrons, splice-junction arcs, read
pileups, coverage signal, chromosome ideograms), readers for the file formats
those tracks come in, a **genomic coordinate model with intron compression**
(`GenomicMapper`), and a track-stacking composer (`plot_tracks`) that aligns any
number of tracks — over one locus or several side by side.

Read the [Documentation](https://ggtracks-python.readthedocs.io/en/latest/) for Concepts & APIs.
Read the [Tutorials](https://github.com/jefferyUstc/ggtracks-python-tutorials) for demo notebooks.

## Install

```bash
pip install ggtracks-python
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

## License

MIT
