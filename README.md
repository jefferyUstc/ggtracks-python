<img src="assets/logo.svg" align="right" width="128" height="128" alt="ggtracks logo"/>

# ggtracks

**Grammar-of-graphics genomic tracks for the [ggplot2-python](https://github.com/Bio-Babel/ggplot2-python) ecosystem**

`ggtracks` gives you real ggplot2-python `Geom`/`Stat` citizens for genomic tracks
(exon/feature ranges, introns with strand chevrons, splice-junction arcs, read
pileups), a **genomic coordinate model with intron compression**
(`GenomicMapper`), and a track-stacking composer (`plot_tracks`) that aligns any
number of tracks on one shared, intron-compressed genomic x-axis.


## Install

```bash
pip install ggtracks-python
```

## Quick start

```python
import pandas as pd
import ggplot2_py as gg
import ggtracks as ggt

# 1. exon features in genomic coordinates. xstart/xend are half-open
#    [xstart, xend) — exon 1 spans bases 1000-1099 (100 bp) — plus a
#    numeric y, a grouping column, and a fill category.
exons = pd.DataFrame({
    "xstart":  [1000, 3000, 5000],
    "xend":    [1100, 3100, 5100],
    "y":       [1.0, 1.0, 1.0],
    "feature": ["exon", "exon", "exon"],
    "track":   "MyGene",
})
introns = ggt.to_intron(exons, group_var="track")   # gaps between exons

# 2. a coordinate model that compresses those introns at render time.
mapper = ggt.GenomicMapper.from_intervals(
    list(zip(exons["xstart"], exons["xend"])),
    intron_mode="clamp", target_gap_width=100,
)

# 3. compose a track from real ggplot2 geoms and stack it.
track = ggt.Track("MyGene", [
    ggt.geom_range(gg.aes(xstart="xstart", xend="xend", y="y", fill="feature"), data=exons),
    ggt.geom_intron(gg.aes(xstart="xstart", xend="xend", y="y"), data=introns),
])
ggt.plot_tracks([track], mapper, title="demo")
```

## Data contract

Geoms consume genomic-coordinate DataFrames (`xstart`/`xend`, a numeric `y`, and
geom-specific columns — see each geom's docstring). For `plot_tracks`, every
layer's data carries a `track` column (the facet-row key) and a numeric `y`.
Genomic→display intron compression is applied by the shared `GenomicMapper` via
`scale_x_genomic`.

**Coordinates.** Intervals are **half-open** — `[xstart, xend)`, so length is
`xend - xstart`. The origin is yours: 0- or 1-based both work, because
`GenomicMapper` only computes *relative* layout and reports tick labels back in
whatever frame you fed it. Just stay consistent within a figure.

## License

MIT
