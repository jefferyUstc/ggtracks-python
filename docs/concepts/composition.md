# Composing figures

## The data contract

Every geom consumes a plain DataFrame in genomic coordinates:

- position aesthetics are **`xstart` / `xend`** (half-open — see
  [One coordinate dialect](coordinates.md)) plus a **numeric `y`**
  (transcript rank, depth, arc height). `plot_tracks` shares one y
  aesthetic across free-y facets, so a string y breaks type consistency;
- for `plot_tracks`, each layer's data carries a **`track`** column naming
  its panel, and — in a multi-locus figure — a **`locus`** column;
- a layer that *omits* the facet column is repeated into every panel.
  That is what `geom_highlight` exploits deliberately, and what
  `plot_tracks` refuses inside a `Track` (pass such layers as
  `background=` instead).

## Tracks and `plot_tracks`

`plot_tracks(tracks, mapper)` stacks tracks as facet rows over one shared
intron-compressed axis, with per-track panel heights and (via
`new_scale=`) an independent fill/colour scale per track.

```python
tracks = [
    ggt.Track("coverage", [ggt.geom_coverage(...)],
              y_limits=ggt.signal_limits(values), range_label=True),
    ggt.Track("gene model", ggt.gene_model_layers(ann, track="gene model"),
              height=0.4, y_breaks=[1.0], y_labels=[""]),
]
p = ggt.plot_tracks(tracks, mapper, title="Actb")
```

Things worth knowing, all stated in the API docs but easy to miss:

- **`height`** as a plain number is a *relative* share (that also counts
  as one inch of figure height); a `grid_py.Unit` pins the panel to an
  absolute size.
- **Free y is right for a browser and wrong for a comparison.** Two tracks
  each scaled to their own maximum look equally tall whatever the ratio.
  Give them the same `y_limits`, computed by `signal_limits` over the
  pooled values (it clips at a quantile so one spike cannot flatten the
  rest). `range_label=True` draws the `[0-1479]` browser badge.
- **Guardrails raise rather than silently dropping rows**: a layer with no
  `track` column, a `track`/`locus` value matching no panel, empty
  `y_breaks`, inverted `y_limits`. A track with no data warns and is
  omitted.
- `plot_tracks` **copies** each layer before rewriting its facet column,
  so one `Track` list can safely build several figures.

### Several loci at once

Pass a mapping instead of a single mapper and each locus becomes a facet
*column* with its own coordinate system:

```python
ggt.plot_tracks(tracks, mappers={"Actb": m1, "Myc": m2}, n_breaks=3)
```

Layer data selects its column through a `locus` column; a layer without
one (a highlight, a reference line) spans them all.

## Sizing: measured, not assumed

A figure's chrome — axes, title, facet strips, margins — is not a
constant; it grows with `base_size`. `natural_height` *measures* it from
the plot's own gtable, and `plot_tracks` sets the figure height to
measured chrome + the panel allowance. `finalize_gg` pins the display
size to the same inches used for `save=`, so the notebook figure and the
saved file are dimensionally identical.

There is deliberately no `show=` anywhere: a `ggplot` is a value that
renders once as a cell result, not a handle into a global figure registry.

## Stacking whole figures: `vstack_gg`

`plot_tracks` composes rows that share one x scale. Some figures need
rows that deliberately do **not** share it — and grid facets cannot
express that, because x scales run per *column*. `vstack_gg` stacks whole
plots instead (via patchwork-python), aligning panel edges across rows;
each plot defaults to the height it was measured at, so stacking never
squashes a panel.

### Recipe: focus + context zoom

The same `Track` list rendered twice — once whole, once clipped — plus a
`geom_zoom_link` connector, in **one** saved artifact:

```python
lo, hi = 3800, 4600
link = ggt.Track("zoom",
                 [ggt.geom_zoom_link(xstart=lo, xend=hi, track="zoom")],
                 height=0.3, y_breaks=[0.0], y_labels=[""])

overview = ggt.plot_tracks(tracks + [link], mapper)
detail = ggt.plot_tracks(tracks, mapper, genomic_xlim=(lo, hi))
ggt.vstack_gg([overview, detail], save="zoom.png")
```

### Recipe: chromosome context above a browser figure

A whole-chromosome ideogram strip (its x is the chromosome — a different
domain from the gene panels) stacked over the locus figure:

```python
bands = ggt.read_cytoband("mm10_cytoband.txt.gz", chrom="chr7")
ctx = (gg.ggplot()
       + ggt.geom_ideogram(gg.aes(xstart="xstart", xend="xend", y="y",
                                  stain="stain", fill="stain"),
                           data=bands.assign(y=1.0))
       + ggt.geom_highlight(xstart=locus.start, xend=locus.end)
       + ggt.scale_fill_giemsa() + ggt.theme_tracks()
       + gg.labs(x="", y=""))

main = ggt.plot_tracks(tracks, locus.mapper)
ggt.vstack_gg([ctx, main], heights=[0.6, main.fig_height])
```

!!! note "Hand-built plots need an explicit height"
    `plot_tracks` output carries a *measured* `fig_height`; a hand-built
    ggplot like the ideogram strip above falls back to the ggplot default
    (5 in), so pass `heights=` for such rows, as the recipe does.

## Locus preparation

`Locus.from_features` bundles the three things every locus figure starts
with — the features, the exonic-union mapper, and the genomic range a
signal query needs:

```python
locus = ggt.Locus.from_features(ann, flank=300,
                                intron_mode="clamp", target_gap_width=150)
cov = ggt.read_bigwig("signal.bw", *locus.region, bins=400)
```

Flanks enter the mapper as *uncompressed* spans: they are genomic context,
not introns. `gene_model_layers` is the companion — the standard collapsed
gene model (exon boxes, taller CDS boxes, introns with strand arrows) as a
ready-made layer list for a `Track`.
