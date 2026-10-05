# Licensed offline Landsat tutorial snapshot

This 3.4 MB teaching bundle contains real measured outputs, not synthetic
predictions. `manifest.json` pins every data file and records its original local
research artifact, transformations and license. Examples 29/30 verify hashes
before using cached products. They prepare inputs and reevaluate existing
polygons offline; model inference requires explicit live mode and pinned weights.

Camargue bounds: 4.55, 43.55, 4.59, 43.58 (EPSG:4326). Landsat 8/9 window:
2024-05-01 through exclusive 2024-09-30; scene cloud threshold 20%. The native
PAN, six-band SR and established fused visible input are byte-identical copies.
`scene_manifest.json` records paired identifiers, unmatched observations, QA,
common support and the exact nested grids. Export imagery uses the original
600 m metric buffer around the AOI; final polygon inclusion uses the AOI itself.
SR files use reflectance ×10000;
PAN uses unit TOA reflectance. The tutorial converts SR to unit surface
reflectance before constructing D–H. Fusion is an experimental derived product.

Attribution and redistribution:

- Landsat imagery: USGS/NASA, Collection 2 Landsat 8/9; public domain, with
  source acknowledgment requested. [USGS redistribution policy](https://www.usgs.gov/faqs/are-there-any-restrictions-use-or-redistribution-landsat-data).
- Camargue reference declarations: IGN / ASP, RPG 2024, Licence Ouverte 2.0
  (Etalab-2.0). [Provider catalogue](https://www.data.gouv.fr/datasets/rpg).
  The original conservative research status is retained in the inventory;
  this audited public RPG subset and its overlay maps may be redistributed
  with this attribution. They describe declared crop units, not exhaustive
  independent physical-field truth. No numerical positional error is known.
- Published FTW Global predictions: Fields of The World / Taylor Geospatial
  Engine, nominal 2024 Sentinel-2 predictions, CC-BY-4.0.
  [Product and license](https://source.coop/ftw/global-data).
  `ftw_raw.parquet` preserves all 131 buffered polygons, identities and confidence.
  The tutorial makes a separate bounding-box-indexed working copy; it does not
  clip or modify provider geometry. Confidence is a sampled product score, not
  an independently calibrated probability. Unknown confidence is not low confidence.
- Agribound derived predictions/statistics/coverage and fused input: repository
  Apache-2.0 license, with underlying data attributions above retained. Model
  weights are not distributed. Checkpoint `large_v2`, revision
  `369d0b4c44cf9bec2bd3a27bc81810cadd2c963e`, SHA-256
  `46700b8a279b07922953a11adaeb5e658d9a2384b6334c8e0a3090886218915a`.

Six summary locations were fixed as all six additional sites in the existing
FTW configuration, before tutorial figures were created. Their inventory retains
climate/crop evidence, years, reuse, coverage and training uncertainty. This
selection includes every outcome, rather than selecting sites with good scores.
France and Netherlands use declaration parcels; Québec uses FADQ active crop
units; Vietnamese annotations are 2021 while imagery/FTW are 2024. All-site
research results remain in the full suite. These summaries contain statistics,
not international reference vectors. No restricted South African, Montana or
Washington vectors or overlay maps are included.

`model_provenance.json` retains original per-method channel selection, BGR order,
percentile bounds, checkpoint and engine settings with local paths/hostnames
omitted. These are original measured-run records, not new inference timings.

Measured input pipeline: native CPU FP32, seed 42, `large_v2`, confidence 0.15,
batch 1, super-resolution 1, model tiles 512 pixels with step 0.5; minimum area
1000 m², simplification 2 m, no SAM/smoothing/LULC. Tile footprints are 7.68 km
on 15 m grids versus 15.36 km on 30 m grids. Native preprocessing selects the
logical RGB triplet, reverses it to BGR for the NumPy Ultralytics interface,
then applies scene-wide per-channel percentile normalization. Existing research
provenance records the bounds; no calibrated common PAN/SR radiometry is claimed.

Scores are conditional on known reference coverage. Final AOI representative
points are selected after postprocessing without clipping polygons; boundary
evaluation uses original lines inside the coverage mask. No model-training
independence is established. Agreement CSVs compare peer predictions and must
never be labelled reference accuracy. Nominal-year matches do not establish the
physical vintage of declaration edges.
