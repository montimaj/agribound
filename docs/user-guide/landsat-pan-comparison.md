# Landsat PAN versus surface reflectance for field delineation

Example 23 (`examples/23_landsat_pan_sr_comparison.py`)
compares three real delineation runs on a small Beauce agricultural area in
France: 15 m panchromatic imagery, 30 m surface reflectance, and experimental
15 m PAN+SR RGB image fusion. The live run on **October 1, 2026** used
`ee-rappjer`, CPU inference, 12 matched Landsat 8/9 scenes from May-August 2023,
and 329 public IGN RPG 2023 reference parcels. No unmatched scenes were found.

## Run the example

From the repository root, with Python >=3.12:

```bash
python -m pip install -e ".[gee,delineate-anything]" matplotlib
agribound auth --project YOUR_GEE_PROJECT
python examples/23_landsat_pan_sr_comparison.py --gee-project YOUR_GEE_PROJECT
```

The example downloads a small reference extract from the public
[IGN WFS](https://data.geopf.fr/wfs/ows?SERVICE=WFS&VERSION=2.0.0&REQUEST=GetCapabilities),
the paired imagery, and the pinned published `large_v2` checkpoint. It verifies
the checkpoint SHA-256. The default output folder is
`outputs/landsat_pan_sr_comparison`. Set `--output-dir PATH` to change it and
`--device cuda` to use a CUDA-enabled environment. `--prepare-only` downloads
the reference and prepares the three inputs without downloading weights or
running inference. Existing input rasters are reused only when their scene
manifest and grid agree; use a new output directory for a different area.

For this workspace, the executed command was:

```bash
python examples/23_landsat_pan_sr_comparison.py --gee-project ee-rappjer
```

After preparing inputs, an offline run is also possible:

```bash
python examples/23_landsat_pan_sr_comparison.py --offline \
  --gee-project YOUR_GEE_PROJECT --checkpoint /path/to/DelineateAnythingv2.pt
```

The project argument identifies the prepared configuration in offline mode;
no Earth Engine calls are made. Pass the released checkpoint already downloaded
by the online run. Inference gets a new cache directory each run so timings
do not measure cached polygon loading. Input preparation may be cached and is
timed separately. Model download is excluded from inference timings.

For ordinary use outside this controlled comparison, the source interface is:

```python
import agribound

fields = agribound.delineate(
    source="landsat-pan",  # use "landsat" for the normal 30 m SR stack
    year=2023,
    study_area="bbox:1.62,48.13,1.68,48.17",
    gee_project="YOUR_GEE_PROJECT",
    engine="delineate-anything",
    output_path="fields_pan.gpkg",
    lulc_filter=False,
)
```

The ordinary builders can include Landsat 7 in 2023. Example 23 uses a separate
matched-export helper to restrict both sources to Landsat 8/9 and identical
scene IDs; that stricter selection is essential for this comparison.

## What is held constant

| Setting | All three experiments |
|---|---|
| Evaluation area | 1.62-1.68 E, 48.13-48.17 N; imagery has a 600 m context buffer |
| Dates | 2023-05-01 through 2023-08-31 inclusive |
| Scenes | Same Landsat 8/9 `LANDSAT_SCENE_ID`; both products have scene cloud cover <=20% |
| Masking | QA_PIXEL bits 0-4 on both products, plus their existing data masks; each 30 m cell needs all SR bands and all four PAN subpixels |
| Composite | Per-band temporal median using the matched, jointly masked observations |
| Engine | Delineate-Anything native backend, published `large_v2`; no fine-tuning |
| Weights | Revision `369d0b4c44cf9bec2bd3a27bc81810cadd2c963e`; SHA-256 `46700b8a279b07922953a11adaeb5e658d9a2384b6334c8e0a3090886218915a` |
| Inference | Confidence 0.15, batch size 1, FP32, tile step 0.5, super-resolution setting 1, seed 42 |
| Polygon processing | Representative point inside the evaluation area; minimum 1000 m²; 2 m simplification; no smoothing or SAM |
| LULC filter | Disabled for all three |
| Evaluation | One-to-one IoU >=0.5; common 10, 15 and 30 m boundary tolerances; 1 m boundary sampling; common size bins |

`source="landsat-pan"` maps B8 to all three RGB channels. `source="landsat"`
writes the usual six-band SR stack, but this RGB engine reads only red, green
and blue. This is therefore a comparison against **SR RGB with the same
engine**, not against a separately trained NIR/SWIR model. FTW and Prithvi
require spectral bands absent from PAN, so using them would change the engine
as well as the input. GeoAI/DINOv3 require additional trained field checkpoints.

The native sampling differs by design. With the same pixel-based inference
settings, 15 m inputs use nine tiles and 30 m SR uses four in this example;
their physical context differs. Each input also gets the engine's standard
per-band 1-99 percentile stretch. These results measure the complete configured
workflows, rather than isolating spatial resolution from every other effect.
Both 15 and 30 m are outside this checkpoint's documented 0.25-10 m training
resolution range. The example is not a geographically held-out model benchmark:
we have not established whether this region occurs in its training data.

## How PAN+SR is combined

The combined experiment fuses **imagery**, before applying the same field
detector. It does not combine predictions or use reference boundaries in fusion.
It uses a conservative experimental high-pass detail-injection method from
`agribound.composites.pan_fusion`, applied to the matched temporal composites:

1. Convert SR RGB from `reflectance_x10000` to unit reflectance. Preserve PAN
   as unit TOA reflectance; do not reinterpret it as surface reflectance.
2. Average each aligned 2x2 PAN block to the 30 m grid. Fit a nonnegative
   regression slope between this low-pass PAN and SR RGB mean intensity.
   An intercept is fitted implicitly by centring both variables; it is not
   injected into the image.
3. Nearest-neighbour resample RGB to 15 m, and add the scaled within-block
   PAN residual equally to R, G and B. Each residual has zero block mean.
4. If detail would produce negative values, attenuate the whole block's detail
   consistently across channels. This preserves coarse band means instead of
   distorting them through pixelwise clipping. Invalidate incomplete blocks.

Conceptually, for each band `b`, `fused_b = upsample(SR_b) + alpha * gain *
(PAN - upsample(block_mean(PAN)))`. The gain transfers only fine spatial
contrast between different radiometric products. The result is an experimental
RGB input, **not a calibrated native 15 m SR product**. It is passed as
`source="local"` with explicit RGB indices, not advertised as a new global source.
Plain SR resampling alone adds no native spatial detail.

[USGS band specifications](https://www.usgs.gov/landsat-missions/landsat-8)
place PAN at 0.50-0.68 micrometres; NIR and SWIR are outside it, and blue only
partially overlaps. This motivates limiting injection to visible RGB and
checking colour distortion. High-pass injection belongs to the spatial-detail
fusion family described in
[Schowengerdt (1980), bibliographic record](https://agris.fao.org/search/en/providers/123819/records/64735f6408fd68d546036fa2);
this simple block method is an experimental adaptation, not a reproduction of
that paper or an MTF-matched sensor reconstruction.
The original publisher PDF URL currently returns 404 to the link checker;
the citation is *Reconstruction of multispatial, multispectral image data using
spatial frequency content*, Photogrammetric Engineering & Remote Sensing
46(10), 1325–1334.

Validation in this run found:

- Maximum absolute error after degrading fused RGB back to 30 m:
  **7.45e-9** unit reflectance. Only **0.119%** of cells needed attenuation.
- In a reduced-resolution check (SR 60 m + PAN 30 m, compared with known
  SR 30 m), red/green/blue RMSE changed from **0.0130/0.00795/0.00551** for
  resampled SR to **0.00612/0.00456/0.00568** for fusion.
- Median spectral angle worsened from **0.922° to 1.229°**. Fusion improves
  some detail while introducing colour error, particularly in blue.

This box-degradation check is not MTF-matched and cannot establish actual
15 m SR accuracy. It supports an explicit tradeoff, not a claim that all
spectral information is improved.

## Measured results

Field detection uses one-to-one IoU >=0.5; boundary metrics below use a 15 m
tolerance. CPU inference wall times include model loading and tiling, exclude
downloads and evaluation, and are single-run measurements.

| Input | Resolution | Predicted fields | Precision | Recall | Detection F1 | Matched IoU | Boundary F1 | OS mean | US mean | Inference |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PAN | 15 m | 196 | 0.561 | 0.334 | 0.419 | 0.745 | **0.621** | 0.367 | **0.596** | 20.1 s |
| SR RGB | 30 m | 124 | 0.581 | 0.219 | 0.318 | 0.670 | 0.377 | 0.489 | 0.693 | 9.3 s |
| PAN+SR RGB | 15 m | 172 | **0.610** | 0.319 | 0.419 | 0.745 | 0.599 | **0.325** | 0.624 | 17.6 s |

OS/US are Agribound's per-reference-field segmentation error measures; lower
is better. The inference order was PAN, SR, combined, so first-use warmup can
affect runtime. Shared input preparation and fusion times are recorded
separately; the final run reused prepared inputs. Do not compare its cached
acquisition timing with a fresh imagery download. An offline repeat with fresh
inference caches reproduced the same polygon counts and accuracy scores;
the table uses that final run's timings. The first live run took
25.8/9.8/16.9 s respectively, illustrating runtime variability.

| Reference size | Parcels | PAN detection F1 | SR detection F1 | Combined detection F1 |
|---|---:|---:|---:|---:|
| <1 ha | 138 | 0.000 | 0.000 | 0.000 |
| 1-5 ha | 74 | **0.308** | 0.045 | 0.252 |
| 5-20 ha | 104 | 0.734 | 0.629 | **0.749** |
| 20-100 ha | 13 | 0.686 | 0.581 | **0.706** |

PAN improves boundary agreement and detection relative to SR RGB in this
area, particularly in the 1-5 ha class. SR remains the simpler calibrated
multispectral input and the faster workflow here; its NIR/SWIR information
could help a suitable spectral model, which this comparison does not test.
Combined RGB slightly improves precision and oversegmentation over PAN,
but misses more fields and has lower boundary F1. Its detection F1 is nearly
identical to PAN, so this run does **not** demonstrate a clear overall gain
that justifies fusion's complexity.

No sub-hectare reference parcel is matched at IoU >=0.5. Some boundaries can
still be located while adjacent parcels are merged. The 0.1 ha output-area
threshold also prevents very small detections. Reference sizes range from
0.0065 to 34.45 ha (median 2.03 ha). RPG declarations can contain management
splits that are invisible in imagery and omit undeclared land. Imagery and
reference years match, but declaration date and physical boundary accuracy
are not independently verified. Three invalid reference geometries were
repaired by the evaluator. Individual-field bootstrap intervals (200 resamples)
are supplied in JSON; spatial dependence limits their interpretation. This
single area, season and checkpoint do not establish general superiority.

Three-way delineation comparison on a common SR background (`outputs/landsat_pan_sr_comparison/comparison.png`; generated locally)

All columns show the same unsharpened SR RGB background, extent, reference
outlines and stretches. Cyan is reference and magenta is prediction. Zooms
are selected from reference geometry only: the smallest parcel >=0.1 ha,
and the first shared edge >30 m. The exact bounds are saved in JSON.

## Outputs and reproducibility

For countries, climates and crop contexts outside France, see the
[international comparison](landsat-international-comparison.md): six frozen
sites in four countries, five measured three-way comparisons and one retained
unmeasured site. It includes cases where SR or fusion improves over PAN.

### Additional landscape comparisons

Example 24 (`examples/24_landsat_landscape_comparison.py`)
repeats all three experiments in three additional areas. All nine runs completed
using the same 2023 RPG reference vintage, May–August Landsat 8/9 window,
cloud threshold, QA, checkpoint and inference/postprocessing settings as Beauce.
Scene selection is identical across experiments **within each site**; different
sites naturally have different acquisitions. There were no unmatched scenes.
Each site uses its local UTM grid (Brittany/Landes: zone 30N; Alsace: zone 32N).

| Site | Reference parcels | Matched scenes | Median area (ha) | Median elongation |
| --- | ---: | ---: | ---: | ---: |
| Beauce | 329 | 12 | 2.03 | 2.73 |
| Brittany, near Saint-Caradec | 566 | 8 | 1.50 | 2.25 |
| Landes, near Ychoux | 25 | 5 | 1.37 | 2.89 |
| Alsace, west of Benfeld | 644 | 17 | 1.14 | 4.17 |

Elongation is the long/short side ratio of each reference parcel's minimum
rotated rectangle, then the median across parcels. Sites were fixed before
measuring prediction scores. Landes was initially a candidate for large fields,
but the actual study box is predominantly forest with sparse agricultural
references; its median declared parcel is not large. We retain this difficult
case rather than replace it after seeing the scores.

| Site | Detection F1: PAN / SR / combined | Boundary F1 at 15 m: PAN / SR / combined | CPU inference seconds: PAN / SR / combined |
| --- | --- | --- | --- |
| Beauce | 0.419 / 0.318 / 0.419 | 0.621 / 0.377 / 0.599 | 20.1 / 9.3 / 17.6 |
| Brittany | 0.253 / 0.092 / 0.245 | 0.479 / 0.190 / 0.442 | 25.2 / 7.1 / 19.3 |
| Landes | 0.035 / 0.054 / 0.039 | 0.102 / 0.073 / 0.098 | 21.5 / 8.1 / 16.3 |
| Alsace | 0.241 / 0.063 / 0.177 | 0.529 / 0.239 / 0.451 | 22.1 / 6.9 / 15.9 |

Detection uses the same one-to-one IoU >=0.5 matching. These are measured
single-run timings, not timing confidence intervals. PAN has the highest
boundary F1 in all four boxes, while SR has the highest detection F1 in Landes.
There, PAN produces 257 polygons against only 25 reference parcels, with
precision 0.019; the figure shows many predictions in forest. With LULC filtering
disabled equally, extra spatial detail can increase unwanted delineations.
RPG is not exhaustive physical-field truth, so unmatched predictions are not
all proven non-fields. Fusion does not improve boundary F1 over PAN in any box,
and these measurements do not justify its extra complexity for this checkpoint.
All four sites remain in France and share one reference system; this is not
evidence of general performance across countries or engines.

Run the suite after installing/authenticating as above:

```bash
python examples/24_landsat_landscape_comparison.py --gee-project YOUR_GEE_PROJECT
python examples/24_landsat_landscape_comparison.py --summarize-only
```

Use `--sites brittany alsace` to choose a subset, `--checkpoint PATH` to reuse
the same pinned local weights, and `--baseline PATH` to include a completed
example 23 run. Every site directory in `outputs/landsat_landscapes/` contains
the full GeoPackages, provenance, scene manifest, size-stratified metrics and
common-background maps described below. Top-level `landscape_comparison.csv`
retains all size classes and 10/15/30 m tolerances; `headline_15m.csv` is the
all-size slice, and `suite_status.json` records scene counts and parcel profiles.

Measured comparison across four landscapes (`outputs/landsat_landscapes/landscape_comparison.png`; generated locally)

Recorded full results (`outputs/landsat_landscapes/landscape_comparison.csv`; generated locally)
and site metadata (`outputs/landsat_landscapes/suite_status.json`; generated locally)
accompany the Brittany (`outputs/landsat_landscapes/brittany.png`; generated locally),
Landes (`outputs/landsat_landscapes/landes.png`; generated locally)
and Alsace (`outputs/landsat_landscapes/alsace.png`; generated locally)
maps, each with identical extents and SR background across its three experiments.

### Per-site artifacts

The output directory contains:

- `inputs/landsat-pan.tif`, `inputs/landsat.tif`, `inputs/pan_sr_rgb.tif`, and
  `inputs/scene_manifest.json` with paired acquisition and product IDs,
  sensor IDs, unmatched scenes, masks and radiometry.
- `reference.gpkg`, `reference_evaluation.gpkg` and reference metadata with
  request URLs, vintage, attribution and a feature-content checksum.
- `fields_pan.gpkg`, `fields_sr.gpkg`, `fields_combined.gpkg` and each one's
  `.gpkg.provenance.json`: configuration, actual weights checksum, package
  versions, engine settings, timings, warnings and source manifest.
- `comparison.csv`, metrics JSON at 10/15/30 m, size classes and bootstrap
  intervals, and `per_field_*.csv` with matches and boundary errors.
- `comparison.png`, `comparison.pdf`, `figure_windows.json`,
  `fusion_validation.json`, `preparation_timing.json` and `run_status.json`.

Recorded copies of the
comparison CSV (`outputs/landsat_pan_sr_comparison/comparison.csv`; generated locally),
scene manifest (`outputs/landsat_pan_sr_comparison/scene_manifest.json`; generated locally) and
fusion validation (`outputs/landsat_pan_sr_comparison/fusion_validation.json`; generated locally)
accompany this guide. The large weights and raw output data stay in the
ignored output directory. The example is a prepared-input workflow using
Agribound's engine, selection, polygon utilities, evaluator and recorder;
it bypasses automatic composites equally for all three experiments.

Offline tests cover real detail injection, coarse spectral mean preservation,
nodata, negative-value prevention, zero-strength resampling, reduced-resolution
validation, scene pairing and duplicate rejection:

```bash
python -m pytest tests/unit/test_pan_fusion.py tests/unit/test_landsat_comparison.py -q
```

Public offline demonstration maps and source attributions are in the [tutorial guide](landsat-tutorials.md). Full research imagery, vectors and large/restricted overlay maps remain local and are not part of the proposed PR.
