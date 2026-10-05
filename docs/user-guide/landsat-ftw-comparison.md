# Landsat inputs versus published Fields of The World

Example 28 is a separate, frozen comparison of eight Landsat inputs and published
FTW Global predictions. It preserves examples 23–27 and their outputs. The suite
retains all 16 previous locations, including blocked Red River, and adds six AOIs
in France, the Netherlands, Canada and Vietnam. All results stay under
`outputs/landsat_ftw_comparison/`.

Three polygon types have different roles:

- **Published FTW Global**: Sentinel-2/PRUE model predictions. Direct comparison
  with Agribound measures agreement, correspondence and disagreement.
- **FTW benchmark labels**: source-derived annotations, declarations or survey
  units. Their source licenses, coverage and split membership govern evaluation.
- **Other provider references**: suitable source geometries with their own
  limitations. An independent provider does not establish independence from
  either model's training corpus. No spatially independent holdout is verified
  for this suite.

## Install, authenticate and execute

Use Python 3.12+ at the repository root. This Windows environment was tested
with the exact versions in `examples/landsat_comparison_constraints.txt`.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -c examples/landsat_comparison_constraints.txt -e ".[gee,delineate-anything]" matplotlib pytest
.venv\Scripts\agribound.exe auth --project ee-rappjer
```

Earth Engine requires an enabled account with permission to use `ee-rappjer`.
Public FTW and reference queries require no account. Cached pinned weights are
required for new Landsat inference. Supply `--checkpoint PATH` if the existing
model cache is unavailable; the runner verifies SHA-256, without downloading an
untracked checkpoint. Existing historical inference is copied only after input,
output, channel-order, checkpoint and inference-setting verification.

Checkpoint revision: `369d0b4c44cf9bec2bd3a27bc81810cadd2c963e`.
SHA-256: `46700b8a279b07922953a11adaeb5e658d9a2384b6334c8e0a3090886218915a`.

Clear incompatible GIS paths inherited from another Windows GIS installation
before using the geospatial wheels:

```powershell
Remove-Item Env:PROJ_LIB,Env:PROJ_DATA,Env:GDAL_DATA -ErrorAction SilentlyContinue

# Prepare references, frozen reference-only zooms and paired new-site imagery:
.venv\Scripts\python.exe examples/28_landsat_ftw_comparison.py --prepare --skip-figures
# Query once, retain immutable raw FTW snapshots and derive sensitivities locally:
.venv\Scripts\python.exe examples/28_landsat_ftw_comparison.py --query-ftw --query-workers 2 --skip-figures
# Audit published benchmark chip splits without downloading imagery chips:
.venv\Scripts\python.exe examples/landsat_ftw_training_audit.py
# Run/resume A-H; compatible historical inference is reused:
.venv\Scripts\python.exe examples/28_landsat_ftw_comparison.py --run-landsat --offline --skip-figures
# Score vectors without loading weights or repeating inference:
.venv\Scripts\python.exe examples/28_landsat_ftw_comparison.py --evaluate-only --offline --evaluation-workers 4 --skip-figures
# Regenerate maps and quantitative figures without inference or network:
.venv\Scripts\python.exe examples/28_landsat_ftw_comparison.py --figures-only --offline
# Update plates/quantitative plots/input illustrations using existing site maps:
.venv\Scripts\python.exe examples/28_landsat_ftw_comparison.py --figures-only --skip-site-maps --offline
# Rebuild inventories, summaries, report and artifact manifest:
.venv\Scripts\python.exe examples/28_landsat_ftw_comparison.py --summarize-only --offline --skip-figures
```

The default command performs all stages. Select sites/methods with, for example:

```powershell
.venv\Scripts\python.exe examples/28_landsat_ftw_comparison.py --sites nl_flevoland_2025 qc_yamaska_2024 --methods pan_nir_red coarse_pan_nir_red hybrid --offline
```

Prepare examples 23–27 first when historical reference/input/output caches are
absent. Missing historical data does not prevent FTW retrieval. New sites can
run from the public adapters. A new output directory must remain within the
repository. Never edit a frozen run's site/year configuration or refresh a
mutable remote FTW source inside that run; use a separately versioned run for
changed inputs. Product and evaluation failures are independently recorded as
unmeasured. A complete supported query yielding zero polygons is a measured
empty prediction. An unsupported year, missing partition, truncated query or
failed download is not a zero result.

## Frozen sites and sources

Selection precedes new prediction scores and agreement. Bounds, dates, source
years, reference inclusion, crop evidence and representative selection are in
`examples/landsat_ftw_comparison_sites.json`. Reference-only 900 m small-field
and shared-edge windows are saved before new inference. When exact shared edges
are absent, the existing selector uses a median-area reference and discloses it.
Reference-free views use a fixed AOI grid. Sites are never replaced after scores.

| New site | Landsat / FTW year | Reference | Landscape context and limitation |
| --- | --- | --- | --- |
| Camargue, France | 2024 | RPG 2024 crop declarations | Mediterranean rice/annual-crop context; actual crop codes retained |
| Fronsac/Bordeaux, France | 2024 | RPG 2024 crop declarations | Oceanic vineyard/perennial context; actual crop codes retained |
| Flevoland, Netherlands | 2025 | BRP 2025 definitive declarations | Maritime commercial arable fields; provider crop names retained |
| Bollenstreek, Netherlands | 2025 | BRP 2025 definitive declarations | Maritime horticultural/flower-bulb context; spring acquisition window |
| Yamaska, Quebec, Canada | 2024 | BDPPAD 2024 active production units | Temperate maize/soy/hay context; declaration units and uncertain physical geometry date |
| Adjacent Mekong patch, Vietnam | 2024 | Digitized 2021 physical fields | Tropical monsoonal rice region; species unknown and explicit reference-year mismatch |

The live Dutch layer was verified as **2025**, rather than assuming that an
older 2024 announcement described the current service. Reference crop codes
and names verify declarations inside the AOI; broad regional descriptions do
not establish the crop inside every zoom. Unknown crops remain unknown.
No crop species, irrigation or growth stage is inferred from appearance.
New 2024 tropical or arid independent physical-field surveys were not found.
Existing arid irrigated and Southern Hemisphere sites remain represented; this
gap is not filled with unsuitable cadastral or irrigation-district polygons.

Reference inventory and training/temporal audit CSVs preserve original source,
year, license, direct links, geometry vintage, coverage restrictions and unknown
parcel-level training overlap. Reused international benchmark-converted
geometry and the adjacent Vietnam survey are not new independent datasets.

The independently cached v1 benchmark `chips_*.parquet` files are also audited.
Historical Netherlands, Spain and Vietnam sites intersect train-chip footprints;
South Africa intersects train/validation footprints under its restrictive source
license. The new French and Dutch AOIs do not intersect the published v1 chips.
These are chip-footprint findings, not proof of exact parcel membership in the
published FTW or Delineate Anything training corpus. The United States and Canada
are unlisted in the v1 country inventory, which also does not prove independence.
These categories remain visible in aggregate tables. HTTP chip URLs returned 403;
the documented anonymous S3 endpoint worked. Both attempts and remote size/mtime
are preserved. Recheck the frozen snapshots offline with:

```powershell
.venv\Scripts\python.exe examples/landsat_ftw_training_audit.py --offline
```

Authoritative source documentation:

- [FTW Global product and current metadata](https://source.coop/ftw/global-data):
  nominal 2024/2025, native 10 m Sentinel-2 inputs, CC-BY-4.0, predicted connected
  field-interior units rather than legal parcels.
- [Global FTW map paper](https://arxiv.org/html/2605.11055) and
  [PRUE paper](https://arxiv.org/html/2603.27101): training and calendar-dependent
  planting/harvest composites; polygon-level acquisition dates are not supplied
  in the queried vectors.
- [FTW source dataset inventory](https://github.com/fieldsoftheworld/ftw-datasets-list)
  and [benchmark baseline code](https://github.com/fieldsoftheworld/ftw-baselines):
  source/split information does not prove holdout independence from another model.
- [Delineate Anything v2 paper](https://arxiv.org/html/2607.19069v1): RGB-trained
  FBIS-73M model. False-color and stacked inputs remain input experiments.
- [France RPG metadata](https://www.data.gouv.fr/datasets/rpg) and
  [IGN product description](https://geoservices.ign.fr/sites/default/files/2025-11/DC_DL_RPG_3-0.pdf):
  annual declarations, Licence Ouverte 2.0. Administrative crop splits may not
  be visible physical edges.
- [PDOK BRP documentation](https://www.pdok.nl/introductie/-/article/gewaspercelen-inspire-geharmoniseerd-):
  annual crop declarations, CC0; live year, category and crop attributes verified.
- [FADQ BDPPAD downloads](https://www.fadq.qc.ca/documents/donnees/base-de-donnees-des-parcelles-et-productions-agricoles-declarees/)
  and [provider guide](https://www.fadq.qc.ca/fileadmin/geo4tb/BDPPAD/bdppad-v03-guide-utilisateur.pdf):
  CC-BY-4.0; declaration updates do not guarantee contemporaneous physical edges.
- [Vietnam survey](https://doi.org/10.17026/dans-xy6-ngg6): digitized 2021 fields,
  CC-BY-4.0. Climate and regional rice evidence remain separately linked in the
  frozen configuration.

## Inputs and experimental controls

The implementation shares example 27's six-band SR/PAN observation support,
QA masks, paired Landsat 8/9 scene IDs, median compositing, 20% cloud threshold,
normalization and visible-band fusion. Unmatched observations and support checks
are recorded. `landsat_multispectral.py`, `pan_fusion.py`, the existing reference
adapters/evaluator and map rendering helpers are reused.

| Method | Logical model channels | Native information / inference grid |
| --- | --- | --- |
| A PAN | B8 TOA / B8 TOA / B8 TOA | 15 m / 15 m |
| B SR RGB | B4 / B3 / B2 surface reflectance | 30 m / 30 m |
| C fused RGB | fused Red / Green / Blue | Visible PAN detail / 15 m |
| D SR false color | B5 NIR / B4 Red / B3 Green | 30 m / 30 m |
| E nearest false color | nearest NIR / Red / Green | 30 m / 15 m grid |
| F direct stack | native PAN / nearest NIR / nearest Red | Mixed 15/30 m / 15 m grid |
| G coarse-PAN control | aligned 2×2 valid PAN means, nearest upsample / NIR / Red | 30 m / 15 m grid |
| H hybrid | nearest NIR / existing fused Red / existing fused Green | Unsharpened 30 m NIR / 15 m grid |

The native engine takes BGR internally and restores logical RGB for the model;
channel order is verified in provenance. PAN TOA and SR surface reflectance
remain distinct. Each channel uses recorded percentile normalization. Resampling
creates no native spatial detail. Landsat PAN does not overlap NIR spectrally;
H leaves NIR unchanged and reuses established visible fusion. Reduced-resolution
fusion validation is retained, with integrity checks rather than invented
fusion diagnostics for stacking/resampling.

CPU FP32, seed 42, batch 1, confidence 0.15, tile step 0.5 and 512-pixel model
tiles match the established suite. A 512-pixel tile spans 7.68 km on a 15 m grid
and 15.36 km on a 30 m grid. E controls grid/context changes without fully
isolating those effects. Postprocessing retains the 1000 m² minimum, 2 m
simplification, and no SAM, smoothing or LULC filter.

FTW queries use explicit supported years, a 600 m buffered query, `clip=False`,
deduplication, and no `max_features` truncation. Full intersecting polygons,
identifiers and null confidence are preserved in the raw Parquet snapshot.
Unfiltered FTW is primary. Offline sensitivities are confidence ≥69 keeping
nulls, confidence ≥69 excluding nulls, and area ≥1000 m². Raw geometry is never
simplified. Current provider advice cautions against hard confidence cuts in
smallholder landscapes; 69 is a predeclared sensitivity, not a calibrated score
equivalent to the Landsat confidence. Null rates and excluded counts are explicit.

## Evaluation and agreement

Reference evaluation uses common metre tolerances 10/15/30, one-to-one IoU ≥0.5
matching, overlap, existing area fragmentation/leakage measures, and size bins
0–1, 1–5, 5–20, 20–100 and ≥100 ha. Headline measures are boundary F1 at 15 m
and detection F1. Crop strata use provider labels only, with counts, an unknown
category and a disclosed sparse-label group below five references.

Whole polygons whose representative points fall inside the frozen AOI are
included. This final selection uses EPSG:4326 for every delivered product,
after postprocessing. Historical inference vectors remain byte-identical;
separate `fields_*_aoi.gpkg` files record exclusions before rescoring. The final
edge audit found 30 Landsat polygons outside this rule across 168 exports; they
are retained in the original files and excluded from both comparison tracks.
The primary prediction-independent mask consists of known reference
footprints, except the existing full Vietnamese survey AOIs. Unknown land is
not negative truth. Precision is conditional on mapped footprints. Original
boundary lines are measured inside the mask; edges created by polygon clipping
are never scored. Invalid working geometry repairs are counted and raw vectors
are preserved. Whole-AOI descriptive metrics remain separately labelled.

Agreement CSVs contain no accuracy claims. Landsat is the **left** product and
FTW the peer anchor: left correspondence fraction divides matched pairs by
Landsat count; FTW correspondence fraction divides by FTW count. Boundary
agreement averages forward/reverse boundary F1; directional near-boundary
fractions and matched IoU are also retained. Split/merge adjacency counts
require intersection at least 10% of the smaller polygon, separately from
one-to-one matching. Count and area differences are left minus FTW.

Historical cohorts, same-nominal-year declarations with uncertain geometry,
and Vietnam reference-year mismatch stay separate in category aggregates.
Equal-site and field-count-weighted means average site scores; neither is
pooled precision/recall. Descriptive paired-site bootstrap intervals do not
remove spatial dependence or selection bias. Crop/climate associations are
not causal. Delivered products differ in sensors, seasons, models, training and
postprocessing, so this is a comparison of systems rather than an isolated
sensor experiment. FTW retrieval time is never reported as inference time.

## Outputs and verification

`REPORT.md` contains measured per-site results and tradeoffs. Separate
`reference_accuracy.csv` and `prediction_agreement.csv` tables prevent the two
tracks being confused. Additional outputs include confidence diagnostics,
size/crop strata, category aggregates, paired contrasts, reference/training
inventories, runtime stages, immutable snapshot/scene manifests, GeoPackages,
and a SHA-256 artifact manifest. Maps use identical extents and one common
unsharpened background per site; missing imagery is disclosed on neutral
backgrounds. Six-column representative maps use the frozen eight-site selection;
controls are shown on companion pages. PDF, 300 dpi PNG and SVG are generated.
Restricted vectors and reference-overlay maps remain local.

`aoi_edge_audit.csv` reports retained polygons crossing the AOI edge and FTW
intersecting polygons excluded by the frozen representative-point rule. Original
preselection exclusion counts for historical Landsat/reference exports are
unknown where their preselection vectors were not retained; these are missing
values, never asserted zeros. The split audit supplements the original frozen
source metadata in `training_temporal_audit.csv` without changing site selection.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_landsat_ftw_comparison.py tests/unit/test_ftw_query.py tests/unit/test_boundary_coverage.py tests/unit/test_landsat_multispectral.py tests/unit/test_pan_fusion.py -m "not network" -q
```

Offline tests cover years, confidence/nulls, duplicates, complete polygons,
CRS/area, AOI edges, coverage, empty predictions, frozen snapshots, reuse
validation and accuracy/agreement separation. The existing Landsat fusion and
preprocessing tests remain the shared scientific checks.
