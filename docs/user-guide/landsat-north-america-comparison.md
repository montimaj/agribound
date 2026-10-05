# North American Landsat PAN/SR comparison

Example 26 extends the [matched comparison](landsat-pan-comparison.md) with
six frozen areas across California, Washington, Montana, Utah and Québec.
On October 1, 2026, **all 18 delineations completed** using `ee-rappjer` and
the existing published checkpoint. The four French and five measured
international sites were read for comparison; their inference was not repeated.

These references measure agreement with **crop-mapping units, irrigation
units, or declarations**. None is claimed to be an independently surveyed
physical-field gold standard. The [reference acquisition workflow](north-america-reference-workflow.md)
documents the remaining Corn Belt, dryland-cereal and Mexico gaps, useful
organizations, and a blank annotation kit for obtaining stronger references.

## Reproduce

Run from the repository root with Python >=3.12:

```bash
python -m pip install -c examples/landsat_comparison_constraints.txt -e ".[gee,delineate-anything]" matplotlib
agribound auth --project ee-rappjer
python examples/26_landsat_north_america_comparison.py --gee-project ee-rappjer
```

The constraints record the live Windows/Python 3.12 environment; wheel
availability on other systems can differ. GeoPandas/pyogrio read the ZIP
shapefiles and file geodatabase; rasterio, pyproj, Shapely, pandas and
matplotlib handle evaluation and figures. The native inference backend
needs the Delineate-Anything extra, without separate `osgeo` bindings.
Reference downloads are public and need no account. Earth Engine requires
an enabled account, authenticated credentials, and permission to use project
`ee-rappjer`. The default downloads the published model if unavailable locally.

The actual live command reused these weights:

```bash
python examples/26_landsat_north_america_comparison.py --gee-project ee-rappjer --checkpoint outputs/landsat_pan_sr_comparison/model_cache/hub/models--MykolaL--DelineateAnything/snapshots/369d0b4c44cf9bec2bd3a27bc81810cadd2c963e/DelineateAnythingv2.pt
```

Pinned revision: `369d0b4c44cf9bec2bd3a27bc81810cadd2c963e`;
checkpoint SHA-256:
`46700b8a279b07922953a11adaeb5e658d9a2384b6334c8e0a3090886218915a`.
The example rejects a different checkpoint. On Windows, clear incompatible
GIS data variables inherited from another application's environment:

```powershell
Remove-Item Env:PROJ_LIB,Env:PROJ_DATA,Env:GDAL_DATA -ErrorAction SilentlyContinue
```

Useful commands:

```bash
# References, dated aerial inspection views, and inputs without inference:
python examples/26_landsat_north_america_comparison.py --prepare-only
# References/aerial views only:
python examples/26_landsat_north_america_comparison.py --audit-only
# A frozen subset:
python examples/26_landsat_north_america_comparison.py --sites ca_colusa qc2023
# Recompute coverage evaluation and maps from completed polygons, without credentials/weights:
python examples/26_landsat_north_america_comparison.py --reevaluate-only --offline
# Rebuild cross-site tables/figure without inference:
python examples/26_landsat_north_america_comparison.py --summarize-only
```

`--offline --checkpoint PATH` runs inference from previously prepared imagery,
cached references and dated aerial views. Completed sites are reused;
`--output-dir NEW_DIRECTORY` is required for changed site selection. The
runner continues independent sites after a failure and exits nonzero when
a requested site remains unmeasured. A changed provider download fails its
checksum rather than silently becoming a new reference version.

## Frozen sites and reference inventory

`examples/landsat_north_america_sites.json` fixes boxes in WGS84, filters,
inclusive acquisition windows, source endpoints/checksums, crop/climate
evidence, dated aerial IDs, and selection rationale. Its canonical SHA-256
is `3baa3f5271362cf52b357906b4a55ba45afc39d301eba813fbb08907cd6e2a83`.
Reference availability, provider classes, aerial spot checks and eligible
matched scenes were inspected before inference. `sites_frozen.json` records
the selection. No difficult site was replaced after scores were observed.

| Site / county or equivalent | Crop/use context | Imagery window | References; median size | Dated audit |
| --- | --- | --- | --- | --- |
| Colusa County, CA | Rice, DWR R1 | June–September 2022 | 14; 15.61 ha | NAIP July 9, 2022 |
| Tulare County, CA | Orchard/vineyard blocks | May–August 2022 | 333; 3.75 ha | NAIP June 24, 2022 |
| Grant County, WA | Cereal, hay/silage, vegetable/herb groups; pivots | May–August 2023 | 29; 12.28 ha | NAIP June 22, 2023 |
| Beaverhead County, MT | Irrigation-equipped fields; species unknown | June–September 2020 | 42; 8.29 ha | NAIP July 21, **2019** |
| Cache County, UT | Grass hay, alfalfa, wheat, barley and minor crops | May–August 2018 | 132; 3.36 ha | NAIP August 29, 2018 |
| Les Maskoutains / Saint-Hyacinthe, Montérégie, QC | Maize, soya, hay declarations; narrow strips | June–September 2023 | 166; 5.16 ha | GeoMont spring 2023 |

All windows cover the Northern Hemisphere growing season. Species/group
labels come from provider attributes, not visual interpretation. The
[DWR legend](https://data.cnra.ca.gov/dataset/6c3d65e3-35bb-49e1-a51e-49d5a2cf09a9/resource/9a00d123-7d5f-46a0-8e89-b6b0591a01f0/download/2022-dwr-standard-land-use-legend-remote-sensing-version.pdf)
identifies rice R1, peaches/nectarines D5, almonds D12 and plums D7.
Washington groups do not establish crop species or purely dryland production.
Montana has irrigation-type attributes, without crop labels. Québec crops
are client declarations, with retained inactive parcels removed using
`NBPRO >= 1`, `TYPPAR = PAC` and explicit cultivated-use exclusions.

NOAA state summaries support California's dry-summer seasonality,
Washington's dry interior, and the mountain/interior temperature seasons in
[California](https://statesummaries.ncics.org/chapter/ca/),
[Washington](https://statesummaries.ncics.org/chapter/wa/),
[Montana](https://statesummaries.ncics.org/report_section/mt/1/) and
[Utah](https://statesummaries.ncics.org/chapter/ut/).
[MAPAQ's agricultural landscape report](https://cdn-contenu.quebec.ca/cdn-contenu/adm/min/agriculture-pecheries-alimentation/consultation-publique/RA_Ruiz_Lavoie_evolution-spatiale-activites-agricoles_MAPAQ.pdf)
documents Montérégie's Saint Lawrence lowland agricultural context.
These regional descriptions are not per-field climate measurements or
formal Köppen classifications. They do not establish causal performance effects.

| Source | Representation, year and completeness | License / direct provider documentation |
| --- | --- | --- |
| DWR / Land IQ 2022 | Homogeneous cropped-area units; water-year mapping. Crop/age splits can occur inside continuous cultivation. Exact polygon image dates and boundary accuracy are unavailable. Aerial and satellite information contribute to mapping. Only selected crop classes are scored. | Public domain; [catalog](https://lab.data.ca.gov/dataset/statewide-crop-mapping), [2022 download](https://data.cnra.ca.gov/dataset/6c3d65e3-35bb-49e1-a51e-49d5a2cf09a9/resource/b92e0daf-6e2e-4b5c-a112-09474138d1cd/download/i15_crop_mapping_2022_shp.zip) |
| WSDA 2025 snapshot | Annual publication filtered to WSDA observations surveyed in 2023. Publication/survey date does not establish geometry-update date. Other survey years and NASS observations are excluded, leaving unknown coverage. | No explicit reuse license found; extracted vectors and reference-overlay maps stay local. [Program](https://agr.wa.gov/departments/land-and-water/natural-resources/agricultural-land-use), [download](https://cms.agr.wa.gov/WSDAKentico/Documents/DO/NRAS/2025WSDACropDistribution-gdb.zip) |
| Montana DNRC / UM Climate Office | Approximately 2020 irrigation-equipped units, constructed with 2019/2020 NAIP and 2018–2021 satellite indices. Small/ambiguous fields may be omitted; pivot/corner or irrigation splits can lack physical crop edges. Audit is a year earlier. | Public service, provider copyright, no explicit redistribution license found; vectors/overlay maps stay local. [Service and metadata](https://gis.dnrc.mt.gov/arcgis/rest/services/WRD/MontanaStatewideIrrigation/FeatureServer/0) |
| Utah DWRe 2018 | Irrigated land-use mapping with survey/crop attributes. All selected SURV_YEAR values are 2018; exact digitization dates unknown. Wet/riparian interfaces and subdivisions can be visually uncertain. | CC BY-SA, version unspecified in [provider item](https://www.arcgis.com/home/item.html?id=4f1671cc9da441b3825ead1512a08e7b); [SGID](https://gis.utah.gov/products/sgid/planning/water-related-land-use/), [2018 service](https://services.arcgis.com/ZzrwjTRez6FJiOq4/arcgis/rest/services/Water_Related_Land_Use_Statewide_2018/FeatureServer/0) |
| FADQ BDPPAD 2023 | Active annual declarations, not exhaustive cultivation fields. Original geometry dates unknown. Insurance/declaration splits can be invisible; spring ortho and summer Landsat differ seasonally. No crop-label ground truth claim. | CC BY 4.0; [program](https://www.fadq.qc.ca/documents/donnees/base-de-donnees-des-parcelles-et-productions-agricoles-declarees/), [license/catalog](https://www.donneesquebec.ca/recherche/dataset/base-de-donnees-des-parcelles-et-productions-agricoles-declarees-bdppad), [2023 download](https://www.fadq.qc.ca/fileadmin/geo4tb/BDPPAD/BDPPAD_V03_2023.zip), [user guide](https://www.fadq.qc.ca/fileadmin/geo4tb/BDPPAD/bdppad-v03-guide-utilisateur.pdf) |

The suite writes a machine-readable `reference_inventory.csv` and complete
per-site `reference_source.json`. ArcGIS adapters freeze object IDs before
fetching batches, verify every requested ID, and request EPSG:4326 explicitly.
ZIP data are spatially subset in their declared CRS. FADQ CP1252 encoding is
explicit. Missing CRS, truncated responses, changed checksums and duplicate
IDs fail. Invalid geometries are repaired with Shapely's structure method,
and repair counts are recorded. Whole polygons are retained by representative
point in the frozen box; neither reference nor prediction is clipped for IoU.

## What the evaluation means

All six references received prediction-blind **single-reviewer qualitative
spot checks**, not exhaustive validation or independent double review.
Inspected DWR orchard lanes/rice bunds and WSDA rims generally align with
visible edges. Montana corner splits, Utah wet-zone divisions and Québec
declaration strips often have ambiguous or absent physical separations.
Irrigation and declaration scores are therefore explicitly separate
reference categories. Administrative splits are never described as proven
physical edges. Original per-polygon positional errors are unspecified;
this limits interpretation at a 10 m tolerance. Model training geographic
overlap is undocumented, so these are not proven held-out sites.

For incomplete/class-filtered references, the headline scores use the fixed
union of selected reference footprints. Whole predictions enter object
evaluation when their representative point lies in this mask. Boundary
precision/recall uses **original line segments inside the mask**, not newly
created polygon-clipping edges. Unmatched predictions outside the mask
remain unknown. Precision is conditional on mapped footprints and can be
optimistic compared with an exhaustively labeled area. Whole-field IoU and
size are unchanged. Adjacent reference polygons can count a shared edge
twice; the established per-polygon evaluator is retained consistently.

Both conditional `known_footprints` and original whole-AOI `aoi` tables/maps
are retained. Full-AOI precision treats unsupported regions as unmatched
and is unsuitable as exhaustive field accuracy. Coverage masks are not
adjusted after predictions. Masks retain their original component vertices,
projected directly into the boundary UTM grid, with a **1 mm** outward
numerical guard. Validation caught floating-point loss, area-densification
differences and an invalid geographic dissolve. Coverage scores were
recomputed from the same frozen footprint components and saved predictions,
without repeating inference. All six sites retain 100% of their reference
perimeter in the corrected mask check. `coverage_evaluation.json`
records old/new table checksums, evaluator checksum, and the correction.

## Identical experiment settings and fusion

Example 23 supplies the unchanged matched L8/9 workflow. Scene cloud cover
is <=20%; exact `LANDSAT_SCENE_ID` pairs and unmatched observations are
recorded. Both products use existing QA_PIXEL bits 0–4 masking. A 30 m cell
is usable for an observation only when all six SR bands and all four PAN
subpixels are valid. Medians use these identical scene/mask supports.
Aligned local UTM grids use nearest-neighbour regridding at 15/30 m,
with 600 m context. This is regridding of native samples, not new measured
SR detail. Additional QA criteria were not introduced for this cohort.

PAN is B8 **TOA** unit reflectance replicated into three model channels.
SR is the existing six-band **surface-reflectance** composite, with RGB
selected for this engine. The combined experiment injects zero-mean PAN
block detail into SR RGB, preserving each 30 m band's mean. It combines
**imagery**, not predictions or boundaries, and produces experimental RGB
on a 15 m grid, not calibrated 15 m surface reflectance. L8/9 B8 covers
approximately 0.50–0.68 µm, overlapping green/red and not the full blue,
NIR or SWIR response; this does not validate NIR/SWIR pansharpening.
The [original method documentation](landsat-pan-comparison.md#how-pansr-is-combined)
explains the spectral limitations and image-level choice.

Per-site `fusion_validation.json` retains coarse-band conservation checks
and the reduced-resolution 60 m SR + 30 m PAN comparison with known 30 m SR
(RMSE and spectral angle). Box degradation is not MTF matched, and cannot
prove 15 m SR radiometric accuracy. Validation is reported even when fusion
does not improve boundaries.

All experiments use `large_v2`, native backend, CPU FP32, batch 1,
confidence 0.15, tile step 0.5, super-resolution 1 and seed 42. Minimum
area is 1,000 m², simplification 2 m, with no smoothing, SAM or LULC filter.
The same pixel-based model context spans different ground distances at
15 and 30 m; PAN lacks colour while SR/fusion have RGB. Both resolutions
exceed the model's reported 0.25–10 m training range. These necessary
input differences prevent attributing every score change solely to resolution.
SR's NIR/SWIR benefit is not tested by an RGB engine.

## Measured comparisons

Headline scores are conditional on mapped footprints. Each cell shows
**detection F1 / boundary F1 (15 m)**; PAN/fusion inputs are 15 m and SR is 30 m.

| Site | Reference type | PAN | SR | PAN+SR |
| --- | --- | ---: | ---: | ---: |
| ca_colusa | mapped_crop_unit | 0.410 / 0.542 | 0.051 / 0.266 | 0.186 / 0.506 |
| ca_orchards | mapped_crop_unit | 0.640 / 0.723 | 0.417 / 0.425 | 0.581 / 0.687 |
| wa_grant | mapped_crop_unit | 0.600 / 0.516 | 0.680 / 0.409 | 0.566 / 0.456 |
| mt_beaverhead | irrigation_mapping_unit | 0.317 / 0.299 | 0.197 / 0.184 | 0.364 / 0.308 |
| ut_cache | irrigation_mapping_unit | 0.142 / 0.310 | 0.070 / 0.118 | 0.122 / 0.244 |
| qc2023 | declaration_parcel | 0.311 / 0.420 | 0.200 / 0.283 | 0.405 / 0.402 |

| Site | Matched scenes | Unmatched PAN / SR | PAN / SR / fused inference | Shared acquisition / fusion |
| --- | ---: | ---: | ---: | ---: |
| ca_colusa | 11 | 4 / 0 | 14.6 / 7.7 / 13.9 s | 5.1 / 0.081 s |
| ca_orchards | 10 | 0 / 0 | 21.6 / 11.6 / 18.4 s | 5.1 / 0.076 s |
| wa_grant | 26 | 0 / 0 | 13.7 / 5.4 / 10.0 s | 7.4 / 0.079 s |
| mt_beaverhead | 11 | 0 / 0 | 11.0 / 4.5 / 9.3 s | 4.3 / 0.080 s |
| ut_cache | 8 | 1 / 0 | 11.8 / 4.0 / 8.9 s | 4.1 / 0.072 s |
| qc2023 | 5 | 0 / 0 | 15.0 / 6.7 / 11.8 s | 4.5 / 0.082 s |

All six sites have **100% valid composite pixels** within both the AOI and
known footprints on the common support. This is composite coverage, not
100% valid observations per scene. All selected reference geometry lies
within the imagery export to numerical precision.

PAN has the highest boundary F1 at five sites, and higher detection F1
than SR at five. Grant County is a counterexample: SR detection F1 is 0.680
versus PAN 0.600, although PAN locates more boundaries within 15 m.
Fusion improves detection over PAN in Montana (0.364 versus 0.317) and
Québec (0.405 versus 0.311). Its boundary F1 improves slightly in Montana
(0.308 versus 0.299), and is lower in Québec (0.402 versus 0.420).
It reduces detection in the two California sites, Washington and Utah.
These are measured unit-agreement results; irrigation/declaration splits
are not independent physical edges.

Tulare orchard blocks show a substantial PAN detection gain over SR
(0.640 versus 0.417). Cache County remains difficult: PAN recall is only
0.083, SR 0.038 and fusion 0.068. Higher boundary precision alone does not
mean that most fields were detected. Colusa has just 14 rice references;
one-to-one detection varies sharply and needs more independent sites.

Fusion is therefore a site-specific option, not a demonstrated default
improvement over PAN. SR is faster and remains the calibrated spectral
product; its additional NIR/SWIR channels need a compatible engine for
their delineation value to be tested.

| Reference category / cohort | Input | Equal-site detection / boundary F1 | Reference-count-weighted detection / boundary F1 |
| --- | --- | ---: | ---: |
| mapped_crop_unit (3 sites; 376 refs) | pan | 0.550 / 0.594 | 0.629 / 0.701 |
| mapped_crop_unit (3 sites; 376 refs) | sr | 0.383 / 0.367 | 0.423 / 0.418 |
| mapped_crop_unit (3 sites; 376 refs) | combined | 0.444 / 0.550 | 0.565 / 0.663 |
| irrigation_mapping_unit (2 sites; 174 refs) | pan | 0.230 / 0.304 | 0.184 / 0.307 |
| irrigation_mapping_unit (2 sites; 174 refs) | sr | 0.134 / 0.151 | 0.101 / 0.134 |
| irrigation_mapping_unit (2 sites; 174 refs) | combined | 0.243 / 0.276 | 0.180 / 0.260 |
| declaration_parcel (1 site; 166 refs) | pan | 0.311 / 0.420 | 0.311 / 0.420 |
| declaration_parcel (1 site; 166 refs) | sr | 0.200 / 0.283 | 0.200 / 0.283 |
| declaration_parcel (1 site; 166 refs) | combined | 0.405 / 0.402 | 0.405 / 0.402 |
| France (legacy AOI) | pan | 0.237 / 0.433 | 0.280 / 0.524 |
| France (legacy AOI) | sr | 0.132 / 0.220 | 0.127 / 0.248 |
| France (legacy AOI) | combined | 0.220 / 0.397 | 0.250 / 0.473 |
| international (legacy AOI) | pan | 0.203 / 0.447 | 0.158 / 0.442 |
| international (legacy AOI) | sr | 0.139 / 0.301 | 0.070 / 0.297 |
| international (legacy AOI) | combined | 0.213 / 0.440 | 0.144 / 0.435 |

Aggregates are means of site metrics, not pooled matching or global
length-weighted boundary F1. Reference-count weighting is dominated by
larger mapped inventories; equal-site weighting gives the 14 Colusa rice
references the same site influence as 333 orchard units. Reference kinds
are kept separate. Legacy French/international figures use whole-AOI
evaluation and different reference semantics, so cross-cohort absolute
scores are not directly interchangeable. The earlier French results
also generally favour PAN over SR, while South Africa favours fusion
detection and the Mekong has no matched fields for any input. The new
Washington result reinforces that PAN is not always best at detection.

| Site | Reduced-resolution RGB RMSE: resampled SR → fused | Median spectral angle: resampled → fused |
| --- | --- | ---: |
| ca_colusa | 0.0105/0.0069/0.0053 → 0.0053/0.0046/0.0052 | 0.472° → 0.664° |
| ca_orchards | 0.0162/0.0113/0.0092 → 0.0078/0.0065/0.0073 | 0.767° → 1.090° |
| wa_grant | 0.0096/0.0059/0.0050 → 0.0049/0.0041/0.0044 | 0.580° → 0.750° |
| mt_beaverhead | 0.0133/0.0103/0.0087 → 0.0074/0.0066/0.0071 | 0.502° → 0.790° |
| ut_cache | 0.0189/0.0160/0.0151 → 0.0098/0.0091/0.0096 | 1.136° → 1.335° |
| qc2023 | 0.0055/0.0042/0.0037 → 0.0032/0.0027/0.0029 | 0.552° → 0.658° |

All sites conserve coarse RGB means with maximum absolute error
<1.5e-8 unit reflectance. Reduced-resolution RMSE improves in all three bands
while median spectral angle worsens at every site. Conservation and
lower RMSE therefore do not establish better colour or delineation.
These values validate an experimental tradeoff, not true 15 m SR.

The complete CSV (`outputs/landsat_north_america/north_america_comparison.csv`; generated locally)
includes precision, recall and F1 at 10/15/30 m, mean IoU
of one-to-one matches at IoU >=0.5, per-reference oversegmentation and
undersegmentation errors (lower is better), counts, inference wall time,
resolution, and imagery coverage. Means over matched fields exclude missed
fields, so IoU must be read with detection recall. Size bins are
0–1, 1–5, 5–20, 20–100 and 100–100,000 ha, with reference counts and empty
bins marked missing. Individual-field bootstrap intervals (200 resamples)
are in JSON; spatial dependence and label uncertainty limit interpretation.
The 15 m tolerance size-class CSV (`outputs/landsat_north_america/size_class_15m.csv`; generated locally)
provides a smaller per-site table for all three inputs. None of the **41
sub-hectare references** is detected at IoU >=0.5 by any input: 12 orchard
units, one Montana unit, 25 Utah units and three Québec declarations.
The 0.1 ha output threshold and coarse model inputs limit small-field
recall; ambiguous provider splits remain another explanation. A visible
nearby boundary does not imply successful one-to-one field detection.
`resolution_m` denotes the imagery/inference grid. GeoPackage vectors have
continuous coordinates with 2 m simplification, not an independent raster
resolution or a claim of 2 m boundary accuracy.

CPU runtimes are single runs in PAN/SR/combined order, including loading,
tiling and postprocessing, excluding shared downloads/fusion and evaluation.
`preparation_timing.json` records those separate costs and input-cache reuse.
They are not controlled throughput benchmarks. Valid coverage is reported
at 30 m cell centres for all six SR bands, on the same common PAN/SR support;
vector area within the export is separate from pixel validity. Coverage
does not imply every scene is usable at every pixel.

Cross-site comparison, with explicit reference type and coverage scope (`outputs/landsat_north_america/north_america_comparison.png`; generated locally)

All per-site maps have identical extents in three columns, the same
unsharpened SR RGB background/stretch, cyan references and magenta predictions.
The two 900 m enlargements use the smallest eligible reference field and
the first shared edge >30 m, with a median-field fallback when no shared
edge exists. Zooms are reference-selected, not chosen for attractive model
results. `figure_windows.json` stores their bounds. The audit figure instead
uses dated aerial imagery, with overview/smallest/median-field views.

## Outputs and checks

Default output root: `outputs/landsat_north_america/`.

- `sites_frozen.json`, `suite_status.json`, `reference_inventory.csv`.
- `north_america_comparison.csv`: all tolerances/size bins, both coverage
  scopes, and previously completed cohorts.
- `headline_15m.csv`, `headline_15m_all_scopes.csv`, `aggregate_metrics.csv`,
  `north_america_comparison.png` and `.pdf`.
- Each site: raw `fields_pan.gpkg`, `fields_sr.gpkg`, `fields_combined.gpkg`
  and their `.gpkg.provenance.json`; `fields_*_evaluated.gpkg` and provenance;
  reference/coverage GeoPackages; inputs, scene manifests, fusion validation,
  imagery coverage, source/audit metadata, metrics, per-field CSVs, runtime
  logs and common-background figures in both original and `coverage/` views.

Published assets contain numerical tables, manifests and permitted maps.
WSDA and Montana vector data/reference-overlay maps stay in local outputs
pending license clarification. Summary bar charts expose no reference geometry.
No data provider was contacted, and no changes were committed or pushed.
The local `artifact_manifest.json` records checksums for all 36 raw/conditional
GeoPackages and provenance files, inputs, evaluation masks, tables and maps.

The machine-readable source inventory (`outputs/landsat_north_america/reference_inventory.csv`; generated locally)
and aggregate CSV (`outputs/landsat_north_america/aggregate_metrics.csv`; generated locally)
retain source and weighting distinctions. Permitted per-site maps:

Colusa rice units, DWR public-domain reference (`outputs/landsat_north_america/comparison.png`; generated locally)

Tulare orchard units, DWR public-domain reference (`outputs/landsat_north_america/comparison.png`; generated locally)

Cache County irrigation units, Utah DWRe reference under provider CC BY-SA terms (`outputs/landsat_north_america/comparison.png`; generated locally)

Montérégie declarations, FADQ reference under CC BY 4.0 (`outputs/landsat_north_america/comparison.png`; generated locally)

Reference attribution: California DWR / Land IQ; Utah Division of Water
Resources (provider CC BY-SA, version unspecified); La Financière agricole
du Québec (CC BY 4.0). The maps transform/project selected reference
outlines and overlay Agribound predictions on public Landsat imagery.
Washington and Montana maps are available in local output folders.

```bash
python -m pytest tests/unit/test_evaluate.py tests/unit/test_boundary_coverage.py tests/unit/test_landsat_comparison.py tests/unit/test_landsat_international.py tests/unit/test_landsat_north_america.py tests/unit/test_pan_fusion.py -q
```

Offline tests cover ArcGIS ID batching and truncation, pinned sources,
CRS-aware subsets, survey/declaration years, archive paths, Unicode crops,
whole-polygon selection, mask holes and reprojection, unchanged IoU,
valid-image nodata handling, scope-separated aggregation and uncertified
manual coverage. Imagery matching/fusion tests remain shared with example 23.

Public offline demonstration maps and source attributions are in the [tutorial guide](landsat-tutorials.md). Full research imagery, vectors and large/restricted overlay maps remain local and are not part of the proposed PR.
