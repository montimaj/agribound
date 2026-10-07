# Satellite Sources

Agribound builds one input raster per run (the "composite", stage A of the
pipeline) from one of eleven sources. The source metadata below is the content
of `agribound.registry.SOURCE_REGISTRY` (`agribound list-sources`,
`agribound.list_sources()`); the facts in it were checked against the Earth
Engine catalogue and the upstream packages on 2026-09-26 to 2026-09-28.

## Source overview

| Source | Key | Export resolution | Bands written | Value scale | Years | Coverage | Earth Engine |
|---|---|---|---|---|---|---|---|
| Sentinel-2 MSI L2A (harmonized) | `sentinel2` | 10 m | B1-B12 without B10 (12 bands) | reflectance × 10000 | 2017-present | Global. The Earth Engine collection also holds earlier L2A images (the first on 2015-07-04, counted 2026-09-28), which agribound does not accept; the catalogue lists the extent from 2017-03-28 and warns that 2017-2018 L2A coverage is not global | yes |
| Landsat 5/7/8/9 Collection 2 Level-2 | `landsat` | 30 m | SR_B2-SR_B7 in Landsat 8/9 naming (6 bands) | reflectance × 10000 | 1984-present | Global; L5 1984-2012, L7 1999-2024, L8 2013-, L9 2021- | yes |
| Landsat 7/8/9 panchromatic | `landsat-pan` | 15 m | B8 | `unit` TOA reflectance | 1999-present | Global; L7 1999-2024, L8 2013-, L9 2021-; one PAN bandpass per composite by default ([below](#landsat-panchromatic-landsat-pan)) | yes |
| Harmonized Landsat Sentinel-2 v2.0 | `hls` | 30 m | B1-B7 in HLSL30 naming (7 bands) | reflectance × 10000 | 2013-present | Global land; HLSL30 2013-, HLSS30 2015- | yes |
| NAIP | `naip` | 1 m (`naip_resolution_m`) | R, G, B, N | uint8 | 2002-2023 | Conterminous US; about 2-3 year revisit per state | yes |
| USGS NAIP Plus ImageServer | `usgs-naip-plus` | finest resolution of the selected footprints (0.3-0.6 m) | R, G, B, N | uint8 | 2012-2023 | Latest NAIP/HRO vintage per state only (see below) | no |
| SPOT 6/7 multispectral | `spot` | 6 m | R, G, B, N | `dn` (uncalibrated) | 2012-2023 | Global, **restricted** | yes |
| SPOT 6/7 panchromatic | `spot-pan` | 1.5 m | P | `dn` (uncalibrated) | 2012-2023 | Global, **restricted** | yes |
| Local GeoTIFF | `local` | the file's | the file's | unknown | any | user-provided | no |
| Google Satellite Embedding V1 (AlphaEarth Foundations) | `google-embedding` | 10 m | 64-D embedding `A00`-`A63` | embedding | 2017-2025 | Global land | yes with the default `google_embedding_backend="gee"`; no with `"source_coop"` |
| TESSERA embeddings | `tessera-embedding` | 10 m | 128-D embedding `T000`-`T127` | embedding | depends on `tessera_version` (below) | depends on version | no |

The LULC crop filter (on by default) reads its land-cover datasets from Earth
Engine for **every** source, including `local`, `usgs-naip-plus` and
`tessera-embedding` (see [LULC crop filter](#lulc-crop-filter)).

### Value scales

`value_scale` tells the engines what the pixel values mean
(`agribound.registry.source_value_scale`):

- `reflectance_x10000`: float32 surface reflectance × 10000. Sentinel-2 values
  are used as stored; Landsat C2 L2 digital numbers are converted to
  reflectance (`DN × 2.75e-5 − 0.2`), clipped at 0 and multiplied by 10000
  before compositing; HLS (stored as 0-1 reflectance in Earth Engine) is
  multiplied by 10000.
- `uint8`: 8-bit digital numbers 0-255 (NAIP, USGS NAIP Plus).
- `unit`: Landsat PAN calibrated TOA reflectance, exported as float32 as stored.
- `dn`: SPOT 6/7 per-band medians (or greenest-pixel selections) of the
  scenes' raw digital numbers, written as float32 (a median of an even number
  of scenes can be a half-integer). Their radiometry has not been verified, so
  the reflectance conversions (`agribound.io.raster.to_unit_reflectance`,
  `to_s2_dn`) refuse them; engines that use a percentile stretch
  (Delineate-Anything, GeoAI, DINOv3, SAM) accept them.
- `embedding`: pre-computed embedding vectors.
- `unknown`: local rasters. Engines that need a value scale take it from
  `engine_params["value_scale"]`.

### Canonical bands

Engines ask for canonical bands (`R, G, B, NIR, NIR_NARROW, SWIR1, SWIR2`) and
`agribound.engines.base.get_canonical_band_indices` maps them to band
positions of the composite:

| Source | R | G | B | NIR | NIR_NARROW | SWIR1 | SWIR2 |
|---|---|---|---|---|---|---|---|
| `sentinel2` | B4 | B3 | B2 | B8 | B8A | B11 | B12 |
| `landsat` | SR_B4 | SR_B3 | SR_B2 | SR_B5 | - | SR_B6 | SR_B7 |
| `hls` | B4 | B3 | B2 | B5 | B5 | B6 | B7 |
| `naip`, `usgs-naip-plus` | R | G | B | N | - | - | - |
| `spot` | R | G | B | N | - | - | - |
| `spot-pan` | P | P | P | - | - | - | - |
| `landsat-pan` | B8 | B8 | B8 | - | - | - | - |

HLSS30 bands `B1, B2, B3, B4, B8A, B11, B12` are renamed to the HLSL30 names
`B1`-`B7`, so `B5` is the narrow NIR and `B6`/`B7` are SWIR 1/SWIR 2 for both
HLS sensors (agribound 0.1.x put the HLSS30 red-edge bands B6/B7 into these
slots; see the [CHANGELOG](https://github.com/montimaj/agribound/blob/main/CHANGELOG.md)).
For `local` rasters (and to override any source) pass `bands`, a mapping of
canonical name to 1-based band index, e.g. `{"R": 1, "G": 2, "B": 3, "NIR": 4}`.

## How composites are built

### Extent and grid

The export grid covers the **bounding box of the study area** in
`export_crs`, with pixel edges on multiples of the resolution:

- `export_crs="utm"` (default): the WGS 84 / UTM zone of the study-area
  centroid, used for the whole study area. Any `"EPSG:<code>"` is accepted; a
  geographic CRS logs a WARNING because metre scales then become non-square
  pixels.
- Pixels are **not** masked to the study-area polygons. When the study area is
  a set of field polygons (for example reference boundaries), masking would
  imprint their outlines on the engine input. Predictions outside an irregular
  study area are removed later by `aoi_selection` (see
  [Configuration](configuration.md#study-area-selection)).
- Composites of the same study area, CRS and resolution (for example FTW's two
  date windows) share one grid.

### Date window, cloud filtering and masking

Without `date_range` the window is the calendar year of `year`; `date_range`
(`("YYYY-MM-DD", "YYYY-MM-DD")`, end inclusive) replaces it. Scenes are first
filtered on their scene cloud-cover property (`cloud_cover_max`, default 20 %:
`CLOUDY_PIXEL_PERCENTAGE` for Sentinel-2, `CLOUD_COVER` for Landsat,
`CLOUD_COVERAGE` for HLS, `cloud_coverage_percentage` for SPOT), then masked
per pixel:

| Source | Pixel mask |
|---|---|
| `sentinel2` | `s2_cloud_mask="scl"` (default): SCL classes 3 (cloud shadow), 8 and 9 (cloud medium/high probability), 10 (thin cirrus). `s2_cloud_mask="cloud_score_plus"`: keeps pixels with Cloud Score+ `cs_cdf >= cloud_score_threshold` (default 0.60). |
| `landsat` | `QA_PIXEL` bits 0-4 (fill, dilated cloud, cirrus, cloud, cloud shadow). |
| `landsat-pan` | `QA_PIXEL` bits 0, 1, 3 and 4 (fill, dilated cloud, cloud, cloud shadow) on Landsat 7; bits 0-4 (also bit 2, cirrus) on Landsat 8/9. |
| `hls` | `Fmask` bits 1-3 (cloud, adjacent to cloud/shadow, cloud shadow). |
| `spot`, `spot-pan` | no pixel mask; scene filter only. |
| `naip` | none (mosaic, see below). |

Masked pixels are NaN in the float32 composites.

`landsat` composites merge Landsat 5, 7, 8 and 9. Landsat 7 contributes for
windows between 1999-05-28 and 2024-01-19 (including its SLC-off stripes since
2003); there is no option to exclude it. `landsat-pan` chooses its missions
with `landsat_pan_missions` (see
[Landsat panchromatic](#landsat-panchromatic-landsat-pan)).

### Compositing methods

| `composite_method` | What it does |
|---|---|
| `median` (default) | Per-band median of the unmasked values. |
| `greenest` | Per pixel, the observation with the highest NDVI (`qualityMosaic`), computed from the source's canonical `NIR` and `R` bands. Not available for sources without both. |
| `max_ndvi` | An alias of `greenest` (the same computation). |

NAIP is mosaicked, not composited, so only the default `median` is accepted
for `naip` (it is ignored).

### NAIP (`naip`)

- Only 4-band (R, G, B, N) images are used; some early years are RGB only and
  are skipped.
- Without `date_range`, images from `year - 1` to `year + 1` are mosaicked,
  exact-year images on top and the newest image on top within each group, so
  gaps in the requested year are filled from the neighbouring years (a
  WARNING names the neighbouring years whenever they have images in the
  window).
- With `date_range`, only images inside that range are used, newest on top.
- 0.1.x mosaicked the `year ± 1` images in unsorted collection order and
  ignored `date_range`; see the
  [CHANGELOG](https://github.com/montimaj/agribound/blob/main/CHANGELOG.md#results-produced-with-agribound--013-that-are-affected).
- Exported as uint8 (nodata 0) at `naip_resolution_m` (default 1.0 m). The
  native resolution is 0.6 m in most states since 2018 (0.3 m in some) and
  1-2 m before.
- Earth Engine holds NAIP for 2002-2023 (no 2024/2025 imagery as of 2026-09).

### USGS NAIP Plus (`usgs-naip-plus`)

- Read directly from the
  [USGS NAIP Plus ImageServer](https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer);
  no Earth Engine account is needed for the imagery.
- The service holds only the **latest NAIP/HRO vintage of each state**
  (2012-2023, most states 2019-2023), not the historical archive, so most
  years are unavailable for a given state; a year without imagery raises an
  error that lists the years the service has for the area. Use `naip` for other
  years. `usgs_allow_year_fallback=True` also accepts footprints from
  `year ± 1`; `usgs_state` restricts the query to one state.
- Footprints exist for the conterminous states, Alaska (2020), Hawaii (2013),
  Puerto Rico (2018), Guam (2013), the Northern Mariana Islands (2012) and
  American Samoa (2012), and none for the US Virgin Islands (service query,
  2026-09-27).
- The export is at the finest ground resolution of the selected footprints
  (0.3-0.6 m depending on the state vintage), so it can be several times larger
  than a 1 m NAIP export of the same area. The band order R, G, B, N is assumed
  (the service reports generic band names).
- A WARNING is logged when the selected footprints cover less than 99 % of the
  grid outline or less than 99 % of the study-area pixels have imagery
  (GeoTIFF tags `AGRIBOUND_FOOTPRINT_COVERAGE`, `AGRIBOUND_VALID_FRACTION`).

### Landsat panchromatic (`landsat-pan`)

Use `source="landsat-pan"` with `year` or `date_range` for a median composite
of native 15 m B8 (panchromatic) observations from `LANDSAT/LE07/C02/T1_TOA`,
`LANDSAT/LC08/C02/T1_TOA` or `LANDSAT/LC09/C02/T1_TOA`, chosen as described
under *Missions* below. Scene filtering uses `cloud_cover_max`; `QA_PIXEL`
masks fill, dilated cloud, cloud and cloud shadow, plus cirrus on Landsat 8/9.
Only median compositing is available because PAN has no separate NIR/red bands.
RGB engines read B8 three times, as with `spot-pan`; FTW and Prithvi require
multispectral inputs and do not support this source. Delineate-Anything was
trained on 0.25-10 m imagery, so it logs a WARNING for these 15 m composites
(see [Engines](engines.md#delineate-anything-delineate-anything)).

**Missions.** The PAN bands of the two sensor generations cover different
wavelengths: Landsat 7 ETM+ PAN 0.52-0.90 µm, which includes the near
infrared, and Landsat 8/9 OLI PAN 0.50-0.68 µm, which does not. Their values
therefore differ, most over vegetation, which reflects strongly in the near
infrared. Over oil palm, for example, the median TOA reflectance was 0.220 in
Landsat 7 PAN against 0.096 in Landsat 8 PAN and 0.081 in Landsat 9 PAN.
`landsat_pan_missions` (CLI `--landsat-pan-missions`) chooses the missions:

- `"auto"` (default) never mixes the two bandpasses. A date window that
  overlaps the Landsat 8/9 record uses Landsat 8 (from 2013-03-18) and
  Landsat 9 (from 2021-10-31) only; an earlier window uses Landsat 7
  (1999-05-28 to 2024-01-19). A 2013 composite therefore holds only Landsat 8
  images, from 18 March on. The choice depends on the dates alone: when
  Landsat 8 and 9 have no image over the study area that passes the scene
  filter, the run raises `NoDataError` instead of falling back to Landsat 7.
  The message names the missions searched and the years that have images of
  them; when Landsat 7 has images in the window that pass the same filters,
  it also gives their number and says that `landsat_pan_missions="LE07"`
  uses them.
- A list of mission IDs from `"LE07"`, `"LC08"` and `"LC09"` (or a
  comma-separated string, as on the command line:
  `--landsat-pan-missions LC08,LC09`) uses exactly those missions, where their
  record overlaps the window; for example `"LE07"` gives Landsat 7 composites
  after 2013 too. A list with Landsat 7 and Landsat 8 or 9 mixes the two
  bandpasses in one median, and a WARNING is logged when images of both
  contribute. A list none of whose missions has a record overlapping the year
  or `date_range` (for example `"LE07"` with 2025, or `"LC09"` with 2020) is
  rejected when the configuration is created, with a `ValueError`.

**Composite tags.** The GeoTIFF tags record the setting and what it
delivered; the provenance record keeps them under `facts.composite`:

| Tag | Content | Example (Lost Hills, California, 2023) |
|---|---|---|
| `AGRIBOUND_LANDSAT_PAN_MISSIONS` | the `landsat_pan_missions` setting | `auto` |
| `AGRIBOUND_MISSIONS_SELECTED` | the missions searched for the date window | `LC08,LC09` |
| `AGRIBOUND_SENSORS` | the missions with at least one image after the bounds, date and scene cloud filters | `LC08,LC09` |
| `AGRIBOUND_SENSOR_IMAGES` | the number of images of each of these missions | `LC08:16,LC09:16` |
| `AGRIBOUND_SPECTRAL_RESPONSE` | the PAN bandpasses of these missions | `L8/9 PAN 0.50-0.68 um` |

A selected mission can contribute no image, for example when every scene of
it over the area exceeds `cloud_cover_max`, so `AGRIBOUND_SENSORS` can list
fewer missions than `AGRIBOUND_MISSIONS_SELECTED`. `AGRIBOUND_COLLECTIONS`
lists the collections of the selected missions, `AGRIBOUND_CLOUD_MASK` and
`AGRIBOUND_SCALING` the masking and radiometry, and `AGRIBOUND_SLC_OFF` is
written when Landsat 7 is selected. The composite cache key includes
`landsat_pan_missions`.

Values remain calibrated TOA reflectance (`unit`), separate from the `landsat`
Level-2 surface-reflectance stack. This source performs no pansharpening, and
no special gap filling of the Landsat 7 SLC-off stripes after 2003. See the
Earth Engine catalogues for
[Landsat 7](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_TOA),
[Landsat 8](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA)
and [Landsat 9](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_TOA).

### SPOT 6/7 (`spot`, `spot-pan`)

!!! warning "Restricted source"
    The SPOT 6/7 collection (`AIRBUS/SPOT6_7`) is not in the public Earth
    Engine catalogue. Access is limited to select Earth Engine users; in
    agribound it is for internal DRI use. Selecting `spot` or `spot-pan` emits
    a `UserWarning`. External users who need SPOT-based field boundaries
    should contact the package author.

`spot-pan` writes the single panchromatic band; engines that need R, G, B read
it three times.

### Local GeoTIFF (`local`)

`local_tif_path` is validated (it must have a CRS; `bands` indices must lie
within the band count) and, when a study area is given, cropped to the study
area's bounding box without changing the pixel values. A raster without a
nodata value is accepted with a WARNING (zeros are then valid data). A study
area that does not overlap the raster raises `agribound.composites.NoDataError`.

### Embeddings

**Google Satellite Embedding** (AlphaEarth Foundations; 64 unit-length
dimensions, 10 m, annual 2017-2025) is read from:

- `google_embedding_backend="gee"` (default): Earth Engine
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`.
- `google_embedding_backend="source_coop"`: the public COG mirror on Source
  Cooperative, read window by window with rasterio (no Earth Engine compute,
  no geoai-py). Its tile index (`aef_index.parquet`, about 78 MB) is downloaded
  once to `embedding_cache_dir` (default `~/.cache/agribound`); on HPC point
  `embedding_cache_dir` at shared storage. Each pixel is taken from the tile of
  its own UTM zone (or hemisphere), so values can differ from the Earth Engine
  mosaic in narrow strips along zone boundaries, where both tiles hold data
  (96.6-100 % identical pixels in tests on 2023 data, 2026-09-27).

The AlphaEarth Foundations Satellite Embedding dataset is produced by Google
and Google DeepMind (CC-BY 4.0).

**TESSERA** embeddings (128 dimensions, 10 m) are streamed from the public
Zarr stores with `geotessera.GeoTesseraZarr` (geotessera >= 0.10; older
releases can no longer download anything because the old host returns HTTP
410). The dataset is chosen with `tessera_version`:

| `tessera_version` | Years with published tiles | Coverage |
|---|---|---|
| `v1` (default) | 2017-2025 | near-global for 2024; regional for 2017-2023 and 2025 |
| `v1.1` | 2015-2025 | regional |
| `v2` | 2017-2025 | beta, sparse (mostly Europe) |

`tessera_variant` selects a dataset variant (default: geotessera's default for
the version). A year outside the version's range is rejected when the
configuration is validated; a year without tiles over the study area raises
`NoDataError`. `agribound.composites.local.tessera_coverage` reports tile
coverage of a bounding box from the dataset manifest without downloading
embeddings.

TESSERA pixels are placed on the grid the Zarr store publishes (read with
`read_region`); agribound copies them without resampling when the export CRS
is the same UTM zone. On that grid, the TESSERA v1 2024 raster did not line up
exactly with optical composites of the same areas. With phase correlation of
edge maps (2026-09-29), it sat about 9-10 m east of the Sentinel-2 and SPOT
6/7 composites of example 15 (Pampas, Argentina) and about 5 m east and 5 m
south of those of example 02 (West Bengal, India). The store and the v1
GeoTIFF tiles that agribound 0.1.x downloaded hold the same values, on average
about a third of a pixel (3.3 m) apart in the Pampas (1-5.5 m, depending on
the tile); the 0.1.x mosaic of those tiles, however, placed the data about 9 m
west of the store's grid there (about 7 m west of the tiles' own
georeferencing), which happened to cancel most of the offset. Other areas were
not checked. Polygons of TESSERA clusters carry the offset (about 8 m in the
Pampas, measured against Sentinel-2 and SPOT edges and Delineate-Anything
polygons). SAM refinement on an optical composite redraws the outlines it
refines from that image: there, the polygons SAM 2 refined on Sentinel-2 sat
about 1-4 m east, while the polygons it left unrefined kept the offset of the
clusters.

Both embedding readers hold the whole area in memory while it is assembled
(about 512 bytes per pixel for TESSERA and 256 for Google embeddings, roughly
twice that at peak), so very large regions should be tiled (see
[HPC and large areas](hpc.md)).

### Valid-pixel check

After a composite is written, the share of pixels inside the study-area
polygons that are valid in every band (in at least one band for the uint8
`naip` and `usgs-naip-plus` composites) is stored in the GeoTIFF tag
`AGRIBOUND_VALID_FRACTION`. A share of 0 raises `NoDataError` (an Earth
Engine composite is then deleted); a WARNING is logged below 95 % for Earth
Engine imagery, below 99 % for USGS NAIP Plus and below 50 % for embeddings.

### Batch exports

`export_method="gdrive"` or `"gcs"` (with `gcs_bucket`) starts an Earth Engine
batch export task and stops with `ExportTaskStartedError`: the pipeline cannot
continue until the file exists locally. The task is recorded in
`<composite stem>_task.json` in the cache; a later run finds a task that is
`READY`, `RUNNING` or `COMPLETED` and does not start a duplicate. When the task
has finished, download the GeoTIFF and run with `source="local"`. The default
`export_method="local"` downloads the composite directly (in tiles of at most
`tile_size` pixels per side, assembled into one GeoTIFF).

### Caching

Every composite, window composite and embedding raster is cached in the
working directory (`cache_dir`, default `<output dir>/.agribound_cache`) under
a name that contains a key over the study area, source, year, date range and
compositing/export settings. See [Reproducibility](reproducibility.md#cache-keys).

## LULC crop filter

After post-processing, polygons whose crop fraction is below
`lulc_crop_threshold` (default 0.3) are removed. The dataset is chosen with
`lulc_dataset`:

| `lulc_dataset` | Earth Engine asset, crop rule | Years | Scale |
|---|---|---|---|
| `nlcd` | `projects/sat-io/open-datasets/USGS/ANNUAL_NLCD/LANDCOVER`, classes 81 (pasture/hay) and 82 (cultivated crops); fraction of pixels | 1985-2025 | 30 m |
| `cdl` | `USDA/NASS/CDL` band `cultivated` = 2; fraction of pixels (CONUS) | 2013-2023 | 30 m |
| `dynamic_world` | `GOOGLE/DYNAMICWORLD/V1`, annual median of the `crops` probability (of `crops` + `trees` with [`lulc_tree_crops`](#tree-crops)); mean probability (not a pixel fraction) | 2016 to the last complete calendar year | 10 m |
| `c3s` | `projects/sat-io/open-datasets/ESA/C3S-LC-L4-LCCS`, classes 10, 11, 12, 20, 30 (and the tree-cover classes with [`lulc_tree_crops`](#tree-crops)); fraction of pixels | 2000-2022 | 300 m |

Year ranges are those of the assets on 2026-09-26. Outside a dataset's range
the nearest available year is used (recorded in `lulc:year`, WARNING). Annual
NLCD and C3S come from the community catalogue (`projects/sat-io`), which has
no Google service-level agreement; if the Annual NLCD asset cannot be read, the
official `USGS/NLCD_RELEASES/2021_REL/NLCD` (2021 only) is used with a WARNING.

`lulc_dataset="auto"` (default) routes deterministically
(`agribound.postprocess.lulc_filter.select_lulc_dataset`):

1. NLCD, if the area intersects the conterminous-US envelope (-125, 24, -66,
   50) **and** at least 90 % of it has valid Annual NLCD pixels (checked on
   Earth Engine). Northern Mexico and southern Canada lie inside the envelope
   but have no NLCD pixels, so they are not routed to NLCD.
2. Otherwise Dynamic World for 2016 up to the previous calendar year (later
   years use the last complete year).
3. Otherwise (before 2016) C3S.

The Earth Engine catalogue notes that Dynamic World crop probabilities can be
comparatively low in the absence of obvious distinguishing features and on
high-return surfaces in arid climates, so the default threshold may remove
real fields in arid regions.

Output columns: `lulc:crop_fraction` (NaN when the dataset has no valid pixel
under the polygon, never 0), `lulc:dataset`, `lulc:year`, `lulc:valid`.
Other settings:

- `lulc_nodata_policy`: `"keep"` (default) keeps and flags NaN polygons;
  `"drop"` removes them.
- `lulc_mode`: `"server"` (default) computes per-polygon means with Earth
  Engine `reduceRegions` (pixel-area weighted, in batches of
  `lulc_batch_size`); `"raster"` downloads a float32 crop raster during the
  composite stage and averages the pixels whose centres fall inside each
  polygon locally, so the delineation stage can run offline.
- `lulc_on_error`: `"raise"` (default) aborts the run when the filter fails
  (for example without Earth Engine credentials); `"warn"` keeps the
  unfiltered polygons, logs a WARNING and records the failure in the
  provenance record. `lulc_filter=False` (CLI `--no-lulc-filter`) skips the
  filter.
- `lulc_tree_crops`: counts tree cover as crop; see [Tree crops](#tree-crops).

### Tree crops

Dynamic World files tree crops under `trees`, not `crops`: Table 1 of Brown
et al. (2022) lists "Plantations such as apples, bananas, citrus, and rubber"
among the examples of trees, while `crops` is "Human planted/plotted cereals,
grasses, and crops". The `crops` probability of orchards and plantations is
therefore low, and the default filter can remove them where it uses Dynamic
World. Of 95 oil-palm blocks at Twifo Praso, Ghana (RSPO GeoRSPO concession
boundaries), it kept none for 2020 and 1 for 2023.

`lulc_tree_crops=True` (CLI `--lulc-tree-crops`) counts tree cover as crop:

| `lulc_dataset` | Crop value with `lulc_tree_crops=True` |
|---|---|
| `dynamic_world` | mean over the polygon of the annual median of the per-image sum of the `crops` and `trees` probabilities, i.e. the probability of crops or trees (`lulc_stats["band"]` is `crops+trees`) |
| `c3s` | fraction of pixels in the cropland classes or the tree-cover classes 50, 60, 61, 62, 70, 71, 72, 80, 81, 82 and 90 (not 100, the tree and shrub mosaic, nor 160 and 170, flooded tree cover) |
| `nlcd`, `cdl` | unchanged: NLCD class 82 (cultivated crops) includes "perennial woody crops such as orchards and vineyards" ([NLCD legend](https://www.mrlc.gov/data/legends/national-land-cover-database-class-legend-and-description)), and the CDL cultivated layer counts tree crops such as apples, citrus, almonds and olives as cultivated ([NASS](https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_Cultivated-Layer-2023.htm)) |

With the option, the filter kept all 95 Twifo Praso blocks with Dynamic World
for both years (for 2023 in both `lulc_mode`s); with C3S (2015) it kept all
95 with and without the option. Where `"auto"` selects NLCD (most of the
conterminous US), the option changes nothing: orchards and vineyards mapped as
class 82 are kept by default.

The trade-off: the filter then keeps forest and other tree cover as well
(Dynamic World `trees`, the C3S tree-cover classes); it still removes polygons
on water, built-up land, bare ground, grassland and shrubland. Use it where the
fields of interest are tree crops; polygons that the engine draws in forest
are then kept.

The setting is recorded in `lulc_stats["tree_crops"]` (also in the provenance
record). With `lulc_mode="raster"`, the LULC raster's tag
`AGRIBOUND_LULC_TREE_CROPS` describes the raster, not the run: it is `True`
only for a Dynamic World or C3S raster made with the option, which counts tree
cover and has its own cache key, so Dynamic World and C3S rasters made with
and without the option are not shared. NLCD and CDL rasters do not change
with the option, so runs with and without it share one raster, tagged `False`
even in a run with `lulc_tree_crops=True`; a raster cached by agribound 1.0.0
or 1.0.1, which is still reused, has no tag. The option enters the configuration hash
only when True, so outputs made without it are still reused.

## Data citations

- Earth Engine: Gorelick et al. (2017), *Remote Sensing of Environment* 202,
  18-27, <https://doi.org/10.1016/j.rse.2017.06.031>.
- HLS: Claverie et al. (2018), *Remote Sensing of Environment* 219, 145-161,
  <https://doi.org/10.1016/j.rse.2018.09.002>.
- Dynamic World: Brown et al. (2022), *Scientific Data* 9, 251,
  <https://doi.org/10.1038/s41597-022-01307-4>.
- Cloud Score+: Pasquarella et al. (2023), CVPR Workshops, 2125-2135,
  <https://doi.org/10.1109/CVPRW59228.2023.00206>.
- Annual NLCD: U.S. Geological Survey (2024), Annual NLCD Collection 1 Science
  Products, <https://doi.org/10.5066/P94UXNTS>.
- C3S land cover: Copernicus Climate Change Service (2019),
  <https://doi.org/10.24381/cds.006f2c9a>.
- AlphaEarth Foundations: Brown et al. (2025), arXiv:2507.22291.
- TESSERA: Feng et al. (2026), CVPR 2026, arXiv:2506.20380.

See [Citation & References](../citation.md) for the full list.
