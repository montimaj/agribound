# Satellite Sources

Agribound builds one input raster per run (the "composite", stage A of the
pipeline) from one of sixteen sources. The source metadata below is the content
of `agribound.registry.SOURCE_REGISTRY` (`agribound list-sources`,
`agribound.list_sources()`); the facts in it were checked against the Earth
Engine catalogue and the upstream packages on 2026-09-26 to 2026-09-28, those
of Sentinel-1 on 2026-10-07, those of the Umbra and Capella open data on
2026-10-07 to 2026-10-09, and those of the CDL and ACI collections on
2026-10-09. Two of the sources are not imagery: `cdl-sequence` and
`aci-sequence` are stacks of annual crop-type maps (class codes), read only
by the `crop-sequence` engine.

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
| Sentinel-1 C-band SAR GRD | `sentinel1` | 10 m (about 20 × 22 m resolution) | VV, VH, VV_VH (VV − VH) | uint8 (dB scaled to 1-255) | 2015-present | Global where IW images with VV and VH exist; S1A 2014-2026, S1B 2016-2021, S1C 2025-, S1D 2026- ([below](#sentinel-1-sentinel1)) | yes |
| Umbra open SAR (X-band, GEC) | `umbra` | 1 m (`vhr_sar_resolution_m`) | MEDIAN (read as R, G, B) | `dn` (uncalibrated log-scaled amplitude) | 2023-present | Sites of the Umbra Open Data Program on AWS; one collect is a scene of about 5 × 5 km ([below](#open-vhr-sar-umbra-capella)) | no |
| Capella open SAR (X-band, GEO) | `capella` | 1 m (`vhr_sar_resolution_m`) | MEDIAN (read as R, G, B) | `db` (sigma0 dB + 50) | 2021-present | Capella Space open data on AWS; few farmland sites ([below](#open-vhr-sar-umbra-capella)) | no |
| Local GeoTIFF | `local` | the file's | the file's | unknown | any | user-provided | no |
| Google Satellite Embedding V1 (AlphaEarth Foundations) | `google-embedding` | 10 m | 64-D embedding `A00`-`A63` | embedding | 2017-2025 | Global land | yes with the default `google_embedding_backend="gee"`; no with `"source_coop"` |
| TESSERA embeddings | `tessera-embedding` | 10 m | 128-D embedding `T000`-`T127` | embedding | depends on `tessera_version` (below) | depends on version | no |
| USDA NASS Cropland Data Layer, multi-year stack | `cdl-sequence` | 30 m (10 m maps from 2024 read at 30 m) | one band of class codes per year, `CDL<year>` | categorical | 2008-present | Conterminous US; public domain ([below](#crop-type-map-sequences-cdl-sequence-aci-sequence)) | yes |
| AAFC Annual Crop Inventory, multi-year stack | `aci-sequence` | 30 m | one band of class codes per year, `ACI<year>` | categorical | 2011-present | Canada; Open Government Licence - Canada ([below](#crop-type-map-sequences-cdl-sequence-aci-sequence)) | yes |

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
- `uint8`: 8-bit digital numbers 0-255 (NAIP, USGS NAIP Plus). For
  `sentinel1`, radar backscatter mapped from fixed dB ranges to 1-255, with 0
  as nodata (see [Sentinel-1](#sentinel-1-sentinel1)); the reflectance
  conversions (`agribound.io.raster.to_unit_reflectance`, `to_s2_dn`) refuse
  it unless `value_scale` is passed.
- `unit`: Landsat PAN calibrated TOA reflectance, exported as float32 as stored.
- `dn`: SPOT 6/7 per-band medians (or greenest-pixel selections) of the
  scenes' raw digital numbers, written as float32 (a median of an even number
  of scenes can be a half-integer). Their radiometry has not been verified, so
  the reflectance conversions (`agribound.io.raster.to_unit_reflectance`,
  `to_s2_dn`) refuse them; engines that use a percentile stretch
  (Delineate-Anything, GeoAI, DINOv3, SAM) accept them. For `umbra`: the
  median of level-normalised 8-bit GEC digital numbers, an uncalibrated
  log-scaled radar amplitude (about 6.6 DN per dB), float32 with 0 as nodata.
- `db`: `capella`, float32 radar backscatter in decibels plus 50 (sigma0 dB
  + 50, so that valid pixels are positive), with 0 as nodata. The reflectance
  conversions refuse the SAR sources unless `value_scale` is passed.
- `embedding`: pre-computed embedding vectors.
- `categorical`: class codes of crop-type maps (`cdl-sequence`,
  `aci-sequence`, and local stacks tagged so), uint8 with 0 as nodata. The
  codes are nominal, so only the engines of
  `agribound.registry.CATEGORICAL_ENGINES` (`crop-sequence`) read them; the
  other engines, SAM refinement and fine-tuning refuse a raster whose
  `AGRIBOUND_VALUE_SCALE` tag is `categorical`, and the local builder keeps
  that tag when it crops such a file. The reflectance conversions
  (`agribound.io.raster.to_unit_reflectance`, `to_s2_dn`) take arrays, not
  rasters: they refuse the two crop-map sources and
  `value_scale="categorical"`.
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
| `sentinel1` | VV | VH | VV_VH | - | - | - | - |
| `umbra`, `capella` | MEDIAN | MEDIAN | MEDIAN | - | - | - | - |

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
(`("YYYY-MM-DD", "YYYY-MM-DD")`, end inclusive) replaces it. Scenes of
`landsat`, `landsat-pan`, `sentinel2`, `hls`, `spot` and `spot-pan` are first
filtered on their scene cloud-cover property (`cloud_cover_max`, default 20 %:
`CLOUDY_PIXEL_PERCENTAGE` for Sentinel-2, `CLOUD_COVER` for Landsat,
`CLOUD_COVERAGE` for HLS, `cloud_coverage_percentage` for SPOT), then masked
per pixel. `naip` and `sentinel1` ignore `cloud_cover_max`. Radar sees through
clouds, so `sentinel1` has no cloud filter or mask and also ignores
`s2_cloud_mask`; its scenes are selected by mode, resolution, polarisation and
orbit pass instead (see [Sentinel-1](#sentinel-1-sentinel1)). `umbra` and
`capella` (radar too) select collects by geometry
([below](#open-vhr-sar-umbra-capella)) and ignore both.

| Source | Pixel mask |
|---|---|
| `sentinel2` | `s2_cloud_mask="scl"` (default): SCL classes 3 (cloud shadow), 8 and 9 (cloud medium/high probability), 10 (thin cirrus). `s2_cloud_mask="cloud_score_plus"`: keeps pixels with Cloud Score+ `cs_cdf >= cloud_score_threshold` (default 0.60). |
| `landsat` | `QA_PIXEL` bits 0-4 (fill, dilated cloud, cirrus, cloud, cloud shadow). |
| `landsat-pan` | `QA_PIXEL` bits 0, 1, 3 and 4 (fill, dilated cloud, cloud, cloud shadow) on Landsat 7; bits 0-4 (also bit 2, cirrus) on Landsat 8/9. |
| `hls` | `Fmask` bits 1-3 (cloud, adjacent to cloud/shadow, cloud shadow). |
| `spot`, `spot-pan` | no pixel mask; scene filter only. |
| `sentinel1` | VV below -30 dB (border noise and invalid swath edges); no cloud mask. |
| `naip` | none (mosaic, see below). |

Masked pixels are NaN in the float32 composites and 0 in the uint8 ones.

`landsat` composites merge Landsat 5, 7, 8 and 9. Landsat 7 contributes for
windows between 1999-05-28 and 2024-01-19 (including its SLC-off stripes since
2003); there is no option to exclude it. `landsat-pan` chooses its missions
with `landsat_pan_missions` (see
[Landsat panchromatic](#landsat-panchromatic-landsat-pan)).

### Compositing methods

| `composite_method` | What it does |
|---|---|
| `median` (default) | Per-band median of the unmasked values. |
| `greenest` | Per pixel, the observation with the highest NDVI (`qualityMosaic`), computed from the source's canonical `NIR` and `R` bands. Not available for sources without both (`landsat-pan`, `spot-pan`, `sentinel1`, `umbra`, `capella`): the configuration is rejected with a `ValueError`. |
| `max_ndvi` | An alias of `greenest` (the same computation). |

NAIP is mosaicked, not composited, so only the default `median` is accepted
for `naip` (it is ignored).

### Dates in a composite

A `median` composite combines every image of the window that passes the scene
filter, per pixel and per band, so neighbouring pixels, and the bands of one
pixel, can come from different dates. The Sentinel-2 median of a 5 km square
at Twifo Praso, Ghana, for 2020 (`cloud_cover_max=80` and Cloud Score+
masking, as in example 23) is made from 52 images, one per date: each pixel
has 10 to 30 valid observations (median 20), and the first and the last of
them are a median 355 days apart. `greenest` keeps one
observation per pixel, but its pixels take them from many dates: 33 in the
same square.

#### What is recorded

Every Earth Engine composite lists the images it was made from in its tags,
which the provenance record copies (`facts["composite"]`):

| Tag | Content |
|---|---|
| `AGRIBOUND_IMAGE_LIST` | JSON list of `[image ID, UTC date, scene cloud %]`, sorted by date. The image ID is the Earth Engine asset ID, for example `COPERNICUS/S2_SR_HARMONIZED/20200804T101559_20200804T103905_T30NXM`; the scene cloud cover is the image property that `cloud_cover_max` filters on (`null` for `naip` and `sentinel1`). At most 1,000 images, the first by date. |
| `AGRIBOUND_IMAGE_DATES` | The number of distinct UTC dates of all the images. |
| `AGRIBOUND_IMAGE_DATE_FIRST`, `AGRIBOUND_IMAGE_DATE_LAST` | The first and the last of those dates. |
| `AGRIBOUND_IMAGE_LIST_TRUNCATED` | Only with more than 1,000 images: `first 1000 of <n> by date`. |

The list comes from the image-count request, so it costs no extra request,
and it lists the images that `AGRIBOUND_N_IMAGES` counts (for `sentinel1`,
one GRD slice per acquisition). Pixels, cache keys and configuration hashes
do not change; a composite cached by an earlier version is reused without
these tags (delete it to make it again with them). 1,000 Sentinel-2 rows take
about 90 KB.

`agribound composite --date-layers` (Python:
`agribound.build_date_layers(config)`) also downloads the composite's date
layers: a separate uint16 GeoTIFF on the composite's grid, cached next to it
as `<source>_date_layers_<key>.tif`.

| Band | Content |
|---|---|
| `n_valid` | The number of images with every band unmasked at the pixel. |
| `date_first`, `date_last` | The first and the last date of those observations, in days since 1970-01-01 (UTC). For `greenest`, both are the date of the observation the composite took. |

0 means no observation (nodata). Days convert with
`numpy.datetime64("1970-01-01") + days` or
`agribound.composites.scenes.day_to_date`, and the tags
`AGRIBOUND_DATE_ORIGIN`, `AGRIBOUND_DATE_LAYERS_RULE` and
`AGRIBOUND_COMPOSITE_CACHE_KEY` (the composite they describe) record this.
`date_first` and `date_last` bound the dates that enter a median; no single
date gives a median pixel its value. The layers cost one image-count request
and one download, are never a band of the composite (the engines read its
bands by position), and are available for `sentinel2`, `landsat`,
`landsat-pan`, `hls`, `spot` and `spot-pan` with `export_method="local"`. The
counts for the Twifo Praso median above come from them.

#### Single-date composites

`agribound scenes` (Python: `agribound.survey_scenes(config)`) lists the
acquisitions of the composite's window over the study area and ranks them by
the share of the study area that each shows clear under the composite's own
pixel mask. An acquisition is the images of one UTC date that pass
`cloud_cover_max`: in practice one satellite pass, whose granules or rows
count once. When two passes fall on one date, both are in one row, because a
one-day window composites them together: in the Landsat survey of the Twifo
Praso square for 2022 (`cloud_cover_max=80`), 8 of the 31 dates have two
images, Landsat 7 with Landsat 9 on seven of them and with Landsat 8 on one.

| Source | Ranked by | Ties |
|---|---|---|
| `sentinel2` | Clear share under `s2_cloud_mask` (SCL or Cloud Score+) | Higher mean Cloud Score+ `cs_cdf` |
| `landsat`, `landsat-pan`, `hls` | Clear share under the source's pixel mask | Lower scene cloud cover |
| `spot`, `spot-pan` | Share of the study area the scenes cover (no pixel mask) | Lower scene cloud cover |

Remaining ties go to the earlier date. Each row also gives the number of
images and their asset IDs, the scene cloud cover, the footprint share and,
for Sentinel-2, the SCL and Cloud Score+ clear shares and the mean aerosol
optical thickness (`AOT`) over the clear pixels, which is reported but not
ranked on. The survey selects the images as the composite does (for `hls`
with the footprint rules [above](#hls-hls)), reduces each date over the study
area at 60 m (coarser for study areas above 250,000 pixels; `--scale-m`),
makes one Earth Engine request and writes nothing (`--csv` writes the table,
`--json` prints it as JSON). A study area that still has more than 5,000
vertices once simplified (thousands of field polygons, for example) is
replaced by the outline of the export grid, and the default scale then keeps
that outline at 250,000 pixels or fewer. When Earth Engine runs out of memory
or time on that request, the survey takes the images selected for the whole
window month by month, one request per month, with the same rows (each date
is reduced on its own, and the `landsat-pan` missions are those of the
whole window): example 23's Sentinel-2 surveys at Madera, 2022 (177 images
on 96 dates, two on 81 of them), and Úbeda, 2023 (83 images on 46 dates),
took 13 requests each, and those at Twifo Praso (52 images, one per date)
and Oro (40) one. `naip` (a mosaic), `sentinel1` (no cloud filter
or mask), `umbra` and `capella` (they select their collects themselves) and
the sources not composited on Earth Engine are refused.

A one-day `date_range` (start = end) with the default `median` then builds
the composite of that acquisition alone. Masked pixels stay nodata, with the
usual WARNING below 95 % valid pixels (see [Valid-pixel
check](#valid-pixel-check)).

```bash
agribound scenes --study-area twifo.geojson --source sentinel2 --year 2020 \
    --cloud-cover-max 80 --s2-cloud-mask cloud_score_plus --gee-project my-project
agribound composite --study-area twifo.geojson --source sentinel2 --year 2020 \
    --cloud-cover-max 80 --s2-cloud-mask cloud_score_plus \
    --date-range 2020-08-04 2020-08-04 --gee-project my-project
```

```python
import agribound

table = agribound.survey_scenes(config)
one_date = config.merged(date_range=table.iloc[0]["date_range"])
agribound.build_composite(one_date)
```

The date is part of the configuration, so the configuration hash and the
cache key pin it. At Twifo Praso, the composite of
`date_range=("2020-08-04", "2020-08-04")` is identical, pixel for pixel, to
the single-acquisition composite of the pilot below. The first rows of the
survey of that square for 2020 (52 acquisitions):

| Rank | Date | Clear share | Mean `cs_cdf` | Mean AOT |
|---|---|---|---|---|
| 1 | 2020-08-04 | 1.000 | 0.903 | 0.06 |
| 2 | 2020-01-12 | 1.000 | 0.851 | 0.59 |
| 3 | 2020-01-27 | 1.000 | 0.679 | 1.18 |
| 4 | 2020-01-07 | 1.000 | 0.672 | 1.08 |
| 5 | 2020-04-21 | 0.982 | 0.882 | 0.17 |

The ranking follows the mask. SCL misses haze, and Cloud Score+ `cs_cdf` is
relative to each pixel's history, so hazy acquisitions can rank as clear:
ranks 2 to 4 above are fully clear under Cloud Score+, with a mean aerosol
optical thickness of 0.59 to 1.18.

#### What a single date does and does not do

A pilot (2026-10-09) compared composites as input for Delineate-Anything
(released weights, `max_detections=1000`) at four squares with reference
boundaries, with a pass rule fixed before any run:

- Twifo Praso, Ghana (5 km, 2020): 55 oil palm estate blocks, median 39.9 ha.
- Oro, Papua New Guinea (3 km, 2021): 302 oil palm smallholder parcels,
  median 1.53 ha.
- Chenderiang, Perak, Malaysia (3 km, 2024): 137 oil palm smallholder
  parcels, median 1.20 ha.
- Madera, California (5 km, 2022): 122 fields, 112 of them tree crops; 121 of
  the 122 are among Delineate-Anything's training labels (FBIS-73M).

Delineate-Anything ran on Sentinel-2 composites with example 23's settings at
the three tropical squares (Cloud Score+, `cloud_cover_max=80`) and the
defaults at Madera. The scores are F1 at IoU ≥ 0.5 with one-to-one matches,
as example 23 computes them (`F1_ov`). The RSPO maps of the tropical squares
do not cover every field, so there precision is counted over the predictions
that overlap a reference polygon (`count_tp / (count_tp + count_fp -
count_fp_unassigned)` of `evaluate()`). Madera's reference maps every field,
and its score is the F1 of the 112 tree-crop fields (`per_stratum`).
`evaluate()`'s own `f1`, whose precision counts every prediction, gives the
median and the single date 0.244 and 0.095 at Twifo Praso, and 0.112 and
0.083 at Chenderiang.

| Composite | Twifo Praso | Oro | Chenderiang | Madera |
|---|---|---|---|---|
| `median` (default) | 0.248 | 0.000 | 0.163 | 0.802 |
| Single clearest acquisition | 0.097 | 0.000 | 0.107 | 0.837 |
| Median of the 3 clearest acquisitions | 0.078 | 0.000 | 0.125 | 0.847 |
| Per-pixel Cloud Score+ quality mosaic | 0.167 | 0.000 | 0.037 | 0.539 |
| `greenest` | 0.356 | 0.000 | 0.061 | 0.717 |

- The median was a better Delineate-Anything input than the single
  clearest acquisition at both tropical squares where Delineate-Anything
  found fields: 0.248 against 0.097 at Twifo Praso and 0.163 against 0.107 at
  Chenderiang (boundary F-score within 20 m: 0.537 against 0.214 and 0.437
  against 0.350). At Oro it found no field on either. At Madera, inside its
  training data, the single date scored 0.837 against 0.802.
- Each of the four acquisitions fully clear over the Twifo Praso square in
  2020 scored lower than the median (0.000 to 0.105), and there the score
  rose with the number of images in the composite (one: 0.000 to 0.105;
  three: 0.078; 52: 0.248). The pilot did not find why: the minimum polygon
  area, holes in the composite and misregistration between dates (at most
  0.2 pixels) did not explain it, and the block boundaries were more visible
  on the single date than on the median.
- The per-pixel Cloud Score+ quality mosaic (each pixel from its clearest
  observation) scored lower than the median at the three squares where
  Delineate-Anything found fields, Madera included. agribound does not offer
  it.
- For scale, the same median at Twifo Praso scored 0.186 in 2021 against
  0.248 in 2020, and the clearest acquisition of 2021 gave no field.
- **Date seams.** On the default median, Delineate-Anything's lines were not
  nearer to the composite's date seams (where the number of valid
  observations changes by 3 or more between neighbouring pixels) than chance
  at Twifo Praso (1.21 times the share expected by chance, p = 0.12) or Madera
  (0.53 times); at Chenderiang they were (1.91 times, p = 0.03), over 0.27 km
  of 20.5 km of lines. On two composites with sharper seams at Twifo Praso, a
  QA60-masked median over 25 months (the recipe of the SNIC script credited
  at the end of this section) and the per-pixel quality mosaic, the lines
  were nearer to the seams (3.44 and 3.01 times, p = 0.01): an excess of 5 %
  and 12 % of the line length, mixed with leftover cirrus and haze.

A single date avoids mixing dates. It can show whether polygons follow
compositing artefacts, serve windows with land change such as replanting, or
feed methods tuned to one image; for Delineate-Anything at the tropical
squares the median remained the better input. Not tested: seams between
orbits (each tropical square lay in one Sentinel-2 tile and one relative
orbit), land change inside the window, filling the pixels a single date
masks from a second date, study areas that no single acquisition covers
clear, and other sensors.

Example 23's `--s2-single-date` option (off by default) repeats the
comparison with the released code: in each of its four areas it adds
Delineate-Anything on the Sentinel-2 composite of the acquisition the survey
ranks first, under the area's own mask (Cloud Score+ at Twifo Praso and Oro,
SCL at Madera and Úbeda), scored like the example's other runs. Where that
date has several granules over the square, the one-day composite is their
per-pixel median, while the pilot used one granule. At Madera on 2022-01-24
the composite is the median of tiles T10SGF and T10SGG (their mean where both
are clear), which differ by about 5 (reflectance × 10,000) on average
because Level-2A processing is done per tile; the pilot used T10SGF alone.
So the pilot's Madera scores are not expected exactly, also because the
pilot ran the Delineate-Anything backend of its day on CPU. Run with
agribound 1.1.0 on 2026-10-11, the single date scored `F1_ov` 0.095
(2020-08-04) against 0.244 for the median at Twifo Praso and 0 against 0 at
Oro (2021-08-21), and tree-crop F1 0.843 (2022-01-24) against 0.802 at
Madera and 0.057 (2023-12-19) against 0.159 at Úbeda: the median was the
better input at Twifo Praso and Úbeda, the single date at Madera.

#### If you composite in your own Earth Engine script

- **QA60.** In `COPERNICUS/S2_SR_HARMONIZED`, images of processing baselines
  04.00 and 05.09 have no QA60 cloud bits in their whole granules. At Twifo
  Praso, 5 of the 12 images of January 2021 to January 2023 with scene cloud
  cover below 25 % were such images, and 6 of the 12 had cloud over more than
  5 % of the square that QA60 did not mask. QA60 also missed the cloud of a
  baseline 03.01 image (2022-01-06, 64 % cloudy by Cloud Score+) and flagged
  cirrus over more than half of the square on 2021-12-22, which Cloud Score+
  rated clear. agribound masks with SCL or Cloud Score+, not QA60.
- **Projection and scale.** An `ImageCollection` reduction such as `median()`
  has no fixed projection (Earth Engine gives it EPSG:4326 at 1°), so
  scale-dependent steps such as SNIC run at whatever scale is requested: the
  zoom level of the Code Editor map, or the export scale. In the Twifo Praso
  square, SNIC with one setting gave 289 clusters on a 10 m grid and 72 on a
  20 m grid, and one SNIC output vectorised at 10, 20 and 30 m gave 584, 209
  and 107 polygons. agribound exports composites on a fixed UTM grid;
  reproject to a fixed grid before such steps.

#### Superpixel segmentation (SNIC, SLIC)

agribound has no SNIC or other superpixel engine. For agribound 1.1,
superpixel segmentation of the Sentinel-2 median was tested as a candidate
engine for oil palm estate blocks (2026-10-10). The squares, the settings and
a pass rule were fixed before any composite was built. No setting passed. The
results hold for these squares, this composite and these settings.

**What was tested.**

- **Squares.** Four 5 km squares where the RSPO GeoRSPO concession maps (the
  version example 23 uses) map an oil palm estate block by block: Western
  Region, Ghana; Morobe, Papua New Guinea; San Martín, Peru; and Petén,
  Guatemala.
- **How they were chosen.** A fixed rule took the square with the most estate
  blocks of 10-100 ha in each of Africa, Asia-Pacific and the Americas, then
  the best such square in a fourth country. A square qualified with at least
  15 blocks of one RSPO member, and that member's blocks had to cover at least
  half of it. Only 14 squares met the rule, from five estates in four
  countries. Twifo Praso (example 23) was left out, because its blocks set the
  segment size. 90 % of the Ghana square lies inside the area on which
  example 23 fine-tunes Delineate-Anything; only the released weights were run
  there.
- **Imagery.** agribound's Sentinel-2 median of 2025 with example 23's
  settings (`cloud_cover_max=80`, Cloud Score+ masking at 0.60), on the 10 m
  UTM grid of each square: 46 to 130 images, every pixel valid.
- **SNIC, size 30.** SNIC in Earth Engine with size 30, compactness 0.1 and
  8-connectivity (seeds 300 m apart, segments of about 9 ha), on the 12 bands
  as reflectance. This is the setting of the oil palm SNIC script credited
  below, run on agribound's composite and grid.
- **SNIC, size 63.** The same with size 63 (seeds 630 m apart, about 40 ha).
  The spacing is the median block of Twifo Praso, 39.9 ha.
- **SLIC.** scikit-image 0.26.0 with the same 630 m spacing (63 segments per
  square), on the downloaded composite.
- **Comparison.** Delineate-Anything v2 as released (`max_detections=1000`, no
  crop filter, on CPU) on the same composite. Its training data (FBIS-73M)
  have no patches in Ghana or Papua New Guinea; the Peru and Guatemala squares
  were not checked.
- **Scoring.** F1 at IoU ≥ 0.5 with one-to-one matches, against all of the
  estate's polygons of 0.25-100 ha in the square (40 to 162). Precision is
  computed over the polygons that overlap a reference polygon, because the
  maps do not cover every field: the reference polygons cover 55-88 % of each
  square. Polygons under 2,500 m² and polygons whose representative point lies
  outside the square are dropped.
- **Chance baselines.** Two baselines were scored like a segmenter: the
  reference shifted east and north by half its median block width (128 to
  322 m), and an image-blind square grid with cells of the area of the
  segmenter's median polygon (the mean over 36 offsets).
- **Pass rule.** A setting passes a square if its F1 is at least 0.10 above
  both baselines and no more than 0.05 below Delineate-Anything's. A setting
  needed passes at 3 of the 4 squares.

**What was found.** The table gives F1 at IoU ≥ 0.5. Brackets give the score
of the same segmenter's image-blind grid, and bold marks a pass. The shifted
reference scored 0 at every square.

| Square (2025) | Reference polygons, median area | SNIC, size 30 | SNIC, size 63 | SLIC | Delineate-Anything |
|---|---|---|---|---|---|
| Western Region, Ghana | 40, 40.2 ha | 0.016 (0.002) | **0.192** (0.013) | **0.304** (0.095) | 0.025 |
| Morobe, Papua New Guinea | 162, 9.2 ha | **0.137** (0.022) | **0.123** (0.020) | 0.010 (0.000) | 0.012 |
| San Martín, Peru | 99, 25.7 ha | 0.128 (0.009) | 0.144 (0.011) | 0.026 (0.004) | 0.894 |
| Petén, Guatemala | 61, 25.8 ha | 0.021 (0.008) | 0.031 (0.042) | 0.107 (0.053) | 0.057 |
| Mean | | 0.076 | 0.123 | 0.112 | 0.247 |
| Squares passed (3 needed) | | 1 | 2 | 1 | |

- **Passes.** All four were at the two squares where Delineate-Anything itself
  scored below 0.05, so the comparison with it cost almost nothing there. In
  Ghana, Delineate-Anything's polygons were much smaller than the blocks
  (median 3.1 ha against 40.2 ha); in Papua New Guinea it returned 7
  polygons. The passing settings matched more polygons than
  Delineate-Anything at these two squares (exact McNemar tests, p ≤ 0.008).
  Even so, they matched at most 12 of the 40 polygons in Ghana (SLIC) and 30
  of the 162 in Papua New Guinea (SNIC, size 30).
- **Peru.** The reference is regular: 72 of the 99 polygons are rectangles
  (they fill at least 95 % of their minimum rotated rectangle), and 57 have the
  same area, 25.65 ha. At the other three squares, 0 to 7 polygons are
  rectangles by this measure. Delineate-Anything matched 89 of the 99
  polygons. SNIC with size 63 matched 13, although its median segment
  (25.7 ha) was as large as the median block: for 86 % of the polygons, the
  segment that covers most of a polygon also covers at least a quarter of
  another one. SNIC with size 30 split 80 % of the polygons: at least two
  segments each covered 10 % or more of a polygon.
- **Guatemala.** SNIC with size 63 scored below its image-blind grid (0.031
  against 0.042). Delineate-Anything's polygons were again much smaller than
  the blocks (median 2.4 ha against 25.8 ha).
- **Segments did not follow block outlines at either spacing.** Over the four
  squares, SNIC with size 30 (median segment 6.4-7.0 ha) split 71-98 % of the
  reference polygons. SNIC with size 63 (16-26 ha) merged 58-93 % of them
  with a neighbour and still split 31-79 %, and SLIC (43-48 ha) merged
  70-99 %. The median reference polygon was 9.2 ha in Papua New Guinea, where
  86 of the 162 are under 10 ha, and 25.7-40.2 ha at the other squares.
- **Outside the rule.** Without the comparison with Delineate-Anything, SNIC
  with size 63 would have cleared the chance baselines at 3 of the 4 squares
  (Ghana, Papua New Guinea and Peru), with F1 0.12-0.19. SNIC on a 20 m grid
  with size 32 (seeds 640 m apart), reported only, scored 0.048-0.194 (mean
  0.105).
- **Twifo Praso** (example 23's 2020 composite). It was not counted, because
  its blocks set the 630 m spacing. SNIC with size 63 scored 0.224, against
  Delineate-Anything's 0.248. SNIC with size 30 scored 0.029 and SLIC 0.051.

**Earlier pilot.** A pilot (2026-10-09) ran the size-30 setting on the same
kind of Sentinel-2 median at three squares:

- Twifo Praso (2020);
- oil palm smallholder parcels at Oro, Papua New Guinea (2021);
- oil palm smallholder parcels at Chenderiang, Malaysia (2024).

SNIC scored 0.027, 0.023 and 0.000, against Delineate-Anything's 0.248, 0.000
and 0.163. It split 98 % of the Twifo Praso blocks. A boundary score with a
20 m tolerance could not separate SNIC from an image-blind grid at the two
smallholder squares, where 77-81 % of the scored area lies within 20 m of a
reference line.

**Not tested.**

- **Single-date composites as input.** The pilot gives one result, at Twifo
  Praso. There, SNIC with size 30 ran on a 20 m grid (600 m seeds), a scale
  chosen after seeing the blocks. It matched more of the 55 blocks on the
  clearest 2020 date than on the 2020 median (21 against 12), but not in 2021
  (18 against 17).
- Compactness values other than 0.1.
- Merging segments into fields.
- Other sensors and resolutions.
- Other years at the four squares.
- SNIC on other inputs, such as stacks of yearly crop-type maps.

**SNIC in your own Earth Engine script.**

- Reproject the input to a fixed grid first (see
  [above](#if-you-composite-in-your-own-earth-engine-script)).
- Choose `size` from that grid's pixel size and the expected block size.
- With `connectivity: 8`, vectorise with `eightConnected: true`. In the Twifo
  Praso square, 289 SNIC clusters gave 584 polygons with `false` (276 of them
  under 0.25 ha) and 290 with `true`.
- Score the segments at object level, not only with a boundary tolerance.
  Compare them with a shifted copy of the reference and with an image-blind
  grid whose cells have the segments' size; `agribound.evaluate.chance_baselines`
  scores both (see [Chance baselines](evaluation.md#chance-baselines)).

Thanks to Jacob Abramowitz (The University of Alabama in Huntsville), who
raised the question of date mixing in tropical composites and shared his SNIC
script (its size-30 setting is the one tested above), and to Jacob T. Stid
(Michigan State University) and Jeremy Rapp, who shared the crop-sequence
recipe that the crop-sequence route follows (see
[Crop-sequence zones](engines.md#crop-sequence-zones-crop-sequence)).

### NAIP (`naip`)

- Only 4-band (R, G, B, N) images are used; some early years are RGB only and
  are skipped.
- Without `date_range`, images from `year - 1` to `year + 1` are mosaicked,
  exact-year images on top and the newest image on top within each group, so
  gaps in the requested year are filled from the neighbouring years. When
  neighbouring-year images intersect the study-area extent, one more Earth
  Engine request finds where they show, by image footprint: a WARNING names
  the years whose images fill more than one export pixel and their share of
  the extent; the others are logged at INFO as lying under the `year`
  images. `AGRIBOUND_IMAGE_YEARS` lists the years whose images show in the
  mosaic and `AGRIBOUND_IMAGE_YEARS_UNDERNEATH` the others. A footprint
  that holds no data would count as covered. Checked against the valid
  pixels: in the 7 squares of example 25 with neighbouring-year images, the
  `year` images covered every 2 m pixel, as their footprints said; in a
  4.1 × 5.6 km box across the eastern edge of Oregon's 2022 images, the 2023
  images of Idaho filled 65.1 % of the extent by footprint and 65.0 % of the
  10 m pixels.
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

### HLS (`hls`)

`hls` composites merge `NASA/HLS/HLSL30/v002` and `NASA/HLS/HLSS30/v002`
(bands and masks above).

**Footprints at 180°.** In Earth Engine (checked 2026-10-09), 10,324 HLSS30
images of 31 of the MGRS tiles that straddle 180° (acquired 2015-12-22 to
2025-05-30) have a `system:footprint` ring that runs clockwise. Earth Engine
reads such a ring as the whole globe except the tile, so `filterBounds` with a
polygon returns these images everywhere except over their own tile. HLSL30 has
no such image; nor had the Landsat 7/8/9, Sentinel-2, Sentinel-1 and SPOT 6/7
collections in the years checked (2022 and 2024). agribound
(`agribound.composites.gee`):

- excludes the images whose footprint meets one of two points on the far side
  of the Earth: the antipode of the centroid of the export grid's outline, and
  that point moved 90° east (`hls_far_points`, `hls_footprint_filter`). They
  are at least 10,007 km from the centroid, so the footprint of an image that
  covers the study area meets neither, and the tile of a reversed footprint
  cannot hold both;
- when the outline comes within 5° of 180° (`HLS_ANTIMERIDIAN_MARGIN_DEG`;
  every affected tile reaches at most 3.0006° from it), lists with one request
  the images of those tiles that have such a footprint and whose footprint
  ring, read with the even-odd rule, meets the outline, and loads them by ID
  (`hls_antimeridian_images`; their pixels, taken from the collection, depend
  on the extent of each pixel request). HLSL30 is listed as well, although it
  had no such image, because the exclusion above applies to it too. The tags
  `AGRIBOUND_HLSS30_IMAGES_BY_ID` and `AGRIBOUND_HLSL30_IMAGES_BY_ID` give
  their numbers;
- records the change in the composite cache key (tag
  `AGRIBOUND_HLS_RECIPE_VERSION`, `"2"`), so cached `hls` composites are made
  again once. The configuration hash and the results versions are unchanged.

What this changes:

- **Away from 180°** the excluded images hold no pixel in the export grid, so
  the composites are unchanged, but they were counted in `AGRIBOUND_N_IMAGES`,
  `AGRIBOUND_IMAGE_YEARS`, the image-count log line and the years listed by the
  no-image error. For example 03's Murray-Darling box (2022) the count falls
  from 298 to 77 images (20 HLSL30 and 57 HLSS30, all of tile T55JGG); for a
  5 km square at Twifo Praso (2022, cloud cover ≤ 80 %) from 713 to 45. The
  composites of five such configurations, including those of examples 03 and
  14, are identical before and after, pixel for pixel. The runs of examples
  03 and 14 with agribound 1.1.0 (2026-10-11) recorded 77 and 10 images
  (against 298 and 21 in the development run of 2026-10-09), and example 03's
  three outputs were the same as before. A window whose only images were
  reversed footprints (Lea County, 2022-10-13 to 2022-10-31, cloud
  cover ≤ 20 %: seven such images) now raises the no-image error instead of
  downloading a composite without valid pixels.
- **Inside an affected tile** the composites lacked the tile's affected
  images. In a 5 km square in Fiji (2022, cloud cover ≤ 80 %) they are 29 of
  the 45 HLSS30 images of tile T01KAA: the selection now holds 206 images
  instead of 816 (639 from other tiles), the mean number of observations per
  pixel rises from 90.2 to 106.5, and every pixel changes. Where a study area
  straddles the edge of an affected tile, these images were selected and
  their pixels were used, depending on the extent of the download requests.
- An existing output is still reused (its configuration hash and results
  versions are unchanged), except one made before this change at the
  affected tiles: its provenance record has no `AGRIBOUND_HLS_RECIPE_VERSION`
  among the composite's tags, and its study area lies within 5° of 180° and
  within 0.5° of the latitudes of those tiles (50.6° S to 46.9° S, 20.0° S to
  9.0° S, 50.4° N to 52.3° N and 62.1° N to 72.1° N;
  `near_hls_reversed_footprint_tiles`). Such an output raises
  `FileExistsError` instead (pass `overwrite=True`, CLI `--overwrite`), and
  its HPC tile shows as `stale`. The study area is read without Earth Engine;
  a GEE-asset study area without its local copy is reused with a WARNING.
- HPC tiles whose `hls` composite was staged before this change show as not
  staged (`pending`), because the stage markers of `hls` tiles include
  `HLS_RECIPE_VERSION`: the delineation stage reads the composite under its
  new cache key. Stage them again on a node with internet access
  (`agribound tiles run --stage composite`, or `submit_region.sh --resume`)
  before `--stage delineate`. Tiles recorded as having no data are tried
  again.
- The agent's `check_availability(live=True)` counts HLS images by the same
  rules, over the study area's bounding box.

**Grids that reach across 180°.** The export grid of a study area that ends
at 180°, or a little short of it, can reach across 180°: its edges are
rounded outwards to whole pixels, and away from the equator the edges of a
UTM grid are not meridians. How far short depends on the latitude and on the
study area's north-south extent: 0.2 km for a 0.1°-tall area at 17° S,
3.4 km for a 2°-tall one, 1.4 km at 52° N, 5.4 km at 71° N, 10 km for a
2°-tall area at 64° N to 66° N. The outline of such a grid is now split at
180° before it selects images, for every Earth Engine source (`ee_geometry`,
`split_at_antimeridian`). As one polygon, its longitudes ran from −180° to
180°, and `filterBounds` selected images all around the globe at that
latitude: for a 5 km square that ends at 180° in Fiji (2022, cloud cover
≤ 80 %), agribound 1.0.1 selected 20,806 HLS images; the 76 of tile T01KAB
are selected now. The composites of the other sources cached before for such a grid
are reused with their old tags (image counts and years that include those
images; for `sentinel1` the orbit pass was chosen from those counts); delete
them to make them again.

Study areas that cross 180° are still not supported: with longitudes in
−180° to 180°, their outline spans the globe, and so does their export grid.

### SPOT 6/7 (`spot`, `spot-pan`)

!!! warning "Restricted source"
    The SPOT 6/7 collection (`AIRBUS/SPOT6_7`) is not in the public Earth
    Engine catalogue. Access is limited to select Earth Engine users; in
    agribound it is for internal DRI use. Selecting `spot` or `spot-pan` emits
    a `UserWarning`. External users who need SPOT-based field boundaries
    should contact the package author.

`spot-pan` writes the single panchromatic band; engines that need R, G, B read
it three times.

### Sentinel-1 (`sentinel1`)

`source="sentinel1"` builds a median composite of Sentinel-1 C-band radar
backscatter from the Earth Engine collection
[`COPERNICUS/S1_GRD`](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD):
ground range detected (GRD) scenes that Earth Engine has processed to sigma0
in dB (orbit file, border-noise removal, thermal-noise removal, radiometric
calibration, and terrain correction with SRTM 30 m, or ASTER above 60°
latitude; see the
[Sentinel-1 guide](https://developers.google.com/earth-engine/guides/sentinel1)).
Earth Engine applies no speckle filter and no radiometric terrain flattening,
and neither does agribound.

**Scenes.** The composite uses the images over the study-area extent in the
date window that have `instrumentMode` IW (Interferometric Wide swath),
`resolution_meters` 10 and both VV and VH in
`transmitterReceiverPolarisation`. Earth Engine stores each acquisition (one
pass of one satellite over the area) as GRD slices of about 25 s, so a study
area that crosses a slice boundary has two or more images of one acquisition.
agribound therefore counts acquisitions: images with the same
`platform_number` and `orbitNumber_start` (the absolute orbit) count once,
while the composite uses every slice. Of the images it uses one orbit pass:
the one (`ASCENDING` or `DESCENDING`) with more acquisitions, `DESCENDING` on a
tie. Ascending and descending acquisitions view the ground from opposite
sides, at different incidence angles and times of day, so mixing them could
create seams that look like edges. One pass can still combine several
relative orbits, which view the area at different incidence angles, and no
incidence-angle normalisation is applied, so the brightness can change where
the swath of one relative orbit ends inside the study area. The pass with more
acquisitions need not cover the whole study area; pixels without an
acquisition of that pass are nodata (see `AGRIBOUND_VALID_FRACTION`). Each
tile of a tiled run ([HPC and large areas](hpc.md)) chooses its own pass and
records it in its composite's `AGRIBOUND_ORBIT_PASS`, so neighbouring tiles
can use different passes. There is no cloud filter:
`cloud_cover_max` and `s2_cloud_mask` are ignored, although they still enter
the cache key and the configuration hash.

**Pixels.** In each image, pixels with VV below -30 dB are masked in every
band (border noise and invalid swath edges, the mask of the Earth Engine
catalogue example), and `VV_VH` = VV − VH in dB (the VV/VH ratio of the
image in dB) is added. Each band is then mapped from a fixed dB range
`(lo, hi)` to 1-255:

`value = 1 + round(254 × clamp((dB − lo) / (hi − lo), 0, 1))`

| Composite band | Canonical band | dB range (lo to hi) | Step |
|---|---|---|---|
| 1 `VV` | R | -25 to 0 | 0.098 dB |
| 2 `VH` | G | -32 to -8 | 0.094 dB |
| 3 `VV_VH` (VV − VH) | B | 0 to 15 | 0.059 dB |

Values beyond a range become 1 or 255, and 0 is nodata. The per-band median of
the scaled images is rounded to whole values (halves up) and exported as
uint8. Because the scaling keeps the order of the values, the median of an odd
number of images is exactly the scaled median of the dB values; the median of
an even number is the mean of the two middle values, rounded, and can differ
from it by one step. `agribound.composites.gee.sentinel1_uint8_to_db`
converts composite values back to dB (`dB = lo + (value − 1) × (hi − lo) / 254`;
pass it the composite's `AGRIBOUND_S1_DB_RANGES` tag).

**Composite tags.** Besides the tags of every composite, a `sentinel1`
composite records (the provenance record keeps them under `facts.composite`):

| Tag | Content |
|---|---|
| `AGRIBOUND_ORBIT_PASS` | the orbit pass used |
| `AGRIBOUND_ORBIT_PASS_ACQUISITIONS` | the number of acquisitions of each pass after the scene filters, e.g. `ASCENDING:14,DESCENDING:31` |
| `AGRIBOUND_SENSORS`, `AGRIBOUND_SENSOR_IMAGES` | the satellites (`S1A` to `S1D`, from `platform_number`) with acquisitions in the composite, and the number of acquisitions of each |
| `AGRIBOUND_IMAGE_COUNTS` | the note that these counts and `AGRIBOUND_N_IMAGES` count acquisitions, not GRD slices |
| `AGRIBOUND_S1_DB_RANGES` | the dB range of each band, as JSON: `{"VV": [-25.0, 0.0], "VH": [-32.0, -8.0], "VV_VH": [0.0, 15.0]}` |
| `AGRIBOUND_RGB_MAPPING`, `AGRIBOUND_SCALING`, `AGRIBOUND_EDGE_MASK` | the band roles, the scaling formula and the edge mask |
| `AGRIBOUND_INSTRUMENT_MODE`, `AGRIBOUND_POLARISATION`, `AGRIBOUND_CLOUD_FILTER` | `IW`, `VV+VH`, and the note that no cloud filter is applied |
| `AGRIBOUND_S1_RECIPE_VERSION` | the version of this recipe, part of the composite cache key |

`AGRIBOUND_N_IMAGES` is the number of acquisitions of the pass used.
`AGRIBOUND_CLOUD_COVER_MAX` is written, as for every composite, but not
applied.

**Satellites.** In Earth Engine (checked 2026-10-07), Sentinel-1A images run
from 2014-10-03 to 2026-06-29, Sentinel-1B images from 2016-09-26 to
2021-12-23 (the satellite failed), Sentinel-1C images start on 2025-03-27 and
Sentinel-1D images on 2026-04-15. ESA's Sentinel-1C products have preliminary
calibration until 2025-05-19 and are fully calibrated from then on. Each
satellite repeats its orbit every 12 days, but how often an area is imaged
depends on the mission's observation plan. Where both satellites imaged each
track, windows from 2022 to early 2025, when Sentinel-1A flew alone, have
about half the acquisitions of 2017-2021 windows; elsewhere they have about
as many. In boxes of 0.09° × 0.09° (2026-10-07), a calendar-year composite
would use 118 acquisitions in 2021 and 61 in 2023 at Beauce (France), and 108
and 57 in the Pampas (Argentina), but 59 and 47 in Iowa, 60 and 61 in the
Punjab (India) and 31 and 30 at Namoi (Australia). 2015, and in some regions
2016, are sparse: a 2015 composite would use 1 acquisition in the Pampas, 2 in
the Punjab, 3 in Lea County (New Mexico) and 4 in central Kenya.
`AGRIBOUND_N_IMAGES` gives the number a composite used, and the agent's
`check_availability(live=True)` counts them before a run. Earth Engine cannot
remove thermal noise from images produced before July 2015, so a 2015 window
can include images without it, and its guide lists GRD border-noise removal as
of 2018-01-12, so earlier images can keep border noise that the edge mask does
not remove.

**Engines.** Delineate-Anything, GeoAI and DINOv3 read VV, VH and VV_VH as R,
G and B, unchanged (uint8 input is not stretched), and so does SAM
refinement. The SAM 2.1 automatic-mask engine `sam-auto` reads them the same
way but has no default setting for `sentinel1`: it runs only with all four
of its per-source settings given in `engine_params` (see [SAM 2.1 automatic
masks](engines.md#sam-21-automatic-masks-sam-auto)). GeoAI's Mask R-CNN
(COCO) and DINOv3's SAT-493M backbone were pre-trained on optical imagery,
so for them these composites are out of distribution. Radar is not new to
Delineate-Anything v2: the training list of
FBIS-73M, its training data (`train.txt` of the Hugging Face dataset
`MykolaL/FBIS-73M` at revision 9e4cdbd, checked 2026-10-08), names 76,448
Sentinel-1 patches (`*_S1_10m_256_custom`) and 7,852 patches named
`*_SYN-S1_256_custom` among its 1,432,187 (5.9 % together). How those patches
render the radar bands is not documented, so agribound's composites (VV, VH
and VV − VH on fixed dB ranges) may still differ from them. The published
training list of FBIS-22M, the data of the v1 models (`train.txt` of
`MykolaL/FBIS-22M` at revision 8713461, checked 2026-10-08), names no
Sentinel-1 patch: by their names, its 599,178 patches are Sentinel-2,
PlanetScope, Maxar and orthophoto images (its dataset card says that Pleiades
images were left out of the published data). All the same, zero-shot scores
were low in a smoke test on two 5 km squares: Delineate-Anything v2 as
released (default settings, without the LULC crop filter) scored F1 0.18 at
IoU ≥ 0.5 in Lea County, New Mexico (2022, against the NMOSE WUCB polygons,
the New Mexico reference before V18; 9 of the 25 fields matched, precision
0.12), and 0.39 near Davis, California (2024, against the provisional DWR
2024 crop map; 102 of the 292 fields matched, precision 0.45; FBIS-73M uses
DWR polygons as its California labels, so this is not a held-out result).

A pilot (2026-10-08) ran the same model and settings on `sentinel1`
composites of example 23's four squares (with `max_detections=1000`, as
example 23 sets for its other 10-15 m sources), of 20.5 km² near Davis that
lie inside the Davis smoke square, and of a 415 km² box of eastern Lea
County that contains the Lea County smoke square, with reference fields
under 2,500 m² dropped:

| Area (year) | Reference | Evaluated area | F1 |
|---|---|---|---|
| Madera County, California (2022): orchards | DWR / Land IQ, tree-crop fields | example 23's 5 km square | 0.80 |
| Near Davis, California (2024) | provisional DWR 2024 crop map | 20.5 km² | 0.455 |
| Eastern Lea County, New Mexico (2024) | DRI/NMOSE field boundaries (V18) | 415 km² (the box of examples 12 and 14) | 0.173 |
| Twifo Praso, Ghana (2020): oil palm estate | RSPO blocks | example 23's 5 km square | 0.11 |
| Úbeda, Spain (2023): olive groves | SIGPAC, tree-crop recintos | example 23's 4 km square | 0.06 |
| Oro, Papua New Guinea (2021): oil palm smallholders | RSPO parcels | example 23's 3 km square | 0.00 |

The scores in example 23's squares follow its rules: the tree-crop fields at
Madera and Úbeda, and at Twifo and Oro a precision that counts only the
predictions overlapping a reference polygon, because the RSPO maps do not
map every field (see `examples/23_tree_crops.py`); the other two are plain
F1. On Madera's large, regular orchard blocks Sentinel-1 scored about as well
as Sentinel-2: in the development run of agribound 1.1 (2026-10-10), with one
backend for both, example 23's Sentinel-1 runs gave the pilot's scores in
the four squares (0.800, 0.059, 0.106 and 0.000 at Madera, Úbeda, Twifo
Praso and Oro) and its Sentinel-2 run 0.802 at Madera. The California and
Spanish scores are not held out: FBIS-73M
uses DWR polygons as its California labels and SIGPAC polygons in Spain, and
its training list holds Sentinel-1 patches in or near these areas: 2 that
intersect the Davis area and about 38 within 10 km of it, and 18 within 10 km of the
Madera square (none within 10 km of the Lea County box; Spain was not
checked; FBIS-73M has no patches in Ghana or Papua New Guinea).
`engine_params["stretch_percentiles"]` has no effect on this uint8
composite, which Delineate-Anything reads unchanged; a float copy of the
composite stretched at (0.1, 99.9) scored lower in 4 of the 6 areas (Twifo,
Madera, Úbeda, Davis), higher in Lea County (0.24 against 0.17) and the same
at Oro. The agent's `recommend_configurations` ranks `sentinel1` after every
other source. Where reference boundaries exist, fine-tune Delineate-Anything,
GeoAI or DINOv3 on Sentinel-1 composites (GeoAI and DINOv3 need a fine-tuned
checkpoint anyway); agribound does not fine-tune SAM. Delineate-Anything's
0.25-10 m training-range check sees the 10 m pixels and does not warn,
although the resolution of IW GRD images is about 20 × 22 m. FTW and Prithvi
need optical reflectance and do not support this source, and
`engine="ensemble"` needs explicit members (`engine_params["engines"]`)
because its default members include FTW. Single images are speckled (grainy);
the median of many images reduces the speckle, so prefer windows with many
acquisitions. On speckled radar images neighbouring detections can overlap
across tile edges, and Delineate-Anything's `merge_tile_pieces` (native
backend, default True) can then chain tile-cut pieces into very large
polygons: in a test on 1 m single-look SAR images (not an agribound source),
689 of 770 detections were joined into 13 polygons, the largest 1,619 ha. On
Sentinel-1 composites it did not happen: on the two smoke squares the largest
polygons were 50 ha (Lea County) and 42 ha (Davis), and in the six pilot
areas, up to 415 km², `merge_tile_pieces=False` changed F1 by at most 0.005
at 10 m. Still, look at the polygon sizes of a Sentinel-1 run, and set
`engine_params={"merge_tile_pieces": False}` if pieces chain.

### Open VHR SAR (`umbra`, `capella`)

`source="umbra"` and `source="capella"` build an amplitude composite from the
open very-high-resolution (VHR) X-band SAR archives of the AWS Registry of
Open Data: the [Umbra Open Data Program](https://registry.opendata.aws/umbra-open-data/)
(`s3://umbra-open-data-catalog`) and
[Capella Space Open Data](https://registry.opendata.aws/capella_opendata/)
(`s3://capella-open-data`). They do not use Earth Engine: agribound lists the
bucket, reads the per-collect metadata, downloads the amplitude products over
anonymous HTTPS and builds the composite locally
(`agribound.composites.vhr_sar`). Every collect has one polarization (HH or
VV), so these sources carry no polarimetry. Complex and derived products
(CPHD, SICD, SLC, SIDD, CSI) are never downloaded. The LULC crop filter (on by
default) still reads Earth Engine.

**The archives** (bucket listings of 2026-10-07 to 2026-10-09):

| | `umbra` | `capella` |
|---|---|---|
| Licence | CC BY 4.0 ([Umbra open licence](https://umbra.space/open-license)) | CC BY 4.0 ([Capella data licensing](https://www.capellaspace.com/legal/data-licensing)); users must follow US sanctions and export rules |
| Credit | Umbra publishes no credit line; agribound writes "SAR data: Umbra Lab, Inc., CC BY 4.0" | "© Capella Space Corp, All Rights Reserved." plus a note of the changes (in the composite's tags); the gallery's figures print "© Capella Space Corp, CC BY 4.0, modified" |
| Amplitude product | GEC: 8-bit digital numbers on a rotated EPSG:4326 grid with RPCs; not terrain-corrected; uncalibrated (the DN behaves as a log scale, about 6.6 DN per dB, and DN 1 holds 23-40 % of the pixels) | GEO: UInt16 amplitude, sigma0 dB = 20 log10(scale_factor × DN), terrain-corrected with a filtered 30 m AW3D30 DSM, north-up UTM. About 19 % of the products are not fully calibrated (`"limited"` or `"partial"`) |
| Collects | 17,810 GEC folders (VV in 14,416, HH in 3,388); 10,212 collects after merging copies, from January 2023 (5,209 folders without a `.stac.v2.json` sidecar are not used) | 2,291 collects from 2021-02-08 to 2026-09-09 (HH 2,094, VV 197), with a GEO product in 2,286 |
| Scenes | spotlight, about 5 × 5 km, 0.16-1 m resolution; a GEC is 0.04-1.2 GB (5-95 %; median 0.28 GB) | spotlight, sliding spotlight and stripmap (median GEO pixel 0.34, 0.60 and 0.84 m); a GEO is 0.25-1.7 GB (median 0.53 GB) |
| Farmland | 166-204 collects each at UC Davis (CA), Purdue (IN), Texas A&M, NDSU, Johnston (IA) and Centerfield (UT); most sites have few dates | no long series in one geometry: 89 % of the collects lie at sites with under 10 % cropland |

Neither archive covers most study areas; check one with
`check_availability(live=True)` in the agent, or run the composite stage
(`agribound composite`), whose error lists the nearest dates.

**Index.** The first run lists the bucket and reads the metadata into an
index cached as `<vhr_sar_cache_dir>/<source>/index.json.gz` (default
`~/.cache/agribound/vhr_sar`); on 2026-10-09 the Umbra index read 95,520 keys
and 12,601 sidecars in 35 s and the Capella index 24,816 keys and 2,286 GEO
items in 17 s. Umbra copies of one collect (task and open-data folders) are
merged by `umbra:collect_id`, keeping the copy with the newest GEC, whose GEC
and sidecar come from the same folder; two distinct collects seconds apart
stay separate. Polarization and geometry come from the `.stac.v2.json`
sidecar: `METADATA.json` disagreed with it for 36 of 204 collects at
Centerfield, so folders that have only `METADATA.json` are not used. Collects
dated after the index snapshot are dropped (the Umbra listing holds 8 dated
2035 and 2036). Capella records take the incidence angle at full precision
from the head of each GEO `_extended.json` (the STAC items round it to 0.1°),
with the calibration label and the scale factor. The snapshot time is
recorded (`AGRIBOUND_INDEX_SNAPSHOT`) and enters the composite's cache key;
the selection is deterministic for one snapshot. A cached index is refreshed
(incrementally: only new or changed metadata files are read) when the date
window ends after the snapshot and the snapshot is more than a day old;
otherwise call `agribound.composites.vhr_sar.refresh_index(source)` or delete
the file to pick up newer uploads. To reproduce a composite exactly, pin its
collects (`vhr_sar_collect_ids`, listed in `AGRIBOUND_SAR_COLLECTS`).

**Selection** (`agribound.composites.vhr_sar.select_collects`):

1. collects in the date window whose footprint covers the study area (at
   least 98 % of it: the metadata footprints are approximate, and the
   prototype's evaluation areas, drawn inside the image data, were 99.0-99.99 %
   inside the footprints of the collects it used);
2. grouped by orbit pass, look side and polarization, so HH and VV, and the
   two look sides, are never mixed (`vhr_sar_polarization` keeps one);
   with `vhr_sar_calibration="full"`, Capella collects whose calibration is
   `"limited"` or `"partial"` are dropped first;
3. in each group, the largest subset within 5° of incidence and 25° of view
   azimuth (measured around the circle, so 355° and 5° are 10° apart); the
   group with the largest such subset wins (ties: smaller spreads, then the
   longer time span);
4. up to `vhr_sar_max_collects` (default 6) of its collects, spread evenly
   over time (always the first and the last).

The reference date for co-registration is the chosen collect closest to the
median incidence and the middle of the dates. `vhr_sar_collect_ids` replaces
steps 1-4 by the given collects, which must still share orbit pass, look side
and polarization, lie within the two tolerances, cover the study area and fall
in the date window. On the four sites of the Umbra prototype (its evaluation
areas and date windows) the automatic selection picks the prototype's own
collects at Centerfield (4) and Purdue (6) with the same reference dates, the
two VV dates on Mallorca (the prototype also used an HH date), and at UC Davis
the prototype's geometry group (9 consistent collects), from which it takes 6
(the prototype took 4 within its download budget).

**Orthorectification.** Umbra GECs are warped with their RPCs over the
Copernicus GLO-30 DEM (public COG tiles on AWS; GLO-90, with a WARNING, for the
1° tiles GLO-30 does not publish, which include Armenia and Azerbaijan, where
Umbra has collects; a tile missing from both is taken as sea, height 0, with a
WARNING; the manifest records the source of every tile), converted from EGM2008
heights to ellipsoidal heights with the EGM2008 geoid grid of the PROJ data
CDN (`us_nga_egm08_25.tif`); the geoid matters, because the RPC heights are
ellipsoidal, and a DEM 50 m too high moves a UC Davis GEC by about 100 m
(measured with this module; h / tan(incidence) is 96 m at its 27° incidence). A GEC without RPCs is warped with its own
geotransform, with a WARNING (no terrain correction; tile-to-tile geolocation
scatter was 3.7-6.6 m with it against 1.0-3.5 m with RPCs and the DEM, and
18-24 m on Mallorca). Capella GEO products are already terrain-corrected and
are only resampled. The GEO terrain correction uses a 30 m DSM, which does not
resolve trees, so tree tops are displaced (about 12 m for a 10 m tree at 40°
incidence; a WARNING says so). Capella GEC, which has no terrain correction,
is not used.

**Radiometry.** Each date is averaged onto the composite grid: Umbra DN with
GDAL `average` (the mean of a log scale), Capella with GDAL `rms`, so that
(scale_factor × rms DN)² is the mean sigma0 intensity. Capella values become
sigma0 dB + 50, so that every valid pixel is positive (the calibration label
of each collect is kept in `AGRIBOUND_SAR_CALIBRATION`); Umbra DN stay
uncalibrated. Each date is then level-normalised: its median over the pixels
all dates cover is shifted to the mean of the dates' medians (the scaling of
Umbra's complex products differed by 12-16 dB between dates, and without the
shift the median follows the brightest date). There is no beam flattening: removing the spotlight
illumination fall-off lowered F1 in the prototype.

**Co-registration** (`vhr_sar_coregister`):

- `"auto"` (default): every date is registered to the reference date
  (normalised cross-correlation of edge maps on 1 km tiles, falling back to 2 m
  and 4 m block means, which single-look 1 m products need), then the stack to
  an optical base of 1 m or finer (in the base's CRS unit: a State Plane
  orthophoto with 2 ft pixels has 0.61 m pixels): `vhr_sar_coregister_base` when given, else
  the latest USGS NAIP Plus vintage, read at 1 m, where it has imagery
  (the conterminous US and some territories). Without a base the dates are
  registered to each other only, with a WARNING: the composite then keeps the
  reference date's absolute geolocation error.
- `"optical"`: as `"auto"`, but an error without a base.
- `"sar"`: dates to the reference date only. `"none"`: no co-registration.

A registration counts only when the tile estimates agree. The shift is the
median of the inlier tiles, those within 3 MAD of the first median (at least
the tolerance, 3 m date to date and 6 m to the base, and at most three times
it), and the inliers must be a majority of the tiles with a correlation peak.
Unrelated images (another site's orthophoto or SAR median) fail every level of
this test; before the radius was capped, 98-100 % of their scattered estimates
(spread 55-133 m) counted as inliers at the 2 m and 4 m levels, and a shift of
tens of metres was applied. Every real build so far passed at full
resolution, with 89-100 % inliers. A failed registration stops the build with
a `RuntimeError`.
Each date is warped once, with its total shift (no double resampling); the
shift and the registration quality of every collect are recorded. In the
prototype, offsets after the RPC and DEM correction were 2.3-25.9 m (median
6.8 m) and 0.9 m or less after this correction. Sentinel-2 is never used as
the base: it was 4.5-7 m off at two of the prototype sites.

**Output.** float32 on the study-area grid at `vhr_sar_resolution_m`
(default 1 m; 0.5 m scored lower in the prototype), 0 = nodata: Umbra the
median DN (value scale `dn`, clipped to 1-255), Capella the median sigma0
dB + 50 (value scale `db`). The median needs two valid dates per pixel (one in
a single-date composite). With `vhr_sar_bands="median"` (default) the one band
`MEDIAN` is read as R, G and B. `vhr_sar_bands="median+dates"`
(experimental) adds `PAIR_MEAN` (the mean of the two dates closest in time)
and one band per date (`D01_<date>`, ...); it needs at least 2 collects. The
engines then read band 1 as R and B and band 2 as G unless `bands` is given, so
Delineate-Anything reads the median and the pair mean (the mapping follows
`vhr_sar_bands` when the composite is read; `bands` itself is not changed). A
median plus a second amplitude band reached F1 0.46-0.48 (up to 0.505) at UC
Davis in the phase pilot, with amplitudes from SICD data and variants picked on
that area; in the held-out test below it brought no consistent gain (0.378
against 0.390 for the median at the Texas site, 0.303 against 0.261 at the Ohio
site).

**Delineate-Anything** reads these composites best at a coarser processing
resolution and without tile-piece merging, which chain-merges detections on
speckled SAR into scene-sized polygons (689 of 770 detections joined into 13
polygons at Purdue). Set `engine_params={"processing_gsd_m": 3,
"merge_tile_pieces": False}`: the recipe fixed before the prototype was
scored, not an engine default (the agent's `recommend_configurations`
proposes it). Zero-shot F1 (IoU ≥ 0.5, one-to-one, reference fields of at
least 2,500 m²) in the Umbra prototype:

| | UC Davis 2024 | Centerfield 2025 | Purdue 2024 | Mallorca 2026 |
|---|---|---|---|---|
| Reference fields | 235 | 262 | 103 | 524 |
| One date, 1 m, default settings | 0.008 | 0.000 | 0.081 | 0.065 |
| Median, 1 m, read at 3 m, no merging | 0.439 | 0.163 | 0.257 | 0.360 |
| Orthophoto + DA (NAIP; PNOA on Mallorca) | 0.426 | 0.360 | 0.240 | 0.460 |

At UC Davis and Purdue the median matched the orthophotos; at Centerfield
(1 m single-look products) and Mallorca (winter collects, when the almond and
fruit trees are bare; that median mixed HH and VV, which agribound does not
do) it trailed them by 0.10-0.20. Tree crops were no harder than field crops
relative to optical (UC Davis tree crops: F1 0.49-0.53 against 0.47), and the
errors on trees were mostly merges of neighbouring blocks. A single date
with the default settings matched none of the 48 (UC Davis) and 102 (Mallorca)
tree fields. On Capella GEO near Sevilla (two VV dates of January 2023, 1,128
SIGPAC parcels of 2025) F1 was 0.50-0.54. None of these is held out: the recipe was chosen on these
areas, the UC Davis and Mallorca areas and Spain lie in Delineate-Anything's
training geography (DWR and SIGPAC labels), and whether FBIS-73M holds X-band
patches was not checked; Purdue's reference is synthetic (USDA Crop Sequence
Boundaries).

**Held-out test** (2026-10-09). The recipe was then fixed (automatic
selection, every setting at its default, `processing_gsd_m=3` and
`merge_tile_pieces=False`) and the sites, the evaluation rule and the verdict
rule were written down before any build or scoring. The sites are new US
farmland with a series of consistent collects in one growing season; the
reference is the OpenET field boundaries v2.1.1 (agricultural fields of at
least 2,500 m² whose representative point lies in the evaluation area: the
study area within the composite's valid pixels); F1 at IoU ≥ 0.5, one-to-one,
with 95 % bootstrap intervals. The optical control is the closest NAIP year in
Earth Engine (2022 in Texas, 2023 in Ohio and Vermont) + Delineate-Anything,
with the engine defaults and with the SAR settings:

| Site | Collects | Fields | SAR (fixed recipe) | NAIP, DA defaults | NAIP, DA at 3 m, no merging |
|---|---|---|---|---|---|
| Texas A&M AgriLife farm, Burleson Co., TX | Umbra, 6 VV, 2024-04-12 to 06-24 | 136 | 0.390 [0.324-0.461] | 0.284 [0.234-0.343] | 0.403 [0.338-0.471] |
| Intel Ohio site, Licking Co., OH | Umbra, 4 HH, 2024-06-11 to 09-06 | 77 | 0.261 [0.208-0.313] | 0.193 [0.137-0.248] | 0.310 [0.240-0.375] |
| Orleans Co., VT | Capella GEO, 2 HH, 2022-06-18 and 06-26 | 112 | 0.274 [0.207-0.350] | 0.261 [0.196-0.337] | 0.249 [0.188-0.319] |

By the rule fixed in advance (parity: SAR at most 0.05 below the optical
control with its defaults) the recipe holds at all three sites; against the
optical control read the same way, the differences were -0.013, -0.048 and
+0.025. The absolute scores are low (0.26-0.39), and Capella's only held-out
site scored 0.27: its 0.50-0.54 near Sevilla is in-sample.

Example 26
([`examples/26_sar_field_boundaries.py`](https://github.com/montimaj/agribound/blob/main/examples/26_sar_field_boundaries.py))
repeats these runs with the same areas, collects and settings, and adds
Sentinel-1: Delineate-Anything v2 as released on radar composites at seven
sites with field references, beside optical controls on the same areas
(`--uavsar` adds a UAVSAR L-band quad-pol composite near UC Davis). In its
run of 2026-10-10 with the development code of agribound 1.1 the three
held-out sites scored 0.370, 0.261 and 0.274 on Umbra or Capella, against
0.282, 0.193 and 0.261 for NAIP with the defaults and 0.406, 0.310 and 0.249
with the SAR settings, and 0.159, 0.132 and 0.035 on Sentinel-1 (at Texas A&M
the held-out test had given 0.390 on Umbra, and 0.284 and 0.403 on NAIP; why
they differ was not checked). See the
[gallery](../gallery.md).

GeoAI and DINOv3 accept these sources but need a checkpoint
fine-tuned on such composites (out of distribution for their optical
pre-training); FTW and Prithvi need optical reflectance and refuse them, and
`engine="ensemble"` needs explicit members.

**Errors.** The composite stage raises:

- `NoDataError` when no collect in the window covers the study area (the
  message lists the nearest dates with collects over it) or covers all of it,
  and when fewer than 2 consistent collects exist (set
  `vhr_sar_max_collects=1` for a single-date composite; it logs a WARNING:
  in the Umbra prototype single dates scored F1 0.00-0.08 with
  Delineate-Anything's default settings, and read at 2 m without tile merging
  0.36-0.40 near UC Davis, 0.15-0.25 near Purdue and 0.16-0.27 on Mallorca
  against 0.42, 0.22 and 0.31 for the median, and about 0 at Centerfield, with
  reference fields down to 100 m²);
- `ValueError` when the study area is larger than one scene (every collect
  must cover the whole study area; split it), when pinned collects are
  unknown or inconsistent, and when the downloads would exceed
  `vhr_sar_max_download_gb` (default 5 GB; cached products do not count);
- `RuntimeError` when the DEM cannot be built or the co-registration fails,
  that is, when the tile estimates do not agree (`vhr_sar_coregister="none"`
  skips it; `"sar"` skips the optical base).

Downloads resume from a partial file named after the ETag its bytes came
from (a partial file of another version is deleted), are checked against the
size and ETag of the index (every request carries `If-Match`, so a download
never mixes two versions of an object), and are cached per collect and copy
under `<vhr_sar_cache_dir>/<source>/products/<collect>/<copy>/` (copies of one
Umbra collect share the file name). A lock file next to each product and DEM
makes processes that share the cache directory, such as `agribound tiles` array
tasks, download it once: the others wait and reuse it. On file systems without
locks (for example Lustre mounted without `flock`) a WARNING says so and the
processes may each download.

**Composite tags.** Besides the tags of every composite: `AGRIBOUND_ATTRIBUTION`
(the credit line), `AGRIBOUND_LICENSE`, `AGRIBOUND_LICENSE_URL`,
`AGRIBOUND_DATA_CITATION` (the AWS registry's citation with the access date,
e.g. "Umbra Synthetic Aperture Radar (SAR) Open Data was accessed on
2026-10-09 from https://registry.opendata.aws/umbra-open-data."),
`AGRIBOUND_CHANGES` (the note of changes), `AGRIBOUND_SAR_COLLECTS` (one JSON
entry per collect: ID, time, platform, polarization, pass, look side,
incidence, view azimuth, resolution, looks, processor, calibration, bucket
key, ETag, warp method, applied shift, registration quality, level offset),
`AGRIBOUND_SAR_COREGISTRATION`, `AGRIBOUND_SAR_RECIPE`
(`vhr-sar-amplitude-1`), `AGRIBOUND_INDEX_SNAPSHOT`, `AGRIBOUND_SAR_PRODUCT`,
`AGRIBOUND_SAR_POLARIZATION`, `AGRIBOUND_SAR_ORBIT_STATE`,
`AGRIBOUND_SAR_LOOK_SIDE` and, for Capella, `AGRIBOUND_DB_OFFSET` and
`AGRIBOUND_SAR_CALIBRATION`. The pipeline copies them into the provenance
record; a JSON manifest next to the composite holds the selection, the
downloads and the per-tile registration. Credit the vendor wherever the
composite or a figure made from it is shown.

```python
from agribound import AgriboundConfig, delineate

config = AgriboundConfig(
    source="umbra",
    year=2024,
    date_range=("2024-06-01", "2024-09-15"),
    study_area="davis.geojson",
    engine="delineate-anything",
    engine_params={"processing_gsd_m": 3, "merge_tile_pieces": False},
    output_path="fields_umbra.gpkg",
)
gdf = delineate(config=config)
```

```bash
agribound composite --source capella --year 2023 --date-range 2023-01-01 2023-01-31 \
    --study-area sevilla.geojson --vhr-sar-coregister-base pnoa_ortho.tif
```

**Checks of this implementation** (2026-10-09): rebuilding the UC Davis
prototype composite from its 4 collects (`vhr_sar_collect_ids`) gave the
prototype's date-to-date shifts within 0.1 m; the whole stack sat 2.2 m from
the prototype's, the difference between the two NAIP 2022 bases (USGS NAIP
Plus here, the Earth Engine mosaic there), and with that offset removed the
two composites correlate at r = 0.954 (1 m) and 0.994 (3 m blocks). Near
Sevilla the composite of the two VV dates agrees with the pilot's to r = 0.993
(1 m), a mean difference of -0.05 dB, and a date-to-date shift of 0.71/0.25 m
against the pilot's 0.8/0.2 m.

### Local GeoTIFF (`local`)

`local_tif_path` is validated (it must have a CRS; `bands` indices must lie
within the band count) and, when a study area is given, cropped to the study
area's bounding box without changing the pixel values (for the
`crop-sequence` engine with a margin of 10 pixels, as far as the raster
reaches, so that its focal windows are complete at the edge). A raster without a
nodata value is accepted with a WARNING (zeros are then valid data, except
that the embedding engine and Delineate-Anything (native and reference
backends) skip pixels whose bands are all zero). A study
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
is the same UTM zone. The v1 store was converted again upstream on
2026-10-06/08 with geotessera's fix of the tile placement (issue #429, commit
ce2d8256fd "zarr: use floor not round when placing tiles", release v0.11.0;
the fix is in the code that builds the store, and the geotessera 0.10.x that
agribound requires reads the reconverted store unchanged); the earlier
conversion had placed about three quarters of the tiles 10 m too
far east, too far south, or both. Read on 2026-10-09, the 2024 rasters of
examples 15 (Pampas, Argentina) and 02 (West Bengal, India) lie within about
1 m of the Sentinel-2 composites (at most 1.1 m and 0.7 m east-west or
north-south, by cross-correlation of edge maps); other areas were not
measured. Against the rasters read before, the reconversion moved 91 % of
example 15's raster one pixel west or north-west and left 9 % in place, and
moved 99.60-99.75 % of the pixels of example 23's three TESSERA areas by one
pixel. Rasters read before the reconversion, such as those of the
agribound 1.0.0 and 1.0.1 example runs, sat 9-10 m east, south or both of the
Sentinel-2 composites in most 0.1° tiles, and polygons of TESSERA clusters
carried that offset (the 0.1.x mosaic of the v1 GeoTIFF tiles happened to
cancel most of it in the Pampas).

The cache key of a TESSERA raster does not change when the store is rewritten
under the same URL, so agribound records the state of the store's objects it
reads (the shards of the `embeddings` and `scales` arrays and the chunks of
the `x` and `y` coordinate arrays under the box read, by their ETags, and the
zones' grid) in the raster's tags and in the provenance record, and checks
it again before a cached raster or an output made from it is reused. A raster
whose shards or coordinates have changed since it was read, or that was cached
before agribound recorded them (such as one read before the reconversion), is
read again once, and an output made from it is not reused (it raises
`FileExistsError` until it is made again with `overwrite=True`). Without
network access the raster is reused with a WARNING, and so is an output,
unless the cached raster has been read again since the output was made. The
check costs a few HTTP requests per read (see
[Reproducibility](reproducibility.md#cache-keys) and
[output reuse](reproducibility.md#output-reuse)).

Both embedding readers hold the whole area in memory while it is assembled
(about 512 bytes per pixel for TESSERA and 256 for Google embeddings, roughly
twice that at peak), so very large regions should be tiled (see
[HPC and large areas](hpc.md)).

### Crop-type map sequences (`cdl-sequence`, `aci-sequence`)

These two sources download stacks of annual crop-type maps instead of
imagery: one band of class codes per year, for `crop_sequence_years` years
(default 8; CLI `--crop-sequence-years`) ending in `year`. Only the
`crop-sequence` engine reads them (see
[Crop-sequence zones](engines.md#crop-sequence-zones-crop-sequence)), and
`agribound.attach_crop_history` adds their per-year classes to any polygons.

| | `cdl-sequence` | `aci-sequence` |
|---|---|---|
| Map | USDA NASS Cropland Data Layer (CDL) | AAFC Annual Crop Inventory (ACI) |
| Earth Engine collection, band | `USDA/NASS/CDL`, `cropland` | `AAFC/ACI`, `landcover` |
| Coverage | conterminous US | Canada (the agricultural area AAFC maps) |
| First year used | 2008 | 2011 |
| Native pixel size | 30 m to 2023, 10 m from 2024 | 30 m |
| Band names of the stack | `CDL2016` ... `CDL2023` | `ACI2016` ... `ACI2023` |
| Class codes | 1-254 | 10-230 |
| Licence | public domain | Open Government Licence - Canada 2.0 (attribution required) |

**Window.** `year` is the last map year, so the defaults (`year=2024`,
`crop_sequence_years=8`) read 2017-2024; the examples use `year=2023`
(2016-2023, the window of the published Crop Sequence Boundaries that the
pilot compared with). NASS builds its Crop Sequence Boundaries from eight-year
windows. `crop_sequence_years` must be 2-20, and a window that starts before
the first year is refused when the configuration is validated. A window year
without a map over the study area raises `NoDataError`, which lists the years
that have maps there; a year with two maps raises `ValueError`.
`date_range` is refused (the maps are annual), and so are a
`composite_method` other than `"median"`, `bands`, `export_method="gdrive"`
or `"gcs"` (the stack is downloaded directly), `sam_refine=True` and an
`export_crs` that is geographic or not in metres (the grid's 30 is in CRS
units, so a CRS in feet would give 9.144 m pixels).

**Grid.** The stack is written on the export grid (`export_crs`; by default
the UTM zone of the study area) at 30 m, padded by 10 pixels (300 m) on every
side so that the engine's focal windows are complete at the edge of the
study area. Earth Engine reprojects each map onto the grid with its default
nearest-neighbour resampling, so the codes stay classes of the map and are
never averaged. A 10 m CDL map (2024 on) is read from the 20 m level of its
pyramid, whose cells hold the most frequent class of their 2 × 2 10 m cells
(the asset's pyramiding policy is `MODE`; Earth Engine reads the level with
the closest scale less than or equal to the requested one): each 30 m pixel
takes the class of the 20 m cell under its centre. In three checks in Iowa on
2026-10-10, 93-96 % of the 30 m pixels held the class of the 10 m cell under
their centre. Reproducing a stack needs the same `export_crs`: in the
crop-sequence pilot, zones built on a UTM grid and on an EPSG:5070 grid
scored within 0.02 of each other, but their polygons agreed at F1 0.23-0.72.

**Value scale.** The codes are nominal: they are written as Earth Engine
returns them, as uint8 with 0 as nodata, and the stack's `value_scale` is
`categorical`. The other engines, the embedding engine, ensembles, SAM
refinement and fine-tuning refuse such rasters, and the reflectance
conversions refuse the two sources. A stack of codes is not a basemap for
`show_boundaries(satellite_tif=...)`.

**Tags** (`AGRIBOUND_*`): `SOURCE`, `VALUE_SCALE` (`categorical`), `YEAR`,
`CROP_SEQUENCE_YEARS`, `YEARS`, `COLLECTIONS`, `IMAGE_IDS` and
`NATIVE_SCALES_M` (per year), `EXPORT_CRS`, `RESOLUTION_M`, `RESAMPLING`
(`nearest`; with 10 m maps in the window, for example
`nearest; 2024 (10 m maps): 20 m MODE pyramid level`), `PAD_PX`, `LEGEND`
(class code to name, from the newest map of the window) and `LEGEND_YEAR`,
`VALID_FRACTION` (the share of the study area with a class in every year)
and `VALID_FRACTION_BY_YEAR`, `CACHE_KEY`, `RECIPE_VERSION` and `ATTRIBUTION`
(the credit line to cite).

**Requests.** One Earth Engine request lists the maps of the window (image
IDs, pixel sizes and the legend), and one download fetches the stack: a 5 km
square is about 190 × 190 pixels per year.

**Dataset details.** The Earth Engine CDL collection holds two images each for
2005 and 2007 and 56 m images for 2005-2007, so `cdl-sequence` starts in
2008; CDL code 62 (Pasture/Grass) has not been used since the 2013 CDL (NASS
merged it into 176, Grass/Pasture, which also holds rangeland). The ACI maps
of 2009-2010 are 56 m, so `aci-sequence` starts in 2011; ACI codes 121
(Cropland) and 159 (Other Oilseeds) appear from 2024.

**LULC crop filter.** The engine applies its own cultivated mask. The crop
filter (on by default) would filter the zones a second time, for example
removing pasture zones that NLCD maps as grassland, so the examples set
`lulc_filter=False` for these sources.

**Other series.** Stacks of other annual crop-type maps (for example the
Thünen maps of Germany or INTA's maps of Argentina) can be read with
`source="local"` and explicit class lists (see the engine section); only CDL
and ACI were measured.

**Attribution.** CDL: USDA National Agricultural Statistics Service Cropland
Data Layer, published crop-specific data layer, USDA-NASS, Washington, DC
(public domain). ACI: "Contains information licensed under the Open
Government Licence – Canada" (Agriculture and Agri-Food Canada, Annual Crop
Inventory). The stack's `AGRIBOUND_ATTRIBUTION` tag holds the line for its
years.

### Valid-pixel check

After a composite is written, the share of pixels inside the study-area
polygons that are valid in every band (in at least one band for the uint8
`naip`, `usgs-naip-plus` and `sentinel1` composites; in every year for the
crop-type map stacks, whose share per year is stored in
`AGRIBOUND_VALID_FRACTION_BY_YEAR`) is stored in the GeoTIFF tag
`AGRIBOUND_VALID_FRACTION`. A share of 0 raises `NoDataError` (an Earth
Engine composite is then deleted); a WARNING is logged below 95 % for Earth
Engine imagery and the `umbra` and `capella` composites, below 99 % for USGS
NAIP Plus and below 50 % for embeddings. A single-date composite (a one-day
`date_range`, see [Dates in a composite](#dates-in-a-composite)) is not
filled from other dates: the pixels its acquisition masks stay nodata, so its
valid share is about the clear share that `agribound scenes` reports for that
date.

### Batch exports

`export_method="gdrive"` or `"gcs"` (with `gcs_bucket`) starts an Earth Engine
batch export task and stops with `ExportTaskStartedError`: the pipeline cannot
continue until the file exists locally. The task is recorded in
`<composite stem>_task.json` in the cache; a later run finds a task that is
`READY`, `RUNNING` or `COMPLETED` and does not start a duplicate. When the task
has finished, download the GeoTIFF and run with `source="local"`. The default
`export_method="local"` downloads the composite directly (in tiles of at most
`tile_size` pixels per side, assembled into one GeoTIFF). The crop-type map
stacks (`cdl-sequence`, `aci-sequence`) are always downloaded directly; batch
exports are refused for them.

### Caching

Every composite, window composite and embedding raster is cached in the
working directory (`cache_dir`, default `<output dir>/.agribound_cache`) under
a name that contains a key over the study area, source, year, date range and
compositing/export settings. See [Reproducibility](reproducibility.md#cache-keys).
The date layers of an Earth Engine composite (`--date-layers`) are cached next
to it as `<source>_date_layers_<key>.tif`, whose key is the composite's plus
`date-layers-1`.
The `umbra` and `capella` keys also hold the recipe tag, the `vhr_sar_*`
settings, the sorted collect IDs with their ETags and the index snapshot; their
archive indexes, downloaded products and DEMs are shared between runs in
`vhr_sar_cache_dir` (default `~/.cache/agribound/vhr_sar`).
The crop-type map stacks are cached as `<source>_stack_<key>.tif`; their key
also holds `crop_sequence_years`, the recipe version and the pad, and the
`crop-sequence` engine caches its label raster next to them, so another
engine parameter or `min_field_area_m2` reuses the download.

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

Composites of cloudy tropical areas such as Twifo Praso combine many dates;
[Dates in a composite](#dates-in-a-composite) describes what is recorded about
them, how to build a composite of one date, and what a pilot at four
tree-crop squares measured.

## Data citations

- Earth Engine: Gorelick et al. (2017), *Remote Sensing of Environment* 202,
  18-27, <https://doi.org/10.1016/j.rse.2017.06.031>.
- HLS: Claverie et al. (2018), *Remote Sensing of Environment* 219, 145-161,
  <https://doi.org/10.1016/j.rse.2018.09.002>.
- Sentinel-1: Torres et al. (2012), *Remote Sensing of Environment* 120, 9-24,
  <https://doi.org/10.1016/j.rse.2011.05.028>.
- Umbra: "Umbra Synthetic Aperture Radar (SAR) Open Data was accessed on DATE
  from <https://registry.opendata.aws/umbra-open-data>." (the registry's citation;
  CC BY 4.0).
- Capella: "Capella Space Synthetic Aperture Radar (SAR) Open Dataset was
  accessed on DATE from <https://registry.opendata.aws/capella_opendata>.", with
  the credit "© Capella Space Corp, All Rights Reserved." and a note of the
  changes (CC BY 4.0); the gallery's figures print the shorter "© Capella
  Space Corp, CC BY 4.0, modified".
- Copernicus DEM GLO-30, and GLO-90 where GLO-30 has no public tile (used to
  orthorectify Umbra GEC): European Space Agency (2022), Copernicus DEM,
  <https://doi.org/10.5270/ESA-c5d3d65>.
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
- Cropland Data Layer: USDA National Agricultural Statistics Service Cropland
  Data Layer. {year}. Published crop-specific data layer. Available at
  <https://www.nass.usda.gov/Research_and_Science/Cropland/SARS1a.php>.
  USDA-NASS, Washington, DC (public domain; the citation of the Earth Engine
  catalogue, linked to NASS's Cropland Data Layer page).
- Annual Crop Inventory: Agriculture and Agri-Food Canada Annual Crop
  Inventory. {year}. Open Government Licence - Canada 2.0
  (<https://developers.google.com/earth-engine/datasets/catalog/AAFC_ACI>).
- Crop Sequence Boundaries: Hunt et al. (2024), *Statistical Journal of the
  IAOS* 40, 237-246, <https://doi.org/10.3233/SJI-230078>.

See [Citation & References](../citation.md) for the full list.
