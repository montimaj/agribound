"""
Source and engine registries: the single source of truth for Agribound metadata.

This module imports nothing heavy (standard library only) so it can be used by
the configuration layer, the CLI and documentation tooling without pulling in
GDAL, torch or Earth Engine.

Facts encoded here were checked against the upstream catalogues and packages
on 2026-09-26 (GEE STAC entries and live collection queries, ftw-tools 2.0.0b5,
Delineate-Anything v1/v2 model cards, geoai-py 0.43.1, terratorch 1.2.13,
geotessera 0.10.2); the Sentinel-1 facts on 2026-10-07 (the Earth Engine
catalogue page and Sentinel-1 guide, and live queries of ``COPERNICUS/S1_GRD``);
the Sentinel-1 patches of FBIS-73M and FBIS-22M on 2026-10-08 (their training
lists, ``train.txt`` of the Hugging Face datasets ``MykolaL/FBIS-73M`` at
9e4cdbd and ``MykolaL/FBIS-22M`` at 8713461); the Umbra and Capella open-data
facts on 2026-10-07/08 (listings of ``s3://umbra-open-data-catalog`` and
``s3://capella-open-data``, their STAC metadata and the AWS Registry of Open
Data pages); the crop-type map facts on 2026-10-09 (a live listing of the
images of ``USDA/NASS/CDL`` and ``AAFC/ACI``: system index, start date, bands
and nominal scale).

Registry entries
----------------
``SOURCE_REGISTRY[name]`` keys (every source has all of them):

``name``, ``collection``, ``resolution_m`` (default export resolution in metres,
*None* when it depends on the input), ``native_resolution_m``, ``all_bands``
(band order of the composite written by the builder, *None* for local files),
``canonical_bands`` (mapping of canonical names ``R, G, B, NIR, NIR_NARROW,
SWIR1, SWIR2`` to native band names, where available), ``value_scale`` (see
below), ``year_range`` (``(first, last)``; ``last`` is *None* for missions that
are still acquiring; *None* for local files), ``coverage`` (free text),
``requires_gee`` and ``restricted``.

``value_scale`` is one of:

- ``"reflectance_x10000"`` -- float32 surface reflectance multiplied by 10000
  (Sentinel-2, Landsat and HLS composites after the 1.0 harmonisation).
- ``"uint8"`` -- 8-bit values 0-255: digital numbers (NAIP, USGS NAIP Plus), or
  Sentinel-1 backscatter mapped from fixed dB ranges to 1-255, with 0 as nodata
  (see :data:`agribound.composites.gee.S1_DB_RANGES`).
- ``"unit"`` -- float32 TOA reflectance as stored (Landsat PAN), without x10000 scaling.
- ``"dn"`` -- per-band medians (or greenest-pixel selections) of the scenes' raw
  integer digital numbers, exported as float32 (a median of an even number of
  scenes can be a half-integer); radiometry unverified (SPOT 6/7). For
  ``umbra``: the median of level-normalised 8-bit GEC digital numbers, an
  uncalibrated log-scaled amplitude (about 6.6 DN per dB), with 0 as nodata.
- ``"db"`` -- float32 radar backscatter in decibels plus a fixed offset, so that
  valid pixels are positive and 0 is nodata (``capella``: sigma0 dB + 50, see
  :data:`agribound.composites.vhr_sar.CAPELLA_DB_OFFSET`).
- ``"embedding"`` -- pre-computed embedding vectors (Google, TESSERA).
- ``"categorical"`` -- class codes of annual crop-type maps, one uint8 band per
  year, with 0 as nodata (``cdl-sequence``, ``aci-sequence``). The codes are
  nominal: no distance or mean of codes is meaningful, so only the engines of
  :data:`CATEGORICAL_ENGINES` read them; the other engines refuse a raster
  whose ``AGRIBOUND_VALUE_SCALE`` tag is ``"categorical"``.
- ``"unknown"`` -- user-provided rasters.

``ENGINE_REGISTRY[name]`` keys: ``name``, ``approach``, ``strengths``,
``gpu_recommended``, ``requires_bands``, ``supported_sources``, ``label_free``
(runs without a user-supplied or fine-tuned checkpoint), ``fine_tunable``,
``reference``, ``install_extra``, ``notes`` (applies to every source) and
``source_notes`` (source name -> note that applies only to that source; use
:func:`engine_notes` to get the notes for one source).
"""

from __future__ import annotations

import copy
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANONICAL_BAND_NAMES: tuple[str, ...] = ("R", "G", "B", "NIR", "NIR_NARROW", "SWIR1", "SWIR2")
"""Canonical band names understood by :func:`agribound.engines.base.get_canonical_band_indices`."""

VALUE_SCALES: tuple[str, ...] = (
    "reflectance_x10000",
    "unit",
    "uint8",
    "dn",
    "db",
    "embedding",
    "unknown",
    "categorical",
)
"""Allowed values of ``SOURCE_REGISTRY[...]["value_scale"]``."""

SAM_REFINE_BACKENDS: tuple[str, ...] = ("sam2", "sam2.1", "sam3", "sam3-hf")
"""SAM backends accepted by ``AgriboundConfig.sam_backend`` for the refinement stage."""

# Module-level registry key requested by the 1.0 contract (same values as a list).
sam_refine_backends: list[str] = list(SAM_REFINE_BACKENDS)

TESSERA_YEAR_RANGES: dict[str, tuple[int, int]] = {
    # v1 (variant "vultr"): near-global for 2024, regional 2017-2023 and 2025.
    "v1": (2017, 2025),
    # v1.1 (variant "cambridge"): regional, 2015-2025.
    "v1.1": (2015, 2025),
    # v2 (beta variants "2B-L~beta1"/"2B-L~beta2"): sparse, mostly Europe.
    "v2": (2017, 2025),
}
"""Years with any published TESSERA tiles, per dataset version (geotessera 0.10.2 manifests)."""

_OPTICAL_GEE_SOURCES = ["landsat", "landsat-pan", "sentinel2", "hls", "naip", "spot", "spot-pan"]
_SAR_GEE_SOURCES = ["sentinel1"]
#: Open very-high-resolution SAR on AWS (amplitude composites built locally,
#: :mod:`agribound.composites.vhr_sar`).
_VHR_SAR_SOURCES = ["umbra", "capella"]
#: Stacks of annual crop-type maps on Earth Engine (class codes, not imagery).
_CROP_SEQUENCE_SOURCES = ["cdl-sequence", "aci-sequence"]
_ALL_IMAGERY_SOURCES = [
    "landsat",
    "landsat-pan",
    "sentinel2",
    "hls",
    "naip",
    "usgs-naip-plus",
    "spot",
    "spot-pan",
    "sentinel1",
    "umbra",
    "capella",
    "local",
]

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "landsat": {
        "name": "Landsat 5/7/8/9 Collection 2 Level-2",
        "collection": (
            "LANDSAT/LT05/C02/T1_L2 + LANDSAT/LE07/C02/T1_L2 + "
            "LANDSAT/LC08/C02/T1_L2 + LANDSAT/LC09/C02/T1_L2"
        ),
        "resolution_m": 30,
        "native_resolution_m": 30,
        # Spectral SR bands common to L5/7/8/9, in L8/9 naming. L5/7 bands are
        # renamed to the L8/9 convention when the collection is built.
        "all_bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "canonical_bands": {
            "R": "SR_B4",
            "G": "SR_B3",
            "B": "SR_B2",
            "NIR": "SR_B5",
            "SWIR1": "SR_B6",
            "SWIR2": "SR_B7",
        },
        "value_scale": "reflectance_x10000",
        "year_range": (1984, None),
        "coverage": (
            "Global. Landsat 5 TM 1984-03-16 to 2012-05-05, Landsat 7 ETM+ 1999-05-28 to "
            "2024-01-19, Landsat 8 OLI 2013-03-18 to present, Landsat 9 OLI-2 2021-10-31 to "
            "present"
        ),
        "requires_gee": True,
        "restricted": False,
    },
    "landsat-pan": {
        "name": "Landsat 7/8/9 Collection 2 Tier 1 TOA panchromatic",
        "collection": (
            "LANDSAT/LE07/C02/T1_TOA + LANDSAT/LC08/C02/T1_TOA + LANDSAT/LC09/C02/T1_TOA"
        ),
        "resolution_m": 15,
        "native_resolution_m": 15,
        "all_bands": ["B8"],
        "canonical_bands": {"R": "B8", "G": "B8", "B": "B8"},
        "value_scale": "unit",
        "year_range": (1999, None),
        "coverage": (
            "Global. Landsat 7 ETM+ 1999-05-28 to 2024-01-19 (SLC-off gaps after 2003), "
            "Landsat 8 OLI from 2013-03-18, Landsat 9 OLI-2 from 2021-10-31. "
            "The PAN bandpasses differ (Landsat 7 0.52-0.90 um, Landsat 8/9 0.50-0.68 um), so "
            "by default (landsat_pan_missions='auto') a composite uses Landsat 8/9 when the date "
            "window overlaps their record and Landsat 7 only for earlier windows, never both"
        ),
        "requires_gee": True,
        "restricted": False,
    },
    "sentinel2": {
        "name": "Sentinel-2 MSI L2A (harmonized)",
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "resolution_m": 10,
        "native_resolution_m": 10,
        # All 12 spectral bands (no QA/SCL). 20 m and 60 m bands are resampled.
        "all_bands": [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B11",
            "B12",
        ],
        "canonical_bands": {
            "R": "B4",
            "G": "B3",
            "B": "B2",
            "NIR": "B8",
            "NIR_NARROW": "B8A",
            "SWIR1": "B11",
            "SWIR2": "B12",
        },
        "value_scale": "reflectance_x10000",
        "year_range": (2017, None),
        "coverage": (
            "Global, 2017 to present (the years Agribound accepts). The Earth Engine "
            "collection also holds earlier L2A images (the first on 2015-07-04, counted "
            "2026-09-28), which Agribound does not use; the catalogue's listed extent starts "
            "2017-03-28, and it warns that 2017-2018 L2A coverage is not yet global"
        ),
        "requires_gee": True,
        "restricted": False,
    },
    "hls": {
        "name": "Harmonized Landsat Sentinel-2 v2.0 (HLSL30 + HLSS30)",
        "collection": "NASA/HLS/HLSL30/v002 + NASA/HLS/HLSS30/v002",
        "resolution_m": 30,
        "native_resolution_m": 30,
        # Seven harmonised bands in HLSL30 naming. HLSS30 [B1, B2, B3, B4, B8A,
        # B11, B12] are mapped onto [B1..B7] when the collection is built.
        "all_bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        "canonical_bands": {
            "R": "B4",
            "G": "B3",
            "B": "B2",
            # L30 B5 (OLI narrow NIR) / S30 B8A (narrow NIR), harmonised.
            "NIR": "B5",
            "NIR_NARROW": "B5",
            "SWIR1": "B6",
            "SWIR2": "B7",
        },
        "value_scale": "reflectance_x10000",
        "year_range": (2013, None),
        "coverage": "Global land. HLSL30 2013-04-11 to present, HLSS30 2015-11-28 to present",
        "requires_gee": True,
        "restricted": False,
    },
    "naip": {
        "name": "NAIP (USDA National Agriculture Imagery Program)",
        "collection": "USDA/NAIP/DOQQ",
        # Default export resolution (AgriboundConfig.naip_resolution_m).
        "resolution_m": 1.0,
        # USDA standard since 2018 (0.3 m option); 1-2 m in earlier years; varies
        # by state and year.
        "native_resolution_m": 0.6,
        "all_bands": ["R", "G", "B", "N"],
        "canonical_bands": {"R": "R", "G": "G", "B": "B", "NIR": "N"},
        "value_scale": "uint8",
        "year_range": (2002, 2023),
        "coverage": (
            "Conterminous US, 2002-2023 in GEE (no 2024/2025 imagery as of 2026-09), about "
            "2-3 year revisit per state. 0.6 m standard since 2018 (0.3 m in some states); "
            "some early years are RGB only"
        ),
        "requires_gee": True,
        "restricted": False,
    },
    "usgs-naip-plus": {
        "name": "USGS NAIP Plus ImageServer",
        "collection": (
            "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
        ),
        # Exported at the finest resolution among the selected footprints.
        "resolution_m": None,
        # Service pixel size is 0.3 m; state vintages are 0.3-0.6 m.
        "native_resolution_m": None,
        "all_bands": ["R", "G", "B", "N"],
        "canonical_bands": {"R": "R", "G": "G", "B": "B", "NIR": "N"},
        "value_scale": "uint8",
        "year_range": (2012, 2023),
        # Checked on 2026-09-27 with ImageServer queries (distinct State/Year of
        # the footprints; service extent -179.3 to 179.8 E, -14.6 to 71.4 N).
        "coverage": (
            "US states and territories: footprints for the conterminous states, Alaska "
            "(2020), Hawaii (2013), Puerto Rico (2018), Guam (2013), the Northern Mariana "
            "Islands (2012) and American Samoa (2012); none for the US Virgin Islands "
            "(service query, 2026-09-27). Only the latest NAIP/HRO vintage per state (years "
            "2012-2023, dense from 2019); not a historical archive. 0.3-0.6 m depending on "
            "the state vintage"
        ),
        "requires_gee": False,
        "restricted": False,
    },
    "spot": {
        "name": "SPOT 6/7 multispectral",
        "collection": "AIRBUS/SPOT6_7",
        "resolution_m": 6,
        "native_resolution_m": 6,
        "all_bands": ["R", "G", "B", "N"],
        "canonical_bands": {"R": "R", "G": "G", "B": "B", "NIR": "N"},
        "value_scale": "dn",
        "year_range": (2012, 2023),
        "coverage": (
            "Global, restricted access (AIRBUS/SPOT6_7 is not in the public GEE catalogue), "
            "2012-10-17 to 2023-11-15. Composites hold per-band medians of raw DN "
            "(unverified radiometry)"
        ),
        "requires_gee": True,
        "restricted": True,
    },
    "spot-pan": {
        "name": "SPOT 6/7 panchromatic",
        "collection": "AIRBUS/SPOT6_7",
        "resolution_m": 1.5,
        "native_resolution_m": 1.5,
        "all_bands": ["P"],
        # The single panchromatic band is replicated into the three RGB slots.
        "canonical_bands": {"R": "P", "G": "P", "B": "P"},
        "value_scale": "dn",
        "year_range": (2012, 2023),
        "coverage": (
            "Global, restricted access (AIRBUS/SPOT6_7 is not in the public GEE catalogue), "
            "2012-10-17 to 2023-11-15. Composites hold per-band medians of raw DN "
            "(unverified radiometry)"
        ),
        "requires_gee": True,
        "restricted": True,
    },
    "sentinel1": {
        "name": "Sentinel-1 C-band SAR GRD",
        "collection": "COPERNICUS/S1_GRD",
        "resolution_m": 10,
        # Pixel spacing. The IW GRD high-resolution product resolves about 20 x 22 m
        # (range x azimuth), so the effective resolution is about 20 m.
        "native_resolution_m": 10,
        # VV and VH backscatter and VV - VH (dB), each mapped from a fixed dB range
        # to 1-255 with 0 as nodata (agribound.composites.gee.S1_DB_RANGES).
        "all_bands": ["VV", "VH", "VV_VH"],
        "canonical_bands": {"R": "VV", "G": "VH", "B": "VV_VH"},
        "value_scale": "uint8",
        # Earth Engine cannot remove thermal noise from images produced before July
        # 2015, and its Sentinel-1 guide lists GRD border-noise removal as of 2018-01-12
        # (it does not say whether earlier images were reprocessed).
        "year_range": (2015, None),
        "coverage": (
            "Global, where Interferometric Wide swath (IW) images with VV and VH exist (most "
            "land; HH+HV areas, mainly polar, have none). In Earth Engine: Sentinel-1A "
            "2014-10-03 to 2026-06-29, Sentinel-1B 2016-09-26 to 2021-12-23, Sentinel-1C from "
            "2025-03-27 (preliminary calibration until 2025-05-19), Sentinel-1D from "
            "2026-04-15. 6-12 day revisit from 2017 in most regions (12 days per satellite and "
            "track, 6 where two satellites image a track; the observation plan decides); sparse "
            "in 2015, and in some regions in 2016. 10 m pixels, about 20 x 22 m resolution. "
            "Earth Engine cannot remove thermal noise from images before July 2015, and its "
            "guide lists GRD border-noise removal as of 2018-01-12, so earlier images can keep "
            "border noise"
        ),
        "requires_gee": True,
        "restricted": False,
    },
    "umbra": {
        "name": "Umbra open SAR (X-band, GEC)",
        "collection": "s3://umbra-open-data-catalog (AWS Registry of Open Data: umbra-open-data)",
        # Default composite grid (AgriboundConfig.vhr_sar_resolution_m).
        "resolution_m": 1.0,
        # Spotlight products of 0.16-1 m resolution; the GEC pixel spacing varies per collect.
        "native_resolution_m": None,
        # The median amplitude, copied to R, G and B (vhr_sar_bands="median"). With
        # vhr_sar_bands="median+dates" band 2 is PAIR_MEAN and bands 3+ the single dates.
        "all_bands": ["MEDIAN"],
        "canonical_bands": {"R": "MEDIAN", "G": "MEDIAN", "B": "MEDIAN"},
        "value_scale": "dn",
        "year_range": (2023, None),
        "coverage": (
            "Sites of the Umbra Open Data Program and other tasked collects in "
            "s3://umbra-open-data-catalog (CC BY 4.0), from January 2023: X-band spotlight "
            "scenes of about 5 x 5 km, 0.16-1 m resolution, one polarization per collect (VV in "
            "14,416 and HH in 3,388 GEC folders, listings of 2026-10-07/08). Long series "
            "(166-204 collects) exist at a few farm sites (UC Davis CA, Purdue IN, Texas A&M, "
            "NDSU, Johnston IA, Centerfield UT); most sites have few dates. The 8-bit GEC "
            "amplitude is uncalibrated; collects must each cover the whole study area"
        ),
        "requires_gee": False,
        "restricted": False,
    },
    "capella": {
        "name": "Capella open SAR (X-band, GEO)",
        "collection": "s3://capella-open-data (AWS Registry of Open Data: capella_opendata)",
        "resolution_m": 1.0,
        # Spotlight, sliding spotlight and stripmap: median GEO pixel spacing 0.34, 0.60
        # and 0.84 m.
        "native_resolution_m": None,
        "all_bands": ["MEDIAN"],
        "canonical_bands": {"R": "MEDIAN", "G": "MEDIAN", "B": "MEDIAN"},
        "value_scale": "db",
        "year_range": (2021, None),
        "coverage": (
            "Capella Space open data in s3://capella-open-data (CC BY 4.0): 2,291 X-band "
            "collects from 2021-02-08 to 2026-09-09 (census of 2026-10-08; uploads arrive "
            "about monthly), single polarization (HH in 2,094, VV in 197), spotlight, sliding "
            "spotlight and stripmap, with a terrain-corrected GEO product in 2,286. 89 % of "
            "the collects lie at sites with under 10 % cropland and no farmland site has a "
            "long series in one geometry. About 19 % of the products are not fully calibrated "
            "('limited' or 'partial'); collects must each cover the whole study area"
        ),
        "requires_gee": False,
        "restricted": False,
    },
    "local": {
        "name": "Local GeoTIFF",
        "collection": None,
        "resolution_m": None,
        "native_resolution_m": None,
        "all_bands": None,
        "canonical_bands": None,
        "value_scale": "unknown",
        "year_range": None,
        "coverage": "User-provided",
        "requires_gee": False,
        "restricted": False,
    },
    "google-embedding": {
        "name": "Google Satellite Embedding V1 (AlphaEarth Foundations)",
        "collection": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        "resolution_m": 10,
        "native_resolution_m": 10,
        "all_bands": [f"A{i:02d}" for i in range(64)],
        "canonical_bands": None,
        "value_scale": "embedding",
        "year_range": (2017, 2025),
        "coverage": "Global land, annual 2017-2025 (64-D unit-length embeddings)",
        # True for the default backend (google_embedding_backend="gee").
        "requires_gee": True,
        "restricted": False,
    },
    "tessera-embedding": {
        "name": "TESSERA embeddings",
        "collection": "geotessera (https://data.source.coop/tessera/tessera)",
        "resolution_m": 10,
        "native_resolution_m": 10,
        "all_bands": [f"T{i:03d}" for i in range(128)],
        "canonical_bands": None,
        "value_scale": "embedding",
        "year_range": TESSERA_YEAR_RANGES["v1"],
        "coverage": (
            "v1: near-global for 2024, regional for 2017-2023 and 2025; v1.1: regional "
            "2015-2025; v2: beta, sparse (mostly Europe)"
        ),
        "requires_gee": False,
        "restricted": False,
    },
    "cdl-sequence": {
        "name": "USDA NASS Cropland Data Layer, multi-year stack",
        "collection": "USDA/NASS/CDL",
        "resolution_m": 30,
        # 30 m maps to 2023; the 2024 and later maps are 10 m. Earth Engine reads them at
        # 30 m from the 20 m level of their MODE pyramid (each 20 m cell holds the most
        # frequent class of its 2 x 2 10 m cells), then by nearest neighbour.
        "native_resolution_m": 30,
        # One band of class codes per year (CDL<year>), crop_sequence_years of them.
        "all_bands": None,
        "canonical_bands": None,
        "value_scale": "categorical",
        "year_range": (2008, None),
        "coverage": (
            "Conterminous US; one map per year in Earth Engine, 30 m for 2008-2023 and 10 m "
            "from 2024 (read at 30 m from the 20 m level of their mode pyramid). Earlier "
            "years are not used (two images each for 2005 and 2007, 56 m images in "
            "2005-2007). Public domain"
        ),
        "requires_gee": True,
        "restricted": False,
    },
    "aci-sequence": {
        "name": "AAFC Annual Crop Inventory, multi-year stack",
        "collection": "AAFC/ACI",
        "resolution_m": 30,
        "native_resolution_m": 30,
        # One band of class codes per year (ACI<year>), crop_sequence_years of them.
        "all_bands": None,
        "canonical_bands": None,
        "value_scale": "categorical",
        "year_range": (2011, None),
        "coverage": (
            "Canada; one 30 m map per year in Earth Engine from 2011 (the 56 m maps of "
            "2009-2010 are not used). Open Government Licence - Canada 2.0 (attribution "
            "required)"
        ),
        "requires_gee": True,
        "restricted": False,
    },
}

ENSEMBLE_DEFAULT_MEMBERS: tuple[str, ...] = ("delineate-anything", "ftw")
"""Members of ``engine="ensemble"`` when ``engine_params["engines"]`` is not set.

:data:`agribound.engines.ensemble.DEFAULT_MEMBERS` is this tuple; it lives here so
that :mod:`agribound.config` can validate the defaults without importing the
engine (and geopandas).
"""

SAR_SOURCES: tuple[str, ...] = tuple(_SAR_GEE_SOURCES + _VHR_SAR_SOURCES)
"""Synthetic aperture radar sources (their composites hold backscatter, not reflectance).

The reflectance conversions of :mod:`agribound.io.raster` refuse them unless a
``value_scale`` is passed, and the agent's ``recommend_configurations`` ranks
them after every other source.
"""

VHR_SAR_SOURCES: tuple[str, ...] = tuple(_VHR_SAR_SOURCES)
"""Open very-high-resolution SAR sources (Umbra and Capella open data on AWS).

Their amplitude composites are built locally, without Earth Engine, by
:class:`agribound.composites.vhr_sar.VHRSARCompositeBuilder`; the
``vhr_sar_*`` configuration fields apply only to them.
"""

GEE_IMAGERY_SOURCES: tuple[str, ...] = tuple(_OPTICAL_GEE_SOURCES + _SAR_GEE_SOURCES)
"""Imagery sources whose composites are built on Google Earth Engine."""

SCENE_SURVEY_SOURCES: tuple[str, ...] = (
    "sentinel2",
    "landsat",
    "landsat-pan",
    "hls",
    "spot",
    "spot-pan",
)
"""Sources whose acquisitions :func:`agribound.composites.scenes.survey_scenes` ranks.

They are also the sources of :func:`agribound.composites.scenes.build_date_layers`.
``naip`` is a mosaic, ``sentinel1`` has no cloud filter or mask, and ``umbra`` and
``capella`` select their collects themselves.
"""

SCENE_RANKED_SOURCES: tuple[str, ...] = ("sentinel2", "landsat", "landsat-pan", "hls")
"""Of :data:`SCENE_SURVEY_SOURCES`, those with a per-pixel cloud mask.

Their acquisitions are ranked by the share of the study area that is clear under
the mask; ``spot`` and ``spot-pan`` have no pixel mask, so their clear share is the
share of the study area their scenes cover.
"""

EMBEDDING_SOURCES: tuple[str, ...] = ("google-embedding", "tessera-embedding")
"""Pre-computed embedding sources."""

CROP_SEQUENCE_SOURCES: tuple[str, ...] = tuple(_CROP_SEQUENCE_SOURCES)
"""Stacks of annual crop-type maps (value scale ``"categorical"``).

Their builder, :class:`agribound.composites.crop_sequence.CropSequenceCompositeBuilder`,
downloads the raw class codes of ``crop_sequence_years`` annual maps ending in
``year`` from Earth Engine, one uint8 band per year. Only the engines of
:data:`CATEGORICAL_ENGINES` support them; ``crop_sequence_years`` applies only
to them.
"""

CATEGORICAL_ENGINES: tuple[str, ...] = ("crop-sequence",)
"""Engines that read class codes (rasters whose value scale is ``"categorical"``).

Every other engine refuses such a raster
(:meth:`agribound.engines.base.DelineationEngine.validate_input`), and so do
SAM refinement and fine-tuning. The reflectance conversions of
:mod:`agribound.io.raster` take arrays, not rasters: they refuse the sources of
:data:`CROP_SEQUENCE_SOURCES` and ``value_scale="categorical"``.
"""

# ---------------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------------

ENGINE_REGISTRY: dict[str, dict[str, Any]] = {
    "delineate-anything": {
        "name": "Delineate-Anything",
        "approach": "YOLO11-seg instance segmentation (Ultralytics)",
        "strengths": (
            "Resolution-agnostic; trained on 0.25-10 m imagery (FBIS-22M for v1, FBIS-73M "
            "for v2); outputs field instances directly"
        ),
        "gpu_recommended": True,
        "requires_bands": ["R", "G", "B"],
        "supported_sources": list(_ALL_IMAGERY_SOURCES),
        "label_free": True,
        "fine_tunable": True,
        "reference": (
            "Lavreniuk et al. (2025) Delineate Anything: Resolution-Agnostic Field Boundary "
            "Delineation on Satellite Imagery, ECAI 2025, arXiv:2504.02534; Lavreniuk et al. "
            "(2026) Delineate Anything v2: A Global Foundation Model for Field Delineation, "
            "ECCV 2026 Workshops (GAIA), arXiv:2607.19069"
        ),
        "install_extra": "delineate-anything",
        "notes": (
            "Training imagery spans 0.25-10 m; engine_meta['gsd_outside_training_range'] is "
            "True (and a WARNING is logged) when the input GSD is more than 5 % outside that "
            "range. Model code and weights are AGPL-3.0."
        ),
        "source_notes": {
            "landsat": "30 m Landsat composites are outside the 0.25-10 m training range.",
            "landsat-pan": (
                "15 m Landsat PAN composites are outside the 0.25-10 m training range; the "
                "single band is replicated to grey R, G, B."
            ),
            "hls": "30 m HLS composites are outside the 0.25-10 m training range.",
            "sentinel1": (
                "Sentinel-1 SAR backscatter (VV, VH and VV-VH as R, G, B). FBIS-73M, the "
                "training set of the v2 model (large_v2), includes Sentinel-1 patches (76,448 "
                "S1 and 7,852 SYN-S1 of the 1,432,187 in its training list; the published "
                "training list of FBIS-22M, of the v1 models, has none), but how their bands "
                "were rendered is not documented, so agribound's composites may differ from "
                "them. Zero-shot F1 (IoU >= 0.5, no LULC filter) was 0.18 and 0.39 on two 5 "
                "km smoke squares (default settings; Lea County, NM, 2022; near Davis, CA, "
                "2024) and, in a pilot (2026-10-08; max_detections=1000 on example 23's "
                "squares), 0.80 on the tree crops of example 23's Madera square (2022), 0.455 "
                "on 20.5 km2 near Davis (2024; inside the Davis smoke square), 0.173 on 415 km2 "
                "of eastern Lea County (2024; a box that contains the Lea smoke square), and "
                "0.11, 0.06 and 0.00 on example 23's Twifo Praso (2020), Ubeda (2023) and Oro "
                "(2021) squares. The California and Spanish scores are not held out: "
                "FBIS-73M's labels there are DWR and SIGPAC polygons, and 2 of its S1 patches "
                "intersect the Davis area (about 38 lie within 10 km of it) and 18 lie within "
                "10 km of the Madera square; expect to need "
                "fine-tuning. stretch_percentiles has no effect on this uint8 composite (a "
                "float copy stretched at (0.1, 99.9) scored lower in 4 of the 6 pilot areas), "
                "and merge_tile_pieces=False changed F1 by at most 0.005 there. The effective "
                "resolution is about 20 m (IW GRD, 20 x 22 m) although the pixels are 10 m, so "
                "the GSD check does not warn."
            ),
            "umbra": (
                "Umbra GEC amplitude (one polarization; the median of co-registered, "
                "level-normalised dates in R, G and B). In a prototype (2026-10-08; reference "
                "fields of at least 2,500 m2, IoU >= 0.5), zero-shot F1 with a recipe fixed in "
                "advance (median of 4-6 collects of one geometry, 1 m composite, engine_params "
                "processing_gsd_m=3 and merge_tile_pieces=False) was 0.439 near UC Davis, CA "
                "(2024), 0.163 at Centerfield, UT (2025), 0.257 near Purdue, IN (2024) and 0.360 "
                "on Mallorca (2026; that median mixed HH and VV, which agribound does not do), "
                "against 0.426, 0.360, 0.240 and 0.460 for NAIP or PNOA orthophotos + "
                "Delineate-Anything; one date with the default settings scored 0.008, 0.000, "
                "0.081 and 0.065. Set processing_gsd_m=3 and merge_tile_pieces=False (they are "
                "not engine defaults). These prototype scores are not held out: the recipe was "
                "chosen on these areas, and the UC Davis and Mallorca areas lie in the training "
                "geography of the v2 model (DWR and SIGPAC labels); whether FBIS-73M holds X-band "
                "patches was not checked. In a held-out test of the fixed recipe (2026-10-09; "
                "OpenET field boundaries v2.1.1, fields of at least 2,500 m2) F1 was 0.390 at the "
                "Texas A&M AgriLife farm, TX (6 VV collects, April-June 2024; 136 fields) and "
                "0.261 near the Intel Ohio site, OH (4 HH collects, June-September 2024; 77 "
                "fields), against 0.284 and 0.193 for NAIP + Delineate-Anything with its defaults "
                "and 0.403 and 0.310 with processing_gsd_m=3 and merge_tile_pieces=False."
            ),
            "capella": (
                "Capella GEO amplitude (sigma0 dB + 50, one polarization; the median of "
                "co-registered, level-normalised dates in R, G and B). In a held-out test of the "
                "fixed recipe (2026-10-09) in Orleans County, VT (two HH collects of June 2022; "
                "112 OpenET field boundaries v2.1.1 of at least 2,500 m2), zero-shot F1 (IoU >= "
                "0.5) was 0.274, against 0.261 for NAIP + Delineate-Anything with its defaults "
                "and 0.249 with processing_gsd_m=3 and merge_tile_pieces=False. In a pilot "
                "(2026-10-08) near Sevilla, Spain (January 2023 collects, 1,128 SIGPAC parcels of "
                "2025), F1 was 0.50-0.54 reading at 2-3 m without tile merging; that is "
                "in-sample, not held out (the v2 model was trained with SIGPAC labels in Spain, "
                "and the reference is two years newer than the images). Set processing_gsd_m=3 "
                "and merge_tile_pieces=False (they are not engine defaults)."
            ),
        },
    },
    "ftw": {
        "name": "Fields of The World (FTW)",
        "approach": (
            "Semantic segmentation (field / boundary / background) with ftw-tools "
            "pre-trained checkpoints (e.g. PRUE U-Net + EfficientNet), polygonised"
        ),
        "strengths": (
            "Pre-trained on the FTW benchmark (24 countries); bi-temporal Sentinel-2 input "
            "(two windows of R, G, B, NIR = 8 bands for the PRUE models)"
        ),
        "gpu_recommended": True,
        "requires_bands": ["R", "G", "B", "NIR"],
        "supported_sources": ["sentinel2", "hls", "landsat", "local"],
        "label_free": True,
        "fine_tunable": False,
        "reference": (
            "Kerner et al. (2025) Fields of The World: A Machine Learning Benchmark Dataset for "
            "Global Agricultural Field Boundary Segmentation, AAAI 39(27):28151-28159, "
            "doi:10.1609/aaai.v39i27.35034"
        ),
        "install_extra": "ftw",
        "notes": "Checkpoints are calibrated on Sentinel-2 L2A.",
        "source_notes": {
            "landsat": (
                "Landsat composites are passed as harmonised surface reflectance and are out of "
                "distribution (engine_meta['out_of_distribution_source'])."
            ),
            "hls": (
                "HLS composites are passed as harmonised surface reflectance and are out of "
                "distribution (engine_meta['out_of_distribution_source'])."
            ),
        },
    },
    "geoai": {
        "name": "GeoAI field boundary",
        "approach": "Mask R-CNN instance segmentation (geoai-py)",
        "strengths": "Flexible multi-band input; trainable on user reference boundaries",
        "gpu_recommended": True,
        "requires_bands": ["R", "G", "B"],
        "supported_sources": list(_ALL_IMAGERY_SOURCES),
        "label_free": False,
        "fine_tunable": True,
        "reference": (
            "Wu (2026) GeoAI: A Python package for integrating artificial intelligence with "
            "geospatial data analysis and visualization, JOSS 11(118):9605, "
            "doi:10.21105/joss.09605"
        ),
        "install_extra": "geoai",
        "notes": (
            "No field-boundary weights are published; a checkpoint from fine-tuning "
            "(fine_tune=True) or engine_params['checkpoint_path'] is required."
        ),
        "source_notes": {
            "sentinel1": (
                "Sentinel-1 SAR backscatter (VV, VH and VV-VH as R, G, B) is out of "
                "distribution for the COCO-pretrained Mask R-CNN; fine-tune on Sentinel-1 "
                "composites. The effective resolution is about 20 m (IW GRD, 20 x 22 m) "
                "although the pixels are 10 m."
            ),
            "umbra": (
                "Umbra X-band SAR amplitude (one polarization, copied to R, G and B) is out of "
                "distribution for the COCO-pretrained Mask R-CNN; fine-tune on these composites."
            ),
            "capella": (
                "Capella X-band SAR amplitude (one polarization, copied to R, G and B) is out of "
                "distribution for the COCO-pretrained Mask R-CNN; fine-tune on these composites."
            ),
        },
    },
    "dinov3": {
        "name": "DINOv3",
        "approach": "DINOv3 ViT backbone + DPT segmentation head (geoai-py)",
        "strengths": "Strong self-supervised ViT features; optional LoRA fine-tuning",
        "gpu_recommended": True,
        "requires_bands": ["R", "G", "B"],
        "supported_sources": list(_ALL_IMAGERY_SOURCES),
        "label_free": False,
        "fine_tunable": True,
        "reference": "Siméoni et al. (2025) DINOv3, arXiv:2508.10104",
        "install_extra": "dinov3",
        "notes": "Requires a fine-tuned checkpoint (fine_tune=True with reference boundaries).",
        "source_notes": {
            "sentinel1": (
                "Sentinel-1 SAR backscatter (VV, VH and VV-VH as R, G, B) is out of "
                "distribution for the SAT-493M backbone, which was pre-trained on optical "
                "satellite imagery; fine-tune on Sentinel-1 composites. The effective resolution "
                "is about 20 m (IW GRD, 20 x 22 m) although the pixels are 10 m."
            ),
            "umbra": (
                "Umbra X-band SAR amplitude (one polarization, copied to R, G and B) is out of "
                "distribution for the SAT-493M backbone, which was pre-trained on optical "
                "satellite imagery; fine-tune on these composites."
            ),
            "capella": (
                "Capella X-band SAR amplitude (one polarization, copied to R, G and B) is out of "
                "distribution for the SAT-493M backbone, which was pre-trained on optical "
                "satellite imagery; fine-tune on these composites."
            ),
        },
    },
    "prithvi": {
        "name": "Prithvi-EO-2.0",
        "approach": (
            "Prithvi-EO-2.0 ViT foundation model (terratorch): fine-tuned segmentation, or "
            "unsupervised clustering of patch embeddings (mode='embed')"
        ),
        "strengths": (
            "Pre-trained on HLS (Blue, Green, Red, NIR narrow, SWIR1, SWIR2); single-frame "
            "inference in Agribound"
        ),
        "gpu_recommended": True,
        "requires_bands": ["R", "G", "B", "NIR", "SWIR1", "SWIR2"],
        "supported_sources": ["landsat", "sentinel2", "hls", "local"],
        "label_free": True,
        "fine_tunable": True,
        "reference": (
            "Szwarcman et al. (2026) Prithvi-EO-2.0: A versatile multitemporal foundation model "
            "for Earth observation applications, IEEE TGRS 64:1-20, "
            "doi:10.1109/TGRS.2025.3642610"
        ),
        "install_extra": "prithvi",
        "notes": (
            "Label-free only with engine_params mode='embed' (clustering); segmentation mode "
            "needs a fine-tuned checkpoint. The NIR input is NIR_NARROW where the source "
            "defines it (Sentinel-2 B8A, HLS B5), else NIR. The 'prithvi' extra conflicts with "
            "'ftw' (lightning versions)."
        ),
        "source_notes": {
            "landsat": (
                "Landsat defines only NIR (SR_B5): the OLI narrow NIR on Landsat 8/9, but the "
                "broad TM/ETM+ NIR on Landsat 5/7, which differs from the pre-training band."
            ),
        },
    },
    "embedding": {
        "name": "Embedding clustering",
        "approach": (
            "Unsupervised K-means clustering of pre-computed per-pixel embeddings or of the "
            "bands of a local feature raster"
        ),
        "strengths": "Runs on CPU; no labels or model weights needed",
        "gpu_recommended": False,
        "requires_bands": [],
        "supported_sources": [*EMBEDDING_SOURCES, "local"],
        "label_free": True,
        "fine_tunable": False,
        "reference": (
            "Feng et al. (2026) TESSERA: Temporal Embeddings of Surface Spectra for Earth "
            "Representation and Analysis, CVPR 2026 (arXiv:2506.20380); Brown et al. (2025) "
            "AlphaEarth Foundations, arXiv:2507.22291"
        ),
        "install_extra": "embedding",
        "notes": (
            "Clusters are land-cover segments, not field instances. With source='local' it "
            "clusters any feature GeoTIFF (e.g. SAR polarimetric features); set "
            "engine_params['standardize'] when the bands have different units."
        ),
    },
    "sam-auto": {
        "name": "SAM 2.1 automatic masks",
        "approach": (
            "Segment Anything 2.1 automatic mask generation (a grid of point prompts) on large "
            "overlapping tiles, partitioned by score into non-overlapping polygons"
        ),
        "strengths": (
            "No field labels or field-specific training; SAM 2.1 code and weights are Apache-2.0; "
            "tight outlines of fields that look uniform in the image"
        ),
        "gpu_recommended": True,
        "requires_bands": ["R", "G", "B"],
        "supported_sources": list(_ALL_IMAGERY_SOURCES),
        "label_free": True,
        "fine_tunable": False,
        "reference": (
            "Kirillov et al. (2023) Segment Anything, ICCV 2023, arXiv:2304.02643; Ravi et al. "
            "(2025) SAM 2: Segment Anything in Images and Videos, ICLR 2025, arXiv:2408.00714"
        ),
        "install_extra": "sam-auto",
        "notes": (
            "Opt-in: never an engine default, and the agent proposes it only when it is named; "
            "example 28 runs it on NAIP, and examples 24 and 25 with --sam-auto. SAM outlines "
            "every region it sees as one object, not only fields (house lots, rangeland, roads), "
            "and merges look-alike neighbours; keep the LULC filter on (the default). At the "
            "default 50 % tile overlap an object is whole only if it is at most about half a "
            "tile across. The per-source settings were chosen on two Nevada development squares "
            "(2026-10-12), so their scores there are in-sample. A pre-registered evaluation "
            "against Delineate-Anything v2 as released (2026-10-13) kept it on NAIP (F1 0.017 "
            "higher on 46 of example 25's squares, 95 % interval 0.004 to 0.030, both near F1 "
            "0.3: better by its rule), and on SPOT 6/7 and Landsat (8/9; 7/8/9 on the San Juan "
            "box) over four Nevada and New Mexico areas, with no 'better' claim (the Nevada "
            "pairs scored without the crop filter and the New Mexico pairs with it; both Nevada "
            "areas re-scored by an amendment of 2026-10-13); on Sentinel-2 it did not transfer "
            "(F1 0.152 lower). Sources without a piloted setting (usgs-naip-plus, hls, "
            "landsat-pan, spot-pan, umbra, capella, local) run at native scale (1,024-pixel "
            "tiles) with sam2's thresholds and a WARNING. Output is identical on one device and "
            "differs between devices. GPU-time ratios were measured back to back with "
            "Delineate-Anything on the same composites of two New Mexico boxes. SAM 2.1 code "
            "and weights are Apache-2.0; "
            "the sam2 1.1.0 package on PyPI is a third-party build of Meta's code, whose files "
            "equal Meta's commit 2b90b9f (2024-12-15). HPC tiling is not supported."
        ),
        "source_notes": {
            "naip": (
                "Default: 2,048 m tiles, 32 points per side, predicted IoU > 0.8, stability >= "
                "0.95 (sam2's own thresholds). Pre-registered evaluation against "
                "Delineate-Anything v2 as released: F1 0.017 higher on 46 of example 25's squares "
                "(95 % interval 0.004 to 0.030; both near F1 0.3). GPU time 6.1 times "
                "Delineate-Anything's, measured back to back on two large New Mexico boxes (Apple "
                "M2 Max, MPS); 3.1 times on example 25's 5 km squares. In the pilot (development "
                "evidence), on one New Mexico square of small fields that played no part in the "
                "choice, F1 was 0.51 against Delineate-Anything's 0.43 (paired difference +0.07, "
                "1 km-block bootstrap interval +0.03 to +0.12)."
            ),
            "usgs-naip-plus": (
                "The same imagery as NAIP, but no setting was piloted on this source: it runs at "
                "native scale (1,024-pixel tiles) with sam2's thresholds and a WARNING."
            ),
            "spot": (
                "Default: 6,144 m tiles, 32 points per side, predicted IoU > 0.6, stability >= "
                "0.95, chosen on growing-season composites (one scene in Diamond Valley, the "
                "median of five in Mason Valley). Pre-registered evaluation: mean F1 difference "
                "+0.26 over four Nevada and New Mexico areas (median fields 1.1-48 ha), positive "
                "in all four (Mason Valley's 1 km-block interval, -0.003 to +0.161, just includes "
                "0); as pre-registered, the Nevada pairs are scored without the crop filter and "
                "the New Mexico pairs with it (+0.21 with all four unfiltered, a secondary "
                "result), and both Nevada areas were scored again by an amendment of 2026-10-13, "
                "after the first scores had been read; four areas support no 'better' claim. GPU "
                "time 6.4 times Delineate-Anything's, measured back to back on two large New "
                "Mexico boxes. In the "
                "pilot (development evidence), on one New Mexico square of small fields (a "
                "winter-only median) F1 was 0.26 against Delineate-Anything's 0.39 (paired "
                "difference -0.13, 1 km-block bootstrap interval -0.20 to -0.05). In a multi-tile "
                "development check (2026-10-12: three development squares scored inside runs on "
                "25 km composites of many tiles, against their single-tile runs) F1 changed by "
                "-0.09, +0.12 and +0.02: not consistent by the pre-registered rule (each change "
                ">= -0.05). Where it fell (Diamond Valley, the same scene in both runs) more "
                "polygons lay on unmapped ground; part of the +0.12 (Mason Valley) comes from a "
                "different composite, 7 scenes against 5."
            ),
            "sentinel2": (
                "Did not transfer: in the pre-registered evaluation its F1 was 0.15 below "
                "Delineate-Anything v2's on 46 of example 25's squares (95 % interval -0.18 to "
                "-0.13), lower on 45 of them, with about a quarter as many polygons; it was ahead "
                "in three Nevada and New Mexico areas with median fields of 7.7-48 ha and level in "
                "a fourth (1.1 ha). The setting is unchanged and was not tried again. Available "
                "with a WARNING: 5,120 m tiles, 32 points per side, predicted IoU > 0.8, "
                "stability >= 0.9; 17.0 times Delineate-Anything's GPU time per km2 (0.59 against "
                "0.035 s), measured back to back on two large New Mexico boxes. In the pilot "
                "(development evidence), on one New Mexico square of small fields F1 was 0.35 "
                "against Delineate-Anything's 0.40 (paired difference -0.05; its 1 km-block "
                "bootstrap interval, -0.12 to +0.01, includes 0). In a multi-tile development "
                "check (2026-10-12: three development squares scored inside runs on 25 km "
                "composites of many tiles) F1 stayed within 0.02 of their single-tile runs."
            ),
            "landsat": (
                "Available with a WARNING about its cost: 7,680 m tiles, 32 points per side, "
                "predicted IoU > 0.6, stability >= 0.8; 20.7 times Delineate-Anything's GPU time "
                "per km2 (0.23 against 0.011 s), measured back to back on two large New Mexico "
                "boxes. Pre-registered evaluation (Landsat 8/9; 7/8/9 on the San Juan box): mean "
                "F1 difference +0.13 over four Nevada and New Mexico areas (Mason Valley -0.04, "
                "interval -0.11 to +0.03), with the same filter states and amendment as SPOT 6/7 "
                "(+0.09 with all four pairs unfiltered, a secondary result); four "
                "areas support no 'better' claim, and at 30 m Delineate-Anything runs outside its "
                "training range (0.25-10 m). Outlines are coarse at 30 m. In a multi-tile "
                "development check (2026-10-12: three development squares scored inside runs on "
                "25 km composites of many tiles, against their single-tile runs) F1 changed by "
                "-0.11, +0.09 and +0.06: not consistent by the pre-registered rule (each change "
                ">= -0.05)."
            ),
            "sentinel1": (
                "No default: set sam_tile_m, points_per_side, pred_iou_thresh and "
                "stability_score_thresh. On the two development squares the setting the "
                "selection rule chose was level with Delineate-Anything (mean F1 0.437 against "
                "0.441, in-sample; the highest of its 72 settings had 0.443) at about 22 times "
                "its GPU time on 5 km squares."
            ),
        },
    },
    "crop-sequence": {
        "name": "Crop-sequence zones",
        "approach": (
            "Zones of uniform multi-year crop history from annual crop-type maps (CSB-like): "
            "cultivated mask, per-year focal mode, barrier classes, equal-sequence regions, "
            "small-region merge"
        ),
        "strengths": (
            "No imagery, model weights or GPU; seconds of CPU per 25 km2; per-zone crop history"
        ),
        "gpu_recommended": False,
        "requires_bands": [],
        "supported_sources": [*_CROP_SEQUENCE_SOURCES, "local"],
        "label_free": True,
        "fine_tunable": False,
        "reference": (
            "Hunt et al. (2024) Crop sequence boundaries using USDA National Agricultural "
            "Statistics Service historic cropland data layers, Statistical Journal of the IAOS "
            "40:237-246, doi:10.3233/SJI-230078"
        ),
        "install_extra": "crop-sequence",
        "notes": (
            "Zones of uniform crop history, not field instances. Local rasters need "
            "engine_params value_scale='categorical' and a dataset or class lists."
        ),
    },
    "ensemble": {
        "name": "Ensemble",
        "approach": "Multi-engine consensus (intersection, union or vote)",
        "strengths": "Combines several engines or models on the same composite",
        "gpu_recommended": True,
        "requires_bands": [],
        "supported_sources": list(_ALL_IMAGERY_SOURCES),
        "label_free": True,
        "fine_tunable": False,
        "reference": "N/A (combines the member engines' references)",
        "install_extra": "all",
        "notes": (
            "label_free and source support depend on the members (default "
            "delineate-anything + ftw, both label-free); the members, explicit or default, "
            "are checked against the source when the configuration is validated."
        ),
    },
}

ENGINE_CLASSES: dict[str, str] = {
    "delineate-anything": "agribound.engines.delineate_anything:DelineateAnythingEngine",
    "ftw": "agribound.engines.ftw:FTWEngine",
    "geoai": "agribound.engines.geoai_field:GeoAIEngine",
    "dinov3": "agribound.engines.dinov3:DINOv3Engine",
    "prithvi": "agribound.engines.prithvi:PrithviEngine",
    "embedding": "agribound.engines.embedding:EmbeddingEngine",
    "sam-auto": "agribound.engines.sam_auto:SAMAutoEngine",
    "crop-sequence": "agribound.engines.crop_sequence:CropSequenceEngine",
    "ensemble": "agribound.engines.ensemble:EnsembleEngine",
}
"""Engine name -> ``"module:Class"`` import path used by :func:`agribound.engines.get_engine`."""


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def list_sources() -> dict[str, dict[str, Any]]:
    """List all satellite sources and their metadata.

    Returns
    -------
    dict[str, dict]
        Deep copy of :data:`SOURCE_REGISTRY` (safe to mutate).

    Examples
    --------
    >>> from agribound import list_sources
    >>> for name, info in list_sources().items():
    ...     print(name, info["resolution_m"], info["value_scale"])
    """
    return copy.deepcopy(SOURCE_REGISTRY)


def list_engines() -> dict[str, dict[str, Any]]:
    """List all delineation engines and their metadata.

    Returns
    -------
    dict[str, dict]
        Deep copy of :data:`ENGINE_REGISTRY` (safe to mutate).

    Examples
    --------
    >>> from agribound import list_engines
    >>> for name, info in list_engines().items():
    ...     print(name, info["approach"])
    """
    return copy.deepcopy(ENGINE_REGISTRY)


def list_sam_backends() -> list[str]:
    """Return the SAM backends accepted for the refinement stage."""
    return list(SAM_REFINE_BACKENDS)


def engine_supports_source(engine: str, source: str) -> bool:
    """Return *True* if *engine* accepts rasters from *source*.

    Parameters
    ----------
    engine : str
        Engine name (case-insensitive).
    source : str
        Source name (case-insensitive).

    Returns
    -------
    bool
        *False* for unknown engine or source names.
    """
    info = ENGINE_REGISTRY.get(str(engine).lower().strip())
    if info is None:
        return False
    return str(source).lower().strip() in info["supported_sources"]


def _source_info(source: str) -> dict[str, Any]:
    key = str(source).lower().strip()
    info = SOURCE_REGISTRY.get(key)
    if info is None:
        raise ValueError(f"Unknown source {source!r}. Available: {list(SOURCE_REGISTRY)}")
    return info


def source_value_scale(source: str) -> str:
    """Return the pixel value scale of the composite built for *source*.

    Parameters
    ----------
    source : str
        Source name.

    Returns
    -------
    str
        One of :data:`VALUE_SCALES`.

    Raises
    ------
    ValueError
        If the source is unknown.
    """
    return _source_info(source)["value_scale"]


def source_year_range(
    source: str, tessera_version: str | None = None
) -> tuple[int, int | None] | None:
    """Return the ``(first, last)`` years with data for *source*.

    Parameters
    ----------
    source : str
        Source name.
    tessera_version : str or None
        For ``"tessera-embedding"`` only: dataset version (``"v1"``, ``"v1.1"``
        or ``"v2"``). *None* returns the registry default (v1).

    Returns
    -------
    tuple[int, int or None] or None
        ``last`` is *None* for missions that are still acquiring. *None* means
        no constraint (local files).

    Raises
    ------
    ValueError
        If the source or TESSERA version is unknown.
    """
    info = _source_info(source)
    if str(source).lower().strip() == "tessera-embedding" and tessera_version is not None:
        if tessera_version not in TESSERA_YEAR_RANGES:
            raise ValueError(
                f"Unknown tessera_version {tessera_version!r}. "
                f"Choose from {tuple(TESSERA_YEAR_RANGES)}"
            )
        return TESSERA_YEAR_RANGES[tessera_version]
    return info["year_range"]


def engine_notes(engine: str, source: str | None = None) -> list[str]:
    """Return the notes of *engine* that apply to *source*.

    Parameters
    ----------
    engine : str
        Engine name (case-insensitive).
    source : str or None
        Source name. *None* returns only the notes that apply to every source.

    Returns
    -------
    list[str]
        ``ENGINE_REGISTRY[engine]["notes"]`` followed by the entry of
        ``ENGINE_REGISTRY[engine]["source_notes"]`` for *source*, if any.

    Raises
    ------
    ValueError
        If the engine is unknown.
    """
    info = ENGINE_REGISTRY.get(str(engine).lower().strip())
    if info is None:
        raise ValueError(f"Unknown engine {engine!r}. Available: {list(ENGINE_REGISTRY)}")
    notes = [info["notes"]] if info.get("notes") else []
    if source is not None:
        extra = (info.get("source_notes") or {}).get(str(source).lower().strip())
        if extra:
            notes.append(extra)
    return notes


def canonical_bands(source: str) -> dict[str, str]:
    """Return the canonical-name -> native-band mapping for *source* (empty if none)."""
    return dict(_source_info(source).get("canonical_bands") or {})


def supported_sources(engine: str) -> list[str]:
    """Return the sources accepted by *engine*.

    Raises
    ------
    ValueError
        If the engine is unknown.
    """
    info = ENGINE_REGISTRY.get(str(engine).lower().strip())
    if info is None:
        raise ValueError(f"Unknown engine {engine!r}. Available: {list(ENGINE_REGISTRY)}")
    return list(info["supported_sources"])


__all__ = [
    "CANONICAL_BAND_NAMES",
    "CATEGORICAL_ENGINES",
    "CROP_SEQUENCE_SOURCES",
    "EMBEDDING_SOURCES",
    "ENGINE_CLASSES",
    "ENGINE_REGISTRY",
    "ENSEMBLE_DEFAULT_MEMBERS",
    "GEE_IMAGERY_SOURCES",
    "SAM_REFINE_BACKENDS",
    "SAR_SOURCES",
    "SCENE_RANKED_SOURCES",
    "SCENE_SURVEY_SOURCES",
    "SOURCE_REGISTRY",
    "TESSERA_YEAR_RANGES",
    "VALUE_SCALES",
    "VHR_SAR_SOURCES",
    "canonical_bands",
    "engine_notes",
    "engine_supports_source",
    "list_engines",
    "list_sam_backends",
    "list_sources",
    "sam_refine_backends",
    "source_value_scale",
    "source_year_range",
    "supported_sources",
]
