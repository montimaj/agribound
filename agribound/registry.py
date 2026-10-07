"""
Source and engine registries: the single source of truth for Agribound metadata.

This module imports nothing heavy (standard library only) so it can be used by
the configuration layer, the CLI and documentation tooling without pulling in
GDAL, torch or Earth Engine.

Facts encoded here were checked against the upstream catalogues and packages
on 2026-09-26 (GEE STAC entries and live collection queries, ftw-tools 2.0.0b5,
Delineate-Anything v1/v2 model cards, geoai-py 0.43.1, terratorch 1.2.13,
geotessera 0.10.2).

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
- ``"uint8"`` -- 8-bit digital numbers 0-255 (NAIP, USGS NAIP Plus).
- ``"unit"`` -- float32 TOA reflectance as stored (Landsat PAN), without x10000 scaling.
- ``"dn"`` -- per-band medians (or greenest-pixel selections) of the scenes' raw
  integer digital numbers, exported as float32 (a median of an even number of
  scenes can be a half-integer); radiometry unverified (SPOT 6/7).
- ``"embedding"`` -- pre-computed embedding vectors (Google, TESSERA).
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
    "embedding",
    "unknown",
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
_ALL_IMAGERY_SOURCES = [
    "landsat",
    "landsat-pan",
    "sentinel2",
    "hls",
    "naip",
    "usgs-naip-plus",
    "spot",
    "spot-pan",
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
}

ENSEMBLE_DEFAULT_MEMBERS: tuple[str, ...] = ("delineate-anything", "ftw")
"""Members of ``engine="ensemble"`` when ``engine_params["engines"]`` is not set.

:data:`agribound.engines.ensemble.DEFAULT_MEMBERS` is this tuple; it lives here so
that :mod:`agribound.config` can validate the defaults without importing the
engine (and geopandas).
"""

GEE_IMAGERY_SOURCES: tuple[str, ...] = tuple(_OPTICAL_GEE_SOURCES)
"""Imagery sources whose composites are built on Google Earth Engine."""

EMBEDDING_SOURCES: tuple[str, ...] = ("google-embedding", "tessera-embedding")
"""Pre-computed embedding sources."""

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
        "approach": "Unsupervised K-means clustering of pre-computed per-pixel embeddings",
        "strengths": "Runs on CPU; no labels or model weights needed",
        "gpu_recommended": False,
        "requires_bands": [],
        "supported_sources": list(EMBEDDING_SOURCES),
        "label_free": True,
        "fine_tunable": False,
        "reference": (
            "Feng et al. (2026) TESSERA: Temporal Embeddings of Surface Spectra for Earth "
            "Representation and Analysis, CVPR 2026 (arXiv:2506.20380); Brown et al. (2025) "
            "AlphaEarth Foundations, arXiv:2507.22291"
        ),
        "install_extra": "embedding",
        "notes": "Clusters are land-cover segments, not field instances.",
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
    "EMBEDDING_SOURCES",
    "ENGINE_CLASSES",
    "ENGINE_REGISTRY",
    "ENSEMBLE_DEFAULT_MEMBERS",
    "GEE_IMAGERY_SOURCES",
    "SAM_REFINE_BACKENDS",
    "SOURCE_REGISTRY",
    "TESSERA_YEAR_RANGES",
    "VALUE_SCALES",
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
