"""
Configuration management for Agribound.

Provides the ``AgriboundConfig`` dataclass that controls every stage of the
delineation pipeline: satellite source, delineation engine, Earth Engine
export and authentication settings, compositing, post-processing, LULC crop
filtering, optional SAM refinement, fine-tuning, caching and provenance.

Configurations can be created programmatically, loaded from YAML files, or
built from CLI flags. Every construction path runs the same validation
(:meth:`AgriboundConfig._validate`); unknown keys are rejected.
"""

from __future__ import annotations

import copy
import datetime as _dt
import logging
import math
import os
import re
import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from agribound.registry import (
    CATEGORICAL_ENGINES,
    CROP_SEQUENCE_SOURCES,
    ENGINE_REGISTRY,
    ENSEMBLE_DEFAULT_MEMBERS,
    GEE_IMAGERY_SOURCES,
    SAM_REFINE_BACKENDS,
    SOURCE_REGISTRY,
    TESSERA_YEAR_RANGES,
    VHR_SAR_SOURCES,
    engine_supports_source,
    source_year_range,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid choices
# ---------------------------------------------------------------------------

VALID_SOURCES = tuple(SOURCE_REGISTRY)
VALID_ENGINES = tuple(ENGINE_REGISTRY)

VALID_EXPORT_METHODS = ("local", "gdrive", "gcs")
VALID_OUTPUT_FORMATS = ("gpkg", "geojson", "parquet")
VALID_COMPOSITE_METHODS = ("median", "greenest", "max_ndvi")
VALID_DEVICES = ("auto", "cuda", "cpu", "mps")
VALID_S2_CLOUD_MASKS = ("scl", "cloud_score_plus")
VALID_TESSERA_VERSIONS = tuple(TESSERA_YEAR_RANGES)
VALID_LULC_DATASETS = ("auto", "nlcd", "cdl", "dynamic_world", "c3s")
VALID_LULC_ON_ERROR = ("raise", "warn")
VALID_LULC_MODES = ("server", "raster")
VALID_LULC_NODATA_POLICIES = ("keep", "drop")
VALID_SAM_BACKENDS = SAM_REFINE_BACKENDS
VALID_FINE_TUNE_SPLITS = ("block", "random", "column")
VALID_GOOGLE_EMBEDDING_BACKENDS = ("gee", "source_coop")
#: Landsat missions whose panchromatic band ``source="landsat-pan"`` can use, in
#: canonical order (see ``landsat_pan_missions``).
VALID_LANDSAT_PAN_MISSIONS = ("LE07", "LC08", "LC09")
VALID_AOI_SELECTIONS = ("representative_point", "intersects", "clip", "none")
#: Settings of the open VHR SAR sources (``umbra``, ``capella``).
VALID_VHR_SAR_POLARIZATIONS = ("auto", "HH", "VV")
VALID_VHR_SAR_COREGISTER = ("auto", "optical", "sar", "none")
VALID_VHR_SAR_BANDS = ("median", "median+dates")
VALID_VHR_SAR_CALIBRATION = ("any", "full")
#: Band mapping the engines read for ``vhr_sar_bands="median+dates"`` when *bands* is
#: not given (see :func:`effective_bands`): the median in R and B and the mean of the
#: closest pair of dates in G.
VHR_SAR_MEDIAN_DATES_BANDS: dict[str, int] = {"R": 1, "G": 2, "B": 1}

#: File extensions that imply an output format.
OUTPUT_EXTENSIONS: dict[str, str] = {
    ".gpkg": "gpkg",
    ".geojson": "geojson",
    ".json": "geojson",
    ".parquet": "parquet",
    ".geoparquet": "parquet",
}

_DEFAULT_OUTPUT_PATH = "fields.gpkg"
_MAX_SEED = 2**32 - 1  # numpy's legacy seeding range
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EPSG_RE = re.compile(r"^EPSG:(\d+)$", re.IGNORECASE)
# ee.data.setWorkloadTag: 1-63 chars, alphanumeric at both ends; '-', '_', '.' inside.
_WORKLOAD_TAG_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]{0,61}[A-Za-z0-9])?$")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgriboundConfig:
    """Configuration for an Agribound delineation run.

    Parameters
    ----------
    source : str
        Satellite source: ``"landsat"``, ``"sentinel2"``, ``"hls"``,
        ``"landsat-pan"``, ``"naip"``, ``"usgs-naip-plus"``, ``"spot"``, ``"spot-pan"``,
        ``"sentinel1"``, ``"umbra"``, ``"capella"``, ``"local"``, ``"google-embedding"``,
        ``"tessera-embedding"``, ``"cdl-sequence"`` or ``"aci-sequence"`` (see
        :func:`agribound.list_sources`).
    engine : str
        Delineation engine: ``"delineate-anything"``, ``"ftw"``, ``"geoai"``,
        ``"dinov3"``, ``"prithvi"``, ``"embedding"``, ``"sam-auto"``,
        ``"crop-sequence"`` or ``"ensemble"`` (see :func:`agribound.list_engines`).
        The engine must support the source. ``"sam-auto"`` checks its
        ``engine_params`` here (:func:`agribound.engines.sam_auto.check_params`)
        and refuses ``sam_refine`` and ``sam_model`` (its model is
        ``engine_params["sam_model"]``).
    year : int
        Target year. Must lie within the source's available years
        (:func:`agribound.registry.source_year_range`). For ``cdl-sequence``
        and ``aci-sequence`` it is the last map year of the window (see
        *crop_sequence_years*).
    study_area : str
        Path to a GeoJSON / Shapefile / GeoParquet / GeoPackage, a GEE vector
        asset ID (e.g. ``"projects/my-project/assets/my_aoi"``), a bounding box
        ``"bbox:minx,miny,maxx,maxy"`` in EPSG:4326, or a WKT geometry in
        EPSG:4326. May be empty only for ``source="local"``.
    output_path : str
        Destination file for the output field boundary vectors. Its extension
        (``.gpkg``, ``.geojson``/``.json``, ``.parquet``/``.geoparquet``)
        must match *output_format*; when *output_format* is left at its
        default it is inferred from the extension.
    output_format : str
        Output vector format: ``"gpkg"`` | ``"geojson"`` | ``"parquet"``
        (GeoParquet with fiboa-style column names).
    gee_project : str or None
        Google Earth Engine project ID. Required for GEE imagery sources;
        when not given it is resolved from ``GEE_PROJECT``, then ``gcloud
        config``, then the ``project_id`` of the credentials file
        (*gee_service_account_key*, else ``AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY``,
        else ``GOOGLE_APPLICATION_CREDENTIALS``; see
        :func:`agribound.auth.project_from_credentials`).
    export_method : str
        GEE export method: ``"local"`` (direct download, default) |
        ``"gdrive"`` | ``"gcs"``.
    gcs_bucket : str or None
        Google Cloud Storage bucket name. Required when *export_method* is
        ``"gcs"``.
    usgs_service_url, usgs_state, usgs_allow_year_fallback, usgs_timeout_s, usgs_retries
        USGS NAIP Plus ImageServer settings (``source="usgs-naip-plus"``).
        ``usgs_allow_year_fallback`` allows footprints from ``year ± 1``.
    composite_method : str
        Compositing strategy: ``"median"`` | ``"greenest"`` | ``"max_ndvi"``.
        ``"greenest"`` and its alias ``"max_ndvi"`` need NIR and red bands, so
        they are rejected for ``landsat-pan``, ``spot-pan`` and ``sentinel1``;
        ``umbra`` and ``capella`` accept only ``"median"``.
    date_range : tuple[str, str] or None
        Override the default full-year window with an explicit
        ``("YYYY-MM-DD", "YYYY-MM-DD")`` window (e.g. a growing season), end
        inclusive. A one-day window (start = end) gives a single-date
        composite; :func:`agribound.composites.scenes.survey_scenes` ranks the
        dates of a window.
    cloud_cover_max : int
        Maximum scene cloud cover percentage, 0-100 (default 20): the scene
        filter of the ``landsat``, ``landsat-pan``, ``sentinel2``, ``hls``,
        ``spot`` and ``spot-pan`` composites. The other sources, including
        ``naip`` and ``sentinel1`` (radar), ignore it.
    export_crs : str
        CRS of exported composites: ``"utm"`` (UTM zone of the AOI centroid,
        default) or an ``"EPSG:XXXX"`` code.
    s2_cloud_mask : str
        Sentinel-2 cloud mask: ``"scl"`` (scene classification, default) or
        ``"cloud_score_plus"`` (Google Cloud Score+ ``cs_cdf``).
    cloud_score_threshold : float
        Minimum Cloud Score+ ``cs_cdf`` kept when
        ``s2_cloud_mask="cloud_score_plus"`` (default 0.60).
    naip_resolution_m : float
        NAIP export resolution in metres (default 1.0; the native GSD is
        0.6 m in most states since 2018).
    landsat_pan_missions : str or tuple[str, ...]
        Missions whose panchromatic band ``source="landsat-pan"`` composites
        (ignored for other sources). ``"auto"`` (default) never mixes the two
        PAN bandpasses: Landsat 8/9 OLI (0.50-0.68 µm; ``LC08``, ``LC09``)
        whenever the date window overlaps their record (from 2013-03-18), else
        Landsat 7 ETM+ (0.52-0.90 µm; ``LE07``). Or a list of ``"LE07"``,
        ``"LC08"`` and ``"LC09"`` (a comma-separated string is accepted):
        exactly those missions, where their record overlaps the window; a list
        with ``LE07`` and an OLI mission mixes the two bandpasses in one
        median, with a WARNING. For ``landsat-pan``, a setting with no mission
        whose record overlaps the year (or ``date_range``) is rejected with a
        :class:`ValueError` (Landsat 7 1999-05-28 to 2024-01-19, Landsat 8
        from 2013-03-18, Landsat 9 from 2021-10-31).
    crop_sequence_years : int
        The number of annual crop-type maps, ending in *year*, stacked for
        ``cdl-sequence`` and ``aci-sequence`` (default 8, as the USDA NASS Crop
        Sequence Boundaries; 2-20); other sources ignore it. The first map
        year, ``year - crop_sequence_years + 1``, must be at least the
        source's first year (CDL 2008, ACI 2011). Engine ``crop-sequence`` on
        a local stack uses every band of the file instead.
    vhr_sar_collect_ids : tuple[str, ...] or None
        ``umbra``/``capella`` only (as are all ``vhr_sar_*`` fields): the
        collect IDs to composite (a list or a comma-separated string), instead
        of the automatic selection. They must share orbit pass, look side and
        polarization, lie within 5 degrees of incidence and 25 degrees of view
        azimuth, cover the study area and fall in the date window.
        ``vhr_sar_max_collects`` does not apply then.
    vhr_sar_max_collects : int
        Maximum number of collects of the automatic selection (default 6),
        spread over the date window. 1 builds a single-date composite (with a
        WARNING: in the Umbra prototype single dates scored F1 0.00-0.08 with
        Delineate-Anything's default settings; read at 2 m without tile merging,
        0.15-0.40 against 0.22-0.42 for the median at three of the four sites,
        and about 0 at the fourth).
    vhr_sar_polarization : str
        ``"auto"`` (default: the polarization of the largest consistent group of
        collects), ``"HH"`` or ``"VV"``. HH and VV are never mixed.
    vhr_sar_resolution_m : float
        Pixel size of the composite in metres (default 1.0; 0.5 m scored worse).
    vhr_sar_coregister : str
        Co-registration of the dates: ``"auto"`` (default: every date to a
        reference date, then the stack to an optical base of 1 m or finer:
        *vhr_sar_coregister_base* when given, else USGS NAIP Plus where it has
        imagery; without a base, SAR to SAR only, with a WARNING), ``"optical"``
        (as ``"auto"``, but an error without a base), ``"sar"`` (dates to the
        reference date only) or ``"none"``. Sentinel-2 is never used as the base.
    vhr_sar_coregister_base : str or None
        GeoTIFF orthophoto of 1 m or finer (any CRS) used as the co-registration
        base, e.g. outside the US. Not used with ``vhr_sar_coregister`` ``"sar"``
        or ``"none"``.
    vhr_sar_bands : str
        ``"median"`` (default: one band, the median amplitude, read as R, G and
        B) or ``"median+dates"`` (experimental: the median, the mean of the two
        dates closest in time and each date; needs at least 2 collects). With
        ``"median+dates"`` and no *bands*, the engines read
        ``{"R": 1, "G": 2, "B": 1}`` (:func:`effective_bands`); *bands* itself
        is not changed.
    vhr_sar_calibration : str
        ``capella`` only: ``"any"`` (default) or ``"full"`` (drop collects whose
        calibration is ``"limited"`` or ``"partial"``, about 19 % of products).
        Umbra products are not calibrated.
    vhr_sar_max_download_gb : float
        Maximum gigabytes (1e9 bytes) of products one composite may download
        (default 5.0); products already in the cache do not count. Not hashed.
    vhr_sar_cache_dir : str or None
        Directory for the archive indexes and the downloaded products (default
        ``$XDG_CACHE_HOME/agribound/vhr_sar``, else ``~/.cache/agribound/vhr_sar``).
        Not hashed.
    tessera_version : str
        TESSERA dataset version: ``"v1"`` (default), ``"v1.1"`` or ``"v2"``
        (beta).
    tessera_variant : str or None
        TESSERA dataset variant (*None* = geotessera's default for the version).
    embedding_cache_dir : str or None
        Shared directory for embedding registries/tiles (e.g. HPC scratch).
    google_embedding_backend : str
        Where Google Satellite Embeddings are read from: ``"gee"`` (default)
        or ``"source_coop"`` (public COG mirror, no Earth Engine compute).
    local_tif_path : str or None
        Path to a local GeoTIFF when *source* is ``"local"``.
    bands : dict or None
        Canonical band name -> 1-based band index, e.g.
        ``{"R": 1, "G": 2, "B": 3, "NIR": 4}``; passed to
        :func:`agribound.engines.base.get_canonical_band_indices` by engines
        that support it.
    aoi_selection : str
        How predicted polygons are restricted to the study-area geometry.
        Composites cover the study area's bounding box in the export CRS
        (no polygon masking), so engines can return polygons inside that
        box but outside an irregular study area. Applied after delineation
        and SAM refinement, before post-processing, the LULC filter and
        evaluation:

        - ``"representative_point"`` (default): keep polygons whose
          :meth:`~shapely.Geometry.representative_point` (a point guaranteed
          to lie inside the polygon) intersects the study area, i.e. lies
          inside it or on its boundary. Fields crossing the outline are kept
          whole or dropped whole.
        - ``"intersects"``: keep polygons that intersect the study area.
        - ``"clip"``: clip polygons to the study area (pieces that are not
          polygonal are discarded; slivers below *min_field_area_m2* are
          then removed by post-processing).
        - ``"none"``: keep every polygon.

        Skipped when no study area is set (allowed only for
        ``source="local"``). The rule and the polygon counts before and
        after are recorded in the provenance record (``aoi_selection``). The
        reference boundaries used for evaluation are selected with the same
        rule (with ``"none"``: the references intersecting the study area).
    min_field_area_m2 : float
        Minimum field polygon area in m² (default 2500), in EPSG:6933. It is
        applied before smoothing/simplification (holes smaller than it are
        filled then) and again after them, so no output polygon is smaller.
    simplify_tolerance : float
        Douglas-Peucker simplification tolerance in **metres** (default 2.0;
        0 disables simplification).
    lulc_filter : bool
        Drop polygons whose crop fraction in a land-use/land-cover dataset is
        below *lulc_crop_threshold* (default *True*). The LULC datasets are
        read from Earth Engine, so this needs GEE access even for non-GEE
        sources (see *lulc_on_error*).
    lulc_crop_threshold : float
        Minimum crop fraction, 0-1 (default 0.3).
    lulc_batch_size : int
        Polygons per Earth Engine ``reduceRegions`` request (default 200).
    lulc_dataset : str
        ``"auto"`` (documented routing by region and year) | ``"nlcd"`` |
        ``"cdl"`` | ``"dynamic_world"`` | ``"c3s"``.
    lulc_on_error : str
        ``"raise"`` (default): a failed LULC filter aborts the run.
        ``"warn"``: continue with unfiltered polygons, log a warning and flag
        the failure in the provenance record.
    lulc_mode : str
        ``"server"`` (Earth Engine ``reduceRegions``, default) or ``"raster"``
        (download a LULC raster during the composite stage, then compute zonal
        statistics locally, so the delineation stage can run offline).
    lulc_nodata_policy : str
        Polygons without valid LULC pixels are kept and flagged (``"keep"``,
        default) or dropped (``"drop"``).
    lulc_tree_crops : bool
        Count tree cover as crop in the LULC filter (default *False*), for
        tree crops such as orchards and plantations: Dynamic World files
        plantations under ``trees``, so its crop value becomes the annual
        median of the ``crops`` + ``trees`` probability, and C3S adds its tree
        cover classes (50, 60-62, 70-72, 80-82, 90) to the cropland classes;
        with these two datasets the filter then no longer removes forest.
        NLCD and CDL are unchanged (NLCD class 82 already includes orchards
        and vineyards), so with them the filter still removes forest.
    sam_refine : bool
        Refine polygons with box-prompted SAM after delineation (default
        *False*). ``engine_params["sam_refine"]`` is still honoured. Refused
        with engine ``"sam-auto"``, whose polygons are already SAM masks.
    sam_backend : str
        ``"sam2"`` (default), ``"sam2.1"``, ``"sam3"`` or ``"sam3-hf"``. The two SAM 3
        backends are untested in 1.1.0: they have not been run end to end (the
        ``facebook/sam3`` weights are gated), and a WARNING is logged when one is loaded.
    sam_model : str or None
        SAM model id or size (*None* = backend default).
    sam_min_crop_px : int
        Polygons whose padded bounding box is smaller than this many pixels
        are not refined (default 64).
    sam_crop_padding : float
        Padding around each polygon's bounding box as a fraction of its size
        (default 0.15).
    device : str
        Compute device: ``"auto"`` | ``"cuda"`` | ``"cpu"`` | ``"mps"``.
    tile_size : int
        Maximum tile dimension in pixels used when splitting large GEE/USGS
        downloads (default 10 000).
    n_workers : int
        Number of worker processes for PyTorch data loaders (default 4; 0
        loads data in the main process).
    seed : int
        Random seed for Python, NumPy, torch and Lightning (default 42).
    reference_boundaries : str or None
        Existing field boundaries (``.shp``, ``.gpkg``, ``.geojson``,
        ``.parquet``) for fine-tuning or evaluation.
    fine_tune : bool
        When *True*, fine-tune the engine on *reference_boundaries* before
        inference.
    fine_tune_epochs : int
        Number of fine-tuning epochs (default 20).
    fine_tune_val_split : float
        Fraction of chips reserved for validation, in (0, 1) (default 0.2).
    fine_tune_split : str
        How chips are split into train/validation: ``"block"`` (spatial
        blocks, default), ``"random"`` or ``"column"`` (groups from
        *fine_tune_split_column*).
    fine_tune_block_size_m : float
        Block edge length in metres for ``fine_tune_split="block"``
        (default 5000).
    fine_tune_split_column : str or None
        Column of *reference_boundaries* used as group id when
        ``fine_tune_split="column"``.
    gee_service_account_key : str or None
        Path to a GEE service-account JSON key (also read from
        ``AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY``; Application Default Credentials
        honour ``GOOGLE_APPLICATION_CREDENTIALS``).
    gee_high_volume : bool
        Use the Earth Engine high-volume endpoint (default *False*).
    gee_max_requests : int
        Maximum concurrent Earth Engine download requests of this process
        (default 8). The sum over concurrent jobs should stay within the
        project's limit (40 concurrent requests by default).
    gee_workload_tag : str or None
        Earth Engine workload tag for per-run EECU accounting.
    cache_dir : str or None
        Directory for intermediate files. Default ``<output dir>/.agribound_cache``.
    overwrite : bool
        Re-run and overwrite an existing output (default *False*).
    provenance : bool
        Write ``<output_path>.provenance.json`` (default *True*).
    engine_params : dict
        Engine-specific keyword arguments.
    """

    # Required -----------------------------------------------------------------
    source: str = "sentinel2"
    engine: str = "delineate-anything"
    year: int = 2024
    study_area: str = ""
    output_path: str = _DEFAULT_OUTPUT_PATH

    # Output -------------------------------------------------------------------
    output_format: str = "gpkg"

    # GEE ----------------------------------------------------------------------
    gee_project: str | None = None
    export_method: str = "local"
    gcs_bucket: str | None = None
    gee_service_account_key: str | None = None
    gee_high_volume: bool = False
    gee_max_requests: int = 8
    gee_workload_tag: str | None = None

    # USGS ---------------------------------------------------------------------
    usgs_service_url: str = (
        "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer"
    )
    usgs_state: str | None = None
    usgs_allow_year_fallback: bool = False
    usgs_timeout_s: int = 120
    usgs_retries: int = 5

    # Compositing --------------------------------------------------------------
    composite_method: str = "median"
    date_range: tuple[str, str] | None = None
    cloud_cover_max: int = 20
    export_crs: str = "utm"
    s2_cloud_mask: str = "scl"
    cloud_score_threshold: float = 0.60
    naip_resolution_m: float = 1.0
    landsat_pan_missions: str | tuple[str, ...] = "auto"
    crop_sequence_years: int = 8

    # Open VHR SAR (umbra, capella) --------------------------------------------
    vhr_sar_collect_ids: tuple[str, ...] | None = None
    vhr_sar_max_collects: int = 6
    vhr_sar_polarization: str = "auto"
    vhr_sar_resolution_m: float = 1.0
    vhr_sar_coregister: str = "auto"
    vhr_sar_coregister_base: str | None = None
    vhr_sar_bands: str = "median"
    vhr_sar_calibration: str = "any"
    vhr_sar_max_download_gb: float = 5.0
    vhr_sar_cache_dir: str | None = None

    # Embeddings ---------------------------------------------------------------
    tessera_version: str = "v1"
    tessera_variant: str | None = None
    embedding_cache_dir: str | None = None
    google_embedding_backend: str = "gee"

    # Local input --------------------------------------------------------------
    local_tif_path: str | None = None
    bands: dict[str, int] | None = None

    # Post-processing ----------------------------------------------------------
    aoi_selection: str = "representative_point"
    min_field_area_m2: float = 2500.0
    simplify_tolerance: float = 2.0

    # LULC crop filter ---------------------------------------------------------
    lulc_filter: bool = True
    lulc_crop_threshold: float = 0.3
    lulc_batch_size: int = 200
    lulc_dataset: str = "auto"
    lulc_on_error: str = "raise"
    lulc_mode: str = "server"
    lulc_nodata_policy: str = "keep"
    lulc_tree_crops: bool = False

    # SAM refinement -----------------------------------------------------------
    sam_refine: bool = False
    sam_backend: str = "sam2"
    sam_model: str | None = None
    sam_min_crop_px: int = 64
    sam_crop_padding: float = 0.15

    # Compute ------------------------------------------------------------------
    device: str = "auto"
    tile_size: int = 10_000
    n_workers: int = 4
    seed: int = 42

    # Fine-tuning / evaluation -------------------------------------------------
    reference_boundaries: str | None = None
    fine_tune: bool = False
    fine_tune_epochs: int = 20
    fine_tune_val_split: float = 0.2
    fine_tune_split: str = "block"
    fine_tune_block_size_m: float = 5000.0
    fine_tune_split_column: str | None = None

    # Caching / provenance -----------------------------------------------------
    cache_dir: str | None = None
    overwrite: bool = False
    provenance: bool = True

    # Engine pass-through ------------------------------------------------------
    engine_params: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        for name in (
            "source",
            "engine",
            "output_format",
            "export_method",
            "device",
            "composite_method",
            "s2_cloud_mask",
            "lulc_dataset",
            "lulc_on_error",
            "lulc_mode",
            "lulc_nodata_policy",
            "sam_backend",
            "fine_tune_split",
            "google_embedding_backend",
            "aoi_selection",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string, got {type(value).__name__}")
            setattr(self, name, value.lower().strip())

        if isinstance(self.date_range, list):
            self.date_range = tuple(self.date_range)
        self.landsat_pan_missions = _normalise_landsat_pan_missions(self.landsat_pan_missions)
        self.vhr_sar_collect_ids = _normalise_collect_ids(self.vhr_sar_collect_ids)
        for name in ("vhr_sar_coregister", "vhr_sar_bands", "vhr_sar_calibration"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string, got {type(value).__name__}")
            setattr(self, name, value.lower().strip())
        if not isinstance(self.vhr_sar_polarization, str):
            raise TypeError(
                "vhr_sar_polarization must be a string, got "
                f"{type(self.vhr_sar_polarization).__name__}"
            )
        polarization = self.vhr_sar_polarization.strip()
        self.vhr_sar_polarization = (
            "auto" if polarization.lower() == "auto" else polarization.upper()
        )
        for name in ("vhr_sar_coregister_base", "vhr_sar_cache_dir"):
            value = getattr(self, name)
            if isinstance(value, Path):
                setattr(self, name, str(value))
        if self.engine_params is None:
            self.engine_params = {}
        if self.study_area is None:
            self.study_area = ""
        for name in ("study_area", "output_path", "local_tif_path", "reference_boundaries"):
            value = getattr(self, name)
            if isinstance(value, Path):
                setattr(self, name, str(value))

        # Backwards compatibility: engine_params["sam_refine"] enables the stage.
        if not self.sam_refine and bool(self.engine_params.get("sam_refine", False)):
            self.sam_refine = True

        self._resolve_output_format()
        self._validate()

    def _resolve_output_format(self) -> None:
        """Reconcile *output_format* with the *output_path* extension."""
        if not self.output_path:
            self.output_path = _DEFAULT_OUTPUT_PATH
        if self.output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format {self.output_format!r}. Choose from {VALID_OUTPUT_FORMATS}"
            )
        suffix = Path(self.output_path).suffix.lower()
        implied = OUTPUT_EXTENSIONS.get(suffix)
        if implied is None:
            raise ValueError(
                f"output_path {self.output_path!r} has unsupported extension {suffix!r}. "
                f"Use one of {sorted(OUTPUT_EXTENSIONS)}."
            )
        if implied == self.output_format:
            return
        if self.output_path == _DEFAULT_OUTPUT_PATH:
            # Only the format was chosen: derive the default file name from it.
            self.output_path = f"fields{self.get_output_extension()}"
        elif self.output_format == "gpkg":
            # Format left at its default: infer it from the extension.
            logger.debug("output_format inferred from %s: %s", self.output_path, implied)
            self.output_format = implied
        else:
            raise ValueError(
                f"output_path {self.output_path!r} (format {implied!r}) conflicts with "
                f"output_format={self.output_format!r}. Change one of them."
            )

    def _validate(self) -> None:
        """Run validation checks on the configuration."""
        _choice("source", self.source, VALID_SOURCES)
        _choice("engine", self.engine, VALID_ENGINES)
        _choice("export_method", self.export_method, VALID_EXPORT_METHODS)
        _choice("device", self.device, VALID_DEVICES)
        _choice("composite_method", self.composite_method, VALID_COMPOSITE_METHODS)
        _choice("s2_cloud_mask", self.s2_cloud_mask, VALID_S2_CLOUD_MASKS)
        _choice("tessera_version", self.tessera_version, VALID_TESSERA_VERSIONS)
        _choice("lulc_dataset", self.lulc_dataset, VALID_LULC_DATASETS)
        _choice("lulc_on_error", self.lulc_on_error, VALID_LULC_ON_ERROR)
        _choice("lulc_mode", self.lulc_mode, VALID_LULC_MODES)
        _choice("lulc_nodata_policy", self.lulc_nodata_policy, VALID_LULC_NODATA_POLICIES)
        if not isinstance(self.lulc_tree_crops, bool):
            raise TypeError(f"lulc_tree_crops must be a bool, got {self.lulc_tree_crops!r}")
        _choice("sam_backend", self.sam_backend, VALID_SAM_BACKENDS)
        _choice("fine_tune_split", self.fine_tune_split, VALID_FINE_TUNE_SPLITS)
        _choice(
            "google_embedding_backend",
            self.google_embedding_backend,
            VALID_GOOGLE_EMBEDDING_BACKENDS,
        )
        _choice("aoi_selection", self.aoi_selection, VALID_AOI_SELECTIONS)

        # Engine / source compatibility ------------------------------------
        if not engine_supports_source(self.engine, self.source):
            supported = ENGINE_REGISTRY[self.engine]["supported_sources"]
            raise ValueError(
                f"Engine {self.engine!r} does not support source {self.source!r}. "
                f"Supported sources: {supported}"
            )
        if self.engine == "ensemble":
            self._validate_ensemble_members()

        # Numeric fields -----------------------------------------------------
        self.year = _as_int("year", self.year)
        self.seed = _as_int("seed", self.seed)
        if not 0 <= self.seed <= _MAX_SEED:
            raise ValueError(f"seed must be in [0, {_MAX_SEED}], got {self.seed}")
        self.cloud_cover_max = _as_number("cloud_cover_max", self.cloud_cover_max)
        if not 0 <= self.cloud_cover_max <= 100:
            raise ValueError(f"cloud_cover_max must be in [0, 100], got {self.cloud_cover_max}")
        _in_unit_interval("cloud_score_threshold", self.cloud_score_threshold)
        _in_unit_interval("lulc_crop_threshold", self.lulc_crop_threshold)
        val_split = _as_number("fine_tune_val_split", self.fine_tune_val_split)
        if not 0 < val_split < 1:
            raise ValueError(f"fine_tune_val_split must be in (0, 1), got {val_split}")
        _non_negative("min_field_area_m2", self.min_field_area_m2)
        _non_negative("simplify_tolerance", self.simplify_tolerance)
        _non_negative("sam_crop_padding", self.sam_crop_padding)
        _positive("naip_resolution_m", self.naip_resolution_m)
        _positive("fine_tune_block_size_m", self.fine_tune_block_size_m)
        self.n_workers = _as_int("n_workers", self.n_workers)
        if self.n_workers < 0:
            raise ValueError(f"n_workers must be >= 0, got {self.n_workers}")
        for name in (
            "tile_size",
            "lulc_batch_size",
            "fine_tune_epochs",
            "gee_max_requests",
            "sam_min_crop_px",
            "usgs_timeout_s",
        ):
            value = _as_int(name, getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
            setattr(self, name, value)
        self.usgs_retries = _as_int("usgs_retries", self.usgs_retries)
        if self.usgs_retries < 0:
            raise ValueError(f"usgs_retries must be >= 0, got {self.usgs_retries}")
        if self.gee_max_requests > 40:
            logger.warning(
                "gee_max_requests=%d exceeds Earth Engine's default limit of 40 concurrent "
                "requests per project; requests beyond the limit are rejected.",
                self.gee_max_requests,
            )

        # Years and dates -----------------------------------------------------
        self._validate_year()
        self._validate_date_range()
        self._validate_landsat_pan_window()
        self._validate_composite_method()
        self._validate_vhr_sar()

        # CRS -------------------------------------------------------------
        self.export_crs = _normalise_export_crs(self.export_crs)

        # Crop-type map stacks (cdl-sequence, aci-sequence) and engine crop-sequence ----
        self._validate_crop_sequence()

        # Earth Engine options -------------------------------------------------
        if self.gee_workload_tag is not None and not _WORKLOAD_TAG_RE.match(
            str(self.gee_workload_tag)
        ):
            raise ValueError(
                f"Invalid gee_workload_tag {self.gee_workload_tag!r}: use 1-63 characters, "
                "letters/digits at both ends and only letters, digits, '-', '_' or '.' inside."
            )

        # GEE imagery and crop-map sources need a project ID — auto-detect if not set
        needs_project = self.source in GEE_IMAGERY_SOURCES or self.source in CROP_SEQUENCE_SOURCES
        if needs_project and self.gee_project is None:
            self.gee_project = os.environ.get("GEE_PROJECT")
            if self.gee_project is None:
                from agribound.auth import _get_gcloud_project

                self.gee_project = _get_gcloud_project()
            if self.gee_project is None:
                from agribound.auth import project_from_credentials

                self.gee_project = project_from_credentials(self.gee_service_account_key)
            if self.gee_project is None:
                raise ValueError(
                    f"gee_project is required for source={self.source!r}. "
                    "Provide it via one of:\n"
                    "  1. gee_project parameter or --gee-project CLI flag\n"
                    "  2. GEE_PROJECT environment variable\n"
                    "  3. gcloud config: gcloud config set project YOUR_PROJECT\n"
                    "  4. the project_id of the service-account key (gee_service_account_key, "
                    "AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or GOOGLE_APPLICATION_CREDENTIALS)"
                )

        # GCS export needs a bucket
        if self.export_method == "gcs" and self.gcs_bucket is None:
            raise ValueError("gcs_bucket is required when export_method='gcs'.")

        # Local source needs a TIF path
        if self.source == "local" and self.local_tif_path is None:
            raise ValueError("local_tif_path is required when source='local'.")

        # Band override
        if self.bands is not None:
            if not isinstance(self.bands, dict):
                raise TypeError("bands must be a dict of canonical name -> 1-based index")
            for name, idx in self.bands.items():
                if _as_int(f"bands[{name!r}]", idx) < 1:
                    raise ValueError(f"bands[{name!r}] must be a 1-based index, got {idx}")

        # Fine-tuning needs reference data
        if self.fine_tune and self.reference_boundaries is None:
            raise ValueError("reference_boundaries is required when fine_tune=True.")
        if self.fine_tune and not ENGINE_REGISTRY[self.engine]["fine_tunable"]:
            tunable = [n for n, i in ENGINE_REGISTRY.items() if i["fine_tunable"]]
            raise ValueError(
                f"Engine {self.engine!r} does not support fine-tuning "
                f"(fine-tunable engines: {tunable}). {_fine_tune_hint(self.engine)}".rstrip()
            )
        if self.fine_tune_split == "column" and not self.fine_tune_split_column:
            raise ValueError("fine_tune_split_column is required when fine_tune_split='column'.")

        if not isinstance(self.engine_params, dict):
            raise TypeError("engine_params must be a dict")
        if self.engine == "sam-auto":
            self._validate_sam_auto()
        _check_dilate_interior_px(self.engine, self.engine_params)
        if "pivots" in self.engine_params:  # the opt-in centre-pivot post-process
            from agribound.postprocess.pivots import resolve_pivot_mode

            resolve_pivot_mode(self.engine_params["pivots"])

        # SPOT access warning
        if self.source in ("spot", "spot-pan"):
            warnings.warn(
                "SPOT 6/7 imagery (AIRBUS/SPOT6_7) is restricted to select GEE "
                "users and is for internal DRI use only. External users who need "
                "SPOT-based field boundaries should contact the package author "
                "(sayantan.majumdar@dri.edu) to request processing.",
                UserWarning,
                stacklevel=3,
            )

    def _validate_ensemble_members(self) -> None:
        """Check the ensemble members against the source.

        Without ``engine_params["engines"]`` the default members
        (:data:`agribound.registry.ENSEMBLE_DEFAULT_MEMBERS`, the same tuple
        as :data:`agribound.engines.ensemble.DEFAULT_MEMBERS`) are checked, so
        an unsupported default member fails here rather than after the
        composite has been built. A member's own ``engine_params`` are checked
        where the check needs no model: ``pivots`` is refused there (the
        pipeline applies it to the ensemble's output), a ``"sam-auto"``
        member's keys and values are checked as the engine's
        (:func:`agribound.engines.sam_auto.check_params`), and so is
        ``dilate_interior_px`` (:func:`_check_dilate_interior_px`).
        """
        members = self.engine_params.get("engines")
        if members is None:
            for name in ENSEMBLE_DEFAULT_MEMBERS:
                if not engine_supports_source(name, self.source):
                    alternatives = [
                        n
                        for n in ENGINE_REGISTRY
                        if n not in ("ensemble", *CATEGORICAL_ENGINES)
                        and engine_supports_source(n, self.source)
                    ]
                    raise ValueError(
                        f"Ensemble member {name!r} (a default member) does not support source "
                        f"{self.source!r} (supported: "
                        f"{ENGINE_REGISTRY[name]['supported_sources']}). The default members "
                        f"are {list(ENSEMBLE_DEFAULT_MEMBERS)}; set engine_params['engines'] to "
                        f"members that support {self.source!r}: {alternatives}."
                    )
            return
        if not isinstance(members, list | tuple) or not members:
            raise ValueError("engine_params['engines'] must be a non-empty list for ensemble")
        for spec in members:
            name = spec.get("engine") if isinstance(spec, dict) else spec
            if not isinstance(name, str):
                raise ValueError(f"Invalid ensemble member spec: {spec!r}")
            name = name.lower().strip()
            if name in CATEGORICAL_ENGINES:
                raise ValueError(
                    f"Engine {name!r} cannot be an ensemble member: it reads class codes of "
                    "crop-type maps, not imagery, and its zones are crop-history zones, not field "
                    f"instances. Run engine={name!r} on its own."
                )
            if name not in ENGINE_REGISTRY or name == "ensemble":
                raise ValueError(
                    f"Invalid ensemble member {name!r}. Choose from "
                    f"{[n for n in ENGINE_REGISTRY if n not in ('ensemble', *CATEGORICAL_ENGINES)]}"
                )
            if not engine_supports_source(name, self.source):
                raise ValueError(
                    f"Ensemble member {name!r} does not support source {self.source!r}. "
                    f"Supported sources: {ENGINE_REGISTRY[name]['supported_sources']}"
                )
            params = spec.get("engine_params") if isinstance(spec, dict) else None
            if params is not None and not isinstance(params, dict):
                continue  # malformed: the ensemble engine refuses this spec
            params = params or {}
            if "pivots" in params:
                raise ValueError(
                    f"engine_params['pivots'] of ensemble member {name!r} would not be used: "
                    "the centre-pivot post-process runs on the ensemble's output. Set it in the "
                    "ensemble's own engine_params."
                )
            if name == "sam-auto":
                from agribound.engines.sam_auto import check_params

                check_params(self.source, params)
            _check_dilate_interior_px(name, params)

    def _validate_sam_auto(self) -> None:
        """Engine ``"sam-auto"``: refuse the SAM refinement fields and check ``engine_params``.

        ``sam_refine`` is refused (refining SAM masks with SAM adds nothing), and
        so is ``sam_model``, the refinement stage's model (the engine's model is
        ``engine_params["sam_model"]``). ``engine_params`` is checked by
        :func:`agribound.engines.sam_auto.check_params`: unknown keys, invalid
        values, and ``source="sentinel1"`` (no default setting) without every
        per-source setting. Categorical and embedding sources are refused
        before, by the engine's ``supported_sources``.
        """
        if self.sam_refine:
            raise ValueError(
                "sam_refine does not apply to engine 'sam-auto': its polygons are already SAM "
                "masks."
            )
        if self.sam_model is not None:
            raise ValueError(
                "sam_model selects the model of the SAM refinement stage, which engine "
                "'sam-auto' does not use; set engine_params['sam_model'] instead."
            )
        from agribound.engines.sam_auto import check_params

        check_params(self.source, self.engine_params)

    def _validate_year(self) -> None:
        year_range = source_year_range(
            self.source,
            tessera_version=self.tessera_version if self.source == "tessera-embedding" else None,
        )
        if year_range is None:
            return
        first, last = year_range
        upper = last if last is not None else _dt.datetime.now(_dt.UTC).year
        if not first <= self.year <= upper:
            label = f"{first}-{last}" if last is not None else f"{first}-present"
            extra = (
                f" for tessera_version={self.tessera_version!r}"
                if self.source == "tessera-embedding"
                else ""
            )
            raise ValueError(
                f"year={self.year} is outside the available range for source "
                f"{self.source!r}{extra}: {label}."
            )

    def _validate_date_range(self) -> None:
        if self.date_range is None:
            return
        if not isinstance(self.date_range, tuple) or len(self.date_range) != 2:
            raise ValueError("date_range must be a (start, end) pair of 'YYYY-MM-DD' strings")
        parsed = []
        for value in self.date_range:
            text = value.isoformat() if isinstance(value, _dt.date) else str(value)
            if not _DATE_RE.match(text):
                raise ValueError(f"date_range entries must be 'YYYY-MM-DD', got {value!r}")
            try:
                parsed.append(_dt.date.fromisoformat(text))
            except ValueError as exc:
                raise ValueError(f"Invalid date in date_range: {value!r}") from exc
        if parsed[0] > parsed[1]:
            raise ValueError(f"date_range start {parsed[0]} is after end {parsed[1]}")
        self.date_range = (parsed[0].isoformat(), parsed[1].isoformat())

    def _validate_landsat_pan_window(self) -> None:
        """For ``landsat-pan``, require a selected mission whose record overlaps the window.

        Uses the composite builder's own rule
        (:func:`agribound.composites.gee.landsat_pan_missions_for_window` on
        :func:`~agribound.composites.gee.date_window`), so a mission list that
        cannot match the year or ``date_range`` fails here instead of at the
        composite stage. Other sources ignore ``landsat_pan_missions``.
        """
        if self.source != "landsat-pan":
            return
        from agribound.composites.gee import date_window, landsat_pan_missions_for_window

        window = date_window(self)
        setting = self.landsat_pan_missions
        if landsat_pan_missions_for_window(setting, window):
            return
        text = setting if isinstance(setting, str) else ",".join(setting)
        if self.date_range is not None:
            period = f"date_range {self.date_range[0]} to {self.date_range[1]}"
        else:
            period = f"year={self.year}"
        hint = (
            "Landsat PAN images start on 1999-05-28."
            if setting == "auto"
            else "Use 'auto' or missions whose record overlaps the window."
        )
        raise ValueError(
            f"landsat_pan_missions={text!r} has no mission whose record overlaps {period} "
            "(Landsat 7 1999-05-28 to 2024-01-19, Landsat 8 from 2013-03-18, Landsat 9 from "
            f"2021-10-31). {hint}"
        )

    def _validate_composite_method(self) -> None:
        """Reject ``greenest``/``max_ndvi`` for Earth Engine sources without NIR and red bands.

        These methods keep, per pixel, the observation with the highest NDVI
        (:func:`agribound.composites.gee.apply_composite_method`), which
        ``landsat-pan``, ``spot-pan`` and ``sentinel1`` cannot compute. The
        check uses the composite builder's own rule, so such a configuration
        fails here instead of at the composite stage. ``umbra`` and ``capella``
        accept only ``"median"`` (:meth:`_validate_vhr_sar`); the other sources
        whose rasters are not composited on Earth Engine ignore
        ``composite_method``.
        """
        if self.composite_method not in ("greenest", "max_ndvi"):
            return
        if self.source not in GEE_IMAGERY_SOURCES:
            return
        from agribound.composites.gee import _ndvi_bands

        _ndvi_bands(self.source, self.composite_method)

    def _validate_vhr_sar(self) -> None:
        """Check the ``vhr_sar_*`` fields (they apply only to ``umbra`` and ``capella``).

        The fields are checked for every source, so a configuration does not
        carry an invalid value that only fails once the source is changed.
        ``vhr_sar_bands="median+dates"`` needs at least 2 collects; it does not
        change *bands* (the engines derive the band mapping with
        :func:`effective_bands`, so switching back to ``"median"`` leaves no
        stale mapping).
        """
        _choice("vhr_sar_polarization", self.vhr_sar_polarization, VALID_VHR_SAR_POLARIZATIONS)
        _choice("vhr_sar_coregister", self.vhr_sar_coregister, VALID_VHR_SAR_COREGISTER)
        _choice("vhr_sar_bands", self.vhr_sar_bands, VALID_VHR_SAR_BANDS)
        _choice("vhr_sar_calibration", self.vhr_sar_calibration, VALID_VHR_SAR_CALIBRATION)
        self.vhr_sar_max_collects = _as_int("vhr_sar_max_collects", self.vhr_sar_max_collects)
        if not 1 <= self.vhr_sar_max_collects <= 50:
            raise ValueError(
                f"vhr_sar_max_collects must be in [1, 50], got {self.vhr_sar_max_collects}"
            )
        _positive("vhr_sar_resolution_m", self.vhr_sar_resolution_m)
        if self.vhr_sar_resolution_m > 10:
            raise ValueError(
                "vhr_sar_resolution_m must be at most 10 m (the products have 0.16-1.2 m "
                f"resolution), got {self.vhr_sar_resolution_m}"
            )
        _positive("vhr_sar_max_download_gb", self.vhr_sar_max_download_gb)
        if self.vhr_sar_coregister_base is not None:
            if not isinstance(self.vhr_sar_coregister_base, str):
                raise TypeError("vhr_sar_coregister_base must be a path string or None")
            if self.vhr_sar_coregister in ("sar", "none"):
                raise ValueError(
                    "vhr_sar_coregister_base is set but vhr_sar_coregister="
                    f"{self.vhr_sar_coregister!r} does not use an optical base; use 'auto' or "
                    "'optical', or remove the base."
                )
        if self.vhr_sar_cache_dir is not None and not isinstance(self.vhr_sar_cache_dir, str):
            raise TypeError("vhr_sar_cache_dir must be a path string or None")
        if self.source not in VHR_SAR_SOURCES:
            return
        if self.composite_method != "median":
            raise ValueError(
                f"composite_method={self.composite_method!r} does not apply to source "
                f"{self.source!r}: its composite is the median of co-registered, "
                "level-normalised SAR amplitudes. Use 'median'."
            )
        if self.source == "umbra" and self.vhr_sar_calibration != "any":
            raise ValueError(
                f"vhr_sar_calibration={self.vhr_sar_calibration!r} applies to source='capella' "
                "only: Umbra GEC products are not calibrated. Use 'any'."
            )
        if self.vhr_sar_bands == "median+dates":
            n_pinned = len(self.vhr_sar_collect_ids or ())
            if n_pinned == 1 or (n_pinned == 0 and self.vhr_sar_max_collects == 1):
                raise ValueError(
                    "vhr_sar_bands='median+dates' needs at least 2 collects (Delineate-Anything "
                    "reads the mean of a pair of dates as G); use vhr_sar_bands='median' for a "
                    "single-date composite."
                )

    def _validate_crop_sequence(self) -> None:
        """Check ``crop_sequence_years`` and the settings of the crop-sequence route.

        ``crop_sequence_years`` must be an integer in [2, 20] for every source
        (a configuration does not carry a value that only fails once the
        source is changed). For ``cdl-sequence`` and ``aci-sequence``: the
        window ``year - crop_sequence_years + 1 .. year`` must start at or after
        the source's first year; ``date_range`` and ``bands`` must be unset,
        ``composite_method`` must be ``"median"`` (the maps are not
        composited), ``export_method`` must be ``"local"`` (no batch exports),
        ``sam_refine`` must be off (the stack holds no imagery), and
        ``export_crs`` must be a projected CRS in metres
        (:func:`crop_sequence_crs_problem`: the stack's 30 m pixel size is set
        in CRS units, and the engine's focal window and 1 ha merge are in
        pixels and square metres). For engine
        ``crop-sequence``: ``sam_refine`` must be off and, with
        ``source="local"``, ``engine_params`` must declare
        ``value_scale="categorical"`` and either ``dataset`` (``"cdl"`` or
        ``"aci"``) or both ``classes`` and ``barrier_classes``.
        """
        self.crop_sequence_years = _as_int("crop_sequence_years", self.crop_sequence_years)
        n = self.crop_sequence_years
        if not 2 <= n <= 20:
            raise ValueError(f"crop_sequence_years must be in [2, 20], got {n}")
        if self.source in CROP_SEQUENCE_SOURCES:
            first, _ = source_year_range(self.source)
            start = self.year - n + 1
            if start < first:
                raise ValueError(
                    f"year={self.year} with crop_sequence_years={n} needs the maps of "
                    f"{start}-{self.year}, but source {self.source!r} starts in {first}. Use "
                    f"year >= {first + n - 1} or a smaller crop_sequence_years."
                )
            if self.date_range is not None:
                raise ValueError(
                    f"date_range does not apply to source {self.source!r}: the maps are annual. "
                    "Set year (the last map year) and crop_sequence_years instead."
                )
            if self.composite_method != "median":
                raise ValueError(
                    f"composite_method={self.composite_method!r} does not apply to source "
                    f"{self.source!r}: the annual maps are stacked, not composited. Use the "
                    "default 'median' (ignored)."
                )
            if self.export_method != "local":
                raise ValueError(
                    f"export_method={self.export_method!r} is not supported for source "
                    f"{self.source!r}: the stack (one uint8 band per year) is downloaded "
                    "directly. Use export_method='local'."
                )
            if self.sam_refine:
                raise ValueError(
                    f"sam_refine does not apply to source {self.source!r}: the stack holds class "
                    "codes, not imagery."
                )
            if self.bands is not None:
                raise ValueError(
                    f"bands does not apply to source {self.source!r}: the stack has one band of "
                    "class codes per year."
                )
            problem = crop_sequence_crs_problem(self.export_crs, self.source)
            if problem is not None:
                raise ValueError(problem)
        if self.engine in CATEGORICAL_ENGINES:
            if self.sam_refine:
                raise ValueError(
                    f"sam_refine does not apply to engine {self.engine!r}: its input holds class "
                    "codes, not imagery, and its zones are not refined."
                )
            if self.source == "local":
                self._validate_local_categorical()

    def _validate_local_categorical(self) -> None:
        """Engine ``crop-sequence`` on a local raster: the stack must be declared and described."""
        params = self.engine_params or {}
        scale = params.get("value_scale")
        if not isinstance(scale, str) or scale.strip().lower() != "categorical":
            raise ValueError(
                f"engine={self.engine!r} with source='local' needs "
                "engine_params['value_scale']='categorical': the local GeoTIFF must be a stack "
                "of annual crop-type maps (one band of class codes per year)."
            )
        dataset = params.get("dataset")
        if dataset is not None:
            from agribound.composites.crop_sequence import DATASET_SOURCES

            if not isinstance(dataset, str) or dataset.strip().lower() not in DATASET_SOURCES:
                raise ValueError(
                    f"engine_params['dataset'] must be one of {tuple(DATASET_SOURCES)} (or "
                    f"None with explicit classes and barrier_classes), got {dataset!r}"
                )
            return
        missing = [
            key
            for key in ("classes", "barrier_classes")
            if params.get(key) is None or params.get(key) == "auto"
        ]
        if missing:
            raise ValueError(
                f"engine={self.engine!r} with source='local' needs engine_params['dataset'] "
                "('cdl' or 'aci') for the default class lists, or explicit lists of "
                f"{' and '.join(repr(k) for k in missing)} (class codes)."
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the names of all configuration fields."""
        return tuple(f.name for f in fields(cls))

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a plain dictionary.

        Tuples (``date_range``) are returned as lists so the result can be
        written to YAML/JSON; :meth:`from_dict` converts them back.
        """
        data = {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}
        if data.get("date_range") is not None:
            data["date_range"] = list(data["date_range"])
        if isinstance(data.get("landsat_pan_missions"), tuple):
            data["landsat_pan_missions"] = list(data["landsat_pan_missions"])
        if isinstance(data.get("vhr_sar_collect_ids"), tuple):
            data["vhr_sar_collect_ids"] = list(data["vhr_sar_collect_ids"])
        return data

    def to_yaml(self, path: str | Path) -> None:
        """Write the configuration to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml_str())

    def to_yaml_str(self) -> str:
        """Return the configuration as a YAML string (tuples written as lists)."""
        return yaml.safe_dump(
            _plain(self.to_dict()), default_flow_style=False, sort_keys=False, allow_unicode=True
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgriboundConfig:
        """Load a configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        AgriboundConfig
            Loaded configuration instance.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file is not a mapping or contains unknown keys.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(f"Configuration file {path} must contain a mapping of keys to values")
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgriboundConfig:
        """Create a configuration from a dictionary.

        The input dictionary is not modified.

        Parameters
        ----------
        data : dict
            Configuration dictionary.

        Returns
        -------
        AgriboundConfig
            Configuration instance.

        Raises
        ------
        ValueError
            If *data* contains unknown keys.
        """
        known = set(cls.field_names())
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown configuration key(s): {unknown}. Valid keys: {sorted(known)}"
            )
        data = copy.deepcopy(dict(data))
        if isinstance(data.get("date_range"), list):
            data["date_range"] = tuple(data["date_range"])
        return cls(**data)

    def merged(self, **overrides: Any) -> AgriboundConfig:
        """Return a validated copy of this configuration with *overrides* applied.

        Parameters
        ----------
        **overrides
            Field values to replace.

        Returns
        -------
        AgriboundConfig
            New configuration (this instance is not modified).

        Raises
        ------
        ValueError
            If an override key is unknown or the result is invalid.
        """
        unknown = sorted(set(overrides) - set(self.field_names()))
        if unknown:
            raise ValueError(f"Unknown configuration key(s): {unknown}")
        data = self.to_dict()
        data.update(copy.deepcopy(overrides))
        if "sam_refine" in overrides and "sam_refine" in data.get("engine_params", {}):
            data["engine_params"]["sam_refine"] = bool(overrides["sam_refine"])
        return type(self).from_dict(data)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def resolve_device(self) -> str:
        """Resolve ``"auto"`` to an actual device string.

        Returns
        -------
        str
            One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def is_gee_source(self) -> bool:
        """Return *True* if the imagery composite is built on Google Earth Engine.

        Covers ``landsat``, ``landsat-pan``, ``sentinel2``, ``hls``, ``naip``,
        ``spot``, ``spot-pan`` and ``sentinel1``. Embedding sources are not
        included (see :meth:`requires_gee`).
        """
        return self.source in GEE_IMAGERY_SOURCES

    def is_embedding_source(self) -> bool:
        """Return *True* if the source is a pre-computed embedding dataset."""
        return self.source in {"google-embedding", "tessera-embedding"}

    def requires_gee(self) -> bool:
        """Return *True* if any stage of this run needs Earth Engine.

        True for GEE imagery sources, for the crop-type map stacks
        (``cdl-sequence``, ``aci-sequence``), for ``google-embedding`` with the
        ``"gee"`` backend, and whenever the LULC filter is enabled.
        """
        if self.is_gee_source() or self.source in CROP_SEQUENCE_SOURCES or self.lulc_filter:
            return True
        return self.source == "google-embedding" and self.google_embedding_backend == "gee"

    def get_output_extension(self) -> str:
        """Return the file extension for the configured output format."""
        ext_map = {"gpkg": ".gpkg", "geojson": ".geojson", "parquet": ".parquet"}
        return ext_map[self.output_format]

    def get_working_dir(self) -> Path:
        """Return (and create) the directory for intermediate files.

        Uses *cache_dir* when set, otherwise ``<output dir>/.agribound_cache``.
        Cache file names are keyed by :func:`agribound._cache.cache_key`, so
        several runs can share one directory.
        """
        if self.cache_dir:
            cache_dir = Path(self.cache_dir).expanduser()
        else:
            cache_dir = Path(self.output_path).parent / ".agribound_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def effective_bands(config: Any) -> dict[str, int] | None:
    """The band mapping the engines read for *config*.

    ``config.bands`` when given; else, for ``umbra`` and ``capella`` with
    ``vhr_sar_bands="median+dates"``, :data:`VHR_SAR_MEDIAN_DATES_BANDS` (the
    median in R and B, the pair mean in G); else None (the source registry's
    mapping). The mapping is derived when the composite is read, not stored in
    the configuration, so it follows ``vhr_sar_bands``.
    """
    bands = getattr(config, "bands", None)
    if bands is not None:
        return bands
    if (
        getattr(config, "source", None) in VHR_SAR_SOURCES
        and getattr(config, "vhr_sar_bands", None) == "median+dates"
    ):
        return dict(VHR_SAR_MEDIAN_DATES_BANDS)
    return None


def crop_sequence_crs_problem(export_crs: str, source: str) -> str | None:
    """Why *export_crs* cannot hold the stack of a crop-type map *source*, or None.

    The stack is built at 30 CRS units per pixel
    (:func:`agribound.composites.gee.compute_export_grid`) and the
    crop-sequence engine's focal window is in pixels, so the CRS must be
    projected with axes in metres: ``"utm"`` always is; a geographic CRS, or a
    projected CRS in feet (whose pixels would be 30 ft, 9.144 m, and whose
    3 px focal radius would be 27 m instead of 90 m), is refused.

    Parameters
    ----------
    export_crs : str
        ``"utm"`` or ``"EPSG:<code>"`` (``AgriboundConfig.export_crs``).
    source : str
        Source name, for the message.

    Returns
    -------
    str or None
        The reason, or *None* when the CRS is usable.
    """
    if str(export_crs).strip().lower() == "utm":
        return None
    import pyproj

    crs = pyproj.CRS.from_user_input(export_crs)
    if crs.is_geographic:
        return (
            f"export_crs={export_crs!r} is geographic; source {source!r} needs a projected CRS "
            "in metres (the crop-sequence engine works in pixels and square metres). Use 'utm' "
            "(default) or a projected EPSG code in metres."
        )
    axis = crs.axis_info[0] if crs.axis_info else None
    factor = getattr(axis, "unit_conversion_factor", None)
    if not crs.is_projected or factor is None or not math.isclose(float(factor), 1.0):
        unit = getattr(axis, "unit_name", None) or "unknown"
        return (
            f"export_crs={export_crs!r} has axes in {unit}; source {source!r} needs a projected "
            "CRS in metres: the stack's pixel size of 30 is set in CRS units, so its pixels "
            "would not be 30 m, and the crop-sequence engine's focal window is in pixels. Use "
            "'utm' (default) or a projected EPSG code in metres."
        )
    return None


def _check_dilate_interior_px(engine: str, engine_params: dict | None) -> None:
    """Check ``engine_params["dilate_interior_px"]`` of *engine* where it has a rule.

    ``"dinov3"``: a number of pixels, a string of an integer or ``"half_band"``
    (:func:`agribound.engines.dinov3.resolve_dilate_rule`). ``"prithvi"``:
    ``"half_band"`` is refused, since that growth rule is DINOv3's. Other
    engines do not read the key.

    Raises
    ------
    ValueError
        For another string for DINOv3, or ``"half_band"`` for Prithvi.
    """
    value = (engine_params or {}).get("dilate_interior_px")
    if value is None:
        return
    if engine == "dinov3":
        from agribound.engines.dinov3 import resolve_dilate_rule

        resolve_dilate_rule(value)
    elif engine == "prithvi" and isinstance(value, str) and value.strip().lower() == "half_band":
        raise ValueError(
            "engine_params['dilate_interior_px']='half_band' is DINOv3's growth rule (measured "
            "on DINOv3's predicted boundary band and piloted only on DINOv3); engine 'prithvi' "
            "takes a number of pixels."
        )


def _fine_tune_hint(engine: str) -> str:
    """Return the fine-tuning dispatcher's instructions for *engine*, if any."""
    try:
        from agribound.engines.finetune import _NOT_FINE_TUNABLE_HINTS
    except Exception:  # pragma: no cover - optional hint only
        return ""
    return _NOT_FINE_TUNABLE_HINTS.get(engine, "")


def _plain(value: Any) -> Any:
    """Recursively convert tuples/sets/Paths to YAML-safe builtins."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(_plain(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    return value


def _choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"Invalid {name} {value!r}. Choose from {choices}")


def _normalise_landsat_pan_missions(value: Any) -> str | tuple[str, ...]:
    """``"auto"``, or the distinct missions of *value* in canonical order.

    Accepts ``"auto"`` (any case), a comma-separated string such as
    ``"LC08,LC09"`` or a list/tuple of mission IDs (any case).
    """
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "auto":
            return "auto"
        items: list[Any] = [part for part in text.split(",") if part.strip()]
    elif isinstance(value, list | tuple):
        items = list(value)
    else:
        raise TypeError(
            "landsat_pan_missions must be 'auto' or a list of mission IDs, got "
            f"{type(value).__name__}"
        )
    missions = set()
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"landsat_pan_missions entries must be strings, got {item!r}")
        missions.add(item.strip().upper())
    unknown = sorted(missions - set(VALID_LANDSAT_PAN_MISSIONS))
    if unknown:
        raise ValueError(
            f"Invalid landsat_pan_missions {unknown}. Choose 'auto' or from "
            f"{VALID_LANDSAT_PAN_MISSIONS}"
        )
    if not missions:
        raise ValueError(
            "landsat_pan_missions is empty. Use 'auto' or name at least one of "
            f"{VALID_LANDSAT_PAN_MISSIONS}"
        )
    return tuple(m for m in VALID_LANDSAT_PAN_MISSIONS if m in missions)


def _normalise_collect_ids(value: Any) -> tuple[str, ...] | None:
    """*None*, or the distinct collect IDs of *value* in the given order.

    Accepts a comma-separated string or a list/tuple of strings (surrounding
    whitespace is removed; IDs are case-sensitive).
    """
    if value is None:
        return None
    if isinstance(value, str):
        items: list[Any] = [part for part in value.split(",") if part.strip()]
    elif isinstance(value, list | tuple):
        items = list(value)
    else:
        raise TypeError(
            "vhr_sar_collect_ids must be None, a list of collect IDs or a comma-separated "
            f"string, got {type(value).__name__}"
        )
    ids: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise TypeError(f"vhr_sar_collect_ids entries must be non-empty strings, got {item!r}")
        text = item.strip()
        if text not in ids:
            ids.append(text)
    if not ids:
        raise ValueError("vhr_sar_collect_ids is empty; use None for the automatic selection")
    return tuple(ids)


def _as_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got a bool")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    raise TypeError(f"{name} must be an integer, got {value!r}")


def _as_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {value!r}")
    return value


def _in_unit_interval(name: str, value: Any) -> None:
    value = _as_number(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _non_negative(name: str, value: Any) -> None:
    number = _as_number(name, value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value}")


def _positive(name: str, value: Any) -> None:
    number = _as_number(name, value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a finite number > 0, got {value}")


def _normalise_export_crs(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"export_crs must be a string, got {value!r}")
    text = value.strip()
    if text.lower() == "utm":
        return "utm"
    match = _EPSG_RE.match(text)
    if match is None:
        raise ValueError(f"Invalid export_crs {value!r}: use 'utm' or 'EPSG:<code>'")
    code = int(match.group(1))
    import pyproj

    try:
        crs = pyproj.CRS.from_epsg(code)
    except pyproj.exceptions.CRSError as exc:
        raise ValueError(f"Invalid export_crs {value!r}: unknown EPSG code") from exc
    if crs.is_geographic:
        logger.warning(
            "export_crs=%s is geographic: GEE converts metre scales to degrees, which gives "
            "non-square pixels away from the equator. 'utm' is recommended.",
            text,
        )
    return f"EPSG:{code}"
