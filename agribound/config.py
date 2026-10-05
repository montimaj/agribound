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
    ENGINE_REGISTRY,
    ENSEMBLE_DEFAULT_MEMBERS,
    GEE_IMAGERY_SOURCES,
    SAM_REFINE_BACKENDS,
    SOURCE_REGISTRY,
    TESSERA_YEAR_RANGES,
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
VALID_AOI_SELECTIONS = ("representative_point", "intersects", "clip", "none")

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
        ``"local"``, ``"google-embedding"`` or ``"tessera-embedding"``
        (see :func:`agribound.list_sources`).
    engine : str
        Delineation engine: ``"delineate-anything"``, ``"ftw"``, ``"geoai"``,
        ``"dinov3"``, ``"prithvi"``, ``"embedding"`` or ``"ensemble"`` (see
        :func:`agribound.list_engines`). The engine must support the source.
    year : int
        Target year. Must lie within the source's available years
        (:func:`agribound.registry.source_year_range`).
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
    date_range : tuple[str, str] or None
        Override the default full-year window with an explicit
        ``("YYYY-MM-DD", "YYYY-MM-DD")`` window (e.g. a growing season).
    cloud_cover_max : int
        Maximum scene cloud cover percentage, 0-100 (default 20).
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
    sam_refine : bool
        Refine polygons with box-prompted SAM after delineation (default
        *False*). ``engine_params["sam_refine"]`` is still honoured.
    sam_backend : str
        ``"sam2"`` (default), ``"sam2.1"``, ``"sam3"`` or ``"sam3-hf"``. The two SAM 3
        backends are untested in 1.0.1: they have not been run end to end (the
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
    usgs_retries: int = 3

    # Compositing --------------------------------------------------------------
    composite_method: str = "median"
    date_range: tuple[str, str] | None = None
    cloud_cover_max: int = 20
    export_crs: str = "utm"
    s2_cloud_mask: str = "scl"
    cloud_score_threshold: float = 0.60
    naip_resolution_m: float = 1.0

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

        # CRS -------------------------------------------------------------
        self.export_crs = _normalise_export_crs(self.export_crs)

        # Earth Engine options -------------------------------------------------
        if self.gee_workload_tag is not None and not _WORKLOAD_TAG_RE.match(
            str(self.gee_workload_tag)
        ):
            raise ValueError(
                f"Invalid gee_workload_tag {self.gee_workload_tag!r}: use 1-63 characters, "
                "letters/digits at both ends and only letters, digits, '-', '_' or '.' inside."
            )

        # GEE imagery sources need a project ID — auto-detect if not set
        if self.source in GEE_IMAGERY_SOURCES and self.gee_project is None:
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
        composite has been built.
        """
        members = self.engine_params.get("engines")
        if members is None:
            for name in ENSEMBLE_DEFAULT_MEMBERS:
                if not engine_supports_source(name, self.source):
                    alternatives = [
                        n
                        for n in ENGINE_REGISTRY
                        if n != "ensemble" and engine_supports_source(n, self.source)
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
            if name not in ENGINE_REGISTRY or name == "ensemble":
                raise ValueError(
                    f"Invalid ensemble member {name!r}. Choose from "
                    f"{[n for n in ENGINE_REGISTRY if n != 'ensemble']}"
                )
            if not engine_supports_source(name, self.source):
                raise ValueError(
                    f"Ensemble member {name!r} does not support source {self.source!r}. "
                    f"Supported sources: {ENGINE_REGISTRY[name]['supported_sources']}"
                )

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

        Covers ``landsat``, ``sentinel2``, ``hls``, ``naip``, ``spot`` and
        ``spot-pan``. Embedding sources are not included (see
        :meth:`requires_gee`).
        """
        return self.source in GEE_IMAGERY_SOURCES

    def is_embedding_source(self) -> bool:
        """Return *True* if the source is a pre-computed embedding dataset."""
        return self.source in {"google-embedding", "tessera-embedding"}

    def requires_gee(self) -> bool:
        """Return *True* if any stage of this run needs Earth Engine.

        True for GEE imagery sources, for ``google-embedding`` with the
        ``"gee"`` backend, and whenever the LULC filter is enabled.
        """
        if self.is_gee_source() or self.lulc_filter:
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
