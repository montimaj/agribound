"""
Configuration management for Agribound.

Provides the ``AgriboundConfig`` dataclass for controlling every aspect of the
delineation pipeline — satellite source, delineation engine, GEE export
settings, post-processing parameters, and optional fine-tuning.

Configurations can be created programmatically, loaded from YAML files, or
passed via CLI flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Valid choices
# ---------------------------------------------------------------------------

VALID_SOURCES = (
    "landsat",
    "sentinel2",
    "hls",
    "naip",
    "spot",
    "local",
    "google-embedding",
    "tessera-embedding",
)

VALID_ENGINES = (
    "delineate-anything",
    "ftw",
    "geoai",
    "prithvi",
    "embedding",
    "ensemble",
)

VALID_EXPORT_METHODS = ("local", "gdrive", "gcs")
VALID_OUTPUT_FORMATS = ("gpkg", "geojson", "parquet")
VALID_COMPOSITE_METHODS = ("median", "greenest", "max_ndvi")
VALID_DEVICES = ("auto", "cuda", "cpu", "mps")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgriboundConfig:
    """Configuration for an Agribound delineation run.

    Parameters
    ----------
    source : str
        Satellite source. One of ``"landsat"``, ``"sentinel2"``, ``"hls"``,
        ``"naip"``, ``"spot"``, ``"local"``, ``"google-embedding"``,
        ``"tessera-embedding"``.
    engine : str
        Delineation engine. One of ``"delineate-anything"``, ``"ftw"``,
        ``"geoai"``, ``"prithvi"``, ``"embedding"``, ``"ensemble"``.
    year : int
        Target year for the annual composite.
    study_area : str
        Path to a GeoJSON / Shapefile / GeoParquet **or** a GEE vector asset
        ID (e.g. ``"projects/my-project/assets/my_aoi"``).
    output_path : str
        Destination file for the output field boundary vectors.
    output_format : str
        Output vector format: ``"gpkg"`` | ``"geojson"`` | ``"parquet"``
        (fiboa-compliant GeoParquet).
    gee_project : str or None
        Google Earth Engine project ID. Required when *source* is a GEE-based
        satellite.
    export_method : str
        GEE export method: ``"local"`` (direct download, default) |
        ``"gdrive"`` | ``"gcs"``.
    gcs_bucket : str or None
        Google Cloud Storage bucket name. Required when *export_method* is
        ``"gcs"``.
    composite_method : str
        Compositing strategy: ``"median"`` | ``"greenest"`` | ``"max_ndvi"``.
    date_range : tuple[str, str] or None
        Override the default full-year date range with an explicit
        ``("YYYY-MM-DD", "YYYY-MM-DD")`` window (e.g. a growing season).
    cloud_cover_max : int
        Maximum cloud cover percentage for scene filtering (default 20).
    local_tif_path : str or None
        Path to a local GeoTIFF when *source* is ``"local"``.
    bands : dict or None
        Band mapping override, e.g. ``{"R": 1, "G": 2, "B": 3, "NIR": 4}``.
    min_field_area_m2 : float
        Minimum field polygon area in m** (default 2500).
    simplify_tolerance : float
        Ramer-Douglas-Peucker simplification tolerance in pixels (default 2.0).
    device : str
        Compute device: ``"auto"`` | ``"cuda"`` | ``"cpu"`` | ``"mps"``.
    tile_size : int
        Max tile dimension (pixels) for auto-chunking large composites
        (default 10 000).
    n_workers : int
        Number of parallel workers for dask tiling / download (default 4).
    reference_boundaries : str or None
        Path to existing field boundaries (``.shp``, ``.gpkg``, ``.geojson``,
        ``.parquet``) for fine-tuning or evaluation.
    fine_tune : bool
        When *True* and *reference_boundaries* is provided, fine-tune the
        engine on the reference data before inference.
    fine_tune_epochs : int
        Number of epochs for fine-tuning (default 20).
    fine_tune_val_split : float
        Fraction of reference data reserved for validation (default 0.2).
    engine_params : dict
        Arbitrary keyword arguments forwarded to the selected engine.
    """

    # Required -----------------------------------------------------------------
    source: str = "sentinel2"
    engine: str = "delineate-anything"
    year: int = 2024
    study_area: str = ""
    output_path: str = "fields.gpkg"

    # Output -------------------------------------------------------------------
    output_format: str = "gpkg"

    # GEE ----------------------------------------------------------------------
    gee_project: str | None = None
    export_method: str = "local"
    gcs_bucket: str | None = None

    # Compositing --------------------------------------------------------------
    composite_method: str = "median"
    date_range: tuple[str, str] | None = None
    cloud_cover_max: int = 20

    # Local input --------------------------------------------------------------
    local_tif_path: str | None = None
    bands: dict[str, int] | None = None

    # Post-processing ----------------------------------------------------------
    min_field_area_m2: float = 2500.0
    simplify_tolerance: float = 2.0

    # Compute ------------------------------------------------------------------
    device: str = "auto"
    tile_size: int = 10_000
    n_workers: int = 4

    # Fine-tuning / evaluation -------------------------------------------------
    reference_boundaries: str | None = None
    fine_tune: bool = False
    fine_tune_epochs: int = 20
    fine_tune_val_split: float = 0.2

    # Engine pass-through ------------------------------------------------------
    engine_params: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        self.source = self.source.lower().strip()
        self.engine = self.engine.lower().strip()
        self.output_format = self.output_format.lower().strip()
        self.export_method = self.export_method.lower().strip()
        self.device = self.device.lower().strip()
        self.composite_method = self.composite_method.lower().strip()

        self._validate()

    def _validate(self) -> None:
        """Run validation checks on the configuration."""
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"Invalid source {self.source!r}. Choose from {VALID_SOURCES}"
            )
        if self.engine not in VALID_ENGINES:
            raise ValueError(
                f"Invalid engine {self.engine!r}. Choose from {VALID_ENGINES}"
            )
        if self.output_format not in VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output_format {self.output_format!r}. "
                f"Choose from {VALID_OUTPUT_FORMATS}"
            )
        if self.export_method not in VALID_EXPORT_METHODS:
            raise ValueError(
                f"Invalid export_method {self.export_method!r}. "
                f"Choose from {VALID_EXPORT_METHODS}"
            )
        if self.device not in VALID_DEVICES:
            raise ValueError(
                f"Invalid device {self.device!r}. Choose from {VALID_DEVICES}"
            )
        if self.composite_method not in VALID_COMPOSITE_METHODS:
            raise ValueError(
                f"Invalid composite_method {self.composite_method!r}. "
                f"Choose from {VALID_COMPOSITE_METHODS}"
            )

        # GEE-based sources need a project ID
        gee_sources = {"landsat", "sentinel2", "hls", "naip", "spot"}
        if self.source in gee_sources and self.gee_project is None:
            raise ValueError(
                f"gee_project is required for source={self.source!r}. "
                "Set it to your GEE cloud project ID."
            )

        # GCS export needs a bucket
        if self.export_method == "gcs" and self.gcs_bucket is None:
            raise ValueError("gcs_bucket is required when export_method='gcs'.")

        # Local source needs a TIF path
        if self.source == "local" and self.local_tif_path is None:
            raise ValueError(
                "local_tif_path is required when source='local'."
            )

        # Fine-tuning needs reference data
        if self.fine_tune and self.reference_boundaries is None:
            raise ValueError(
                "reference_boundaries is required when fine_tune=True."
            )

        # SPOT access warning
        if self.source == "spot":
            import warnings

            warnings.warn(
                "SPOT 6/7 imagery (AIRBUS/SPOT6_7) is restricted to select GEE "
                "users and is for internal DRI use only. External users who need "
                "SPOT-based field boundaries should contact the package author "
                "(sayantan.majumdar@dri.edu) to request processing.",
                UserWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a plain dictionary."""
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write configuration to a YAML file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        # Convert tuple to list for safe YAML serialization
        if data.get("date_range") is not None:
            data["date_range"] = list(data["date_range"])
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgriboundConfig":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        AgriboundConfig
            Loaded configuration instance.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        # Convert date_range list back to tuple
        if "date_range" in data and isinstance(data["date_range"], list):
            data["date_range"] = tuple(data["date_range"])
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgriboundConfig":
        """Create configuration from a dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary.

        Returns
        -------
        AgriboundConfig
            Configuration instance.
        """
        if "date_range" in data and isinstance(data["date_range"], list):
            data["date_range"] = tuple(data["date_range"])
        return cls(**data)

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
        """Return *True* if the source requires Google Earth Engine."""
        return self.source in {"landsat", "sentinel2", "hls", "naip", "spot"}

    def is_embedding_source(self) -> bool:
        """Return *True* if the source is a pre-computed embedding dataset."""
        return self.source in {"google-embedding", "tessera-embedding"}

    def get_output_extension(self) -> str:
        """Return the file extension for the configured output format."""
        ext_map = {"gpkg": ".gpkg", "geojson": ".geojson", "parquet": ".parquet"}
        return ext_map[self.output_format]

    def get_working_dir(self) -> Path:
        """Return a working directory for intermediate files.

        Creates a ``.agribound_cache`` directory next to the output path.
        """
        output_dir = Path(self.output_path).parent
        cache_dir = output_dir / ".agribound_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
