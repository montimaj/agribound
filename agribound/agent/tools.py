"""
Provider-neutral, typed tools of the Agribound agent.

Every tool is a plain function ``tool(context, validated_input) -> output``
with a pydantic input model and a pydantic output model. The same
:class:`ToolRegistry` drives the local tool-use loop
(:mod:`agribound.agent.agent`) and the MCP server
(:mod:`agribound.agent.mcp_server`). This module imports neither
``anthropic`` nor ``mcp``.

Read-only tools
    ``list_sources``, ``list_engines``, ``describe_study_area``,
    ``check_availability``, ``estimate_resolvability``,
    ``recommend_configurations``, ``evaluate_against_reference`` and
    ``query_published_ftw``. None of them runs the pipeline or modifies user
    files. ``query_published_ftw`` writes the downloaded polygons into the
    session work directory, and the network-backed checks may fill download
    caches (published FTW tiles under ``<workdir>/ftw_cache``, the TESSERA tile
    manifest in the geotessera cache).

Gated tools
    ``propose_run`` freezes a validated configuration into a
    :class:`~agribound.agent.plans.Plan` (and writes its YAML); it never runs
    anything. ``execute_plan`` runs :func:`agribound.pipeline.delineate` for a
    plan only after the :class:`~agribound.agent.gate.ConfirmationGate`
    approved that plan's exact hash. Before the reviewer is asked,
    :func:`preflight_execution` refuses plans that cannot run in this session
    (execution disabled, no executions left, changed inputs, or remote
    services needed while network access is off).

Network access
    With ``ToolContext.allow_network=False`` no tool contacts Earth Engine,
    TESSERA, Source Cooperative or the USGS NAIP Plus ImageServer: the
    read-only tools refuse (or skip) their network parts, and
    ``execute_plan`` refuses every plan whose pipeline run would need one of
    these services (:func:`plan_network_services`). Model weights are a
    separate matter: engines download them from Hugging Face unless they are
    already cached (``agribound prefetch``); this flag does not block that.

Expected failures raise :class:`~agribound.agent.errors.AgentToolError`.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agribound.agent.errors import (
    AgentToolError,
    ExecutionDisabledError,
    NetworkDisabledError,
    UnknownPlanError,
)
from agribound.registry import (
    ENGINE_REGISTRY,
    GEE_IMAGERY_SOURCES,
    SOURCE_REGISTRY,
    TESSERA_YEAR_RANGES,
    engine_notes,
    source_year_range,
)

logger = logging.getLogger(__name__)

SourceName = Literal[tuple(SOURCE_REGISTRY)]  # type: ignore[valid-type]
EngineName = Literal[tuple(ENGINE_REGISTRY)]  # type: ignore[valid-type]
TesseraVersion = Literal[tuple(TESSERA_YEAR_RANGES)]  # type: ignore[valid-type]

#: Python modules whose presence is reported per engine by ``list_engines``.
ENGINE_MODULES: dict[str, tuple[str, ...]] = {
    "delineate-anything": ("ultralytics",),
    "ftw": ("ftw_tools",),
    "geoai": ("geoai",),
    "dinov3": ("geoai",),
    "prithvi": ("terratorch",),
    "embedding": ("sklearn",),
    "ensemble": (),
}

#: Configuration fields a proposal may not set (the agent layer controls them).
PROPOSAL_RESERVED_FIELDS: dict[str, str] = {
    "output_path": "outputs are written inside the session work directory (use output_name)",
    "cache_dir": "the session cache directory is used",
    "overwrite": "agent runs never overwrite existing outputs",
    "provenance": "agent runs always write a provenance record",
    "gee_service_account_key": "credentials are configured by the user, not by the agent",
    "gee_project": (
        "the Earth Engine project is configured by the user (the session's gee_project, "
        "GEE_PROJECT, gcloud, or the project_id of the configured credentials file), "
        "not by the agent"
    ),
    "embedding_cache_dir": "embedding caches are kept in the session cache directory",
}

#: Named constants used by the recommendation rules (reported in the tool output).
RECOMMENDATION_PARAMETERS: dict[str, dict[str, Any]] = {
    "conus_bbox_4326": {
        "value": [-125.0, 24.0, -66.5, 49.5],
        "rationale": (
            "Approximate bounding box (min lon, min lat, max lon, max lat) of the conterminous "
            "United States. NAIP in Earth Engine (USDA/NAIP/DOQQ) covers the conterminous US "
            "only; a study-area centroid outside this box excludes 'naip' and 'usgs-naip-plus'. "
            "For 'usgs-naip-plus' the box is a simplification: the service also has footprints "
            "in Alaska, Hawaii, Puerto Rico, Guam, the Northern Mariana Islands and American "
            "Samoa (registry coverage text), and this rule excludes study areas there (propose "
            "them explicitly with propose_run in that case). A centroid inside the box does not "
            "prove coverage; for 'naip' use check_availability with live=True."
        ),
        "citation": (
            "Agribound source registry coverage text; GEE catalog entry USDA/NAIP/DOQQ (NAIP only)"
        ),
    },
    "sentinel2_partial_coverage_years": {
        "value": [2017, 2018],
        "rationale": (
            "Sentinel-2 L2A surface reflectance in Earth Engine is not global in these years; "
            "a candidate for one of them carries a warning (it is not excluded)."
        ),
        "citation": "GEE catalog entry COPERNICUS/S2_SR_HARMONIZED; Agribound source registry",
    },
    "tessera_v1_near_global_years": {
        "value": [2024],
        "rationale": (
            "TESSERA v1 precomputed embeddings are near-global only for 2024; other v1 years "
            "(and every v1.1 year, 2015-2025) are regional. A tessera-embedding candidate for "
            "another year carries a warning (it is not excluded)."
        ),
        "citation": "TESSERA manifests on Source Cooperative (data.source.coop/tessera/tessera)",
    },
}

_EXTENSIONS = (".gpkg", ".geojson", ".json", ".parquet", ".geoparquet")


def _utc_year() -> int:
    import datetime as _dt

    return _dt.datetime.now(_dt.UTC).year


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """State shared by the tools of one agent session or MCP server process.

    Parameters
    ----------
    workdir : str or Path
        Session directory. Plans, plan outputs, downloaded FTW polygons and
        the shared cache (``<workdir>/cache``) are written here.
    study_area : str or None
        Default study area for tools called without one.
    gee_project : str or None
        Default Earth Engine project (plans and live checks).
    gee_service_account_key : str or None
        Service-account key injected into plans (never set by the model).
    reference_boundaries : str or None
        Default reference layer for resolvability and evaluation tools.
    allow_network : bool
        Whether tools may contact Earth Engine, TESSERA, Source Cooperative
        or the USGS NAIP Plus ImageServer. *False* disables the live
        availability checks, published FTW polygons and GEE-asset study areas
        of the read-only tools, and makes ``execute_plan`` refuse plans whose
        pipeline run needs one of these services (:func:`plan_network_services`;
        see the module docstring for model-weight downloads).
    execution_enabled : bool
        Whether ``execute_plan`` may run at all (*False* for dry runs).
    gate : ConfirmationGate or None
        The confirmation gate (required for execution).
    embedding_cache_dir : str or None
        Cache directory for the TESSERA manifest used by live checks.
    """

    workdir: Path
    study_area: str | None = None
    gee_project: str | None = None
    gee_service_account_key: str | None = None
    reference_boundaries: str | None = None
    allow_network: bool = True
    execution_enabled: bool = False
    gate: Any = None
    embedding_cache_dir: str | None = None
    session: Any = None
    plans: dict[str, Any] = field(default_factory=dict)
    executions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir).expanduser().resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_dir(self) -> Path:
        return self.workdir / "cache"

    def resolve_study_area(self, value: str | None) -> str:
        area = value or self.study_area
        if not area:
            raise AgentToolError(
                "No study area was given and the session has no default study area. Pass "
                "study_area (a vector file, GEE asset ID, 'bbox:minx,miny,maxx,maxy' or WKT)."
            )
        return str(area)

    def require_network(self, what: str) -> None:
        if not self.allow_network:
            raise NetworkDisabledError(
                f"{what} needs network access, which is disabled for this session."
            )

    def get_plan(self, plan_id: str) -> Any:
        plan = self.plans.get(plan_id)
        if plan is None:
            known = sorted(self.plans) or "none"
            raise UnknownPlanError(f"Unknown plan_id {plan_id!r}. Plans in this session: {known}")
        return plan


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _export_resolution(source: str) -> float | None:
    """Pixel size (m) of the composite the pipeline writes for *source* by default."""
    if source == "naip":
        from agribound.agent.plans import config_defaults

        return float(config_defaults()["naip_resolution_m"])
    res = SOURCE_REGISTRY[source]["resolution_m"]
    return None if res is None else float(res)


def _init_gee(ctx: ToolContext) -> None:
    """Initialise Earth Engine without any interactive authentication."""
    ctx.require_network("Earth Engine access")
    try:
        from agribound.auth import setup_gee

        setup_gee(
            project=ctx.gee_project,
            service_account_key=ctx.gee_service_account_key,
            interactive=False,
        )
    except ImportError as exc:
        raise AgentToolError(
            'earthengine-api is not installed (pip install "agribound[gee]").'
        ) from exc
    except Exception as exc:
        raise AgentToolError(f"Earth Engine initialisation failed: {exc}") from exc


def _read_aoi(ctx: ToolContext, study_area: str) -> Any:
    """Read a study area as a GeoDataFrame in EPSG:4326."""
    from agribound.io.vector import read_study_area

    if study_area.startswith(("projects/", "users/")):
        _init_gee(ctx)
    try:
        gdf = read_study_area(study_area)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise AgentToolError(f"Could not read study area {study_area!r}: {exc}") from exc
    if len(gdf) == 0:
        raise AgentToolError(f"Study area {study_area!r} contains no features.")
    if gdf.crs is None:
        raise AgentToolError(f"Study area {study_area!r} has no CRS.")
    return gdf.to_crs("EPSG:4326")


def _read_layer(path: str) -> Any:
    from agribound.io.vector import read_vector

    try:
        gdf = read_vector(path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise AgentToolError(f"Could not read vector layer {path!r}: {exc}") from exc
    if gdf.crs is None:
        raise AgentToolError(f"Vector layer {path!r} has no CRS.")
    return gdf


def _restrict_to_aoi(gdf: Any, aoi_4326: Any) -> Any:
    aoi = aoi_4326.to_crs(gdf.crs).geometry.union_all()
    return gdf[gdf.geometry.intersects(aoi)]


def _clean_polygons(gdf: Any) -> tuple[Any, int]:
    """Drop missing/empty geometries; return the layer and the number dropped."""
    keep = gdf.geometry.notna() & ~gdf.geometry.is_empty
    return gdf[keep], int((~keep).sum())


def _quantiles(values: Any) -> dict[str, float] | None:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    q = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "p10": float(q[0]),
        "p25": float(q[1]),
        "median": float(q[2]),
        "p75": float(q[3]),
        "p90": float(q[4]),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _area_stats_ha(gdf: Any) -> dict[str, float] | None:
    """Area quantiles and total in hectares (EPSG:6933; invalid geometries ignored)."""
    import numpy as np

    if len(gdf) == 0:
        return None
    areas = _field_areas_m2(gdf) / 1e4
    areas = areas[np.isfinite(areas)]
    stats = _quantiles(areas)
    if stats is None:
        return None
    stats["total"] = float(areas.sum())
    return stats


def _field_areas_m2(gdf: Any) -> Any:
    """Field areas in m^2, via ``agribound.evaluate.pixels_per_field``.

    ``pixels_per_field(gdf, gsd_m=1.0)`` is ``A / 1 m^2``, i.e. the area in m^2
    in the equal-area CRS (EPSG:6933) used by the evaluation module. Invalid
    geometries are repaired first (as in :func:`agribound.evaluate.evaluate`);
    the area is NaN only for null, empty, non-polygonal or zero-area
    geometries. Raises ImportError when a repair is needed and the installed
    shapely lacks ``make_valid(method="structure")`` (shapely < 2.1).
    """
    from agribound.evaluate import pixels_per_field

    return pixels_per_field(gdf, gsd_m=1.0).to_numpy(dtype=float)


def _load_is_refinable() -> Callable[..., bool] | None:
    try:
        from agribound.engines.samgeo_engine import is_refinable
    except ImportError:
        return None
    return is_refinable


def _min_refinable_square_side_m(
    gsd_m: float, min_crop_px: int, padding: float, is_refinable: Callable[..., bool]
) -> float | None:
    """Smallest square side (m, rounded up to 0.1 m) that ``is_refinable`` accepts at *gsd_m*.

    Found by bisection, so it reproduces whatever rule ``is_refinable`` applies.
    The bisection stops after :data:`_BISECTION_MAX_STEPS` halvings or when the
    midpoint no longer differs from an end point (float spacing wider than
    0.1 m for huge inputs), so it always terminates. Returns None when no
    finite square is refinable.
    """
    lo, hi = 0.0, float(gsd_m) * float(min_crop_px) * 4.0 + 1.0
    if not math.isfinite(hi):
        return None
    pixel = (gsd_m, gsd_m)
    if not is_refinable((0.0, 0.0, hi, hi), pixel, min_crop_px=min_crop_px, padding=padding):
        return None
    for _ in range(_BISECTION_MAX_STEPS):
        if hi - lo <= 0.1:
            break
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if is_refinable(
            (0.0, 0.0, mid, mid), (gsd_m, gsd_m), min_crop_px=min_crop_px, padding=padding
        ):
            hi = mid
        else:
            lo = mid
    return math.ceil(hi * 10.0) / 10.0  # rounded up, so the reported side is refinable


#: Upper bound on the halvings in :func:`_min_refinable_square_side_m` (a float
#: interval shrinks below its spacing after at most ~1100 halvings; 200 reaches
#: 0.1 m from any side below about 1.6e59 m).
_BISECTION_MAX_STEPS = 200


def _package_found(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _utm_extent(aoi_4326: Any) -> tuple[Any, tuple[float, float, float, float]]:
    import geopandas as gpd

    from agribound.io.crs import utm_crs_for_geometry

    union = aoi_4326.geometry.union_all()
    utm = utm_crs_for_geometry(union)
    projected = gpd.GeoSeries([union], crs="EPSG:4326").to_crs(utm)
    bounds = tuple(float(v) for v in projected.total_bounds)
    return utm, bounds


def _composite_estimate(
    source: str,
    utm_bounds: tuple[float, float, float, float],
    tile_size: int,
    resolution_m: float | None = None,
) -> CompositeEstimate:
    info = SOURCE_REGISTRY[source]
    res = resolution_m if resolution_m is not None else _export_resolution(source)
    bands = info["all_bands"]
    n_bands = len(bands) if bands else None
    dtype = "uint8" if info["value_scale"] == "uint8" else "float32"
    if res is None or n_bands is None:
        note = (
            "resolution depends on the input (state vintage or local file); no estimate"
            if res is None
            else "band count depends on the local file; no estimate"
        )
        return CompositeEstimate(source=source, resolution_m=res, dtype=dtype, note=note)
    width = max(1, math.ceil((utm_bounds[2] - utm_bounds[0]) / res))
    height = max(1, math.ceil((utm_bounds[3] - utm_bounds[1]) / res))
    nbytes = 1 if dtype == "uint8" else 4
    return CompositeEstimate(
        source=source,
        resolution_m=res,
        width_px=width,
        height_px=height,
        n_bands=n_bands,
        dtype=dtype,
        uncompressed_mb=round(width * height * n_bands * nbytes / 1e6, 1),
        download_tiles=math.ceil(width / tile_size) * math.ceil(height / tile_size),
        note="bounding box of the study area in the UTM zone of its centroid (export_crs='utm')",
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EmptyInput(_Model):
    """No arguments."""


class SourceInfo(BaseModel):
    source: str
    name: str
    collection: str | None
    export_resolution_m: float | None
    native_resolution_m: float | None
    year_first: int | None
    year_last: int | None = Field(description="None: still acquiring (or no constraint for local)")
    coverage: str
    value_scale: str
    canonical_bands: dict[str, str] | None
    requires_gee: bool
    restricted: bool


class ListSourcesOutput(BaseModel):
    sources: list[SourceInfo]


class EngineInfo(BaseModel):
    engine: str
    name: str
    approach: str
    strengths: str
    label_free: bool = Field(description="Runs without reference boundaries or a user checkpoint")
    fine_tunable: bool
    gpu_recommended: bool
    requires_bands: list[str]
    supported_sources: list[str]
    reference: str
    install_extra: str
    notes: str = Field(description="Notes that apply to every source")
    source_notes: dict[str, str] = Field(
        default_factory=dict,
        description="Notes that apply only to one source (source name -> note)",
    )
    python_packages_found: dict[str, bool] = Field(
        description="Whether the engine's main Python package is importable here "
        "(importable does not guarantee a working install)"
    )


class ListEnginesOutput(BaseModel):
    engines: list[EngineInfo]


class StudyAreaInput(_Model):
    study_area: str | None = Field(
        None,
        description="Vector file, GEE asset ID, 'bbox:minx,miny,maxx,maxy' or WKT (EPSG:4326). "
        "Defaults to the session's study area.",
    )


class CompositeEstimate(BaseModel):
    source: str
    resolution_m: float | None
    width_px: int | None = None
    height_px: int | None = None
    n_bands: int | None = None
    dtype: str
    uncompressed_mb: float | None = None
    download_tiles: int | None = Field(
        None, description="Number of tiles at the default tile_size (10000 px)"
    )
    note: str = ""


class StudyAreaDescription(BaseModel):
    study_area: str
    n_features: int
    area_km2: float = Field(description="Area of the union in EPSG:6933 (equal-area)")
    bbox_4326: list[float]
    centroid_lon_lat: list[float]
    utm_zone: int
    utm_epsg: int
    utm_zones_spanned: list[int]
    composite_estimates: list[CompositeEstimate]


class AvailabilityInput(_Model):
    year: int = Field(ge=1950, le=2100, description="Target year")
    sources: list[SourceName] | None = Field(
        None, description="Sources to check (default: every source except 'local')"
    )
    study_area: str | None = Field(None, description="Needed for live checks; session default")
    live: bool = Field(
        False,
        description="Also query Earth Engine image counts / TESSERA tile counts for the study "
        "area's bounding box (network; GEE needs credentials; TESSERA downloads its tile "
        "manifest on first use).",
    )
    tessera_version: TesseraVersion = "v1"
    tessera_variant: str | None = Field(
        None,
        description="TESSERA dataset variant for the live check (geotessera dataset_variant); "
        "None uses geotessera's default variant for the version",
    )


class LiveCheck(BaseModel):
    status: Literal["ok", "not_run", "error", "not_available"]
    method: str = ""
    image_count: int | None = None
    per_collection: dict[str, int] | None = None
    tile_count: int | None = None
    message: str | None = None


class SourceAvailability(BaseModel):
    source: str
    year: int
    registry_year_range: list[int | None] | None
    in_registry_range: bool
    coverage: str
    restricted: bool
    requires_gee: bool
    live: LiveCheck | None = None


class AvailabilityOutput(BaseModel):
    results: list[SourceAvailability]
    notes: list[str]


class ResolvabilityInput(_Model):
    study_area: str | None = Field(
        None, description="Restricts a reference layer; needed for published FTW polygons"
    )
    reference_path: str | None = Field(
        None, description="Reference field boundaries (field sizes are read from them)"
    )
    use_published_ftw: bool = Field(
        False,
        description="Use published FTW prediction polygons as the field-size sample (network; "
        "model predictions, not ground truth)",
    )
    median_field_area_ha: float | None = Field(
        None, gt=0, description="A representative field area in hectares (one square field)"
    )
    year: int | None = Field(
        None,
        description="Prediction year of the published FTW polygons. Default: the latest year "
        "present, so that a field predicted in several years is counted once.",
    )
    sources: list[SourceName] | None = Field(
        None, description="Sources to evaluate (default: all with a known export resolution)"
    )
    gsd_m: dict[str, Annotated[float, Field(gt=0, le=10_000, allow_inf_nan=False)]] | None = Field(
        None,
        description="Per-source ground sampling distance overrides in metres (0-10,000 m)",
    )
    min_crop_px: int = Field(
        64,
        ge=1,
        le=100_000,
        description="SAM refinement minimum crop in pixels (AgriboundConfig.sam_min_crop_px "
        "default)",
    )
    crop_padding: float = Field(
        0.15,
        ge=0,
        le=100,
        allow_inf_nan=False,
        description="SAM crop padding fraction (AgriboundConfig.sam_crop_padding default)",
    )

    @model_validator(mode="after")
    def _one_field_size_source(self) -> ResolvabilityInput:
        given = [
            name
            for name, present in (
                ("reference_path", self.reference_path is not None),
                ("use_published_ftw", self.use_published_ftw),
                ("median_field_area_ha", self.median_field_area_ha is not None),
            )
            if present
        ]
        if len(given) > 1:
            raise ValueError(f"give exactly one field-size source, got {given}")
        return self


class SamEligibility(BaseModel):
    available: bool
    eligible_count: int | None = None
    eligible_fraction: float | None = None
    eligible_area_fraction: float | None = None
    min_refinable_square_side_m: float | None = None
    message: str | None = None


class SourceResolvability(BaseModel):
    source: str
    gsd_m: float
    gsd_basis: str
    native_resolution_m: float | None
    pixels_per_field: dict[str, float] | None
    sam_refinement: SamEligibility


class ResolvabilityOutput(BaseModel):
    field_size_source: str
    n_fields: int
    n_dropped_empty: int
    published_ftw_years: dict[str, int] | None = Field(
        None,
        description="Published FTW polygons per prediction year returned by the query, before "
        "a single year was selected (None for other field-size sources)",
    )
    published_ftw_year_used: int | None = Field(
        None, description="Prediction year whose published FTW polygons form the sample"
    )
    field_area_ha: dict[str, float] | None
    per_source: list[SourceResolvability]
    skipped_sources: dict[str, str]
    definitions: dict[str, str]
    parameters: dict[str, Any]
    notes: list[str]


class RecommendInput(_Model):
    year: int = Field(ge=1950, le=2100)
    study_area: str | None = None
    reference_path: str | None = Field(
        None, description="Reference boundaries available for fine-tuning (and field sizes)"
    )
    median_field_area_ha: float | None = Field(
        None, gt=0, description="Representative field size if no reference layer is given"
    )
    prefer_label_free: bool = Field(True, description="Rank runs without fine-tuning first")
    want_sam_refine: bool = Field(False, description="The user asked for SAM edge refinement")
    include_restricted: bool = Field(
        False, description="Include restricted sources (SPOT 6/7) the user has access to"
    )
    sources: list[SourceName] | None = None
    engines: list[EngineName] | None = None
    tessera_version: TesseraVersion = Field(
        "v1",
        description="TESSERA dataset version whose year range is checked for "
        "tessera-embedding candidates (AgriboundConfig.tessera_version default 'v1'); a "
        "non-default version is written into the candidates' config",
    )
    min_median_pixels_per_field: float | None = Field(
        None,
        gt=0,
        description="Optional user-chosen exclusion threshold on median pixels per field "
        "(A/GSD^2). No threshold is applied when omitted.",
    )
    max_candidates: int = Field(10, ge=1, le=50)


class Candidate(BaseModel):
    rank: int
    source: str
    engine: str
    proposal: dict[str, Any] = Field(description="Arguments for propose_run")
    rules_applied: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


class Excluded(BaseModel):
    source: str
    engine: str | None
    reasons: list[str]


class RecommendOutput(BaseModel):
    candidates: list[Candidate]
    excluded: list[Excluded]
    ordering: str
    parameters: dict[str, Any]
    notes: list[str]


class FtwQueryInput(_Model):
    study_area: str | None = None
    year: int | None = Field(None, description="Filter on the polygons' year/time column")
    min_confidence: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Keep polygons with confidence >= this value, on the published 0-100 "
        "scale (query_ftw min_confidence; the dataset README recommends 69)",
    )
    keep_null_confidence: bool = Field(
        True,
        description="Keep polygons without a confidence value (outside the confidence layer)",
    )
    max_features: int | None = Field(None, ge=1, description="Row limit for previews")


class FtwQueryOutput(BaseModel):
    n_polygons: int = Field(
        description="Rows returned; with several prediction years a field can appear once per "
        "year (see polygons_per_year)"
    )
    polygons_per_year: dict[str, int] | None = Field(
        None,
        description="Rows per prediction year (agribound.ftw_arrow.row_years; 'unknown' for "
        "rows without a year); None when the layer has no year column",
    )
    output_path: str | None
    area_ha: dict[str, float] | None
    columns: list[str]
    notes: list[str]


class EvaluateInput(_Model):
    predicted_path: str = Field(description="Predicted field boundaries (vector file)")
    reference_path: str | None = Field(None, description="Reference layer; session default")
    iou_threshold: float = Field(0.5, gt=0, le=1)
    boundary_tolerance_m: float | None = Field(
        None, gt=0, description="Passed to agribound.evaluate.evaluate when given"
    )
    restrict_to_study_area: bool = Field(
        True, description="Use only reference polygons intersecting the study area"
    )
    study_area: str | None = None


class EvaluateOutput(BaseModel):
    n_predicted: int
    n_reference: int
    reference_selection: str
    metrics: dict[str, Any]


class ProposeRunInput(_Model):
    source: SourceName
    engine: EngineName
    year: int
    study_area: str | None = Field(None, description="Defaults to the session's study area")
    reference_boundaries: str | None = Field(
        None, description="Reference boundaries (fine-tuning, or evaluation when not fine-tuning)"
    )
    fine_tune: bool = False
    local_tif_path: str | None = None
    output_name: str | None = Field(
        None,
        description="Output file name (no directories) with extension .gpkg, .geojson, "
        ".parquet; the file goes into the plan directory",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Other AgriboundConfig fields (validated). Keep package defaults unless "
        "the user explicitly asked for a value.",
    )
    rationale: str = Field("", description="Why this configuration answers the request")
    limitations: list[str] = Field(
        default_factory=list,
        description="Limitations of this run, taken from tool outputs (GSD vs field size, "
        "labels, out-of-distribution inputs, imagery access)",
    )
    alternatives: list[str] = Field(
        default_factory=list, description="Alternative configurations the user could choose"
    )


class ProposeRunOutput(BaseModel):
    plan_id: str
    plan_hash: str
    yaml_path: str
    output_path: str
    config: dict[str, Any] = Field(
        description="The frozen AgriboundConfig dictionary of the plan (what execute_plan runs)"
    )
    warnings: list[str]
    non_default_fields: dict[str, Any]
    estimated_cost: dict[str, Any]
    network_services: list[str] = Field(
        description="Remote services the pipeline run contacts, judged from the configuration "
        "(caches are not inspected; model-weight downloads are not listed)"
    )
    execution_enabled: bool
    next_step: str


class ExecutePlanInput(_Model):
    plan_id: str = Field(description="plan_id returned by propose_run")


class ExecutionSummary(BaseModel):
    plan_id: str
    plan_hash: str
    status: Literal["success"]
    run_id: str | None
    n_polygons: int
    stage_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Polygon counts after each pipeline stage, from the provenance record "
        "(n_detected, n_postprocessed, n_after_lulc, n_output)",
    )
    area_ha: dict[str, float] | None
    lulc_status: str | None = Field(
        None, description="LULC filter status from the provenance record"
    )
    lulc_stats: dict[str, Any] | None
    sam_stats: dict[str, Any] | None
    evaluation: dict[str, Any] | None
    engine_meta: dict[str, Any] | None
    reused_existing_output: bool
    output_path: str
    provenance_path: str | None
    provenance_warnings: list[str]
    wall_s: float
    approval: dict[str, Any]


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------


def list_sources_tool(ctx: ToolContext, inp: EmptyInput) -> ListSourcesOutput:
    out = []
    for key, info in SOURCE_REGISTRY.items():
        yr = info["year_range"]
        out.append(
            SourceInfo(
                source=key,
                name=info["name"],
                collection=info["collection"],
                export_resolution_m=_export_resolution(key),
                native_resolution_m=info["native_resolution_m"],
                year_first=None if yr is None else yr[0],
                year_last=None if yr is None else yr[1],
                coverage=info["coverage"],
                value_scale=info["value_scale"],
                canonical_bands=info["canonical_bands"],
                requires_gee=info["requires_gee"],
                restricted=info["restricted"],
            )
        )
    return ListSourcesOutput(sources=out)


def list_engines_tool(ctx: ToolContext, inp: EmptyInput) -> ListEnginesOutput:
    out = []
    for key, info in ENGINE_REGISTRY.items():
        out.append(
            EngineInfo(
                engine=key,
                name=info["name"],
                approach=info["approach"],
                strengths=info["strengths"],
                label_free=info["label_free"],
                fine_tunable=info["fine_tunable"],
                gpu_recommended=info["gpu_recommended"],
                requires_bands=list(info["requires_bands"]),
                supported_sources=list(info["supported_sources"]),
                reference=info["reference"],
                install_extra=info["install_extra"],
                notes=info.get("notes", ""),
                source_notes=dict(info.get("source_notes") or {}),
                python_packages_found={m: _package_found(m) for m in ENGINE_MODULES.get(key, ())},
            )
        )
    return ListEnginesOutput(engines=out)


def describe_study_area_tool(ctx: ToolContext, inp: StudyAreaInput) -> StudyAreaDescription:
    from agribound.agent.plans import config_defaults
    from agribound.io.crs import get_equal_area_crs, utm_zones_for_bounds

    study_area = ctx.resolve_study_area(inp.study_area)
    aoi = _read_aoi(ctx, study_area)
    union = aoi.geometry.union_all()
    import geopandas as gpd

    area_m2 = float(gpd.GeoSeries([union], crs="EPSG:4326").to_crs(get_equal_area_crs()).area[0])
    utm, utm_bounds = _utm_extent(aoi)
    bbox = [float(v) for v in union.bounds]
    centroid = union.centroid
    epsg = int(utm.to_epsg())
    tile_size = int(config_defaults()["tile_size"])
    estimates = [
        _composite_estimate(src, utm_bounds, tile_size) for src in SOURCE_REGISTRY if src != "local"
    ]
    return StudyAreaDescription(
        study_area=study_area,
        n_features=len(aoi),
        area_km2=round(area_m2 / 1e6, 4),
        bbox_4326=bbox,
        centroid_lon_lat=[float(centroid.x), float(centroid.y)],
        utm_zone=epsg % 100,
        utm_epsg=epsg,
        utm_zones_spanned=utm_zones_for_bounds(tuple(bbox)),
        composite_estimates=estimates,
    )


def _gee_collections(source: str) -> list[str]:
    collection = SOURCE_REGISTRY[source]["collection"] or ""
    return [c.strip() for c in collection.split(" + ") if c.strip()]


def _live_gee(ctx: ToolContext, source: str, bbox: list[float], year: int) -> LiveCheck:
    _init_gee(ctx)
    import ee

    region = ee.Geometry.Rectangle(list(bbox), "EPSG:4326", False)
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    counts: dict[str, int] = {}
    for cid in _gee_collections(source):
        n = ee.ImageCollection(cid).filterBounds(region).filterDate(start, end).size().getInfo()
        counts[cid] = int(n)
    return LiveCheck(
        status="ok",
        method=(
            "ee.ImageCollection(...).filterBounds(study-area bbox).filterDate(year).size(), "
            "before cloud filtering"
        ),
        image_count=sum(counts.values()),
        per_collection=counts,
    )


def _live_tessera(
    ctx: ToolContext, bbox: list[float], year: int, version: str, variant: str | None = None
) -> LiveCheck:
    ctx.require_network("A TESSERA coverage check")
    try:
        from geotessera import GeoTessera
    except ImportError as exc:
        raise AgentToolError(
            'geotessera is not installed (pip install "agribound[tessera]").'
        ) from exc
    gt = GeoTessera(
        dataset_version=version, dataset_variant=variant, cache_dir=ctx.embedding_cache_dir
    )
    n = gt.embeddings_count(tuple(bbox), year=int(year))
    return LiveCheck(
        status="ok",
        method=(
            f"geotessera GeoTessera(dataset_version={version!r}, dataset_variant={variant!r})"
            ".embeddings_count(bbox, year)"
        ),
        tile_count=int(n),
        message="0.1-degree tiles intersecting the bounding box; partial coverage is not assessed",
    )


def check_availability_tool(ctx: ToolContext, inp: AvailabilityInput) -> AvailabilityOutput:
    sources = inp.sources or [s for s in SOURCE_REGISTRY if s != "local"]
    bbox = None
    if inp.live:
        aoi = _read_aoi(ctx, ctx.resolve_study_area(inp.study_area))
        bbox = [float(v) for v in aoi.geometry.union_all().bounds]
    results = []
    for source in sources:
        info = SOURCE_REGISTRY[source]
        yr = source_year_range(
            source, tessera_version=inp.tessera_version if source == "tessera-embedding" else None
        )
        if yr is None:
            in_range = True
        else:
            upper = yr[1] if yr[1] is not None else _utc_year()
            in_range = yr[0] <= inp.year <= upper
        live = None
        if inp.live:
            live = _live_check(
                ctx, source, bbox, inp.year, inp.tessera_version, in_range, inp.tessera_variant
            )
        results.append(
            SourceAvailability(
                source=source,
                year=inp.year,
                registry_year_range=None if yr is None else [yr[0], yr[1]],
                in_registry_range=in_range,
                coverage=info["coverage"],
                restricted=info["restricted"],
                requires_gee=info["requires_gee"],
                live=live,
            )
        )
    notes = [
        "Registry year ranges are the years with any data; coverage within a year can be "
        "partial (see 'coverage').",
    ]
    if not inp.live:
        notes.append("No live check was run (live=False).")
    return AvailabilityOutput(results=results, notes=notes)


def _live_check(
    ctx: ToolContext,
    source: str,
    bbox: list[float],
    year: int,
    version: str,
    in_range: bool,
    variant: str | None = None,
) -> LiveCheck:
    if not ctx.allow_network:
        return LiveCheck(status="not_run", message="network access is disabled for this session")
    if not in_range:
        return LiveCheck(status="not_run", message="year outside the registry range")
    try:
        if source in GEE_IMAGERY_SOURCES or source == "google-embedding":
            return _live_gee(ctx, source, bbox, year)
        if source == "tessera-embedding":
            return _live_tessera(ctx, bbox, year, version, variant)
    except AgentToolError as exc:
        return LiveCheck(status="error", message=str(exc))
    except Exception as exc:
        return LiveCheck(status="error", message=f"{type(exc).__name__}: {exc}")
    if source == "usgs-naip-plus":
        return LiveCheck(
            status="not_available",
            message="no live check is implemented for the USGS NAIP Plus ImageServer; the "
            "composite stage reports the state's available years when the year is missing",
        )
    return LiveCheck(status="not_available", message="no live check for this source")


def _row_years(gdf: Any) -> Any:
    """Per-row prediction years of published FTW polygons (``agribound.ftw_arrow.row_years``)."""
    try:
        from agribound.ftw_arrow import row_years
    except ImportError:  # pragma: no cover - part of the package
        return None
    return row_years(gdf)


def _year_counts(gdf: Any) -> dict[str, int] | None:
    """Rows per prediction year (``"unknown"`` for rows without one); None without a year column."""
    import pandas as pd

    years = _row_years(gdf)
    if years is None:
        return None
    counts: dict[str, int] = {}
    for value in years:
        key = "unknown" if pd.isna(value) else str(int(value))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _field_sample(ctx: ToolContext, inp: Any) -> tuple[Any, str, int, list[str], dict[str, Any]]:
    """Return (polygons, description, n_dropped, notes, extra) for the field-size source of *inp*.

    *extra* holds the published-FTW year information (empty for the other
    field-size sources).
    """
    notes: list[str] = []
    reference = inp.reference_path
    use_ftw = getattr(inp, "use_published_ftw", False)
    if reference is None and not use_ftw and inp.median_field_area_ha is None:
        reference = ctx.reference_boundaries
    if reference is not None:
        gdf = _read_layer(reference)
        study_area = inp.study_area or ctx.study_area
        if study_area:
            gdf = _restrict_to_aoi(gdf, _read_aoi(ctx, study_area))
            notes.append("Reference polygons intersecting the study area only.")
        gdf, dropped = _clean_polygons(gdf)
        if len(gdf) == 0:
            raise AgentToolError(f"No reference polygons in {reference!r} (after AOI selection).")
        return gdf, f"reference:{reference}", dropped, notes, {}
    if use_ftw:
        gdf = _query_ftw(ctx, inp.study_area, inp.year, None, None, clip=False)
        notes.append(
            "Field sizes from published FTW predictions (model output, not ground truth); "
            "polygons crossing the study-area edge are kept whole."
        )
        gdf, dropped = _clean_polygons(gdf)
        if len(gdf) == 0:
            raise AgentToolError("No published FTW polygons were found for the study area.")
        per_year = _year_counts(gdf)
        extra: dict[str, Any] = {"published_ftw_years": per_year}
        years = _row_years(gdf)
        if inp.year is not None:
            extra["published_ftw_year_used"] = int(inp.year)
        elif years is not None and years.notna().any():
            latest = int(years.max())
            extra["published_ftw_year_used"] = latest
            if len(per_year or {}) > 1:
                mask = (years == latest).fillna(False).to_numpy(dtype=bool)
                gdf = gdf[mask]
                notes.append(
                    f"The published FTW polygons span several prediction years {per_year}; only "
                    f"the latest year ({latest}) is used, so that a field predicted in several "
                    "years is counted once. Pass year to choose another year."
                )
        else:
            notes.append(
                "The prediction year of the published FTW polygons is unknown; if the layer "
                "holds several years, a field can be counted once per year."
            )
        return gdf, "published_ftw", dropped, notes, extra
    if inp.median_field_area_ha is None:
        raise AgentToolError(
            "Give reference_path, use_published_ftw=True or median_field_area_ha (the session "
            "has no default reference layer)."
        )
    import geopandas as gpd
    from shapely.geometry import box

    side = math.sqrt(float(inp.median_field_area_ha) * 1e4)
    # A square field in the equal-area CRS used for areas; only its size matters.
    from agribound.io.crs import get_equal_area_crs

    gdf = gpd.GeoDataFrame(geometry=[box(0.0, 0.0, side, side)], crs=get_equal_area_crs())
    notes.append(
        f"One square field of {inp.median_field_area_ha} ha (side {side:.1f} m) stands for the "
        "field-size distribution; fractions are therefore 0 or 1."
    )
    return gdf, "user_median", 0, notes, {}


def estimate_resolvability_tool(ctx: ToolContext, inp: ResolvabilityInput) -> ResolvabilityOutput:
    import numpy as np

    from agribound.io.crs import utm_crs_for_geometry

    fields_gdf, basis, dropped, notes, sample_extra = _field_sample(ctx, inp)
    areas = _field_areas_m2(fields_gdf)
    valid = np.isfinite(areas) & (areas > 0)
    n_invalid = int((~valid).sum())
    if n_invalid:
        notes.append(f"{n_invalid} polygon(s) without a valid polygonal area were ignored.")
    if not valid.any():
        raise AgentToolError("No field polygon has a valid area.")
    if basis == "user_median":
        metric = fields_gdf
    else:
        from shapely.geometry import box

        extent = box(*fields_gdf.to_crs("EPSG:4326").total_bounds)
        metric = fields_gdf.to_crs(utm_crs_for_geometry(extent))
        notes.append(
            f"Bounding boxes in {metric.crs.to_string()} (UTM zone of the centre of the layer's "
            "extent)."
        )
    bounds = metric.geometry.bounds.to_numpy(dtype=float)[valid]
    areas = areas[valid]
    is_refinable = _load_is_refinable()

    overrides = dict(inp.gsd_m or {})
    unknown = sorted(set(overrides) - set(SOURCE_REGISTRY))
    if unknown:
        raise AgentToolError(f"gsd_m has unknown sources {unknown}")
    sources = inp.sources or list(SOURCE_REGISTRY)
    per_source, skipped = [], {}
    for source in sources:
        if source in overrides:
            gsd, gsd_basis = float(overrides[source]), "user override"
        else:
            gsd = _export_resolution(source)
            gsd_basis = "default export resolution of the composite"
        if gsd is None:
            skipped[source] = "no fixed resolution (pass gsd_m to include it)"
            continue
        if gsd <= 0:
            raise AgentToolError(f"gsd_m for {source!r} must be > 0")
        ppf = _quantiles(areas / gsd**2)
        if is_refinable is None:
            sam = SamEligibility(
                available=False,
                message="agribound.engines.samgeo_engine.is_refinable is not available in "
                "this installation",
            )
        else:
            ok = np.array(
                [
                    bool(
                        is_refinable(
                            tuple(b),
                            (gsd, gsd),
                            min_crop_px=inp.min_crop_px,
                            padding=inp.crop_padding,
                        )
                    )
                    for b in bounds
                ],
                dtype=bool,
            )
            total_area = float(areas.sum())
            sam = SamEligibility(
                available=True,
                eligible_count=int(ok.sum()),
                eligible_fraction=float(ok.mean()),
                eligible_area_fraction=float(areas[ok].sum() / total_area) if total_area else None,
                min_refinable_square_side_m=_min_refinable_square_side_m(
                    gsd, inp.min_crop_px, inp.crop_padding, is_refinable
                ),
            )
        per_source.append(
            SourceResolvability(
                source=source,
                gsd_m=gsd,
                gsd_basis=gsd_basis,
                native_resolution_m=SOURCE_REGISTRY[source]["native_resolution_m"],
                pixels_per_field=ppf,
                sam_refinement=sam,
            )
        )
    area_ha = _quantiles(areas / 1e4)
    if area_ha is not None:
        area_ha["total"] = float(areas.sum() / 1e4)
    return ResolvabilityOutput(
        field_size_source=basis,
        n_fields=int(valid.sum()),
        n_dropped_empty=dropped,
        published_ftw_years=sample_extra.get("published_ftw_years"),
        published_ftw_year_used=sample_extra.get("published_ftw_year_used"),
        field_area_ha=area_ha,
        per_source=per_source,
        skipped_sources=skipped,
        definitions={
            "pixels_per_field": "p = A / GSD^2 with A the field area in m^2 as computed by "
            "agribound.evaluate.pixels_per_field (EPSG:6933 equal-area, repaired geometries)",
            "sam_refinement": "fields accepted by agribound.engines.samgeo_engine.is_refinable("
            "field bounding box, (GSD, GSD), min_crop_px, padding); the SAM refinement stage "
            "leaves the other fields unrefined",
            "min_refinable_square_side_m": "smallest square field side accepted by is_refinable "
            "at this GSD (bisection to 0.1 m)",
        },
        parameters={
            "min_crop_px": inp.min_crop_px,
            "crop_padding": inp.crop_padding,
            "citations": [
                "Duveiller, G., Defourny, P. (2010). A conceptual framework to define the spatial "
                "resolution requirements for agricultural monitoring using remote sensing. "
                "Remote Sensing of Environment 114(11), 2637-2650.",
            ],
        },
        notes=notes,
    )


def recommend_configurations_tool(ctx: ToolContext, inp: RecommendInput) -> RecommendOutput:
    study_area = ctx.resolve_study_area(inp.study_area)
    aoi = _read_aoi(ctx, study_area)
    centroid = aoi.geometry.union_all().centroid
    reference = inp.reference_path or ctx.reference_boundaries

    median_area_m2 = None
    size_basis = None
    if inp.median_field_area_ha is not None:
        median_area_m2, size_basis = float(inp.median_field_area_ha) * 1e4, "user-given"
    elif reference is not None:
        import numpy as np

        ref = _restrict_to_aoi(_read_layer(reference), aoi)
        ref, _ = _clean_polygons(ref)
        ref_areas = _field_areas_m2(ref) if len(ref) > 0 else np.array([])
        ref_areas = ref_areas[np.isfinite(ref_areas) & (ref_areas > 0)]
        if ref_areas.size:
            median_area_m2 = float(np.median(ref_areas))
            size_basis = f"median of {ref_areas.size} reference polygons in the study area"

    from agribound.agent.plans import config_defaults

    is_refinable = _load_is_refinable()
    defaults = config_defaults()
    sam_min_crop_px = int(defaults["sam_min_crop_px"])
    sam_padding = float(defaults["sam_crop_padding"])
    conus = RECOMMENDATION_PARAMETERS["conus_bbox_4326"]["value"]
    in_conus = conus[0] <= centroid.x <= conus[2] and conus[1] <= centroid.y <= conus[3]
    source_order = list(SOURCE_REGISTRY)
    engine_order = list(ENGINE_REGISTRY)
    sources = inp.sources or source_order
    engines = inp.engines or engine_order

    candidates: list[dict[str, Any]] = []
    excluded: list[Excluded] = []
    for source in sources:
        info = SOURCE_REGISTRY[source]
        reasons: list[str] = []
        warn: list[str] = []
        if source == "local":
            reasons.append("needs a user GeoTIFF (local_tif_path); propose it explicitly")
        is_tessera = source == "tessera-embedding"
        yr = source_year_range(source, tessera_version=inp.tessera_version if is_tessera else None)
        if yr is not None:
            upper = yr[1] if yr[1] is not None else _utc_year()
            if not yr[0] <= inp.year <= upper:
                if is_tessera:
                    others = [
                        f"{v} ({a}-{b})"
                        for v, (a, b) in TESSERA_YEAR_RANGES.items()
                        if v != inp.tessera_version and a <= inp.year <= b
                    ]
                    reasons.append(
                        f"year {inp.year} outside the TESSERA {inp.tessera_version} range "
                        f"{yr[0]}-{yr[1]}"
                        + (f"; covered by tessera_version {', '.join(others)}" if others else "")
                    )
                else:
                    reasons.append(f"year {inp.year} outside the registry range {yr[0]}-{yr[1]}")
        if info["restricted"] and not inp.include_restricted:
            reasons.append("restricted access (include_restricted=False)")
        if source == "naip" and not in_conus:
            reasons.append(
                "study-area centroid outside conus_bbox_4326 (NAIP in Earth Engine covers the "
                "conterminous US)"
            )
        if source == "usgs-naip-plus" and not in_conus:
            reasons.append(
                "study-area centroid outside conus_bbox_4326 (the rule covers only the "
                "conterminous part of USGS NAIP Plus; see the conus_bbox_4326 rationale)"
            )
        gsd = _export_resolution(source)
        median_p = None
        if gsd is not None and median_area_m2 is not None:
            median_p = median_area_m2 / gsd**2
            if (
                inp.min_median_pixels_per_field is not None
                and median_p < inp.min_median_pixels_per_field
            ):
                reasons.append(
                    f"median pixels per field {median_p:.0f} < min_median_pixels_per_field "
                    f"{inp.min_median_pixels_per_field}"
                )
        if reasons:
            excluded.append(Excluded(source=source, engine=None, reasons=reasons))
            continue
        s2_partial = RECOMMENDATION_PARAMETERS["sentinel2_partial_coverage_years"]["value"]
        if source == "sentinel2" and inp.year in s2_partial:
            warn.append(
                f"Sentinel-2 L2A coverage in {inp.year} is not global "
                "(sentinel2_partial_coverage_years)."
            )
        tessera_global = RECOMMENDATION_PARAMETERS["tessera_v1_near_global_years"]["value"]
        if is_tessera and inp.tessera_version == "v1" and inp.year not in tessera_global:
            warn.append(
                f"TESSERA v1 is near-global only for {tessera_global}; {inp.year} is regional "
                "in v1 (and v1.1 is regional in every year). Check coverage with "
                "check_availability(live=True) for tessera_version v1 or v1.1."
            )
        elif is_tessera and inp.tessera_version != "v1":
            warn.append(
                f"TESSERA {inp.tessera_version} coverage is regional or sparse (registry "
                f"coverage: {info['coverage']}). Check it with check_availability(live=True, "
                f"tessera_version={inp.tessera_version!r})."
            )
        if source in ("naip", "usgs-naip-plus"):
            warn.append(f"Coverage: {info['coverage']}")
        for engine in engines:
            einfo = ENGINE_REGISTRY[engine]
            if source not in einfo["supported_sources"]:
                continue
            if engine == "ensemble":
                continue
            variants: list[tuple[bool, dict[str, Any], list[str]]] = []
            if einfo["label_free"]:
                params: dict[str, Any] = {}
                rules = ["label_free engine: runs without reference boundaries"]
                if engine == "prithvi":
                    params = {"mode": "embed"}
                    rules = [
                        "prithvi is label-free only with engine_params mode='embed' (clustering)"
                    ]
                variants.append((False, params, rules))
            if reference is not None and einfo["fine_tunable"]:
                variants.append((True, {}, ["reference boundaries available: fine-tuning"]))
            if not variants:
                excluded.append(
                    Excluded(
                        source=source,
                        engine=engine,
                        reasons=["not label-free and no reference boundaries for fine-tuning"],
                    )
                )
                continue
            for fine_tune, params, rules in variants:
                proposal: dict[str, Any] = {
                    "source": source,
                    "engine": engine,
                    "year": inp.year,
                    "study_area": study_area,
                    "fine_tune": fine_tune,
                }
                config: dict[str, Any] = {}
                if params:
                    config["engine_params"] = params
                if is_tessera and inp.tessera_version != "v1":
                    config["tessera_version"] = inp.tessera_version
                if fine_tune:
                    proposal["reference_boundaries"] = reference
                cwarn = list(warn)
                for note in engine_notes(engine, source):
                    cwarn.append(f"{einfo['name']}: {note}")
                if not all(_package_found(m) for m in ENGINE_MODULES.get(engine, ())):
                    cwarn.append(
                        f"Python package for {engine} not found; install "
                        f"'agribound[{einfo['install_extra']}]'."
                    )
                metrics: dict[str, Any] = {"gsd_m": gsd, "median_pixels_per_field": median_p}
                if inp.want_sam_refine and engine != "embedding":
                    config["sam_refine"] = True
                    if is_refinable is not None and gsd is not None:
                        side = _min_refinable_square_side_m(
                            gsd, sam_min_crop_px, sam_padding, is_refinable
                        )
                        metrics["min_refinable_square_side_m"] = side
                        if side is not None:
                            cwarn.append(
                                f"SAM refinement (sam_min_crop_px={sam_min_crop_px}, "
                                f"sam_crop_padding={sam_padding}) at {gsd} m only refines "
                                f"fields at least ~{side} m across (square); use "
                                "estimate_resolvability for the refined fraction."
                            )
                if config:
                    proposal["config"] = config
                candidates.append(
                    {
                        "key": (
                            0 if (fine_tune is not inp.prefer_label_free) else 1,
                            gsd if gsd is not None else math.inf,
                            source_order.index(source),
                            engine_order.index(engine),
                        ),
                        "source": source,
                        "engine": engine,
                        "proposal": proposal,
                        "rules": rules
                        + [
                            f"year {inp.year} within the registry range of {source}"
                            + (f" (tessera_version {inp.tessera_version})" if is_tessera else "")
                        ],
                        "warnings": cwarn,
                        "metrics": metrics,
                    }
                )
    if "ensemble" in engines:
        excluded.append(
            Excluded(
                source="*",
                engine="ensemble",
                reasons=[
                    "ensembles are not generated automatically; choose the members explicitly "
                    "with propose_run (engine_params['engines'])"
                ],
            )
        )
    candidates.sort(key=lambda c: c["key"])
    ranked = [
        Candidate(
            rank=i + 1,
            source=c["source"],
            engine=c["engine"],
            proposal=c["proposal"],
            rules_applied=c["rules"],
            warnings=c["warnings"],
            metrics=c["metrics"],
        )
        for i, c in enumerate(candidates[: inp.max_candidates])
    ]
    parameters = dict(RECOMMENDATION_PARAMETERS)
    parameters["min_median_pixels_per_field"] = {
        "value": inp.min_median_pixels_per_field,
        "rationale": "user-supplied; no pixels-per-field threshold is applied when None",
    }
    parameters["median_field_area_m2"] = {"value": median_area_m2, "basis": size_basis}
    parameters["tessera_version"] = {
        "value": inp.tessera_version,
        "rationale": "TESSERA dataset version whose year range (TESSERA_YEAR_RANGES) decides "
        "whether tessera-embedding is a candidate for the year",
    }
    parameters["sam_min_crop_px"] = {
        "value": sam_min_crop_px,
        "rationale": "AgriboundConfig default used by the SAM refinement stage; with "
        "want_sam_refine=True it sets the smallest refinable field reported in the warnings "
        "(agribound.engines.samgeo_engine.is_refinable)",
    }
    parameters["sam_crop_padding"] = {
        "value": sam_padding,
        "rationale": "AgriboundConfig default crop padding of the SAM refinement stage "
        "(agribound.engines.samgeo_engine.is_refinable)",
    }
    return RecommendOutput(
        candidates=ranked,
        excluded=excluded,
        ordering=(
            "1) runs matching prefer_label_free first (no fine-tuning when True, fine-tuning "
            "first when False); 2) finer export GSD first (equivalently, more pixels per field "
            "for a given field size); 3) registry order of sources, then engines."
        ),
        parameters=parameters,
        notes=[
            "Rule-based; the ranking does not predict accuracy. Candidates keep the package "
            "defaults for all thresholds and filters (including the LULC crop-filter "
            "threshold); they set only sam_refine (when want_sam_refine=True), Prithvi's "
            "engine_params mode='embed' (label-free runs) and, for tessera-embedding, the "
            "requested non-default tessera_version.",
            f"{len(candidates)} candidates before truncation to max_candidates.",
        ],
    )


def _query_ftw(
    ctx: ToolContext,
    study_area: str | None,
    year: int | None,
    min_confidence: float | None,
    max_features: int | None,
    *,
    clip: bool,
    keep_null_confidence: bool = True,
    output_path: Path | None = None,
) -> Any:
    ctx.require_network("Querying published FTW polygons")
    from agribound.ftw_query import query_ftw

    aoi = _read_aoi(ctx, ctx.resolve_study_area(study_area))
    kwargs: dict[str, Any] = {
        "year": year,
        "clip": clip,
        "cache_dir": str(ctx.workdir / "ftw_cache"),
        "max_features": max_features,
    }
    if output_path is not None:
        kwargs["output_path"] = str(output_path)
    if min_confidence is not None:
        kwargs["min_confidence"] = min_confidence
    if not keep_null_confidence:
        kwargs["keep_null_confidence"] = False
    try:
        return query_ftw(aoi, **kwargs)
    except TypeError as exc:
        raise AgentToolError(f"query_ftw rejected the arguments: {exc}") from exc
    except (OSError, ValueError, RuntimeError) as exc:
        raise AgentToolError(f"Published FTW query failed: {exc}") from exc


def query_published_ftw_tool(ctx: ToolContext, inp: FtwQueryInput) -> FtwQueryOutput:
    from agribound._cache import _sha1

    study_area = ctx.resolve_study_area(inp.study_area)
    key = [study_area, inp.year, inp.min_confidence, inp.keep_null_confidence, inp.max_features]
    tag = _sha1(json.dumps(key))[:10]
    out = ctx.workdir / "ftw" / f"ftw_published_{tag}.gpkg"
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf = _query_ftw(
        ctx,
        study_area,
        inp.year,
        inp.min_confidence,
        inp.max_features,
        clip=True,
        keep_null_confidence=inp.keep_null_confidence,
        output_path=out,
    )
    per_year = _year_counts(gdf)
    notes = [
        "Published FTW polygons are predictions of the FTW models, not ground truth.",
        "Polygons are clipped to the study area.",
    ]
    if per_year is not None and len(per_year) > 1:
        notes.append(
            f"The polygons span several prediction years {per_year}: a field predicted in "
            "several years appears once per year, so n_polygons and area_ha count "
            "field-year predictions, not distinct fields. Pass year for a single year."
        )
    return FtwQueryOutput(
        n_polygons=len(gdf),
        polygons_per_year=per_year,
        output_path=str(out) if out.exists() else None,
        area_ha=_area_stats_ha(gdf),
        columns=[str(c) for c in gdf.columns],
        notes=notes,
    )


def evaluate_against_reference_tool(ctx: ToolContext, inp: EvaluateInput) -> EvaluateOutput:
    from agribound.evaluate import evaluate

    reference_path = inp.reference_path or ctx.reference_boundaries
    if not reference_path:
        raise AgentToolError("No reference_path given and the session has no reference layer.")
    predicted = _read_layer(inp.predicted_path)
    reference = _read_layer(reference_path)
    selection = "all reference polygons"
    study_area = inp.study_area or ctx.study_area
    if inp.restrict_to_study_area and study_area:
        reference = _restrict_to_aoi(reference, _read_aoi(ctx, study_area))
        selection = "reference polygons intersecting the study area"
    kwargs: dict[str, Any] = {"iou_threshold": inp.iou_threshold}
    if inp.boundary_tolerance_m is not None:
        kwargs["boundary_tolerance_m"] = inp.boundary_tolerance_m
    try:
        metrics = evaluate(predicted, reference, **kwargs)
    except TypeError as exc:
        raise AgentToolError(f"agribound.evaluate.evaluate rejected the arguments: {exc}") from exc
    from agribound.provenance import to_jsonable

    return EvaluateOutput(
        n_predicted=len(predicted),
        n_reference=len(reference),
        reference_selection=selection,
        metrics=to_jsonable(metrics),
    )


# ---------------------------------------------------------------------------
# Gated tools
# ---------------------------------------------------------------------------


def _resolve_existing_path(value: str | None) -> str | None:
    """Absolute path for an existing file; other strings (bbox:, WKT, asset IDs) unchanged."""
    if not value:
        return value
    try:
        path = Path(value).expanduser()
        exists = path.exists()
    except (OSError, ValueError):  # e.g. a long WKT string exceeds the file-name limit
        return value
    return str(path.resolve()) if exists else value


def plan_network_services(config: Any) -> list[str]:
    """Remote data services the pipeline contacts for *config*.

    Judged from the configuration alone: caches are not inspected, so a plan
    is listed as needing a service even if, for example, its composite is
    already cached. Model-weight downloads (Hugging Face) are not listed.

    Parameters
    ----------
    config : AgriboundConfig
        Validated configuration.

    Returns
    -------
    list of str
        One entry per service and stage, e.g. ``"Earth Engine (composite)"``;
        empty for a ``local`` run without the LULC filter and with a file
        study area.
    """
    services: list[str] = []
    if str(config.study_area or "").startswith(("projects/", "users/")):
        services.append("Earth Engine (study-area asset)")
    if config.is_gee_source():
        services.append("Earth Engine (composite)")
    if config.source == "google-embedding":
        if config.google_embedding_backend == "gee":
            services.append("Earth Engine (Google embeddings)")
        else:
            services.append("Source Cooperative (Google embeddings)")
    if config.source == "tessera-embedding":
        services.append("TESSERA (embedding tiles)")
    if config.source == "usgs-naip-plus":
        from agribound.config import AgriboundConfig

        default_url = AgriboundConfig.__dataclass_fields__["usgs_service_url"].default
        if config.usgs_service_url == default_url:
            services.append("USGS NAIP Plus ImageServer (composite)")
        else:
            from urllib.parse import urlparse

            host = urlparse(str(config.usgs_service_url)).netloc or str(config.usgs_service_url)
            services.append(f"ImageServer at {host} (usgs_service_url; composite)")
    if config.is_gee_source() and config.export_method == "gcs":
        services.append(f"Google Cloud Storage bucket {config.gcs_bucket!r} (batch export)")
    elif config.is_gee_source() and config.export_method == "gdrive":
        services.append("Google Drive of the Earth Engine account (batch export)")
    if config.lulc_filter:
        services.append(f"Earth Engine (LULC filter, lulc_mode={config.lulc_mode!r})")
    return services


def _require_plan_network(ctx: ToolContext, plan: Any) -> None:
    if ctx.allow_network:
        return
    services = plan_network_services(plan.to_config())
    if services:
        raise NetworkDisabledError(
            f"Plan {plan.plan_id} needs network access ({'; '.join(services)}), which is "
            "disabled for this session (allow_network=False / --offline). Nothing was run and "
            "the reviewer was not asked. Report this to the user: they can run the plan YAML "
            f"({plan.yaml_path}) with `agribound delineate --config <yaml>` where these "
            "services are reachable."
        )


def preflight_execution(ctx: ToolContext, plan: Any) -> None:
    """Checks made before the reviewer is asked to approve *plan*.

    None of them involves the reviewer, so a plan that cannot run is refused
    without asking for an approval that could not be used.

    Raises
    ------
    ExecutionDisabledError
        Execution is disabled (dry run) or no gate is configured.
    ExecutionLimitError
        The gate has no executions left.
    PlanChangedError
        The plan no longer matches its hash
        (:func:`agribound.agent.gate.check_plan_current`).
    NetworkDisabledError
        Network access is off and the plan needs a remote service
        (:func:`plan_network_services`).
    """
    from agribound.agent.gate import check_plan_current

    if not ctx.execution_enabled:
        raise ExecutionDisabledError("Execution is disabled in this session (dry run).")
    if ctx.gate is None:
        raise ExecutionDisabledError("No confirmation gate is configured; execution is disabled.")
    ctx.gate.check_limit()
    check_plan_current(plan)
    _require_plan_network(ctx, plan)


def _plan_warnings(config: Any, changes: dict[str, Any], gsd: float | None) -> list[str]:
    from agribound.agent.plans import destination_warnings, method_warnings, threshold_warnings

    warnings = (
        threshold_warnings(changes) + method_warnings(changes) + destination_warnings(changes)
    )
    einfo = ENGINE_REGISTRY[config.engine]
    sinfo = SOURCE_REGISTRY[config.source]
    for note in engine_notes(config.engine, config.source):
        warnings.append(f"{einfo['name']}: {note}")
    if sinfo["restricted"]:
        warnings.append(f"{sinfo['name']} is restricted: {sinfo['coverage']}")
    if config.requires_gee():
        reasons = []
        if config.is_gee_source() or config.source == "google-embedding":
            reasons.append("the composite")
        if config.lulc_filter:
            reasons.append(f"the LULC filter (lulc_mode={config.lulc_mode!r})")
        warnings.append(f"Earth Engine is used for {' and '.join(reasons)}.")
    if config.sam_refine and gsd is not None:
        is_refinable = _load_is_refinable()
        if is_refinable is not None:
            side = _min_refinable_square_side_m(
                gsd, config.sam_min_crop_px, config.sam_crop_padding, is_refinable
            )
            if side is not None:
                warnings.append(
                    f"SAM refinement at {gsd} m refines only fields at least ~{side} m across "
                    f"(square; sam_min_crop_px={config.sam_min_crop_px}, "
                    f"sam_crop_padding={config.sam_crop_padding}); smaller fields keep the "
                    "engine's geometry."
                )
    return warnings


def _plan_cost(config: Any, ctx: ToolContext) -> dict[str, Any]:
    cost: dict[str, Any] = {
        "requires_gee": config.requires_gee(),
        "gpu_recommended": ENGINE_REGISTRY[config.engine]["gpu_recommended"],
        "fine_tune": config.fine_tune,
        "runtime": "not estimated (depends on hardware, Earth Engine load and the engine)",
    }
    if config.study_area and config.source != "local":
        try:
            aoi = _read_aoi(ctx, config.study_area)
            _, utm_bounds = _utm_extent(aoi)
            res = float(config.naip_resolution_m) if config.source == "naip" else None
            est = _composite_estimate(config.source, utm_bounds, config.tile_size, res)
            cost["composite"] = est.model_dump()
        except AgentToolError as exc:
            cost["composite"] = {"note": f"no estimate: {exc}"}
    return cost


def _control_characters(value: Any) -> list[str]:
    """Control or format characters in *value* (recursing into dicts and lists)."""
    from agribound.agent.plans import _UNSAFE_CATEGORIES

    if isinstance(value, str):
        return [ch for ch in value if unicodedata.category(ch) in _UNSAFE_CATEGORIES]
    if isinstance(value, dict):
        return [
            ch
            for k, v in value.items()
            for ch in (*_control_characters(k), *_control_characters(v))
        ]
    if isinstance(value, list | tuple):
        return [ch for item in value for ch in _control_characters(item)]
    return []


def _reject_control_characters(inp: ProposeRunInput) -> None:
    """Refuse control characters in every proposal value shown as a single line.

    Paths, names and configuration values appear in the review screen, the
    plan YAML and the report; a line break or terminal escape sequence in one
    of them could add or hide lines there. The free-text rationale,
    limitations and alternatives may span lines and are escaped when shown.
    """
    checked = {
        "study_area": inp.study_area,
        "reference_boundaries": inp.reference_boundaries,
        "local_tif_path": inp.local_tif_path,
        "output_name": inp.output_name,
        "config": inp.config,
    }
    bad = sorted(name for name, value in checked.items() if _control_characters(value))
    if bad:
        raise AgentToolError(
            f"{', '.join(bad)} must not contain control or format characters (line breaks, "
            "tabs, escape sequences, bidirectional overrides)."
        )


def propose_run_tool(ctx: ToolContext, inp: ProposeRunInput) -> ProposeRunOutput:
    from agribound.agent.plans import make_plan, non_default_fields, write_plan_yaml
    from agribound.config import AgriboundConfig

    _reject_control_characters(inp)
    extra = dict(inp.config)
    top_level = set(ProposeRunInput.model_fields) - {"config", "output_name"}
    duplicated = sorted(set(extra) & top_level)
    if duplicated:
        raise AgentToolError(f"Pass {duplicated} as top-level arguments, not inside 'config'.")
    reserved = sorted(set(extra) & set(PROPOSAL_RESERVED_FIELDS))
    if reserved:
        details = "; ".join(f"{k}: {PROPOSAL_RESERVED_FIELDS[k]}" for k in reserved)
        raise AgentToolError(f"These fields cannot be set by a proposal: {details}.")
    unknown = sorted(set(extra) - set(AgriboundConfig.field_names()))
    if unknown:
        raise AgentToolError(
            f"Unknown configuration field(s) {unknown}. Valid fields: "
            f"{sorted(set(AgriboundConfig.field_names()) - set(PROPOSAL_RESERVED_FIELDS))}"
        )

    study_area = inp.study_area or ctx.study_area or ""
    if inp.source != "local" and not study_area:
        ctx.resolve_study_area(None)  # raises with instructions
    output_name = inp.output_name or f"fields_{inp.source}_{inp.year}_{inp.engine}.gpkg"
    if Path(output_name).name != output_name or output_name in ("", ".", ".."):
        raise AgentToolError("output_name must be a plain file name without directories.")
    if Path(output_name).suffix.lower() not in _EXTENSIONS:
        raise AgentToolError(f"output_name must end with one of {_EXTENSIONS}.")

    fields: dict[str, Any] = {
        **extra,
        "source": inp.source,
        "engine": inp.engine,
        "year": inp.year,
        "study_area": _resolve_existing_path(study_area),
        "reference_boundaries": _resolve_existing_path(
            inp.reference_boundaries or (ctx.reference_boundaries if inp.fine_tune else None)
        ),
        "fine_tune": inp.fine_tune,
        "local_tif_path": _resolve_existing_path(inp.local_tif_path),
        "cache_dir": str(ctx.cache_dir),
        "overwrite": False,
        "provenance": True,
    }
    if ctx.gee_project:
        fields["gee_project"] = ctx.gee_project
    if ctx.gee_service_account_key:
        fields["gee_service_account_key"] = ctx.gee_service_account_key
    # Placeholder output path: the final one needs the plan ID (content hash).
    fields["output_path"] = str(ctx.workdir / "plans" / "pending" / output_name)
    try:
        draft = AgriboundConfig.from_dict(fields)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        raise AgentToolError(f"Invalid configuration: {exc}") from exc

    # The run directory is named after a hash of the configuration (without its
    # output path) and of the input fingerprints: the same proposal on the same
    # inputs maps to the same directory, while a changed study-area geometry or
    # input file gets a new one (so a stale output is never reused).
    from agribound.agent.plans import canonical_json, compute_plan_hash, input_fingerprints

    key_config = draft.to_dict()
    key_config.pop("output_path")
    try:
        inputs_json = canonical_json(input_fingerprints(draft))
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise AgentToolError(f"Could not fingerprint the plan inputs: {exc}") from exc
    run_key = compute_plan_hash(canonical_json(key_config), inputs_json)[:10]
    plan_dir = ctx.workdir / "plans" / f"{inp.source}_{inp.year}_{inp.engine}_{run_key}"
    try:
        config = draft.merged(output_path=str(plan_dir / output_name))
    except (ValueError, TypeError) as exc:
        raise AgentToolError(f"Invalid configuration: {exc}") from exc

    changes = non_default_fields(config.to_dict())
    try:
        plan = make_plan(
            config,
            rationale=inp.rationale,
            limitations=inp.limitations,
            alternatives=inp.alternatives,
            warnings=_plan_warnings(config, changes, _export_resolution(config.source)),
            estimated_cost=_plan_cost(config, ctx),
            yaml_path=str(plan_dir / f"{output_name.rsplit('.', 1)[0]}.plan.yaml"),
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        raise AgentToolError(f"Could not fingerprint the plan inputs: {exc}") from exc
    existing = ctx.plans.get(plan.plan_id)
    if existing is not None and existing.plan_hash != plan.plan_hash:
        # plan_id holds only 12 hex characters of the hash: never let a different
        # plan take over an ID the reviewer may already have been shown.
        raise AgentToolError(
            f"Plan ID {plan.plan_id} is already used by a different plan in this session "
            f"(sha256 {existing.plan_hash}); change the proposal (e.g. output_name) and "
            "propose it again."
        )
    write_plan_yaml(plan, plan.yaml_path)
    ctx.plans[plan.plan_id] = plan
    if ctx.session is not None:
        ctx.session.record_plan(plan)
    services = plan_network_services(config)
    if not ctx.execution_enabled:
        next_step = (
            "Execution is disabled in this session (dry run). The human can run: agribound "
            f"delineate --config {plan.yaml_path}"
        )
    elif not ctx.allow_network and services:
        next_step = (
            "Network access is disabled in this session and this plan needs "
            f"{'; '.join(services)}, so execute_plan will refuse it. Report the plan; the "
            f"human can run it where these services are reachable: agribound delineate "
            f"--config {plan.yaml_path}"
        )
    else:
        next_step = (
            f"Call execute_plan with plan_id={plan.plan_id!r} to ask the human reviewer for "
            "approval. The run starts only if they approve this exact plan."
        )
    return ProposeRunOutput(
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        yaml_path=str(plan.yaml_path),
        output_path=str(config.output_path),
        config=plan.config,
        warnings=list(plan.warnings),
        non_default_fields=changes,
        estimated_cost=plan.estimated_cost,
        network_services=services,
        execution_enabled=ctx.execution_enabled,
        next_step=next_step,
    )


def _write_session(ctx: ToolContext) -> None:
    """Snapshot the gate into the transcript and write it (a write failure only logs)."""
    session = ctx.session
    if session is None:
        return
    session.set_gate(ctx.gate)
    try:
        session.write()
    except OSError as exc:
        logger.warning("Could not write the agent transcript %s: %s", session.path, exc)


def run_approved_plan(ctx: ToolContext, plan: Any) -> ExecutionSummary:
    """Consume the approval for *plan* and run the pipeline.

    The approval is consumed (and the execution counted) before the pipeline
    starts, so a failed run still uses up the session's execution. The
    transcript (if the context has a session) is written right after that,
    with the approval and a ``"running"`` execution record, and again when
    the run ends.

    Raises
    ------
    ExecutionDisabledError, NetworkDisabledError
        If execution is disabled or the plan needs network access that is
        disabled (checked before the approval is consumed).
    GateError
        If the gate refuses (see :meth:`ConfirmationGate.authorize_execution`).
    AgentToolError
        If the pipeline fails.
    """
    if not ctx.execution_enabled:
        raise ExecutionDisabledError("Execution is disabled in this session (dry run).")
    if ctx.gate is None:
        raise ExecutionDisabledError("No confirmation gate is configured; execution is disabled.")
    _require_plan_network(ctx, plan)
    approval = ctx.gate.authorize_execution(plan)
    record: dict[str, Any] = {
        "attempt": len(ctx.executions) + 1,
        "plan_id": plan.plan_id,
        "plan_hash": plan.plan_hash,
        "approval": approval.to_dict(),
        "started_utc": approval.used_utc,
        "status": "running",
    }
    ctx.executions.append(record)
    if ctx.session is not None:
        ctx.session.record_execution(record)
        _write_session(ctx)
    t0 = time.perf_counter()
    try:
        from agribound.pipeline import delineate

        gdf = delineate(config=plan.to_config())
    except Exception as exc:
        record.update(
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            wall_s=round(time.perf_counter() - t0, 3),
        )
        if ctx.session is not None:
            ctx.session.record_execution(record)
        logger.warning("Execution of %s failed", plan.plan_id, exc_info=True)
        raise AgentToolError(
            f"Execution of plan {plan.plan_id} failed: {type(exc).__name__}: {exc}"
        ) from exc
    wall = round(time.perf_counter() - t0, 3)
    summary = _execution_summary(plan, gdf, approval, wall)
    record.update(status="success", summary=summary.model_dump(mode="json"), wall_s=wall)
    if ctx.session is not None:
        ctx.session.record_execution(record)
    return summary


_STAGE_COUNT_FACTS = ("n_detected", "n_postprocessed", "n_after_lulc", "n_output")


def _execution_summary(plan: Any, gdf: Any, approval: Any, wall_s: float) -> ExecutionSummary:
    """Summarise a pipeline result.

    Statistics come from ``gdf.attrs`` and, where the attrs lack them (for
    example when the pipeline reused an existing output), from the facts of
    the output's provenance record.
    """
    from agribound.provenance import read_provenance, to_jsonable

    config = plan.config
    area = None
    if len(gdf) > 0 and "metrics:area" in gdf.columns:
        values = gdf["metrics:area"].to_numpy(dtype=float) / 1e4
        area = _quantiles(values) or {}
        area["total"] = float(values.sum())
    record = read_provenance(config["output_path"]) or {}
    facts = record.get("facts") or {}

    def attr_or_fact(attr: str, fact: str) -> Any:
        value = gdf.attrs.get(attr)
        return to_jsonable(value if value is not None else facts.get(fact))

    counts = {k: int(facts[k]) for k in _STAGE_COUNT_FACTS if isinstance(facts.get(k), int)}
    return ExecutionSummary(
        plan_id=plan.plan_id,
        plan_hash=plan.plan_hash,
        status="success",
        run_id=gdf.attrs.get("run_id"),
        n_polygons=len(gdf),
        stage_counts=counts,
        area_ha=area,
        lulc_status=facts.get("lulc_status"),
        lulc_stats=attr_or_fact("lulc_stats", "lulc_stats"),
        sam_stats=attr_or_fact("sam_stats", "sam_stats"),
        evaluation=attr_or_fact("evaluation_metrics", "evaluation"),
        engine_meta=to_jsonable(gdf.attrs.get("engine_meta") or record.get("engine_meta")),
        reused_existing_output=bool(gdf.attrs.get("reused", False)),
        output_path=str(config["output_path"]),
        provenance_path=gdf.attrs.get("provenance_path"),
        provenance_warnings=[str(w) for w in record.get("warnings", []) or []],
        wall_s=wall_s,
        approval=approval.to_dict(),
    )


def execute_plan_tool(ctx: ToolContext, inp: ExecutePlanInput) -> ExecutionSummary:
    if not ctx.execution_enabled:
        raise ExecutionDisabledError("Execution is disabled in this session (dry run).")
    if ctx.gate is None:
        raise ExecutionDisabledError("No confirmation gate is configured; execution is disabled.")
    plan = ctx.get_plan(inp.plan_id)
    preflight_execution(ctx, plan)  # refuse without asking the reviewer when it cannot run
    ctx.gate.request_approval(plan)
    return run_approved_plan(ctx, plan)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """A tool: name, description, typed input/output and implementation.

    ``read_only`` and ``open_world`` become the MCP ``read_only_hint`` and
    ``open_world_hint`` annotations. ``open_world`` is *True* for every tool
    that may contact a remote service, including the tools that read a study
    area (a GEE-asset study area is read through Earth Engine).
    """

    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    func: Callable[[ToolContext, Any], BaseModel]
    read_only: bool = True
    open_world: bool = False

    def input_schema(self) -> dict[str, Any]:
        """JSON Schema of the input model."""
        return self.input_model.model_json_schema()

    def definition(self) -> Any:
        from agribound.agent.backends.base import ToolDefinition

        return ToolDefinition(
            name=self.name, description=self.description, input_schema=self.input_schema()
        )


@dataclass
class ToolOutcome:
    """Result of :meth:`ToolRegistry.call`."""

    name: str
    ok: bool
    output: dict[str, Any] | None = None
    error: str | None = None
    arguments: Any = None
    arguments_valid: bool = False

    def content(self) -> str:
        """Text returned to the model: JSON output, or the error message."""
        if self.ok:
            return json.dumps(self.output, indent=None, sort_keys=False)
        return str(self.error)


def format_validation_error(tool: str, exc: ValidationError) -> str:
    """One-line description of a pydantic validation error (input values omitted)."""
    parts = []
    for err in exc.errors(include_url=False, include_input=False, include_context=False):
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(arguments)"
        parts.append(f"{loc}: {err.get('msg')}")
    return f"Invalid arguments for {tool}: " + "; ".join(parts)


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "list_sources",
        "List every imagery and embedding source Agribound supports, with export and native "
        "resolution, available years, coverage, pixel value scale, and whether Earth Engine or "
        "restricted access is needed. Use it before choosing a source.",
        EmptyInput,
        ListSourcesOutput,
        list_sources_tool,
    ),
    ToolSpec(
        "list_engines",
        "List the delineation engines with their approach, whether they run without labels "
        "(label_free), whether they can be fine-tuned, supported sources, required bands, "
        "references, caveats (notes; source_notes for caveats that apply to one source only) "
        "and whether their Python package is installed.",
        EmptyInput,
        ListEnginesOutput,
        list_engines_tool,
    ),
    ToolSpec(
        "describe_study_area",
        "Describe a study area: area in km^2, bounding box, centroid, UTM zone(s), and the "
        "estimated size of the composite each source would produce (pixels, bands, "
        "uncompressed MB, download tiles).",
        StudyAreaInput,
        StudyAreaDescription,
        describe_study_area_tool,
        open_world=True,
    ),
    ToolSpec(
        "check_availability",
        "Check whether sources have data for a year: registry year ranges and coverage notes, "
        "and optionally (live=True) Earth Engine image counts or TESSERA tile counts over the "
        "study area's bounding box. Live checks need network access and credentials.",
        AvailabilityInput,
        AvailabilityOutput,
        check_availability_tool,
        open_world=True,
    ),
    ToolSpec(
        "estimate_resolvability",
        "Estimate how well each sensor resolves the fields: the distribution of pixels per "
        "field (area / GSD^2) per source, and the fraction of fields (by count and area) that "
        "the SAM refinement stage would refine. Field sizes come from exactly one of: a "
        "reference layer, published FTW polygons (network; one prediction year, the latest "
        "unless year is given), or a representative field area.",
        ResolvabilityInput,
        ResolvabilityOutput,
        estimate_resolvability_tool,
        open_world=True,
    ),
    ToolSpec(
        "recommend_configurations",
        "Rank candidate (source, engine) configurations with deterministic, documented rules: "
        "year availability, access restrictions, US-only coverage, engine-source support, "
        "label availability (label-free vs fine-tuning) and export resolution. Every rule and "
        "threshold is listed in the output. It does not predict accuracy.",
        RecommendInput,
        RecommendOutput,
        recommend_configurations_tool,
        open_world=True,
    ),
    ToolSpec(
        "query_published_ftw",
        "Download the published Fields of The World prediction polygons for the study area "
        "(network) into the session work directory and summarise them. These are model "
        "predictions, not ground truth.",
        FtwQueryInput,
        FtwQueryOutput,
        query_published_ftw_tool,
        read_only=False,
        open_world=True,
    ),
    ToolSpec(
        "evaluate_against_reference",
        "Evaluate a predicted boundary layer against reference boundaries with "
        "agribound.evaluate.evaluate (IoU matching, precision, recall, F1, and more). Read-only.",
        EvaluateInput,
        EvaluateOutput,
        evaluate_against_reference_tool,
        open_world=True,
    ),
    ToolSpec(
        "propose_run",
        "Validate an Agribound configuration and freeze it into a plan (plan_id + sha256 "
        "hash), write its YAML, and report warnings, fields that differ from the package "
        "defaults, and a size estimate. It does NOT run anything. Include your rationale, the "
        "limitations found with the other tools, and alternatives; the human reviewer sees "
        "them.",
        ProposeRunInput,
        ProposeRunOutput,
        propose_run_tool,
        read_only=False,
        open_world=True,
    ),
    ToolSpec(
        "execute_plan",
        "Ask the human reviewer to approve a plan and, only if they approve that exact plan, "
        "run the Agribound pipeline for it. At most one plan runs per session; the session "
        "ends after the run (or after a denial).",
        ExecutePlanInput,
        ExecutionSummary,
        execute_plan_tool,
        read_only=False,
        open_world=True,
    ),
)
"""All tools, in the order they are offered to the model."""


class ToolRegistry:
    """Validating dispatcher over :data:`TOOL_SPECS` for one :class:`ToolContext`.

    Parameters
    ----------
    context : ToolContext
        Shared session state.
    include_execute : bool or None
        Offer ``execute_plan``; default ``context.execution_enabled``.
    """

    def __init__(self, context: ToolContext, *, include_execute: bool | None = None) -> None:
        self.context = context
        include = context.execution_enabled if include_execute is None else include_execute
        self.specs: dict[str, ToolSpec] = {
            s.name: s for s in TOOL_SPECS if include or s.name != "execute_plan"
        }

    def definitions(self) -> list[Any]:
        """Tool definitions for a backend, in a stable order."""
        return [spec.definition() for spec in self.specs.values()]

    def call(self, name: str, arguments: Any) -> ToolOutcome:
        """Validate *arguments* and run tool *name*; never raises for tool failures."""
        spec = self.specs.get(name)
        if spec is None:
            return ToolOutcome(
                name=name,
                ok=False,
                error=f"Unknown tool {name!r}. Available: {list(self.specs)}",
                arguments=arguments,
            )
        if not isinstance(arguments, dict):
            return ToolOutcome(
                name=name,
                ok=False,
                error=f"Arguments for {name} must be a JSON object, got {type(arguments).__name__}",
                arguments=arguments,
            )
        try:
            validated = spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            return ToolOutcome(
                name=name, ok=False, error=format_validation_error(name, exc), arguments=arguments
            )
        dumped = validated.model_dump(mode="json")
        try:
            result = spec.func(self.context, validated)
        except AgentToolError as exc:
            return ToolOutcome(
                name=name, ok=False, error=str(exc), arguments=dumped, arguments_valid=True
            )
        except Exception as exc:
            logger.warning("Tool %s raised an unexpected error", name, exc_info=True)
            return ToolOutcome(
                name=name,
                ok=False,
                error=f"Unexpected error in {name}: {type(exc).__name__}: {exc}",
                arguments=dumped,
                arguments_valid=True,
            )
        return ToolOutcome(
            name=name,
            ok=True,
            output=result.model_dump(mode="json"),
            arguments=dumped,
            arguments_valid=True,
        )


__all__ = [
    "ENGINE_MODULES",
    "PROPOSAL_RESERVED_FIELDS",
    "RECOMMENDATION_PARAMETERS",
    "TOOL_SPECS",
    "ToolContext",
    "ToolOutcome",
    "ToolRegistry",
    "ToolSpec",
    "format_validation_error",
    "plan_network_services",
    "preflight_execution",
    "run_approved_plan",
]
