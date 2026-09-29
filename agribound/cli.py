"""
Command-line interface for Agribound.

Provides the ``agribound`` CLI:

- ``delineate``: run the full pipeline (composite, optional fine-tuning,
  delineation, post-processing, export).
- ``composite``: run only stage A (composite / embedding download, plus the
  LULC raster prefetch with ``--lulc-mode raster``).
- ``prefetch``: download an engine's weights (and the SAM refinement weights
  when ``sam_refine`` is set) ahead of an offline run.
- ``evaluate``: score predicted boundaries against reference boundaries.
- ``query-ftw``: query published FTW prediction polygons.
- ``list-engines``, ``list-sources``, ``list-ftw-models``: registry listings.
- ``auth``: authenticate with Google Earth Engine.

``delineate``, ``composite`` and ``prefetch`` accept ``--config run.yaml``.
Flags given explicitly on the command line override the YAML values; flags that
are not given keep the YAML value (or the :class:`~agribound.config.AgriboundConfig`
default when the YAML does not set it). ``--dry-run`` prints the resolved
configuration as YAML and exits without running anything.

Optional command groups (``tiles`` from :mod:`agribound.hpc.cli`, ``agent`` and
``mcp`` from :mod:`agribound.agent.cli`) are registered when those modules exist.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import sys
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

import click
from click.core import ParameterSource

from agribound._version import __version__
from agribound.config import (
    VALID_AOI_SELECTIONS,
    VALID_ENGINES,
    VALID_SOURCES,
    AgriboundConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Parameter sources that count as "explicitly given" and override --config values.
_EXPLICIT_SOURCES = frozenset({ParameterSource.COMMANDLINE, ParameterSource.ENVIRONMENT})

#: click parameter name -> AgriboundConfig field.
_PARAM_TO_FIELD: dict[str, str] = {
    "study_area": "study_area",
    "source": "source",
    "year": "year",
    "engine": "engine",
    "output": "output_path",
    "output_format": "output_format",
    "gee_project": "gee_project",
    "export_method": "export_method",
    "gcs_bucket": "gcs_bucket",
    "composite_method": "composite_method",
    "date_range": "date_range",
    "cloud_cover_max": "cloud_cover_max",
    "s2_cloud_mask": "s2_cloud_mask",
    "naip_resolution": "naip_resolution_m",
    "export_crs": "export_crs",
    "tessera_version": "tessera_version",
    "embedding_cache_dir": "embedding_cache_dir",
    "local_tif": "local_tif_path",
    "usgs_state": "usgs_state",
    "tile_size": "tile_size",
    "lulc_filter": "lulc_filter",
    "lulc_threshold": "lulc_crop_threshold",
    "lulc_dataset": "lulc_dataset",
    "lulc_mode": "lulc_mode",
    "lulc_on_error": "lulc_on_error",
    "seed": "seed",
    "cache_dir": "cache_dir",
    "gee_service_account_key": "gee_service_account_key",
    "gee_high_volume": "gee_high_volume",
    "gee_max_requests": "gee_max_requests",
    "gee_workload_tag": "gee_workload_tag",
    "engine_param": "engine_params",
    "aoi_selection": "aoi_selection",
    "min_area": "min_field_area_m2",
    "simplify": "simplify_tolerance",
    "device": "device",
    "n_workers": "n_workers",
    "reference": "reference_boundaries",
    "fine_tune": "fine_tune",
    "fine_tune_epochs": "fine_tune_epochs",
    "fine_tune_split": "fine_tune_split",
    "fine_tune_block_size": "fine_tune_block_size_m",
    "fine_tune_split_column": "fine_tune_split_column",
    "sam_refine": "sam_refine",
    "sam_backend": "sam_backend",
    "sam_model": "sam_model",
    "overwrite": "overwrite",
    "provenance": "provenance",
}

_SAM_BACKENDS = ("sam2", "sam2.1", "sam3", "sam3-hf")
try:  # single source of truth when the registry is available
    from agribound.registry import sam_refine_backends as _registry_sam_backends

    _SAM_BACKENDS = tuple(_registry_sam_backends)
except ImportError:  # pragma: no cover - registry is part of the package
    pass


def _config_default(name: str) -> Any:
    """Return the :class:`AgriboundConfig` default for field *name* (or ``MISSING``)."""
    for f in fields(AgriboundConfig):
        if f.name == name:
            if f.default is not MISSING:
                return f.default
            if f.default_factory is not MISSING:
                return f.default_factory()
    return MISSING


def _d(name: str) -> str:
    """Help-text suffix showing the configuration default of field *name*."""
    value = _config_default(name)
    if value is MISSING:
        return ""
    return f" [default: {value}]"


def _parse_engine_params(
    ctx: click.Context, param: click.Parameter, values: tuple[str, ...]
) -> dict[str, Any]:
    """Parse repeated ``KEY=VALUE`` options into a dict (VALUE as JSON, else string)."""
    parsed: dict[str, Any] = {}
    for item in values:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise click.BadParameter(f"expected KEY=VALUE, got {item!r}", ctx=ctx, param=param)
        try:
            parsed[key] = json.loads(raw)
        except ValueError:
            parsed[key] = raw
    return parsed


def _load_yaml_mapping(path: str) -> dict[str, Any]:
    """Load a YAML config file as a plain dict."""
    import yaml

    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise click.UsageError(f"Could not parse --config {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise click.UsageError(
            f"--config {path} must contain a YAML mapping of AgriboundConfig fields, "
            f"got {type(data).__name__}."
        )
    return data


def _default_output_path(data: dict[str, Any]) -> str:
    """``fields_<source>_<year>.<format>`` from the resolved values (or config defaults)."""
    source = str(data.get("source", _config_default("source"))).lower().strip()
    year = data.get("year", _config_default("year"))
    fmt = str(data.get("output_format", _config_default("output_format"))).lower().strip()
    return f"fields_{source}_{year}.{fmt}"


def _resolve_config_data(ctx: click.Context, *, require_study_area: bool) -> dict[str, Any]:
    """Merge ``--config`` YAML values with explicitly passed CLI flags."""
    params = ctx.params
    config_file = params.get("config_file")
    data = _load_yaml_mapping(config_file) if config_file else {}

    for name, field_name in _PARAM_TO_FIELD.items():
        if name not in params or ctx.get_parameter_source(name) not in _EXPLICIT_SOURCES:
            continue
        value = params[name]
        if name == "engine_param":
            engine_params = dict(data.get("engine_params") or {})
            engine_params.update(value)
            data["engine_params"] = engine_params
        elif name == "date_range":
            data["date_range"] = tuple(value)
        else:
            data[field_name] = value

    source = str(data.get("source", _config_default("source"))).lower().strip()
    if require_study_area and not data.get("study_area") and source != "local":
        raise click.UsageError(
            "Missing option '--study-area' (required unless the --config YAML sets "
            "study_area, or --source is local).",
            ctx=ctx,
        )
    if not data.get("output_path"):
        data["output_path"] = _default_output_path(data)
    return data


def _build_config(data: dict[str, Any], ctx: click.Context) -> AgriboundConfig:
    """Validate *data* into an :class:`AgriboundConfig`, reporting errors as usage errors."""
    try:
        return AgriboundConfig.from_dict(dict(data))
    except (ValueError, TypeError, FileNotFoundError) as exc:
        raise click.UsageError(f"Invalid configuration: {exc}", ctx=ctx) from exc


def _plain(obj: Any) -> Any:
    """Convert tuples/Paths (recursively) so :func:`yaml.safe_dump` can serialise them."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _config_yaml(config: AgriboundConfig) -> str:
    """Serialise *config* as YAML that ``--config`` can read back."""
    import yaml

    return yaml.safe_dump(_plain(config.to_dict()), sort_keys=False, default_flow_style=False)


def _jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays, tuples and non-finite floats for strict JSON."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, str | bytes):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj
    return str(obj)


def _apply(options: list) -> Any:
    """Apply a list of click option decorators in display order."""

    def decorator(func: Any) -> Any:
        for option in reversed(options):
            func = option(func)
        return func

    return decorator


# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

_CONFIG_OPTION = click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "YAML file of AgriboundConfig fields. Options given explicitly on the command "
        "line override its values."
    ),
)

_DRY_RUN_OPTION = click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the resolved configuration as YAML and exit without running anything.",
)

_ENGINE_PARAM_OPTION = click.option(
    "--engine-param",
    "engine_param",
    multiple=True,
    metavar="KEY=VALUE",
    callback=_parse_engine_params,
    help=(
        "Engine parameter (repeatable). VALUE is parsed as JSON (numbers, true/false, "
        "null, lists, objects) and otherwise kept as a string. Merged into the "
        "engine_params of --config; keys given here win."
    ),
)

#: Stage-A options: everything that determines the composite / embedding raster.
_STAGE_A_OPTIONS = [
    click.option(
        "--study-area",
        default=None,
        help=(
            "Area of interest: vector file (GeoJSON/GPKG/Shapefile/GeoParquet), GEE "
            "asset ID, 'bbox:minx,miny,maxx,maxy' (EPSG:4326) or WKT. Required "
            "unless --config sets study_area; optional with --source local (the "
            "whole raster is used)."
        ),
    ),
    click.option(
        "--source",
        default=None,
        type=click.Choice(VALID_SOURCES, case_sensitive=False),
        help=f"Imagery or embedding source.{_d('source')}",
    ),
    click.option("--year", default=None, type=int, help=f"Target year.{_d('year')}"),
    click.option(
        "--engine",
        default=None,
        type=click.Choice(VALID_ENGINES, case_sensitive=False),
        help=f"Delineation engine (validated against --source).{_d('engine')}",
    ),
    click.option(
        "--output",
        "-o",
        default=None,
        help=(
            "Output vector path. Intermediates are cached in <output dir>/.agribound_cache "
            "unless --cache-dir is given. [default: fields_<source>_<year>.<format>]"
        ),
    ),
    click.option(
        "--output-format",
        default=None,
        type=click.Choice(["gpkg", "geojson", "parquet"]),
        help=f"Output vector format.{_d('output_format')}",
    ),
    click.option("--gee-project", default=None, help="Google Earth Engine project ID."),
    click.option(
        "--export-method",
        default=None,
        type=click.Choice(["local", "gdrive", "gcs"]),
        help=f"GEE export method.{_d('export_method')}",
    ),
    click.option("--gcs-bucket", default=None, help="GCS bucket (required with gcs export)."),
    click.option(
        "--composite-method",
        default=None,
        type=click.Choice(["median", "greenest", "max_ndvi"]),
        help=f"Compositing method.{_d('composite_method')}",
    ),
    click.option(
        "--date-range",
        nargs=2,
        default=None,
        metavar="START END",
        help="Composite date window (YYYY-MM-DD YYYY-MM-DD) instead of the calendar year.",
    ),
    click.option(
        "--cloud-cover-max",
        default=None,
        type=int,
        help=f"Maximum scene cloud cover in percent (GEE scene filter).{_d('cloud_cover_max')}",
    ),
    click.option(
        "--s2-cloud-mask",
        default=None,
        type=click.Choice(["scl", "cloud_score_plus"]),
        help=f"Sentinel-2 pixel cloud mask.{_d('s2_cloud_mask')}",
    ),
    click.option(
        "--naip-resolution",
        default=None,
        type=float,
        help=f"NAIP export resolution in meters.{_d('naip_resolution_m')}",
    ),
    click.option(
        "--export-crs",
        default=None,
        help=(
            "Export CRS: 'utm' (UTM zone of the AOI centroid) or an EPSG code such as "
            f"EPSG:5070.{_d('export_crs')}"
        ),
    ),
    click.option(
        "--tessera-version",
        default=None,
        type=click.Choice(["v1", "v1.1", "v2"]),
        help=f"TESSERA embedding dataset version.{_d('tessera_version')}",
    ),
    click.option(
        "--embedding-cache-dir",
        default=None,
        help=(
            "Cache directory of the embedding readers (e.g. shared HPC scratch): geotessera's "
            "TESSERA Zarr read cache, and the Google Satellite Embedding tile index "
            "(aef_index.parquet, about 78 MB) of google_embedding_backend='source_coop'. "
            "[default: TESSERA reads are not cached on disk; the tile index goes to "
            "~/.cache/agribound]"
        ),
    ),
    click.option("--local-tif", default=None, help="Local GeoTIFF (required with --source local)."),
    click.option(
        "--usgs-state",
        default=None,
        help="Two-letter state code filter for USGS NAIP Plus scenes (e.g. NM).",
    ),
    click.option(
        "--tile-size",
        default=None,
        type=int,
        help=f"Maximum tile size in pixels for chunked composite downloads.{_d('tile_size')}",
    ),
    click.option(
        "--lulc-filter/--no-lulc-filter",
        default=None,
        help=f"Keep only polygons on cropland according to a LULC dataset.{_d('lulc_filter')}",
    ),
    click.option(
        "--lulc-threshold",
        default=None,
        type=float,
        help=(
            "Minimum cropland fraction (0-1) of a polygon for the LULC filter to keep it."
            f"{_d('lulc_crop_threshold')}"
        ),
    ),
    click.option(
        "--lulc-dataset",
        default=None,
        type=click.Choice(["auto", "nlcd", "cdl", "dynamic_world", "c3s"]),
        help=f"LULC dataset for the crop filter.{_d('lulc_dataset')}",
    ),
    click.option(
        "--lulc-mode",
        default=None,
        type=click.Choice(["server", "raster"]),
        help=(
            "'server': GEE reduceRegions; 'raster': prefetch the LULC raster during the "
            f"composite stage and compute zonal statistics locally.{_d('lulc_mode')}"
        ),
    ),
    click.option(
        "--lulc-on-error",
        default=None,
        type=click.Choice(["raise", "warn"]),
        help=f"What to do when the LULC filter fails.{_d('lulc_on_error')}",
    ),
    click.option("--seed", default=None, type=int, help=f"Random seed.{_d('seed')}"),
    click.option(
        "--cache-dir",
        default=None,
        help="Cache directory for intermediates [default: <output dir>/.agribound_cache].",
    ),
    click.option(
        "--gee-service-account-key",
        default=None,
        help="Path to a GEE service-account JSON key (non-interactive authentication).",
    ),
    click.option(
        "--gee-high-volume/--no-gee-high-volume",
        default=None,
        help=f"Use the Earth Engine high-volume endpoint.{_d('gee_high_volume')}",
    ),
    click.option(
        "--gee-max-requests",
        default=None,
        type=int,
        help=f"Maximum concurrent Earth Engine download requests.{_d('gee_max_requests')}",
    ),
    click.option(
        "--gee-workload-tag",
        default=None,
        help="Earth Engine workload tag for usage attribution.",
    ),
    _CONFIG_OPTION,
    _DRY_RUN_OPTION,
]

#: Options used only after stage A (engine, fine-tuning, refinement, post-processing).
_DELINEATE_OPTIONS = [
    _ENGINE_PARAM_OPTION,
    click.option(
        "--aoi-selection",
        default=None,
        type=click.Choice(VALID_AOI_SELECTIONS),
        help=(
            "How predictions are restricted to the study-area geometry (composites cover its "
            "bounding box): keep polygons whose representative point is inside, polygons "
            "that intersect it, clip them to it, or keep all (none)."
            f"{_d('aoi_selection')}"
        ),
    ),
    click.option(
        "--min-area",
        default=None,
        type=float,
        help=f"Minimum field area in m^2.{_d('min_field_area_m2')}",
    ),
    click.option(
        "--simplify",
        default=None,
        type=float,
        help=(
            "Douglas-Peucker simplification tolerance in meters (0 disables)."
            f"{_d('simplify_tolerance')}"
        ),
    ),
    click.option(
        "--device",
        default=None,
        type=click.Choice(["auto", "cuda", "cpu", "mps"]),
        help=f"Compute device.{_d('device')}",
    ),
    click.option(
        "--n-workers",
        default=None,
        type=int,
        help=f"Data-loader worker processes for the engines.{_d('n_workers')}",
    ),
    click.option(
        "--reference",
        default=None,
        help="Reference boundaries for evaluation, or for training with --fine-tune.",
    ),
    click.option(
        "--fine-tune/--no-fine-tune",
        default=None,
        help=f"Fine-tune the engine on --reference before inference.{_d('fine_tune')}",
    ),
    click.option(
        "--fine-tune-epochs",
        default=None,
        type=int,
        help=f"Fine-tuning epochs.{_d('fine_tune_epochs')}",
    ),
    click.option(
        "--fine-tune-split",
        default=None,
        type=click.Choice(["block", "random", "column"]),
        help=(
            "Train/validation split: spatial blocks, random chips, or groups from "
            f"--fine-tune-split-column.{_d('fine_tune_split')}"
        ),
    ),
    click.option(
        "--fine-tune-block-size",
        default=None,
        type=float,
        help=f"Block edge length in meters for the block split.{_d('fine_tune_block_size_m')}",
    ),
    click.option(
        "--fine-tune-split-column",
        default=None,
        help="Reference-file column used as the group id for the column split.",
    ),
    click.option(
        "--sam-refine/--no-sam-refine",
        default=None,
        help=f"Refine polygons with SAM after delineation.{_d('sam_refine')}",
    ),
    click.option(
        "--sam-backend",
        default=None,
        type=click.Choice(_SAM_BACKENDS),
        help=f"SAM backend for refinement.{_d('sam_backend')}",
    ),
    click.option("--sam-model", default=None, help="SAM model id override for refinement."),
    click.option(
        "--overwrite/--no-overwrite",
        default=None,
        help=f"Re-run even if the output file already exists.{_d('overwrite')}",
    ),
    click.option(
        "--provenance/--no-provenance",
        default=None,
        help=f"Write <output>.provenance.json next to the output.{_d('provenance')}",
    ),
]


# ---------------------------------------------------------------------------
# Command group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version=__version__, prog_name="agribound")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """Agribound: Unified agricultural field boundary delineation toolkit."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# delineate / composite / prefetch
# ---------------------------------------------------------------------------


@main.command()
@_apply(_STAGE_A_OPTIONS + _DELINEATE_OPTIONS)
@click.pass_context
def delineate(ctx: click.Context, dry_run: bool, **_: Any) -> None:
    """Run field boundary delineation.

    With --config, the YAML supplies every value that is not given explicitly on
    the command line (including --study-area).
    """
    data = _resolve_config_data(ctx, require_study_area=True)
    config = _build_config(data, ctx)
    if dry_run:
        click.echo(_config_yaml(config), nl=False)
        return

    from agribound.pipeline import delineate as run_delineate

    try:
        gdf = run_delineate(config=config)
    except FileExistsError as exc:  # output from a different configuration
        raise click.ClickException(str(exc)) from exc
    except ImportError as exc:  # an optional extra is missing; the message says which
        raise _missing_dependency(exc) from exc
    click.echo(f"Delineated {len(gdf)} field boundaries → {config.output_path}")

    metrics = gdf.attrs.get("evaluation_metrics") if hasattr(gdf, "attrs") else None
    if metrics:
        summary = ", ".join(
            f"{key}={metrics[key]:.3f}"
            for key in ("precision", "recall", "f1", "iou_mean")
            if isinstance(metrics.get(key), int | float)
        )
        if summary:
            click.echo(f"Evaluation against --reference: {summary}")

    if config.provenance:
        from agribound.provenance import provenance_path

        record = provenance_path(config.output_path)
        if record.exists():
            click.echo(f"Provenance → {record}")


@main.command()
@_apply(_STAGE_A_OPTIONS)
@click.pass_context
def composite(ctx: click.Context, dry_run: bool, **_: Any) -> None:
    """Build the stage-A raster (composite or embeddings) and print its path.

    Downloaded or computed rasters go to the cache (see --output / --cache-dir),
    and so does a GeoJSON copy of a GEE-asset study area; with --lulc-mode raster
    the LULC raster is prefetched as well. A later 'agribound delineate' with the
    same stage-A options and cache location reuses the cached files.
    """
    data = _resolve_config_data(ctx, require_study_area=True)
    config = _build_config(data, ctx)
    if dry_run:
        click.echo(_config_yaml(config), nl=False)
        return

    from agribound.pipeline import build_composite

    try:
        raster_path = build_composite(config)
    except ImportError as exc:  # an optional extra is missing; the message says which
        raise _missing_dependency(exc) from exc
    click.echo(f"Composite ready → {raster_path}")


def _missing_dependency(exc: ImportError) -> click.ClickException:
    """One-line CLI error for a missing optional dependency (traceback logged at DEBUG).

    The engines' ImportErrors name the extra to install
    (``pip install 'agribound[ftw]'``); the traceback is shown with ``-v``.
    """
    logging.getLogger("agribound.cli").debug("ImportError traceback", exc_info=exc)
    return click.ClickException(str(exc))


def _default_prefetch_source(engine: str) -> tuple[str, dict[str, Any]]:
    """Pick a source for a prefetch-only config that needs no GEE project."""
    from agribound.registry import ENGINE_REGISTRY

    sources = list(ENGINE_REGISTRY[engine]["supported_sources"])
    if "local" in sources:
        # The raster is never read by prefetch; the path only satisfies validation.
        return "local", {"local_tif_path": "prefetch-placeholder.tif"}
    for candidate in ("tessera-embedding", "usgs-naip-plus"):
        if candidate in sources:
            return candidate, {}
    return sources[0], {}


@main.command()
@click.option(
    "--engine",
    default=None,
    type=click.Choice(VALID_ENGINES, case_sensitive=False),
    help="Engine whose weights to download (required unless --config sets engine).",
)
@_ENGINE_PARAM_OPTION
@click.option(
    "--source",
    default=None,
    type=click.Choice(VALID_SOURCES, case_sensitive=False),
    help=(
        "Source the engine will run on (passed to prefetch() in the config). "
        "[default: from --config, else a source the engine supports that needs no GEE "
        "project, e.g. local]"
    ),
)
@click.option(
    "--gee-project",
    default=None,
    help="Google Earth Engine project ID (only needed with a GEE --source).",
)
@click.option(
    "--sam-refine/--no-sam-refine",
    default=None,
    help=f"Also download the SAM refinement weights.{_d('sam_refine')}",
)
@click.option(
    "--sam-backend",
    default=None,
    type=click.Choice(_SAM_BACKENDS),
    help=f"SAM backend whose weights to download with --sam-refine.{_d('sam_backend')}",
)
@click.option("--sam-model", default=None, help="SAM model id override for refinement.")
@_CONFIG_OPTION
@click.pass_context
def prefetch(ctx: click.Context, **_: Any) -> None:
    """Download an engine's weights so later runs can work offline (e.g. HPC nodes).

    Calls the engine's prefetch() and prints the files it reports. When the
    configuration enables sam_refine (--sam-refine or --config), the SAM
    refinement weights of agribound.engines.samgeo_engine.prefetch() are
    downloaded as well; the embedding engine's own prefetch() already covers
    them. Engines that do not implement prefetch() report nothing and
    download their weights on first use.
    """
    # prefetch needs no AOI; "source" is only used for validation / weight choice.
    data = _resolve_config_data(ctx, require_study_area=False)
    engine = str(data.get("engine") or "").lower().strip()
    if not engine:
        raise click.UsageError("Missing option '--engine' (or engine in --config).", ctx=ctx)
    if "source" not in data and engine in VALID_ENGINES:
        source, extra = _default_prefetch_source(engine)
        data["source"] = source
        for key, value in extra.items():
            data.setdefault(key, value)
    config = _build_config(data, ctx)

    from agribound.engines import get_engine

    try:
        engine_obj = get_engine(config.engine)
        paths = list(engine_obj.prefetch(config) or [])
        # The pipeline's SAM stage runs for every engine except "embedding", which
        # refines inside the engine and prefetches the SAM weights itself.
        sam_paths: list[Any] = []
        if config.sam_refine and config.engine != "embedding":
            from agribound.engines.samgeo_engine import prefetch as sam_prefetch

            sam_paths = list(sam_prefetch(config) or [])
    except ImportError as exc:  # an optional extra is missing; the message says which
        raise _missing_dependency(exc) from exc
    for path in paths + sam_paths:
        click.echo(str(path))
    if paths:
        click.echo(f"Prefetched {len(paths)} file(s) for engine {config.engine!r}.")
    else:
        click.echo(f"Engine {config.engine!r} reported no files to prefetch.")
    if sam_paths:
        click.echo(
            f"Prefetched {len(sam_paths)} file(s) for SAM refinement "
            f"(sam_backend={config.sam_backend!r})."
        )


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _parse_size_bins(
    ctx: click.Context, param: click.Parameter, value: str | None
) -> str | list[float] | None:
    """``"auto"`` or comma-separated, strictly increasing, non-negative edges (hectares)."""
    if value is None:
        return None
    if value.strip().lower() == "auto":
        return "auto"
    try:
        bins = [float(v) for v in value.split(",") if v.strip()]
    except ValueError:
        raise click.BadParameter(
            f"expected 'auto' or comma-separated numbers, got {value!r}", ctx=ctx, param=param
        ) from None
    if len(bins) < 2 or any(hi <= lo for lo, hi in zip(bins, bins[1:], strict=False)):
        raise click.BadParameter(
            "expected at least two strictly increasing bin edges", ctx=ctx, param=param
        )
    if bins[0] < 0 or any(math.isnan(b) for b in bins):
        raise click.BadParameter("bin edges must be non-negative numbers", ctx=ctx, param=param)
    return bins


def _parse_spacing(ctx: click.Context, param: click.Parameter, value: str | None) -> Any:
    """A positive float in metres, ``"none"`` (-> None), or unset (-> the sentinel ``...``)."""
    if value is None:
        return ...
    if value.strip().lower() == "none":
        return None
    try:
        spacing = float(value)
    except ValueError:
        raise click.BadParameter(
            f"expected a distance in meters or 'none', got {value!r}", ctx=ctx, param=param
        ) from None
    if not math.isfinite(spacing) or spacing <= 0:
        raise click.BadParameter("the spacing must be a positive number", ctx=ctx, param=param)
    return spacing


@main.command()
@click.option(
    "--predicted",
    "-p",
    required=True,
    type=click.Path(exists=True),
    help="Predicted field boundaries (any vector format agribound reads).",
)
@click.option(
    "--reference",
    "-r",
    required=True,
    type=click.Path(exists=True),
    help="Reference field boundaries.",
)
@click.option(
    "--iou-threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="IoU needed for a predicted polygon to match a reference polygon.",
)
@click.option(
    "--strata-column",
    default=None,
    help="Reference column defining strata; metrics are also reported per stratum.",
)
@click.option(
    "--size-bins",
    default=None,
    callback=_parse_size_bins,
    metavar="auto|EDGES",
    help=(
        "Field-size classes for per-size-class metrics: 'auto' (1-2-5 series spanning the "
        "reference areas) or comma-separated, strictly increasing reference-field area edges "
        "in hectares, e.g. 0,0.5,1,2,5,10,inf."
    ),
)
@click.option(
    "--matching",
    default=None,
    type=click.Choice(["one_to_one", "many_to_one"]),
    help=(
        "Matching of predictions to reference fields: one_to_one (greedy by IoU; each "
        "polygon matched at most once) or many_to_one (the 0.1.x rule; one prediction may "
        "match several reference fields). [default: one_to_one]"
    ),
)
@click.option(
    "--bootstrap",
    default=0,
    show_default=True,
    type=click.IntRange(min=0),
    help="Bootstrap resamples for confidence intervals (0 disables).",
)
@click.option(
    "--bootstrap-seed", default=42, show_default=True, type=int, help="Bootstrap random seed."
)
@click.option(
    "--boundary-tolerance-m",
    default=None,
    type=float,
    help="Distance tolerance in meters for the boundary metrics.",
)
@click.option(
    "--boundary-sample-spacing-m",
    default=None,
    callback=_parse_spacing,
    metavar="FLOAT|none",
    help=(
        "Maximum spacing in meters of the boundary sample points for the Hausdorff and mean "
        "boundary distances; 'none' selects the faster length-dependent spacing. "
        "[default: 1 m]"
    ),
)
@click.option(
    "--equal-area-crs",
    default=None,
    help="Equal-area CRS for area and IoU computations [default: evaluate()'s default].",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help="Write the metrics as JSON to this file [default: print to stdout].",
)
def evaluate(
    predicted: str,
    reference: str,
    iou_threshold: float,
    strata_column: str | None,
    size_bins: str | list[float] | None,
    matching: str | None,
    bootstrap: int,
    bootstrap_seed: int,
    boundary_tolerance_m: float | None,
    boundary_sample_spacing_m: Any,
    equal_area_crs: str | None,
    output: str | None,
) -> None:
    """Evaluate predicted field boundaries against reference boundaries."""
    from agribound.evaluate import evaluate as run_evaluate
    from agribound.io.vector import read_vector

    pred_gdf = read_vector(predicted)
    ref_gdf = read_vector(reference)

    # Only pass options the user set, so evaluate()'s own defaults apply otherwise.
    kwargs: dict[str, Any] = {"iou_threshold": iou_threshold}
    if strata_column is not None:
        if strata_column not in ref_gdf.columns:
            raise click.BadParameter(
                f"column {strata_column!r} not found in {reference} "
                f"(columns: {', '.join(map(str, ref_gdf.columns))})",
                param_hint="'--strata-column'",
            )
        kwargs["strata"] = strata_column
    if size_bins is not None:
        kwargs["size_bins"] = size_bins
    if matching is not None:
        kwargs["matching"] = matching
    if bootstrap:
        kwargs["bootstrap"] = bootstrap
        kwargs["bootstrap_seed"] = bootstrap_seed
    if boundary_tolerance_m is not None:
        kwargs["boundary_tolerance_m"] = boundary_tolerance_m
    if boundary_sample_spacing_m is not ...:
        kwargs["boundary_sample_spacing_m"] = boundary_sample_spacing_m
    if equal_area_crs is not None:
        kwargs["equal_area_crs"] = equal_area_crs

    metrics = run_evaluate(pred_gdf, ref_gdf, **kwargs)
    text = json.dumps(_jsonable(metrics), indent=2)
    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        click.echo(f"Metrics → {out}")
    else:
        click.echo(text)


# ---------------------------------------------------------------------------
# query-ftw
# ---------------------------------------------------------------------------


@main.command("query-ftw")
@click.option(
    "--study-area",
    required=True,
    help="AOI: vector file, bbox:minx,miny,maxx,maxy (EPSG:4326), or WKT.",
)
@click.option("--year", default=None, type=int, help="Optional FTW prediction year filter.")
@click.option("--label", default="field", help="Optional FTW label filter. Default: field.")
@click.option(
    "--clip/--no-clip",
    default=True,
    show_default=True,
    help=(
        "Clip polygons to the AOI (the area and perimeter of clipped polygons are "
        "recomputed; agribound:clipped marks them)."
    ),
)
@click.option("--output", "-o", default=None, help="Output vector path.")
@click.option(
    "--output-format",
    default=None,
    type=click.Choice(["gpkg", "geojson", "parquet"]),
    help="Output format override. Inferred from --output by default.",
)
@click.option(
    "--source-url",
    default=None,
    help="Manifest URL or base URL for relative tile paths.",
)
@click.option("--manifest-path", default=None, help="Local or HTTP(S) FTW tile manifest path.")
@click.option("--tile-dir", default=None, help="Local directory containing FTW GeoParquet tiles.")
@click.option("--cache-dir", default=None, help="Directory for downloaded remote manifest/tiles.")
@click.option(
    "--source-backend",
    default="auto",
    type=click.Choice(["auto", "pyarrow", "manifest"]),
    help=(
        "FTW source backend. Auto uses public PyArrow source unless manifest/tile inputs are given."
    ),
)
@click.option(
    "--max-features",
    default=None,
    type=int,
    help="Optional row limit for PyArrow-backed preview/smoke-test queries.",
)
@click.option(
    "--columns",
    multiple=True,
    help="Tile columns to read and return. May be passed multiple times or comma-separated.",
)
@click.option(
    "--deduplicate/--no-deduplicate",
    default=True,
    show_default=True,
    help="Remove repeated FTW polygons (identical geometry in the same prediction year).",
)
@click.option("--dst-crs", default=None, help="Optional output CRS, e.g. EPSG:5070.")
@click.option(
    "--min-confidence",
    default=None,
    type=float,
    help="Keep only polygons whose published confidence is at least this value.",
)
@click.option(
    "--keep-null-confidence/--drop-null-confidence",
    default=True,
    show_default=True,
    help="With --min-confidence: keep or drop polygons that have no confidence value.",
)
@click.option(
    "--layout",
    default="by-admin-conf",
    show_default=True,
    type=click.Choice(["by-admin-conf", "raw"]),
    help=(
        "Published FTW layout: 'by-admin-conf' (alpha/results-by-admin-conf, partitioned "
        "by country/subdivision) or 'raw' (legacy alpha/results)."
    ),
)
def query_ftw_cmd(
    study_area,
    year,
    label,
    clip,
    output,
    output_format,
    source_url,
    manifest_path,
    tile_dir,
    cache_dir,
    source_backend,
    max_features,
    columns,
    deduplicate,
    dst_crs,
    min_confidence,
    keep_null_confidence,
    layout,
):
    """Query already-published FTW prediction polygons for an AOI."""
    from agribound.ftw_query import query_ftw

    # Confidence/layout options are forwarded when given; otherwise query_ftw's
    # own defaults apply (the same values shown in --help).
    ctx = click.get_current_context()
    confidence_options = {
        name: value
        for name, value in (
            ("min_confidence", min_confidence),
            ("keep_null_confidence", keep_null_confidence),
            ("layout", layout),
        )
        if ctx.get_parameter_source(name) in _EXPLICIT_SOURCES
    }
    parsed_columns = list(columns) if columns else None
    gdf = query_ftw(
        study_area=study_area,
        year=year,
        label=label or None,
        clip=clip,
        output_path=output,
        output_format=output_format,
        source_url=source_url,
        manifest_path=manifest_path,
        tile_dir=tile_dir,
        cache_dir=cache_dir,
        source_backend=source_backend,
        max_features=max_features,
        columns=parsed_columns,
        deduplicate=deduplicate,
        dst_crs=dst_crs,
        **confidence_options,
    )

    if output:
        click.echo(f"Queried {len(gdf)} published FTW polygons → {output}")
        if gdf.attrs.get("provenance_path"):
            click.echo(f"Provenance → {gdf.attrs['provenance_path']}")
    else:
        click.echo(f"Queried {len(gdf)} published FTW polygons")


# ---------------------------------------------------------------------------
# Listings
# ---------------------------------------------------------------------------


@main.command("list-engines")
def list_engines_cmd():
    """List available delineation engines."""
    from agribound.engines import list_engines

    engines = list_engines()
    click.echo("\nAvailable Engines:")
    click.echo("-" * 100)
    for name, info in engines.items():
        gpu = info.get("gpu_recommended", info.get("gpu_required", False))
        device = "GPU recommended" if gpu else "CPU"
        flags = [
            label
            for key, label in (("label_free", "label-free"), ("fine_tunable", "fine-tunable"))
            if info.get(key)
        ]
        suffix = f" ({', '.join(flags)})" if flags else ""
        click.echo(f"  {name:<20} {info['approach']:<50} [{device}]{suffix}")
    click.echo()


@main.command("list-sources")
def list_sources_cmd():
    """List available satellite sources."""
    from agribound.composites import list_sources

    sources = list_sources()
    click.echo("\nAvailable Satellite Sources:")
    click.echo("-" * 100)
    for name, info in sources.items():
        res = f"{info['resolution_m']}m" if info.get("resolution_m") else "varies"
        gee = "GEE" if info.get("requires_gee") else "no GEE"
        restricted = " (restricted)" if info.get("restricted") else ""
        years = info.get("year_range")
        if years:
            first, last = years
            span = f"{first}-{last if last is not None else 'present'}"
        else:
            span = "any"
        click.echo(f"  {name:<20} {info['name']:<34} {res:<8} {span:<13} [{gee}]{restricted}")
    click.echo()


@main.command("list-ftw-models")
@click.option("--all", "include_all", is_flag=True, help="Include legacy FTW models.")
def list_ftw_models_cmd(include_all: bool) -> None:
    """List the FTW models available from the installed ftw-tools."""
    from agribound.engines.ftw import list_ftw_models

    try:
        models = list_ftw_models(include_legacy=include_all)
    except ImportError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("\nFTW Models:")
    click.echo("-" * 100)
    for name, info in models.items():
        tags = []
        if info.get("default"):
            tags.append("default")
        if info.get("instance_segmentation"):
            tags.append("instance")
        if info.get("requires_window"):
            tags.append("two-window")
        if info.get("legacy"):
            tags.append("legacy")
        suffix = f" [{', '.join(tags)}]" if tags else ""
        title = info.get("title") or ""
        click.echo(f"  {name:<36} {title}{suffix}")
    click.echo()


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------


@main.command()
@click.option("--project", default=None, help="GEE project ID.")
@click.option("--service-account-key", default=None, help="Path to service account JSON key.")
def auth(project, service_account_key):
    """Authenticate with Google Earth Engine."""
    from agribound.auth import setup_gee

    try:
        setup_gee(project=project, service_account_key=service_account_key)
        click.echo("GEE authentication successful!")
    except Exception as exc:
        click.echo(f"Authentication failed: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Optional command groups
# ---------------------------------------------------------------------------

#: (module, attribute) of optional commands registered when the module exists.
_OPTIONAL_COMMANDS: tuple[tuple[str, str], ...] = (
    ("agribound.hpc.cli", "tiles"),
    ("agribound.agent.cli", "agent"),
    ("agribound.agent.cli", "mcp"),
)


def _is_missing_module(exc: ModuleNotFoundError, module_name: str) -> bool:
    """True if *exc* reports *module_name* itself (or its agribound parent package) missing."""
    missing = exc.name or ""
    if missing == module_name:
        return True
    return missing.startswith("agribound.") and module_name.startswith(missing + ".")


def _register_optional_commands(
    group: click.Group, specs: tuple[tuple[str, str], ...] = _OPTIONAL_COMMANDS
) -> list[str]:
    """Add optional commands to *group*; return the names that were registered.

    A missing optional module is skipped. Any other import error, including a
    missing third-party dependency inside an existing module, is re-raised.
    """
    registered = []
    for module_name, attr in specs:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if _is_missing_module(exc, module_name):
                logger.debug("Optional CLI module %s not available", module_name)
                continue
            raise
        command = getattr(module, attr)
        group.add_command(command)
        registered.append(command.name)
    return registered


_register_optional_commands(main)


if __name__ == "__main__":
    main()
