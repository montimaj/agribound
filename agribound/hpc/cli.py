"""
``agribound tiles``: tile large study areas and run them as batch jobs.

Subcommands:

- ``make``: cut a study area into tiles and write the manifest and one
  configuration per tile (:func:`agribound.hpc.tiles.write_tile_manifest`).
- ``run``: run one tile (``--index``, default ``$SLURM_ARRAY_TASK_ID``) for a
  stage (``composite``, ``delineate`` or ``all``).
- ``merge``: merge finished tiles into one file with a provenance summary.
- ``status``: per-tile progress, and index lists for ``sbatch --array``.
- ``prefetch``: download the weights a tiled run needs (engine weights and,
  with ``sam_refine``, the SAM weights) before compute nodes go offline.
- ``region``: print a region definition (``examples/regions/*.yaml``) as shell
  variables for the example scripts.
- ``matrix``: expand years x sources x engines into runs and skipped
  combinations (registry rules; used by the example scripts).
- ``gee-project``: print the Earth Engine project the runs will use, or fail
  with instructions when they need one and none is configured (used by the
  example scripts before they run or submit anything).

``make``, ``run``, ``merge`` and ``prefetch`` accept ``--dry-run``, which
prints what would be done and exits without writing, downloading or running
anything; ``status``, ``region``, ``matrix`` and ``gee-project`` only read. The module imports only
:mod:`click` at import time (``agribound`` registers this group on start-up);
everything else is imported inside the commands.
"""

from __future__ import annotations

import json
import logging
import os
import signal
from typing import Any

import click

logger = logging.getLogger(__name__)


def _parse_engine_params(ctx: click.Context, param: click.Parameter, values: tuple[str, ...]):
    """Parse repeated ``KEY=VALUE`` options (VALUE as JSON, else string)."""
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


def _load_base_config(config_file: str, study_area: str | None, engine_params: dict):
    """Load the base YAML and apply --study-area / --engine-param."""
    from agribound.config import AgriboundConfig

    try:
        config = AgriboundConfig.from_yaml(config_file)
        overrides: dict[str, Any] = {}
        if study_area:
            overrides["study_area"] = study_area
        if engine_params:
            merged = dict(config.engine_params)
            merged.update(engine_params)
            overrides["engine_params"] = merged
        if overrides:
            config = config.merged(**overrides)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        raise click.UsageError(f"Invalid base configuration {config_file}: {exc}") from exc
    if not config.study_area:
        raise click.UsageError("No study area: pass --study-area or set study_area in --config.")
    return config


@click.group("tiles")
def tiles() -> None:
    """Tile large study areas and run them as independent (HPC) jobs.

    Typical sequence: 'tiles make' -> 'tiles run --stage composite' (nodes with
    internet) -> 'tiles run --stage delineate' (GPU nodes) -> 'tiles merge'.
    See examples/hpc/README.md.
    """


# ---------------------------------------------------------------------------
# make
# ---------------------------------------------------------------------------


@tiles.command("make")
@click.option(
    "--config",
    "config_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Base AgriboundConfig YAML shared by all tiles (e.g. from 'agribound delineate "
    "--dry-run ... > base.yaml').",
)
@click.option(
    "--study-area",
    default=None,
    help="Area to tile (vector file, 'bbox:minx,miny,maxx,maxy', WKT, GEE asset). "
    "[default: study_area of --config]",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory for the manifest, tile configurations and tile outputs.",
)
@click.option(
    "--tile-size-m",
    default=20000.0,
    show_default=True,
    type=float,
    help="Core tile edge length in metres.",
)
@click.option(
    "--halo-m",
    default=1000.0,
    show_default=True,
    type=float,
    help="Halo around each core in metres; must exceed the largest expected field dimension.",
)
@click.option(
    "--grid",
    "grid_kind",
    default="utm",
    show_default=True,
    type=click.Choice(["utm", "equal-area"]),
    help="'utm': one grid per UTM zone; 'equal-area': one Lambert azimuthal equal-area grid.",
)
@click.option(
    "--clip/--no-clip",
    default=True,
    show_default=True,
    help="Clip tile cores to the study area.",
)
@click.option(
    "--engine-param",
    "engine_param",
    multiple=True,
    metavar="KEY=VALUE",
    callback=_parse_engine_params,
    help="Engine parameter added to the base configuration (repeatable; VALUE parsed as JSON).",
)
@click.option(
    "--cache-root",
    default=None,
    type=click.Path(file_okay=False),
    help="Parent of the per-tile cache directories <cache-root>/<tile_id>. "
    "[default: cache_dir of --config, else <out-dir>/tiles/<tile_id>/cache]",
)
@click.option(
    "--keep-reference",
    is_flag=True,
    default=False,
    help="Keep reference_boundaries in the tile configurations (per-tile evaluation).",
)
@click.option(
    "--allow-fine-tune-per-tile",
    is_flag=True,
    default=False,
    help="Allow fine_tune=True (one separately fine-tuned model per tile).",
)
@click.option("--overwrite", is_flag=True, default=False, help="Replace a different manifest.")
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the tiling summary (JSON) and exit without writing files.",
)
def make_cmd(
    config_file: str,
    study_area: str | None,
    out_dir: str,
    tile_size_m: float,
    halo_m: float,
    grid_kind: str,
    clip: bool,
    engine_param: dict,
    cache_root: str | None,
    keep_reference: bool,
    allow_fine_tune_per_tile: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """Cut a study area into tiles and write the manifest."""
    from agribound.hpc.tiles import make_tiles, write_tile_manifest

    config = _load_base_config(config_file, study_area, engine_param)
    try:
        tiles_gdf = make_tiles(
            config.study_area,
            tile_size_m=tile_size_m,
            halo_m=halo_m,
            crs=grid_kind,
            clip=clip,
            config=config,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    summary = {
        "n_tiles": len(tiles_gdf),
        "grid": tiles_gdf.attrs["grid"]["kind"],
        "systems": sorted(tiles_gdf.attrs["grid"]["systems"]),
        "tile_size_m": tile_size_m,
        "halo_m": halo_m,
        "clip": clip,
        "core_area_km2": round(float(tiles_gdf["core_area_km2"].sum()), 3),
        "first_tiles": list(tiles_gdf["tile_id"].head(5)),
        "out_dir": os.path.abspath(out_dir),
    }
    if dry_run:
        click.echo(json.dumps(summary, indent=2))
        return
    try:
        path = write_tile_manifest(
            tiles_gdf,
            config,
            out_dir,
            cache_root=cache_root,
            overwrite=overwrite,
            keep_reference=keep_reference,
            allow_fine_tune_per_tile=allow_fine_tune_per_tile,
        )
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(f"Wrote {len(tiles_gdf)} tiles -> {path}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _raise_on_sigterm(signum: int, frame: Any) -> None:  # pragma: no cover - signal path
    raise SystemExit(128 + signum)


@tiles.command("run")
@click.option(
    "--manifest",
    required=True,
    type=click.Path(exists=True),
    help="manifest.json or the directory containing it.",
)
@click.option(
    "--index",
    default=None,
    type=int,
    help="0-based tile index [default: $SLURM_ARRAY_TASK_ID]. --index-offset is added.",
)
@click.option(
    "--index-offset",
    default=0,
    show_default=True,
    type=int,
    envvar="AGB_INDEX_OFFSET",
    help="Added to the index (for arrays split to respect MaxArraySize; env AGB_INDEX_OFFSET).",
)
@click.option("--tile-id", default=None, help="Run the tile with this ID instead of an index.")
@click.option(
    "--stage",
    default="all",
    show_default=True,
    type=click.Choice(["all", "composite", "delineate"]),
    help="'composite': download only; 'delineate': delineate a staged tile; 'all': both.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Re-run a stage that is done, and retry a tile recorded as no-data.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the tile, its configuration and its current status; run nothing.",
)
def run_cmd(
    manifest: str,
    index: int | None,
    index_offset: int,
    tile_id: str | None,
    stage: str,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """Run one tile (idempotent; finished stages are skipped).

    Prints the result as JSON. A tile without input data (water, outside the
    source's coverage) is recorded as no-data and exits with status 0.
    """
    from agribound.hpc.tiles import load_manifest, run_tile, tile_status

    m = load_manifest(manifest)
    if tile_id is not None:
        target: int | str = tile_id
    else:
        if index is None:
            env = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env is None:
                raise click.UsageError("Pass --index or --tile-id (or run inside a Slurm array).")
            index = int(env)
        target = index + index_offset
        if not 0 <= target < len(m["tiles"]):
            raise click.UsageError(
                f"Tile index {target} (index {index} + offset {index_offset}) is out of range: "
                f"the manifest has {len(m['tiles'])} tiles."
            )

    if dry_run:
        status = tile_status(m)
        if isinstance(target, str):
            row = status[status["tile_id"] == target]
        else:
            row = status[status["index"] == target]
        if row.empty:
            raise click.UsageError(f"Tile {target!r} is not in the manifest")
        info = row.iloc[0].to_dict()
        entry = m["tiles"][int(info["index"])]
        click.echo(
            json.dumps(
                {
                    "stage": stage,
                    "tile_id": entry["tile_id"],
                    "index": entry["index"],
                    "config": os.path.join(m["_root"], entry["config"]),
                    "study_area": entry["study_area"],
                    "composite": info["composite"],
                    "delineate": info["delineate"],
                },
                indent=2,
            )
        )
        return

    signal.signal(signal.SIGTERM, _raise_on_sigterm)
    try:
        result = run_tile(m, target, stage=stage, overwrite=overwrite)
    except FileExistsError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        logger.exception("Tile %s failed", target)
        raise click.ClickException(f"Tile {target} failed ({stage}): {exc}") from exc
    click.echo(json.dumps(result))


# ---------------------------------------------------------------------------
# merge
# ---------------------------------------------------------------------------


@tiles.command("merge")
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    default=None,
    help="Merged output file [default: <manifest dir>/fields_merged.<format>].",
)
@click.option("--crs", default="EPSG:4326", show_default=True, help="CRS of the merged output.")
@click.option(
    "--allow-missing",
    is_flag=True,
    default=False,
    help="Merge even if some tiles are not done (no-data tiles never count as missing).",
)
@click.option(
    "--reference",
    default=None,
    type=click.Path(exists=True),
    help="Evaluate the merged output against these reference boundaries.",
)
@click.option(
    "--overlap-check/--no-overlap-check",
    default=True,
    show_default=True,
    help="Count overlapping polygons from different tiles.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    default=False,
    help="Recompute and replace an existing merged output.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print how many tiles are done, no-data or not done, the output path and whether a "
    "merged output exists (and how many tiles it lacks); merge nothing.",
)
def merge_cmd(
    manifest: str,
    output: str | None,
    crs: str,
    allow_missing: bool,
    reference: str | None,
    overlap_check: bool,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """Merge finished tiles (representative-point rule) into one file."""
    from agribound.hpc.tiles import (
        default_merge_output,
        format_index_ranges,
        load_manifest,
        merge_tiles,
        tile_status,
    )

    m = load_manifest(manifest)
    if dry_run:
        from pathlib import Path

        from agribound.provenance import read_provenance

        status = tile_status(m)
        done = status["delineate"] == "done"
        no_data = status["delineate"] == "no-data"
        out_path = Path(output).expanduser() if output else default_merge_output(m)
        previous = read_provenance(out_path) if out_path.exists() else None
        click.echo(
            json.dumps(
                {
                    "output": str(out_path),
                    "output_exists": out_path.exists(),
                    # Tiles missing from an existing merged output (None: no summary).
                    "output_missing_tiles": (
                        len(previous.get("missing_tiles") or []) if previous else None
                    ),
                    "n_tiles": len(status),
                    "n_done": int(done.sum()),
                    "n_no_data": int(no_data.sum()),
                    "not_done_indices": format_index_ranges(status.loc[~done & ~no_data, "index"]),
                },
                indent=2,
            )
        )
        return
    try:
        gdf = merge_tiles(
            m,
            output,
            crs=crs,
            allow_missing=allow_missing,
            overwrite=overwrite,
            reference=reference,
            check_overlaps=overlap_check,
        )
    except (RuntimeError, FileExistsError) as exc:
        raise click.ClickException(str(exc)) from exc
    summary = gdf.attrs.get("merge_summary", {})
    click.echo(
        f"Merged {len(gdf)} polygons from {summary.get('n_merged_tiles')} tiles -> "
        f"{summary.get('output_path')}"
    )
    if summary.get("n_no_data_tiles"):
        click.echo(
            f"{summary['n_no_data_tiles']} tiles have no input data "
            f"({summary.get('no_data_core_area_km2')} of {summary.get('core_area_km2_total')} km2 "
            "of tile cores); see no_data_tiles in the provenance summary."
        )
    metrics = gdf.attrs.get("evaluation_metrics")
    if metrics:
        parts = [
            f"{k}={metrics[k]:.3f}"
            for k in ("precision", "recall", "f1", "iou_mean")
            if isinstance(metrics.get(k), int | float)
        ]
        if parts:
            click.echo("Evaluation: " + ", ".join(parts))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@tiles.command("status")
@click.option("--manifest", required=True, type=click.Path(exists=True))
@click.option(
    "--json", "as_json", is_flag=True, default=False, help="Print one JSON record per tile."
)
@click.option(
    "--list",
    "list_state",
    default=None,
    type=click.Choice(["done", "failed", "pending", "stale", "no-data", "not-done"]),
    help="Print only the indices of tiles in this state as an sbatch --array list. "
    "'not-done': tiles that still need work (neither done nor no-data).",
)
@click.option(
    "--stage",
    default="delineate",
    show_default=True,
    type=click.Choice(["composite", "delineate"]),
    help="Which stage --list refers to.",
)
@click.option(
    "--lines",
    is_flag=True,
    default=False,
    help="With --list: print one index per line instead of ranges.",
)
def status_cmd(
    manifest: str, as_json: bool, list_state: str | None, stage: str, lines: bool
) -> None:
    """Show per-tile progress.

    States: done, failed, pending, stale (output from another configuration),
    no-data (the source has no data for the tile: water, or outside its
    coverage; final, merged as empty) and error (unreadable tile configuration).
    """
    from agribound.hpc.tiles import format_index_ranges, load_manifest, tile_status

    status = tile_status(load_manifest(manifest))
    if list_state is not None:
        column = status[stage]
        if list_state == "not-done":
            mask = ~column.isin(["done", "no-data"])
        else:
            mask = column == list_state
        indices = [int(i) for i in status.loc[mask, "index"]]
        if lines:
            for index in indices:
                click.echo(str(index))
        else:
            click.echo(format_index_ranges(indices))
        return
    if as_json:
        for record in status.to_dict(orient="records"):
            click.echo(json.dumps(record, default=str))
        return
    click.echo(f"{len(status)} tiles")
    for column in ("composite", "delineate"):
        counts = status[column].value_counts().to_dict()
        click.echo(f"  {column:<10} " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    failed = status[(status["composite"] == "failed") | (status["delineate"] == "failed")]
    for row in failed.head(20).itertuples(index=False):
        click.echo(f"  FAILED {row.index:>6} {row.tile_id}: {row.error}")
    if len(failed) > 20:
        click.echo(f"  ... {len(failed) - 20} more failed tiles")
    no_data = status[status["delineate"] == "no-data"]
    for row in no_data.head(5).itertuples(index=False):
        click.echo(f"  NO-DATA {row.index:>5} {row.tile_id}: {row.no_data_reason}")
    if len(no_data) > 5:
        click.echo(f"  ... {len(no_data) - 5} more no-data tiles (--list no-data)")
    n_polys = status["n_output"].dropna()
    if len(n_polys):
        click.echo(f"  polygons in finished tiles: {int(n_polys.sum())}")


# ---------------------------------------------------------------------------
# prefetch
# ---------------------------------------------------------------------------


@tiles.command("prefetch")
@click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Base AgriboundConfig YAML (the one given to 'tiles make').",
)
@click.option(
    "--manifest",
    default=None,
    type=click.Path(exists=True),
    help="Use the base configuration stored in this tile manifest instead of --config.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print what would be downloaded (engine, SAM backend) and exit.",
)
def prefetch_cmd(config_file: str | None, manifest: str | None, dry_run: bool) -> None:
    """Download engine weights (and SAM weights with sam_refine) for offline nodes.

    Runs the engine's prefetch() and, when the configuration enables
    sam_refine, the SAM refinement prefetch of agribound.engines.samgeo_engine
    (the same downloads as 'agribound prefetch --config'). Run it on a node with
    internet access, with HF_HOME / TORCH_HOME / FTW_CACHE_DIR pointing to
    storage the compute nodes can read (examples/hpc/common.sh sets them from
    AGB_WEIGHTS_DIR).
    """
    from agribound.config import AgriboundConfig

    if (config_file is None) == (manifest is None):
        raise click.UsageError("Pass exactly one of --config or --manifest.")
    try:
        if manifest is not None:
            from agribound.hpc.tiles import load_manifest

            config = AgriboundConfig.from_dict(load_manifest(manifest)["base_config"])
        else:
            config = AgriboundConfig.from_yaml(config_file)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        raise click.UsageError(f"Invalid configuration: {exc}") from exc

    plan = {
        "engine": config.engine,
        "engine_params": config.engine_params,
        "sam_refine": bool(config.sam_refine and config.engine != "embedding"),
        "sam_backend": config.sam_backend if config.sam_refine else None,
        "sam_model": config.sam_model if config.sam_refine else None,
        "HF_HOME": os.environ.get("HF_HOME"),
        "TORCH_HOME": os.environ.get("TORCH_HOME"),
    }
    if dry_run:
        click.echo(json.dumps(plan, indent=2, default=str))
        return

    from agribound.engines import get_engine

    paths = [str(p) for p in (get_engine(config.engine).prefetch(config) or [])]
    if plan["sam_refine"]:
        from agribound.engines.samgeo_engine import prefetch as sam_prefetch

        paths += [str(p) for p in (sam_prefetch(config) or [])]
    for path in paths:
        click.echo(path)
    if not paths:
        click.echo(f"Engine {config.engine!r} reported no files to prefetch.")
    else:
        click.echo(f"Prefetched {len(paths)} file(s).")


# ---------------------------------------------------------------------------
# region
# ---------------------------------------------------------------------------


def _split_list(value: str) -> list[str]:
    return [v for v in value.replace(",", " ").split() if v]


@tiles.command("matrix")
@click.option("--years", required=True, help="Years, comma or space separated.")
@click.option("--sources", required=True, help="Sources, comma or space separated.")
@click.option("--engines", required=True, help="Engines, comma or space separated.")
@click.option("--tessera-version", default=None, help="TESSERA version for the year check.")
@click.option(
    "--fine-tune", is_flag=True, default=False, help="Fine-tune engines that are not label-free."
)
@click.option(
    "--has-checkpoint",
    is_flag=True,
    default=False,
    help="A checkpoint is supplied, so engines that are not label-free can run.",
)
@click.option(
    "--include-restricted", is_flag=True, default=False, help="Allow restricted sources (SPOT)."
)
def matrix_cmd(
    years: str,
    sources: str,
    engines: str,
    tessera_version: str | None,
    fine_tune: bool,
    has_checkpoint: bool,
    include_restricted: bool,
) -> None:
    """Print the year x source x engine runs (tab-separated; used by the example scripts).

    Columns: action (run|skip), year, source, engine, fine_tune (yes|no),
    note (comma-separated: gfm-env when the engine needs environment-gfm.yml,
    cpu when the registry does not recommend a GPU; "-" for none), reason
    ("-" for runs).
    """
    from agribound.hpc.regions import plan_runs

    try:
        year_list = [int(y) for y in _split_list(years)]
        plan = plan_runs(
            year_list,
            [s.lower() for s in _split_list(sources)],
            [e.lower() for e in _split_list(engines)],
            tessera_version=tessera_version or None,
            fine_tune=fine_tune,
            has_checkpoint=has_checkpoint,
            include_restricted=include_restricted,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    for row in plan:
        click.echo(
            "\t".join(
                [
                    row["action"],
                    str(row["year"]),
                    row["source"],
                    row["engine"],
                    "yes" if row["fine_tune"] else "no",
                    row["note"] or "-",
                    row["reason"] or "-",
                ]
            )
        )


@tiles.command("region")
@click.option(
    "--region",
    "region_name",
    required=True,
    help="Region name (examples/regions/<name>.yaml) or path to a region YAML.",
)
@click.option(
    "--format",
    "fmt",
    default="shell",
    show_default=True,
    type=click.Choice(["shell", "json"]),
    help="'shell': AGB_REGION_*=... assignments for eval; 'json': the parsed file.",
)
@click.option("--test", is_flag=True, default=False, help="Use the region's small test_bbox.")
def region_cmd(region_name: str, fmt: str, test: bool) -> None:
    """Print a region definition (used by examples/run_region_delineation.sh)."""
    from agribound.hpc.regions import load_region, region_shell_assignments

    try:
        region = load_region(region_name)
        text = (
            region_shell_assignments(region, test=test)
            if fmt == "shell"
            else json.dumps(region, indent=2, default=str) + "\n"
        )
    except (FileNotFoundError, ValueError) as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(text, nl=False)


# ---------------------------------------------------------------------------
# gee-project
# ---------------------------------------------------------------------------

#: Stand-in project used only to validate a base configuration without running
#: AgriboundConfig's own project lookup (gee-project resolves the project itself).
_STAND_IN_PROJECT = "agribound-gee-project-check"


def _no_project_error(uses: list[str], *, from_config: bool) -> click.ClickException:
    first = (
        "gee_project in the base configuration (agribound delineate --dry-run --gee-project ID "
        "... > base.yaml)"
        if from_config
        else "--gee-project ID"
    )
    what = f" ({', '.join(uses)})" if uses else ""
    return click.ClickException(
        f"No Google Earth Engine project found, and these runs use Earth Engine{what}. "
        f"Provide one with: {first}; the GEE_PROJECT environment variable; "
        "'gcloud config set project ID'; or a service-account key whose project_id is the "
        "project (--gee-service-account-key, AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY or "
        "GOOGLE_APPLICATION_CREDENTIALS). Use your own Google Cloud project registered "
        "for Earth Engine."
    )


@tiles.command("gee-project")
@click.option("--project", default=None, help="Project given explicitly (printed as is).")
@click.option(
    "--service-account-key",
    default=None,
    help="Service-account JSON key whose project_id is the last fallback.",
)
@click.option(
    "--sources",
    default=None,
    help="Sources of the runs, comma or space separated. Without it a project is always "
    "needed; with it, only if a source uses Earth Engine or the LULC filter is on.",
)
@click.option(
    "--no-lulc-filter",
    is_flag=True,
    default=False,
    help="The runs skip the LULC filter (which uses Earth Engine for every source).",
)
@click.option(
    "--config",
    "config_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Base AgriboundConfig YAML instead of the options above: its gee_project, "
    "gee_service_account_key and requires_gee() are used.",
)
def gee_project_cmd(
    project: str | None,
    service_account_key: str | None,
    sources: str | None,
    no_lulc_filter: bool,
    config_file: str | None,
) -> None:
    """Print the Earth Engine project the runs will use (used by the example scripts).

    The project is resolved as agribound resolves it, without contacting Earth
    Engine: --project (or gee_project of --config), then GEE_PROJECT, then the
    gcloud configuration, then the project_id of the credentials file
    (--service-account-key or gee_service_account_key of --config, else
    AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY, else GOOGLE_APPLICATION_CREDENTIALS).
    Prints nothing when no run uses Earth Engine. Exits with status 1 and the
    ways to provide a project when a run uses Earth Engine and none is found.
    """
    from agribound.auth import project_from_credentials, resolve_project
    from agribound.registry import SOURCE_REGISTRY

    uses: list[str] = []
    if config_file:
        if project or service_account_key or sources or no_lulc_filter:
            raise click.UsageError(
                "--config cannot be combined with --project, --service-account-key, --sources "
                "or --no-lulc-filter (the base configuration defines them)."
            )
        import yaml

        from agribound.config import AgriboundConfig

        try:
            with open(config_file) as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError("the file must contain a mapping of AgriboundConfig fields")
            project = data.get("gee_project") or None
            # The stand-in project skips AgriboundConfig's own lookup: only the other
            # fields are validated and requires_gee() is evaluated here.
            config = AgriboundConfig.from_dict(
                {**data, "gee_project": project or _STAND_IN_PROJECT}
            )
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            raise click.UsageError(f"Invalid base configuration {config_file}: {exc}") from exc
        service_account_key = config.gee_service_account_key
        if not config.requires_gee():
            return
        if config.is_gee_source():
            uses.append(f"source {config.source}")
        elif config.source == "google-embedding":
            uses.append("google-embedding with the gee backend")
        if config.lulc_filter:
            uses.append("the LULC filter")
    elif sources is not None:
        names = [s.lower() for s in _split_list(sources)]
        unknown = [s for s in names if s not in SOURCE_REGISTRY]
        if unknown:
            raise click.UsageError(f"Unknown sources {unknown}. Sources: {sorted(SOURCE_REGISTRY)}")
        uses = [f"source {s}" for s in names if SOURCE_REGISTRY[s].get("requires_gee")]
        if not no_lulc_filter:
            uses.append("the LULC filter")
        if not uses:
            return
    resolved = resolve_project(project) or project_from_credentials(service_account_key)
    if not resolved:
        raise _no_project_error(uses, from_config=bool(config_file))
    click.echo(resolved)


__all__ = ["tiles"]
