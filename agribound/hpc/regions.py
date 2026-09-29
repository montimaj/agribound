"""
Region definitions for large-area runs (``examples/regions/*.yaml``).

A region file describes one agricultural region: a verified EPSG:4326
bounding box, a small test box, recommended years, sources and engines,
data availability notes and tiling parameters. The ``run`` block holds the
machine-read defaults used by ``examples/run_region_delineation.sh``::

    name: iowa_corn_belt_us
    title: ...
    bbox: [minx, miny, maxx, maxy]        # EPSG:4326
    test_bbox: [minx, miny, maxx, maxy]   # optional small box for quick checks
    study_area: null                      # optional vector file (relative to this file)
    run:
      years: [2023, 2024]
      sources: [sentinel2, landsat]
      engines: [delineate-anything, ftw]
      tile_size_km: 20
      halo_m: 1500
      reference: null                     # optional, relative to this file
      tessera_version: v1                 # optional
      lulc_dataset: auto                  # optional (--lulc-dataset)

All other keys are free-form documentation (verification results, source
availability, reference data, FTW notes). :func:`plan_runs` expands the
``years x sources x engines`` matrix with the registry rules.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any

import yaml

#: Environment variable with extra directories to search for region files.
REGIONS_DIR_ENV = "AGRIBOUND_REGIONS_DIR"

_REQUIRED = ("name", "title", "bbox", "run")
_RUN_REQUIRED = ("years", "sources", "engines", "tile_size_km", "halo_m")


def _search_dirs() -> list[Path]:
    dirs = []
    env = os.environ.get(REGIONS_DIR_ENV)
    if env:
        dirs += [Path(p).expanduser() for p in env.split(os.pathsep) if p]
    dirs.append(Path.cwd() / "examples" / "regions")
    # Source checkout (editable install): <repo>/examples/regions
    dirs.append(Path(__file__).resolve().parents[2] / "examples" / "regions")
    return dirs


def find_region_file(name_or_path: str | Path) -> Path:
    """Resolve a region name (e.g. ``"punjab_in"``) or a path to a YAML file.

    Names are looked up as ``<name>.yaml`` in ``$AGRIBOUND_REGIONS_DIR``
    (``os.pathsep``-separated), ``./examples/regions`` and the
    ``examples/regions`` directory of the agribound source checkout.

    Raises
    ------
    FileNotFoundError
        If no region file is found (the message lists the searched places).
    """
    path = Path(name_or_path).expanduser()
    if path.suffix in (".yaml", ".yml") or path.exists():
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"Region file not found: {path}")
    searched = []
    for directory in _search_dirs():
        candidate = directory / f"{name_or_path}.yaml"
        searched.append(str(candidate))
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Region {name_or_path!r} not found. Searched: {searched}. Pass a YAML path or set "
        f"{REGIONS_DIR_ENV}."
    )


def _check_bbox(value: Any, key: str, name: str) -> list[float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"Region {name!r}: {key} must be [minx, miny, maxx, maxy]")
    minx, miny, maxx, maxy = (float(v) for v in value)
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise ValueError(f"Region {name!r}: {key} {value} is not a valid EPSG:4326 box")
    return [minx, miny, maxx, maxy]


def load_region(name_or_path: str | Path) -> dict[str, Any]:
    """Load and validate a region file.

    Returns
    -------
    dict
        The YAML content plus ``"_path"`` (absolute file path),
        ``"study_area"`` resolved to an absolute path or a ``"bbox:..."``
        string, ``"test_study_area"`` (``"bbox:..."`` or *None*) and
        ``run["reference"]`` resolved to an absolute path (or *None*).

    Raises
    ------
    ValueError
        If required keys are missing or the boxes are invalid.
    """
    path = find_region_file(name_or_path)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Region file {path} must contain a mapping")
    missing = [k for k in _REQUIRED if k not in data]
    if missing:
        raise ValueError(f"Region file {path} is missing keys {missing}")
    name = str(data["name"])
    run = data["run"] or {}
    missing = [k for k in _RUN_REQUIRED if k not in run]
    if missing:
        raise ValueError(f"Region file {path}: run block is missing keys {missing}")
    bbox = _check_bbox(data["bbox"], "bbox", name)
    test_bbox = data.get("test_bbox")
    if test_bbox is not None:
        test_bbox = _check_bbox(test_bbox, "test_bbox", name)

    base = path.parent
    study_area = data.get("study_area")
    if study_area:
        resolved = (base / study_area).resolve()
        if not resolved.exists():
            raise ValueError(f"Region {name!r}: study_area file {resolved} does not exist")
        data["study_area"] = str(resolved)
    else:
        data["study_area"] = bbox_string(bbox)
    data["test_study_area"] = bbox_string(test_bbox) if test_bbox else None
    reference = run.get("reference")
    run["reference"] = str((base / reference).resolve()) if reference else None
    for key in ("years", "sources", "engines"):
        value = run[key]
        run[key] = [value] if isinstance(value, str | int) else list(value)
    data["run"] = run
    data["_path"] = str(path)
    return data


def bbox_string(bbox: list[float]) -> str:
    """``[minx, miny, maxx, maxy]`` -> ``"bbox:minx,miny,maxx,maxy"``."""
    return "bbox:" + ",".join(_fmt_coord(v) for v in bbox)


def _fmt_coord(value: float) -> str:
    text = f"{float(value):.7f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def region_shell_assignments(region: dict[str, Any], test: bool = False) -> str:
    """Return ``AGB_REGION_*=...`` shell assignments (values quoted with :func:`shlex.quote`).

    Parameters
    ----------
    region : dict
        Output of :func:`load_region`.
    test : bool
        Use the region's ``test_bbox`` as the study area.

    Raises
    ------
    ValueError
        If *test* is True and the region has no ``test_bbox``.
    """
    run = region["run"]
    if test and not region.get("test_study_area"):
        raise ValueError(f"Region {region['name']!r} has no test_bbox")
    values = {
        "AGB_REGION_NAME": region["name"],
        "AGB_REGION_TITLE": region["title"],
        "AGB_REGION_FILE": region["_path"],
        "AGB_REGION_STUDY_AREA": region["test_study_area"] if test else region["study_area"],
        "AGB_REGION_YEARS": " ".join(str(y) for y in run["years"]),
        "AGB_REGION_SOURCES": " ".join(run["sources"]),
        "AGB_REGION_ENGINES": " ".join(run["engines"]),
        "AGB_REGION_TILE_SIZE_KM": str(run["tile_size_km"]),
        "AGB_REGION_HALO_M": str(run["halo_m"]),
        "AGB_REGION_REFERENCE": run.get("reference") or "",
        "AGB_REGION_TESSERA_VERSION": str(run.get("tessera_version") or ""),
        "AGB_REGION_LULC_DATASET": str(run.get("lulc_dataset") or ""),
    }
    return "\n".join(f"{k}={shlex.quote(str(v))}" for k, v in values.items()) + "\n"


#: Engines whose environment is ``environment-gfm.yml`` (terratorch), not the core env.
GFM_ENV_ENGINES = frozenset({"prithvi"})


def plan_runs(
    years: list[int],
    sources: list[str],
    engines: list[str],
    *,
    tessera_version: str | None = None,
    fine_tune: bool = False,
    has_checkpoint: bool = False,
    include_restricted: bool = False,
) -> list[dict[str, Any]]:
    """Expand years x sources x engines into runs and skipped combinations.

    A combination is skipped (with a reason) when the engine does not support
    the source (:func:`agribound.registry.engine_supports_source`), the year
    is outside the source's range
    (:func:`agribound.registry.source_year_range`, TESSERA by version), the
    source is restricted (SPOT) and *include_restricted* is False, or the
    engine is not label-free and neither *fine_tune* (the engine must be
    fine-tunable) nor *has_checkpoint* is set.

    Returns
    -------
    list of dict
        ``action`` (``"run"`` or ``"skip"``), ``year``, ``source``,
        ``engine``, ``fine_tune`` (bool: fine-tune this engine on the
        reference), ``note`` (comma-separated flags: ``"gfm-env"`` for
        engines that need the GFM environment, ``"cpu"`` for engines whose
        registry entry has ``gpu_recommended=False``; empty otherwise) and
        ``reason`` (for skips).

    Raises
    ------
    ValueError
        For unknown sources or engines.
    """
    from agribound.registry import (
        ENGINE_REGISTRY,
        SOURCE_REGISTRY,
        engine_supports_source,
        source_year_range,
    )

    unknown = [s for s in sources if s not in SOURCE_REGISTRY]
    unknown += [e for e in engines if e not in ENGINE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown source/engine names {unknown}. Sources: {sorted(SOURCE_REGISTRY)}; "
            f"engines: {sorted(ENGINE_REGISTRY)}"
        )
    plan = []
    for year in years:
        for source in sources:
            version = tessera_version if source == "tessera-embedding" else None
            years_ok = source_year_range(source, tessera_version=version)
            for engine in engines:
                info = ENGINE_REGISTRY[engine]
                notes = ["gfm-env"] if engine in GFM_ENV_ENGINES else []
                if not info["gpu_recommended"]:
                    notes.append("cpu")
                row: dict[str, Any] = {
                    "action": "run",
                    "year": int(year),
                    "source": source,
                    "engine": engine,
                    "fine_tune": False,
                    "note": ",".join(notes),
                    "reason": "",
                }
                if not engine_supports_source(engine, source):
                    row.update(action="skip", reason=f"{engine} does not support {source}")
                elif years_ok is not None and not (
                    years_ok[0] <= int(year) <= (years_ok[1] or 9999)
                ):
                    last = years_ok[1] if years_ok[1] is not None else "present"
                    label = f" {version}" if version else ""
                    row.update(
                        action="skip",
                        reason=f"no {source}{label} data for {year} ({years_ok[0]}-{last})",
                    )
                elif SOURCE_REGISTRY[source].get("restricted") and not include_restricted:
                    row.update(action="skip", reason=f"{source} is restricted (--include-spot)")
                elif not info["label_free"]:
                    if fine_tune and info["fine_tunable"]:
                        row["fine_tune"] = True
                    elif not has_checkpoint:
                        row.update(
                            action="skip",
                            reason=(
                                f"{engine} has no label-free weights: pass --fine-tune (needs a "
                                "reference) or --engine-param checkpoint_path=..."
                            ),
                        )
                plan.append(row)
    return plan


__all__ = [
    "GFM_ENV_ENGINES",
    "REGIONS_DIR_ENV",
    "bbox_string",
    "find_region_file",
    "load_region",
    "plan_runs",
    "region_shell_assignments",
]
