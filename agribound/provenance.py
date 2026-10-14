"""
Run provenance: what was run, with which inputs, versions and resources.

:class:`RunRecorder` collects a JSON-serialisable record of one pipeline run
(configuration and its hash, seed, package versions, platform, device, step
timings, peak memory, engine metadata, facts and warnings). The pipeline writes
it next to the output as ``<output_path>.provenance.json``
(:func:`provenance_path`) and uses :func:`reuse_mismatch` (the
:func:`config_hash`, the study-area fingerprint, the results versions of
:mod:`agribound._results`, for TESSERA the state of the store and, for HLS, the
recipe of a composite made at 180 degrees) to decide whether an existing output
can be reused.
"""

from __future__ import annotations

import contextlib
import datetime as _dt
import functools
import hashlib
import json
import logging
import math
import os
import platform
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROVENANCE_SCHEMA_VERSION = "1"

#: Configuration fields excluded from :func:`config_hash`: output location and
#: format, caching and provenance switches, credentials and request tuning,
#: execution resources and transport settings. Most of them do not change the
#: delineated polygons. ``device`` (like ``n_workers``) is excluded so that an
#: output can be reused on other hardware, although it can change the result:
#: the native Delineate-Anything backend runs in FP16 on CUDA and MPS and in FP32
#: on CPU (its ``engine_meta`` records ``precision`` and ``device``). An output
#: computed on one device is therefore reused as it is on another; pass
#: ``overwrite=True`` to recompute it there.
HASH_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {
        "output_path",
        "output_format",
        "overwrite",
        "provenance",
        "cache_dir",
        "embedding_cache_dir",
        "gee_project",
        "gee_service_account_key",
        "gee_high_volume",
        "gee_max_requests",
        "gee_workload_tag",
        "export_method",
        "gcs_bucket",
        "usgs_timeout_s",
        "usgs_retries",
        "lulc_batch_size",
        "n_workers",
        "device",
    }
)

#: Environment variables recorded when present (scheduler context).
_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_CLUSTER_NAME",
    "PBS_JOBID",
    "CUDA_VISIBLE_DEVICES",
)


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Recursively convert *value* to JSON-serialisable builtins.

    NumPy scalars/arrays, tuples, sets, paths, datetimes and objects with
    ``to_dict``/``isoformat`` are converted; non-finite floats become
    *None*; anything else becomes ``str(value)``.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted((to_jsonable(v) for v in value), key=str)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, _dt.datetime | _dt.date):
        return value.isoformat()
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return to_jsonable(value.item())
        if isinstance(value, np.ndarray):
            return to_jsonable(value.tolist())
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return to_jsonable(value.to_dict())
        except Exception:
            pass
    return str(value)


def _utc_now() -> str:
    return _utc_text(_dt.datetime.now(_dt.UTC))


def _utc_text(moment: _dt.datetime) -> str:
    """``YYYY-MM-DDTHH:MM:SSZ`` of a timezone-aware UTC datetime."""
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Hashing and file helpers
# ---------------------------------------------------------------------------


#: The open VHR SAR sources (:data:`agribound.registry.VHR_SAR_SOURCES`).
_VHR_SAR: tuple[str, ...] = ("umbra", "capella")
#: The crop-type map stacks (:data:`agribound.registry.CROP_SEQUENCE_SOURCES`).
_CROP_SEQUENCE: tuple[str, ...] = ("cdl-sequence", "aci-sequence")

#: Fields added after agribound 1.0.1, and when each one does not apply
#: (field -> predicate on the configuration dictionary). Such a field is left
#: out of :func:`config_hash` where it does not apply, so the hashes of earlier
#: configurations do not change and their outputs are still reused. The
#: identities computed from the whole configuration dictionary, the tile-manifest
#: signature (:func:`agribound.hpc.tiles.write_tile_manifest`) and the agent's
#: plan directory (``propose_run``), leave it out where it does not apply and
#: holds its default (:func:`drop_inapplicable_fields` with
#: ``keep_non_default=True``), so tile manifests and plan directories of
#: configurations that 1.0.1 could express are unchanged too. A field added
#: later must be listed here. ``landsat_pan_missions`` always applies to
#: ``source="landsat-pan"``, even at its default, so an output made before the
#: mission rule existed is not reused (that code mixed Landsat 7 and 8/9 PAN).
#: The ``vhr_sar_*`` fields apply only to the open VHR SAR sources (``umbra``,
#: ``capella``; ``vhr_sar_calibration`` only to ``capella``), and
#: ``vhr_sar_max_download_gb`` and ``vhr_sar_cache_dir`` never apply: a download
#: cap and a cache location do not change the composite, so they never enter the
#: hash (they are listed here rather than in :data:`HASH_EXCLUDED_FIELDS` so that
#: they also stay out of the tile-manifest signature and the plan directory while
#: they hold their defaults). ``crop_sequence_years`` applies only to the crop-type
#: map stacks (``cdl-sequence``, ``aci-sequence``), where it is hashed at any value.
HASH_CONDITIONAL_FIELDS: dict[str, Any] = {
    "lulc_tree_crops": lambda data: data.get("lulc_tree_crops") is not True,
    "landsat_pan_missions": lambda data: data.get("source") != "landsat-pan",
    "vhr_sar_collect_ids": lambda data: data.get("source") not in _VHR_SAR,
    "vhr_sar_max_collects": lambda data: data.get("source") not in _VHR_SAR,
    "vhr_sar_polarization": lambda data: data.get("source") not in _VHR_SAR,
    "vhr_sar_resolution_m": lambda data: data.get("source") not in _VHR_SAR,
    "vhr_sar_coregister": lambda data: data.get("source") not in _VHR_SAR,
    "vhr_sar_coregister_base": lambda data: data.get("source") not in _VHR_SAR,
    "vhr_sar_bands": lambda data: data.get("source") not in _VHR_SAR,
    "vhr_sar_calibration": lambda data: data.get("source") != "capella",
    "vhr_sar_max_download_gb": lambda data: True,
    "vhr_sar_cache_dir": lambda data: True,
    "crop_sequence_years": lambda data: data.get("source") not in _CROP_SEQUENCE,
}

#: Fields whose default changed after agribound 1.0.1, with their 1.0.1 default (JSON
#: form). Each one is in :data:`HASH_EXCLUDED_FIELDS`: it does not change the result. The
#: identities computed from the whole configuration dictionary (the tile-manifest signature
#: and the agent's plan directory; :func:`drop_inapplicable_fields` with
#: ``keep_non_default=True``) take such a field at its 1.0.1 default while it holds its
#: current default, so that a configuration that 1.0.1 could express keeps the identity that
#: 1.0.1 gave it. An explicit value equal to the 1.0.1 default then has the identity of the
#: current default. ``usgs_retries``: 3 in agribound 1.0.1, 5 since 1.1.0. A field whose
#: default changes later must be listed here.
DEFAULTS_OF_1_0_1: dict[str, Any] = {"usgs_retries": 3}


def _field_defaults(names: Any) -> dict[str, Any]:
    """Current defaults (JSON form) of the ``AgriboundConfig`` fields *names*."""
    import dataclasses

    from agribound.config import AgriboundConfig

    defaults = {}
    for f in dataclasses.fields(AgriboundConfig):
        if f.name in names:
            default = f.default if f.default is not dataclasses.MISSING else f.default_factory()
            defaults[f.name] = to_jsonable(default)
    return defaults


@functools.cache
def _conditional_field_defaults() -> dict[str, Any]:
    """Defaults of the :data:`HASH_CONDITIONAL_FIELDS` (JSON form), from ``AgriboundConfig``."""
    return _field_defaults(HASH_CONDITIONAL_FIELDS)


@functools.cache
def _changed_field_defaults() -> dict[str, Any]:
    """Current defaults of the :data:`DEFAULTS_OF_1_0_1` fields (JSON form)."""
    return _field_defaults(DEFAULTS_OF_1_0_1)


def drop_inapplicable_fields(
    data: dict[str, Any], *, keep_non_default: bool = False
) -> dict[str, Any]:
    """Return *data* without the :data:`HASH_CONDITIONAL_FIELDS` that do not apply to it.

    Every predicate sees the whole of *data*. :func:`canonical_config` (and so
    :func:`config_hash`) uses the default form: a field that does not apply
    cannot change the result, whatever its value.

    Parameters
    ----------
    data : dict
        Configuration dictionary (``AgriboundConfig.to_dict()``).
    keep_non_default : bool
        Leave a field out only when it does not apply **and** holds its
        default (or is missing). Every configuration written before the field
        existed holds the default, so identities computed from the whole
        configuration dictionary (the tile-manifest signature and the agent's
        plan directory) stay those of agribound 1.0.1 for the configurations
        1.0.1 could express, while an explicit value still changes them. A
        field of :data:`DEFAULTS_OF_1_0_1` that holds its current default is
        given its 1.0.1 default, for the same reason.

    Returns
    -------
    dict
        A new dictionary; *data* is not modified.
    """
    defaults = _conditional_field_defaults() if keep_non_default else {}
    drop = set()
    for name, not_applicable in HASH_CONDITIONAL_FIELDS.items():
        if name not in data or not not_applicable(data):
            continue
        if keep_non_default and to_jsonable(data[name]) != defaults[name]:
            continue
        drop.add(name)
    out = {k: v for k, v in data.items() if k not in drop}
    if keep_non_default:
        current = _changed_field_defaults()
        for name, old in DEFAULTS_OF_1_0_1.items():
            if name in out and to_jsonable(out[name]) == current[name]:
                out[name] = old
    return out


def canonical_config(config: Any) -> dict[str, Any]:
    """Return the configuration fields that define the result.

    Fields in :data:`HASH_EXCLUDED_FIELDS` are left out, and so are the fields
    of :data:`HASH_CONDITIONAL_FIELDS` where they do not apply
    (:func:`drop_inapplicable_fields`).
    """
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return {
        k: to_jsonable(v)
        for k, v in sorted(drop_inapplicable_fields(data).items())
        if k not in HASH_EXCLUDED_FIELDS
    }


def config_hash(config: Any) -> str:
    """Return the SHA-1 hex digest of the canonical YAML of *config*.

    Fields listed in :data:`HASH_EXCLUDED_FIELDS` (output location and format,
    caching/provenance switches, credentials, request tuning and execution
    resources) are excluded, so configurations that differ only in these
    fields hash identically. This includes ``device``, which can change the
    polygons slightly (see :data:`HASH_EXCLUDED_FIELDS`). Fields added after
    1.0.1 are hashed only where they apply (:data:`HASH_CONDITIONAL_FIELDS`):
    ``lulc_tree_crops`` when it is *True*, ``landsat_pan_missions`` for
    ``source="landsat-pan"``, the ``vhr_sar_*`` settings for ``umbra`` and
    ``capella`` (never ``vhr_sar_max_download_gb`` or ``vhr_sar_cache_dir``),
    and ``crop_sequence_years`` for ``cdl-sequence`` and ``aci-sequence``.

    Parameters
    ----------
    config : AgriboundConfig or dict
        Configuration.

    Returns
    -------
    str
        40-character hexadecimal digest.
    """
    text = yaml.safe_dump(canonical_config(config), sort_keys=True, default_flow_style=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def provenance_path(output_path: str | Path) -> Path:
    """Return the provenance sidecar path ``f"{output_path}.provenance.json"``."""
    return Path(f"{output_path}.provenance.json")


def write_provenance(output_path: str | Path, record: dict) -> Path:
    """Write *record* as JSON next to *output_path* (atomic replace).

    Parameters
    ----------
    output_path : str or Path
        Output vector path the record describes.
    record : dict
        Provenance record (converted with :func:`to_jsonable`).

    Returns
    -------
    pathlib.Path
        Path of the written sidecar.
    """
    path = provenance_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(to_jsonable(record), f, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


def read_provenance(output_path: str | Path) -> dict | None:
    """Read the provenance sidecar of *output_path*.

    Returns
    -------
    dict or None
        The record, or *None* if the sidecar is missing or not valid JSON.
    """
    path = provenance_path(output_path)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read provenance file %s: %s", path, exc)
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Output reuse
# ---------------------------------------------------------------------------


def reuse_facts(config: Any) -> dict[str, Any]:
    """Return the facts that :func:`reuse_mismatch` checks besides :func:`config_hash`.

    ``aoi_fingerprint`` is :func:`agribound._cache.aoi_fingerprint`: the
    study-area geometry (read from the file, ``bbox:`` string or WKT), the ID
    string of a GEE asset (so no Earth Engine access is needed; the asset's
    features are not covered), or the local raster's resolved path, size and
    modification time when there is no study area. It is left out when the
    study area cannot be read (the composite stage then reports the error).
    ``results_versions`` is :func:`agribound._results.results_versions`.

    Parameters
    ----------
    config : AgriboundConfig
        Configuration of the run.

    Returns
    -------
    dict
        ``{"aoi_fingerprint": str, "results_versions": dict}``.
    """
    from agribound._cache import aoi_fingerprint
    from agribound._results import results_versions

    facts: dict[str, Any] = {}
    try:
        facts["aoi_fingerprint"] = aoi_fingerprint(config)
    except Exception as exc:
        logger.debug("No study-area fingerprint for %r: %s", config.study_area, exc)
    facts["results_versions"] = results_versions(config)
    return facts


def _fingerprinted_input(config: Any) -> str | None:
    """Name the input whose contents only the study-area fingerprint covers, or *None*.

    :func:`config_hash` covers a ``bbox:`` or WKT study area (the text is the
    geometry) and a GEE asset (fingerprinted by its ID), but only the path of
    a study-area file or, without a study area, of the local raster. The
    string formats are told apart as in :func:`agribound.io.vector.read_study_area`.
    """
    from agribound._cache import _is_gee_asset
    from agribound.io.vector import _WKT_RE

    study_area = str(getattr(config, "study_area", "") or "").strip()
    if study_area:
        if _is_gee_asset(study_area) or study_area.lower().startswith("bbox:"):
            return None
        if _WKT_RE.match(study_area):
            return None
        return f"study-area file {study_area!r}"
    local_tif = getattr(config, "local_tif_path", None)
    return f"local raster {str(local_tif)!r}" if local_tif else None


def reuse_mismatch(
    record: dict,
    config: Any,
    output: str | Path | None = None,
    *,
    unverified: list[str] | None = None,
) -> str | None:
    """Return why the output that *record* describes cannot be reused for *config*.

    The output of a successful run is reused only when:

    - its ``config_hash`` equals :func:`config_hash` of *config*;
    - its ``facts["results_versions"]`` equal
      :func:`agribound._results.results_versions` of *config* (a missing fact
      or component counts as version 1, i.e. agribound <= 1.0.0), so an
      output of a component whose results have changed since is not reused;
    - its ``facts["aoi_fingerprint"]`` equals
      :func:`agribound._cache.aoi_fingerprint` of *config* when the study
      area is a file (a file whose geometry changed at the same path is not
      reused) or, without a study area, for the local raster (compared by
      resolved path, size and modification time). A ``bbox:``, WKT or GEE
      asset study area is covered by the configuration hash already;
    - for ``source="tessera-embedding"``, the store stamp of the TESSERA
      raster it was made from (``facts["composite"]["TESSERA_STORE_FINGERPRINT"]``,
      copied from the raster's tags) equals the store's current stamp for the
      box that was read (``TESSERA_READ_BBOX``;
      :func:`agribound.composites.local.current_tessera_stamp`). A record
      without a stamp (made before agribound recorded it, or when the store
      could not be checked) does not match when the current stamp can be
      computed. This check runs only when nothing else differs. When the
      current stamp cannot be computed (no network, an HTTP error,
      ``AGRIBOUND_TESSERA_STORE_CHECK=0``), the recorded stamp is compared,
      without network, with the stamp in the tags of the cached TESSERA
      raster (:func:`agribound.composites.local.cached_tessera_tags`: the
      raster a run of *config* uses, else ``facts["raster_path"]``): a raster
      read from the store again since the output was made (a different
      fingerprint for the same box, or one where the record has none) does
      not match;
    - for ``source="hls"``, an output whose composite tags
      (``facts["composite"]``) have no ``AGRIBOUND_HLS_RECIPE_VERSION`` (its
      composite was selected before HLS recipe 2,
      :data:`agribound.composites.gee.HLS_RECIPE_VERSION`) is not reused when
      its study area may meet one of the MGRS tiles at 180 degrees whose
      HLSS30 images Earth Engine selected everywhere except over their own
      tile (:func:`agribound.composites.gee.near_hls_reversed_footprint_tiles`):
      only there can the composite differ now. The study area is read
      without Earth Engine (a GEE asset from its local copy). This check
      runs only when nothing else differs.

    A record without ``aoi_fingerprint`` (agribound <= 1.0.0) is accepted,
    with a WARNING that the study-area file (or local raster) could not be
    verified; so is a record whose fingerprint cannot be compared because
    the file cannot be read now. A TESSERA output is accepted, with a
    WARNING, when the store's current state cannot be determined and the
    cached raster does not show another one; so is an HLS output made before
    HLS recipe 2 whose study area cannot be read without Earth Engine. These
    WARNINGs are logged only when the output is reused.

    Parameters
    ----------
    record : dict
        Provenance record of a successful run (:func:`read_provenance`).
    config : AgriboundConfig
        Configuration of the new run.
    output : str, Path or None
        The existing output, named in the warnings.
    unverified : list or None
        When given, a TESSERA output whose store state cannot be checked is
        noted here (``"<output>: <why>"``) instead of in a WARNING, so that a
        caller checking many outputs (:func:`agribound.hpc.tiles.tile_status`)
        can report them once.

    Returns
    -------
    str or None
        *None* when the output can be reused, else the reason as a clause
        (``"it was produced ..."``) for an error message.
    """
    from agribound._cache import aoi_fingerprint
    from agribound._results import results_versions
    from agribound._version import __version__

    current_hash = config_hash(config)
    if record.get("config_hash") != current_hash:
        return (
            f"it was produced with a different configuration (config_hash "
            f"{str(record.get('config_hash'))[:12]} != {current_hash[:12]})"
        )

    facts = record.get("facts") or {}
    version = record.get("agribound_version")
    made_by = f"agribound {version}" if version else "an unknown agribound version"
    target = "the existing output" + (f" {str(output)!r}" if output is not None else "")
    reasons: list[str] = []
    # WARNINGs that say the output is reused: logged only if it is (no reason below).
    warnings: list[tuple[str, tuple[Any, ...]]] = []

    current_versions = results_versions(config)
    recorded_versions = facts.get("results_versions")
    if not isinstance(recorded_versions, dict):
        recorded_versions = {}
    changed = {
        name: (recorded_versions.get(name, 1), current)
        for name, current in current_versions.items()
        if recorded_versions.get(name, 1) != current
    }
    if changed:
        changes = ", ".join(f"{name} {old} -> {new}" for name, (old, new) in changed.items())
        reasons.append(
            f"it was produced by {made_by}, whose results for this configuration differ from "
            f"those of agribound {__version__} (results versions of agribound._results: "
            f"{changes})"
        )

    checked = _fingerprinted_input(config)
    recorded_aoi = facts.get("aoi_fingerprint")
    if checked is not None and recorded_aoi is None:
        warnings.append(
            (
                "The provenance record of %s (written by %s) has no study-area fingerprint, so "
                "it cannot be verified that the output was made from the current %s; reusing "
                "it. Pass overwrite=True (CLI: --overwrite) if the file has changed since that "
                "run.",
                (target, made_by, checked),
            )
        )
    elif checked is not None:
        study_area = str(getattr(config, "study_area", "") or "").strip()
        try:
            if not study_area and not Path(config.local_tif_path).expanduser().exists():
                raise FileNotFoundError(f"{config.local_tif_path} does not exist")
            current_aoi = aoi_fingerprint(config)
        except Exception as exc:
            current_aoi = None
            warnings.append(
                (
                    "Could not read the %s to check that %s was made from it (%s: %s); "
                    "reusing the output without that check",
                    (checked, target, type(exc).__name__, exc),
                )
            )
        if current_aoi is not None and current_aoi != recorded_aoi:
            if study_area:
                what = "for a different study area: the geometry in the"
                how = "study-area fingerprint"
            else:
                what = "from a different local raster: the path, size or modification time of the"
                how = "fingerprint"
            reasons.append(
                f"it was produced {what} {checked} differs from the one recorded for that run "
                f"({how} {recorded_aoi} -> {current_aoi})"
            )
    if not reasons and str(getattr(config, "source", "") or "") == "tessera-embedding":
        reason = _tessera_store_mismatch(facts, config, target, warnings, unverified)
        if reason:
            reasons.append(reason)
    if not reasons and str(getattr(config, "source", "") or "") == "hls":
        reason = _hls_footprint_mismatch(facts, config, target, warnings)
        if reason:
            reasons.append(reason)
    if reasons:
        return " and ".join(reasons)
    for message, args in warnings:
        logger.warning(message, *args)
    return None


def _study_area_4326_offline(config: Any) -> Any:
    """The study area of *config* in EPSG:4326 (one geometry), read without Earth Engine.

    Files, ``bbox:`` strings and WKT are read with
    :func:`agribound.io.vector.read_study_area`; a GEE asset only from its
    local copy (:func:`agribound.io.vector.study_area_cache_file`), which
    :func:`agribound.io.vector.read_config_study_area` writes when it reads the
    asset.

    Raises
    ------
    FileNotFoundError
        For a GEE asset without a local copy (reading it needs Earth Engine).
    ValueError
        For a local copy that cannot be read, or a study area without geometry.
    """
    import geopandas as gpd
    from shapely.geometry import shape

    from agribound.io.vector import read_study_area, study_area_cache_file

    copy = study_area_cache_file(config)
    if copy is None:
        gdf = read_study_area(config.study_area)
    elif not copy.exists():
        raise FileNotFoundError(
            f"the GEE asset has no local copy ({copy}), and reading it needs Earth Engine"
        )
    else:
        try:
            features = json.loads(copy.read_text())["features"]
            geometries = [shape(f["geometry"]) for f in features if f.get("geometry")]
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"the local copy of the GEE asset ({copy}) cannot be read") from exc
        gdf = gpd.GeoDataFrame(geometry=geometries, crs="EPSG:4326")
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    geometry = gdf.geometry.dropna()
    geometry = geometry[~geometry.is_empty]
    if geometry.empty:
        raise ValueError("the study area has no geometry")
    return geometry.union_all()


def _hls_footprint_mismatch(
    facts: dict, config: Any, target: str, warnings: list[tuple[str, tuple[Any, ...]]]
) -> str | None:
    """Why an HLS output made before HLS recipe 2 at 180 degrees cannot be reused, or *None*.

    See :func:`reuse_mismatch`. Without a recorded
    ``AGRIBOUND_HLS_RECIPE_VERSION``, the output's composite was selected
    with the reversed HLSS30 footprints; it can differ from one selected now
    only where :func:`agribound.composites.gee.near_hls_reversed_footprint_tiles`
    holds for the study area. A study area that cannot be read without Earth
    Engine adds a WARNING to *warnings* and returns *None*.
    """
    from agribound.composites.gee import near_hls_reversed_footprint_tiles

    composite = facts.get("composite")
    composite = composite if isinstance(composite, dict) else {}
    if composite.get("AGRIBOUND_HLS_RECIPE_VERSION"):
        return None
    try:
        geometry = _study_area_4326_offline(config)
    except Exception as exc:
        warnings.append(
            (
                "Could not read the study area without Earth Engine to check whether %s, made "
                "from an HLS composite selected before HLS recipe 2, lies at the MGRS tiles at "
                "180 degrees where that selection missed images (%s: %s); reusing it. Pass "
                "overwrite=True (CLI: --overwrite) if the study area lies within 5 degrees of "
                "180 degrees.",
                (target, type(exc).__name__, exc),
            )
        )
        return None
    if not near_hls_reversed_footprint_tiles(geometry):
        return None
    return (
        "it was made from an HLS composite selected before HLS recipe 2 (its record has no "
        "AGRIBOUND_HLS_RECIPE_VERSION), and its study area lies at the MGRS tiles at 180 "
        "degrees whose HLSS30 images Earth Engine selected everywhere except over their own "
        "tile, so that composite can lack images of those tiles (see the HLS section of the "
        "satellite-sources guide)"
    )


def _tessera_store_mismatch(
    facts: dict,
    config: Any,
    target: str,
    warnings: list[tuple[str, tuple[Any, ...]]],
    unverified: list[str] | None = None,
) -> str | None:
    """Why a TESSERA output cannot be reused because of the store's state, or *None*.

    Compares the store stamp recorded in ``facts["composite"]`` with the
    store's current stamp (see :func:`reuse_mismatch`). When the current
    stamp cannot be computed, compares it with the stamp of the cached
    raster (:func:`_cached_raster_mismatch`); when that shows nothing, adds a
    WARNING to *warnings* (or a note to *unverified*) and returns *None*.
    """
    from agribound.composites import tessera_stamp
    from agribound.composites.local import (
        TESSERA_READ_BBOX_TAG,
        current_tessera_stamp,
        parse_read_bbox,
    )

    composite = facts.get("composite")
    composite = composite if isinstance(composite, dict) else {}
    recorded = composite.get(tessera_stamp.TAG_FINGERPRINT) or None
    try:
        stamp = current_tessera_stamp(config, parse_read_bbox(composite.get(TESSERA_READ_BBOX_TAG)))
    except Exception as exc:  # StoreCheckError (no network, ...) or an unreadable study area
        why = f"{type(exc).__name__}: {exc}"
        reason = _cached_raster_mismatch(facts, config, composite, recorded, why)
        if reason:
            return reason
        if unverified is not None:
            unverified.append(f"{target}: {why}")
            return None
        warnings.append(
            (
                "Could not check whether the TESSERA store has changed since %s was made (%s); "
                "reusing it. If the store has been rewritten since (as the v1 store was on "
                "2026-10-06/08), the output was made from data that the store no longer has: "
                "pass overwrite=True (CLI: --overwrite) with network access to make it again.",
                (target, why),
            )
        )
        return None
    if recorded is None:
        return (
            "its provenance record does not hold the state of the TESSERA store it was read from "
            "(it was made before agribound recorded it, or when the store could not be checked), "
            "so it may come from data that the store no longer has (the v1 store was converted "
            f"again upstream on 2026-10-06/08); the store now: {stamp.describe()}"
        )
    if recorded != stamp.fingerprint:
        before = composite.get(tessera_stamp.TAG_LAST_MODIFIED) or "unknown"
        return (
            "the TESSERA store has changed since it was made: the store's objects it was read "
            f"from differ (fingerprint {recorded[:12]}, newest modified {before}; the store "
            f"now: {stamp.describe()})"
        )
    return None


def _cached_raster_mismatch(
    facts: dict, config: Any, composite: dict, recorded: str | None, why: str
) -> str | None:
    """Why the cached TESSERA raster shows that an output is out of date, or *None*.

    Used when the store cannot be checked: the raster a run of *config* uses
    (or the one the output was made from) may have been read from the store
    again since, by a run with network access (the HPC composite stage, or
    ``agribound composite``). A stamp in its tags that differs from the
    recorded one, for the same box read, or that the record lacks, shows that
    the output was made from another state of the store. Nothing is
    requested from the store.
    """
    from agribound.composites import tessera_stamp
    from agribound.composites.local import (
        TESSERA_READ_BBOX_TAG,
        cached_tessera_tags,
        parse_read_bbox,
    )

    found = cached_tessera_tags(config, facts.get("raster_path"))
    if found is None:
        return None
    path, tags = found
    current = tags.get(tessera_stamp.TAG_FINGERPRINT) or None
    if current is None or current == recorded:
        return None
    recorded_box = parse_read_bbox(composite.get(TESSERA_READ_BBOX_TAG))
    raster_box = parse_read_bbox(tags.get(TESSERA_READ_BBOX_TAG))
    if (
        recorded_box is not None
        and raster_box is not None
        and any(abs(a - b) > 1e-9 for a, b in zip(recorded_box, raster_box, strict=True))
    ):
        return None  # stamps of different reads: they cannot be compared
    newest = tags.get(tessera_stamp.TAG_LAST_MODIFIED) or "unknown"
    raster = f"the cached TESSERA raster {str(path)!r}"
    if recorded is None:
        state = (
            f"its provenance record does not hold the state of the TESSERA store, while {raster} "
            f"does (fingerprint {current[:12]}, newest modified {newest})"
        )
    else:
        before = composite.get(tessera_stamp.TAG_LAST_MODIFIED) or "unknown"
        state = (
            f"it was made from another state of the TESSERA store than {raster} holds "
            f"(fingerprint {recorded[:12]}, newest modified {before} in its provenance record; "
            f"{current[:12]}, newest modified {newest} in the raster's tags)"
        )
    return (
        f"{state}: the raster has been read from the store again since the output was made "
        f"(the store itself could not be checked now: {why})"
    )


# ---------------------------------------------------------------------------
# Environment probes
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _git_info() -> dict[str, Any] | None:
    """Commit and dirty flag of the agribound checkout, if it is a git work tree."""
    repo = Path(__file__).resolve().parent.parent
    if not (repo / ".git").exists():
        return None
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": bool(status.strip())}


def _peak_rss_mb() -> float | None:
    """Peak resident set size of this process in MiB."""
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS, kilobytes on Linux.
        divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
        return round(peak / divisor, 1)
    except (ImportError, OSError, ValueError):
        pass
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss
        return round(peak / (1024 * 1024), 1)
    except Exception:  # pragma: no cover
        return None


def _torch_max_memory_mb() -> float | None:
    """Peak CUDA memory allocated by torch in MiB (only if torch is already imported)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:  # pragma: no cover
        return None
    return None


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

#: Most warnings kept in one record; later ones are counted, not stored.
MAX_RECORDED_WARNINGS = 200


class _WarningCollector(logging.Handler):
    """Forward WARNING (and higher) records of the ``agribound`` loggers to a recorder."""

    def __init__(self, recorder: RunRecorder) -> None:
        super().__init__(level=logging.WARNING)
        self.recorder = recorder

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed format arguments
            message = str(record.msg)
        self.recorder.add_warning(message)


class RunRecorder:
    """Collect provenance for one pipeline run.

    Use as a context manager; the block's duration and outcome
    (``"success"`` or ``"failed"`` with the error) are recorded on exit.
    The duration is recorded twice: ``wall_s`` with :func:`time.perf_counter`
    (as each step's ``wall_s``), which does not advance while the computer
    sleeps (macOS; Linux's monotonic clock also stops during suspend), and
    ``elapsed_utc_s``, the seconds from ``started_utc`` to ``finished_utc``
    by the system clock, which does (and also follows clock adjustments). A
    difference between the two is mostly time the computer slept.
    While the block runs, every WARNING (or higher) logged by an
    ``agribound`` logger (``agribound.*``, in any thread of the process) is
    added to the record's ``warnings`` as well, e.g. an engine's note that
    the input resolution is outside its training range. Identical messages
    are kept once, and at most :data:`MAX_RECORDED_WARNINGS` are stored
    (``warnings_not_recorded`` counts the rest).

    Parameters
    ----------
    config : AgriboundConfig
        Configuration of the run. Its hash and dictionary are captured when
        the recorder is created, before any stage can modify it.
    run_id : str or None
        Run identifier; a new one from :func:`agribound._repro.new_run_id` is
        generated when *None*.

    Examples
    --------
    >>> with RunRecorder(config) as rec:
    ...     with rec.step("composite"):
    ...         raster = build_composite(config)
    ...     rec.set("raster_path", raster)
    >>> write_provenance(config.output_path, rec.to_dict())
    """

    def __init__(self, config: Any, run_id: str | None = None) -> None:
        from agribound._repro import new_run_id

        self.config = config
        self.run_id = run_id or new_run_id()
        self.config_dict = to_jsonable(config.to_dict())
        self.config_hash = config_hash(config)
        self.steps: list[dict[str, Any]] = []
        self.facts: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.warnings_not_recorded = 0
        self._warning_handler: _WarningCollector | None = None
        self.engine_meta: dict[str, Any] = {}
        self.status = "created"
        self.error: str | None = None
        self.started_utc: str | None = None
        self.finished_utc: str | None = None
        self._t0: float | None = None
        self._wall_s: float | None = None
        # The system clock at the start and the seconds it advanced until the end; unlike
        # perf_counter (wall_s), it also advances while the computer sleeps.
        self._t0_utc: _dt.datetime | None = None
        self._elapsed_utc_s: float | None = None
        # Resource figures frozen when the run finishes (live values before that).
        self._peak_rss_mb: float | None = None
        self._torch_mem_mb: float | None = None
        self._versions: dict[str, str] | None = None

    # Context manager -----------------------------------------------------

    def __enter__(self) -> RunRecorder:
        self._t0_utc = _dt.datetime.now(_dt.UTC)
        self.started_utc = _utc_text(self._t0_utc)
        self._t0 = time.perf_counter()
        self.status = "running"
        if self._warning_handler is None:
            self._warning_handler = _WarningCollector(self)
            logging.getLogger("agribound").addHandler(self._warning_handler)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._warning_handler is not None:
            logging.getLogger("agribound").removeHandler(self._warning_handler)
            self._warning_handler = None
        finished = _dt.datetime.now(_dt.UTC)
        self.finished_utc = _utc_text(finished)
        if self._t0 is not None:
            self._wall_s = round(time.perf_counter() - self._t0, 3)
        if self._t0_utc is not None:
            self._elapsed_utc_s = round((finished - self._t0_utc).total_seconds(), 3)
        self._peak_rss_mb = _peak_rss_mb()
        self._torch_mem_mb = _torch_max_memory_mb()
        if exc is None:
            self.status = "success"
        else:
            self.status = "failed"
            self.error = f"{exc_type.__name__}: {exc}"
        return False

    # Recording -----------------------------------------------------------

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[dict[str, Any]]:
        """Time a pipeline step.

        Yields the step's record (a dict) so callers can attach details.
        A failing step is recorded with ``status="failed"`` and the error is
        re-raised.
        """
        record: dict[str, Any] = {"name": name, "started_utc": _utc_now(), "status": "running"}
        self.steps.append(record)
        t0 = time.perf_counter()
        try:
            yield record
        except BaseException as exc:
            record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        else:
            record["status"] = "success"
        finally:
            record["wall_s"] = round(time.perf_counter() - t0, 3)

    def set(self, key: str, value: Any) -> None:
        """Record a fact (counts, backend, weights, dataset, ...)."""
        self.facts[str(key)] = to_jsonable(value)

    def add_warning(self, msg: str) -> None:
        """Record a warning (a message already recorded is not added again).

        Inside the ``with`` block, WARNING records of the ``agribound``
        loggers are recorded automatically, so a caller that also logs the
        message does not create a duplicate.
        """
        text = str(msg)
        if text in self.warnings:
            return
        if len(self.warnings) >= MAX_RECORDED_WARNINGS:
            self.warnings_not_recorded += 1
            return
        self.warnings.append(text)

    def record_engine_meta(self, meta: dict) -> None:
        """Merge engine metadata (``gdf.attrs["engine_meta"]``) into the record."""
        if not meta:
            return
        if not isinstance(meta, dict):
            meta = {"value": meta}
        self.engine_meta.update(to_jsonable(meta))

    # Output --------------------------------------------------------------

    def _device(self) -> str | None:
        try:
            return self.config.resolve_device()
        except Exception:  # pragma: no cover
            return None

    def to_dict(self) -> dict[str, Any]:
        """Return the provenance record as a JSON-serialisable dictionary."""
        from agribound._repro import collect_versions
        from agribound._version import __version__

        wall_s = self._wall_s
        if wall_s is None and self._t0 is not None:
            wall_s = round(time.perf_counter() - self._t0, 3)
        elapsed_utc_s = self._elapsed_utc_s
        if elapsed_utc_s is None and self._t0_utc is not None:
            elapsed_utc_s = round((_dt.datetime.now(_dt.UTC) - self._t0_utc).total_seconds(), 3)
        finished = self.finished_utc is not None
        if self._versions is None:
            self._versions = collect_versions()
        env = {k: os.environ[k] for k in _ENV_KEYS if k in os.environ}
        record = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "agribound_version": __version__,
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "config_hash": self.config_hash,
            "seed": getattr(self.config, "seed", None),
            "config": self.config_dict,
            "versions": dict(self._versions),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
            "python": sys.version,
            "device": self._device(),
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "wall_s": wall_s,
            "elapsed_utc_s": elapsed_utc_s,
            "peak_rss_mb": self._peak_rss_mb if finished else _peak_rss_mb(),
            "torch_max_memory_mb": self._torch_mem_mb if finished else _torch_max_memory_mb(),
            "steps": list(self.steps),
            "facts": dict(self.facts),
            "warnings": list(self.warnings),
            "warnings_not_recorded": self.warnings_not_recorded,
            "engine_meta": dict(self.engine_meta),
            "gee_workload_tag": getattr(self.config, "gee_workload_tag", None),
            "git": _git_info(),
            "environment": env,
        }
        return to_jsonable(record)


__all__ = [
    "DEFAULTS_OF_1_0_1",
    "HASH_CONDITIONAL_FIELDS",
    "HASH_EXCLUDED_FIELDS",
    "MAX_RECORDED_WARNINGS",
    "PROVENANCE_SCHEMA_VERSION",
    "RunRecorder",
    "canonical_config",
    "config_hash",
    "drop_inapplicable_fields",
    "provenance_path",
    "read_provenance",
    "reuse_facts",
    "reuse_mismatch",
    "to_jsonable",
    "write_provenance",
]
