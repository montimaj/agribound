"""
Run provenance: what was run, with which inputs, versions and resources.

:class:`RunRecorder` collects a JSON-serialisable record of one pipeline run
(configuration and its hash, seed, package versions, platform, device, step
timings, peak memory, engine metadata, facts and warnings). The pipeline writes
it next to the output as ``<output_path>.provenance.json``
(:func:`provenance_path`) and uses :func:`reuse_mismatch` (the
:func:`config_hash`, the study-area fingerprint and the results versions of
:mod:`agribound._results`) to decide whether an existing output can be reused.
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
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Hashing and file helpers
# ---------------------------------------------------------------------------


def canonical_config(config: Any) -> dict[str, Any]:
    """Return the configuration fields that define the result (see :data:`HASH_EXCLUDED_FIELDS`)."""
    data = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    return {k: to_jsonable(v) for k, v in sorted(data.items()) if k not in HASH_EXCLUDED_FIELDS}


def config_hash(config: Any) -> str:
    """Return the SHA-1 hex digest of the canonical YAML of *config*.

    Fields listed in :data:`HASH_EXCLUDED_FIELDS` (output location and format,
    caching/provenance switches, credentials, request tuning and execution
    resources) are excluded, so configurations that differ only in these
    fields hash identically. This includes ``device``, which can change the
    polygons slightly (see :data:`HASH_EXCLUDED_FIELDS`).

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


def reuse_mismatch(record: dict, config: Any, output: str | Path | None = None) -> str | None:
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
      asset study area is covered by the configuration hash already.

    A record without ``aoi_fingerprint`` (agribound <= 1.0.0) is accepted,
    with a WARNING that the study-area file (or local raster) could not be
    verified; so is a record whose fingerprint cannot be compared because
    the file cannot be read now.

    Parameters
    ----------
    record : dict
        Provenance record of a successful run (:func:`read_provenance`).
    config : AgriboundConfig
        Configuration of the new run.
    output : str, Path or None
        The existing output, named in the warnings.

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
        if not reasons:
            logger.warning(
                "The provenance record of %s (written by %s) has no study-area fingerprint, so "
                "it cannot be verified that the output was made from the current %s; reusing "
                "it. Pass overwrite=True (CLI: --overwrite) if the file has changed since that "
                "run.",
                target,
                made_by,
                checked,
            )
    elif checked is not None:
        study_area = str(getattr(config, "study_area", "") or "").strip()
        try:
            if not study_area and not Path(config.local_tif_path).expanduser().exists():
                raise FileNotFoundError(f"{config.local_tif_path} does not exist")
            current_aoi = aoi_fingerprint(config)
        except Exception as exc:
            current_aoi = None
            if not reasons:
                logger.warning(
                    "Could not read the %s to check that %s was made from it (%s: %s); "
                    "reusing the output without that check",
                    checked,
                    target,
                    type(exc).__name__,
                    exc,
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
    return " and ".join(reasons) or None


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

    Use as a context manager; the block's wall time and outcome
    (``"success"`` or ``"failed"`` with the error) are recorded on exit.
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
        # Resource figures frozen when the run finishes (live values before that).
        self._peak_rss_mb: float | None = None
        self._torch_mem_mb: float | None = None
        self._versions: dict[str, str] | None = None

    # Context manager -----------------------------------------------------

    def __enter__(self) -> RunRecorder:
        self.started_utc = _utc_now()
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
        self.finished_utc = _utc_now()
        if self._t0 is not None:
            self._wall_s = round(time.perf_counter() - self._t0, 3)
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
    "HASH_EXCLUDED_FIELDS",
    "MAX_RECORDED_WARNINGS",
    "PROVENANCE_SCHEMA_VERSION",
    "RunRecorder",
    "canonical_config",
    "config_hash",
    "provenance_path",
    "read_provenance",
    "reuse_facts",
    "reuse_mismatch",
    "to_jsonable",
    "write_provenance",
]
