"""
Content-addressed cache keys for intermediate files.

Every module that writes an intermediate artefact (composites, window
composites, embeddings, engine inputs/outputs, fine-tuning data, LULC rasters)
names it with :func:`cache_path`, so that runs over different study areas,
years, date ranges or compositing settings never reuse each other's files,
even when they share one cache directory.

The key is a 12-character SHA-1 prefix over :data:`CACHE_SCHEMA_VERSION`, a
fingerprint of the study area (:func:`aoi_fingerprint`), the source, year and
date range, the compositing and export settings, source-specific options and
any extra ``parts`` supplied by the caller (for example a model name).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CACHE_SCHEMA_VERSION: str = "2"
"""Bump when radiometry or export semantics change, to invalidate old caches."""

_KEY_LENGTH = 12
_COORD_PRECISION = 1e-7

# (study_area string, mtime_ns or None, size or None) -> fingerprint
_AOI_FINGERPRINT_CACHE: dict[tuple[str, int | None, int | None], str] = {}


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _file_signature(path: str) -> tuple[int | None, int | None]:
    try:
        st = os.stat(path)
    except OSError:
        return None, None
    return st.st_mtime_ns, st.st_size


def _path_exists(text: str) -> bool:
    # Long WKT strings can exceed the OS file-name limit (OSError: ENAMETOOLONG).
    try:
        return Path(text).exists()
    except (OSError, ValueError):
        return False


def _is_gee_asset(study_area: str) -> bool:
    return study_area.startswith(("projects/", "users/"))


def gee_asset_fingerprint(asset_id: str) -> str:
    """Return the 12-hex fingerprint of a GEE asset ID, as used by :func:`aoi_fingerprint`.

    The asset is not read: the fingerprint depends on the ID string only.
    """
    return _sha1(f"gee-asset:{asset_id}")[:_KEY_LENGTH]


def aoi_fingerprint(config: Any) -> str:
    """Return a 12-hex SHA-1 fingerprint of the configured study area.

    - GEE asset IDs are fingerprinted by the asset ID string (the asset is not
      downloaded; :func:`gee_asset_fingerprint`). Cached files therefore do
      not change when the asset's features change under the same ID; the
      local copy of the asset kept by
      :func:`agribound.io.vector.read_config_study_area` has the same key.
    - Files, ``"bbox:..."`` strings and WKT are read, reprojected to
      EPSG:4326, unioned, snapped to a 1e-7 degree grid, normalised, and the
      2-D WKB is hashed. Results are memoised per path and modification time.
    - An empty study area (``source="local"`` without clipping) is
      fingerprinted by the local raster's resolved path, size and
      modification time.

    Parameters
    ----------
    config : AgriboundConfig
        Configuration providing ``study_area`` (and ``local_tif_path``).

    Returns
    -------
    str
        12 hexadecimal characters.
    """
    study_area = str(getattr(config, "study_area", "") or "").strip()

    if not study_area:
        tif = getattr(config, "local_tif_path", None)
        if tif:
            resolved = str(Path(tif).expanduser().resolve())
            mtime, size = _file_signature(resolved)
            return _sha1(f"local-raster:{resolved}:{size}:{mtime}")[:_KEY_LENGTH]
        return _sha1("no-study-area")[:_KEY_LENGTH]

    if _is_gee_asset(study_area):
        return gee_asset_fingerprint(study_area)

    is_path = not study_area.lower().startswith("bbox:") and _path_exists(study_area)
    if is_path:
        resolved = str(Path(study_area).resolve())
        mtime, size = _file_signature(resolved)
        memo_key = (resolved, mtime, size)
    else:
        memo_key = (study_area, None, None)
    cached = _AOI_FINGERPRINT_CACHE.get(memo_key)
    if cached is not None:
        return cached

    import shapely

    from agribound.io.vector import read_study_area

    gdf = read_study_area(study_area)
    if gdf.crs is None:
        logger.warning("Study area %s has no CRS; assuming EPSG:4326", study_area)
        gdf = gdf.set_crs("EPSG:4326")
    elif not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    geom = gdf.geometry.union_all()
    geom = shapely.set_precision(geom, _COORD_PRECISION)
    geom = shapely.normalize(geom)
    wkb = shapely.to_wkb(geom, output_dimension=2, byte_order=1, include_srid=False)
    fingerprint = hashlib.sha1(wkb).hexdigest()[:_KEY_LENGTH]
    _AOI_FINGERPRINT_CACHE[memo_key] = fingerprint
    return fingerprint


def _canonical(value: Any) -> Any:
    """Convert *value* into a JSON-stable structure."""
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(str(v) for v in value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def _key_fields(config: Any, include_temporal: bool) -> list[tuple[str, Any]]:
    source = getattr(config, "source", None)
    items: list[tuple[str, Any]] = [
        ("schema", CACHE_SCHEMA_VERSION),
        ("aoi", aoi_fingerprint(config)),
        ("source", source),
    ]
    if include_temporal:
        items.append(("year", getattr(config, "year", None)))
        items.append(("date_range", getattr(config, "date_range", None)))
    items += [
        ("composite_method", getattr(config, "composite_method", None)),
        ("cloud_cover_max", getattr(config, "cloud_cover_max", None)),
        ("export_crs", getattr(config, "export_crs", None)),
        ("s2_cloud_mask", getattr(config, "s2_cloud_mask", None)),
        ("naip_resolution_m", getattr(config, "naip_resolution_m", None)),
    ]
    if getattr(config, "s2_cloud_mask", None) == "cloud_score_plus":
        items.append(("cloud_score_threshold", getattr(config, "cloud_score_threshold", None)))
    if source in ("google-embedding", "tessera-embedding"):
        items.append(("tessera_version", getattr(config, "tessera_version", None)))
        items.append(("tessera_variant", getattr(config, "tessera_variant", None)))
    if source == "google-embedding":
        items.append(
            ("google_embedding_backend", getattr(config, "google_embedding_backend", None))
        )
    if source == "usgs-naip-plus":
        items.append(("usgs_service_url", getattr(config, "usgs_service_url", None)))
        items.append(("usgs_state", getattr(config, "usgs_state", None)))
        items.append(
            ("usgs_allow_year_fallback", getattr(config, "usgs_allow_year_fallback", None))
        )
    if source == "local":
        tif = getattr(config, "local_tif_path", None)
        if tif:
            resolved = str(Path(tif).expanduser().resolve())
            mtime, size = _file_signature(resolved)
            items.append(("local_tif", f"{resolved}:{size}:{mtime}"))
    return items


def cache_key(config: Any, *parts: object, include_temporal: bool = True) -> str:
    """Return a 12-hex cache key for *config* and optional extra *parts*.

    Parameters
    ----------
    config : AgriboundConfig
        Pipeline configuration.
    *parts : object
        Additional values (model names, window labels, parameters) that
        distinguish the artefact. They are hashed as ``str(part)`` in order.
    include_temporal : bool
        When *False*, ``year`` and ``date_range`` are left out, for artefacts
        that do not depend on time.

    Returns
    -------
    str
        12 hexadecimal characters.

    Notes
    -----
    Hashed fields: :data:`CACHE_SCHEMA_VERSION`, :func:`aoi_fingerprint`,
    ``source``, ``year`` and ``date_range`` (if *include_temporal*),
    ``composite_method``, ``cloud_cover_max``, ``export_crs``,
    ``s2_cloud_mask``, ``naip_resolution_m``; ``cloud_score_threshold`` when
    Cloud Score+ masking is selected; ``tessera_version``/``tessera_variant``
    for embedding sources; ``google_embedding_backend`` for Google embeddings;
    the USGS service URL, state and year-fallback flag for USGS NAIP Plus; and
    the local raster's path, size and modification time for local sources.
    """
    payload = {
        "fields": [
            [name, _canonical(value)] for name, value in _key_fields(config, include_temporal)
        ],
        "parts": [str(p) for p in parts],
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha1(text)[:_KEY_LENGTH]


def cache_path(
    config: Any,
    stem: str,
    suffix: str,
    *parts: object,
    include_temporal: bool = True,
) -> Path:
    """Return ``config.get_working_dir() / f"{stem}_{key}{suffix}"``.

    Parameters
    ----------
    config : AgriboundConfig
        Pipeline configuration.
    stem : str
        Human-readable file name prefix (may contain sub-directories).
    suffix : str
        File suffix including the dot (e.g. ``".tif"``), or ``""`` for a
        directory name.
    *parts : object
        Extra key parts (see :func:`cache_key`).
    include_temporal : bool
        See :func:`cache_key`.

    Returns
    -------
    pathlib.Path
        Path inside the working directory; its parent directory exists.
    """
    key = cache_key(config, *parts, include_temporal=include_temporal)
    path = Path(config.get_working_dir()) / f"{stem}_{key}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def clear_fingerprint_cache() -> None:
    """Forget memoised study-area fingerprints (mainly for tests)."""
    _AOI_FINGERPRINT_CACHE.clear()


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "aoi_fingerprint",
    "cache_key",
    "cache_path",
    "clear_fingerprint_cache",
    "gee_asset_fingerprint",
]
