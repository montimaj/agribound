"""
Tiling of large study areas for batch (HPC) runs.

A region that is too large for one composite download or one GPU job is cut
into square tiles. Each tile is an ordinary Agribound run with its own YAML
configuration, output file and provenance record, so tiles can be executed as
independent Slurm array tasks and restarted at any time.

Workflow
--------
1. :func:`make_tiles` cuts the study area into a grid of *core* cells and
   adds a *halo* around each core.
2. :func:`write_tile_manifest` writes ``tiles.gpkg``, ``tiles.txt``, one
   configuration per tile (``tiles/<tile_id>/config.yaml``) and
   ``manifest.json``.
3. :func:`run_tile` runs one tile: ``stage="composite"`` (downloads only:
   composite or embeddings, the LULC raster for ``lulc_mode="raster"`` and
   the FTW window composites of two-window FTW models, also as ensemble
   members),
   ``stage="delineate"`` (delineation of an already staged tile) or
   ``stage="all"`` (the composite stage, then the delineation), using
   :func:`agribound.pipeline.build_composite` and
   :func:`agribound.pipeline.delineate`.
4. :func:`merge_tiles` combines the tile outputs into one file and writes a
   merged provenance summary; :func:`tile_status` reports progress.

No-data tiles
-------------
A rectangular grid over a real region contains tiles without input data:
open water, or areas outside a source's coverage (NAIP outside the US,
missing TESSERA tiles). The composite builders raise
:class:`agribound.composites.base.NoDataError` (a :class:`ValueError`) for
them: no image intersects the study-area extent, no valid pixel inside the
study area, no TESSERA / Google embedding data, no USGS NAIP Plus imagery,
a local raster that does not overlap. :func:`run_tile` recognises these
errors by their type (:func:`no_data_reason`; a :class:`ValueError` whose
message matches :data:`NO_DATA_PATTERNS` is accepted as a fallback) and
records the tile as ``"no-data"`` instead of failing: it writes a content-addressed
``nodata_<key>.json`` marker in the tile's cache directory (plus
``no_data.json`` in the tile directory) with the error, and returns
``status="no-data"``. The marker is keyed like the stage markers, so it is
reused by later runs with the same inputs and ignored after a configuration
change; ``overwrite=True`` (``--overwrite``) retries the download. Any other
error still fails the tile. :func:`merge_tiles` merges the other tiles and
lists the no-data tiles, their reasons and their core area in the summary; it
raises if no tile produced output, which usually means that the source has
no data for the year.

Grids
-----
``crs="utm"`` (default)
    The study area is split at UTM zone boundaries (6 degree longitude bands,
    ``zone = floor((lon + 180) / 6) + 1``) and at the equator. Each part is
    tiled on its own WGS 84 / UTM grid (EPSG:326xx north, 327xx south), whose
    origin is the lower-left corner of that part rounded down to whole metres.
    Tile IDs are ``"<zone><N|S>_<col>_<row>"``, e.g. ``"55S_003_012"``.
``crs="equal-area"``
    One Lambert azimuthal equal-area grid centred on the study-area centroid
    (``+proj=laea``, WGS 84). Tile IDs are ``"ea_<col>_<row>"``. Distances
    and tile sizes are close to true within about 1,000 km of the centre.

In both cases every tile's composite is exported in the UTM zone of the tile
(``export_crs="EPSG:<utm_epsg>"``) unless the base configuration sets an
explicit ``EPSG:`` code.

Cores and halos
---------------
The *core* of a tile is its grid cell; with ``clip=True`` (default) it is the
cell intersected with the study area, otherwise the full cell (always cut at
UTM zone and equator boundaries for ``crs="utm"``). The *halo* is the core's
bounding box in the grid CRS expanded by ``halo_m`` on every side. The tile's
``study_area`` is the EPSG:4326 bounding box of the halo (a ``"bbox:..."``
string), so the downloaded composite covers at least the halo.

Halo rule
---------
A field is delineated whole only if it lies completely inside the halo of the
tile that owns it. Every field that crosses a core boundary extends at most
its own size past that boundary, so the halo must exceed the largest expected
field dimension: fields larger than the halo that cross a core boundary can be
truncated. Centre pivots are about 800 m across, so the default
``halo_m=1000`` is a minimum; use more for regions with larger fields.
:func:`merge_tiles` counts kept polygons that reach the edge of their halo
(``n_reaching_halo_edge``), which flags possible truncation.

Merge rule
----------
A polygon from tile *T* is kept only if *T* owns its representative point
(:func:`shapely.point_on_surface`, computed in EPSG:4326). Ownership is
decided arithmetically from the grid definition (zone/hemisphere of the point,
then ``floor`` of its grid coordinates), which assigns every point to exactly
one grid cell; with ``clip=True`` the point must also lie in the study area.
Identical polygons delineated by two neighbouring tiles are therefore kept
exactly once. Two *different* detections of the same field (one per tile)
have different representative points and can, rarely, both be kept or both be
dropped; :func:`merge_tiles` reports overlapping polygons from different tiles
(``n_cross_tile_overlap_pairs``) so this can be checked. Study areas that
cross the antimeridian are not supported.

With ``clip=True`` this representative-point test is the only selection at
the region's study-area outline; with ``clip=False`` there is none. The base
configuration's ``aoi_selection`` is applied in each tile run to the tile's
study area, which is the tile's halo box, not to the region's outline.
Reference polygons given to :func:`merge_tiles` are
selected with the matching rule (representative point with ``clip=True``,
else intersecting the study area).
"""

from __future__ import annotations

import contextlib
import copy
import datetime as _dt
import functools
import hashlib
import json
import logging
import math
import os
import re
import socket
import time
import traceback
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import shapely
from shapely.geometry import box

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = "1"
"""Version of the ``manifest.json`` layout written by :func:`write_tile_manifest`."""

MANIFEST_FILENAME = "manifest.json"
TILES_GPKG_FILENAME = "tiles.gpkg"
TILE_LIST_FILENAME = "tiles.txt"
TILE_CONFIG_FILENAME = "config.yaml"

GRID_KINDS: tuple[str, ...] = ("utm", "equal-area")
STAGES: tuple[str, ...] = ("all", "composite", "delineate")

#: Densification step (degrees) applied to study-area outlines before projecting them.
_AOI_SEGMENT_DEG = 0.01
#: Status strings reported by :func:`tile_status`.
_DONE, _FAILED, _PENDING, _STALE = "done", "failed", "pending", "stale"
_NO_DATA = "no-data"

#: Messages of the composite builders' no-data errors
#: (``agribound/composites/gee.py``, ``local.py``, ``usgs.py``). The builders
#: raise :class:`~agribound.composites.base.NoDataError`, which
#: :func:`no_data_reason` checks first; a plain :class:`ValueError` matching
#: one of these patterns is accepted as a fallback (e.g. from a third-party
#: builder or an older error), so a tile whose error chain contains either is
#: recorded as ``"no-data"`` (see the module docstring).
NO_DATA_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        # GEE imagery (_years_error) and Google embeddings on Earth Engine.
        r"\bNo \S+ images?\b.*\bintersects? the study-area extent",
        # Downloaded composite / embedding without a valid pixel in the study area.
        r"has no valid pixels? inside the study area",
        # TESSERA embeddings.
        r"\breturned no data for \d{4}",
        # USGS NAIP Plus.
        r"has no imagery\b.*\b(?:inside the study area|over the study-area extent)",
        # Google embeddings from the Source Cooperative mirror.
        r"has no Google Satellite Embedding tile for \d{4} over the study-area extent",
        r"tiles for \d{4} do not overlap the study-area extent",
        # Local raster that does not overlap the tile.
        r"the study-area extent does not overlap",
    )
)

#: Configuration fields holding file paths that tile jobs must find from any
#: working directory (made absolute by :func:`write_tile_manifest`).
_PATH_FIELDS: tuple[str, ...] = (
    "local_tif_path",
    "gee_service_account_key",
    "reference_boundaries",
    "embedding_cache_dir",
    "sam_model",
)
#: Engine parameters holding file paths (made absolute when the file exists).
_PATH_ENGINE_PARAMS: tuple[str, ...] = ("checkpoint_path", "weights_path", "model_path")

_SLURM_ENV_KEYS = (
    "SLURM_JOB_ID",
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_JOB_NAME",
    "SLURM_CLUSTER_NAME",
    "SLURMD_NODENAME",
    "CUDA_VISIBLE_DEVICES",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha1_json(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, data: Any) -> None:
    from agribound.provenance import to_jsonable

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        json.dump(to_jsonable(data), f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def no_data_reason(exc: BaseException) -> str | None:
    """Return the no-data message in *exc*'s chain, or *None*.

    Walks ``exc``, its ``__cause__`` and ``__context__`` (e.g. the FTW
    engine's :class:`RuntimeError` wrapping a window composite's
    :class:`~agribound.composites.base.NoDataError`) and returns
    ``"<ExceptionType>: <message>"`` of the first
    :class:`~agribound.composites.base.NoDataError` in the chain; if there is
    none, of the first :class:`ValueError` whose message matches
    :data:`NO_DATA_PATTERNS` (fallback).
    """
    from agribound.composites.base import NoDataError

    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    for item in chain:
        if isinstance(item, NoDataError):
            return f"{type(item).__name__}: {item}"
    for item in chain:
        if isinstance(item, ValueError):
            message = str(item)
            if any(p.search(message) for p in NO_DATA_PATTERNS):
                return f"{type(item).__name__}: {message}"
    return None


def _looks_like_remote(value: str) -> bool:
    return "://" in value or value.startswith(("/vsi", "projects/", "users/", "gs:", "s3:"))


def _absolute_path_overrides(config: Any) -> dict[str, Any]:
    """Overrides that make relative file paths of *config* absolute (against the cwd).

    Tile jobs start in arbitrary working directories, so relative paths in the
    base configuration are resolved here, the way ``agribound delineate
    --config`` resolves them from the current directory. Existing files and
    directories are resolved; ``embedding_cache_dir`` is resolved even if it
    does not exist yet (it is created on first use). A relative path that does
    not exist is kept, with a WARNING. ``sam_model`` and the engine parameters
    in ``_PATH_ENGINE_PARAMS`` are only resolved when the file exists (they can
    also be model names).
    """
    overrides: dict[str, Any] = {}
    for name in _PATH_FIELDS:
        value = getattr(config, name, None)
        if not isinstance(value, str) or not value or _looks_like_remote(value):
            continue
        path = Path(value).expanduser()
        if path.is_absolute():
            continue
        if path.exists() or name == "embedding_cache_dir":
            overrides[name] = str(path.resolve())
        elif name != "sam_model":
            logger.warning(
                "%s=%r is a relative path that does not exist in the current directory %s; "
                "tile jobs resolve it against their own working directory.",
                name,
                value,
                os.getcwd(),
            )
    params = dict(config.engine_params or {})
    changed = False
    for name in _PATH_ENGINE_PARAMS:
        value = params.get(name)
        if isinstance(value, str) and value and not _looks_like_remote(value):
            path = Path(value).expanduser()
            if not path.is_absolute() and path.exists():
                params[name] = str(path.resolve())
                changed = True
    if changed:
        overrides["engine_params"] = params
    if overrides:
        logger.info("Tile configurations use absolute paths: %s", overrides)
    return overrides


def _runtime_context() -> dict[str, Any]:
    """Host and scheduler facts recorded in markers."""
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "environment": {k: os.environ[k] for k in _SLURM_ENV_KEYS if k in os.environ},
    }


def _transform_xy(
    transformer: pyproj.Transformer, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised ``transformer.transform`` that also accepts length-1 arrays.

    pyproj routes length-1 arrays through its scalar code path, which converts
    them to floats (a NumPy deprecation); the point is duplicated instead.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 1:
        tx, ty = transformer.transform(np.repeat(x, 2), np.repeat(y, 2))
        return np.asarray(tx, dtype=float)[:1], np.asarray(ty, dtype=float)[:1]
    tx, ty = transformer.transform(x, y)
    return np.asarray(tx, dtype=float), np.asarray(ty, dtype=float)


def _project(geom: Any, transformer: pyproj.Transformer) -> Any:
    """Apply a pyproj transformer (``always_xy=True``) to a shapely geometry."""

    def _fn(coords: np.ndarray) -> np.ndarray:
        x, y = _transform_xy(transformer, coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geom, _fn)


def _polygonal(geom: Any) -> Any:
    """Keep only the polygonal parts of *geom* (drops points/lines from intersections)."""
    if geom is None or geom.is_empty:
        return shapely.Polygon()
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    parts = [g for g in shapely.get_parts(geom) if g.geom_type in ("Polygon", "MultiPolygon")]
    if not parts:
        return shapely.Polygon()
    return shapely.union_all(parts)


def _study_area_geometry(study_area: Any, config: Any = None) -> tuple[Any, str]:
    """Return the study area as one polygonal EPSG:4326 geometry and a label.

    *config* is passed to :func:`agribound.io.vector.read_study_area`, so a
    GEE asset ID is read with the configured Earth Engine project and
    credentials.
    """
    if isinstance(study_area, gpd.GeoDataFrame | gpd.GeoSeries):
        series = study_area.geometry if isinstance(study_area, gpd.GeoDataFrame) else study_area
        if series.crs is None:
            logger.warning("Study area has no CRS; assuming EPSG:4326")
        elif not series.crs.equals("EPSG:4326"):
            series = series.to_crs("EPSG:4326")
        geom, label = shapely.union_all(series.values), "<GeoDataFrame>"
    elif isinstance(study_area, shapely.Geometry):
        geom, label = study_area, "<shapely geometry, EPSG:4326>"
    else:
        from agribound.io.vector import read_study_area

        gdf = read_study_area(str(study_area), config=config)
        if gdf.crs is None:
            logger.warning("Study area %s has no CRS; assuming EPSG:4326", study_area)
        elif not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs("EPSG:4326")
        geom, label = shapely.union_all(gdf.geometry.values), str(study_area)
    geom = _polygonal(shapely.make_valid(geom))
    if geom.is_empty or geom.area <= 0:
        raise ValueError(f"Study area {label} has no polygonal area")
    minx, miny, maxx, maxy = geom.bounds
    if not (-180 <= minx < maxx <= 180 and -90 <= miny < maxy <= 90):
        raise ValueError(
            f"Study area {label} has bounds {geom.bounds}, which are not valid EPSG:4326 "
            "longitude/latitude (study areas crossing the antimeridian are not supported)."
        )
    return geom, label


def _fmt_bbox(bounds: Sequence[float]) -> str:
    """``"bbox:minx,miny,maxx,maxy"`` rounded outwards to 1e-7 degrees."""
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    scale = 1e7
    minx = max(-180.0, math.floor(minx * scale) / scale)
    miny = max(-90.0, math.floor(miny * scale) / scale)
    maxx = min(180.0, math.ceil(maxx * scale) / scale)
    maxy = min(90.0, math.ceil(maxy * scale) / scale)
    return f"bbox:{minx:.7f},{miny:.7f},{maxx:.7f},{maxy:.7f}"


def _tile_id(system: str, col: int, row: int) -> str:
    return f"{system}_{int(col):03d}_{int(row):03d}"


def format_index_ranges(indices: Iterable[int]) -> str:
    """Compress integers into a Slurm ``--array`` style list, e.g. ``"0-3,7,9-10"``.

    Parameters
    ----------
    indices : iterable of int
        Indices (duplicates and order are ignored).

    Returns
    -------
    str
        Comma-separated ranges; empty string for no indices.
    """
    values = sorted({int(i) for i in indices})
    if not values:
        return ""
    ranges: list[str] = []
    start = prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = value
    ranges.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(ranges)


# ---------------------------------------------------------------------------
# Grid systems
# ---------------------------------------------------------------------------


def _utm_parts(aoi: Any) -> list[dict[str, Any]]:
    """Split an EPSG:4326 study area into (UTM zone, hemisphere) parts."""
    from agribound.io.crs import utm_epsg, utm_zones_for_bounds

    minx, miny, maxx, maxy = aoi.bounds
    parts = []
    for zone in utm_zones_for_bounds((minx, miny, maxx, maxy)):
        west = -180.0 + 6.0 * (zone - 1)
        east = west + 6.0
        for hemisphere, (lat_lo, lat_hi) in (("S", (-90.0, 0.0)), ("N", (0.0, 90.0))):
            if maxy < lat_lo or miny >= lat_hi:
                continue
            band = box(west, lat_lo, east, lat_hi)
            part = _polygonal(shapely.intersection(aoi, band))
            if part.is_empty or part.area <= 0:
                continue
            parts.append(
                {
                    "system": f"{zone:02d}{hemisphere}",
                    "zone": zone,
                    "hemisphere": hemisphere,
                    "epsg": utm_epsg(zone, south=hemisphere == "S"),
                    "part": part,
                    "band": band,
                }
            )
    return parts


def _laea_definition(aoi: Any) -> str:
    """PROJ definition of the Lambert azimuthal equal-area grid centred on *aoi*."""
    centroid = aoi.centroid
    return (
        f"+proj=laea +lat_0={centroid.y:.4f} +lon_0={centroid.x:.4f} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def _grid_cells(
    region: Any, origin: tuple[float, float], size: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(cols, rows, boxes)`` of grid cells intersecting *region* (grid CRS).

    An overhang of less than ``1e-6 * size`` past a grid line (projection
    round-off, e.g. 2 cm for 20 km tiles) does not start a new column or row.
    """
    x0, y0 = origin
    _minx, _miny, maxx, maxy = region.bounds
    ncols = max(1, math.ceil((maxx - x0) / size - 1e-6))
    nrows = max(1, math.ceil((maxy - y0) / size - 1e-6))
    cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows), indexing="xy")
    cols, rows = cols.ravel(), rows.ravel()
    boxes = shapely.box(
        x0 + cols * size, y0 + rows * size, x0 + (cols + 1) * size, y0 + (rows + 1) * size
    )
    shapely.prepare(region)
    hit = shapely.intersects(region, boxes)
    return cols[hit], rows[hit], boxes[hit]


# ---------------------------------------------------------------------------
# make_tiles
# ---------------------------------------------------------------------------


def make_tiles(
    study_area: Any,
    tile_size_m: float = 20000.0,
    halo_m: float = 1000.0,
    crs: str = "utm",
    clip: bool = True,
    config: Any = None,
) -> gpd.GeoDataFrame:
    """Cut a study area into square tiles with halos.

    Parameters
    ----------
    study_area : str, GeoDataFrame, GeoSeries or shapely geometry
        Anything :func:`agribound.io.vector.read_study_area` accepts (vector
        file, ``"bbox:minx,miny,maxx,maxy"``, WKT, GEE asset ID), or an
        in-memory geometry (a bare shapely geometry is taken as EPSG:4326).
    tile_size_m : float
        Edge length of the core cells in metres of the grid CRS (default
        20,000).
    halo_m : float
        Margin added around each core's bounding box (default 1,000 m). See
        the halo rule in the module docstring.
    crs : str
        ``"utm"`` (per-zone UTM grids, default) or ``"equal-area"`` (one
        Lambert azimuthal equal-area grid centred on the study area).
    clip : bool
        Intersect the cores with the study area (default *True*). With
        *False* the cores are whole grid cells (still cut at UTM zone and
        equator boundaries for ``crs="utm"``).
    config : AgriboundConfig or None
        Base configuration; used only to read a GEE-asset study area with the
        configured Earth Engine project and credentials
        (:func:`agribound.auth.ensure_gee`).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per tile in EPSG:4326, ordered by grid system, row and column,
        with columns ``index``, ``tile_id``, ``system``, ``grid_crs``,
        ``zone`` and ``hemisphere`` (UTM grids; *None* otherwise), ``col``,
        ``row``, ``utm_epsg`` (EPSG code of the tile's UTM zone, used as the
        export CRS), ``core_area_km2`` (in the grid CRS), ``halo_minx``,
        ``halo_miny``, ``halo_maxx``, ``halo_maxy`` (halo box in the grid
        CRS), ``bounds`` (EPSG:4326 bounds of the halo), ``study_area`` (the
        same bounds as a ``"bbox:..."`` string), the core as the active
        ``geometry`` and the halo as a second geometry column ``halo``.
        ``attrs["grid"]`` holds the grid definition used by
        :func:`assign_tile_ids`, ``attrs["study_area_geometry"]`` the unioned
        study area and ``attrs["study_area_label"]`` its description.

    Raises
    ------
    ValueError
        For invalid sizes, an unknown *crs*, or an empty study area.
    """
    tile_size_m = float(tile_size_m)
    halo_m = float(halo_m)
    if not math.isfinite(tile_size_m) or tile_size_m <= 0:
        raise ValueError(f"tile_size_m must be > 0, got {tile_size_m}")
    if not math.isfinite(halo_m) or halo_m < 0:
        raise ValueError(f"halo_m must be >= 0, got {halo_m}")
    crs = str(crs).lower().strip()
    if crs not in GRID_KINDS:
        raise ValueError(f"crs must be one of {GRID_KINDS}, got {crs!r}")
    if halo_m < 1000:
        logger.warning(
            "halo_m=%.0f m: fields larger than the halo that cross a core boundary can be "
            "truncated (centre pivots are ~800 m across).",
            halo_m,
        )
    if halo_m > tile_size_m:
        logger.warning(
            "halo_m=%.0f m exceeds tile_size_m=%.0f m: each composite covers >9x its core.",
            halo_m,
            tile_size_m,
        )

    from agribound.io.crs import get_utm_crs

    aoi, label = _study_area_geometry(study_area, config=config)

    if crs == "utm":
        parts = _utm_parts(aoi)
    else:
        parts = [
            {
                "system": "ea",
                "zone": None,
                "hemisphere": None,
                "crs_string": _laea_definition(aoi),
                "part": aoi,
            }
        ]

    systems: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    halos_4326: list[Any] = []
    seg_m = max(10.0, tile_size_m / 20.0)

    for part in parts:
        crs_string = f"EPSG:{part['epsg']}" if crs == "utm" else part["crs_string"]
        grid_crs = pyproj.CRS.from_user_input(crs_string)
        to_grid = pyproj.Transformer.from_crs("EPSG:4326", grid_crs, always_xy=True)
        to_4326 = pyproj.Transformer.from_crs(grid_crs, "EPSG:4326", always_xy=True)
        part_dense = shapely.segmentize(part["part"], _AOI_SEGMENT_DEG)
        part_grid = shapely.make_valid(_project(part_dense, to_grid))
        # Whole-metre origin; the 1e-6 m slack absorbs reprojection round-off so that
        # an AOI starting at 600000 m does not get its origin at 599999 m.
        origin = (
            math.floor(part_grid.bounds[0] + 1e-6),
            math.floor(part_grid.bounds[1] + 1e-6),
        )

        if clip:
            region = part_grid
        elif crs == "utm":
            # Whole cells, cut only at the zone/equator boundaries.
            minx, miny, maxx, maxy = part["part"].bounds
            lat_abs = min(89.0, max(abs(miny), abs(maxy)))
            margin = 2.0 * tile_size_m / (111_320.0 * max(math.cos(math.radians(lat_abs)), 0.05))
            band = part["band"]
            window = box(
                max(band.bounds[0], minx - margin),
                max(band.bounds[1], miny - margin),
                min(band.bounds[2], maxx + margin),
                min(band.bounds[3], maxy + margin),
            )
            region = shapely.make_valid(
                _project(shapely.segmentize(window, _AOI_SEGMENT_DEG), to_grid)
            )
        else:
            region = None

        cols, rows, cells = _grid_cells(part_grid, origin, tile_size_m)
        if region is None:
            cores = cells
        else:
            cores = np.array([_polygonal(g) for g in shapely.intersection(cells, region)])
        systems[part["system"]] = {
            "crs": crs_string,
            "zone": part["zone"],
            "hemisphere": part["hemisphere"],
            "origin": [float(origin[0]), float(origin[1])],
        }

        for col, row, core in zip(cols, rows, cores, strict=True):
            if core.is_empty or core.area <= 0:
                continue
            cx0, cy0, cx1, cy1 = core.bounds
            halo = box(cx0 - halo_m, cy0 - halo_m, cx1 + halo_m, cy1 + halo_m)
            core_4326 = _polygonal(
                shapely.make_valid(_project(shapely.segmentize(core, seg_m), to_4326))
            )
            halo_4326 = _project(shapely.segmentize(halo, seg_m), to_4326)
            if crs == "utm":
                utm_epsg = int(part["epsg"])
            else:
                c = core_4326.centroid
                utm_epsg = int(get_utm_crs(c.x, c.y).to_epsg())
            hb = tuple(float(v) for v in halo_4326.bounds)
            records.append(
                {
                    "tile_id": _tile_id(part["system"], col, row),
                    "system": part["system"],
                    "grid_crs": systems[part["system"]]["crs"],
                    "zone": part["zone"],
                    "hemisphere": part["hemisphere"],
                    "col": int(col),
                    "row": int(row),
                    "utm_epsg": utm_epsg,
                    "core_area_km2": float(core.area) / 1e6,
                    "halo_minx": float(halo.bounds[0]),
                    "halo_miny": float(halo.bounds[1]),
                    "halo_maxx": float(halo.bounds[2]),
                    "halo_maxy": float(halo.bounds[3]),
                    "bounds": hb,
                    "study_area": _fmt_bbox(hb),
                    "geometry": core_4326,
                }
            )
            halos_4326.append(halo_4326)

    if not records:
        raise ValueError(f"No tiles intersect the study area {label}")

    order = sorted(
        range(len(records)),
        key=lambda i: (records[i]["system"], records[i]["row"], records[i]["col"]),
    )
    records = [records[i] for i in order]
    halos_4326 = [halos_4326[i] for i in order]
    for i, rec in enumerate(records):
        rec["index"] = i

    columns = ["index", "tile_id"] + [c for c in records[0] if c not in ("index", "tile_id")]
    frame = pd.DataFrame.from_records(records, columns=columns)
    tiles = gpd.GeoDataFrame(frame, geometry="geometry", crs="EPSG:4326")
    tiles["halo"] = gpd.GeoSeries(halos_4326, crs="EPSG:4326", index=tiles.index)
    tiles.attrs["grid"] = {
        "kind": crs,
        "tile_size_m": tile_size_m,
        "halo_m": halo_m,
        "clip": bool(clip),
        "systems": systems,
    }
    tiles.attrs["study_area_geometry"] = aoi
    tiles.attrs["study_area_label"] = label
    logger.info(
        "make_tiles: %d tiles (%s grid, %.0f m cores, %.0f m halo) over %s",
        len(tiles),
        crs,
        tile_size_m,
        halo_m,
        label,
    )
    return tiles


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------


def representative_points(geometries: gpd.GeoSeries) -> np.ndarray:
    """Return :func:`shapely.point_on_surface` of each geometry, computed in EPSG:4326."""
    series = geometries
    if series.crs is not None and not series.crs.equals("EPSG:4326"):
        series = series.to_crs("EPSG:4326")
    return shapely.point_on_surface(np.asarray(series.values, dtype=object))


def assign_tile_ids(points_4326: Any, grid: dict[str, Any]) -> np.ndarray:
    """Return the ID of the grid cell that owns each EPSG:4326 point.

    The cell is found arithmetically: for UTM grids the point's zone
    (``floor((lon + 180) / 6) + 1``) and hemisphere (``lat < 0`` is south)
    select the grid system, then ``col = floor((x - x0) / size)`` and
    ``row = floor((y - y0) / size)`` in that system's CRS. Every point is
    assigned to exactly one cell; cells on a shared edge belong to the cell
    on their east/north side.

    Parameters
    ----------
    points_4326 : array-like of shapely Points
        Points in EPSG:4326 (e.g. from :func:`representative_points`).
    grid : dict
        Grid definition (``tiles.attrs["grid"]`` or ``manifest["grid"]``).

    Returns
    -------
    numpy.ndarray
        Object array of tile IDs; *None* for empty points or points outside
        every grid system. An ID is returned whether or not a tile with that
        ID exists; callers compare it with their tile's ID.
    """
    points = np.asarray(points_4326, dtype=object)
    result = np.full(points.shape, None, dtype=object)
    if points.size == 0:
        return result
    valid = ~shapely.is_empty(points) & ~shapely.is_missing(points)
    xs = np.full(points.shape, np.nan)
    ys = np.full(points.shape, np.nan)
    xs[valid] = shapely.get_x(points[valid])
    ys[valid] = shapely.get_y(points[valid])

    systems = grid["systems"]
    keys = np.full(points.shape, None, dtype=object)
    if grid["kind"] == "utm":
        # Same formula as agribound.io.crs.utm_zone_for_lon, vectorised.
        wrapped = np.mod(xs[valid] + 180.0, 360.0) - 180.0
        zones = np.mod(np.floor((wrapped + 180.0) / 6.0).astype(np.int64), 60) + 1
        hemis = np.where(ys[valid] < 0, "S", "N")
        keys[valid] = [f"{z:02d}{h}" for z, h in zip(zones, hemis, strict=True)]
    else:
        keys[valid] = "ea"

    size = float(grid["tile_size_m"])
    for system, info in systems.items():
        mask = keys == system
        if not mask.any():
            continue
        transformer = pyproj.Transformer.from_crs("EPSG:4326", info["crs"], always_xy=True)
        gx, gy = _transform_xy(transformer, xs[mask], ys[mask])
        x0, y0 = info["origin"]
        cols = np.floor((gx - x0) / size).astype(np.int64)
        rows = np.floor((gy - y0) / size).astype(np.int64)
        result[mask] = [_tile_id(system, c, r) for c, r in zip(cols, rows, strict=True)]
    return result


def cell_box(grid: dict[str, Any], system: str, col: int, row: int) -> Any:
    """Return the full grid cell ``(system, col, row)`` as a box in the system's CRS."""
    size = float(grid["tile_size_m"])
    x0, y0 = grid["systems"][system]["origin"]
    return box(x0 + col * size, y0 + row * size, x0 + (col + 1) * size, y0 + (row + 1) * size)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _as_config(base_config: Any) -> Any:
    from agribound.config import AgriboundConfig

    if isinstance(base_config, AgriboundConfig):
        return copy.deepcopy(base_config)
    if isinstance(base_config, dict):
        return AgriboundConfig.from_dict(base_config)
    return AgriboundConfig.from_yaml(base_config)


def _estimate_tile_raster_mb(config: Any, tiles: gpd.GeoDataFrame) -> float | None:
    """Rough size (MB) of the largest tile composite, from the source registry."""
    from agribound.registry import SOURCE_REGISTRY

    info = SOURCE_REGISTRY.get(config.source, {})
    res = config.naip_resolution_m if config.source == "naip" else info.get("resolution_m")
    bands = info.get("all_bands")
    if not res or not bands:
        return None
    width = float((tiles["halo_maxx"] - tiles["halo_minx"]).max())
    height = float((tiles["halo_maxy"] - tiles["halo_miny"]).max())
    nbytes = 1 if info.get("value_scale") == "uint8" else 4
    return round(width / res * height / res * len(bands) * nbytes / 1e6, 1)


def write_tile_manifest(
    tiles: gpd.GeoDataFrame,
    base_config: Any,
    out_dir: str | Path,
    *,
    cache_root: str | Path | None = None,
    overwrite: bool = False,
    keep_reference: bool = False,
    allow_fine_tune_per_tile: bool = False,
) -> Path:
    """Write the tile manifest, per-tile configurations and ``tiles.gpkg``.

    Files written under *out_dir*:

    - ``tiles.gpkg`` with layers ``tiles`` (cores), ``halos`` and
      ``study_area`` (EPSG:4326);
    - ``tiles.txt``: one tile ID per line, in index order (line ``i + 1`` is
      tile ``i``, i.e. Slurm array index ``i``);
    - ``tiles/<tile_id>/config.yaml``: the base configuration with
      ``study_area`` set to the tile's halo bbox, ``output_path`` set to
      ``tiles/<tile_id>/fields.<ext>``, ``cache_dir`` set to
      ``<cache_root>/<tile_id>``, ``export_crs`` set to the tile's UTM EPSG
      code when the base uses ``"utm"``, ``overwrite=False`` and
      ``provenance=True`` (both needed for idempotent restarts; a base value
      that differs is replaced with a WARNING), and relative paths of the
      base configuration made absolute against the current directory, so
      tile jobs can start in any directory: ``local_tif_path``,
      ``gee_service_account_key``, ``reference_boundaries``, ``sam_model``
      and the engine parameters ``checkpoint_path``, ``weights_path``,
      ``model_path`` when they name an existing file or directory (a
      relative ``local_tif_path``, key or reference that does not exist is
      kept, with a WARNING), and ``embedding_cache_dir`` always;
    - ``manifest.json`` (paths relative to *out_dir*).

    Writing is idempotent: an existing manifest with the same signature
    (output directory, grid, tile entries, base configuration and cache
    root) is kept, and a different one is refused unless *overwrite*.

    Parameters
    ----------
    tiles : geopandas.GeoDataFrame
        Output of :func:`make_tiles`.
    base_config : AgriboundConfig, dict or path
        Configuration shared by all tiles (a YAML path is loaded with
        :meth:`AgriboundConfig.from_yaml`). Its ``study_area`` and
        ``output_path`` are replaced per tile.
    out_dir : str or Path
        Directory for the manifest and all tile outputs.
    cache_root : str, Path or None
        Parent of the per-tile cache directories. Default: the base
        configuration's ``cache_dir`` if set, else ``<out_dir>/tiles``
        (i.e. ``tiles/<tile_id>/cache``). Tile IDs depend only on the study
        area and tiling parameters, so manifests for several engines over the
        same source and year can share one *cache_root* and hence one
        download per tile -- provided only one job writes a given tile's
        cache at a time (see ``examples/hpc/README.md``).
    overwrite : bool
        Replace an existing manifest that differs from the new one. Tile
        outputs are never deleted; outputs whose configuration changed are
        protected by their provenance hash.
    keep_reference : bool
        Keep ``reference_boundaries`` in the tile configurations. By default
        it is removed (with an INFO message), because per-tile evaluation
        against the reference polygons that intersect each halo double-counts
        fields; evaluate the merged output instead (``merge_tiles(...,
        reference=...)``).
    allow_fine_tune_per_tile : bool
        Allow ``fine_tune=True``, which fine-tunes a separate model in every
        tile on the reference polygons inside that tile. Refused by default:
        fine-tune once, then pass the checkpoint with
        ``engine_params["checkpoint_path"]``.

    Returns
    -------
    pathlib.Path
        Path of ``manifest.json``.

    Raises
    ------
    FileExistsError
        If a different manifest exists in *out_dir* and *overwrite* is False.
    ValueError
        If *tiles* did not come from :func:`make_tiles`, or the base
        configuration fine-tunes and *allow_fine_tune_per_tile* is False.
    """
    from agribound._version import __version__
    from agribound.provenance import config_hash, to_jsonable

    grid = tiles.attrs.get("grid")
    if not grid or len(tiles) == 0:
        raise ValueError("tiles must be a non-empty GeoDataFrame returned by make_tiles()")
    config = _as_config(base_config)
    path_overrides = _absolute_path_overrides(config)
    if path_overrides:
        config = config.merged(**path_overrides)
    if config.fine_tune and not allow_fine_tune_per_tile:
        raise ValueError(
            "The base configuration has fine_tune=True, which would fine-tune a separate model "
            "in every tile. Fine-tune once (e.g. 'agribound delineate --fine-tune --reference "
            "REF' over the reference area), then pass the checkpoint to the tiles with "
            "engine_params checkpoint_path=<path> (CLI: --engine-param checkpoint_path=...). "
            "Pass allow_fine_tune_per_tile=True (--allow-fine-tune-per-tile) to override."
        )

    out_dir = Path(out_dir).expanduser().resolve()
    tiles_dir = out_dir / "tiles"
    if cache_root is None:
        cache_root_path = (
            Path(config.cache_dir).expanduser().resolve() if config.cache_dir else None
        )
    else:
        cache_root_path = Path(cache_root).expanduser().resolve()

    if config.overwrite:
        logger.warning(
            "Tile configurations use overwrite=False (base had overwrite=True); "
            "use 'agribound tiles run --overwrite' to recompute tiles."
        )
    if not config.provenance:
        logger.warning(
            "Tile configurations use provenance=True (base had provenance=False): "
            "provenance records are required to skip finished tiles on restart."
        )
    drop_reference = (
        bool(config.reference_boundaries) and not keep_reference and not config.fine_tune
    )
    if drop_reference:
        logger.info(
            "reference_boundaries removed from the tile configurations; evaluate the merged "
            "output instead ('agribound tiles merge --reference %s').",
            config.reference_boundaries,
        )

    ext = config.get_output_extension()
    tile_entries: list[dict[str, Any]] = []
    tile_configs: list[Any] = []
    for row in tiles.itertuples(index=False):
        tile_dir = tiles_dir / row.tile_id
        cache_dir = (cache_root_path / row.tile_id) if cache_root_path else (tile_dir / "cache")
        overrides: dict[str, Any] = {
            "study_area": row.study_area,
            "output_path": str(tile_dir / f"fields{ext}"),
            "cache_dir": str(cache_dir),
            "overwrite": False,
            "provenance": True,
        }
        if config.export_crs == "utm":
            overrides["export_crs"] = f"EPSG:{int(row.utm_epsg)}"
        if drop_reference:
            overrides["reference_boundaries"] = None
        tile_config = config.merged(**overrides)
        tile_configs.append(tile_config)
        rel_dir = tile_dir.relative_to(out_dir)
        tile_entries.append(
            {
                "index": int(row.index),
                "tile_id": row.tile_id,
                "system": row.system,
                "col": int(row.col),
                "row": int(row.row),
                "utm_epsg": int(row.utm_epsg),
                "core_area_km2": round(float(row.core_area_km2), 6),
                "halo_grid_bounds": [
                    float(row.halo_minx),
                    float(row.halo_miny),
                    float(row.halo_maxx),
                    float(row.halo_maxy),
                ],
                "study_area": row.study_area,
                "dir": str(rel_dir),
                "config": str(rel_dir / TILE_CONFIG_FILENAME),
                "output": str(rel_dir / f"fields{ext}"),
                "config_hash": config_hash(tile_config),
            }
        )

    base_dict = to_jsonable(config.to_dict())
    # cache_dir is excluded from the configuration hash, so the cache root is
    # part of the signature explicitly (a different root must not be ignored).
    signature = _sha1_json(
        {
            "out_dir": str(out_dir),
            "grid": grid,
            "tiles": tile_entries,
            "base": base_dict,
            "cache_root": str(cache_root_path) if cache_root_path else None,
        }
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "agribound_version": __version__,
        "created_utc": _utc_now(),
        "signature": signature,
        "study_area": tiles.attrs.get("study_area_label"),
        "grid": grid,
        "base_config": base_dict,
        "base_config_hash": config_hash(config),
        "output_format": config.output_format,
        "cache_root": str(cache_root_path) if cache_root_path else None,
        "reference_boundaries": config.reference_boundaries,
        "approx_tile_raster_mb": _estimate_tile_raster_mb(config, tiles),
        "n_tiles": len(tile_entries),
        "files": {
            "tiles_gpkg": TILES_GPKG_FILENAME,
            "tile_list": TILE_LIST_FILENAME,
        },
        "tiles": tile_entries,
    }

    manifest_path = out_dir / MANIFEST_FILENAME
    existing = _read_json(manifest_path) if manifest_path.exists() else None
    if existing is not None and existing.get("signature") == signature:
        logger.info("Manifest %s is up to date (%d tiles)", manifest_path, len(tile_entries))
        _write_tile_configs(out_dir, tile_entries, tile_configs, only_missing=True)
        if not (out_dir / TILES_GPKG_FILENAME).exists():
            _write_tiles_gpkg(tiles, out_dir / TILES_GPKG_FILENAME)
        if not (out_dir / TILE_LIST_FILENAME).exists():
            _write_tile_list(out_dir, tile_entries)
        return manifest_path
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"{manifest_path} exists and describes different tiles, a different base "
            "configuration or a different cache root. Use another --out-dir, or "
            "overwrite=True (--overwrite) to replace the manifest and tile configurations "
            "(tile outputs are kept)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_tile_configs(out_dir, tile_entries, tile_configs, only_missing=False)
    _write_tiles_gpkg(tiles, out_dir / TILES_GPKG_FILENAME)
    _write_tile_list(out_dir, tile_entries)
    _write_json_atomic(manifest_path, manifest)
    logger.info("Wrote %s (%d tiles)", manifest_path, len(tile_entries))
    return manifest_path


def _write_tile_configs(
    out_dir: Path, entries: list[dict], configs: list[Any], *, only_missing: bool
) -> None:
    for entry, config in zip(entries, configs, strict=True):
        path = out_dir / entry["config"]
        if only_missing and path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(config.to_yaml_str())
        os.replace(tmp, path)


def _write_tile_list(out_dir: Path, entries: list[dict]) -> None:
    tmp = out_dir / f".{TILE_LIST_FILENAME}.{os.getpid()}.tmp"
    tmp.write_text("".join(f"{e['tile_id']}\n" for e in entries))
    os.replace(tmp, out_dir / TILE_LIST_FILENAME)


def _write_tiles_gpkg(tiles: gpd.GeoDataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.gpkg")
    if tmp.exists():
        tmp.unlink()
    attrs_cols = [c for c in tiles.columns if c not in ("geometry", "halo", "bounds")]
    cores = gpd.GeoDataFrame(tiles[attrs_cols].copy(), geometry=tiles.geometry, crs=tiles.crs)
    cores.attrs = {}
    cores.to_file(tmp, layer="tiles", driver="GPKG")
    halos = gpd.GeoDataFrame(
        {"index": tiles["index"].to_numpy(), "tile_id": tiles["tile_id"].to_numpy()},
        geometry=gpd.GeoSeries(tiles["halo"].values, crs=tiles.crs),
        crs=tiles.crs,
    )
    halos.to_file(tmp, layer="halos", driver="GPKG")
    aoi = tiles.attrs.get("study_area_geometry")
    if aoi is not None:
        gpd.GeoDataFrame(
            {"label": [str(tiles.attrs.get("study_area_label"))]}, geometry=[aoi], crs="EPSG:4326"
        ).to_file(tmp, layer="study_area", driver="GPKG")
    os.replace(tmp, path)


def load_manifest(manifest: str | Path | dict) -> dict[str, Any]:
    """Load ``manifest.json`` (or a directory containing it).

    Returns
    -------
    dict
        The manifest with an extra key ``"_root"`` (absolute directory of the
        manifest, against which its relative paths are resolved).

    Raises
    ------
    FileNotFoundError
        If the manifest does not exist.
    ValueError
        If the file is not a manifest of a supported schema version.
    """
    if isinstance(manifest, dict):
        if "_root" not in manifest:
            raise ValueError("A manifest dict must come from load_manifest() (missing '_root')")
        return manifest
    path = Path(manifest).expanduser()
    if path.is_dir():
        path = path / MANIFEST_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Tile manifest not found: {path}")
    data = _read_json(path)
    if data is None or "tiles" not in data or "grid" not in data:
        raise ValueError(f"{path} is not an agribound tile manifest")
    if str(data.get("schema_version")) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has manifest schema {data.get('schema_version')!r}; this agribound reads "
            f"schema {MANIFEST_SCHEMA_VERSION!r}. Re-create it with 'agribound tiles make'."
        )
    data["_root"] = str(path.parent.resolve())
    return data


def _tile_entry(manifest: dict[str, Any], tile: int | str) -> dict[str, Any]:
    entries = manifest["tiles"]
    if isinstance(tile, str) and not tile.lstrip("-").isdigit():
        for entry in entries:
            if entry["tile_id"] == tile:
                return entry
        raise ValueError(f"Tile {tile!r} is not in the manifest")
    index = int(tile)
    if not 0 <= index < len(entries):
        raise ValueError(
            f"Tile index {index} is out of range: the manifest has {len(entries)} tiles "
            f"(indices 0-{len(entries) - 1})"
        )
    return entries[index]


def _tile_paths(manifest: dict[str, Any], entry: dict[str, Any]) -> dict[str, Path]:
    root = Path(manifest["_root"])
    tile_dir = root / entry["dir"]
    return {
        "dir": tile_dir,
        "config": root / entry["config"],
        "output": root / entry["output"],
        "composite_done": tile_dir / "composite.done.json",
        "composite_failed": tile_dir / "composite.failed.json",
        "delineate_done": tile_dir / "delineate.done.json",
        "delineate_failed": tile_dir / "delineate.failed.json",
        "composite_provenance": tile_dir / "composite.provenance.json",
        "no_data": tile_dir / "no_data.json",
    }


def load_tile_config(manifest: str | Path | dict, tile: int | str) -> Any:
    """Load the :class:`~agribound.config.AgriboundConfig` of one tile."""
    from agribound.config import AgriboundConfig

    m = load_manifest(manifest)
    return AgriboundConfig.from_yaml(_tile_paths(m, _tile_entry(m, tile))["config"])


# ---------------------------------------------------------------------------
# Stage markers
# ---------------------------------------------------------------------------


def _working_dir(config: Any) -> Path:
    """``config.get_working_dir()`` without creating the directory."""
    if config.cache_dir:
        return Path(config.cache_dir).expanduser()
    return Path(config.output_path).parent / ".agribound_cache"


def _stage_marker_path(config: Any) -> Path:
    """Content-addressed "stage A done" marker inside the tile's cache directory.

    Named by :func:`agribound._cache.cache_key` of the configuration, so
    configurations that share the composite (same study area, source, year,
    compositing and export settings) also share the marker.
    """
    from agribound._cache import cache_key

    return _working_dir(config) / f"stage_{cache_key(config, 'stage-a')}.json"


def _needs_lulc_raster(config: Any) -> bool:
    return bool(config.lulc_filter) and config.lulc_mode == "raster"


@functools.lru_cache(maxsize=64)
def _ftw_n_windows(engine_params_json: str) -> int:
    """Number of input windows of the FTW model selected by the engine parameters."""
    from agribound.engines.ftw import resolve_ftw_model

    return int(resolve_ftw_model(json.loads(engine_params_json)).n_windows)


def _engine_params_json(config: Any) -> str:
    return json.dumps(config.engine_params, sort_keys=True, default=str)


def _engine_staging(config: Any, *, strict: bool = True) -> str | None:
    """Kind of engine input that the composite stage must also build, or *None*.

    ``"ftw-windows"``: engine ``"ftw"`` on an Earth Engine imagery source with
    a two-window model. The FTW engine builds its two seasonal window
    composites inside ``delineate()``, which needs network access, so the
    composite stage builds them in advance (see :func:`_stage_engine_inputs`).

    ``"ensemble-inputs"``: engine ``"ensemble"`` on an Earth Engine imagery
    source with at least one two-window FTW member; the members' window
    composites are built in the members' cache directories
    (:meth:`agribound.engines.ensemble.EnsembleEngine.stage_inputs`).

    With ``strict=False`` a model that cannot be resolved here (e.g. ftw-tools
    not installed in the environment running ``tiles status``) counts as
    two-window, so a tile is only reported as staged once its windows are.
    """
    if not config.is_gee_source():
        return None
    if config.engine == "ftw":
        try:
            n_windows = _ftw_n_windows(_engine_params_json(config))
        except Exception:
            if strict:
                raise
            return "ftw-windows"
        return "ftw-windows" if n_windows == 2 else None
    if config.engine == "ensemble":
        from agribound.engines.ensemble import EnsembleEngine

        for spec in EnsembleEngine.member_specs(config):
            if spec["engine"] != "ftw":
                continue
            try:
                n_windows = _ftw_n_windows(
                    json.dumps(spec["engine_params"], sort_keys=True, default=str)
                )
            except Exception:
                if strict:
                    raise
                return "ensemble-inputs"
            if n_windows == 2:
                return "ensemble-inputs"
    return None


def _engine_marker_path(config: Any) -> Path:
    """Stage marker for engine inputs, keyed by the stage-A key, engine and its parameters."""
    from agribound._cache import cache_key

    key = cache_key(config, "stage-engine", config.engine, _engine_params_json(config))
    return _working_dir(config) / f"stage_{key}.json"


def _engine_marker_valid(marker: dict | None) -> bool:
    if not marker:
        return False
    rasters = marker.get("rasters") or []
    if not all(Path(p).exists() for p in rasters):
        return False
    if rasters:
        return True
    # An ensemble whose members all failed to stage with on_member_error="skip"
    # has nothing left to stage (those members are skipped during delineation).
    return marker.get("kind") == "ensemble-inputs" and bool(
        (marker.get("record") or {}).get("failed_members")
    )


def _stage_engine_inputs(config: Any, raster_path: str) -> dict[str, Any] | None:
    """Build engine inputs that would otherwise be downloaded during delineation.

    For ``"ftw-windows"`` this calls
    :meth:`agribound.engines.ftw.FTWEngine.stage_inputs`, the input stage of
    ``FTWEngine.delineate``, so the window composites land in the tile's
    cache under the keys the engine looks up later (same windows, same
    ``date_range``, same builder); for ``"ensemble-inputs"``
    :meth:`agribound.engines.ensemble.EnsembleEngine.stage_inputs` does the
    same for every member in its own cache directory. Returns the marker
    content (``kind``, ``rasters``, ``record``) or *None* when the engine
    needs nothing.
    """
    kind = _engine_staging(config)
    if kind is None:
        return None
    if kind == "ensemble-inputs":
        from agribound.engines.ensemble import EnsembleEngine

        staged = EnsembleEngine.stage_inputs(config, raster_path)
        record = {"members": staged["members"], "failed_members": staged["failed_members"]}
        return {"kind": kind, "rasters": staged["rasters"], "record": record}
    from agribound.engines.ftw import FTWEngine

    staged = FTWEngine.stage_inputs(config, raster_path)
    return {"kind": kind, "rasters": staged["rasters"], "record": staged["windows"]}


def _lulc_marker_path(config: Any) -> Path:
    """Stage marker for the LULC raster, keyed by the stage-A fields and ``lulc_dataset``.

    The LULC raster depends on the study area, year, export CRS and the
    dataset choice, not on the composite settings alone, so configurations
    that share a composite but select another LULC dataset do not share it.
    """
    from agribound._cache import cache_key

    key = cache_key(config, "stage-lulc", config.lulc_dataset)
    return _working_dir(config) / f"stage_{key}.json"


def _lulc_marker(config: Any) -> dict | None:
    """The LULC marker if the raster it names exists, else *None*."""
    marker = _read_json(_lulc_marker_path(config))
    path = (marker or {}).get("lulc_raster_path")
    return marker if path and Path(path).exists() else None


def _write_lulc_marker(config: Any, entry: dict, lulc_path: str, run_id: str | None) -> None:
    _write_json_atomic(
        _lulc_marker_path(config),
        {
            "tile_id": entry["tile_id"],
            "lulc_dataset": config.lulc_dataset,
            "lulc_raster_path": str(lulc_path),
            "run_id": run_id,
            "finished_utc": _utc_now(),
        },
    )


def _prefetch_lulc(config: Any) -> str:
    """Download the LULC raster of a tile for the composite stage; raise on failure.

    The composite stage exists so that the delineation stage can run without
    network access, so a failed LULC prefetch fails the stage whatever
    ``lulc_on_error`` says (that policy applies when the filter itself runs).
    """
    from agribound.postprocess.lulc_filter import prefetch_lulc_raster

    try:
        path = prefetch_lulc_raster(config)
    except Exception as exc:
        raise RuntimeError(
            f"LULC raster prefetch failed ({type(exc).__name__}: {exc}). The composite stage "
            "downloads the LULC raster (lulc_mode='raster') so that the delineation stage can "
            "run offline; it fails whatever lulc_on_error is set to. Re-run the stage to "
            "retry, or rebuild the base configuration with --no-lulc-filter."
        ) from exc
    if not path or not Path(path).exists():
        raise RuntimeError(f"prefetch_lulc_raster returned {path!r}, which does not exist")
    return str(path)


def _staged_marker(config: Any, *, strict: bool = False, require_lulc: bool = True) -> dict | None:
    """Return the stage-A marker if it is valid for *config*, else *None*.

    Valid means: the composite raster exists; the LULC raster exists when
    ``lulc_mode="raster"`` (unless *require_lulc* is False); and the engine
    inputs (FTW window composites) exist when the engine needs them
    (:func:`_engine_staging`).
    """
    marker = _read_json(_stage_marker_path(config))
    if marker is None:
        return None
    raster = marker.get("raster_path")
    if not raster or not Path(raster).exists():
        return None
    if require_lulc and _needs_lulc_raster(config):
        lulc = _lulc_marker(config)
        if lulc is None:
            return None
        marker = {**marker, "lulc_raster_path": lulc["lulc_raster_path"]}
    if _engine_staging(config, strict=strict) is not None:
        engine_marker = _read_json(_engine_marker_path(config))
        if not _engine_marker_valid(engine_marker):
            return None
        marker = {**marker, "engine_inputs": engine_marker}
    return marker


def _no_data_marker_paths(config: Any) -> dict[str, Path]:
    """No-data markers: ``composite`` (shared by engines) and ``engine`` (engine-specific)."""
    stage_key = _stage_marker_path(config).stem.removeprefix("stage_")
    engine_key = _engine_marker_path(config).stem.removeprefix("stage_")
    work = _working_dir(config)
    return {
        "composite": work / f"nodata_{stage_key}.json",
        "engine": work / f"nodata_{engine_key}.json",
    }


def _no_data_marker(config: Any) -> dict | None:
    """Return the no-data record that applies to *config*, or *None*."""
    for path in _no_data_marker_paths(config).values():
        marker = _read_json(path)
        if marker is not None and marker.get("reason"):
            return marker
    return None


def _record_no_data(
    config: Any, entry: dict, paths: dict[str, Path], level: str, stage: str, exc: BaseException
) -> dict[str, Any]:
    """Write the no-data markers for a tile and return the record."""
    from agribound.provenance import config_hash

    record = {
        "tile_id": entry["tile_id"],
        "index": entry["index"],
        "level": level,
        "stage": stage,
        "reason": no_data_reason(exc),
        "error": f"{type(exc).__name__}: {exc}",
        "config_hash": config_hash(config),
        "recorded_utc": _utc_now(),
        **_runtime_context(),
    }
    _write_json_atomic(_no_data_marker_paths(config)[level], record)
    _write_json_atomic(paths["no_data"], record)
    for key in ("composite_failed", "delineate_failed"):
        paths[key].unlink(missing_ok=True)
    logger.warning(
        "Tile %s has no input data (%s); recorded as no-data: %s",
        entry["tile_id"],
        level,
        record["reason"],
    )
    return record


def _write_failed_marker(path: Path, stage: str, entry: dict, exc: BaseException) -> None:
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    record = {
        "tile_id": entry["tile_id"],
        "index": entry["index"],
        "stage": stage,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": "".join(tb)[-20000:],
        "failed_utc": _utc_now(),
        **_runtime_context(),
    }
    try:
        _write_json_atomic(path, record)
    except Exception as write_exc:  # pragma: no cover - best effort
        logger.warning("Could not write failure marker %s: %s", path, write_exc)


def _output_is_current(config: Any, output: Path) -> dict | None:
    """Return the provenance record if *output* is a successful run of *config*.

    The same test as the pipeline's reuse check
    (:func:`agribound.provenance.reuse_mismatch`): configuration hash,
    study-area fingerprint and results versions.
    """
    from agribound.provenance import read_provenance, reuse_mismatch

    if not output.exists() or output.stat().st_size == 0:
        return None
    record = read_provenance(output)
    if record is None or record.get("status") != "success":
        return None
    if reuse_mismatch(record, config, output) is not None:
        return None
    return record


# ---------------------------------------------------------------------------
# run_tile
# ---------------------------------------------------------------------------


def run_tile(
    manifest: str | Path | dict,
    index: int | str,
    stage: str = "all",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one tile of a manifest (idempotent).

    Parameters
    ----------
    manifest : str, Path or dict
        ``manifest.json``, its directory, or a loaded manifest.
    index : int or str
        Tile index (Slurm array index, 0-based) or tile ID.
    stage : str
        ``"composite"``: build the composite / embeddings with
        :func:`agribound.pipeline.build_composite`, download the LULC raster
        when ``lulc_mode="raster"`` (a failed LULC download fails the stage,
        whatever ``lulc_on_error`` says, because the delineation stage would
        need network access), and for ``engine="ftw"`` with a two-window
        model on an Earth Engine source build the two seasonal window
        composites with the FTW engine's own window builder; then write the
        stage markers.
        ``"delineate"``: run :func:`agribound.pipeline.delineate` on a tile
        whose composite stage is done (raises otherwise), so the composite,
        LULC raster and FTW windows are read from the cache.
        ``"all"``: the composite stage without the LULC download, then
        :func:`agribound.pipeline.delineate`, which downloads the LULC raster
        itself and applies ``lulc_on_error``.
    overwrite : bool
        Re-run even if the stage is already done, and retry tiles recorded as
        no-data (the delineation output is replaced; builders still reuse
        cached rasters -- delete the tile's cache directory to force new
        downloads). A re-run delineation gets a new run ID, so a merged
        output made before it no longer matches the tile outputs:
        :func:`merge_tiles` then raises :class:`FileExistsError` unless it is
        also given ``overwrite=True`` (``agribound tiles merge --overwrite``).

    Returns
    -------
    dict
        ``{"tile_id", "index", "stage", "status"}`` where *status* is
        ``"done"`` (ran now), ``"skipped"`` (already done) or ``"no-data"``
        (no input data for the tile, see the module docstring; with
        ``"reason"``), plus ``raster_path`` (composite stage) or ``output``,
        ``n_output``, ``run_id`` (delineation), and ``wall_s``.

    Raises
    ------
    RuntimeError
        For ``stage="delineate"`` on a tile that has not been staged, or a
        failed LULC download in the composite stage.
    Exception
        Any other pipeline error is re-raised after ``<stage>.failed.json``
        (error and traceback) is written to the tile directory.

    Notes
    -----
    Completion is decided from content, not from markers alone: a
    delineation is done when the tile output exists and its provenance
    record reports success with the current configuration hash and results
    versions (the pipeline's reuse test,
    :func:`agribound.provenance.reuse_mismatch`); a composite
    stage is done when the content-addressed stage markers in the tile's
    cache directory point to existing rasters (composite; LULC raster when
    needed; FTW window rasters when needed).

    ``stage="delineate"`` guarantees only that these inputs are cached.
    Other network access during delineation is not prevented: model weights
    (``agribound tiles prefetch``), and Earth Engine for
    ``lulc_mode="server"``. Ensemble tiles stage the window composites of
    their two-window FTW members; an ensemble member whose staging failed
    with ``on_member_error="skip"`` builds its inputs again (online) during
    delineation or is skipped.
    """
    stage = str(stage).lower().strip()
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    m = load_manifest(manifest)
    entry = _tile_entry(m, index)
    paths = _tile_paths(m, entry)
    from agribound.config import AgriboundConfig

    config = AgriboundConfig.from_yaml(paths["config"])
    t0 = time.perf_counter()
    base = {"tile_id": entry["tile_id"], "index": entry["index"], "stage": stage}

    if stage == "composite":
        return _run_composite_stage(config, entry, paths, base, t0, overwrite, require_lulc=True)
    if not overwrite:
        reused = _reuse_current_output(config, entry, paths, base)
        if reused is not None:
            return reused
    if stage == "all":
        staged = _run_composite_stage(config, entry, paths, base, t0, overwrite, require_lulc=False)
        if staged["status"] == _NO_DATA:
            return staged
    return _run_delineate_stage(config, entry, paths, base, t0, overwrite, stage)


def _no_data_result(base: dict, marker: dict, t0: float) -> dict[str, Any]:
    return {
        **base,
        "status": _NO_DATA,
        "reason": marker.get("reason"),
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def _clear_no_data(config: Any, paths: dict[str, Path], levels: Sequence[str]) -> None:
    markers = _no_data_marker_paths(config)
    for level in levels:
        markers[level].unlink(missing_ok=True)
    paths["no_data"].unlink(missing_ok=True)


def _run_composite_stage(
    config: Any,
    entry: dict,
    paths: dict[str, Path],
    base: dict,
    t0: float,
    overwrite: bool,
    *,
    require_lulc: bool,
) -> dict[str, Any]:
    from agribound.pipeline import build_composite
    from agribound.provenance import RunRecorder, write_provenance

    if not overwrite:
        marker = _staged_marker(config, strict=True, require_lulc=require_lulc)
        if marker is not None:
            logger.info(
                "Tile %s: composite already staged (%s)", entry["tile_id"], marker["raster_path"]
            )
            _ensure_local_composite_marker(paths, marker)
            return {
                **base,
                "status": "skipped",
                "raster_path": marker["raster_path"],
                "wall_s": 0.0,
            }
        no_data = _no_data_marker(config)
        if no_data is not None:
            logger.info("Tile %s: no input data (%s)", entry["tile_id"], no_data.get("reason"))
            if not paths["no_data"].exists():
                _write_json_atomic(paths["no_data"], no_data)
            return _no_data_result(base, no_data, t0)

    recorder = RunRecorder(config)
    level = "composite"
    lulc_path = None
    engine_inputs = None
    raster_path = None
    marker: dict[str, Any] = {}
    try:
        with recorder:
            recorder.set("tile_id", entry["tile_id"])
            with recorder.step("composite"):
                # The composite cache key does not include the LULC settings; the LULC
                # raster is handled below (composite stage) or by delineate() (stage all).
                composite_config = (
                    config.merged(lulc_filter=False) if config.lulc_filter else config
                )
                raster_path = build_composite(composite_config)
            recorder.set("raster_path", raster_path)
            marker = {
                "tile_id": entry["tile_id"],
                "raster_path": str(raster_path),
                "run_id": recorder.run_id,
                "finished_utc": _utc_now(),
                **_runtime_context(),
            }
            # Written at once, so other engines sharing the cache can reuse the composite.
            _write_json_atomic(_stage_marker_path(config), marker)
            if require_lulc and _needs_lulc_raster(config):
                level = "lulc"
                with recorder.step("lulc_raster"):
                    lulc_path = _prefetch_lulc(config)
                recorder.set("lulc_raster_path", lulc_path)
                _write_lulc_marker(config, entry, lulc_path, recorder.run_id)
            if _engine_staging(config) is not None:
                level = "engine"
                with recorder.step("engine_inputs"):
                    engine_inputs = _stage_engine_inputs(config, raster_path)
                recorder.set("engine_inputs", engine_inputs)
                _write_json_atomic(_engine_marker_path(config), engine_inputs)
    except BaseException as exc:
        with contextlib.suppress(Exception):  # best effort; the failure marker follows
            write_provenance(paths["dir"] / "composite", recorder.to_dict())
        if level in ("composite", "engine") and no_data_reason(exc) is not None:
            record = _record_no_data(config, entry, paths, level, base["stage"], exc)
            return _no_data_result(base, record, t0)
        _write_failed_marker(paths["composite_failed"], "composite", entry, exc)
        raise
    wall_s = round(time.perf_counter() - t0, 3)
    local = {**marker, "wall_s": wall_s}
    if lulc_path is not None:
        local["lulc_raster_path"] = lulc_path
    if engine_inputs is not None:
        local["engine_inputs"] = engine_inputs
    _write_json_atomic(paths["composite_done"], local)
    write_provenance(paths["dir"] / "composite", recorder.to_dict())
    paths["composite_failed"].unlink(missing_ok=True)
    _clear_no_data(config, paths, ("composite", "engine") if engine_inputs else ("composite",))
    logger.info("Tile %s: composite staged in %.1f s -> %s", entry["tile_id"], wall_s, raster_path)
    return {**base, "status": "done", "raster_path": str(raster_path), "wall_s": wall_s}


def _ensure_local_composite_marker(paths: dict[str, Path], marker: dict) -> None:
    if not paths["composite_done"].exists():
        _write_json_atomic(paths["composite_done"], marker)
    paths["composite_failed"].unlink(missing_ok=True)


def _reuse_current_output(
    config: Any, entry: dict, paths: dict[str, Path], base: dict
) -> dict[str, Any] | None:
    """``"skipped"`` result if the tile output is a successful run of *config*."""
    output = paths["output"]
    record = _output_is_current(config, output)
    if record is None:
        return None
    logger.info("Tile %s: output is current (%s)", entry["tile_id"], output)
    done = _done_record(entry, record, output, reused=True)
    if not paths["delineate_done"].exists():
        _write_json_atomic(paths["delineate_done"], done)
    paths["delineate_failed"].unlink(missing_ok=True)
    return {
        **base,
        "status": "skipped",
        "output": str(output),
        "n_output": done["n_output"],
        "run_id": done["run_id"],
        "wall_s": 0.0,
    }


def _run_delineate_stage(
    config: Any,
    entry: dict,
    paths: dict[str, Path],
    base: dict,
    t0: float,
    overwrite: bool,
    stage: str,
) -> dict[str, Any]:
    from agribound.pipeline import delineate
    from agribound.provenance import provenance_path, read_provenance

    output = paths["output"]
    if stage == "delineate":
        # A no-data tile has nothing to delineate (also with overwrite; the
        # composite stage with --overwrite retries the download).
        no_data = _no_data_marker(config)
        if no_data is not None:
            logger.info("Tile %s: no input data (%s)", entry["tile_id"], no_data.get("reason"))
            if not paths["no_data"].exists():
                _write_json_atomic(paths["no_data"], no_data)
            return _no_data_result(base, no_data, t0)
        try:
            staged = _staged_marker(config, strict=True)
        except Exception as exc:
            _write_failed_marker(paths["delineate_failed"], stage, entry, exc)
            raise
        if staged is None:
            exc = RuntimeError(
                f"Tile {entry['tile_id']} (index {entry['index']}) has not been staged: run "
                "'agribound tiles run --stage composite' for it first (e.g. the stage array of "
                "examples/hpc/submit_region.sh), or use --stage all on nodes with internet "
                "access."
            )
            _write_failed_marker(paths["delineate_failed"], stage, entry, exc)
            raise exc

    run_config = config.merged(overwrite=True) if overwrite else config
    try:
        gdf = delineate(config=run_config)
    except BaseException as exc:
        if no_data_reason(exc) is not None:
            record = _record_no_data(config, entry, paths, "engine", stage, exc)
            return _no_data_result(base, record, t0)
        _write_failed_marker(paths["delineate_failed"], stage, entry, exc)
        raise
    record = read_provenance(output) or {}
    done = _done_record(entry, record, output, reused=bool(gdf.attrs.get("reused")))
    done["n_output"] = len(gdf)
    _write_json_atomic(paths["delineate_done"], done)
    paths["delineate_failed"].unlink(missing_ok=True)
    _clear_no_data(config, paths, ("composite", "engine"))

    if stage == "all" and _needs_lulc_raster(config):
        # delineate() downloaded the LULC raster itself; record it for later
        # --stage delineate runs of configurations sharing this cache.
        lulc_path = (record.get("facts") or {}).get("lulc_raster_path")
        if lulc_path and Path(lulc_path).exists():
            _write_lulc_marker(config, entry, lulc_path, record.get("run_id"))

    wall_s = round(time.perf_counter() - t0, 3)
    logger.info(
        "Tile %s: %d polygons in %.1f s -> %s (provenance %s)",
        entry["tile_id"],
        len(gdf),
        wall_s,
        output,
        provenance_path(output).name,
    )
    return {
        **base,
        "status": "done",
        "output": str(output),
        "n_output": len(gdf),
        "run_id": done["run_id"],
        "wall_s": wall_s,
    }


def _done_record(entry: dict, record: dict, output: Path, *, reused: bool) -> dict[str, Any]:
    facts = record.get("facts") or {}
    return {
        "tile_id": entry["tile_id"],
        "index": entry["index"],
        "output": str(output),
        "run_id": record.get("run_id"),
        "config_hash": record.get("config_hash"),
        "n_output": facts.get("n_output"),
        "wall_s": record.get("wall_s"),
        "reused": reused,
        "finished_utc": _utc_now(),
        **_runtime_context(),
    }


# ---------------------------------------------------------------------------
# tile_status
# ---------------------------------------------------------------------------


def tile_status(manifest: str | Path | dict) -> pd.DataFrame:
    """Report the state of every tile.

    Returns
    -------
    pandas.DataFrame
        One row per tile with columns ``index``, ``tile_id``, ``composite``
        and ``delineate`` (``"done"``, ``"failed"``, ``"pending"`` or
        ``"no-data"`` -- no input data, a final state, see the module
        docstring; the delineation can also be ``"stale"`` -- an output
        exists but was made with a different configuration, or by an
        agribound release whose results for it differ
        (:mod:`agribound._results`) -- and either
        column is ``"error"`` when the tile configuration cannot be loaded),
        ``n_output``, ``run_id``, ``error`` (first line of the latest
        failure, if any) and ``no_data_reason``.
    """
    from agribound.config import AgriboundConfig
    from agribound.provenance import read_provenance, reuse_mismatch

    m = load_manifest(manifest)
    rows = []
    for entry in m["tiles"]:
        paths = _tile_paths(m, entry)
        row: dict[str, Any] = {
            "index": entry["index"],
            "tile_id": entry["tile_id"],
            "composite": _PENDING,
            "delineate": _PENDING,
            "n_output": None,
            "run_id": None,
            "error": None,
            "no_data_reason": None,
        }
        try:
            config = AgriboundConfig.from_yaml(paths["config"])
        except Exception as exc:
            row.update(composite="error", delineate="error", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            continue

        no_data = _no_data_marker(config)
        if _staged_marker(config) is not None:
            row["composite"] = _DONE
        elif no_data is not None:
            row["composite"] = _NO_DATA
        elif paths["composite_failed"].exists():
            row["composite"] = _FAILED
            row["error"] = _short_error(paths["composite_failed"])

        output = paths["output"]
        record = read_provenance(output) if output.exists() else None
        if record is not None and record.get("status") == "success":
            if reuse_mismatch(record, config, output) is None:
                row["delineate"] = _DONE
                row["n_output"] = (record.get("facts") or {}).get("n_output")
                row["run_id"] = record.get("run_id")
                row["error"] = None
            else:
                row["delineate"] = _STALE
        elif no_data is not None:
            row["delineate"] = _NO_DATA
            row["no_data_reason"] = no_data.get("reason")
            row["error"] = None
        elif paths["delineate_failed"].exists():
            row["delineate"] = _FAILED
            row["error"] = _short_error(paths["delineate_failed"])
        elif record is not None and record.get("status") == "failed":
            row["delineate"] = _FAILED
            row["error"] = record.get("error")
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "index",
            "tile_id",
            "composite",
            "delineate",
            "n_output",
            "run_id",
            "error",
            "no_data_reason",
        ],
    )


def _short_error(path: Path) -> str | None:
    data = _read_json(path) or {}
    error = data.get("error")
    return str(error).splitlines()[0] if error else None


# ---------------------------------------------------------------------------
# merge_tiles
# ---------------------------------------------------------------------------


def default_merge_output(manifest: str | Path | dict) -> Path:
    """Default merged output path: ``<manifest dir>/fields_merged.<ext>``."""
    m = load_manifest(manifest)
    ext = {"gpkg": ".gpkg", "geojson": ".geojson", "parquet": ".parquet"}[m["output_format"]]
    return Path(m["_root"]) / f"fields_merged{ext}"


def _read_study_area_layer(m: dict[str, Any]) -> Any | None:
    path = Path(m["_root"]) / m["files"]["tiles_gpkg"]
    try:
        gdf = gpd.read_file(path, layer="study_area")
    except Exception as exc:
        raise RuntimeError(f"Could not read the study_area layer of {path}: {exc}") from exc
    if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
        gdf = gdf.to_crs("EPSG:4326")
    return shapely.union_all(gdf.geometry.values)


def merge_tiles(
    manifest: str | Path | dict,
    output: str | Path | None = None,
    *,
    crs: str = "EPSG:4326",
    allow_missing: bool = False,
    overwrite: bool = False,
    reference: str | Path | None = None,
    check_overlaps: bool = True,
) -> gpd.GeoDataFrame:
    """Merge finished tile outputs into one vector file.

    Each tile keeps only the polygons whose representative point it owns
    (see the merge rule in the module docstring); the kept polygons get an
    ``agribound:tile_id`` column, are reprojected to *crs* and concatenated.
    A summary is written to ``<output>.provenance.json``.

    Parameters
    ----------
    manifest : str, Path or dict
        Tile manifest.
    output : str, Path or None
        Output file (format from the extension). Default
        :func:`default_merge_output`.
    crs : str
        CRS of the merged output (default EPSG:4326, valid for any region).
    allow_missing : bool
        Merge even if some tiles are not done (they are listed in the summary
        and a WARNING is logged). Default: raise. Tiles recorded as
        ``"no-data"`` (see the module docstring) are never "missing": they
        are merged as empty and listed under ``no_data_tiles`` with their
        reasons, with a WARNING.
    overwrite : bool
        Recompute and replace an existing merged output. Without it, an
        output made from exactly the same tile runs is returned without
        recomputation, and one made from other tile runs raises
        :class:`FileExistsError`.
    reference : str, Path or None
        Reference boundaries; when given, they are restricted to the study
        area (the ``study_area`` layer of ``tiles.gpkg``) with the rule the
        merged polygons follow: by representative point with ``clip=True``,
        else those intersecting the study area. They are compared with the
        merged output using :func:`agribound.evaluate.evaluate`; the metrics
        are stored in the summary (``evaluation``, with the reference counts
        and rule in ``evaluation_reference``) and in
        ``gdf.attrs["evaluation_metrics"]``. Reference polygons in no-data
        tiles count as false negatives.
    check_overlaps : bool
        Count pairs of kept polygons from different tiles whose overlap
        exceeds half of the smaller polygon's area (default *True*).

    Returns
    -------
    geopandas.GeoDataFrame
        The merged polygons. ``attrs["merge_summary"]`` holds the summary.

    Raises
    ------
    RuntimeError
        If tiles are missing and *allow_missing* is False, or if no tile is
        done and some are no-data (nothing to merge; usually the source has
        no data for the year).
    FileExistsError
        If *output* exists, was made from different tile runs, and
        *overwrite* is False.
    """
    from agribound._repro import collect_versions
    from agribound._version import __version__
    from agribound.io.vector import read_vector, write_vector
    from agribound.provenance import provenance_path, read_provenance, write_provenance

    m = load_manifest(manifest)
    grid = m["grid"]
    output = Path(output).expanduser() if output else default_merge_output(m)
    status = tile_status(m)
    done_mask = status["delineate"] == _DONE
    no_data_mask = status["delineate"] == _NO_DATA
    missing = status.loc[~(done_mask | no_data_mask), ["index", "tile_id", "delineate", "error"]]
    no_data = status.loc[no_data_mask, ["index", "tile_id", "no_data_reason"]].rename(
        columns={"no_data_reason": "reason"}
    )
    if len(missing) and not allow_missing:
        preview = ", ".join(missing["tile_id"].head(20))
        raise RuntimeError(
            f"{len(missing)} of {len(status)} tiles are not done ({preview}"
            f"{', ...' if len(missing) > 20 else ''}). Indices: "
            f"{format_index_ranges(missing['index'])}. Re-run them ('agribound tiles run') or "
            "pass allow_missing=True (--allow-missing)."
        )
    if not done_mask.any() and len(no_data):
        raise RuntimeError(
            f"No tile produced output: {len(no_data)} of {len(status)} tiles have no input data "
            f"(e.g. {no_data['tile_id'].iloc[0]}: {no_data['reason'].iloc[0]}). Check that the "
            "source covers the study area in this year ('agribound tiles status')."
        )

    signature = _sha1_json(
        {
            "tiles": sorted(
                zip(status.loc[done_mask, "tile_id"], status.loc[done_mask, "run_id"], strict=True)
            ),
            "crs": crs,
            "missing": sorted(missing["tile_id"]),
            "no_data": sorted(no_data["tile_id"]),
            "reference": str(reference) if reference else None,
            "manifest": m.get("signature"),
        }
    )
    if output.exists():
        previous = read_provenance(output)
        if not overwrite:
            if previous is not None and previous.get("inputs_signature") == signature:
                logger.info("Merged output %s is up to date", output)
                gdf = read_vector(output)
                gdf.attrs["merge_summary"] = previous
                gdf.attrs["reused"] = True
                return gdf
            raise FileExistsError(
                f"{output} exists and was not made from the current tile outputs. Pass "
                "overwrite=True (--overwrite) or choose another output."
            )

    study_area = _read_study_area_layer(m)
    shapely.prepare(study_area)
    aoi = study_area if grid.get("clip") else None
    entries = {e["tile_id"]: e for e in m["tiles"]}
    parts: list[gpd.GeoDataFrame] = []
    per_tile: list[dict[str, Any]] = []
    tile_records: list[dict[str, Any]] = []
    warnings: list[str] = []
    candidates: list[np.ndarray] = []

    for tile_id in status.loc[done_mask, "tile_id"]:
        entry = entries[tile_id]
        paths = _tile_paths(m, entry)
        record = read_provenance(paths["output"]) or {}
        tile_records.append(record)
        gdf = read_vector(paths["output"])
        info = {
            "tile_id": tile_id,
            "index": entry["index"],
            "run_id": record.get("run_id"),
            "config_hash": record.get("config_hash"),
            "n_in": len(gdf),
            "n_kept": 0,
            "n_reaching_halo_edge": 0,
            "wall_s": record.get("wall_s"),
        }
        if len(gdf) == 0:
            per_tile.append(info)
            continue
        if gdf.crs is None:
            raise ValueError(f"Tile output {paths['output']} has no CRS")
        points = representative_points(gdf.geometry)
        owner = assign_tile_ids(points, grid)
        keep = owner == tile_id
        if aoi is not None:
            keep &= shapely.covers(aoi, points)
        kept = gdf.loc[keep].copy()
        info["n_kept"] = len(kept)
        if len(kept):
            system = entry["system"]
            grid_crs = grid["systems"][system]["crs"]
            geoms_grid = np.asarray(kept.geometry.to_crs(grid_crs).values, dtype=object)
            halo_box = box(*entry["halo_grid_bounds"])
            reaching = ~shapely.within(geoms_grid, halo_box)
            info["n_reaching_halo_edge"] = int(reaching.sum())
            cell = cell_box(grid, system, entry["col"], entry["row"])
            candidates.append(~shapely.within(geoms_grid, cell))
            kept["agribound:tile_id"] = tile_id
            parts.append(kept.to_crs(crs))
        per_tile.append(info)

    if parts:
        merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=crs)
    else:
        merged = gpd.GeoDataFrame(
            {"agribound:tile_id": []}, geometry=gpd.GeoSeries([], crs=crs), crs=crs
        )

    n_edge = int(sum(t["n_reaching_halo_edge"] for t in per_tile))
    if n_edge:
        msg = (
            f"{n_edge} kept polygons reach the edge of their tile's halo and may be truncated; "
            f"consider a halo larger than the largest field (current halo_m={grid['halo_m']})."
        )
        logger.warning(msg)
        warnings.append(msg)
    n_overlap = None
    if check_overlaps and len(merged):
        n_overlap = _cross_tile_overlap_pairs(merged, np.concatenate(candidates))
        if n_overlap:
            msg = (
                f"{n_overlap} pairs of polygons from different tiles overlap by more than half "
                "of the smaller polygon (the same field delineated by two tiles)."
            )
            logger.warning(msg)
            warnings.append(msg)
    if len(missing):
        msg = f"Merged with {len(missing)} tiles missing: {format_index_ranges(missing['index'])}"
        logger.warning(msg)
        warnings.append(msg)
    if len(no_data):
        msg = (
            f"{len(no_data)} tiles have no input data and contribute no polygons: "
            f"{format_index_ranges(no_data['index'])} (reasons in no_data_tiles)"
        )
        logger.warning(msg)
        warnings.append(msg)

    metrics = None
    reference_selection = None
    if reference:
        metrics, reference_selection = _evaluate_merged(
            merged, reference, study_area, bool(grid.get("clip"))
        )

    write_vector(merged, output)
    summary = _merge_summary(
        m,
        per_tile,
        tile_records,
        missing,
        no_data,
        merged,
        crs,
        n_edge,
        n_overlap,
        metrics,
        warnings,
        signature,
        output,
    )
    summary["evaluation_reference"] = reference_selection
    summary["agribound_version"] = __version__
    summary["versions"] = collect_versions()
    write_provenance(output, summary)
    merged.attrs["merge_summary"] = summary
    if metrics is not None:
        merged.attrs["evaluation_metrics"] = metrics
    logger.info(
        "Merged %d polygons from %d tiles -> %s (summary %s)",
        len(merged),
        int(done_mask.sum()),
        output,
        provenance_path(output).name,
    )
    return merged


def _cross_tile_overlap_pairs(merged: gpd.GeoDataFrame, candidate_mask: np.ndarray) -> int:
    """Count pairs (from different tiles) overlapping > 50 % of the smaller polygon."""
    from agribound.io.crs import get_equal_area_crs

    idx = np.flatnonzero(candidate_mask)
    if idx.size == 0:
        return 0
    geoms = np.asarray(merged.geometry.to_crs(get_equal_area_crs()).values, dtype=object)
    tiles_arr = merged["agribound:tile_id"].to_numpy()
    tree = shapely.STRtree(geoms)
    left, right = tree.query(geoms[idx], predicate="intersects")
    left = idx[left]
    pairs = {
        (min(a, b), max(a, b))
        for a, b in zip(left, right, strict=True)
        if a != b and tiles_arr[a] != tiles_arr[b]
    }
    count = 0
    for a, b in pairs:
        inter = shapely.area(shapely.intersection(geoms[a], geoms[b]))
        smaller = min(shapely.area(geoms[a]), shapely.area(geoms[b]))
        if smaller > 0 and inter > 0.5 * smaller:
            count += 1
    return count


def _evaluate_merged(
    merged: gpd.GeoDataFrame, reference: str | Path, study_area: Any, clip: bool
) -> tuple[dict, dict[str, Any]]:
    """Evaluate *merged* against the reference polygons in *study_area* (EPSG:4326).

    The references are selected like the merged predictions: with *clip*
    (predictions kept only if their representative point lies in the study
    area) by representative point, else (predictions not restricted to the
    study-area outline) those intersecting the study area, as in
    :func:`agribound.pipeline._evaluate` with ``aoi_selection="none"``.

    Returns
    -------
    tuple
        ``(metrics, selection)`` with ``selection = {"n_reference_total",
        "n_reference_used", "selection"}``.
    """
    from agribound.evaluate import evaluate
    from agribound.io.vector import read_vector
    from agribound.pipeline import _REFERENCE_SELECTION_LABELS, select_in_study_area

    ref = read_vector(reference)
    n_ref = len(ref)
    selection: dict[str, Any] = {"n_reference_total": n_ref, "selection": "all"}
    rule = "representative_point" if clip else "intersects"
    if n_ref and ref.crs is None:
        logger.warning("Reference %s has no CRS; it is not restricted to the study area", reference)
    elif n_ref:
        aoi_ref = gpd.GeoSeries([study_area], crs="EPSG:4326").to_crs(ref.crs).iloc[0]
        ref, _ = select_in_study_area(ref, aoi_ref, rule)
        selection["selection"] = _REFERENCE_SELECTION_LABELS[rule]
    selection["n_reference_used"] = len(ref)
    logger.info(
        "Evaluating the merged output against %d of %d reference polygons (%s)",
        len(ref),
        n_ref,
        selection["selection"],
    )
    return evaluate(merged, ref), selection


def _split_engine_meta(metas: list[dict]) -> tuple[dict[str, Any], list[str]]:
    """Split tile ``engine_meta`` dicts into keys equal in every tile and keys that vary.

    Returns
    -------
    tuple
        ``(common, varying)``: the entries whose value is identical in all
        tiles (e.g. model and weights), and the sorted names of the other keys
        (per-tile counts, timings or per-raster statistics such as a
        percentile stretch).
    """
    if not metas:
        return {}, []
    keys = set().union(*metas)
    common: dict[str, Any] = {}
    varying: list[str] = []
    for key in sorted(keys):
        values = {json.dumps(m.get(key, "<missing>"), sort_keys=True, default=str) for m in metas}
        if len(values) == 1 and all(key in m for m in metas):
            common[key] = metas[0][key]
        else:
            varying.append(key)
    return common, varying


def _merge_summary(
    m: dict,
    per_tile: list[dict],
    records: list[dict],
    missing: pd.DataFrame,
    no_data: pd.DataFrame,
    merged: gpd.GeoDataFrame,
    crs: str,
    n_edge: int,
    n_overlap: int | None,
    metrics: dict | None,
    warnings: list[str],
    signature: str,
    output: Path,
) -> dict[str, Any]:
    core_area = {e["tile_id"]: float(e.get("core_area_km2") or 0.0) for e in m["tiles"]}
    step_totals: dict[str, float] = {}
    for record in records:
        for step in record.get("steps") or []:
            if isinstance(step.get("wall_s"), int | float):
                step_totals[step["name"]] = step_totals.get(step["name"], 0.0) + step["wall_s"]
    walls = [r["wall_s"] for r in records if isinstance(r.get("wall_s"), int | float)]
    rss = [r["peak_rss_mb"] for r in records if isinstance(r.get("peak_rss_mb"), int | float)]
    gpu = [
        r["torch_max_memory_mb"]
        for r in records
        if isinstance(r.get("torch_max_memory_mb"), int | float)
    ]
    meta_common, meta_varying = _split_engine_meta([r.get("engine_meta") or {} for r in records])
    versions = {r.get("agribound_version") for r in records if r.get("agribound_version")}
    if len(versions) > 1:
        warnings.append(f"Tiles were produced by different agribound versions: {sorted(versions)}")
    tile_warnings: dict[str, int] = {}
    for record in records:
        for w in record.get("warnings") or []:
            tile_warnings[w] = tile_warnings.get(w, 0) + 1
    return {
        "schema_version": "1",
        "kind": "agribound-tile-merge",
        "status": "success",
        "created_utc": _utc_now(),
        "output_path": str(output),
        "manifest": str(Path(m["_root"]) / MANIFEST_FILENAME),
        "manifest_signature": m.get("signature"),
        "inputs_signature": signature,
        "base_config": m.get("base_config"),
        "base_config_hash": m.get("base_config_hash"),
        "grid": m["grid"],
        "merge_rule": (
            "keep a tile's polygon iff the tile owns its representative point "
            "(shapely.point_on_surface in EPSG:4326; owner = floor of grid coordinates"
            + ("; point must lie in the study area)" if m["grid"].get("clip") else ")")
        ),
        "crs": crs,
        "n_tiles": len(m["tiles"]),
        "n_merged_tiles": len(per_tile),
        "missing_tiles": missing.to_dict(orient="records"),
        "n_no_data_tiles": len(no_data),
        "no_data_tiles": no_data.to_dict(orient="records"),
        "no_data_core_area_km2": round(
            sum(float(core_area.get(t, 0.0)) for t in no_data["tile_id"]), 3
        ),
        "core_area_km2_total": round(sum(core_area.values()), 3),
        "n_polygons_in": int(sum(t["n_in"] for t in per_tile)),
        "n_polygons_out": len(merged),
        "n_reaching_halo_edge": n_edge,
        "n_cross_tile_overlap_pairs": n_overlap,
        "wall_s_total": round(float(sum(walls)), 3),
        "step_wall_s_total": {k: round(v, 3) for k, v in step_totals.items()},
        "peak_rss_mb_max": max(rss) if rss else None,
        "torch_max_memory_mb_max": max(gpu) if gpu else None,
        "hosts": sorted({str(r.get("hostname")) for r in records if r.get("hostname")}),
        "devices": sorted({str(r.get("device")) for r in records if r.get("device")}),
        "config_hashes": sorted({r.get("config_hash") for r in records if r.get("config_hash")}),
        "engine_meta": meta_common,
        "engine_meta_varying_keys": meta_varying,
        "engine_meta_consistent": not meta_varying,
        "tile_warnings": tile_warnings,
        "evaluation": metrics,
        "warnings": warnings,
        "tiles": per_tile,
    }


__all__ = [
    "GRID_KINDS",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "NO_DATA_PATTERNS",
    "STAGES",
    "assign_tile_ids",
    "cell_box",
    "default_merge_output",
    "format_index_ranges",
    "load_manifest",
    "load_tile_config",
    "make_tiles",
    "merge_tiles",
    "no_data_reason",
    "representative_points",
    "run_tile",
    "tile_status",
    "write_tile_manifest",
]
