"""
Query the published USDA NASS Crop Sequence Boundaries (CSB).

CSB are polygons that USDA NASS derives from eight years of the Cropland Data
Layer (CDL) for the conterminous US: zones of uniform crop history, made from
crop maps, not field outlines drawn on imagery ("the boundaries captured are of
crops grown only, not ownership boundaries or tax parcels", NASS metadata;
Hunt et al. 2024). NASS publishes one national File Geodatabase per eight-year
window, from 2008-2015 to 2018-2025 (revision rev23), in the public domain.
Each polygon carries ``CSBID``, ``CSBYEARS``, ``CSBACRES``, the majority CDL
class of each year (``CDL<year>``), state, agricultural statistics district
and county codes, an inside point (``INSIDE_X``, ``INSIDE_Y``) and its length
and area.

:func:`query_csb` reads the polygons of a study area from a release zip in
place with GDAL (``/vsizip//vsicurl/``), without extracting or storing the
zip. The geodatabase in the zip is compressed, so GDAL reads the zip up to the
records it needs: on 2026-10-10, with the 2016-2023 release, a 5 km square in
Iowa (139 polygons) took 195 s and a 1 km box in Arizona 462 s. For repeated
queries, download the zip once (:func:`csb_release` gives its URL) and pass it
as ``source``. The
``crop-sequence`` engine makes zones of the same kind from the
``cdl-sequence`` stack (see the engines guide).

Reference
---------
Hunt, K.A., Abernethy, J., Beeson, P.C., Bowman, M., Wallander, S., Williams,
R., 2024. Crop sequence boundaries using USDA National Agricultural Statistics
Service historic cropland data layers. Statistical Journal of the IAOS 40,
237-246. https://doi.org/10.3233/SJI-230078
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd

logger = logging.getLogger(__name__)

#: Revision of the national releases that :func:`query_csb` reads.
CSB_REVISION = "rev23"
#: Folder of the national release zips on the NASS web site.
CSB_BASE_URL = "https://www.nass.usda.gov/Research_and_Science/Crop-Sequence-Boundaries/datasets"
#: Years of a CSB window.
CSB_WINDOW_YEARS = 8
#: Last years of the published eight-year windows (2008-2015 to 2018-2025), checked on
#: 2026-10-10.
CSB_YEARS: tuple[int, ...] = tuple(range(2015, 2026))
#: GDAL configuration of a remote read (``/vsicurl/``), set for the read only: GDAL
#: fetches byte ranges of the zip and keeps up to 200 MB of them in memory.
CSB_GDAL_OPTIONS: dict[str, str] = {
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".zip",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "200000000",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
}
#: Description of the data, recorded in the provenance of a saved query.
CSB_DATA_LABEL = (
    "USDA NASS Crop Sequence Boundaries (rev23, public domain): crop-history zones made "
    "from eight years of the Cropland Data Layer, not field outlines or reference boundaries"
)


def csb_release(year: int) -> dict[str, Any]:
    """Names of the national CSB release whose eight-year window ends in *year*.

    Parameters
    ----------
    year : int
        Last year of the window, one of :data:`CSB_YEARS` (2023 is the
        2016-2023 release).

    Returns
    -------
    dict
        ``window`` (``(2016, 2023)``), ``name`` (``NationalCSB_2016-2023_rev23``),
        ``url`` (the zip on the NASS web site), ``gdb`` (the geodatabase's path
        inside the zip, ``NationalCSB_2016-2023_rev23/CSB1623.gdb``) and
        ``layer`` (``national1623``).

    Raises
    ------
    ValueError
        If no release ends in *year*.
    """
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError(f"year must be an integer (the last year of the window), got {year!r}")
    if year not in CSB_YEARS:
        raise ValueError(
            f"No national CSB release ends in {year}. The {CSB_REVISION} releases cover the "
            f"eight-year windows ending in {CSB_YEARS[0]}-{CSB_YEARS[-1]} "
            f"({CSB_YEARS[0] - CSB_WINDOW_YEARS + 1}-{CSB_YEARS[0]} to "
            f"{CSB_YEARS[-1] - CSB_WINDOW_YEARS + 1}-{CSB_YEARS[-1]})."
        )
    start = year - CSB_WINDOW_YEARS + 1
    name = f"NationalCSB_{start}-{year}_{CSB_REVISION}"
    short = f"{start % 100:02d}{year % 100:02d}"
    return {
        "window": (start, year),
        "name": name,
        "url": f"{CSB_BASE_URL}/{name}.zip",
        "gdb": f"{name}/CSB{short}.gdb",
        "layer": f"national{short}",
    }


def csb_dataset(year: int, source: str | Path | None = None) -> tuple[str, str, bool]:
    """GDAL path and layer of a CSB release, and whether reading it is remote.

    Parameters
    ----------
    year : int
        Last year of the window (:func:`csb_release`).
    source : str or Path, optional
        ``None``: the release zip on the NASS web site, read in place. A path
        or HTTP(S) URL ending in ``.zip``: a copy of the release zip (the
        geodatabase is read inside it). Anything else is used as given, for
        example an extracted ``CSB1623.gdb`` folder or a GDAL path.

    Returns
    -------
    tuple[str, str, bool]
        ``(path, layer, remote)``.
    """
    release = csb_release(year)
    layer = release["layer"]
    if source is None:
        return f"/vsizip//vsicurl/{release['url']}/{release['gdb']}", layer, True
    text = str(source)
    if text.lower().endswith(".zip"):
        if text.startswith(("http://", "https://")):
            return f"/vsizip//vsicurl/{text}/{release['gdb']}", layer, True
        local = Path(text).expanduser().resolve()
        return f"/vsizip/{local}/{release['gdb']}", layer, False
    return text, layer, text.startswith(("/vsicurl/", "/vsizip//vsicurl/", "http"))


@contextlib.contextmanager
def _gdal_options(options: dict[str, str], enabled: bool) -> Iterator[None]:
    """Set GDAL configuration *options* (pyogrio's GDAL) and restore the previous values."""
    if not enabled:
        yield
        return
    import pyogrio

    previous = {name: pyogrio.get_gdal_config_option(name) for name in options}
    pyogrio.set_gdal_config_options(dict(options))
    try:
        yield
    finally:
        pyogrio.set_gdal_config_options(previous)


def query_csb(
    study_area: Any,
    year: int,
    *,
    clip: bool = False,
    columns: Sequence[str] | None = None,
    dst_crs: Any = None,
    source: str | Path | None = None,
    layer: str | None = None,
    output_path: str | Path | None = None,
    output_format: str | None = None,
    provenance: bool = True,
) -> gpd.GeoDataFrame:
    """Read the published CSB polygons of a study area.

    Parameters
    ----------
    study_area : object
        Area of interest: a vector file path, ``"bbox:minx,miny,maxx,maxy"``,
        a bbox tuple ``(minx, miny, maxx, maxy)`` in EPSG:4326, a WKT string, a
        Shapely geometry (EPSG:4326), a GeoSeries or a GeoDataFrame (as for
        :func:`agribound.query_ftw`).
    year : int
        Last year of the CSB window: 2023 reads the 2016-2023 release (one of
        :data:`CSB_YEARS`).
    clip : bool
        Clip the polygons to the study area (default False: whole polygons
        that intersect it). Only the polygonal part of a cut is kept (lines or
        points where a polygon touches the study area's edge are dropped), so
        a polygon that only touches the study area is left out. The published
        attributes (``CSBACRES``, ``Shape_Area``, ``Shape_Length``) describe
        the whole polygon; ``agribound:clipped`` marks the polygons that were
        cut.
    columns : sequence of str, optional
        Attribute columns to read (default: all).
    dst_crs : CRS-like, optional
        CRS of the result (default: the release's, an Albers equal-area
        projection with the parameters of EPSG:5070).
    source : str or Path, optional
        Where to read the release (:func:`csb_dataset`): default the zip on the
        NASS web site; a local copy of the zip; or an extracted geodatabase.
    layer : str, optional
        Layer name (default ``national<yy><yy>``, e.g. ``national1623``).
    output_path : str or Path, optional
        Also write the result (GeoPackage, GeoJSON or GeoParquet, see
        :func:`agribound.io.vector.write_vector`).
    output_format : str, optional
        Format of *output_path* when its extension does not say.
    provenance : bool
        With *output_path*, also write ``<output_path>.provenance.json``: the
        query, the release, the counts and the package versions (default
        True).

    Returns
    -------
    geopandas.GeoDataFrame
        The CSB polygons that intersect the study area, with the release's
        attributes. ``attrs["csb_query"]`` records the release, the GDAL path
        and layer, the window, the counts (``n_bbox``: polygons read by the
        study area's bounding box; ``n_returned``) and the seconds the read
        took.

    Raises
    ------
    ValueError
        If no release ends in *year*, or the study area is empty.

    Notes
    -----
    CSB covers the conterminous US only; a study area elsewhere returns no
    polygon. CSB polygons are at least about 1 ha (NASS drops smaller ones),
    and adjacent fields with the same crop sequence form one polygon, so CSB is
    not a field-boundary reference (see the crop-sequence engine's limits in
    the engines guide).

    Examples
    --------
    >>> from agribound import query_csb
    >>> csb = query_csb("bbox:-90.85,42.21,-90.79,42.25", 2023)  # 2016-2023, Iowa
    >>> csb[["CSBID", "CSBACRES", "CDL2022", "CDL2023"]].head()
    """
    import pyogrio
    import shapely

    from agribound.ftw_query import _coerce_study_area, _ensure_crs

    start_time = time.perf_counter()
    release = csb_release(year)
    path, default_layer, remote = csb_dataset(year, source)
    layer = layer or default_layer
    aoi = _ensure_crs(_coerce_study_area(study_area), "EPSG:4326")
    geoms = aoi.geometry[~(aoi.geometry.isna() | aoi.geometry.is_empty)]
    if geoms.empty:
        raise ValueError("study_area holds no geometry")
    # Edges are densified before reprojection, so that a straight edge in the study
    # area's CRS keeps its curve in the release's CRS (about 100 m apart).
    step = 0.001 if geoms.crs.is_geographic else 100.0
    with _gdal_options(CSB_GDAL_OPTIONS, remote):
        info = pyogrio.read_info(path, layer=layer)
        aoi_layer = shapely.union_all(geoms.segmentize(step).to_crs(info["crs"]).values)
        frame = pyogrio.read_dataframe(
            path,
            layer=layer,
            bbox=tuple(aoi_layer.bounds),
            columns=None if columns is None else list(columns),
        )
    if frame.crs is None:
        frame = frame.set_crs(info["crs"])
    n_bbox = len(frame)
    shapely.prepare(aoi_layer)
    frame = frame[shapely.intersects(aoi_layer, frame.geometry.values)].copy()
    if clip and len(frame):
        from agribound.postprocess.simplify import make_polygonal

        clipped = shapely.intersection(frame.geometry.values, aoi_layer)
        changed = ~shapely.equals(clipped, frame.geometry.values)
        # A polygon that overlaps the study area and also touches its edge elsewhere is
        # cut to a GeometryCollection (its area plus a line or point): keep the area.
        # A polygon that only touches the study area leaves nothing polygonal.
        polygonal = [make_polygonal(g) for g in clipped]
        frame = frame.set_geometry(gpd.GeoSeries(polygonal, index=frame.index, crs=frame.crs))
        frame["agribound:clipped"] = changed
        keep = ~(frame.geometry.isna() | frame.geometry.is_empty)
        frame = frame[keep].copy()
    if dst_crs is not None:
        frame = frame.to_crs(dst_crs)
    frame = frame.reset_index(drop=True)
    seconds = round(time.perf_counter() - start_time, 1)
    query = {
        "release": release["name"],
        "window": list(release["window"]),
        "path": path,
        "layer": layer,
        "remote": remote,
        "clip": bool(clip),
        "n_bbox": int(n_bbox),
        "n_returned": int(len(frame)),
        "seconds": seconds,
    }
    frame.attrs["csb_query"] = query
    logger.info(
        "CSB %s: %d polygons intersect the study area (%d in its bounding box), read in %.1f s",
        release["name"],
        len(frame),
        n_bbox,
        seconds,
    )
    if output_path is not None:
        from agribound.io.vector import write_vector

        write_vector(frame, output_path, format=output_format)
        if provenance:
            from agribound.provenance import write_provenance

            record = _provenance_record(study_area, year, query, aoi.to_crs("EPSG:4326"))
            frame.attrs["provenance_path"] = str(write_provenance(output_path, record))
    return frame


def _provenance_record(
    study_area: Any, year: int, query: dict[str, Any], aoi_4326: gpd.GeoDataFrame
) -> dict[str, Any]:
    """Provenance record written next to a saved :func:`query_csb` result."""
    import datetime as _dt
    import platform
    import sys

    from agribound._repro import collect_versions
    from agribound._version import __version__
    from agribound.ftw_query import _describe_study_area
    from agribound.provenance import PROVENANCE_SCHEMA_VERSION, to_jsonable

    return to_jsonable(
        {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "kind": "query_csb",
            "agribound_version": __version__,
            "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "data": CSB_DATA_LABEL,
            "parameters": {"study_area": _describe_study_area(study_area), "year": year},
            "aoi_bounds_4326": [float(v) for v in aoi_4326.total_bounds],
            "query": query,
            "versions": collect_versions(("pyogrio",)),
            "platform": platform.platform(),
            "python": sys.version,
        }
    )


__all__ = [
    "CSB_BASE_URL",
    "CSB_DATA_LABEL",
    "CSB_GDAL_OPTIONS",
    "CSB_REVISION",
    "CSB_WINDOW_YEARS",
    "CSB_YEARS",
    "csb_dataset",
    "csb_release",
    "query_csb",
]
