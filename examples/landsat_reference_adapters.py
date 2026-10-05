"""Reference adapters for the North American Landsat comparison (example 26)."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import box


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def get_json(url, params=None):
    if params:
        url += "?" + urlencode(params)
    with urlopen(url, timeout=120) as response:
        data = json.load(response)
    if "error" in data:
        raise ValueError(f"Provider error: {data['error']}")
    return data


def arcgis_snapshot(url, bbox, *, page_size=500, request=get_json):
    """Freeze object IDs first, then fetch every ID exactly once in WGS84.

    Avoid offset pagination on mutable services. A truncated, duplicated or
    changed response fails rather than silently yielding an incomplete reference.
    """
    metadata = request(url, {"f": "json"})
    oid = metadata.get("objectIdField") or metadata.get("objectIdFieldName")
    if not oid:
        oid = next((f["name"] for f in metadata["fields"] if f["type"] == "esriFieldTypeOID"), None)
    if not oid:
        raise ValueError("Feature service lacks an object ID field")
    query = url.rstrip("/") + "/query"
    selection = {
        "f": "json",
        "where": "1=1",
        "geometry": ",".join(map(str, bbox)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "returnIdsOnly": "true",
    }
    ids = request(query, selection).get("objectIds") or []
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in feature-service selection")
    ids = sorted(ids)
    limit = min(page_size, metadata.get("maxRecordCount", page_size))
    if limit < 1:
        raise ValueError("Invalid page size")
    features, pages = [], []
    for offset in range(0, len(ids), limit):
        wanted = ids[offset : offset + limit]
        params = {
            "f": "geojson",
            "objectIds": ",".join(map(str, wanted)),
            "outFields": "*",
            "outSR": 4326,
            "returnGeometry": "true",
        }
        data = request(query, params)
        rows = data.get("features", [])
        got = [r["properties"].get(oid, r.get("id")) for r in rows]
        if data.get("exceededTransferLimit") or sorted(got) != wanted:
            raise ValueError("Feature-service page is truncated, duplicated or changed")
        rows.sort(key=lambda r: r["properties"].get(oid, r.get("id")))
        features.extend(rows)
        pages.append({"url": query, "params": params, "n_features": len(rows)})
    return {"type": "FeatureCollection", "features": features}, {
        "metadata": metadata,
        "selection": selection,
        "object_ids": ids,
        "pages": pages,
        "output_crs": "EPSG:4326",
    }


def fetch_source(source, cache_dir, *, offline=False):
    """Download/query a pinned source, with atomic writes and checksum validation."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / source["filename"]
    if target.exists():
        if file_sha256(target) != source["sha256"]:
            raise ValueError(f"Reference checksum mismatch: {target}")
        return target
    if offline:
        raise ValueError(f"Offline reference missing: {target}")
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        if source["adapter"] == "arcgis":
            data, manifest = arcgis_snapshot(source["url"], source["query_bbox"])
            temporary.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")
            target.with_suffix(".query.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        else:
            with urlopen(source["url"], timeout=180) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        if file_sha256(temporary) != source["sha256"]:
            raise ValueError("Downloaded source differs from the frozen reference checksum")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def safe_extract(archive_path, destination):
    """Extract only after checking every resolved path stays in the destination."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination):
                raise ValueError("Archive member escapes extraction directory")
        archive.extractall(destination)


def read_reference(path, source, bbox):
    """Spatially subset in the file's CRS, without guessing coordinate systems."""
    if source["adapter"] == "zip-shapefile":
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(".shp")]
        member = source.get("member")
        if member is None:
            if len(members) != 1:
                raise ValueError("Choose an explicit shapefile archive member")
            member = members[0]
        if member not in members:
            raise ValueError("Shapefile member missing from archive")
        dataset = f"/vsizip/{path.resolve().as_posix()}/{member}"
    elif source["adapter"] == "zip-geodatabase":
        destination = path.with_suffix("")
        if not destination.exists():
            safe_extract(path, destination)
        databases = list(destination.rglob("*.gdb"))
        if len(databases) != 1:
            raise ValueError("Expected one file geodatabase")
        dataset = databases[0]
    elif source["adapter"] in ("arcgis", "local"):
        dataset = path
    else:
        raise ValueError(f"Unknown reference adapter: {source['adapter']}")
    options = {"layer": source["layer"]} if source.get("layer") else {}
    info = pyogrio.read_info(dataset, **options)
    if not info["crs"]:
        raise ValueError("Reference lacks CRS")
    bounds = tuple(gpd.GeoSeries([box(*bbox)], crs=4326).to_crs(info["crs"]).total_bounds)
    if source.get("encoding"):
        options["encoding"] = source["encoding"]
    return gpd.read_file(dataset, bbox=bounds, **options)


def filter_reference(frame, rules, *, year=None):
    """Explicit attribute rules; active declarations require a known production."""
    keep = pd.Series(True, index=frame.index)
    for rule in rules:
        column, operation, value = rule["column"], rule["op"], rule["value"]
        if column not in frame:
            raise ValueError(f"Reference lacks filter column: {column}")
        data = frame[column]
        if operation == "in":
            selected = data.isin(value)
        elif operation == "not-in":
            selected = ~data.isin(value)
        elif operation == "ge":
            selected = pd.to_numeric(data, errors="coerce") >= value
        elif operation == "year":
            selected = pd.to_datetime(data, errors="coerce").dt.year == value
        else:
            raise ValueError(f"Unsupported filter operation: {operation}")
        keep &= selected.fillna(False)
    if year is not None and "AN" in frame and not (frame.loc[keep, "AN"] == year).all():
        raise ValueError("Declaration year disagrees with the frozen reference year")
    return frame.loc[keep].copy()


def select_coverage(predicted, coverage):
    """Keep whole predictions whose representative points are in known coverage.

    Precision is conditional on these mapped footprints. Unknown territory is
    excluded, not asserted to be non-agricultural. Polygons are never clipped.
    """
    if predicted.crs is None or coverage.crs is None:
        raise ValueError("Predictions and coverage must have a CRS")
    mask = coverage.to_crs(predicted.crs).geometry.union_all()
    keep = predicted.geometry.representative_point().map(mask.covers)
    return predicted.loc[keep].copy(), {
        "criterion": "Whole polygons with representative points covered by the fixed mask",
        "n_predictions_in_aoi": len(predicted),
        "n_predictions_evaluated": int(keep.sum()),
        "n_predictions_unknown_coverage": int((~keep).sum()),
    }
