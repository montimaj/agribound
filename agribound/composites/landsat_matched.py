"""Matched Landsat 8/9 PAN and SR exports for controlled comparisons.

The regular source builders can merge different missions. This experiment
helper deliberately restricts both products to the same L8/9 scene IDs and
uses complete, common 30 m mask support before computing temporal medians.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from agribound.composites.gee import (
    LANDSAT_BANDS,
    ExportGrid,
    compute_export_grid,
    date_window,
    ee_geometry,
    export_ee_image,
    grid_footprint_4326,
    prepare_landsat_image,
    prepare_landsat_pan_image,
    resolve_export_crs,
)
from agribound.config import AgriboundConfig


class NoMatchedScenesError(ValueError):
    """Scene selection failed, with an inspectable manifest of eligible records."""

    def __init__(self, manifest):
        super().__init__("No matched Landsat 8/9 PAN and SR scenes in the requested window")
        self.manifest = manifest


def match_scenes(pan_records: list[dict], sr_records: list[dict]) -> dict:
    """Pair exact LANDSAT_SCENE_IDs, retaining product IDs and unmatched records."""

    def index(records):
        result = {}
        for row in records:
            key = row["scene_id"]
            if not key or key in result:
                raise ValueError(f"Missing or duplicate Landsat scene ID: {key!r}")
            result[key] = row
        return result

    pan, sr = index(pan_records), index(sr_records)
    common = sorted(pan.keys() & sr.keys())
    return {
        "pairs": [{"scene_id": key, "pan": pan[key], "sr": sr[key]} for key in common],
        "unmatched_pan": [pan[key] for key in sorted(pan.keys() - sr.keys())],
        "unmatched_sr": [sr[key] for key in sorted(sr.keys() - pan.keys())],
    }


def paired_collections(config: AgriboundConfig, geometry: Any) -> tuple[Any, Any, dict]:
    """Return exact scene-matched, jointly masked L8/9 image collections."""
    import ee

    crs = resolve_export_crs(config.export_crs, geometry)
    grid = compute_export_grid(geometry, crs, 30)
    region = ee_geometry(grid_footprint_4326(grid))
    records = {"pan": [], "sr": []}
    for mission in ("LC08", "LC09"):
        for kind, product in (("pan", "T1_TOA"), ("sr", "T1_L2")):
            cid = f"LANDSAT/{mission}/C02/{product}"
            col = (
                ee.ImageCollection(cid)
                .filterBounds(region)
                .filterDate(*date_window(config))
                .filter(ee.Filter.lte("CLOUD_COVER", config.cloud_cover_max))
            )
            info = ee.Dictionary(
                {
                    "scene": col.aggregate_array("LANDSAT_SCENE_ID"),
                    "index": col.aggregate_array("system:index"),
                    "product": col.aggregate_array("LANDSAT_PRODUCT_ID"),
                }
            ).getInfo()
            for scene, index, product_id in zip(
                info["scene"], info["index"], info["product"], strict=True
            ):
                records[kind].append(
                    {
                        "scene_id": scene,
                        "image_id": f"{cid}/{index}",
                        "product_id": product_id,
                        "mission": mission,
                    }
                )
    manifest = match_scenes(records["pan"], records["sr"])
    pans, srs = [], []
    projection30 = ee.Projection(crs, [30, 0, 0, 0, -30, 0])
    projection15 = ee.Projection(crs, [15, 0, 0, 0, -15, 0])
    for pair in manifest["pairs"]:
        pan = prepare_landsat_pan_image(ee.Image(pair["pan"]["image_id"]), pair["pan"]["mission"])
        sr = prepare_landsat_image(ee.Image(pair["sr"]["image_id"]), LANDSAT_BANDS)
        # Default nearest-neighbour regridding; no synthetic fine detail in SR.
        pan = pan.reproject(projection15)
        sr = sr.reproject(projection30)
        pan_valid30 = (
            pan.mask().reduceResolution(ee.Reducer.min(), maxPixels=16).reproject(projection30)
        )
        common = sr.mask().reduce(ee.Reducer.min()).And(pan_valid30)
        pans.append(pan.updateMask(common))
        srs.append(sr.updateMask(common))
    manifest.update(
        {
            "date_window_end_exclusive": list(date_window(config)),
            "cloud_cover_max": config.cloud_cover_max,
            "crs": crs,
            "qa_mask": "QA_PIXEL bits 0-4 on both products; all bands and four PAN subpixels valid",
            "missions": sorted({p["pan"]["mission"] for p in manifest["pairs"]}),
            "composite": "per-band median of identically matched observations with common masks",
            "sr_value_scale": "reflectance_x10000",
            "pan_value_scale": "unit TOA reflectance",
            "regridding": "nearest neighbour to aligned 30 m SR and 15 m PAN export grids",
        }
    )
    if not manifest["pairs"]:
        raise NoMatchedScenesError(manifest)
    return ee.ImageCollection.fromImages(pans), ee.ImageCollection.fromImages(srs), manifest


def export_matched(
    config: AgriboundConfig, geometry: Any, out_dir: Path
) -> tuple[Path, Path, dict]:
    """Export aligned matched median inputs, with manifest checks on reused files."""
    try:
        pan_col, sr_col, manifest = paired_collections(config, geometry)
    except NoMatchedScenesError as exc:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "scene_selection_failure.json").write_text(
            json.dumps({"status": "unmeasured", "error": str(exc), **exc.manifest}, indent=2),
            encoding="utf-8",
        )
        raise
    signature = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest["sha256"] = signature
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    grid30 = compute_export_grid(geometry, manifest["crs"], 30)
    from rasterio.transform import Affine

    grid15 = ExportGrid(
        grid30.crs, grid30.transform * Affine.scale(0.5), grid30.width * 2, grid30.height * 2
    )
    for source, collection, grid, bands, scale in (
        ("landsat-pan", pan_col, grid15, ["B8"], "unit"),
        ("landsat", sr_col, grid30, LANDSAT_BANDS, "reflectance_x10000"),
    ):
        path = out_dir / f"{source}.tif"
        if path.exists():
            import rasterio

            with rasterio.open(path) as src:
                if (
                    src.tags().get("AGRIBOUND_SCENE_MANIFEST_SHA256") == signature
                    and str(src.crs) == grid.crs
                    and src.transform == grid.transform
                    and src.width == grid.width
                    and src.height == grid.height
                    and src.count == len(bands)
                ):
                    continue
            raise ValueError(
                f"Input {path} belongs to a different scene manifest; use a new output directory"
            )
        tags = {
            "AGRIBOUND_SOURCE": source,
            "AGRIBOUND_VALUE_SCALE": scale,
            "AGRIBOUND_SCENE_MANIFEST_SHA256": signature,
            "AGRIBOUND_N_IMAGES": len(manifest["pairs"]),
            "AGRIBOUND_COLLECTIONS": ",".join(
                sorted(
                    {
                        p["pan" if source == "landsat-pan" else "sr"]["image_id"].rsplit("/", 1)[0]
                        for p in manifest["pairs"]
                    }
                )
            ),
            "AGRIBOUND_SENSORS": ",".join(manifest["missions"]),
            "AGRIBOUND_CLOUD_MASK": manifest["qa_mask"],
            "AGRIBOUND_DATE_START": date_window(config)[0],
            "AGRIBOUND_DATE_END_EXCLUSIVE": date_window(config)[1],
            "AGRIBOUND_RESOLUTION_M": grid.transform.a,
            "AGRIBOUND_COMPOSITE_METHOD": "median",
        }
        export_ee_image(
            collection.median(),
            path,
            grid=grid,
            dtype="float32",
            band_names=bands,
            tags=tags,
            max_requests=config.gee_max_requests,
        )
    return out_dir / "landsat-pan.tif", out_dir / "landsat.tif", manifest


def validate_matched_inputs(pan_path: Path, sr_path: Path, manifest: dict, geometry: Any) -> None:
    """Reject changed manifests, mismatched grids and different cloud support."""
    import numpy as np
    import rasterio

    from agribound.composites.pan_fusion import block_mean

    payload = {k: v for k, v in manifest.items() if k != "sha256"}
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if signature != manifest.get("sha256"):
        raise ValueError("Scene manifest checksum does not match its contents")
    grid = compute_export_grid(geometry, manifest["crs"], 30)
    with rasterio.open(sr_path) as sr, rasterio.open(pan_path) as pan:
        if (
            str(sr.crs) != grid.crs
            or sr.transform != grid.transform
            or (sr.width, sr.height) != (grid.width, grid.height)
            or sr.count != 6
            or pan.count != 1
            or pan.crs != sr.crs
            or pan.transform != sr.transform * rasterio.Affine.scale(0.5)
            or (pan.width, pan.height) != (2 * sr.width, 2 * sr.height)
        ):
            raise ValueError("Prepared input grids do not match this study area")
        if any(s.tags().get("AGRIBOUND_SCENE_MANIFEST_SHA256") != signature for s in (sr, pan)):
            raise ValueError("Prepared inputs do not match the scene manifest")
        sr_valid = np.isfinite(sr.read(masked=True).filled(np.nan)).all(axis=0)
        pan_valid = np.isfinite(pan.read(1, masked=True).filled(np.nan))
        coarse_pan_valid = block_mean(pan_valid.astype(float)) == 1
        if not np.array_equal(sr_valid, coarse_pan_valid):
            raise ValueError("Prepared PAN/SR inputs have different valid support")
