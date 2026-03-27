"""
SamGeo boundary refinement.

Uses Meta's Segment Anything Model 2 (SAM2) via the segment-geospatial
package to **refine** field boundaries produced by other engines.

Instead of running SAM in automatic mode (which segments everything
blindly), the bounding boxes of existing field polygons are fed to SAM
as prompts.  SAM then produces pixel-accurate masks for each field,
yielding smoother and more precise boundaries than the original
raster-derived polygons.

This is intended as a **post-processing step**, not a standalone engine.
The pipeline activates it when ``engine_params["sam_refine"]`` is True.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.validation import make_valid

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

# SAM2 model variants (hosted on HuggingFace)
SAM2_MODELS = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small",
    "base_plus": "facebook/sam2-hiera-base-plus",
    "large": "facebook/sam2-hiera-large",
}


def refine_boundaries(
    gdf: gpd.GeoDataFrame,
    raster_path: str,
    config: AgriboundConfig,
) -> gpd.GeoDataFrame:
    """Refine field boundaries using SAM2 box-prompted segmentation.

    Each polygon's bounding box is used as a SAM prompt.  SAM produces a
    precise mask that is vectorized and replaces the original geometry.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Field boundaries from a delineation engine.
    raster_path : str
        Path to the input GeoTIFF (must contain RGB bands).
    config : AgriboundConfig
        Pipeline configuration.  Relevant ``engine_params``:

        - ``sam_model`` : str — SAM2 variant (``"tiny"``, ``"small"``,
          ``"base_plus"``, ``"large"``).  Default ``"large"``.

    Returns
    -------
    geopandas.GeoDataFrame
        Refined field boundaries.
    """
    try:
        from samgeo import SamGeo2
    except ImportError:
        raise ImportError(
            "segment-geospatial is required for SAM refinement. "
            "Install with: pip install agribound[samgeo]"
        ) from None

    if len(gdf) == 0:
        return gdf

    model_id = config.engine_params.get("sam_model", "large")
    model_id = SAM2_MODELS.get(model_id, model_id)
    device = config.resolve_device()
    cache_dir = config.get_working_dir()

    # Prepare RGB raster for SAM
    rgb_raster = str(cache_dir / "sam_refine_rgb.tif")
    if not Path(rgb_raster).exists():
        from agribound.engines.base import get_canonical_band_indices
        from agribound.io.raster import select_and_reorder_bands

        if config.source != "local":
            rgb_indices = get_canonical_band_indices(config.source, ["R", "G", "B"])
        else:
            rgb_indices = [1, 2, 3]
        select_and_reorder_bands(raster_path, rgb_raster, rgb_indices)

    # Read raster metadata for coordinate transforms
    with rasterio.open(rgb_raster) as src:
        raster_crs = src.crs
        transform = src.transform
        img_h, img_w = src.shape
        # Read as RGB numpy array for SAM
        rgb_array = src.read([1, 2, 3]).transpose(1, 2, 0)  # (H, W, 3)

    # Normalize to uint8 for SAM
    rgb_array = _normalize_to_uint8(rgb_array)

    # Ensure polygons are in raster CRS
    original_crs = gdf.crs
    if original_crs is not None and original_crs != raster_crs:
        gdf_proj = gdf.to_crs(raster_crs)
    else:
        gdf_proj = gdf

    # Convert polygon bounding boxes to pixel coordinates
    boxes_px = []
    valid_indices = []
    for idx, geom in enumerate(gdf_proj.geometry):
        minx, miny, maxx, maxy = geom.bounds
        # Geo coords → pixel coords
        col_min, row_max = ~transform * (minx, miny)
        col_max, row_min = ~transform * (maxx, maxy)
        # Clamp to image bounds
        x0 = max(0, int(col_min))
        y0 = max(0, int(row_min))
        x1 = min(img_w, int(col_max))
        y1 = min(img_h, int(row_max))
        if x1 > x0 and y1 > y0:
            boxes_px.append([x0, y0, x1, y1])
            valid_indices.append(idx)

    if not boxes_px:
        logger.warning("No valid bounding boxes for SAM refinement")
        return gdf

    logger.info(
        "Refining %d field boundaries with SAM2 (model=%s)",
        len(boxes_px),
        model_id,
    )

    # Initialize SAM2
    sam = SamGeo2(
        model_id=model_id,
        device=device,
        automatic=False,
    )
    sam.predictor.set_image(rgb_array)

    # Process in batches to avoid OOM
    batch_size = config.engine_params.get("sam_batch_size", 100)
    total_batches = (len(boxes_px) + batch_size - 1) // batch_size
    refined_geoms = []

    for batch_start in range(0, len(boxes_px), batch_size):
        batch_idx = batch_start // batch_size + 1
        batch_boxes = boxes_px[batch_start : batch_start + batch_size]
        logger.info(
            "SAM2 batch %d/%d: refining %d fields (%d/%d done)",
            batch_idx,
            total_batches,
            len(batch_boxes),
            batch_start,
            len(boxes_px),
        )
        input_boxes = np.array(batch_boxes)

        masks, scores, _ = sam.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        # masks shape: (N, 1, H, W) or (N, H, W) depending on version
        if masks.ndim == 4:
            masks = masks[:, 0]  # (N, H, W)

        for i, mask in enumerate(masks):
            geom = _mask_to_polygon(mask, transform)
            if geom is not None and not geom.is_empty:
                refined_geoms.append((batch_start + i, geom))

    # Replace geometries with SAM-refined versions
    result = gdf.copy()
    n_refined = 0
    for local_idx, geom in refined_geoms:
        orig_idx = valid_indices[local_idx]
        # Reproject refined geometry back to original CRS if needed
        if original_crs is not None and original_crs != raster_crs:
            from pyproj import Transformer
            from shapely.ops import transform as shapely_transform

            proj = Transformer.from_crs(raster_crs, original_crs, always_xy=True).transform
            geom = shapely_transform(proj, geom)
        result.geometry.iloc[orig_idx] = geom
        n_refined += 1

    logger.info("SAM2 refined %d / %d field boundaries", n_refined, len(gdf))
    return result


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a multi-band array to uint8 [0, 255] for SAM input."""
    if arr.dtype == np.uint8:
        return arr
    # Per-band percentile stretch
    out = np.zeros_like(arr, dtype=np.uint8)
    for b in range(arr.shape[2]):
        band = arr[:, :, b].astype(np.float64)
        p2, p98 = np.nanpercentile(band, [2, 98])
        if p98 > p2:
            band = (band - p2) / (p98 - p2) * 255
        band = np.clip(band, 0, 255)
        out[:, :, b] = band.astype(np.uint8)
    return out


def _mask_to_polygon(mask: np.ndarray, transform: rasterio.Affine) -> Polygon | MultiPolygon | None:
    """Convert a binary SAM mask to a shapely polygon in geo-coordinates."""
    from rasterio.features import shapes as rasterio_shapes

    mask_uint8 = mask.astype(np.uint8)
    polys = []
    for geom_dict, val in rasterio_shapes(mask_uint8, transform=transform):
        if val == 1:
            geom = shape(geom_dict)
            if not geom.is_valid:
                geom = make_valid(geom)
            if not geom.is_empty:
                polys.append(geom)

    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    # Return the largest polygon
    return max(polys, key=lambda g: g.area)
