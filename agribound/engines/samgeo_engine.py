"""
SamGeo boundary refinement.

Uses Meta's Segment Anything Model 2 (SAM2) via the segment-geospatial
package to **refine** field boundaries produced by other engines.

Instead of encoding the entire raster (which can be 100M+ pixels and
crash), each field's bounding box is cropped from the raster and fed
individually to SAM2.  SAM then produces a pixel-accurate mask for
each field, yielding smoother and more precise boundaries.

This is intended as a **post-processing step**, not a standalone engine.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.validation import make_valid

from agribound.config import AgriboundConfig

# Suppress noisy warnings from numpy (nodata crops) and GDAL (Memory driver)
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")
warnings.filterwarnings("ignore", message=".*Memory.*driver.*deprecated.*")
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# SAM2 model variants (hosted on HuggingFace)
SAM2_MODELS = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small",
    "base_plus": "facebook/sam2-hiera-base-plus",
    "large": "facebook/sam2-hiera-large",
}

# Minimum crop size for SAM (pixels). Small crops produce poor masks.
MIN_CROP_SIZE = 64
# Padding around the bounding box (fraction of box size)
CROP_PADDING = 0.15


def refine_boundaries(
    gdf: gpd.GeoDataFrame,
    raster_path: str,
    config: AgriboundConfig,
) -> gpd.GeoDataFrame:
    """Refine field boundaries using SAM2 box-prompted segmentation.

    For each polygon, crops a local window from the raster around the
    bounding box, runs SAM2 on the crop, and replaces the original
    geometry with the SAM2-refined mask.

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
        - ``sam_batch_size`` : int — Log progress every N fields.

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

    # Open raster for windowed reads
    src = rasterio.open(rgb_raster)
    raster_crs = src.crs
    raster_transform = src.transform
    img_h, img_w = src.shape

    # Ensure polygons are in raster CRS
    original_crs = gdf.crs
    if original_crs is not None and original_crs != raster_crs:
        gdf_proj = gdf.to_crs(raster_crs)
    else:
        gdf_proj = gdf

    # Initialize SAM2
    logger.info("Initializing SAM2 (model=%s, device=%s)", model_id, device)
    sam = SamGeo2(
        model_id=model_id,
        device=device,
        automatic=False,
    )

    # Process each field individually with cropped windows
    log_interval = config.engine_params.get("sam_batch_size", 100)
    result = gdf.copy()
    n_refined = 0
    n_failed = 0

    for i, (idx, geom) in enumerate(zip(gdf_proj.index, gdf_proj.geometry, strict=True)):
        if i % log_interval == 0:
            logger.info(
                "SAM2 refining field %d/%d (%d refined, %d failed)",
                i,
                len(gdf_proj),
                n_refined,
                n_failed,
            )

        try:
            refined_geom = _refine_single_field(sam, src, geom, raster_transform, img_h, img_w)
            if refined_geom is not None:
                # Map back to original CRS if needed
                if original_crs is not None and original_crs != raster_crs:
                    from pyproj import Transformer
                    from shapely.ops import transform as shapely_transform

                    proj = Transformer.from_crs(raster_crs, original_crs, always_xy=True).transform
                    refined_geom = shapely_transform(proj, refined_geom)
                result.loc[idx, "geometry"] = refined_geom
                n_refined += 1
            else:
                n_failed += 1
        except Exception as exc:
            if i < 5:  # Log first few failures for debugging
                logger.debug("SAM2 failed on field %d: %s", i, exc)
            n_failed += 1

    src.close()

    # Clean up invalid/empty geometries
    result = result[~result.geometry.is_empty & result.geometry.is_valid].reset_index(drop=True)

    logger.info("SAM2 refined %d / %d field boundaries (%d failed)", n_refined, len(gdf), n_failed)
    return result


def _refine_single_field(
    sam,
    src: rasterio.DatasetReader,
    geom: Polygon | MultiPolygon,
    raster_transform: rasterio.Affine,
    img_h: int,
    img_w: int,
) -> Polygon | MultiPolygon | None:
    """Refine a single field polygon using SAM2.

    Crops a padded window around the field's bounding box, runs SAM2,
    and returns the refined polygon in raster CRS.
    """
    minx, miny, maxx, maxy = geom.bounds

    # Geo coords → pixel coords
    col_min, row_max = ~raster_transform * (minx, miny)
    col_max, row_min = ~raster_transform * (maxx, maxy)

    # Add padding
    pad_w = (col_max - col_min) * CROP_PADDING
    pad_h = (row_max - row_min) * CROP_PADDING
    x0 = max(0, int(col_min - pad_w))
    y0 = max(0, int(row_min - pad_h))
    x1 = min(img_w, int(col_max + pad_w))
    y1 = min(img_h, int(row_max + pad_h))

    crop_w = x1 - x0
    crop_h = y1 - y0

    if crop_w < MIN_CROP_SIZE or crop_h < MIN_CROP_SIZE:
        return None

    # Read crop from raster
    window = Window(x0, y0, crop_w, crop_h)
    crop = src.read([1, 2, 3], window=window).transpose(1, 2, 0)  # (H, W, 3)
    crop = _normalize_to_uint8(crop)

    # Box prompt in crop-local pixel coordinates
    box_local = [
        int(col_min - x0),
        int(row_min - y0),
        int(col_max - x0),
        int(row_max - y0),
    ]

    # Run SAM2 on the crop
    sam.predictor.set_image(crop)

    import numpy as np

    masks, scores, _ = sam.predictor.predict(
        point_coords=None,
        point_labels=None,
        box=np.array([box_local]),
        multimask_output=False,
    )

    if masks.ndim == 4:
        masks = masks[:, 0]
    mask = masks[0]  # (H, W)

    # Convert mask to polygon in raster CRS
    crop_transform = rasterio.transform.from_bounds(
        *rasterio.transform.array_bounds(
            crop_h, crop_w, rasterio.windows.transform(window, raster_transform)
        ),
        crop_w,
        crop_h,
    )

    refined = _mask_to_polygon(mask, crop_transform)
    if refined is None or refined.is_empty:
        return None

    return refined


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize a multi-band array to uint8 [0, 255] for SAM input."""
    if arr.dtype == np.uint8:
        return arr
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
    return max(polys, key=lambda g: g.area)
