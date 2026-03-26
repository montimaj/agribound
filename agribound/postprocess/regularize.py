"""
Polygon regularization and orthogonalization.

Snaps field boundary edges to regular shapes (orthogonal edges, smooth
curves) using geoai's geometry utilities when available.
"""

from __future__ import annotations

import logging

import geopandas as gpd

logger = logging.getLogger(__name__)


def regularize_polygons(
    gdf: gpd.GeoDataFrame,
    method: str = "adaptive",
    angle_threshold: float = 15.0,
) -> gpd.GeoDataFrame:
    """Regularize field boundary polygon geometry.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygons.
    method : str
        Regularization method: ``"orthogonal"``, ``"adaptive"``
        (default), or ``"none"``.
    angle_threshold : float
        Maximum deviation angle for orthogonalization (degrees).

    Returns
    -------
    geopandas.GeoDataFrame
        Regularized polygons.
    """
    if len(gdf) == 0 or method == "none":
        return gdf

    try:
        if method == "orthogonal":
            from geoai.utils.geometry import orthogonalize

            result = gdf.copy()
            result["geometry"] = result.geometry.apply(
                lambda g: orthogonalize(g, angle_threshold=angle_threshold) if g.is_valid else g
            )
        elif method == "adaptive":
            from geoai.utils.geometry import adaptive_regularization

            result = gdf.copy()
            result["geometry"] = result.geometry.apply(
                lambda g: adaptive_regularization(g) if g.is_valid else g
            )
        else:
            logger.warning("Unknown regularization method %r, skipping", method)
            return gdf

        # Validate results
        result = result[result.geometry.is_valid & ~result.geometry.is_empty]
        logger.info("Regularized %d polygons (method=%s)", len(result), method)
        return result.reset_index(drop=True)

    except ImportError:
        logger.info(
            "geoai not installed — skipping regularization. "
            "Install with: pip install agribound[geoai]"
        )
        return gdf
    except Exception as exc:
        logger.warning("Regularization failed: %s — returning original polygons", exc)
        return gdf
