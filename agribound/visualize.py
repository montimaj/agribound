"""
Interactive map visualization for field boundaries.

Displays predicted field boundary polygons overlaid on satellite basemaps
using leafmap. Supports static plots and interactive HTML maps.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd

logger = logging.getLogger(__name__)

# Default polygon styling
_DEFAULT_STYLE = {
    "color": "#ff6600",
    "weight": 2,
    "opacity": 0.8,
    "fillColor": "#ff6600",
    "fillOpacity": 0.15,
}


def show_boundaries(
    boundaries: gpd.GeoDataFrame,
    basemap: str = "Esri.WorldImagery",
    style: dict[str, Any] | None = None,
    satellite_tif: str | None = None,
    output_html: str | None = None,
    center: tuple[float, float] | None = None,
    zoom: int | None = None,
    width: str = "100%",
    height: str = "600px",
    layer_name: str = "Field Boundaries",
) -> Any:
    """Display field boundaries on an interactive satellite basemap.

    Parameters
    ----------
    boundaries : geopandas.GeoDataFrame
        Field boundary polygons to display.
    basemap : str
        Basemap tile layer name (default ``"Esri.WorldImagery"``).
        Options: ``"Esri.WorldImagery"``, ``"Google.Satellite"``,
        ``"OpenStreetMap"``, ``"CartoDB.Positron"``.
    style : dict or None
        Polygon styling dict with keys like ``color``, ``weight``,
        ``fillOpacity``. Uses orange outline by default.
    satellite_tif : str or None
        Path to a local GeoTIFF to overlay instead of the web basemap.
    output_html : str or None
        If provided, saves the map as a standalone HTML file.
    center : tuple[float, float] or None
        Map center ``(lat, lon)``. If *None*, auto-computed from data.
    zoom : int or None
        Initial zoom level. If *None*, auto-fit to data bounds.
    width : str
        Map width CSS value (default ``"100%"``).
    height : str
        Map height CSS value (default ``"600px"``).
    layer_name : str
        Name for the polygon layer in the layer control.

    Returns
    -------
    leafmap.Map
        Interactive leafmap Map object (displayable in Jupyter).

    Examples
    --------
    >>> import agribound
    >>> gdf = agribound.delineate(...)
    >>> m = agribound.show_boundaries(gdf)
    >>> m  # displays in Jupyter
    """
    try:
        import leafmap
    except ImportError:
        raise ImportError(
            "leafmap is required for visualization. Install with: pip install leafmap"
        ) from None

    if style is None:
        style = _DEFAULT_STYLE.copy()

    # Ensure EPSG:4326 for web mapping
    if boundaries.crs is not None and not boundaries.crs.equals("EPSG:4326"):
        boundaries = boundaries.to_crs("EPSG:4326")

    # Create map
    m = leafmap.Map(width=width, height=height)

    # Set basemap
    basemap_map = {
        "Esri.WorldImagery": "Esri.WorldImagery",
        "Google.Satellite": "SATELLITE",
        "OpenStreetMap": "OpenStreetMap",
        "CartoDB.Positron": "CartoDB.Positron",
    }
    basemap_key = basemap_map.get(basemap, basemap)
    try:
        m.add_basemap(basemap_key)
    except Exception:
        logger.warning("Could not add basemap %r, using default", basemap)

    # Add satellite TIF if provided
    if satellite_tif is not None:
        try:
            m.add_raster(satellite_tif, layer_name="Satellite Composite")
        except Exception as exc:
            logger.warning("Could not add satellite TIF: %s", exc)

    # Add field boundaries
    if len(boundaries) > 0:
        m.add_gdf(
            boundaries,
            layer_name=layer_name,
            style=style,
            info_mode="on_click",
        )

    # Set view
    if center is not None:
        m.set_center(center[1], center[0], zoom or 12)
    elif len(boundaries) > 0:
        bounds = boundaries.total_bounds  # minx, miny, maxx, maxy
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])

    # Save to HTML if requested
    if output_html is not None:
        output_path = Path(output_html)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        m.to_html(str(output_path))
        logger.info("Map saved to %s", output_path)

    return m


def show_comparison(
    boundaries_list: list[gpd.GeoDataFrame],
    labels: list[str] | None = None,
    basemap: str = "Esri.WorldImagery",
    output_html: str | None = None,
) -> Any:
    """Display multiple sets of field boundaries for comparison.

    Useful for comparing results from different engines or years.

    Parameters
    ----------
    boundaries_list : list[geopandas.GeoDataFrame]
        List of boundary GeoDataFrames to compare.
    labels : list[str] or None
        Labels for each set (e.g., engine names or years).
    basemap : str
        Basemap tile layer name.
    output_html : str or None
        If provided, saves the map as HTML.

    Returns
    -------
    leafmap.Map
        Interactive map with toggle-able layers.
    """
    try:
        import leafmap
    except ImportError:
        raise ImportError("leafmap is required for visualization.") from None

    colors = ["#ff6600", "#0066ff", "#00cc44", "#cc00ff", "#ffcc00"]

    if labels is None:
        labels = [f"Layer {i + 1}" for i in range(len(boundaries_list))]

    m = leafmap.Map()

    with contextlib.suppress(Exception):
        m.add_basemap(basemap)

    for i, (gdf, label) in enumerate(zip(boundaries_list, labels, strict=False)):
        if gdf.crs is not None and not gdf.crs.equals("EPSG:4326"):
            gdf = gdf.to_crs("EPSG:4326")

        color = colors[i % len(colors)]
        style = {
            "color": color,
            "weight": 2,
            "opacity": 0.8,
            "fillColor": color,
            "fillOpacity": 0.15,
        }

        if len(gdf) > 0:
            m.add_gdf(gdf, layer_name=label, style=style)

    # Fit to all bounds
    all_bounds = []
    for gdf in boundaries_list:
        if len(gdf) > 0:
            gdf_4326 = gdf.to_crs("EPSG:4326") if gdf.crs != "EPSG:4326" else gdf
            all_bounds.append(gdf_4326.total_bounds)

    if all_bounds:
        import numpy as np

        bounds = np.array(all_bounds)
        m.fit_bounds(
            [
                [bounds[:, 1].min(), bounds[:, 0].min()],
                [bounds[:, 3].max(), bounds[:, 2].max()],
            ]
        )

    if output_html:
        m.to_html(output_html)

    return m
