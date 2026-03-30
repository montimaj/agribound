"""
Interactive map visualization for field boundaries.

Displays predicted field boundary polygons overlaid on satellite basemaps
using leafmap. Supports static plots and interactive HTML maps.
"""

from __future__ import annotations

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

    # Set view — use set_center which persists to HTML (fit_bounds doesn't)
    if center is not None:
        m.set_center(center[1], center[0], zoom or 12)
    elif len(boundaries) > 0:
        bounds = boundaries.total_bounds  # minx, miny, maxx, maxy
        center_lon = (bounds[0] + bounds[2]) / 2
        center_lat = (bounds[1] + bounds[3]) / 2
        extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1])
        auto_zoom = 14 if extent < 0.1 else 12 if extent < 1 else 9 if extent < 10 else 6
        m.set_center(center_lon, center_lat, zoom or auto_zoom)

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
    import folium
    import numpy as np

    colors = ["#ff6600", "#0066ff", "#00cc44", "#cc00ff", "#ffcc00"]

    if labels is None:
        labels = [f"Layer {i + 1}" for i in range(len(boundaries_list))]

    # Reproject everything to EPSG:4326 upfront and collect bounds
    layers_4326 = []
    all_bounds = []

    for gdf in boundaries_list:
        if len(gdf) == 0:
            layers_4326.append(gdf)
            continue
        gdf_4326 = gdf.to_crs(epsg=4326) if gdf.crs is not None else gdf
        gdf_4326 = gdf_4326.explode(index_parts=False).reset_index(drop=True)
        gdf_4326 = gdf_4326[gdf_4326.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        gdf_4326 = gdf_4326[["geometry"]].copy()
        layers_4326.append(gdf_4326)
        all_bounds.append(gdf_4326.total_bounds)

    # Compute center and zoom from combined bounds
    if all_bounds:
        bounds = np.array(all_bounds)
        min_lon, min_lat = bounds[:, 0].min(), bounds[:, 1].min()
        max_lon, max_lat = bounds[:, 2].max(), bounds[:, 3].max()
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2
        extent = max(max_lon - min_lon, max_lat - min_lat)
        zoom = 14 if extent < 0.1 else 12 if extent < 1 else 9 if extent < 10 else 6
    else:
        center_lon, center_lat, zoom = 0, 0, 2

    # Pure folium map for clean HTML export and stable layer toggling
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=None,
    )

    # Satellite basemap
    basemap_urls = {
        "Esri.WorldImagery": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    }
    tile_url = basemap_urls.get(basemap, basemap_urls["Esri.WorldImagery"])
    folium.TileLayer(
        tiles=tile_url,
        attr="Esri" if "esri" in tile_url.lower() else "OpenStreetMap",
        name="Basemap",
    ).add_to(m)

    # Add each layer
    def _make_style_func(c: str):
        def _style(_feature):
            return {
                "color": c,
                "weight": 1,
                "opacity": 0.9,
                "fillColor": c,
                "fillOpacity": 0.1,
            }

        return _style

    for i, (gdf_4326, label) in enumerate(zip(layers_4326, labels, strict=False)):
        if len(gdf_4326) == 0:
            continue
        color = colors[i % len(colors)]
        folium.GeoJson(
            gdf_4326.__geo_interface__,
            name=label,
            style_function=_make_style_func(color),
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Simple HTML legend
    legend_html = '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
    legend_html += "background:white;padding:10px;border-radius:5px;"
    legend_html += 'box-shadow:0 0 5px rgba(0,0,0,0.3);font-size:13px;">'
    legend_html += "<b>Layers</b><br>"
    for i, label in enumerate(labels):
        color = colors[i % len(colors)]
        legend_html += f'<span style="color:{color};font-size:16px;">&#9632;</span> {label}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    if output_html:
        Path(output_html).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_html)

    return m
