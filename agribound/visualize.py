"""
Interactive map visualization for field boundaries.

Displays predicted field boundary polygons overlaid on satellite basemaps
using leafmap. Supports static plots and interactive HTML maps.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default polygon styling
_DEFAULT_STYLE = {
    "color": "#ff6600",
    "weight": 2,
    "opacity": 0.8,
    "fillColor": "#ff6600",
    "fillOpacity": 0.15,
}

#: Longest side, in pixels, of the composite image embedded by show_boundaries.
OVERLAY_MAX_SIDE = 2048


def _composite_overlay(
    path: str | Path,
    bands: dict[str, int] | None = None,
    max_side: int = OVERLAY_MAX_SIDE,
) -> tuple[str, tuple[tuple[float, float], tuple[float, float]]]:
    """RGB PNG of a GeoTIFF for an image overlay, and its latitude/longitude bounds.

    The R, G and B bands are chosen as the pipeline chooses them for
    ``source="local"``: *bands* (``{"R": 4, "G": 3, "B": 2}``, 1-based) where
    given, otherwise bands 1, 2 and 3; a raster with fewer than 3 bands (a
    panchromatic composite) shows band 1 in grey. They are warped to
    EPSG:3857 (the map's projection, so the image lines up with the basemap)
    with average resampling, at most *max_side* pixels on the long side.
    Non-uint8 data are stretched to 8 bits with
    :func:`agribound.io.raster.percentile_stretch_uint8` (per-band 1-99
    percentiles, as Delineate-Anything sees the image); uint8 data are kept.
    Pixels that are nodata or NaN in any band, or outside the raster, are
    transparent.

    Returns
    -------
    tuple
        ``("data:image/png;base64,...", ((south, west), (north, east)))``.
    """
    import base64
    import io

    import rasterio
    from affine import Affine
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds

    from agribound.engines.base import get_canonical_band_indices
    from agribound.io.raster import percentile_stretch_uint8

    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS")
        if bands is None and src.count < 3:
            indices = [1, 1, 1]  # grey
        else:
            indices = get_canonical_band_indices("local", ["R", "G", "B"], bands=bands)
        bad = [i for i in indices if not 1 <= i <= src.count]
        if bad:
            raise ValueError(f"RGB bands {indices} are out of range for a {src.count}-band raster")
        transform, width, height = calculate_default_transform(
            src.crs, "EPSG:3857", src.width, src.height, *src.bounds
        )
        scale = max(width, height) / max_side
        if scale > 1:
            new_width, new_height = max(1, round(width / scale)), max(1, round(height / scale))
            transform = transform @ Affine.scale(width / new_width, height / new_height)
            width, height = new_width, new_height
        planes = {}
        for band in dict.fromkeys(indices):  # each band once (grey repeats band 1)
            # NaN-initialised: pixels with no valid source data stay NaN (transparent).
            planes[band] = np.full((height, width), np.nan, dtype=np.float32)
            reproject(
                source=rasterio.band(src, band),
                destination=planes[band],
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs="EPSG:3857",
                dst_nodata=np.nan,
                resampling=Resampling.average,
            )
        rgb = np.stack([planes[band] for band in indices])
        is_uint8 = src.dtypes[indices[0] - 1] == "uint8"
    valid = np.isfinite(rgb).all(axis=0)
    if is_uint8:
        image = np.clip(np.nan_to_num(np.rint(rgb)), 0, 255).astype(np.uint8)
    else:
        image = percentile_stretch_uint8(rgb)
    alpha = np.where(valid, 255, 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(np.dstack([*image, alpha])).save(buffer, format="PNG")  # (h, w, 4) uint8: RGBA
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    left, top = transform.c, transform.f
    right, bottom = left + width * transform.a, top + height * transform.e
    west, south, east, north = transform_bounds("EPSG:3857", "EPSG:4326", left, bottom, right, top)
    return url, ((south, west), (north, east))


def _json_safe_value(value: Any) -> Any:
    """Return *value* as a JSON-serialisable Python object.

    Missing values (``None``, ``NaT``, ``NA``) and non-finite floats (which
    ``JSON.parse`` rejects in the exported HTML) become ``None``; dates, times
    and timestamps become ISO-8601 text; timedeltas become ISO-8601 durations;
    NumPy scalars become Python scalars; lists, tuples, arrays and dicts are
    converted element by element; geometries become WKT; anything else that
    JSON cannot encode becomes ``str(value)``.
    """
    from shapely.geometry.base import BaseGeometry

    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, _dt.timedelta):
        return pd.Timedelta(value).isoformat()
    if isinstance(value, _dt.datetime | _dt.date | _dt.time):  # includes pandas.Timestamp
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe_value(v) for v in value.tolist()]
    if isinstance(value, list | tuple | set):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    if isinstance(value, BaseGeometry):
        return value.wkt
    return str(value)


def _json_safe_copy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Copy of *gdf* whose attribute columns can be written to web-map JSON.

    leafmap 0.63.1 converts only ``datetime64[ns]`` and ``datetime64[ns, UTC]``
    columns (and does so in place on the frame it is given); any other
    datetime resolution, such as the ``datetime64[us, UTC]``
    ``determination:datetime`` column of every pipeline output, reaches the
    widget as ``pandas.Timestamp`` values and ``Map.to_html`` fails with
    "Object of type Timestamp is not JSON serializable". Because ipywidgets
    embeds the state of every live widget, that failed layer also breaks
    every later ``to_html`` call in the same process.

    Boolean, numeric (non-complex) and pandas string columns are left as they
    are. Datetime columns become ISO-8601 text (``NaT`` becomes missing) and
    every other column, including extra geometry columns, is converted value
    by value with :func:`_json_safe_value`. The input frame is not modified.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Frame to display.

    Returns
    -------
    geopandas.GeoDataFrame
        Converted copy with the same active geometry column and CRS.
    """
    out = gdf.copy()
    geometry_name = out.geometry.name
    for column in out.columns:
        if column == geometry_name:
            continue
        dtype = out[column].dtype
        if pd.api.types.is_bool_dtype(dtype) or isinstance(dtype, pd.StringDtype):
            continue
        if pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_complex_dtype(dtype):
            continue
        out[column] = out[column].astype(object).map(_json_safe_value).astype(object)
    return out


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
    satellite_bands: dict[str, int] | None = None,
) -> Any:
    """Display field boundaries on an interactive satellite basemap.

    Parameters
    ----------
    boundaries : geopandas.GeoDataFrame
        Field boundary polygons to display. The frame is not modified: the
        map is built from a copy in which datetime columns (such as the
        pipeline's ``determination:datetime``) are ISO-8601 text and other
        values JSON cannot encode are converted to text.
    basemap : str
        Basemap tile layer name (default ``"Esri.WorldImagery"``).
        Options: ``"Esri.WorldImagery"``, ``"Google.Satellite"``,
        ``"OpenStreetMap"``, ``"CartoDB.Positron"``.
    style : dict or None
        Polygon styling dict with keys like ``color``, ``weight``,
        ``fillOpacity``. Uses orange outline by default.
    satellite_tif : str or None
        Path to a local GeoTIFF to show above the basemap, as an RGB image
        embedded in the map (layer "Satellite Composite"), so it also shows
        in the HTML file after Python has exited. The image is at most
        :data:`OVERLAY_MAX_SIDE` pixels on its long side; see
        :func:`_composite_overlay` for bands, stretch and transparency.
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
    satellite_bands : dict or None
        1-based R, G, B bands of *satellite_tif*, as ``AgriboundConfig.bands``
        (``{"R": 4, "G": 3, "B": 2}``). Default: bands 1, 2 and 3 (band 1 in
        grey for a raster with fewer than 3 bands).

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
    # Datetime and other non-JSON attribute values break leafmap's HTML export
    # (and every later export in the process); convert them on a copy.
    boundaries = _json_safe_copy(boundaries)

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

    # Add satellite TIF if provided. Embedded as an image: a localtileserver
    # layer (leafmap's add_raster) points at a tile server that stops with
    # this process, and localtileserver 1.0 answers HTTP 500 for rasters
    # with NaN nodata (every agribound float composite).
    image_bounds = None
    if satellite_tif is not None:
        try:
            import ipyleaflet

            url, image_bounds = _composite_overlay(satellite_tif, bands=satellite_bands)
            m.add(ipyleaflet.ImageOverlay(url=url, bounds=image_bounds, name="Satellite Composite"))
        except Exception as exc:
            image_bounds = None
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
    elif len(boundaries) > 0 or image_bounds is not None:
        if len(boundaries) > 0:
            bounds = boundaries.total_bounds  # minx, miny, maxx, maxy
        else:  # no polygons: centre on the composite
            (south, west), (north, east) = image_bounds
            bounds = (west, south, east, north)
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
        List of boundary GeoDataFrames to compare. Only the (Multi)Polygon
        geometries are drawn; attribute columns are not added to the map, so
        datetime columns need no conversion. The frames are not modified.
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
        gdf_4326 = gdf_4326[[gdf_4326.geometry.name]].copy()
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
