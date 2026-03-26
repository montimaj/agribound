"""
Google Earth Engine composite builder.

Handles annual composite generation for Landsat, Sentinel-2, HLS, NAIP,
and SPOT satellite sources using GEE. Supports multiple export methods
(local download, Google Drive, Google Cloud Storage) and automatic chunking
for large study areas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from agribound.composites.base import SOURCE_REGISTRY, CompositeBuilder
from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cloud masking functions (per source)
# ---------------------------------------------------------------------------


def _mask_landsat_clouds(image):
    """Apply QA_PIXEL cloud mask to a Landsat image."""

    qa = image.select("QA_PIXEL")
    # Bits 3 (cloud) and 4 (cloud shadow)
    cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
    return image.updateMask(cloud_mask)


def _mask_s2_clouds(image):
    """Apply SCL cloud mask to a Sentinel-2 image."""

    scl = image.select("SCL")
    # SCL classes: 3=cloud shadow, 8=cloud medium, 9=cloud high, 10=cirrus
    mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return image.updateMask(mask)


def _mask_hls_clouds(image):
    """Apply Fmask cloud mask to an HLS image."""

    fmask = image.select("Fmask")
    # Bits 1 (cloud) and 3 (cloud shadow)
    cloud_mask = fmask.bitwiseAnd(1 << 1).eq(0).And(fmask.bitwiseAnd(1 << 3).eq(0))
    return image.updateMask(cloud_mask)


# ---------------------------------------------------------------------------
# Source-specific collection builders
# ---------------------------------------------------------------------------


def _build_landsat_collection(config: AgriboundConfig, geometry):
    """Build a Landsat annual composite with all spectral bands.

    Automatically selects the appropriate Landsat missions based on the
    target year:

    - Landsat 5 TM (1984--2012): Collection 2, Level 2
    - Landsat 7 ETM+ (1999--2024): Collection 2, Level 2
    - Landsat 8 OLI (2013--present): Collection 2, Level 2
    - Landsat 9 OLI-2 (2021--present): Collection 2, Level 2

    Multiple overlapping missions are merged for better temporal coverage.
    All spectral SR bands are downloaded. L5/7 band names are harmonized
    to match L8/9 naming for cross-mission merging.

    Output bands (L8/9 naming): SR_B2, SR_B3, SR_B4, SR_B5, SR_B6, SR_B7
    """
    import ee

    start_date, end_date = _get_date_range(config)
    year = config.year

    # L8/9 spectral bands (output naming convention)
    l89_bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
    # L5/7 equivalent bands (different numbering, same wavelengths)
    l57_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]

    collections = []

    # Landsat 5 TM (1984–2012): rename to L8/9 convention for merging
    if year <= 2012:
        lt05 = (
            ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .filter(ee.Filter.lt("CLOUD_COVER", config.cloud_cover_max))
            .map(_mask_landsat_clouds)
            .select(l57_bands, l89_bands)
        )
        collections.append(lt05)

    # Landsat 7 ETM+ (1999–2024): same band names as Landsat 5
    # Note: SLC-off after May 2003 causes striping; median compositing mitigates this
    if 1999 <= year <= 2024:
        le07 = (
            ee.ImageCollection("LANDSAT/LE07/C02/T1_L2")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .filter(ee.Filter.lt("CLOUD_COVER", config.cloud_cover_max))
            .map(_mask_landsat_clouds)
            .select(l57_bands, l89_bands)
        )
        collections.append(le07)

    # Landsat 8 OLI (2013–present): keep native band names
    if year >= 2013:
        lc08 = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .filter(ee.Filter.lt("CLOUD_COVER", config.cloud_cover_max))
            .map(_mask_landsat_clouds)
            .select(l89_bands)
        )
        collections.append(lc08)

    # Landsat 9 OLI-2 (2021–present): same band numbering as Landsat 8
    if year >= 2021:
        lc09 = (
            ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
            .filterDate(start_date, end_date)
            .filterBounds(geometry)
            .filter(ee.Filter.lt("CLOUD_COVER", config.cloud_cover_max))
            .map(_mask_landsat_clouds)
            .select(l89_bands)
        )
        collections.append(lc09)

    if not collections:
        raise ValueError(f"No Landsat missions available for year {year}")

    # Merge all available missions
    merged = collections[0]
    for col in collections[1:]:
        merged = merged.merge(col)

    return merged, 30


def _build_sentinel2_collection(config: AgriboundConfig, geometry):
    """Build a Sentinel-2 annual composite with all spectral bands.

    Output bands: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
    """
    import ee

    start_date, end_date = _get_date_range(config)

    s2_bands = SOURCE_REGISTRY["sentinel2"]["all_bands"]

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", config.cloud_cover_max))
        .map(_mask_s2_clouds)
        .select(s2_bands)
    )

    return collection, 10


def _build_hls_collection(config: AgriboundConfig, geometry):
    """Build a Harmonized Landsat-Sentinel annual composite with all spectral bands.

    Downloads all 7 harmonized spectral bands common to HLSL30 and HLSS30.
    HLSS B8A is renamed to B5 for compatibility with HLSL naming.

    Output bands: B1, B2, B3, B4, B5, B6, B7
    """
    import ee

    start_date, end_date = _get_date_range(config)

    hls_bands = SOURCE_REGISTRY["hls"]["all_bands"]

    hlsl = (
        ee.ImageCollection("NASA/HLS/HLSL30/v002")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUD_COVERAGE", config.cloud_cover_max))
        .map(_mask_hls_clouds)
        .select(hls_bands)
    )

    # HLSS30 uses B8A for NIR narrow; rename to B5 to match HLSL30
    hlss_src = ["B1", "B2", "B3", "B4", "B8A", "B6", "B7"]
    hlss = (
        ee.ImageCollection("NASA/HLS/HLSS30/v002")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .filter(ee.Filter.lt("CLOUD_COVERAGE", config.cloud_cover_max))
        .map(_mask_hls_clouds)
        .select(hlss_src, hls_bands)
    )

    merged = hlsl.merge(hlss)
    return merged, 30


def _build_naip_collection(config: AgriboundConfig, geometry):
    """Build a NAIP imagery collection with all bands.

    NAIP is acquired periodically (every 2-3 years) so no median composite
    is created. Instead, the most recent image for the target year is used.

    Output bands: R, G, B, N
    """
    import ee

    # NAIP may not have imagery for every year — expand window
    year = config.year
    naip_bands = SOURCE_REGISTRY["naip"]["all_bands"]
    collection = (
        ee.ImageCollection("USDA/NAIP/DOQQ")
        .filterBounds(geometry)
        .filter(ee.Filter.calendarRange(year - 1, year + 1, "year"))
        .select(naip_bands)
    )

    return collection, 1


def _build_spot_collection(config: AgriboundConfig, geometry):
    """Build a SPOT 6/7 annual composite with all available bands.

    SPOT data in GEE has TOA reflectance values in [0, 10000].

    Output bands: R, G, B
    """
    import ee

    start_date, end_date = _get_date_range(config)

    spot_bands = SOURCE_REGISTRY["spot"]["all_bands"]
    collection = (
        ee.ImageCollection("AIRBUS/SPOT6_7")
        .filterDate(start_date, end_date)
        .filterBounds(geometry)
        .select(spot_bands)
    )

    return collection, 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_date_range(config: AgriboundConfig) -> tuple[str, str]:
    """Get the date range for filtering imagery."""
    if config.date_range is not None:
        return config.date_range
    return f"{config.year}-01-01", f"{config.year}-12-31"


def _apply_composite_method(collection, method: str, source: str):
    """Apply the compositing method to a collection.

    Parameters
    ----------
    collection : ee.ImageCollection
        Filtered and cloud-masked image collection.
    method : str
        Compositing method: ``"median"``, ``"greenest"``, or ``"max_ndvi"``.
    source : str
        Satellite source name (used to look up NIR/R band names).
    """
    if method == "median":
        return collection.median()
    elif method in ("greenest", "max_ndvi"):
        canonical = SOURCE_REGISTRY.get(source, {}).get("canonical_bands") or {}
        nir_band = canonical.get("NIR")
        red_band = canonical.get("R")
        if not nir_band or not red_band:
            logger.warning(
                "Source %r has no NIR/R bands for NDVI composite; falling back to median",
                source,
            )
            return collection.median()

        # Get the original spectral band names (exclude NDVI after qualityMosaic)
        all_bands = SOURCE_REGISTRY[source]["all_bands"]

        def add_ndvi(image):
            ndvi = image.normalizedDifference([nir_band, red_band]).rename("NDVI")
            return image.addBands(ndvi)

        with_ndvi = collection.map(add_ndvi)
        return with_ndvi.qualityMosaic("NDVI").select(all_bands)
    else:
        return collection.median()


# ---------------------------------------------------------------------------
# Tiling and download
# ---------------------------------------------------------------------------


def _compute_tiles(
    bounds: tuple[float, float, float, float],
    resolution_m: float,
    max_tile_pixels: int,
) -> list[tuple[float, float, float, float]]:
    """Split a bounding box into tiles if it exceeds the pixel limit.

    Parameters
    ----------
    bounds : tuple
        ``(min_lon, min_lat, max_lon, max_lat)``.
    resolution_m : float
        Pixel resolution in meters.
    max_tile_pixels : int
        Maximum pixels per tile dimension.

    Returns
    -------
    list of tuples
        List of tile bounding boxes.
    """
    from agribound.io.crs import estimate_pixel_count

    total_pixels = estimate_pixel_count(bounds, resolution_m)
    max_total = max_tile_pixels * max_tile_pixels

    if total_pixels <= max_total:
        return [bounds]

    min_lon, min_lat, max_lon, max_lat = bounds

    # Compute approximate pixel dimensions
    center_lat = (min_lat + max_lat) / 2
    m_per_deg_lon = 111_320.0 * np.cos(np.radians(center_lat))
    m_per_deg_lat = 111_320.0

    width_px = (max_lon - min_lon) * m_per_deg_lon / resolution_m
    height_px = (max_lat - min_lat) * m_per_deg_lat / resolution_m

    n_cols = max(1, int(np.ceil(width_px / max_tile_pixels)))
    n_rows = max(1, int(np.ceil(height_px / max_tile_pixels)))

    tile_width = (max_lon - min_lon) / n_cols
    tile_height = (max_lat - min_lat) / n_rows

    tiles = []
    for row in range(n_rows):
        for col in range(n_cols):
            tile_bounds = (
                min_lon + col * tile_width,
                min_lat + row * tile_height,
                min_lon + (col + 1) * tile_width,
                min_lat + (row + 1) * tile_height,
            )
            tiles.append(tile_bounds)

    logger.info(
        "Split study area into %d tiles (%d x %d) for parallel download",
        len(tiles),
        n_cols,
        n_rows,
    )
    return tiles


def _download_tile_local(
    composite,
    geometry,
    resolution_m: float,
    output_path: str,
    crs: str = "EPSG:4326",
    max_retries: int = 3,
) -> str:
    """Download a composite tile via ``ee.Image.getDownloadURL``.

    Parameters
    ----------
    composite : ee.Image
        Composited GEE image.
    geometry : ee.Geometry
        Region of interest.
    resolution_m : float
        Pixel resolution in meters.
    output_path : str
        Local file path for the GeoTIFF.
    crs : str
        Output CRS.
    max_retries : int
        Number of retry attempts on failure or corrupted download.

    Returns
    -------
    str
        Path to the downloaded GeoTIFF.
    """
    import time

    import rasterio

    try:
        import geedim  # noqa: F401 — registers the ee.Image.gd accessor
    except ImportError:
        raise ImportError(
            "geedim is required for local GEE downloads. Install with: pip install agribound[gee]"
        ) from None

    logger.info("Downloading composite to %s", output_path)

    for attempt in range(1, max_retries + 1):
        try:
            prepared = composite.gd.prepareForExport(
                region=geometry,
                scale=resolution_m,
                crs=crs,
            )
            prepared.gd.toGeoTIFF(output_path, overwrite=True)

            # Validate the downloaded file
            with rasterio.open(output_path) as ds:
                if ds.width == 0 or ds.height == 0:
                    raise RuntimeError("Downloaded raster has zero dimensions")
            logger.info("  Download OK: %d×%d, %d bands", ds.width, ds.height, ds.count)
            return output_path

        except Exception as exc:
            logger.warning("  Attempt %d/%d failed: %s", attempt, max_retries, exc)
            if Path(output_path).exists():
                Path(output_path).unlink()
            if attempt < max_retries:
                wait = 2**attempt
                logger.info("  Retrying in %ds...", wait)
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Failed to download composite after {max_retries} attempts: {exc}"
                ) from exc

    return output_path  # unreachable, but satisfies type checkers


def _export_to_drive(
    composite,
    geometry,
    resolution_m: float,
    description: str,
    folder: str = "agribound_exports",
    crs: str = "EPSG:4326",
) -> str:
    """Export a composite to Google Drive.

    Parameters
    ----------
    composite : ee.Image
        Composited GEE image.
    geometry : ee.Geometry
        Region of interest.
    resolution_m : float
        Pixel resolution in meters.
    description : str
        Task description and filename.
    folder : str
        Google Drive folder.
    crs : str
        Output CRS.

    Returns
    -------
    str
        Description of the export task (download manually from Drive).
    """
    import ee

    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=description,
        folder=folder,
        region=geometry,
        scale=resolution_m,
        crs=crs,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    logger.info("GEE export task started: %s (check Google Drive/%s/)", description, folder)
    return f"gdrive://{folder}/{description}"


def _export_to_gcs(
    composite,
    geometry,
    resolution_m: float,
    bucket: str,
    file_prefix: str,
    crs: str = "EPSG:4326",
) -> str:
    """Export a composite to Google Cloud Storage.

    Parameters
    ----------
    composite : ee.Image
        Composited GEE image.
    geometry : ee.Geometry
        Region of interest.
    resolution_m : float
        Pixel resolution in meters.
    bucket : str
        GCS bucket name.
    file_prefix : str
        File prefix in the bucket.
    crs : str
        Output CRS.

    Returns
    -------
    str
        GCS URI for the exported file.
    """
    import ee

    task = ee.batch.Export.image.toCloudStorage(
        image=composite,
        description=file_prefix,
        bucket=bucket,
        fileNamePrefix=f"agribound/{file_prefix}",
        region=geometry,
        scale=resolution_m,
        crs=crs,
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    logger.info("GEE export to GCS started: gs://%s/agribound/%s", bucket, file_prefix)
    return f"gs://{bucket}/agribound/{file_prefix}"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

# Map source names to collection builder functions
_COLLECTION_BUILDERS = {
    "landsat": _build_landsat_collection,
    "sentinel2": _build_sentinel2_collection,
    "hls": _build_hls_collection,
    "naip": _build_naip_collection,
    "spot": _build_spot_collection,
}


class GEECompositeBuilder(CompositeBuilder):
    """Composite builder for Google Earth Engine satellite sources.

    Handles Landsat, Sentinel-2, HLS, NAIP, and SPOT imagery. Creates
    annual cloud-free composites and downloads them as local GeoTIFFs.

    For large study areas, automatically tiles the download using dask
    for parallel processing.
    """

    def build(self, config: AgriboundConfig) -> str:
        """Build and download an annual satellite composite.

        Parameters
        ----------
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        str
            Path to the downloaded composite GeoTIFF.
        """

        from agribound.auth import setup_gee
        from agribound.io.vector import get_study_area_bounds, read_study_area

        # Initialize GEE
        setup_gee(project=config.gee_project)

        # Load study area
        study_gdf = read_study_area(config.study_area)
        bounds = get_study_area_bounds(study_gdf)

        # Convert study area to GEE geometry
        geometry = self._gdf_to_ee_geometry(study_gdf)

        # Build collection
        builder_fn = _COLLECTION_BUILDERS.get(config.source)
        if builder_fn is None:
            raise ValueError(f"No GEE collection builder for source: {config.source}")

        collection, resolution_m = builder_fn(config, geometry)

        # Apply compositing method
        if config.source == "naip":
            # NAIP: use mosaic (most recent on top)
            composite = collection.mosaic()
        else:
            composite = _apply_composite_method(
                collection, config.composite_method, config.source
            )

        # Clip to study area
        composite = composite.clip(geometry)

        # Prepare output
        cache_dir = config.get_working_dir()
        base_name = f"{config.source}_{config.year}_composite"

        # Return cached composite if it already exists (avoids re-downloading
        # when multiple engines use the same source/year).
        cached_tif = cache_dir / f"{base_name}.tif"
        if cached_tif.exists():
            logger.info("Using cached composite: %s", cached_tif)
            return str(cached_tif)

        # Check if tiling is needed
        tiles = _compute_tiles(bounds, resolution_m, config.tile_size)

        if len(tiles) == 1:
            # Single tile — direct download
            output_path = str(cache_dir / f"{base_name}.tif")
            return self._download_single(composite, geometry, resolution_m, output_path, config)
        else:
            # Multiple tiles — parallel download and merge
            return self._download_tiled(
                composite, tiles, resolution_m, base_name, cache_dir, config
            )

    def _download_single(
        self,
        composite,
        geometry,
        resolution_m: float,
        output_path: str,
        config: AgriboundConfig,
    ) -> str:
        """Download a single (non-tiled) composite."""
        if config.export_method == "local":
            return _download_tile_local(composite, geometry, resolution_m, output_path)
        elif config.export_method == "gdrive":
            desc = Path(output_path).stem
            return _export_to_drive(composite, geometry, resolution_m, desc)
        elif config.export_method == "gcs":
            prefix = Path(output_path).stem
            return _export_to_gcs(composite, geometry, resolution_m, config.gcs_bucket, prefix)
        else:
            raise ValueError(f"Unknown export method: {config.export_method}")

    def _download_tiled(
        self,
        composite,
        tiles: list[tuple[float, float, float, float]],
        resolution_m: float,
        base_name: str,
        cache_dir: Path,
        config: AgriboundConfig,
    ) -> str:
        """Download a composite in tiles and merge into a VRT.

        Uses dask for parallel tile downloads.
        """
        import ee

        tile_dir = cache_dir / f"{base_name}_tiles"
        tile_dir.mkdir(parents=True, exist_ok=True)

        tile_paths = []
        for i, tile_bounds in enumerate(tiles):
            tile_path = str(tile_dir / f"tile_{i:04d}.tif")
            tile_geom = ee.Geometry.Rectangle(list(tile_bounds))
            tile_composite = composite.clip(tile_geom)

            if config.export_method == "local":
                if Path(tile_path).exists():
                    logger.info("Using cached tile: %s", tile_path)
                else:
                    _download_tile_local(tile_composite, tile_geom, resolution_m, tile_path)
                tile_paths.append(tile_path)
            else:
                # For GDrive/GCS, export individual tiles
                desc = f"{base_name}_tile_{i:04d}"
                if config.export_method == "gdrive":
                    _export_to_drive(tile_composite, tile_geom, resolution_m, desc)
                elif config.export_method == "gcs":
                    _export_to_gcs(
                        tile_composite,
                        tile_geom,
                        resolution_m,
                        config.gcs_bucket,
                        desc,
                    )

        if config.export_method != "local":
            logger.warning(
                "Tiles exported to %s. Download and merge manually, "
                "or switch to export_method='local'.",
                config.export_method,
            )
            return str(tile_dir)

        # Merge tiles into a single GeoTIFF (all engines expect a plain TIF)
        merged_path = str(cache_dir / f"{base_name}.tif")
        self._merge_tiles(tile_paths, merged_path)
        return merged_path

    @staticmethod
    def _merge_tiles(tile_paths: list[str], output_path: str) -> str:
        """Merge tile GeoTIFFs into a single GeoTIFF.

        Uses GDAL Translate via VRT for efficiency, falling back to
        rasterio merge if GDAL Python bindings are unavailable.

        Parameters
        ----------
        tile_paths : list[str]
            Paths to tile GeoTIFFs.
        output_path : str
            Output GeoTIFF path.

        Returns
        -------
        str
            Path to the merged GeoTIFF.
        """
        try:
            from osgeo import gdal

            gdal.UseExceptions()
            vrt = gdal.BuildVRT("", tile_paths)
            gdal.Translate(output_path, vrt, creationOptions=["COMPRESS=DEFLATE"])
            del vrt
        except ImportError:
            import rasterio
            from rasterio.merge import merge

            datasets = [rasterio.open(p) for p in tile_paths]
            merged, transform = merge(datasets)
            for ds in datasets:
                ds.close()

            profile = rasterio.open(tile_paths[0]).profile.copy()
            profile.update(
                width=merged.shape[2],
                height=merged.shape[1],
                transform=transform,
                count=merged.shape[0],
                compress="deflate",
            )
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(merged)

        logger.info("Merged %d tiles into %s", len(tile_paths), output_path)
        return output_path

    @staticmethod
    def _gdf_to_ee_geometry(gdf):
        """Convert a GeoDataFrame to a GEE Geometry.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Input geodataframe.

        Returns
        -------
        ee.Geometry
            GEE geometry object.
        """
        import ee
        from shapely.geometry import mapping

        gdf_4326 = gdf.to_crs("EPSG:4326") if gdf.crs != "EPSG:4326" else gdf
        union_geom = gdf_4326.union_all()
        geojson = mapping(union_geom)
        return ee.Geometry(geojson)

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Return canonical band mapping for a GEE source.

        Parameters
        ----------
        source : str
            Satellite source name.

        Returns
        -------
        dict[str, str]
            Mapping of canonical names to source band names.
        """
        info = SOURCE_REGISTRY.get(source, {})
        return info.get("canonical_bands") or {}
