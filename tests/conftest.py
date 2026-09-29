"""Shared pytest fixtures for the agribound test suite."""

from __future__ import annotations

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Markers and global state
# ---------------------------------------------------------------------------


def pytest_configure(config):
    """Register the markers used by the suite (also declared in pyproject.toml)."""
    for line in (
        "slow: long-running tests",
        "gpu: tests that need a CUDA/MPS device",
        "gee: tests that need Google Earth Engine credentials",
        "network: tests that need internet access",
    ):
        config.addinivalue_line("markers", line)


@pytest.fixture(autouse=True)
def _reset_agribound_state():
    """Forget process-level caches (GEE init state, AOI fingerprints) between tests."""
    from agribound import _cache, auth

    auth._reset_state()
    _cache.clear_fingerprint_cache()
    yield
    auth._reset_state()
    _cache.clear_fingerprint_cache()


# ---------------------------------------------------------------------------
# Raster fixtures
# ---------------------------------------------------------------------------


def _write_tif(
    path, data, crs="EPSG:32611", bounds=(500000, 4000000, 500640, 4000640), nodata=None
):
    count, height, width = data.shape
    transform = from_bounds(*bounds, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


@pytest.fixture
def s2_reflectance_tif(tmp_path):
    """12-band Sentinel-2-like composite: float32 reflectance x 10000, NaN nodata (1.0 layout)."""
    rng = np.random.default_rng(0)
    data = rng.uniform(200, 4000, (12, 64, 64)).astype(np.float32)
    data[:, :2, :2] = np.nan
    return _write_tif(tmp_path / "s2_x10000.tif", data, nodata=float("nan"))


@pytest.fixture
def naip_uint8_tif(tmp_path):
    """4-band NAIP-like uint8 raster (R, G, B, N)."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, (4, 64, 64), dtype=np.uint8)
    return _write_tif(tmp_path / "naip_uint8.tif", data)


@pytest.fixture
def sample_rgb_tif(tmp_path):
    """Create a tiny 64x64 3-band GeoTIFF with CRS EPSG:32611 (UTM 11N)."""
    path = tmp_path / "rgb.tif"
    height, width, bands = 64, 64, 3
    transform = from_bounds(500000, 4000000, 500640, 4000640, width, height)
    data = np.random.randint(0, 10000, (bands, height, width), dtype=np.uint16)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint16",
        crs="EPSG:32611",
        transform=transform,
    ) as dst:
        dst.write(data)

    return str(path)


@pytest.fixture
def sample_rgbn_tif(tmp_path):
    """Create a tiny 64x64 4-band GeoTIFF with CRS EPSG:32611."""
    path = tmp_path / "rgbn.tif"
    height, width, bands = 64, 64, 4
    transform = from_bounds(500000, 4000000, 500640, 4000640, width, height)
    data = np.random.randint(0, 10000, (bands, height, width), dtype=np.uint16)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint16",
        crs="EPSG:32611",
        transform=transform,
    ) as dst:
        dst.write(data)

    return str(path)


# ---------------------------------------------------------------------------
# Vector / AOI fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_aoi_geojson(tmp_path):
    """Create a small GeoJSON polygon AOI file in EPSG:4326."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "test_aoi"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-117.0, 36.0],
                            [-117.0, 36.01],
                            [-116.99, 36.01],
                            [-116.99, 36.0],
                            [-117.0, 36.0],
                        ]
                    ],
                },
            }
        ],
    }
    path = tmp_path / "aoi.geojson"
    path.write_text(json.dumps(geojson))
    return str(path)


@pytest.fixture
def sample_geodataframe():
    """Create a GeoDataFrame with a few simple polygon geometries in EPSG:32611."""
    polys = [
        box(500000, 4000000, 500200, 4000200),
        box(500100, 4000100, 500300, 4000300),
        box(500500, 4000500, 500600, 4000600),
        box(500800, 4000800, 500810, 4000810),  # very small polygon
    ]
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4]},
        geometry=polys,
        crs="EPSG:32611",
    )
    return gdf


@pytest.fixture
def sample_reference_gdf():
    """Create a reference GeoDataFrame for evaluation tests in EPSG:32611."""
    polys = [
        box(500000, 4000000, 500200, 4000200),
        box(500500, 4000500, 500700, 4000700),
        box(501000, 4001000, 501200, 4001200),
    ]
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=polys,
        crs="EPSG:32611",
    )
    return gdf


# ---------------------------------------------------------------------------
# USGS fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_usgs_service_metadata():
    """Minimal ImageServer metadata used by mocked USGS tests."""
    return {
        "maxImageWidth": 4000,
        "maxImageHeight": 4000,
        "maxMosaicImageCount": 50,
    }


@pytest.fixture
def sample_usgs_query_features():
    """Mock ArcGIS ImageServer query response with two adjacent footprints in EPSG:3857."""
    return {
        "features": [
            {
                "attributes": {
                    "OBJECTID": 101,
                    "Name": "tile_a",
                    "State": "MI",
                    "Year": 2023,
                    "Category": 1,
                    "download_url": None,
                    "acquisition_date": 1688169600000,
                    "resolution_value": 1.0,
                    "resolution_units": "Meters",
                    "band_count": 4,
                },
                "geometry": {
                    "xmin": -13024450.0,
                    "ymin": 5160950.0,
                    "xmax": -13023850.0,
                    "ymax": 5161750.0,
                },
            },
            {
                "attributes": {
                    "OBJECTID": 102,
                    "Name": "tile_b",
                    "State": "MI",
                    "Year": 2023,
                    "Category": 1,
                    "download_url": None,
                    "acquisition_date": 1688256000000,
                    "resolution_value": 1.0,
                    "resolution_units": "Meters",
                    "band_count": 4,
                },
                "geometry": {
                    "xmin": -13023850.0,
                    "ymin": 5160950.0,
                    "xmax": -13023250.0,
                    "ymax": 5161750.0,
                },
            },
        ]
    }


@pytest.fixture
def sample_usgs_aoi_geojson(tmp_path):
    """Create a small GeoJSON AOI file in EPSG:4326 for USGS pipeline tests."""
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "test_usgs_aoi"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-117.0000, 42.0000],
                            [-117.0000, 42.0050],
                            [-116.9950, 42.0050],
                            [-116.9950, 42.0000],
                            [-117.0000, 42.0000],
                        ]
                    ],
                },
            }
        ],
    }
    path = tmp_path / "usgs_aoi.geojson"
    path.write_text(json.dumps(geojson))
    return str(path)


@pytest.fixture
def sample_export_tif(tmp_path):
    """Create a tiny 4-band exported TIFF in EPSG:3857."""
    path = tmp_path / "export.tif"
    height, width, bands = 64, 64, 4
    transform = from_bounds(-13024450, 5160950, -13023250, 5161750, width, height)
    data = np.random.randint(0, 255, (bands, height, width), dtype=np.uint8)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        dst.write(data)

    return str(path)
