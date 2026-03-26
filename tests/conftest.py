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
# Raster fixtures
# ---------------------------------------------------------------------------


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
