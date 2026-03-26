"""
I/O utilities for raster and vector data.

Provides functions for reading/writing GeoTIFF files, vector formats
(GeoJSON, GeoPackage, GeoParquet), and CRS handling.
"""

from agribound.io.crs import get_equal_area_crs, get_utm_crs, reproject_raster
from agribound.io.raster import get_raster_info, read_raster, write_raster
from agribound.io.vector import read_vector, write_vector

__all__ = [
    "read_raster",
    "write_raster",
    "get_raster_info",
    "read_vector",
    "write_vector",
    "reproject_raster",
    "get_utm_crs",
    "get_equal_area_crs",
]
