"""
I/O utilities for raster and vector data.

Provides functions for reading/writing GeoTIFF files, vector formats
(GeoJSON, GeoPackage, GeoParquet), and CRS handling.
"""

from agribound.io.raster import read_raster, write_raster, get_raster_info
from agribound.io.vector import read_vector, write_vector
from agribound.io.crs import reproject_raster, get_utm_crs, get_equal_area_crs

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
