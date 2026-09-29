"""
I/O utilities for raster and vector data.

Provides functions for reading/writing GeoTIFF files, vector formats
(GeoJSON, GeoPackage, GeoParquet), study-area parsing (files, GEE assets,
``bbox:`` strings, WKT), CRS/UTM helpers and radiometric value-scale
conversions.
"""

from agribound.io.crs import (
    get_equal_area_crs,
    get_utm_crs,
    reproject_raster,
    utm_crs_for_geometry,
    utm_zones_for_bounds,
)
from agribound.io.raster import (
    get_raster_info,
    percentile_stretch_uint8,
    read_raster,
    to_s2_dn,
    to_unit_reflectance,
    write_raster,
)
from agribound.io.vector import (
    read_config_study_area,
    read_study_area,
    read_vector,
    write_vector,
)

__all__ = [
    "read_raster",
    "write_raster",
    "get_raster_info",
    "to_unit_reflectance",
    "to_s2_dn",
    "percentile_stretch_uint8",
    "read_vector",
    "write_vector",
    "read_study_area",
    "read_config_study_area",
    "reproject_raster",
    "get_utm_crs",
    "get_equal_area_crs",
    "utm_crs_for_geometry",
    "utm_zones_for_bounds",
]
