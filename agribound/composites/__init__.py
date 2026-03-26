"""
Composite builders for satellite imagery acquisition.

Handles creating annual composites from various satellite sources via
Google Earth Engine or loading local GeoTIFF files.
"""

from agribound.composites.base import CompositeBuilder, get_composite_builder, list_sources

__all__ = ["CompositeBuilder", "get_composite_builder", "list_sources"]
