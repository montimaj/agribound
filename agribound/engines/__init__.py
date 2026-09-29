"""
Delineation engines for agricultural field boundary detection.

Each engine wraps a different model or approach for extracting field
boundary polygons from satellite imagery or embeddings. Engine modules are
imported lazily by :func:`get_engine`, so importing this package does not
pull in torch or other optional dependencies.
"""

from agribound.engines.base import (
    ENGINE_REGISTRY,
    DelineationEngine,
    get_canonical_band_indices,
    get_engine,
    get_engine_class,
    list_engines,
)

__all__ = [
    "ENGINE_REGISTRY",
    "DelineationEngine",
    "get_canonical_band_indices",
    "get_engine",
    "get_engine_class",
    "list_engines",
]
