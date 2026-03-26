"""
Agribound: Unified Agricultural Field Boundary Delineation Toolkit.

A Python package that combines satellite foundation models, embeddings, and
global training data for accurate agricultural field boundary mapping. Supports
multiple satellite sources (Landsat, Sentinel-2, HLS, NAIP, SPOT) and
delineation engines (Delineate-Anything, FTW, GeoAI, Prithvi, embeddings).

Basic Usage
-----------
>>> import agribound
>>> gdf = agribound.delineate(
...     study_area="area.geojson",
...     source="sentinel2",
...     year=2024,
...     engine="delineate-anything",
...     gee_project="my-project",
... )
"""

from agribound._version import __version__
from agribound.composites import list_sources
from agribound.config import AgriboundConfig
from agribound.engines import list_engines
from agribound.evaluate import evaluate
from agribound.pipeline import delineate
from agribound.visualize import show_boundaries

__all__ = [
    "__version__",
    "AgriboundConfig",
    "delineate",
    "evaluate",
    "list_engines",
    "list_sources",
    "show_boundaries",
]
