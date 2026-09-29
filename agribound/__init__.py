"""
Agribound: agricultural field boundary delineation toolkit.

Agribound runs published field-boundary models and embedding-based methods on
satellite and aerial imagery through one configuration and pipeline:

- **Sources**: Landsat 5/7/8/9, Sentinel-2, HLS, NAIP and SPOT 6/7 composites
  built on Google Earth Engine, USGS NAIP Plus, local GeoTIFFs, and
  pre-computed Google Satellite Embedding (AlphaEarth) and TESSERA embeddings
  (:func:`list_sources`).
- **Engines**: Delineate-Anything, Fields of The World (FTW), GeoAI Mask R-CNN,
  DINOv3, Prithvi-EO-2.0, embedding clustering and ensembles
  (:func:`list_engines`), with optional fine-tuning on reference boundaries and
  optional SAM refinement.
- **Post-processing**: polygon merging, area filtering, simplification, LULC
  crop filtering, fiboa-style metadata, evaluation against reference
  boundaries, and a provenance record for every run.

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

Heavy or optional parts are imported lazily: :func:`list_ftw_models`,
:func:`query_ftw`, :func:`show_boundaries` and the optional ``agent``
subpackage (``pip install "agribound[agent]"``).
"""

from __future__ import annotations

import importlib
from typing import Any

from agribound._version import __version__
from agribound.config import AgriboundConfig
from agribound.evaluate import evaluate
from agribound.pipeline import build_composite, delineate
from agribound.registry import list_engines, list_sources

# name -> (module, attribute); resolved on first access.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "list_ftw_models": ("agribound.engines.ftw", "list_ftw_models"),
    "query_ftw": ("agribound.ftw_query", "query_ftw"),
    "show_boundaries": ("agribound.visualize", "show_boundaries"),
}

__all__ = [
    "__version__",
    "AgriboundConfig",
    "build_composite",
    "delineate",
    "evaluate",
    "list_engines",
    "list_ftw_models",
    "list_sources",
    "query_ftw",
    "show_boundaries",
]


def _load_agent() -> Any:
    """Import ``agribound.agent`` and return the package.

    ``agribound.agent.agent`` imports only the standard library at module
    level, so this succeeds without the ``agent`` extra. The missing optional
    dependencies are reported when :func:`agribound.agent.agent` is called,
    as :class:`agribound.agent.errors.AgentDependencyError` (an
    :class:`ImportError` subclass with the ``pip install "agribound[agent]"``
    hint). The ImportError raised here covers only a broken installation.
    """
    try:
        importlib.import_module("agribound.agent.agent")
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ImportError(
            f"agribound.agent is unavailable ({missing} could not be imported). "
            'Install the optional dependencies with: pip install "agribound[agent]"'
        ) from exc
    return importlib.import_module("agribound.agent")


def __getattr__(name: str) -> Any:
    if name == "agent":
        return _load_agent()
    target = _LAZY_ATTRS.get(name)
    if target is not None:
        module_name, attr = target
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'agribound' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | {"agent"})
