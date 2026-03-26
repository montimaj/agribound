"""
Abstract base class for delineation engines and engine registry.

Each engine wraps a different model or approach for extracting field
boundary polygons from satellite imagery or embeddings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import geopandas as gpd

from agribound.config import AgriboundConfig

# ---------------------------------------------------------------------------
# Engine metadata registry
# ---------------------------------------------------------------------------

ENGINE_REGISTRY: dict[str, dict[str, Any]] = {
    "delineate-anything": {
        "name": "Delineate-Anything",
        "approach": "YOLO instance segmentation",
        "strengths": "Resolution-agnostic (1m–10m+), best for high-res imagery (NAIP, SPOT)",
        "gpu_required": True,
        "requires_bands": ["R", "G", "B"],
        "supported_sources": [
            "landsat",
            "sentinel2",
            "hls",
            "naip",
            "spot",
            "local",
        ],
        "reference": "arXiv:2504.02534",
        "install_extra": "delineate-anything",
    },
    "ftw": {
        "name": "Fields of The World (FTW)",
        "approach": "Semantic segmentation (UNet/UPerNet/DeepLabV3+)",
        "strengths": "16+ pre-trained models, 25 countries, best for Sentinel-2",
        "gpu_required": True,
        "requires_bands": ["R", "G", "B"],
        "supported_sources": ["landsat", "sentinel2", "hls", "local"],
        "reference": "Fields of The World (FTW) dataset",
        "install_extra": "ftw",
    },
    "geoai": {
        "name": "GeoAI Field Delineator",
        "approach": "Mask R-CNN instance segmentation",
        "strengths": "Built-in NDVI, flexible multi-spectral input",
        "gpu_required": True,
        "requires_bands": ["R", "G", "B"],
        "supported_sources": ["sentinel2", "naip", "local"],
        "reference": "geoai-py package",
        "install_extra": "geoai",
    },
    "prithvi": {
        "name": "Prithvi-EO-2.0",
        "approach": "Foundation model (ViT) + segmentation or embedding clustering",
        "strengths": "Foundation model features, multi-temporal, best for HLS/Landsat",
        "gpu_required": True,
        "requires_bands": ["R", "G", "B", "NIR"],
        "supported_sources": ["landsat", "sentinel2", "hls", "local"],
        "reference": "NASA/IBM Prithvi-EO-2.0",
        "install_extra": "prithvi",
    },
    "embedding": {
        "name": "Embedding Clustering",
        "approach": "Unsupervised K-means/spectral clustering on pixel embeddings",
        "strengths": "No GPU needed, pre-computed data, change detection",
        "gpu_required": False,
        "requires_bands": [],
        "supported_sources": ["google-embedding", "tessera-embedding"],
        "reference": "TESSERA (CVPR 2026) + Google AlphaEarth",
        "install_extra": "geoai",
    },
    "ensemble": {
        "name": "Ensemble",
        "approach": "Multi-engine consensus via majority vote or intersection",
        "strengths": "Combines strengths of multiple engines",
        "gpu_required": True,
        "requires_bands": [],
        "supported_sources": [
            "landsat",
            "sentinel2",
            "hls",
            "naip",
            "spot",
            "local",
        ],
        "reference": "N/A",
        "install_extra": "all",
    },
}


def list_engines() -> dict[str, dict[str, Any]]:
    """List all available delineation engines and their metadata.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping engine names to their metadata.

    Examples
    --------
    >>> from agribound import list_engines
    >>> engines = list_engines()
    >>> for name, info in engines.items():
    ...     print(f"{name}: {info['approach']}")
    """
    return dict(ENGINE_REGISTRY)


class DelineationEngine(ABC):
    """Abstract base class for delineation engines.

    Subclasses must implement :meth:`delineate` to perform field boundary
    extraction from a raster file.
    """

    name: str = "base"
    supported_sources: list[str] = []
    requires_bands: list[str] = []

    @abstractmethod
    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run field boundary delineation on a raster file.

        Parameters
        ----------
        raster_path : str
            Path to the input GeoTIFF (composite or local file).
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        geopandas.GeoDataFrame
            Field boundary polygons with at minimum a ``geometry`` column.
        """

    def validate_input(self, raster_path: str, config: AgriboundConfig) -> None:
        """Validate that the input raster is compatible with this engine.

        Parameters
        ----------
        raster_path : str
            Path to the input raster.
        config : AgriboundConfig
            Pipeline configuration.

        Raises
        ------
        ValueError
            If the input is incompatible.
        """
        from agribound.io.raster import get_raster_info

        info = get_raster_info(raster_path)
        required_bands = len(self.requires_bands)
        if required_bands > 0 and info.count < required_bands:
            raise ValueError(
                f"Engine {self.name!r} requires at least {required_bands} bands "
                f"({self.requires_bands}), but the raster has {info.count} bands."
            )


def get_canonical_band_indices(
    source: str,
    canonical_names: list[str],
) -> list[int]:
    """Get 1-based raster band indices for canonical band names.

    Looks up each canonical name (e.g. ``"R"``, ``"G"``, ``"B"``, ``"NIR"``)
    in the source registry and returns the corresponding 1-based band indices
    in the downloaded composite.

    Parameters
    ----------
    source : str
        Satellite source name.
    canonical_names : list[str]
        Canonical band names to look up (e.g. ``["R", "G", "B"]``).

    Returns
    -------
    list[int]
        1-based band indices in the composite raster.

    Raises
    ------
    ValueError
        If the source or a canonical band name is not found.
    """
    from agribound.composites.base import SOURCE_REGISTRY

    info = SOURCE_REGISTRY.get(source)
    if info is None:
        raise ValueError(f"Unknown source {source!r}")

    all_bands = info.get("all_bands")
    canonical = info.get("canonical_bands") or {}

    if all_bands is None:
        # Local source — fall back to positional (1, 2, 3, ...)
        return list(range(1, len(canonical_names) + 1))

    indices = []
    for name in canonical_names:
        native = canonical.get(name)
        if native is None:
            raise ValueError(
                f"Canonical band {name!r} not defined for source {source!r}. "
                f"Available: {list(canonical.keys())}"
            )
        idx = all_bands.index(native) + 1  # 1-based
        indices.append(idx)
    return indices


def get_engine(engine_name: str) -> DelineationEngine:
    """Factory function to get a delineation engine by name.

    Parameters
    ----------
    engine_name : str
        Engine name (e.g. ``"delineate-anything"``, ``"ftw"``).

    Returns
    -------
    DelineationEngine
        Engine instance.

    Raises
    ------
    ValueError
        If the engine name is not recognized.
    """
    engine_name = engine_name.lower().strip()

    if engine_name not in ENGINE_REGISTRY:
        raise ValueError(
            f"Unknown engine {engine_name!r}. Available: {list(ENGINE_REGISTRY.keys())}"
        )

    if engine_name == "delineate-anything":
        from agribound.engines.delineate_anything import DelineateAnythingEngine

        return DelineateAnythingEngine()
    elif engine_name == "ftw":
        from agribound.engines.ftw import FTWEngine

        return FTWEngine()
    elif engine_name == "geoai":
        from agribound.engines.geoai_field import GeoAIEngine

        return GeoAIEngine()
    elif engine_name == "prithvi":
        from agribound.engines.prithvi import PrithviEngine

        return PrithviEngine()
    elif engine_name == "embedding":
        from agribound.engines.embedding import EmbeddingEngine

        return EmbeddingEngine()
    elif engine_name == "ensemble":
        from agribound.engines.ensemble import EnsembleEngine

        return EnsembleEngine()
    else:
        raise ValueError(f"Engine {engine_name!r} is not implemented.")
