"""
Abstract base class for composite builders and source registry.

Each satellite source has a corresponding builder that handles data
acquisition, cloud masking, compositing, and download to local GeoTIFF.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agribound.config import AgriboundConfig

# ---------------------------------------------------------------------------
# Source metadata registry
# ---------------------------------------------------------------------------

SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "landsat": {
        "name": "Landsat 8/9",
        "collection": "LANDSAT/LC08/C02/T1_L2 + LANDSAT/LC09/C02/T1_L2",
        "resolution_m": 30,
        "bands": {"R": "SR_B4", "G": "SR_B3", "B": "SR_B2", "NIR": "SR_B5"},
        "coverage": "Global, 1985–present (L5/7/8/9)",
        "requires_gee": True,
    },
    "sentinel2": {
        "name": "Sentinel-2 L2A",
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "resolution_m": 10,
        "bands": {"R": "B4", "G": "B3", "B": "B2", "NIR": "B8"},
        "coverage": "Global, 2017–present",
        "requires_gee": True,
    },
    "hls": {
        "name": "Harmonized Landsat-Sentinel",
        "collection": "NASA/HLS/HLSL30/v002 + NASA/HLS/HLSS30/v002",
        "resolution_m": 30,
        "bands": {"R": "B4", "G": "B3", "B": "B2", "NIR": "B5"},
        "coverage": "Global, 2013–present",
        "requires_gee": True,
    },
    "naip": {
        "name": "NAIP",
        "collection": "USDA/NAIP/DOQQ",
        "resolution_m": 1,
        "bands": {"R": "R", "G": "G", "B": "B", "NIR": "N"},
        "coverage": "Continental US, ~2-3 year cycle",
        "requires_gee": True,
    },
    "spot": {
        "name": "SPOT 6/7",
        "collection": "AIRBUS/SPOT6_7",
        "resolution_m": 6,
        "bands": {"R": "R", "G": "G", "B": "B", "NIR": "NIR"},
        "coverage": "Global (restricted access), 2012–2023",
        "requires_gee": True,
        "restricted": True,
    },
    "local": {
        "name": "Local GeoTIFF",
        "collection": None,
        "resolution_m": None,
        "bands": None,
        "coverage": "User-provided",
        "requires_gee": False,
    },
    "google-embedding": {
        "name": "Google Satellite Embedding V1",
        "collection": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        "resolution_m": 10,
        "bands": {f"A{i:02d}": f"A{i:02d}" for i in range(64)},
        "coverage": "Global, 2017–2025",
        "requires_gee": False,
    },
    "tessera-embedding": {
        "name": "TESSERA Embeddings",
        "collection": None,
        "resolution_m": 10,
        "bands": {f"T{i:03d}": f"T{i:03d}" for i in range(128)},
        "coverage": "Global, 2017–2024",
        "requires_gee": False,
    },
}


def list_sources() -> dict[str, dict[str, Any]]:
    """List all available satellite sources and their metadata.

    Returns
    -------
    dict[str, dict]
        Dictionary mapping source names to their metadata.

    Examples
    --------
    >>> from agribound import list_sources
    >>> sources = list_sources()
    >>> for name, info in sources.items():
    ...     print(f"{name}: {info['resolution_m']}m")
    """
    return dict(SOURCE_REGISTRY)


class CompositeBuilder(ABC):
    """Abstract base class for satellite composite builders.

    Subclasses must implement :meth:`build` and :meth:`get_band_mapping`.
    """

    @abstractmethod
    def build(self, config: AgriboundConfig) -> str:
        """Build an annual composite and download it as a local GeoTIFF.

        Parameters
        ----------
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        str
            Path to the downloaded composite GeoTIFF.
        """

    @abstractmethod
    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Return the band name mapping for a satellite source.

        Parameters
        ----------
        source : str
            Satellite source name.

        Returns
        -------
        dict[str, str]
            Mapping of canonical names (R, G, B, NIR) to source band names.
        """

    def get_resolution(self, source: str) -> float | None:
        """Return the native resolution in meters for a source.

        Parameters
        ----------
        source : str
            Satellite source name.

        Returns
        -------
        float or None
            Resolution in meters, or *None* for local sources.
        """
        return SOURCE_REGISTRY.get(source, {}).get("resolution_m")


def get_composite_builder(source: str) -> CompositeBuilder:
    """Factory function to get the appropriate composite builder.

    Parameters
    ----------
    source : str
        Satellite source name.

    Returns
    -------
    CompositeBuilder
        Builder instance for the given source.

    Raises
    ------
    ValueError
        If the source is not recognized.
    """
    if source not in SOURCE_REGISTRY:
        raise ValueError(
            f"Unknown source {source!r}. Available: {list(SOURCE_REGISTRY.keys())}"
        )

    if source == "local":
        from agribound.composites.local import LocalCompositeBuilder

        return LocalCompositeBuilder()
    elif source in ("google-embedding", "tessera-embedding"):
        from agribound.composites.local import EmbeddingCompositeBuilder

        return EmbeddingCompositeBuilder()
    else:
        from agribound.composites.gee import GEECompositeBuilder

        return GEECompositeBuilder()
