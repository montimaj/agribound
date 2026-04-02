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
        # All spectral SR bands common across L5/7/8/9 (L8/9 naming convention).
        # L5/7 bands are renamed to match L8/9 during collection building.
        "all_bands": ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        "canonical_bands": {"R": "SR_B4", "G": "SR_B3", "B": "SR_B2", "NIR": "SR_B5"},
        "coverage": "Global, 1985–present (L5/7/8/9)",
        "requires_gee": True,
    },
    "sentinel2": {
        "name": "Sentinel-2 L2A",
        "collection": "COPERNICUS/S2_SR_HARMONIZED",
        "resolution_m": 10,
        # All spectral bands (no QA/SCL).
        "all_bands": [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B11",
            "B12",
        ],
        "canonical_bands": {"R": "B4", "G": "B3", "B": "B2", "NIR": "B8"},
        "coverage": "Global, 2017–present",
        "requires_gee": True,
    },
    "hls": {
        "name": "Harmonized Landsat-Sentinel",
        "collection": "NASA/HLS/HLSL30/v002 + NASA/HLS/HLSS30/v002",
        "resolution_m": 30,
        # 7 harmonized spectral bands common to HLSL30 and HLSS30.
        "all_bands": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
        "canonical_bands": {"R": "B4", "G": "B3", "B": "B2", "NIR": "B5"},
        "coverage": "Global, 2013–present",
        "requires_gee": True,
    },
    "naip": {
        "name": "NAIP",
        "collection": "USDA/NAIP/DOQQ",
        "resolution_m": 1,
        "all_bands": ["R", "G", "B", "N"],
        "canonical_bands": {"R": "R", "G": "G", "B": "B", "NIR": "N"},
        "coverage": "Continental US, ~2-3 year cycle",
        "requires_gee": True,
    },
        "usgs-naip-plus": {
        "name": "USGS NAIP Plus ImageServer",
        "collection": "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPPlus/ImageServer",
        "resolution_m": None,
        "all_bands": ["R", "G", "B", "N"],
        "canonical_bands": {"R": "R", "G": "G", "B": "B", "NIR": "N"},
        "coverage": "USGS The National Map USGSNAIPPlus ImageServer",
        "requires_gee": False,
    },
    "spot": {
        "name": "SPOT 6/7",
        "collection": "AIRBUS/SPOT6_7",
        "resolution_m": 6,
        "all_bands": ["R", "G", "B"],
        "canonical_bands": {"R": "R", "G": "G", "B": "B"},
        "coverage": "Global (restricted access), 2012–2023",
        "requires_gee": True,
        "restricted": True,
    },
    "spot-pan": {
        "name": "SPOT 6/7 Panchromatic",
        "collection": "AIRBUS/SPOT6_7",
        "resolution_m": 1.5,
        "all_bands": ["P"],
        "canonical_bands": {"R": "P", "G": "P", "B": "P"},
        "coverage": "Global (restricted access), 2012–2023",
        "requires_gee": True,
        "restricted": True,
    },
    "local": {
        "name": "Local GeoTIFF",
        "collection": None,
        "resolution_m": None,
        "all_bands": None,
        "canonical_bands": None,
        "coverage": "User-provided",
        "requires_gee": False,
    },
    "google-embedding": {
        "name": "Google Satellite Embedding V1",
        "collection": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        "resolution_m": 10,
        "all_bands": [f"A{i:02d}" for i in range(64)],
        "canonical_bands": None,
        "coverage": "Global, 2017–2025",
        "requires_gee": False,
    },
    "tessera-embedding": {
        "name": "TESSERA Embeddings",
        "collection": None,
        "resolution_m": 10,
        "all_bands": [f"T{i:03d}" for i in range(128)],
        "canonical_bands": None,
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

    elif source == "usgs-naip-plus":
        from agribound.composites.usgs import USGSNAIPPlusCompositeBuilder

        return USGSNAIPPlusCompositeBuilder()

    else:
        from agribound.composites.gee import GEECompositeBuilder

        return GEECompositeBuilder()
