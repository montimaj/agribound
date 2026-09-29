"""
Abstract base class for composite builders and the builder factory.

Each satellite source has a builder that handles data acquisition, cloud
masking, compositing and download to a local GeoTIFF. Source metadata lives in
:mod:`agribound.registry`; ``SOURCE_REGISTRY``, ``ENGINE_REGISTRY``,
``list_sources`` and ``list_engines`` are re-exported here for backwards
compatibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agribound.config import AgriboundConfig
from agribound.registry import (
    EMBEDDING_SOURCES,
    ENGINE_REGISTRY,
    SOURCE_REGISTRY,
    list_engines,
    list_sources,
)

__all__ = [
    "ENGINE_REGISTRY",
    "SOURCE_REGISTRY",
    "CompositeBuilder",
    "NoDataError",
    "get_composite_builder",
    "list_engines",
    "list_sources",
]


class NoDataError(ValueError):
    """No input imagery or embeddings for this study area and year.

    Raised by the composite builders when the source has no data for the
    study area in the requested period: no image intersects the study-area
    extent (the message lists the years that have images there), the
    downloaded composite or embedding has no valid pixel inside the study
    area, TESSERA or the Source Cooperative mirror returns no tile, USGS NAIP
    Plus has no imagery there, or a local raster does not overlap the study
    area. Invalid configurations and service errors raise other exceptions.
    A subclass of :class:`ValueError`, so ``except ValueError`` handlers
    keep catching it; :func:`agribound.hpc.tiles.no_data_reason` uses it to
    record such tiles as "no-data" instead of failed.
    """


class CompositeBuilder(ABC):
    """Abstract base class for satellite composite builders.

    Subclasses must implement :meth:`build` and :meth:`get_band_mapping`.
    """

    @abstractmethod
    def build(self, config: AgriboundConfig) -> str:
        """Build a composite (or fetch embeddings) and write it as a local GeoTIFF.

        Parameters
        ----------
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        str
            Path to the composite GeoTIFF.
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
            Mapping of canonical names (R, G, B, NIR, ...) to source band names.
        """

    def get_resolution(self, source: str) -> float | None:
        """Return the default export resolution in metres for a source.

        Parameters
        ----------
        source : str
            Satellite source name.

        Returns
        -------
        float or None
            Resolution in metres, or *None* when it depends on the input.
        """
        return SOURCE_REGISTRY.get(source, {}).get("resolution_m")


_BUILDERS: dict[str, str] = {
    "local": "agribound.composites.local:LocalCompositeBuilder",
    "usgs-naip-plus": "agribound.composites.usgs:USGSNAIPPlusCompositeBuilder",
    **{src: "agribound.composites.local:EmbeddingCompositeBuilder" for src in EMBEDDING_SOURCES},
}
_DEFAULT_BUILDER = "agribound.composites.gee:GEECompositeBuilder"


def get_composite_builder(source: str) -> CompositeBuilder:
    """Factory function to get the composite builder for a source.

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
        If the source is not recognised.
    """
    import importlib

    if source not in SOURCE_REGISTRY:
        raise ValueError(f"Unknown source {source!r}. Available: {list(SOURCE_REGISTRY.keys())}")

    target = _BUILDERS.get(source, _DEFAULT_BUILDER)
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)()
