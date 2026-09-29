"""
Abstract base class for delineation engines and the engine factory.

Each engine wraps a different model or approach for extracting field
boundary polygons from satellite imagery or embeddings. Engine metadata lives
in :mod:`agribound.registry`; ``ENGINE_REGISTRY``, ``SOURCE_REGISTRY``,
``list_engines`` and ``list_sources`` are re-exported here for backwards
compatibility.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

import geopandas as gpd

from agribound.config import AgriboundConfig
from agribound.registry import (
    CANONICAL_BAND_NAMES,
    ENGINE_CLASSES,
    ENGINE_REGISTRY,
    SOURCE_REGISTRY,
    list_engines,
    list_sources,
)

__all__ = [
    "ENGINE_CLASSES",
    "ENGINE_REGISTRY",
    "SOURCE_REGISTRY",
    "DelineationEngine",
    "get_canonical_band_indices",
    "get_engine",
    "get_engine_class",
    "list_engines",
    "list_sources",
]


class DelineationEngine(ABC):
    """Abstract base class for delineation engines.

    Subclasses must implement :meth:`delineate`. They may attach
    JSON-serialisable run metadata (backend, model id, weights repository,
    revision, sha256, thresholds, window dates, ...) to the returned frame as
    ``gdf.attrs["engine_meta"]``; the pipeline copies it into the provenance
    record.
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
            ``gdf.attrs["engine_meta"]`` may hold engine metadata.
        """

    def validate_input(self, raster_path: str, config: AgriboundConfig) -> None:
        """Validate that the input raster is compatible with this engine.

        Checks that the raster has enough bands for the engine's
        ``requires_bands`` (class attribute, falling back to the registry
        entry): at least the highest 1-based index those canonical bands map
        to for the configured source, with ``config.bands`` taking precedence
        (:func:`get_canonical_band_indices`). For local rasters without
        ``config.bands`` the indices are positional, so this is
        ``len(requires_bands)``.

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

        required = list(self.requires_bands) or list(
            ENGINE_REGISTRY.get(self.name, {}).get("requires_bands", [])
        )
        info = get_raster_info(raster_path)
        n_needed = 0
        if required:
            try:
                n_needed = max(
                    get_canonical_band_indices(config.source, required, bands=config.bands)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Engine {self.name!r} requires bands {required}, which source "
                    f"{config.source!r} does not provide: {exc}"
                ) from exc
        if n_needed > 0 and info.count < n_needed:
            override = f" with bands={config.bands}" if config.bands else ""
            raise ValueError(
                f"Engine {self.name!r} requires bands {required}{override}, i.e. at least "
                f"{n_needed} bands, but the raster has {info.count} bands."
            )

    @classmethod
    def prefetch(cls, config: AgriboundConfig) -> list[str]:
        """Download model weights so that inference can run offline.

        Engines that load remote weights override this and return the local
        paths (or cache directories) they populated. The base implementation
        downloads nothing.

        Parameters
        ----------
        config : AgriboundConfig
            Pipeline configuration (engine parameters select the model).

        Returns
        -------
        list[str]
            Local paths of the downloaded artefacts (empty here).
        """
        return []


def get_canonical_band_indices(
    source: str,
    canonical_names: list[str],
    bands: dict[str, int] | None = None,
) -> list[int]:
    """Get 1-based raster band indices for canonical band names.

    Looks up each canonical name (``"R"``, ``"G"``, ``"B"``, ``"NIR"``,
    ``"NIR_NARROW"``, ``"SWIR1"``, ``"SWIR2"``) in the source registry and
    returns the corresponding 1-based band index in the composite written by
    the source's builder.

    Parameters
    ----------
    source : str
        Satellite source name.
    canonical_names : list[str]
        Canonical band names to look up (e.g. ``["R", "G", "B"]``).
    bands : dict[str, int] or None
        Optional explicit mapping of canonical names to 1-based indices (for
        example ``AgriboundConfig.bands``). When given it takes precedence
        over the registry for every name it contains.

    Returns
    -------
    list[int]
        1-based band indices in the composite raster.

    Raises
    ------
    ValueError
        If the source is unknown or a canonical band is not available.

    Notes
    -----
    For ``source="local"`` without an explicit mapping the indices are
    positional (``1, 2, 3, ...`` in the order requested), i.e. the local file
    is assumed to store the requested bands first and in that order.
    """
    info = SOURCE_REGISTRY.get(source)
    if info is None:
        raise ValueError(f"Unknown source {source!r}")

    all_bands = info.get("all_bands")
    canonical = info.get("canonical_bands") or {}
    bands = dict(bands or {})

    if all_bands is None and not bands:
        # Local source -- positional (1, 2, 3, ...)
        return list(range(1, len(canonical_names) + 1))

    indices = []
    for position, name in enumerate(canonical_names):
        if name in bands:
            idx = int(bands[name])
            if idx < 1:
                raise ValueError(f"Band index for {name!r} must be >= 1, got {idx}")
            indices.append(idx)
            continue
        if all_bands is None:
            # Local source with a partial mapping: remaining names positional.
            indices.append(position + 1)
            continue
        native = canonical.get(name)
        if native is None:
            known = [n for n in CANONICAL_BAND_NAMES if n in canonical]
            raise ValueError(
                f"Canonical band {name!r} not defined for source {source!r}. Available: {known}"
            )
        indices.append(all_bands.index(native) + 1)  # 1-based
    return indices


def get_engine_class(engine_name: str) -> type[DelineationEngine]:
    """Import and return the engine class for *engine_name* without instantiating it.

    Parameters
    ----------
    engine_name : str
        Engine name (e.g. ``"delineate-anything"``).

    Returns
    -------
    type[DelineationEngine]
        Engine class.

    Raises
    ------
    ValueError
        If the engine name is not recognised.
    """
    key = str(engine_name).lower().strip()
    target = ENGINE_CLASSES.get(key)
    if target is None or key not in ENGINE_REGISTRY:
        raise ValueError(f"Unknown engine {engine_name!r}. Available: {list(ENGINE_REGISTRY)}")
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_engine(engine_name: str) -> DelineationEngine:
    """Factory function to get a delineation engine instance by name.

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
        If the engine name is not recognised.
    """
    return get_engine_class(engine_name)()
