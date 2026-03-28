"""
Local GeoTIFF and embedding composite handlers.

Handles the ``source="local"`` case (user-provided GeoTIFFs) and the
embedding sources (Google Satellite Embeddings, TESSERA) which download
pre-computed data rather than building composites from GEE.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agribound.composites.base import SOURCE_REGISTRY, CompositeBuilder
from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)


class LocalCompositeBuilder(CompositeBuilder):
    """Passthrough handler for user-provided local GeoTIFF files.

    Validates that the file exists, optionally clips it to the study area,
    and returns the path.
    """

    def build(self, config: AgriboundConfig) -> str:
        """Validate and optionally clip a local GeoTIFF.

        Parameters
        ----------
        config : AgriboundConfig
            Pipeline configuration with ``local_tif_path`` set.

        Returns
        -------
        str
            Path to the (optionally clipped) GeoTIFF.

        Raises
        ------
        FileNotFoundError
            If ``local_tif_path`` does not exist.
        """
        if config.local_tif_path is None:
            raise ValueError("local_tif_path must be set when source='local'")

        src_path = Path(config.local_tif_path)
        if not src_path.exists():
            raise FileNotFoundError(f"Local TIF not found: {src_path}")

        logger.info("Using local GeoTIFF: %s", src_path)

        # If study area is provided, clip the raster
        if config.study_area:
            from agribound.io.raster import clip_raster_to_geometry
            from agribound.io.vector import get_study_area_geometry, read_study_area

            try:
                study_gdf = read_study_area(config.study_area)
                geometry = get_study_area_geometry(study_gdf)

                cache_dir = config.get_working_dir()
                clipped_path = cache_dir / f"local_clipped_{src_path.stem}.tif"

                if not clipped_path.exists():
                    logger.info("Clipping local TIF to study area")
                    clip_raster_to_geometry(str(src_path), str(clipped_path), geometry)
                return str(clipped_path)
            except Exception as exc:
                logger.warning("Could not clip to study area, using full raster: %s", exc)

        return str(src_path)

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Return band mapping for local source.

        For local files, band mapping depends on user configuration.
        Default assumes R=1, G=2, B=3.

        Parameters
        ----------
        source : str
            Source name (``"local"``).

        Returns
        -------
        dict[str, str]
            Default band mapping.
        """
        return {"R": "1", "G": "2", "B": "3"}


class EmbeddingCompositeBuilder(CompositeBuilder):
    """Handler for pre-computed embedding downloads.

    Downloads Google Satellite Embeddings or TESSERA embeddings for the
    study area and year, saving them as local GeoTIFFs.
    """

    def build(self, config: AgriboundConfig) -> str:
        """Download embedding data for the study area.

        Parameters
        ----------
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        str
            Path to the downloaded embedding GeoTIFF.
        """
        from agribound.io.vector import get_study_area_bounds, read_study_area

        study_gdf = read_study_area(config.study_area)
        bbox = get_study_area_bounds(study_gdf)
        cache_dir = config.get_working_dir()

        if config.source == "google-embedding":
            return self._download_google_embedding(bbox, config.year, cache_dir, config.gee_project)
        elif config.source == "tessera-embedding":
            return self._download_tessera_embedding(bbox, config.year, cache_dir)
        else:
            raise ValueError(f"Unknown embedding source: {config.source}")

    def _download_google_embedding(
        self,
        bbox: tuple[float, float, float, float],
        year: int,
        output_dir: Path,
        gee_project: str | None = None,
    ) -> str:
        """Download Google Satellite Embeddings.

        Tries the ``geoai`` direct download first; falls back to exporting
        from the GEE ImageCollection ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL``
        if the direct download fails (e.g. HTTP 403).

        Parameters
        ----------
        bbox : tuple
            ``(min_lon, min_lat, max_lon, max_lat)``.
        year : int
            Target year (2018-2024).
        output_dir : Path
            Directory for downloaded files.
        gee_project : str or None
            GEE project ID (used for GEE fallback).

        Returns
        -------
        str
            Path to the embedding GeoTIFF.
        """
        out_path = output_dir / f"google_embedding_{year}.tif"
        if out_path.exists():
            logger.info("Using cached Google embedding: %s", out_path)
            return str(out_path)

        # Try geoai direct download first
        try:
            from geoai import download_google_satellite_embedding

            logger.info("Downloading Google Satellite Embeddings (year=%d, bbox=%s)", year, bbox)
            paths = download_google_satellite_embedding(
                bbox=bbox,
                output_dir=str(output_dir),
                years=year,
                dequantize=True,
            )
            if paths:
                return paths[0]
        except Exception as exc:
            logger.warning("geoai direct download failed: %s. Falling back to GEE export.", exc)

        # Fallback: export from GEE ImageCollection
        return self._export_google_embedding_gee(bbox, year, out_path, gee_project)

    def _export_google_embedding_gee(
        self,
        bbox: tuple[float, float, float, float],
        year: int,
        out_path: Path,
        gee_project: str | None = None,
    ) -> str:
        """Export Google Satellite Embeddings from GEE.

        The ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`` ImageCollection has
        64 embedding bands at 10 m resolution.  For large study areas the
        download is split into tiles and merged afterwards to stay within
        GEE download limits.
        """
        try:
            import ee
        except ImportError:
            raise ImportError(
                "earthengine-api is required for GEE-based embedding export. "
                "Install with: pip install agribound[gee]"
            ) from None

        from agribound.auth import check_gee_initialized, setup_gee

        if not check_gee_initialized():
            setup_gee(project=gee_project)

        logger.info("Exporting Google Satellite Embeddings from GEE (year=%d)", year)

        region = ee.Geometry.BBox(*bbox)
        ic = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        img = ic.filter(ee.Filter.calendarRange(year, year, "year")).mosaic().clip(region)

        # Export via geedim which handles tiling automatically
        try:
            import geedim as gd

            gd_img = gd.MaskedImage(img)
            gd_img.download(str(out_path), region=region, crs="EPSG:4326", scale=10)
        except ImportError:
            import urllib.request

            url = img.getDownloadURL(
                {
                    "region": region,
                    "scale": 10,
                    "crs": "EPSG:4326",
                    "format": "GEO_TIFF",
                }
            )
            urllib.request.urlretrieve(url, str(out_path))

        if not out_path.exists():
            raise RuntimeError(f"GEE export failed: no file at {out_path}")

        logger.info("Google embedding exported: %s", out_path)
        return str(out_path)

    def _download_tessera_embedding(
        self,
        bbox: tuple[float, float, float, float],
        year: int,
        output_dir: Path,
    ) -> str:
        """Download TESSERA embeddings.

        Parameters
        ----------
        bbox : tuple
            ``(min_lon, min_lat, max_lon, max_lat)``.
        year : int
            Target year (2017-2024).
        output_dir : Path
            Directory for downloaded files.

        Returns
        -------
        str
            Path to the embedding GeoTIFF.
        """
        try:
            from geoai import tessera_download
        except ImportError:
            raise ImportError(
                "geoai-py is required for TESSERA embedding downloads. "
                "Install with: pip install agribound[tessera]"
            ) from None

        logger.info("Downloading TESSERA embeddings (year=%d, bbox=%s)", year, bbox)
        paths = tessera_download(
            bbox=bbox,
            year=year,
            output_dir=str(output_dir),
            output_format="tiff",
        )
        if not paths:
            raise RuntimeError(f"No TESSERA data found for year={year}, bbox={bbox}")
        return paths[0]

    def get_band_mapping(self, source: str) -> dict[str, str]:
        """Return band mapping for embedding sources.

        Parameters
        ----------
        source : str
            ``"google-embedding"`` or ``"tessera-embedding"``.

        Returns
        -------
        dict[str, str]
            Band mapping (all embedding dimensions).
        """
        info = SOURCE_REGISTRY.get(source, {})
        return info.get("canonical_bands") or {}
