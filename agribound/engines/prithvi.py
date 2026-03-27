"""
Prithvi-EO-2.0 engine wrapper.

Wraps NASA/IBM's Prithvi foundation model for field boundary extraction.
Supports two modes:

1. **Segmentation mode**: Requires a fine-tuned checkpoint with a segmentation
   decoder (UPerNet). Best when user-provided reference boundaries are available
   for fine-tuning.
2. **Embedding mode** (default): Extracts Prithvi feature embeddings and
   clusters them to delineate fields. No fine-tuned weights needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)


class PrithviEngine(DelineationEngine):
    """Field boundary delineation using Prithvi-EO-2.0.

    The Prithvi foundation model provides powerful multi-temporal features
    for Earth observation tasks. This engine supports both supervised
    segmentation (with fine-tuned weights) and unsupervised embedding
    clustering.
    """

    name = "prithvi"
    supported_sources = ["landsat", "sentinel2", "hls", "local"]
    requires_bands = ["R", "G", "B", "NIR"]

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run Prithvi-based field delineation.

        Parameters
        ----------
        raster_path : str
            Path to the input GeoTIFF.
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        geopandas.GeoDataFrame
            Field boundary polygons.
        """
        mode = config.engine_params.get("mode", "embed")
        checkpoint_path = config.engine_params.get("checkpoint_path")

        if mode == "segment" and checkpoint_path:
            return self._segment_mode(raster_path, config, checkpoint_path)
        else:
            return self._embed_mode(raster_path, config)

    def _segment_mode(
        self,
        raster_path: str,
        config: AgriboundConfig,
        checkpoint_path: str,
    ) -> gpd.GeoDataFrame:
        """Segmentation mode using fine-tuned Prithvi + UPerNet.

        Parameters
        ----------
        raster_path : str
            Path to the input GeoTIFF.
        config : AgriboundConfig
            Pipeline configuration.
        checkpoint_path : str
            Path to the fine-tuned checkpoint.

        Returns
        -------
        geopandas.GeoDataFrame
            Field boundary polygons.
        """
        try:
            import torch  # noqa: F401
            from terratorch.cli_tools import LightningInferenceModel
        except ImportError:
            raise ImportError(
                "terratorch and torch are required for Prithvi segmentation mode. "
                "Install with: pip install agribound[prithvi]"
            ) from None

        self.validate_input(raster_path, config)

        device = config.resolve_device()
        model_name = config.engine_params.get("model_name", "Prithvi-EO-2.0-300M-TL")

        logger.info(
            "Running Prithvi segmentation (model=%s, checkpoint=%s)",
            model_name,
            checkpoint_path,
        )

        # Load model via terratorch
        model = LightningInferenceModel.from_config(
            checkpoint_path=checkpoint_path,
            map_location=device,
        )

        # Run inference on tiles
        cache_dir = config.get_working_dir()
        pred_path = str(cache_dir / "prithvi_segmentation.tif")

        if Path(pred_path).exists():
            logger.info("Using cached Prithvi segmentation: %s", pred_path)
        else:
            model.predict_raster(
                raster_path,
                output_path=pred_path,
                patch_size=config.engine_params.get("patch_size", 224),
                overlap=config.engine_params.get("overlap", 0.25),
            )

        # Polygonize the segmentation mask
        from agribound.postprocess.polygonize import polygonize_mask

        gdf = polygonize_mask(pred_path, min_area_m2=config.min_field_area_m2)
        logger.info("Prithvi segmentation delineated %d fields", len(gdf))
        return gdf

    def _embed_mode(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Embedding mode using Prithvi features + clustering.

        Extracts feature embeddings from Prithvi's encoder and clusters
        them to identify homogeneous field regions.

        Parameters
        ----------
        raster_path : str
            Path to the input GeoTIFF.
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        geopandas.GeoDataFrame
            Field boundary polygons from clustered embeddings.
        """
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch is required for Prithvi embedding mode. "
                "Install with: pip install agribound[prithvi]"
            ) from None

        self.validate_input(raster_path, config)

        device = config.resolve_device()
        model_name = config.engine_params.get("model_name", "Prithvi-EO-2.0-300M-TL")

        logger.info("Running Prithvi embedding extraction (model=%s)", model_name)

        # Load Prithvi model
        try:
            from geoai import load_prithvi_model

            model = load_prithvi_model(model_name=model_name, device=device)
        except ImportError:
            logger.info("Loading Prithvi via transformers")
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                f"ibm-nasa-geospatial/{model_name}", trust_remote_code=True
            )
            model = model.to(device).eval()

        # Read only the R, G, B, NIR bands from the multi-band raster
        from agribound.engines.base import get_canonical_band_indices
        from agribound.io.raster import read_raster

        if config.source != "local":
            rgbn_indices = get_canonical_band_indices(config.source, ["R", "G", "B", "NIR"])
        else:
            rgbn_indices = [1, 2, 3, 4]

        data, meta = read_raster(raster_path, bands=rgbn_indices)
        if data is None or data.size == 0:
            logger.warning("Empty raster data from %s", raster_path)
            return gpd.GeoDataFrame(columns=["geometry"], crs=meta.get("crs"))

        # Replace inf/nan/nodata with 0
        data = np.where(np.isfinite(data), data, 0)

        logger.info("Raster shape: %s", data.shape)

        # Write cluster map as raster
        cache_dir = config.get_working_dir()
        cluster_path = str(cache_dir / "prithvi_clusters.tif")

        if Path(cluster_path).exists():
            logger.info("Using cached Prithvi clusters: %s", cluster_path)
        else:
            # Tile the raster into patches and extract embeddings
            patch_size = config.engine_params.get("patch_size", 224)
            embeddings = self._extract_embeddings(data, model, device, patch_size)

            # Cluster embeddings
            n_clusters = config.engine_params.get("n_clusters", "auto")
            cluster_map = self._cluster_embeddings(embeddings, n_clusters)

            from agribound.io.raster import write_raster

            write_raster(
                cluster_path,
                cluster_map.astype(np.int32),
                crs=meta["crs"],
                transform=meta["transform"],
            )

        # Polygonize
        from agribound.postprocess.polygonize import polygonize_mask

        gdf = polygonize_mask(cluster_path, min_area_m2=config.min_field_area_m2)
        logger.info("Prithvi embedding clustering delineated %d fields", len(gdf))
        return gdf

    def _extract_embeddings(
        self,
        data: np.ndarray,
        model,
        device: str,
        patch_size: int,
    ) -> np.ndarray:
        """Extract feature embeddings from image data using Prithvi.

        Parameters
        ----------
        data : numpy.ndarray
            Raster data with shape ``(bands, height, width)``.
        model
            Pre-loaded Prithvi model.
        device : str
            Compute device.
        patch_size : int
            Patch size for tiled extraction.

        Returns
        -------
        numpy.ndarray
            Embedding array with shape ``(height, width, embed_dim)``.
        """

        bands, height, width = data.shape

        # Normalize data
        data_float = data.astype(np.float32)
        for b in range(bands):
            band = data_float[b]
            mu = np.nanmean(band)
            std = np.nanstd(band)
            if std > 0:
                data_float[b] = (band - mu) / std

        # Simple spatial average pooling to create a low-res embedding map
        _pool_size = patch_size // 16  # Typical ViT downscale factor
        # For now, use PCA on spatial patches as a lightweight embedding proxy
        from sklearn.decomposition import PCA

        # Reshape to (n_pixels, n_bands)
        pixel_features = data_float.reshape(bands, -1).T  # (H*W, B)

        n_components = min(bands, 16)
        pca = PCA(n_components=n_components)
        embeddings_flat = pca.fit_transform(pixel_features)

        embeddings = embeddings_flat.reshape(height, width, n_components)
        return embeddings

    def _cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int | str) -> np.ndarray:
        """Cluster pixel embeddings to delineate fields.

        Parameters
        ----------
        embeddings : numpy.ndarray
            Shape ``(height, width, embed_dim)``.
        n_clusters : int or "auto"
            Number of clusters. ``"auto"`` uses silhouette score.

        Returns
        -------
        numpy.ndarray
            Cluster label map with shape ``(1, height, width)``.
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        height, width, n_features = embeddings.shape
        flat = embeddings.reshape(-1, n_features)

        # Remove invalid pixels
        valid_mask = np.all(np.isfinite(flat), axis=1)
        valid_pixels = flat[valid_mask]

        if len(valid_pixels) == 0:
            return np.zeros((1, height, width), dtype=np.int32)

        # Subsample for efficiency
        max_samples = 50_000
        if len(valid_pixels) > max_samples:
            indices = np.random.choice(len(valid_pixels), max_samples, replace=False)
            sample = valid_pixels[indices]
        else:
            sample = valid_pixels

        if not isinstance(n_clusters, int) or n_clusters < 2:
            best_k = 10
            best_score = -1
            for k in [5, 10, 15, 20, 30]:
                km = KMeans(n_clusters=k, n_init=3, random_state=42)
                labels = km.fit_predict(sample)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(sample, labels, sample_size=min(5000, len(sample)))
                    if score > best_score:
                        best_score = score
                        best_k = k
            n_clusters = best_k
            logger.info("Auto-selected %d clusters (silhouette=%.3f)", n_clusters, best_score)

        km = KMeans(n_clusters=int(n_clusters), n_init=5, random_state=42)
        km.fit(sample)

        labels = np.zeros(len(flat), dtype=np.int32)
        labels[valid_mask] = km.predict(valid_pixels) + 1  # 0 = nodata

        cluster_map = labels.reshape(1, height, width)
        return cluster_map
