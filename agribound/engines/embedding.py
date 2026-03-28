"""
Embedding-based clustering engine.

Uses pre-computed pixel embeddings (Google Satellite Embeddings or TESSERA)
to delineate field boundaries via unsupervised clustering. No GPU required.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)


class EmbeddingEngine(DelineationEngine):
    """Field boundary delineation via embedding clustering.

    Downloads pre-computed pixel embeddings and clusters them to identify
    homogeneous field regions. Supports Google Satellite Embeddings (64-D)
    and TESSERA embeddings (128-D).

    This engine requires no GPU — clustering runs on CPU.
    """

    name = "embedding"
    supported_sources = ["google-embedding", "tessera-embedding"]
    requires_bands = []

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run embedding-based field delineation.

        The ``raster_path`` points to a pre-downloaded embedding GeoTIFF
        (produced by the ``EmbeddingCompositeBuilder``).

        Parameters
        ----------
        raster_path : str
            Path to the embedding GeoTIFF (64 or 128 bands).
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        geopandas.GeoDataFrame
            Field boundary polygons.
        """
        from agribound.io.raster import get_raster_info, read_raster

        info = get_raster_info(raster_path)
        logger.info(
            "Loading embeddings: %d bands, %dx%d pixels",
            info.count,
            info.width,
            info.height,
        )

        data, meta = read_raster(raster_path)
        bands, height, width = data.shape

        # Reshape to (H*W, D) for clustering
        embeddings = data.reshape(bands, -1).T.astype(np.float32)

        # Handle nodata
        valid_mask = np.all(np.isfinite(embeddings), axis=1) & np.any(embeddings != 0, axis=1)

        # Optional: PCA dimensionality reduction for speed
        use_pca = config.engine_params.get("use_pca", True)
        n_components = config.engine_params.get("pca_components", 16)

        if use_pca and bands > n_components:
            from sklearn.decomposition import PCA

            logger.info("Reducing embeddings from %d to %d dimensions via PCA", bands, n_components)
            pca = PCA(n_components=n_components)
            valid_embeddings = embeddings[valid_mask]
            # Fit on subsample for efficiency
            max_fit = 100_000
            if len(valid_embeddings) > max_fit:
                fit_sample = valid_embeddings[
                    np.random.choice(len(valid_embeddings), max_fit, replace=False)
                ]
            else:
                fit_sample = valid_embeddings
            pca.fit(fit_sample)
            embeddings_reduced = np.zeros((len(embeddings), n_components), dtype=np.float32)
            embeddings_reduced[valid_mask] = pca.transform(valid_embeddings)
        else:
            embeddings_reduced = embeddings

        # Write cluster raster
        cache_dir = config.get_working_dir()
        source_tag = config.source.replace("-", "_")
        cluster_path = str(cache_dir / f"embedding_clusters_{source_tag}.tif")

        if Path(cluster_path).exists():
            logger.info("Using cached embedding clusters: %s", cluster_path)
        else:
            # Clustering
            n_clusters = config.engine_params.get("n_clusters", "auto")
            clustering_method = config.engine_params.get("clustering_method", "kmeans")

            cluster_labels = self._cluster(
                embeddings_reduced, valid_mask, n_clusters, clustering_method
            )

            # Reshape to raster
            cluster_map = cluster_labels.reshape(1, height, width)

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
        logger.info("Embedding clustering delineated %d field boundaries", len(gdf))

        # Optional SAM2 refinement using first 3 bands as pseudo-RGB
        if config.engine_params.get("sam_refine", False) and len(gdf) > 0:
            try:
                from agribound.engines.samgeo_engine import refine_boundaries

                logger.info("Refining %d boundaries with SAM2 (bands 1-3 as RGB)", len(gdf))
                gdf = refine_boundaries(gdf, raster_path, config)
                logger.info("SAM2 refined to %d boundaries", len(gdf))
            except Exception as exc:
                logger.warning("SAM2 refinement failed, using unrefined boundaries: %s", exc)

        return gdf

    def _cluster(
        self,
        embeddings: np.ndarray,
        valid_mask: np.ndarray,
        n_clusters: int | str,
        method: str,
    ) -> np.ndarray:
        """Cluster pixel embeddings.

        Parameters
        ----------
        embeddings : numpy.ndarray
            Shape ``(n_pixels, n_features)``.
        valid_mask : numpy.ndarray
            Boolean mask for valid pixels.
        n_clusters : int or "auto"
            Number of clusters.
        method : str
            ``"kmeans"`` or ``"spectral"``.

        Returns
        -------
        numpy.ndarray
            Cluster labels with shape ``(n_pixels,)``.
        """
        from sklearn.cluster import KMeans, MiniBatchKMeans

        valid_data = embeddings[valid_mask]

        if len(valid_data) == 0:
            return np.zeros(len(embeddings), dtype=np.int32)

        # Subsample for auto-k selection
        max_samples = 50_000
        if len(valid_data) > max_samples:
            sample_idx = np.random.choice(len(valid_data), max_samples, replace=False)
            sample = valid_data[sample_idx]
        else:
            sample = valid_data

        if n_clusters == "auto":
            n_clusters = self._auto_select_k(sample)

        logger.info(
            "Clustering %d valid pixels into %d clusters (%s)",
            len(valid_data),
            n_clusters,
            method,
        )

        if method == "kmeans":
            # Use MiniBatchKMeans for large datasets
            if len(valid_data) > 100_000:
                clusterer = MiniBatchKMeans(
                    n_clusters=n_clusters, batch_size=10_000, n_init=3, random_state=42
                )
            else:
                clusterer = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)

            clusterer.fit(sample)
            all_labels = np.zeros(len(embeddings), dtype=np.int32)
            all_labels[valid_mask] = clusterer.predict(valid_data) + 1
        else:
            # Spectral clustering (only feasible for smaller datasets)
            from sklearn.cluster import SpectralClustering

            if len(sample) > 10_000:
                logger.warning(
                    "Spectral clustering with %d samples is slow, consider kmeans",
                    len(sample),
                )
            sc = SpectralClustering(
                n_clusters=n_clusters, random_state=42, affinity="nearest_neighbors"
            )
            sc.fit(sample)
            # Predict for all valid pixels using nearest centroid
            from sklearn.neighbors import NearestCentroid

            nc = NearestCentroid()
            nc.fit(sample, sc.labels_)
            all_labels = np.zeros(len(embeddings), dtype=np.int32)
            all_labels[valid_mask] = nc.predict(valid_data) + 1

        return all_labels

    @staticmethod
    def _auto_select_k(sample: np.ndarray, k_range: list[int] | None = None) -> int:
        """Auto-select number of clusters using silhouette score.

        Parameters
        ----------
        sample : numpy.ndarray
            Subsampled embedding data.
        k_range : list[int] or None
            Range of k values to test.

        Returns
        -------
        int
            Best number of clusters.
        """
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.metrics import silhouette_score

        if k_range is None:
            k_range = [5, 10, 15, 20, 30, 50]

        best_k = 15
        best_score = -1

        eval_sample = sample[: min(5000, len(sample))]

        for k in k_range:
            if k >= len(eval_sample):
                continue
            km = MiniBatchKMeans(n_clusters=k, n_init=3, random_state=42, batch_size=5000)
            labels = km.fit_predict(eval_sample)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(eval_sample, labels)
                if score > best_score:
                    best_score = score
                    best_k = k

        logger.info("Auto-selected k=%d (silhouette=%.3f)", best_k, best_score)
        return best_k
