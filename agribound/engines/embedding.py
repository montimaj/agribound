"""
Embedding-based clustering engine.

Clusters pre-computed per-pixel embeddings (Google Satellite Embedding V1,
64-D, or TESSERA, 128-D) with scikit-learn and polygonizes the cluster map.
No labels, model weights or GPU are needed. Clusters are land-cover
segments, not field instances: every connected region of every cluster
becomes a polygon, and non-cropland segments are only removed by the
downstream area and LULC filters.

Clustering is memory-bounded: the raster is read in row blocks of at most
``engine_params["max_block_mb"]`` MiB, a seeded uniform random sample of
valid pixels is drawn in one pass, the dimensionality reduction and the
clusterer are fitted on that sample, and a second pass predicts the label of
every valid pixel block by block into an int32 label raster. Polygonization
(:func:`agribound.postprocess.polygonize.polygonize_mask`) then reads that
label raster at once (4 bytes per pixel).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine
from agribound.registry import ENGINE_REGISTRY

logger = logging.getLogger(__name__)

#: Default ``engine_params`` of the embedding engine.
DEFAULT_PARAMS: dict[str, Any] = {
    "use_pca": True,
    "pca_components": 16,
    "matryoshka_depth": None,
    "n_clusters": "auto",
    "clustering_method": "kmeans",
    "k_candidates": [5, 10, 15, 20, 30, 50],
    "pca_sample_size": 100_000,
    "cluster_sample_size": 50_000,
    "silhouette_sample_size": 5_000,
    "max_block_mb": 256,
}

#: Matryoshka prefix lengths of TESSERA v2 embeddings: 16, 32 and 64 are the
#: nested dimensions described for the v2 models (ucam-eo/tessera README);
#: 4 and 16 are also published as separate depth arrays in the v2 Zarr stores
#: (``geoemb:depths``), which geotessera documents as prefixes of the full
#: 128-D embedding.
MATRYOSHKA_DEPTHS: tuple[int, ...] = (4, 16, 32, 64)

#: Complete k-means restarts (``KMeans(n_init=...)``) of the final fit and of each
#: silhouette fit; the lowest-inertia run is kept.
_KMEANS_N_INIT = 10
#: Pixels transformed and predicted per call in the prediction pass.
_PREDICT_CHUNK = 262_144
#: Bump when the clustering result changes for the same inputs, since the key does not
#: name the clusterer (v3: finite nodata is invalid; v4: KMeans(n_init=10) replaces
#: MiniBatchKMeans in the final fit and the k selection).
_CACHE_VERSION = "embedding-clusters-v4"


class EmbeddingEngine(DelineationEngine):
    """Field delineation by unsupervised clustering of pixel embeddings.

    Engine parameters (``config.engine_params``)
    --------------------------------------------
    use_pca : bool
        Reduce the embeddings with PCA before clustering (default *True*;
        only when the raster has more bands than *pca_components*).
    pca_components : int
        PCA dimensions (default 16).
    matryoshka_depth : int or None
        TESSERA v2 only (``source="tessera-embedding"`` with
        ``tessera_version="v2"``): cluster the first *depth* dimensions,
        a Matryoshka prefix, instead of PCA. One of :data:`MATRYOSHKA_DEPTHS`.
        Any other source or version raises ``ValueError``.
    n_clusters : int or "auto"
        Number of clusters, or ``"auto"`` (default): the candidate in
        *k_candidates* with the highest silhouette score of a
        ``KMeans(n_init=10)`` fit on the first *silhouette_sample_size*
        sampled pixels.
    clustering_method : str
        ``"kmeans"`` (default) or ``"spectral"``. ``"kmeans"`` fits
        ``KMeans(n_init=10)`` on the cluster sample whatever the raster size
        and keeps the lowest-inertia of the ten complete restarts (0.1.x and
        1.0.0 used ``MiniBatchKMeans(batch_size=10000, n_init=3)`` above
        100 000 valid pixels, which ends in a higher-inertia solution for
        many samples, and ``KMeans(n_init=5)`` otherwise). ``"spectral"`` is
        ``SpectralClustering(affinity="nearest_neighbors")`` on the sample,
        extended to all pixels with ``NearestCentroid`` (slow).
    k_candidates : list[int]
        Candidates for ``n_clusters="auto"`` (default 5, 10, 15, 20, 30, 50).
    pca_sample_size, cluster_sample_size, silhouette_sample_size : int
        Sizes of the nested random samples used to fit PCA, the clusterer and
        the silhouette selection (defaults 100 000, 50 000, 5 000).
    max_block_mb : float
        Maximum size of one row block read from the raster (default 256 MiB).
        Peak memory is a few times this (a block, the copy of its valid
        pixels and per-band masks) plus the fitting sample; GDAL's block
        cache is limited to ``max(64, max_block_mb)`` MiB during the reads
        unless ``GDAL_CACHEMAX`` is set in the environment.

    A pixel is valid when all bands used (every band, or the Matryoshka
    prefix) are finite, not all zero and, when the raster declares a finite
    nodata value, not all equal to it. All
    random choices (pixel sample, PCA solver, k-means initialisation) are
    seeded from ``config.seed``. scikit-learn computes the k-means sums in
    parallel, so a different number of OpenMP threads can move a small
    fraction of pixels to another cluster. The cluster raster (int32;
    0 = invalid, ``1..k`` = cluster) is cached with :func:`agribound._cache.cache_path`,
    keyed by the study area, source, year, TESSERA version, every parameter
    above except *max_block_mb* (the result does not depend on the block
    size), the seed and the input raster's path, size and modification time.

    When ``config.sam_refine`` is *True* (which also absorbs the legacy
    ``engine_params["sam_refine"]``), the polygons are refined with
    :func:`agribound.engines.samgeo_engine.refine_boundaries` on this raster.
    An embedding raster has no RGB bands, so this requires
    ``engine_params["sam_rgb_bands"]`` (three 1-based dimensions used as a
    pseudo-RGB image); without it the engine raises before clustering.
    """

    name = "embedding"
    supported_sources = list(ENGINE_REGISTRY["embedding"]["supported_sources"])
    requires_bands = list(ENGINE_REGISTRY["embedding"]["requires_bands"])

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_params(config: AgriboundConfig) -> dict[str, Any]:
        """Return the validated engine parameters (defaults filled in).

        Raises
        ------
        ValueError
            For invalid values.
        """
        user = dict(config.engine_params or {})
        params = {k: user.get(k, v) for k, v in DEFAULT_PARAMS.items()}
        method = str(params["clustering_method"]).lower().strip()
        if method not in ("kmeans", "spectral"):
            raise ValueError(f"clustering_method must be 'kmeans' or 'spectral', got {method!r}")
        params["clustering_method"] = method
        n_clusters = params["n_clusters"]
        if isinstance(n_clusters, str):
            if n_clusters.lower().strip() != "auto":
                raise ValueError(
                    f"n_clusters must be an integer >= 2 or 'auto', got {n_clusters!r}"
                )
            params["n_clusters"] = "auto"
        elif isinstance(n_clusters, bool) or int(n_clusters) != n_clusters or n_clusters < 2:
            raise ValueError(f"n_clusters must be an integer >= 2 or 'auto', got {n_clusters!r}")
        else:
            params["n_clusters"] = int(n_clusters)
        candidates = [int(k) for k in params["k_candidates"]]
        if not candidates or min(candidates) < 2:
            raise ValueError(f"k_candidates must be integers >= 2, got {params['k_candidates']!r}")
        params["k_candidates"] = sorted(set(candidates))
        for key in ("pca_components", "pca_sample_size", "cluster_sample_size"):
            params[key] = int(params[key])
            if params[key] < 1:
                raise ValueError(f"{key} must be >= 1, got {params[key]}")
        params["silhouette_sample_size"] = int(params["silhouette_sample_size"])
        if params["silhouette_sample_size"] < 3:
            raise ValueError("silhouette_sample_size must be >= 3")
        params["use_pca"] = bool(params["use_pca"])
        params["max_block_mb"] = float(params["max_block_mb"])
        if params["max_block_mb"] <= 0:
            raise ValueError("max_block_mb must be > 0")

        depth = params["matryoshka_depth"]
        if depth is not None:
            depth = int(depth)
            if config.source != "tessera-embedding" or config.tessera_version != "v2":
                raise ValueError(
                    "matryoshka_depth needs TESSERA v2 embeddings (source='tessera-embedding', "
                    f"tessera_version='v2'); got source={config.source!r}, "
                    f"tessera_version={config.tessera_version!r}. Remove it to use PCA."
                )
            if depth not in MATRYOSHKA_DEPTHS:
                raise ValueError(
                    f"matryoshka_depth must be one of {MATRYOSHKA_DEPTHS}, got {depth}"
                )
            params["matryoshka_depth"] = depth
        return params

    # ------------------------------------------------------------------
    # Engine API
    # ------------------------------------------------------------------

    @staticmethod
    def cluster_cache_path(
        raster_path: str, config: AgriboundConfig, params: dict[str, Any] | None = None
    ) -> Path:
        """Return the cache path of the cluster raster for *raster_path* and *config*.

        The key covers :func:`agribound._cache.cache_key`'s fields plus the
        seed, the raster's resolved path, size and modification time, and all
        engine parameters except ``max_block_mb``.
        """
        from agribound._cache import cache_path

        params = params if params is not None else EmbeddingEngine.resolve_params(config)
        # max_block_mb only changes how the raster is read, not the result.
        parts = [_CACHE_VERSION, f"seed={config.seed}", _file_signature(raster_path)] + [
            f"{k}={params[k]}" for k in sorted(params) if k != "max_block_mb"
        ]
        return cache_path(config, "embedding_clusters", ".tif", *parts)

    @classmethod
    def prefetch(cls, config: AgriboundConfig) -> list[str]:
        """Download the SAM weights when ``config.sam_refine`` is set (nothing else is remote)."""
        if not config.sam_refine:
            return []
        from agribound.engines.samgeo_engine import prefetch as sam_prefetch

        return sam_prefetch(config)

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Cluster the embedding raster and polygonize the clusters.

        Parameters
        ----------
        raster_path : str
            Embedding GeoTIFF (float32, one band per embedding dimension).
        config : AgriboundConfig
            Pipeline configuration.

        Returns
        -------
        geopandas.GeoDataFrame
            Polygons with ``class_value`` (cluster label ``1..k``), in the
            raster CRS. ``attrs["engine_meta"]`` describes the clustering;
            ``attrs["sam_stats"]`` is set when SAM refinement ran.

        Raises
        ------
        ValueError
            For an unsupported source, invalid parameters, a raster with too
            few valid pixels, or SAM refinement without ``sam_rgb_bands``.
        """
        from agribound.io.raster import get_raster_info

        if config.source not in self.supported_sources:
            raise ValueError(
                f"The embedding engine needs an embedding source {self.supported_sources}, "
                f"got {config.source!r}"
            )
        params = self.resolve_params(config)
        info = get_raster_info(raster_path)
        depth = params["matryoshka_depth"]
        if depth is not None and depth > info.count:
            raise ValueError(f"matryoshka_depth={depth} but the raster has only {info.count} bands")
        if config.sam_refine:
            from agribound.engines.samgeo_engine import _rgb_band_indices

            _rgb_band_indices(config, info.count)  # fail before clustering

        cluster_path = self.cluster_cache_path(raster_path, config, params)
        meta_path = cluster_path.with_suffix(".json")

        if cluster_path.exists() and meta_path.exists():
            logger.info("Using cached embedding clusters: %s", cluster_path)
            meta = json.loads(meta_path.read_text())
            meta["cache_hit"] = True
        else:
            logger.info(
                "Clustering embeddings: %d bands, %dx%d pixels", info.count, info.width, info.height
            )
            meta = self._cluster_raster(raster_path, cluster_path, params, config)
            _atomic_write_text(meta_path, json.dumps(meta, indent=2, sort_keys=True))
            meta["cache_hit"] = False
        meta["cluster_raster"] = str(cluster_path)

        from agribound.postprocess.polygonize import polygonize_mask

        gdf = polygonize_mask(str(cluster_path), min_area_m2=config.min_field_area_m2)
        logger.info("Embedding clustering delineated %d polygons", len(gdf))

        meta["sam_refine"] = bool(config.sam_refine)
        if config.sam_refine and len(gdf) > 0:
            from agribound.engines.samgeo_engine import refine_boundaries

            gdf = refine_boundaries(gdf, raster_path, config)
            meta["sam_stats"] = gdf.attrs.get("sam_stats")
        gdf.attrs["engine_meta"] = meta
        return gdf

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_raster(
        self,
        raster_path: str,
        out_path: Path,
        params: dict[str, Any],
        config: AgriboundConfig,
    ) -> dict[str, Any]:
        """Fit on a seeded sample, predict block by block, write the int32 label raster."""
        import rasterio
        import sklearn
        from rasterio.windows import Window

        from agribound._repro import get_rng

        seed = int(config.seed)
        depth = params["matryoshka_depth"]
        # Bound GDAL's block cache too (default: 5 % of RAM) unless the user set it.
        env = {} if "GDAL_CACHEMAX" in os.environ else {"GDAL_CACHEMAX": _gdal_cache_mb(params)}
        with rasterio.Env(**env), rasterio.open(raster_path) as src:
            n_bands = src.count
            bands = list(range(1, (depth or n_bands) + 1))
            height, width = src.height, src.width
            nodata = src.nodata
            bytes_per_row = max(1, width * len(bands) * 4)
            block_rows = int(max(1, min(height, params["max_block_mb"] * 2**20 // bytes_per_row)))

            def blocks():
                for row in range(0, height, block_rows):
                    h = min(block_rows, height - row)
                    data = src.read(bands, window=Window(0, row, width, h)).astype(
                        np.float32, copy=False
                    )
                    flat = data.reshape(len(bands), -1).T
                    yield row, h, flat, _valid_mask(data, nodata)

            # Pass 1: uniform random sample without replacement (bottom-k random keys).
            n_keep = max(params["pca_sample_size"], params["cluster_sample_size"])
            rng = get_rng(config, "embedding-sample")
            keys = np.empty(0, dtype=np.float64)
            sample = np.empty((0, len(bands)), dtype=np.float32)
            n_valid = 0
            for _, _, flat, valid in blocks():
                vals = flat[valid]
                n_valid += len(vals)
                block_keys = rng.random(len(vals))
                if len(block_keys) > n_keep:
                    sel = np.argpartition(block_keys, n_keep - 1)[:n_keep]
                    block_keys, vals = block_keys[sel], vals[sel]
                keys = np.concatenate([keys, block_keys])
                sample = np.concatenate([sample, vals])
                if len(keys) > n_keep:
                    sel = np.argpartition(keys, n_keep - 1)[:n_keep]
                    keys, sample = keys[sel], sample[sel]
            sample = sample[np.argsort(keys, kind="stable")]
            if n_valid < 3:
                raise ValueError(
                    f"{raster_path} has {n_valid} valid embedding pixels; at least 3 are needed"
                )

            # Dimensionality reduction.
            reducer = None
            meta: dict[str, Any] = {
                "backend": "scikit-learn",
                "sklearn_version": sklearn.__version__,
                "source": config.source,
                "tessera_version": config.tessera_version
                if config.source == "tessera-embedding"
                else None,
                "n_bands": int(n_bands),
                "n_pixels": int(height * width),
                "n_valid_pixels": int(n_valid),
                "seed": seed,
                "block_rows": block_rows,
                "params": _jsonable(params),
            }
            if depth is not None:
                meta["reduction"] = "matryoshka"
                meta["n_features"] = depth
            elif params["use_pca"] and n_bands > params["pca_components"]:
                from sklearn.decomposition import PCA

                pca_sample = sample[: params["pca_sample_size"]]
                reducer = PCA(n_components=params["pca_components"], random_state=seed)
                reducer.fit(pca_sample)
                meta["reduction"] = "pca"
                meta["n_features"] = params["pca_components"]
                meta["pca_sample_size"] = int(len(pca_sample))
                meta["pca_explained_variance_ratio"] = float(
                    np.sum(reducer.explained_variance_ratio_)
                )
            else:
                meta["reduction"] = "none"
                meta["n_features"] = int(n_bands)

            fit_x = sample[: params["cluster_sample_size"]]
            if reducer is not None:
                fit_x = reducer.transform(fit_x)
            meta["cluster_sample_size"] = int(len(fit_x))

            k = params["n_clusters"]
            if k == "auto":
                k, scores, n_eval = _select_k(
                    fit_x[: params["silhouette_sample_size"]], params["k_candidates"], seed
                )
                meta["auto_k"] = {
                    "candidates": params["k_candidates"],
                    "silhouette": scores,
                    "sample_size": n_eval,
                }
            if k > len(fit_x):
                raise ValueError(f"n_clusters={k} exceeds the {len(fit_x)} sampled pixels")
            meta["n_clusters"] = int(k)

            predict = self._fit_clusterer(fit_x, k, params["clustering_method"], seed, meta)
            logger.info(
                "Clustering %d valid pixels into %d clusters (%s, reduction=%s)",
                n_valid,
                k,
                meta["clusterer"],
                meta["reduction"],
            )

            # Pass 2: predict per block and write the label raster.
            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": 1,
                "dtype": "int32",
                "crs": src.crs,
                "transform": src.transform,
                "nodata": 0,
                "compress": "lzw",
            }
            tmp_path = out_path.with_name(out_path.stem + ".partial.tif")
            with rasterio.open(tmp_path, "w", **profile) as dst:
                for row, h, flat, valid in blocks():
                    labels = np.zeros(len(flat), dtype=np.int32)
                    idx = np.flatnonzero(valid)
                    for start in range(0, len(idx), _PREDICT_CHUNK):
                        sel = idx[start : start + _PREDICT_CHUNK]
                        x = flat[sel]
                        if reducer is not None:
                            x = reducer.transform(x)
                        labels[sel] = predict(x).astype(np.int32) + 1
                    dst.write(labels.reshape(1, h, width), window=Window(0, row, width, h))
            os.replace(tmp_path, out_path)
        return meta

    @staticmethod
    def _fit_clusterer(x: np.ndarray, k: int, method: str, seed: int, meta: dict[str, Any]):
        """Fit the clusterer on *x* and return a ``predict(array) -> labels`` callable.

        ``"kmeans"`` is ``KMeans(n_init=10)`` on *x* for every raster size; *x*
        has at most *cluster_sample_size* rows, so the ten complete restarts
        stay cheap, and unlike ``MiniBatchKMeans`` they reach the
        lowest-inertia solution for most samples.
        """
        if method == "kmeans":
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=k, n_init=_KMEANS_N_INIT, random_state=seed)
            model.fit(x)
            meta["clusterer"] = "KMeans"
            meta["kmeans_n_init"] = _KMEANS_N_INIT
            meta["inertia"] = float(model.inertia_)
            return model.predict

        from sklearn.cluster import SpectralClustering
        from sklearn.neighbors import NearestCentroid

        if len(x) > 10_000:
            logger.warning("Spectral clustering with %d samples is slow; consider kmeans", len(x))
        sc = SpectralClustering(n_clusters=k, random_state=seed, affinity="nearest_neighbors")
        sc.fit(x)
        nc = NearestCentroid()
        nc.fit(x, sc.labels_)
        meta["clusterer"] = "SpectralClustering+NearestCentroid"
        return nc.predict

    @staticmethod
    def _auto_select_k(
        sample: np.ndarray, k_range: list[int] | None = None, random_state: int = 42
    ) -> int:
        """Return the silhouette-best k (see :func:`_select_k`)."""
        k_range = k_range if k_range is not None else list(DEFAULT_PARAMS["k_candidates"])
        return _select_k(sample, k_range, random_state)[0]


def _select_k(
    sample: np.ndarray, candidates: list[int], seed: int
) -> tuple[int, dict[str, float], int]:
    """Pick k by silhouette score of ``KMeans(n_init=10)`` fits.

    The fits use the same algorithm as the final ``"kmeans"`` fit (1.0.0 used
    ``MiniBatchKMeans(n_init=3, batch_size=5000)``). Candidates with
    ``k >= len(sample)`` are skipped, as are fits that return a single label.
    Returns ``(best_k, {str(k): score}, len(sample))``.

    Raises
    ------
    ValueError
        If no candidate can be evaluated.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    scores: dict[str, float] = {}
    best_k, best_score = None, -np.inf
    for k in candidates:
        if k >= len(sample):
            continue
        labels = KMeans(n_clusters=k, n_init=_KMEANS_N_INIT, random_state=seed).fit_predict(sample)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(sample, labels))
        scores[str(k)] = score
        if score > best_score:
            best_k, best_score = k, score
    if best_k is None:
        raise ValueError(
            f"Could not choose n_clusters automatically: none of {candidates} can be evaluated "
            f"on {len(sample)} sampled pixels. Set engine_params['n_clusters'] explicitly."
        )
    logger.info("Auto-selected k=%d (silhouette=%.3f)", best_k, best_score)
    return int(best_k), scores, int(len(sample))


def _valid_mask(data: np.ndarray, nodata: float | None = None) -> np.ndarray:
    """Valid pixels of a ``(bands, h, w)`` block, flattened, computed band by band.

    A pixel is valid when all its bands are finite, not all zero and, for a
    finite *nodata*, not all equal to *nodata*.
    """
    finite = np.ones(data.shape[1] * data.shape[2], dtype=bool)
    nonzero = np.zeros_like(finite)
    check_nodata = nodata is not None and np.isfinite(nodata)
    # Compare in the block's dtype: a float32 pixel equal to float32(1e-3) differs
    # from the float64 value 1e-3.
    nd = data.dtype.type(nodata) if check_nodata else None
    not_nodata = np.zeros_like(finite)
    for band in data:
        values = band.ravel()
        finite &= np.isfinite(values)
        nonzero |= values != 0
        if check_nodata:
            not_nodata |= values != nd
    valid = finite & nonzero
    return valid & not_nodata if check_nodata else valid


def _gdal_cache_mb(params: dict[str, Any]) -> int:
    return int(max(64, params["max_block_mb"]))


def _file_signature(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    st = os.stat(resolved)
    return f"raster={resolved}:{st.st_size}:{st.st_mtime_ns}"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text)
    os.replace(tmp, path)


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out
