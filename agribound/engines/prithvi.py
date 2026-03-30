"""
Prithvi-EO-2.0 engine wrapper.

Wraps NASA/IBM's Prithvi foundation model for field boundary extraction.
Supports three modes:

1. **Segmentation mode**: Requires a fine-tuned checkpoint with a segmentation
   decoder (UPerNet). Best when user-provided reference boundaries are available
   for fine-tuning.
2. **Embedding mode** (default ``"embed"``): Extracts Prithvi feature embeddings
   via the ViT encoder and clusters them to delineate fields. No fine-tuned
   weights needed. Requires ``transformers`` (``pip install agribound[prithvi]``).
3. **PCA mode** (``"pca"``): Lightweight fallback that clusters PCA-reduced
   spectral bands without running the ViT encoder. No GPU or ``transformers``
   needed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine

logger = logging.getLogger(__name__)

# Prithvi-EO-2.0 expects 6 HLS bands in this order:
# Blue(B02), Green(B03), Red(B04), NIR(B05), SWIR1(B06), SWIR2(B07)
PRITHVI_BANDS = ["B", "G", "R", "NIR", "SWIR1", "SWIR2"]

# Per-band normalization from Prithvi-EO-2.0 pre-training (HLS reflectance)
PRITHVI_MEAN = np.array([1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0], dtype=np.float32)
PRITHVI_STD = np.array([2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0], dtype=np.float32)

# Canonical band name → SWIR mapping for sources that have SWIR bands.
# Used by _get_prithvi_band_indices to select the 6 bands the model needs.
_SWIR_CANONICAL = {
    "hls": {"SWIR1": "B6", "SWIR2": "B7"},
    "landsat": {"SWIR1": "SR_B6", "SWIR2": "SR_B7"},
    "sentinel2": {"SWIR1": "B11", "SWIR2": "B12"},
}


class PrithviEngine(DelineationEngine):
    """Field boundary delineation using Prithvi-EO-2.0.

    The Prithvi foundation model provides powerful multi-temporal features
    for Earth observation tasks. This engine supports supervised
    segmentation (with fine-tuned weights), ViT embedding clustering,
    and a lightweight PCA fallback.
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
            Pipeline configuration.  ``engine_params``:

            - ``mode`` : ``"embed"`` (default) | ``"pca"`` | ``"segment"``
            - ``checkpoint_path`` : str — required for ``"segment"`` mode
            - ``patch_size`` : int — ViT input size (default 224)
            - ``n_clusters`` : int or ``"auto"``
            - ``batch_size`` : int — patches per GPU batch (default 8)

        Returns
        -------
        geopandas.GeoDataFrame
            Field boundary polygons.
        """
        mode = config.engine_params.get("mode", "embed")
        checkpoint_path = config.engine_params.get("checkpoint_path")

        if mode == "segment" and checkpoint_path:
            return self._segment_mode(raster_path, config, checkpoint_path)
        elif mode == "pca":
            return self._pca_mode(raster_path, config)
        else:
            return self._embed_mode(raster_path, config)

    # ── Segmentation mode ───────────────────────────────────────────────

    def _segment_mode(
        self,
        raster_path: str,
        config: AgriboundConfig,
        checkpoint_path: str,
    ) -> gpd.GeoDataFrame:
        """Segmentation mode using fine-tuned Prithvi + UPerNet."""
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

        model = LightningInferenceModel.from_config(
            checkpoint_path=checkpoint_path,
            map_location=device,
        )

        cache_dir = config.get_working_dir()
        source_tag = config.source.replace("-", "_")
        pred_path = str(cache_dir / f"prithvi_segmentation_{source_tag}.tif")

        if Path(pred_path).exists():
            logger.info("Using cached Prithvi segmentation: %s", pred_path)
        else:
            model.predict_raster(
                raster_path,
                output_path=pred_path,
                patch_size=config.engine_params.get("patch_size", 224),
                overlap=config.engine_params.get("overlap", 0.25),
            )

        from agribound.postprocess.polygonize import polygonize_mask

        gdf = polygonize_mask(pred_path, min_area_m2=config.min_field_area_m2)
        logger.info("Prithvi segmentation delineated %d fields", len(gdf))
        return gdf

    # ── ViT embedding mode ──────────────────────────────────────────────

    def _embed_mode(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Embedding mode using Prithvi ViT encoder + clustering.

        Tiles the raster into 224×224 patches, runs each through the
        Prithvi encoder, and reassembles the per-token embeddings into
        a spatial feature map that is then K-means clustered.
        """
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for Prithvi embedding mode. "
                "Install with: pip install agribound[prithvi]"
            ) from None

        self.validate_input(raster_path, config)

        device = config.resolve_device()
        model_name = config.engine_params.get("model_name", "Prithvi-EO-2.0-300M-TL")

        cache_dir = config.get_working_dir()
        source_tag = config.source.replace("-", "_")
        cluster_path = str(cache_dir / f"prithvi_clusters_{source_tag}.tif")

        if Path(cluster_path).exists():
            logger.info("Using cached Prithvi clusters: %s", cluster_path)
        else:
            # Read 6 Prithvi bands
            data, meta = self._read_prithvi_bands(raster_path, config)
            if data is None or data.size == 0:
                logger.warning("Empty raster data from %s", raster_path)
                return gpd.GeoDataFrame(columns=["geometry"], crs=meta.get("crs"))

            logger.info("Raster shape: %s (6-band Prithvi input)", data.shape)

            # Load Prithvi model
            logger.info("Loading Prithvi model: %s", model_name)
            model = self._load_prithvi_model(model_name, device)

            # Extract ViT embeddings via tiled inference
            patch_size = config.engine_params.get("patch_size", 224)
            batch_size = config.engine_params.get("batch_size", 8)
            embeddings = self._extract_vit_embeddings(
                data, model, device, patch_size, batch_size, torch
            )
            del model
            if hasattr(torch, "mps") and device == "mps":
                torch.mps.empty_cache()
            elif device.startswith("cuda"):
                torch.cuda.empty_cache()

            # Cluster
            n_clusters = config.engine_params.get("n_clusters", "auto")
            cluster_map = self._cluster_embeddings(embeddings, n_clusters)

            from agribound.io.raster import write_raster

            write_raster(
                cluster_path,
                cluster_map.astype(np.int32),
                crs=meta["crs"],
                transform=meta["transform"],
            )

        from agribound.postprocess.polygonize import polygonize_mask

        gdf = polygonize_mask(cluster_path, min_area_m2=config.min_field_area_m2)
        logger.info("Prithvi ViT embedding clustering delineated %d fields", len(gdf))
        return gdf

    # ── PCA fallback mode ───────────────────────────────────────────────

    def _pca_mode(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Lightweight PCA mode — clusters spectral bands without ViT.

        Useful as a baseline comparison or when ``transformers`` is not
        installed.
        """
        self.validate_input(raster_path, config)

        cache_dir = config.get_working_dir()
        source_tag = config.source.replace("-", "_")
        cluster_path = str(cache_dir / f"prithvi_pca_clusters_{source_tag}.tif")

        if Path(cluster_path).exists():
            logger.info("Using cached PCA clusters: %s", cluster_path)
        else:
            from agribound.engines.base import get_canonical_band_indices
            from agribound.io.raster import read_raster

            if config.source != "local":
                band_indices = get_canonical_band_indices(
                    config.source, ["R", "G", "B", "NIR"]
                )
            else:
                band_indices = [1, 2, 3, 4]

            data, meta = read_raster(raster_path, bands=band_indices)
            if data is None or data.size == 0:
                logger.warning("Empty raster data from %s", raster_path)
                return gpd.GeoDataFrame(columns=["geometry"], crs=meta.get("crs"))

            data = np.where(np.isfinite(data), data, 0)
            logger.info("PCA mode — raster shape: %s", data.shape)

            embeddings = self._pca_embeddings(data)

            n_clusters = config.engine_params.get("n_clusters", "auto")
            cluster_map = self._cluster_embeddings(embeddings, n_clusters)

            from agribound.io.raster import write_raster

            write_raster(
                cluster_path,
                cluster_map.astype(np.int32),
                crs=meta["crs"],
                transform=meta["transform"],
            )

        from agribound.postprocess.polygonize import polygonize_mask

        gdf = polygonize_mask(cluster_path, min_area_m2=config.min_field_area_m2)
        logger.info("PCA clustering delineated %d fields", len(gdf))
        return gdf

    # ── Helpers ──────────────────────────────────────────────────────────

    def _read_prithvi_bands(
        self, raster_path: str, config: AgriboundConfig
    ) -> tuple[np.ndarray | None, dict]:
        """Read the 6 bands Prithvi expects (Blue, Green, Red, NIR, SWIR1, SWIR2)."""
        from agribound.composites.base import SOURCE_REGISTRY
        from agribound.io.raster import read_raster

        source = config.source
        if source == "local":
            # Assume first 6 bands are B, G, R, NIR, SWIR1, SWIR2
            data, meta = read_raster(raster_path, bands=[1, 2, 3, 4, 5, 6])
        else:
            info = SOURCE_REGISTRY.get(source, {})
            all_bands = info.get("all_bands", [])
            canonical = info.get("canonical_bands", {})
            swir_map = _SWIR_CANONICAL.get(source, {})

            # Build the 6-band index list
            indices = []
            for canon_name in PRITHVI_BANDS:
                if canon_name in ("SWIR1", "SWIR2"):
                    band_name = swir_map.get(canon_name)
                else:
                    band_name = canonical.get(canon_name)
                if band_name is None or band_name not in all_bands:
                    raise ValueError(
                        f"Source '{source}' does not have band '{canon_name}' "
                        f"required by Prithvi. Supported sources: hls, landsat, sentinel2."
                    )
                indices.append(all_bands.index(band_name) + 1)  # 1-based

            data, meta = read_raster(raster_path, bands=indices)

        if data is not None:
            data = np.where(np.isfinite(data), data, 0).astype(np.float32)
        return data, meta

    @staticmethod
    def _load_prithvi_model(model_name: str, device: str):
        """Load Prithvi encoder from HuggingFace Hub.

        Downloads ``prithvi_mae.py`` and the pre-trained weights directly,
        bypassing ``transformers.AutoModel`` which is incompatible with
        the Prithvi config on some ``transformers`` versions.
        """
        import importlib.util
        import json

        import torch

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for Prithvi embedding mode. "
                "Install with: pip install huggingface-hub"
            ) from None

        repo_id = f"ibm-nasa-geospatial/{model_name}"
        logger.info("Downloading Prithvi model from %s", repo_id)

        config_path = hf_hub_download(repo_id, "config.json")
        mae_path = hf_hub_download(repo_id, "prithvi_mae.py")

        # Find the .pt weight file (name varies across model variants)
        from huggingface_hub import list_repo_files

        pt_files = [f for f in list_repo_files(repo_id) if f.endswith(".pt")]
        if not pt_files:
            raise FileNotFoundError(f"No .pt weight file found in {repo_id}")
        weights_path = hf_hub_download(repo_id, pt_files[0])

        # Load config
        with open(config_path) as f:
            cfg = json.load(f)["pretrained_cfg"]
        cfg["num_frames"] = 1  # single-date composite

        # Import PrithviMAE class from the downloaded module
        spec = importlib.util.spec_from_file_location("prithvi_mae", mae_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        model = mod.PrithviMAE(**cfg)

        # Load pre-trained weights (drop pos_embed — recomputed for variable input)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        for k in list(state_dict.keys()):
            if "pos_embed" in k:
                del state_dict[k]
        model.load_state_dict(state_dict, strict=False)

        model = model.to(device).eval()
        if device == "mps":
            torch.set_default_dtype(torch.float32)
            model = model.float()

        logger.info("Prithvi model loaded (embed_dim=%d)", cfg["embed_dim"])
        return model

    def _extract_vit_embeddings(
        self,
        data: np.ndarray,
        model,
        device: str,
        patch_size: int,
        batch_size: int,
        torch_module,
    ) -> np.ndarray:
        """Tile raster into patches, run through Prithvi ViT, reassemble.

        Parameters
        ----------
        data : numpy.ndarray
            Shape ``(6, H, W)`` — the 6 Prithvi bands.
        model
            Loaded Prithvi model with ``forward_features()``.
        device : str
            Compute device.
        patch_size : int
            Spatial tile size (default 224).
        batch_size : int
            Number of patches per forward pass.
        torch_module
            The ``torch`` module (avoids re-importing).

        Returns
        -------
        numpy.ndarray
            Shape ``(H, W, embed_dim)`` spatial embedding map.
        """
        torch = torch_module
        bands, height, width = data.shape

        # Normalize with Prithvi pre-training stats
        for b in range(bands):
            data[b] = (data[b] - PRITHVI_MEAN[b]) / PRITHVI_STD[b]

        # Determine ViT token grid for one patch
        token_h = patch_size // 16  # 14 for 224
        token_w = patch_size // 16

        # Figure out embed_dim and CLS token presence from a dummy forward pass
        dummy = torch.zeros(1, bands, 1, patch_size, patch_size, device=device)
        with torch.no_grad():
            dummy_out = model.forward_features(dummy)
        last_dummy = dummy_out[-1]  # (1, num_tokens, embed_dim)
        embed_dim = last_dummy.shape[-1]
        n_spatial_tokens = token_h * token_w
        # CLS token is present if output has more tokens than the spatial grid
        has_cls = last_dummy.shape[1] > n_spatial_tokens
        del dummy, dummy_out, last_dummy

        logger.info(
            "Prithvi: embed_dim=%d, token_grid=%dx%d per %dx%d patch",
            embed_dim, token_h, token_w, patch_size, patch_size,
        )

        # Pad raster to be divisible by patch_size
        pad_h = (patch_size - height % patch_size) % patch_size
        pad_w = (patch_size - width % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            data = np.pad(data, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        padded_h, padded_w = data.shape[1], data.shape[2]

        n_rows = padded_h // patch_size
        n_cols = padded_w // patch_size
        total_patches = n_rows * n_cols
        logger.info("Tiling: %d×%d = %d patches", n_rows, n_cols, total_patches)

        # Allocate output embedding map at token resolution
        embed_map = np.zeros(
            (n_rows * token_h, n_cols * token_w, embed_dim), dtype=np.float32
        )

        # Collect patch coordinates
        patches_coords = []
        for r in range(n_rows):
            for c in range(n_cols):
                y0 = r * patch_size
                x0 = c * patch_size
                patches_coords.append((r, c, y0, x0))

        # Process in batches
        for b_start in range(0, total_patches, batch_size):
            b_end = min(b_start + batch_size, total_patches)
            batch_coords = patches_coords[b_start:b_end]

            # Build batch tensor: (B, C, T=1, H, W)
            batch_data = np.stack(
                [data[:, y:y + patch_size, x:x + patch_size] for _, _, y, x in batch_coords]
            )
            # (B, C, H, W) → (B, C, 1, H, W)
            batch_tensor = torch.from_numpy(batch_data[:, :, np.newaxis, :, :]).to(device)

            with torch.no_grad():
                features = model.forward_features(batch_tensor)
                # Last encoder layer: (B, [cls] + T*token_h*token_w, embed_dim)
                last_features = features[-1].cpu().numpy()

            # Scatter tokens back into spatial map
            for i, (r, c, _, _) in enumerate(batch_coords):
                feat = last_features[i]  # ([cls] + n_spatial, embed_dim)
                if has_cls:
                    feat = feat[1:]  # drop CLS token
                tokens = feat.reshape(token_h, token_w, embed_dim)
                tr = r * token_h
                tc = c * token_w
                embed_map[tr:tr + token_h, tc:tc + token_w, :] = tokens

            if (b_start // batch_size) % 10 == 0 and b_start > 0:
                logger.info("  Processed %d/%d patches", b_end, total_patches)

        # Upsample token-resolution map to pixel resolution
        from scipy.ndimage import zoom

        scale_h = padded_h / embed_map.shape[0]
        scale_w = padded_w / embed_map.shape[1]
        embed_full = zoom(embed_map, (scale_h, scale_w, 1), order=1)

        # Crop to original size
        embed_full = embed_full[:height, :width, :]
        logger.info("Embedding map shape: %s", embed_full.shape)
        return embed_full

    @staticmethod
    def _pca_embeddings(data: np.ndarray) -> np.ndarray:
        """PCA-based spectral embeddings (lightweight fallback)."""
        from sklearn.decomposition import PCA

        bands, height, width = data.shape

        # Per-band z-score normalization
        data_float = data.astype(np.float32)
        for b in range(bands):
            band = data_float[b]
            mu = np.nanmean(band)
            std = np.nanstd(band)
            if std > 0:
                data_float[b] = (band - mu) / std

        pixel_features = data_float.reshape(bands, -1).T  # (H*W, B)

        n_components = min(bands, 16)
        pca = PCA(n_components=n_components)
        embeddings_flat = pca.fit_transform(pixel_features)

        return embeddings_flat.reshape(height, width, n_components)

    @staticmethod
    def _cluster_embeddings(embeddings: np.ndarray, n_clusters: int | str) -> np.ndarray:
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

        valid_mask = np.all(np.isfinite(flat), axis=1)
        valid_pixels = flat[valid_mask]

        if len(valid_pixels) == 0:
            return np.zeros((1, height, width), dtype=np.int32)

        # Subsample for efficiency
        max_samples = 50_000
        if len(valid_pixels) > max_samples:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(valid_pixels), max_samples, replace=False)
            sample = valid_pixels[indices]
        else:
            sample = valid_pixels

        if not isinstance(n_clusters, int) or n_clusters < 2:
            best_k = 10
            best_score = -1.0
            for k in [5, 10, 15, 20, 30]:
                km = KMeans(n_clusters=k, n_init=3, random_state=42)
                labels = km.fit_predict(sample)
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(
                        sample, labels, sample_size=min(5000, len(sample))
                    )
                    if score > best_score:
                        best_score = score
                        best_k = k
            n_clusters = best_k
            logger.info("Auto-selected %d clusters (silhouette=%.3f)", n_clusters, best_score)

        km = KMeans(n_clusters=int(n_clusters), n_init=5, random_state=42)
        km.fit(sample)

        labels = np.zeros(len(flat), dtype=np.int32)
        labels[valid_mask] = km.predict(valid_pixels) + 1  # 0 = nodata

        return labels.reshape(1, height, width)
