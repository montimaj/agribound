"""
Prithvi-EO-2.0 engine (terratorch).

Prithvi-EO-2.0 (Szwarcman et al., 2026, IEEE TGRS, doi:10.1109/TGRS.2025.3642610)
is a ViT masked-autoencoder pre-trained on HLS Blue, Green, Red, narrow NIR,
SWIR 1 and SWIR 2 surface reflectance. Agribound builds the encoder from the
terratorch backbone registry and runs it on single-date composites
(``num_frames=1``). Three modes (``engine_params["mode"]``):

``"embed"`` (label-free)
    Patch-token features of one encoder layer (the last, normalised layer by
    default) from non-overlapping tiles (tiles that extend past the raster
    are filled by mirror reflection of the raster), interpolated bilinearly
    between patch-token centres to pixel resolution and clustered with
    K-means (fitted on a seeded sample of at most 50 000 pixels); 4-connected
    regions of one cluster become polygons. Clusters are land-cover
    segments, not field instances.
``"segment"``
    A Prithvi + UPerNet segmentation model fine-tuned by agribound
    (``fine_tune=True``, see ``agribound.engines.finetune._prithvi``) or any
    terratorch ``SemanticSegmentationTask`` checkpoint trained on the same
    six bands and normalisation with class 1 = field interior and class 2 =
    field boundary, run with terratorch's ``tiled_inference`` on the whole
    raster (held in memory with
    its class logits). Each field interior region (class 1) is grown back
    over the predicted boundary class (2) by the boundary width used in
    training, so neighbouring fields do not overlap.
``"pca"``
    Baseline without the ViT: K-means on the PCA of per-band z-scores of R,
    G, B, NIR.

The default mode is ``"segment"`` when ``engine_params["checkpoint_path"]``
is set (as after fine-tuning) and ``"embed"`` otherwise.

Inputs are the six bands of :func:`agribound.engines.finetune._data.prithvi_band_names`
in surface reflectance x 10000 (the scale of agribound's Sentinel-2, Landsat
and HLS composites), normalised with the Prithvi-EO-2.0 means and standard
deviations (:data:`PRITHVI_MEAN`, :data:`PRITHVI_STD`); no other scaling is
applied.
Invalid pixels are set to the band means (0 after normalisation) and to
label 0 in the output. The ``embed`` and ``segment`` modes need terratorch
(``pip install agribound[prithvi]`` or ``environment-gfm.yml``).
"""

from __future__ import annotations

import datetime as _dt
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine
from agribound.registry import ENGINE_REGISTRY

logger = logging.getLogger(__name__)

#: Prithvi-EO-2.0 normalisation statistics (HLS reflectance x 10000); identical
#: to terratorch's ``PRITHVI_V2_MEAN``/``PRITHVI_V2_STD``.
PRITHVI_MEAN = [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0]
PRITHVI_STD = [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0]

#: terratorch ``HLSBands`` names in model input order.
PRITHVI_HLS_BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]

#: Hugging Face model name -> terratorch backbone registry name.
PRITHVI_MODELS: dict[str, str] = {
    "Prithvi-EO-2.0-tiny-TL": "prithvi_eo_v2_tiny_tl",
    "Prithvi-EO-2.0-100M-TL": "prithvi_eo_v2_100_tl",
    "Prithvi-EO-2.0-300M": "prithvi_eo_v2_300",
    "Prithvi-EO-2.0-300M-TL": "prithvi_eo_v2_300_tl",
    "Prithvi-EO-2.0-600M": "prithvi_eo_v2_600",
    "Prithvi-EO-2.0-600M-TL": "prithvi_eo_v2_600_tl",
}

#: Registry name -> (Hugging Face repo, weights file), as in terratorch 1.2.13
#: ``prithvi_vit.pretrained_weights``.
PRITHVI_WEIGHTS: dict[str, tuple[str, str]] = {
    "prithvi_eo_v2_tiny_tl": (
        "ibm-nasa-geospatial/Prithvi-EO-2.0-tiny-TL",
        "Prithvi_EO_V2_tiny_TL.pt",
    ),
    "prithvi_eo_v2_100_tl": (
        "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL",
        "Prithvi_EO_V2_100M_TL.pt",
    ),
    "prithvi_eo_v2_300": ("ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "Prithvi_EO_V2_300M.pt"),
    "prithvi_eo_v2_300_tl": (
        "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL",
        "Prithvi_EO_V2_300M_TL.pt",
    ),
    "prithvi_eo_v2_600": ("ibm-nasa-geospatial/Prithvi-EO-2.0-600M", "Prithvi_EO_V2_600M.pt"),
    "prithvi_eo_v2_600_tl": (
        "ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL",
        "Prithvi_EO_V2_600M_TL.pt",
    ),
}

#: Registry name -> (transformer depth, patch size), from terratorch 1.2.13
#: ``prithvi_vit.prithvi_cfgs``.
PRITHVI_ARCH: dict[str, tuple[int, int]] = {
    "prithvi_eo_v2_tiny_tl": (12, 16),
    "prithvi_eo_v2_100_tl": (12, 16),
    "prithvi_eo_v2_300": (24, 16),
    "prithvi_eo_v2_300_tl": (24, 16),
    "prithvi_eo_v2_600": (32, 14),
    "prithvi_eo_v2_600_tl": (32, 14),
}

DEFAULT_PRITHVI_MODEL = "Prithvi-EO-2.0-300M-TL"
_MODES = ("embed", "segment", "pca")
_PREDICT_BUDGET_BYTES = 256 * 2**20


def resolve_prithvi_model(name: str | None) -> str:
    """Return the terratorch registry name for a Prithvi-EO-2.0 model.

    Accepts a Hugging Face name (``"Prithvi-EO-2.0-300M-TL"``, with or without
    the ``"ibm-nasa-geospatial/"`` prefix) or a registry name
    (``"prithvi_eo_v2_300_tl"``, optionally prefixed ``"terratorch_"``).

    Raises
    ------
    ValueError
        For unknown names.
    """
    text = str(name or DEFAULT_PRITHVI_MODEL).strip()
    text = text.removeprefix("ibm-nasa-geospatial/").removeprefix("terratorch_")
    if text in PRITHVI_MODELS:
        return PRITHVI_MODELS[text]
    if text in PRITHVI_WEIGHTS:
        return text
    raise ValueError(
        f"Unknown Prithvi model {name!r}. Choose one of {list(PRITHVI_MODELS)} "
        f"(or registry names {list(PRITHVI_WEIGHTS)})"
    )


def select_indices(depth: int) -> list[int]:
    """Encoder layers fed to the UPerNet decoder: the ends of the four quarters.

    Gives ``[2, 5, 8, 11]`` (12 layers), ``[5, 11, 17, 23]`` (24) and
    ``[7, 15, 23, 31]`` (32), the indices used in the Prithvi-EO-2.0 and
    terratorch example configurations.
    """
    return [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]


#: Pool scales of the pyramid pooling module of terratorch's ``UperNetDecoder``
#: (its default; agribound does not change it).
UPERNET_POOL_SCALES = (1, 2, 3, 6)


def _hw(size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(size, int | np.integer):
        return int(size), int(size)
    h, w = size
    return int(h), int(w)


def upernet_coarsest_side(size: int | tuple[int, int], patch: int) -> int | tuple[int, int]:
    """Size in pixels of the coarsest Prithvi + UPerNet feature map.

    terratorch's ``PixelWiseModel`` reflect-pads inputs to a multiple of twice
    the patch size, the encoder gives one token per patch, and
    ``LearnedInterpolateToPyramidal`` max-pools the token grid by 2 for the
    coarsest level: ``ceil(side / (2 * patch))`` per side. Returns an int for
    a square *size* (int) and ``(height, width)`` for a ``(height, width)``
    input.
    """
    h, w = _hw(size)
    ch, cw = -(-h // (2 * int(patch))), -(-w // (2 * int(patch)))
    return ch if isinstance(size, int | np.integer) else (ch, cw)


def mps_adaptive_pool_supported(height: int, width: int, output: int) -> bool:
    """Whether PyTorch's MPS backend runs ``AdaptiveAvgPool2d(output)`` on *height* x *width*.

    Checked against torch 2.10 on Apple MPS for every input of 1-14 px per
    side and outputs 1, 2, 3, 6: both sides must be at least the output or
    both at most the output; when the height is at least the output, both
    sides must be divisible by it, otherwise the output must be divisible by
    both sides.
    """
    h, w, s = int(height), int(width), int(output)
    if not ((h >= s and w >= s) or (h <= s and w <= s)):
        return False
    if h >= s:
        return h % s == 0 and w % s == 0
    return s % h == 0 and s % w == 0


def upernet_mps_compatible(size: int | tuple[int, int], patch: int) -> bool:
    """Whether Prithvi + UPerNet can run on Apple MPS with inputs of *size* px.

    UPerNet's pyramid pooling applies ``AdaptiveAvgPool2d`` with output sizes
    :data:`UPERNET_POOL_SCALES` (1, 2, 3, 6) to the coarsest feature map
    (:func:`upernet_coarsest_side`), and MPS supports only some input/output
    size pairs (:func:`mps_adaptive_pool_supported`). For square inputs the
    coarsest side must be a multiple of 6 or equal to 1, e.g. 161-192 or
    353-384 px tiles for patch 16 and 141-168 px for patch 14, or tiles of at
    most 2 x patch px. The default 224 px gives a side of 7 (patch 16) or 8
    (patch 14) and fails on MPS. *size* is an int (square) or
    ``(height, width)``.
    """
    ch, cw = _hw(upernet_coarsest_side(_hw(size), patch))
    return (
        ch > 0
        and cw > 0
        and all(mps_adaptive_pool_supported(ch, cw, s) for s in UPERNET_POOL_SCALES)
    )


def upernet_device(device: str, size: int | tuple[int, int], patch: int, what: str) -> str:
    """Return *device*, or ``"cpu"`` (logged at WARNING) where MPS cannot run UPerNet.

    *size* is the model input size in pixels (int for square tiles, or
    ``(height, width)``). See :func:`upernet_mps_compatible`. Only the
    compute device changes; the model and its inputs are the same.
    """
    if str(device).startswith("mps") and not upernet_mps_compatible(size, patch):
        h, w = _hw(size)
        ch, cw = _hw(upernet_coarsest_side((h, w), patch))
        logger.warning(
            "Prithvi + UPerNet %s runs on CPU instead of MPS: with %d x %d px inputs and patch "
            "size %d the coarsest feature map is %d x %d px, which PyTorch's MPS adaptive "
            "pooling (UPerNet pool scales %s) does not support. Tiles of %d px run on MPS.",
            what,
            h,
            w,
            patch,
            ch,
            cw,
            UPERNET_POOL_SCALES,
            12 * patch,
        )
        return "cpu"
    return device


def _patch_size_of(model_args: dict[str, Any], model: Any = None) -> int | None:
    """Patch size of a Prithvi segmentation model (registry table, else the model)."""
    backbone = str(model_args.get("backbone") or "").removeprefix("terratorch_")
    if backbone in PRITHVI_ARCH:
        return PRITHVI_ARCH[backbone][1]
    patch = getattr(model, "patch_size", None)
    if patch is None:
        return None
    return int(patch[-1]) if isinstance(patch, list | tuple) else int(patch)


def composite_mid_date(config: AgriboundConfig) -> tuple[int, int]:
    """``(year, day of year)`` of the midpoint of the compositing window.

    The window is ``config.date_range`` or the calendar year ``config.year``
    (midpoint 2 July in common years and 1 July in leap years; day of year
    183 in both). The midpoint date is rounded down to whole days.
    """
    if config.date_range:
        start = _dt.date.fromisoformat(str(config.date_range[0]))
        end = _dt.date.fromisoformat(str(config.date_range[1]))
    else:
        start = _dt.date(int(config.year), 1, 1)
        end = _dt.date(int(config.year), 12, 31)
    mid = start + (end - start) / 2
    return mid.year, mid.timetuple().tm_yday


def raster_centre_latlon(raster_path: str) -> tuple[float, float]:
    """``(lat, lon)`` in EPSG:4326 of the centre of a raster's footprint."""
    import rasterio
    from rasterio.warp import transform as warp_transform

    with rasterio.open(raster_path) as src:
        b = src.bounds
        x, y = (b.left + b.right) / 2.0, (b.bottom + b.top) / 2.0
        if src.crs is None:
            raise ValueError(f"{raster_path} has no CRS")
        lon, lat = warp_transform(src.crs, "EPSG:4326", [x], [y])
    return float(lat[0]), float(lon[0])


def _import_terratorch(feature: str):
    try:
        import terratorch  # noqa: F401
        import torch
    except ImportError:
        raise ImportError(
            f"terratorch is required for Prithvi {feature}. Install the GFM environment "
            "(environment-gfm.yml) or 'pip install agribound[prithvi]' (it cannot be "
            "installed together with the 'ftw' extra)."
        ) from None
    return torch


class PrithviEngine(DelineationEngine):
    """Field delineation with Prithvi-EO-2.0 (see the module docstring)."""

    name = "prithvi"
    supported_sources = list(ENGINE_REGISTRY["prithvi"]["supported_sources"])
    requires_bands = list(ENGINE_REGISTRY["prithvi"]["requires_bands"])

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run Prithvi-based field delineation.

        Parameters
        ----------
        raster_path : str
            Composite GeoTIFF.
        config : AgriboundConfig
            Pipeline configuration. ``engine_params``:

            - ``mode``: ``"embed"`` | ``"segment"`` | ``"pca"`` (default: see
              the module docstring).
            - ``checkpoint_path``: terratorch ``SemanticSegmentationTask``
              checkpoint (``segment`` mode).
            - ``model_name``: Prithvi-EO-2.0 variant for ``embed`` mode
              (default ``"Prithvi-EO-2.0-300M-TL"``; see :data:`PRITHVI_MODELS`).
            - ``tile_size``: tile edge in pixels (default 224). In ``embed``
              mode it must be a multiple of the patch size (16; 14 for the
              600M models); tiles that extend past the raster are filled by
              mirror reflection of the raster. In ``segment`` mode a raster
              that fits in one tile is passed whole (terratorch reflect-pads
              it to a multiple of twice the patch size); a larger raster is
              run through terratorch's ``tiled_inference``, after mirror
              padding a side shorter than the tile to the tile size. On
              Apple MPS the segmentation model runs on CPU (logged at
              WARNING) unless the model input size passes
              :func:`upernet_mps_compatible` (e.g. 192 px tiles for patch
              16). ``patch_size`` is accepted as a legacy alias.
            - ``stride``: tile step for ``segment`` mode (default
              ``tile_size - 32``).
            - ``batch_size``: tiles per forward pass (default 8).
            - ``n_clusters``: int or ``"auto"`` (silhouette over 5, 10, 15,
              20, 30 on a seeded sample) for ``embed``/``pca``.
            - ``embed_layer``: encoder layer used in ``embed`` mode (default
              -1, the normalised last layer).
            - ``temporal_coords``: ``[year, day_of_year]``, or *False* to not
              pass them (default: :func:`composite_mid_date`); ``embed``
              mode with a ``*-TL`` model only (``segment`` mode and
              fine-tuning pass no coordinates).
            - ``location_coords``: ``[lat, lon]``, or *False* (default: the
              raster centre); ``embed`` mode with a ``*-TL`` model only.
            - ``dilate_interior_px``: ``segment`` mode; pixels by which each
              interior region is grown over the predicted boundary class
              (default: the checkpoint's training ``boundary_erosion`` if
              recorded, else ``engine_params["boundary_erosion"]`` or 2; see
              :func:`agribound.engines.finetune._data.interior_polygons`).
            - ``value_scale``: required for ``source="local"``
              (``"reflectance_x10000"`` or ``"unit"``). Without
              ``config.bands`` a local raster's bands 1-6 are read as Blue,
              Green, Red, narrow NIR, SWIR 1, SWIR 2 (``pca`` mode: bands
              1-4 as R, G, B, NIR).
            - ``pretrained``: *False* builds a randomly initialised encoder in
              ``embed`` mode (for tests only; logged as a warning).

        Returns
        -------
        geopandas.GeoDataFrame
            Polygons with ``gdf.attrs["engine_meta"]``.

        Raises
        ------
        RuntimeError
            If ``mode="segment"`` is requested without a checkpoint.
        ValueError
            For unknown modes, models or non-reflectance inputs.
        """
        params = config.engine_params
        checkpoint = params.get("checkpoint_path")
        mode = str(params.get("mode") or ("segment" if checkpoint else "embed")).lower()
        if mode not in _MODES:
            raise ValueError(f"Unknown Prithvi mode {mode!r}. Choose from {_MODES}")
        self.validate_input(raster_path, config)
        if mode == "segment":
            if not checkpoint:
                raise RuntimeError(
                    "Prithvi mode='segment' needs a fine-tuned checkpoint: set fine_tune=True "
                    "with reference_boundaries, or engine_params['checkpoint_path']. Use "
                    "mode='embed' for label-free clustering."
                )
            return self._segment_mode(raster_path, config, str(checkpoint))
        if checkpoint:
            logger.warning(
                "Prithvi mode=%r ignores engine_params['checkpoint_path'] (%s)", mode, checkpoint
            )
        if mode == "pca":
            return self._pca_mode(raster_path, config)
        return self._embed_mode(raster_path, config)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    @staticmethod
    def _bands(config: AgriboundConfig) -> tuple[list[str], list[int]]:
        from agribound.engines.base import get_canonical_band_indices
        from agribound.engines.finetune._data import prithvi_band_names

        names = prithvi_band_names(config.source, config.bands)
        return names, get_canonical_band_indices(config.source, names, bands=config.bands)

    @staticmethod
    def _normalise(
        data: np.ndarray, source: str, value_scale: str | None, nodata: float | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (normalised float32 bands with invalid = 0, valid mask)."""
        from agribound.engines.finetune._data import prithvi_reflectance, valid_pixels

        valid = valid_pixels(data, nodata)
        refl = prithvi_reflectance(data, source, value_scale)
        mean = np.asarray(PRITHVI_MEAN, dtype=np.float32)[:, None, None]
        std = np.asarray(PRITHVI_STD, dtype=np.float32)[:, None, None]
        norm = (refl - mean) / std
        norm = np.where(valid[None], norm, 0.0).astype(np.float32)
        return norm, valid

    @staticmethod
    def _coords(config: AgriboundConfig, raster_path: str) -> dict[str, Any]:
        params = config.engine_params
        out: dict[str, Any] = {}
        tc = params.get("temporal_coords")
        if tc is not False:
            year, doy = composite_mid_date(config) if tc is None else (tc[0], tc[1])
            out["temporal_coords"] = [float(year), float(doy)]
        lc = params.get("location_coords")
        if lc is not False:
            lat, lon = raster_centre_latlon(raster_path) if lc is None else (lc[0], lc[1])
            out["location_coords"] = [float(lat), float(lon)]
        return out

    # ------------------------------------------------------------------
    # Embed mode
    # ------------------------------------------------------------------

    def _embed_mode(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Label-free clustering of Prithvi patch-token embeddings."""
        import rasterio
        from rasterio.windows import Window

        from agribound._cache import cache_path
        from agribound._repro import get_rng
        from agribound.engines.finetune._data import (
            hf_cached_file_info,
            package_versions,
            raster_fingerprint,
            read_json,
            write_json,
        )

        torch = _import_terratorch("embedding mode")
        from terratorch.registry import BACKBONE_REGISTRY

        params = config.engine_params
        registry_name = resolve_prithvi_model(params.get("model_name"))
        depth, patch = PRITHVI_ARCH[registry_name]
        tile = int(params.get("tile_size", params.get("patch_size", 224)))
        if tile % patch:
            raise ValueError(f"tile_size {tile} must be a multiple of the patch size {patch}")
        batch_size = max(1, int(params.get("batch_size", 8)))
        layer = int(params.get("embed_layer", -1))
        n_clusters = params.get("n_clusters", "auto")
        value_scale = params.get("value_scale")
        pretrained = bool(params.get("pretrained", True))
        is_tl = registry_name.endswith("_tl")
        coords = self._coords(config, raster_path) if is_tl else {}
        names, indices = self._bands(config)
        device = config.resolve_device()

        cluster_path = cache_path(
            config,
            "prithvi_embed_clusters",
            ".tif",
            raster_fingerprint(raster_path),
            registry_name,
            f"pretrained={pretrained}",
            f"bands={names}:{indices}",
            f"value_scale={value_scale}",
            f"tile={tile}",
            "pad=reflect",
            f"layer={layer}",
            f"coords={coords}",
            f"k={n_clusters}",
            f"seed={config.seed}",
        )
        meta: dict[str, Any] = {
            "backend": "terratorch",
            "mode": "embed",
            "model": registry_name,
            "weights_repo": PRITHVI_WEIGHTS[registry_name][0] if pretrained else None,
            "weights_file": PRITHVI_WEIGHTS[registry_name][1] if pretrained else None,
            "pretrained": pretrained,
            "num_frames": 1,
            "band_names": names,
            "band_indices": indices,
            "input_units": "surface reflectance x 10000",
            "normalisation": {"mean": PRITHVI_MEAN, "std": PRITHVI_STD},
            "tile_size": tile,
            "tile_padding": "reflect",
            "patch_size": patch,
            "embed_layer": layer,
            "coords": coords,
            "seed": config.seed,
            "device": device,
        }
        meta.update(package_versions("terratorch", "torch"))
        if config.source == "landsat":
            meta["nir_note"] = "Landsat SR_B5: narrow NIR on L8/9, broad NIR on L5/7"

        sidecar = cluster_path.with_suffix(".json")
        if cluster_path.exists() and sidecar.exists():
            logger.info("Using cached Prithvi clusters: %s", cluster_path)
            meta.update(read_json(sidecar), cache_reused=True)
        else:
            if not pretrained:
                logger.warning("Prithvi embed mode with a randomly initialised encoder")
            logger.info("Building Prithvi encoder %s (bands %s)", registry_name, names)
            model = BACKBONE_REGISTRY.build(
                registry_name, pretrained=pretrained, bands=PRITHVI_HLS_BANDS, num_frames=1
            )
            model_patch = int(model.patch_embed.patch_size[-1])
            if model_patch != patch:
                raise RuntimeError(
                    f"{registry_name} has patch size {model_patch}, expected {patch} "
                    "(agribound's PRITHVI_ARCH table is out of date for this terratorch)"
                )
            if pretrained:
                hub = hf_cached_file_info(*PRITHVI_WEIGHTS[registry_name])
                meta["weights_revision"] = hub["revision"]
                meta["weights_sha256"] = hub["sha256"]
            model = model.float().eval().to(device)
            with rasterio.open(raster_path) as src:
                height, width, nodata = src.height, src.width, src.nodata
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
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                }

                def read_rows(row0: int, nrows: int) -> tuple[np.ndarray, np.ndarray]:
                    data = src.read(indices, window=Window(0, row0, width, nrows))
                    return self._normalise(data, config.source, value_scale, nodata)

                tokens, valid = extract_token_map(
                    read_rows,
                    height,
                    width,
                    model,
                    tile=tile,
                    patch=patch,
                    batch_size=batch_size,
                    device=device,
                    coords=coords,
                    layer=layer,
                    torch_module=torch,
                )
            del model
            _empty_cache(torch, device)

            rng = get_rng(config, "prithvi", "embed", "kmeans-sample")
            flat_valid = np.flatnonzero(valid)
            if flat_valid.size == 0:
                logger.warning("No valid pixels in %s", raster_path)
                return _empty(profile["crs"], meta)
            n_sample = min(50_000, flat_valid.size)
            pick = np.sort(rng.choice(flat_valid, n_sample, replace=False))
            ys, xs = np.divmod(pick, width)
            sample = interpolate_tokens(tokens, patch, ys, xs)
            km, k, score = fit_kmeans(sample, n_clusters, config.seed)
            meta.update({"n_clusters": int(k), "silhouette": score, "n_fit_samples": int(n_sample)})

            tmp = cluster_path.with_name(cluster_path.stem + ".partial.tif")
            rows_per_strip = max(1, int(_PREDICT_BUDGET_BYTES // (width * tokens.shape[-1] * 4)))
            with rasterio.open(tmp, "w", **profile) as dst:
                for row0 in range(0, height, rows_per_strip):
                    nrows = min(rows_per_strip, height - row0)
                    ys_strip = np.arange(row0, row0 + nrows)
                    emb = interpolate_token_rows(tokens, patch, ys_strip, width)
                    labels = km.predict(emb.reshape(-1, emb.shape[-1])).reshape(nrows, width) + 1
                    labels = np.where(valid[row0 : row0 + nrows], labels, 0).astype(np.int32)
                    dst.write(labels, 1, window=Window(0, row0, width, nrows))
            tmp.replace(cluster_path)
            write_json(
                sidecar,
                {
                    k_: meta.get(k_)
                    for k_ in (
                        "n_clusters",
                        "silhouette",
                        "n_fit_samples",
                        "device",
                        "weights_revision",
                        "weights_sha256",
                    )
                },
            )

        from agribound.postprocess.polygonize import polygonize_mask

        gdf = polygonize_mask(str(cluster_path), min_area_m2=config.min_field_area_m2)
        gdf.attrs["engine_meta"] = meta
        logger.info("Prithvi embedding clustering delineated %d polygons", len(gdf))
        return gdf

    # ------------------------------------------------------------------
    # Segment mode
    # ------------------------------------------------------------------

    def _segment_mode(
        self, raster_path: str, config: AgriboundConfig, checkpoint: str
    ) -> gpd.GeoDataFrame:
        """Prithvi + UPerNet segmentation with a terratorch checkpoint."""
        import rasterio

        from agribound._cache import cache_path
        from agribound.engines.finetune._data import (
            file_sha256,
            interior_polygons,
            package_versions,
            raster_fingerprint,
            read_checkpoint_hparams,
            read_json,
            write_json,
        )

        torch = _import_terratorch("segmentation mode")
        from terratorch.tasks import SemanticSegmentationTask
        from terratorch.tasks.tiled_inference import tiled_inference

        ckpt = Path(checkpoint).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"Prithvi checkpoint not found: {ckpt}")
        params = config.engine_params
        tile = int(params.get("tile_size", params.get("patch_size", 224)))
        stride = int(params.get("stride", max(tile - 32, 1)))
        batch_size = max(1, int(params.get("batch_size", 8)))
        value_scale = params.get("value_scale")
        names, indices = self._bands(config)
        device = config.resolve_device()
        training = _read_training_meta(ckpt)
        dilate_px = params.get("dilate_interior_px")
        if dilate_px is None:
            dilate_px = training.get("boundary_erosion", params.get("boundary_erosion", 2))
        dilate_px = int(dilate_px)

        pred_path = cache_path(
            config,
            "prithvi_segmentation",
            ".tif",
            raster_fingerprint(raster_path),
            raster_fingerprint(ckpt),
            f"bands={names}:{indices}",
            f"value_scale={value_scale}",
            f"tile={tile}",
            f"stride={stride}",
            "pad=reflect-v2",
        )
        hparams = read_checkpoint_hparams(ckpt)
        model_args = dict(hparams.get("model_args") or {})
        meta: dict[str, Any] = {
            "backend": "terratorch",
            "mode": "segment",
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_sha256": file_sha256(ckpt),
            "model": model_args.get("backbone"),
            "decoder": model_args.get("decoder"),
            "necks": model_args.get("necks"),
            "peft_config": model_args.get("peft_config"),
            "num_classes": model_args.get("num_classes"),
            "band_names": names,
            "band_indices": indices,
            "input_units": "surface reflectance x 10000",
            "normalisation": {"mean": PRITHVI_MEAN, "std": PRITHVI_STD},
            "tile_size": tile,
            "stride": stride,
            "field_class": 1,
            "dilate_interior_px": dilate_px,
            "device": device,
            "training": training or None,
        }
        meta.update(package_versions("terratorch", "torch"))

        pred_info = pred_path.with_suffix(".json")
        if pred_path.exists() and pred_info.exists():
            logger.info("Using cached Prithvi segmentation: %s", pred_path)
            meta.update(read_json(pred_info), cache_reused=True)
        else:
            # Weights come from the checkpoint; do not download the backbone again.
            load_args = {**model_args, "backbone_pretrained": False}
            task = SemanticSegmentationTask.load_from_checkpoint(
                str(ckpt), map_location="cpu", model_args=load_args
            )
            with rasterio.open(raster_path) as src:
                data = src.read(indices)
                profile = {
                    "driver": "GTiff",
                    "height": src.height,
                    "width": src.width,
                    "count": 1,
                    "dtype": "uint8",
                    "crs": src.crs,
                    "transform": src.transform,
                    "compress": "lzw",
                }
                nodata = src.nodata
            norm, valid = self._normalise(data, config.source, value_scale, nodata)
            del data
            height, width = valid.shape
            single_pass = height <= tile and width <= tile
            if single_pass:
                # tiled_inference runs one forward pass on the input as given;
                # the model reflect-pads it to a multiple of 2 x patch.
                model_input = (height, width)
                pad_h = pad_w = 0
            else:
                # tiled_inference needs both sides >= the tile: mirror-pad a
                # shorter side (bottom/right) to the tile size.
                model_input = (tile, tile)
                pad_h, pad_w = max(0, tile - height), max(0, tile - width)
                if pad_h or pad_w:
                    norm = np.pad(norm, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
            meta["model_input_size"] = list(model_input)
            meta["input_padding"] = (
                {"mode": "reflect", "rows": pad_h, "cols": pad_w} if pad_h or pad_w else None
            )
            patch = _patch_size_of(model_args, task.model)
            if patch is not None:
                device = upernet_device(device, model_input, patch, "segmentation")
                meta["device"] = device
            model = task.model.float().eval().to(device)
            x = torch.from_numpy(norm)[None]
            if single_pass:
                x = x.to(device)

            def forward(t):
                return model(t).output

            logits = tiled_inference(
                forward,
                x,
                h_crop=tile,
                w_crop=tile,
                h_stride=stride,
                w_stride=stride,
                batch_size=batch_size,
                device=device,
            )
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)[:height, :width]
            pred[~valid] = 0
            del model, task, logits
            _empty_cache(torch, device)
            tmp = pred_path.with_name(pred_path.stem + ".partial.tif")
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(pred, 1)
            tmp.replace(pred_path)
            write_json(
                pred_info,
                {k: meta[k] for k in ("device", "model_input_size", "input_padding")},
            )

        gdf = interior_polygons(pred_path, dilate_px, config.min_field_area_m2)
        gdf.attrs["engine_meta"] = meta
        logger.info("Prithvi segmentation delineated %d fields", len(gdf))
        return gdf

    # ------------------------------------------------------------------
    # PCA mode
    # ------------------------------------------------------------------

    def _pca_mode(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """K-means on PCA of per-band z-scores of R, G, B, NIR (no ViT)."""
        from agribound._cache import cache_path
        from agribound._repro import get_rng
        from agribound.engines.base import get_canonical_band_indices
        from agribound.engines.finetune._data import (
            raster_fingerprint,
            read_json,
            valid_pixels,
            write_json,
        )
        from agribound.io.raster import read_raster, write_raster

        params = config.engine_params
        n_clusters = params.get("n_clusters", "auto")
        indices = get_canonical_band_indices(
            config.source, ["R", "G", "B", "NIR"], bands=config.bands
        )
        cluster_path = cache_path(
            config,
            "prithvi_pca_clusters",
            ".tif",
            raster_fingerprint(raster_path),
            f"bands={indices}",
            f"k={n_clusters}",
            f"seed={config.seed}",
        )
        meta: dict[str, Any] = {
            "backend": "scikit-learn",
            "mode": "pca",
            "band_indices": indices,
            "seed": config.seed,
        }
        sidecar = cluster_path.with_suffix(".json")
        if cluster_path.exists() and sidecar.exists():
            logger.info("Using cached PCA clusters: %s", cluster_path)
            meta.update(read_json(sidecar), cache_reused=True)
        else:
            data, raster_meta = read_raster(raster_path, bands=indices)
            valid = valid_pixels(data, raster_meta.get("nodata"))
            if not valid.any():
                logger.warning("No valid pixels in %s", raster_path)
                return _empty(raster_meta.get("crs"), meta)
            features = pca_features(data, valid, config.seed)
            rng = get_rng(config, "prithvi", "pca", "kmeans-sample")
            n_valid = features.shape[0]
            pick = rng.choice(n_valid, min(50_000, n_valid), replace=False)
            km, k, score = fit_kmeans(features[np.sort(pick)], n_clusters, config.seed)
            labels = np.zeros(valid.shape, dtype=np.int32)
            labels[valid] = km.predict(features) + 1
            write_raster(
                cluster_path,
                labels[np.newaxis],
                crs=raster_meta["crs"],
                transform=raster_meta["transform"],
                nodata=0,
            )
            meta.update({"n_clusters": int(k), "silhouette": score})
            write_json(sidecar, {"n_clusters": int(k), "silhouette": score})

        from agribound.postprocess.polygonize import polygonize_mask

        gdf = polygonize_mask(str(cluster_path), min_area_m2=config.min_field_area_m2)
        gdf.attrs["engine_meta"] = meta
        logger.info("PCA clustering delineated %d polygons", len(gdf))
        return gdf

    # ------------------------------------------------------------------
    # Prefetch
    # ------------------------------------------------------------------

    @classmethod
    def prefetch(cls, config: AgriboundConfig) -> list[str]:
        """Download the Prithvi-EO-2.0 weights of ``engine_params["model_name"]``.

        The pre-trained weights are needed by ``embed`` mode (unless
        ``engine_params["pretrained"]`` is *False*) and by fine-tuning
        (``config.fine_tune``, unless ``backbone_pretrained`` is *False*).
        ``segment`` inference loads every weight from its checkpoint and
        ``pca`` mode uses no model, so nothing is downloaded for them. Files
        go to the Hugging Face cache (``HF_HOME``); set ``HF_HUB_OFFLINE=1``
        on nodes without internet access afterwards.

        Returns
        -------
        list[str]
            Local path of the weights file when needed, plus the checkpoint
            if ``engine_params["checkpoint_path"]`` exists.
        """
        from huggingface_hub import hf_hub_download

        params = config.engine_params
        checkpoint = params.get("checkpoint_path")
        mode = str(params.get("mode") or ("segment" if checkpoint else "embed")).lower()
        needs_weights = (config.fine_tune and params.get("backbone_pretrained", True)) or (
            mode == "embed" and params.get("pretrained", True)
        )
        paths: list[str] = []
        if needs_weights:
            registry_name = resolve_prithvi_model(params.get("model_name"))
            repo, filename = PRITHVI_WEIGHTS[registry_name]
            paths.append(hf_hub_download(repo_id=repo, filename=filename))
        else:
            logger.info("Prithvi mode=%r needs no pre-trained weights; nothing downloaded", mode)
        if checkpoint and Path(checkpoint).is_file():
            paths.append(str(Path(checkpoint).resolve()))
        return paths


# ---------------------------------------------------------------------------
# Helpers (module level so they can be tested without a model)
# ---------------------------------------------------------------------------


def extract_token_map(
    read_rows,
    height: int,
    width: int,
    model,
    *,
    tile: int,
    patch: int,
    batch_size: int,
    device: str,
    coords: dict[str, Any],
    layer: int,
    torch_module,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the encoder over non-overlapping tiles and assemble patch tokens.

    Parameters
    ----------
    read_rows : callable
        ``read_rows(row0, nrows) -> (normalised (C, nrows, width) array,
        valid (nrows, width) mask)``.
    height, width : int
        Raster size.
    model : torch.nn.Module
        Encoder returning a list of ``(B, 1 + N, D)`` token tensors (CLS
        first), as built by terratorch's backbone registry.
    tile, patch : int
        Tile and patch size in pixels (``tile % patch == 0``).
    batch_size : int
        Tiles per forward pass.
    device : str
        Torch device.
    coords : dict
        Optional ``temporal_coords`` ``[year, doy]`` and ``location_coords``
        ``[lat, lon]``, repeated for every tile.
    layer : int
        Index into the encoder outputs.
    torch_module : module
        ``torch``.

    Returns
    -------
    tokens : numpy.ndarray
        ``(n_rows * t, n_cols * t, D)`` float32, ``t = tile // patch``; the
        grid covers the raster extended at the bottom and right to whole
        tiles by mirror reflection of the raster
        (:func:`agribound.engines.finetune._data.reflect_indices`), so no
        tile holds constant padding.
    valid : numpy.ndarray
        ``(height, width)`` bool mask of valid pixels.
    """
    from agribound.engines.finetune._data import reflect_indices

    torch = torch_module
    t = tile // patch
    n_rows = -(-height // tile)
    n_cols = -(-width // tile)
    col_map = reflect_indices(np.arange(n_cols * tile), width)
    valid = np.zeros((height, width), dtype=bool)
    tokens: np.ndarray | None = None

    def _run(batch: list[np.ndarray]) -> np.ndarray:
        x = torch.from_numpy(np.stack(batch)).to(device)
        kwargs = {}
        if "temporal_coords" in coords:
            kwargs["temporal_coords"] = torch.tensor(
                [[coords["temporal_coords"]]] * len(batch), dtype=torch.float32, device=device
            )
        if "location_coords" in coords:
            kwargs["location_coords"] = torch.tensor(
                [coords["location_coords"]] * len(batch), dtype=torch.float32, device=device
            )
        with torch.no_grad():
            feats = model(x, **kwargs)[layer]
        return feats[:, 1:, :].float().cpu().numpy()

    for r in range(n_rows):
        row0 = r * tile
        nrows = min(tile, height - row0)
        rows = reflect_indices(np.arange(row0, row0 + tile), height)
        lo, hi = int(rows.min()), int(rows.max()) + 1
        norm, valid_rows = read_rows(lo, hi - lo)
        valid[row0 : row0 + nrows] = valid_rows[row0 - lo : row0 - lo + nrows]
        strip = np.ascontiguousarray(norm[:, rows - lo][:, :, col_map], dtype=np.float32)
        tiles = [strip[:, :, c * tile : (c + 1) * tile] for c in range(n_cols)]
        for b0 in range(0, n_cols, batch_size):
            out = _run(tiles[b0 : b0 + batch_size])
            if tokens is None:
                tokens = np.zeros((n_rows * t, n_cols * t, out.shape[-1]), dtype=np.float32)
            if out.shape[1] != t * t:
                raise RuntimeError(
                    f"Encoder returned {out.shape[1]} patch tokens per tile, expected {t * t}"
                )
            for i in range(out.shape[0]):
                c = b0 + i
                tokens[r * t : (r + 1) * t, c * t : (c + 1) * t] = out[i].reshape(t, t, -1)
    assert tokens is not None
    return tokens, valid


def _axis_weights(
    positions: np.ndarray, patch: int, n_tokens: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear-interpolation indices/weights between patch-token centres.

    Pixel ``p`` (centre at ``p + 0.5``) lies at token coordinate
    ``(p + 0.5) / patch - 0.5``, clamped to ``[0, n_tokens - 1]``.
    """
    src = (np.asarray(positions, dtype=np.float64) + 0.5) / patch - 0.5
    src = np.clip(src, 0.0, n_tokens - 1)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, n_tokens - 1)
    w = (src - i0).astype(np.float32)
    return i0, i1, w


def interpolate_tokens(
    tokens: np.ndarray, patch: int, ys: np.ndarray, xs: np.ndarray
) -> np.ndarray:
    """Bilinear token embeddings at pixel positions ``(ys, xs)`` -> ``(N, D)``."""
    y0, y1, wy = _axis_weights(ys, patch, tokens.shape[0])
    x0, x1, wx = _axis_weights(xs, patch, tokens.shape[1])
    wy = wy[:, None]
    wx = wx[:, None]
    top = tokens[y0, x0] * (1 - wx) + tokens[y0, x1] * wx
    bottom = tokens[y1, x0] * (1 - wx) + tokens[y1, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def interpolate_token_rows(
    tokens: np.ndarray, patch: int, ys: np.ndarray, width: int
) -> np.ndarray:
    """Bilinear token embeddings for whole pixel rows -> ``(len(ys), width, D)``."""
    y0, y1, wy = _axis_weights(ys, patch, tokens.shape[0])
    x0, x1, wx = _axis_weights(np.arange(width), patch, tokens.shape[1])
    wx = wx[None, :, None]
    top = tokens[y0][:, x0] * (1 - wx) + tokens[y0][:, x1] * wx
    bottom = tokens[y1][:, x0] * (1 - wx) + tokens[y1][:, x1] * wx
    wy = wy[:, None, None]
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def fit_kmeans(sample: np.ndarray, n_clusters: int | str, seed: int):
    """Fit K-means; ``"auto"`` picks k in (5, 10, 15, 20, 30) by silhouette.

    Returns
    -------
    tuple
        ``(fitted KMeans, k, silhouette or None)``.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    score: float | None = None
    if isinstance(n_clusters, str) or n_clusters is None:
        if str(n_clusters).lower() != "auto":
            raise ValueError(f"n_clusters must be an int >= 2 or 'auto', got {n_clusters!r}")
        best_k, best = None, -np.inf
        for k in (5, 10, 15, 20, 30):
            if k >= len(sample):
                break
            labels = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(sample)
            if len(np.unique(labels)) < 2:
                continue
            s = float(
                silhouette_score(
                    sample, labels, sample_size=min(5000, len(sample)), random_state=seed
                )
            )
            if s > best:
                best_k, best = k, s
        if best_k is None:
            raise ValueError("Could not choose n_clusters automatically (too few samples)")
        n_clusters, score = best_k, best
        logger.info("Auto-selected %d clusters (silhouette=%.3f)", n_clusters, score)
    k = int(n_clusters)
    if k < 2:
        raise ValueError(f"n_clusters must be >= 2, got {k}")
    km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(sample)
    return km, k, score


def pca_features(data: np.ndarray, valid: np.ndarray, seed: int) -> np.ndarray:
    """PCA of per-band z-scores of the valid pixels -> ``(n_valid, n_bands)``."""
    from sklearn.decomposition import PCA

    pixels = np.asarray(data, dtype=np.float64)[:, valid].T
    mu = pixels.mean(axis=0)
    sd = pixels.std(axis=0)
    sd[sd == 0] = 1.0
    z = (pixels - mu) / sd
    return PCA(n_components=z.shape[1], random_state=seed).fit_transform(z).astype(np.float32)


def _read_training_meta(ckpt: Path) -> dict[str, Any]:
    """Training metadata written next to an agribound checkpoint, if any."""
    from agribound.engines.finetune._data import read_training_meta

    return read_training_meta(ckpt)


def _empty_cache(torch, device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def _empty(crs: Any, meta: dict[str, Any]) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
    gdf.attrs["engine_meta"] = meta
    return gdf
