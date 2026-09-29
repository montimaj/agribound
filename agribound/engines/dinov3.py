"""
DINOv3 semantic segmentation engine (geoai-py).

Runs geoai's ``DINOv3Segmenter`` -- a DINOv3 ViT backbone (Siméoni et al.,
2025, arXiv:2508.10104) with a DPT decoder -- trained by agribound on
reference boundaries (``fine_tune=True``, see
``agribound.engines.finetune._dinov3``) into background / field interior /
field boundary classes. There are no published field-boundary weights, so a
fine-tuned Lightning ``.ckpt`` is required. Each field interior region is
grown back over the predicted boundary class by the boundary width used in
training (:func:`agribound.engines.finetune._data.interior_polygons`), so
neighbouring fields do not overlap.

Input
-----
Canonical R, G, B bands with a scene-level 1-99 percentile stretch to uint8
(:func:`agribound.engines.finetune._data.write_rgb_input`), stored as
float32 ``uint8 / 255``. For ``source="local"`` without ``config.bands``
bands 1, 2, 3 are read as R, G, B. geoai divides a window by 255 only when
its maximum exceeds 1 and applies no mean/std normalisation, so the model
sees exactly these [0, 1] values at training and inference. The SAT-493M
backbone was pre-trained with the normalisation mean (0.430, 0.411, 0.296)
and standard deviation (0.213, 0.156, 0.143) (facebookresearch/dinov3
README), so its pre-trained features receive inputs that are not normalised
as in pre-training; this matters most when the backbone is frozen
(``use_lora`` or ``freeze_backbone``). Normalising the chips beforehand is
not a workaround: geoai divides any chip or window whose maximum exceeds 1
by 255.

Sliding window
--------------
geoai's ``dinov3_segment_geotiff`` zero-pads every window that extends past
the raster to the full window size, and its last window along each axis
starts at ``min(i * stride, size - 1)``, so it can extend past the raster.
The ViT attends over the whole window, so zero padding changes the features
of the real pixels. Agribound therefore (:func:`plan_dinov3_windows`) uses
the training chip size as the window by default, caps the window at the
larger raster side (rounded up to the 16 px patch size), and extends the
RGB input at the bottom and right by mirror reflection so that every window
lies inside it; the prediction is then cropped back to the raster grid.

Weights and offline use
-----------------------
geoai 0.43.1 builds the backbone with ``torch.hub.load`` from
``facebookresearch/dinov3`` (GitHub, or the local clone named by the
``DINOV3_LOCATION`` environment variable) and then loads the SAT-493M ViT-L/16
weights ``giswqs/geoai`` / ``dinov3_vitl16_sat493m.pth`` from Hugging Face
unless ``weights_path`` is given. This also happens when a fine-tuned
checkpoint is loaded for inference (with the ``weights_path`` recorded in the
checkpoint); the checkpoint's weights then replace them. SAT-493M weights
exist only for ViT-L/16 and ViT-7B/16, so fine-tuning other backbone sizes
needs ``engine_params["weights_path"]``. :meth:`DINOv3Engine.prefetch`
downloads the hub repository and the weights for nodes without internet
access.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import geopandas as gpd

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine
from agribound.registry import ENGINE_REGISTRY

logger = logging.getLogger(__name__)

#: Size aliases -> torch.hub entry points of facebookresearch/dinov3. Only
#: "large" (ViT-L/16) works without ``weights_path``: geoai's default weights
#: are the ViT-L/16 SAT-493M checkpoint.
DINOV3_MODELS = {
    "small": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "large": "dinov3_vitl16",
}

#: Backbone that geoai's default (SAT-493M) weights belong to.
DINOV3_DEFAULT_BACKBONE = "dinov3_vitl16"

#: Hugging Face location of geoai's default backbone weights.
DINOV3_DEFAULT_WEIGHTS = ("giswqs/geoai", "dinov3_vitl16_sat493m.pth")

#: Patch size of the DINOv3 ViT backbones (every ``dinov3_vit*16*`` entry point).
DINOV3_PATCH_SIZE = 16

#: geoai's default ``window_size`` for ``dinov3_segment_geotiff``.
_GEOAI_DEFAULT_WINDOW = 512


def _round_up(value: int, multiple: int) -> int:
    return -(-int(value) // multiple) * multiple


def _full_window_extent(size: int, window: int, stride: int) -> int:
    """Smallest extent >= *size* covered exactly by windows at multiples of *stride*.

    With geoai's window starts ``min(i * stride, extent - 1)`` and
    ``max(1, ceil((extent - overlap) / stride))`` windows, an extent of
    ``window + k * stride`` gives ``k + 1`` windows that all end inside it.
    """
    if size <= window:
        return window
    return window + _round_up(size - window, stride)


def plan_dinov3_windows(
    height: int,
    width: int,
    *,
    window_size: int | None = None,
    overlap: int | None = None,
    training_chip_size: int | None = None,
) -> dict[str, Any]:
    """Sliding-window settings for ``geoai.dinov3_finetune.dinov3_segment_geotiff``.

    - Window: ``window_size`` if given (must be a multiple of the 16 px
      patch size), else the training chip size rounded up to a multiple of
      16, else geoai's default 512. It is then capped at the larger raster
      side rounded up to a multiple of 16, so a small raster is not padded
      far beyond its size (DINOv3 does not resize its input, so a smaller
      window changes the attention context, not the object scale).
    - Overlap: ``overlap`` if given, else half the window; it must be at
      least 0 and smaller than the window. A capped window covers the whole
      raster in one window, so the overlap has no effect; an overlap that
      does not fit the capped window is then replaced by half of it.
    - Padded extent: the RGB input is extended by mirror reflection to
      ``window + k * stride`` rows and columns (the smallest such size at
      least the raster size), so every geoai window lies inside it and none
      is zero-padded.

    Returns
    -------
    dict
        ``window_size``, ``overlap``, ``padded_height``, ``padded_width``,
        ``window_source`` (``"engine_params"``, ``"training_chip_size"`` or
        ``"geoai_default"``), ``capped`` (bool) and ``training_chip_size``.

    Raises
    ------
    ValueError
        For a window that is not a positive multiple of 16, or an overlap
        outside ``[0, window)``.
    """
    patch = DINOV3_PATCH_SIZE
    if window_size is not None:
        window = int(window_size)
        source = "engine_params"
        if window <= 0 or window % patch:
            raise ValueError(
                f"DINOv3 window_size must be a positive multiple of the {patch} px patch size "
                f"(geoai zero-pads other windows), got {window}"
            )
    elif training_chip_size:
        window = _round_up(int(training_chip_size), patch)
        source = "training_chip_size"
    else:
        window = _GEOAI_DEFAULT_WINDOW
        source = "geoai_default"
    cap = _round_up(max(int(height), int(width)), patch)
    capped = window > cap
    if capped:
        window = cap
    ov = window // 2 if overlap is None else int(overlap)
    if capped and ov >= window:
        # One window covers the raster: geoai makes a single window for any overlap.
        ov = window // 2
    if not 0 <= ov < window:
        raise ValueError(f"DINOv3 overlap must be in [0, window_size={window}), got {ov}")
    stride = window - ov
    return {
        "window_size": window,
        "overlap": ov,
        "padded_height": _full_window_extent(int(height), window, stride),
        "padded_width": _full_window_extent(int(width), window, stride),
        "window_source": source,
        "capped": capped,
        "training_chip_size": int(training_chip_size) if training_chip_size else None,
    }


def _crop_raster(src_path: Path, dst_path: Path, height: int, width: int) -> None:
    """Copy the top-left ``height x width`` pixels of a single-band raster."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(src_path) as src:
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": src.dtypes[0],
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
            "compress": "lzw",
        }
        with rasterio.open(dst_path, "w", **profile) as dst:
            for row in range(0, height, 2048):
                window = Window(0, row, width, min(2048, height - row))
                dst.write(src.read(1, window=window), 1, window=window)


def resolve_dinov3_model(name: str | None, weights_path: str | None = None) -> str:
    """Return the torch.hub entry point for ``engine_params["dinov3_model"]``.

    Parameters
    ----------
    name : str or None
        ``"large"`` (default), ``"small"``, ``"base"`` or a hub entry point
        such as ``"dinov3_vitl16"``.
    weights_path : str or None
        Local backbone weights; required for every backbone except
        ViT-L/16.

    Raises
    ------
    ValueError
        For unknown names, or a non-ViT-L/16 backbone without
        *weights_path*.
    """
    text = str(name or "large").strip()
    hub = DINOV3_MODELS.get(text, text)
    if not hub.startswith("dinov3_"):
        raise ValueError(
            f"Unknown DINOv3 model {name!r}. Use 'large' (ViT-L/16, SAT-493M weights) or a "
            "facebookresearch/dinov3 hub entry point with engine_params['weights_path']"
        )
    if hub != DINOV3_DEFAULT_BACKBONE and not weights_path:
        raise ValueError(
            f"DINOv3 model {name!r} ({hub}) needs engine_params['weights_path']: geoai loads "
            f"the ViT-L/16 SAT-493M weights ({DINOV3_DEFAULT_WEIGHTS[0]}/"
            f"{DINOV3_DEFAULT_WEIGHTS[1]}) unless weights are given, which do not fit other "
            "backbone sizes (SAT-493M weights exist only for ViT-L/16 and ViT-7B/16)."
        )
    return hub


def resolve_lora(params: dict[str, Any]) -> tuple[bool, bool]:
    """Return ``(use_lora, freeze_backbone)`` for DINOv3 fine-tuning.

    - Default: ``(False, False)``, full fine-tuning of the backbone and
      decoder (for ViT-L/16 about 303 M backbone parameters are trained).
    - ``use_lora=True`` implies ``freeze_backbone=True``: in geoai 0.43.1
      ``freeze_backbone=False`` with LoRA would still train every backbone
      weight except the adapted ``qkv`` layers, so only the rank-``lora_rank``
      adapters (about 0.39 M parameters for ViT-L/16 at rank 4) and the
      decoder are trained.
    - ``freeze_backbone=True`` without LoRA trains the decoder only.

    Raises
    ------
    ValueError
        If ``use_lora=True`` is combined with an explicit
        ``freeze_backbone=False``.
    """
    use_lora = bool(params.get("use_lora", False))
    freeze = params.get("freeze_backbone")
    if use_lora:
        if freeze is not None and not bool(freeze):
            raise ValueError(
                "use_lora=True requires freeze_backbone=True (geoai would otherwise train "
                "the full backbone in addition to the LoRA adapters). Remove "
                "freeze_backbone or set use_lora=False for full fine-tuning."
            )
        return True, True
    return False, bool(freeze) if freeze is not None else False


class DINOv3Engine(DelineationEngine):
    """Field delineation with a fine-tuned DINOv3 + DPT model (see module docstring)."""

    name = "dinov3"
    supported_sources = list(ENGINE_REGISTRY["dinov3"]["supported_sources"])
    requires_bands = list(ENGINE_REGISTRY["dinov3"]["requires_bands"])

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run DINOv3 segmentation on a composite.

        Parameters
        ----------
        raster_path : str
            Composite GeoTIFF.
        config : AgriboundConfig
            Pipeline configuration. ``engine_params``:

            - ``checkpoint_path``: Lightning ``.ckpt`` written by
              ``train_dinov3_segmentation`` (required; set automatically by
              ``fine_tune=True``).
            - ``dinov3_model``: only used to check the checkpoint; the
              backbone recorded in the checkpoint is what runs.
            - ``weights_path``: not used for inference. geoai rebuilds the
              backbone from the ``weights_path`` recorded in the checkpoint
              and then loads the fine-tuned weights; a different value here
              only logs a warning.
            - ``window_size``: sliding window in pixels, a multiple of 16
              (default: the training chip size recorded next to the
              checkpoint, else 512; capped at the larger raster side, see
              :func:`plan_dinov3_windows`).
            - ``overlap``: window overlap in pixels (default: half the
              window).
            - ``batch_size``: windows per forward pass (default 4).
            - ``dilate_interior_px``: pixels by which each interior region is
              grown over the predicted boundary class (default: the training
              ``boundary_erosion`` recorded next to the checkpoint, else
              ``engine_params["boundary_erosion"]`` or 2).

        Returns
        -------
        geopandas.GeoDataFrame
            Field polygons with ``gdf.attrs["engine_meta"]``.

        Raises
        ------
        RuntimeError
            Without a checkpoint, or if geoai writes no output.
        ValueError
            If the checkpoint is not a Lightning ``.ckpt``.
        """
        try:
            from geoai.dinov3_finetune import dinov3_segment_geotiff
        except ImportError:
            raise ImportError(
                "geoai-py is required for the DINOv3 engine. "
                "Install with: pip install agribound[dinov3]"
            ) from None

        import rasterio

        from agribound._cache import cache_path
        from agribound.engines.base import get_canonical_band_indices
        from agribound.engines.finetune._data import (
            file_sha256,
            interior_polygons,
            mask_invalid_predictions,
            package_versions,
            raster_fingerprint,
            read_checkpoint_hparams,
            read_json,
            read_training_meta,
            write_json,
            write_rgb_input,
        )

        self.validate_input(raster_path, config)
        params = config.engine_params
        checkpoint = params.get("checkpoint_path")
        if not checkpoint:
            raise RuntimeError(
                "DINOv3 requires a fine-tuned checkpoint (no field-boundary weights are "
                "published). Set fine_tune=True with reference_boundaries, or provide "
                "engine_params={'checkpoint_path': '/path/to/dinov3.ckpt'}."
            )
        ckpt = Path(checkpoint).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {ckpt}")
        if ckpt.suffix != ".ckpt":
            raise ValueError(
                f"DINOv3 checkpoint {ckpt} is not a Lightning .ckpt. geoai loads other files "
                "as a plain state dict with strict=False, which can silently skip weights; "
                "use the .ckpt written by fine-tuning."
            )

        hparams = read_checkpoint_hparams(ckpt)
        model_name = hparams.get("model_name", DINOV3_DEFAULT_BACKBONE)
        requested = params.get("dinov3_model")
        if requested is not None:
            wanted = DINOV3_MODELS.get(str(requested), str(requested))
            if wanted != model_name:
                logger.warning(
                    "engine_params['dinov3_model']=%r differs from the checkpoint's backbone "
                    "%r; the checkpoint's backbone is used",
                    requested,
                    model_name,
                )
        # geoai rebuilds the model from the checkpoint's hyper-parameters
        # (DINOv3Segmenter.load_from_checkpoint): the backbone is initialised
        # from hparams["weights_path"] (the SAT-493M ViT-L/16 file when it is
        # None or missing), then the checkpoint's state dict replaces every
        # weight (strict loading).
        weights_path = hparams.get("weights_path")
        user_weights = params.get("weights_path")
        if user_weights and str(user_weights) != str(weights_path):
            logger.warning(
                "engine_params['weights_path']=%r is not used for inference: geoai rebuilds "
                "the backbone from the checkpoint's own weights_path (%r) before loading the "
                "fine-tuned weights",
                user_weights,
                weights_path,
            )
        if weights_path and not Path(weights_path).is_file():
            if model_name != DINOV3_DEFAULT_BACKBONE:
                raise FileNotFoundError(
                    f"The {model_name} checkpoint {ckpt} was trained from backbone weights "
                    f"{weights_path!r}, which do not exist on this machine. geoai would "
                    "initialise the backbone from the ViT-L/16 SAT-493M weights instead, which "
                    f"do not fit {model_name}. Copy the weights file to that path."
                )
            logger.warning(
                "Backbone weights %r recorded in %s do not exist here; geoai initialises the "
                "ViT-L/16 backbone from %s/%s instead. The fine-tuned checkpoint then replaces "
                "every weight, so the prediction is unchanged.",
                weights_path,
                ckpt.name,
                *DINOV3_DEFAULT_WEIGHTS,
            )
            weights_path = None
        batch_size = int(params.get("batch_size", 4))
        training = read_training_meta(ckpt)
        dilate_px = params.get("dilate_interior_px")
        if dilate_px is None:
            dilate_px = training.get("boundary_erosion", params.get("boundary_erosion", 2))
        dilate_px = int(dilate_px)
        with rasterio.open(raster_path) as src:
            height, width = src.height, src.width
        plan = plan_dinov3_windows(
            height,
            width,
            window_size=params.get("window_size"),
            overlap=params.get("overlap"),
            training_chip_size=training.get("chip_size"),
        )
        window_size, overlap = plan["window_size"], plan["overlap"]
        padded = (plan["padded_height"], plan["padded_width"])
        if plan["capped"]:
            logger.info(
                "DINOv3 window capped at %d px (raster %d x %d px)", window_size, height, width
            )
        if plan["training_chip_size"] and window_size != plan["training_chip_size"]:
            logger.info(
                "DINOv3 window %d px differs from the %d px training chips (the attention "
                "context differs; the object scale does not)",
                window_size,
                plan["training_chip_size"],
            )
        indices = get_canonical_band_indices(config.source, ["R", "G", "B"], bands=config.bands)
        device = config.resolve_device()
        raster_fp = raster_fingerprint(raster_path)

        rgb_path = cache_path(
            config,
            "dinov3_rgb",
            ".tif",
            raster_fp,
            f"bands={indices}",
            "unit",
            f"padded={padded[0]}x{padded[1]}:reflect",
        )
        rgb_info_path = rgb_path.with_suffix(".json")
        if rgb_path.exists() and rgb_info_path.exists():
            rgb_info = read_json(rgb_info_path)
        else:
            rgb_info = write_rgb_input(
                raster_path, rgb_path, indices, unit_float=True, pad_to=padded
            )
            write_json(rgb_info_path, rgb_info)

        seg_path = cache_path(
            config,
            "dinov3_segmentation",
            ".tif",
            raster_fp,
            raster_fingerprint(ckpt),
            f"bands={indices}",
            f"window={window_size}",
            f"overlap={overlap}",
            f"padded={padded[0]}x{padded[1]}:reflect",
            f"weights={weights_path}",
        )
        seg_info_path = seg_path.with_suffix(".json")
        run_info = {"device": device}
        if seg_path.exists() and seg_info_path.exists():
            logger.info("Using cached DINOv3 segmentation: %s", seg_path)
            run_info = {**read_json(seg_info_path), "cache_reused": True}
        else:
            logger.info(
                "Running DINOv3 segmentation (backbone=%s, checkpoint=%s, window=%d, "
                "overlap=%d, device=%s)",
                model_name,
                ckpt,
                window_size,
                overlap,
                device,
            )
            raw = seg_path.with_name(seg_path.stem + ".padded.partial.tif")
            tmp = seg_path.with_name(seg_path.stem + ".partial.tif")
            dinov3_segment_geotiff(
                input_path=str(rgb_path),
                output_path=str(raw),
                checkpoint_path=str(ckpt),
                model_name=model_name,
                weights_path=weights_path,
                num_classes=int(hparams.get("num_classes", 3)),
                window_size=window_size,
                overlap=overlap,
                batch_size=batch_size,
                device=device,
                quiet=True,
            )
            if not raw.exists():
                raise RuntimeError(f"DINOv3 segmentation failed: no output at {raw}")
            if padded != (height, width):
                _crop_raster(raw, tmp, height, width)
                raw.unlink()
            else:
                raw.replace(tmp)
            mask_invalid_predictions(tmp, raster_path, indices)
            tmp.replace(seg_path)
            write_json(seg_info_path, run_info)

        gdf = interior_polygons(seg_path, dilate_px, config.min_field_area_m2)
        gdf.attrs["engine_meta"] = {
            "backend": "geoai",
            **package_versions("geoai-py", "torch"),
            "model_name": model_name,
            # Pre-trained backbone weights the checkpoint was fine-tuned from.
            "weights": training.get("weights")
            or hparams.get("weights_path")
            or "/".join(DINOV3_DEFAULT_WEIGHTS),
            "weights_revision": training.get("weights_revision"),
            "weights_sha256": training.get("weights_sha256"),
            "use_lora": hparams.get("use_lora"),
            "freeze_backbone": hparams.get("freeze_backbone"),
            "lora_rank": hparams.get("lora_rank") if hparams.get("use_lora") else None,
            "trainable_params": training.get("trainable_params"),
            "checkpoint": str(ckpt.resolve()),
            "checkpoint_sha256": file_sha256(ckpt),
            "band_indices": indices,
            "input": rgb_info,
            "window_size": window_size,
            "overlap": overlap,
            "window_source": plan["window_source"],
            "window_capped": plan["capped"],
            "training_chip_size": plan["training_chip_size"],
            "input_padded_to": list(padded),
            "input_padding": "reflect" if padded != (height, width) else None,
            "batch_size": batch_size,
            "field_class": 1,
            "dilate_interior_px": dilate_px,
            "dinov3_location": os.environ.get("DINOV3_LOCATION", "facebookresearch/dinov3"),
            **run_info,
        }
        logger.info("DINOv3 delineated %d field boundaries", len(gdf))
        return gdf

    @classmethod
    def prefetch(cls, config: AgriboundConfig) -> list[str]:
        """Download what geoai needs to build the DINOv3 backbone offline.

        - The ``facebookresearch/dinov3`` torch.hub repository (skipped when
          ``DINOV3_LOCATION`` points to a local clone). On nodes without
          internet access set ``DINOV3_LOCATION`` to the returned directory.
        - The SAT-493M ViT-L/16 weights from Hugging Face (skipped when
          ``engine_params["weights_path"]`` is given); set
          ``HF_HUB_OFFLINE=1`` on offline nodes.

        Returns
        -------
        list[str]
            Local paths (hub repository directory, weights file).
        """
        import torch
        from huggingface_hub import hf_hub_download

        paths: list[str] = []
        location = os.environ.get("DINOV3_LOCATION")
        if location and Path(location).is_dir():
            paths.append(str(Path(location).resolve()))
        else:
            torch.hub.list("facebookresearch/dinov3", trust_repo=True, skip_validation=True)
            hub_dir = Path(torch.hub.get_dir())
            candidates = sorted(hub_dir.glob("facebookresearch_dinov3_*"))
            if not candidates:
                raise RuntimeError(f"torch.hub did not cache facebookresearch/dinov3 in {hub_dir}")
            main = hub_dir / "facebookresearch_dinov3_main"
            repo_dir = main if main in candidates else candidates[0]
            paths.append(str(repo_dir))
            logger.info("Cached facebookresearch/dinov3 in %s (use as DINOV3_LOCATION)", repo_dir)
        weights_path = config.engine_params.get("weights_path")
        if weights_path:
            paths.append(str(Path(weights_path).resolve()))
        else:
            paths.append(
                hf_hub_download(
                    repo_id=DINOV3_DEFAULT_WEIGHTS[0], filename=DINOV3_DEFAULT_WEIGHTS[1]
                )
            )
        return paths
