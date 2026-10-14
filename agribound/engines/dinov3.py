"""
DINOv3 semantic segmentation engine (geoai-py model, agribound sliding window).

Runs geoai's ``DINOv3Segmenter`` -- a DINOv3 ViT backbone (Siméoni et al.,
2025, arXiv:2508.10104) with a DPT decoder -- trained by agribound on
reference boundaries (``fine_tune=True``, see
``agribound.engines.finetune._dinov3``) into background / field interior /
field boundary classes. There are no published field-boundary weights, so a
fine-tuned Lightning ``.ckpt`` is required. Each field interior region is
grown back over the predicted boundary class by the boundary width used in
training (:func:`agribound.engines.finetune._data.interior_polygons`), so
neighbouring fields do not overlap. Models can predict a band wider than
the one they were trained on, which leaves the outlines inside the fields;
the opt-in ``engine_params["dilate_interior_px"] = "half_band"`` grows each
interior by half the predicted band instead (:func:`half_band_dilate_px`).

Input
-----
Canonical R, G, B bands with a scene-level 1-99 percentile stretch to uint8
(:func:`agribound.engines.finetune._data.write_rgb_input`), stored as
float32 ``uint8 / 255``. For ``source="local"`` without ``config.bands``
bands 1, 2, 3 are read as R, G, B. geoai divides a chip by 255 only when its
maximum exceeds 1, so the model receives exactly these [0, 1] values. The
model then normalises them with the mean and standard deviation recorded in
its checkpoint (:func:`dinov3_segmenter_class`): since agribound's recipe
version 2 (:data:`agribound.engines.finetune._dinov3.RECIPE_VERSION`), the
statistics the backbone was pre-trained with
(:data:`DINOV3_INPUT_NORMALIZATIONS`; for the SAT-493M weights mean (0.430,
0.411, 0.296) and standard deviation (0.213, 0.156, 0.143), from the
facebookresearch/dinov3 README). The normalisation is part of the model's
``forward``, so training, validation and inference apply it identically.
Checkpoints trained by agribound 1.0.1 or earlier, or by geoai, record none
and get none, as in their training.

Sliding window
--------------
agribound runs its own sliding window (:func:`segment_raster`); geoai's
``dinov3_segment_geotiff`` is not used. The window is the training chip size
by default (from ``<checkpoint>.agribound.json``, or else from the
checkpoint's hyper-parameters, which hold it from recipe version 2 on),
capped at the larger raster side (rounded up to the 16 px patch size), and
the RGB input is extended at the bottom and right by mirror
reflection so that every window lies inside it (:func:`plan_dinov3_windows`):
no window is zero-padded (the ViT attends over the whole window, so zero
padding would change the features of the real pixels). With the default
``window_stitching="blend"`` the class probabilities of overlapping windows
are averaged, each window weighted by a cosine taper that is 1 in its centre
and falls towards its edges over the overlap (:func:`window_taper`; the
weights of neighbouring windows sum to 1), so no window edge leaves a seam.
``"crop"`` keeps, for each pixel, the prediction of the window whose
central part it is in, the rule of geoai's ``dinov3_segment_geotiff`` used by
agribound 1.0.1 and earlier, which leaves straight class changes along the
middle of each overlap. The prediction is cropped back to the raster grid.

Processing grid
---------------
A checkpoint fine-tuned on a raster coarser than 10 m was trained on a
bilinear copy with 10 m pixels (``processing_gsd_m`` in
``<checkpoint>.agribound.json`` and in the checkpoint's hyper-parameters; see
:func:`agribound.engines.finetune._data.dinov3_processing_gsd`), and
inference reads the same grid (:func:`inference_processing_gsd`): the
window, the RGB input and the predicted classes are on the copy's grid, and
the polygons are in the raster's CRS. Other checkpoints read the raster as it
is, unless ``engine_params["processing_gsd_m"]`` says otherwise.

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

import functools
import logging
import math
import os
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig, effective_bands
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

#: Mean and standard deviation of the RGB input (in [0, 1]) that the DINOv3 backbones
#: were pre-trained with, from the facebookresearch/dinov3 README ("For models using the
#: SAT-493M weights (pretrained on satellite imagery), please use the following
#: transform"; the LVD-1689M web-image weights use the ImageNet values).
DINOV3_INPUT_NORMALIZATIONS: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "sat493m": ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),
    "lvd1689m": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}

#: How the predictions of overlapping windows are combined (:func:`segment_raster`).
WINDOW_STITCHING = ("blend", "crop")

#: ``engine_params["dilate_interior_px"]`` value that grows each interior by half the
#: thickness of the predicted boundary band (:func:`half_band_dilate_px`).
DILATE_HALF_BAND = "half_band"

#: Version of agribound's DINOv3 inference, part of the segmentation cache key
#: (1: geoai's ``dinov3_segment_geotiff``, agribound <= 1.0.1; 2: :func:`segment_raster`).
_INFERENCE_VERSION = 2

#: Names of the three classes, for the validation IoU of each class.
DINOV3_CLASS_NAMES = ("background", "interior", "boundary")


def resolve_input_normalization(
    value: Any = None, weights_path: str | None = None
) -> dict[str, Any] | None:
    """Input normalisation of a DINOv3 fine-tuning (``engine_params["input_normalization"]``).

    Parameters
    ----------
    value : str, dict or None
        ``None`` (default): ``"lvd1689m"`` when the file name of
        *weights_path* contains ``lvd1689m`` (Meta's web-image weights),
        otherwise ``"sat493m"`` (geoai's default weights and Meta's satellite
        weights). ``"sat493m"`` or ``"lvd1689m"``: the statistics of
        :data:`DINOV3_INPUT_NORMALIZATIONS`. ``"none"`` (or ``False``): no
        normalisation, the recipe of agribound 1.0.1 and earlier. A dict
        ``{"mean": [r, g, b], "std": [r, g, b]}``: these values (standard
        deviations > 0).
    weights_path : str or None
        Backbone weights of the fine-tuning (``engine_params["weights_path"]``).

    Returns
    -------
    dict or None
        ``{"name", "mean", "std"}`` (``name`` is ``"custom"`` for a dict), or
        None for no normalisation.

    Raises
    ------
    ValueError
        For other values.
    """
    if value is None:
        name = Path(str(weights_path or "")).name.lower()
        value = "lvd1689m" if "lvd1689m" in name else "sat493m"
    if value is False or (isinstance(value, str) and value.strip().lower() == "none"):
        return None
    if isinstance(value, str) and value.strip().lower() in DINOV3_INPUT_NORMALIZATIONS:
        key = value.strip().lower()
        mean, std = DINOV3_INPUT_NORMALIZATIONS[key]
        return {"name": key, "mean": list(mean), "std": list(std)}
    if isinstance(value, dict) and set(value) == {"mean", "std"}:
        try:
            mean = [float(v) for v in value["mean"]]
            std = [float(v) for v in value["std"]]
        except (TypeError, ValueError):
            mean = std = []
        finite = all(math.isfinite(v) for v in mean + std)
        if len(mean) == 3 and len(std) == 3 and finite and min(std) > 0:
            return {"name": "custom", "mean": mean, "std": std}
    raise ValueError(
        "engine_params['input_normalization'] must be 'sat493m', 'lvd1689m', 'none' or "
        f"{{'mean': [r, g, b], 'std': [r, g, b]}} with standard deviations > 0, got {value!r}"
    )


@functools.cache
def dinov3_segmenter_class() -> type:
    """geoai's ``DINOv3Segmenter`` with input normalisation and a per-epoch validation IoU.

    The subclass (``AgriboundDINOv3Segmenter``) takes five more arguments,
    which are saved with the other hyper-parameters in its checkpoints:

    - ``input_mean``, ``input_std``: per-band statistics; ``forward``
      normalises its input with them (``(x - mean) / std``) before the
      backbone. ``None`` (the default, and what every checkpoint without them
      loads with) leaves the input unchanged, as geoai does.
    - ``boundary_erosion``: the boundary width of the training masks, read at
      inference as the default ``dilate_interior_px``.
    - ``chip_size``: the side of the training chips in pixels, read at
      inference as the default window when the checkpoint has no
      ``<checkpoint>.agribound.json`` next to it (which records it too).
    - ``processing_gsd_m``: the pixel size in metres of the processing grid
      the training chips were cut on (None: the raster's own grid), read at
      inference when that file is missing (:func:`inference_processing_gsd`).

    Validation logs ``val_loss`` and, over all validation chips of the epoch,
    the IoU of each class (``val_iou_background``, ``val_iou_interior``,
    ``val_iou_boundary``; pixels with the ignore index excluded) and their
    mean ``val_miou`` (classes absent from both the masks and the prediction
    left out). geoai's ``val_iou`` is a mean of per-batch values instead.

    The statistics are held in buffers that are not saved, so the state dict
    is that of geoai's class, and geoai's own ``DINOv3Segmenter`` can still
    load such a checkpoint (Lightning drops the extra hyper-parameters), but
    then without the normalisation.
    """
    import torch
    from geoai.dinov3_finetune import DINOv3Segmenter

    class AgriboundDINOv3Segmenter(DINOv3Segmenter):
        def __init__(
            self,
            input_mean: list[float] | None = None,
            input_std: list[float] | None = None,
            boundary_erosion: int | None = None,
            chip_size: int | None = None,
            processing_gsd_m: float | None = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            if (input_mean is None) != (input_std is None):
                raise ValueError("input_mean and input_std must both be given or both be None")
            self.normalizes_input = input_mean is not None
            channels = int(self.hparams.get("num_channels", 3))
            shape = (1, channels, 1, 1)
            mean = torch.zeros(shape) if input_mean is None else torch.tensor(input_mean)
            std = torch.ones(shape) if input_std is None else torch.tensor(input_std)
            self.register_buffer("input_mean", mean.float().reshape(shape), persistent=False)
            self.register_buffer("input_std", std.float().reshape(shape), persistent=False)
            self._val_counts: torch.Tensor | None = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.normalizes_input:
                x = (x - self.input_mean) / self.input_std
            return super().forward(x)

        def on_validation_epoch_start(self) -> None:
            k = int(self.hparams.num_classes)
            self._val_counts = torch.zeros(2, k, dtype=torch.long)

        def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
            x, y = self._unpack_batch(batch)
            logits = self(x)
            loss = self.loss_fn(logits, y)
            self.log("val_loss", loss, on_epoch=True, prog_bar=True)
            counts = class_overlap_counts(
                logits.argmax(dim=1), y, int(self.hparams.num_classes), self.hparams.ignore_index
            )
            if self._val_counts is None:
                self.on_validation_epoch_start()
            self._val_counts += counts.cpu()
            return loss

        def on_validation_epoch_end(self) -> None:
            if self._val_counts is None:
                return
            ious = iou_from_counts(self._val_counts)
            valid = [v for v in ious if not math.isnan(v)]
            self.log("val_miou", float(np.mean(valid)) if valid else 0.0, prog_bar=True)
            names = DINOV3_CLASS_NAMES if len(ious) == 3 else [str(c) for c in range(len(ious))]
            for name, value in zip(names, ious, strict=True):
                self.log(f"val_iou_{name}", value)

    return AgriboundDINOv3Segmenter


def class_overlap_counts(pred: Any, target: Any, num_classes: int, ignore_index: int) -> Any:
    """Per-class intersection and union pixel counts of a predicted and a target class map.

    Parameters
    ----------
    pred, target : torch.Tensor
        Integer class maps of the same shape; *target* pixels equal to
        *ignore_index* are excluded.
    num_classes : int
        Number of classes.
    ignore_index : int
        Target value of pixels without a label.

    Returns
    -------
    torch.Tensor
        ``(2, num_classes)`` int64: intersections, then unions.
    """
    import torch

    valid = target != ignore_index
    rows = []
    for c in range(num_classes):
        p, t = (pred == c) & valid, (target == c) & valid
        rows.append(torch.stack([(p & t).sum(), (p | t).sum()]))
    return torch.stack(rows, dim=1).long()


def iou_from_counts(counts: Any) -> list[float]:
    """IoU of each class from ``(2, k)`` intersection and union counts (NaN for a union of 0)."""
    inter, union = (np.asarray(c, dtype=np.float64) for c in counts.tolist())
    return [float(i / u) if u > 0 else float("nan") for i, u in zip(inter, union, strict=True)]


def window_taper(window: int, overlap: int) -> np.ndarray:
    """Weights of the pixels of a window along one axis, for ``window_stitching="blend"``.

    1 in the centre; over the first *overlap* pixels the weight rises from
    near 0 to near 1 (``sin²``), and over the last *overlap* pixels it falls
    back (a tapered cosine). Neighbouring windows, ``window - overlap``
    pixels apart, share *overlap* pixels, and over them the falling side of
    one window and the rising side of the next sum to 1 (for ``overlap`` up
    to half the window), so averaging with these weights is a partition of
    unity. For ``overlap == window / 2`` this is the Hann window
    ``sin²(π (i + 0.5) / window)``. With ``overlap`` 0 every weight is 1.
    The weights are never 0, so a pixel covered by one window (at the raster
    edges) takes that window's prediction.

    Parameters
    ----------
    window : int
        Window size in pixels.
    overlap : int
        Overlap of neighbouring windows in pixels, in ``[0, window)``.

    Returns
    -------
    numpy.ndarray
        float32 weights of length *window*.
    """
    weights = np.ones(int(window), dtype=np.float64)
    overlap = int(overlap)
    if overlap > 0:
        ramp = np.sin(0.5 * np.pi * (np.arange(overlap) + 0.5) / overlap) ** 2
        weights[:overlap] = np.minimum(weights[:overlap], ramp)
        weights[window - overlap :] = np.minimum(weights[window - overlap :], ramp[::-1])
    return weights.astype(np.float32)


def _crop_weights(index: int, n: int, window: int, overlap: int) -> np.ndarray:
    """0/1 weights of window *index* of *n* under geoai's centre-crop rule (one axis)."""
    weights = np.zeros(int(window), dtype=np.float32)
    start = 0 if index == 0 else overlap // 2
    stop = window if index == n - 1 else window - overlap + overlap // 2
    weights[start:stop] = 1.0
    return weights


def segment_raster(
    model: Any,
    rgb_path: str | Path,
    out_path: str | Path,
    *,
    height: int,
    width: int,
    window_size: int,
    overlap: int,
    batch_size: int = 4,
    device: str = "cpu",
    stitching: str = "blend",
) -> dict[str, Any]:
    """Classify a raster with a DINOv3 model in overlapping windows.

    Windows of *window_size* pixels start every ``window_size - overlap``
    pixels from the top-left corner of the RGB input, which must be the
    padded input of :func:`plan_dinov3_windows` (``window + k * stride``
    rows and columns), so every window lies inside it. The model's softmax
    probabilities are combined per pixel as a weighted average over the
    windows that cover it (``stitching="blend"``: :func:`window_taper` along
    both axes; ``"crop"``: weight 1 in the window's central part up to the
    middle of its overlaps, as in geoai's ``dinov3_segment_geotiff``), and
    the class with the highest averaged probability is written for the top-left
    ``height x width`` pixels. Rows are written as soon as no later window
    covers them, so memory grows with the raster width, not its height.

    Parameters
    ----------
    model : torch.nn.Module
        Model in evaluation mode on *device*; ``model(x)`` returns logits
        ``(batch, classes, window, window)`` for float input
        ``(batch, bands, window, window)``.
    rgb_path : str or Path
        Model input (float32, the padded RGB input).
    out_path : str or Path
        Output GeoTIFF: uint8 classes on the grid of *rgb_path*, cropped to
        *height* x *width*.
    height, width : int
        Size of the raster the input was made from.
    window_size, overlap : int
        Window and overlap in pixels.
    batch_size : int
        Windows per forward pass.
    device : str
        Torch device of the model.
    stitching : str
        ``"blend"`` or ``"crop"``.

    Returns
    -------
    dict
        ``n_windows``, ``stitching``.

    Raises
    ------
    ValueError
        For an unknown *stitching*, or an input whose size is not covered
        exactly by the windows.
    """
    import rasterio
    import torch
    from rasterio.windows import Window

    if stitching not in WINDOW_STITCHING:
        raise ValueError(f"window_stitching must be one of {WINDOW_STITCHING}, got {stitching!r}")
    window, overlap = int(window_size), int(overlap)
    stride = window - overlap
    dev = torch.device(device)
    with rasterio.open(rgb_path) as src:
        pad_h, pad_w = src.height, src.width
        if (
            (pad_h - window) % stride
            or (pad_w - window) % stride
            or pad_h < window
            or pad_w < window
        ):
            raise ValueError(
                f"The input ({pad_h} x {pad_w} px) is not covered exactly by {window} px windows "
                f"every {stride} px; pad it with plan_dinov3_windows"
            )
        rows = list(range(0, pad_h - window + 1, stride))
        cols = list(range(0, pad_w - window + 1, stride))
        if stitching == "blend":
            taper = window_taper(window, overlap)
            row_w = [taper] * len(rows)
            col_w = [taper] * len(cols)
        else:
            row_w = [_crop_weights(i, len(rows), window, overlap) for i in range(len(rows))]
            col_w = [_crop_weights(j, len(cols), window, overlap) for j in range(len(cols))]
        profile = {
            "driver": "GTiff",
            "height": int(height),
            "width": int(width),
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "compress": "lzw",
        }
        n_classes = None
        acc = wsum = None
        n_windows = 0
        with rasterio.open(out_path, "w", **profile) as dst:
            for i, top in enumerate(rows):
                strip = src.read(window=Window(0, top, pad_w, window)).astype(np.float32)
                for k in range(0, len(cols), max(int(batch_size), 1)):
                    batch_cols = list(range(k, min(k + max(int(batch_size), 1), len(cols))))
                    x = np.stack([strip[:, :, cols[j] : cols[j] + window] for j in batch_cols])
                    with torch.inference_mode():
                        logits = model(torch.from_numpy(x).to(dev))
                        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                    if acc is None:
                        n_classes = probs.shape[1]
                        acc = np.zeros((n_classes, window, pad_w), dtype=np.float32)
                        wsum = np.zeros((window, pad_w), dtype=np.float32)
                    for p, j in zip(probs, batch_cols, strict=True):
                        weight = np.outer(row_w[i], col_w[j])
                        acc[:, :, cols[j] : cols[j] + window] += p * weight
                        wsum[:, cols[j] : cols[j] + window] += weight
                    n_windows += len(batch_cols)
                # Rows above the next window are final.
                done = stride if i < len(rows) - 1 else window
                n_out = max(0, min(done, int(height) - top))
                if n_out:
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean = acc[:, :n_out, : int(width)] / wsum[None, :n_out, : int(width)]
                    classes = np.nan_to_num(mean, nan=-1.0).argmax(axis=0).astype(np.uint8)
                    dst.write(classes, 1, window=Window(0, top, int(width), n_out))
                acc = np.concatenate(
                    [acc[:, stride:], np.zeros((n_classes, stride, pad_w), np.float32)], axis=1
                )
                wsum = np.concatenate(
                    [wsum[stride:], np.zeros((stride, pad_w), np.float32)], axis=0
                )
    return {"n_windows": n_windows, "stitching": stitching}


def _round_up(value: int, multiple: int) -> int:
    return -(-int(value) // multiple) * multiple


def _full_window_extent(size: int, window: int, stride: int) -> int:
    """Smallest extent >= *size* covered exactly by windows at multiples of *stride*.

    An extent of ``window + k * stride`` holds ``k + 1`` windows starting at
    ``0, stride, ..., k * stride`` that all end inside it (also with geoai's
    window starts ``min(i * stride, extent - 1)`` and
    ``max(1, ceil((extent - overlap) / stride))`` windows).
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
    """Sliding-window settings of :func:`segment_raster`.

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
      least the raster size), so every window lies inside it and none is
      zero-padded.

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


def resolve_window_stitching(value: Any = None) -> str:
    """``engine_params["window_stitching"]``: ``"blend"`` (default) or ``"crop"``.

    Raises
    ------
    ValueError
        For other values.
    """
    text = "blend" if value is None else str(value).strip().lower()
    if text not in WINDOW_STITCHING:
        raise ValueError(
            f"engine_params['window_stitching'] must be one of {WINDOW_STITCHING}, got {value!r}"
        )
    return text


def resolve_dilate_rule(value: Any = None) -> str:
    """How ``engine_params["dilate_interior_px"]`` sets the growth: ``"fixed"`` or ``"half_band"``.

    ``"half_band"`` (any case, surrounding spaces ignored) selects
    :func:`half_band_dilate_px`; None, a number of pixels or a string of an
    integer selects ``"fixed"`` (:func:`resolve_dilate_px`).

    Raises
    ------
    ValueError
        For any other string.
    """
    if not isinstance(value, str):
        return "fixed"
    text = value.strip().lower()
    if text == DILATE_HALF_BAND:
        return DILATE_HALF_BAND
    try:
        int(text)
    except ValueError:
        raise ValueError(
            "engine_params['dilate_interior_px'] must be a number of pixels or "
            f"{DILATE_HALF_BAND!r}, got {value!r}"
        ) from None
    return "fixed"


def resolve_dilate_px(params: dict[str, Any], training: dict[str, Any], hparams: dict) -> int:
    """Default growth of the interiors over the boundary class (``dilate_interior_px``).

    ``engine_params["dilate_interior_px"]`` if set to a number; else the
    training ``boundary_erosion`` recorded next to the checkpoint
    (``<checkpoint>.agribound.json``), else the one in the checkpoint's
    hyper-parameters (agribound's recipe version 2 and later), else
    ``engine_params["boundary_erosion"]``, else 2 (the boundary width of
    agribound 1.0.1 and earlier). With ``dilate_interior_px="half_band"``
    the growth is measured on the prediction (:func:`half_band_dilate_px`)
    and this value (the trained width) is used only where it cannot be.
    """
    requested = params.get("dilate_interior_px")
    if resolve_dilate_rule(requested) == DILATE_HALF_BAND:
        requested = None
    for value in (
        requested,
        training.get("boundary_erosion"),
        hparams.get("boundary_erosion"),
        params.get("boundary_erosion"),
    ):
        if value is not None:
            return int(value)
    return 2


def half_band_dilate_px(pred: np.ndarray, fallback_px: int) -> dict[str, Any]:
    """Growth of half the predicted boundary band (``dilate_interior_px="half_band"``).

    ``g = ceil(T / 2)``, with T the median thickness of the predicted
    boundary band where a field meets background
    (:func:`agribound.engines.finetune._data.predicted_band_thickness`).
    Where a field meets background the outline then lies in the middle of
    the band, which is the field edge when the model draws the band across
    the edge rather than inside the field; where two fields meet, both
    interiors grow into the band between them until they meet or have grown
    g px. A model that predicts its trained band of k px, inside the field,
    gets ``ceil(k / 2)``, less than the default growth k. Without an
    interior pixel or a boundary pixel next to background, T is undefined
    and *fallback_px* is used. The rule (B-half) passed a pre-registered
    pilot on development squares (2026-10-12): boundary F1 rose on 8 of 9
    held-out runs and F1 changed by -0.003 to +0.033; the outlines
    overshoot where the boundary class covers the ground between fields.

    Parameters
    ----------
    pred : numpy.ndarray
        ``(H, W)`` class raster (0 background, 1 interior, 2 boundary), as
        written by inference (invalid pixels 0).
    fallback_px : int
        Growth used when T is undefined: the trained width
        (:func:`resolve_dilate_px`).

    Returns
    -------
    dict
        ``dilate_interior_px`` (the growth), ``band_thickness_px`` (T or
        None), ``band_edge_px`` (the boundary pixels next to background
        that T is the median over) and ``fallback`` (True when
        *fallback_px* was used).
    """
    from agribound.engines.finetune._data import predicted_band_thickness

    thickness, n_edge = predicted_band_thickness(pred)
    if thickness is None:
        growth, fallback = int(fallback_px), True
    else:
        growth, fallback = int(math.ceil(thickness / 2)), False
    return {
        "dilate_interior_px": growth,
        "band_thickness_px": thickness,
        "band_edge_px": n_edge,
        "fallback": fallback,
    }


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


def inference_processing_gsd(
    raster_path: str,
    params: dict[str, Any],
    training: dict[str, Any],
    hparams: dict[str, Any] | None = None,
) -> float | None:
    """Pixel size of the grid DINOv3 inference reads *raster_path* on (None: its own).

    - ``engine_params["processing_gsd_m"]``, when set (not None):
      :func:`agribound.engines.finetune._data.dinov3_processing_gsd`;
    - else the grid the checkpoint was trained on, when the raster is coarser
      than it: ``processing_gsd_m`` of ``<checkpoint>.agribound.json``
      (*training*) or, without it, of the checkpoint's hyper-parameters
      (*hparams*; a checkpoint copied or renamed without that file);
    - else None: checkpoints trained on their raster's own grid (every
      checkpoint of agribound 1.0.1, and later ones of rasters of 10 m or
      finer) and checkpoints without the record read the raster as it is.
    """
    from agribound.engines.finetune._data import dinov3_processing_gsd

    if params.get("processing_gsd_m") is not None:
        return dinov3_processing_gsd(raster_path, params)
    trained = training.get("processing_gsd_m")
    if trained is None:
        trained = (hparams or {}).get("processing_gsd_m")
    if trained is None:
        return None
    try:
        return dinov3_processing_gsd(raster_path, {"processing_gsd_m": float(trained)})
    except ValueError:  # a raster finer than the training grid is read as it is
        return None


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

            - ``checkpoint_path``: Lightning ``.ckpt`` written by DINOv3
              fine-tuning (required; set automatically by ``fine_tune=True``).
              Its input normalisation (see the module docstring) is applied.
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
            - ``window_stitching``: ``"blend"`` (default) or ``"crop"``
              (agribound 1.0.1), see :func:`segment_raster`.
            - ``batch_size``: windows per forward pass (default 4).
            - ``dilate_interior_px``: pixels by which each interior region is
              grown over the predicted boundary class (default: the training
              boundary width, see :func:`resolve_dilate_px`), or
              ``"half_band"``: half the thickness of the predicted band,
              measured on the prediction (:func:`half_band_dilate_px`; opt-in).
            - ``processing_gsd_m``: the grid the raster is read on (default:
              the checkpoint's training grid, see
              :func:`inference_processing_gsd`).

        Returns
        -------
        geopandas.GeoDataFrame
            Field polygons with ``gdf.attrs["engine_meta"]``.

        Raises
        ------
        RuntimeError
            Without a checkpoint.
        ValueError
            If the checkpoint is not a Lightning ``.ckpt``, or for an unknown
            ``window_stitching``.
        """
        try:
            segmenter_cls = dinov3_segmenter_class()
        except ImportError:
            raise ImportError(
                "geoai-py is required for the DINOv3 engine. "
                "Install with: pip install agribound[dinov3]"
            ) from None

        import rasterio

        from agribound._cache import cache_path
        from agribound.engines.base import get_canonical_band_indices
        from agribound.engines.finetune._data import (
            dinov3_processing_raster,
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

        stitching = resolve_window_stitching(params.get("window_stitching"))
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
        dilate_rule = resolve_dilate_rule(params.get("dilate_interior_px"))
        dilate_px = resolve_dilate_px(params, training, hparams)
        normalization = None
        if hparams.get("input_mean") is not None:
            normalization = {
                "mean": list(hparams["input_mean"]),
                "std": list(hparams.get("input_std") or []),
            }
        indices = get_canonical_band_indices(
            config.source, ["R", "G", "B"], bands=effective_bands(config)
        )
        # The raster itself, or the copy on the checkpoint's processing grid.
        read_path, read_indices, processing = dinov3_processing_raster(
            raster_path,
            config,
            inference_processing_gsd(raster_path, params, training, hparams),
            indices,
        )
        with rasterio.open(read_path) as src:
            height, width = src.height, src.width
        plan = plan_dinov3_windows(
            height,
            width,
            window_size=params.get("window_size"),
            overlap=params.get("overlap"),
            # The checkpoint's own record (recipe 2) when the metadata file is missing.
            training_chip_size=training.get("chip_size") or hparams.get("chip_size"),
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
        device = config.resolve_device()
        raster_fp = raster_fingerprint(read_path)

        rgb_path = cache_path(
            config,
            "dinov3_rgb",
            ".tif",
            raster_fp,
            f"bands={read_indices}",
            "unit",
            f"padded={padded[0]}x{padded[1]}:reflect",
        )
        rgb_info_path = rgb_path.with_suffix(".json")
        if rgb_path.exists() and rgb_info_path.exists():
            rgb_info = read_json(rgb_info_path)
        else:
            rgb_info = write_rgb_input(
                read_path, rgb_path, read_indices, unit_float=True, pad_to=padded
            )
            write_json(rgb_info_path, rgb_info)

        seg_path = cache_path(
            config,
            "dinov3_segmentation",
            ".tif",
            raster_fp,
            raster_fingerprint(ckpt),
            f"bands={read_indices}",
            f"window={window_size}",
            f"overlap={overlap}",
            f"padded={padded[0]}x{padded[1]}:reflect",
            f"weights={weights_path}",
            f"stitching={stitching}",
            f"inference={_INFERENCE_VERSION}",
        )
        seg_info_path = seg_path.with_suffix(".json")
        run_info: dict[str, Any] = {"device": device}
        if seg_path.exists() and seg_info_path.exists():
            logger.info("Using cached DINOv3 segmentation: %s", seg_path)
            run_info = {**read_json(seg_info_path), "cache_reused": True}
        else:
            logger.info(
                "Running DINOv3 segmentation (backbone=%s, checkpoint=%s, window=%d, "
                "overlap=%d, stitching=%s, device=%s)",
                model_name,
                ckpt,
                window_size,
                overlap,
                stitching,
                device,
            )
            # The checkpoint's hyper-parameters rebuild the model (geoai initialises the
            # backbone from weights_path, the SAT-493M ViT-L/16 file when it is None); then
            # its state dict replaces every weight (strict loading). The full checkpoint
            # (optimizer state included) is loaded on the CPU.
            model = segmenter_cls.load_from_checkpoint(
                str(ckpt), map_location="cpu", weights_path=weights_path
            )
            model = model.eval().to(device)
            tmp = seg_path.with_name(seg_path.stem + ".partial.tif")
            try:
                stats = segment_raster(
                    model,
                    rgb_path,
                    tmp,
                    height=height,
                    width=width,
                    window_size=window_size,
                    overlap=overlap,
                    batch_size=batch_size,
                    device=device,
                    stitching=stitching,
                )
                mask_invalid_predictions(tmp, read_path, read_indices)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            finally:
                del model
            tmp.replace(seg_path)
            run_info["n_windows"] = stats["n_windows"]
            write_json(seg_info_path, run_info)

        half_band = None
        if dilate_rule == DILATE_HALF_BAND:
            with rasterio.open(seg_path) as src:
                half_band = half_band_dilate_px(src.read(1), fallback_px=dilate_px)
            half_band["trained_dilate_px"] = dilate_px
            dilate_px = half_band["dilate_interior_px"]
            logger.info(
                "DINOv3 growth 'half_band': predicted band %s px at %d edge pixels -> %d px "
                "(trained growth %d px)",
                half_band["band_thickness_px"],
                half_band["band_edge_px"],
                dilate_px,
                half_band["trained_dilate_px"],
            )
        gdf = interior_polygons(seg_path, dilate_px, config.min_field_area_m2)
        gdf.attrs["engine_meta"] = {
            "backend": "geoai",
            "inference": "agribound sliding window",
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
            # The bilinear copy read instead of the raster (processing grid), if any.
            **({"processing": processing} if processing is not None else {}),
            "input": rgb_info,
            # Applied inside the model (recorded in the checkpoint); None for checkpoints of
            # agribound 1.0.1 and earlier.
            "input_normalization": normalization,
            "window_size": window_size,
            "overlap": overlap,
            "window_stitching": stitching,
            "window_source": plan["window_source"],
            "window_capped": plan["capped"],
            "training_chip_size": plan["training_chip_size"],
            "input_padded_to": list(padded),
            "input_padding": "reflect" if padded != (height, width) else None,
            "batch_size": batch_size,
            "field_class": 1,
            "dilate_interior_px": dilate_px,
            # Only with dilate_interior_px="half_band": the measured band and the rule.
            **(
                {
                    "dilate_interior_rule": DILATE_HALF_BAND,
                    "band_thickness_px": half_band["band_thickness_px"],
                    "band_edge_px": half_band["band_edge_px"],
                    "dilate_fallback": half_band["fallback"],
                    "trained_dilate_px": half_band["trained_dilate_px"],
                }
                if half_band is not None
                else {}
            ),
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
