"""
Box-prompted SAM refinement of field boundaries.

This is a post-processing stage, not a delineation engine: every polygon's
bounding box is given to a Segment Anything model as a single-instance box
prompt, and the polygon is replaced by the mask SAM returns, unless that mask
covers too little of it (step 7). The pipeline runs it after delineation when
``config.sam_refine`` is *True* (the embedding engine calls it itself).

Algorithm (:func:`refine_boundaries`)
-------------------------------------
1. Polygons are reprojected to the raster CRS. The raster must not be
   rotated or sheared.
2. **Gating.** Missing or empty geometries are skipped. So is every polygon
   whose bounding box extends more than half a pixel
   (:data:`EDGE_TOLERANCE_PX`) beyond the raster (counted in
   ``n_skipped_outside``): SAM would see only part of such a field, and a
   mask of the visible part would truncate it. Polygons delineated from the
   raster itself normally lie inside it. Of the remaining polygons, one is refined
   only if both sides of its padded bounding box are at least
   ``config.sam_min_crop_px`` pixels, where the padded side is
   ``floor(side_px * (1 + 2 * config.sam_crop_padding))`` and ``side_px`` is
   the bounding-box width (height) divided by the pixel width (height).
   :func:`crop_window_px` and :func:`is_refinable` implement exactly this
   test (pass ``raster_bounds`` to :func:`is_refinable` to include the
   inside-the-raster test). Skipped polygons keep their geometry. Boxes
   within the half-pixel tolerance, and padded boxes, are clipped to the
   raster.
3. **Image.** The canonical R, G, B bands of ``config.source`` (with
   ``config.bands`` taking precedence, or ``engine_params["sam_rgb_bands"]``)
   are converted to uint8 with one scene-wide percentile stretch: the 1st and
   99th percentiles of each band are computed by
   :func:`agribound.io.raster.percentile_stretch_uint8` (valid, finite,
   positive pixels) on a nearest-neighbour decimated read of at most 4096
   pixels per side, and every window is stretched with those same bounds.
   uint8 rasters are used as-is. For embedding rasters (signed values) the
   percentiles are taken over all finite values.
4. **Windows.** Polygons whose padded box (clipped to the raster) fits in
   ``window_px - window_px // 2`` pixels on both axes are assigned to a grid
   of ``window_px`` x ``window_px`` windows with a stride of
   ``window_px // 2`` (windows at the raster edge are shifted inwards to
   keep their full size); each window is encoded once and all its boxes are
   decoded in batches of ``batch_size``. Larger polygons get their own
   square window, with the side of their padded box's longer axis, centred
   on the box. A field window whose side exceeds
   ``max_window_px = max(2 * window_px, 2048)`` is read with
   nearest-neighbour decimation so that its longer side is
   ``max_window_px``, which bounds memory (SAM downsamples it to its input
   size in any case). SAM resizes every window to a fixed square input
   without keeping the aspect ratio (1024 x 1024 px for SAM 2/2.1, 1008 x
   1008 px for SAM 3), so windows that are not square (the raster is
   smaller than the window on one axis, or a field window is clipped by the
   raster) are padded with black (0) pixels on the right and bottom to a
   square first; box coordinates are unaffected. A full grid window is
   encoded at about its native scale. This differs from agribound 0.1.x,
   which encoded each field's padded crop on its own, so that SAM upsampled
   a 64 px crop 16-fold; a smaller ``engine_params["sam_window_px"]`` (at
   least ``2 * sam_min_crop_px``) restores part of that zoom at the cost of
   more encoder passes.
5. **Masks.** One mask per box (``multimask_output=False``). As in 0.1.x, where
   the image was the field's padded crop, only the part of the mask inside
   the field's padded box (rounded outwards to whole pixels and clipped to
   the raster) is used. It is vectorised with
   :func:`rasterio.features.shapes`, the largest polygon is kept (holes
   included), repaired if invalid, and reprojected to the input CRS. Boxes
   are passed as continuous pixel coordinates (not truncated). Steps 6 and 7
   decide whether the mask replaces the input polygon.
6. **Overlaps** (``engine_params["sam_overlaps"]``, default ``"trim"``). A
   mask may grow over a neighbouring polygon, and both would be kept. With
   ``"trim"`` a refined polygon never takes area that another input polygon
   covered and it did not (:func:`trim_refinement_overlaps`), and where two
   refined masks grew over the same new area, the one with the higher SAM
   score keeps it. The overlap between the refined polygons and the others is
   therefore never larger than between the input polygons, so SAM adds no
   overlap to an engine output without overlaps (Delineate-Anything resolves
   them). The trimmed mask keeps its largest part (step 7 then tests it); a
   mask with nothing left keeps the input geometry and counts as failed.
   ``"keep"`` keeps the masks as SAM drew them (with ``sam_min_coverage=0``,
   the behaviour before 1.0.0), so outputs may overlap.
   Trade-off, measured on 2026-09-28 with Delineate-Anything (``large_v2``) +
   SAM 2 (``sam2-hiera-large``, MPS) on the Namoi test area (Sentinel-2,
   2023; 16 of 230 polygons refined): the post-processed output had 3.59 ha
   of overlap with ``"keep"`` (3.45 ha between a refined polygon and a
   neighbour) and 0.20 ha with ``"trim"``, against 0.18 ha without SAM. With
   ``"keep"`` one refined mask grew over the neighbours of one of the four
   reference fields in the area, raising that field's best IoU from 0.636
   (no SAM) to 0.773; with ``"trim"`` it is 0.676 (the other three fields:
   unchanged or within 0.02). SAM's growth over an engine boundary can be a
   correction (a field split in two) or a leak into a real neighbour; the
   default keeps the engine's boundaries between polygons.
7. **Coverage** (``engine_params["sam_min_coverage"]``, default
   :data:`DEFAULT_MIN_COVERAGE` = 0.5). SAM returns one object per box, so
   when an input polygon holds several fields (an embedding cluster, or
   fields an engine merged) the mask can follow one of them, and the rest of
   the polygon's area would be left without a polygon. A mask that covers
   less than ``sam_min_coverage`` of its input polygon, ``area(mask & input)
   / area(input)`` after step 6 (an invalid input is repaired first), is not
   used: the polygon keeps its input geometry, ``agribound:sam_refined`` is
   False and ``agribound:sam_score`` NaN, and ``n_low_coverage`` counts it.
   With ``"trim"`` each mask is tested right after its trim, in step 6's
   score order, so a rejected mask claims no area from the lower-scoring
   masks. An input without area is never rejected; 0 turns the test off (the
   1.0.0 behaviour). Measured on 2026-09-29 (SAM 2 ``sam2-hiera-large``,
   MPS, outputs after the area filter, smoothing and simplification), no
   test -> 0.5: example 15's Sentinel-2 refinement of the TESSERA (Google)
   crop polygons left 8 -> 1 (14 -> 2) of 29 checked centre pivots less than
   half covered and lost 15.7 -> 7.4 % (24.1 -> 8.1 %) of the input area
   (48 of 510, 95 of 439 masks rejected); example 13's input
   (Delineate-Anything, Sentinel-2, 67 masks, each covering >= 92 % of its
   polygon) did not change. On example 14's DINOv3 NAIP (SPOT) polygons of
   Lea County the reference fields less than half covered went from 35 (67)
   without SAM to 80 (105) with SAM and 53 (85) with 0.5, and F1 from 0.604
   (0.423) to 0.609 (0.479) and 0.590 (0.445): a rejected mask often
   matched one of the several fields its polygon held. 0.7 restored more
   coverage (no pivot missing; 40 (77) fields) at an F1 of 0.595 (0.418).

Backends (``config.sam_backend``)
---------------------------------
- ``"sam2"``: ``samgeo.SamGeo2(model_id, automatic=False).predictor``
  (``SAM2ImagePredictor``); SAM 2.0 checkpoints ``facebook/sam2-hiera-*``.
- ``"sam2.1"``: ``sam2.sam2_image_predictor.SAM2ImagePredictor.from_pretrained``
  with ``facebook/sam2.1-hiera-*`` checkpoints (SamGeo2 accepts only SAM 2.0
  ids). ``apply_postprocessing=False`` as in SamGeo2, so the two differ only
  in their weights.
- ``"sam3"``: ``samgeo.SamGeo3(backend="meta", enable_inst_interactivity=True)``
  and ``predict_inst(box=...)`` (SAM 3 instance-interactive, i.e. SAM 1/2
  style single-object prompts). Needs a CUDA GPU and ``triton``, which the Meta
  package imports at import time: Linux is the platform Meta supports; Windows
  works only through the community ``triton-windows`` wheel (not verified by
  agribound; a WARNING is logged); macOS is not supported. Gated weights
  ``facebook/sam3`` or ``facebook/sam3.1``.
- ``"sam3-hf"``: ``transformers.Sam3TrackerModel`` / ``Sam3TrackerProcessor``
  (promptable visual segmentation, one mask per box). No triton dependency, so it
  is the SAM 3 option for platforms without triton (Windows without triton-windows,
  macOS); it imports on macOS, but agribound has not yet run it end to end on any
  platform because the weights are gated. CUDA recommended. Gated weights
  ``facebook/sam3``.

Concept-exemplar (PCS) box prompts such as ``SamGeo3.generate_masks_by_boxes``,
which segment *all* objects similar to the box, are never used.
"""

from __future__ import annotations

import logging
import math
import numbers
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import numpy as np

if TYPE_CHECKING:
    from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Legacy size aliases -> SAM 2.0 Hugging Face ids (``engine_params["sam_model"]``).
SAM2_MODELS: dict[str, str] = {
    "tiny": "facebook/sam2-hiera-tiny",
    "small": "facebook/sam2-hiera-small",
    "base_plus": "facebook/sam2-hiera-base-plus",
    "large": "facebook/sam2-hiera-large",
}

_SAM2_SIZES = {"tiny": "tiny", "small": "small", "base_plus": "base-plus", "large": "large"}

#: Default model per backend.
DEFAULT_SAM_MODELS: dict[str, str] = {
    "sam2": "facebook/sam2-hiera-large",
    "sam2.1": "facebook/sam2.1-hiera-large",
    "sam3": "facebook/sam3",
    "sam3-hf": "facebook/sam3",
}

#: Model ids accepted per backend (``sam2``/``sam2.1`` also accept size aliases).
ALLOWED_SAM_MODELS: dict[str, tuple[str, ...]] = {
    "sam2": tuple(f"facebook/sam2-hiera-{s}" for s in _SAM2_SIZES.values()),
    "sam2.1": tuple(f"facebook/sam2.1-hiera-{s}" for s in _SAM2_SIZES.values()),
    "sam3": ("facebook/sam3", "facebook/sam3.1"),
    "sam3-hf": ("facebook/sam3",),
}

#: Checkpoint file names in the Hugging Face repositories (sam2 1.1.0
#: ``build_sam.HF_MODEL_ID_TO_FILENAMES``; sam3 0.1.4 ``download_ckpt_from_hf``;
#: samgeo 1.4.2 ``SAM31_CKPT_NAME``).
_SAM3_CHECKPOINTS = {"facebook/sam3": "sam3.pt", "facebook/sam3.1": "sam3.1_multiplex.pt"}

#: SAM 3 text-encoder vocabulary, from the location samgeo 1.4.2 downloads it from.
_SAM3_BPE = ("giswqs/geospatial", "bpe_simple_vocab_16e6.txt.gz", "dataset")

#: Defaults mirrored from :class:`agribound.config.AgriboundConfig`.
MIN_CROP_SIZE = 64
CROP_PADDING = 0.15

DEFAULT_WINDOW_PX = 1024
DEFAULT_BATCH_SIZE = 32
#: Field windows are read decimated above ``max(2 * window_px, MIN_MAX_WINDOW_PX)`` px.
MIN_MAX_WINDOW_PX = 2048
#: A bounding box may extend this many pixels beyond the raster and still be refined.
EDGE_TOLERANCE_PX = 0.5
STRETCH_PERCENTILES = (1.0, 99.0)
REFINED_COLUMN = "agribound:sam_refined"
SCORE_COLUMN = "agribound:sam_score"

_EPS = 1e-6


# ---------------------------------------------------------------------------
# Gating (pure functions, used by the agent's resolvability estimate)
# ---------------------------------------------------------------------------


def crop_window_px(
    bounds: tuple[float, float, float, float],
    pixel_size: tuple[float, float],
    padding: float,
) -> tuple[int, int]:
    """Return the padded bounding-box size in whole pixels, as used for gating.

    ``width_px = floor((maxx - minx) / |pixel_size[0]| * (1 + 2 * padding))``
    and likewise for the height with ``pixel_size[1]`` (a tolerance of 1e-6
    pixel absorbs floating-point error).

    Parameters
    ----------
    bounds : tuple of float
        ``(minx, miny, maxx, maxy)`` in the raster CRS.
    pixel_size : tuple of float
        Pixel width and height in CRS units (sign ignored).
    padding : float
        Padding on each side as a fraction of the box size (>= 0).

    Returns
    -------
    tuple[int, int]
        ``(width_px, height_px)``; ``(0, 0)`` for non-finite bounds.

    Raises
    ------
    ValueError
        If a pixel size is not positive or *padding* is negative.
    """
    px, py = abs(float(pixel_size[0])), abs(float(pixel_size[1]))
    if not (px > 0 and py > 0):
        raise ValueError(f"pixel_size must be non-zero, got {pixel_size}")
    if padding < 0:
        raise ValueError(f"padding must be >= 0, got {padding}")
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        return 0, 0
    scale = 1.0 + 2.0 * float(padding)
    width = max(0.0, (maxx - minx) / px * scale)
    height = max(0.0, (maxy - miny) / py * scale)
    return int(math.floor(width + _EPS)), int(math.floor(height + _EPS))


def is_refinable(
    bounds: tuple[float, float, float, float],
    pixel_size: tuple[float, float],
    min_crop_px: int = MIN_CROP_SIZE,
    padding: float = CROP_PADDING,
    raster_bounds: tuple[float, float, float, float] | None = None,
) -> bool:
    """Return *True* if :func:`refine_boundaries` would prompt SAM with this box.

    A polygon is refined only when both sides of its padded bounding box
    (:func:`crop_window_px`) are at least *min_crop_px* pixels. With the
    defaults (64 px, 15 % padding) the unpadded box must be at least
    ``64 / 1.3 = 49.2`` pixels on each side.

    :func:`refine_boundaries` also skips every polygon whose bounding box
    extends more than :data:`EDGE_TOLERANCE_PX` (half a pixel) beyond the
    raster. Without *raster_bounds* this function assumes the polygon lies
    inside the raster (normally true for polygons delineated from it); with
    *raster_bounds* it applies that test too and then reproduces
    :func:`refine_boundaries` exactly.

    Parameters
    ----------
    bounds : tuple of float
        ``(minx, miny, maxx, maxy)`` in the raster CRS.
    pixel_size : tuple of float
        Pixel width and height in CRS units.
    min_crop_px : int
        Minimum padded side in pixels (``config.sam_min_crop_px``).
    padding : float
        Padding fraction (``config.sam_crop_padding``).
    raster_bounds : tuple of float or None
        Raster extent ``(minx, miny, maxx, maxy)`` in its CRS (the order of
        the two x and the two y values does not matter).

    Returns
    -------
    bool
    """
    if raster_bounds is not None and not _inside_raster(bounds, raster_bounds, pixel_size):
        return False
    width, height = crop_window_px(bounds, pixel_size, padding)
    return min(width, height) >= int(min_crop_px)


def _inside_raster(
    bounds: tuple[float, float, float, float],
    raster_bounds: tuple[float, float, float, float],
    pixel_size: tuple[float, float],
) -> bool:
    """True if *bounds* lie inside *raster_bounds*, up to :data:`EDGE_TOLERANCE_PX` pixels.

    The box must also overlap the raster (so that clipping it to the raster
    leaves a non-empty box).
    """
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    if not all(math.isfinite(v) for v in (minx, miny, maxx, maxy)):
        return False
    rx0, ry0, rx1, ry1 = (float(v) for v in raster_bounds)
    rx0, rx1 = min(rx0, rx1), max(rx0, rx1)
    ry0, ry1 = min(ry0, ry1), max(ry0, ry1)
    tol_x = EDGE_TOLERANCE_PX * abs(float(pixel_size[0]))
    tol_y = EDGE_TOLERANCE_PX * abs(float(pixel_size[1]))
    within = (
        minx >= rx0 - tol_x and maxx <= rx1 + tol_x and miny >= ry0 - tol_y and maxy <= ry1 + tol_y
    )
    overlaps = minx < rx1 and maxx > rx0 and miny < ry1 and maxy > ry0
    return within and overlaps


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------


def resolve_sam_model(backend: str, model: str | None = None) -> str:
    """Return the Hugging Face model id for *backend* and an optional *model*.

    Parameters
    ----------
    backend : str
        ``"sam2"``, ``"sam2.1"``, ``"sam3"`` or ``"sam3-hf"``.
    model : str or None
        Model id (``"facebook/..."`` or without the ``facebook/`` prefix) or,
        for SAM 2/2.1, a size alias ``"tiny"``, ``"small"``, ``"base_plus"``
        or ``"large"``. *None* selects :data:`DEFAULT_SAM_MODELS`.

    Returns
    -------
    str

    Raises
    ------
    ValueError
        For an unknown backend or a model the backend cannot load.
    """
    from agribound.registry import SAM_REFINE_BACKENDS

    backend = str(backend).lower().strip()
    if backend not in SAM_REFINE_BACKENDS:
        raise ValueError(f"Unknown SAM backend {backend!r}. Choose from {SAM_REFINE_BACKENDS}")
    if model is None or str(model).strip() == "":
        return DEFAULT_SAM_MODELS[backend]
    text = str(model).strip()
    if backend in ("sam2", "sam2.1"):
        alias = text.lower().replace("-", "_")
        if alias in _SAM2_SIZES:
            return f"facebook/{backend}-hiera-{_SAM2_SIZES[alias]}"
    model_id = text if text.startswith("facebook/") else f"facebook/{text}"
    allowed = ALLOWED_SAM_MODELS[backend]
    if model_id not in allowed:
        hint = (
            " or a size alias (tiny, small, base_plus, large)" if backend.startswith("sam2") else ""
        )
        raise ValueError(
            f"SAM backend {backend!r} cannot load model {model!r}. "
            f"Use one of {list(allowed)}{hint}."
        )
    return model_id


def _configured_model(config: AgriboundConfig) -> str | None:
    """``config.sam_model``, else the legacy ``engine_params["sam_model"]``."""
    if config.sam_model:
        return config.sam_model
    legacy = (config.engine_params or {}).get("sam_model")
    return str(legacy) if legacy else None


# ---------------------------------------------------------------------------
# Predictor adapters: set_image(uint8 HxWx3) then predict_boxes((B, 4)) ->
# (masks (B, H, W) bool, scores (B,) float)
# ---------------------------------------------------------------------------


def _normalise_sam_output(masks: Any, scores: Any, n_boxes: int) -> tuple[np.ndarray, np.ndarray]:
    """Bring SAM2-style predictor output to ``(B, H, W)`` bool masks and ``(B,)`` scores.

    ``SAM2ImagePredictor.predict`` (and SAM 3's instance predictor) squeeze
    the batch axis for a single box: masks are ``(C, H, W)`` for one box and
    ``(B, C, H, W)`` for several; with ``multimask_output=False`` ``C == 1``.
    """
    m = np.asarray(masks)
    s = np.asarray(scores, dtype=np.float64)
    if m.ndim == 3:
        m = m[None]
    if s.ndim == 1:
        s = s[None]
    if m.ndim != 4 or m.shape[0] != n_boxes:
        raise RuntimeError(
            f"Unexpected SAM mask shape {np.asarray(masks).shape} for {n_boxes} boxes"
        )
    return m[:, 0] > 0.5, s[:, 0]


class _SAM2Adapter:
    """Wraps ``sam2.sam2_image_predictor.SAM2ImagePredictor`` (also used by SamGeo2)."""

    def __init__(self, predictor: Any, backend: str, model_id: str, device: str) -> None:
        self._predictor = predictor
        self.backend = backend
        self.model_id = model_id
        self.device = device

    def set_image(self, image: np.ndarray) -> None:
        self._predictor.set_image(image)

    def predict_boxes(self, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        masks, scores, _ = self._predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.asarray(boxes, dtype=np.float32),
            multimask_output=False,
        )
        return _normalise_sam_output(masks, scores, len(boxes))


class _SAM3MetaAdapter:
    """Wraps ``samgeo.SamGeo3(backend="meta", enable_inst_interactivity=True)``."""

    def __init__(self, sam: Any, model_id: str, device: str) -> None:
        self._sam = sam
        self.backend = "sam3"
        self.model_id = model_id
        self.device = device

    def set_image(self, image: np.ndarray) -> None:
        self._sam.set_image(image)

    def predict_boxes(self, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        masks, scores, _ = self._sam.predict_inst(
            box=np.asarray(boxes, dtype=np.float32), multimask_output=False
        )
        return _normalise_sam_output(masks, scores, len(boxes))


class _SAM3HFAdapter:
    """Wraps ``transformers.Sam3TrackerModel`` + ``Sam3TrackerProcessor`` (one encode per image)."""

    def __init__(self, model: Any, processor: Any, model_id: str, device: str) -> None:
        self._model = model
        self._processor = processor
        self.backend = "sam3-hf"
        self.model_id = model_id
        self.device = device
        self._embeddings = None
        self._sizes = None
        try:
            self._dtype = next(model.parameters()).dtype
        except (StopIteration, AttributeError, TypeError):
            self._dtype = None

    def set_image(self, image: np.ndarray) -> None:
        import torch
        from PIL import Image

        encoded = self._processor(images=Image.fromarray(image), return_tensors="pt")
        self._sizes = encoded["original_sizes"]
        pixel_values = encoded["pixel_values"].to(self.device)
        if self._dtype is not None:
            pixel_values = pixel_values.to(self._dtype)
        with torch.no_grad():
            self._embeddings = self._model.get_image_embeddings(pixel_values)

    def predict_boxes(self, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        prompts = self._processor(
            original_sizes=self._sizes,
            input_boxes=[np.asarray(boxes, dtype=np.float64).tolist()],
            return_tensors="pt",
        )
        input_boxes = prompts["input_boxes"].to(self.device)
        if self._dtype is not None:
            input_boxes = input_boxes.to(self._dtype)
        with torch.no_grad():
            out = self._model(
                image_embeddings=self._embeddings,
                input_boxes=input_boxes,
                multimask_output=False,
            )
        masks = self._processor.post_process_masks(out.pred_masks.cpu(), self._sizes)[0]
        masks = np.asarray(masks.numpy() if hasattr(masks, "numpy") else masks)
        scores = out.iou_scores[0, :, 0].float().cpu().numpy()
        if masks.ndim != 4 or masks.shape[0] != len(boxes):
            raise RuntimeError(f"Unexpected SAM 3 mask shape {masks.shape} for {len(boxes)} boxes")
        return masks[:, 0].astype(bool), scores.astype(np.float64)


def _is_gated(exc: BaseException) -> bool:
    """True if *exc* is, or was raised from, a ``huggingface_hub`` ``GatedRepoError``.

    transformers re-raises gated-repository errors as ``OSError(...) from
    GatedRepoError`` (``transformers.utils.hub.cached_file``), so the cause
    chain is searched.
    """
    from huggingface_hub.errors import GatedRepoError

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, GatedRepoError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _gated_error(model_id: str, exc: Exception) -> RuntimeError:
    return RuntimeError(
        f"Access to the gated Hugging Face repository {model_id!r} was refused ({exc}). "
        f"Request access at https://huggingface.co/{model_id}, then log in with "
        "`hf auth login` (or set HF_TOKEN). For the 'sam3' backend on offline nodes you "
        "can instead set SAM3_CHECKPOINT_PATH to a downloaded checkpoint."
    )


def _require_meta_sam3_runtime(device: str) -> None:
    """Check the runtime the Meta ``sam3`` package needs: a CUDA device and ``triton``.

    ``sam3`` 0.1.4 imports ``triton`` at module import time (``sam3.model_builder`` ->
    ``sam3_tracking_predictor`` -> ``sam3_tracker_utils`` -> ``edt``), even for
    single-image inference, and Meta supports Linux with CUDA only. On Windows the
    community ``triton-windows`` wheel (github.com/triton-lang/triton-windows; it
    installs the ``triton`` module) is accepted but unverified by agribound. macOS has
    no CUDA, so it always fails here; use ``sam_backend="sam3-hf"`` there.
    """
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            "sam_backend='sam3' (Meta SAM 3 via samgeo) needs a CUDA GPU "
            f"(platform={sys.platform!r}, device={device!r}). "
            "Use sam_backend='sam3-hf' (transformers; no triton dependency) instead."
        )
    try:
        import triton  # noqa: F401
    except ImportError as exc:
        hint = (
            "pip install 'agribound[sam3]' (Linux: triton; Windows: the community "
            "triton-windows wheel)"
            if sys.platform in ("linux", "win32")
            else "use sam_backend='sam3-hf'"
        )
        raise RuntimeError(
            "sam_backend='sam3' needs triton, which the Meta sam3 package imports at "
            f"import time (platform={sys.platform!r}). Fix: {hint}."
        ) from exc
    if sys.platform == "win32":
        logger.warning(
            "sam_backend='sam3' on Windows: Meta supports Linux + CUDA only; running with a "
            "community triton build (triton-windows). This combination is not verified by "
            "agribound. sam_backend='sam3-hf' avoids triton entirely."
        )
    elif not sys.platform.startswith("linux"):
        raise RuntimeError(
            f"sam_backend='sam3' is not supported on platform {sys.platform!r}; "
            "use sam_backend='sam3-hf'."
        )


#: Logged whenever a SAM 3 backend is loaded (agribound 1.0.1 has not run either end to end).
SAM3_UNTESTED_WARNING = (
    "sam_backend=%r: the SAM 3 backends (sam3, sam3-hf) are untested in agribound 1.0.1. They "
    "have not been run end to end, because the facebook/sam3 weights are gated; only their "
    "imports and argument handling are covered by tests. Check the refined polygons before "
    "relying on them, or use sam_backend='sam2' (tested)."
)


def _load_predictor(backend: str, model_id: str, device: str) -> Any:
    """Load the SAM predictor for *backend* (weights are fetched from the HF cache/hub)."""
    if backend in ("sam3", "sam3-hf"):
        logger.warning(SAM3_UNTESTED_WARNING, backend)
    if backend == "sam2":
        try:
            from samgeo.samgeo2 import SamGeo2
        except ImportError as exc:
            raise ImportError(
                "sam_backend='sam2' needs segment-geospatial with SAM 2. "
                "Install with: pip install 'agribound[samgeo]'"
            ) from exc
        sam = SamGeo2(model_id=model_id, device=device, automatic=False)
        return _SAM2Adapter(sam.predictor, "sam2", model_id, device)

    if backend == "sam2.1":
        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "sam_backend='sam2.1' needs the sam2 package. "
                "Install with: pip install 'agribound[samgeo]'"
            ) from exc
        predictor = SAM2ImagePredictor.from_pretrained(
            model_id, device=device, mode="eval", apply_postprocessing=False
        )
        return _SAM2Adapter(predictor, "sam2.1", model_id, device)

    if backend == "sam3":
        _require_meta_sam3_runtime(device)
        try:
            from samgeo import samgeo3
        except ImportError as exc:
            raise ImportError(
                "sam_backend='sam3' needs segment-geospatial with SAM 3. "
                "Install with: pip install 'agribound[sam3]' (CUDA; Linux, or Windows "
                "via triton-windows)"
            ) from exc
        if not getattr(samgeo3, "SAM3_META_AVAILABLE", False):
            raise ImportError(
                "Meta SAM 3 could not be imported "
                f"({getattr(samgeo3, 'SAM3_META_IMPORT_ERROR', 'unknown error')}). "
                "Install with: pip install 'agribound[sam3]'"
            )
        from huggingface_hub import hf_hub_download

        try:
            bpe_path = hf_hub_download(_SAM3_BPE[0], _SAM3_BPE[1], repo_type=_SAM3_BPE[2])
            sam = samgeo3.SamGeo3(
                backend="meta",
                model_id=model_id,
                device=device,
                bpe_path=bpe_path,
                enable_inst_interactivity=True,
            )
        except Exception as exc:
            if _is_gated(exc):
                raise _gated_error(model_id, exc) from exc
            raise
        return _SAM3MetaAdapter(sam, model_id, device)

    if backend == "sam3-hf":
        try:
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        except ImportError as exc:
            raise ImportError(
                "sam_backend='sam3-hf' needs transformers >= 5 with Sam3TrackerModel. "
                "Install with: pip install 'transformers>=5'"
            ) from exc
        try:
            processor = Sam3TrackerProcessor.from_pretrained(model_id)
            model = Sam3TrackerModel.from_pretrained(model_id).to(device)
        except Exception as exc:
            if _is_gated(exc):
                raise _gated_error(model_id, exc) from exc
            raise
        model.eval()
        return _SAM3HFAdapter(model, processor, model_id, device)

    raise ValueError(f"Unknown SAM backend {backend!r}")


def prefetch(config: AgriboundConfig) -> list[str]:
    """Download the weights of the configured SAM backend into the Hugging Face cache.

    Afterwards refinement can run with ``HF_HUB_OFFLINE=1``. Files:
    ``sam2``/``sam2.1``: the checkpoint named in
    ``sam2.build_sam.HF_MODEL_ID_TO_FILENAMES``; ``sam3``: ``config.json``,
    ``sam3.pt`` (or ``sam3.1_multiplex.pt``) and the text-encoder vocabulary;
    ``sam3-hf``: the repository's ``*.json`` and ``*.safetensors`` files.

    Parameters
    ----------
    config : AgriboundConfig
        Uses ``sam_backend`` and ``sam_model`` (or legacy
        ``engine_params["sam_model"]``).

    Returns
    -------
    list[str]
        Local paths of the downloaded files (or snapshot directory).

    Raises
    ------
    RuntimeError
        If a gated repository refuses access.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    backend = config.sam_backend
    model_id = resolve_sam_model(backend, _configured_model(config))
    try:
        if backend in ("sam2", "sam2.1"):
            try:
                from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
            except ImportError as exc:
                raise ImportError(
                    "Prefetching SAM 2 weights needs the sam2 package: "
                    "pip install 'agribound[samgeo]'"
                ) from exc
            checkpoint = HF_MODEL_ID_TO_FILENAMES[model_id][1]
            return [hf_hub_download(model_id, checkpoint)]
        if backend == "sam3":
            return [
                hf_hub_download(model_id, "config.json"),
                hf_hub_download(model_id, _SAM3_CHECKPOINTS[model_id]),
                hf_hub_download(_SAM3_BPE[0], _SAM3_BPE[1], repo_type=_SAM3_BPE[2]),
            ]
        return [snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"])]
    except Exception as exc:
        if _is_gated(exc):
            raise _gated_error(model_id, exc) from exc
        raise


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------


def _rgb_band_indices(config: AgriboundConfig, n_bands: int, rgb_bands: Any = None) -> list[int]:
    """1-based band indices used as SAM's R, G, B for *config.source*."""
    from agribound.engines.base import get_canonical_band_indices
    from agribound.registry import source_value_scale

    explicit = rgb_bands if rgb_bands is not None else config.engine_params.get("sam_rgb_bands")
    if explicit is not None:
        indices = [int(i) for i in explicit]
        if len(indices) != 3:
            raise ValueError(f"sam_rgb_bands must list 3 band indices, got {explicit!r}")
    elif source_value_scale(config.source) == "embedding":
        raise ValueError(
            f"SAM refinement needs an RGB image, but source {config.source!r} is an embedding "
            "raster without RGB bands. Set engine_params['sam_rgb_bands'] = [i, j, k] "
            "(1-based) to use three embedding dimensions as a pseudo-RGB image (recorded "
            "in sam_stats), or call refine_boundaries() with an imagery raster and its config."
        )
    else:
        indices = get_canonical_band_indices(config.source, ["R", "G", "B"], bands=config.bands)
    bad = [i for i in indices if not 1 <= i <= n_bands]
    if bad:
        raise ValueError(f"RGB band indices {indices} are out of range for a {n_bands}-band raster")
    return indices


def _stretch_bounds(
    src: Any, bands: list[int], embedding: bool
) -> tuple[list[float], list[float], bool]:
    """Scene-wide stretch bounds from a decimated read (<= 4096 px per side).

    The sample is read by :func:`agribound.io.raster.read_stretch_sample`
    (full-resolution data: a raster with overviews is reopened with
    ``OVERVIEW_LEVEL=NONE`` when decimating). Returns
    ``(lows, highs, passthrough)``; *passthrough* is True for uint8 data.
    """
    from agribound.io.raster import percentile_stretch_uint8, read_stretch_sample

    if all(src.dtypes[b - 1] == "uint8" for b in bands):
        return [0.0] * 3, [255.0] * 3, True
    sample = read_stretch_sample(src, bands)
    low, high = STRETCH_PERCENTILES
    if not embedding:
        lows, highs = percentile_stretch_uint8(
            sample, nodata=src.nodata, low=low, high=high, per_band=True, return_bounds_only=True
        )
        return list(lows), list(highs), False
    # Embedding dimensions are signed: percentiles over all finite values.
    lows, highs = [], []
    for i in range(sample.shape[0]):
        band = sample[i].astype(np.float64)
        finite = band[np.isfinite(band)]
        if finite.size == 0:
            raise ValueError(f"Band {bands[i]} has no finite values for the SAM stretch")
        lo, hi = np.percentile(finite, (low, high))
        lows.append(float(lo))
        highs.append(float(hi))
    return lows, highs, False


def _apply_stretch(
    data: np.ndarray, lows: list[float], highs: list[float], nodata: float | None, passthrough: bool
) -> np.ndarray:
    """Map a ``(3, h, w)`` window to uint8 ``(h, w, 3)`` with fixed bounds.

    Same mapping as :func:`agribound.io.raster.percentile_stretch_uint8`:
    ``clip(255 * (v - lo) / (hi - lo), 0, 255)`` truncated to uint8, with
    non-finite pixels and all-band nodata pixels set to 0.
    """
    if passthrough:
        out = data.astype(np.uint8, copy=False)
    else:
        valid = (
            np.all(np.isfinite(data), axis=0)
            if np.issubdtype(data.dtype, np.floating)
            else np.ones(data.shape[1:], dtype=bool)
        )
        if nodata is not None and np.isfinite(nodata):
            valid &= ~np.all(data == nodata, axis=0)
        out = np.zeros(data.shape, dtype=np.uint8)
        for i, (lo, hi) in enumerate(zip(lows, highs, strict=True)):
            span = hi - lo if hi > lo else 1e-12
            band = data[i].astype(np.float64)
            with np.errstate(invalid="ignore"):
                stretched = np.clip(255.0 * ((band - lo) / span), 0, 255)
            out[i] = np.where(valid & np.isfinite(stretched), stretched, 0).astype(np.uint8)
    return np.ascontiguousarray(out.transpose(1, 2, 0))


def _pad_to_square(image: np.ndarray) -> np.ndarray:
    """Pad an ``(h, w, 3)`` uint8 image with zeros on the right/bottom to ``(s, s, 3)``.

    SAM resizes its input to a fixed square without keeping the aspect
    ratio, so padding keeps a non-square window undistorted. Pixel
    coordinates (boxes, the top-left part of the masks) are unchanged.
    """
    h, w = image.shape[:2]
    if h == w:
        return image
    side = max(h, w)
    out = np.zeros((side, side, image.shape[2]), dtype=image.dtype)
    out[:h, :w] = image
    return out


# ---------------------------------------------------------------------------
# Masks -> polygons
# ---------------------------------------------------------------------------


def _mask_to_polygon(mask: np.ndarray, transform: Any) -> Any:
    """Largest polygon (with holes) of a binary mask, in the transform's CRS, or *None*."""
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape

    from agribound.postprocess.simplify import make_polygonal

    mask_uint8 = np.ascontiguousarray(mask, dtype=np.uint8)
    polys = []
    for geom_dict, val in rasterio_shapes(
        mask_uint8, mask=mask_uint8.astype(bool), transform=transform
    ):
        if val != 1:
            continue
        geom = make_polygonal(shape(geom_dict))
        if geom is not None and not geom.is_empty:
            polys.extend(geom.geoms if geom.geom_type == "MultiPolygon" else [geom])
    if not polys:
        return None
    return max(polys, key=lambda g: g.area)


@contextmanager
def _quiet_root_info() -> Iterator[None]:
    """Drop INFO records logged directly on the root logger (SAM 2 logs per image)."""

    class _Filter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not (record.name == "root" and record.levelno <= logging.INFO)

    flt = _Filter()
    root = logging.getLogger()
    root.addFilter(flt)
    try:
        yield
    finally:
        root.removeFilter(flt)


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------


def _square_start(lo: int, hi: int, side: int, size: int) -> int:
    """Start of a *side*-long interval centred on ``[lo, hi]`` and kept inside ``[0, size]``.

    Requires ``hi - lo <= side``; the result always contains ``[lo, hi]``.
    """
    start = int(math.floor((lo + hi - side) / 2))
    return min(max(start, 0), max(size - side, 0))


def _grid_transform(
    transform: Any, col_off: float, row_off: float, sx: float = 1.0, sy: float = 1.0
) -> Any:
    """Transform of a grid whose origin is raster pixel ``(col_off, row_off)``.

    Each grid pixel spans ``sx`` x ``sy`` raster pixels. *transform* must
    have no rotation (``refine_boundaries`` checks this). Built with the
    ``Affine`` constructor, which behaves the same in every ``affine``
    version (``*`` composition is deprecated in affine 3, ``@`` needs >= 3).
    """
    from rasterio import Affine

    return Affine(
        transform.a * sx,
        0.0,
        transform.c + transform.a * col_off,
        0.0,
        transform.e * sy,
        transform.f + transform.e * row_off,
    )


def _max_window_px(window_px: int) -> int:
    """Longest side a field window is read at (larger windows are decimated)."""
    return max(2 * int(window_px), MIN_MAX_WINDOW_PX)


def _plan_windows(
    bounds_px: np.ndarray,
    refinable: np.ndarray,
    padding: float,
    width: int,
    height: int,
    window_px: int,
) -> dict[tuple, dict]:
    """Assign refinable polygons to encoder windows.

    *bounds_px* holds ``(col0, row0, col1, row1)`` pixel bounds per polygon;
    the polygons selected by *refinable* must lie inside the raster up to
    :data:`EDGE_TOLERANCE_PX` (``refine_boundaries`` checks this). A polygon
    whose padded box (clipped to the raster, rounded outwards) is at most
    ``window_px - window_px // 2`` pixels on both axes goes to the grid
    window whose origin (a multiple of ``window_px // 2``) precedes the box;
    the window is shifted back inside the raster so that it keeps its full
    ``window_px`` size wherever the raster is large enough. Larger polygons
    get a square window with the side of their padded box's longer axis,
    centred on it and shifted inside the raster; when that side exceeds
    :func:`_max_window_px` the window is read decimated to an output shape
    whose longer side is that limit. Every padded box lies inside its window.

    Returns ``{key: {"window": (col, row, w, h), "out_shape": (out_h, out_w),
    "items": [(pos, box, clip)]}}``: *window* in raster pixels; *out_shape*
    the shape it is read at (``(h, w)`` unless decimated); *box* the prompt
    ``(x0, y0, x1, y1)`` (the bounding box clipped to the raster) and *clip*
    the integer padded box ``(x0, y0, x1, y1)`` the mask is confined to,
    both in output pixel coordinates.
    """
    stride = window_px // 2
    fit = window_px - stride
    max_side = _max_window_px(window_px)
    windows: dict[tuple, dict] = {}
    for pos in np.flatnonzero(refinable):
        c0, r0, c1, r1 = bounds_px[pos]
        pad_c = (c1 - c0) * padding
        pad_r = (r1 - r0) * padding
        ic0, ir0 = max(c0 - pad_c, 0.0), max(r0 - pad_r, 0.0)
        ic1, ir1 = min(c1 + pad_c, float(width)), min(r1 + pad_r, float(height))
        x0, y0 = int(math.floor(ic0)), int(math.floor(ir0))
        x1, y1 = int(math.ceil(ic1)), int(math.ceil(ir1))
        if x1 - x0 <= fit and y1 - y0 <= fit:
            wx = min((x0 // stride) * stride, max(width - window_px, 0))
            wy = min((y0 // stride) * stride, max(height - window_px, 0))
            key: tuple = ("grid", wy, wx)
            window = (wx, wy, min(window_px, width), min(window_px, height))
        else:
            side = max(x1 - x0, y1 - y0)
            wx = _square_start(x0, x1, side, width)
            wy = _square_start(y0, y1, side, height)
            key = ("field", int(pos))
            window = (wx, wy, min(side, width), min(side, height))
        w, h = window[2], window[3]
        if max(w, h) > max_side:
            factor = max(w, h) / max_side
            out_shape = (max(1, int(round(h / factor))), max(1, int(round(w / factor))))
        else:
            out_shape = (h, w)
        sx, sy = w / out_shape[1], h / out_shape[0]
        box = (
            (max(c0, 0.0) - wx) / sx,
            (max(r0, 0.0) - wy) / sy,
            (min(c1, float(width)) - wx) / sx,
            (min(r1, float(height)) - wy) / sy,
        )
        clip = (
            max(0, int(math.floor((x0 - wx) / sx))),
            max(0, int(math.floor((y0 - wy) / sy))),
            min(out_shape[1], int(math.ceil((x1 - wx) / sx))),
            min(out_shape[0], int(math.ceil((y1 - wy) / sy))),
        )
        entry = windows.setdefault(key, {"window": window, "out_shape": out_shape, "items": []})
        entry["items"].append((int(pos), box, clip))
    return dict(sorted(windows.items()))


#: Values of ``engine_params["sam_overlaps"]`` (module docstring, step 6).
SAM_OVERLAP_MODES = ("trim", "keep")

#: Default of ``engine_params["sam_min_coverage"]`` (module docstring, step 7).
DEFAULT_MIN_COVERAGE = 0.5


def _min_coverage(value: Any) -> float:
    """Validate ``engine_params["sam_min_coverage"]``: a number in [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"sam_min_coverage must be a number in [0, 1], got {value!r}")
    if not 0.0 <= float(value) <= 1.0:  # also rejects NaN
        raise ValueError(f"sam_min_coverage must be in [0, 1], got {value!r}")
    return float(value)


def _repaired(originals: list[Any]) -> list[Any]:
    """*originals* with every invalid geometry made valid (its polygonal parts)."""
    from agribound.postprocess.simplify import make_polygonal

    return [
        make_polygonal(g) if g is not None and not g.is_empty and not g.is_valid else g
        for g in originals
    ]


def _covers_too_little(geom: Any, own: Any, min_coverage: float) -> bool:
    """Whether mask *geom* covers less than *min_coverage* of its valid input *own* (step 7).

    Coverage is ``area(geom & own) / area(own)``. An input without area never
    counts, and no mask does when *min_coverage* is 0.
    """
    if min_coverage <= 0 or own is None or own.is_empty or not own.area > 0:
        return False
    return geom.intersection(own).area / own.area < min_coverage


def _largest_polygon(geom: Any) -> Any:
    """The largest polygon of a (repaired) polygonal geometry, or None if none is left."""
    from agribound.postprocess.simplify import make_polygonal

    fixed = make_polygonal(geom)
    if fixed is None or fixed.is_empty:
        return None
    if fixed.geom_type == "MultiPolygon":
        fixed = max(fixed.geoms, key=lambda part: part.area)
    return fixed if fixed.area > 0 else None


def trim_refinement_overlaps(
    originals: list[Any],
    refined: dict[int, Any],
    scores: np.ndarray,
) -> tuple[dict[int, Any], list[int], float]:
    """Keep SAM masks from growing over other polygons (see the module docstring, step 6).

    Refined polygons are processed by decreasing SAM score (ties by
    position). Each loses (a) the part of every *other* input polygon that
    its own input polygon did not cover and (b) the area already given to a
    higher-scoring refined polygon, again outside its own input polygon;
    of what remains the largest polygon is kept.

    Parameters
    ----------
    originals : list of shapely geometry or None
        Input polygons (one per row, same CRS as *refined*).
    refined : dict
        Row position -> refined polygon.
    scores : numpy.ndarray
        SAM score per row (NaN where not refined).

    Returns
    -------
    (dict, list of int, float)
        The trimmed refined polygons (rows whose mask had nothing left are
        omitted), the positions whose mask was trimmed or dropped, and the
        total area removed (in CRS units squared).
    """
    kept, trimmed, removed, _ = _trim_overlaps(originals, refined, scores, 0.0)
    return kept, trimmed, removed


def _trim_overlaps(
    originals: list[Any],
    refined: dict[int, Any],
    scores: np.ndarray,
    min_coverage: float,
) -> tuple[dict[int, Any], list[int], float, list[int]]:
    """:func:`trim_refinement_overlaps` plus the coverage test (module docstring, step 7).

    A trimmed mask that covers less than *min_coverage* of its input polygon
    is left out of the result like a dropped one, but it claims no area, so
    the lower-scoring masks are trimmed as if it had never been refined, and
    it is not counted as trimmed. Returns ``(kept, trimmed, removed, low)``,
    with *low* the positions of those masks in ascending order.
    """
    import shapely

    work = _repaired(originals)
    idx = [i for i, g in enumerate(work) if g is not None and not g.is_empty]
    tree = shapely.STRtree([work[i] for i in idx]) if idx else None

    def rank(pos: int) -> tuple[float, int]:
        score = scores[pos]
        return (-(float(score) if np.isfinite(score) else -np.inf), pos)

    kept: dict[int, Any] = {}
    claimed: list[Any] = []
    trimmed: list[int] = []
    low: list[int] = []
    removed = 0.0
    for pos in sorted(refined, key=rank):
        geom = refined[pos]
        own = work[pos] if pos < len(work) else None
        blockers = []
        if tree is not None:
            hits = tree.query(geom, predicate="intersects")
            blockers += [work[idx[h]] for h in hits if idx[h] != pos]
        blockers += [c for c in claimed if c.intersects(geom)]
        new = geom
        if blockers:
            forbidden = shapely.union_all(blockers)
            if own is not None and not own.is_empty:
                forbidden = forbidden.difference(own)
            if not forbidden.is_empty and forbidden.intersection(geom).area > 0:
                new = _largest_polygon(geom.difference(forbidden))
        if new is not None and _covers_too_little(new, own, min_coverage):
            low.append(pos)  # the input geometry stays, so this mask claims nothing
            continue
        if new is not geom:
            trimmed.append(pos)
            removed += geom.area - (new.area if new is not None else 0.0)
        if new is None:
            continue
        kept[pos] = new
        claimed.append(new)
    return kept, trimmed, float(removed), sorted(low)


def refine_boundaries(
    gdf: gpd.GeoDataFrame,
    raster_path: str,
    config: AgriboundConfig,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Refine field boundaries with box-prompted SAM (see the module docstring).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Field boundaries from a delineation engine.
    raster_path : str
        GeoTIFF the polygons were delineated from (not rotated or sheared).
    config : AgriboundConfig
        Uses ``source`` (RGB band lookup), ``bands``, ``sam_backend``,
        ``sam_model`` (legacy: ``engine_params["sam_model"]``),
        ``sam_min_crop_px``, ``sam_crop_padding`` and ``device``, plus
        ``engine_params`` ``"sam_rgb_bands"`` (three 1-based band indices,
        required for embedding rasters), ``"sam_window_px"`` (default 1024),
        ``"sam_batch_size"`` (boxes per decoder call, default 32),
        ``"sam_overlaps"`` (``"trim"``, default, or ``"keep"``; step 6 of the
        module docstring) and ``"sam_min_coverage"`` (number in [0, 1],
        default :data:`DEFAULT_MIN_COVERAGE`; 0 disables it; step 7).
    **kwargs
        ``rgb_bands``, ``window_px`` and ``batch_size`` override the
        corresponding ``engine_params``; ``predictor`` supplies an already
        loaded predictor object with ``set_image(uint8 HxWx3)`` and
        ``predict_boxes((B, 4) xyxy) -> ((B, H, W) bool, (B,) scores)``
        (plus ``backend``, ``model_id`` and ``device`` attributes).

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of *gdf* (same rows, order, index and columns) with geometry
        replaced where refined, a bool column ``"agribound:sam_refined"`` and
        a float column ``"agribound:sam_score"`` (SAM's predicted IoU for
        refined rows, NaN otherwise). ``attrs["sam_stats"]`` holds
        ``backend, model, device, n_total, n_refined, n_skipped_small,
        n_skipped_outside, n_failed, min_crop_px, padding, window_px,
        max_window_px, batch_size, n_windows, n_windows_decimated,
        rgb_bands, rgb_source, stretch, errors, overlaps, n_overlap_trimmed,
        overlap_trimmed_fraction, min_coverage, n_low_coverage`` (plus
        ``multimask_output``, ``mask_selection`` and, for SAM 2/2.1,
        ``apply_postprocessing`` when SAM ran), with ``n_total == n_refined +
        n_skipped_small + n_skipped_outside + n_failed + n_low_coverage``.
        ``model`` and ``device`` are the configured ones when no polygon is
        prompted (no model is loaded then). ``n_skipped_outside`` counts
        missing/empty geometries and polygons whose bounding box is not
        inside the raster (half-pixel tolerance); a polygon that is both
        outside and small counts as outside. ``n_failed`` counts empty masks,
        masks that lay entirely on other polygons (step 6 of the module
        docstring) and fields in windows where SAM raised.
        ``n_overlap_trimmed`` counts the masks trimmed so as not to overlap
        other polygons, and ``overlap_trimmed_fraction`` is the share of the
        refined mask area removed by that trimming (masks counted in
        ``n_low_coverage`` excluded). ``n_low_coverage`` counts the polygons
        that keep their input geometry because their mask covered less than
        ``min_coverage`` of it (step 7).

    Raises
    ------
    ValueError
        For a rotated raster, bad band indices, an embedding raster without
        ``sam_rgb_bands``, or an invalid ``sam_window_px``, ``sam_batch_size``,
        ``sam_overlaps`` or ``sam_min_coverage``.
    RuntimeError
        If SAM raised for every window that had prompts.

    Notes
    -----
    Masks depend on the compute device: on a Sentinel-2 test crop, SAM 2
    (``sam2-hiera-tiny``) masks computed on Apple MPS overlapped the CPU
    masks of the same fields with IoU between 0.59 and 0.97. ``sam_stats``
    records the device.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    from agribound.registry import source_value_scale

    backend = config.sam_backend
    min_crop_px = int(config.sam_min_crop_px)
    padding = float(config.sam_crop_padding)
    params = config.engine_params or {}
    window_px = int(kwargs.get("window_px") or params.get("sam_window_px") or DEFAULT_WINDOW_PX)
    batch_size = int(kwargs.get("batch_size") or params.get("sam_batch_size") or DEFAULT_BATCH_SIZE)
    if window_px < 2 * min_crop_px:
        raise ValueError(f"sam_window_px ({window_px}) must be >= 2 * sam_min_crop_px")
    if batch_size < 1:
        raise ValueError(f"sam_batch_size must be >= 1, got {batch_size}")
    overlaps = str(params.get("sam_overlaps", "trim"))
    if overlaps not in SAM_OVERLAP_MODES:
        raise ValueError(f"sam_overlaps must be one of {SAM_OVERLAP_MODES}, got {overlaps!r}")
    min_coverage = _min_coverage(params.get("sam_min_coverage", DEFAULT_MIN_COVERAGE))

    result = gdf.copy()
    result.attrs = dict(gdf.attrs)
    n_total = len(gdf)
    refined_flags = np.zeros(n_total, dtype=bool)
    scores_out = np.full(n_total, np.nan, dtype=np.float64)
    stats: dict[str, Any] = {
        "backend": backend,
        "model": None,
        "device": None,
        "n_total": n_total,
        "n_refined": 0,
        "n_skipped_small": 0,
        "n_skipped_outside": 0,
        "n_failed": 0,
        "min_crop_px": min_crop_px,
        "padding": padding,
        "window_px": window_px,
        "max_window_px": _max_window_px(window_px),
        "batch_size": batch_size,
        "n_windows": 0,
        "n_windows_decimated": 0,
        "rgb_bands": None,
        "stretch": None,
        "errors": [],
        "overlaps": overlaps,
        "n_overlap_trimmed": 0,
        "overlap_trimmed_fraction": 0.0,
        "min_coverage": min_coverage,
        "n_low_coverage": 0,
    }
    predictor = kwargs.get("predictor")
    device = config.resolve_device()
    if predictor is None:
        # The configured model and device, recorded even if nothing is prompted.
        model_id = resolve_sam_model(backend, _configured_model(config))
        stats["model"], stats["device"] = model_id, str(device)
    else:
        stats["backend"] = getattr(predictor, "backend", backend)
        stats["model"] = getattr(predictor, "model_id", None)
        stats["device"] = str(getattr(predictor, "device", device))

    with rasterio.open(raster_path) as src:
        transform = src.transform
        if transform.b != 0 or transform.d != 0:
            raise ValueError(
                f"{raster_path} has a rotated/sheared transform, which SAM refinement does "
                "not support"
            )
        rgb = _rgb_band_indices(config, src.count, kwargs.get("rgb_bands"))
        embedding = source_value_scale(config.source) == "embedding"
        stats["rgb_bands"] = rgb
        stats["rgb_source"] = "embedding dimensions (pseudo-RGB)" if embedding else "imagery"

        raster_crs = src.crs
        if gdf.crs is None:
            logger.warning("refine_boundaries: polygons have no CRS; assuming the raster CRS")
            proj = gdf.set_crs(raster_crs, allow_override=True)
        elif raster_crs is not None and gdf.crs != raster_crs:
            proj = gdf.to_crs(raster_crs)
        else:
            proj = gdf

        pixel_size = (abs(transform.a), abs(transform.e))
        xs = (transform.c, transform.c + transform.a * src.width)
        ys = (transform.f, transform.f + transform.e * src.height)
        raster_bounds = (min(xs), min(ys), max(xs), max(ys))
        geoms = list(proj.geometry)
        present = np.array([g is not None and not g.is_empty for g in geoms], dtype=bool)
        bounds = np.array(
            [g.bounds if ok else (np.nan,) * 4 for g, ok in zip(geoms, present, strict=True)],
            dtype=np.float64,
        ).reshape(n_total, 4)
        inside = np.array(
            [
                ok and _inside_raster(tuple(b), raster_bounds, pixel_size)
                for b, ok in zip(bounds, present, strict=True)
            ],
            dtype=bool,
        )
        # Same decision as is_refinable(..., raster_bounds=raster_bounds).
        refinable = np.array(
            [
                ok and is_refinable(tuple(b), pixel_size, min_crop_px, padding)
                for b, ok in zip(bounds, inside, strict=True)
            ],
            dtype=bool,
        )
        skipped_small = inside & ~refinable

        # Pixel bounds (col0, row0, col1, row1) of each box. The transform has no
        # rotation (checked above), so col = (x - c) / a and row = (y - f) / e;
        # sorting the two corners handles north-up and south-up rasters alike.
        cols_a = (bounds[:, 0] - transform.c) / transform.a
        cols_b = (bounds[:, 2] - transform.c) / transform.a
        rows_a = (bounds[:, 3] - transform.f) / transform.e
        rows_b = (bounds[:, 1] - transform.f) / transform.e
        bounds_px = np.column_stack(
            [
                np.minimum(cols_a, cols_b),
                np.minimum(rows_a, rows_b),
                np.maximum(cols_a, cols_b),
                np.maximum(rows_a, rows_b),
            ]
        )
        windows = _plan_windows(bounds_px, refinable, padding, src.width, src.height, window_px)
        stats["n_skipped_small"] = int(skipped_small.sum())
        stats["n_skipped_outside"] = int((~inside).sum())
        stats["n_windows"] = len(windows)
        stats["n_windows_decimated"] = sum(
            1 for e in windows.values() if e["out_shape"] != (e["window"][3], e["window"][2])
        )

        n_prompts = sum(len(w["items"]) for w in windows.values())
        refined_geoms: dict[int, Any] = {}
        n_failed = 0
        n_window_errors = 0
        if n_prompts:
            lows, highs, passthrough = _stretch_bounds(src, rgb, embedding)
            stats["stretch"] = {
                "method": "none (uint8)" if passthrough else "percentile",
                "percentiles": None if passthrough else list(STRETCH_PERCENTILES),
                "lows": lows,
                "highs": highs,
            }
            if predictor is None:
                logger.info("Loading SAM backend %s (%s) on %s", backend, model_id, device)
                predictor = _load_predictor(backend, model_id, device)
                stats["backend"] = getattr(predictor, "backend", backend)
                stats["model"] = getattr(predictor, "model_id", model_id)
                stats["device"] = str(getattr(predictor, "device", device))
            if backend in ("sam2", "sam2.1"):
                stats["apply_postprocessing"] = False
            stats["multimask_output"] = False
            stats["mask_selection"] = (
                "largest polygon of the mask inside the field's padded box (whole pixels)"
            )

            logger.info(
                "SAM refinement: %d of %d polygons prompted in %d windows "
                "(%d below %d px, %d missing/outside)",
                n_prompts,
                n_total,
                len(windows),
                stats["n_skipped_small"],
                min_crop_px,
                stats["n_skipped_outside"],
            )
            with _quiet_root_info():
                for i_win, entry in enumerate(windows.values()):
                    col, row, w, h = entry["window"]
                    out_h, out_w = entry["out_shape"]
                    items = entry["items"]
                    win = Window(col, row, w, h)
                    done: set[int] = set()
                    try:
                        if (out_h, out_w) == (h, w):
                            data = src.read(rgb, window=win)
                        else:  # oversized field window: decimated read
                            data = src.read(
                                rgb,
                                window=win,
                                out_shape=(len(rgb), out_h, out_w),
                                resampling=Resampling.nearest,
                            )
                        image = _pad_to_square(
                            _apply_stretch(data, lows, highs, src.nodata, passthrough)
                        )
                        predictor.set_image(image)
                        # Raster pixels per output pixel (1 unless decimated).
                        sx, sy = w / out_w, h / out_h
                        for start in range(0, len(items), batch_size):
                            chunk = items[start : start + batch_size]
                            boxes = np.array([b for _, b, _ in chunk], dtype=np.float64)
                            masks, scores = predictor.predict_boxes(boxes)
                            for (pos, _, clip), mask, score in zip(
                                chunk, masks, scores, strict=True
                            ):
                                cx0, cy0, cx1, cy1 = clip
                                poly = _mask_to_polygon(
                                    mask[cy0:cy1, cx0:cx1],
                                    _grid_transform(
                                        transform, col + cx0 * sx, row + cy0 * sy, sx, sy
                                    ),
                                )
                                done.add(pos)
                                if poly is None:
                                    n_failed += 1
                                    continue
                                refined_geoms[pos] = poly
                                scores_out[pos] = float(score)
                    except Exception as exc:  # the rest of this window failed
                        n_window_errors += 1
                        n_failed += sum(1 for p, _, _ in items if p not in done)
                        if len(stats["errors"]) < 5:
                            stats["errors"].append(f"{type(exc).__name__}: {exc}")
                        logger.debug("SAM window %s failed: %s", entry["window"], exc)
                    if (i_win + 1) % 50 == 0:
                        logger.info(
                            "SAM refinement: %d/%d windows done (%d refined)",
                            i_win + 1,
                            len(windows),
                            len(refined_geoms),
                        )

        if n_prompts and n_window_errors == len(windows):
            raise RuntimeError(
                f"SAM refinement failed in all {len(windows)} windows; first error: "
                f"{stats['errors'][0] if stats['errors'] else 'unknown'}"
            )

    low_coverage: list[int] = []
    if refined_geoms and overlaps == "trim":
        # Steps 6 and 7: no refined mask takes area of another polygon, and a trimmed
        # mask covering too little of its input polygon is not used.
        kept, trimmed, removed, low_coverage = _trim_overlaps(
            geoms, refined_geoms, scores_out, min_coverage
        )
        used = set(refined_geoms) - set(low_coverage)
        total_area = float(sum(refined_geoms[p].area for p in used))
        dropped = sorted(used - set(kept))
        for pos in dropped:  # nothing left: keep the input geometry
            scores_out[pos] = np.nan
        n_failed += len(dropped)
        refined_geoms = kept
        stats["n_overlap_trimmed"] = len(trimmed)
        stats["overlap_trimmed_fraction"] = round(removed / total_area, 6) if total_area else 0.0
        if trimmed:
            logger.info(
                "SAM refinement: %d masks trimmed where they overlapped other polygons "
                "(%.1f %% of the refined area; %d dropped)",
                len(trimmed),
                100.0 * stats["overlap_trimmed_fraction"],
                len(dropped),
            )
    elif refined_geoms and min_coverage > 0:  # "keep": step 7 on the masks as SAM drew them
        work = _repaired(geoms)
        for pos in sorted(refined_geoms):
            if _covers_too_little(refined_geoms[pos], work[pos], min_coverage):
                low_coverage.append(pos)
                del refined_geoms[pos]
    scores_out[low_coverage] = np.nan  # these rows keep their input geometry
    stats["n_low_coverage"] = len(low_coverage)
    if low_coverage:
        logger.info(
            "SAM refinement: %d masks covered less than %g %% of their input polygon; those "
            "polygons keep their input geometry (sam_min_coverage=%g)",
            len(low_coverage),
            100.0 * min_coverage,
            min_coverage,
        )

    if refined_geoms:
        positions = sorted(refined_geoms)
        new_geoms = gpd.GeoSeries([refined_geoms[p] for p in positions], crs=raster_crs)
        if gdf.crs is not None and raster_crs is not None and gdf.crs != raster_crs:
            new_geoms = new_geoms.to_crs(gdf.crs)
        geom_col = result.geometry.name
        values = list(result.geometry)
        for p, g in zip(positions, new_geoms, strict=True):
            values[p] = g
        result[geom_col] = gpd.GeoSeries(values, index=result.index, crs=gdf.crs)
        refined_flags[positions] = True

    stats["n_refined"] = int(refined_flags.sum())
    stats["n_failed"] = int(n_failed)
    result[REFINED_COLUMN] = refined_flags
    result[SCORE_COLUMN] = scores_out
    result.attrs["sam_stats"] = stats

    if stats["n_failed"]:
        logger.warning(
            "SAM refinement: %d polygons could not be refined and keep their geometry%s",
            stats["n_failed"],
            f" (first error: {stats['errors'][0]})" if stats["errors"] else " (empty masks)",
        )
    logger.info(
        "SAM refinement (%s, %s): %d refined, %d below %d px, %d missing/outside, %d failed, "
        "%d below sam_min_coverage=%g, of %d polygons",
        stats["backend"],
        stats["model"],
        stats["n_refined"],
        stats["n_skipped_small"],
        min_crop_px,
        stats["n_skipped_outside"],
        stats["n_failed"],
        stats["n_low_coverage"],
        min_coverage,
        n_total,
    )
    return result


__all__ = [
    "ALLOWED_SAM_MODELS",
    "CROP_PADDING",
    "DEFAULT_SAM_MODELS",
    "MIN_CROP_SIZE",
    "REFINED_COLUMN",
    "SAM2_MODELS",
    "SCORE_COLUMN",
    "crop_window_px",
    "is_refinable",
    "prefetch",
    "refine_boundaries",
    "resolve_sam_model",
]
