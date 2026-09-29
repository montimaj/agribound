"""
GeoAI Mask R-CNN field instance segmentation (geoai-py).

Uses geoai's instance-segmentation workflow (Wu, 2026, JOSS 11(118):9605):
``geoai.train.instance_segmentation`` runs a torchvision Mask R-CNN
ResNet50-FPN (2 classes: background, field) with a sliding window and
class-aware NMS and writes an instance-id raster; each instance is then
vectorised with ``geoai.utils.raster.raster_to_vector``.

No field-boundary weights are published for geoai: as of 2026-09 the Hugging
Face repository ``giswqs/geoai`` holds building, car, ship, solar-panel,
parking-spot, water and wetland models and DINOv3 backbone weights; the
``field_boundary_detector.pth`` that ``geoai.AgricultureFieldDelineator``
names by default is not among them, and geoai's default detector weights
(``building_footprints_usa.pth``) detect buildings. The engine therefore
needs a checkpoint: from ``fine_tune=True``
with reference boundaries (``agribound.engines.finetune._geoai``) or given as
``engine_params["checkpoint_path"]`` -- a Mask R-CNN ResNet50-FPN state dict
with 2 classes and 3 input channels, as written by geoai's
``train_MaskRCNN_model``/agribound fine-tuning. It never falls back to other
weights.

Input: canonical R, G, B bands with a scene-level 1-99 percentile stretch to
uint8 (:func:`agribound.engines.finetune._data.write_rgb_input`), the same
radiometry as the fine-tuning chips; geoai divides by 255. For
``source="local"`` without ``config.bands`` bands 1, 2, 3 are read as R, G, B.

Scale: torchvision's Mask R-CNN resizes every input image so that its
shorter side is 800 px (``GeneralizedRCNNTransform``, ``min_size=800``) at
training and at inference. The apparent size of a field therefore depends on
the image size: a 256 px training chip is enlarged 3.125 times, a 512 px
inference window 1.5625 times. The inference window defaults to the training
chip size recorded next to the checkpoint so that fields appear at the scale
the model was trained on (:func:`plan_geoai_windows`).
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine
from agribound.registry import ENGINE_REGISTRY

logger = logging.getLogger(__name__)

_NO_WEIGHTS_MESSAGE = (
    "The GeoAI engine needs a field-boundary checkpoint, and none is published "
    "(giswqs/geoai on Hugging Face has no field model; geoai's default weights detect "
    "buildings). Set fine_tune=True with reference_boundaries to train one, or pass "
    "engine_params={'checkpoint_path': '/path/to/best_model.pth'}."
)


def resolve_geoai_checkpoint(params: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return the local checkpoint path and where it came from.

    Keys: ``checkpoint_path`` (or legacy ``model_path``) -- a local file; or
    ``repo_id`` + ``filename`` (or legacy ``model_path``) (+ optional
    ``revision``) -- a file downloaded explicitly from a Hugging Face
    repository.

    Raises
    ------
    RuntimeError
        If no checkpoint is configured.
    ValueError
        If both ``checkpoint_path`` and ``repo_id`` are set (e.g. after
        ``fine_tune=True``, which sets ``checkpoint_path``): it would be
        ambiguous which model runs.
    FileNotFoundError
        If the local file does not exist.
    """
    repo_id = params.get("repo_id")
    local = params.get("checkpoint_path") or params.get("model_path")
    if repo_id and params.get("checkpoint_path"):
        raise ValueError(
            "GeoAI engine_params has both 'checkpoint_path' "
            f"({params['checkpoint_path']!r}) and 'repo_id' ({repo_id!r}); set only one. "
            "Fine-tuning sets checkpoint_path and always starts from the COCO weights."
        )
    if repo_id:
        filename = params.get("filename") or params.get("model_path")
        if not filename:
            raise RuntimeError(
                "engine_params['repo_id'] needs engine_params['filename'] (the weights file)"
            )
        from huggingface_hub import hf_hub_download

        revision = params.get("revision")
        path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
        return path, {"repo_id": repo_id, "filename": filename, "revision": revision}
    if not local:
        raise RuntimeError(_NO_WEIGHTS_MESSAGE)
    path = Path(local).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"GeoAI checkpoint not found: {path}")
    return str(path), {}


#: geoai's default ``window_size`` for ``instance_segmentation``.
_GEOAI_DEFAULT_WINDOW = 512


def maskrcnn_limits() -> dict[str, Any]:
    """Inference settings of the Mask R-CNN that ``geoai.train.instance_segmentation`` builds.

    geoai 0.43.1 builds the model with
    ``torchvision.models.detection.maskrcnn_resnet50_fpn`` without changing
    these arguments, so they are torchvision's ``MaskRCNN`` defaults (read
    from its signature): ``min_size``/``max_size`` (input resizing),
    ``box_score_thresh`` (detections below it are discarded before
    agribound's ``confidence_threshold`` is applied) and
    ``box_detections_per_img`` (at most this many detections per window).
    """
    import inspect

    from torchvision.models.detection import MaskRCNN

    params = inspect.signature(MaskRCNN.__init__).parameters
    return {
        "min_size": params["min_size"].default,
        "max_size": params["max_size"].default,
        "box_score_thresh": params["box_score_thresh"].default,
        "box_detections_per_img": params["box_detections_per_img"].default,
    }


def plan_geoai_windows(
    height: int,
    width: int,
    *,
    window_size: int | None = None,
    overlap: int | None = None,
    batch_size: int = 4,
    training_chip_size: int | None = None,
    min_size: int = 800,
) -> dict[str, Any]:
    """Sliding-window settings for ``geoai.train.instance_segmentation``.

    - Window: ``window_size`` if given, else the training chip size, else
      geoai's default 512. It is never shrunk for small rasters: Mask R-CNN
      resizes each window to ``min_size`` px, so a smaller window would
      enlarge the fields; geoai zero-pads windows that extend past the
      raster instead. A window that differs from a known training chip size
      is logged at WARNING.
    - Overlap: ``overlap`` if given, else half the window; it must be at
      least 0 and smaller than the window.
    - Batch size: geoai 0.43.1 processes its last, partial batch of windows
      only at the last grid position, which it skips when a raster side is
      shorter than the window but longer than the overlap, so those windows
      would be dropped. For rasters with a side shorter than the window the
      batch size is set to 1 (every window is then processed
      when it is read). This changes only speed: geoai pads every window to
      the same size and the model is in evaluation mode, so each window's
      detections do not depend on the other windows in its batch (up to
      floating-point rounding).

    Returns
    -------
    dict
        ``window_size``, ``overlap``, ``batch_size``,
        ``requested_batch_size``, ``window_source`` (``"engine_params"``,
        ``"training_chip_size"`` or ``"geoai_default"``),
        ``training_chip_size`` and ``resize_factor`` (``min_size /
        window_size``, the enlargement Mask R-CNN applies to each window).

    Raises
    ------
    ValueError
        For a non-positive window or batch size, or an overlap outside
        ``[0, window)``.
    """
    if window_size is not None:
        window, source = int(window_size), "engine_params"
    elif training_chip_size:
        window, source = int(training_chip_size), "training_chip_size"
    else:
        window, source = _GEOAI_DEFAULT_WINDOW, "geoai_default"
    if window <= 0:
        raise ValueError(f"GeoAI window_size must be positive, got {window}")
    ov = window // 2 if overlap is None else int(overlap)
    if not 0 <= ov < window:
        raise ValueError(f"GeoAI overlap must be in [0, window_size={window}), got {ov}")
    if min(int(height), int(width)) <= 2 * ov - window:
        # geoai's window count ceil((side - overlap) / (window - overlap)) + 1 is then <= 0.
        raise ValueError(
            f"GeoAI overlap {ov} is too large for a {height} x {width} px raster with "
            f"window_size {window}: geoai would process no window. Use overlap <= "
            f"{window // 2}."
        )
    requested = int(batch_size)
    if requested < 1:
        raise ValueError(f"GeoAI batch_size must be >= 1, got {requested}")
    batch = requested
    if min(int(height), int(width)) < window and batch != 1:
        batch = 1
        logger.info(
            "Raster %d x %d px has a side shorter than the %d px window; GeoAI runs with "
            "batch_size=1 so that geoai 0.43.1 processes every window",
            height,
            width,
            window,
        )
    if training_chip_size and window != int(training_chip_size):
        logger.warning(
            "GeoAI window_size %d px differs from the %d px chips the checkpoint was trained "
            "on. Mask R-CNN resizes every image to a %d px shorter side, so fields appear at "
            "%.2fx their training scale.",
            window,
            int(training_chip_size),
            min_size,
            int(training_chip_size) / window,
        )
    elif not training_chip_size and window_size is None:
        logger.info(
            "The checkpoint records no training chip size; GeoAI uses geoai's default "
            "window_size=%d. Set engine_params['window_size'] to the tile size the model was "
            "trained on (Mask R-CNN resizes each window to %d px).",
            window,
            min_size,
        )
    return {
        "window_size": window,
        "overlap": ov,
        "batch_size": batch,
        "requested_batch_size": requested,
        "window_source": source,
        "training_chip_size": int(training_chip_size) if training_chip_size else None,
        "resize_factor": float(min_size) / window,
    }


def window_edges(size: int, window: int, overlap: int) -> list[int]:
    """Pixel offsets of the interior window edges geoai's sliding window uses on one axis.

    geoai 0.43.1 places windows at ``min(k * (window - overlap), size - window)`` (at least
    0) for ``k = 0 .. ceil((size - overlap) / (window - overlap))``; each window spans
    ``[start, start + window)``. The raster borders (0 and *size*) are left out.
    """
    stride = window - overlap
    steps = math.ceil((size - overlap) / stride)
    starts = {max(0, min(k * stride, size - window)) for k in range(steps + 1)}
    edges = starts | {s + window for s in starts}
    return sorted(e for e in edges if 0 < e < size)


def _nearest_across(band: np.ndarray, g: int) -> tuple[np.ndarray, ...]:
    """Nearest instance on each side of a seam in a ``(n, 2 g + 2)`` band.

    Columns ``0 .. g`` lie before the seam and ``g + 1 .. 2 g + 1`` after it. Returns the
    nearest non-zero id before and after the seam for every row and its distance (in
    pixels) from the seam; 0 where there is none within *g* pixels.
    """
    before = band[:, : g + 1][:, ::-1]
    after = band[:, g + 1 :]
    has_b, has_a = (before > 0).any(axis=1), (after > 0).any(axis=1)
    db, da = np.argmax(before > 0, axis=1), np.argmax(after > 0, axis=1)
    rows = np.arange(band.shape[0])
    id_b = np.where(has_b, before[rows, db], 0).astype(np.int64)
    id_a = np.where(has_a, after[rows, da], 0).astype(np.int64)
    return id_b, db, id_a, da


def merge_window_seams(
    instance_path: str,
    output_path: str,
    window: int,
    overlap: int,
    min_seam_px: int = 16,
    min_seam_fraction: float = 0.5,
    max_gap_px: int = 2,
) -> dict[str, Any]:
    """Join instances that one field split into at geoai's window edges.

    geoai paints every detection's full mask into one instance raster and keeps the
    partial detections of a field from overlapping windows when their boxes overlap by less
    than the NMS threshold, so a field larger than the overlap is split along a window edge
    (an axis-aligned line at a window start or end), sometimes with a thin gap of
    background where neither partial mask reaches the edge. For every interior window edge
    the nearest instances on its two sides are compared row by row (at most *max_gap_px*
    background pixels between them): two different instances that meet across the edge
    along at least *min_seam_px* pixels, and along at least *min_seam_fraction* of the
    shorter of their two runs on that edge, are joined (union-find, so a field cut by
    several edges becomes one instance). Gaps of at most *max_gap_px* pixels across an
    edge between two parts of one (joined) instance are then filled, so no slit is left.
    Instances that meet anywhere else are left alone.

    Writes the relabelled raster to *output_path* and returns counts for ``engine_meta``.
    """
    import rasterio
    from rasterio.windows import Window

    g = max(int(max_gap_px), 0)
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent.get(a, a) != a:
            parent[a] = parent.get(parent[a], parent[a])
            a = parent[a]
        return a

    def band_windows(width: int, height: int):
        """(window, transpose) for every interior window edge, both axes."""
        for x in window_edges(width, window, overlap):
            lo, hi = max(x - 1 - g, 0), min(x + 1 + g, width)
            yield Window(lo, 0, hi - lo, height), False, x - lo - 1
        for y in window_edges(height, window, overlap):
            lo, hi = max(y - 1 - g, 0), min(y + 1 + g, height)
            yield Window(0, lo, width, hi - lo), True, y - lo - 1

    def as_band(arr: np.ndarray, transpose: bool, before: int) -> tuple[np.ndarray, int]:
        band = arr.T if transpose else arr
        # Re-centre so that the seam lies between columns gg and gg + 1.
        gg = min(before, band.shape[1] - before - 2)
        return band[:, before - gg : before + gg + 2], gg

    with rasterio.open(instance_path) as src:
        profile = src.profile.copy()
        width, height = src.width, src.height
        for win, transpose, before in band_windows(width, height):
            band, gg = as_band(src.read(1, window=win), transpose, before)
            id_b, db, id_a, da = _nearest_across(band, gg)
            meet = (id_b > 0) & (id_a > 0) & (id_b != id_a) & (db + da <= g)
            if not meet.any():
                continue
            key = np.stack(
                [np.minimum(id_b[meet], id_a[meet]), np.maximum(id_b[meet], id_a[meet])], 1
            )
            uniq, counts = np.unique(key, axis=0, return_counts=True)
            run = np.bincount(np.r_[id_b[id_b > 0], id_a[id_a > 0]])
            for (i, j), n in zip(uniq.tolist(), counts.tolist(), strict=True):
                shorter = min(run[i], run[j])
                if n >= min_seam_px and n >= min_seam_fraction * shorter:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[max(ri, rj)] = min(ri, rj)

        merged_ids = {k for k in parent if find(k) != k}
        lut = None
        if parent:
            lut = np.arange(max(max(parent), max(parent.values())) + 1, dtype=np.int64)
            for k in list(parent):
                lut[k] = find(k)
        with rasterio.open(output_path, "w", **profile) as dst:
            for _, win in src.block_windows(1):
                block = src.read(1, window=win)
                if lut is not None:
                    inside = (block > 0) & (block < len(lut))
                    block = block.copy()
                    block[inside] = lut[block[inside]].astype(block.dtype)
                dst.write(block, 1, window=win)

    n_filled = 0
    if g > 0:
        with rasterio.open(output_path, "r+") as dst:
            for win, transpose, before in band_windows(width, height):
                arr = dst.read(1, window=win)
                band, gg = as_band(arr, transpose, before)
                id_b, db, id_a, da = _nearest_across(band, gg)
                fill = (id_b > 0) & (id_b == id_a) & (db + da > 0) & (db + da <= g)
                if not fill.any():
                    continue
                band = band.copy()
                for r in np.flatnonzero(fill):
                    band[r, gg - db[r] + 1 : gg + 1 + da[r]] = id_b[r]
                    n_filled += int(db[r] + da[r])
                full = arr.T.copy() if transpose else arr.copy()
                full[:, before - gg : before + gg + 2] = band
                dst.write(full.T if transpose else full, 1, window=win)
    return {
        "seam_merge": True,
        "seam_min_px": int(min_seam_px),
        "seam_min_fraction": float(min_seam_fraction),
        "seam_max_gap_px": g,
        "n_instances_merged_at_seams": len(merged_ids),
        "n_seam_gap_pixels_filled": n_filled,
    }


def instances_to_polygons(instance_path: str, score_path: str | None = None) -> gpd.GeoDataFrame:
    """Vectorise an instance-id raster into one (multi)polygon per instance.

    Uses ``geoai.utils.raster.raster_to_vector`` (connected regions of equal
    id), dissolves the regions of each id, and adds the instance's detection
    ``score`` (constant per instance in geoai's score raster) when
    *score_path* is given.
    """
    import rasterio
    from geoai.utils.raster import raster_to_vector

    gdf = raster_to_vector(
        instance_path, output_path=None, threshold=0, min_area=0, attribute_name="instance_id"
    )
    if len(gdf) == 0:
        with rasterio.open(instance_path) as src:
            crs = src.crs
        return gpd.GeoDataFrame({"instance_id": [], "geometry": []}, geometry="geometry", crs=crs)
    gdf = gdf.dissolve(by="instance_id", as_index=False)
    if score_path is not None and Path(score_path).exists():
        with rasterio.open(instance_path) as src_i, rasterio.open(score_path) as src_s:
            ids = src_i.read(1).astype(np.int64).ravel()
            scores = src_s.read(1).astype(np.float64).ravel()
        fg = ids > 0
        counts = np.bincount(ids[fg])
        sums = np.bincount(ids[fg], weights=scores[fg], minlength=len(counts))
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = sums / counts
        gdf["score"] = [
            float(mean[i]) if i < len(mean) else float("nan") for i in gdf["instance_id"]
        ]
    return gdf


class GeoAIEngine(DelineationEngine):
    """Field delineation with a fine-tuned geoai Mask R-CNN (see module docstring)."""

    name = "geoai"
    supported_sources = list(ENGINE_REGISTRY["geoai"]["supported_sources"])
    requires_bands = list(ENGINE_REGISTRY["geoai"]["requires_bands"])

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run geoai instance segmentation on a composite.

        Parameters
        ----------
        raster_path : str
            Composite GeoTIFF.
        config : AgriboundConfig
            Pipeline configuration. ``engine_params``:

            - ``checkpoint_path``: Mask R-CNN weights (``.pth``); see
              :func:`resolve_geoai_checkpoint` for the Hugging Face option.
            - ``window_size``: sliding window in pixels (default: the
              training chip size recorded next to the checkpoint, else
              geoai's 512). Keep it equal to the training chip size: Mask
              R-CNN resizes each window to an 800 px shorter side, so the
              window size sets the apparent field size (a different value is
              logged at WARNING). Windows that extend past a small raster are
              zero-padded by geoai. See :func:`plan_geoai_windows`.
            - ``overlap``: window overlap in pixels (default: half the
              window).
            - ``batch_size``: windows per forward pass (default 4; 1 when a
              raster side is shorter than the window, see
              :func:`plan_geoai_windows`).
            - ``confidence_threshold`` (default 0.5) and ``nms_threshold``
              (default 0.3, geoai's cross-window NMS).
            - ``merge_window_seams`` (default *True*), ``seam_min_px``
              (default 16) and ``seam_max_gap_px`` (default 2): join
              instances that a field was split into at the window edges and
              fill thin gaps along those edges (see
              :func:`merge_window_seams`).
            - ``clean_instance_mask``: *False* (default), *True* or a dict of
              keyword arguments for ``geoai.utils.raster.clean_instance_mask``
              (removes small instances, fills holes, smooths boundaries;
              runtime grows with instances x pixels).

        Returns
        -------
        geopandas.GeoDataFrame
            One polygon per detected field with ``instance_id`` and ``score``
            columns and ``gdf.attrs["engine_meta"]``.

        Notes
        -----
        geoai does not expose two limits of torchvision's Mask R-CNN
        (:func:`maskrcnn_limits`, recorded in ``engine_meta``): at most 100
        detections are kept per window (``box_detections_per_img``), so
        where more fields fit in one window (small fields at 10-30 m) the
        rest are lost, and detections scoring below 0.05
        (``box_score_thresh``) are discarded, so a ``confidence_threshold``
        below 0.05 has the same effect as 0.05 (logged at WARNING). For
        dense small fields, fine-tune with a smaller
        ``engine_params["chip_size"]``; inference then uses windows of the
        same size.
        """
        try:
            import torch  # noqa: F401
            from geoai.train import instance_segmentation
        except ImportError:
            raise ImportError(
                "geoai-py is required for the GeoAI engine. "
                "Install with: pip install agribound[geoai]"
            ) from None
        import rasterio

        from agribound._cache import cache_path
        from agribound.engines.base import get_canonical_band_indices
        from agribound.engines.finetune._data import (
            file_sha256,
            mask_invalid_predictions,
            package_versions,
            raster_fingerprint,
            read_json,
            read_training_meta,
            write_json,
            write_rgb_input,
        )

        self.validate_input(raster_path, config)
        params = config.engine_params
        checkpoint, hub_info = resolve_geoai_checkpoint(params)
        indices = get_canonical_band_indices(config.source, ["R", "G", "B"], bands=config.bands)
        confidence = float(params.get("confidence_threshold", 0.5))
        nms = float(params.get("nms_threshold", 0.3))
        limits = maskrcnn_limits()
        if confidence < limits["box_score_thresh"]:
            logger.warning(
                "GeoAI confidence_threshold %.3g is below Mask R-CNN's box_score_thresh %.3g, "
                "which discards lower-scoring detections first; it acts as %.3g",
                confidence,
                limits["box_score_thresh"],
                limits["box_score_thresh"],
            )
        training = read_training_meta(checkpoint)
        with rasterio.open(raster_path) as src:
            height, width = src.height, src.width
        plan = plan_geoai_windows(
            height,
            width,
            window_size=params.get("window_size"),
            overlap=params.get("overlap"),
            batch_size=int(params.get("batch_size", 4)),
            training_chip_size=training.get("chip_size"),
            min_size=int(limits["min_size"]),
        )
        window, overlap, batch_size = plan["window_size"], plan["overlap"], plan["batch_size"]
        clean = params.get("clean_instance_mask", False)

        device = config.resolve_device()
        if device == "mps":
            logger.warning(
                "GeoAI Mask R-CNN runs on CPU instead of MPS: torchvision Mask R-CNN on MPS "
                "reports Metal command-buffer errors and its detections differ from CPU "
                "(checked with torch 2.10 and geoai-py 0.43.1)"
            )
            device = "cpu"

        raster_fp = raster_fingerprint(raster_path)
        rgb_path = cache_path(config, "geoai_rgb", ".tif", raster_fp, f"bands={indices}", "uint8")
        rgb_info_path = rgb_path.with_suffix(".json")
        if rgb_path.exists() and rgb_info_path.exists():
            rgb_info = read_json(rgb_info_path)
        else:
            rgb_info = write_rgb_input(raster_path, rgb_path, indices, unit_float=False)
            write_json(rgb_info_path, rgb_info)

        inst_path = cache_path(
            config,
            "geoai_instances",
            ".tif",
            raster_fp,
            raster_fingerprint(checkpoint),
            f"bands={indices}",
            f"window={window}",
            f"overlap={overlap}",
            f"conf={confidence}",
            f"nms={nms}",
        )
        score_path = inst_path.with_name(f"{inst_path.stem}_score{inst_path.suffix}")
        inst_info_path = inst_path.with_suffix(".json")
        run_info: dict[str, Any] = {"device": device}
        if inst_path.exists() and score_path.exists() and inst_info_path.exists():
            logger.info("Using cached GeoAI instances: %s", inst_path)
            run_info = {**read_json(inst_info_path), "cache_reused": True}
        else:
            logger.info("Running GeoAI instance segmentation (device=%s)", device)
            instance_segmentation(
                input_path=str(rgb_path),
                output_path=str(inst_path),
                model_path=checkpoint,
                window_size=window,
                overlap=overlap,
                confidence_threshold=confidence,
                nms_threshold=nms,
                batch_size=batch_size,
                num_channels=3,
                num_classes=2,
                vectorize=False,
                device=device,
            )
            mask_invalid_predictions(inst_path, raster_path, indices)
            write_json(inst_info_path, run_info)

        vector_source = str(inst_path)
        seam_info: dict[str, Any] = {"seam_merge": False}
        if params.get("merge_window_seams", True):
            seam_min = int(params.get("seam_min_px", 16))
            seam_gap = int(params.get("seam_max_gap_px", 2))
            merged_path = inst_path.with_name(
                f"{inst_path.stem}_seams{seam_min}_gap{seam_gap}{inst_path.suffix}"
            )
            seam_json = merged_path.with_suffix(".json")
            if merged_path.exists() and seam_json.exists():
                seam_info = read_json(seam_json)
            else:
                seam_info = merge_window_seams(
                    str(inst_path),
                    str(merged_path),
                    window,
                    overlap,
                    min_seam_px=seam_min,
                    max_gap_px=seam_gap,
                )
                write_json(seam_json, seam_info)
            logger.info(
                "GeoAI: joined %d instances split at the %d px window edges",
                seam_info["n_instances_merged_at_seams"],
                window,
            )
            vector_source = str(merged_path)
        if clean:
            from geoai.utils.raster import clean_instance_mask

            kwargs = dict(clean) if isinstance(clean, dict) else {}
            # Clean the seam-merged raster when there is one, so the join is kept.
            src = Path(vector_source)
            cleaned = src.with_name(f"{src.stem}_cleaned{src.suffix}")
            vector_source = clean_instance_mask(str(src), str(cleaned), **kwargs)

        gdf = instances_to_polygons(vector_source, str(score_path))
        gdf.attrs["engine_meta"] = {
            "backend": "geoai.instance_segmentation",
            **package_versions("geoai-py", "torch", "torchvision"),
            "model": "maskrcnn_resnet50_fpn",
            "num_classes": 2,
            "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            **({"hub": hub_info} if hub_info else {}),
            "training": training or None,
            "band_indices": indices,
            "input": rgb_info,
            "window_size": window,
            "overlap": overlap,
            "window_source": plan["window_source"],
            "training_chip_size": plan["training_chip_size"],
            "resize_factor": plan["resize_factor"],
            "confidence_threshold": confidence,
            "nms_threshold": nms,
            "batch_size": batch_size,
            "requested_batch_size": plan["requested_batch_size"],
            "maskrcnn_limits": limits,
            "clean_instance_mask": clean,
            **seam_info,
            **run_info,
        }
        if len(gdf) == 0:
            logger.warning("No field boundaries detected by GeoAI")
        logger.info("GeoAI delineated %d field boundaries", len(gdf))
        return gdf

    @classmethod
    def prefetch(cls, config: AgriboundConfig) -> list[str]:
        """Download the weights GeoAI inference and fine-tuning need offline.

        - torchvision's COCO Mask R-CNN ResNet50-FPN weights
          (``MaskRCNN_ResNet50_FPN_Weights.DEFAULT``): geoai builds its model
          with them before loading a checkpoint, and fine-tuning starts from
          them. They go to ``$TORCH_HOME/hub/checkpoints``.
        - The Hugging Face checkpoint, when ``engine_params["repo_id"]`` is
          set.

        Returns
        -------
        list[str]
            Local paths.
        """
        import os

        import torch
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        weights.get_state_dict(progress=True)
        paths = [os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(weights.url))]
        params = config.engine_params
        if params.get("repo_id"):
            paths.append(resolve_geoai_checkpoint(params)[0])
        elif params.get("checkpoint_path") and Path(params["checkpoint_path"]).is_file():
            paths.append(str(Path(params["checkpoint_path"]).resolve()))
        return paths
