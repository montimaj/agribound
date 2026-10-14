"""
Delineate-Anything (Ultralytics YOLO11-seg) fine-tuning.

:func:`_finetune_yolo` is called by :func:`agribound.engines.finetune.fine_tune`
with the directory written by
:func:`agribound.engines.finetune._data._prepare_training_data` for
``engine="delineate-anything"``: georeferenced uint8 R, G, B chips with the
scene-level percentile stretch already applied (the stretch the engine
applies at inference: 1-99 %, or ``engine_params["stretch_percentiles"]``),
cut from the raster the native backend reads (resampled to
``engine_params["processing_gsd_m"]`` when that is set), split into
``images/`` (train) and ``val_images/`` (validation) by
:func:`agribound.engines.finetune._data.assign_splits`. The training
metadata records both options, and the native backend logs a WARNING when it
runs the checkpoint with other values.

Labels
------
YOLO segmentation labels are built from the reference polygons themselves
(``config.reference_boundaries``), not from the rasterised masks: every
reference polygon is clipped to the chip footprint and to the chip's valid
pixels (read from the chip's ``_data`` mask, where invalid pixels are
``IGNORE_INDEX``; see :func:`chip_valid_pixels`), and every resulting polygon
part of at least ``yolo_min_instance_px`` pixels (default 4) becomes one
instance of class 0 (``field``). Touching fields therefore stay separate
instances. YOLO labels hold one ring per instance, so the holes of a polygon
are joined to its exterior by zero-width bridges (the fill of such a ring
leaves the holes empty); a field split into several parts by the chip edge
gives one instance per part.

Slivers get no instance: a part that a side of the chip frame cuts from its
reference polygon (the part reaches that side and the polygon extends beyond
it) and that reaches at most ``yolo_sliver_depth_px`` pixels of the model
input into the chip from that side (default 16,
:data:`DEFAULT_SLIVER_DEPTH_PX`; see :func:`frame_cut_depth`). Such a strip
shows little of its field, and its edge along the frame is not in the image;
a model fine-tuned on a few dozen chips can learn to draw strips along the
frame of every tile. 16 px is the depth at which upstream DelAnyFlow's
tile-seam merge treats a tile-cut piece as "only a strip along the border"
(``seam_max_strip_depth_px``, also used by agribound's native backend). Parts
cut by invalid pixels rather than by the frame are kept, and the chips
overlap (half the chip size apart by default, see
:func:`agribound.engines.finetune._data.chip_stride`), so a field cut to a
sliver in one chip appears whole, or deeper, in a neighbouring chip.

Scale and model
---------------
Images are upsampled by the Delineate-Anything super-resolution factor (2
with bicubic interpolation for a GSD of 4 m or more, else 1;
``engine_params["super_resolution"]`` overrides it), computed from the chip's
pixel size exactly as at inference
(:func:`agribound.engines.delineate_anything.pixel_size_m`, the mean of the
east-west and north-south pixel sides), and ``imgsz`` is the upsampled chip
size, so the model sees the same pixel scale during training and inference.
``imgsz`` is 512 (the model input size at inference) when the chip size times
the factor is 512. ``_data`` chooses the default chip size (256 px at 4 m or
more, 512 px below) with the same rule,
:func:`~agribound.engines.delineate_anything.pixel_size_m`, but for the whole
raster, while the factor here comes from the first training chip. For a
geographic raster the two differ with the latitude of the chip (about 0.7 %
per degree between the raster centre and the chip at 45°), so a pixel size
close to 4 m can still put them on different sides of the threshold, and an
explicit ``engine_params["chip_size"]`` can also give an ``imgsz`` of 256 or
1024. A
WARNING is then logged and the training metadata records
``imgsz_matches_model_input: false``. The base weights are
the selected Delineate-Anything model
(``engine_params["da_model"]``, default ``large_v2``), downloaded at its
pinned revision with its SHA-256 checked
(:func:`agribound.engines.delineate_anything.download_da_weights`).

Training settings: ``seed=config.seed`` and ``deterministic=True``,
``mosaic=0.0`` (as in the Delineate Anything v2 recipe), ``optimizer="AdamW"``
with ``lr0`` 0.002 (the value Ultralytics' ``optimizer="auto"`` picks for one
class and at most 10,000 iterations; ``engine_params["yolo_lr0"]``
overrides) and ``warmup_bias_lr=0`` (what ``optimizer="auto"`` sets;
Ultralytics' default of 0.1 applies only to a named optimizer;
``yolo_warmup_bias_lr`` overrides), horizontal and vertical flips with probability 0.5,
``plots=False``, ``epochs=config.fine_tune_epochs``, ``batch`` 16
(``yolo_batch``), ``nbs`` equal to the batch (``yolo_nbs``; see below),
``workers=config.n_workers``. Ultralytics selects the checkpoint with the best
validation fitness (``best.pt``).

Small training sets. Ultralytics steps the optimizer once every
``round(nbs / batch)`` batches (at least 1), accumulating the gradients in
between, and during the warmup (3 epochs) it raises that number from 1. With
its default ``nbs`` of 64 and batches of 16, 22 training chips (two batches
per epoch) got 11 optimizer steps in 20 epochs. agribound sets ``nbs`` to the
batch size, so the optimizer steps after every batch (40 steps there);
``yolo_nbs=64`` restores Ultralytics' accumulation. With ``yolo_batch=-1``
(Ultralytics' AutoBatch) the batch size is not known in advance, and ``nbs``
stays at Ultralytics' 64 unless ``yolo_nbs`` is set. The default ``lr0`` can
also move the pretrained weights too far for them to recover. On 62 training
chips of 512 px (SPOT 6/7 panchromatic, an oil palm estate in Ghana; example
23), with chips that did not overlap and ``nbs`` 64 (the recipe of
2026-10-05), the validation mask mAP50 at the default ``lr0`` was 0.02 after
the first epoch (the checkpoint Ultralytics kept) and 0.00 from the third
epoch on, and 0.47 after 20 epochs with ``yolo_lr0=1e-4``, against 0.17 for
the released weights (with the ``warmup_bias_lr`` of 0.1 that agribound <=
1.0.1 used, the default ``lr0`` scored 0.00). Use a smaller ``yolo_lr0`` for
small training sets, and compare the validation scores with those of the
released weights.

``RECIPE_VERSION`` is part of the fine-tuning cache keys, so checkpoints
trained with an earlier recipe are not reused.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from agribound.config import AgriboundConfig

logger = logging.getLogger(__name__)

#: Default minimum size of a label instance, in chip pixels.
DEFAULT_MIN_INSTANCE_PX = 4.0

#: Default AdamW learning rate (Ultralytics ``optimizer="auto"`` for nc=1).
DEFAULT_LR0 = 0.002

#: Default warmup learning rate of the bias parameters. Ultralytics'
#: ``optimizer="auto"`` sets 0 ("no higher than 0.01 for Adam"); its default of
#: 0.1, which applies to a named optimizer, perturbs the pretrained biases during
#: warmup with AdamW (agribound <= 1.0.1 used it).
DEFAULT_WARMUP_BIAS_LR = 0.0

#: Default depth, in pixels of the model input, up to which a label part cut by the chip
#: frame is a sliver and is dropped (``yolo_sliver_depth_px``; 0 keeps every part): upstream
#: DelAnyFlow's ``seam_max_strip_depth_px``, at which its tile-seam merge treats a tile-cut
#: piece as only a strip along the border.
DEFAULT_SLIVER_DEPTH_PX = 16.0

#: Ultralytics' default nominal batch size (``nbs``), kept when the batch size is chosen by
#: Ultralytics' AutoBatch (``yolo_batch=-1``) and ``yolo_nbs`` is not set.
_ULTRALYTICS_NBS = 64

#: Tolerance, in chip pixels, of the test that a label part reaches a side of the chip.
_FRAME_TOL_PX = 1e-6

#: Version of the fine-tuning recipe (part of the cache keys). 2: ``warmup_bias_lr=0``.
#: 3: chips half a chip apart, split by chip-size cells; ``nbs`` equal to the batch size;
#: sliver labels dropped; pixels that are 0 in every band are nodata in the chips.
RECIPE_VERSION = 3


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def bridge_holes(polygon: Any) -> np.ndarray:
    """Return the vertices ``(N, 2)`` of one ring outlining *polygon*, holes bridged.

    The exterior is oriented counter-clockwise and the holes clockwise; each
    hole is spliced into the ring at its vertex closest to the ring, going
    out and back along the same zero-width bridge. Filling the returned ring
    (e.g. with ``cv2.fillPoly``, as Ultralytics does) leaves the holes empty.
    The last vertex is not a repeat of the first.
    """
    from scipy.spatial import cKDTree
    from shapely.geometry.polygon import orient

    poly = orient(polygon, sign=1.0)
    ring = np.asarray(poly.exterior.coords, dtype=float)[:-1]
    for interior in poly.interiors:
        hole = np.asarray(interior.coords, dtype=float)[:-1]
        if len(hole) < 3:
            continue
        dist, idx = cKDTree(ring).query(hole)
        j = int(np.argmin(dist))
        i = int(idx[j])
        hole_from_j = np.roll(hole, -j, axis=0)
        ring = np.concatenate(
            [ring[: i + 1], hole_from_j, hole_from_j[:1], ring[i : i + 1], ring[i + 1 :]]
        )
    return ring


def _polygon_parts(geom: Any) -> list[Any]:
    """Polygons contained in a (multi)polygon or geometry collection."""
    if geom is None or geom.is_empty:
        return []
    kind = geom.geom_type
    if kind == "Polygon":
        return [geom]
    if kind in ("MultiPolygon", "GeometryCollection"):
        parts = []
        for part in geom.geoms:
            parts.extend(_polygon_parts(part))
        return parts
    return []


def frame_cut_depth(part: Any, polygon: Any, width: int, height: int) -> float | None:
    """How far a label part reaches into the chip from the chip sides that cut it.

    *part* is a part of *polygon* inside a ``width`` x ``height`` chip, both
    in chip pixel coordinates (x right, y down). A side of the chip frame cuts
    the part when the part reaches that side and the polygon it belongs to
    (for a multi-part *polygon*, the part's own component) extends beyond it.
    The depth from a side is the largest distance of the part from that side
    (from the left side: the part's largest x).

    Returns
    -------
    float or None
        The smallest depth over the sides that cut *part*; None when no side
        cuts it (the part lies inside the chip, or ends at invalid pixels).
    """
    if polygon.geom_type != "Polygon":
        inside = part.representative_point()
        polygon = next((p for p in _polygon_parts(polygon) if p.intersects(inside)), polygon)
    minx, miny, maxx, maxy = part.bounds
    gminx, gminy, gmaxx, gmaxy = polygon.bounds
    tol = _FRAME_TOL_PX
    depths = []
    if minx <= tol and gminx < -tol:
        depths.append(maxx)
    if maxx >= width - tol and gmaxx > width + tol:
        depths.append(width - minx)
    if miny <= tol and gminy < -tol:
        depths.append(maxy)
    if maxy >= height - tol and gmaxy > height + tol:
        depths.append(height - miny)
    return min(depths) if depths else None


def chip_labels(
    polygons: list[Any],
    chip_transform: Any,
    width: int,
    height: int,
    valid_area_px: Any | None = None,
    min_area_px: float = DEFAULT_MIN_INSTANCE_PX,
    sliver_depth_px: float = 0.0,
    counts: dict[str, int] | None = None,
) -> list[str]:
    """YOLO segmentation label lines for reference polygons over one chip.

    Parameters
    ----------
    polygons : list of shapely geometries
        Reference polygons in the chip CRS.
    chip_transform : affine.Affine
        Chip transform (pixel -> CRS).
    width, height : int
        Chip size in pixels.
    valid_area_px : shapely geometry or None
        Valid image area in pixel coordinates; labels are clipped to it.
    min_area_px : float
        Minimum area, in square pixels, of a label polygon part.
    sliver_depth_px : float
        Parts cut by the chip frame that reach at most this many chip pixels
        into the chip from a side that cuts them (:func:`frame_cut_depth`)
        are dropped; 0 keeps them.
    counts : dict or None
        If given, ``counts["n_slivers"]`` is increased by the number of
        dropped slivers.

    Returns
    -------
    list[str]
        ``"0 x1 y1 x2 y2 ..."`` lines with coordinates normalised to [0, 1]
        (x by *width*, y by *height*, y down), one per polygon part.
    """
    import shapely
    from shapely.affinity import affine_transform
    from shapely.geometry import box

    inv = ~chip_transform
    matrix = [inv.a, inv.b, inv.d, inv.e, inv.c, inv.f]
    frame = box(0.0, 0.0, float(width), float(height))
    clip = frame if valid_area_px is None else frame.intersection(valid_area_px)
    lines: list[str] = []
    for geom in polygons:
        if geom is None or geom.is_empty:
            continue
        pixel_geom = shapely.make_valid(affine_transform(geom, matrix))
        for part in _polygon_parts(pixel_geom.intersection(clip)):
            if part.area < min_area_px:
                continue
            if sliver_depth_px > 0:
                depth = frame_cut_depth(part, pixel_geom, width, height)
                if depth is not None and depth <= sliver_depth_px:
                    if counts is not None:
                        counts["n_slivers"] = counts.get("n_slivers", 0) + 1
                    continue
            ring = bridge_holes(part)
            if len(ring) < 3:
                continue
            xy = np.column_stack(
                [np.clip(ring[:, 0] / width, 0.0, 1.0), np.clip(ring[:, 1] / height, 0.0, 1.0)]
            )
            lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in xy))
    return lines


def _mask_path(image_path: Path) -> Path:
    """Mask chip written by ``_data`` next to an image chip (``masks/`` or ``val_masks/``)."""
    folder = image_path.parent
    return folder.with_name(folder.name.replace("images", "masks")) / image_path.name


def chip_valid_pixels(image: np.ndarray, mask_path: Path | None = None) -> np.ndarray:
    """Boolean ``(H, W)`` mask of the valid pixels of a training chip.

    ``_data`` writes the semantic mask of every chip with invalid image pixels
    set to :data:`agribound.engines.finetune._data.IGNORE_INDEX`; when that
    mask exists it defines validity. Without it, pixels whose bands are all 0
    (the value ``_data`` gives invalid pixels in the stretched image) are
    treated as invalid.
    """
    if mask_path is not None and Path(mask_path).is_file():
        import rasterio

        from agribound.engines.finetune._data import IGNORE_INDEX

        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"Mask chip {mask_path} has shape {mask.shape}, image chip {image.shape[1:]}"
            )
        return mask != IGNORE_INDEX
    return ~np.all(image == 0, axis=0)


def _valid_area_px(valid: np.ndarray) -> Any | None:
    """Polygon (pixel coordinates) of the True pixels of *valid*; None if all are valid."""
    from rasterio.features import shapes
    from shapely.geometry import Polygon, shape
    from shapely.ops import unary_union

    if valid.all():
        return None
    parts = [shape(g) for g, v in shapes(valid.astype(np.uint8), mask=valid) if v == 1]
    return unary_union(parts) if parts else Polygon()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _read_chip(path: Path) -> tuple[np.ndarray, Any, Any]:
    import rasterio

    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"Training chip {path} has {src.count} bands; R, G, B are required")
        data = src.read([1, 2, 3])
        return data, src.transform, src.crs


def prepare_yolo_dataset(
    train_dir: Path,
    config: AgriboundConfig,
    out_dir: Path,
    super_resolution: int,
    min_area_px: float = DEFAULT_MIN_INSTANCE_PX,
    sliver_depth_px: float = DEFAULT_SLIVER_DEPTH_PX,
) -> dict[str, Any]:
    """Write the Ultralytics dataset (PNG images, label files, ``data.yaml``).

    Parameters
    ----------
    train_dir : Path
        Output of ``_prepare_training_data`` (uint8 R, G, B chips in
        ``images/`` and ``val_images/``).
    config : AgriboundConfig
        Provides ``reference_boundaries``.
    out_dir : Path
        Dataset directory (replaced if it exists).
    super_resolution : int
        Image upsampling factor (bicubic).
    min_area_px : float
        Minimum label polygon area in chip pixels.
    sliver_depth_px : float
        Depth, in pixels of the model input (chip pixels times
        *super_resolution*), up to which a label part cut by the chip frame
        is dropped, in both splits (see :func:`chip_labels`); 0 keeps them.

    Returns
    -------
    dict
        ``data_yaml``, ``imgsz``, ``chip_size`` and per-split image,
        instance and dropped-sliver counts (``n_<split>_images``,
        ``n_<split>_instances``, ``n_<split>_slivers``).
    """
    import cv2
    import yaml
    from PIL import Image

    from agribound.engines.finetune._data import split_files
    from agribound.io.vector import read_vector

    splits = {s: split_files(train_dir, s, "images") for s in ("train", "val")}
    if not splits["train"] or not splits["val"]:
        raise RuntimeError(
            f"{train_dir} must contain training and validation chips (images/ and "
            f"val_images/); found {len(splits['train'])} and {len(splits['val'])}."
        )
    reference = read_vector(config.reference_boundaries)
    if reference.crs is None:
        raise ValueError(f"Reference layer {config.reference_boundaries} has no CRS")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    stats: dict[str, Any] = {}
    chip_size: int | None = None
    ref_crs_cache: dict[str, Any] = {}
    chip_sliver_px = float(sliver_depth_px) / float(super_resolution)
    for split, files in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        n_instances = 0
        counts: dict[str, int] = {"n_slivers": 0}
        for path in files:
            data, transform, crs = _read_chip(path)
            if data.dtype != np.uint8:
                raise ValueError(
                    f"Training chip {path} is {data.dtype}; expected uint8 R, G, B after the "
                    "scene-level stretch (agribound.engines.finetune._data)."
                )
            height, width = data.shape[1:]
            if height != width:
                raise ValueError(f"Training chip {path} is not square ({width}x{height})")
            if chip_size is None:
                chip_size = width
            elif width != chip_size:
                raise ValueError(f"Training chips have different sizes ({width} vs {chip_size})")
            key = str(crs)
            if key not in ref_crs_cache:
                ref = reference if reference.crs.equals(crs) else reference.to_crs(crs)
                ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
                ref_crs_cache[key] = (ref.geometry.to_numpy(), ref.sindex)
            geoms, sindex = ref_crs_cache[key]
            from shapely.geometry import box

            footprint = box(*_chip_bounds(transform, width, height))
            hits = np.sort(sindex.query(footprint, predicate="intersects"))
            lines = chip_labels(
                [geoms[i] for i in hits],
                transform,
                width,
                height,
                valid_area_px=_valid_area_px(chip_valid_pixels(data, _mask_path(path))),
                min_area_px=min_area_px,
                sliver_depth_px=chip_sliver_px,
                counts=counts,
            )
            n_instances += len(lines)
            hwc = np.ascontiguousarray(np.transpose(data, (1, 2, 0)))  # R, G, B
            if super_resolution != 1:
                size = width * super_resolution
                hwc = cv2.resize(hwc, (size, size), interpolation=cv2.INTER_CUBIC)
            Image.fromarray(hwc).save(img_dir / f"{path.stem}.png")
            (lbl_dir / f"{path.stem}.txt").write_text("\n".join(lines))
        stats[f"n_{split}_images"] = len(files)
        stats[f"n_{split}_instances"] = n_instances
        stats[f"n_{split}_slivers"] = counts["n_slivers"]

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "field"},
            }
        )
    )
    return {
        "data_yaml": str(data_yaml),
        "chip_size": int(chip_size),
        "imgsz": int(chip_size) * int(super_resolution),
        **stats,
    }


def _chip_bounds(transform: Any, width: int, height: int) -> tuple[float, float, float, float]:
    xs = [transform.c, transform.c + transform.a * width]
    ys = [transform.f, transform.f + transform.e * height]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _nominal_batch_size(value: Any, batch: int) -> int:
    """Ultralytics ``nbs`` from ``engine_params["yolo_nbs"]`` (see the module docstring).

    None gives *batch* (no gradient accumulation), or Ultralytics' default of
    64 when *batch* is not positive (AutoBatch, ``yolo_batch=-1``).

    Raises
    ------
    ValueError
        If *value* is not an integer of at least 1.
    """
    import numbers

    if value is None:
        return batch if batch >= 1 else _ULTRALYTICS_NBS
    ok = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not (ok and np.isfinite(value) and float(value).is_integer() and value >= 1):
        raise ValueError(
            f"engine_params['yolo_nbs'] must be an integer of at least 1, got {value!r}"
        )
    return int(value)


def _sliver_depth_px(value: Any) -> float:
    """``engine_params["yolo_sliver_depth_px"]`` as a float (a finite number of at least 0)."""
    import numbers

    ok = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not (ok and np.isfinite(value) and value >= 0):
        raise ValueError(
            f"engine_params['yolo_sliver_depth_px'] must be a number of at least 0, got {value!r}"
        )
    return float(value)


def _finetune_yolo(train_dir: Path, config: AgriboundConfig, model_key: str) -> str:
    """Fine-tune Delineate-Anything with Ultralytics YOLO (see the module docstring).

    Parameters
    ----------
    train_dir : Path
        Training chips from ``_prepare_training_data(..., "delineate-anything")``.
    config : AgriboundConfig
        Configuration whose working directory is the fine-tuning run
        directory. Relevant ``engine_params``: ``da_model``/``model_size``
        (base model), ``super_resolution``, ``yolo_lr0``,
        ``yolo_warmup_bias_lr``, ``yolo_batch``, ``yolo_nbs`` (Ultralytics'
        nominal batch size: an integer of at least 1; default the batch size,
        i.e. no gradient accumulation), ``yolo_min_instance_px`` and
        ``yolo_sliver_depth_px`` (a number of at least 0; default
        :data:`DEFAULT_SLIVER_DEPTH_PX`).
    model_key : str
        Name the dispatcher uses for this model variant (logged and recorded
        as ``dispatcher_model_key``; the base weights follow
        ``engine_params`` via
        :func:`agribound.engines.delineate_anything.resolve_da_model_key`).

    Returns
    -------
    str
        Path of the ``best.pt`` checkpoint.

    Raises
    ------
    ImportError
        If ultralytics is not installed.
    ValueError
        If the engine parameters select the ``ftw`` backend, which cannot
        load a fine-tuned checkpoint, or ``yolo_warmup_bias_lr``,
        ``yolo_nbs`` or ``yolo_sliver_depth_px`` is out of range.
    RuntimeError
        If training produced no checkpoint.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for Delineate-Anything fine-tuning. Install with: "
            "pip install 'agribound[delineate-anything]'"
        ) from None
    import rasterio

    from agribound._cache import cache_path
    from agribound.engines.delineate_anything import (
        DA_HF_REPO,
        DA_MODELS,
        DAOptions,
        download_da_weights,
        pixel_size_m,
        select_super_resolution,
    )
    from agribound.engines.finetune._data import read_chip_meta, split_files, write_training_meta

    params = dict(config.engine_params)
    opts = DAOptions.from_engine_params(params)
    if opts.backend == "ftw":
        raise ValueError(
            "engine_params['backend']='ftw' cannot run fine-tuned weights (ftw-tools loads only "
            "its own checkpoints); fine-tune with backend='native' (default) or 'reference'."
        )
    if opts.checkpoint_path:
        logger.info(
            "engine_params['checkpoint_path'] is not used as base weights; fine-tuning starts "
            "from the pinned %s weights",
            opts.model_key,
        )
    spec = DA_MODELS[opts.model_key]

    first_chip = split_files(train_dir, "train", "images")
    if not first_chip:
        raise RuntimeError(f"No training chips in {train_dir / 'images'}")
    with rasterio.open(first_chip[0]) as src:
        gsd = pixel_size_m(src.crs, src.transform, src.height, src.width)
    sr = select_super_resolution(gsd, opts.super_resolution)

    lr0 = float(params.get("yolo_lr0", DEFAULT_LR0))
    warmup_bias_lr = float(params.get("yolo_warmup_bias_lr", DEFAULT_WARMUP_BIAS_LR))
    if not 0 <= warmup_bias_lr <= 1:  # Ultralytics' range for this key; also rejects NaN
        raise ValueError(
            f"engine_params['yolo_warmup_bias_lr'] must be between 0 and 1, got {warmup_bias_lr}"
        )
    batch = int(params.get("yolo_batch", 16))
    nbs = _nominal_batch_size(params.get("yolo_nbs"), batch)
    min_px = float(params.get("yolo_min_instance_px", DEFAULT_MIN_INSTANCE_PX))
    sliver_px = _sliver_depth_px(params.get("yolo_sliver_depth_px", DEFAULT_SLIVER_DEPTH_PX))
    device = config.resolve_device()
    weights = download_da_weights(spec.key)

    run_root = cache_path(
        config,
        f"yolo_{spec.key}",
        "",
        spec.sha256,
        Path(train_dir).name,
        sr,
        config.fine_tune_epochs,
        lr0,
        batch,
        min_px,
        config.seed,
        f"warmup_bias_lr={warmup_bias_lr}",
        f"nbs={nbs}",
        f"sliver_depth_px={sliver_px}",
        f"recipe={RECIPE_VERSION}",
    )
    run_root.mkdir(parents=True, exist_ok=True)
    dataset = prepare_yolo_dataset(
        Path(train_dir), config, run_root / "dataset", sr, min_px, sliver_px
    )

    train_kwargs: dict[str, Any] = {
        "data": dataset["data_yaml"],
        "epochs": int(config.fine_tune_epochs),
        "imgsz": dataset["imgsz"],
        "batch": batch,
        "nbs": nbs,
        "project": str(run_root.resolve()),
        "name": "train",
        "exist_ok": True,
        "device": device,
        "workers": int(config.n_workers),
        "seed": int(config.seed),
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": lr0,
        "warmup_bias_lr": warmup_bias_lr,
        "mosaic": 0.0,
        "fliplr": 0.5,
        "flipud": 0.5,
        "plots": False,
    }
    from agribound.engines.delineate_anything import MODEL_INPUT_PX

    imgsz_matches = dataset["imgsz"] == MODEL_INPUT_PX
    if not imgsz_matches:
        logger.warning(
            "Fine-tuning Delineate-Anything at imgsz=%d (%d px chips x super_resolution %d), "
            "not the %d px model input used at inference: the pixel scale matches inference, "
            "the spatial context per image does not. Set engine_params['chip_size'] to %d.",
            dataset["imgsz"],
            dataset["chip_size"],
            sr,
            MODEL_INPUT_PX,
            MODEL_INPUT_PX // sr,
        )
    logger.info(
        "Fine-tuning Delineate-Anything %s (%s) for %d epochs: imgsz=%d (super_resolution=%d), "
        "lr0=%g, warmup_bias_lr=%g, batch=%d, nbs=%d, %d train / %d val chips, %d / %d "
        "instances (%d / %d slivers of at most %g px dropped)",
        spec.key,
        model_key,
        config.fine_tune_epochs,
        dataset["imgsz"],
        sr,
        lr0,
        warmup_bias_lr,
        batch,
        nbs,
        dataset["n_train_images"],
        dataset["n_val_images"],
        dataset["n_train_instances"],
        dataset["n_val_instances"],
        dataset["n_train_slivers"],
        dataset["n_val_slivers"],
        sliver_px,
    )
    from agribound.engines.delineate_anything import no_nms_time_limit

    # Validation NMS without Ultralytics' wall-clock limit: best.pt is chosen on it.
    with no_nms_time_limit():
        results = YOLO(weights).train(**train_kwargs)

    save_dir = Path(getattr(results, "save_dir", "") or (run_root / "train"))
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        last = save_dir / "weights" / "last.pt"
        if not last.is_file():
            raise RuntimeError(
                f"YOLO fine-tuning wrote no checkpoint in {save_dir / 'weights'}; see the "
                "Ultralytics log above."
            )
        logger.warning("No best.pt in %s; returning last.pt", save_dir / "weights")
        best = last

    try:
        chips_meta = read_chip_meta(train_dir)
    except (OSError, ValueError):
        chips_meta = {}
    write_training_meta(
        best,
        {
            "engine": "delineate-anything",
            "base_model_key": spec.key,
            "dispatcher_model_key": model_key,
            "base_weights_repo": DA_HF_REPO,
            "base_weights_filename": spec.filename,
            "base_weights_revision": spec.revision,
            "base_weights_sha256": spec.sha256,
            "gsd_m": gsd,
            "super_resolution": sr,
            "chip_size": dataset["chip_size"],
            "imgsz": dataset["imgsz"],
            "imgsz_matches_model_input": imgsz_matches,
            "labels": "reference polygons clipped per chip; holes bridged",
            "min_instance_px": min_px,
            "sliver_depth_px": sliver_px,
            "chip_stride": chips_meta.get("chip_stride"),
            # The options the chips were made with (recorded by _data); with gsd_m,
            # super_resolution and raster_dtype (uint8: not stretched) the input the model
            # saw, which the native backend compares with its own input.
            "raster_dtype": chips_meta.get("raster_dtype"),
            "processing_gsd_m": chips_meta.get("processing_gsd_m", opts.processing_gsd_m),
            "stretch_percentiles": (chips_meta.get("image") or {}).get(
                "stretch_percentiles", list(opts.stretch_percentiles)
            ),
            "processing": chips_meta.get("processing"),
            "recipe_version": RECIPE_VERSION,
            "train_kwargs": {k: v for k, v in train_kwargs.items() if k not in ("data", "project")},
            "dataset": {k: v for k, v in dataset.items() if k != "data_yaml"},
            "split": chips_meta.get("split"),
            "chips_stretch": chips_meta.get("image"),
        },
    )
    (run_root / "train_kwargs.json").write_text(json.dumps(train_kwargs, indent=2, default=str))
    logger.info("Fine-tuned Delineate-Anything weights: %s", best)
    return str(best)
