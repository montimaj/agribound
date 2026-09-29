"""
Delineate-Anything (Ultralytics YOLO11-seg) fine-tuning.

:func:`_finetune_yolo` is called by :func:`agribound.engines.finetune.fine_tune`
with the directory written by
:func:`agribound.engines.finetune._data._prepare_training_data` for
``engine="delineate-anything"``: georeferenced uint8 R, G, B chips with the
scene-level 1-99 percentile stretch already applied (the stretch the engine
applies at inference), split into ``images/`` (train) and ``val_images/``
(validation) by :func:`agribound.engines.finetune._data.assign_splits`.

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
overrides), horizontal and vertical flips with probability 0.5,
``plots=False``, ``epochs=config.fine_tune_epochs``, ``batch`` 16
(``yolo_batch``), ``workers=config.n_workers``. Ultralytics selects the
checkpoint with the best validation fitness (``best.pt``).
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


def chip_labels(
    polygons: list[Any],
    chip_transform: Any,
    width: int,
    height: int,
    valid_area_px: Any | None = None,
    min_area_px: float = DEFAULT_MIN_INSTANCE_PX,
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

    Returns
    -------
    dict
        ``data_yaml``, ``imgsz``, ``chip_size`` and per-split image and
        instance counts.
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
    for split, files in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        n_instances = 0
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


def _finetune_yolo(train_dir: Path, config: AgriboundConfig, model_key: str) -> str:
    """Fine-tune Delineate-Anything with Ultralytics YOLO (see the module docstring).

    Parameters
    ----------
    train_dir : Path
        Training chips from ``_prepare_training_data(..., "delineate-anything")``.
    config : AgriboundConfig
        Configuration whose working directory is the fine-tuning run
        directory. Relevant ``engine_params``: ``da_model``/``model_size``
        (base model), ``super_resolution``, ``yolo_lr0``, ``yolo_batch``,
        ``yolo_min_instance_px``.
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
        load a fine-tuned checkpoint.
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
    batch = int(params.get("yolo_batch", 16))
    min_px = float(params.get("yolo_min_instance_px", DEFAULT_MIN_INSTANCE_PX))
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
    )
    run_root.mkdir(parents=True, exist_ok=True)
    dataset = prepare_yolo_dataset(Path(train_dir), config, run_root / "dataset", sr, min_px)

    train_kwargs: dict[str, Any] = {
        "data": dataset["data_yaml"],
        "epochs": int(config.fine_tune_epochs),
        "imgsz": dataset["imgsz"],
        "batch": batch,
        "project": str(run_root.resolve()),
        "name": "train",
        "exist_ok": True,
        "device": device,
        "workers": int(config.n_workers),
        "seed": int(config.seed),
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": lr0,
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
        "%d train / %d val chips, %d / %d instances",
        spec.key,
        model_key,
        config.fine_tune_epochs,
        dataset["imgsz"],
        sr,
        dataset["n_train_images"],
        dataset["n_val_images"],
        dataset["n_train_instances"],
        dataset["n_val_instances"],
    )
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
            "train_kwargs": {k: v for k, v in train_kwargs.items() if k not in ("data", "project")},
            "dataset": {k: v for k, v in dataset.items() if k != "data_yaml"},
            "split": chips_meta.get("split"),
            "chips_stretch": chips_meta.get("image"),
        },
    )
    (run_root / "train_kwargs.json").write_text(json.dumps(train_kwargs, indent=2, default=str))
    logger.info("Fine-tuned Delineate-Anything weights: %s", best)
    return str(best)
