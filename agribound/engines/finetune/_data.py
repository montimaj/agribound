"""
Training data for fine-tuning: engine inputs, chips, masks and the split.

This module is shared by the fine-tuning trainers and by the engines that
consume their checkpoints (``geoai``, ``dinov3``, ``prithvi`` use
:func:`write_rgb_input`, :func:`prithvi_reflectance`,
:func:`mask_invalid_predictions` and :func:`interior_polygons`), so that a
model sees the same kind of input at training and at inference time.

Engine inputs
-------------
- RGB engines (``delineate-anything``, ``geoai``, ``dinov3``) read the
  canonical R, G, B bands and apply a **scene-level** 1-99 percentile stretch
  to uint8 (:func:`scene_stretch_bounds` + :func:`apply_stretch`, the same
  formula as :func:`agribound.io.raster.percentile_stretch_uint8`). The
  percentiles are computed once per raster, on a nearest-neighbour sample of
  at most 4096 x 4096 pixels, from valid, finite, strictly positive values.
  uint8 rasters (NAIP) are used unchanged. ``dinov3`` chips hold the same
  values as float32 ``uint8 / 255`` (see :data:`CHIP_FORMATS`); on rasters
  coarser than :data:`DINOV3_PROCESSING_GSD_M` (10 m: Landsat and HLS, 30 m;
  Landsat PAN, 15 m) they are cut from a bilinear copy with 10 m pixels,
  which DINOv3 also reads at inference (:func:`dinov3_processing_gsd`,
  :func:`dinov3_processing_raster`).
  ``delineate-anything`` chips follow the engine's ``engine_params``
  ``stretch_percentiles`` (the percentiles) and ``processing_gsd_m`` (the
  chips are cut from the raster resampled to that pixel size,
  :func:`agribound.engines.delineate_anything.processing_raster`), and treat
  pixels that are 0 in every band as nodata (:func:`valid_pixels` with
  ``zero_invalid=True``), as its native backend does at inference.
- ``prithvi`` reads six bands in Prithvi-EO-2.0 order -- Blue, Green, Red,
  narrow NIR, SWIR 1, SWIR 2 (:func:`prithvi_band_names`) -- as surface
  reflectance x 10000 (:func:`prithvi_reflectance`), the scale of the
  Prithvi-EO-2.0 normalisation statistics. No other scaling is applied.

Training directory layout
-------------------------
:func:`_prepare_training_data` writes::

    <train_dir>/
        images/, masks/, instances/              training chips
        val_images/, val_masks/, val_instances/  validation chips
        chips.gpkg                               chip footprints (layer "chips")
        chips_meta.json                          how the chips were made

All three directories of one split hold files with the same names
(``chip_<id>.tif``). ``masks`` are uint8 semantic masks: 0 background, 1 field
interior, 2 field boundary (the reference-polygon pixels within
``boundary_erosion`` pixels, city-block distance, of a pixel of another
polygon or of background), and 255 (:data:`IGNORE_INDEX`) where the image has
no valid data. ``instances`` are int32 masks holding, per pixel, the 1-based
position of the reference polygon in the reference layer (0 = background;
where polygons overlap, the later polygon wins). ``instances`` have no ignore
value: pixels without valid image data are 0 (background) even inside a
reference polygon, and their image values are 0 (RGB chips) or the
Prithvi-EO-2.0 band means (Prithvi chips), so the GeoAI trainer, which reads
``instances``, learns such holes as background. ``chips.gpkg`` has one row
per written chip with ``chip_id``, ``split``, ``row_off``, ``col_off``,
``size`` and the footprint in the raster CRS. ``chips_meta.json`` is written
last; a directory without it is incomplete and is rebuilt.

Pixels outside every reference polygon are labelled background, so the
reference layer must be complete inside the chips that contain its polygons.

Split
-----
:func:`assign_splits` implements the train/validation split used by every
trainer (``config.fine_tune_split``: ``"block"``, ``"random"`` or
``"column"``), seeded through :func:`agribound._repro.get_rng`. Its units are
the chip-size cells that the chips cover, so that chips that overlap
(``delineate-anything`` by default, see :func:`chip_stride`) never put one
pixel in both splits; see :func:`_prepare_training_data`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from agribound.config import AgriboundConfig, effective_bands

logger = logging.getLogger(__name__)

__all__ = [
    "CHIP_FORMATS",
    "DEFAULT_BOUNDARY_EROSION",
    "DINOV3_PROCESSING_GSD_M",
    "IGNORE_INDEX",
    "RGB_ENGINES",
    "SEMANTIC_CLASSES",
    "apply_stretch",
    "assign_splits",
    "chip_stride",
    "default_boundary_erosion",
    "dinov3_grid_gsd",
    "dinov3_processing_gsd",
    "dinov3_processing_raster",
    "engine_band_names",
    "field_interior_depths",
    "file_sha256",
    "grow_labels",
    "hf_cached_file_info",
    "interior_polygons",
    "mask_invalid_predictions",
    "package_versions",
    "predicted_band_thickness",
    "prithvi_band_names",
    "prithvi_reflectance",
    "raster_fingerprint",
    "read_checkpoint_hparams",
    "read_chip_meta",
    "read_json",
    "read_training_meta",
    "reflect_indices",
    "scene_stretch_bounds",
    "semantic_from_instances",
    "split_files",
    "training_meta_path",
    "training_output_dir",
    "valid_pixels",
    "widest_boundary_erosion",
    "write_json",
    "write_rgb_input",
    "write_training_meta",
]

#: Semantic mask classes written to ``masks/``.
SEMANTIC_CLASSES: dict[int, str] = {0: "background", 1: "field_interior", 2: "field_boundary"}

#: Mask value for pixels without valid image data (ignored by the losses).
IGNORE_INDEX = 255

#: Engines whose input is an RGB image.
RGB_ENGINES = ("delineate-anything", "geoai", "dinov3")

#: Image-chip format written per engine.
CHIP_FORMATS: dict[str, str] = {
    # 3-band uint8 after the scene-level stretch.
    "delineate-anything": "rgb_uint8",
    "geoai": "rgb_uint8",
    # The uint8 values divided by 255, stored as float32 in [0, 1]. geoai's
    # DINOv3 dataset and inference divide a chip/window by 255 only when its
    # maximum exceeds 1; with [0, 1] input that rule never applies, so every
    # chip and window is scaled identically.
    "dinov3": "rgb_unit_float32",
    # 6-band float32 surface reflectance x 10000.
    "prithvi": "prithvi_x10000",
}

#: Default boundary width of the masks in pixels per engine
#: (``engine_params["boundary_erosion"]`` overrides; other engines: 2). DINOv3's decoder
#: predicts on the 16 px patch grid, and its default was chosen on the validation chips
#: of example 23's four training areas (see the fine-tuning guide); 2 px before. The
#: default is part of DINOv3's training recipe: the fine-tuning cache key holds
#: ``_dinov3.RECIPE_VERSION``, not this value, so changing it needs a new recipe version.
DEFAULT_BOUNDARY_EROSION: dict[str, int] = {"dinov3": 4}

#: Default chip edge length in pixels per engine (``engine_params["chip_size"]``
#: overrides). Delineate-Anything uses 512 px below 4 m GSD and 256 px otherwise
#: (its native tiling rule); 224 px is divisible by both Prithvi patch sizes
#: (16 and 14).
_DEFAULT_CHIP_SIZE: dict[str, int] = {"geoai": 256, "dinov3": 256, "prithvi": 224}

#: GeoAI (Mask R-CNN instance segmentation) predicts at most one instance per field per
#: window, inference uses windows of the training chip size, and no window sees the whole
#: of a larger field (the engine rejoins the pieces only where they meet at a window edge,
#: ``geoai_field.merge_window_seams``). Its default chip is therefore sized from the
#: reference fields: at least GEOAI_FIELD_MARGIN times the 90th-percentile field
#: bounding-box side, in pixels, rounded up to a multiple of 32 and clamped to
#: [GEOAI_MIN_CHIP, GEOAI_MAX_CHIP] (Mask R-CNN resizes every chip to 800 px and keeps at
#: most 100 detections per window, so very large chips would drop fields in landscapes of
#: small fields).
GEOAI_FIELD_QUANTILE = 0.9
GEOAI_FIELD_MARGIN = 1.25
GEOAI_MIN_CHIP = 256
GEOAI_MAX_CHIP = 1024
#: Warn when more than this share of the reference fields is larger than a chip.
GEOAI_LARGE_FIELD_WARN_SHARE = 0.10

#: DINOv3 reads rasters with coarser pixels than this many metres on a copy with pixels of
#: this size (training chips and inference; :func:`dinov3_processing_gsd`). The ViT reads
#: 16 x 16 px patches and geoai's head predicts the classes on that patch grid, upsampled
#: bilinearly: at 30 m a patch is 480 m wide, more than the shorter side of about half of
#: example 14's 393 training fields (median 566 m), and the eastern Lea County composite
#: holds only 4 chips of 256 px (3 training, 1 validation). On the 10 m copy of the same
#: Landsat 2024 composite (52 chips, 42 training) the fine-tuned model's field F1 against
#: the 388 V18 polygons was 0.538 instead of 0.023 (CPU runs, seed 42).
DINOV3_PROCESSING_GSD_M = 10.0
#: Rasters within this relative tolerance of the processing pixel size are read as they are.
_DINOV3_GRID_TOLERANCE = 0.01
#: Version of DINOv3's processing copy (part of its cache key and of the chips' key).
DINOV3_GRID_VERSION = 1

#: Engines trained on the 3-class masks (``masks/``); their chips count the reference
#: fields that keep no interior pixel inside the boundary band.
MASK_ENGINES = ("dinov3", "prithvi")
#: Warn when more than this share of the reference fields in the chips of a mask engine
#: has no interior pixel (all of it within ``boundary_erosion`` px of its edge).
NO_INTERIOR_WARN_SHARE = 0.10


def chip_size_rule(engine: str) -> str:
    """Identifier of the default chip-size rule of *engine* (part of the fine-tuning cache key).

    Changing a default rule changes this string, so checkpoints trained with chips chosen
    by an older rule are not reused.
    """
    if engine == "geoai":
        return (
            f"fields-q{GEOAI_FIELD_QUANTILE:g}x{GEOAI_FIELD_MARGIN:g}-"
            f"{GEOAI_MIN_CHIP}-{GEOAI_MAX_CHIP}"
        )
    if engine == "delineate-anything":
        return "gsd-512-below-4m-else-256"
    return f"fixed-{_DEFAULT_CHIP_SIZE.get(engine, 'na')}"


_SPLIT_TRAIN = "train"
_SPLIT_VAL = "val"
_META_NAME = "chips_meta.json"
_MAX_SAMPLE_SIDE = 4096
#: Percentiles of the scene stretch of RGB chips (Delineate-Anything:
#: ``engine_params["stretch_percentiles"]``).
_DEFAULT_PERCENTILES = (1.0, 99.0)
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


# ---------------------------------------------------------------------------
# Train/validation split
# ---------------------------------------------------------------------------


def assign_splits(
    units: Any,
    config: AgriboundConfig,
    reference: Any | None = None,
    *,
    info: dict[str, Any] | None = None,
) -> np.ndarray:
    """Assign each unit (e.g. a chip footprint) to ``"train"`` or ``"val"``.

    Parameters
    ----------
    units : geopandas.GeoDataFrame
        Unit footprints in any CRS (the CRS must be set).
    config : AgriboundConfig
        Uses ``fine_tune_split``, ``fine_tune_val_split`` (target fraction of
        units in validation), ``fine_tune_block_size_m``,
        ``fine_tune_split_column`` and ``seed``.
    reference : geopandas.GeoDataFrame or None
        Reference polygons; required for ``fine_tune_split="column"``.
    info : dict or None
        If given, filled with a summary: ``strategy``, ``n_train``, ``n_val``,
        ``n_groups`` and, for blocks, ``block_size_m`` (the size used) and
        ``requested_block_size_m``.

    Returns
    -------
    numpy.ndarray
        Array of ``"train"``/``"val"`` strings with ``len(units)`` entries, in
        the order of *units*. At least one unit is in each split.

    Raises
    ------
    ValueError
        If there are fewer than two units, the CRS is missing, the strategy
        is unknown, or (``"column"``) the reference layer or column is
        missing or all units fall in one group.

    Notes
    -----
    Randomness comes only from ``agribound._repro.get_rng(config,
    "fine_tune_split", strategy)``, so the split depends on the seed, the
    strategy and the units, and is the same in every process.

    - ``"random"``: a seeded permutation; ``round(fine_tune_val_split * n)``
      units (at least 1, at most ``n - 1``) go to validation.
    - ``"block"``: units are grouped by the square block, of edge
      ``fine_tune_block_size_m`` metres, that contains their centroid. The
      block grid starts at the south-west corner of the units' bounding box
      in a Lambert azimuthal equal-area projection centred on the centroid of
      all units (distances are close to true metres within a few hundred
      kilometres of the centre). Units of one block always share a split.
      If every unit falls in one block, the block edge is halved until at
      least two blocks are occupied; this is logged at WARNING and reported as
      ``info["block_size_m"]``.
    - ``"column"``: each unit is assigned the value of
      ``fine_tune_split_column`` that is most frequent among the reference
      polygons overlapping it with positive area (ties: larger total overlap
      area, then the smallest value as a string); units overlapping no
      polygon form one extra group. Units of one group always share a split.

    For ``"block"`` and ``"column"`` whole groups go to validation: the
    groups are visited in a seeded random order and a group is added when it
    brings the number of validation units closer to
    ``fine_tune_val_split * n``. If no group was added, the smallest group is
    used; if every group was added, the smallest validation group is moved
    back to training. No buffer is kept between training and validation
    groups, so units on either side of a group edge can still be spatially
    correlated.
    """
    from agribound._repro import get_rng

    n = len(units)
    if n < 2:
        raise ValueError(f"At least 2 units are needed for a train/validation split, got {n}")
    if getattr(units, "crs", None) is None:
        raise ValueError("units must have a CRS")
    strategy = str(config.fine_tune_split).lower()
    fraction = float(config.fine_tune_val_split)
    rng = get_rng(config, "fine_tune_split", strategy)
    summary: dict[str, Any] = {"strategy": strategy}

    if strategy == "random":
        n_val = int(min(max(round(fraction * n), 1), n - 1))
        order = rng.permutation(n)
        is_val = np.zeros(n, dtype=bool)
        is_val[order[:n_val]] = True
        summary["n_groups"] = n
    elif strategy == "block":
        requested = float(config.fine_tune_block_size_m)
        groups, size_used = _block_groups(units, requested)
        if size_used != requested:
            logger.warning(
                "All %d units fall in one %.0f m block; using %.1f m blocks so that the "
                "block split has at least two blocks. Set fine_tune_block_size_m smaller or "
                "fine_tune_split='random' to control this.",
                n,
                requested,
                size_used,
            )
        summary["requested_block_size_m"] = requested
        summary["block_size_m"] = size_used
        is_val = _groups_to_val(groups, fraction, rng)
        summary["n_groups"] = len(np.unique(groups))
    elif strategy == "column":
        groups = _column_groups(units, config, reference)
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError(
                f"fine_tune_split='column' needs at least 2 groups, but all {n} units map to "
                f"one value of column {config.fine_tune_split_column!r}. Use "
                "fine_tune_split='block' or 'random'."
            )
        is_val = _groups_to_val(groups, fraction, rng)
        summary["n_groups"] = n_groups
    else:
        raise ValueError(f"Unknown fine_tune_split {strategy!r}. Choose block, random or column")

    splits = np.where(is_val, _SPLIT_VAL, _SPLIT_TRAIN)
    summary["n_train"] = int((~is_val).sum())
    summary["n_val"] = int(is_val.sum())
    if info is not None:
        info.update(summary)
    return splits


def _block_groups(units: Any, block_size_m: float) -> tuple[np.ndarray, float]:
    """Return per-unit block labels and the block size actually used."""
    import pyproj

    if block_size_m <= 0:
        raise ValueError(f"fine_tune_block_size_m must be > 0, got {block_size_m}")
    units_4326 = units.to_crs("EPSG:4326") if not units.crs.equals("EPSG:4326") else units
    centre = units_4326.geometry.union_all().centroid
    laea = pyproj.CRS.from_proj4(
        f"+proj=laea +lat_0={round(centre.y, 3)} +lon_0={round(centre.x, 3)} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    projected = units.to_crs(laea).geometry
    cent = projected.centroid
    xy = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
    if not np.all(np.isfinite(xy)):
        raise ValueError("Some unit geometries are empty or invalid")
    minx, miny, _, _ = projected.total_bounds
    xy = xy - np.array([minx, miny])

    size = float(block_size_m)
    while True:
        cells = np.floor(xy / size).astype(np.int64)
        _, labels = np.unique(cells, axis=0, return_inverse=True)
        labels = np.asarray(labels).reshape(-1)
        if labels.max() >= 1:
            return labels, size
        if size < 1e-3:
            raise ValueError("All units share the same centroid; they cannot be split by block")
        size /= 2.0


def _column_groups(units: Any, config: AgriboundConfig, reference: Any | None) -> np.ndarray:
    """Return per-unit group labels from the majority reference-column value."""
    column = config.fine_tune_split_column
    if not column:
        raise ValueError("fine_tune_split='column' requires fine_tune_split_column")
    if reference is None:
        raise ValueError("fine_tune_split='column' requires the reference polygons")
    if column not in reference.columns:
        raise ValueError(
            f"fine_tune_split_column {column!r} is not a column of the reference layer "
            f"(columns: {[c for c in reference.columns if c != reference.geometry.name]})"
        )
    ref = reference.to_crs(units.crs) if not reference.crs.equals(units.crs) else reference
    ref = ref[ref.geometry.notna() & ~ref.geometry.is_empty]
    values = ref[column].to_numpy()
    ref_geoms = ref.geometry.to_numpy()
    tree_idx_units, tree_idx_ref = ref.sindex.query(units.geometry, predicate="intersects")

    labels: list[str] = []
    none_label = "\x00no-reference"
    per_unit: dict[int, list[int]] = {}
    for u, r in zip(tree_idx_units.tolist(), tree_idx_ref.tolist(), strict=True):
        per_unit.setdefault(u, []).append(r)
    unit_geoms = units.geometry.to_numpy()
    for u in range(len(units)):
        hits = per_unit.get(u)
        if not hits:
            labels.append(none_label)
            continue
        counts: dict[str, int] = {}
        areas: dict[str, float] = {}
        for r in hits:
            overlap = float(unit_geoms[u].intersection(ref_geoms[r]).area)
            if overlap <= 0.0:
                continue  # touching only
            key = str(values[r])
            counts[key] = counts.get(key, 0) + 1
            areas[key] = areas.get(key, 0.0) + overlap
        if not counts:
            labels.append(none_label)
            continue
        best = sorted(counts, key=lambda k: (-counts[k], -areas[k], k))[0]
        labels.append(best)
    _, inverse = np.unique(np.asarray(labels, dtype=object).astype(str), return_inverse=True)
    return np.asarray(inverse).reshape(-1)


def _groups_to_val(groups: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Select whole groups for validation (see :func:`assign_splits`)."""
    n = len(groups)
    ids, sizes = np.unique(groups, return_counts=True)
    size_of = dict(zip(ids.tolist(), sizes.tolist(), strict=True))
    target = fraction * n
    order = [ids[i] for i in rng.permutation(len(ids))]
    chosen: list[int] = []
    n_val = 0
    for g in order:
        s = size_of[int(g)]
        if abs(n_val + s - target) < abs(n_val - target):
            chosen.append(int(g))
            n_val += s
    if not chosen:
        chosen.append(min(order, key=lambda g: size_of[int(g)]))
    elif len(chosen) == len(ids):
        chosen.remove(min(chosen, key=lambda g: size_of[int(g)]))
    return np.isin(groups, chosen)


# ---------------------------------------------------------------------------
# Bands and radiometry
# ---------------------------------------------------------------------------


def prithvi_band_names(source: str, bands: dict[str, int] | None = None) -> list[str]:
    """Canonical band names in Prithvi-EO-2.0 order for *source*.

    Prithvi-EO-2.0 was pre-trained on HLS Blue, Green, Red, narrow NIR (HLS
    B5: OLI B5 / MSI B8A), SWIR 1 and SWIR 2. The narrow NIR is
    ``NIR_NARROW`` where the source defines it (Sentinel-2 B8A, HLS B5).
    Landsat defines only ``NIR`` (``SR_B5`` after renaming): the OLI narrow
    NIR on Landsat 8/9, but the broad NIR on Landsat 5/7 (TM band 4,
    0.76-0.90 um; ETM+ band 4, 0.77-0.90 um), which differs from the
    pre-training band. For ``source="local"`` ``NIR_NARROW`` is used unless
    *bands* maps only ``NIR``; without *bands* the local raster's bands 1-6
    are read as Blue, Green, Red, narrow NIR, SWIR 1, SWIR 2 (positional, see
    :func:`agribound.engines.base.get_canonical_band_indices`).

    Parameters
    ----------
    source : str
        Source name.
    bands : dict or None
        Explicit canonical-name -> 1-based index mapping (``config.bands``).

    Returns
    -------
    list[str]
        ``["B", "G", "R", <NIR_NARROW or NIR>, "SWIR1", "SWIR2"]``.
    """
    from agribound.registry import canonical_bands

    mapping = dict(bands or {})
    canonical = canonical_bands(source) if source != "local" else {}
    if "NIR_NARROW" in mapping or "NIR_NARROW" in canonical:
        nir = "NIR_NARROW"
    elif "NIR" in mapping or "NIR" in canonical:
        nir = "NIR"
    else:
        nir = "NIR_NARROW"
    return ["B", "G", "R", nir, "SWIR1", "SWIR2"]


def engine_band_names(engine: str, source: str, bands: dict[str, int] | None = None) -> list[str]:
    """Canonical band names an engine reads from the composite.

    Raises
    ------
    ValueError
        For engines without fine-tuning chip support here.
    """
    if engine in RGB_ENGINES:
        return ["R", "G", "B"]
    if engine == "prithvi":
        return prithvi_band_names(source, bands)
    raise ValueError(
        f"No training-chip definition for engine {engine!r} "
        f"(supported: {', '.join((*RGB_ENGINES, 'prithvi'))})"
    )


def valid_pixels(
    data: np.ndarray, nodata: float | None = None, *, zero_invalid: bool = False
) -> np.ndarray:
    """Return a ``(H, W)`` mask of pixels with valid data in all bands.

    A pixel is invalid if any band is non-finite, or if every band equals a
    finite *nodata* value (the rule of
    :func:`agribound.io.raster.percentile_stretch_uint8`). With
    *zero_invalid*, a pixel whose bands are all 0 is invalid too: the rule of
    the Delineate-Anything native backend (upstream's default nodata value
    ``[0, 0, 0]``), which its training chips follow.
    """
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if np.issubdtype(arr.dtype, np.floating):
        valid = np.all(np.isfinite(arr), axis=0)
    else:
        valid = np.ones(arr.shape[1:], dtype=bool)
    if nodata is not None and np.isfinite(nodata):
        valid &= ~np.all(arr == nodata, axis=0)
    if zero_invalid:
        valid &= ~np.all(arr == 0, axis=0)
    return valid


def prithvi_reflectance(
    data: np.ndarray, source: str, value_scale: str | None = None
) -> np.ndarray:
    """Return Prithvi input bands as float32 surface reflectance x 10000.

    Parameters
    ----------
    data : numpy.ndarray
        Bands as written by the source's composite builder.
    source : str
        Source name (its registry ``value_scale`` is used).
    value_scale : str or None
        Override: ``"reflectance_x10000"`` (used unchanged) or ``"unit"``
        (0-1 reflectance, multiplied by 10000). Required for
        ``source="local"``.

    Raises
    ------
    ValueError
        If the data are not surface reflectance (e.g. uint8 NAIP, SPOT DN,
        embeddings, or a local raster without *value_scale*).
    """
    from agribound.registry import source_value_scale

    scale = value_scale or source_value_scale(source)
    if scale == "reflectance_x10000":
        return np.asarray(data, dtype=np.float32)
    if scale == "unit":
        return (np.asarray(data, dtype=np.float32) * np.float32(10000.0)).astype(np.float32)
    raise ValueError(
        f"Prithvi-EO-2.0 needs surface reflectance; source {source!r} has value scale "
        f"{scale!r}. For a local reflectance raster set "
        "engine_params['value_scale'] to 'reflectance_x10000' or 'unit'."
    )


def scene_stretch_bounds(
    raster_path: str | Path,
    band_indices: list[int],
    low: float = 1.0,
    high: float = 99.0,
    max_sample_side: int = _MAX_SAMPLE_SIDE,
) -> tuple[list[float], list[float]]:
    """Per-band stretch percentiles of a whole raster.

    The bands are read on a nearest-neighbour grid of at most
    ``max_sample_side`` pixels per side (a decimated read of the
    full-resolution data by :func:`agribound.io.raster.read_stretch_sample`:
    a raster with overviews is reopened with ``OVERVIEW_LEVEL=NONE``, as in
    :func:`agribound.engines.delineate_anything.scene_stretch_bounds`) and
    passed to :func:`agribound.io.raster.percentile_stretch_uint8`, so the
    bounds are the *low*/*high* percentiles of valid, finite, strictly
    positive values. uint8 rasters return ``([0, ...], [255, ...])`` (no
    stretch).

    Returns
    -------
    tuple[list[float], list[float]]
        ``(lows, highs)``, one value per band.
    """
    import rasterio

    from agribound.io.raster import percentile_stretch_uint8, read_stretch_sample

    with rasterio.open(raster_path) as src:
        dtype = np.dtype(src.dtypes[band_indices[0] - 1])
        if dtype == np.uint8:
            return [0.0] * len(band_indices), [255.0] * len(band_indices)
        sample = read_stretch_sample(src, band_indices, max_sample_side)
        nodata = src.nodata
    lows, highs = percentile_stretch_uint8(
        sample, nodata=nodata, low=low, high=high, per_band=True, return_bounds_only=True
    )
    return [float(v) for v in lows], [float(v) for v in highs]


def apply_stretch(
    data: np.ndarray,
    lows: list[float],
    highs: list[float],
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Map *data* to uint8 with known per-band bounds.

    Uses the formula of :func:`agribound.io.raster.percentile_stretch_uint8`:
    ``clip(255 * (v - lo) / (hi - lo), 0, 255)`` truncated to uint8, 0 where
    *valid* is False. uint8 input is returned unchanged (copied).
    """
    arr = np.asarray(data)
    if arr.dtype == np.uint8:
        return arr.copy()
    if valid is None:
        valid = valid_pixels(arr)
    out = np.zeros(arr.shape, dtype=np.uint8)
    for i, (lo, hi) in enumerate(zip(lows, highs, strict=True)):
        span = hi - lo if hi > lo else 1e-12
        band = arr[i].astype(np.float64)
        stretched = np.clip(255.0 * ((band - lo) / span), 0, 255)
        stretched = np.where(valid & np.isfinite(stretched), stretched, 0)
        out[i] = stretched.astype(np.uint8)
    return out


def reflect_indices(positions: Any, n: int) -> np.ndarray:
    """Map pixel positions onto ``0 .. n - 1`` by mirroring at the edges.

    The mirror excludes the edge pixel and repeats for positions more than
    one length away, exactly as ``numpy.pad(..., mode="reflect")``: for
    ``n = 4``, positions ``4, 5, 6, 7`` map to ``2, 1, 0, 1`` and ``-1`` maps
    to ``1``. ``n = 1`` maps everything to 0.
    """
    pos = np.asarray(positions, dtype=np.int64)
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if n == 1:
        return np.zeros_like(pos)
    period = 2 * (n - 1)
    r = np.mod(pos, period)
    return np.where(r > n - 1, period - r, r)


def write_rgb_input(
    raster_path: str | Path,
    out_path: str | Path,
    band_indices: list[int],
    *,
    unit_float: bool = False,
    bounds: tuple[list[float], list[float]] | None = None,
    pad_to: tuple[int, int] | None = None,
    block_rows: int = 1024,
) -> dict[str, Any]:
    """Write the scene-stretched RGB input of an RGB engine.

    Parameters
    ----------
    raster_path : str or Path
        Composite raster.
    out_path : str or Path
        Destination GeoTIFF (no nodata value). Same grid as *raster_path*,
        extended at the bottom and right when *pad_to* is given.
    band_indices : list[int]
        1-based R, G, B band indices.
    unit_float : bool
        Write ``uint8 / 255`` as float32 (DINOv3) instead of uint8.
    bounds : tuple or None
        ``(lows, highs)``; computed with :func:`scene_stretch_bounds` if None.
    pad_to : tuple of int or None
        ``(height, width)`` of the output, at least the raster size. The
        extra rows and columns (bottom and right; the top-left origin and the
        transform are unchanged) repeat the raster by mirror reflection
        (:func:`reflect_indices`), so a window that reaches past the raster
        edge holds image content rather than zeros.
    block_rows : int
        Rows processed per block (memory bound).

    Returns
    -------
    dict
        ``{"lows", "highs", "format"}`` describing the transform applied,
        plus ``"padded_to"`` (``[height, width]``) and ``"padding":
        "reflect"`` when the output is larger than the raster.
    """
    import rasterio
    from rasterio.windows import Window

    lows, highs = bounds if bounds is not None else scene_stretch_bounds(raster_path, band_indices)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".partial" + out_path.suffix)
    with rasterio.open(raster_path) as src:
        out_h, out_w = (src.height, src.width) if pad_to is None else map(int, pad_to)
        if out_h < src.height or out_w < src.width:
            raise ValueError(
                f"pad_to {pad_to} is smaller than the raster ({src.height} x {src.width})"
            )
        padded = (out_h, out_w) != (src.height, src.width)
        row_map = reflect_indices(np.arange(out_h), src.height)
        col_map = reflect_indices(np.arange(out_w), src.width)
        profile = {
            "driver": "GTiff",
            "height": out_h,
            "width": out_w,
            "count": len(band_indices),
            "dtype": "float32" if unit_float else "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        with rasterio.open(tmp, "w", **profile) as dst:
            for row in range(0, out_h, block_rows):
                nrows = min(block_rows, out_h - row)
                rows = row_map[row : row + nrows]
                lo, hi = int(rows.min()), int(rows.max()) + 1
                data = src.read(band_indices, window=Window(0, lo, src.width, hi - lo))
                if padded:
                    data = data[:, rows - lo][:, :, col_map]
                rgb = apply_stretch(data, lows, highs, valid_pixels(data, src.nodata))
                window = Window(0, row, out_w, nrows)
                if unit_float:
                    dst.write((rgb.astype(np.float32) / np.float32(255.0)), window=window)
                else:
                    dst.write(rgb, window=window)
    os.replace(tmp, out_path)
    info: dict[str, Any] = {
        "lows": lows,
        "highs": highs,
        "format": "rgb_unit_float32" if unit_float else "rgb_uint8",
    }
    if padded:
        info.update(padded_to=[out_h, out_w], padding="reflect")
    return info


def raster_fingerprint(path: str | Path) -> str:
    """``"<file name>|<size>|<mtime_ns>"`` of a file, for cache keys."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return p.name
    return f"{p.name}|{st.st_size}|{st.st_mtime_ns}"


_SHA256_MEMO: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | Path) -> str:
    """SHA-256 hex digest of a file (memoised per path, size and mtime)."""
    import hashlib

    p = Path(path).expanduser().resolve()
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime_ns)
    if key not in _SHA256_MEMO:
        digest = hashlib.sha256()
        with open(p, "rb") as fh:
            for block in iter(lambda: fh.read(8 * 2**20), b""):
                digest.update(block)
        _SHA256_MEMO[key] = digest.hexdigest()
    return _SHA256_MEMO[key]


def training_output_dir(config: AgriboundConfig, engine: str, train_dir: str | Path) -> Path:
    """Directory for one training run's checkpoints and logs.

    Keyed with :func:`agribound._cache.cache_path` by the engine, the chip
    directory, the epochs, the seed and ``engine_params``, so different
    trainings never share (or clear) each other's checkpoints; a retry with
    the same settings reuses the directory.
    """
    from agribound._cache import cache_path

    params = json.dumps(config.engine_params, sort_keys=True, default=str)
    return cache_path(
        config,
        f"{engine}_training",
        "",
        f"chips={Path(train_dir).name}",
        f"epochs={config.fine_tune_epochs}",
        f"seed={config.seed}",
        f"params={params}",
    )


def training_meta_path(checkpoint: str | Path) -> Path:
    """Path of the metadata file written next to a fine-tuned checkpoint."""
    return Path(f"{checkpoint}.agribound.json")


def write_training_meta(checkpoint: str | Path, meta: dict[str, Any]) -> Path:
    """Write training metadata for *checkpoint* (``<checkpoint>.agribound.json``)."""
    path = training_meta_path(checkpoint)
    record = {"checkpoint": Path(checkpoint).name, **meta}
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def read_training_meta(checkpoint: str | Path) -> dict[str, Any]:
    """Training metadata of *checkpoint*, or ``{}`` if none was written."""
    path = training_meta_path(checkpoint)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if record.get("checkpoint") != Path(checkpoint).name:
        return {}
    return record


def read_checkpoint_hparams(path: str | Path) -> dict[str, Any]:
    """Return ``hyper_parameters`` of a Lightning checkpoint (``{}`` if absent).

    The file is unpickled with ``weights_only=False`` (memory-mapped when
    possible), as Lightning does when the checkpoint is loaded for
    inference, so only use checkpoints you trust.
    """
    import torch

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return {}
    return dict(ckpt.get("hyper_parameters") or {})


def grow_labels(labels: np.ndarray, allowed: np.ndarray, steps: int) -> np.ndarray:
    """Grow labelled regions into *allowed* pixels by up to *steps* 4-neighbour steps.

    Breadth-first growth in city-block distance: in each step, every
    unlabelled (0) pixel where *allowed* is True and that has a labelled
    4-neighbour takes the largest label among those neighbours. Each pixel
    ends with at most one label, so grown regions never overlap, and every
    grown pixel is 4-connected to its region through pixels of the same
    label.

    Parameters
    ----------
    labels : numpy.ndarray
        ``(H, W)`` integer labels, 0 = unlabelled.
    allowed : numpy.ndarray
        ``(H, W)`` bool mask of pixels the regions may grow into.
    steps : int
        Maximum growth distance in pixels (city-block).

    Returns
    -------
    numpy.ndarray
        Grown labels (a new array; *labels* is not modified).
    """
    out = np.array(labels, copy=True)
    open_px = np.asarray(allowed, dtype=bool) & (out == 0)
    for _ in range(max(int(steps), 0)):
        if not open_px.any():
            break
        cand = np.zeros_like(out)
        np.maximum(cand[1:, :], out[:-1, :], out=cand[1:, :])
        np.maximum(cand[:-1, :], out[1:, :], out=cand[:-1, :])
        np.maximum(cand[:, 1:], out[:, :-1], out=cand[:, 1:])
        np.maximum(cand[:, :-1], out[:, 1:], out=cand[:, :-1])
        grow = open_px & (cand > 0)
        if not grow.any():
            break
        out[grow] = cand[grow]
        open_px &= ~grow
    return out


def interior_polygons(
    pred_path: str | Path,
    dilate_px: int,
    min_area_m2: float,
    *,
    out_dir: str | Path | None = None,
) -> Any:
    """Field polygons from a 3-class prediction, grown back over the boundary class.

    The training masks label as boundary (class 2) the pixels of a reference
    field within ``boundary_erosion`` pixels (city-block distance) of another
    field or of background, so a predicted interior (class 1) is inset by
    that width. Each 4-connected interior region gets its own label and is
    grown by *dilate_px* 4-neighbour steps into predicted boundary pixels
    (:func:`grow_labels`). Regions never grow into background (class 0,
    which includes pixels without valid data) or into another interior, and
    a pixel reached by two regions in the same step goes to one of them, so
    the polygons of neighbouring fields can share edges but never overlap.

    When the prediction equals the training labels of a field and
    ``dilate_px`` equals ``boundary_erosion``, the result is the field's
    morphological opening by a city-block disc of radius ``dilate_px``: the
    field itself, except that ``dilate_px * (dilate_px + 1) / 2`` pixels are
    missing at each convex right-angled corner of an axis-aligned field
    (e.g. 3 pixels per corner for 2 px).

    The grown label raster is written as ``<stem>_fields_d<dilate_px>.tif``
    (int32, 0 = no field) next to *pred_path*, or in *out_dir*, and
    polygonised with :func:`agribound.postprocess.polygonize.polygonize_mask`.
    An existing file of that name is replaced. Polygons smaller than
    *min_area_m2* are dropped. The whole prediction is held in memory.

    Parameters
    ----------
    pred_path : str or Path
        Single-band class raster (0 background, 1 interior, 2 boundary).
    dilate_px : int
        Growth distance in pixels (0 = interiors only).
    min_area_m2 : float
        Minimum polygon area in m².
    out_dir : str or Path, optional
        Directory for the label raster (created if missing). Default: the
        directory of *pred_path*, as the engines use it (their cache). Pass
        one to polygonise a prediction without writing beside it, for
        example in a finished run's cache.

    Returns
    -------
    geopandas.GeoDataFrame
        One polygon per interior region, with its label as ``instance_id``.
    """
    import rasterio
    from scipy.ndimage import label as nd_label

    from agribound.postprocess.filter import filter_polygons
    from agribound.postprocess.polygonize import polygonize_mask

    pred_path = Path(pred_path)
    with rasterio.open(pred_path) as src:
        pred = src.read(1)
        crs, transform = src.crs, src.transform
    labels, _ = nd_label(pred == 1, structure=_CROSS)
    labels = labels.astype(np.int32, copy=False)
    if dilate_px > 0:
        labels = grow_labels(labels, pred == 2, int(dilate_px))
    label_dir = pred_path.parent if out_dir is None else Path(out_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    label_path = label_dir / f"{pred_path.stem}_fields_d{int(dilate_px)}.tif"
    tmp = label_path.with_name(label_path.stem + ".partial.tif")
    with rasterio.open(
        tmp,
        "w",
        driver="GTiff",
        height=labels.shape[0],
        width=labels.shape[1],
        count=1,
        dtype="int32",
        crs=crs,
        transform=transform,
        nodata=0,
        compress="lzw",
    ) as dst:
        dst.write(labels, 1)
    os.replace(tmp, label_path)
    gdf = polygonize_mask(str(label_path), min_area_m2=0)
    gdf = gdf.rename(columns={"class_value": "instance_id"})
    if len(gdf) and min_area_m2 > 0:
        gdf = filter_polygons(gdf, min_area_m2=min_area_m2)
    return gdf


def predicted_band_thickness(pred: np.ndarray) -> tuple[int | None, int]:
    """Thickness, in pixels, of the predicted boundary band where a field meets background.

    The median, over the boundary pixels (class 2) that have a background
    4-neighbour (class 0), of their city-block distance to the nearest
    interior pixel (class 1), truncated to an integer (a median of 12.5
    gives 12). A model that predicts the band of its training masks
    (:func:`semantic_from_instances`) gives ``boundary_erosion`` here; the
    edge of the array is not background. The whole prediction is held in
    memory, with an int32 distance raster of the same size.

    Parameters
    ----------
    pred : numpy.ndarray
        ``(H, W)`` class raster: 0 background, 1 interior, 2 boundary.

    Returns
    -------
    tuple
        ``(thickness, n_edge)``: the thickness (None when the prediction has
        no interior pixel or no boundary pixel next to background) and the
        number of boundary pixels next to background.
    """
    from scipy import ndimage

    pred = np.asarray(pred)
    bg = pred == 0
    adj = np.zeros_like(bg)
    adj[1:, :] |= bg[:-1, :]
    adj[:-1, :] |= bg[1:, :]
    adj[:, 1:] |= bg[:, :-1]
    adj[:, :-1] |= bg[:, 1:]
    outer = (pred == 2) & adj
    n_edge = int(outer.sum())
    if not (pred == 1).any() or n_edge == 0:
        return None, n_edge
    dist = ndimage.distance_transform_cdt(pred != 1, metric="taxicab")
    return int(np.median(dist[outer])), n_edge


def hf_cached_file_info(
    repo_id: str, filename: str, revision: str | None = None
) -> dict[str, str | None]:
    """Revision and SHA-256 of a file in the local Hugging Face cache.

    Looks the file up with ``huggingface_hub.try_to_load_from_cache`` (no
    network access). The revision is the commit of the cached snapshot that
    *revision* (default: the cached ``main`` ref) points to. The SHA-256 is
    the name of the cached blob, which the Hugging Face cache sets to the
    SHA-256 of the content for Git LFS files (weights files are stored with
    LFS); it is *None* for other files, whose blobs are named by their git
    SHA-1.

    Returns
    -------
    dict
        ``{"repo_id", "filename", "revision", "sha256"}``; ``revision`` and
        ``sha256`` are *None* when the file is not in the cache.
    """
    import re

    info: dict[str, str | None] = {
        "repo_id": repo_id,
        "filename": filename,
        "revision": None,
        "sha256": None,
    }
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return info
    path = try_to_load_from_cache(repo_id, filename, revision=revision)
    if not isinstance(path, str):
        return info
    parts = Path(path).parts
    if "snapshots" in parts:
        idx = len(parts) - 1 - parts[::-1].index("snapshots")
        if idx + 1 < len(parts):
            info["revision"] = parts[idx + 1]
    blob = Path(os.path.realpath(path)).name
    if re.fullmatch(r"[0-9a-f]{64}", blob):
        info["sha256"] = blob
    return info


def package_versions(*dists: str) -> dict[str, str]:
    """``{"<dist>_version": version}`` for the installed distributions."""
    import importlib.metadata as md

    out: dict[str, str] = {}
    for dist in dists:
        try:
            out[f"{dist}_version"] = md.version(dist)
        except md.PackageNotFoundError:
            continue
    return out


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object (``{}`` if missing or unreadable)."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write *data* as indented JSON."""
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def mask_invalid_predictions(
    pred_path: str | Path,
    raster_path: str | Path,
    band_indices: list[int],
    block_rows: int = 1024,
) -> int:
    """Set prediction pixels to 0 where the source bands have no valid data.

    *pred_path* must be a single-band raster on the grid of *raster_path*.
    Validity follows :func:`valid_pixels`. The file is updated in place.

    Returns
    -------
    int
        Number of pixels set to 0.
    """
    import rasterio
    from rasterio.windows import Window

    n_masked = 0
    with rasterio.open(raster_path) as src, rasterio.open(pred_path, "r+") as dst:
        if (src.height, src.width) != (dst.height, dst.width):
            raise ValueError(f"{pred_path} and {raster_path} have different sizes")
        for row in range(0, src.height, block_rows):
            window = Window(0, row, src.width, min(block_rows, src.height - row))
            valid = valid_pixels(src.read(band_indices, window=window), src.nodata)
            if valid.all():
                continue
            pred = dst.read(1, window=window)
            n_masked += int(np.count_nonzero(pred[~valid]))
            pred[~valid] = 0
            dst.write(pred, 1, window=window)
    return n_masked


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def semantic_from_instances(instances: np.ndarray, boundary_erosion: int = 2) -> np.ndarray:
    """Three-class mask from an instance-id mask.

    A reference pixel is *boundary* (2) if a pixel of another instance or of
    background (0) lies within city-block distance ``boundary_erosion``,
    otherwise *interior* (1); background stays 0. For an isolated polygon
    this equals ``scipy.ndimage.binary_erosion`` with the 4-connected
    structuring element applied ``boundary_erosion`` times; unlike eroding
    the union of all polygons, it also separates touching polygons. The
    array edge is not treated as a boundary (pad the input to get exact
    results near an edge).

    Parameters
    ----------
    instances : numpy.ndarray
        ``(H, W)`` integer array, 0 = background.
    boundary_erosion : int
        Boundary width in pixels (0 = no boundary class).

    Returns
    -------
    numpy.ndarray
        uint8 array with values 0, 1, 2.
    """
    from scipy.ndimage import binary_dilation

    inst = np.asarray(instances)
    fg = inst > 0
    sem = np.zeros(inst.shape, dtype=np.uint8)
    sem[fg] = 1
    k = int(boundary_erosion)
    if k <= 0 or not fg.any():
        return sem
    edge = np.zeros(inst.shape, dtype=bool)
    diff_v = inst[1:, :] != inst[:-1, :]
    edge[1:, :] |= diff_v
    edge[:-1, :] |= diff_v
    diff_h = inst[:, 1:] != inst[:, :-1]
    edge[:, 1:] |= diff_h
    edge[:, :-1] |= diff_h
    edge &= fg
    if k > 1:
        edge = binary_dilation(edge, structure=_CROSS, iterations=k - 1) & fg
    sem[edge] = 2
    return sem


def field_interior_depths(
    reference: Any, raster_path: str | Path, grid_gsd_m: float | None = None
) -> np.ndarray:
    """Widest boundary band, in pixels, below which each reference field keeps an interior pixel.

    Each polygon of *reference* (a vector file or a GeoDataFrame) is
    rasterised on the pixel grid of *raster_path* as the training masks are
    (the pixels whose centre it covers; the raster's own edge is not a
    boundary), and its depth is the largest city-block distance of one of its
    pixels from the nearest pixel outside it. In masks with a
    ``boundary_erosion`` of *k* pixels (:func:`semantic_from_instances`) the
    field has an interior pixel exactly when its depth exceeds *k*. Each
    polygon is measured on its own, which is the masks' rule for reference
    polygons that do not overlap; polygons that cover no pixel centre of the
    raster are left out. Only the raster's grid is read, not its pixels. With
    *grid_gsd_m* the depths are those on the grid of DINOv3's processing
    copy with pixels of that size (:func:`dinov3_processing_raster`; for
    example the result of :func:`dinov3_processing_gsd`), not on the
    raster's own grid.

    Returns
    -------
    numpy.ndarray
        int64 depths, one per polygon that covers a pixel centre.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from scipy.ndimage import distance_transform_cdt

    from agribound.io.vector import read_vector

    ref = reference if isinstance(reference, gpd.GeoDataFrame) else read_vector(reference)
    with rasterio.open(raster_path) as src:
        crs, transform, height, width = src.crs, src.transform, src.height, src.width
        if grid_gsd_m is not None:
            transform, height, width, _ = _processing_grid(src, grid_gsd_m)
    if ref.crs is not None and crs is not None and not ref.crs.equals(crs):
        ref = ref.to_crs(crs)
    inv = ~transform
    depths: list[int] = []
    for geom in ref.geometry:
        if geom is None or geom.is_empty:
            continue
        x0, y0, x1, y1 = geom.bounds
        corners = [inv * xy for xy in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]
        cols, rows = [c for c, _ in corners], [r for _, r in corners]
        # One pixel beyond the polygon on every side that is inside the raster.
        c0, c1 = max(int(np.floor(min(cols))) - 1, 0), min(int(np.ceil(max(cols))) + 1, width)
        r0, r1 = max(int(np.floor(min(rows))) - 1, 0), min(int(np.ceil(max(rows))) + 1, height)
        if c1 <= c0 or r1 <= r0:
            continue
        mask = rasterize(
            [(geom, 1)],
            out_shape=(r1 - r0, c1 - c0),
            transform=transform @ Affine.translation(c0, r0),
            fill=0,
            dtype="uint8",
        )
        if not mask.any():
            continue
        if mask.all():  # the polygon covers the whole raster
            depths.append(max(mask.shape))
            continue
        depths.append(int(distance_transform_cdt(mask, metric="taxicab").max()))
    return np.asarray(depths, dtype=np.int64)


def widest_boundary_erosion(
    reference: Any,
    raster_path: str | Path,
    max_px: int = 4,
    max_share: float = NO_INTERIOR_WARN_SHARE,
    grid_gsd_m: float | None = None,
) -> tuple[int, dict[int, float]]:
    """The widest ``boundary_erosion`` that leaves most reference fields an interior pixel.

    A rule on the training labels alone, for coarse imagery: the widest band
    of 1 to *max_px* pixels for which at most *max_share* of the reference
    fields on the grid of *raster_path* have no interior pixel
    (:func:`field_interior_depths`). The default share is the one above which
    the training chips log a WARNING (:data:`NO_INTERIOR_WARN_SHARE`, 10 %),
    and the default *max_px* is DINOv3's default band (4 px). If even a 1 px
    band leaves more fields than that without an interior, the result is 1
    (the narrowest band that still separates touching fields). With
    *grid_gsd_m* the fields are measured on the grid of DINOv3's processing
    copy (see :func:`field_interior_depths`), where DINOv3's chips of a
    coarser raster are cut.

    Returns
    -------
    tuple
        ``(band, shares)``: the band in pixels and, for every band of 1 to
        *max_px* pixels, the share of the fields without an interior pixel
        (0 for every band when no field covers a pixel centre).
    """
    if int(max_px) < 1:
        raise ValueError(f"max_px must be at least 1, got {max_px}")
    depths = field_interior_depths(reference, raster_path, grid_gsd_m=grid_gsd_m)
    shares = {
        k: float(np.mean(depths <= k)) if depths.size else 0.0 for k in range(1, int(max_px) + 1)
    }
    band = max((k for k, share in shares.items() if share <= max_share), default=1)
    return band, shares


# ---------------------------------------------------------------------------
# Chip extraction
# ---------------------------------------------------------------------------


def _gsd_m(src: Any) -> float:
    """Approximate ground sample distance of an open raster, in its CRS's units.

    In a geographic CRS: the pixel width in metres along the parallel through
    the raster's centre (111,320 m per degree times the cosine of the
    latitude). In a projected CRS: the pixel width in the CRS's units, which
    are metres for most projected CRSs but feet for some (US State Plane);
    DINOv3's processing grid converts them (:func:`_grid_pixel_size_m`).
    """
    res = abs(float(src.res[0]))
    if src.crs is not None and src.crs.is_geographic:
        lat = (src.bounds.top + src.bounds.bottom) / 2.0
        return res * 111_320.0 * max(math.cos(math.radians(lat)), 1e-6)
    return res


def _metres_per_unit(crs: Any) -> float:
    """Metres per unit of a projected *crs* (0.3048006 for US survey feet; 1 when unknown)."""
    from rasterio.errors import CRSError

    try:
        return float(crs.linear_units_factor[1])
    except (CRSError, AttributeError, TypeError, ValueError, IndexError):
        return 1.0


def _grid_pixel_size_m(src: Any) -> float:
    """Pixel size in metres that DINOv3's grid rule reads (:func:`dinov3_processing_gsd`).

    :func:`_gsd_m`, with the units of a projected CRS converted to metres
    (``CRS.linear_units_factor``): a 10 m raster in a CRS in US survey feet
    (32.8 ft pixels) measures 10 m. For CRSs in metres, geographic CRSs and
    rasters without a CRS it equals :func:`_gsd_m`.
    """
    if src.crs is not None and src.crs.is_projected:
        return abs(float(src.res[0])) * _metres_per_unit(src.crs)
    return _gsd_m(src)


def _grid_pixel_size_range_m(src: Any) -> tuple[float, float]:
    """Smallest and largest :func:`_grid_pixel_size_m` of a crop of *src*, in metres.

    In a geographic CRS the pixel width in metres depends on the latitude it
    is measured at, the centre of the raster that is read (a crop of *src*
    has its own centre): the extremes are at the latitude of *src* farthest
    from the equator and at the one nearest to it (0 when *src* spans the
    equator). Otherwise both are :func:`_grid_pixel_size_m`.
    """
    if src.crs is not None and src.crs.is_geographic:
        res = abs(float(src.res[0]))
        lo, hi = sorted(max(-90.0, min(90.0, v)) for v in (src.bounds.bottom, src.bounds.top))
        nearest = 0.0 if lo <= 0.0 <= hi else min(lo, hi, key=abs)
        farthest = max(lo, hi, key=abs)

        def size(lat: float) -> float:
            return res * 111_320.0 * max(math.cos(math.radians(lat)), 1e-6)

        return size(farthest), size(nearest)
    size_m = _grid_pixel_size_m(src)
    return size_m, size_m


def _processing_grid(src: Any, gsd_m: float) -> tuple[Any, int, int, float]:
    """``(transform, height, width, scale)`` of the grid with *gsd_m* m pixels over *src*.

    The grid starts at the raster's origin; its pixels are ``scale`` (the
    raster's pixel size in metres, :func:`_grid_pixel_size_m`, over *gsd_m*)
    times smaller on both axes. It holds the grid pixels that lie inside the
    raster (``floor(size x scale)`` per axis): where ``size x scale`` is not a
    whole number (an odd size at 15 m, scale 1.5), a strip narrower than one
    grid pixel along the raster's last column or row is left out, so that no
    pixel of the copy lies outside the raster.
    """
    from rasterio.transform import Affine

    scale = _grid_pixel_size_m(src) / float(gsd_m)
    t = src.transform
    transform = Affine(t.a / scale, t.b / scale, t.c, t.d / scale, t.e / scale, t.f)
    # The tolerance keeps whole-number scales exact (30 m to 10 m: 3 x size).
    height = max(1, int(math.floor(src.height * scale + 1e-6)))
    width = max(1, int(math.floor(src.width * scale + 1e-6)))
    return transform, height, width, scale


def dinov3_processing_gsd(
    raster_path: str | Path, params: dict[str, Any] | None = None
) -> float | None:
    """Pixel size, in metres, of the grid DINOv3 reads *raster_path* on (None: its own).

    ``engine_params["processing_gsd_m"]`` (*params*):

    - ``"auto"`` (the default; also when unset or None):
      :data:`DINOV3_PROCESSING_GSD_M` (10 m) when the raster's pixels are
      more than 1 % larger than that (Landsat and HLS, 30 m; Landsat PAN,
      15 m), else None (Sentinel-1, Sentinel-2, SPOT 6/7, NAIP and finer
      rasters are read as they are);
    - ``"native"``: None, the raster as it is (agribound's behaviour before
      the processing grid);
    - a number of metres finer than the raster's pixels: that size (None
      within 1 % of the raster's own).

    The raster's pixel size is measured in metres (:func:`_grid_pixel_size_m`:
    in a projected CRS in feet, the width in feet converted to metres; in a
    geographic CRS, the width along the parallel through the raster's
    centre). The copy is bilinear (:func:`dinov3_processing_raster`). DINOv3
    reads 16 x 16 px patches and predicts on that patch grid, so a finer grid
    makes each patch cover less ground; it adds no image detail.

    Raises
    ------
    ValueError
        For other values, or a number not finer than the raster's pixels.
    """
    import rasterio

    with rasterio.open(raster_path) as src:
        gsd = _grid_pixel_size_m(src)
    return dinov3_grid_gsd(gsd, params)


def dinov3_grid_gsd(raster_gsd_m: float, params: dict[str, Any] | None = None) -> float | None:
    """The rule of :func:`dinov3_processing_gsd` for a raster with *raster_gsd_m* m pixels.

    For when the raster is not at hand: :func:`agribound._results.results_versions`
    applies it to the pixel size of a source's composite.

    Raises
    ------
    ValueError
        As :func:`dinov3_processing_gsd`.
    """
    import numbers

    value = (params or {}).get("processing_gsd_m", "auto")
    gsd = float(raster_gsd_m)
    if value is None or (isinstance(value, str) and value.strip().lower() == "auto"):
        target = DINOV3_PROCESSING_GSD_M
        return target if gsd > target * (1 + _DINOV3_GRID_TOLERANCE) else None
    if isinstance(value, str) and value.strip().lower() == "native":
        return None
    ok = isinstance(value, numbers.Real) and not isinstance(value, bool)
    if not (ok and math.isfinite(value) and value > 0):
        raise ValueError(
            "engine_params['processing_gsd_m'] for DINOv3 must be 'auto', 'native' or a "
            f"pixel size in metres, got {value!r}"
        )
    target = float(value)
    if abs(gsd / target - 1.0) <= _DINOV3_GRID_TOLERANCE:
        return None
    if target > gsd:
        raise ValueError(
            f"engine_params['processing_gsd_m']={target!r} m is coarser than the raster's "
            f"{gsd:.2f} m pixels; DINOv3's processing grid only refines the grid"
        )
    return target


def _write_dinov3_grid(src_path: str | Path, dst_path: Path, bands: list[int], gsd_m: float):
    """Write the bilinear copy of *bands* of *src_path* on the grid of :func:`_processing_grid`."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    tmp = dst_path.with_name(dst_path.stem + ".partial" + dst_path.suffix)
    with rasterio.open(src_path) as src:
        transform, height, width, scale = _processing_grid(src, gsd_m)
        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": len(bands),
            "dtype": src.dtypes[bands[0] - 1],
            "crs": src.crs,
            "transform": transform,
            "nodata": src.nodata,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "BIGTIFF": "IF_SAFER",
        }
        vrt_options: dict[str, Any] = {
            "crs": src.crs,
            "transform": transform,
            "width": width,
            "height": height,
            "resampling": Resampling.bilinear,
        }
        if src.nodata is not None:
            vrt_options.update(src_nodata=src.nodata, nodata=src.nodata)
        tags = src.tags()
        descriptions = [src.descriptions[b - 1] for b in bands]
        with WarpedVRT(src, **vrt_options) as vrt, rasterio.open(tmp, "w", **profile) as dst:
            for row in range(0, height, 1024):
                window = Window(0, row, width, min(1024, height - row))
                dst.write(vrt.read(bands, window=window), window=window)
            dst.update_tags(
                **tags,
                AGRIBOUND_DINOV3_GRID=(
                    f"bilinear copy of bands {bands} of {Path(src_path).name}, "
                    f"{float(gsd_m):g} m pixels (x{scale:g})"
                ),
            )
            for i, description in enumerate(descriptions, start=1):
                if description:
                    dst.set_band_description(i, description)
    os.replace(tmp, dst_path)


def dinov3_processing_raster(
    raster_path: str | Path,
    config: AgriboundConfig,
    gsd_m: float | None,
    bands: list[int],
) -> tuple[str, list[int], dict[str, Any] | None]:
    """The raster DINOv3 reads: *raster_path*, or its copy on a grid of *gsd_m* pixels.

    Without *gsd_m* (see :func:`dinov3_processing_gsd`) this is the raster
    itself. Otherwise it is a copy of the distinct *bands*, in increasing
    order, resampled bilinearly (GDAL) onto the grid that starts at the
    raster's origin with pixels of *gsd_m* metres (in the raster's CRS
    units) and holds the grid pixels inside the raster (see
    :func:`_processing_grid`; nodata pixels are left out of the
    interpolation). It is written once to the working directory
    (``dinov3_grid_<key>.tif``, keyed by the raster's name, size and
    modification time, *gsd_m*, the bands and :data:`DINOV3_GRID_VERSION`)
    and reused. Polygons delineated on it are in the raster's CRS.

    Returns
    -------
    tuple
        ``(path, band indices in that raster, info)``; *info* is None for the
        raster itself, else ``{"gsd_m", "scale", "size_px", "resampling",
        "raster", "version"}``.
    """
    import rasterio

    from agribound._cache import cache_path

    if gsd_m is None:
        return str(raster_path), list(bands), None
    distinct = sorted({int(b) for b in bands})
    path = cache_path(
        config,
        "dinov3_grid",
        ".tif",
        raster_fingerprint(raster_path),
        f"gsd_m={float(gsd_m)!r}",
        f"bands={distinct}",
        f"version={DINOV3_GRID_VERSION}",
    )
    with rasterio.open(raster_path) as src:
        raster_gsd = _grid_pixel_size_m(src)
    if path.exists():
        logger.info("Using the cached %.2f m copy of the raster for DINOv3: %s", gsd_m, path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_dinov3_grid(raster_path, path, distinct, float(gsd_m))
        logger.info(
            "Resampled the raster from %.2f m to %.2f m pixels for DINOv3 (bilinear; "
            "engine_params['processing_gsd_m']='native' reads it as it is): %s",
            raster_gsd,
            gsd_m,
            path,
        )
    scale = raster_gsd / float(gsd_m)
    with rasterio.open(path) as out:
        info = {
            "gsd_m": float(gsd_m),
            "scale": round(scale, 6),
            "size_px": [out.height, out.width],
            "resampling": "bilinear",
            "raster": str(path),
            "version": DINOV3_GRID_VERSION,
        }
    return str(path), [distinct.index(int(b)) + 1 for b in bands], info


def _chip_gsd_m(src: Any, engine: str) -> float:
    """GSD in metres used to choose the default chip size.

    For ``delineate-anything`` this is the engine's own rule,
    :func:`agribound.engines.delineate_anything.pixel_size_m` (mean of the
    east-west and north-south pixel sides; the rule the super-resolution
    factor of inference and of ``_yolo`` uses), so the 4 m threshold of the
    chip size and of the super-resolution factor agree. ``dinov3`` uses the
    measure of its processing grid, :func:`_grid_pixel_size_m` (metres also
    in a projected CRS in feet; its chip size does not depend on it). Other
    engines (and rasters without a CRS) use :func:`_gsd_m`.
    """
    if engine == "delineate-anything" and src.crs is not None:
        from agribound.engines.delineate_anything import pixel_size_m

        return float(pixel_size_m(src.crs, src.transform, src.height, src.width))
    if engine == "dinov3":
        return _grid_pixel_size_m(src)
    return _gsd_m(src)


def default_boundary_erosion(engine: str) -> int:
    """Default boundary width of the training masks of *engine*, in pixels.

    :data:`DEFAULT_BOUNDARY_EROSION` (4 for ``dinov3``), else 2.
    """
    return DEFAULT_BOUNDARY_EROSION.get(engine, 2)


def _default_chip_size(engine: str, gsd_m: float) -> int:
    if engine == "delineate-anything":
        return 512 if gsd_m < 4.0 else 256
    return _DEFAULT_CHIP_SIZE[engine]


def chip_stride(engine: str, chip_size: int, params: dict[str, Any] | None = None) -> int:
    """Step between the origins of neighbouring training chips, in pixels.

    ``engine_params["chip_stride"]`` when it is set (an integer from 1 to
    *chip_size*). Otherwise half the chip size for ``delineate-anything``, so
    that the chips overlap, as its inference tiles do (512 px tiles every
    256 px), and every field less than half a chip across lies whole in some
    chip; and the chip size for the other engines (chips that do not overlap).

    Raises
    ------
    ValueError
        If ``chip_stride`` is not an integer from 1 to *chip_size*.
    """
    import numbers

    value = (params or {}).get("chip_stride")
    if value is None:
        return chip_size // 2 if engine == "delineate-anything" else chip_size
    ok = isinstance(value, numbers.Real) and not isinstance(value, bool)
    ok = ok and math.isfinite(value) and float(value).is_integer() and 1 <= value <= chip_size
    if not ok:
        raise ValueError(
            f"engine_params['chip_stride'] must be an integer from 1 to the chip size "
            f"({chip_size} px), got {value!r}"
        )
    return int(value)


def _cell_splits(
    records: list[dict[str, Any]],
    chip_size: int,
    height: int,
    width: int,
    transform: Any,
    crs: Any,
    config: AgriboundConfig,
    reference: Any,
    info: dict[str, Any],
) -> list[str | None]:
    """Split chips through the chip-size cells they cover (see ``_prepare_training_data``).

    The cells tile the raster from its origin in steps of *chip_size*; a cell
    is a unit of :func:`assign_splits` when a chip covers part of it (cells in
    row-major order, footprints clipped to the raster). A chip takes the split
    of its cells when they all have the same split; otherwise it is dropped
    (``None``). Chips that share a pixel share the cell that holds it, so a
    pixel is never in a training and a validation chip. With chips that do not
    overlap (stride = chip size), every chip is its own cell and the split is
    that of :func:`assign_splits` on the chips themselves.

    Returns
    -------
    list
        ``"train"``, ``"val"`` or ``None`` (dropped) per record, in order.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window
    from shapely.geometry import box

    def cell_range(offset: int, size: int) -> range:
        return range(offset // chip_size, (offset + size - 1) // chip_size + 1)

    chip_cells = [
        [
            (r, c)
            for r in cell_range(rec["row_off"], rec["size"])
            for c in cell_range(rec["col_off"], rec["size"])
        ]
        for rec in records
    ]
    cells = sorted({cell for covered in chip_cells for cell in covered})
    index = {cell: i for i, cell in enumerate(cells)}
    footprints = []
    for r, c in cells:
        row, col = r * chip_size, c * chip_size
        window = Window(col, row, min(chip_size, width - col), min(chip_size, height - row))
        footprints.append(box(*rasterio.windows.bounds(window, transform)))
    units = gpd.GeoDataFrame(geometry=footprints, crs=crs)
    cell_split = assign_splits(units, config, reference=reference, info=info)
    out: list[str | None] = []
    for covered in chip_cells:
        found = {str(cell_split[index[cell]]) for cell in covered}
        out.append(found.pop() if len(found) == 1 else None)
    return out


def reference_field_sides_m(reference: Any) -> np.ndarray:
    """Bounding-box side (the longer of width and height) of each reference field, in metres.

    Measured in the UTM zone of the layer (``estimate_utm_crs``); empty geometries are
    dropped.
    """
    ref = reference[reference.geometry.notna() & ~reference.geometry.is_empty]
    if len(ref) == 0:
        return np.zeros(0)
    if ref.crs is not None and not ref.crs.is_projected:
        ref = ref.to_crs(ref.estimate_utm_crs())
    b = ref.geometry.bounds
    return np.maximum(b["maxx"] - b["minx"], b["maxy"] - b["miny"]).to_numpy(dtype=float)


def geoai_chip_size_for_fields(sides_m: np.ndarray, gsd_m: float) -> int:
    """Default GeoAI chip size (pixels) that fits the reference fields; see GEOAI_* above."""
    if sides_m.size == 0 or not gsd_m > 0:
        return _DEFAULT_CHIP_SIZE["geoai"]
    side_px = float(np.quantile(sides_m, GEOAI_FIELD_QUANTILE)) / gsd_m
    chip = int(math.ceil(GEOAI_FIELD_MARGIN * side_px / 32.0) * 32)
    return int(min(max(chip, GEOAI_MIN_CHIP), GEOAI_MAX_CHIP))


def _prepare_training_data(raster_path: str, config: AgriboundConfig, engine: str) -> Path:
    """Cut training chips and masks from a composite and split them.

    Parameters
    ----------
    raster_path : str
        Composite GeoTIFF.
    config : AgriboundConfig
        Configuration with ``reference_boundaries``. Relevant
        ``engine_params``: ``chip_size`` (pixels; default 224 for prithvi,
        256 for dinov3, for geoai sized from the reference fields (see
        :func:`geoai_chip_size_for_fields`: 1.25 times the 90th-percentile
        field bounding-box side, 256-1024 px), and 512 below 4 m GSD / 256 otherwise for
        delineate-anything, with the GSD of
        :func:`agribound.engines.delineate_anything.pixel_size_m` for the
        whole raster), ``boundary_erosion`` (boundary width in pixels,
        default :func:`default_boundary_erosion`: 4 for dinov3, else 2),
        ``min_label_fraction`` (minimum fraction of chip pixels
        inside reference polygons, default 0.01), ``min_valid_fraction``
        (minimum fraction of valid image pixels, default 0.5) and, for
        prithvi on local rasters, ``value_scale``. For delineate-anything,
        ``processing_gsd_m`` and ``stretch_percentiles`` as at inference: the
        chips are cut from the raster the native backend reads
        (:func:`agribound.engines.delineate_anything.processing_raster`, the
        copy at ``processing_gsd_m``, if set; its pixel size then sets the
        default chip size), with the same stretch percentiles, and both enter
        the cache key when set (with the version of the copy when one is
        read). For dinov3, ``processing_gsd_m`` (see
        :func:`dinov3_processing_gsd`; by default 10 m for rasters coarser than
        10 m): the chips, and the default chip size, are those of the bilinear
        copy on that grid (:func:`dinov3_processing_raster`), and the grid enters
        the cache key when a copy is read. ``chip_stride`` (pixels; see
        :func:`chip_stride`: half the chip size for delineate-anything, the
        chip size for the other engines). ``config.bands`` overrides the band
        lookup.
    engine : str
        Engine the chips are for (selects bands and chip format, see the
        module docstring).

    Returns
    -------
    pathlib.Path
        Training directory (layout in the module docstring). A complete
        directory with the same cache key is reused.

    Raises
    ------
    ValueError
        If the reference layer is empty or does not overlap the raster, if
        fewer than two chips qualify or one split is left without chips, or
        for unsupported engines/radiometry (including a raster of class codes,
        whose ``AGRIBOUND_VALUE_SCALE`` tag is ``"categorical"``).

    Notes
    -----
    Chips are ``chip_size`` windows on the raster grid, starting at the
    top-left corner every ``chip_stride`` pixels on both axes; partial windows
    at the right and bottom edges are skipped. Masks are rasterised per chip
    from the reference polygons (pixel centres inside a polygon), with a
    ``boundary_erosion`` margin so that boundaries at chip edges are exact.

    Pixels are valid as in :func:`valid_pixels`; delineate-anything chips
    also treat pixels that are 0 in every band as invalid
    (``zero_invalid=True``), as its native backend does at inference, and
    their image is 0 at every invalid pixel (uint8 rasters included), as the
    backend's tiles are.

    The split is made on chip-size cells, not on the chips: the raster is cut
    into cells of ``chip_size`` pixels from its origin, every cell that a
    qualifying chip covers part of is a unit of :func:`assign_splits`
    (``fine_tune_val_split`` is then the fraction of these cells), and a chip
    takes the split of its cells. A chip whose cells are in different splits
    is dropped, so two chips that share a pixel (and so the cell that holds
    it) are always in the same split. When the chips do not overlap
    (``chip_stride`` equal to ``chip_size``) each chip is its own cell, and
    no chip is dropped.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from rasterio.windows import transform as window_transform
    from shapely.geometry import box

    from agribound._cache import cache_path
    from agribound.engines.base import get_canonical_band_indices
    from agribound.engines.finetune import _reference_fingerprint
    from agribound.io.raster import refuse_categorical
    from agribound.io.vector import read_vector

    refuse_categorical(raster_path, config, "Fine-tuning")
    if engine not in CHIP_FORMATS:
        raise ValueError(
            f"No training-chip definition for engine {engine!r} "
            f"(supported: {', '.join(CHIP_FORMATS)})"
        )
    if not config.reference_boundaries:
        raise ValueError("reference_boundaries is required to prepare training data")
    params = config.engine_params
    chip_format = CHIP_FORMATS[engine]
    bands = effective_bands(config)
    names = engine_band_names(engine, config.source, bands)
    band_indices = get_canonical_band_indices(config.source, names, bands=bands)
    erosion = int(params.get("boundary_erosion", default_boundary_erosion(engine)))
    min_label = float(params.get("min_label_fraction", 0.01))
    min_valid = float(params.get("min_valid_fraction", 0.5))
    value_scale = params.get("value_scale")

    with rasterio.open(raster_path) as src:
        if max(band_indices) > src.count:
            raise ValueError(
                f"Engine {engine!r} needs bands {names} at indices {band_indices}, but "
                f"{raster_path} has {src.count} bands"
            )
    # Delineate-Anything chips are cut from the raster its native backend reads (the copy
    # at engine_params["processing_gsd_m"], if set) with its stretch percentiles.
    read_path, read_bands, processing = str(raster_path), list(band_indices), None
    processing_gsd = None
    percentiles = _DEFAULT_PERCENTILES
    options_key: list[str] = []
    if engine == "delineate-anything":
        from agribound.engines.delineate_anything import (
            _PROCESSING_RASTER_VERSION,
            DAOptions,
            processing_raster,
        )

        da_opts = DAOptions.from_engine_params(params)
        processing_gsd, percentiles = da_opts.processing_gsd_m, da_opts.stretch_percentiles
        read_path, read_bands, processing = processing_raster(
            str(raster_path), config, processing_gsd, band_indices
        )
        if processing_gsd is not None:
            options_key.append(f"processing_gsd_m={processing_gsd!r}")
        if processing is not None:  # chips cut from an older copy are not reused
            options_key.append(f"processing_raster={_PROCESSING_RASTER_VERSION}")
        if percentiles != _DEFAULT_PERCENTILES:
            options_key.append(f"stretch_percentiles={list(percentiles)!r}")
    elif engine == "dinov3":
        # DINOv3 chips of rasters coarser than 10 m are cut from the bilinear 10 m copy
        # that its inference reads too (engine_params["processing_gsd_m"]).
        dinov3_gsd = dinov3_processing_gsd(raster_path, params)
        if dinov3_gsd is not None:
            read_path, read_bands, processing = dinov3_processing_raster(
                str(raster_path), config, dinov3_gsd, band_indices
            )
            processing_gsd = dinov3_gsd
            options_key += [
                f"processing_gsd_m={dinov3_gsd!r}",
                f"dinov3_grid={DINOV3_GRID_VERSION}",
            ]
    with rasterio.open(read_path) as src:
        gsd = _chip_gsd_m(src, engine)
    field_sides_m = None
    if engine == "geoai":
        field_sides_m = reference_field_sides_m(read_vector(config.reference_boundaries))
    if params.get("chip_size"):
        chip_size = int(params["chip_size"])
    elif engine == "geoai":
        chip_size = geoai_chip_size_for_fields(field_sides_m, gsd)
        logger.info(
            "GeoAI chip size %d px (%.0f m): fits the reference fields (90th-percentile "
            "bounding-box side %.0f m x %.2f, clamped to %d-%d px; engine_params['chip_size'] "
            "overrides)",
            chip_size,
            chip_size * gsd,
            float(np.quantile(field_sides_m, GEOAI_FIELD_QUANTILE)) if field_sides_m.size else 0.0,
            GEOAI_FIELD_MARGIN,
            GEOAI_MIN_CHIP,
            GEOAI_MAX_CHIP,
        )
    else:
        chip_size = _default_chip_size(engine, gsd)
    if field_sides_m is not None and field_sides_m.size:
        share = float(np.mean(field_sides_m > chip_size * gsd))
        if share > GEOAI_LARGE_FIELD_WARN_SHARE:
            logger.warning(
                "%.0f %% of the reference fields are larger than a %d px (%.0f m) GeoAI chip. "
                "GeoAI predicts fields within inference windows of the chip size; no window sees "
                "the whole of those fields, and the engine rejoins their pieces only where they "
                "meet at a window edge (engine_params['merge_window_seams']). Use a larger "
                "engine_params['chip_size'] (Mask R-CNN keeps at most 100 detections per "
                "window) or coarser imagery.",
                100 * share,
                chip_size,
                chip_size * gsd,
            )
    if chip_size < 16:
        raise ValueError(f"chip_size must be >= 16 pixels, got {chip_size}")
    stride = chip_stride(engine, chip_size, params)
    # Delineate-Anything's native backend treats pixels that are 0 in every band as nodata.
    zero_invalid = engine == "delineate-anything"
    if zero_invalid:
        options_key += [f"stride={stride}", "valid=zero-invalid"]
    elif stride != chip_size:
        options_key.append(f"stride={stride}")
    if chip_format == "prithvi_x10000":
        # Fail before any file is written if the source is not reflectance.
        prithvi_reflectance(np.zeros((1, 1, 1), np.float32), config.source, value_scale)

    key_parts = [
        f"engine={engine}",
        f"format={chip_format}",
        f"bands={names}:{band_indices}",
        f"chip={chip_size}",
        f"erosion={erosion}",
        f"min_label={min_label}",
        f"min_valid={min_valid}",
        f"value_scale={value_scale}",
        f"reference={_reference_fingerprint(config.reference_boundaries)}",
        f"raster={raster_fingerprint(raster_path)}",
        f"split={config.fine_tune_split}:{config.fine_tune_val_split}:"
        f"{config.fine_tune_block_size_m}:{config.fine_tune_split_column}",
        f"seed={config.seed}",
        *options_key,
    ]
    train_dir = cache_path(config, f"finetune_data_{engine}", "", *key_parts)
    if (train_dir / _META_NAME).is_file():
        logger.info("Using cached training data: %s", train_dir)
        return train_dir
    if train_dir.exists():
        logger.info("Removing incomplete training data directory %s", train_dir)
        shutil.rmtree(train_dir)
    staging = train_dir.with_name(train_dir.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    kinds = ("images", "masks", "instances")
    dirs = {
        (split, kind): staging / (f"{prefix}{kind}")
        for split, prefix in ((_SPLIT_TRAIN, ""), (_SPLIT_VAL, "val_"))
        for kind in kinds
    }
    pending = {kind: staging / "_pending" / kind for kind in kinds}
    for d in (*dirs.values(), *pending.values()):
        d.mkdir(parents=True, exist_ok=True)

    ref = read_vector(config.reference_boundaries)
    if len(ref) == 0:
        raise ValueError(f"Reference layer {config.reference_boundaries} is empty")
    if ref.crs is None:
        raise ValueError(f"Reference layer {config.reference_boundaries} has no CRS")

    with rasterio.open(read_path) as src:
        crs = src.crs
        nodata = src.nodata
        dtype = np.dtype(src.dtypes[read_bands[0] - 1])
        ref_r = ref.to_crs(crs) if not ref.crs.equals(crs) else ref
        ref_r = ref_r.reset_index(drop=True)
        keep = ref_r.geometry.notna() & ~ref_r.geometry.is_empty
        ref_r = ref_r[keep]
        ref_ids = (ref_r.index.to_numpy() + 1).astype(np.int64)
        ref_geoms = ref_r.geometry.to_numpy()
        sindex = ref_r.sindex

        transform_info: dict[str, Any]
        if chip_format == "prithvi_x10000":
            from agribound.engines.prithvi import PRITHVI_MEAN

            fill = np.asarray(PRITHVI_MEAN, dtype=np.float32)[:, None, None]
            transform_info = {
                "format": chip_format,
                "value_scale": value_scale or None,
                "invalid_fill": "prithvi_mean",
            }
            lows = highs = None
        else:
            lows, highs = scene_stretch_bounds(read_path, read_bands, *percentiles)
            transform_info = {
                "format": chip_format,
                "stretch_percentiles": list(percentiles),
                "stretch_lows": lows,
                "stretch_highs": highs,
            }

        records: list[dict[str, Any]] = []
        n_skipped_label = n_skipped_valid = 0
        ids_in_chips: set[int] = set()
        ids_with_interior: set[int] = set()
        chip_id = 0
        pad = max(erosion, 0)
        for row in range(0, src.height - chip_size + 1, stride):
            for col in range(0, src.width - chip_size + 1, stride):
                window = Window(col, row, chip_size, chip_size)
                chip_tf = window_transform(window, src.transform)
                chip_box = box(*rasterio.windows.bounds(window, src.transform))
                pad_window = Window(col - pad, row - pad, chip_size + 2 * pad, chip_size + 2 * pad)
                pad_tf = window_transform(pad_window, src.transform)
                pad_box = box(*rasterio.windows.bounds(pad_window, src.transform))
                hits = sindex.query(pad_box, predicate="intersects")
                if len(hits) == 0:
                    continue
                order = np.sort(hits)
                inst_pad = rasterize(
                    [(ref_geoms[i], int(ref_ids[i])) for i in order],
                    out_shape=(chip_size + 2 * pad, chip_size + 2 * pad),
                    transform=pad_tf,
                    fill=0,
                    dtype="int32",
                )
                inst = inst_pad[pad : pad + chip_size, pad : pad + chip_size]
                label_frac = float(np.count_nonzero(inst)) / inst.size
                if label_frac < min_label:
                    n_skipped_label += 1
                    continue
                data = src.read(read_bands, window=window)
                valid = valid_pixels(data, nodata, zero_invalid=zero_invalid)
                if float(valid.mean()) < min_valid:
                    n_skipped_valid += 1
                    continue
                sem = semantic_from_instances(inst_pad, erosion)[
                    pad : pad + chip_size, pad : pad + chip_size
                ]
                sem = sem.copy()
                sem[~valid] = IGNORE_INDEX
                inst = np.where(valid, inst, 0).astype(np.int32)
                ids_in_chips.update(np.unique(inst[inst > 0]).tolist())
                ids_with_interior.update(np.unique(inst[sem == 1]).tolist())
                if chip_format == "prithvi_x10000":
                    image = prithvi_reflectance(data, config.source, value_scale)
                    image = np.where(valid[None], image, fill).astype(np.float32)
                else:
                    image = apply_stretch(data, lows, highs, valid)
                    if zero_invalid:
                        image[:, ~valid] = 0  # as in the native backend's tiles, uint8 too
                    if chip_format == "rgb_unit_float32":
                        image = image.astype(np.float32) / np.float32(255.0)
                name = f"chip_{chip_id:05d}.tif"
                _write_chip(pending["images"] / name, image, crs, chip_tf)
                _write_chip(pending["masks"] / name, sem[np.newaxis], crs, chip_tf)
                _write_chip(pending["instances"] / name, inst[np.newaxis], crs, chip_tf)
                records.append(
                    {
                        "chip_id": chip_id,
                        "row_off": row,
                        "col_off": col,
                        "size": chip_size,
                        "label_fraction": label_frac,
                        "geometry": chip_box,
                    }
                )
                chip_id += 1

    if len(records) < 2:
        shutil.rmtree(staging, ignore_errors=True)
        raise ValueError(
            f"Only {len(records)} training chip(s) of {chip_size} px qualified "
            f"(skipped {n_skipped_label} with < {min_label:.0%} reference pixels and "
            f"{n_skipped_valid} with < {min_valid:.0%} valid pixels); at least 2 are needed. "
            "Check that the reference polygons overlap the composite, or lower "
            "engine_params['chip_size']."
        )

    split_info: dict[str, Any] = {}
    with rasterio.open(read_path) as src:
        height, width, transform = src.height, src.width, src.transform
    splits = _cell_splits(
        records, chip_size, height, width, transform, crs, config, ref_r, split_info
    )
    kept = [split is not None for split in splits]
    n_dropped = len(records) - sum(kept)
    if stride < chip_size:
        # assign_splits counted the cells; report the chips too.
        split_info.update(
            unit="chip-size cell",
            n_train_cells=split_info.pop("n_train"),
            n_val_cells=split_info.pop("n_val"),
            n_train=splits.count(_SPLIT_TRAIN),
            n_val=splits.count(_SPLIT_VAL),
            n_dropped_both_splits=n_dropped,
        )
    for split in (_SPLIT_TRAIN, _SPLIT_VAL):
        if split not in splits:
            shutil.rmtree(staging, ignore_errors=True)
            raise ValueError(
                f"No {'training' if split == _SPLIT_TRAIN else 'validation'} chip is left "
                "after dropping the chips that cover chip-size cells of both splits "
                f"({n_dropped} of {len(records)}). Use a larger reference area, another "
                "fine_tune_split or fine_tune_val_split, or "
                f"engine_params['chip_stride']={chip_size} (chips that do not overlap)."
            )

    units = gpd.GeoDataFrame(
        [rec for rec, keep in zip(records, kept, strict=True) if keep], geometry="geometry", crs=crs
    )
    units["split"] = [split for split in splits if split is not None]
    for rec, split in zip(records, splits, strict=True):
        name = f"chip_{rec['chip_id']:05d}.tif"
        for kind in kinds:
            if split is None:
                (pending[kind] / name).unlink()
            else:
                os.replace(pending[kind] / name, dirs[(split, kind)] / name)
    shutil.rmtree(staging / "_pending")

    units.to_file(staging / "chips.gpkg", layer="chips", driver="GPKG")
    n_fields = len(ids_in_chips)
    n_no_interior = len(ids_in_chips - ids_with_interior)
    if engine in MASK_ENGINES and n_fields and n_no_interior > NO_INTERIOR_WARN_SHARE * n_fields:
        logger.warning(
            "%d of the %d reference fields in the training chips (%.0f %%) have no field-interior "
            "pixel with a boundary band of %d px: fields less than about %d px (%.0f m) across "
            "are labelled boundary only, so the model cannot learn them as fields. Use a smaller "
            "engine_params['boundary_erosion'] (the default for %s is %d px).",
            n_no_interior,
            n_fields,
            100 * n_no_interior / n_fields,
            erosion,
            2 * erosion + 1,
            (2 * erosion + 1) * gsd,
            engine,
            default_boundary_erosion(engine),
        )
    meta = {
        "engine": engine,
        "source": config.source,
        "raster": raster_fingerprint(raster_path),
        "raster_dtype": str(dtype),
        "band_names": names,
        "band_indices": band_indices,
        "chip_size": chip_size,
        "chip_stride": stride,
        "gsd_m": gsd,
        "processing_gsd_m": processing_gsd,
        "processing": processing,
        "boundary_erosion": erosion,
        "ignore_index": IGNORE_INDEX,
        "zero_pixels_invalid": zero_invalid,
        "semantic_classes": {str(k): v for k, v in SEMANTIC_CLASSES.items()},
        "instance_ids": "1-based row position in the reference layer",
        "image": transform_info,
        "n_chips": len(units),
        "n_skipped_low_label": n_skipped_label,
        "n_skipped_low_valid": n_skipped_valid,
        "n_reference_fields": n_fields,
        "n_reference_fields_without_interior": n_no_interior if engine in MASK_ENGINES else None,
        "split": split_info,
        "seed": config.seed,
        "reference": _reference_fingerprint(config.reference_boundaries),
    }
    (staging / _META_NAME).write_text(json.dumps(meta, indent=2, default=str))
    os.replace(staging, train_dir)
    logger.info(
        "Prepared %d training chips (%d px every %d px, %s): %d train, %d val (%s split)%s",
        len(units),
        chip_size,
        stride,
        chip_format,
        split_info.get("n_train", 0),
        split_info.get("n_val", 0),
        split_info.get("strategy"),
        f"; {n_dropped} chips covering cells of both splits dropped" if n_dropped else "",
    )
    return train_dir


def _write_chip(path: Path, data: np.ndarray, crs: Any, transform: Any) -> None:
    """Write one chip GeoTIFF (no nodata value, LZW)."""
    import rasterio

    count, height, width = data.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs=crs,
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(data)


def read_chip_meta(train_dir: str | Path) -> dict[str, Any]:
    """Return the ``chips_meta.json`` of a training directory."""
    return json.loads((Path(train_dir) / _META_NAME).read_text())


def split_files(train_dir: str | Path, split: str, kind: str) -> list[Path]:
    """Sorted chip files of one split (``"train"``/``"val"``) and kind.

    *kind* is ``"images"``, ``"masks"`` or ``"instances"``.
    """
    prefix = "" if split == _SPLIT_TRAIN else "val_"
    return sorted((Path(train_dir) / f"{prefix}{kind}").glob("chip_*.tif"))
