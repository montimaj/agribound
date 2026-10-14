"""
Delineate-Anything engine (YOLO11-seg instance segmentation of field boundaries).

Models
------
Weights come from the Hugging Face repository ``MykolaL/DelineateAnything`` at
pinned revisions; the SHA-256 of every downloaded file is checked against
:data:`DA_MODELS` before it is used.

- ``large_v2`` (default): ``DelineateAnythingv2.pt``, Delineate Anything v2,
  YOLO11x-seg trained on FBIS-73M; default confidence 0.15; FTW registry name
  ``DelineateAnythingV2``.
- ``large``: ``DelineateAnything.pt``, YOLO11x-seg trained on FBIS-22M;
  default confidence 0.005; FTW name ``DelineateAnything``.
- ``small``: ``DelineateAnything-S.pt``, YOLO11n-seg trained on FBIS-22M;
  default confidence 0.005; FTW name ``DelineateAnything-S``.

The default confidences are those of the upstream ``conf_sample.yaml``: 0.15
for ``large_v2`` (Lavreniuk/Delineate-Anything a6f30b2) and 0.005 for the v1
models (the v1-era sample configuration). Select a model with
``engine_params["da_model"]`` (a key or one of the aliases
``"DelineateAnythingV2"``, ``"DelineateAnything"``, ``"DelineateAnything-S"``);
the legacy ``engine_params["model_size"]`` (``"large"``/``"small"``) selects
the v1 models.

Backends
--------
``engine_params["backend"]`` chooses the implementation explicitly (default
``"native"``). There is no automatic fallback between backends: a backend that
cannot run raises an error that says what is missing.

``"native"``
    Agribound's own tiled Ultralytics inference. It reproduces the
    *preprocessing* of the reference Delineate-Anything pipeline (DelAnyFlow,
    upstream ``methods/main``): a scene-level per-band 1-99 percentile stretch
    (``stretch_percentiles``) to uint8, computed on valid, strictly positive
    pixels sampled (nearest neighbour, full-resolution data, never overviews)
    on a grid of at most 4096 px per side (uint8 rasters are used unchanged);
    tiles of 512 native pixels when the ground sampling distance (GSD) is
    below 4 m, else 256 native pixels upsampled 2x with bicubic interpolation,
    so the model input is always 512 x 512 (``super_resolution`` = 1, 2 or 4
    overrides the factor); 50 % tile overlap starting half a tile before the
    raster origin, as the upstream ``ExecutionPlanner`` does; BGR channel
    order for NumPy input to Ultralytics; ``retina_masks=True``; masks cast to
    float before the upstream 3 x 3 erode / dilate / dilate / erode morphology
    (Ultralytics >= 8.3.217 returns uint8 masks, on which the upstream
    negation trick would turn the erosion into a dilation); FP16 on GPU/MPS.
    Each detection becomes the largest polygon of its mask, clipped to valid
    pixels, and is *tile-cut* when that polygon reaches into the strip of
    :data:`TILE_EDGE_STRIP_PX` (4) model-input pixels along an interior tile
    edge, the edge strip DelAnyFlow marks (``merging_edge_width``). The
    detections of all tiles are then combined at polygon level, with
    duplicates defined as IoU >= ``dedup_iou`` or intersection >=
    ``dedup_containment`` of the smaller polygon: (1) tile-cut duplicates of
    one another are merged into their union, which stays tile-cut
    (:func:`merge_tile_pieces`; ``merge_tile_pieces=False`` skips it), so a
    field too large to be complete in any tile is rebuilt from its pieces
    (the polygon-level counterpart of DelAnyFlow's merging of fields that
    touch a tile border); pieces that duplicate a complete (not tile-cut), at
    least as large detection are not merged; (2) greedy non-maximum
    suppression visits the polygons by higher confidence, then larger area,
    and drops every polygon that duplicates one already kept, except that a
    complete detection is visited before each tile-cut duplicate that is not
    larger than it (:func:`deduplicate_detections`); (3) remaining overlaps
    go to complete polygons before tile-cut ones, and within each group to
    the higher confidence, then the larger area, so a polygon cut by its tile
    edge never cuts that edge into a field that another tile saw whole
    (:func:`resolve_overlaps`; ``resolve_overlaps=False`` keeps them); (4)
    tile-cut polygons that meet along an interior tile edge are joined when
    they pass DelAnyFlow's seam test (:func:`merge_tile_seams`, a port of
    upstream ``merge_tile_seams``, commit 488e537, with its thresholds;
    ``merge_tile_seams=False`` skips it). Pieces that only meet at a tile
    edge are otherwise never joined, because they do not overlap. This is
    simpler than DelAnyFlow's raster-level region merging (which, for
    example, lets smaller fields carve their area out of larger ones), so
    results are close to, but not identical with, the ``"reference"``
    backend. NMS uses IoU 0.3 by default, the value of the authors' openEO
    UDP (``openeo_udp/udf/delineate_onnx.py``); the upstream ``execute()``
    keeps Ultralytics' default (0.7). The raster is read tile by tile.
    ``config.min_field_area_m2`` is applied to the raw detections, before
    step (1), and again after step (4); upstream applies its
    ``minimum_area_m2`` only after its seam merge, so there a tile-cut piece
    smaller than the minimum can still be joined to its neighbour. Keeping
    the tile-cut detections until the second filter was tried on cached
    SPOT-Pan composites (example 23's Twifo Praso and example 22's Cauvery
    delta with the released weights, example 23's Oro with the published
    Higaturu fine-tuning): 2, 0 and 7 more seam joins, and scores that moved
    by one polygon either way (one more false positive at Twifo Praso, one
    more split field at Oro), so the filter order was kept.
``"reference"``
    Runs the upstream DelAnyFlow pipeline (``methods.main.inference.execute``)
    in a subprocess, from a Delineate-Anything checkout given by
    ``engine_params["da_repo"]`` or the ``AGRIBOUND_DA_REPO`` environment
    variable. Requires the GDAL Python bindings (``osgeo``) and a checkout that
    contains the uint8-mask fix (upstream commit 34eddf7 or later).
``"ftw"``
    ``ftw_tools.inference.inference.run_instance_segmentation``. FTW's wrapper
    divides the first three bands by 3000 (Sentinel-2 L2A units), clips to
    [0, 1] and resizes bilinearly, so this backend accepts only
    ``reflectance_x10000`` composites. ``large_v2`` needs an ftw-tools build
    whose ``MODEL_REGISTRY`` contains ``DelineateAnythingV2`` (ftw-baselines
    main at fa86d4a or later; not in ftw-tools 2.0.0b5).

Engine parameters
-----------------
All optional. A Delineate-Anything parameter that the selected backend cannot
honour raises :class:`ValueError`, as do the names ``confidence`` and
``minimal_confidence`` (the confidence is ``conf_threshold`` for every
backend); parameters not listed here are left to other pipeline stages.

- All backends: ``backend``; ``da_model``/``model_size`` (``large_v2``);
  ``conf_threshold`` (per model; the ``ftw`` backend uses 0.15 for v2 and
  ftw-tools' 0.05 for v1); ``batch_size`` (tiles per forward pass, 4).
- ``native`` and ``reference``: ``checkpoint_path`` (fine-tuned YOLO weights;
  set by the pipeline after fine-tuning); ``super_resolution`` (1, 2 or 4;
  default automatic); ``tile_step`` (fraction of the tile, 0.5); ``half``
  (FP16 on GPU/MPS, True).
- ``native`` and ``ftw``: ``iou_threshold`` (NMS IoU, 0.3);
  ``max_detections`` (per tile, 300). Both predict without Ultralytics'
  wall-clock limit on non-maximum suppression (:func:`no_nms_time_limit`),
  as does fine-tuning's validation.
- ``native``: ``dedup_iou`` (0.3), ``dedup_containment`` (0.8),
  ``merge_tile_pieces`` (True), ``resolve_overlaps`` (True),
  ``merge_tile_seams`` (True when ``resolve_overlaps`` is True and
  ``tile_step`` is 0.5, which it needs; False otherwise, and True with
  either of them changed raises :class:`ValueError`), ``processing_gsd_m``
  (None) and ``stretch_percentiles`` (``(1, 99)``); see below.
- ``reference``: ``da_repo``; ``min_hole_area_m2`` (holes smaller than this
  are filled, 2500 m², the upstream ``conf_sample.yaml`` value).
- ``ftw``: ``patch_size`` (256; a multiple of 32 smaller than the raster's
  smaller side), ``resize_factor`` (2), ``padding`` (FTW
  default), ``close_interiors`` (True), ``simplify`` (FTW simplification
  tolerance, applied in EPSG:6933 coordinates, i.e. metres that are exact
  only near 30° latitude; 0), ``max_size`` (m², None),
  ``overlap_iou_threshold`` (0.3), ``overlap_contain_threshold`` (0.8),
  ``value_scale`` (for ``local`` rasters).

Two ``native`` options change the input the model sees, for example for
fields larger than its tiles. Neither is set by default, and neither changes
the results, cache keys or configuration hash of runs that do not set it:

- ``processing_gsd_m`` (metres): the backend reads the raster at this
  coarser pixel size. The raster is resampled once by average resampling
  (each new pixel is the area-weighted mean of the valid raster pixels it
  covers) and cached in the working directory (:func:`processing_raster`);
  everything then runs on that copy as on any raster of that pixel size,
  including the 4 m super-resolution rule, so at 3 m the 512 px tiles cover
  1536 m instead of the 768 m of 1.5 m imagery. The polygons are in the
  raster's CRS. Values outside the 0.25-10 m training range, or finer than
  the raster's pixels (by more than 1 %), raise :class:`ValueError`; a value
  within 1 % of the raster's pixel size reads the raster as it is.
  ``engine_meta`` records ``processing_gsd_m`` and ``processing`` (pixel
  size, factor, size and path of the copy; None when the raster is read as
  it is).
- ``stretch_percentiles`` ``(low, high)``, ``0 <= low < high <= 100``: the
  percentiles of the scene stretch (upstream's 1 and 99 by default). uint8
  rasters are not stretched, so it has no effect on them (WARNING).

The fine-tuning chips of this engine are cut from the same raster with the
same stretch (:mod:`agribound.engines.finetune._data`), and the training
metadata of a fine-tuned checkpoint records the input they gave the model:
pixel size, super-resolution factor and stretch percentiles (none for a
uint8 raster). A run that gives the checkpoint it loads another input (a
pixel size more than 1 % off, e.g. a 1.5 m checkpoint on a 3 m raster,
another factor or another stretch) logs a WARNING and records both in
``engine_meta["checkpoint_input_mismatch"]``.

``config.min_field_area_m2`` is applied as an absolute area in m², computed
in the equal-area EPSG:6933, by every backend (the reference pipeline's
``automatic_area_scale`` is disabled; upstream and ftw-tools compute areas in
EPSG:6933 too). Holes: the ``native`` backend keeps holes of any size, the
``reference`` backend fills holes smaller than ``min_hole_area_m2`` and the
``ftw`` backend fills all holes while ``close_interiors`` is True.
The returned frame carries ``gdf.attrs["engine_meta"]`` (backend, model key,
weights repository, revision and SHA-256, thresholds, super-resolution
factor, tile size, device, pixel size, ...); the ``native`` backend also
returns a ``confidence`` column. The published models were trained on
0.25-10 m imagery: for rasters outside that range (e.g. 30 m Landsat/HLS) a
WARNING is logged and ``engine_meta["gsd_outside_training_range"]`` is True
(:func:`gsd_outside_training_range`; with ``processing_gsd_m``, the pixel size
the model sees is checked).

References
----------
Lavreniuk, M., et al. (2025). Delineate Anything: Resolution-Agnostic Field
Boundary Delineation on Satellite Imagery. European Conference on Artificial
Intelligence (ECAI 2025). arXiv:2504.02534.

Lavreniuk, M., et al. (2025). Delineate Anything Flow: Fast, Country-Level
Field Boundary Detection from Any Source. arXiv:2511.13417.

Lavreniuk, M., et al. (2026). Delineate Anything v2: A Global Foundation Model
for Field Delineation. European Conference on Computer Vision Workshops (ECCVW
2026), GAIA workshop. arXiv:2607.19069.

Model code and weights are AGPL-3.0; Ultralytics is AGPL-3.0.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig, effective_bands
from agribound.engines.base import DelineationEngine, get_canonical_band_indices
from agribound.registry import ENGINE_REGISTRY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

DA_HF_REPO = "MykolaL/DelineateAnything"
"""Hugging Face repository holding the Delineate-Anything weights."""


@dataclass(frozen=True)
class DAModel:
    """A pinned Delineate-Anything checkpoint.

    Attributes
    ----------
    key : str
        Model key used by the upstream configuration (``config["model"]``).
    filename : str
        File name in :data:`DA_HF_REPO`.
    revision : str
        Hugging Face commit the file is downloaded from.
    sha256 : str
        Expected SHA-256 of the file.
    size_bytes : int
        Expected file size.
    default_conf : float
        Default minimum confidence (upstream sample configuration).
    ftw_name : str
        Name of the model in the ftw-tools ``MODEL_REGISTRY``.
    architecture : str
        Ultralytics architecture.
    training_data : str
        Training dataset.
    """

    key: str
    filename: str
    revision: str
    sha256: str
    size_bytes: int
    default_conf: float
    ftw_name: str
    architecture: str
    training_data: str


DA_MODELS: dict[str, DAModel] = {
    "large_v2": DAModel(
        key="large_v2",
        filename="DelineateAnythingv2.pt",
        revision="369d0b4c44cf9bec2bd3a27bc81810cadd2c963e",
        sha256="46700b8a279b07922953a11adaeb5e658d9a2384b6334c8e0a3090886218915a",
        size_bytes=124_747_297,
        default_conf=0.15,
        ftw_name="DelineateAnythingV2",
        architecture="YOLO11x-seg",
        training_data="FBIS-73M",
    ),
    "large": DAModel(
        key="large",
        filename="DelineateAnything.pt",
        revision="029e9a94c6abc51c67cebdc9b9a9b6c1ac2b1187",
        sha256="e3dcda35780083aeaefe9425b73b15a30561cdc12c43277041d04f9e88ede029",
        size_bytes=124_746_842,
        default_conf=0.005,
        ftw_name="DelineateAnything",
        architecture="YOLO11x-seg",
        training_data="FBIS-22M",
    ),
    "small": DAModel(
        key="small",
        filename="DelineateAnything-S.pt",
        revision="029e9a94c6abc51c67cebdc9b9a9b6c1ac2b1187",
        sha256="5463cdfb73690fc506035e4f7dce26a4c06af6ef4d207570513110c4b879d643",
        size_bytes=17_635_629,
        default_conf=0.005,
        ftw_name="DelineateAnything-S",
        architecture="YOLO11n-seg",
        training_data="FBIS-22M",
    ),
}
"""Pinned Delineate-Anything checkpoints (see the module docstring)."""

DEFAULT_DA_MODEL = "large_v2"
"""Model used when no ``da_model``/``model_size`` engine parameter is given."""

_DA_ALIASES: dict[str, str] = {
    "large_v2": "large_v2",
    "delineateanythingv2": "large_v2",
    "large": "large",
    "delineateanything": "large",
    "small": "small",
    "delineateanything-s": "small",
}
_MODEL_SIZE_KEYS = {"large": "large", "small": "small"}

VALID_BACKENDS = ("native", "reference", "ftw")
"""Values accepted by ``engine_params["backend"]``."""

#: Side of the square model input in pixels (all published DA models use 512).
MODEL_INPUT_PX = 512

#: Default confidence of the ``ftw`` backend for v1 models (ftw-tools default).
_FTW_V1_DEFAULT_CONF = 0.05

#: Detection overlaps the native backend treats as duplicates. The values are
#: ftw-tools' ``overlap_iou_threshold``/``overlap_contain_threshold`` defaults.
DEFAULT_DEDUP_IOU = 0.3
DEFAULT_DEDUP_CONTAINMENT = 0.8

#: Bump when the native algorithm changes, to invalidate cached native results
#: (3: tile-piece merging, complete-first precedence only over smaller-or-equal
#: tile-cut duplicates, EPSG:6933 areas, stretch bounds never from overviews;
#: 4: tile-cut within a 4 px edge strip of the kept polygon, complete polygons
#: first in overlap resolution, tile-seam merge).
_NATIVE_IMPL_VERSION = "4"

#: Width, in model-input pixels, of the strip along each interior tile edge in
#: which a detection counts as tile-cut: DelAnyFlow's ``merging_edge_width``.
TILE_EDGE_STRIP_PX = 4

#: Thresholds of :func:`merge_tile_seams`, in model-input pixels: the defaults
#: of upstream DelAnyFlow's ``merge_tile_seams`` (Lavreniuk/Delineate-Anything
#: 488e537, ``conf_sample.yaml``).
TILE_SEAM_PARAMS: dict[str, float] = {
    "seam_min_contact_px": 8,
    "seam_min_coverage": 0.3,
    "seam_min_evidence": 0.5,
    "seam_split_window_px": 1,
    "seam_max_colour_diff": 0.2,
    "seam_end_tolerance_px": 8,
    "seam_end_tolerance_rel": 0.25,
    "seam_max_strip_depth_px": 16,
}

#: Upstream ``SEAM_STRIP``: the colour test compares the ``SEAM_STRIP // 2``
#: pixels on each side of a tile edge.
_SEAM_STRIP_PX = 8

#: Upstream per-tile composition (``pixel_area_threshold``, ``compose_merge_iou``
#: of ``conf_sample.yaml``), used for the view of the tile centred on a seam.
_COMPOSE_MIN_AREA_PX = 512
_COMPOSE_MERGE_IOU = 0.8

#: Distance, in model-input pixels, within which :func:`merge_tile_seams` moves a
#: vertex onto the pixel corner of the model-input grid before joining pieces.
_GRID_SNAP_PX = 1e-6

#: Upstream ``conf_sample.yaml`` hole threshold of the reference pipeline (m²).
DEFAULT_MIN_HOLE_AREA_M2 = 2500.0

#: Percentiles of the scene stretch (upstream ``DataAnalyser``); the native
#: backend's ``engine_params["stretch_percentiles"]`` replaces them.
DEFAULT_STRETCH_PERCENTILES: tuple[float, float] = (1.0, 99.0)

#: A ``processing_gsd_m`` within this fraction of the raster's pixel size reads the
#: raster as it is; a finer one is refused.
PROCESSING_GSD_TOLERANCE = 0.01

#: Version of the resampled raster of ``processing_gsd_m``: part of the cache key of the
#: copy, and of the native results and fine-tuning chips made from a copy.
_PROCESSING_RASTER_VERSION = "1"

#: Values (pixels times bands) of the raster read at a time while it is resampled.
_PROCESSING_CHUNK_VALUES = 1 << 22

#: String that marks the uint8-mask fix in upstream ``methods/main/inference.py``
#: (commit 34eddf7). Checkouts without it silently dilate masks.
_UINT8_FIX_MARKER = "masks.data.float()"

_COMMON_KEYS = frozenset({"backend", "da_model", "model_size", "conf_threshold", "batch_size"})
_BACKEND_KEYS: dict[str, frozenset[str]] = {
    "native": _COMMON_KEYS
    | {
        "checkpoint_path",
        "iou_threshold",
        "max_detections",
        "super_resolution",
        "tile_step",
        "half",
        "dedup_iou",
        "dedup_containment",
        "merge_tile_pieces",
        "resolve_overlaps",
        "merge_tile_seams",
        "processing_gsd_m",
        "stretch_percentiles",
    },
    "reference": _COMMON_KEYS
    | {"checkpoint_path", "super_resolution", "tile_step", "half", "da_repo", "min_hole_area_m2"},
    "ftw": _COMMON_KEYS
    | {
        "iou_threshold",
        "max_detections",
        "patch_size",
        "resize_factor",
        "padding",
        "close_interiors",
        "simplify",
        "max_size",
        "overlap_iou_threshold",
        "overlap_contain_threshold",
        "value_scale",
    },
}
_ALL_DA_KEYS = frozenset().union(*_BACKEND_KEYS.values())

#: Names used for the detection confidence by agribound 0.x examples and by the
#: upstream configuration. They are rejected (not silently ignored) in favour of
#: the single ``conf_threshold`` parameter.
_RENAMED_KEYS: dict[str, str] = {
    "confidence": "conf_threshold",
    "minimal_confidence": "conf_threshold",
}


def resolve_da_model_key(engine_params: dict[str, Any] | None) -> str:
    """Return the :data:`DA_MODELS` key selected by *engine_params*.

    Parameters
    ----------
    engine_params : dict or None
        Engine parameters. ``da_model`` accepts a key (``"large_v2"``,
        ``"large"``, ``"small"``) or an alias (``"DelineateAnythingV2"``,
        ``"DelineateAnything"``, ``"DelineateAnything-S"``; case-insensitive).
        The legacy ``model_size`` accepts ``"large"`` or ``"small"`` (the v1
        models).

    Returns
    -------
    str
        Model key; :data:`DEFAULT_DA_MODEL` when neither parameter is given.

    Raises
    ------
    ValueError
        For unknown values, or when ``da_model`` and ``model_size`` disagree.
    """
    params = engine_params or {}
    da_model = params.get("da_model")
    model_size = params.get("model_size")
    key_from_model = None
    if da_model is not None:
        key_from_model = _DA_ALIASES.get(str(da_model).strip().lower())
        if key_from_model is None:
            raise ValueError(
                f"Unknown da_model {da_model!r}. Choose one of {sorted(DA_MODELS)} or an alias "
                "('DelineateAnythingV2', 'DelineateAnything', 'DelineateAnything-S')."
            )
    key_from_size = None
    if model_size is not None:
        key_from_size = _MODEL_SIZE_KEYS.get(str(model_size).strip().lower())
        if key_from_size is None:
            raise ValueError(
                f"Unknown model_size {model_size!r}; use 'large' or 'small' (v1 models), or "
                f"da_model={sorted(DA_MODELS)}."
            )
    if key_from_model and key_from_size and key_from_model != key_from_size:
        raise ValueError(
            f"da_model={da_model!r} and model_size={model_size!r} select different models "
            f"({key_from_model!r} vs {key_from_size!r}); pass only da_model."
        )
    return key_from_model or key_from_size or DEFAULT_DA_MODEL


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

_SHA256_MEMO: dict[tuple[str, int, int], str] = {}


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file (memoised per path, size and mtime)."""
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    memo_key = (str(resolved), stat.st_size, stat.st_mtime_ns)
    cached = _SHA256_MEMO.get(memo_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with open(resolved, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _SHA256_MEMO[memo_key] = value
    return value


def download_da_weights(model_key: str, verify: bool = True) -> str:
    """Download (or reuse from the Hugging Face cache) pinned DA weights.

    Parameters
    ----------
    model_key : str
        Key of :data:`DA_MODELS`.
    verify : bool
        Check the file's SHA-256 against :data:`DA_MODELS` (default True).

    Returns
    -------
    str
        Local path of the weights file.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is not installed.
    RuntimeError
        If the downloaded file does not have the expected SHA-256.
    """
    spec = DA_MODELS[model_key]
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface-hub is required to download Delineate-Anything weights. "
            "Install with: pip install 'agribound[delineate-anything]'"
        ) from None
    path = hf_hub_download(repo_id=DA_HF_REPO, filename=spec.filename, revision=spec.revision)
    if verify:
        actual = file_sha256(path)
        if actual != spec.sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {DA_HF_REPO}/{spec.filename}@{spec.revision}: expected "
                f"{spec.sha256}, got {actual} ({path}). Delete the cached file and retry."
            )
    return str(path)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class DAOptions:
    """Validated engine parameters for one Delineate-Anything run.

    Build with :meth:`from_engine_params`; see the module docstring for the
    meaning and defaults of each field.
    """

    backend: str
    model_key: str
    checkpoint_path: str | None
    conf_threshold: float
    conf_source: str
    iou_threshold: float
    max_detections: int
    super_resolution: int | None
    tile_step: float
    batch_size: int
    half: bool
    dedup_iou: float
    dedup_containment: float
    merge_tile_pieces: bool
    resolve_overlaps: bool
    merge_tile_seams: bool
    da_repo: str | None
    min_hole_area_m2: float = DEFAULT_MIN_HOLE_AREA_M2
    processing_gsd_m: float | None = None
    stretch_percentiles: tuple[float, float] = DEFAULT_STRETCH_PERCENTILES
    ftw: dict[str, Any] = field(default_factory=dict)

    @property
    def model(self) -> DAModel:
        """The pinned model spec of :attr:`model_key`."""
        return DA_MODELS[self.model_key]

    @classmethod
    def from_engine_params(cls, engine_params: dict[str, Any] | None) -> DAOptions:
        """Parse and validate *engine_params*.

        Raises
        ------
        ValueError
            For an unknown backend or model, invalid values, or parameters
            the selected backend cannot honour.
        """
        params = dict(engine_params or {})
        renamed = sorted(k for k in params if k in _RENAMED_KEYS)
        if renamed:
            raise ValueError(
                f"engine_params {renamed} are not Delineate-Anything parameters; the detection "
                "confidence is set with engine_params['conf_threshold'] (all backends)."
            )
        backend = str(params.get("backend", "native")).strip().lower()
        if backend not in VALID_BACKENDS:
            raise ValueError(
                f"Unknown Delineate-Anything backend {params.get('backend')!r}. "
                f"Choose from {VALID_BACKENDS}."
            )
        unsupported = sorted(
            k for k in params if k in _ALL_DA_KEYS and k not in _BACKEND_KEYS[backend]
        )
        # checkpoint_path is set by the pipeline after fine-tuning; the ftw backend cannot use it.
        if unsupported:
            hint = (
                " FTW's run_instance_segmentation cannot load custom weights; use "
                "backend='native' or 'reference' for fine-tuned checkpoints."
                if "checkpoint_path" in unsupported
                else ""
            )
            raise ValueError(
                f"engine_params {unsupported} are not supported by the {backend!r} "
                f"Delineate-Anything backend (supported: {sorted(_BACKEND_KEYS[backend])}).{hint}"
            )

        model_key = resolve_da_model_key(params)
        checkpoint = params.get("checkpoint_path")
        if checkpoint is not None:
            checkpoint = str(checkpoint)

        if params.get("conf_threshold") is not None:
            conf = _float_in("conf_threshold", params["conf_threshold"], 0.0, 1.0)
            conf_source = "engine_params"
        elif backend == "ftw" and model_key != "large_v2":
            conf, conf_source = _FTW_V1_DEFAULT_CONF, "ftw-tools default"
        else:
            conf = DA_MODELS[model_key].default_conf
            conf_source = f"upstream default for {model_key}"

        sr = params.get("super_resolution")
        if sr is not None:
            if isinstance(sr, bool) or int(sr) != sr or int(sr) not in (1, 2, 4):
                raise ValueError(f"super_resolution must be None, 1, 2 or 4, got {sr!r}")
            sr = int(sr)

        tile_step = _float_in("tile_step", params.get("tile_step", 0.5), 0.0, 1.0, low_open=True)
        batch_size = _positive_int("batch_size", params.get("batch_size", 4))
        max_det = _positive_int("max_detections", params.get("max_detections", 300))
        iou = _float_in("iou_threshold", params.get("iou_threshold", 0.3), 0.0, 1.0)
        dedup_iou = _float_in("dedup_iou", params.get("dedup_iou", DEFAULT_DEDUP_IOU), 0.0, 1.0)
        dedup_contain = _float_in(
            "dedup_containment",
            params.get("dedup_containment", DEFAULT_DEDUP_CONTAINMENT),
            0.0,
            1.0,
        )
        half = params.get("half", True)
        if not isinstance(half, bool):
            raise ValueError(f"half must be True or False, got {half!r}")
        resolve = params.get("resolve_overlaps", True)
        if not isinstance(resolve, bool):
            raise ValueError(f"resolve_overlaps must be True or False, got {resolve!r}")
        merge = params.get("merge_tile_pieces", True)
        if not isinstance(merge, bool):
            raise ValueError(f"merge_tile_pieces must be True or False, got {merge!r}")
        seams = params.get("merge_tile_seams")
        if seams is None:
            seams = resolve and tile_step == 0.5
        elif not isinstance(seams, bool):
            raise ValueError(f"merge_tile_seams must be True or False, got {seams!r}")
        elif seams and not resolve:
            raise ValueError(
                "merge_tile_seams=True needs resolve_overlaps=True: the seam test joins "
                "non-overlapping polygons."
            )
        elif seams and tile_step != 0.5:
            raise ValueError(
                f"merge_tile_seams=True needs tile_step=0.5 (got {tile_step!r}): the seam test "
                "checks each tile edge against the tile centred on it."
            )
        hole = params.get("min_hole_area_m2", DEFAULT_MIN_HOLE_AREA_M2)
        if isinstance(hole, bool) or not isinstance(hole, int | float) or hole < 0:
            raise ValueError(f"min_hole_area_m2 must be a number >= 0, got {hole!r}")
        processing_gsd = _processing_gsd(params.get("processing_gsd_m"))
        percentiles = _stretch_percentiles(params.get("stretch_percentiles"))

        ftw_opts = {}
        if backend == "ftw":
            patch_size = _positive_int("patch_size", params.get("patch_size", 256))
            if patch_size % 32:
                # ftw-tools setup_inference asserts this; fail before the input is written.
                raise ValueError(f"patch_size must be a multiple of 32, got {patch_size}")
            ftw_opts = {
                "patch_size": patch_size,
                "resize_factor": _positive_int("resize_factor", params.get("resize_factor", 2)),
                "padding": params.get("padding"),
                "close_interiors": bool(params.get("close_interiors", True)),
                "simplify": int(params.get("simplify", 0)),
                "max_size": params.get("max_size"),
                "overlap_iou_threshold": _float_in(
                    "overlap_iou_threshold", params.get("overlap_iou_threshold", 0.3), 0.0, 1.0
                ),
                "overlap_contain_threshold": _float_in(
                    "overlap_contain_threshold",
                    params.get("overlap_contain_threshold", 0.8),
                    0.0,
                    1.0,
                ),
                "value_scale": params.get("value_scale"),
            }

        return cls(
            backend=backend,
            model_key=model_key,
            checkpoint_path=checkpoint,
            conf_threshold=conf,
            conf_source=conf_source,
            iou_threshold=iou,
            max_detections=max_det,
            super_resolution=sr,
            tile_step=tile_step,
            batch_size=batch_size,
            half=half,
            dedup_iou=dedup_iou,
            dedup_containment=dedup_contain,
            merge_tile_pieces=merge,
            resolve_overlaps=resolve,
            merge_tile_seams=seams,
            da_repo=params.get("da_repo"),
            min_hole_area_m2=float(hole),
            processing_gsd_m=processing_gsd,
            stretch_percentiles=percentiles,
            ftw=ftw_opts,
        )


def _is_number(value: Any) -> bool:
    """True for a real number (int, float or NumPy scalar), False for bool and other types."""
    import numbers

    return isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)


def _processing_gsd(value: Any) -> float | None:
    """Validate ``engine_params["processing_gsd_m"]`` (None or metres in the training range)."""
    if value is None:
        return None
    low, high = TRAINING_GSD_RANGE_M
    if not _is_number(value) or not low <= float(value) <= high:
        raise ValueError(
            f"processing_gsd_m must be a pixel size in metres within the {low:g}-{high:g} m "
            f"range of Delineate-Anything's training imagery, got {value!r}"
        )
    return float(value)


def _stretch_percentiles(value: Any) -> tuple[float, float]:
    """Validate ``engine_params["stretch_percentiles"]`` (``0 <= low < high <= 100``)."""
    if value is None:
        return DEFAULT_STRETCH_PERCENTILES
    pair = tuple(value) if isinstance(value, list | tuple) else ()
    if (
        len(pair) != 2
        or not all(_is_number(v) for v in pair)
        or not 0.0 <= float(pair[0]) < float(pair[1]) <= 100.0
    ):
        raise ValueError(
            "stretch_percentiles must be two percentiles (low, high) with "
            f"0 <= low < high <= 100, got {value!r}"
        )
    return float(pair[0]), float(pair[1])


def _float_in(name: str, value: Any, low: float, high: float, *, low_open: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not (low < number if low_open else low <= number) or number > high:
        bracket = "(" if low_open else "["
        raise ValueError(f"{name} must be in {bracket}{low}, {high}], got {value!r}")
    return number


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float) or int(value) != value:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if int(value) < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return int(value)


# ---------------------------------------------------------------------------
# Pure helpers (tiling, GSD, morphology, stretch, de-duplication)
# ---------------------------------------------------------------------------


def pixel_size_m(crs: Any, transform: Any, height: int, width: int) -> float:
    """Return the mean pixel size in metres, as Delineate-Anything computes it.

    Mirrors upstream ``DataAnalyser.evaluate_pixel_size`` /
    ``get_pixel_size_meters``: for projected CRSs the pixel size times the
    linear unit; for geographic CRSs the pixel size in degrees times
    ``6371000 * pi / 180`` metres per degree (north-south) and that value
    times ``cos(latitude of the raster centre)`` (east-west). The result is
    the mean of the two sides.

    Parameters
    ----------
    crs : rasterio CRS, pyproj CRS or str
        Raster CRS.
    transform : affine.Affine
        Raster transform.
    height, width : int
        Raster size in pixels.

    Returns
    -------
    float
        Mean pixel side in metres.
    """
    import math

    import pyproj

    if crs is None:
        raise ValueError("The raster has no CRS; cannot determine its pixel size in metres")
    crs_obj = pyproj.CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
    width_units = abs(transform.a)
    height_units = abs(transform.e)
    if crs_obj.is_projected:
        factor = crs_obj.axis_info[0].unit_conversion_factor
        width_m = width_units * factor
        height_m = height_units * factor
    else:
        center_y = transform.f + (width / 2) * transform.d + (height / 2) * transform.e
        radians_per_unit = crs_obj.axis_info[0].unit_conversion_factor
        lat_m = 6371000.0 * radians_per_unit
        lon_m = lat_m * math.cos(center_y * radians_per_unit)
        width_m = width_units * lon_m
        height_m = height_units * lat_m
    return 0.5 * (width_m + height_m)


def select_super_resolution(gsd_m: float, override: int | None = None) -> int:
    """Return the upsampling factor applied to tiles before inference.

    Upstream rule (``DataAnalyser.isCompatible``): 1 when the GSD is below
    4 m, else 2. *override* (1, 2 or 4) replaces the rule.
    """
    if override is not None:
        if override not in (1, 2, 4):
            raise ValueError(f"super_resolution must be 1, 2 or 4, got {override!r}")
        return int(override)
    return 1 if gsd_m < 4 else 2


#: Ground sampling distances of the training imagery (FBIS-22M and FBIS-73M: 0.25-10 m).
TRAINING_GSD_RANGE_M = (0.25, 10.0)


def gsd_outside_training_range(gsd_m: float, tolerance: float = 0.05) -> bool:
    """Return True if *gsd_m* is more than *tolerance* (relative) outside 0.25-10 m.

    The tolerance absorbs resampling noise, e.g. Sentinel-2 reprojected to
    9.99 or 10.02 m.
    """
    low, high = TRAINING_GSD_RANGE_M
    return gsd_m < low * (1 - tolerance) or gsd_m > high * (1 + tolerance)


def _record_training_gsd(gsd_m: float, meta: dict[str, Any]) -> None:
    """Set ``meta["gsd_outside_training_range"]`` and log a WARNING when it is True."""
    outside = gsd_outside_training_range(gsd_m)
    meta["gsd_outside_training_range"] = outside
    if outside:
        logger.warning(
            "Delineate-Anything was trained on 0.25-10 m imagery; this raster's pixel size "
            "(%.2f m) is outside that range, so accuracy is not established.",
            gsd_m,
        )


@contextlib.contextmanager
def no_nms_time_limit():
    """Lift Ultralytics' wall-clock limit on non-maximum suppression.

    ``ultralytics.utils.nms.non_max_suppression`` (8.4) stops after
    ``2 s + 0.05 s`` per image of a batch and returns no detections for the
    batch's remaining images, logging "NMS time limit ... exceeded". On a slow
    or busy device a prediction would then lose tiles, and a fine-tuning
    validation would underrate the epoch on which ``best.pt`` is chosen.
    Inside this context the limit is infinite (``max_time_img=inf``); the
    detections are otherwise unchanged. Ultralytics' validator and predictor
    call the function through the ``nms`` module, which is what is patched.
    """
    try:
        from ultralytics.utils import nms
    except ImportError:  # Ultralytics missing: nothing to patch
        yield
        return
    original = nms.non_max_suppression
    if getattr(original, "_agribound_no_time_limit", False):  # nested use
        yield
        return

    @functools.wraps(original)
    def unlimited(*args: Any, **kwargs: Any) -> Any:
        kwargs["max_time_img"] = float("inf")
        return original(*args, **kwargs)

    unlimited._agribound_no_time_limit = True
    nms.non_max_suppression = unlimited
    try:
        yield
    finally:
        nms.non_max_suppression = original


def tile_origins(size: int, tile: int, step: int) -> list[int]:
    """Return tile start offsets along one axis, as the upstream planner does.

    The first tile starts ``tile - step`` pixels (rounded down to a multiple
    of *step*) before the raster origin, and tiles continue while their start
    is inside the raster (upstream ``ExecutionPlanner.get_plan``). With the
    default half-tile step every pixel therefore falls in the central half of
    some tile.
    """
    if tile < 1 or step < 1:
        raise ValueError("tile and step must be >= 1")
    start = ((0 - (tile - step)) // step) * step
    return list(range(start, size, step))


def refine_masks(masks: Any) -> Any:
    """Apply the upstream 3 x 3 mask morphology (erode, dilate, dilate, erode).

    The masks are cast to float first: Ultralytics >= 8.3.217 returns uint8
    masks, and ``-max_pool2d(-x)`` on uint8 wraps around, which turns the
    intended erosion into a dilation.

    Parameters
    ----------
    masks : torch.Tensor
        ``(N, H, W)`` masks (any dtype).

    Returns
    -------
    torch.Tensor
        float32 masks of the same shape.
    """
    import torch
    from torch.nn.functional import max_pool2d

    with torch.no_grad():
        m = masks.float()
        m = -max_pool2d(-m, kernel_size=3, stride=1, padding=1)
        m = max_pool2d(m, kernel_size=3, stride=1, padding=1)
        m = max_pool2d(m, kernel_size=3, stride=1, padding=1)
        m = -max_pool2d(-m, kernel_size=3, stride=1, padding=1)
    return m


def stretch_with_bounds(
    data: np.ndarray, lows: list[float], highs: list[float], valid: np.ndarray | None = None
) -> np.ndarray:
    """Map ``(bands, H, W)`` values to uint8 with fixed per-band bounds.

    Uses ``clip(255 * (v - lo) / (hi - lo), 0, 255)`` truncated to uint8 (the
    formula of upstream ``DataLoaderCached`` and
    :func:`agribound.io.raster.percentile_stretch_uint8`). Pixels outside
    *valid* and non-finite results are set to 0.
    """
    out = np.zeros(data.shape, dtype=np.uint8)
    for i, (lo, hi) in enumerate(zip(lows, highs, strict=True)):
        span = hi - lo if hi > lo else 1e-12
        with np.errstate(invalid="ignore"):
            band = np.clip(255.0 * ((data[i].astype(np.float64) - lo) / span), 0, 255)
        band = np.where(np.isfinite(band), band, 0)
        if valid is not None:
            band = np.where(valid, band, 0)
        out[i] = band.astype(np.uint8)
    return out


def _detection_arrays(
    gdf: gpd.GeoDataFrame, score_column: str, edge_column: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Valid geometries, areas, scores (0 if absent) and tile-cut flags (False if absent)."""
    import shapely

    n = len(gdf)
    geoms = shapely.make_valid(np.asarray(gdf.geometry.values, dtype=object))
    areas = shapely.area(geoms)
    scores = gdf[score_column].to_numpy(dtype=float) if score_column in gdf else np.zeros(n)
    edges = (
        gdf[edge_column].to_numpy(dtype=bool)
        if edge_column is not None and edge_column in gdf
        else np.zeros(n, dtype=bool)
    )
    return geoms, areas, scores, edges


def _duplicate_pairs(
    geoms: np.ndarray, areas: np.ndarray, iou_threshold: float, containment_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs ``(i, j)``, ``i < j``, with IoU >= *iou_threshold* or containment >= threshold.

    Containment is the intersection area divided by the smaller area.
    """
    import shapely

    tree = shapely.STRtree(geoms)
    left, right = tree.query(geoms, predicate="intersects")
    pair = left < right
    left, right = left[pair], right[pair]
    if not left.size:
        return left, right
    inter = shapely.area(shapely.intersection(geoms[left], geoms[right]))
    union = areas[left] + areas[right] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    smaller = np.minimum(areas[left], areas[right])
    contain = np.divide(inter, smaller, out=np.zeros_like(inter), where=smaller > 0)
    dup = (iou >= iou_threshold) | (contain >= containment_threshold)
    return left[dup], right[dup]


def _complete_view_pairs(
    left: np.ndarray, right: np.ndarray, areas: np.ndarray, edges: np.ndarray
) -> list[tuple[int, int]]:
    """``(complete, cut)`` duplicate pairs in which the complete detection is at least as large.

    A tile-cut detection is a truncated view of its field; a complete
    (not tile-cut) duplicate that is at least as large is a better view of
    the same field.
    """
    out = []
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        for whole, cut in ((a, b), (b, a)):
            if not edges[whole] and edges[cut] and areas[whole] >= areas[cut]:
                out.append((whole, cut))
    return out


def _visit_order(
    areas: np.ndarray,
    scores: np.ndarray,
    edges: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Greedy visiting order of the detections (see :func:`deduplicate_detections`).

    Higher score first, then larger area, except that a complete detection
    is visited before every tile-cut duplicate that is not larger than it
    (:func:`_complete_view_pairs`); among the detections whose complete
    views have been visited, the score/area order applies.
    """
    import heapq

    n = len(areas)
    base = np.lexsort((-areas, -scores))
    rank = np.empty(n, dtype=np.int64)
    rank[base] = np.arange(n)
    followers: dict[int, list[int]] = {}
    waiting = np.zeros(n, dtype=np.int64)
    for whole, cut in _complete_view_pairs(left, right, areas, edges):
        followers.setdefault(whole, []).append(cut)
        waiting[cut] += 1
    # Constraints only point from complete to tile-cut detections, so they
    # cannot form a cycle and every detection is visited exactly once.
    heap = [(int(rank[i]), i) for i in range(n) if waiting[i] == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        _, idx = heapq.heappop(heap)
        order.append(idx)
        for other in followers.get(idx, ()):
            waiting[other] -= 1
            if waiting[other] == 0:
                heapq.heappush(heap, (int(rank[other]), other))
    return np.asarray(order, dtype=np.int64)


def merge_tile_pieces(
    gdf: gpd.GeoDataFrame,
    iou_threshold: float = DEFAULT_DEDUP_IOU,
    containment_threshold: float = DEFAULT_DEDUP_CONTAINMENT,
    *,
    score_column: str = "confidence",
    edge_column: str = "_tile_edge",
) -> gpd.GeoDataFrame:
    """Merge the tile-cut pieces of a field seen in overlapping tiles into their union.

    With the default half-tile step, a field wider than half a tile can be
    cut by a tile edge in every tile that sees it, and a field wider than a
    tile always is, so no single detection covers it. Tile-cut
    detections (*edge_column* True) that duplicate one another (IoU >=
    *iou_threshold* or intersection >= *containment_threshold* of the
    smaller polygon) are grouped by connected components, and every group
    of two or more is replaced by the largest polygon of the union of its
    members, with the members' highest *score_column* and *edge_column*
    True. The union stays tile-cut because no tile saw it whole: where no
    member reaches past a tile edge, the union still ends at that edge. It
    therefore gives way to complete detections in :func:`resolve_overlaps`,
    and :func:`merge_tile_seams` can join it to a piece across a tile edge. A
    tile-cut detection that duplicates a complete detection at least as large
    as itself is a partial view of a field detected whole in another tile: it
    is not merged and is left for :func:`deduplicate_detections`. Complete
    detections are unchanged.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Detections.
    iou_threshold, containment_threshold : float
        Duplicate criteria.
    score_column : str
        Detection confidence column.
    edge_column : str
        Boolean column marking tile-cut detections.

    Returns
    -------
    geopandas.GeoDataFrame
        Unmerged rows (original order) followed by one row per merged group
        (the other columns are those of the group's highest-scoring member),
        with an integer column ``_n_pieces`` (1 for unmerged rows).
    """
    import pandas as pd
    import shapely

    out = gdf.copy()
    out["_n_pieces"] = 1
    n = len(gdf)
    if n <= 1 or edge_column not in gdf:
        return out
    geoms, areas, scores, edges = _detection_arrays(gdf, score_column, edge_column)
    left, right = _duplicate_pairs(geoms, areas, iou_threshold, containment_threshold)
    mergeable = edges.copy()
    for _, cut in _complete_view_pairs(left, right, areas, edges):
        mergeable[cut] = False

    root = np.arange(n)

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = int(root[i])
        return i

    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        if mergeable[a] and mergeable[b]:
            ra, rb = find(a), find(b)
            if ra != rb:
                root[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in np.flatnonzero(mergeable).tolist():
        groups.setdefault(find(i), []).append(i)
    groups = {k: v for k, v in groups.items() if len(v) > 1}
    if not groups:
        return out

    replaced: list[int] = []
    leaders, shapes, best_scores, sizes = [], [], [], []
    for members in groups.values():
        idx = np.asarray(members)
        union = shapely.make_valid(shapely.union_all(geoms[idx]))
        parts = [
            g for g in getattr(union, "geoms", [union]) if g.geom_type == "Polygon" and g.area > 0
        ]
        if not parts:  # not expected: duplicates overlap with a positive area
            continue
        replaced.extend(members)
        leaders.append(int(idx[np.argmax(scores[idx])]))
        shapes.append(max(parts, key=lambda g: g.area))
        best_scores.append(float(scores[idx].max()))
        sizes.append(len(members))
    if not leaders:
        return out
    merged = out.iloc[leaders].copy()
    merged[gdf.geometry.name] = gpd.GeoSeries(shapes, index=merged.index, crs=gdf.crs)
    if score_column in merged:
        merged[score_column] = best_scores
    merged[edge_column] = True
    merged["_n_pieces"] = sizes
    kept = out.iloc[np.setdiff1d(np.arange(n), replaced)]  # positional: labels may repeat
    result = pd.concat([kept, merged], ignore_index=True)
    return gpd.GeoDataFrame(result, geometry=gdf.geometry.name, crs=gdf.crs)


def deduplicate_detections(
    gdf: gpd.GeoDataFrame,
    iou_threshold: float = DEFAULT_DEDUP_IOU,
    containment_threshold: float = DEFAULT_DEDUP_CONTAINMENT,
    *,
    score_column: str = "confidence",
    edge_column: str | None = "_tile_edge",
) -> gpd.GeoDataFrame:
    """Remove duplicate detections produced by overlapping tiles.

    Two polygons are duplicates when their IoU is at least *iou_threshold*
    or when their intersection covers at least *containment_threshold* of
    the smaller one. Polygons are visited in order and a polygon is dropped
    if it duplicates one that was already kept (greedy non-maximum
    suppression). Order: higher *score_column* first, then larger area,
    except that a complete detection (*edge_column* False) is visited before
    every tile-cut duplicate (*edge_column* True) that is not larger than it:
    a tile-cut detection is a truncated view of its field. A tile-cut
    detection is therefore never suppressed in favour of a smaller complete
    detection nested inside it unless that one has the higher score. The
    kept geometries are unchanged (no union; see :func:`merge_tile_pieces`).

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Detections.
    iou_threshold, containment_threshold : float
        Duplicate criteria (defaults are ftw-tools' overlap thresholds).
    score_column : str
        Detection confidence column.
    edge_column : str or None
        Boolean column marking polygons that touch an interior tile edge.

    Returns
    -------
    geopandas.GeoDataFrame
        Kept rows, in the original order, with all their columns (so the
        tile-cut flag reaches :func:`resolve_overlaps`).
    """
    n = len(gdf)
    if n <= 1:
        return gdf.copy()
    geoms, areas, scores, edges = _detection_arrays(gdf, score_column, edge_column)
    left, right = _duplicate_pairs(geoms, areas, iou_threshold, containment_threshold)
    order = _visit_order(areas, scores, edges, left, right)
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)
    later: dict[int, list[int]] = {}
    for a, b in zip(left.tolist(), right.tolist(), strict=True):
        first, second = (a, b) if position[a] < position[b] else (b, a)
        later.setdefault(first, []).append(second)

    suppressed = np.zeros(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    for idx in order.tolist():
        if suppressed[idx]:
            continue
        keep[idx] = True
        for other in later.get(idx, ()):
            suppressed[other] = True
    return gdf.iloc[np.flatnonzero(keep)].copy()


def resolve_overlaps(
    gdf: gpd.GeoDataFrame,
    *,
    score_column: str = "confidence",
    edge_column: str | None = "_tile_edge",
) -> gpd.GeoDataFrame:
    """Make detections non-overlapping by giving contested area to the one visited first.

    Complete detections (*edge_column* False) are visited before tile-cut
    ones (*edge_column* True), and within each group the higher
    *score_column* first, then the larger area; each polygon loses the parts
    that overlap polygons visited before it and keeps its largest remaining
    part (empty results are dropped). A tile-cut polygon ends with a straight
    line where its tile cut it; visited after the complete ones, it can no
    longer cut that line into a field that another tile saw whole. This
    mimics the per-pixel assignment of DelAnyFlow, whose output fields do not
    overlap.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Detections (usually the output of :func:`deduplicate_detections`).
    score_column : str
        Detection confidence column (0 if absent).
    edge_column : str or None
        Boolean column marking tile-cut detections (all complete if absent).

    Returns
    -------
    geopandas.GeoDataFrame
        Rows in the original order, with modified geometries.
    """
    import shapely

    n = len(gdf)
    if n <= 1:
        return gdf.copy()
    original, areas, scores, edges = _detection_arrays(gdf, score_column, edge_column)
    tree = shapely.STRtree(original)
    placed = np.zeros(n, dtype=bool)
    result = np.empty(n, dtype=object)
    # np.lexsort sorts by the last key first: complete, then score, then area.
    order = np.lexsort((-areas, -scores, edges.astype(np.int8)))
    for idx in order.tolist():
        geom = original[idx]
        neighbours = [j for j in tree.query(geom, predicate="intersects") if placed[j]]
        if neighbours:
            blockers = [result[j] for j in neighbours if result[j] is not None]
            if blockers:
                geom = shapely.make_valid(geom.difference(shapely.union_all(blockers)))
        parts = [g for g in getattr(geom, "geoms", [geom]) if g.geom_type == "Polygon"]
        parts = [g for g in parts if not g.is_empty and g.area > 0]
        result[idx] = max(parts, key=lambda g: g.area) if parts else None
        placed[idx] = True
    keep = np.array([g is not None for g in result])
    out = gdf.iloc[np.flatnonzero(keep)].copy()
    out = out.set_geometry(gpd.GeoSeries(list(result[keep]), index=out.index, crs=gdf.crs))
    return out


# ---------------------------------------------------------------------------
# Tile seams (port of DelAnyFlow's merge_tile_seams)
# ---------------------------------------------------------------------------


def _pixel_extent(geom: Any, transform: Any) -> tuple[int, int, int, int]:
    """Pixel bounds ``(col0, row0, col1, row1)`` (end-exclusive) of a polygon on *transform*'s grid.

    A polygon traced from a mask has its vertices on pixel corners, so its
    exterior mapped back with the inverse transform, rounded, gives the
    extent of the mask pixels it covers.
    """
    inv = ~transform
    xy = np.asarray(geom.exterior.coords)
    cols = np.rint(inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c)
    rows = np.rint(inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f)
    return int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())


def _pixel_window_polygon(transform: Any, col0: int, row0: int, col1: int, row1: int) -> Any:
    """The pixel window ``[col0, col1) x [row0, row1)`` of *transform*'s grid as a polygon."""
    from shapely.geometry import Polygon

    corners = [(col0, row0), (col1, row0), (col1, row1), (col0, row1)]
    return Polygon(
        [
            (
                transform.a * c + transform.b * r + transform.c,
                transform.d * c + transform.e * r + transform.f,
            )
            for c, r in corners
        ]
    )


def _compose_tile_view(geoms: list[Any], transform: Any, size: int = MODEL_INPUT_PX) -> np.ndarray:
    """Field ids (0: none) of one tile's detections, composed as DelAnyFlow composes a tile.

    The compose step of upstream ``PostprocWorker.process_fields``, without
    land-cover masks: every detection (its polygon with holes filled, as
    upstream fills each field's outer contour) is rasterised on the tile's
    *size* x *size* grid (*transform*), and the detections are written from
    the largest to the smallest, so that a smaller one overwrites a larger
    one. A detection of fewer than ``pixel_area_threshold`` (512) pixels is
    left out; one whose IoU with the visible part of an earlier one exceeds
    ``compose_merge_iou`` (0.8) takes its id; one left with fewer than 512
    visible pixels is left out.
    """
    from rasterio.features import rasterize
    from rasterio.transform import Affine
    from shapely.geometry import Polygon

    fields: list[tuple[tuple[int, int, int, int], np.ndarray] | None] = []
    for geom in geoms:
        c0, r0, c1, r1 = _pixel_extent(geom, transform)
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, size), min(r1, size)
        if c1 <= c0 or r1 <= r0:
            fields.append(None)
            continue
        mask = rasterize(
            [(Polygon(geom.exterior), 1)],
            out_shape=(r1 - r0, c1 - c0),
            transform=transform @ Affine.translation(c0, r0),
            fill=0,
            dtype="uint8",
        ).astype(bool)
        fields.append(((r0, r1, c0, c1), mask))
    areas = np.array([0 if f is None else int(f[1].sum()) for f in fields], dtype=np.int64)
    ids = 2 * (np.arange(len(fields), dtype=np.int64) + 1)  # even ids, as upstream
    order = np.argsort(-areas, kind="stable")
    view = np.zeros((size, size), dtype=np.int32)
    write_id = ids.copy()
    id_area = {int(i): int(a) for i, a in zip(ids, areas, strict=True)}
    for index in order.tolist():
        area = int(areas[index])
        if area < _COMPOSE_MIN_AREA_PX:
            write_id[index] = 0
            continue
        own = int(ids[index])
        (r0, r1, c0, c1), mask = fields[index]
        window = view[r0:r1, c0:c1]
        under, counts = np.unique(window[mask], return_counts=True)
        overlaps = [(int(u), int(c)) for u, c in zip(under, counts, strict=True) if u != 0]
        for key, inter in overlaps:
            if id_area[key] < _COMPOSE_MIN_AREA_PX:
                continue
            if inter / max(1, area + id_area[key] - inter) > _COMPOSE_MERGE_IOU:
                id_area[key] += area - inter
                id_area[own] = 0
                window[mask] = key
                write_id[index] = key
                break
        if id_area[own] >= _COMPOSE_MIN_AREA_PX:
            window[mask] = own
            for key, inter in overlaps:
                id_area[key] -= inter
    view[:] = 0
    for index in order.tolist():
        wid = int(write_id[index])
        if wid == 0 or id_area[wid] < _COMPOSE_MIN_AREA_PX:
            continue
        (r0, r1, c0, c1), mask = fields[index]
        view[r0:r1, c0:c1][mask] = wid
    return view


def _seam_evidence(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What the tile centred on a tile edge saw next to it (upstream ``__evidence``).

    *block* holds that tile's field ids (0: none) in a window across the
    edge, one row per position along it. Returns two boolean arrays: one
    field across the whole window (*crossing*), and a boundary in it, i.e.
    two fields or a gap between fields (*split*).
    """
    field = block > 0
    lowest = np.where(field, block, np.iinfo(block.dtype).max).min(axis=1)
    highest = np.where(field, block, -1).max(axis=1)
    two_fields = field.any(axis=1) & (lowest != highest)
    gap = field[:, 0] & field[:, -1] & ~field.all(axis=1)
    return field.all(axis=1) & ~two_fields, two_fields | gap


def _run_bounds(mask: np.ndarray, first: int, last: int) -> tuple[int, int]:
    """Extent of the run of True in *mask* through positions *first*..*last* (``__run_bounds``)."""
    before = np.flatnonzero(~mask[:first])
    after = np.flatnonzero(~mask[last + 1 :])
    start = int(before[-1]) + 1 if before.size else 0
    end = last + int(after[0]) if after.size else len(mask) - 1
    return start, end


def _strip_depth(block: np.ndarray, field_id: int) -> int:
    """How far *field_id* reaches from the edge (column 0 of *block*), the maximum over rows."""
    inside = block == field_id
    if inside.size == 0:
        return 0
    runs = np.where(inside.all(axis=1), inside.shape[1], np.argmin(inside, axis=1))
    return int(runs.max())


def _colour_diff(pixels_a: np.ndarray, pixels_b: np.ndarray) -> float:
    """Distance between the mean colours of two pixel sets, on a 0-1 scale (0 if one is empty)."""
    if len(pixels_a) == 0 or len(pixels_b) == 0:
        return 0.0
    return float(np.linalg.norm(pixels_a.mean(axis=0) - pixels_b.mean(axis=0)) / 255.0)


def _mutual_best(line_pairs: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """Pairs in which each piece is the other's longest contact on this edge (``__mutual_best``)."""
    best_a: dict[int, tuple[int, int]] = {}
    best_b: dict[int, tuple[int, int]] = {}
    for id_a, id_b, contact in line_pairs:
        if contact > best_a.get(id_a, (-1, -1))[1]:
            best_a[id_a] = (id_b, contact)
        if contact > best_b.get(id_b, (-1, -1))[1]:
            best_b[id_b] = (id_a, contact)
    return [(a, b) for a, b, _ in line_pairs if best_a[a][0] == b and best_b[b][0] == a]


def _border_pairs(
    ids_a: np.ndarray,
    ids_b: np.ndarray,
    cut: np.ndarray,
    crossing: np.ndarray,
    split: np.ndarray,
    load_image: Callable[[], tuple[np.ndarray, np.ndarray] | None],
) -> list[tuple[int, int]]:
    """Upstream DelAnyFlow's seam test on one tile edge (``__border_pairs``).

    *ids_a*, *ids_b*: polygon ids (0: none) on the two sides of the edge,
    one row per position along it and one column per pixel away from it
    (column 0 touches the edge); *cut*: tile-cut flag of every id;
    *crossing*, *split*: :func:`_seam_evidence` of the tile centred on the
    edge, per position; *load_image*: the model input on the two sides in
    the same layout (with a band axis), or None. Only pairs of tile-cut
    polygons are tested. Returns the ``(id_a, id_b)`` pairs to join.
    """
    p = TILE_SEAM_PARAMS
    side_a, side_b = ids_a[:, 0], ids_b[:, 0]
    touching = (side_a > 0) & (side_b > 0) & (side_a != side_b) & cut[side_a] & cut[side_b]
    if not touching.any():
        return []
    keys, inverse, contacts = np.unique(
        (side_a[touching].astype(np.int64) << 32) | side_b[touching].astype(np.int64),
        return_inverse=True,
        return_counts=True,
    )
    n_crossing = np.bincount(inverse, weights=crossing[touching], minlength=len(keys))
    n_split = np.bincount(inverse, weights=split[touching], minlength=len(keys))
    len_a = np.bincount(side_a, minlength=len(cut))
    len_b = np.bincount(side_b, minlength=len(cut))
    positions = np.flatnonzero(touching)
    max_depth = int(p["seam_max_strip_depth_px"])
    k = _SEAM_STRIP_PX // 2
    image: tuple[np.ndarray, np.ndarray] | None = None
    image_loaded = False
    line_pairs: list[tuple[int, int, int]] = []
    for i, (key, contact) in enumerate(zip(keys.tolist(), contacts.tolist(), strict=True)):
        id_a, id_b = key >> 32, key & 0xFFFFFFFF
        # Both cut edges are mostly in contact: a split field has matching cuts on both sides.
        if contact < p["seam_min_contact_px"] or contact < p["seam_min_coverage"] * max(
            len_a[id_a], len_b[id_b]
        ):
            continue
        # The tile centred on the edge must not have seen a boundary there (where it saw
        # no field at all, the geometry decides).
        if n_split[i] > 0 and n_crossing[i] < p["seam_min_evidence"] * (n_crossing[i] + n_split[i]):
            continue
        rows = positions[inverse == i]
        a0, a1 = _run_bounds(side_a == id_a, int(rows.min()), int(rows.max()))
        b0, b1 = _run_bounds(side_b == id_b, int(rows.min()), int(rows.max()))
        tolerance = max(p["seam_end_tolerance_px"], p["seam_end_tolerance_rel"] * contact)
        if abs(a0 - b0) > tolerance or abs(a1 - b1) > tolerance:
            # Cut edges of different extent: two fields, unless one piece is a thin strip.
            depth_a = _strip_depth(ids_a[rows, : max_depth + 1], id_a)
            depth_b = _strip_depth(ids_b[rows, : max_depth + 1], id_b)
            if min(depth_a, depth_b) > max_depth:
                continue
        if not image_loaded:
            image, image_loaded = load_image(), True
        if image is not None and ids_a.shape[1] >= k and ids_b.shape[1] >= k:
            img_a, img_b = image
            in_a, in_b = ids_a[rows, :k] == id_a, ids_b[rows, :k] == id_b
            if (
                _colour_diff(img_a[rows, :k][in_a], img_b[rows, :k][in_b])
                > p["seam_max_colour_diff"]
            ):
                continue
        line_pairs.append((id_a, id_b, contact))
    return _mutual_best(line_pairs)


class _CentreViews:
    """Views of the tiles centred on the tile edges, composed lazily from the raw detections."""

    def __init__(self, raw: gpd.GeoDataFrame | None, transform: Any, window: int) -> None:
        self.transform = transform
        self.window = window
        self.groups: dict[tuple[int, int], list[Any]] = {}
        self.cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray] | None] = {}
        if raw is None or not len(raw) or not {"_tile_x0", "_tile_y0"} <= set(raw.columns):
            return
        x0s = raw["_tile_x0"].to_numpy(dtype=np.int64)
        y0s = raw["_tile_y0"].to_numpy(dtype=np.int64)
        for geom, x0, y0 in zip(raw.geometry.to_numpy(), x0s, y0s, strict=True):
            if geom is not None and not geom.is_empty and geom.geom_type == "Polygon":
                self.groups.setdefault((int(x0), int(y0)), []).append(geom)

    def crosses(self, key: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
        """The tile's ids across its vertical and horizontal centre lines (positions first)."""
        if key not in self.cache:
            geoms = self.groups.get(key)
            if not geoms:
                self.cache[key] = None
            else:
                from rasterio.transform import Affine

                view = _compose_tile_view(geoms, self.transform @ Affine.translation(*key))
                c, w = MODEL_INPUT_PX // 2, self.window
                self.cache[key] = (view[:, c - w : c + w].copy(), view[c - w : c + w, :].T.copy())
        return self.cache[key]

    def evidence(
        self, vertical: bool, pos: int, positions: np.ndarray, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """:func:`_seam_evidence` at *positions* along the edge at *pos* (False elsewhere)."""
        crossing = np.zeros(length, dtype=bool)
        split = np.zeros(length, dtype=bool)
        step = MODEL_INPUT_PX // 2
        # With a half-tile step the central halves of the tiles partition the raster;
        # the tile centred on the edge whose central half holds a position saw it.
        origins = step * ((positions + step // 2) // step) - step
        for origin in np.unique(origins).tolist():
            sel = positions[origins == origin]
            key = (pos - step, origin) if vertical else (origin, pos - step)
            crosses = self.crosses(key)
            if crosses is None:
                continue
            block = crosses[0 if vertical else 1][sel - origin]
            crossing[sel], split[sel] = _seam_evidence(block)
        return crossing, split


def merge_tile_seams(
    gdf: gpd.GeoDataFrame,
    raw: gpd.GeoDataFrame | None,
    transform: Any,
    shape: tuple[int, int],
    image: Callable[[int, int, int, int], np.ndarray] | None = None,
    *,
    score_column: str = "confidence",
    edge_column: str = "_tile_edge",
) -> tuple[gpd.GeoDataFrame, int]:
    """Join tile-cut polygons that meet along an interior tile edge (DelAnyFlow's seam test).

    A field that no tile saw whole, and whose pieces from neighbouring tiles
    do not overlap enough for :func:`merge_tile_pieces`, is left cut by a
    straight line on the tile grid. This is a port of upstream DelAnyFlow's
    ``merge_tile_seams`` (Lavreniuk/Delineate-Anything 488e537,
    ``methods/main/PostprocHandler.py``) to polygons, with its thresholds
    (:data:`TILE_SEAM_PARAMS`). The polygons are rasterised on the
    model-input grid along every interior tile edge (the native backend's
    tiles overlap by half, so the edges are every ``MODEL_INPUT_PX // 2``
    model-input pixels), and two tile-cut polygons (*edge_column* True) on
    opposite sides of an edge are joined when:

    - they touch along the edge for at least ``seam_min_contact_px`` pixels,
      and for at least ``seam_min_coverage`` of the longer of their two
      edges on that line;
    - the tile centred on the edge did not see a boundary there: where that
      tile saw one field across the edge or a boundary within
      ``seam_split_window_px`` of it, it saw one field at least
      ``seam_min_evidence`` of the time (where it saw no field, the
      geometry decides); its view is its own detections in *raw*, composed
      as DelAnyFlow composes a tile (smaller fields over larger ones);
    - their two cut edges start and end within
      ``max(seam_end_tolerance_px, seam_end_tolerance_rel * contact)``
      pixels of each other, or the smaller piece reaches at most
      ``seam_max_strip_depth_px`` pixels from the edge;
    - the mean colour of the model input differs by at most
      ``seam_max_colour_diff`` (0-1 scale) between the two pieces, within
      ``SEAM_STRIP // 2`` (4) pixels of the edge (skipped without *image*);
    - each is the other's longest contact among the pairs that pass these
      tests on that edge.

    Joins are transitive, as upstream. Unlike upstream, only tile-cut
    polygons are joined: a tile saw a complete polygon whole, so its outline
    is the model's, not the tiling's. The members of a group are snapped to
    the model-input pixel grid before their union (pieces from different
    tiles can otherwise be left apart by about 1e-10 m where the pixel size
    or origin is not exact in binary); a member that the union still does
    not connect to another one is left as it was and not counted as a join.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Non-overlapping polygons (:func:`resolve_overlaps` output) in the
        CRS of *transform*.
    raw : geopandas.GeoDataFrame or None
        Every tile's raw detections, with the tile origin in model-input
        pixels in ``_tile_x0`` and ``_tile_y0``. None, or no such columns:
        no centre-tile evidence.
    transform : affine.Affine
        Transform of the model-input pixel grid (the raster transform scaled
        by ``1 / super_resolution``), on which the tiles start every
        ``MODEL_INPUT_PX // 2`` pixels from ``-MODEL_INPUT_PX // 2``.
    shape : tuple of int
        ``(height, width)`` of the raster in model-input pixels.
    image : callable or None
        ``image(col0, row0, col1, row1)`` returns the model input (uint8,
        rows x columns x bands) of that window of model-input pixels, zero
        outside the raster. None: no colour test.
    score_column, edge_column : str
        Confidence and tile-cut columns.

    Returns
    -------
    tuple of (geopandas.GeoDataFrame, int)
        The polygons, unjoined rows first (original order), then one row per
        joined group (the union, with the members' highest *score_column*,
        *edge_column* True and the other columns of the highest-scoring
        member); and the number of joins (polygons removed).
    """
    import pandas as pd
    import shapely
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    out = gdf.copy()
    n = len(gdf)
    if n < 2 or edge_column not in gdf:
        return out, 0
    geoms, _areas, scores, edges = _detection_arrays(gdf, score_column, edge_column)
    if int(edges.sum()) < 2:
        return out, 0
    height, width = (int(v) for v in shape)
    step = MODEL_INPUT_PX // 2
    depth = max(int(TILE_SEAM_PARAMS["seam_max_strip_depth_px"]) + 1, _SEAM_STRIP_PX // 2)
    k = _SEAM_STRIP_PX // 2
    cut = np.concatenate([[False], edges])  # indexed by id = row position + 1
    tree = shapely.STRtree(geoms)
    views = _CentreViews(raw, transform, int(TILE_SEAM_PARAMS["seam_split_window_px"]))

    def load_image(vertical: bool, pos: int) -> tuple[np.ndarray, np.ndarray] | None:
        if image is None:
            return None
        if vertical:
            strip = image(pos - k, 0, pos + k, height)  # (rows, 2k, bands)
        else:
            strip = np.transpose(image(0, pos - k, width, pos + k), (1, 0, 2))
        return strip[:, :k][:, ::-1], strip[:, k:]

    pairs: list[tuple[int, int]] = []
    for vertical in (True, False):
        across, along = (width, height) if vertical else (height, width)
        for pos in range(step, across, step):
            lo, hi = max(pos - depth, 0), min(pos + depth, across)
            win = (lo, 0, hi, height) if vertical else (0, lo, width, hi)
            cand = tree.query(_pixel_window_polygon(transform, *win))
            if int(edges[cand].sum()) < 2:
                continue
            strip = rasterize(
                [(geoms[i], int(i) + 1) for i in cand.tolist()],
                out_shape=(win[3] - win[1], win[2] - win[0]),
                transform=transform @ Affine.translation(win[0], win[1]),
                fill=0,
                dtype="int32",
            )
            band = strip if vertical else strip.T  # (along the edge, across it)
            ids_a, ids_b = band[:, : pos - lo][:, ::-1], band[:, pos - lo :]
            side_a, side_b = ids_a[:, 0], ids_b[:, 0]
            touching = (side_a > 0) & (side_b > 0) & (side_a != side_b) & cut[side_a] & cut[side_b]
            if not touching.any():
                continue
            crossing, split = views.evidence(vertical, pos, np.flatnonzero(touching), along)
            pairs += _border_pairs(
                ids_a,
                ids_b,
                cut,
                crossing,
                split,
                lambda vertical=vertical, pos=pos: load_image(vertical, pos),
            )
    if not pairs:
        return out, 0

    root = np.arange(n)

    def find(i: int) -> int:
        while root[i] != i:
            root[i] = root[root[i]]
            i = int(root[i])
        return i

    for a, b in pairs:
        ra, rb = find(a - 1), find(b - 1)
        if ra != rb:
            root[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    joined_members: list[int] = []
    leaders, shapes, best_scores = [], [], []
    for members in (m for m in groups.values() if len(m) > 1):
        for shape, idx in _seam_union_parts(geoms, np.asarray(members), transform):
            if len(idx) < 2:  # not connected to another member: kept as it was
                continue
            joined_members.extend(idx.tolist())
            leaders.append(int(idx[np.argmax(scores[idx])]))
            shapes.append(shape)
            best_scores.append(float(scores[idx].max()))
    if not leaders:
        return out, 0
    joined = out.iloc[leaders].copy()
    joined[gdf.geometry.name] = gpd.GeoSeries(shapes, index=joined.index, crs=gdf.crs)
    if score_column in joined:
        joined[score_column] = best_scores
    joined[edge_column] = True
    kept = out.iloc[np.setdiff1d(np.arange(n), joined_members)]  # positional: labels may repeat
    result = pd.concat([kept, joined], ignore_index=True)
    n_joins = len(joined_members) - len(leaders)
    return gpd.GeoDataFrame(result, geometry=gdf.geometry.name, crs=gdf.crs), n_joins


def _snap_to_pixel_grid(geom: Any, transform: Any, tolerance_px: float = _GRID_SNAP_PX) -> Any:
    """*geom* with the vertices near a pixel corner of *transform* moved onto that corner.

    A vertex within *tolerance_px* pixels of a corner on both axes gets the
    corner's coordinates, recomputed from its integer column and row with
    *transform*, so two polygons traced on different tiles of the same grid
    get bitwise identical coordinates on the edges they share. Their own
    tile transforms (the raster transform composed with the tile and crop
    offsets) can differ from each other by about 1e-10 m when the pixel size
    or the origin is not exact in binary (e.g. 1.2 m pixels, a
    ``processing_gsd_m`` copy at 2.4 m, the 0.59994 m Web Mercator grid of
    ``usgs-naip-plus``), enough to leave a gap between two pieces that touch.
    Other vertices are left unchanged.
    """
    import shapely

    inv = ~transform

    def snap(xy: np.ndarray) -> np.ndarray:
        cols = inv.a * xy[:, 0] + inv.b * xy[:, 1] + inv.c
        rows = inv.d * xy[:, 0] + inv.e * xy[:, 1] + inv.f
        c, r = np.rint(cols), np.rint(rows)
        on = (np.abs(cols - c) <= tolerance_px) & (np.abs(rows - r) <= tolerance_px)
        out = np.array(xy, dtype=float, copy=True)
        out[on, 0] = transform.a * c[on] + transform.b * r[on] + transform.c
        out[on, 1] = transform.d * c[on] + transform.e * r[on] + transform.f
        return out

    return shapely.transform(geom, snap)


def _seam_union_parts(
    geoms: np.ndarray, members: np.ndarray, transform: Any
) -> list[tuple[Any, np.ndarray]]:
    """The union of a group joined by :func:`merge_tile_seams`, as ``(polygon, members)`` parts.

    The members (positions in *geoms*) are snapped to the model-input pixel
    grid (:func:`_snap_to_pixel_grid`) before the union, so that pieces whose
    shared tile edge was computed from different tile transforms are not
    left apart by a gap of about 1e-10 m. A connected union is one part with
    every member. If the union still has several polygon parts, each part
    is returned with the members whose representative point it holds, so
    that the caller joins only the members a part connects; if a part holds
    no member (not expected), every member is returned alone. No member's
    area is ever dropped.
    """
    import shapely

    snapped = [_snap_to_pixel_grid(geoms[i], transform) for i in members.tolist()]
    union = shapely.make_valid(shapely.union_all(snapped))
    parts = [g for g in getattr(union, "geoms", [union]) if g.geom_type == "Polygon" and g.area > 0]
    if len(parts) == 1:
        return [(parts[0], members)]
    owner = np.full(len(members), -1)
    for i, geom in enumerate(snapped):
        point = geom.representative_point()
        owner[i] = next((k for k, part in enumerate(parts) if part.intersects(point)), -1)
    if not parts or (owner < 0).any() or len(set(owner.tolist())) != len(parts):
        return [(geoms[i], members[[j]]) for j, i in enumerate(members.tolist())]
    return [(part, members[owner == k]) for k, part in enumerate(parts)]


def _area_m2(gdf: gpd.GeoDataFrame) -> np.ndarray:
    """Polygon areas in m², computed in the equal-area EPSG:6933.

    Upstream DelAnyFlow and ftw-tools compute field areas in EPSG:6933 too.
    Areas in the raster CRS would be wrong for, e.g., Web Mercator (inflated
    by about sec²(latitude)) or feet-based CRSs.
    """
    if len(gdf) == 0:
        return np.zeros(0)
    if gdf.crs is None:
        raise ValueError("The detections have no CRS; cannot compute their areas in m²")
    from agribound.io.crs import get_equal_area_crs

    return gdf.geometry.to_crs(get_equal_area_crs()).area.to_numpy(dtype=float)


def _raster_fingerprint(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return f"{resolved}|{stat.st_size}|{stat.st_mtime_ns}"


def _package_version(dist: str) -> str | None:
    try:
        import importlib.metadata as md

        return md.version(dist)
    except Exception:
        return None


def _precision_kwargs(device: str, half: bool) -> dict[str, Any]:
    """FP16 keyword for Ultralytics predict: ``quantize`` (>= 8.4.80) or ``half``."""
    if device == "cpu" or not half:
        return {}
    from ultralytics.cfg import DEFAULT_CFG_DICT

    return {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

_REGISTRY_ENTRY = ENGINE_REGISTRY["delineate-anything"]


class DelineateAnythingEngine(DelineationEngine):
    """Field boundary delineation with Delineate-Anything (see the module docstring)."""

    name = "delineate-anything"
    supported_sources = list(_REGISTRY_ENTRY["supported_sources"])
    requires_bands = list(_REGISTRY_ENTRY["requires_bands"])

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Run Delineate-Anything on a raster.

        Parameters
        ----------
        raster_path : str
            Input GeoTIFF (composite or local file).
        config : AgriboundConfig
            Pipeline configuration; ``engine_params`` select the backend and
            model (module docstring).

        Returns
        -------
        geopandas.GeoDataFrame
            Field polygons in the raster CRS with
            ``gdf.attrs["engine_meta"]``.
        """
        self.validate_input(raster_path, config)
        opts = DAOptions.from_engine_params(config.engine_params)
        rgb = get_canonical_band_indices(
            config.source, ["R", "G", "B"], bands=effective_bands(config)
        )
        if opts.backend == "native":
            return _run_native(raster_path, config, opts, rgb)
        if opts.backend == "reference":
            return _run_reference(raster_path, config, opts, rgb)
        return _run_ftw(raster_path, config, opts, rgb)

    @classmethod
    def prefetch(cls, config: AgriboundConfig) -> list[str]:
        """Download the weights selected by ``config.engine_params``.

        ``native``/``reference``: the pinned Hugging Face file (SHA-256
        checked), unless ``checkpoint_path`` is set (then that file is
        returned if it exists). ``ftw``: ftw-tools' checkpoint URL, which
        Ultralytics resolves relative to the current working directory, is
        downloaded into the current working directory, so the later run must
        start from the same directory to find it offline.

        Returns
        -------
        list[str]
            Local paths of the weights.
        """
        opts = DAOptions.from_engine_params(config.engine_params)
        if opts.checkpoint_path:
            path = Path(opts.checkpoint_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"checkpoint_path does not exist: {path}")
            return [str(path.resolve())]
        if opts.backend in ("native", "reference"):
            return [download_da_weights(opts.model_key)]
        url = _ftw_checkpoint_url(opts.model.ftw_name)
        if url is None:
            raise RuntimeError(
                f"The installed ftw-tools has no checkpoint URL for {opts.model.ftw_name!r}."
            )
        from urllib.parse import unquote, urlparse

        import torch

        target = Path.cwd() / Path(unquote(urlparse(url).path)).name
        if not target.is_file():
            torch.hub.download_url_to_file(url, str(target), progress=True)
        return [str(target)]


# ---------------------------------------------------------------------------
# Native backend
# ---------------------------------------------------------------------------


def _weights_for(opts: DAOptions) -> tuple[str, dict[str, Any]]:
    """Return the weights path and its provenance for native/reference runs."""
    if opts.checkpoint_path:
        path = Path(opts.checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint_path does not exist: {path}")
        return str(path.resolve()), {
            "weights": "checkpoint",
            "checkpoint_path": str(path.resolve()),
            "checkpoint_sha256": file_sha256(path),
            "base_model_key": opts.model_key,
        }
    spec = opts.model
    path = download_da_weights(spec.key)
    return path, {
        "weights": "pinned",
        "weights_repo": DA_HF_REPO,
        "weights_filename": spec.filename,
        "weights_revision": spec.revision,
        "weights_sha256": spec.sha256,
    }


def _trained_input(record: dict[str, Any]) -> dict[str, Any]:
    """The input a fine-tuned checkpoint saw in training, from its training metadata.

    ``gsd_m`` (pixel size of the chips: the raster's, or that of its copy at
    ``processing_gsd_m``), ``super_resolution`` and ``stretch_percentiles``
    (None when the chips were cut from a uint8 raster, which is used without
    a stretch; 1-99 in records of agribound 1.0.1 and earlier, which did not
    record them). Whether the chips were uint8 is read from
    ``raster_dtype``, or, in older records, from the chips' stretch bounds
    (0 and 255 in every band only for uint8 rasters: the bounds of other
    rasters are percentiles of their positive values). A value the record
    does not give is left out.
    """
    out: dict[str, Any] = {}
    if record.get("gsd_m") is not None:
        out["gsd_m"] = round(float(record["gsd_m"]), 4)
    if record.get("super_resolution") is not None:
        out["super_resolution"] = int(record["super_resolution"])
    chips = record.get("chips_stretch") or {}
    dtype = record.get("raster_dtype")
    if dtype is not None:
        unstretched = dtype == "uint8"
    elif chips.get("stretch_lows") is not None and chips.get("stretch_highs") is not None:
        unstretched = all(v == 0 for v in chips["stretch_lows"]) and all(
            v == 255 for v in chips["stretch_highs"]
        )
    else:
        return out
    percentiles = record.get("stretch_percentiles", DEFAULT_STRETCH_PERCENTILES)
    out["stretch_percentiles"] = None if unstretched else [float(v) for v in percentiles]
    return out


def _check_checkpoint_input(
    weights_meta: dict[str, Any], meta: dict[str, Any], run: dict[str, Any]
) -> None:
    """Warn when a fine-tuned checkpoint saw another input in training than in this run.

    *run* is this run's input as the model sees it: ``gsd_m``, the pixel
    size of the raster read (the raster's, or that of its copy at
    ``processing_gsd_m``), ``super_resolution`` and ``stretch_percentiles``
    (None for a uint8 raster, which is not stretched). It is compared with
    the training input recorded next to the checkpoint
    (``<checkpoint>.agribound.json``, see :func:`_trained_input`): pixel
    sizes that differ by more than :data:`PROCESSING_GSD_TOLERANCE` (1 %),
    another super-resolution factor or another stretch log a WARNING, and
    ``meta["checkpoint_input_mismatch"]`` then holds both. The options
    themselves are not compared: ``processing_gsd_m`` set to the raster's own
    pixel size changes nothing, and a checkpoint fine-tuned on 1.5 m chips
    sees another input on a 3 m raster without any option. Values a record
    lacks are not compared; checkpoints without such metadata are not
    checked.
    """
    if weights_meta.get("weights") != "checkpoint":
        return
    from agribound.engines.finetune._data import read_training_meta

    record = read_training_meta(weights_meta["checkpoint_path"])
    if record.get("engine") != "delineate-anything":
        return
    trained = _trained_input(record)
    run = {
        "gsd_m": round(float(run["gsd_m"]), 4),
        "super_resolution": int(run["super_resolution"]),
        "stretch_percentiles": None
        if run["stretch_percentiles"] is None
        else [float(v) for v in run["stretch_percentiles"]],
    }
    differs = [
        key
        for key, value in trained.items()
        if (
            abs(run[key] - value) > PROCESSING_GSD_TOLERANCE * value
            if key == "gsd_m"
            else run[key] != value
        )
    ]
    if differs:
        logger.warning(
            "The checkpoint was fine-tuned on %s, but this run gives the model %s "
            "(stretch_percentiles None: a uint8 raster, used without a stretch): the model sees "
            "a different input than in training.",
            trained,
            run,
        )
        meta["checkpoint_input_mismatch"] = {"trained": trained, "run": run}


def _read_tile(
    src: Any, bands: list[int], x0: int, y0: int, size: int, height: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``size`` x ``size`` window (zero-padded outside the raster).

    Returns ``(data float32 (bands, size, size), valid bool (size, size))``;
    with *height*, the window is *height* rows by *size* columns. A pixel is
    invalid if any band is non-finite, all bands are 0 (the upstream default
    ``nodata_value: [0, 0, 0]``), or all bands equal the raster's nodata
    value.
    """
    from rasterio.windows import Window

    rows = size if height is None else height
    data = np.zeros((len(bands), rows, size), dtype=np.float32)
    valid = np.zeros((rows, size), dtype=bool)
    c0, r0 = max(x0, 0), max(y0, 0)
    c1, r1 = min(x0 + size, src.width), min(y0 + rows, src.height)
    if c1 <= c0 or r1 <= r0:
        return data, valid
    arr = src.read(bands, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    data[:, r0 - y0 : r1 - y0, c0 - x0 : c1 - x0] = np.where(np.isfinite(arr), arr, 0)
    valid[r0 - y0 : r1 - y0, c0 - x0 : c1 - x0] = _valid_pixels(arr, src.nodata)
    return data, valid


def _valid_pixels(arr: np.ndarray, nodata: float | None) -> np.ndarray:
    """``(H, W)`` mask of the valid pixels of ``(bands, H, W)`` *arr* (see :func:`_read_tile`)."""
    ok = np.all(np.isfinite(arr), axis=0) & ~np.all(arr == 0, axis=0)
    if nodata is not None and np.isfinite(nodata):
        ok &= ~np.all(arr == nodata, axis=0)
    return ok


def processing_scale(raster_gsd_m: float, processing_gsd_m: float) -> float:
    """Factor by which ``processing_gsd_m`` coarsens a raster of *raster_gsd_m* pixels.

    Returns 1.0 (the raster is read as it is) when *processing_gsd_m* is
    within :data:`PROCESSING_GSD_TOLERANCE` (1 %) of the raster's pixel size.

    Raises
    ------
    ValueError
        If *processing_gsd_m* is finer than the raster's pixels by more than
        that tolerance: the option only makes pixels coarser.
    """
    scale = float(processing_gsd_m) / float(raster_gsd_m)
    if scale < 1.0 - PROCESSING_GSD_TOLERANCE:
        raise ValueError(
            f"processing_gsd_m={processing_gsd_m:g} m is finer than the raster's pixels "
            f"({raster_gsd_m:.4g} m); it can only make them coarser. Leave it unset to read "
            "the raster at its own pixel size."
        )
    return 1.0 if scale <= 1.0 + PROCESSING_GSD_TOLERANCE else scale


def _area_weights(n_src: int, scale: float, n_dst: int) -> Any:
    """``(n_dst, n_src)`` sparse matrix of the length of source pixel *i* inside coarser pixel *j*.

    Coarser pixel *j* covers ``[j * scale, (j + 1) * scale)`` in source
    pixels, clipped to the ``n_src`` pixels of the raster.
    """
    from scipy import sparse

    edges = np.minimum(np.arange(n_dst + 1, dtype=np.float64) * scale, float(n_src))
    first = np.floor(edges[:-1]).astype(np.int64)
    counts = np.maximum(np.minimum(np.ceil(edges[1:]).astype(np.int64), n_src) - first, 0)
    rows = np.repeat(np.arange(n_dst, dtype=np.int64), counts)
    offsets = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    cols = first[rows] + offsets
    lengths = np.minimum(edges[1:][rows], cols + 1.0) - np.maximum(edges[:-1][rows], cols)
    keep = lengths > 1e-9
    return sparse.csr_matrix(
        (lengths[keep], (rows[keep], cols[keep])), shape=(n_dst, n_src), dtype=np.float64
    )


def _write_processing_raster(src: Any, bands: list[int], scale: float, path: Path) -> None:
    """Write *bands* of *src* on pixels *scale* times larger (see :func:`processing_raster`)."""
    import rasterio
    from rasterio.windows import Window

    height = int(np.ceil(src.height / scale - 1e-9))
    width = int(np.ceil(src.width / scale - 1e-9))
    row_weights = _area_weights(src.height, scale, height)
    col_weights = _area_weights(src.width, scale, width)
    uint8 = all(src.dtypes[b - 1] == "uint8" for b in bands)
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(bands),
        "dtype": "uint8" if uint8 else "float32",
        "crs": src.crs,
        "transform": src.transform @ rasterio.Affine.scale(scale),
        "nodata": 0 if uint8 else float("nan"),
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": "deflate",
        "BIGTIFF": "IF_SAFER",
    }
    step = max(1, int(_PROCESSING_CHUNK_VALUES // (src.width * len(bands) * scale)))
    # Written under a temporary name and renamed when complete, so an interrupted run
    # leaves no partial copy for the next run to reuse.
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with rasterio.open(tmp, "w", **profile) as dst:
            dst.update_tags(
                AGRIBOUND_DA_PROCESSING_SCALE=repr(float(scale)),
                AGRIBOUND_DA_PROCESSING_SOURCE=_raster_fingerprint(src.name),
                AGRIBOUND_DA_PROCESSING_BANDS=json.dumps(bands),
                AGRIBOUND_DA_PROCESSING_RESAMPLING="average",
            )
            for j0 in range(0, height, step):
                j1 = min(j0 + step, height)
                r0, r1 = int(np.floor(j0 * scale)), min(int(np.ceil(j1 * scale)), src.height)
                window = Window(0, r0, src.width, r1 - r0)
                data = src.read(bands, window=window).astype(np.float64)
                valid = _valid_pixels(data, src.nodata)
                w_rows = row_weights[j0:j1, r0:r1]

                def area_sum(values: np.ndarray, w_rows: Any = w_rows) -> np.ndarray:
                    return np.asarray((col_weights @ (w_rows @ values).T).T)

                covered = area_sum(valid.astype(np.float64))
                ok = covered > 0
                out = np.zeros((len(bands), j1 - j0, width), dtype=profile["dtype"])
                for k in range(len(bands)):
                    total = area_sum(np.where(valid, data[k], 0.0))
                    mean = np.divide(total, covered, out=np.zeros_like(total), where=ok)
                    if uint8:
                        out[k] = np.where(ok, np.clip(np.floor(mean + 0.5), 0, 255), 0)
                    else:
                        out[k] = np.where(ok, mean, np.nan)
                dst.write(out, window=Window(0, j0, width, j1 - j0))
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def processing_raster(
    raster_path: str,
    config: AgriboundConfig,
    processing_gsd_m: float | None,
    bands: list[int],
) -> tuple[str, list[int], dict[str, Any] | None]:
    """The raster that the native backend, and its fine-tuning chips, read.

    Without *processing_gsd_m*, or when it is within 1 % of the raster's
    pixel size (:func:`processing_scale`, with the pixel size of
    :func:`pixel_size_m`), this is the raster itself. Otherwise it is the
    raster resampled to *processing_gsd_m* by average resampling: the grid
    starts at the raster's origin, its pixels are ``scale = processing_gsd_m
    / pixel size`` times larger on both axes and cover the whole raster, and
    each pixel is the area-weighted mean of the valid raster pixels it covers
    (valid as in :func:`_read_tile`; in the interior this equals GDAL's
    ``average`` resampling). Pixels that cover no valid pixel are nodata: NaN
    in float32, or 0 in every band when all the bands are uint8, which stay
    uint8 (means rounded) so that they are still used unstretched. The copy
    holds only the distinct *bands*, in increasing order, and is written
    once to the working directory (``da_processing_<key>.tif``, keyed by the
    raster's path, size and modification time, *processing_gsd_m* and the
    bands) and reused. Polygons delineated on it are in the raster's CRS.

    Parameters
    ----------
    raster_path : str
        Input raster.
    config : AgriboundConfig
        Configuration; its working directory holds the copy.
    processing_gsd_m : float or None
        Pixel size to read the raster at, in metres (validated by
        :class:`DAOptions`).
    bands : list[int]
        1-based band indices to read (repeats allowed, e.g. ``[1, 1, 1]``).

    Returns
    -------
    tuple
        ``(path, band indices in that raster, info)``. *info* is None when
        the raster is read as it is, else ``{"gsd_m", "scale", "size_px",
        "resampling", "raster"}`` (pixel size of the copy, the factor, its
        ``[height, width]``, ``"average"`` and its path).

    Raises
    ------
    ValueError
        If *processing_gsd_m* is finer than the raster's pixels.
    """
    import rasterio

    from agribound._cache import cache_path

    if processing_gsd_m is None:
        return str(raster_path), list(bands), None
    with rasterio.open(raster_path) as src:
        gsd = pixel_size_m(src.crs, src.transform, src.height, src.width)
        scale = processing_scale(gsd, processing_gsd_m)
        if scale == 1.0:
            return str(raster_path), list(bands), None
        distinct = sorted(set(int(b) for b in bands))
        path = cache_path(
            config,
            "da_processing",
            ".tif",
            _raster_fingerprint(raster_path),
            f"processing_gsd_m={float(processing_gsd_m)!r}",
            f"bands={distinct}",
            f"version={_PROCESSING_RASTER_VERSION}",
        )
        if path.exists():
            logger.info("Using the cached %.2f m copy of the raster: %s", processing_gsd_m, path)
        else:
            _write_processing_raster(src, distinct, scale, path)
            logger.info(
                "Resampled the raster from %.2f m to %.2f m pixels (average resampling): %s",
                gsd,
                processing_gsd_m,
                path,
            )
    with rasterio.open(path) as out:
        info = {
            "gsd_m": round(float(pixel_size_m(out.crs, out.transform, out.height, out.width)), 4),
            "scale": round(scale, 6),
            "size_px": [out.height, out.width],
            "resampling": "average",
            "raster": str(path),
        }
    return str(path), [distinct.index(int(b)) + 1 for b in bands], info


def scene_stretch_bounds(
    src: Any,
    bands: list[int],
    percentiles: tuple[float, float] = DEFAULT_STRETCH_PERCENTILES,
) -> tuple[list[float], list[float]]:
    """Scene-level stretch bounds, sampled as upstream ``DataAnalyser`` does.

    uint8 rasters get fixed bounds 0/255. Otherwise the bands are read on a
    nearest-neighbour grid of at most 4096 px per side (GDAL/rasterio
    decimated read of the full-resolution data: a raster with overviews is
    reopened with ``OVERVIEW_LEVEL=NONE``, as upstream does, because GDAL
    would otherwise sample the (usually averaged) overviews, which narrows
    the percentile range), pixels with all bands 0, equal to nodata or
    non-finite are excluded, and
    :func:`agribound.io.raster.percentile_stretch_uint8` computes the
    *percentiles* (default 1 and 99, upstream's) of the strictly positive
    values of each band.
    """
    import rasterio
    from rasterio.enums import Resampling

    from agribound.io.raster import MAX_SAMPLE_SIDE, percentile_stretch_uint8

    if src.dtypes[bands[0] - 1] == "uint8":
        return [0.0] * len(bands), [255.0] * len(bands)
    scale = max(src.width, src.height) / MAX_SAMPLE_SIDE
    buf_x = src.width if scale <= 1 else max(1, int(src.width / scale))
    buf_y = src.height if scale <= 1 else max(1, int(src.height / scale))
    out_shape = (len(bands), buf_y, buf_x)
    if scale > 1 and any(src.overviews(b) for b in bands):
        with rasterio.open(src.name, OVERVIEW_LEVEL="NONE") as full:
            sample = full.read(bands, out_shape=out_shape, resampling=Resampling.nearest)
    else:
        sample = src.read(bands, out_shape=out_shape, resampling=Resampling.nearest)
    sample = sample.astype(np.float32)
    invalid = np.all(sample == 0, axis=0)
    if src.nodata is not None and np.isfinite(src.nodata):
        invalid |= np.all(sample == src.nodata, axis=0)
    sample[:, invalid] = np.nan
    low, high = percentiles
    lows, highs = percentile_stretch_uint8(sample, low=low, high=high, return_bounds_only=True)
    return [float(v) for v in lows], [float(v) for v in highs]


def _mask_polygon(mask: np.ndarray, transform: Any) -> Any | None:
    """Largest polygon (with holes) of a boolean mask, or None."""
    from rasterio.features import shapes
    from shapely.geometry import shape

    best, best_area = None, 0.0
    for geom, value in shapes(mask.astype(np.uint8), mask=mask, transform=transform):
        if value != 1:
            continue
        poly = shape(geom)
        if poly.area > best_area:
            best, best_area = poly, poly.area
    return best


def _model_input_reader(
    src: Any, bands_bgr: list[int], lows: list[float], highs: list[float], sr: int
) -> Callable[[int, int, int, int], np.ndarray]:
    """``read(col0, row0, col1, row1)``: the model input of a window of model-input pixels.

    The window is built as the tiles are: stretched to uint8 with the scene
    bounds (invalid pixels and pixels outside the raster 0), in BGR order,
    then for ``sr > 1`` upsampled with bicubic interpolation from a native
    window with a 2-pixel margin, so the values equal those of a tile that
    holds the window away from its own edges. Returns ``(rows, cols, 3)``.
    """
    import cv2

    def read(col0: int, row0: int, col1: int, row1: int) -> np.ndarray:
        if sr == 1:
            data, valid = _read_tile(src, bands_bgr, col0, row0, col1 - col0, row1 - row0)
            return np.transpose(stretch_with_bounds(data, lows, highs, valid), (1, 2, 0))
        margin = 2  # native pixels: the reach of the bicubic kernel
        n_c0, n_r0 = col0 // sr - margin, row0 // sr - margin
        n_c1, n_r1 = -(-col1 // sr) + margin, -(-row1 // sr) + margin
        data, valid = _read_tile(src, bands_bgr, n_c0, n_r0, n_c1 - n_c0, n_r1 - n_r0)
        img = np.ascontiguousarray(
            np.transpose(stretch_with_bounds(data, lows, highs, valid), (1, 2, 0))
        )
        up = cv2.resize(
            img, ((n_c1 - n_c0) * sr, (n_r1 - n_r0) * sr), interpolation=cv2.INTER_CUBIC
        )
        dc, dr = col0 - n_c0 * sr, row0 - n_r0 * sr
        return up[dr : dr + row1 - row0, dc : dc + col1 - col0]

    return read


def _native_detections(
    model: Any,
    src: Any,
    bands_bgr: list[int],
    lows: list[float],
    highs: list[float],
    sr: int,
    opts: DAOptions,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Tile the raster, run the model and polygonise every detection.

    Each row holds the largest polygon of the detection's mask within the
    valid pixels, its ``confidence``, ``_tile_edge`` (the polygon reaches
    into the :data:`TILE_EDGE_STRIP_PX` strip along an interior side of its
    tile) and the tile origin in model-input pixels (``_tile_x0``,
    ``_tile_y0``).
    """
    import cv2
    import rasterio.transform

    tile = MODEL_INPUT_PX // sr
    step = max(1, int(tile * opts.tile_step))
    xs = tile_origins(src.width, tile, step)
    ys = tile_origins(src.height, tile, step)
    precision = _precision_kwargs(device, opts.half)
    base = src.transform
    stats = {"n_tiles_total": len(xs) * len(ys), "n_tiles_processed": 0, "n_raw": 0}
    rows: list[dict[str, Any]] = []

    def run_batch(batch: list[tuple[int, int, np.ndarray, np.ndarray]]) -> None:
        images = [item[2] for item in batch]
        with no_nms_time_limit():  # a slow NMS must not drop the batch's later tiles
            results = model.predict(
                images,
                conf=opts.conf_threshold,
                iou=opts.iou_threshold,
                max_det=opts.max_detections,
                imgsz=MODEL_INPUT_PX,
                retina_masks=True,
                verbose=False,
                device=device,
                **precision,
            )
        for (x0, y0, _img, valid_sr), result in zip(batch, results, strict=True):
            if result.masks is None or result.masks.data.shape[0] == 0:
                continue
            masks = (refine_masks(result.masks.data) > 0.5).cpu().numpy()
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            tile_tf = base @ rasterio.transform.Affine.translation(x0, y0)
            tile_tf = tile_tf @ rasterio.transform.Affine.scale(1.0 / sr)
            interior = {
                "left": x0 > 0,
                "right": x0 + tile < src.width,
                "top": y0 > 0,
                "bottom": y0 + tile < src.height,
            }
            size = MODEL_INPUT_PX
            strip = TILE_EDGE_STRIP_PX
            for k in range(masks.shape[0]):
                bx0, by0, bx1, by1 = boxes[k]
                cx0 = max(int(np.floor(bx0)) - 3, 0)
                cy0 = max(int(np.floor(by0)) - 3, 0)
                cx1 = min(int(np.ceil(bx1)) + 3, size)
                cy1 = min(int(np.ceil(by1)) + 3, size)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                crop = masks[k, cy0:cy1, cx0:cx1] & valid_sr[cy0:cy1, cx0:cx1]
                if not crop.any():
                    continue
                crop_tf = tile_tf @ rasterio.transform.Affine.translation(cx0, cy0)
                poly = _mask_polygon(crop, crop_tf)
                if poly is None or poly.is_empty:
                    continue
                # Tile-cut: the kept polygon reaches into the edge strip of an interior
                # side (upstream marks the merging_edge_width strip of the composed field).
                p0, q0, p1, q1 = _pixel_extent(poly, crop_tf)
                edge = (
                    (interior["left"] and cx0 + p0 < strip)
                    or (interior["right"] and cx0 + p1 > size - strip)
                    or (interior["top"] and cy0 + q0 < strip)
                    or (interior["bottom"] and cy0 + q1 > size - strip)
                )
                rows.append(
                    {
                        "geometry": poly,
                        "confidence": float(confs[k]),
                        "_tile_edge": edge,
                        "_tile_x0": x0 * sr,
                        "_tile_y0": y0 * sr,
                    }
                )
                stats["n_raw"] += 1
        del results

    batch: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for y0 in ys:
        for x0 in xs:
            data, valid = _read_tile(src, bands_bgr, x0, y0, tile)
            if not valid.any():
                continue
            img = stretch_with_bounds(data, lows, highs, valid)  # (3, tile, tile) BGR
            hwc = np.ascontiguousarray(np.transpose(img, (1, 2, 0)))
            valid_sr = valid
            if sr != 1:
                hwc = cv2.resize(
                    hwc, (MODEL_INPUT_PX, MODEL_INPUT_PX), interpolation=cv2.INTER_CUBIC
                )
                valid_sr = cv2.resize(
                    valid.astype(np.uint8),
                    (MODEL_INPUT_PX, MODEL_INPUT_PX),
                    interpolation=cv2.INTER_NEAREST_EXACT,
                ).astype(bool)
            batch.append((x0, y0, hwc, valid_sr))
            stats["n_tiles_processed"] += 1
            if len(batch) == opts.batch_size:
                run_batch(batch)
                batch = []
    if batch:
        run_batch(batch)
    return rows, stats


def _run_native(
    raster_path: str, config: AgriboundConfig, opts: DAOptions, rgb: list[int]
) -> gpd.GeoDataFrame:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics is required for the native Delineate-Anything backend. Install with: "
            "pip install 'agribound[delineate-anything]'"
        ) from None
    import rasterio

    from agribound._cache import cache_path
    from agribound.registry import source_value_scale

    device = config.resolve_device()
    weights, weights_meta = _weights_for(opts)
    bands_bgr = list(reversed(rgb))
    with rasterio.open(raster_path) as src:
        gsd = pixel_size_m(src.crs, src.transform, src.height, src.width)
    # The raster itself, or its copy at processing_gsd_m; everything below reads it.
    read_path, read_bands, processing = processing_raster(
        raster_path, config, opts.processing_gsd_m, bands_bgr
    )
    with rasterio.open(read_path) as src:
        read_gsd = pixel_size_m(src.crs, src.transform, src.height, src.width)
        sr = select_super_resolution(read_gsd, opts.super_resolution)
        lows, highs = scene_stretch_bounds(src, read_bands, opts.stretch_percentiles)
        unstretched = src.dtypes[read_bands[0] - 1] == "uint8"
        crs = src.crs
        # The model-input pixel grid, on which the tiles and their edges lie.
        model_transform = src.transform @ rasterio.Affine.scale(1.0 / sr)
        model_shape = (src.height * sr, src.width * sr)
    if unstretched and opts.stretch_percentiles != DEFAULT_STRETCH_PERCENTILES:
        logger.warning(
            "stretch_percentiles=%s has no effect: uint8 rasters are used without a stretch.",
            list(opts.stretch_percentiles),
        )

    precision = "fp16" if (device != "cpu" and opts.half) else "fp32"
    meta: dict[str, Any] = {
        "backend": "native",
        "implementation": f"agribound-native-{_NATIVE_IMPL_VERSION}",
        "model_key": opts.model_key,
        "model_architecture": opts.model.architecture,
        "model_training_data": opts.model.training_data,
        **weights_meta,
        "conf_threshold": opts.conf_threshold,
        "conf_threshold_source": opts.conf_source,
        "iou_threshold": opts.iou_threshold,
        "max_detections": opts.max_detections,
        "gsd_m": round(float(gsd), 4),
        "processing_gsd_m": opts.processing_gsd_m,
        "processing": processing,
        "super_resolution": sr,
        "tile_size_native_px": MODEL_INPUT_PX // sr,
        "model_input_px": MODEL_INPUT_PX,
        "tile_step": opts.tile_step,
        "batch_size": opts.batch_size,
        "precision": precision,
        "device": device,
        "band_indices_bgr": bands_bgr,
        "value_scale": source_value_scale(config.source),
        "stretch_percentiles": list(opts.stretch_percentiles),
        "stretch_lows_bgr": lows,
        "stretch_highs_bgr": highs,
        "dedup_iou": opts.dedup_iou,
        "dedup_containment": opts.dedup_containment,
        "merge_tile_pieces": opts.merge_tile_pieces,
        "resolve_overlaps": opts.resolve_overlaps,
        "merge_tile_seams": opts.merge_tile_seams,
        "tile_edge_strip_px": TILE_EDGE_STRIP_PX,
        "tile_seam_params": dict(TILE_SEAM_PARAMS) if opts.merge_tile_seams else None,
        "min_field_area_m2": float(config.min_field_area_m2),
        "area_crs": "EPSG:6933",
        "ultralytics_version": _package_version("ultralytics"),
    }
    _record_training_gsd(read_gsd, meta)
    _check_checkpoint_input(
        weights_meta,
        meta,
        {
            "gsd_m": read_gsd,
            "super_resolution": sr,
            "stretch_percentiles": None if unstretched else opts.stretch_percentiles,
        },
    )
    # The options enter the key only when set, so other runs keep their cached results.
    options_key = []
    if opts.processing_gsd_m is not None:
        options_key.append(f"processing_gsd_m={opts.processing_gsd_m!r}")
    if processing is not None:  # results made from an older copy are not reused
        options_key.append(f"processing_raster={_PROCESSING_RASTER_VERSION}")
    if opts.stretch_percentiles != DEFAULT_STRETCH_PERCENTILES:
        options_key.append(f"stretch_percentiles={list(opts.stretch_percentiles)!r}")
    out_path = cache_path(
        config,
        "da_native",
        ".parquet",
        _raster_fingerprint(raster_path),
        _NATIVE_IMPL_VERSION,
        weights_meta.get("weights_sha256") or weights_meta.get("checkpoint_sha256"),
        opts.model_key,
        opts.conf_threshold,
        opts.iou_threshold,
        opts.max_detections,
        sr,
        opts.tile_step,
        opts.dedup_iou,
        opts.dedup_containment,
        opts.merge_tile_pieces,
        opts.resolve_overlaps,
        opts.merge_tile_seams,
        float(config.min_field_area_m2),
        bands_bgr,
        precision,
        device,
        # Model outputs (mask format, NMS, FP16 path) can change between releases.
        meta["ultralytics_version"],
        *options_key,
    )
    if out_path.exists():
        gdf = gpd.read_parquet(out_path)
        logger.info("Using cached native DA result (%d polygons): %s", len(gdf), out_path)
        meta["cached_result"] = str(out_path)
        meta["n_output"] = len(gdf)
        gdf.attrs["engine_meta"] = meta
        return gdf

    logger.info(
        "Delineate-Anything native: model=%s conf=%.3g GSD=%.2f m super_resolution=%d device=%s",
        opts.model_key,
        opts.conf_threshold,
        read_gsd,
        sr,
        device,
    )
    if processing is not None:
        logger.info(
            "Delineate-Anything native: the %.2f m raster is read at %.2f m; tiles of %d px "
            "cover %.0f m",
            gsd,
            read_gsd,
            MODEL_INPUT_PX // sr,
            MODEL_INPUT_PX // sr * read_gsd,
        )
    t0 = time.perf_counter()
    model = YOLO(weights)
    with rasterio.open(read_path) as src:
        rows, stats = _native_detections(model, src, read_bands, lows, highs, sr, opts, device)
    if rows:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    else:
        gdf = gpd.GeoDataFrame(
            {"confidence": [], "_tile_edge": [], "_tile_x0": [], "_tile_y0": []},
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )
    raw = gdf  # every tile's own detections: the centre-tile views of the seam test
    n_raw = len(gdf)
    if n_raw:
        gdf = gdf[_area_m2(gdf) >= float(config.min_field_area_m2)]
    n_area = len(gdf)
    n_merged_groups = n_pieces_merged = 0
    if opts.merge_tile_pieces and n_area > 1:
        gdf = merge_tile_pieces(gdf, opts.dedup_iou, opts.dedup_containment)
        merged = gdf["_n_pieces"].to_numpy() > 1
        n_merged_groups = int(merged.sum())
        n_pieces_merged = int(gdf["_n_pieces"].to_numpy()[merged].sum())
    n_merge = len(gdf)
    gdf = deduplicate_detections(gdf, opts.dedup_iou, opts.dedup_containment)
    n_dedup = len(gdf)
    n_seam_joins = 0
    if opts.resolve_overlaps and n_dedup > 1:
        gdf = resolve_overlaps(gdf)
        if opts.merge_tile_seams and len(gdf) > 1:
            with rasterio.open(read_path) as src:
                gdf, n_seam_joins = merge_tile_seams(
                    gdf,
                    raw,
                    model_transform,
                    model_shape,
                    _model_input_reader(src, read_bands, lows, highs, sr),
                )
        if len(gdf):
            gdf = gdf[_area_m2(gdf) >= float(config.min_field_area_m2)]
    gdf = gdf.drop(
        columns=["_tile_edge", "_n_pieces", "_tile_x0", "_tile_y0"], errors="ignore"
    ).reset_index(drop=True)
    meta.update(
        {
            **stats,
            "n_after_min_area": n_area,
            "n_merged_groups": n_merged_groups,
            "n_pieces_merged": n_pieces_merged,
            "n_after_merge": n_merge,
            "n_after_dedup": n_dedup,
            "n_seam_joins": n_seam_joins,
            "n_output": len(gdf),
            "inference_s": round(time.perf_counter() - t0, 2),
        }
    )
    tmp = out_path.with_suffix(".parquet.tmp")
    gdf.to_parquet(tmp)
    os.replace(tmp, out_path)
    logger.info(
        "Delineate-Anything native: %d detections -> %d after area filter -> %d after "
        "merging %d tile-cut pieces into %d fields -> %d after de-duplication -> %d tile-seam "
        "joins -> %d output (%d tiles, %.1f s)",
        n_raw,
        n_area,
        n_merge,
        n_pieces_merged,
        n_merged_groups,
        n_dedup,
        n_seam_joins,
        len(gdf),
        stats["n_tiles_processed"],
        meta["inference_s"],
    )
    gdf.attrs["engine_meta"] = meta
    return gdf


# ---------------------------------------------------------------------------
# Reference backend (upstream DelAnyFlow in a subprocess)
# ---------------------------------------------------------------------------

#: Upstream ``conf_sample.yaml`` (Lavreniuk/Delineate-Anything a6f30b2) with
#: agribound's fixed choices: no LCLU mask (so no mask classes), absolute
#: area filters, and DA's own simplification disabled (agribound simplifies
#: later in metres).
_REFERENCE_CONFIG: dict[str, Any] = {
    "model": ["large_v2"],
    "method": "main",
    "execution_args": {
        "src_folder": "",
        "temp_folder": "",
        "output_path": "",
        "keep_temp": False,
        "mask_filepath": None,
    },
    "super_resolution": None,
    "treat_as_vrt": False,
    "mask_info": {"range": 40, "filter_classes": [], "clip_classes": []},
    "background_info": {"background_classes_from_mask": [], "additional_source": None},
    "data_loader": {
        "skip": False,
        "bands": [3, 2, 1],
        "nodata_band": None,
        "nodata_value": [0, 0, 0],
        "min": None,
        "max": None,
    },
    "execution_planner": {"region_width": -1, "region_height": -1, "pixel_offset": [0, 0]},
    "postprocess_limits": {"num_workers": -1, "queue_tiles_capacity": 4, "max_tiles_inflight": 8},
    "passes": [
        {
            "batch_size": -1,
            "tile_size": None,
            "tile_step": 0.5,
            "model_args": [{"name": "large_v2", "minimal_confidence": 0.15, "use_half": True}],
            "delineation_config": {
                "pixel_area_threshold": 512,
                "remaining_area_threshold": 0.8,
                "compose_merge_iou": 0.8,
                "merge_iou": 0.8,
                "merge_relative_area_threshold": 0.5,
                "merge_asymetric_pixel_area_threshold": 32,
                "merge_asymetric_relative_area_threshold": 0.7,
                "merging_edge_width": 4,
                "merge_edge_iou": 0.6,
                "merge_edge_pixels": 192,
            },
        }
    ],
    "polygonization_args": {"layer_name": "fields", "override_if_exists": True},
    "filtering_args": {
        "automatic_area_scale": False,
        "minimum_area_m2": 2500,
        "minimum_part_area_m2": 0,
        "minimum_hole_area_m2": 2500,
        "minimum_background_field_area_m2": 50000,
        "minimum_background_field_hole_area_m2": 25000,
        "middleground_offset": None,
        "minimum_middleground_field_area_m2": 10000,
        "minimum_middleground_field_hole_area_m2": 5000,
    },
    "simplification_args": {
        "simplify": False,
        "epsilon_scale": 2,
        "num_workers": -1,
        "raster_resolution": -1,
    },
}


#: Third-party modules imported by upstream ``methods/main`` and ``simplification``
#: (Lavreniuk/Delineate-Anything a6f30b2).
_REFERENCE_IMPORTS = (
    "osgeo",
    "numba",
    "cv2",
    "psutil",
    "scipy",
    "tqdm",
    "affine",
    "ultralytics",
)
_PIP_NAMES = {"cv2": "opencv-python"}


def check_reference_repo(repo: str | Path) -> Path:
    """Validate a Delineate-Anything checkout for the reference backend.

    Raises
    ------
    FileNotFoundError
        If ``methods/main/inference.py`` is missing.
    RuntimeError
        If the checkout predates the uint8-mask fix (upstream 34eddf7).
    """
    root = Path(repo).expanduser().resolve()
    inference = root / "methods" / "main" / "inference.py"
    if not inference.is_file():
        raise FileNotFoundError(
            f"{root} is not a Delineate-Anything checkout (missing methods/main/inference.py). "
            "Clone https://github.com/Lavreniuk/Delineate-Anything and set AGRIBOUND_DA_REPO "
            "or engine_params['da_repo']."
        )
    if _UINT8_FIX_MARKER not in inference.read_text(errors="replace"):
        raise RuntimeError(
            f"The Delineate-Anything checkout {root} predates upstream commit 34eddf7: its "
            "mask morphology runs on uint8 masks, which Ultralytics >= 8.3.217 returns, and "
            "turns the erosion into a dilation. Update the checkout (git pull)."
        )
    return root


def _repo_revision(root: Path) -> str:
    """Git commit of *root*, or the SHA-1 of its inference.py when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return out.stdout.strip()
    except Exception:
        digest = hashlib.sha1((root / "methods" / "main" / "inference.py").read_bytes())
        return f"inference.py-sha1:{digest.hexdigest()}"


def build_reference_config(
    opts: DAOptions,
    bands_bgr: list[int],
    min_area_m2: float,
    src_folder: str,
    temp_folder: str,
    output_path: str,
) -> dict[str, Any]:
    """Return the upstream DA configuration for one reference run.

    Sets ``config["model"] = [key]`` and the first pass's
    ``model_args[0].name`` to the same key (they must match, or upstream
    raises ``KeyError``), ``pixel_offset = [0, 0]``, the per-model
    confidence, absolute area thresholds (``minimum_area_m2`` =
    *min_area_m2*; ``minimum_hole_area_m2`` = ``opts.min_hole_area_m2``,
    default 2500 m² as in the upstream sample configuration) and the
    explicit batch size, tile step and super-resolution.
    """
    cfg = copy.deepcopy(_REFERENCE_CONFIG)
    key = opts.model_key
    cfg["model"] = [key]
    cfg["passes"][0]["model_args"][0].update(
        {"name": key, "minimal_confidence": opts.conf_threshold, "use_half": opts.half}
    )
    cfg["passes"][0]["batch_size"] = opts.batch_size
    cfg["passes"][0]["tile_step"] = opts.tile_step
    cfg["super_resolution"] = opts.super_resolution
    cfg["execution_planner"]["pixel_offset"] = [0, 0]
    cfg["data_loader"]["bands"] = list(bands_bgr)
    cfg["filtering_args"]["automatic_area_scale"] = False
    cfg["filtering_args"]["minimum_area_m2"] = float(min_area_m2)
    cfg["filtering_args"]["minimum_hole_area_m2"] = float(opts.min_hole_area_m2)
    cfg["execution_args"].update(
        {"src_folder": src_folder, "temp_folder": temp_folder, "output_path": output_path}
    )
    return cfg


def _run_reference(
    raster_path: str, config: AgriboundConfig, opts: DAOptions, rgb: list[int]
) -> gpd.GeoDataFrame:
    import shutil

    import rasterio

    from agribound._cache import cache_path

    repo_value = opts.da_repo or os.environ.get("AGRIBOUND_DA_REPO")
    if not repo_value:
        raise RuntimeError(
            "backend='reference' needs a Delineate-Anything checkout: set the AGRIBOUND_DA_REPO "
            "environment variable or engine_params['da_repo'] to a clone of "
            "https://github.com/Lavreniuk/Delineate-Anything (upstream 34eddf7 or later)."
        )
    repo = check_reference_repo(repo_value)
    missing = [m for m in _REFERENCE_IMPORTS if importlib.util.find_spec(m) is None]
    if missing:
        hints = {"osgeo": "conda install -c conda-forge gdal (matching the installed libgdal)"}
        raise ImportError(
            "backend='reference' runs the upstream Delineate-Anything code, which imports "
            f"{', '.join(missing)} (not installed). Install: "
            + "; ".join(hints.get(m, f"pip install {_PIP_NAMES.get(m, m)}") for m in missing)
            + ". Or use backend='native'."
        )
    device = config.resolve_device()
    weights, weights_meta = _weights_for(opts)
    bands_bgr = list(reversed(rgb))
    with rasterio.open(raster_path) as src:
        gsd = pixel_size_m(src.crs, src.transform, src.height, src.width)
    sr = select_super_resolution(gsd, opts.super_resolution)
    revision = _repo_revision(repo)

    run_dir = cache_path(
        config,
        "da_reference",
        "",
        _raster_fingerprint(raster_path),
        revision,
        weights_meta.get("weights_sha256") or weights_meta.get("checkpoint_sha256"),
        opts.model_key,
        opts.conf_threshold,
        opts.super_resolution,
        opts.tile_step,
        float(config.min_field_area_m2),
        opts.min_hole_area_m2,
        bands_bgr,
        opts.half,
        device,
        _package_version("ultralytics"),
    )
    output = run_dir / "output.gpkg"
    done = run_dir / "done.json"
    meta: dict[str, Any] = {
        "backend": "reference",
        "da_repo": str(repo),
        "da_repo_revision": revision,
        "model_key": opts.model_key,
        "model_architecture": opts.model.architecture,
        "model_training_data": opts.model.training_data,
        **weights_meta,
        "conf_threshold": opts.conf_threshold,
        "conf_threshold_source": opts.conf_source,
        "nms_iou": "ultralytics default (upstream execute() passes no iou)",
        "gsd_m": round(float(gsd), 4),
        "super_resolution": sr,
        "tile_size_native_px": MODEL_INPUT_PX // sr,
        "tile_step": opts.tile_step,
        "batch_size": opts.batch_size,
        "pixel_offset": [0, 0],
        "band_indices_bgr": bands_bgr,
        "device": device,
        "use_half": opts.half and device != "cpu",
        "min_field_area_m2": float(config.min_field_area_m2),
        "min_hole_area_m2": opts.min_hole_area_m2,
        "ultralytics_version": _package_version("ultralytics"),
        "run_dir": str(run_dir),
    }
    _record_training_gsd(gsd, meta)
    if not (done.exists() and output.exists()):
        # Fresh, private input folder: upstream processes every .tif in src_folder.
        if run_dir.exists():
            shutil.rmtree(run_dir)
        (run_dir / "input").mkdir(parents=True)
        link = run_dir / "input" / Path(raster_path).name
        try:
            link.symlink_to(Path(raster_path).resolve())
        except OSError:
            shutil.copy2(raster_path, link)
        cfg = build_reference_config(
            opts,
            bands_bgr,
            config.min_field_area_m2,
            str(run_dir / "input"),
            str(run_dir / "temp"),
            str(output),
        )
        spec = {
            "repo": str(repo),
            "model_paths": [weights],
            "config": cfg,
            "device": device,
            "done_marker": str(done),
        }
        spec_path = run_dir / "run_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        env = dict(os.environ)
        if device in ("cpu", "mps"):
            env["CUDA_VISIBLE_DEVICES"] = ""
        log_path = run_dir / "run.log"
        logger.info(
            "Running upstream Delineate-Anything (%s @ %s) in a subprocess; log: %s",
            repo,
            revision[:12],
            log_path,
        )
        t0 = time.perf_counter()
        with open(log_path, "w") as log:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "agribound.engines.delineate_anything",
                    "--reference-run",
                    str(spec_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                check=False,
            )
        meta["inference_s"] = round(time.perf_counter() - t0, 2)
        if proc.returncode != 0 or not output.exists():
            tail = log_path.read_text(errors="replace")[-3000:]
            raise RuntimeError(
                f"Upstream Delineate-Anything failed (exit code {proc.returncode}); see "
                f"{log_path}. Last output:\n{tail}"
            )
    else:
        meta["cached_result"] = str(output)
    gdf = gpd.read_file(output, layer="fields")
    meta["n_output"] = len(gdf)
    logger.info("Delineate-Anything reference: %d field polygons", len(gdf))
    gdf.attrs["engine_meta"] = meta
    return gdf


def _reference_child_main(spec_path: str) -> int:
    """Subprocess entry point: run upstream ``execute`` for one run spec."""
    spec = json.loads(Path(spec_path).read_text())
    repo = Path(spec["repo"]).resolve()
    device = spec["device"]
    import torch

    if device == "cpu":
        # Upstream execute() picks cuda > mps > cpu by itself; hide MPS (CUDA is
        # hidden with CUDA_VISIBLE_DEVICES by the parent) so it runs on the CPU.
        torch.backends.mps.is_available = lambda: False
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("device='mps' was requested but MPS is not available")
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' was requested but CUDA is not available")
    sys.path.insert(0, str(repo))
    inference = importlib.import_module("methods.main.inference")
    module_file = Path(inference.__file__).resolve()
    if not module_file.is_relative_to(repo):
        raise RuntimeError(
            f"'methods.main.inference' was imported from {module_file}, not from {repo}; "
            "another package named 'methods' shadows the Delineate-Anything checkout."
        )
    inference.execute(spec["model_paths"], spec["config"], False)
    Path(spec["done_marker"]).write_text(json.dumps({"status": "ok"}))
    return 0


# ---------------------------------------------------------------------------
# FTW backend
# ---------------------------------------------------------------------------


def _ftw_checkpoint_url(ftw_name: str) -> str | None:
    """Checkpoint URL that ftw-tools' DelineateAnything wrapper loads for *ftw_name*."""
    try:
        from ftw_tools.inference.models import DelineateAnything
    except ImportError:
        return None
    return getattr(DelineateAnything, "checkpoints", {}).get(ftw_name)


def _run_ftw(
    raster_path: str, config: AgriboundConfig, opts: DAOptions, rgb: list[int]
) -> gpd.GeoDataFrame:
    import inspect

    try:
        from ftw_tools.inference.inference import run_instance_segmentation
        from ftw_tools.inference.model_registry import MODEL_REGISTRY
    except ImportError:
        raise ImportError(
            "ftw-tools is required for backend='ftw'. Install with: pip install 'agribound[ftw]'"
        ) from None
    import rasterio

    from agribound._cache import cache_path
    from agribound.registry import source_value_scale

    ftw_name = opts.model.ftw_name
    if ftw_name not in MODEL_REGISTRY:
        raise RuntimeError(
            f"The installed ftw-tools ({_package_version('ftw-tools')}) has no {ftw_name!r} in "
            "its MODEL_REGISTRY. DelineateAnythingV2 needs ftw-baselines main at fa86d4a or "
            "later (pip install 'ftw-tools @ git+https://github.com/fieldsoftheworld/"
            "ftw-baselines'); or use backend='native'."
        )
    value_scale = opts.ftw.get("value_scale") or source_value_scale(config.source)
    if value_scale != "reflectance_x10000":
        raise ValueError(
            f"backend='ftw' divides the input by 3000 (Sentinel-2 L2A units) and needs a "
            f"reflectance x 10000 composite; source {config.source!r} has value scale "
            f"{value_scale!r}. Use backend='native' (scene-level percentile stretch) instead"
            + (
                ", or set engine_params['value_scale']='reflectance_x10000' for a local file "
                "on that scale."
                if config.source == "local"
                else "."
            )
        )
    from agribound.engines.ftw import FTW_INPUT_VERSION, write_ftw_input

    device = config.resolve_device()
    ftw_version = _package_version("ftw-tools")
    rgb_path = cache_path(
        config,
        "da_ftw_rgb",
        ".tif",
        _raster_fingerprint(raster_path),
        rgb,
        value_scale,
        FTW_INPUT_VERSION,
    )
    if not rgb_path.exists():
        # float32 S2 L2A units (to_s2_dn), NaN/inf/nodata -> 0, written in strips.
        write_ftw_input(rgb_path, [(raster_path, rgb)], config.source, value_scale=value_scale)

    f = opts.ftw
    with rasterio.open(rgb_path) as src:
        gsd = pixel_size_m(src.crs, src.transform, src.height, src.width)
        if f["patch_size"] >= min(src.height, src.width):
            # torchgeo 0.10's GridGeoSampler can yield no patch when the patch spans the
            # raster (see agribound.engines.ftw.select_patch_size); ftw-tools then fails.
            raise ValueError(
                f"backend='ftw' needs patch_size ({f['patch_size']}) smaller than the raster's "
                f"smaller side ({min(src.height, src.width)} px); set engine_params['patch_size']."
            )
    kwargs: dict[str, Any] = {
        "input": str(rgb_path),
        "model": ftw_name,
        "gpu": 0 if device == "cuda" else None,
        "num_workers": config.n_workers,
        "patch_size": f["patch_size"],
        "resize_factor": f["resize_factor"],
        "batch_size": opts.batch_size,
        "max_detections": opts.max_detections,
        "iou_threshold": opts.iou_threshold,
        "conf_threshold": opts.conf_threshold,
        "padding": f["padding"],
        "overwrite": True,
        "mps_mode": device == "mps",
        "simplify": f["simplify"],
        "min_size": float(config.min_field_area_m2),
        "max_size": f["max_size"],
        "close_interiors": f["close_interiors"],
        "overlap_iou_threshold": f["overlap_iou_threshold"],
        "overlap_contain_threshold": f["overlap_contain_threshold"],
    }
    params = inspect.signature(run_instance_segmentation).parameters
    if "nan_fill_value" in params:
        kwargs["nan_fill_value"] = 0.0
    out = cache_path(
        config,
        "da_ftw",
        ".gpkg",
        _raster_fingerprint(rgb_path),
        json.dumps(
            {k: v for k, v in kwargs.items() if k not in ("input", "num_workers")},
            sort_keys=True,
            default=str,
        ),
        device,
        ftw_version,
    )
    meta: dict[str, Any] = {
        "backend": "ftw",
        "ftw_tools_version": ftw_version,
        "model_key": opts.model_key,
        "ftw_model": ftw_name,
        "weights_url": _ftw_checkpoint_url(ftw_name),
        "conf_threshold": opts.conf_threshold,
        "conf_threshold_source": opts.conf_source,
        "iou_threshold": opts.iou_threshold,
        "max_detections": opts.max_detections,
        "patch_size": f["patch_size"],
        "resize_factor": f["resize_factor"],
        "input_units": "S2 L2A reflectance x10000 (composite values; NaN, inf and nodata -> 0)",
        "preprocessing": "ftw-tools: first 3 bands / 3000, clip [0, 1], bilinear resize",
        "min_field_area_m2": float(config.min_field_area_m2),
        "device": device,
        "gsd_m": round(float(gsd), 4),
    }
    _record_training_gsd(gsd, meta)
    if not out.exists():
        logger.info("Running Delineate-Anything via ftw-tools (model=%s)", ftw_name)
        # Written under a temporary name and renamed, so an interrupted run
        # never leaves a partial file that a later run would take as cached.
        partial = out.with_name(out.stem + ".partial.gpkg")
        try:
            with no_nms_time_limit():  # ftw-tools predicts with Ultralytics as well
                run_instance_segmentation(out=str(partial), **kwargs)
        except ValueError as exc:
            # ftw-tools concatenates the per-patch results with pd.concat, which
            # raises this when no patch produced a detection.
            if "No objects to concatenate" not in str(exc):
                raise
            logger.warning("ftw-tools produced no detections for %s", raster_path)
            with rasterio.open(raster_path) as src:
                crs = src.crs
            gdf = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)
            meta["n_output"] = 0
            gdf.attrs["engine_meta"] = meta
            return gdf
        if not partial.exists():
            raise RuntimeError(f"ftw-tools instance segmentation produced no output at {partial}")
        os.replace(partial, out)
    else:
        meta["cached_result"] = str(out)
    gdf = gpd.read_file(out)
    meta["n_output"] = len(gdf)
    gdf.attrs["engine_meta"] = meta
    return gdf


if __name__ == "__main__":  # pragma: no cover - exercised through the reference backend
    if len(sys.argv) == 3 and sys.argv[1] == "--reference-run":
        sys.exit(_reference_child_main(sys.argv[2]))
    sys.exit("usage: python -m agribound.engines.delineate_anything --reference-run SPEC.json")
