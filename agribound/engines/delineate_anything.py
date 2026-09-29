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
    to uint8, computed on valid, strictly positive pixels sampled
    (nearest neighbour, full-resolution data, never overviews) on a grid of
    at most 4096 px per side (uint8 rasters are used unchanged); tiles of 512
    native pixels when the ground sampling distance (GSD) is below 4 m, else
    256 native pixels upsampled 2x with bicubic interpolation, so the model
    input is always 512 x 512 (``super_resolution`` = 1, 2 or 4 overrides the
    factor); 50 % tile overlap starting half a tile before the raster origin,
    as the upstream ``ExecutionPlanner`` does; BGR channel order for NumPy
    input to Ultralytics; ``retina_masks=True``; masks cast to float before
    the upstream 3 x 3 erode / dilate / dilate / erode morphology (Ultralytics
    >= 8.3.217 returns uint8 masks, on which the upstream negation trick would
    turn the erosion into a dilation); FP16 on GPU/MPS. Each detection becomes
    the largest polygon of its mask, clipped to valid pixels, and is flagged
    when it touches an interior tile edge (a *tile-cut* detection). The
    detections of all tiles are then combined at polygon level, with
    duplicates defined as IoU >= ``dedup_iou`` or intersection >=
    ``dedup_containment`` of the smaller polygon: (1) tile-cut duplicates of
    one another are merged into their union (:func:`merge_tile_pieces`;
    ``merge_tile_pieces=False`` skips it), so a field too large to be
    complete in any tile is rebuilt from its pieces (the polygon-level
    counterpart of DelAnyFlow's merging of fields that touch a tile border);
    pieces that duplicate a complete (not tile-cut), at least as large
    detection are not merged; (2) greedy non-maximum suppression visits the
    polygons by higher confidence, then larger area, and drops every polygon
    that duplicates one already kept, except that a complete detection is
    visited before each tile-cut duplicate that is not larger than it
    (:func:`deduplicate_detections`); (3) remaining overlaps go to the
    polygon visited first in that order (:func:`resolve_overlaps`;
    ``resolve_overlaps=False`` keeps them). This is simpler than DelAnyFlow's
    raster-level region merging (which, for example, lets smaller fields
    carve their area out of larger ones), so results are close to, but not
    identical with, the ``"reference"`` backend. NMS uses IoU 0.3 by
    default, the value of the authors' openEO UDP
    (``openeo_udp/udf/delineate_onnx.py``); the upstream ``execute()`` keeps
    Ultralytics' default (0.7). The raster is read tile by tile.
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
  ``max_detections`` (per tile, 300).
- ``native``: ``dedup_iou`` (0.3), ``dedup_containment`` (0.8),
  ``merge_tile_pieces`` (True) and ``resolve_overlaps`` (True).
- ``reference``: ``da_repo``; ``min_hole_area_m2`` (holes smaller than this
  are filled, 2500 m², the upstream ``conf_sample.yaml`` value).
- ``ftw``: ``patch_size`` (256; a multiple of 32 smaller than the raster's
  smaller side), ``resize_factor`` (2), ``padding`` (FTW
  default), ``close_interiors`` (True), ``simplify`` (FTW simplification
  tolerance, applied in EPSG:6933 coordinates, i.e. metres that are exact
  only near 30° latitude; 0), ``max_size`` (m², None),
  ``overlap_iou_threshold`` (0.3), ``overlap_contain_threshold`` (0.8),
  ``value_scale`` (for ``local`` rasters).

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
(:func:`gsd_outside_training_range`).

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

import copy
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
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
#: tile-cut duplicates, EPSG:6933 areas, stretch bounds never from overviews).
_NATIVE_IMPL_VERSION = "3"

#: Upstream ``conf_sample.yaml`` hole threshold of the reference pipeline (m²).
DEFAULT_MIN_HOLE_AREA_M2 = 2500.0

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
    da_repo: str | None
    min_hole_area_m2: float = DEFAULT_MIN_HOLE_AREA_M2
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
        hole = params.get("min_hole_area_m2", DEFAULT_MIN_HOLE_AREA_M2)
        if isinstance(hole, bool) or not isinstance(hole, int | float) or hole < 0:
            raise ValueError(f"min_hole_area_m2 must be a number >= 0, got {hole!r}")

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
            da_repo=params.get("da_repo"),
            min_hole_area_m2=float(hole),
            ftw=ftw_opts,
        )


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
    True. A tile-cut detection that duplicates a complete detection at least
    as large as itself is a partial view of a field detected whole in
    another tile: it is not merged and is left for
    :func:`deduplicate_detections`. Complete detections are unchanged.

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
        Kept rows, in the original order.
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
    iou_threshold: float = DEFAULT_DEDUP_IOU,
    containment_threshold: float = DEFAULT_DEDUP_CONTAINMENT,
    *,
    score_column: str = "confidence",
    edge_column: str | None = "_tile_edge",
) -> gpd.GeoDataFrame:
    """Make detections non-overlapping by giving contested area to the one visited first.

    Polygons are visited in the order of :func:`deduplicate_detections`
    (with the same duplicate criteria; after de-duplication this is simply
    higher score, then larger area); each polygon loses the parts that
    overlap polygons visited before it and keeps its largest remaining part
    (empty results are dropped). This mimics the per-pixel assignment of
    DelAnyFlow, whose output fields do not overlap.

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
    left, right = _duplicate_pairs(original, areas, iou_threshold, containment_threshold)
    tree = shapely.STRtree(original)
    placed = np.zeros(n, dtype=bool)
    result = np.empty(n, dtype=object)
    for idx in _visit_order(areas, scores, edges, left, right).tolist():
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
        rgb = get_canonical_band_indices(config.source, ["R", "G", "B"], bands=config.bands)
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


def _read_tile(
    src: Any, bands: list[int], x0: int, y0: int, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Read a ``size`` x ``size`` window (zero-padded outside the raster).

    Returns ``(data float32 (bands, size, size), valid bool (size, size))``.
    A pixel is invalid if any band is non-finite, all bands are 0 (the
    upstream default ``nodata_value: [0, 0, 0]``), or all bands equal the
    raster's nodata value.
    """
    from rasterio.windows import Window

    data = np.zeros((len(bands), size, size), dtype=np.float32)
    valid = np.zeros((size, size), dtype=bool)
    c0, r0 = max(x0, 0), max(y0, 0)
    c1, r1 = min(x0 + size, src.width), min(y0 + size, src.height)
    if c1 <= c0 or r1 <= r0:
        return data, valid
    arr = src.read(bands, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    ok = np.all(np.isfinite(arr), axis=0) & ~np.all(arr == 0, axis=0)
    nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        ok &= ~np.all(arr == nodata, axis=0)
    data[:, r0 - y0 : r1 - y0, c0 - x0 : c1 - x0] = np.where(np.isfinite(arr), arr, 0)
    valid[r0 - y0 : r1 - y0, c0 - x0 : c1 - x0] = ok
    return data, valid


def scene_stretch_bounds(src: Any, bands: list[int]) -> tuple[list[float], list[float]]:
    """Scene-level 1-99 % stretch bounds, sampled as upstream ``DataAnalyser`` does.

    uint8 rasters get fixed bounds 0/255. Otherwise the bands are read on a
    nearest-neighbour grid of at most 4096 px per side (GDAL/rasterio
    decimated read of the full-resolution data: a raster with overviews is
    reopened with ``OVERVIEW_LEVEL=NONE``, as upstream does, because GDAL
    would otherwise sample the (usually averaged) overviews, which narrows
    the percentile range), pixels with all bands 0, equal to nodata or
    non-finite are excluded, and
    :func:`agribound.io.raster.percentile_stretch_uint8` computes the
    percentiles of the strictly positive values.
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
    lows, highs = percentile_stretch_uint8(sample, return_bounds_only=True)
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
    """Tile the raster, run the model and polygonise every detection."""
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
                poly = _mask_polygon(
                    crop, tile_tf @ rasterio.transform.Affine.translation(cx0, cy0)
                )
                if poly is None or poly.is_empty:
                    continue
                edge = (
                    (interior["left"] and cx0 == 0 and bool(crop[:, 0].any()))
                    or (interior["right"] and cx1 == size and bool(crop[:, -1].any()))
                    or (interior["top"] and cy0 == 0 and bool(crop[0, :].any()))
                    or (interior["bottom"] and cy1 == size and bool(crop[-1, :].any()))
                )
                rows.append({"geometry": poly, "confidence": float(confs[k]), "_tile_edge": edge})
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
        sr = select_super_resolution(gsd, opts.super_resolution)
        lows, highs = scene_stretch_bounds(src, bands_bgr)
        crs = src.crs

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
        "super_resolution": sr,
        "tile_size_native_px": MODEL_INPUT_PX // sr,
        "model_input_px": MODEL_INPUT_PX,
        "tile_step": opts.tile_step,
        "batch_size": opts.batch_size,
        "precision": precision,
        "device": device,
        "band_indices_bgr": bands_bgr,
        "value_scale": source_value_scale(config.source),
        "stretch_lows_bgr": lows,
        "stretch_highs_bgr": highs,
        "dedup_iou": opts.dedup_iou,
        "dedup_containment": opts.dedup_containment,
        "merge_tile_pieces": opts.merge_tile_pieces,
        "resolve_overlaps": opts.resolve_overlaps,
        "min_field_area_m2": float(config.min_field_area_m2),
        "area_crs": "EPSG:6933",
        "ultralytics_version": _package_version("ultralytics"),
    }
    _record_training_gsd(gsd, meta)
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
        float(config.min_field_area_m2),
        bands_bgr,
        precision,
        device,
        # Model outputs (mask format, NMS, FP16 path) can change between releases.
        meta["ultralytics_version"],
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
        gsd,
        sr,
        device,
    )
    t0 = time.perf_counter()
    model = YOLO(weights)
    with rasterio.open(raster_path) as src:
        rows, stats = _native_detections(model, src, bands_bgr, lows, highs, sr, opts, device)
    if rows:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    else:
        gdf = gpd.GeoDataFrame(
            {"confidence": [], "_tile_edge": []}, geometry=gpd.GeoSeries([], crs=crs), crs=crs
        )
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
    if opts.resolve_overlaps and n_dedup > 1:
        gdf = resolve_overlaps(gdf, opts.dedup_iou, opts.dedup_containment)
        if len(gdf):
            gdf = gdf[_area_m2(gdf) >= float(config.min_field_area_m2)]
    gdf = gdf.drop(columns=["_tile_edge", "_n_pieces"], errors="ignore").reset_index(drop=True)
    meta.update(
        {
            **stats,
            "n_after_min_area": n_area,
            "n_merged_groups": n_merged_groups,
            "n_pieces_merged": n_pieces_merged,
            "n_after_merge": n_merge,
            "n_after_dedup": n_dedup,
            "n_output": len(gdf),
            "inference_s": round(time.perf_counter() - t0, 2),
        }
    )
    tmp = out_path.with_suffix(".parquet.tmp")
    gdf.to_parquet(tmp)
    os.replace(tmp, out_path)
    logger.info(
        "Delineate-Anything native: %d detections -> %d after area filter -> %d after "
        "merging %d tile-cut pieces into %d fields -> %d after de-duplication -> %d output "
        "(%d tiles, %.1f s)",
        n_raw,
        n_area,
        n_merge,
        n_pieces_merged,
        n_merged_groups,
        n_dedup,
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
