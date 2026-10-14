"""
Direct SAM 2.1 delineation (``engine="sam-auto"``): automatic masks on large tiles.

Segment Anything 2.1 (Ravi et al. 2025) finds objects from a regular grid of
point prompts, with the automatic mask generator of Segment Anything
(Kirillov et al. 2023). This engine runs that generator on large overlapping
tiles of the composite and turns its masks into one set of non-overlapping
polygons. It needs no field labels and no fine-tuning. SAM outlines every region
it sees as one object, not only fields: house lots, rangeland patches, road
strips and image seams come out as polygons too, and the LULC crop filter (on
by default) removes only part of them. Look-alike neighbours are often merged
into one polygon. The engine is opt-in: it is never a default, and agribound's
agent proposes it only when it is named. Example 28 runs it on NAIP
(``--no-sam-auto`` drops that run), and examples 24 and 25 run it with
``--sam-auto``.

Algorithm (:meth:`SAMAutoEngine.delineate`)
-------------------------------------------
1. **Checks.** The raster must not be rotated or sheared, must be in a
   projected CRS (tile sizes and area limits are in metres) and must have
   square pixels. Rasters of class codes are refused
   (:func:`agribound.io.raster.refuse_categorical`).
2. **Image.** The canonical R, G, B bands of ``config.source`` (or
   ``config.bands``, or ``engine_params["sam_rgb_bands"]``), with the
   refinement stage's scene-wide stretch: the 1st and 99th percentiles of each
   band on a decimated read of at most 4,096 pixels per side, the same bounds
   for every tile; uint8 rasters are used as they are
   (:mod:`agribound.engines.samgeo_engine`).
3. **Tiles.** Square tiles of ``sam_tile_m`` metres (``tile_px`` pixels) with
   origins every ``floor(tile_px * (1 - sam_tile_overlap))`` pixels; the last
   row and column are shifted inwards so that every tile has its full size,
   and a raster smaller than a tile is one tile padded with zeros
   (:func:`tile_origins`). Each tile is resampled to SAM's 1,024 x 1,024 pixel
   input with OpenCV (bilinear when enlarged, area average when reduced).
4. **Validity.** A pixel is valid when it lies in the raster, is finite in the
   three bands, is not ``nodata`` in all three and is not 0 in all three, all
   tested on the raw data, as Delineate-Anything does. The stretched image is
   never used for this: in the pilot, a test on the stretched image cut every
   valid pixel darker than the 1st percentile out of the masks.
5. **Masks per tile.** The tile is encoded once, and ``points_per_side**2``
   point prompts are decoded in batches of ``points_per_batch``, three masks
   each, with the steps of ``SAM2AutomaticMaskGenerator`` (sam2 1.1.0) for one
   crop: logits binarised at 0, stability score with offset 1.0 on the
   upsampled logits, boxes from the masks. On the device, masks with predicted
   IoU <= 0.6 or stability < 0.8 are dropped (lower thresholds lower these
   floors); on the CPU a mask is kept when its predicted IoU >
   ``pred_iou_thresh`` and its stability >= ``stability_score_thresh``
   (float64 comparisons), and then by box non-maximum suppression at
   ``box_nms_thresh`` (:func:`box_nms`). Only the survivors are resampled to
   the tile's pixels (value >= 0.5 after bilinear enlargement or area
   reduction) and clipped to the valid pixels. These are the generator's
   masks and scores, but only bit-packed candidates are moved off the device:
   in the pilot, sam2's own path grew past 50 GB at 64 points per side on MPS.
   The one difference: sam2's generator compares the float32 scores with the
   thresholds rounded to float32, this engine in float64, so a score exactly
   at a float32-rounded threshold can pass in one and fail in the other (none
   did in the real-weights test).
6. **Mask filters.** Dropped are masks whose box comes within
   ``tile_edge_margin_px`` (2) pixels of an interior tile edge (the tile cut
   them; an edge on the raster's border is not interior), masks of fewer than
   ``min_mask_px`` (16) pixels or under ``config.min_field_area_m2``, and masks
   over ``max_field_area_m2`` (2.59 km2, one square mile) or over
   ``max_tile_fraction`` (0.5) of a tile's area.
7. **Partition.** The masks of all tiles are ordered
   (``partition_order="score"``: higher predicted IoU, then higher stability,
   then smaller area; ``"area_asc"``: smaller area, then higher predicted IoU;
   remaining ties in tile order, rows first, then in decoding order) and
   painted into one int32 label raster: each mask claims only pixels no mask
   has claimed, and is kept only if it claims at least one pixel and at least
   ``min_keep`` (0.5) of its pixels. Duplicates from overlapping tiles and
   nested masks (a field and the group of fields around it) are resolved here
   (:func:`partition_masks`).
8. **Polygons.** Each label is traced (8-connected,
   :func:`agribound.postprocess.regions.polygonize_labels`) and keeps its
   largest part. The pipeline then applies the study-area rule and its usual
   post-processing; the partition has no overlaps, so its merge step changes
   nothing.

An object is whole in some tile, wherever it lies, only if it is at most
``tile_px - stride - 2 * tile_edge_margin_px - 1`` pixels across
(:func:`largest_whole_object_px`): about half a tile at the default 50 %
overlap. Larger objects can be cut in every tile and dropped (step 6).

Settings per source
-------------------
The four settings that depend on the source (``sam_tile_m``,
``points_per_side``, ``pred_iou_thresh``, ``stability_score_thresh``;
:data:`SOURCE_KEYS`) were chosen on 2026-10-12 on two Nevada development
squares, example 24's training squares in Diamond Valley and Mason Valley,
by mean F1 among 72 settings per source (sam-direct pilot): settings within
0.01 of the highest mean F1 tied, then the fewest SAM forward passes per km2,
then the higher mean boundary F1. The NAIP, Sentinel-2 and Landsat settings
have the highest mean F1 of their 72; for SPOT 6/7 ten settings tied at the
same cost, and boundary F1 chose one of mean F1 0.578 (the highest was
0.587). Their development scores are in-sample, and no held-out accuracy is
established by that choice. Values given in ``engine_params`` replace them.

==============  ==========  ======  ========  ==========  ====================================
source          sam_tile_m  points  IoU >     stability   status
==============  ==========  ======  ========  ==========  ====================================
naip            2,048       32      0.8       >= 0.95     default
spot            6,144       32      0.6       >= 0.95     default
sentinel2       5,120       32      0.8       >= 0.9      available; did not transfer; WARNING
landsat         7,680       32      0.6       >= 0.8      available; WARNING about its cost
sentinel1       (none)                                    every setting must be given
other sources   1,024 px    32      0.8       >= 0.95     native scale; WARNING (not piloted)
==============  ==========  ======  ========  ==========  ====================================

``usgs-naip-plus`` shows the same imagery as ``naip`` but no setting was
piloted on it, so it is one of the other sources.

Every source shares the other defaults (:data:`DEFAULT_PARAMS`): SAM 2.1
large, 50 % tile overlap, 64 prompts per batch, box NMS 0.7, the 2 px cut-edge
rule, 16 px, 2.59 km2 and half a tile as mask limits, the score partition and
``min_keep`` 0.5.

Pre-registered evaluation (fixed on 2026-10-12, run on 2026-10-13): paired
F1 differences against Delineate-Anything v2 as released, on the same
composites, each with its example's own scoring (one-to-one matching at
IoU >= 0.5). NAIP: +0.017 over
46 of example 25's 5 km squares (California left out: Delineate-Anything's
training data hold its DWR labels; 95 % interval +0.004 to +0.030; both
engines near F1 0.3): kept, and better by the pre-registered rule.
Sentinel-2: -0.152 on the same squares (-0.178 to -0.127; lower on 45 of
the 46): it did not transfer, and its setting was neither changed nor tried
again. SPOT 6/7: a mean of +0.256 over four Nevada and New Mexico areas;
Landsat (8/9; 7/8/9 on the San Juan box): +0.126 over the same areas (Mason
Valley -0.04). As pre-registered, these means pool pairs scored without the
crop filter in Nevada and with it in New Mexico (with all four pairs
unfiltered the means are +0.211 and +0.091, a secondary result). Both are
kept; four areas support no claim that the engine is better there.
Amendment of 2026-10-13: Mason Valley's pairs had been scored against a
crop-filter copy of its reference (176 of its 187 fields), and both Nevada
areas were scored again with the reference after the first scores and the
endpoints had been read, a recorded deviation from the pre-registration; no
outcome changed.

GPU time per km2 at these defaults, measured in that evaluation back to back
with Delineate-Anything on the same composites of two New Mexico boxes (415
and 496 km2; the seconds per km2 divide by the composites' export grids,
about 430 and 526 km2; Apple M2 Max, MPS, SAM 2.1 large; the pipeline's
``delineate`` step, model loading included; the ratio is that of the summed
seconds): NAIP
4.48 s against 0.74 s (6.1 times), SPOT 6/7 0.44 s against 0.068 s (6.4
times), Sentinel-2 0.59 s against 0.035 s (17.0 times) and Landsat 0.23 s
against 0.011 s (20.7 times). On example 25's 5 km squares, against
Delineate-Anything runs made in other sessions, the ratios were 3.1 (NAIP)
and 2.1 (Sentinel-2). Sentinel-1 has no default: on the development squares
the setting the same rule chose was level with Delineate-Anything (mean F1
0.437 against 0.441, in-sample; the highest of its 72 settings had 0.443) at
about 22 times its GPU time on 5 km squares.

The pilot ran SPOT, Sentinel-2 and Landsat as one tile per 5 km square. In a
pre-registered multi-tile development check (2026-10-12), three development
squares were scored inside runs on 25 km composites of 36-90 tiles and
compared with their single-tile runs. Sentinel-2's F1 changed by at most
0.02. SPOT and Landsat lost 0.09 and 0.11 on one square (Diamond Valley,
with the same images in both runs; more polygons on ground the reference does
not map) and gained on the other two, which fails the check's rule (every
change >= -0.05). Part of SPOT's gain of 0.12 in Mason Valley comes from a
different composite: the 25 km composite holds 7 scenes there, the square's
5.

Determinism and devices
-----------------------
Nothing is sampled: the point grid, the tile order, the partition order and
its tie-breaks are fixed, so a run gives identical polygons on the same device
with the same versions. On MPS, the pilot's prototype gave identical masks on
repeated runs, and on 2026-10-12 this engine reproduced the prototype's
polygons and scores exactly on the two development squares (NAIP, SPOT,
Sentinel-2, Landsat). Masks differ between devices: the refinement stage
measured mask IoU 0.59-0.97 between MPS and CPU masks of the same fields
(CUDA was not measured). ``engine_meta`` records the device and the
versions, and :data:`agribound.provenance.HASH_EXCLUDED_FIELDS`
leaves the device out of the configuration hash, so an output made on one
device is reused on another unless ``overwrite=True``.

Output and provenance
---------------------
One polygon per kept mask, in the raster's CRS, with ``sam_label`` (its id in
the label raster, in partition order), ``sam_iou`` (SAM's predicted IoU) and
``sam_stability``. ``gdf.attrs["engine_meta"]`` records the model id, the
Hugging Face revision and SHA-256 of the checkpoint, the sam2 distribution
and its origin, the torch, torchvision and OpenCV versions, the device, every
setting and where it came from, the tile grid, the stretch bounds, the counts
of each step and the seconds of each stage.

Licences: SAM 2.1's code and weights are Apache-2.0. The ``sam2`` 1.1.0
package on PyPI is a third-party build of Meta's Apache-2.0 code (its home
page is not Meta's repository); the fast path in step 5 calls predictor
internals checked with that version only, so the ``sam-auto`` extra pins it.

HPC tiling (:mod:`agribound.hpc.tiles`) is not supported in this version: a
tile run would need a halo of at least one SAM tile, which was never tested
(:func:`hpc_tiling_problem`).

References
----------
Kirillov, A., et al. (2023). Segment Anything. IEEE/CVF International
Conference on Computer Vision (ICCV 2023). arXiv:2304.02643.

Ravi, N., et al. (2025). SAM 2: Segment Anything in Images and Videos.
International Conference on Learning Representations (ICLR 2025).
arXiv:2408.00714.
"""

from __future__ import annotations

import logging
import math
import numbers
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import geopandas as gpd
import numpy as np

from agribound.config import AgriboundConfig
from agribound.engines.base import DelineationEngine
from agribound.registry import ENGINE_REGISTRY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The engine's name in :data:`agribound.registry.ENGINE_REGISTRY`.
ENGINE_NAME = "sam-auto"

#: Default model (Apache-2.0 code and weights).
DEFAULT_MODEL = "facebook/sam2.1-hiera-large"

#: Checkpoint file of every SAM 2 / 2.1 model id (as ``sam2.build_sam.HF_MODEL_ID_TO_FILENAMES``
#: of sam2 1.1.0), so that weights can be prefetched without the sam2 package.
CHECKPOINT_FILES: dict[str, str] = {
    "facebook/sam2-hiera-tiny": "sam2_hiera_tiny.pt",
    "facebook/sam2-hiera-small": "sam2_hiera_small.pt",
    "facebook/sam2-hiera-base-plus": "sam2_hiera_base_plus.pt",
    "facebook/sam2-hiera-large": "sam2_hiera_large.pt",
    "facebook/sam2.1-hiera-tiny": "sam2.1_hiera_tiny.pt",
    "facebook/sam2.1-hiera-small": "sam2.1_hiera_small.pt",
    "facebook/sam2.1-hiera-base-plus": "sam2.1_hiera_base_plus.pt",
    "facebook/sam2.1-hiera-large": "sam2.1_hiera_large.pt",
}

#: Pinned checkpoints: model id -> (Hugging Face revision, SHA-256, size in bytes). The
#: default model is downloaded at this revision and its SHA-256 is checked; other models are
#: downloaded at ``main`` and their revision and SHA-256 are recorded.
PINNED_CHECKPOINTS: dict[str, tuple[str, str, int]] = {
    "facebook/sam2.1-hiera-large": (
        "665f8e2ad61cf5f53d65644ff27c8ee525124610",
        "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
        898_083_611,
    ),
}

#: Size aliases accepted by ``engine_params["sam_model"]`` (SAM 2.1 models).
MODEL_ALIASES: dict[str, str] = {
    "tiny": "facebook/sam2.1-hiera-tiny",
    "small": "facebook/sam2.1-hiera-small",
    "base_plus": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}

#: SAM's square input size in pixels (SAM 2 / 2.1).
SAM_INPUT_PX = 1024

#: Device-side floors (step 5 of the module docstring): masks with predicted IoU at or below
#: this value, or stability below the next one (float32 comparisons on the device), are never
#: moved to the CPU. These are the pilot's generation floors; a threshold below a floor lowers
#: it. A mask can differ from a pure float64 test only for values within float32 rounding of
#: the floors.
DEVICE_FLOOR_IOU = 0.6
DEVICE_FLOOR_STABILITY = 0.8

#: ``SAM2AutomaticMaskGenerator`` defaults used by the fast path (sam2 1.1.0).
MASK_THRESHOLD = 0.0
STABILITY_OFFSET = 1.0

#: The sam2 version whose predictor internals the fast path was checked with.
SAM2_VERSION_CHECKED = "1.1.0"

#: Meta's commit (github.com/facebookresearch/sam2, main, 2024-12-15) whose files the ``sam2``
#: 1.1.0 package on PyPI holds: its 62 code, configuration and licence files are byte-identical
#: to the files of this commit, with the four symbolic links among Meta's configuration files
#: resolved (checked 2026-10-12 against the commit's git objects).
SAM2_META_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"

#: :func:`sam2_files_sha256` of that package, as installed from PyPI.
SAM2_1_1_0_FILES_SHA256 = "17bcd3b68a0d4699111402ef2899c7619758e696583a7b31a423f9b072c9d07e"

#: Settings that depend on the source (module docstring, "Settings per source").
SOURCE_KEYS: tuple[str, ...] = (
    "sam_tile_m",
    "points_per_side",
    "pred_iou_thresh",
    "stability_score_thresh",
)

#: Per-source defaults chosen in the 2026-10-12 pilot (in-sample, two Nevada squares).
#: ``usgs-naip-plus`` (the same imagery as ``naip``, not piloted) has none: it runs at
#: native scale (:data:`NATIVE_SETTINGS`) with a WARNING.
SOURCE_SETTINGS: dict[str, dict[str, Any]] = {
    "naip": {
        "sam_tile_m": 2048.0,
        "points_per_side": 32,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.95,
    },
    "spot": {
        "sam_tile_m": 6144.0,
        "points_per_side": 32,
        "pred_iou_thresh": 0.6,
        "stability_score_thresh": 0.95,
    },
    "sentinel2": {
        "sam_tile_m": 5120.0,
        "points_per_side": 32,
        "pred_iou_thresh": 0.8,
        "stability_score_thresh": 0.9,
    },
    "landsat": {
        "sam_tile_m": 7680.0,
        "points_per_side": 32,
        "pred_iou_thresh": 0.6,
        "stability_score_thresh": 0.8,
    },
}

#: Sources whose default is costly at scale: GPU seconds per km2 of SAM and of
#: Delineate-Anything, measured back to back on the same composites of two New Mexico boxes in
#: the pre-registered evaluation (Apple M2 Max, MPS; module docstring), and the ratio of the
#: summed seconds.
COSTLY_SOURCES: dict[str, dict[str, float]] = {
    "sentinel2": {"sam_s_per_km2": 0.59, "da_s_per_km2": 0.035, "ratio": 17.0},
    "landsat": {"sam_s_per_km2": 0.23, "da_s_per_km2": 0.011, "ratio": 20.7},
}

#: Sources on which the engine did not transfer in its pre-registered evaluation (primary
#: paired F1 difference against Delineate-Anything below -0.05; module docstring), with the
#: result as :func:`settings_warning` states it. Their settings were not changed or tried again.
DID_NOT_TRANSFER: dict[str, str] = {
    "sentinel2": (
        "on 46 squares of example 25 its F1 was 0.15 below Delineate-Anything's (95 % interval "
        "-0.18 to -0.13), with about a quarter as many polygons (mean recall 0.09 against 0.26)"
    ),
}

#: Sources without a default: every key of :data:`SOURCE_KEYS` must be given.
NO_DEFAULT_SOURCES: tuple[str, ...] = ("sentinel1",)

#: Settings of the sources without a piloted setting (other than :data:`NO_DEFAULT_SOURCES`):
#: native scale and sam2's own thresholds.
NATIVE_TILE_PX = 1024
NATIVE_SETTINGS: dict[str, Any] = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.8,
    "stability_score_thresh": 0.95,
}

#: Values of ``engine_params["partition_order"]``.
PARTITION_ORDERS: tuple[str, ...] = ("score", "area_asc")

#: Default ``engine_params`` (None: per source, :data:`SOURCE_SETTINGS`).
DEFAULT_PARAMS: dict[str, Any] = {
    "sam_model": DEFAULT_MODEL,
    "sam_tile_m": None,
    "sam_tile_overlap": 0.5,
    "points_per_side": None,
    "points_per_batch": 64,
    "pred_iou_thresh": None,
    "stability_score_thresh": None,
    "box_nms_thresh": 0.7,
    "tile_edge_margin_px": 2,
    "min_mask_px": 16,
    "max_field_area_m2": 2.59e6,
    "max_tile_fraction": 0.5,
    "partition_order": "score",
    "min_keep": 0.5,
    "sam_rgb_bands": None,
}

#: ``engine_params`` keys read by the pipeline rather than by the engine; accepted here
#: (``pivots``: the opt-in centre-pivot post-process, :mod:`agribound.postprocess.pivots`).
PIPELINE_KEYS: frozenset[str] = frozenset(
    {"smooth_iterations", "regularize", "sam_refine", "pivots"}
)

#: A WARNING is logged when the label raster (4 bytes per pixel) exceeds this size.
LABEL_RASTER_WARN_BYTES = 2_000_000_000

#: Relative difference of the two pixel sides above which a raster is refused.
_SQUARE_PIXEL_TOLERANCE = 1e-3

#: Numeric ``engine_params``: name -> (low, high, low end open, high end open).
_RANGES: dict[str, tuple[float, float, bool, bool]] = {
    "sam_tile_m": (0.0, math.inf, True, True),
    "sam_tile_overlap": (0.0, 0.9, False, True),
    "pred_iou_thresh": (0.0, 1.0, False, False),
    "stability_score_thresh": (0.0, 1.0, False, False),
    "box_nms_thresh": (0.0, 1.0, False, False),
    "max_field_area_m2": (0.0, math.inf, True, True),
    "max_tile_fraction": (0.0, 1.0, True, False),
    "min_keep": (0.0, 1.0, False, False),
}

#: Integer ``engine_params``: name -> smallest value.
_MINIMA: dict[str, int] = {
    "points_per_side": 4,
    "points_per_batch": 1,
    "tile_edge_margin_px": 0,
    "min_mask_px": 1,
}


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _number(name: str, value: Any, low: float, high: float, *, low_open: bool, high_open: bool):
    """Return *value* as a float after checking it lies in the given interval."""
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ValueError(f"engine_params[{name!r}] must be a finite number, got {value!r}")
    v = float(value)
    too_low = v <= low if low_open else v < low
    too_high = v >= high if high_open else v > high
    if too_low or too_high:
        lo, hi = ("(" if low_open else "["), (")" if high_open else "]")
        bound = f"{lo}{low:g}, {high:g}{hi}" if math.isfinite(high) else f"{lo}{low:g}, inf)"
        raise ValueError(f"engine_params[{name!r}] must be in {bound}, got {value!r}")
    return v


def _integer(name: str, value: Any, low: int) -> int:
    """Return *value* as an int (whole-number floats accepted) after checking ``>= low``."""
    if isinstance(value, bool) or not (
        isinstance(value, numbers.Integral) or (_is_number(value) and float(value).is_integer())
    ):
        raise ValueError(f"engine_params[{name!r}] must be an integer, got {value!r}")
    v = int(value)
    if v < low:
        raise ValueError(f"engine_params[{name!r}] must be >= {low}, got {value!r}")
    return v


def resolve_model(value: Any = None) -> str:
    """Return the Hugging Face id of the SAM model of ``engine_params["sam_model"]``.

    Parameters
    ----------
    value : str or None
        *None* (the default model, SAM 2.1 large), a SAM 2 or SAM 2.1 model id
        (with or without the ``facebook/`` prefix) or a size alias
        (``"tiny"``, ``"small"``, ``"base_plus"``, ``"large"``: SAM 2.1).

    Returns
    -------
    str

    Raises
    ------
    ValueError
        For any other value (SAM 3 is not supported by this engine).
    """
    if value is None:
        return DEFAULT_MODEL
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"engine_params['sam_model'] must be a model id or alias, got {value!r}")
    text = value.strip()
    alias = text.lower().replace("-", "_")
    if alias in MODEL_ALIASES:
        return MODEL_ALIASES[alias]
    model_id = text if text.startswith("facebook/") else f"facebook/{text}"
    if model_id not in CHECKPOINT_FILES:
        raise ValueError(
            f"engine_params['sam_model']={value!r} is not a SAM 2 or SAM 2.1 model. Use one of "
            f"{sorted(CHECKPOINT_FILES)} or a size alias {sorted(MODEL_ALIASES)} (SAM 2.1)."
        )
    return model_id


def check_params(source: str, engine_params: Mapping[str, Any] | None) -> None:
    """Check the ``engine_params`` of engine ``"sam-auto"`` for *source*.

    Called when an :class:`~agribound.config.AgriboundConfig` with this engine
    is validated, and again by the engine.

    Parameters
    ----------
    source : str
        Source of the run.
    engine_params : mapping or None
        The configuration's ``engine_params``.

    Raises
    ------
    ValueError
        For a key that neither the engine nor the pipeline reads, an invalid
        value (``sam_tile_m`` <= 0, ``sam_tile_overlap`` outside [0, 0.9),
        ``points_per_side`` below 4, a threshold outside [0, 1], an unknown
        ``partition_order``, ...), ``sam_refine`` set, or a source of
        :data:`NO_DEFAULT_SOURCES` without every key of :data:`SOURCE_KEYS`.
    """
    params = dict(engine_params or {})
    unknown = sorted(set(params) - set(DEFAULT_PARAMS) - PIPELINE_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown engine_params for engine {ENGINE_NAME!r}: {unknown}. Valid keys: "
            f"{sorted(DEFAULT_PARAMS)} (and {sorted(PIPELINE_KEYS)}, read by the pipeline)"
        )
    if params.get("sam_refine"):
        raise ValueError(
            f"engine_params['sam_refine'] does not apply to engine {ENGINE_NAME!r}: its "
            "polygons are already SAM masks."
        )
    if params.get("sam_model") is not None:
        resolve_model(params["sam_model"])
    for name, (low, high, low_open, high_open) in _RANGES.items():
        if params.get(name) is not None:
            _number(name, params[name], low, high, low_open=low_open, high_open=high_open)
    for name, low in _MINIMA.items():
        if params.get(name) is not None:
            _integer(name, params[name], low)
    order = params.get("partition_order")
    if order is not None and (not isinstance(order, str) or order.strip() not in PARTITION_ORDERS):
        raise ValueError(
            f"engine_params['partition_order'] must be one of {PARTITION_ORDERS}, got {order!r}"
        )
    bands = params.get("sam_rgb_bands")
    if bands is not None:
        ok = isinstance(bands, list | tuple) and len(bands) == 3
        try:
            ok = ok and all(_integer("sam_rgb_bands", b, 1) for b in bands)
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(
                f"engine_params['sam_rgb_bands'] must list 3 band indices (1-based), got {bands!r}"
            )
    if str(source).lower().strip() in NO_DEFAULT_SOURCES:
        missing = [k for k in SOURCE_KEYS if params.get(k) is None]
        if missing:
            raise ValueError(
                f"Engine {ENGINE_NAME!r} has no default setting for source {source!r}: on two "
                "Nevada development squares the setting the selection rule chose (2,560 m "
                "tiles, 32 points per side, predicted IoU > 0.6, stability >= 0.8; chosen "
                "there, so in-sample) was level with Delineate-Anything (mean F1 0.437 against "
                "0.441; the highest of its 72 settings had 0.443) at about 22 times its GPU "
                "time per km2 on 5 km squares. To run it anyway, set every one of "
                f"engine_params {list(SOURCE_KEYS)}; missing: {missing}."
            )


@dataclass
class SAMAutoSettings:
    """The settings one run uses (:func:`resolve_settings`).

    Attributes
    ----------
    params : dict
        Every key of :data:`DEFAULT_PARAMS` with its value (model resolved,
        per-source keys filled in).
    basis : str
        Where the per-source settings come from: ``"default"`` (``naip``,
        ``spot``), ``"default, costly at scale"`` (``sentinel2``,
        ``landsat``), ``"user"`` (``sentinel1``) or ``"native scale, not
        piloted"`` (other sources, ``usgs-naip-plus`` included).
    user_keys : list of str
        Keys of :data:`DEFAULT_PARAMS` given in ``engine_params``.
    """

    params: dict[str, Any]
    basis: str
    user_keys: list[str] = field(default_factory=list)


def resolve_settings(
    source: str, engine_params: Mapping[str, Any] | None, pixel_size_m: float
) -> SAMAutoSettings:
    """Return the settings of a run on *source* with pixels of *pixel_size_m* metres.

    Values given in *engine_params* win; the per-source keys come from
    :data:`SOURCE_SETTINGS`, and sources without a piloted setting get native
    scale (``sam_tile_m = 1,024 * pixel_size_m``) with
    :data:`NATIVE_SETTINGS`. *engine_params* is checked first
    (:func:`check_params`).

    Raises
    ------
    ValueError
        As :func:`check_params`.
    """
    source = str(source).lower().strip()
    check_params(source, engine_params)
    given = {k: v for k, v in dict(engine_params or {}).items() if k in DEFAULT_PARAMS}
    given = {k: v for k, v in given.items() if v is not None}
    if source in SOURCE_SETTINGS:
        per_source = dict(SOURCE_SETTINGS[source])
        basis = "default, costly at scale" if source in COSTLY_SOURCES else "default"
    elif source in NO_DEFAULT_SOURCES:
        per_source = {}
        basis = "user"
    else:
        per_source = {"sam_tile_m": NATIVE_TILE_PX * float(pixel_size_m), **NATIVE_SETTINGS}
        basis = "native scale, not piloted"
    params = {**DEFAULT_PARAMS, **per_source, **given}
    params["sam_model"] = resolve_model(params["sam_model"])
    params["sam_tile_m"] = float(params["sam_tile_m"])
    params["sam_tile_overlap"] = float(params["sam_tile_overlap"])
    for name in ("points_per_side", "points_per_batch", "tile_edge_margin_px", "min_mask_px"):
        params[name] = int(params[name])
    for name in (
        "pred_iou_thresh",
        "stability_score_thresh",
        "box_nms_thresh",
        "max_field_area_m2",
        "max_tile_fraction",
        "min_keep",
    ):
        params[name] = float(params[name])
    params["partition_order"] = str(params["partition_order"]).strip()
    if params["sam_rgb_bands"] is not None:
        params["sam_rgb_bands"] = [int(b) for b in params["sam_rgb_bands"]]
    return SAMAutoSettings(params=params, basis=basis, user_keys=sorted(given))


def settings_warning(source: str, settings: SAMAutoSettings) -> str | None:
    """The WARNING logged for a run's settings, or *None* (``naip``, ``spot``, ``sentinel1``).

    ``sentinel2`` and ``landsat``: the measured cost at scale
    (:data:`COSTLY_SOURCES`) and, for ``sentinel2``, that its default did not
    transfer in the pre-registered evaluation (:data:`DID_NOT_TRANSFER`).
    Sources without a piloted setting: that no setting was piloted for them,
    with the settings used.
    """
    source = str(source).lower().strip()
    p = settings.params
    used = (
        f"{p['sam_tile_m']:,.0f} m tiles, {p['points_per_side']} points per side, predicted "
        f"IoU > {p['pred_iou_thresh']:g}, stability >= {p['stability_score_thresh']:g}"
    )
    if source in COSTLY_SOURCES:
        cost = COSTLY_SOURCES[source]
        result = (
            " In the pre-registered evaluation the default setting did not transfer: "
            f"{DID_NOT_TRANSFER[source]}."
            if source in DID_NOT_TRANSFER
            else ""
        )
        return (
            f"engine {ENGINE_NAME!r} on {source} is costly at scale: at its default setting "
            f"SAM 2.1 large took about {cost['sam_s_per_km2']:g} s of GPU time per km2, "
            f"{cost['ratio']:.1f} times Delineate-Anything's {cost['da_s_per_km2']:g} s per km2, "
            "measured back to back on the same composites of two New Mexico boxes of 415 and "
            "496 km2, whose export grids are about 430 and 526 km2 (Apple M2 Max, MPS; the "
            "pipeline's delineate step, model loading included)."
            f"{result} The setting was chosen on two Nevada development squares (in-sample). "
            f"This run: {used}."
        )
    if settings.basis == "native scale, not piloted":
        origin = (
            f"engine_params given: {', '.join(settings.user_keys)}"
            if settings.user_keys
            else "native scale and sam2's own thresholds"
        )
        return (
            f"engine {ENGINE_NAME!r}: no setting was piloted for source {source!r}; this run "
            f"uses {used} ({origin}). Check the polygons before relying on them."
        )
    return None


# ---------------------------------------------------------------------------
# Tiles, NMS, partition (pure functions)
# ---------------------------------------------------------------------------


def tile_origins(size: int, tile: int, stride: int) -> list[int]:
    """Tile origins covering ``[0, size)``: multiples of *stride*, the last shifted inwards.

    Parameters
    ----------
    size : int
        Raster size along one axis (pixels).
    tile : int
        Tile size (pixels).
    stride : int
        Distance between origins (pixels, >= 1).

    Returns
    -------
    list[int]
        ``[0]`` when ``size <= tile`` (one tile, padded beyond the raster);
        otherwise ``0, stride, 2 * stride, ...`` up to ``size - tile``, plus
        ``size - tile`` when the last multiple does not reach the end.
    """
    if size <= tile:
        return [0]
    origins = list(range(0, size - tile + 1, stride))
    if origins[-1] + tile < size:
        origins.append(size - tile)
    return origins


def tile_stride(tile_px: int, overlap: float) -> int:
    """Origin spacing ``floor(tile_px * (1 - overlap))`` (at least 1); ``tile_px // 2`` at 0.5."""
    return max(1, int(math.floor(tile_px * (1.0 - float(overlap)) + 1e-9)))


def largest_whole_object_px(tile_px: int, stride: int, margin: int) -> int:
    """Largest extent (pixels, along a row or a column) of an object that is whole in some tile.

    A mask is cut when its box comes within *margin* pixels of an interior tile
    edge (:func:`cut_by_tile`), so in one tile an object can span at most
    ``tile_px - 2 * margin - 2`` pixels. Tiles start every *stride* pixels, so
    an object of extent ``L`` lies inside the uncut part of some tile wherever
    it lies only if ``L <= tile_px - stride - 2 * margin - 1``. Larger objects
    can be cut in every tile that holds them (the last row and column, shifted
    inwards, and the raster's own borders only add room). At 50 % overlap this
    is about half a tile.
    """
    return max(0, int(tile_px) - int(stride) - 2 * int(margin) - 1)


def interior_edges(
    col: int, row: int, tile_px: int, width: int, height: int
) -> tuple[bool, bool, bool, bool]:
    """Which edges of the tile at ``(col, row)`` lie inside the raster (left, top, right, bottom).

    An edge on the raster's border is not interior; nor is the far edge of a
    tile that reaches beyond the raster (a padded tile).
    """
    return col > 0, row > 0, col + tile_px < width, row + tile_px < height


def cut_by_tile(
    box: tuple[int, int, int, int],
    interior: tuple[bool, bool, bool, bool],
    tile_px: int,
    margin: int,
) -> bool:
    """Whether a mask with *box* ``(x0, y0, x1, y1)`` (tile pixels, inclusive) is cut by its tile.

    True when the box comes within *margin* pixels of an interior edge
    (:func:`interior_edges`): ``x0 <= margin`` at the left edge, ``y0 <=
    margin`` at the top, ``x1 >= tile_px - 1 - margin`` at the right and ``y1
    >= tile_px - 1 - margin`` at the bottom.
    """
    x0, y0, x1, y1 = box
    left, top, right, bottom = interior
    far = tile_px - 1 - margin
    return (
        (left and x0 <= margin)
        or (top and y0 <= margin)
        or (right and x1 >= far)
        or (bottom and y1 >= far)
    )


def box_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy box non-maximum suppression, as ``torchvision.ops.nms`` on the CPU.

    Boxes are visited by descending score (a stable sort, so equal scores keep
    their order); each kept box suppresses every later box whose IoU with it
    exceeds *iou_threshold*. Areas are ``(x2 - x1) * (y2 - y1)`` and the IoU is
    computed in float32, then compared in float64, as torchvision's kernel
    does; for boxes on whole pixels (SAM's mask boxes) the result equals
    torchvision's.

    Parameters
    ----------
    boxes : numpy.ndarray
        ``(n, 4)`` boxes ``x1, y1, x2, y2``.
    scores : numpy.ndarray
        ``(n,)`` scores (compared in float32).
    iou_threshold : float
        Boxes with a larger IoU are suppressed.

    Returns
    -------
    numpy.ndarray
        Indices of the kept boxes, ascending.
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    n = len(scores)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    x1, y1, x2, y2 = (boxes[:, k] for k in range(4))
    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []
    threshold = float(iou_threshold)
    for pos in range(n):
        i = int(order[pos])
        if suppressed[i]:
            continue
        keep.append(i)
        rest = order[pos + 1 :]
        rest = rest[~suppressed[rest]]
        if rest.size == 0:
            continue
        w = np.maximum(np.float32(0), np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(np.float32(0), np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = inter / (areas[i] + areas[rest] - inter)
        suppressed[rest[iou.astype(np.float64) > threshold]] = True
    return np.sort(np.asarray(keep, dtype=np.int64))


def masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """``(n, 4)`` float32 boxes ``x1, y1, x2, y2`` of ``(n, h, w)`` bool masks (max inclusive).

    As ``sam2.utils.amg.batched_mask_to_box``: an empty mask gives ``0, 0, 0, 0``.
    """
    masks = np.asarray(masks, dtype=bool)
    out = np.zeros((len(masks), 4), dtype=np.float32)
    for k, m in enumerate(masks):
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m.any(axis=0))
        if rows.size:
            out[k] = (cols[0], rows[0], cols[-1], rows[-1])
    return out


@dataclass
class TileCandidates:
    """Masks of one SAM input image that passed the device floors.

    Attributes
    ----------
    masks : numpy.ndarray
        ``(n, h, ceil(w / 8))`` uint8: each mask's rows bit-packed
        (:func:`numpy.packbits` along the last axis).
    width : int
        Image width ``w`` (for unpacking).
    iou : numpy.ndarray
        ``(n,)`` predicted IoU (float64 copies of float32 values).
    stability : numpy.ndarray
        ``(n,)`` stability scores (float64 copies of float32 values).
    boxes : numpy.ndarray
        ``(n, 4)`` float32 boxes in input pixels (:func:`masks_to_boxes`).
    """

    masks: np.ndarray
    width: int
    iou: np.ndarray
    stability: np.ndarray
    boxes: np.ndarray

    @classmethod
    def from_masks(cls, masks: Any, iou: Any, stability: Any) -> TileCandidates:
        """Build from ``(n, h, w)`` bool masks and their scores (boxes computed here)."""
        masks = np.asarray(masks, dtype=bool)
        if masks.ndim != 3:
            raise ValueError(f"masks must be (n, h, w), got shape {masks.shape}")
        return cls(
            masks=np.packbits(masks, axis=-1),
            width=int(masks.shape[2]),
            iou=np.asarray(iou, dtype=np.float32).astype(np.float64).reshape(-1),
            stability=np.asarray(stability, dtype=np.float32).astype(np.float64).reshape(-1),
            boxes=masks_to_boxes(masks),
        )

    def mask(self, i: int) -> np.ndarray:
        """Mask *i* as an ``(h, w)`` bool array."""
        return np.unpackbits(self.masks[i], axis=-1, count=self.width).astype(bool)


@dataclass
class MaskCandidate:
    """A mask that passed the filters (step 6), cropped to its box in raster pixels."""

    col: int
    row: int
    packed: np.ndarray
    shape: tuple[int, int]
    area_px: int
    iou: float
    stability: float

    def mask(self) -> np.ndarray:
        return np.unpackbits(self.packed, axis=-1, count=self.shape[1]).astype(bool)


def partition_masks(
    candidates: list[MaskCandidate],
    height: int,
    width: int,
    order: str = "score",
    min_keep: float = 0.5,
) -> tuple[np.ndarray, dict[int, tuple[float, float]], int]:
    """Paint *candidates* into a non-overlapping label raster (step 7).

    Parameters
    ----------
    candidates : list of MaskCandidate
        In tile order, then decoding order (the order that breaks ties).
    height, width : int
        Raster size.
    order : str
        ``"score"`` (higher predicted IoU, then higher stability, then smaller
        area first) or ``"area_asc"`` (smaller area, then higher predicted IoU).
    min_keep : float
        A mask is kept only if the pixels it claims are at least this share of
        its pixels. A mask that claims no pixel is dropped whatever the value
        (also at 0), so every label holds pixels.

    Returns
    -------
    tuple
        ``(labels, scores, n_dropped)``: int32 labels (0 = none; ids from 1 in
        painting order), ``{label: (iou, stability)}`` and the number of masks
        dropped for claiming too little.
    """
    if order == "score":
        ranked = sorted(candidates, key=lambda c: (-c.iou, -c.stability, c.area_px))
    elif order == "area_asc":
        ranked = sorted(candidates, key=lambda c: (c.area_px, -c.iou))
    else:
        raise ValueError(f"partition order must be one of {PARTITION_ORDERS}, got {order!r}")
    labels = np.zeros((int(height), int(width)), dtype=np.int32)
    scores: dict[int, tuple[float, float]] = {}
    next_id = 1
    n_dropped = 0
    for c in ranked:
        m = c.mask()
        h, w = m.shape
        sub = labels[c.row : c.row + h, c.col : c.col + w]
        free = m & (sub == 0)
        n_free = int(free.sum())
        if n_free == 0 or n_free < min_keep * c.area_px:
            n_dropped += 1
            continue
        sub[free] = next_id
        scores[next_id] = (c.iou, c.stability)
        next_id += 1
    return labels, scores, n_dropped


def labels_to_polygons(
    labels: np.ndarray, scores: Mapping[int, tuple[float, float]], transform: Any, crs: Any
) -> gpd.GeoDataFrame:
    """Trace *labels* (8-connected) and keep each label's largest part (step 8).

    Returns a frame with ``sam_label``, ``sam_iou``, ``sam_stability`` and the
    geometry, ordered by label.
    """
    from agribound.postprocess.regions import polygonize_labels

    gdf = polygonize_labels(labels, transform, crs, column="sam_label")
    if len(gdf):
        gdf["area_"] = gdf.geometry.area
        gdf = gdf.sort_values(["sam_label", "area_"], ascending=[True, False]).drop_duplicates(
            "sam_label"
        )
        gdf = gdf.drop(columns="area_").reset_index(drop=True)
    gdf["sam_iou"] = np.asarray([scores[int(i)][0] for i in gdf["sam_label"]], dtype=np.float64)
    gdf["sam_stability"] = np.asarray(
        [scores[int(i)][1] for i in gdf["sam_label"]], dtype=np.float64
    )
    return gdf[["sam_label", "sam_iou", "sam_stability", gdf.geometry.name]]


# ---------------------------------------------------------------------------
# Raster access
# ---------------------------------------------------------------------------


def pixel_size_m(src: Any, raster_path: str = "") -> float:
    """Pixel size in metres of an open dataset, after the checks of step 1.

    Raises
    ------
    ValueError
        For a rotated or sheared transform, a raster without a projected CRS,
        or pixels that are not square (relative difference above 0.1 %).
    """
    t = src.transform
    if t.b != 0 or t.d != 0:
        raise ValueError(
            f"{raster_path} has a rotated or sheared transform, which engine {ENGINE_NAME!r} "
            "does not support"
        )
    crs = src.crs
    if crs is None or not crs.is_projected:
        raise ValueError(
            f"Engine {ENGINE_NAME!r} needs a raster in a projected CRS (its tile sizes and "
            f"area limits are in metres); {raster_path} is in {crs}. Use export_crs='utm' "
            "(the default) or reproject the raster."
        )
    sx, sy = abs(float(t.a)), abs(float(t.e))
    if abs(sx - sy) > _SQUARE_PIXEL_TOLERANCE * max(sx, sy):
        raise ValueError(
            f"Engine {ENGINE_NAME!r} needs square pixels; {raster_path} has {sx:g} x {sy:g} "
            "CRS units. Resample the raster to square pixels."
        )
    factor = float(crs.linear_units_factor[1])
    return sx * factor


def read_tile(
    src: Any,
    bands: list[int],
    stretch: tuple[list[float], list[float], bool],
    col: int,
    row: int,
    tile_px: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read one tile: the stretched uint8 image and its valid pixels (steps 2 and 4).

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Open dataset.
    bands : list of int
        The three 1-based bands used as R, G, B.
    stretch : tuple
        ``(lows, highs, passthrough)`` of the scene-wide stretch.
    col, row : int
        Tile origin (pixels).
    tile_px : int
        Tile size (pixels).

    Returns
    -------
    tuple
        ``(image, valid, w, h)``: the ``(tile_px, tile_px, 3)`` uint8 image and
        the ``(tile_px, tile_px)`` validity mask, both zero beyond the raster,
        and the size ``w x h`` of the part inside the raster. Validity is read
        from the raw data: finite, not ``nodata`` in all bands, not 0 in all
        bands.
    """
    from rasterio.windows import Window

    from agribound.engines.samgeo_engine import _apply_stretch

    lows, highs, passthrough = stretch
    w = min(tile_px, src.width - col)
    h = min(tile_px, src.height - row)
    data = src.read(bands, window=Window(col, row, w, h))
    img = _apply_stretch(data, lows, highs, src.nodata, passthrough)
    if np.issubdtype(data.dtype, np.floating):
        valid = np.all(np.isfinite(data), axis=0)
    else:
        valid = np.ones(data.shape[1:], dtype=bool)
    if src.nodata is not None and np.isfinite(src.nodata):
        valid &= ~np.all(data == src.nodata, axis=0)
    valid &= ~np.all(data == 0, axis=0)
    out = np.zeros((tile_px, tile_px, 3), dtype=np.uint8)
    out[:h, :w] = img
    vmask = np.zeros((tile_px, tile_px), dtype=bool)
    vmask[:h, :w] = valid
    return out, vmask, w, h


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize a square uint8 image to *size* (bilinear up, area down; unchanged if equal)."""
    h = image.shape[0]
    if h == size and image.shape[1] == size:
        return image
    import cv2

    interp = cv2.INTER_LINEAR if size > h else cv2.INTER_AREA
    return cv2.resize(image, (size, size), interpolation=interp)


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Resample a square bool mask to *size* (area down, bilinear up; value >= 0.5)."""
    if mask.shape[0] == size and mask.shape[1] == size:
        return mask
    import cv2

    interp = cv2.INTER_AREA if size < mask.shape[0] else cv2.INTER_LINEAR
    return cv2.resize(mask.astype(np.float32), (size, size), interpolation=interp) >= 0.5


# ---------------------------------------------------------------------------
# SAM 2 generator (fast path)
# ---------------------------------------------------------------------------


def sam2_files_sha256(name: str = "sam2") -> tuple[str, int] | None:
    """SHA-256 over the installed files of distribution *name*, and their number.

    The files its RECORD lists, read from disk, without bytecode
    (``__pycache__``) and without the installer's metadata (``*.dist-info``
    except its ``licenses/``): one ``"<path>\\t<SHA-256 of the file>\\n"`` line
    per file, in path order. So a file changed after the installation changes
    the result, and the Python version and the installer do not. *None* when
    the distribution is not installed or a file cannot be read.
    """
    import hashlib
    from importlib import metadata
    from pathlib import Path

    try:
        dist = metadata.distribution(name)
        rows = []
        for f in dist.files or []:
            path = str(f).replace("\\", "/")
            if "__pycache__" in path or (".dist-info/" in path and "/licenses/" not in path):
                continue
            digest = hashlib.sha256(Path(dist.locate_file(f)).read_bytes()).hexdigest()
            rows.append(f"{path}\t{digest}\n")
    except (metadata.PackageNotFoundError, OSError):
        return None
    if not rows:
        return None
    return hashlib.sha256("".join(sorted(rows)).encode()).hexdigest(), len(rows)


def sam2_distribution() -> dict[str, Any]:
    """Name, version and origin of the installed ``sam2`` distribution (``{}`` if absent).

    ``origin`` says whether it is Meta's own release or a third-party build:
    the ``sam2`` 1.1.0 package on PyPI (home page
    github.com/JinsuaFeito-dev/segment-anything-2) is a third-party build of
    Meta's Apache-2.0 SAM 2 code (github.com/facebookresearch/sam2), whose
    files equal Meta's commit :data:`SAM2_META_COMMIT`. The installed files
    are hashed (:func:`sam2_files_sha256`, ``files_sha256``), so the origin
    names that commit only when they are that package's files.
    """
    from importlib import metadata

    try:
        names = metadata.packages_distributions().get("sam2") or ["sam2"]
        md = metadata.metadata(names[0])
    except metadata.PackageNotFoundError:
        return {}
    home = md.get("Home-page") or ""
    urls = md.get_all("Project-URL") or []
    where = " ".join([home, *urls])
    files = sam2_files_sha256(names[0])
    same = files is not None and files[0] == SAM2_1_1_0_FILES_SHA256
    if "facebookresearch/sam2" in where or "facebookresearch/segment-anything-2" in where:
        origin = "Meta's release (github.com/facebookresearch/sam2)"
    elif same:
        origin = (
            "a third-party build of Meta's Apache-2.0 SAM 2 code "
            f"(github.com/facebookresearch/sam2): its {files[1]} code, configuration and licence "
            f"files equal Meta's commit {SAM2_META_COMMIT[:7]} of 2024-12-15; home page "
            f"{home or 'not given'}"
        )
    else:
        origin = (
            "a third-party build of Meta's Apache-2.0 SAM 2 code "
            f"(github.com/facebookresearch/sam2); home page {home or 'not given'}; its files "
            "were not compared with Meta's"
        )
    return {
        "distribution": md.get("Name"),
        "version": md.get("Version"),
        "home_page": home or None,
        "author": md.get("Author"),
        "license": md.get("License"),
        "origin": origin,
        "files_sha256": files[0] if files else None,
        "n_files": files[1] if files else None,
        "meta_commit": SAM2_META_COMMIT if same else None,
    }


def checkpoint_path(model_id: str) -> tuple[str, str, str]:
    """Download (or find in the Hugging Face cache) the checkpoint of *model_id*.

    The default model is fetched at its pinned revision and its SHA-256 is
    checked (:data:`PINNED_CHECKPOINTS`); other models are fetched at
    ``main``.

    Returns
    -------
    tuple[str, str, str]
        ``(path, revision, sha256)``.

    Raises
    ------
    RuntimeError
        If a pinned checkpoint's SHA-256 differs.
    """
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    from agribound.engines.delineate_anything import file_sha256

    filename = CHECKPOINT_FILES[model_id]
    pinned = PINNED_CHECKPOINTS.get(model_id)
    path = hf_hub_download(model_id, filename, revision=pinned[0] if pinned else None)
    sha256 = file_sha256(path)
    if pinned and sha256 != pinned[1]:
        raise RuntimeError(
            f"SHA-256 mismatch for {model_id}/{filename}@{pinned[0]}: expected {pinned[1]}, got "
            f"{sha256} ({path}). Delete the cached file and retry."
        )
    revision = pinned[0] if pinned else Path(path).parent.name
    return str(path), revision, sha256


class SAM2PointGridGenerator:
    """SAM 2 automatic-mask candidates of one image, without RLE encoding (step 5).

    The steps of ``SAM2AutomaticMaskGenerator._process_crop`` (sam2 1.1.0)
    for one full-image crop and ``multimask_output=True``, ``use_m2m=False``:
    the point grid scaled to the image, batches of prompts through the
    predictor's ``_predict``, the device floors, stability on the upsampled
    logits, binarisation at 0 and boxes from the masks. Masks are moved to the
    CPU per batch, bit-packed. The box NMS and the thresholds are applied by
    the engine (:func:`box_nms`).

    Parameters
    ----------
    predictor : sam2.sam2_image_predictor.SAM2ImagePredictor
        Predictor with ``mask_threshold=0``, ``max_hole_area=0`` and
        ``max_sprinkle_area=0`` (as the generator builds it).
    model_id, device : str
        Recorded in :meth:`describe`.
    checkpoint : dict
        ``{"file", "revision", "sha256"}`` of the loaded weights.
    """

    def __init__(self, predictor: Any, model_id: str, device: str, checkpoint: dict) -> None:
        self.predictor = predictor
        self.model_id = model_id
        self.device = str(device)
        self.checkpoint = dict(checkpoint)

    def _sync(self) -> None:
        import torch

        if self.device.startswith("mps"):
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            torch.cuda.synchronize()

    def candidates(
        self,
        image: np.ndarray,
        *,
        points_per_side: int,
        points_per_batch: int,
        iou_floor: float,
        stability_floor: float,
        timing: dict[str, float],
    ) -> TileCandidates:
        """Masks of *image* (``(h, w, 3)`` uint8) above the floors; adds to *timing*."""
        import torch
        from sam2.utils.amg import (
            batch_iterator,
            batched_mask_to_box,
            build_point_grid,
            calculate_stability_score,
        )

        pred = self.predictor
        t0 = time.perf_counter()
        pred.set_image(image)
        self._sync()
        timing["encode_s"] = timing.get("encode_s", 0.0) + time.perf_counter() - t0
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        points_for_image = build_point_grid(points_per_side) * np.array((h, w))[None, ::-1]
        masks_out, ious_out, stabs_out, boxes_out = [], [], [], []
        for (points,) in batch_iterator(points_per_batch, points_for_image):
            pts = torch.as_tensor(points, dtype=torch.float32, device=pred.device)
            in_points = pred._transforms.transform_coords(pts, normalize=True, orig_hw=(h, w))
            in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
            masks, iou_preds, _ = pred._predict(
                in_points[:, None, :],
                in_labels[:, None],
                multimask_output=True,
                return_logits=True,
            )
            masks = masks.flatten(0, 1)
            iou_preds = iou_preds.flatten(0, 1)
            keep = iou_preds > iou_floor
            masks, iou_preds = masks[keep], iou_preds[keep]
            stab = calculate_stability_score(masks, MASK_THRESHOLD, STABILITY_OFFSET)
            keep = stab >= stability_floor
            masks, iou_preds, stab = masks[keep], iou_preds[keep], stab[keep]
            masks = masks > MASK_THRESHOLD
            boxes = batched_mask_to_box(masks)
            masks_out.append(np.packbits(masks.cpu().numpy(), axis=-1))
            ious_out.append(iou_preds.float().cpu().numpy())
            stabs_out.append(stab.float().cpu().numpy())
            boxes_out.append(boxes.float().cpu().numpy())
            del masks
        pred.reset_predictor()
        timing["decode_s"] = timing.get("decode_s", 0.0) + time.perf_counter() - t0
        if not masks_out:
            return TileCandidates(
                masks=np.zeros((0, h, (w + 7) // 8), dtype=np.uint8),
                width=w,
                iou=np.zeros(0),
                stability=np.zeros(0),
                boxes=np.zeros((0, 4), dtype=np.float32),
            )
        return TileCandidates(
            masks=np.concatenate(masks_out),
            width=w,
            iou=np.concatenate(ious_out).astype(np.float64),
            stability=np.concatenate(stabs_out).astype(np.float64),
            boxes=np.concatenate(boxes_out).astype(np.float32),
        )

    def describe(self) -> dict[str, Any]:
        """Model, checkpoint and versions for ``engine_meta``."""
        import torch
        import torchvision

        return {
            "model": self.model_id,
            "checkpoint": dict(self.checkpoint),
            "device": self.device,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "sam2": sam2_distribution(),
        }


def load_generator(model_id: str, device: str) -> SAM2PointGridGenerator:
    """Load SAM 2 / 2.1 *model_id* on *device* (``apply_postprocessing=False``).

    The same model and predictor as ``sam2.build_sam.build_sam2_hf`` followed
    by ``SAM2AutomaticMaskGenerator(model, min_mask_region_area=0)``, with the
    checkpoint of :func:`checkpoint_path`.

    Raises
    ------
    ImportError
        Without the sam2 package or OpenCV, before any checkpoint is read
        (``pip install "agribound[sam-auto]"``).
    """
    try:
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as exc:
        raise ImportError(
            f"Engine {ENGINE_NAME!r} needs the sam2 package (and torch, OpenCV). Install with: "
            'pip install "agribound[sam-auto]"'
        ) from exc
    try:  # sam2 does not require OpenCV, which resamples the tiles and masks
        import cv2  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"Engine {ENGINE_NAME!r} needs OpenCV (cv2), which the sam2 package does not "
            'install. Install with: pip install "agribound[sam-auto]"'
        ) from exc
    from agribound.engines.samgeo_engine import _quiet_root_info

    dist = sam2_distribution()
    if dist.get("version") != SAM2_VERSION_CHECKED:
        logger.warning(
            "engine %r: sam2 %s is installed; the fast path calls SAM2ImagePredictor._predict "
            "and _transforms.transform_coords, which were checked with sam2 %s only.",
            ENGINE_NAME,
            dist.get("version"),
            SAM2_VERSION_CHECKED,
        )
    path, revision, sha256 = checkpoint_path(model_id)
    config_name = HF_MODEL_ID_TO_FILENAMES[model_id][0]
    with _quiet_root_info():
        model = build_sam2(
            config_file=config_name,
            ckpt_path=path,
            device=device,
            mode="eval",
            apply_postprocessing=False,
        )
    predictor = SAM2ImagePredictor(model)
    checkpoint = {"file": CHECKPOINT_FILES[model_id], "revision": revision, "sha256": sha256}
    return SAM2PointGridGenerator(predictor, model_id, device, checkpoint)


def hpc_tiling_problem(config: Any) -> str | None:
    """Why *config* cannot run with HPC tiling (:mod:`agribound.hpc.tiles`), or *None*.

    Engine ``"sam-auto"``, directly or as an ensemble member, is refused in
    this version: an HPC tile would need a halo of at least one SAM tile
    (2-8 km at the defaults), which was never tested.
    """
    from agribound._results import _runs_engine

    if not _runs_engine(config, ENGINE_NAME):
        return None
    return (
        f"Engine {ENGINE_NAME!r} does not support HPC tiling in this version: an HPC tile would "
        "need a halo of at least one SAM tile (2-8 km at the default settings), which was never "
        "tested. Run it on the study area as one job (agribound delineate), or use another "
        "engine for tiled runs."
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SAMAutoEngine(DelineationEngine):
    """Direct field delineation with SAM 2.1 automatic masks (see the module docstring).

    Parameters
    ----------
    generator : object or None
        A loaded generator with ``candidates(image, *, points_per_side,
        points_per_batch, iou_floor, stability_floor, timing) ->
        TileCandidates`` and ``describe() -> dict`` (``model_id`` and
        ``device`` attributes); *None* loads SAM (:func:`load_generator`).
        Tests pass a fake generator here.
    """

    name = ENGINE_NAME
    supported_sources: list[str] = list(ENGINE_REGISTRY[ENGINE_NAME]["supported_sources"])
    requires_bands = ["R", "G", "B"]

    def __init__(self, generator: Any = None) -> None:
        self._generator = generator

    @staticmethod
    def resolve_params(config: AgriboundConfig, pixel_size: float) -> SAMAutoSettings:
        """The settings of a run of *config* on pixels of *pixel_size* metres."""
        return resolve_settings(config.source, config.engine_params, pixel_size)

    def validate_input(self, raster_path: str, config: AgriboundConfig) -> None:
        """Refuse class-code rasters and check the R, G, B bands (``sam_rgb_bands`` first).

        Raises
        ------
        ValueError
            For a raster of class codes, or band indices out of range.
        """
        import rasterio

        from agribound.engines.samgeo_engine import _rgb_band_indices
        from agribound.io.raster import refuse_categorical

        refuse_categorical(raster_path, config, f"Engine {self.name!r}")
        with rasterio.open(raster_path) as src:
            _rgb_band_indices(config, src.count)

    @classmethod
    def prefetch(cls, config: AgriboundConfig) -> list[str]:
        """Download the configured SAM checkpoint into the Hugging Face cache.

        Returns
        -------
        list[str]
            The local path of the checkpoint.
        """
        model_id = resolve_model((config.engine_params or {}).get("sam_model"))
        return [checkpoint_path(model_id)[0]]

    def delineate(self, raster_path: str, config: AgriboundConfig) -> gpd.GeoDataFrame:
        """Delineate fields in *raster_path* (steps 1-8 of the module docstring).

        Parameters
        ----------
        raster_path : str
            Composite GeoTIFF.
        config : AgriboundConfig
            Uses ``source``, ``bands``, ``device``, ``min_field_area_m2`` and
            ``engine_params`` (:data:`DEFAULT_PARAMS`).

        Returns
        -------
        geopandas.GeoDataFrame
            ``sam_label``, ``sam_iou``, ``sam_stability`` and the geometry, in
            the raster's CRS; ``attrs["engine_meta"]`` as in the module
            docstring.

        Raises
        ------
        ValueError
            For the refusals of step 1 and invalid ``engine_params``.
        ImportError
            Without the sam2 package or OpenCV when they are needed.
        """
        import rasterio

        from agribound.engines.samgeo_engine import STRETCH_PERCENTILES, _rgb_band_indices
        from agribound.engines.samgeo_engine import _stretch_bounds as stretch_bounds

        t_start = time.perf_counter()
        self.validate_input(raster_path, config)
        timing: dict[str, float] = {
            "load_s": 0.0,
            "encode_s": 0.0,
            "decode_s": 0.0,
            "post_s": 0.0,
            "partition_s": 0.0,
            "polygons_s": 0.0,
        }
        stats = {
            "n_candidates": 0,
            "n_survivors": 0,
            "n_cut": 0,
            "n_small": 0,
            "n_large": 0,
            "n_overlap_dropped": 0,
            "n_kept": 0,
        }
        with rasterio.open(raster_path) as src:
            gsd_m = pixel_size_m(src, raster_path)
            settings = self.resolve_params(config, gsd_m)
            p = settings.params
            warning = settings_warning(config.source, settings)
            if warning:
                logger.warning(warning)
            rgb = _rgb_band_indices(config, src.count, p["sam_rgb_bands"])
            lows, highs, passthrough = stretch_bounds(src, rgb, False)
            tile_px = max(1, int(round(p["sam_tile_m"] / gsd_m)))
            stride = tile_stride(tile_px, p["sam_tile_overlap"])
            cols = tile_origins(src.width, tile_px, stride)
            rows = tile_origins(src.height, tile_px, stride)
            tiles = [(c, r) for r in rows for c in cols]
            whole_m = largest_whole_object_px(tile_px, stride, p["tile_edge_margin_px"]) * gsd_m
            label_bytes = 4 * src.width * src.height
            if label_bytes > LABEL_RASTER_WARN_BYTES:
                logger.warning(
                    "engine %r: the partition holds an int32 label raster of %.1f GB "
                    "(%d x %d pixels); split large areas into several runs.",
                    ENGINE_NAME,
                    label_bytes / 1e9,
                    src.width,
                    src.height,
                )
            logger.info(
                "sam-auto (%s, settings: %s): %d tiles of %d px (%.0f m, SAM sees %.2f m "
                "pixels), %d points per side (one every %.0f m); objects larger than %.0f m "
                "across can be cut by every tile and dropped",
                p["sam_model"],
                settings.basis,
                len(tiles),
                tile_px,
                p["sam_tile_m"],
                p["sam_tile_m"] / SAM_INPUT_PX,
                p["points_per_side"],
                p["sam_tile_m"] / p["points_per_side"],
                whole_m,
            )
            device = config.resolve_device()
            generator = self._generator
            if generator is None:
                t0 = time.perf_counter()
                generator = load_generator(p["sam_model"], device)
                timing["load_s"] = time.perf_counter() - t0
            iou_floor = min(DEVICE_FLOOR_IOU, p["pred_iou_thresh"])
            stability_floor = min(DEVICE_FLOOR_STABILITY, p["stability_score_thresh"])
            px_area_m2 = gsd_m * gsd_m
            max_tile_px = p["max_tile_fraction"] * tile_px * tile_px
            margin = p["tile_edge_margin_px"]
            candidates: list[MaskCandidate] = []
            for i_tile, (col, row) in enumerate(tiles):
                image, valid, w, h = read_tile(
                    src, rgb, (lows, highs, passthrough), col, row, tile_px
                )
                tile = generator.candidates(
                    _resize_image(image, SAM_INPUT_PX),
                    points_per_side=p["points_per_side"],
                    points_per_batch=p["points_per_batch"],
                    iou_floor=iou_floor,
                    stability_floor=stability_floor,
                    timing=timing,
                )
                t0 = time.perf_counter()
                stats["n_candidates"] += len(tile.iou)
                passed = np.flatnonzero(
                    (tile.iou > p["pred_iou_thresh"])
                    & (tile.stability >= p["stability_score_thresh"])
                )
                kept = passed[box_nms(tile.boxes[passed], tile.iou[passed], p["box_nms_thresh"])]
                interior = interior_edges(col, row, tile_px, src.width, src.height)
                for i in kept:
                    stats["n_survivors"] += 1
                    m = _resize_mask(tile.mask(int(i)), tile_px)[:h, :w] & valid[:h, :w]
                    area = int(m.sum())
                    if area == 0:
                        stats["n_small"] += 1
                        continue
                    ys = np.flatnonzero(m.any(axis=1))
                    xs = np.flatnonzero(m.any(axis=0))
                    x0, x1, y0, y1 = int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])
                    if cut_by_tile((x0, y0, x1, y1), interior, tile_px, margin):
                        stats["n_cut"] += 1
                        continue
                    if area < p["min_mask_px"] or area * px_area_m2 < config.min_field_area_m2:
                        stats["n_small"] += 1
                        continue
                    if area * px_area_m2 > p["max_field_area_m2"] or area > max_tile_px:
                        stats["n_large"] += 1
                        continue
                    crop = m[y0 : y1 + 1, x0 : x1 + 1]
                    candidates.append(
                        MaskCandidate(
                            col=col + x0,
                            row=row + y0,
                            packed=np.packbits(crop, axis=-1),
                            shape=(int(crop.shape[0]), int(crop.shape[1])),
                            area_px=area,
                            iou=float(tile.iou[i]),
                            stability=float(tile.stability[i]),
                        )
                    )
                timing["post_s"] += time.perf_counter() - t0
                if (i_tile + 1) % 25 == 0:
                    logger.info(
                        "sam-auto: %d/%d tiles done (%d masks kept)",
                        i_tile + 1,
                        len(tiles),
                        len(candidates),
                    )
            transform, crs, width, height = src.transform, src.crs, src.width, src.height

        t0 = time.perf_counter()
        labels, scores, n_dropped = partition_masks(
            candidates, height, width, p["partition_order"], p["min_keep"]
        )
        stats["n_overlap_dropped"] = n_dropped
        stats["n_kept"] = len(scores)
        timing["partition_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        gdf = labels_to_polygons(labels, scores, transform, crs)
        timing["polygons_s"] = time.perf_counter() - t0
        del labels
        stats["n_polygons"] = len(gdf)
        timing["total_s"] = time.perf_counter() - t_start
        timing = {k: round(v, 3) for k, v in timing.items()}

        meta = {
            "engine": ENGINE_NAME,
            "implementation": "agribound sam-auto 1 (SAM 2 point-grid masks, fast path)",
            **generator.describe(),
            "opencv": _opencv_version(),
            "settings": {
                **p,
                "min_field_area_m2": float(config.min_field_area_m2),
                "device_floor_iou": iou_floor,
                "device_floor_stability": stability_floor,
            },
            "settings_basis": settings.basis,
            "settings_given": settings.user_keys,
            "settings_warning": warning,
            "tiles": {
                "pixel_size_m": gsd_m,
                "tile_px": tile_px,
                "stride_px": stride,
                "n_cols": len(cols),
                "n_rows": len(rows),
                "n_tiles": len(tiles),
                "sam_pixel_m": p["sam_tile_m"] / SAM_INPUT_PX,
                "point_spacing_m": p["sam_tile_m"] / p["points_per_side"],
                "largest_whole_object_m": whole_m,
            },
            "rgb_bands": rgb,
            "stretch": {
                "method": "none (uint8)" if passthrough else "percentile",
                "percentiles": None if passthrough else list(STRETCH_PERCENTILES),
                "lows": list(lows),
                "highs": list(highs),
            },
            "validity": "raw data: finite, not nodata in all bands, not 0 in all bands",
            "counts": stats,
            "seconds": timing,
            "label_raster_mb": round(label_bytes / 1e6, 1),
            "determinism": (
                "identical on the same device with the same versions; differs between devices"
            ),
        }
        gdf.attrs["engine_meta"] = meta
        logger.info(
            "sam-auto: %d polygons from %d masks (%d candidates, %d after the thresholds and NMS; "
            "%d cut by tiles, %d small, %d large, %d overlapped) in %.1f s",
            len(gdf),
            stats["n_kept"],
            stats["n_candidates"],
            stats["n_survivors"],
            stats["n_cut"],
            stats["n_small"],
            stats["n_large"],
            stats["n_overlap_dropped"],
            timing["total_s"],
        )
        return gdf


def _opencv_version() -> str | None:
    """OpenCV's version, or *None* when it is not installed (no tile was resized)."""
    try:
        import cv2
    except ImportError:
        return None
    return str(cv2.__version__)


__all__ = [
    "CHECKPOINT_FILES",
    "COSTLY_SOURCES",
    "DEFAULT_MODEL",
    "DEFAULT_PARAMS",
    "DID_NOT_TRANSFER",
    "NO_DEFAULT_SOURCES",
    "PINNED_CHECKPOINTS",
    "SAM2_1_1_0_FILES_SHA256",
    "SAM2_META_COMMIT",
    "SAMAutoEngine",
    "SAMAutoSettings",
    "SAM2PointGridGenerator",
    "SOURCE_KEYS",
    "SOURCE_SETTINGS",
    "TileCandidates",
    "box_nms",
    "check_params",
    "checkpoint_path",
    "cut_by_tile",
    "hpc_tiling_problem",
    "interior_edges",
    "labels_to_polygons",
    "largest_whole_object_px",
    "load_generator",
    "partition_masks",
    "pixel_size_m",
    "read_tile",
    "resolve_model",
    "resolve_settings",
    "sam2_distribution",
    "sam2_files_sha256",
    "settings_warning",
    "tile_origins",
    "tile_stride",
]
