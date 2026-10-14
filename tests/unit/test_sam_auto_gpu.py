"""Tests of engine ``sam-auto`` with the real SAM 2.1 weights (marked gpu and slow).

They need the sam2 package (``pip install "agribound[sam-auto]"``), a CUDA or MPS device and
the ``facebook/sam2.1-hiera-large`` checkpoint (downloaded into the Hugging Face cache on first
use, about 900 MB). The fast path calls predictor internals of sam2, so these tests guard the
version it was checked with.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

sam2 = pytest.importorskip("sam2")
torch = pytest.importorskip("torch")

from agribound.config import AgriboundConfig  # noqa: E402
from agribound.engines import sam_auto  # noqa: E402
from agribound.engines.sam_auto import SAMAutoEngine, box_nms  # noqa: E402

UTM = "EPSG:32611"


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    pytest.skip("no CUDA or MPS device")


@pytest.fixture(scope="module")
def generator():
    return sam_auto.load_generator(sam_auto.DEFAULT_MODEL, _device())


def _scene(size=1024, seed=0):
    """A synthetic image: a few uniform rectangles and discs on a textured background."""
    rng = np.random.default_rng(seed)
    img = rng.integers(90, 120, (size, size, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    shapes = [
        ((60, 60, 300, 380), (40, 160, 60)),
        ((80, 450, 330, 900), (180, 170, 90)),
        ((420, 100, 700, 420), (60, 90, 170)),
        ((620, 520, 960, 980), (200, 200, 200)),
    ]
    for (r0, c0, r1, c1), colour in shapes:
        img[r0:r1, c0:c1] = colour
    disc = (yy - 800) ** 2 + (xx - 230) ** 2 < 150**2
    img[disc] = (30, 120, 40)
    return img


def test_sam2_version_and_internals(generator):
    """The fast path relies on sam2 1.1.0's predictor internals."""
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

    dist = sam_auto.sam2_distribution()
    assert dist["version"] == sam_auto.SAM2_VERSION_CHECKED
    assert "third-party build" in dist["origin"] or "Meta's release" in dist["origin"]
    predictor = generator.predictor
    assert callable(predictor._predict) and callable(predictor._transforms.transform_coords)
    for model_id, filename in sam_auto.CHECKPOINT_FILES.items():
        assert HF_MODEL_ID_TO_FILENAMES[model_id][1] == filename
    revision, sha256, _ = sam_auto.PINNED_CHECKPOINTS[sam_auto.DEFAULT_MODEL]
    assert generator.checkpoint == {
        "file": "sam2.1_hiera_large.pt",
        "revision": revision,
        "sha256": sha256,
    }


def test_fast_path_equals_sam2_generator(generator):
    """Same survivors, scores and masks as SAM2AutomaticMaskGenerator on one image."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    image = _scene()
    stock = SAM2AutomaticMaskGenerator(
        generator.predictor.model,
        points_per_side=16,
        points_per_batch=64,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.95,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        min_mask_region_area=0,
        output_mode="binary_mask",
        multimask_output=True,
    )
    with torch.inference_mode():
        anns = stock.generate(image)
    cand = generator.candidates(
        image,
        points_per_side=16,
        points_per_batch=64,
        iou_floor=0.6,
        stability_floor=0.8,
        timing={},
    )
    passed = np.flatnonzero((cand.iou > 0.8) & (cand.stability >= 0.95))
    kept = passed[box_nms(cand.boxes[passed], cand.iou[passed], 0.7)]
    assert len(anns) > 0 and len(kept) == len(anns)

    def key(iou, stab, mask):
        return (round(float(iou), 6), round(float(stab), 6), int(mask.sum()))

    ours = sorted(
        (key(cand.iou[i], cand.stability[i], cand.mask(int(i))), cand.mask(int(i))) for i in kept
    )
    theirs = sorted(
        (key(a["predicted_iou"], a["stability_score"], a["segmentation"]), a["segmentation"])
        for a in anns
    )
    for (k1, m1), (k2, m2) in zip(ours, theirs, strict=True):
        assert k1 == k2
        assert np.array_equal(m1, m2)


def _raster(path):
    image = _scene(size=600, seed=1)
    data = np.ascontiguousarray(image.transpose(2, 0, 1))
    profile = {
        "driver": "GTiff",
        "height": 600,
        "width": 600,
        "count": 3,
        "dtype": "uint8",
        "crs": UTM,
        "transform": Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 4002000.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return str(path)


def _config(tmp_path, raster):
    return AgriboundConfig(
        source="local",
        engine="sam-auto",
        local_tif_path=raster,
        output_path=str(tmp_path / "fields.gpkg"),
        lulc_filter=False,
        device=_device(),
    )


def test_same_output_on_one_device(tmp_path, generator):
    """Identical polygons twice with one loaded model and once with a newly loaded one."""
    raster = _raster(tmp_path / "scene.tif")
    cfg = _config(tmp_path, raster)
    first = SAMAutoEngine(generator=generator).delineate(raster, cfg)
    second = SAMAutoEngine(generator=generator).delineate(raster, cfg)
    fresh = SAMAutoEngine().delineate(raster, cfg)
    assert len(first) > 0
    for other in (second, fresh):
        assert [g.wkb for g in first.geometry] == [g.wkb for g in other.geometry]
        assert first["sam_iou"].tolist() == other["sam_iou"].tolist()
    meta = fresh.attrs["engine_meta"]
    assert meta["device"] == _device() and meta["model"] == sam_auto.DEFAULT_MODEL
    assert meta["checkpoint"]["sha256"] == sam_auto.PINNED_CHECKPOINTS[sam_auto.DEFAULT_MODEL][1]
    assert meta["sam2"]["version"] == "1.1.0" and meta["torch"] == torch.__version__
    assert meta["settings_basis"] == "native scale, not piloted"
    assert meta["seconds"]["load_s"] > 0
