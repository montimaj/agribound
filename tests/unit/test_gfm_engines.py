"""Tests for the GeoAI, DINOv3 and Prithvi engines and their fine-tuning modules.

Upstream models are replaced by small fakes unless the test says otherwise;
tests marked with ``terratorch`` in their name run only where terratorch is
installed (the GFM environment).
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from agribound.config import AgriboundConfig
from agribound.engines.finetune import _data
from agribound.registry import ENGINE_REGISTRY

UTM = "EPSG:32755"
X0, Y0 = 700_000.0, 6_600_000.0


def _write(path, data, res=10.0, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=str(data.dtype),
        crs=UTM,
        transform=from_origin(X0, Y0, res, res),
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return str(path)


def _s2_config(tmp_path, engine, **kwargs):
    values = {
        "source": "sentinel2",
        "year": 2023,
        "study_area": "bbox:150.0,-30.8,150.1,-30.7",
        "gee_project": "test-project",
        "engine": engine,
        "output_path": str(tmp_path / "out" / "fields.gpkg"),
        "lulc_filter": False,
        "device": "cpu",
        "seed": 5,
    }
    values.update(kwargs)
    return AgriboundConfig(**values)


@pytest.fixture
def s2_raster(tmp_path):
    """12-band S2-like x10000 raster, 96 x 96 px, with a NaN corner."""
    rng = np.random.default_rng(0)
    data = rng.uniform(300, 3000, (12, 96, 96)).astype(np.float32)
    data[:, :8, :8] = np.nan
    return _write(tmp_path / "s2.tif", data, nodata=float("nan")), data


@pytest.fixture
def s2_chips(tmp_path):
    """Prepared chip directory factory for an S2 scene with 4 reference fields."""
    rng = np.random.default_rng(1)
    data = rng.uniform(300, 600, (12, 128, 128)).astype(np.float32)
    data[:, 20:110, 20:110] += 2000.0
    raster = _write(tmp_path / "scene.tif", data, nodata=float("nan"))
    fields = [
        box(X0 + 200, Y0 - 700, X0 + 600, Y0 - 250),
        box(X0 + 600, Y0 - 700, X0 + 1000, Y0 - 250),
        box(X0 + 200, Y0 - 1100, X0 + 1000, Y0 - 750),
        box(X0 + 1050, Y0 - 1200, X0 + 1250, Y0 - 1000),
    ]
    ref = tmp_path / "ref.gpkg"
    gpd.GeoDataFrame(geometry=fields, crs=UTM).to_file(ref)

    def make(engine, **params):
        cfg = _s2_config(
            tmp_path,
            engine,
            reference_boundaries=str(ref),
            fine_tune_split="random",
            engine_params={"chip_size": 64, "min_label_fraction": 0.0, **params},
        )
        return cfg, _data._prepare_training_data(raster, cfg, engine), raster

    return make


# ---------------------------------------------------------------------------
# Registry consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "cls", "name"),
    [
        ("agribound.engines.geoai_field", "GeoAIEngine", "geoai"),
        ("agribound.engines.dinov3", "DINOv3Engine", "dinov3"),
        ("agribound.engines.prithvi", "PrithviEngine", "prithvi"),
    ],
)
def test_engine_attributes_match_registry(module, cls, name):
    import importlib

    engine_cls = getattr(importlib.import_module(module), cls)
    assert engine_cls.name == name
    assert engine_cls.supported_sources == ENGINE_REGISTRY[name]["supported_sources"]
    assert engine_cls.requires_bands == ENGINE_REGISTRY[name]["requires_bands"]
    assert {"SWIR1", "SWIR2"} <= set(ENGINE_REGISTRY["prithvi"]["requires_bands"])


# ---------------------------------------------------------------------------
# GeoAI
# ---------------------------------------------------------------------------


class TestGeoAICheckpoint:
    def test_no_checkpoint_raises_with_guidance(self):
        from agribound.engines.geoai_field import resolve_geoai_checkpoint

        with pytest.raises(RuntimeError, match="none is published") as err:
            resolve_geoai_checkpoint({})
        assert "fine_tune=True" in str(err.value)

    def test_missing_file_raises(self, tmp_path):
        from agribound.engines.geoai_field import resolve_geoai_checkpoint

        with pytest.raises(FileNotFoundError):
            resolve_geoai_checkpoint({"checkpoint_path": str(tmp_path / "nope.pth")})

    def test_hub_download_is_explicit(self, monkeypatch, tmp_path):
        import huggingface_hub

        from agribound.engines.geoai_field import resolve_geoai_checkpoint

        calls = []

        def fake(repo_id, filename, revision=None):
            calls.append((repo_id, filename, revision))
            return str(tmp_path / filename)

        monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
        path, info = resolve_geoai_checkpoint(
            {"repo_id": "me/fields", "filename": "m.pth", "revision": "abc"}
        )
        assert calls == [("me/fields", "m.pth", "abc")]
        assert info == {"repo_id": "me/fields", "filename": "m.pth", "revision": "abc"}

    def test_checkpoint_path_and_repo_id_conflict(self, tmp_path):
        from agribound.engines.geoai_field import resolve_geoai_checkpoint

        ckpt = tmp_path / "best_model.pth"
        ckpt.write_bytes(b"w")
        # e.g. fine_tune=True (sets checkpoint_path) with a leftover repo_id
        with pytest.raises(ValueError, match="set only one"):
            resolve_geoai_checkpoint(
                {"checkpoint_path": str(ckpt), "repo_id": "me/fields", "filename": "m.pth"}
            )


class TestGeoAIWindows:
    def test_window_defaults_and_validation(self, caplog):
        from agribound.engines.geoai_field import plan_geoai_windows

        # training chip size known: same window, half overlap, batch kept
        plan = plan_geoai_windows(600, 700, training_chip_size=256)
        assert plan["window_size"] == 256 and plan["overlap"] == 128
        assert plan["batch_size"] == 4 and plan["window_source"] == "training_chip_size"
        assert plan["resize_factor"] == pytest.approx(800 / 256)
        # never shrunk for a small raster; batch size 1 instead
        small = plan_geoai_windows(96, 700, training_chip_size=256)
        assert small["window_size"] == 256 and small["batch_size"] == 1
        assert small["requested_batch_size"] == 4
        # unknown chip size: geoai's default
        assert plan_geoai_windows(600, 700)["window_size"] == 512
        # a window different from the training chips is a scale change -> WARNING
        with caplog.at_level(logging.WARNING, logger="agribound.engines.geoai_field"):
            plan_geoai_windows(600, 700, window_size=512, training_chip_size=256)
        assert "differs from the 256 px chips" in caplog.text and "0.50x" in caplog.text
        with pytest.raises(ValueError, match="overlap must be in"):
            plan_geoai_windows(600, 700, window_size=256, overlap=256)
        with pytest.raises(ValueError, match="geoai would process no window"):
            plan_geoai_windows(40, 700, window_size=256, overlap=200)

    def test_plan_covers_every_pixel_with_geoai_loop(self, tmp_path):
        """The planned settings make geoai 0.43.1's real sliding-window loop visit every pixel."""
        torch = pytest.importorskip("torch")

        train = pytest.importorskip("geoai.train")
        import geoai

        from agribound.engines.geoai_field import plan_geoai_windows

        class PositionReader(torch.nn.Module):
            """Records the raster region of each window (encoded in the pixel values)."""

            def __init__(self):
                super().__init__()
                self.regions = []

            def forward(self, images):
                out = []
                for im in images:
                    real = (im[2] > 0).nonzero()
                    row0 = round(float(im[0, 0, 0]) * 255)
                    col0 = round(float(im[1, 0, 0]) * 255)
                    h, w = int(real[:, 0].max()) + 1, int(real[:, 1].max()) + 1
                    self.regions.append((row0, col0, h, w))
                    out.append(
                        {
                            "scores": torch.zeros(0),
                            "masks": torch.zeros(0, 1, *im.shape[-2:]),
                            "boxes": torch.zeros(0, 4),
                            "labels": torch.zeros(0, dtype=torch.int64),
                        }
                    )
                return out

        def covered(height, width, plan):
            rows, cols = np.mgrid[0:height, 0:width]
            rgb = np.stack([rows, cols, np.ones_like(rows)]).astype(np.uint8)
            path = _write(tmp_path / f"rgb_{height}x{width}.tif", rgb)
            model = PositionReader()
            train.instance_segmentation_inference_on_geotiff(
                model=model,
                geotiff_path=path,
                output_path=str(tmp_path / f"inst_{height}x{width}_{plan['batch_size']}.tif"),
                window_size=plan["window_size"],
                overlap=plan["overlap"],
                batch_size=plan["batch_size"],
                device="cpu",
            )
            seen = np.zeros((height, width), dtype=bool)
            for row0, col0, h, w in model.regions:
                seen[row0 : row0 + h, col0 : col0 + w] = True
            return seen.all()

        # both sides < window; one side < window; both sides >= window
        for height, width, chip in ((96, 96, 512), (100, 180, 128), (200, 240, 128)):
            plan = plan_geoai_windows(height, width, training_chip_size=chip)
            assert covered(height, width, plan), (height, width, plan)
        if geoai.__version__ == "0.43.1":
            # the upstream behaviour the batch_size=1 rule works around
            dropped = {**plan_geoai_windows(100, 180, training_chip_size=128), "batch_size": 4}
            assert not covered(100, 180, dropped)

    def test_maskrcnn_limits_match_torchvision_model(self):
        torchvision = pytest.importorskip("torchvision")
        from agribound.engines.geoai_field import maskrcnn_limits

        limits = maskrcnn_limits()
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=2
        )
        assert model.transform.min_size == (limits["min_size"],) == (800,)
        assert model.transform.max_size == limits["max_size"]
        assert model.roi_heads.score_thresh == limits["box_score_thresh"] == 0.05
        assert model.roi_heads.detections_per_img == limits["box_detections_per_img"] == 100


class TestGeoAIEngine:
    @pytest.fixture
    def fake_geoai(self, monkeypatch):
        train = pytest.importorskip("geoai.train")
        calls = {}

        def fake_instance_segmentation(input_path, output_path, model_path, **kwargs):
            with rasterio.open(input_path) as src:
                calls["input"] = src.read()
                calls["input_dtype"] = src.dtypes[0]
                profile = src.profile.copy()
            calls.update(kwargs, output_path=output_path, model_path=model_path)
            inst = np.zeros(calls["input"].shape[1:], dtype=np.uint32)
            inst[0:6, 0:6] = 1  # inside the NaN corner -> must be masked
            inst[20:40, 20:40] = 2
            inst[50:70, 30:60] = 3
            score = np.where(inst == 2, 0.9, np.where(inst == 3, 0.6, 0.0)).astype(np.float32)
            profile.update(count=1, dtype="uint32")
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(inst, 1)
            base, ext = str(output_path).rsplit(".", 1)
            profile.update(dtype="float32")
            with rasterio.open(f"{base}_score.{ext}", "w", **profile) as dst:
                dst.write(score, 1)
            return {}

        monkeypatch.setattr(train, "instance_segmentation", fake_instance_segmentation)
        return calls

    def test_rgb_input_window_and_polygons(self, tmp_path, s2_raster, fake_geoai):
        from agribound.engines.geoai_field import GeoAIEngine

        raster, data = s2_raster
        ckpt = tmp_path / "best_model.pth"
        ckpt.write_bytes(b"weights")
        cfg = _s2_config(
            tmp_path, "geoai", device="mps", engine_params={"checkpoint_path": str(ckpt)}
        )
        gdf = GeoAIEngine().delineate(raster, cfg)

        # canonical R, G, B = S2 B4, B3, B2 with a scene-level stretch to uint8
        assert fake_geoai["input_dtype"] == "uint8" and fake_geoai["input"].shape[0] == 3
        lows, highs = _data.scene_stretch_bounds(raster, [4, 3, 2])
        expected = _data.apply_stretch(data[[3, 2, 1]], lows, highs)
        np.testing.assert_array_equal(fake_geoai["input"], expected)
        # No training chip size recorded: geoai's 512 px window, not shrunk for the
        # 96 px raster (that would change the scale); batch size 1 so that geoai
        # processes the partial last batch.
        assert fake_geoai["window_size"] == 512 and fake_geoai["overlap"] == 256
        assert fake_geoai["batch_size"] == 1
        assert fake_geoai["num_channels"] == 3 and fake_geoai["num_classes"] == 2
        assert fake_geoai["device"] == "cpu"  # MPS is not used for Mask R-CNN
        # instance 1 lay in the nodata corner and was masked out
        assert sorted(gdf["instance_id"]) == [2, 3]
        assert gdf.set_index("instance_id")["score"].to_dict() == pytest.approx({2: 0.9, 3: 0.6})
        meta = gdf.attrs["engine_meta"]
        assert meta["backend"] == "geoai.instance_segmentation"
        assert meta["band_indices"] == [4, 3, 2] and meta["device"] == "cpu"
        assert meta["window_source"] == "geoai_default" and meta["requested_batch_size"] == 4
        assert meta["maskrcnn_limits"]["box_detections_per_img"] == 100
        assert len(meta["checkpoint_sha256"]) == 64
        json.dumps(meta)  # copied into the provenance record

    def test_window_follows_training_chip_size(self, tmp_path, s2_raster, fake_geoai, caplog):
        from agribound.engines.geoai_field import GeoAIEngine

        raster, _ = s2_raster
        ckpt = tmp_path / "best_model.pth"
        ckpt.write_bytes(b"weights")
        _data.write_training_meta(ckpt, {"engine": "geoai", "chip_size": 64})
        cfg = _s2_config(tmp_path, "geoai", engine_params={"checkpoint_path": str(ckpt)})
        with caplog.at_level(logging.WARNING, logger="agribound.engines.geoai_field"):
            gdf = GeoAIEngine().delineate(raster, cfg)
        assert fake_geoai["window_size"] == 64 and fake_geoai["overlap"] == 32
        assert fake_geoai["batch_size"] == 4  # raster sides >= window
        meta = gdf.attrs["engine_meta"]
        assert meta["window_source"] == "training_chip_size" and meta["training_chip_size"] == 64
        assert meta["resize_factor"] == pytest.approx(12.5)
        assert not [r for r in caplog.records if r.name == "agribound.engines.geoai_field"]
        # an explicit different window and a confidence below box_score_thresh warn
        fake_geoai.clear()
        GeoAIEngine().delineate(
            raster,
            cfg.merged(
                engine_params={
                    "checkpoint_path": str(ckpt),
                    "window_size": 128,
                    "confidence_threshold": 0.01,
                }
            ),
        )
        assert fake_geoai["window_size"] == 128 and fake_geoai["batch_size"] == 1
        assert "differs from the 64 px chips" in caplog.text
        assert "below Mask R-CNN's box_score_thresh" in caplog.text

    def test_clean_instance_mask_keeps_the_seam_join(self, tmp_path, s2_raster, monkeypatch):
        train = pytest.importorskip("geoai.train")
        raster_utils = pytest.importorskip("geoai.utils.raster")
        from agribound.engines.geoai_field import GeoAIEngine

        def split_field(input_path, output_path, model_path, **kwargs):
            with rasterio.open(input_path) as src:
                profile = src.profile.copy()
            inst = np.zeros((96, 96), dtype=np.uint32)
            inst[20:40, 40:64] = 1  # one field cut at the 64 px window edge
            inst[20:40, 64:80] = 2
            profile.update(count=1, dtype="uint32")
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(inst, 1)
            base, ext = str(output_path).rsplit(".", 1)
            profile.update(dtype="float32")
            with rasterio.open(f"{base}_score.{ext}", "w", **profile) as dst:
                dst.write((inst > 0).astype(np.float32) * 0.8, 1)
            return {}

        cleaned_inputs = []

        def fake_clean(input_path, output_path=None, **kwargs):
            cleaned_inputs.append(input_path)
            with rasterio.open(input_path) as src:
                profile, arr = src.profile.copy(), src.read(1)
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(arr, 1)
            return output_path

        monkeypatch.setattr(train, "instance_segmentation", split_field)
        monkeypatch.setattr(raster_utils, "clean_instance_mask", fake_clean)
        ckpt = tmp_path / "best_model.pth"
        ckpt.write_bytes(b"weights")
        _data.write_training_meta(ckpt, {"engine": "geoai", "chip_size": 64})
        cfg = _s2_config(
            tmp_path,
            "geoai",
            engine_params={"checkpoint_path": str(ckpt), "clean_instance_mask": True},
        )
        gdf = GeoAIEngine().delineate(s2_raster[0], cfg)
        # the seam-merged raster is cleaned, not the raw one, so the field stays whole
        assert len(cleaned_inputs) == 1 and "_seams16_gap2" in cleaned_inputs[0]
        assert len(gdf) == 1
        assert gdf.attrs["engine_meta"]["n_instances_merged_at_seams"] == 1

    def test_no_checkpoint_never_runs_a_model(self, tmp_path, s2_raster, fake_geoai):
        from agribound.engines.geoai_field import GeoAIEngine

        with pytest.raises(RuntimeError, match="none is published"):
            GeoAIEngine().delineate(s2_raster[0], _s2_config(tmp_path, "geoai"))
        assert "input" not in fake_geoai


class TestFinetuneGeoAI:
    def test_instance_labels_best_iou_and_early_stopping(self, monkeypatch, s2_chips):
        torch = pytest.importorskip("torch")

        train = pytest.importorskip("geoai.train")
        from agribound.engines.finetune import _geoai

        cfg, td, _ = s2_chips("geoai")
        cfg = cfg.merged(
            fine_tune_epochs=10, device="mps", engine_params={"early_stopping_patience": 2}
        )
        datasets = []
        real_ds = train.ObjectDetectionDataset

        def recording_ds(images, labels, **kwargs):
            datasets.append((images, labels, kwargs))
            return real_ds(images, labels, **kwargs)

        ious = iter([0.2, 0.6, 0.5, 0.55, 0.9])
        devices = []
        monkeypatch.setattr(train, "ObjectDetectionDataset", recording_ds)
        monkeypatch.setattr(
            train,
            "get_instance_segmentation_model",
            lambda num_classes, num_channels, pretrained: torch.nn.Linear(2, 2),
        )
        monkeypatch.setattr(
            train,
            "train_one_epoch",
            lambda model, opt, loader, device, epoch, **kw: devices.append(device) or 1.0,
        )
        monkeypatch.setattr(
            train,
            "evaluate",
            lambda model, loader, device, use_mask_iou: {"loss": 0.5, "IoU": next(ious)},
        )
        best = Path(_geoai._finetune_geoai(td, cfg))
        assert best.name == "best_model.pth" and best.is_file()
        meta = _data.read_training_meta(best)
        assert meta["best_epoch"] == 2 and meta["epochs_run"] == 4  # stopped after 2 stale
        assert [h["val_iou"] for h in meta["history"]] == [0.2, 0.6, 0.5, 0.55]
        # instance-id masks and the prepared split are used; no re-split
        (tr_img, tr_lbl, tr_kw), (va_img, va_lbl, va_kw) = datasets
        assert all("/instances/" in p for p in tr_lbl) and all(
            "/val_instances/" in p for p in va_lbl
        )
        assert tr_kw["instance_labels"] is True and va_kw["instance_labels"] is True
        n_train = len(_data.split_files(td, "train", "images"))
        assert len(tr_img) == n_train == meta["n_train"]
        assert set(devices) == {"cpu"}
        # the chip size that inference uses as its default window is recorded
        assert meta["chip_size"] == 64

    def test_failed_fine_tune_warns(self, monkeypatch, s2_chips, caplog):
        """Regression: a best validation IoU of 0.033 produced output without any warning."""
        torch = pytest.importorskip("torch")
        train = pytest.importorskip("geoai.train")
        from agribound.engines.finetune import _geoai

        cfg, td, _ = s2_chips("geoai")
        cfg = cfg.merged(fine_tune_epochs=3, device="cpu", engine_params={})
        ious = iter([0.030, 0.033, 0.020])
        monkeypatch.setattr(
            train,
            "get_instance_segmentation_model",
            lambda num_classes, num_channels, pretrained: torch.nn.Linear(2, 2),
        )
        monkeypatch.setattr(train, "train_one_epoch", lambda *a, **k: 1.0)
        monkeypatch.setattr(
            train,
            "evaluate",
            lambda model, loader, device, use_mask_iou: {"loss": 1, "IoU": next(ious)},
        )
        with caplog.at_level("WARNING", logger="agribound"):
            best = Path(_geoai._finetune_geoai(td, cfg))
        meta = _data.read_training_meta(best)
        assert meta["best_val_iou"] == pytest.approx(0.033)
        low = [w for w in meta["warnings"] if "best validation IoU of only 0.033" in w]
        assert len(low) == 1 and "below 0.1" in low[0]
        assert any("best validation IoU of only 0.033" in r.message for r in caplog.records)
        if meta["n_train"] < _geoai.FEW_TRAIN_CHIPS_WARNING:
            assert any("smoke test" in w for w in meta["warnings"])

    def test_repo_id_is_rejected(self, tmp_path):
        pytest.importorskip("geoai.train")
        from agribound.engines.finetune import _geoai

        cfg = _s2_config(tmp_path, "geoai", engine_params={"repo_id": "me/fields"})
        with pytest.raises(ValueError, match="starts from the COCO"):
            _geoai._finetune_geoai(tmp_path, cfg)


# ---------------------------------------------------------------------------
# DINOv3
# ---------------------------------------------------------------------------


class TestDINOv3Options:
    def test_model_aliases(self, tmp_path):
        from agribound.engines.dinov3 import resolve_dinov3_model

        assert resolve_dinov3_model("large") == "dinov3_vitl16"
        assert resolve_dinov3_model(None) == "dinov3_vitl16"
        with pytest.raises(ValueError, match="weights_path"):
            resolve_dinov3_model("small")
        with pytest.raises(ValueError, match="weights_path"):
            resolve_dinov3_model("dinov3_vitb16")
        assert resolve_dinov3_model("small", weights_path="w.pth") == "dinov3_vits16"
        with pytest.raises(ValueError, match="Unknown DINOv3"):
            resolve_dinov3_model("vit_huge")

    def test_lora_coupling(self):
        from agribound.engines.dinov3 import resolve_lora

        assert resolve_lora({}) == (False, False)  # full fine-tuning
        assert resolve_lora({"use_lora": True}) == (True, True)
        assert resolve_lora({"use_lora": True, "freeze_backbone": True}) == (True, True)
        assert resolve_lora({"freeze_backbone": True}) == (False, True)
        with pytest.raises(ValueError, match="requires freeze_backbone=True"):
            resolve_lora({"use_lora": True, "freeze_backbone": False})


class TestDINOv3Engine:
    @pytest.fixture
    def ckpt(self, tmp_path):
        torch = pytest.importorskip("torch")

        path = tmp_path / "dinov3.ckpt"
        torch.save(
            {
                "hyper_parameters": {
                    "model_name": "dinov3_vitl16",
                    "num_classes": 3,
                    "use_lora": True,
                    "freeze_backbone": True,
                    "lora_rank": 4,
                }
            },
            path,
        )
        _data.write_training_meta(path, {"boundary_erosion": 1, "trainable_params": {"total": 7}})
        return path

    @pytest.fixture
    def fake_segment(self, monkeypatch):
        mod = pytest.importorskip("geoai.dinov3_finetune")
        calls = {}

        def fake(input_path, output_path, checkpoint_path, **kwargs):
            with rasterio.open(input_path) as src:
                calls["input"] = src.read()
                profile = src.profile.copy()
            calls.update(kwargs, checkpoint_path=checkpoint_path)
            pred = np.zeros(calls["input"].shape[1:], dtype=np.uint8)
            pred[:] = 2
            pred[30:60, 30:60] = 1  # a 30 x 30 px interior
            pred[0:4, 0:4] = 1  # inside the nodata corner
            profile.update(count=1, dtype="uint8")
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(pred, 1)

        monkeypatch.setattr(mod, "dinov3_segment_geotiff", fake)
        return calls

    def test_input_hparams_masking_and_dilation(self, tmp_path, s2_raster, ckpt, fake_segment):
        from agribound.engines.dinov3 import DINOv3Engine

        raster, data = s2_raster
        cfg = _s2_config(
            tmp_path,
            "dinov3",
            min_field_area_m2=0,
            engine_params={"checkpoint_path": str(ckpt), "dinov3_model": "large"},
        )
        gdf = DINOv3Engine().delineate(raster, cfg)
        inp = fake_segment["input"]
        assert inp.dtype == np.float32 and inp.max() <= 1.0
        lows, highs = _data.scene_stretch_bounds(raster, [4, 3, 2])
        expected = _data.apply_stretch(data[[3, 2, 1]], lows, highs).astype(np.float32) / 255
        np.testing.assert_allclose(inp, expected)
        assert fake_segment["model_name"] == "dinov3_vitl16" and fake_segment["num_classes"] == 3
        # no training chip size recorded: geoai's 512 px window, capped at the 96 px
        # raster, so no padding is needed
        assert fake_segment["window_size"] == 96 and fake_segment["overlap"] == 48
        # the nodata corner prediction was removed; the 30 x 30 px interior was grown
        # by the training boundary_erosion (1 px, city-block) into the boundary class:
        # 32 x 32 px minus the 4 corner pixels
        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].area == pytest.approx((32 * 32 - 4) * 10.0**2)
        meta = gdf.attrs["engine_meta"]
        assert meta["window_capped"] is True and meta["input_padding"] is None
        assert meta["use_lora"] is True and meta["dilate_interior_px"] == 1
        assert meta["trainable_params"] == {"total": 7}
        assert meta["device"] == "cpu" and "cache_reused" not in meta
        json.dumps(meta)
        # A second run reuses the cached segmentation and reports the device
        # that produced it, not the current one.
        fake_segment.clear()
        again = DINOv3Engine().delineate(raster, cfg.merged(device="mps"))
        assert fake_segment == {}
        assert again.attrs["engine_meta"]["device"] == "cpu"
        assert again.attrs["engine_meta"]["cache_reused"] is True

    def _ckpt_with_weights(self, tmp_path, model_name, weights_path):
        import torch

        path = tmp_path / f"{model_name}.ckpt"
        torch.save(
            {
                "hyper_parameters": {
                    "model_name": model_name,
                    "num_classes": 3,
                    "weights_path": weights_path,
                }
            },
            path,
        )
        return path

    def test_checkpoint_weights_path_is_what_geoai_uses(
        self, tmp_path, s2_raster, fake_segment, caplog
    ):
        from agribound.engines.dinov3 import DINOv3Engine

        raster, _ = s2_raster
        init = tmp_path / "vitl16_init.pth"
        init.write_bytes(b"w")
        ckpt = self._ckpt_with_weights(tmp_path, "dinov3_vitl16", str(init))
        cfg = _s2_config(
            tmp_path,
            "dinov3",
            min_field_area_m2=0,
            engine_params={"checkpoint_path": str(ckpt), "weights_path": "/other/w.pth"},
        )
        with caplog.at_level(logging.WARNING, logger="agribound.engines.dinov3"):
            gdf = DINOv3Engine().delineate(raster, cfg)
        # the user's weights_path is not what geoai loads for a .ckpt: warned, not passed
        assert "is not used for inference" in caplog.text
        assert fake_segment["weights_path"] == str(init)
        assert gdf.attrs["engine_meta"]["weights"] == str(init)

    def test_missing_checkpoint_weights(self, tmp_path, s2_raster, fake_segment, caplog):
        from agribound.engines.dinov3 import DINOv3Engine

        raster, _ = s2_raster
        missing = str(tmp_path / "gone.pth")
        # ViT-L/16: geoai initialises from SAT-493M, the checkpoint then replaces
        # every weight -> warning only
        ckpt = self._ckpt_with_weights(tmp_path, "dinov3_vitl16", missing)
        cfg = _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(ckpt)})
        with caplog.at_level(logging.WARNING, logger="agribound.engines.dinov3"):
            DINOv3Engine().delineate(raster, cfg)
        assert "do not exist here" in caplog.text
        assert fake_segment["weights_path"] is None
        # another backbone: the SAT-493M ViT-L/16 weights do not fit -> error
        fake_segment.clear()
        ckpt_b = self._ckpt_with_weights(tmp_path, "dinov3_vitb16", missing)
        with pytest.raises(FileNotFoundError, match="do not fit dinov3_vitb16"):
            DINOv3Engine().delineate(
                raster,
                _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(ckpt_b)}),
            )
        assert fake_segment == {}

    def test_real_geoai_windows_are_never_zero_padded(self, tmp_path, monkeypatch):
        """geoai's own sliding window, fed the padded input, only sees image content."""
        torch = pytest.importorskip("torch")

        mod = pytest.importorskip("geoai.dinov3_finetune")
        from agribound.engines.dinov3 import DINOv3Engine

        class FakeSegmenter(torch.nn.Module):
            patch_size = 16

            def __init__(self):
                super().__init__()
                self.windows = []

            def forward(self, x):
                self.windows.append(x.detach().cpu().numpy().copy())
                logits = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
                logits[:, 1] = 1.0  # field interior everywhere
                return logits

        fake = FakeSegmenter()
        monkeypatch.setattr(
            mod.DINOv3Segmenter, "load_from_checkpoint", staticmethod(lambda *a, **k: fake)
        )
        rng = np.random.default_rng(4)
        data = rng.uniform(300, 3000, (12, 150, 90)).astype(np.float32)  # no nodata
        raster = _write(tmp_path / "s2.tif", data, nodata=float("nan"))
        ckpt = tmp_path / "dinov3.ckpt"
        torch.save({"hyper_parameters": {"model_name": "dinov3_vitl16", "num_classes": 3}}, ckpt)
        _data.write_training_meta(ckpt, {"chip_size": 64, "boundary_erosion": 2})

        def zero_edges(batches):
            # windows whose last row or last column is 0 in every band (zero padding)
            return sum(
                bool((w[:, -1, :] == 0).all() or (w[:, :, -1] == 0).all())
                for batch in batches
                for w in batch
            )

        # geoai on the unpadded RGB input: its partial last windows are zero-padded
        rgb = tmp_path / "rgb.tif"
        _data.write_rgb_input(raster, rgb, [4, 3, 2], unit_float=True)
        mod.dinov3_segment_geotiff(
            str(rgb),
            str(tmp_path / "raw.tif"),
            str(ckpt),
            window_size=64,
            overlap=32,
            device="cpu",
            quiet=True,
        )
        assert zero_edges(fake.windows) > 0
        fake.windows.clear()

        cfg = _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(ckpt)})
        gdf = DINOv3Engine().delineate(raster, cfg)
        meta = gdf.attrs["engine_meta"]
        # 150 x 90 px, 64 px window, stride 32 -> input extended to 160 x 96 px
        assert meta["window_size"] == 64 and meta["overlap"] == 32
        assert meta["window_source"] == "training_chip_size"
        assert meta["input_padded_to"] == [160, 96] and meta["input_padding"] == "reflect"
        assert fake.windows and zero_edges(fake.windows) == 0
        assert all(w.shape[-2:] == (64, 64) for w in fake.windows)
        # the prediction is cropped back to the raster grid
        (seg,) = [
            p
            for p in Path(cfg.get_working_dir()).glob("dinov3_segmentation_*.tif")
            if "_fields_" not in p.name
        ]
        with rasterio.open(seg) as src, rasterio.open(raster) as ref:
            assert (src.height, src.width) == (150, 90) and src.transform == ref.transform
        assert len(gdf) == 1 and gdf.geometry.iloc[0].area == pytest.approx(150 * 90 * 100.0)

    def test_plan_dinov3_windows(self):
        from agribound.engines.dinov3 import plan_dinov3_windows

        plan = plan_dinov3_windows(600, 700, training_chip_size=256)
        assert (plan["window_size"], plan["overlap"]) == (256, 128)
        # extents window + k * stride that are >= the raster size
        assert (plan["padded_height"], plan["padded_width"]) == (640, 768)
        # geoai: max(1, ceil((extent - overlap) / stride)) windows starting at i * stride
        for extent in (plan["padded_height"], plan["padded_width"]):
            n = max(1, -(-(extent - 128) // 128))
            assert (n - 1) * 128 + 256 == extent
        assert plan_dinov3_windows(600, 700, training_chip_size=200)["window_size"] == 208
        assert plan_dinov3_windows(600, 700)["window_size"] == 512
        small = plan_dinov3_windows(150, 90, training_chip_size=256)
        assert small["window_size"] == 160 and small["capped"] is True
        assert (small["padded_height"], small["padded_width"]) == (160, 160)
        # capped: one window covers the raster, so an overlap too large for it is moot
        moot = plan_dinov3_windows(150, 90, window_size=512, overlap=256)
        assert (moot["window_size"], moot["overlap"], moot["padded_height"]) == (160, 80, 160)
        with pytest.raises(ValueError, match="multiple of the 16 px"):
            plan_dinov3_windows(600, 700, window_size=200)
        with pytest.raises(ValueError, match="overlap must be in"):
            plan_dinov3_windows(600, 700, window_size=256, overlap=256)

    def test_requires_ckpt_file(self, tmp_path, s2_raster, fake_segment):
        from agribound.engines.dinov3 import DINOv3Engine

        raster, _ = s2_raster
        with pytest.raises(RuntimeError, match="fine-tuned checkpoint"):
            DINOv3Engine().delineate(raster, _s2_config(tmp_path, "dinov3"))
        pth = tmp_path / "state.pth"
        pth.write_bytes(b"x")
        with pytest.raises(ValueError, match="not a Lightning .ckpt"):
            DINOv3Engine().delineate(
                raster, _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(pth)})
            )


class TestFinetuneDINOv3:
    def _fake_model(self, best_path, attached=True):
        import torch

        model = types.SimpleNamespace()
        model.backbone = torch.nn.Linear(4, 4)
        model.decoder = torch.nn.Linear(4, 2)
        for p in model.backbone.parameters():
            p.requires_grad = False

        def parameters():
            yield from model.backbone.parameters()
            yield from model.decoder.parameters()

        model.parameters = parameters
        if attached:
            cb = types.SimpleNamespace(best_model_path=str(best_path), best_model_score=0.25)
            model.trainer = types.SimpleNamespace(checkpoint_callback=cb)
        return model

    def test_uses_prepared_split_and_best_checkpoint(self, monkeypatch, s2_chips):
        mod = pytest.importorskip("geoai.dinov3_finetune")
        from agribound.engines.finetune import _dinov3

        cfg, td, _ = s2_chips("dinov3", use_lora=True, lora_rank=8)
        seen = {}

        def fake_train(**kwargs):
            seen.update(kwargs)
            models = Path(kwargs["output_dir"]) / "models"
            models.mkdir(parents=True, exist_ok=True)
            (models / "last.ckpt").write_bytes(b"last")
            best = models / "dinov3_seg_epoch=00_val_loss=0.2500.ckpt"
            best.write_bytes(b"best")
            return self._fake_model(best)

        monkeypatch.setattr(mod, "train_dinov3_segmentation", fake_train)
        best = Path(_dinov3._finetune_dinov3(td, cfg))
        assert best.name.startswith("dinov3_seg_epoch=00")
        # geoai's unused last.ckpt (3.7 GB for ViT-L) is deleted; the best one is kept.
        assert not (best.parent / "last.ckpt").exists() and best.read_bytes() == b"best"
        assert seen["use_lora"] is True and seen["freeze_backbone"] is True
        assert seen["lora_rank"] == 8 and seen["devices"] == 1 and seen["ignore_index"] == 255
        assert len(seen["val_dataset"]) == len(_data.split_files(td, "val", "images"))
        assert len(seen["train_dataset"]) == len(_data.split_files(td, "train", "images"))
        meta = _data.read_training_meta(best)
        assert meta["trainable_params"]["backbone"] == 0
        assert meta["trainable_params"]["decoder"] == 10
        assert meta["best_model_score"] == 0.25

    def test_best_checkpoint_fallback_and_errors(self, tmp_path):
        from agribound.engines.finetune._dinov3 import _best_checkpoint

        models = tmp_path / "models"
        models.mkdir()
        (models / "last.ckpt").write_bytes(b"l")
        only = models / "dinov3_seg_epoch=03.ckpt"
        only.write_bytes(b"b")

        class Detached:
            @property
            def trainer(self):
                raise RuntimeError("not attached")

        assert _best_checkpoint(Detached(), models) == only.resolve()
        (models / "dinov3_seg_epoch=05.ckpt").write_bytes(b"c")
        with pytest.raises(RuntimeError, match="did not record a best checkpoint"):
            _best_checkpoint(Detached(), models)


# ---------------------------------------------------------------------------
# Prithvi (no terratorch needed)
# ---------------------------------------------------------------------------


class TestPrithviHelpers:
    @pytest.mark.parametrize(
        ("name", "registry"),
        [
            ("Prithvi-EO-2.0-tiny-TL", "prithvi_eo_v2_tiny_tl"),
            ("Prithvi-EO-2.0-100M-TL", "prithvi_eo_v2_100_tl"),
            ("Prithvi-EO-2.0-300M", "prithvi_eo_v2_300"),
            ("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL", "prithvi_eo_v2_300_tl"),
            ("Prithvi-EO-2.0-600M", "prithvi_eo_v2_600"),
            ("Prithvi-EO-2.0-600M-TL", "prithvi_eo_v2_600_tl"),
            ("terratorch_prithvi_eo_v2_600_tl", "prithvi_eo_v2_600_tl"),
            (None, "prithvi_eo_v2_300_tl"),
        ],
    )
    def test_registry_names(self, name, registry):
        from agribound.engines.prithvi import resolve_prithvi_model

        assert resolve_prithvi_model(name) == registry

    def test_unknown_model(self):
        from agribound.engines.prithvi import resolve_prithvi_model

        with pytest.raises(ValueError, match="Unknown Prithvi model"):
            resolve_prithvi_model("Prithvi-EO-3.0")

    def test_select_indices_and_patch_sizes(self):
        from agribound.engines.prithvi import PRITHVI_ARCH, select_indices

        assert select_indices(12) == [2, 5, 8, 11]
        assert select_indices(24) == [5, 11, 17, 23]
        assert select_indices(32) == [7, 15, 23, 31]
        assert PRITHVI_ARCH["prithvi_eo_v2_600_tl"][1] == 14
        assert PRITHVI_ARCH["prithvi_eo_v2_300_tl"][1] == 16

    def test_upernet_mps_rule_and_device_choice(self, caplog):
        from agribound.engines.prithvi import (
            upernet_coarsest_side,
            upernet_device,
            upernet_mps_compatible,
        )

        # 224 px: 14 tokens -> max-pooled to 7 (patch 16); 16 tokens -> 8 (patch 14)
        assert upernet_coarsest_side(224, 16) == 7 and upernet_coarsest_side(224, 14) == 8
        assert upernet_coarsest_side(200, 16) == 7  # padded to 224 (2 x patch multiple)
        assert upernet_coarsest_side((64, 160), 16) == (2, 5)
        assert not upernet_mps_compatible(224, 16) and not upernet_mps_compatible(224, 14)
        assert upernet_mps_compatible(192, 16) and upernet_mps_compatible(384, 16)
        assert upernet_mps_compatible(576, 16)  # side 18
        assert upernet_mps_compatible(168, 14) and not upernet_mps_compatible(160, 16)
        # a 1 px coarsest map divides every pool scale; 2 and 3 px maps do not
        assert upernet_mps_compatible(32, 16) and upernet_mps_compatible((20, 32), 16)
        assert not upernet_mps_compatible(64, 16) and not upernet_mps_compatible(96, 16)
        assert not upernet_mps_compatible((32, 224), 16)  # 1 x 7: mixed smaller/larger
        with caplog.at_level(logging.WARNING, logger="agribound.engines.prithvi"):
            assert upernet_device("mps", 224, 16, "segmentation") == "cpu"
        assert "runs on CPU instead of MPS" in caplog.text and "192 px" in caplog.text
        assert "224 x 224 px inputs" in caplog.text and "7 x 7 px" in caplog.text
        caplog.clear()
        assert upernet_device("mps", 192, 16, "segmentation") == "mps"
        assert upernet_device("mps", (32, 32), 16, "segmentation") == "mps"
        assert upernet_device("cuda:0", 224, 16, "segmentation") == "cuda:0"
        assert upernet_device("cpu", 224, 16, "segmentation") == "cpu"
        assert caplog.text == ""

    def test_mps_adaptive_pool_rule(self):
        from agribound.engines.prithvi import mps_adaptive_pool_supported as ok

        assert ok(6, 6, 6) and ok(12, 12, 3) and ok(12, 18, 6)
        assert not ok(7, 7, 6) and not ok(12, 14, 6)  # larger but not divisible
        assert ok(1, 1, 6) and ok(2, 3, 6) and ok(3, 3, 6)  # smaller and dividing
        assert not ok(4, 4, 6) and not ok(2, 2, 3)  # smaller but not dividing
        assert not ok(1, 7, 6)  # one side smaller, one larger
        assert not ok(6, 3, 6)  # height == output: both sides must be divisible

    def test_mid_date(self, tmp_path):
        from agribound.engines.prithvi import composite_mid_date

        assert composite_mid_date(_s2_config(tmp_path, "prithvi", year=2023)) == (2023, 183)
        assert composite_mid_date(_s2_config(tmp_path, "prithvi", year=2024)) == (2024, 183)
        cfg = _s2_config(tmp_path, "prithvi", date_range=("2023-11-01", "2024-02-29"))
        assert composite_mid_date(cfg) == (2023, 365)  # 2023-12-31

    def test_normalisation_uses_prithvi_stats_without_rescaling(self):
        from agribound.engines.prithvi import PRITHVI_MEAN, PRITHVI_STD, PrithviEngine

        data = np.stack(
            [
                np.full((2, 2), m + s, np.float32)
                for m, s in zip(PRITHVI_MEAN, PRITHVI_STD, strict=True)
            ]
        )
        data[:, 0, 0] = np.nan
        norm, valid = PrithviEngine._normalise(data, "hls", None, None)
        assert not valid[0, 0] and valid[1, 1]
        np.testing.assert_allclose(norm[:, 1, 1], 1.0)  # (mean + std - mean) / std
        assert (norm[:, 0, 0] == 0).all()

    def test_token_interpolation(self):
        from agribound.engines.prithvi import interpolate_token_rows, interpolate_tokens

        tokens = np.arange(3 * 4, dtype=np.float32).reshape(3, 4, 1)  # value = 4*row + col
        # pixel centres of token (1, 2) with patch 16: y = 16 + 7.5, x = 32 + 7.5
        assert interpolate_tokens(tokens, 16, np.array([23]), np.array([39]))[
            0, 0
        ] == pytest.approx(4 * (23.5 / 16 - 0.5) + (39.5 / 16 - 0.5))
        # clamped at the border
        assert interpolate_tokens(tokens, 16, np.array([0]), np.array([0]))[0, 0] == 0.0
        rows = interpolate_token_rows(tokens, 16, np.arange(48), 64)
        ys, xs = np.meshgrid(np.arange(48), np.arange(64), indexing="ij")
        pts = interpolate_tokens(tokens, 16, ys.ravel(), xs.ravel()).reshape(48, 64, 1)
        np.testing.assert_allclose(rows, pts, rtol=1e-6)

    def test_extract_token_map_with_fake_encoder(self):
        torch = pytest.importorskip("torch")

        from agribound.engines.prithvi import extract_token_map

        seen = []

        class FakeEncoder(torch.nn.Module):
            def forward(self, x, temporal_coords=None, location_coords=None):
                seen.append((tuple(x.shape), temporal_coords, location_coords))
                b, _, h, w = x.shape
                t = h // 16
                # token value = mean of the tile (identifies which tile was encoded)
                val = x.mean(dim=(1, 2, 3)).view(b, 1, 1).expand(b, 1 + t * t, 2)
                return [val.clone(), val.clone()]

        height, width = 40, 70  # 2 x 3 tiles of 32 px (extended to 64 x 96)
        data = np.zeros((6, height, width), dtype=np.float32)
        data[:, :, 32:64] = 1.0
        data[:, 30:, :] += 0.5
        reads = []

        def read_rows(row0, nrows):
            reads.append((row0, nrows))
            return data[:, row0 : row0 + nrows], np.ones((nrows, width), dtype=bool)

        tokens, valid = extract_token_map(
            read_rows,
            height,
            width,
            FakeEncoder(),
            tile=32,
            patch=16,
            batch_size=2,
            device="cpu",
            coords={"temporal_coords": [2023.0, 183.0], "location_coords": [-30.0, 150.0]},
            layer=-1,
            torch_module=torch,
        )
        assert tokens.shape == (4, 6, 2) and valid.shape == (40, 70) and valid.all()
        # every tile, including those past the raster edge, is the raster extended by
        # mirror reflection (numpy "reflect"), never constant padding
        extended = np.pad(data, ((0, 0), (0, 24), (0, 26)), mode="reflect")
        for r in range(2):
            for c in range(3):
                tile_mean = extended[:, r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32].mean()
                assert tokens[2 * r, 2 * c, 0] == pytest.approx(tile_mean, rel=1e-6)
        assert tokens[0, 4, 0] > 0  # zero padding would have given 0 here
        # the last strip reads the rows it mirrors (rows 15..39 for rows 40..63)
        assert reads == [(0, 32), (15, 25)]
        assert all(s[0][1:] == (6, 32, 32) for s in seen)
        assert tuple(seen[0][1].shape) == (2, 1, 2) and tuple(seen[0][2].shape) == (2, 2)

    def test_fit_kmeans(self):
        from agribound.engines.prithvi import fit_kmeans

        rng = np.random.default_rng(0)
        sample = np.concatenate([rng.normal(c, 0.1, (200, 3)) for c in (0, 5, 10, 15, 20)])
        km, k, score = fit_kmeans(sample, "auto", seed=1)
        assert k == 5 and score > 0.8
        km2, k2, score2 = fit_kmeans(sample, 3, seed=1)
        assert k2 == 3 and score2 is None
        with pytest.raises(ValueError):
            fit_kmeans(sample, 1, seed=1)
        with pytest.raises(ValueError):
            fit_kmeans(sample, "many", seed=1)


class TestPrithviModes:
    def test_segment_without_checkpoint_raises(self, tmp_path, monkeypatch):
        from agribound.engines.prithvi import PrithviEngine

        raster = _write(tmp_path / "hls.tif", np.ones((7, 16, 16), np.float32) * 1000)
        cfg = _s2_config(tmp_path, "prithvi", source="hls", engine_params={"mode": "segment"})
        with pytest.raises(RuntimeError, match="needs a fine-tuned checkpoint"):
            PrithviEngine().delineate(raster, cfg)
        with pytest.raises(ValueError, match="Unknown Prithvi mode"):
            PrithviEngine().delineate(raster, cfg.merged(engine_params={"mode": "cluster"}))

    def test_checkpoint_selects_segment_mode(self, tmp_path, monkeypatch):
        from agribound.engines.prithvi import PrithviEngine

        raster = _write(tmp_path / "hls.tif", np.ones((7, 16, 16), np.float32) * 1000)
        called = []
        monkeypatch.setattr(PrithviEngine, "_segment_mode", lambda self, r, c, k: called.append(k))
        cfg = _s2_config(
            tmp_path, "prithvi", source="hls", engine_params={"checkpoint_path": "x.ckpt"}
        )
        PrithviEngine().delineate(raster, cfg)
        assert called == ["x.ckpt"]

    def test_pca_mode_is_seeded(self, tmp_path):
        from agribound.engines.prithvi import PrithviEngine

        rng = np.random.default_rng(3)
        data = rng.uniform(500, 600, (12, 60, 60)).astype(np.float32)
        data[:, 10:40, 10:40] += 2000
        data[:, :5, :5] = np.nan
        raster = _write(tmp_path / "s2.tif", data, nodata=float("nan"))
        cfg = _s2_config(
            tmp_path, "prithvi", min_field_area_m2=0, engine_params={"mode": "pca", "n_clusters": 2}
        )
        gdf = PrithviEngine().delineate(raster, cfg)
        assert len(gdf) >= 1 and gdf.attrs["engine_meta"]["n_clusters"] == 2
        json.dumps(gdf.attrs["engine_meta"])
        other = PrithviEngine().delineate(
            raster, cfg.merged(output_path=str(tmp_path / "b" / "f.gpkg"))
        )
        assert sorted(gdf.geometry.area.round(3)) == sorted(other.geometry.area.round(3))

    def test_local_source_needs_value_scale(self, tmp_path):
        from agribound.engines.prithvi import PrithviEngine

        data = np.ones((6, 8, 8), np.float32)
        with pytest.raises(ValueError, match="value_scale"):
            PrithviEngine._normalise(data, "local", None, None)
        norm, _ = PrithviEngine._normalise(data * 0.1, "local", "unit", None)
        assert norm.shape == (6, 8, 8)


def test_prithvi_task_kwargs():
    from agribound.engines.finetune._prithvi import build_task_kwargs

    cfg = AgriboundConfig(
        source="hls",
        year=2023,
        study_area="bbox:150,-31,150.1,-30.9",
        gee_project="p",
        engine="prithvi",
        lulc_filter=False,
        engine_params={"model_name": "Prithvi-EO-2.0-600M-TL"},
    )
    kw = build_task_kwargs(cfg)
    args = kw["model_args"]
    assert kw["model_factory"] == "EncoderDecoderFactory" and kw["ignore_index"] == 255
    assert args["backbone"] == "prithvi_eo_v2_600_tl" and args["backbone_num_frames"] == 1
    assert args["backbone_bands"] == ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
    assert [n["name"] for n in args["necks"]] == [
        "SelectIndices",
        "ReshapeTokensToImage",
        "LearnedInterpolateToPyramidal",
    ]
    assert args["necks"][0]["indices"] == [7, 15, 23, 31]
    assert args["decoder"] == "UperNetDecoder" and "decoder_scale_modules" not in args
    assert args["num_classes"] == 3 and "peft_config" not in args
    lora = build_task_kwargs(cfg.merged(engine_params={"use_lora": True, "lora_rank": 8}))
    peft = lora["model_args"]["peft_config"]
    assert peft["method"] == "LORA" and peft["replace_qkv"] == "qkv"
    assert peft["peft_config_kwargs"]["r"] == 8
    assert peft["peft_config_kwargs"]["target_modules"] == ["qkv.q_linear", "qkv.v_linear"]
    with pytest.raises(ValueError, match="cannot be combined"):
        build_task_kwargs(cfg.merged(engine_params={"use_lora": True, "freeze_backbone": True}))


# ---------------------------------------------------------------------------
# Prefetch
# ---------------------------------------------------------------------------


def test_prithvi_prefetch(monkeypatch, tmp_path):
    import huggingface_hub

    from agribound.engines.prithvi import PrithviEngine

    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo_id, filename: calls.append((repo_id, filename)) or f"/cache/{filename}",
    )
    cfg = _s2_config(tmp_path, "prithvi", engine_params={"model_name": "Prithvi-EO-2.0-100M-TL"})
    assert PrithviEngine.prefetch(cfg) == ["/cache/Prithvi_EO_V2_100M_TL.pt"]  # embed mode
    assert calls == [("ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL", "Prithvi_EO_V2_100M_TL.pt")]
    # segment inference and pca mode need no pre-trained weights
    ckpt = tmp_path / "m.ckpt"
    ckpt.write_bytes(b"x")
    calls.clear()
    seg = cfg.merged(engine_params={"checkpoint_path": str(ckpt)})
    assert PrithviEngine.prefetch(seg) == [str(ckpt.resolve())] and calls == []
    assert PrithviEngine.prefetch(cfg.merged(engine_params={"mode": "pca"})) == []
    assert calls == []
    # fine-tuning needs them (default model 300M-TL)
    ref = tmp_path / "ref.gpkg"
    gpd.GeoDataFrame(geometry=[box(X0, Y0 - 100, X0 + 100, Y0)], crs=UTM).to_file(ref)
    ft = cfg.merged(fine_tune=True, reference_boundaries=str(ref), engine_params={})
    assert PrithviEngine.prefetch(ft) == ["/cache/Prithvi_EO_V2_300M_TL.pt"]


def test_dinov3_prefetch(monkeypatch, tmp_path):
    import huggingface_hub

    torch = pytest.importorskip("torch")

    from agribound.engines.dinov3 import DINOv3Engine

    hub = tmp_path / "hub"
    (hub / "facebookresearch_dinov3_main").mkdir(parents=True)
    listed = []
    monkeypatch.delenv("DINOV3_LOCATION", raising=False)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))
    monkeypatch.setattr(torch.hub, "list", lambda repo, **kw: listed.append(repo) or [])
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo_id, filename: f"/hf/{repo_id}/{filename}"
    )
    paths = DINOv3Engine.prefetch(_s2_config(tmp_path, "dinov3"))
    assert listed == ["facebookresearch/dinov3"]
    assert paths == [
        str(hub / "facebookresearch_dinov3_main"),
        "/hf/giswqs/geoai/dinov3_vitl16_sat493m.pth",
    ]
    monkeypatch.setenv("DINOV3_LOCATION", str(hub / "facebookresearch_dinov3_main"))
    listed.clear()
    DINOv3Engine.prefetch(_s2_config(tmp_path, "dinov3"))
    assert listed == []


def test_geoai_prefetch(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

    from agribound.engines.geoai_field import GeoAIEngine

    fetched = []
    monkeypatch.setattr(
        type(MaskRCNN_ResNet50_FPN_Weights.DEFAULT),
        "get_state_dict",
        lambda self, *a, **k: fetched.append(self.url) or {},
    )
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    paths = GeoAIEngine.prefetch(_s2_config(tmp_path, "geoai"))
    assert fetched and paths[0].startswith(str(tmp_path / "checkpoints"))
    assert paths[0].endswith(".pth")


# ---------------------------------------------------------------------------
# terratorch (GFM environment only)
# ---------------------------------------------------------------------------


def test_terratorch_tables_match_upstream():
    pytest.importorskip("terratorch")
    from terratorch.models.backbones import prithvi_vit

    from agribound.engines.prithvi import PRITHVI_ARCH, PRITHVI_MEAN, PRITHVI_STD, PRITHVI_WEIGHTS

    for name, (depth, patch) in PRITHVI_ARCH.items():
        cfg = prithvi_vit.prithvi_cfgs[name]
        assert cfg["depth"] == depth and cfg["patch_size"][-1] == patch
        upstream = prithvi_vit.pretrained_weights[name]
        assert PRITHVI_WEIGHTS[name] == (upstream["hf_hub_id"], upstream["hf_hub_filename"])
    assert PRITHVI_MEAN == prithvi_vit.PRITHVI_V2_MEAN
    assert PRITHVI_STD == prithvi_vit.PRITHVI_V2_STD


def test_terratorch_tiny_encoder_forward():
    pytest.importorskip("terratorch")
    import torch
    from terratorch.registry import BACKBONE_REGISTRY

    from agribound.engines.prithvi import PRITHVI_HLS_BANDS, extract_token_map

    model = BACKBONE_REGISTRY.build(
        "prithvi_eo_v2_tiny_tl", pretrained=False, bands=PRITHVI_HLS_BANDS, num_frames=1
    ).eval()
    assert model.patch_embed.patch_size[-1] == 16 and len(model.blocks) == 12
    data = np.random.default_rng(0).normal(size=(6, 50, 60)).astype(np.float32)
    tokens, _ = extract_token_map(
        lambda r, n: (data[:, r : r + n], np.ones((n, 60), bool)),
        50,
        60,
        model,
        tile=32,
        patch=16,
        batch_size=4,
        device="cpu",
        coords={"temporal_coords": [2023.0, 183.0], "location_coords": [-30.0, 150.0]},
        layer=-1,
        torch_module=torch,
    )
    assert tokens.shape == (4, 4, 192) and np.isfinite(tokens).all()


@pytest.mark.parametrize(
    "tile", [160, 192, 200, 224, 32, 64, 96, (20, 32), (64, 160), (32, 224), (96, 192)]
)
def test_terratorch_upernet_mps_rule_matches_torch(tile):
    """The MPS rule reproduces PyTorch's MPS adaptive-pooling limitation."""
    pytest.importorskip("terratorch")
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("needs Apple MPS")
    from terratorch.tasks import SemanticSegmentationTask

    from agribound.engines.prithvi import upernet_mps_compatible

    cfg = AgriboundConfig(
        source="hls",
        year=2023,
        study_area="bbox:150,-31,150.1,-30.9",
        gee_project="p",
        engine="prithvi",
        lulc_filter=False,
        engine_params={"model_name": "Prithvi-EO-2.0-tiny-TL", "backbone_pretrained": False},
    )
    from agribound.engines.finetune._prithvi import build_task_kwargs

    model = SemanticSegmentationTask(**build_task_kwargs(cfg)).model.eval().to("mps")
    h, w = (tile, tile) if isinstance(tile, int) else tile
    x = torch.zeros(1, 6, h, w, device="mps")
    try:
        with torch.no_grad():
            model(x)
        ran = True
    except RuntimeError as exc:
        assert "Adaptive pool MPS" in str(exc)
        ran = False
    assert ran == upernet_mps_compatible(tile, 16)


def test_terratorch_tiny_finetune_and_segment(s2_chips, caplog):
    pytest.importorskip("terratorch")
    from agribound.engines.finetune._prithvi import _finetune_prithvi
    from agribound.engines.prithvi import PrithviEngine

    cfg, td, raster = s2_chips(
        "prithvi",
        model_name="Prithvi-EO-2.0-tiny-TL",
        backbone_pretrained=False,
        trainer_kwargs={"limit_train_batches": 1, "limit_val_batches": 1},
    )
    # device="mps": 64 px chips and 224 px tiles cannot run UPerNet on MPS -> CPU
    cfg = cfg.merged(fine_tune_epochs=1, n_workers=0, device="mps")
    with caplog.at_level(logging.WARNING):
        ckpt = _finetune_prithvi(td, cfg)
    assert Path(ckpt).is_file() and ckpt.endswith(".ckpt")
    meta = json.loads(Path(ckpt + ".agribound.json").read_text())
    assert meta["model_args"]["backbone"] == "prithvi_eo_v2_tiny_tl"
    assert meta["device"] == "cpu" and meta["requested_device"] == "mps"
    assert meta["backbone_weights"] is None  # backbone_pretrained=False
    gdf = PrithviEngine().delineate(raster, cfg.merged(engine_params={"checkpoint_path": ckpt}))
    meta = gdf.attrs["engine_meta"]
    assert meta["mode"] == "segment" and meta["device"] == "cpu"
    # the 128 px raster fits in one 224 px tile: passed whole, not padded to the tile
    # (the model reflect-pads it to 128 = 4 x 2 x 16; coarsest map 4 px -> CPU)
    assert meta["model_input_size"] == [128, 128] and meta["input_padding"] is None
    assert caplog.text.count("runs on CPU instead of MPS") == 2
    assert "128 x 128 px inputs" in caplog.text
    json.dumps(meta)
    # a side shorter than the tile, the other longer: mirror-padded to the tile
    rng = np.random.default_rng(2)
    wide = _write(
        Path(raster).with_name("wide.tif"),
        rng.uniform(300, 2600, (12, 96, 200)).astype(np.float32),
        nodata=float("nan"),
    )
    seg2 = PrithviEngine().delineate(
        wide,
        cfg.merged(
            device="cpu",
            engine_params={"checkpoint_path": ckpt, "tile_size": 128, "stride": 96},
        ),
    )
    meta2 = seg2.attrs["engine_meta"]
    assert meta2["model_input_size"] == [128, 128]
    assert meta2["input_padding"] == {"mode": "reflect", "rows": 32, "cols": 0}
    with pytest.raises(RuntimeError, match="wrote no checkpoint"):
        _finetune_prithvi(
            td,
            cfg.merged(
                engine_params={**cfg.engine_params, "trainer_kwargs": {"fast_dev_run": True}}
            ),
        )
