"""Tests for the GeoAI, DINOv3 and Prithvi engines and their fine-tuning modules.

Upstream models are replaced by small fakes unless the test says otherwise;
tests marked with ``terratorch`` in their name run only where terratorch is
installed (the GFM environment).
"""

from __future__ import annotations

import json
import logging
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


@pytest.fixture
def fake_backbone(monkeypatch):
    """geoai's DINOv3Segmenter with a tiny backbone instead of torch.hub and the weights.

    Returns the list of inputs the backbone received.
    """
    torch = pytest.importorskip("torch")
    mod = pytest.importorskip("geoai.dinov3_finetune")
    seen = []

    class FakeViT(torch.nn.Module):
        patch_size = 16
        embed_dim = 8

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(3, 8, kernel_size=16, stride=16)
            self.blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(12))

        def get_intermediate_layers(self, x, n, reshape=True, norm=True):
            seen.append(x.detach().clone())
            f = self.proj(x)
            return [f, f, f, f]

    monkeypatch.setattr(
        mod.DINOv3Segmenter, "_load_backbone", staticmethod(lambda name, weights: FakeViT())
    )
    return seen


def _window_model(gamma=4.0):
    """Fake model: interior where band 1 > 0.5, shifted by the column inside the window.

    Its prediction depends on where a pixel lies in the window, so overlapping windows
    disagree, as a real model's do near its window edges.
    """
    torch = pytest.importorskip("torch")

    class WindowModel(torch.nn.Module):
        patch_size = 16

        def forward(self, x):
            b, _, h, w = x.shape
            local = (torch.arange(w, dtype=torch.float32) / w - 0.5)[None, None, :]
            logits = torch.zeros(b, 3, h, w)
            logits[:, 0] = -10.0
            logits[:, 1] = 20.0 * (x[:, 0] - 0.5) + gamma * local
            return logits

    return WindowModel()


def _ramp_input(path, n=320):
    """Padded RGB input whose first band rises from 0 at the top to 1 at the bottom."""
    rgb = np.zeros((3, n, n), np.float32)
    rgb[0] = (np.arange(n, dtype=np.float32) / n)[:, None]
    return _write(path, rgb, res=1.0)


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

    def test_input_normalization(self):
        from agribound.engines.dinov3 import resolve_input_normalization

        # facebookresearch/dinov3 README: the SAT-493M and the LVD-1689M (ImageNet) transforms
        sat = {"name": "sat493m", "mean": [0.430, 0.411, 0.296], "std": [0.213, 0.156, 0.143]}
        assert resolve_input_normalization() == sat
        assert resolve_input_normalization(" SAT493M ") == sat
        assert resolve_input_normalization(None, "/w/dinov3_vitl16_pretrain_sat493m.pth") == sat
        lvd = resolve_input_normalization(None, "/w/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
        assert lvd == {
            "name": "lvd1689m",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
        assert resolve_input_normalization("none") is None
        assert resolve_input_normalization(False) is None
        custom = resolve_input_normalization({"mean": [0.5, 0.5, 0.5], "std": [0.2, 0.2, 0.2]})
        assert custom == {"name": "custom", "mean": [0.5] * 3, "std": [0.2] * 3}
        for bad in (
            "imagenet",
            True,
            {"mean": [0.5] * 3, "std": [0.2, 0.0, 0.2]},
            {"mean": [0.5] * 2, "std": [0.2] * 2},
            {"mean": [0.5] * 3},
        ):
            with pytest.raises(ValueError, match="input_normalization"):
                resolve_input_normalization(bad)

    def test_window_stitching_and_dilation_defaults(self):
        from agribound.engines.dinov3 import resolve_dilate_px, resolve_window_stitching

        assert resolve_window_stitching() == "blend"
        assert resolve_window_stitching(" Crop ") == "crop"
        with pytest.raises(ValueError, match="window_stitching"):
            resolve_window_stitching("average")
        # engine_params, then the training metadata, then the checkpoint, then 2
        assert resolve_dilate_px({"dilate_interior_px": 5}, {"boundary_erosion": 8}, {}) == 5
        assert resolve_dilate_px({}, {"boundary_erosion": 8}, {"boundary_erosion": 3}) == 8
        assert resolve_dilate_px({"boundary_erosion": 4}, {}, {"boundary_erosion": 3}) == 3
        assert resolve_dilate_px({"boundary_erosion": 4}, {}, {}) == 4
        assert resolve_dilate_px({}, {}, {}) == 2

    def test_window_taper_is_a_partition_of_unity(self):
        from agribound.engines.dinov3 import window_taper

        for window, overlap in ((256, 128), (256, 64), (64, 32), (48, 16), (64, 0)):
            taper = window_taper(window, overlap).astype(np.float64)
            assert taper.shape == (window,) and 0 < taper.min() <= taper.max() <= 1.0
            stride = window - overlap
            total = np.zeros(window + 3 * stride)
            for k in range(4):
                total[k * stride : k * stride + window] += taper
            # wherever two windows overlap, and in each window's centre, the weights sum to 1
            np.testing.assert_allclose(total[overlap : 3 * stride + window - overlap], 1.0)
        # half-window overlap: the Hann window
        hann = np.sin(np.pi * (np.arange(256) + 0.5) / 256) ** 2
        np.testing.assert_allclose(window_taper(256, 128), hann, atol=1e-6)
        assert (window_taper(64, 0) == 1).all() and (window_taper(256, 64)[64:192] == 1).all()


class TestDINOv3Stitching:
    @staticmethod
    def _change_rates(classes, stride, overlap):
        """Class changes between neighbouring columns, at the crop seams and elsewhere."""
        change = (classes[:, 1:] != classes[:, :-1]).mean(axis=0)
        cols = np.arange(1, classes.shape[1])
        seam = (cols % stride == overlap // 2) & (cols > overlap // 2)
        return change[seam], change[~seam]

    def test_blend_leaves_no_window_seams_and_crop_does(self, tmp_path):
        from agribound.engines.dinov3 import segment_raster

        rgb = _ramp_input(tmp_path / "rgb.tif")
        out = {}
        for stitching in ("crop", "blend"):
            path = tmp_path / f"{stitching}.tif"
            stats = segment_raster(
                _window_model(),
                rgb,
                path,
                height=300,
                width=290,
                window_size=64,
                overlap=32,
                batch_size=3,
                stitching=stitching,
            )
            assert stats == {"n_windows": 81, "stitching": stitching}
            with rasterio.open(path) as src:
                assert (src.height, src.width) == (300, 290)
                out[stitching] = src.read(1)
        seams, other = self._change_rates(out["crop"], 32, 32)
        # crop: the iso-line jumps at every seam (a straight north-south class change)
        assert seams.min() > 10 * other.max()
        seams, other = self._change_rates(out["blend"], 32, 32)
        assert seams.max() <= other.max()
        assert set(np.unique(out["blend"])) == {1, 2}

    def test_crop_reproduces_geoai_dinov3_segment_geotiff(self, tmp_path, monkeypatch):
        """``window_stitching="crop"`` gives geoai's output (agribound 1.0.1) on the same input."""
        mod = pytest.importorskip("geoai.dinov3_finetune")
        from agribound.engines.dinov3 import segment_raster

        model = _window_model()
        monkeypatch.setattr(
            mod.DINOv3Segmenter, "load_from_checkpoint", staticmethod(lambda *a, **k: model)
        )
        rgb = _ramp_input(tmp_path / "rgb.tif")
        ckpt = tmp_path / "m.ckpt"
        ckpt.write_bytes(b"x")
        mod.dinov3_segment_geotiff(
            rgb,
            str(tmp_path / "geoai.tif"),
            str(ckpt),
            window_size=64,
            overlap=32,
            batch_size=3,
            device="cpu",
            quiet=True,
        )
        segment_raster(
            model,
            rgb,
            tmp_path / "crop.tif",
            height=300,
            width=290,
            window_size=64,
            overlap=32,
            stitching="crop",
        )
        with rasterio.open(tmp_path / "geoai.tif") as a, rasterio.open(tmp_path / "crop.tif") as b:
            np.testing.assert_array_equal(a.read(1)[:300, :290], b.read(1))

    def test_input_must_be_covered_by_whole_windows(self, tmp_path):
        from agribound.engines.dinov3 import segment_raster

        rgb = _write(tmp_path / "rgb.tif", np.zeros((3, 100, 96), np.float32))
        sizes = {"height": 90, "width": 90}
        with pytest.raises(ValueError, match="not covered exactly"):
            segment_raster(
                _window_model(), rgb, tmp_path / "o.tif", window_size=64, overlap=32, **sizes
            )
        with pytest.raises(ValueError, match="window_stitching"):
            segment_raster(
                _window_model(),
                rgb,
                tmp_path / "o.tif",
                window_size=32,
                overlap=0,
                stitching="mean",
                **sizes,
            )


class TestDINOv3Model:
    def test_normalisation_is_part_of_forward(self, fake_backbone):
        import torch

        from agribound.engines.dinov3 import dinov3_segmenter_class

        cls = dinov3_segmenter_class()
        mean, std = [0.43, 0.411, 0.296], [0.213, 0.156, 0.143]
        common = {"model_name": "dinov3_vitl16", "num_classes": 3, "decoder_features": 8}
        model = cls(
            input_mean=mean, input_std=std, boundary_erosion=8, chip_size=128, **common
        ).eval()
        plain = cls(**common).eval()
        x = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            assert model(x).shape == (2, 3, 32, 32)
            plain(x)
        expected = (x - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[
            None, :, None, None
        ]
        torch.testing.assert_close(fake_backbone[0], expected)
        torch.testing.assert_close(fake_backbone[1], x)  # without statistics: unchanged
        # The statistics are hyper-parameters, not weights: the state dict is geoai's.
        assert model.hparams["input_mean"] == mean and model.hparams["boundary_erosion"] == 8
        assert model.hparams["chip_size"] == 128
        assert set(model.state_dict()) == set(plain.state_dict())
        with pytest.raises(ValueError, match="both"):
            cls(input_mean=mean, **common)

    def test_validation_iou_counts(self):
        torch = pytest.importorskip("torch")
        from agribound.engines.dinov3 import class_overlap_counts, iou_from_counts

        target = torch.tensor([[0, 0, 1, 1], [1, 1, 255, 255]])
        pred = torch.tensor([[0, 1, 1, 1], [1, 0, 2, 0]])
        counts = class_overlap_counts(pred, target, 3, 255)
        # class 0: inter 1, union 3; class 1: inter 3, union 5; class 2 only on ignored pixels
        assert counts.tolist() == [[1, 3, 0], [3, 5, 0]]
        ious = iou_from_counts(counts)
        assert ious[:2] == [pytest.approx(1 / 3), pytest.approx(3 / 5)] and np.isnan(ious[2])


class TestDINOv3Engine:
    @pytest.fixture
    def fake_segmenter(self, monkeypatch):
        """Replace the model class: loading returns a fake model that predicts from band 1.

        Interior (class 1) where the first input band exceeds 0.5, else boundary (2).
        Records the loads and the inputs.
        """
        torch = pytest.importorskip("torch")
        pytest.importorskip("geoai.dinov3_finetune")
        from agribound.engines import dinov3

        calls = {"loads": [], "inputs": []}

        class FakeModel(torch.nn.Module):
            def forward(self, x):
                calls["inputs"].append(x.detach().clone())
                logits = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
                logits[:, 1] = torch.where(x[:, 0] > 0.5, 5.0, -5.0)
                logits[:, 2] = 1.0
                return logits

        class FakeSegmenter:
            @staticmethod
            def load_from_checkpoint(path, map_location=None, **kwargs):
                calls["loads"].append({"path": path, "map_location": map_location, **kwargs})
                return FakeModel()

        monkeypatch.setattr(dinov3, "dinov3_segmenter_class", lambda: FakeSegmenter)
        return calls

    @pytest.fixture
    def block_raster(self, tmp_path):
        """S2-like raster, 96 x 96 px: a bright 30 x 30 px block in red and a NaN corner."""
        data = np.full((12, 96, 96), 500.0, dtype=np.float32)
        data[3, 30:60, 30:60] = 2500.0
        data[:, :8, :8] = np.nan
        return _write(tmp_path / "s2.tif", data, nodata=float("nan")), data

    @staticmethod
    def _ckpt(path, **hparams):
        import torch

        torch.save(
            {"hyper_parameters": {"model_name": "dinov3_vitl16", "num_classes": 3, **hparams}},
            path,
        )
        return path

    def test_stitching_masking_dilation_and_normalisation_record(
        self, tmp_path, block_raster, fake_segmenter
    ):
        from agribound.engines.dinov3 import DINOv3Engine

        raster, data = block_raster
        mean, std = [0.43, 0.411, 0.296], [0.213, 0.156, 0.143]
        ckpt = self._ckpt(
            tmp_path / "dinov3.ckpt", input_mean=mean, input_std=std, boundary_erosion=1
        )
        cfg = _s2_config(
            tmp_path,
            "dinov3",
            min_field_area_m2=0,
            engine_params={"checkpoint_path": str(ckpt), "window_size": 64},
        )
        gdf = DINOv3Engine().delineate(raster, cfg)
        # the model gets the stretched [0, 1] input (it normalises it itself) in whole windows
        assert all(x.shape[-2:] == (64, 64) for x in fake_segmenter["inputs"])
        lows, highs = _data.scene_stretch_bounds(raster, [4, 3, 2])
        expected = _data.apply_stretch(data[[3, 2, 1]], lows, highs).astype(np.float32) / 255
        first = fake_segmenter["inputs"][0][0].numpy()
        np.testing.assert_allclose(first, expected[:, :64, :64])
        (load,) = fake_segmenter["loads"]
        assert load["map_location"] == "cpu" and load["weights_path"] is None
        # one interior region grown by the checkpoint's boundary width (1 px, city block):
        # 32 x 32 px minus the 4 corner pixels; the nodata corner (boundary) is removed
        assert len(gdf) == 1
        assert gdf.geometry.iloc[0].area == pytest.approx((32 * 32 - 4) * 10.0**2)
        (seg,) = [
            p
            for p in Path(cfg.get_working_dir()).glob("dinov3_segmentation_*.tif")
            if "_fields_" not in p.name
        ]
        with rasterio.open(seg) as src:
            pred = src.read(1)
        assert (pred[:8, :8] == 0).all() and (pred[8:, 8:] > 0).all()
        meta = gdf.attrs["engine_meta"]
        assert meta["input_normalization"] == {"mean": mean, "std": std}
        assert meta["window_stitching"] == "blend" and meta["dilate_interior_px"] == 1
        # 96 px, 64 px windows every 32 px: the input is extended to 96 px, 2 x 2 windows
        assert meta["window_size"] == 64 and meta["overlap"] == 32 and meta["n_windows"] == 4
        assert meta["device"] == "cpu" and "cache_reused" not in meta
        json.dumps(meta)
        # A second run reuses the cached segmentation and reports the device that made it.
        fake_segmenter["loads"].clear()
        again = DINOv3Engine().delineate(raster, cfg.merged(device="mps"))
        assert fake_segmenter["loads"] == []
        assert again.attrs["engine_meta"]["device"] == "cpu"
        assert again.attrs["engine_meta"]["cache_reused"] is True
        # The stitching is part of the cache key.
        crop = DINOv3Engine().delineate(
            raster,
            cfg.merged(engine_params={**cfg.engine_params, "window_stitching": "crop"}),
        )
        assert len(fake_segmenter["loads"]) == 1
        assert crop.attrs["engine_meta"]["window_stitching"] == "crop"
        with pytest.raises(ValueError, match="window_stitching"):
            DINOv3Engine().delineate(
                raster,
                cfg.merged(engine_params={**cfg.engine_params, "window_stitching": "mean"}),
            )

    def test_window_from_the_checkpoint_without_its_metadata_file(
        self, tmp_path, block_raster, fake_segmenter
    ):
        """A checkpoint moved without <checkpoint>.agribound.json keeps its 64 px window."""
        from agribound.engines.dinov3 import DINOv3Engine

        raster, _ = block_raster
        ckpt = self._ckpt(tmp_path / "moved.ckpt", chip_size=64, boundary_erosion=4)
        assert not _data.training_meta_path(ckpt).exists()
        cfg = _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(ckpt)})
        meta = DINOv3Engine().delineate(raster, cfg).attrs["engine_meta"]
        assert meta["window_source"] == "training_chip_size" and meta["window_size"] == 64
        assert meta["training_chip_size"] == 64 and meta["dilate_interior_px"] == 4
        # The metadata file, when present, is read first.
        _data.write_training_meta(ckpt, {"chip_size": 32})
        other = cfg.merged(cache_dir=str(tmp_path / "other"))
        meta = DINOv3Engine().delineate(raster, other).attrs["engine_meta"]
        assert meta["window_size"] == 32

    def test_checkpoint_without_statistics_gets_none(self, tmp_path, block_raster, fake_segmenter):
        """Checkpoints of agribound 1.0.1 (geoai's class) record no normalisation."""
        from agribound.engines.dinov3 import DINOv3Engine

        raster, _ = block_raster
        ckpt = self._ckpt(tmp_path / "old.ckpt")
        _data.write_training_meta(ckpt, {"boundary_erosion": 2, "chip_size": 64})
        cfg = _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(ckpt)})
        meta = DINOv3Engine().delineate(raster, cfg).attrs["engine_meta"]
        assert meta["input_normalization"] is None and meta["dilate_interior_px"] == 2
        assert meta["window_source"] == "training_chip_size" and meta["window_size"] == 64

    def _ckpt_with_weights(self, tmp_path, model_name, weights_path):
        return self._ckpt(
            tmp_path / f"{model_name}.ckpt", model_name=model_name, weights_path=weights_path
        )

    def test_checkpoint_weights_path_is_what_geoai_uses(
        self, tmp_path, block_raster, fake_segmenter, caplog
    ):
        from agribound.engines.dinov3 import DINOv3Engine

        raster, _ = block_raster
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
        assert fake_segmenter["loads"][0]["weights_path"] == str(init)
        assert gdf.attrs["engine_meta"]["weights"] == str(init)

    def test_missing_checkpoint_weights(self, tmp_path, block_raster, fake_segmenter, caplog):
        from agribound.engines.dinov3 import DINOv3Engine

        raster, _ = block_raster
        missing = str(tmp_path / "gone.pth")
        # ViT-L/16: geoai initialises from SAT-493M, the checkpoint then replaces
        # every weight -> warning only
        ckpt = self._ckpt_with_weights(tmp_path, "dinov3_vitl16", missing)
        cfg = _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(ckpt)})
        with caplog.at_level(logging.WARNING, logger="agribound.engines.dinov3"):
            DINOv3Engine().delineate(raster, cfg)
        assert "do not exist here" in caplog.text
        assert fake_segmenter["loads"][0]["weights_path"] is None
        # another backbone: the SAT-493M ViT-L/16 weights do not fit -> error
        fake_segmenter["loads"].clear()
        ckpt_b = self._ckpt_with_weights(tmp_path, "dinov3_vitb16", missing)
        with pytest.raises(FileNotFoundError, match="do not fit dinov3_vitb16"):
            DINOv3Engine().delineate(
                raster,
                _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(ckpt_b)}),
            )
        assert fake_segmenter["loads"] == []

    def test_windows_are_never_zero_padded(self, tmp_path, fake_segmenter):
        """The input is mirror-padded so that every window is whole and holds image content."""
        from agribound.engines.dinov3 import DINOv3Engine

        rng = np.random.default_rng(4)
        data = rng.uniform(300, 3000, (12, 150, 90)).astype(np.float32)  # no nodata
        raster = _write(tmp_path / "s2.tif", data, nodata=float("nan"))
        ckpt = self._ckpt(tmp_path / "dinov3.ckpt")
        _data.write_training_meta(ckpt, {"chip_size": 64, "boundary_erosion": 2})
        cfg = _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(ckpt)})
        gdf = DINOv3Engine().delineate(raster, cfg)
        meta = gdf.attrs["engine_meta"]
        # 150 x 90 px, 64 px window, stride 32 -> input extended to 160 x 96 px
        assert meta["window_size"] == 64 and meta["overlap"] == 32
        assert meta["window_source"] == "training_chip_size"
        assert meta["input_padded_to"] == [160, 96] and meta["input_padding"] == "reflect"
        windows = fake_segmenter["inputs"]
        assert windows and all(w.shape[-2:] == (64, 64) for w in windows)
        # no window has a last row or column that is 0 in every band (zero padding)
        assert not any(
            bool((w[:, -1, :] == 0).all() or (w[:, :, -1] == 0).all())
            for batch in windows
            for w in batch
        )
        # the prediction is cropped back to the raster grid
        (seg,) = [
            p
            for p in Path(cfg.get_working_dir()).glob("dinov3_segmentation_*.tif")
            if "_fields_" not in p.name
        ]
        with rasterio.open(seg) as src, rasterio.open(raster) as ref:
            assert (src.height, src.width) == (150, 90) and src.transform == ref.transform

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

    def test_requires_ckpt_file(self, tmp_path, block_raster, fake_segmenter):
        from agribound.engines.dinov3 import DINOv3Engine

        raster, _ = block_raster
        with pytest.raises(RuntimeError, match="fine-tuned checkpoint"):
            DINOv3Engine().delineate(raster, _s2_config(tmp_path, "dinov3"))
        pth = tmp_path / "state.pth"
        pth.write_bytes(b"x")
        with pytest.raises(ValueError, match="not a Lightning .ckpt"):
            DINOv3Engine().delineate(
                raster, _s2_config(tmp_path, "dinov3", engine_params={"checkpoint_path": str(pth)})
            )
        assert fake_segmenter["loads"] == []


class TestFinetuneDINOv3:
    def test_class_weights(self, tmp_path):
        from agribound.engines.finetune._dinov3 import (
            DEFAULT_CLASS_WEIGHT_CAP,
            class_pixel_counts,
            class_weights_from_counts,
            resolve_class_weights,
        )

        # N / (K n_c): 1000 / (3 * [900, 90, 10]); the average weight per pixel is 1
        uncapped = class_weights_from_counts([900, 90, 10], cap=None)
        np.testing.assert_allclose(uncapped, [1000 / 2700, 1000 / 270, 1000 / 30])
        assert np.dot(uncapped, [900, 90, 10]) / 1000 == pytest.approx(1.0)
        assert DEFAULT_CLASS_WEIGHT_CAP == 10
        assert class_weights_from_counts([900, 90, 10])[2] == 10.0  # capped
        # a class without pixels counts as one pixel (then capped)
        assert class_weights_from_counts([50, 50, 0], cap=None)[2] == pytest.approx(100 / 3)
        with pytest.raises(ValueError, match="at least 1"):
            class_weights_from_counts([1, 1, 1], cap=0.5)
        with pytest.raises(ValueError, match="no labelled pixels"):
            class_weights_from_counts([0, 0, 0])

        masks = []
        for k, mask in enumerate(
            (np.array([[0, 0, 1, 255]], np.uint8), np.array([[1, 1, 2, 255]], np.uint8))
        ):
            masks.append(_write(tmp_path / f"m{k}.tif", mask[np.newaxis]))
        assert class_pixel_counts([Path(m) for m in masks]).tolist() == [2, 3, 1]
        weights, info = resolve_class_weights({}, [Path(m) for m in masks])
        assert weights == class_weights_from_counts([2, 3, 1])
        assert info["class_weight_rule"] == "inverse_frequency"
        assert info["class_pixel_counts"] == [2, 3, 1] and info["class_weight_cap"] == 10
        no_cap, _ = resolve_class_weights({"class_weight_cap": None}, [Path(m) for m in masks])
        assert no_cap == class_weights_from_counts([2, 3, 1], cap=None)
        assert resolve_class_weights({"class_weights": None}, []) == (
            None,
            {"class_weights": None, "class_weight_rule": "none"},
        )
        assert resolve_class_weights({"class_weights": "none"}, [])[0] is None
        given, info = resolve_class_weights({"class_weights": [1, 2, 3]}, [])
        assert given == [1.0, 2.0, 3.0] and info["class_weight_rule"] == "given"
        for bad in ([1, 2], [0, 0, 0], [1, -1, 1], "balanced", [1, 2, float("nan")]):
            with pytest.raises(ValueError, match="class_weights"):
                resolve_class_weights({"class_weights": bad}, [])
        for cap in (0.5, "10", True):
            with pytest.raises(ValueError, match="class_weight_cap"):
                resolve_class_weights({"class_weight_cap": cap}, [Path(m) for m in masks])

    def test_monitor(self):
        from agribound.engines.finetune._dinov3 import resolve_monitor

        assert resolve_monitor() == ("val_miou", "max")
        assert resolve_monitor("val_loss") == ("val_loss", "min")
        with pytest.raises(ValueError, match="monitor"):
            resolve_monitor("val_iou")

    @pytest.mark.parametrize("monitor", ["val_miou", "val_loss"])
    def test_training_recipe(self, s2_chips, fake_backbone, monitor):
        """A real Lightning run with a tiny backbone: weights, checkpoint metric, records."""
        import csv

        import torch

        mod = pytest.importorskip("geoai.dinov3_finetune")
        from agribound.engines.dinov3 import dinov3_segmenter_class
        from agribound.engines.finetune import _dinov3

        params = {"decoder_features": 8, "trainer_kwargs": {"enable_progress_bar": False}}
        if monitor != "val_miou":
            params["monitor"] = monitor
        cfg, td, _ = s2_chips("dinov3", **params)
        cfg = cfg.merged(fine_tune_epochs=3, n_workers=0)
        best = Path(_dinov3._finetune_dinov3(td, cfg))
        assert monitor in best.name
        # only the best checkpoint is written (no last.ckpt)
        assert [p.name for p in best.parent.glob("*.ckpt")] == [best.name]

        meta = _data.read_training_meta(best)
        masks = _data.split_files(td, "train", "masks")
        counts = _dinov3.class_pixel_counts(masks)
        assert meta["class_weights"] == _dinov3.class_weights_from_counts(counts)
        assert meta["class_pixel_counts"] == counts.tolist()
        assert meta["recipe_version"] == _dinov3.RECIPE_VERSION == 2
        assert meta["monitor"] == monitor
        assert meta["input_normalization"]["name"] == "sat493m"
        # DINOv3 chips have a 4 px boundary band by default; the checkpoint records it
        assert meta["boundary_erosion"] == _data.read_chip_meta(td)["boundary_erosion"] == 4

        # The best epoch by the monitored metric, read from Lightning's CSV log.
        (log,) = best.parent.glob("lightning_logs/*/metrics.csv")
        rows = [r for r in csv.DictReader(log.open()) if r.get(monitor)]
        values = [float(r[monitor]) for r in rows]
        assert {"val_iou_background", "val_iou_interior", "val_iou_boundary"} <= set(rows[0])
        expected = values.index(max(values) if monitor == "val_miou" else min(values))
        assert meta["best_epoch"] == int(rows[expected]["epoch"])
        assert meta["best_model_score"] == pytest.approx(values[expected])

        hparams = _data.read_checkpoint_hparams(best)
        assert hparams["input_mean"] == [0.43, 0.411, 0.296]
        assert hparams["boundary_erosion"] == 4
        assert hparams["chip_size"] == meta["chip_size"] == _data.read_chip_meta(td)["chip_size"]
        # 10 m chips are cut from the raster itself: no processing grid is recorded.
        assert hparams["processing_gsd_m"] is None and "processing_gsd_m" not in meta
        torch.testing.assert_close(
            hparams["class_weights"], torch.tensor(meta["class_weights"], dtype=torch.float32)
        )
        model = dinov3_segmenter_class().load_from_checkpoint(str(best), map_location="cpu")
        assert model.normalizes_input
        torch.testing.assert_close(model.loss_fn.weight, hparams["class_weights"])
        # geoai's own class still loads the checkpoint (strictly), without the normalisation
        plain = mod.DINOv3Segmenter.load_from_checkpoint(str(best), map_location="cpu")
        assert not hasattr(plain, "normalizes_input")

    def test_version_1_settings_train_as_geoai_did(self, s2_chips, fake_backbone):
        """With the 1.0.1 settings the loop trains exactly as geoai's train_dinov3_segmentation."""
        import torch

        mod = pytest.importorskip("geoai.dinov3_finetune")
        from agribound.engines.finetune import _dinov3

        kwargs = {"enable_progress_bar": False, "enable_model_summary": False}
        old = {"class_weights": None, "monitor": "val_loss", "input_normalization": "none"}
        cfg, td, _ = s2_chips(
            "dinov3", boundary_erosion=2, decoder_features=8, trainer_kwargs=kwargs, **old
        )
        cfg = cfg.merged(fine_tune_epochs=2, n_workers=0)
        torch.manual_seed(0)
        ours = Path(_dinov3._finetune_dinov3(td, cfg))
        files = {
            split: [str(p) for p in _data.split_files(td, split, kind)]
            for split, kind in (("train", "images"), ("val", "images"))
        }
        masks = {
            split: [str(p) for p in _data.split_files(td, split, "masks")]
            for split in ("train", "val")
        }
        torch.manual_seed(0)
        model = mod.train_dinov3_segmentation(
            train_dataset=mod.DINOv3SegmentationDataset(files["train"], masks["train"]),
            val_dataset=mod.DINOv3SegmentationDataset(files["val"], masks["val"]),
            model_name="dinov3_vitl16",
            num_classes=3,
            decoder_features=8,
            output_dir=str(td.parent / "geoai_run"),
            batch_size=4,
            num_epochs=2,
            num_workers=0,
            freeze_backbone=False,
            accelerator="cpu",
            devices=1,
            save_top_k=1,
            **kwargs,
        )
        theirs = Path(model.trainer.checkpoint_callback.best_model_path)
        assert ours.name.split("_val_loss")[0] == theirs.name.split("_val_loss")[0]
        a = torch.load(ours, map_location="cpu", weights_only=False)["state_dict"]
        b = torch.load(theirs, map_location="cpu", weights_only=False)["state_dict"]
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)

    def test_coarse_raster_trains_and_infers_on_the_10m_grid(self, tmp_path, fake_backbone):
        """A 30 m raster: chips from the 10 m copy, the grid recorded, inference on that grid.

        The checkpoint records the grid in ``<checkpoint>.agribound.json`` and in its
        hyper-parameters, so a checkpoint moved without that file reads the grid too.
        """
        import shutil

        pytest.importorskip("geoai.dinov3_finetune")
        from agribound.engines.dinov3 import DINOv3Engine
        from agribound.engines.finetune import fine_tune

        rng = np.random.default_rng(3)
        data = rng.uniform(300, 600, (6, 50, 50)).astype(np.float32)
        data[:, 8:40, 8:40] += 2000.0
        raster = _write(tmp_path / "landsat30.tif", data, res=30.0, nodata=float("nan"))
        fields = [
            box(X0 + 300, Y0 - 700, X0 + 700, Y0 - 300),
            box(X0 + 700, Y0 - 700, X0 + 1100, Y0 - 300),
            box(X0 + 300, Y0 - 1150, X0 + 1100, Y0 - 750),
        ]
        ref = tmp_path / "ref.gpkg"
        gpd.GeoDataFrame(geometry=fields, crs=UTM).to_file(ref)
        params = {
            "chip_size": 64,
            "min_label_fraction": 0.0,
            "decoder_features": 8,
            "trainer_kwargs": {"enable_progress_bar": False, "enable_model_summary": False},
        }
        cfg = _s2_config(
            tmp_path,
            "dinov3",
            source="landsat",
            reference_boundaries=str(ref),
            fine_tune_split="random",
            fine_tune_epochs=2,
            n_workers=0,
            min_field_area_m2=0,
            engine_params=params,
        )
        best = Path(fine_tune(raster, cfg))
        meta = _data.read_training_meta(best)
        assert meta["processing_gsd_m"] == 10.0
        assert meta["processing"]["scale"] == 3.0 and meta["processing"]["size_px"] == [150, 150]
        assert _data.read_checkpoint_hparams(best)["processing_gsd_m"] == 10.0
        run = cfg.merged(engine_params={**params, "checkpoint_path": str(best)})
        processing = DINOv3Engine().delineate(raster, run).attrs["engine_meta"]["processing"]
        assert processing["gsd_m"] == 10.0 and processing["size_px"] == [150, 150]
        # Moved and renamed without <checkpoint>.agribound.json.
        moved = tmp_path / "elsewhere" / "landsat_dinov3.ckpt"
        moved.parent.mkdir()
        shutil.copy(best, moved)
        assert _data.read_training_meta(moved) == {}
        run = cfg.merged(engine_params={**params, "checkpoint_path": str(moved)})
        meta_moved = DINOv3Engine().delineate(raster, run).attrs["engine_meta"]
        assert meta_moved["processing"]["gsd_m"] == 10.0
        assert meta_moved["window_size"] == 64  # the chip size, also a hyper-parameter

    def test_misplaced_trainer_kwargs_raise(self, s2_chips):
        pytest.importorskip("geoai.dinov3_finetune")
        from agribound.engines.finetune import _dinov3

        cfg, td, _ = s2_chips("dinov3", trainer_kwargs={"class_weights": [1, 1, 1]})
        with pytest.raises(ValueError, match=r"engine_params\['class_weights'\]"):
            _dinov3._finetune_dinov3(td, cfg)

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
