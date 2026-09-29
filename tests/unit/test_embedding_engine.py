"""Tests for the embedding clustering engine (agribound.engines.embedding)."""

from __future__ import annotations

import os
import time

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from agribound.config import AgriboundConfig
from agribound.engines.embedding import MATRYOSHKA_DEPTHS, EmbeddingEngine
from agribound.registry import ENGINE_REGISTRY

UTM = "EPSG:32611"


def _embedding_raster(path, dims=128, h=60, w=80, noise=0.05, seed=0):
    """Four well-separated classes in quadrants; NaN block and an all-zero block."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (4, dims)).astype(np.float32)
    labels = np.zeros((h, w), dtype=int)
    labels[:, w // 2 :] += 1
    labels[h // 2 :, :] += 2
    data = centers[labels].transpose(2, 0, 1) + rng.normal(0, noise, (dims, h, w))
    data = data.astype(np.float32)
    data[:, :4, :4] = np.nan  # nodata
    data[:, -3:, -3:] = 0.0  # all-zero = invalid
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=dims,
        dtype="float32",
        crs=UTM,
        transform=from_origin(500000, 4000600, 10, 10),
        nodata=float("nan"),
    ) as dst:
        dst.write(data)
    return str(path), labels


def _config(tmp_path, source="tessera-embedding", **kw):
    params = {"k_candidates": [2, 3, 4, 5, 6]}
    params.update(kw.pop("engine_params", {}))
    base = {
        "source": source,
        "engine": "embedding",
        "year": 2024,
        "study_area": "bbox:-117.0,36.0,-116.99,36.01",
        "output_path": str(tmp_path / "out.gpkg"),
        "lulc_filter": False,
        "min_field_area_m2": 0.0,
        "engine_params": params,
    }
    if source == "google-embedding":
        base["gee_project"] = "test-project"
    base.update(kw)
    return AgriboundConfig(**base)


def _labels(meta):
    with rasterio.open(meta["cluster_raster"]) as src:
        return src.read(1)


@pytest.fixture
def emb(tmp_path):
    return _embedding_raster(tmp_path / "emb.tif")


class TestEngineMetadata:
    def test_class_attributes_match_registry(self):
        info = ENGINE_REGISTRY["embedding"]
        assert EmbeddingEngine.name == "embedding"
        assert EmbeddingEngine.supported_sources == info["supported_sources"]
        assert EmbeddingEngine.requires_bands == info["requires_bands"]

    def test_unsupported_source_raises(self, tmp_path, emb):
        cfg = _config(tmp_path)
        cfg.source = "sentinel2"  # bypass config validation on purpose
        with pytest.raises(ValueError, match="embedding source"):
            EmbeddingEngine().delineate(emb[0], cfg)


class TestClustering:
    def test_recovers_known_segments(self, tmp_path, emb):
        path, truth = emb
        gdf = EmbeddingEngine().delineate(path, _config(tmp_path))
        meta = gdf.attrs["engine_meta"]
        assert meta["n_clusters"] == 4  # silhouette picks the true k
        assert meta["reduction"] == "pca" and meta["n_features"] == 16
        assert 0 < meta["pca_explained_variance_ratio"] <= 1
        assert meta["clusterer"] == "KMeans"
        assert set(meta["auto_k"]["silhouette"]) == {"2", "3", "4", "5", "6"}
        labels = _labels(meta)
        # Invalid pixels (NaN, all-zero) are label 0; each true class maps to one label.
        assert (labels[:4, :4] == 0).all() and (labels[-3:, -3:] == 0).all()
        valid = labels > 0
        assert meta["n_valid_pixels"] == int(valid.sum())
        for cls in range(4):
            assert len(np.unique(labels[valid & (truth == cls)])) == 1
        assert len(gdf) == 4 and sorted(gdf.class_value.unique()) == [1, 2, 3, 4]
        assert gdf.crs == UTM

    def test_explicit_k_and_no_pca(self, tmp_path, emb):
        cfg = _config(tmp_path, engine_params={"n_clusters": 3, "use_pca": False})
        meta = EmbeddingEngine().delineate(emb[0], cfg).attrs["engine_meta"]
        assert meta["n_clusters"] == 3 and "auto_k" not in meta
        assert meta["reduction"] == "none" and meta["n_features"] == 128

    def test_spectral(self, tmp_path):
        path, truth = _embedding_raster(tmp_path / "s.tif", dims=16, h=24, w=24)
        cfg = _config(tmp_path, engine_params={"n_clusters": 4, "clustering_method": "spectral"})
        meta = EmbeddingEngine().delineate(path, cfg).attrs["engine_meta"]
        assert meta["clusterer"] == "SpectralClustering+NearestCentroid"
        labels = _labels(meta)
        valid = labels > 0
        for cls in range(4):
            assert len(np.unique(labels[valid & (truth == cls)])) == 1

    def test_sample_is_seeded_and_independent_of_global_state(self, tmp_path, emb, monkeypatch):
        """Regression: 0.1.x drew the PCA/cluster samples with the unseeded global np.random."""
        import sklearn.decomposition

        fitted = []
        original_fit = sklearn.decomposition.PCA.fit

        def spy(self, x, y=None):
            fitted.append(np.array(x))
            return original_fit(self, x, y)

        monkeypatch.setattr(sklearn.decomposition.PCA, "fit", spy)
        engine = EmbeddingEngine()
        small = {"pca_sample_size": 500, "cluster_sample_size": 300}
        for seed, global_seed, name in [(42, 1, "a"), (42, 2, "b"), (7, 1, "c")]:
            cfg = _config(tmp_path, seed=seed, engine_params=small)
            np.random.seed(global_seed)
            engine._cluster_raster(
                emb[0], tmp_path / f"{name}.tif", engine.resolve_params(cfg), cfg
            )
        assert fitted[0].shape == (500, 128)
        np.testing.assert_array_equal(fitted[0], fitted[1])  # global state irrelevant
        assert not np.array_equal(fitted[0], fitted[2])  # config.seed matters
        with rasterio.open(tmp_path / "a.tif") as a, rasterio.open(tmp_path / "b.tif") as b:
            np.testing.assert_array_equal(a.read(1), b.read(1))

    def test_seed_changes_the_sample(self, tmp_path, emb):
        """Different seeds draw different pixel samples (same cluster structure here)."""
        from agribound._repro import get_rng

        cfg_a, cfg_b = _config(tmp_path, seed=1), _config(tmp_path, seed=2)
        ka = get_rng(cfg_a, "embedding-sample").random(10)
        kb = get_rng(cfg_b, "embedding-sample").random(10)
        assert not np.allclose(ka, kb)
        ma = EmbeddingEngine().delineate(emb[0], cfg_a).attrs["engine_meta"]
        mb = EmbeddingEngine().delineate(emb[0], cfg_b).attrs["engine_meta"]
        assert ma["cluster_raster"] != mb["cluster_raster"]
        assert ma["seed"] == 1 and mb["seed"] == 2

    def test_block_size_does_not_change_result(self, tmp_path, emb):
        small = _config(tmp_path, engine_params={"max_block_mb": 0.05})
        engine = EmbeddingEngine()
        params = engine.resolve_params(small)
        meta_small = engine._cluster_raster(emb[0], tmp_path / "s.tif", params, small)
        big = _config(tmp_path, engine_params={"max_block_mb": 512})
        meta_big = engine._cluster_raster(
            emb[0], tmp_path / "b.tif", engine.resolve_params(big), big
        )
        assert meta_small["block_rows"] < 60 <= meta_big["block_rows"]  # really read in blocks
        with rasterio.open(tmp_path / "s.tif") as a, rasterio.open(tmp_path / "b.tif") as b:
            np.testing.assert_array_equal(a.read(1), b.read(1))

    def test_too_few_valid_pixels(self, tmp_path):
        path = tmp_path / "nan.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=5, width=5, count=8, dtype="float32",
            crs=UTM, transform=from_origin(500000, 4000050, 10, 10),
        ) as dst:  # fmt: skip
            dst.write(np.full((8, 5, 5), np.nan, dtype=np.float32))
        with pytest.raises(ValueError, match="valid embedding pixels"):
            EmbeddingEngine().delineate(str(path), _config(tmp_path))

    @pytest.mark.parametrize("nodata", [-9999.0, 1e-3])
    def test_declared_finite_nodata_is_invalid(self, tmp_path, nodata):
        rng = np.random.default_rng(0)
        h, w, d = 40, 40, 8
        data = rng.normal(0, 1, (d, h, w)).astype(np.float32)
        data[:, :, :20] += 5
        data[:, :10, :10] = nodata  # nodata block (all bands)
        data[0, 30, 30] = nodata  # one band only: still valid
        path = tmp_path / "nd.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=h, width=w, count=d, dtype="float32",
            crs=UTM, transform=from_origin(500000, 4000400, 10, 10), nodata=nodata,
        ) as dst:  # fmt: skip
            dst.write(data)
        gdf = EmbeddingEngine().delineate(
            str(path), _config(tmp_path, engine_params={"n_clusters": 2})
        )
        meta = gdf.attrs["engine_meta"]
        assert meta["n_valid_pixels"] == h * w - 100
        labels = _labels(meta)
        assert (labels[:10, :10] == 0).all()
        assert labels[30, 30] > 0


class TestCaching:
    def test_second_run_uses_cache(self, tmp_path, emb, monkeypatch):
        engine = EmbeddingEngine()
        cfg = _config(tmp_path)
        first = engine.delineate(emb[0], cfg)
        assert first.attrs["engine_meta"]["cache_hit"] is False
        monkeypatch.setattr(
            EmbeddingEngine, "_cluster_raster", lambda *a, **k: pytest.fail("recomputed")
        )
        second = engine.delineate(emb[0], cfg)
        meta = second.attrs["engine_meta"]
        assert meta["cache_hit"] is True and meta["n_clusters"] == 4
        assert sorted(second.area) == pytest.approx(sorted(first.area))

    @pytest.mark.parametrize(
        "change",
        [
            {"engine_params": {"n_clusters": 3}},
            {"engine_params": {"pca_components": 8}},
            {"engine_params": {"clustering_method": "spectral"}},
            {"year": 2023},
            {"tessera_version": "v1.1"},
            {"seed": 7},
            {"study_area": "bbox:-117.0,36.0,-116.98,36.02"},
        ],
    )
    def test_cache_key_changes(self, tmp_path, emb, change):
        base = EmbeddingEngine.cluster_cache_path(emb[0], _config(tmp_path))
        other = EmbeddingEngine.cluster_cache_path(emb[0], _config(tmp_path, **change))
        assert base != other
        assert base.parent == other.parent == (tmp_path / ".agribound_cache")

    def test_block_size_shares_cache_entry(self, tmp_path, emb):
        engine = EmbeddingEngine()
        a = engine.delineate(emb[0], _config(tmp_path, engine_params={"max_block_mb": 1}))
        b = engine.delineate(emb[0], _config(tmp_path, engine_params={"max_block_mb": 64}))
        assert a.attrs["engine_meta"]["cluster_raster"] == b.attrs["engine_meta"]["cluster_raster"]
        assert b.attrs["engine_meta"]["cache_hit"] is True

    def test_modified_raster_invalidates_cache(self, tmp_path, emb):
        engine = EmbeddingEngine()
        cfg = _config(tmp_path)
        first = engine.delineate(emb[0], cfg).attrs["engine_meta"]["cluster_raster"]
        time.sleep(0.01)
        _embedding_raster(tmp_path / "emb.tif", seed=3)  # rewrite in place
        os.utime(emb[0])
        second = engine.delineate(emb[0], cfg).attrs["engine_meta"]
        assert second["cluster_raster"] != first and second["cache_hit"] is False


class TestMatryoshka:
    def test_v2_prefix_replaces_pca(self, tmp_path, emb):
        cfg = _config(tmp_path, tessera_version="v2", engine_params={"matryoshka_depth": 16})
        meta = EmbeddingEngine().delineate(emb[0], cfg).attrs["engine_meta"]
        assert meta["reduction"] == "matryoshka" and meta["n_features"] == 16
        assert "pca_explained_variance_ratio" not in meta
        assert meta["tessera_version"] == "v2"

    @pytest.mark.parametrize(
        ("kw", "match"),
        [
            ({"tessera_version": "v1"}, "TESSERA v2"),
            ({"source": "google-embedding", "year": 2024}, "TESSERA v2"),
        ],
    )
    def test_rejected_without_v2(self, tmp_path, emb, kw, match):
        cfg = _config(tmp_path, engine_params={"matryoshka_depth": 16}, **kw)
        with pytest.raises(ValueError, match=match):
            EmbeddingEngine().delineate(emb[0], cfg)

    def test_invalid_depths(self, tmp_path, emb):
        cfg = _config(tmp_path, tessera_version="v2", engine_params={"matryoshka_depth": 20})
        with pytest.raises(ValueError, match="matryoshka_depth must be one of"):
            EmbeddingEngine().delineate(emb[0], cfg)
        small, _ = _embedding_raster(tmp_path / "small.tif", dims=8)
        cfg = _config(tmp_path, tessera_version="v2", engine_params={"matryoshka_depth": 16})
        with pytest.raises(ValueError, match="only 8 bands"):
            EmbeddingEngine().delineate(small, cfg)
        assert MATRYOSHKA_DEPTHS == (4, 16, 32, 64)


class TestParameterValidation:
    @pytest.mark.parametrize(
        "params",
        [
            {"n_clusters": 1},
            {"n_clusters": "many"},
            {"clustering_method": "dbscan"},
            {"k_candidates": [1, 2]},
            {"pca_components": 0},
            {"max_block_mb": 0},
        ],
    )
    def test_invalid(self, tmp_path, params):
        with pytest.raises(ValueError):
            EmbeddingEngine.resolve_params(_config(tmp_path, engine_params=params))


class TestSamRefinement:
    def test_requires_rgb_bands_before_clustering(self, tmp_path, emb, monkeypatch):
        monkeypatch.setattr(
            EmbeddingEngine, "_cluster_raster", lambda *a, **k: pytest.fail("clustered first")
        )
        cfg = _config(tmp_path, sam_refine=True)
        with pytest.raises(ValueError, match="sam_rgb_bands"):
            EmbeddingEngine().delineate(emb[0], cfg)

    def test_legacy_engine_param_triggers_refinement(self, tmp_path, emb, monkeypatch):
        from agribound.engines import samgeo_engine

        calls = {}

        def fake_refine(gdf, raster_path, config, **kwargs):
            calls["n"] = len(gdf)
            calls["raster"] = raster_path
            out = gdf.copy()
            out["agribound:sam_refined"] = False
            out.attrs["sam_stats"] = {"backend": "fake", "n_total": len(gdf)}
            return out

        monkeypatch.setattr(samgeo_engine, "refine_boundaries", fake_refine)
        cfg = _config(tmp_path, engine_params={"sam_refine": True, "sam_rgb_bands": [1, 2, 3]})
        assert cfg.sam_refine is True  # absorbed by the config
        gdf = EmbeddingEngine().delineate(emb[0], cfg)
        assert calls == {"n": 4, "raster": emb[0]}
        meta = gdf.attrs["engine_meta"]
        assert meta["sam_refine"] is True and meta["sam_stats"]["backend"] == "fake"

    def test_prefetch(self, tmp_path, monkeypatch):
        from agribound.engines import samgeo_engine

        assert EmbeddingEngine.prefetch(_config(tmp_path)) == []
        monkeypatch.setattr(samgeo_engine, "prefetch", lambda config: ["/weights.pt"])
        cfg = _config(tmp_path, sam_refine=True, engine_params={"sam_rgb_bands": [1, 2, 3]})
        assert EmbeddingEngine.prefetch(cfg) == ["/weights.pt"]


def test_polygons_have_expected_area(tmp_path, emb):
    """Polygon areas equal the valid pixel counts of each cluster (10 m pixels)."""
    gdf = EmbeddingEngine().delineate(emb[0], _config(tmp_path))
    labels = _labels(gdf.attrs["engine_meta"])
    for value, group in gdf.groupby("class_value"):
        assert group.area.sum() == pytest.approx((labels == value).sum() * 100.0)
    assert isinstance(gdf, gpd.GeoDataFrame)
