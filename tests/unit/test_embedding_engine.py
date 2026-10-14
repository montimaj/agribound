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
        assert meta["clusterer"] == "KMeans" and meta["kmeans_n_init"] == 10
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

    def test_kmeans_restarts_whatever_the_raster_size(self, tmp_path, monkeypatch):
        """Regression: 1.0.0 used MiniBatchKMeans above 100 000 valid pixels.

        The silhouette fits and the final fit are all ``KMeans(n_init=10)``.
        """
        import sklearn.cluster

        calls = []
        kmeans = sklearn.cluster.KMeans

        def spy(**kw):
            calls.append(kw)
            return kmeans(**kw)

        monkeypatch.setattr(sklearn.cluster, "KMeans", spy)
        monkeypatch.setattr(
            sklearn.cluster, "MiniBatchKMeans", lambda **kw: pytest.fail("MiniBatchKMeans")
        )
        path, _ = _embedding_raster(tmp_path / "big.tif", dims=8, h=320, w=320)
        meta = EmbeddingEngine().delineate(path, _config(tmp_path)).attrs["engine_meta"]
        assert meta["n_valid_pixels"] > 100_000
        assert meta["clusterer"] == "KMeans" and meta["kmeans_n_init"] == 10
        assert meta["n_clusters"] == 4
        # Five silhouette fits (k_candidates 2..6), then the final fit at k = 4.
        assert [c["n_clusters"] for c in calls] == [2, 3, 4, 5, 6, 4]
        assert all(c["n_init"] == 10 and c["random_state"] == 42 for c in calls)

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
            {"engine_params": {"standardize": "zscore"}},
        ],
    )
    def test_cache_key_changes(self, tmp_path, emb, change):
        base = EmbeddingEngine.cluster_cache_path(emb[0], _config(tmp_path))
        other = EmbeddingEngine.cluster_cache_path(emb[0], _config(tmp_path, **change))
        assert base != other
        assert base.parent == other.parent == (tmp_path / ".agribound_cache")

    def test_cache_version_is_part_of_the_key(self, tmp_path, emb, monkeypatch):
        """Cluster rasters cached by 1.0.0 (MiniBatchKMeans, cache v3) are not reused."""
        from agribound.engines import embedding

        current = EmbeddingEngine.cluster_cache_path(emb[0], _config(tmp_path))
        monkeypatch.setattr(embedding, "_CACHE_VERSION", "embedding-clusters-v3")
        assert EmbeddingEngine.cluster_cache_path(emb[0], _config(tmp_path)) != current

    def test_standardize_none_keeps_existing_cache_keys(self, tmp_path, emb, monkeypatch):
        """standardize="none" (the default) adds nothing to the key parts."""
        import agribound._cache as cache

        seen = {}
        real = cache.cache_path

        def spy(config, name, ext, *parts):
            seen["parts"] = parts
            return real(config, name, ext, *parts)

        monkeypatch.setattr(cache, "cache_path", spy)
        EmbeddingEngine.cluster_cache_path(emb[0], _config(tmp_path))
        assert not any(str(p).startswith("standardize=") for p in seen["parts"])
        cfg = _config(tmp_path, engine_params={"standardize": "robust"})
        EmbeddingEngine.cluster_cache_path(emb[0], cfg)
        assert "standardize=robust" in seen["parts"]

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
            {"standardize": "minmax"},
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


def _mixed_units_raster(path, h=60, w=80, seed=0):
    """Quadrant classes in two small-unit bands and one large-unit band
    without class signal (like entropy and DpRVI next to a noisy dB band)."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((h, w), dtype=int)
    labels[:, w // 2 :] += 1
    labels[h // 2 :, :] += 2
    centres = np.array([[0.2, 0.3], [0.8, 0.3], [0.2, 0.9], [0.8, 0.9]])
    small = centres[labels].transpose(2, 0, 1) + rng.normal(0, 0.03, (2, h, w))
    large = rng.normal(0, 40.0, (1, h, w))
    data = np.concatenate([small, large]).astype(np.float32)
    data[:, :4, :4] = np.nan
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=3,
        dtype="float32",
        crs=UTM,
        transform=from_origin(500000, 4000600, 10, 10),
        nodata=float("nan"),
    ) as dst:
        dst.write(data)
    return str(path), labels


def _purity(pred, truth):
    valid = pred > 0
    p, t = pred[valid], truth[valid]
    return sum(np.bincount(t[p == c]).max() for c in np.unique(p)) / len(p)


class TestStandardisation:
    def _local_config(self, tmp_path, path, **params):
        return _config(
            tmp_path,
            source="local",
            local_tif_path=path,
            engine_params={"n_clusters": 4, **params},
        )

    def test_local_feature_raster_needs_standardisation(self, tmp_path):
        path, truth = _mixed_units_raster(tmp_path / "feat.tif")
        engine = EmbeddingEngine()
        raw = engine.delineate(path, self._local_config(tmp_path, path)).attrs["engine_meta"]
        std = engine.delineate(path, self._local_config(tmp_path, path, standardize="zscore"))
        meta = std.attrs["engine_meta"]
        assert meta["standardize"]["method"] == "zscore" and len(meta["standardize"]["scale"]) == 3
        assert meta["standardize"]["scale"][2] > 30 > 1 > meta["standardize"]["scale"][0]
        assert "standardize" not in raw
        assert _purity(_labels(meta), truth) > 0.99
        assert _purity(_labels(raw), truth) < 0.6

    @pytest.mark.parametrize("method", ["zscore", "robust"])
    def test_same_as_clustering_a_prestandardised_raster(self, tmp_path, method):
        path, _ = _mixed_units_raster(tmp_path / "feat.tif")
        engine = EmbeddingEngine()
        meta = engine.delineate(path, self._local_config(tmp_path, path, standardize=method))
        meta = meta.attrs["engine_meta"]
        centre = np.array(meta["standardize"]["centre"])
        scale = np.array(meta["standardize"]["scale"])
        with rasterio.open(path) as src:
            data, profile = src.read(), src.profile
        pre = ((data.reshape(3, -1).T - centre) / scale).astype(np.float32).T.reshape(data.shape)
        pre_path = str(tmp_path / "pre.tif")
        with rasterio.open(pre_path, "w", **profile) as dst:
            dst.write(pre)
        other = engine.delineate(pre_path, self._local_config(tmp_path, pre_path)).attrs[
            "engine_meta"
        ]
        assert np.array_equal(_labels(meta), _labels(other))


def _write(path, data, nodata=None):
    """Write a ``(bands, h, w)`` array as a 10 m UTM GeoTIFF of its dtype."""
    count, h, w = data.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=count, dtype=str(data.dtype),
        crs=UTM, transform=from_origin(500000, 4000000 + 10 * h, 10, 10), nodata=nodata,
    ) as dst:  # fmt: skip
        dst.write(data)
    return str(path)


class TestLocalFeatureRasters:
    def _cfg(self, tmp_path, path, **params):
        return _config(
            tmp_path, source="local", local_tif_path=path, engine_params={"n_clusters": 2, **params}
        )

    def test_any_band_at_nodata_or_nan_is_invalid(self, tmp_path):
        """A feature pixel is skipped when any band is nodata or NaN; for embedding
        sources the nodata rule stays "every band"."""
        rng = np.random.default_rng(0)
        h, w = 40, 40
        data = rng.normal(0, 1, (4, h, w)).astype(np.float32)
        data[:, :, :20] += 5
        data[:, :10, :10] = -9999.0  # every band
        data[0, 30, 30] = -9999.0  # one band
        data[2, 31, 31] = np.nan  # one band, NaN
        path = _write(tmp_path / "feat.tif", data, nodata=-9999.0)
        meta = EmbeddingEngine().delineate(path, self._cfg(tmp_path, path)).attrs["engine_meta"]
        assert meta["n_valid_pixels"] == h * w - 102
        labels = _labels(meta)
        assert (labels[:10, :10] == 0).all() and labels[30, 30] == 0 and labels[31, 31] == 0
        emb_cfg = _config(tmp_path, engine_params={"n_clusters": 2})
        emb_meta = EmbeddingEngine().delineate(path, emb_cfg).attrs["engine_meta"]
        assert emb_meta["n_valid_pixels"] == h * w - 101
        assert _labels(emb_meta)[30, 30] > 0

    def test_no_valid_pixel_names_the_rule(self, tmp_path):
        data = np.full((2, 5, 5), -1.0, dtype=np.float32)
        data[1] = np.nan  # one band NaN everywhere: no pixel is valid
        path = _write(tmp_path / "feat.tif", data, nodata=-1.0)
        with pytest.raises(ValueError, match="0 valid feature pixels.*not any band equals it"):
            EmbeddingEngine().delineate(path, self._cfg(tmp_path, path))

    def test_complex_pixels_are_rejected(self, tmp_path):
        data = (np.ones((1, 8, 8)) + 1j * np.arange(64).reshape(1, 8, 8)).astype(np.complex64)
        path = _write(tmp_path / "slc.tif", data)
        with pytest.raises(ValueError, match=r"complex pixels \(complex64\)"):
            EmbeddingEngine().delineate(path, self._cfg(tmp_path, path))
        assert not list(tmp_path.glob("**/embedding_clusters_*.tif"))

    def test_integer_bands_cluster_like_their_float32_copy(self, tmp_path):
        path, truth = _mixed_units_raster(tmp_path / "feat.tif")
        with rasterio.open(path) as src:
            data = src.read()
        ints = np.where(np.isfinite(data), np.round(data * 100), -32768).astype(np.int16)
        int_path = _write(tmp_path / "int16.tif", ints, nodata=-32768)
        float_path = _write(tmp_path / "float32.tif", ints.astype(np.float32), nodata=-32768)
        engine = EmbeddingEngine()
        metas = [
            engine.delineate(p, self._cfg(tmp_path, p, n_clusters=4, standardize="zscore")).attrs[
                "engine_meta"
            ]
            for p in (int_path, float_path)
        ]
        assert metas[0]["n_valid_pixels"] == 60 * 80 - 16
        assert np.array_equal(_labels(metas[0]), _labels(metas[1]))
        assert _purity(_labels(metas[0]), truth) > 0.99

    def test_single_band_and_constant_band(self, tmp_path):
        rng = np.random.default_rng(1)
        truth = np.zeros((30, 40), dtype=int)
        truth[:, 20:] = 1
        db = np.array([-15.0, -8.0])[truth] + rng.normal(0, 0.5, truth.shape)
        single = _write(tmp_path / "one.tif", db[None].astype(np.float32))
        two = _write(tmp_path / "two.tif", np.stack([db, np.full_like(db, 7.0)]).astype(np.float32))
        engine = EmbeddingEngine()
        for path, n_scales in ((single, 1), (two, 2)):
            meta = engine.delineate(path, self._cfg(tmp_path, path, standardize="zscore"))
            meta = meta.attrs["engine_meta"]
            assert len(meta["standardize"]["scale"]) == n_scales
            assert _purity(_labels(meta), truth) == 1.0
        assert meta["standardize"]["scale"][1] == 1.0  # zero spread: scale 1, not a division by 0

    def test_unequal_spread_warns_only_without_standardisation(self, tmp_path, caplog):
        import logging

        path, _ = _mixed_units_raster(tmp_path / "feat.tif")
        engine = EmbeddingEngine()
        logger = "agribound.engines.embedding"
        with caplog.at_level(logging.WARNING, logger=logger):
            engine.delineate(path, self._cfg(tmp_path, path, n_clusters=4))
        assert "band standard deviations range from" in caplog.text
        assert "(band 3)" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger=logger):
            engine.delineate(path, self._cfg(tmp_path, path, n_clusters=4, standardize="robust"))
            emb_cfg = _config(tmp_path, engine_params={"n_clusters": 4})
            engine.delineate(path, emb_cfg)  # embedding sources: no warning
        assert "band standard deviations" not in caplog.text

    def test_params_record_leaves_out_standardize_none(self, tmp_path, emb):
        """engine_meta of an existing configuration is the one written before the parameter."""
        meta = EmbeddingEngine().delineate(emb[0], _config(tmp_path)).attrs["engine_meta"]
        assert "standardize" not in meta["params"] and "standardize" not in meta
        cfg = _config(tmp_path, engine_params={"standardize": "zscore"})
        meta = EmbeddingEngine().delineate(emb[0], cfg).attrs["engine_meta"]
        assert (
            meta["params"]["standardize"] == "zscore" and meta["standardize"]["method"] == "zscore"
        )


def _field_grid(path, n=5, field_px=12, road_px=2, seed=0):
    """``n`` x ``n`` square fields of three classes on a road grid (10 m pixels).

    Two bands in small units carry the classes (fields and roads at the corners
    of the feature square, like entropy and DpRVI); a third band in large units
    (standard deviation 40, like an uncalibrated backscatter band) has no class
    signal. Returns the raster path and the field polygons.
    """
    from shapely.geometry import box

    rng = np.random.default_rng(seed)
    step = field_px + road_px
    size = n * step + road_px
    cls = np.full((size, size), 3)  # roads
    x0, y0 = 500000.0, 4000000.0 + 10 * size
    fields = []
    for i in range(n):
        for j in range(n):
            r, c = road_px + i * step, road_px + j * step
            cls[r : r + field_px, c : c + field_px] = rng.integers(0, 3)
            fields.append(
                box(x0 + 10 * c, y0 - 10 * (r + field_px), x0 + 10 * (c + field_px), y0 - 10 * r)
            )
    centres = np.array([[0.2, 0.3], [0.8, 0.3], [0.2, 0.9], [0.8, 0.9]])
    small = centres[cls].transpose(2, 0, 1) + rng.normal(0, 0.03, (2, size, size))
    large = rng.normal(0, 40.0, (1, size, size))
    data = np.concatenate([small, large]).astype(np.float32)
    return _write(path, data, nodata=float("nan")), gpd.GeoDataFrame(geometry=fields, crs=UTM)


@pytest.mark.parametrize("seed", [0, 1])
def test_pipeline_on_a_local_feature_raster_recovers_fields_only_when_standardised(tmp_path, seed):
    """source="local" through agribound.delineate (builder, engine, post-processing)."""
    import agribound

    path, reference = _field_grid(tmp_path / "features.tif", seed=seed)
    scores = {}
    for standardize in ("none", "zscore"):
        gdf = agribound.delineate(
            source="local",
            local_tif_path=path,
            engine="embedding",
            output_path=str(tmp_path / f"fields_{standardize}.gpkg"),
            lulc_filter=False,
            engine_params={"n_clusters": 4, "standardize": standardize},
        )
        scores[standardize] = agribound.evaluate(gdf, reference)
    assert scores["zscore"]["recall"] == 1.0  # every field; the road grid is one extra polygon
    assert scores["zscore"]["f1"] > 0.95
    assert scores["none"]["f1"] == 0.0  # the large-unit noise band decides the clusters


class TestExistingConfigurationsUnchanged:
    """Configurations that agribound 1.0.1 could express keep their hashes and cache keys."""

    @pytest.mark.parametrize(
        ("source", "year", "square", "expected"),
        [
            ("google-embedding", 2022, "madera", "0f9e5094f9689a4df819f1e8065ccafcaf5103ab"),
            ("tessera-embedding", 2020, "twifo", "c017077622bd1e6243f43bb77c1c9a8b961e158e"),
        ],
    )
    def test_config_hash_is_that_of_agribound_1_0_1(self, source, year, square, expected):
        """Example 23's k = 20 runs: the config_hash their 1.0.1 provenance recorded."""
        from agribound.provenance import config_hash

        config = AgriboundConfig(
            source=source,
            year=year,
            engine="embedding",
            engine_params={"n_clusters": 20},
            lulc_filter=False,
            study_area=f"outputs/tree_crops/{square}/study_area.geojson",
            gee_project="p" if source == "google-embedding" else None,  # not hashed
        )
        assert config_hash(config) == expected
        zscore = config.merged(engine_params={"n_clusters": 20, "standardize": "zscore"})
        assert config_hash(zscore) != expected

    @pytest.mark.parametrize(
        ("kw", "expected"),
        [
            ({"source": "tessera-embedding", "year": 2024}, "embedding_clusters_32b75fbab8e4.tif"),
            (
                {
                    "source": "google-embedding",
                    "year": 2022,
                    "gee_project": "p",
                    "study_area": "bbox:-120.5,36.9,-120.4,37.0",
                    "engine_params": {"n_clusters": 20},
                },
                "embedding_clusters_433ec606b1d1.tif",
            ),
        ],
    )
    def test_cluster_cache_key_is_unchanged(self, tmp_path, monkeypatch, kw, expected):
        """Cache names computed before engine_params["standardize"] existed (2026-10-08)."""
        from agribound._results import results_versions
        from agribound.engines import embedding

        monkeypatch.setattr(
            embedding, "_file_signature", lambda p: "raster=/fixed/emb.tif:1234:5678"
        )
        config = AgriboundConfig(
            **{
                "engine": "embedding",
                "study_area": "bbox:-117.0,36.0,-116.99,36.01",
                "output_path": str(tmp_path / "out.gpkg"),
                **kw,
            }
        )
        assert EmbeddingEngine.cluster_cache_path("/fixed/emb.tif", config).name == expected
        assert results_versions(config) == {"embedding": 2}
