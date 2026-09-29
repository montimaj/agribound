"""Live smoke tests for the composite, embedding and LULC builders.

They download small rasters for a Namoi (New South Wales) test box of about
4.8 x 5.6 km (:data:`NAMOI_AOI`, the ``test_bbox`` of
``examples/regions/namoi_catchment_au.yaml``) and need Earth Engine credentials
(``gee``) and/or internet access (``network``). They are excluded from the
offline suite; run them with::

    AGRIBOUND_SMOKE_DIR=/some/dir pytest -m "gee or network" tests/unit/test_composites_smoke.py -s

Outputs go to ``$AGRIBOUND_SMOKE_DIR`` (default: pytest's ``tmp_path``). The
Earth Engine project is the one AgriboundConfig resolves (``GEE_PROJECT``, then
the gcloud configuration, then the ``project_id`` of the credentials file); set
``GEE_PROJECT`` to choose it, e.g. ``GEE_PROJECT=my-project pytest -m gee ...``.
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

from agribound.config import AgriboundConfig

#: Namoi test box (EPSG:4326 minx, miny, maxx, maxy), UTM zone 56S; fields with SPOT and
#: TESSERA coverage. Defined here (not read from a file) so the tests run on any clone.
NAMOI_BOUNDS = (151.0, -31.245, 151.05, -31.195)
NAMOI_AOI = "bbox:" + ",".join(str(v) for v in NAMOI_BOUNDS)
#: None lets AgriboundConfig resolve the project (GEE_PROJECT, then gcloud, then
#: the project_id of the credentials file), as it does for users.
GEE_PROJECT = os.environ.get("GEE_PROJECT") or None


@pytest.fixture
def out_dir(tmp_path):
    base = os.environ.get("AGRIBOUND_SMOKE_DIR")
    path = Path(base) if base else tmp_path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cfg(out_dir, source, **kwargs):
    params = {
        "source": source,
        "engine": "embedding" if "embedding" in source else "delineate-anything",
        "year": 2023,
        "study_area": NAMOI_AOI,
        "gee_project": GEE_PROJECT,
        "output_path": str(out_dir / f"{source}.gpkg"),
        "lulc_filter": False,
    }
    params.update(kwargs)
    return AgriboundConfig(**params)


def _band_medians(path):
    with rasterio.open(path) as src:
        data = src.read().astype(np.float64)
        names = src.descriptions
        info = {"crs": src.crs.to_string(), "shape": (src.height, src.width), "res": src.res}
    med = {n: float(np.nanmedian(d)) for n, d in zip(names, data, strict=True)}
    valid = float(np.isfinite(data[0]).mean())
    return med, valid, info


@pytest.mark.gee
@pytest.mark.slow
@pytest.mark.parametrize(
    ("source", "visible_nir"),
    [
        ("sentinel2", ["B2", "B3", "B4", "B8"]),
        ("landsat", ["SR_B2", "SR_B3", "SR_B4", "SR_B5"]),
        ("hls", ["B2", "B3", "B4", "B5"]),
    ],
)
def test_optical_composite_reflectance_scale(out_dir, source, visible_nir):
    from agribound.composites import get_composite_builder

    builder = get_composite_builder(source)
    path = builder.build(_cfg(out_dir, source))
    med, valid, info = _band_medians(path)
    print(f"\n{source}: {info} valid={valid:.3f} medians={ {k: round(v) for k, v in med.items()} }")
    assert info["crs"] == "EPSG:32756"
    assert valid > 0.9
    for band in visible_nir:
        assert 100 < med[band] < 6000, (band, med[band])
    assert builder.last_metadata["AGRIBOUND_VALUE_SCALE"] == "reflectance_x10000"


@pytest.mark.network
@pytest.mark.slow
def test_tessera_v11_2023(out_dir):
    from agribound.composites.local import build_tessera_embedding

    path, tags = build_tessera_embedding(
        _cfg(out_dir, "tessera-embedding", tessera_version="v1.1", embedding_cache_dir=None)
    )
    with rasterio.open(path) as src:
        print(f"\nTESSERA: {src.crs} {src.height}x{src.width} {src.count} bands, tags={tags}")
        assert src.count == 128 and src.res == (10.0, 10.0)
        band = src.read(1)
    assert float(tags["AGRIBOUND_VALID_FRACTION"]) > 0.9
    assert np.nanstd(band) > 0


@pytest.mark.gee
@pytest.mark.slow
def test_google_embedding_2023(out_dir):
    from agribound.composites.local import build_google_embedding_gee

    path, tags = build_google_embedding_gee(_cfg(out_dir, "google-embedding"))
    with rasterio.open(path) as src:
        data = src.read()
        print(f"\nGoogle embedding: {src.crs} {src.height}x{src.width} {src.count} bands")
        assert src.count == 64 and src.dtypes[0] == "float32" and src.res == (10.0, 10.0)
    norms = np.sqrt(np.nansum(data**2, axis=0))
    finite = np.isfinite(data[0])
    print(f"valid={finite.mean():.3f}, median L2 norm={np.median(norms[finite]):.4f}")
    assert finite.mean() > 0.9
    assert 0.95 < float(np.median(norms[finite])) < 1.05  # unit-length vectors


SEAM_150E = "bbox:149.99,-30.58,150.01,-30.57"  # crosses the UTM 55/56 boundary


@pytest.mark.network
@pytest.mark.slow
def test_google_embedding_source_coop_across_utm_seam(out_dir):
    # Regression: geoai's corner-based reader failed here ("All tiles must share the
    # same CRS"); the direct reader uses both zones' tiles, each over its own side.
    from agribound.composites.local import build_google_embedding_source_coop

    cfg = _cfg(
        out_dir,
        "google-embedding",
        study_area=SEAM_150E,
        google_embedding_backend="source_coop",
        embedding_cache_dir=os.environ.get("AGRIBOUND_AEF_INDEX_DIR"),
    )
    path, tags = build_google_embedding_source_coop(cfg)
    with rasterio.open(path) as src:
        data = src.read(1)
        print(f"\nSource Cooperative seam: {src.crs} {src.height}x{src.width}, tags={tags}")
        assert src.crs.to_epsg() == 32756 and src.count == 64
    assert int(tags["AGRIBOUND_N_TILES"]) == 2
    assert np.isfinite(data).all()


@pytest.mark.gee
@pytest.mark.network
@pytest.mark.slow
def test_google_embedding_backends_agree_namoi(out_dir):
    from agribound.composites.local import (
        build_google_embedding_gee,
        build_google_embedding_source_coop,
    )

    base = _cfg(out_dir, "google-embedding")
    p_gee, _ = build_google_embedding_gee(base)
    p_sc, _ = build_google_embedding_source_coop(
        base.merged(
            google_embedding_backend="source_coop",
            embedding_cache_dir=os.environ.get("AGRIBOUND_AEF_INDEX_DIR"),
        )
    )
    with rasterio.open(p_gee) as a, rasterio.open(p_sc) as b:
        assert a.transform == b.transform and a.shape == b.shape and a.crs == b.crs
        np.testing.assert_array_equal(a.read(), b.read())


@pytest.mark.gee
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["server", "raster"])
def test_lulc_filter_namoi(out_dir, mode):
    from agribound.postprocess.lulc_filter import filter_by_lulc

    aoi = gpd.GeoSeries([box(*NAMOI_BOUNDS)], crs="EPSG:4326").to_crs("EPSG:32756").iloc[0]
    minx, miny, maxx, maxy = aoi.bounds
    step = (maxx - minx) / 4
    polys = [
        box(
            minx + i * step + 50,
            miny + j * step + 50,
            minx + (i + 1) * step - 50,
            miny + (j + 1) * step - 50,
        )
        for i in range(4)
        for j in range(4)
    ]
    gdf = gpd.GeoDataFrame(geometry=polys, crs="EPSG:32756")
    cfg = _cfg(out_dir, "sentinel2", lulc_filter=True, lulc_mode=mode, lulc_crop_threshold=0.3)
    out = filter_by_lulc(gdf, cfg)
    stats = out.attrs["lulc_stats"]
    print(f"\nLULC {mode}: {stats}")
    assert stats["dataset"] == "dynamic_world" and stats["year_used"] == 2023
    assert stats["n_in"] == 16 and stats["n_nan"] == 0
    assert out["lulc:crop_fraction"].between(0, 1).all()


@pytest.mark.gee
@pytest.mark.slow
def test_nlcd_coverage_routing_live(out_dir):
    from agribound.auth import ensure_gee
    from agribound.postprocess.lulc_filter import _nlcd_valid_fraction

    ensure_gee(_cfg(out_dir, "sentinel2"))

    areas = {
        "iowa": box(-93.60, 42.00, -93.55, 42.05),
        "chihuahua": box(-106.10, 28.60, -106.05, 28.65),
        "ontario": box(-81.00, 43.00, -80.95, 43.05),
    }
    fractions = {k: _nlcd_valid_fraction(v, 2023) for k, v in areas.items()}
    print(f"\nNLCD valid fractions: {fractions}")
    assert fractions["iowa"] > 0.99
    assert fractions["chihuahua"] < 0.01 and fractions["ontario"] < 0.01


@pytest.mark.gee
@pytest.mark.slow
def test_sentinel2_cloud_score_plus(out_dir):
    from agribound.composites import get_composite_builder

    builder = get_composite_builder("sentinel2")
    path = builder.build(_cfg(out_dir, "sentinel2", s2_cloud_mask="cloud_score_plus"))
    med, valid, info = _band_medians(path)
    print(f"\nS2 Cloud Score+: {info} valid={valid:.3f} B4={med['B4']:.0f} B8={med['B8']:.0f}")
    assert valid > 0.9 and 100 < med["B4"] < 6000
    assert "GOOGLE/CLOUD_SCORE_PLUS" in builder.last_metadata["AGRIBOUND_COLLECTIONS"]


IOWA_TINY = "bbox:-93.600,42.000,-93.597,42.002"


@pytest.mark.gee
@pytest.mark.slow
def test_naip_iowa_4band_uint8(out_dir):
    from agribound.composites import get_composite_builder

    cfg = _cfg(out_dir, "naip", year=2023, study_area=IOWA_TINY, naip_resolution_m=1.0)
    path = get_composite_builder("naip").build(cfg)
    with rasterio.open(path) as src:
        data = src.read()
        print(f"\nNAIP: {src.crs} {src.shape} {src.res} medians={np.median(data, axis=(1, 2))}")
        assert src.count == 4 and src.dtypes[0] == "uint8" and src.res == (1.0, 1.0)
        assert src.crs.to_epsg() == 32615 and src.descriptions == ("R", "G", "B", "N")
    assert (data[3] > 0).mean() > 0.99  # NIR present everywhere


@pytest.mark.network
@pytest.mark.slow
def test_usgs_naip_plus_iowa(out_dir):
    from agribound.composites import get_composite_builder

    cfg = _cfg(out_dir, "usgs-naip-plus", year=2023, study_area=IOWA_TINY)
    path = get_composite_builder("usgs-naip-plus").build(cfg)
    with rasterio.open(path) as src:
        data = src.read()
        print(f"\nUSGS NAIP Plus: {src.crs} {src.shape} {src.res}")
        assert src.count == 4 and src.dtypes[0] == "uint8" and src.crs.to_epsg() == 32615
        assert src.res[0] < 1.0  # state vintage (0.3-0.6 m), not resampled to 1 m
    assert (data == 0).all(axis=0).mean() < 0.01
    with pytest.raises(ValueError, match="Years available for this area"):
        get_composite_builder("usgs-naip-plus").build(cfg.merged(year=2016))


@pytest.mark.gee
@pytest.mark.slow
def test_spot_namoi_multispectral_dn(out_dir):
    from agribound.composites import get_composite_builder

    builder = get_composite_builder("spot")
    path = builder.build(_cfg(out_dir, "spot", cloud_cover_max=30))
    med, valid, info = _band_medians(path)
    print(f"\nSPOT: {info} valid={valid:.3f} medians={med}")
    assert list(med) == ["R", "G", "B", "N"] and info["res"] == (6.0, 6.0)
    assert builder.last_metadata["AGRIBOUND_VALUE_SCALE"] == "dn"


@pytest.mark.network
@pytest.mark.slow
def test_tessera_coverage_v11(out_dir):
    from agribound.composites.local import tessera_coverage

    cache = os.environ.get("AGRIBOUND_TESSERA_REGISTRY")  # optional pre-downloaded manifests
    report = tessera_coverage(NAMOI_BOUNDS, 2023, version="v1.1", cache_dir=cache)
    print(f"\nTESSERA coverage: {report}")
    assert report["tiles_expected"] >= 1 and report["fraction"] == 1.0
