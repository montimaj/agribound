"""Tests for agribound.composites.vhr_sar (open VHR SAR: Umbra and Capella).

Everything runs offline: the buckets are served by :class:`FakeBucket`, the
Copernicus DEM and EGM2008 readers are replaced, and the products are synthetic
GeoTIFFs (an Umbra GEC with RPCs, a Capella GEO in UTM). The tests marked
``network`` read real headers from the public buckets.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import time
import urllib.parse
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pytest
import rasterio
from rasterio.rpc import RPC
from rasterio.transform import Affine, from_origin
from shapely.geometry import box

from agribound.composites import vhr_sar as vs
from agribound.composites.base import NoDataError, get_composite_builder
from agribound.composites.gee import ExportGrid, compute_export_grid
from agribound.config import AgriboundConfig, effective_bands
from agribound.provenance import canonical_config, config_hash
from agribound.registry import (
    ENGINE_REGISTRY,
    SAR_SOURCES,
    SOURCE_REGISTRY,
    VHR_SAR_SOURCES,
    engine_notes,
    engine_supports_source,
)

LON0, LAT0 = -100.0, 40.0
M_PER_DEG_LAT = 111_320.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _square(lon, lat, half_m):
    dlat = half_m / M_PER_DEG_LAT
    dlon = half_m / M_PER_DEG_LON
    return ((lon - dlon, lat - dlat), (lon + dlon, lat - dlat), (lon + dlon, lat + dlat),
            (lon - dlon, lat + dlat), (lon - dlon, lat - dlat))  # fmt: skip


def rec(collect_id="c1", when="2024-07-01T10:00:00.000000Z", **kw):
    """A CollectRecord over (LON0, LAT0) with defaults that pass every rule."""
    values = {
        "source": "umbra",
        "collect_id": collect_id,
        "datetime": when,
        "polarization": "HH",
        "orbit_state": "ascending",
        "look_side": "right",
        "incidence_deg": 30.0,
        "view_azimuth_deg": 100.0,
        "footprint": _square(LON0, LAT0, 2500),
        "product_key": f"sar-data/tasks/x/{collect_id}/{collect_id}_GEC.tif",
        "product_bytes": 100,
        "product_etag": f"etag-{collect_id}",
        "product_last_modified": "2024-07-02T00:00:00.000Z",
    }
    values.update(kw)
    return vs.CollectRecord(**values)


AOI = box(LON0 - 0.005, LAT0 - 0.005, LON0 + 0.005, LAT0 + 0.005)
WINDOW = ("2024-01-01", "2025-01-01")


class FakeBucket(vs.PublicBucket):
    """In-memory public bucket: ListObjectsV2 (with delimiter), GET (Range), HEAD, streams."""

    def __init__(self, name, objects, last_modified=None):
        super().__init__(name)
        self.objects = dict(objects)
        self.etags = {k: hashlib.md5(v).hexdigest() for k, v in self.objects.items()}
        self.last_modified = dict(last_modified or {})
        self.requests = []

    def request(self, method, path, headers=None):
        headers = dict(headers or {})
        self.requests.append((method, path, headers))
        if path.startswith("/?"):
            query = dict(urllib.parse.parse_qsl(path[2:]))
            return 200, {}, self._listing(query.get("prefix", ""), query.get("delimiter"))
        key = urllib.parse.unquote(path[1:])
        if key not in self.objects:
            return 404, {}, b""
        data = self.objects[key]
        head = {"etag": f'"{self.etags[key]}"', "content-length": str(len(data))}
        if method == "HEAD":
            return 200, head, b""
        if "Range" in headers:
            first, last = headers["Range"].split("=")[1].split("-")
            return 206, head, data[int(first) : int(last) + 1]
        return 200, head, data

    def _listing(self, prefix, delimiter):
        contents, prefixes = [], set()
        for key in sorted(k for k in self.objects if k.startswith(prefix)):
            rest = key[len(prefix) :]
            if delimiter and delimiter in rest:
                prefixes.add(prefix + rest.split(delimiter)[0] + delimiter)
            else:
                contents.append(key)
        parts = ['<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">']
        for key in contents:
            lm = self.last_modified.get(key, "2026-01-01T00:00:00.000Z")
            parts.append(
                f"<Contents><Key>{escape(key)}</Key><LastModified>{lm}</LastModified>"
                f"<ETag>&quot;{self.etags[key]}&quot;</ETag><Size>{len(self.objects[key])}</Size>"
                "</Contents>"
            )
        for p in sorted(prefixes):
            parts.append(f"<CommonPrefixes><Prefix>{escape(p)}</Prefix></CommonPrefixes>")
        parts.append("<IsTruncated>false</IsTruncated></ListBucketResult>")
        return "".join(parts).encode()

    def _stream(self, path, headers):
        key = urllib.parse.unquote(path[1:])
        self.requests.append(("STREAM", path, dict(headers)))
        if headers.get("If-Match", "").strip('"') != self.etags[key]:
            return 412, {}, iter(()), lambda: None
        data, status = self.objects[key], 200
        if "Range" in headers:
            data, status = data[int(headers["Range"].split("=")[1].rstrip("-")) :], 206
        return status, {"etag": f'"{self.etags[key]}"'}, iter([data]), lambda: None


def _fields_texture(size, seed=0, block=24):
    """Uint8 'fields': random levels per block with jittered edges and speckle (values >= 1)."""
    rng = np.random.default_rng(seed)
    n = size // block + 2
    levels = rng.integers(20, 200, size=(n, n))
    yy, xx = np.mgrid[0:size, 0:size]
    jitter = rng.integers(0, block // 3, size=(n,))
    rows = np.clip((yy + jitter[xx // block % n]) // block, 0, n - 1)
    cols = np.clip((xx + jitter[yy // block % n]) // block, 0, n - 1)
    img = levels[rows, cols] + rng.normal(0, 6, (size, size))
    return np.clip(img, 1, 255).astype(np.uint8)


# Synthetic GEC: 600 x 600 pixels of 1.5 m around (LON0, LAT0); a height h moves a
# ground point by K_SAMP * (h - HEIGHT_OFF) / HEIGHT_SCALE normalised samples, which is
# h / tan(45 deg) metres east (a 45 degree incidence looking east).
GEC_SIZE = 600
GEC_PX_M = 1.5
HEIGHT_OFF, HEIGHT_SCALE = -30.0, 100.0
LAT_SCALE = GEC_SIZE / 2 * GEC_PX_M / M_PER_DEG_LAT
LON_SCALE = GEC_SIZE / 2 * GEC_PX_M / M_PER_DEG_LON
K_SAMP = HEIGHT_SCALE / (GEC_SIZE / 2 * GEC_PX_M)  # 1 m height -> 1 m ground shift


def _rpc(k=K_SAMP):
    zeros = [0.0] * 20
    return RPC(
        height_off=HEIGHT_OFF,
        height_scale=HEIGHT_SCALE,
        lat_off=LAT0,
        lat_scale=LAT_SCALE,
        long_off=LON0,
        long_scale=LON_SCALE,
        # RPC image coordinates refer to pixel centres, the geotransform to corners.
        line_off=GEC_SIZE / 2 - 0.5,
        line_scale=GEC_SIZE / 2,
        samp_off=GEC_SIZE / 2 - 0.5,
        samp_scale=GEC_SIZE / 2,
        line_num_coeff=[0.0, 0.0, -1.0] + zeros[3:],
        line_den_coeff=[1.0] + zeros[1:],
        samp_num_coeff=[0.0, 1.0, 0.0, k] + zeros[4:],
        samp_den_coeff=[1.0] + zeros[1:],
    )


def write_gec(path, *, shift_px=(0, 0), seed=0, rpcs=True, rotation_deg=0.0):
    """Synthetic Umbra GEC: uint8, EPSG:4326 (optionally rotated) geotransform and RPCs.

    *shift_px* ``(rows, cols)`` moves the texture inside the image (content
    displaced south/east for positive values).
    """
    big = _fields_texture(GEC_SIZE + 40, seed)
    r, c = shift_px
    img = big[20 - r : 20 - r + GEC_SIZE, 20 - c : 20 - c + GEC_SIZE].copy()
    img[:3, :] = 0  # a nodata edge, as in the GECs
    dx, dy = 2 * LON_SCALE / GEC_SIZE, 2 * LAT_SCALE / GEC_SIZE
    cos, sin = math.cos(math.radians(rotation_deg)), math.sin(math.radians(rotation_deg))
    # The north-up geotransform, rotated about its origin (like a GEC's rotated grid).
    transform = Affine(
        dx * cos, -dx * sin, LON0 - LON_SCALE, -dy * sin, -dy * cos, LAT0 + LAT_SCALE
    )
    profile = {
        "driver": "GTiff",
        "width": GEC_SIZE,
        "height": GEC_SIZE,
        "count": 1,
        "dtype": "uint8",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 0,
    }
    if rpcs:
        profile["rpcs"] = _rpc()
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(img, 1)
    return path


def _stub_dem(monkeypatch, height_m=0.0, geoid_m=HEIGHT_OFF):
    """Copernicus DEM heights of *height_m* and an EGM2008 undulation of *geoid_m* (offline)."""
    monkeypatch.setattr(
        vs,
        "read_copernicus_dem",
        lambda b, shape: (np.full(shape, height_m, np.float32), {"N40_W100": "GLO-30"}),
    )
    monkeypatch.setattr(vs, "read_egm2008", lambda b, shape: np.full(shape, geoid_m, np.float32))


# ---------------------------------------------------------------------------
# Registry, configuration and hashes
# ---------------------------------------------------------------------------


def test_registry_entries_and_engine_support():
    assert VHR_SAR_SOURCES == ("umbra", "capella")
    assert SAR_SOURCES == ("sentinel1", "umbra", "capella")
    for source in VHR_SAR_SOURCES:
        info = SOURCE_REGISTRY[source]
        assert info["requires_gee"] is False and info["restricted"] is False
        assert info["all_bands"] == ["MEDIAN"] and info["resolution_m"] == 1.0
        assert info["canonical_bands"] == {"R": "MEDIAN", "G": "MEDIAN", "B": "MEDIAN"}
        for engine in ("delineate-anything", "geoai", "dinov3"):
            assert engine_supports_source(engine, source)
            assert ENGINE_REGISTRY[engine]["source_notes"][source]
        for engine in ("ftw", "prithvi", "embedding"):
            assert not engine_supports_source(engine, source)
        da = " ".join(engine_notes("delineate-anything", source))
        assert "processing_gsd_m=3" in da and "merge_tile_pieces=False" in da
    assert SOURCE_REGISTRY["umbra"]["value_scale"] == "dn"
    assert SOURCE_REGISTRY["capella"]["value_scale"] == "db"
    assert "0.439" in ENGINE_REGISTRY["delineate-anything"]["source_notes"]["umbra"]
    assert "0.50-0.54" in ENGINE_REGISTRY["delineate-anything"]["source_notes"]["capella"]
    from agribound.engines.base import get_canonical_band_indices

    assert get_canonical_band_indices("umbra", ["R", "G", "B"]) == [1, 1, 1]
    pair = effective_bands(_cfg("capella", vhr_sar_bands="median+dates"))
    assert get_canonical_band_indices("capella", ["R", "G", "B"], pair) == [1, 2, 1]
    assert type(get_composite_builder("umbra")).__name__ == "VHRSARCompositeBuilder"
    assert type(get_composite_builder("capella")).__name__ == "VHRSARCompositeBuilder"


def _cfg(source="umbra", **kw):
    base = {
        "source": source,
        "year": 2024,
        "study_area": "bbox:-100.005,39.995,-99.995,40.005",
        "lulc_filter": False,
    }
    return AgriboundConfig(**{**base, **kw})


def test_config_defaults_and_normalisation():
    cfg = _cfg()
    assert cfg.vhr_sar_collect_ids is None and cfg.vhr_sar_max_collects == 6
    assert cfg.vhr_sar_polarization == "auto" and cfg.vhr_sar_resolution_m == 1.0
    assert cfg.vhr_sar_coregister == "auto" and cfg.vhr_sar_bands == "median"
    assert cfg.vhr_sar_calibration == "any" and cfg.vhr_sar_max_download_gb == 5.0
    assert not cfg.is_gee_source() and not cfg.requires_gee() and cfg.bands is None
    cfg = _cfg(
        "capella",
        vhr_sar_collect_ids=" b, a ,b",
        vhr_sar_polarization="vv",
        vhr_sar_bands="Median+Dates",
        vhr_sar_calibration="FULL",
    )
    assert cfg.vhr_sar_collect_ids == ("b", "a") and cfg.vhr_sar_polarization == "VV"
    # median+dates feeds G the closest pair; the mapping is derived, not stored in bands.
    assert cfg.bands is None and effective_bands(cfg) == {"R": 1, "G": 2, "B": 1}
    assert cfg.to_dict()["vhr_sar_collect_ids"] == ["b", "a"]
    assert AgriboundConfig.from_dict(cfg.to_dict()) == cfg
    given = _cfg(vhr_sar_bands="median+dates", bands={"R": 1, "G": 3, "B": 1})
    assert effective_bands(given)["G"] == 3
    # Back to the median: no stale mapping, and the hash of a fresh median configuration.
    back = _cfg(vhr_sar_bands="median+dates").merged(vhr_sar_bands="median")
    assert back.bands is None and effective_bands(back) is None
    assert config_hash(back) == config_hash(_cfg())
    s2 = _cfg("sentinel2", vhr_sar_bands="median+dates", gee_project="p")
    assert effective_bands(s2) is None


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"vhr_sar_calibration": "full"}, "applies to source='capella' only"),
        ({"composite_method": "greenest"}, "does not apply to source 'umbra'"),
        ({"vhr_sar_polarization": "HV"}, "Invalid vhr_sar_polarization"),
        ({"vhr_sar_max_collects": 0}, r"\[1, 50\]"),
        ({"vhr_sar_resolution_m": 30}, "at most 10 m"),
        ({"vhr_sar_coregister": "sar", "vhr_sar_coregister_base": "x.tif"}, "does not use"),
        ({"vhr_sar_collect_ids": []}, "empty"),
        ({"engine": "ftw"}, "does not support source 'umbra'"),
        ({"engine": "ensemble"}, "default member"),
        ({"vhr_sar_bands": "median+dates", "vhr_sar_max_collects": 1}, "at least 2 collects"),
        ({"vhr_sar_bands": "median+dates", "vhr_sar_collect_ids": "a"}, "at least 2 collects"),
    ],
)
def test_config_rejects(kw, match):
    with pytest.raises(ValueError, match=match):
        _cfg(**kw)


def test_vhr_fields_enter_the_hash_only_for_their_sources(monkeypatch):
    monkeypatch.setenv("GEE_PROJECT", "p")
    s2 = AgriboundConfig(source="sentinel2", study_area="bbox:0,0,1,1", year=2023)
    changed = s2.merged(
        vhr_sar_max_collects=3,
        vhr_sar_polarization="VV",
        vhr_sar_resolution_m=2.0,
        vhr_sar_coregister="none",
        vhr_sar_bands="median+dates",
        vhr_sar_collect_ids=["x"],
    )
    assert config_hash(changed) == config_hash(s2)
    assert not any(k.startswith("vhr_sar") for k in canonical_config(s2))
    umbra = _cfg()
    keys = {k for k in canonical_config(umbra) if k.startswith("vhr_sar")}
    assert keys == {
        "vhr_sar_collect_ids",
        "vhr_sar_max_collects",
        "vhr_sar_polarization",
        "vhr_sar_resolution_m",
        "vhr_sar_coregister",
        "vhr_sar_coregister_base",
        "vhr_sar_bands",
    }
    assert "vhr_sar_calibration" in canonical_config(_cfg("capella"))
    for change in (
        {"vhr_sar_max_collects": 4},
        {"vhr_sar_resolution_m": 2.0},
        {"vhr_sar_collect_ids": "a,b"},
        {"vhr_sar_coregister": "sar"},
    ):
        assert config_hash(umbra.merged(**change)) != config_hash(umbra)
    # A download cap and a cache location never change the result.
    same = umbra.merged(vhr_sar_max_download_gb=50.0, vhr_sar_cache_dir="/scratch/v")
    assert config_hash(same) == config_hash(umbra)
    assert config_hash(_cfg("capella").merged(vhr_sar_calibration="full")) != config_hash(
        _cfg("capella")
    )


# ---------------------------------------------------------------------------
# Index parsing, copies and archive churn
# ---------------------------------------------------------------------------


def _umbra_item(collect_id, when, pol="HH", inc=30.0, az=100.0, z=-30.0):
    ring = [[x, y, z] for x, y in _square(LON0, LAT0, 2500)]
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [ring]},
        "properties": {
            "umbra:collect_id": collect_id,
            "datetime": when,
            "platform": "Umbra-05",
            "sar:polarizations": [pol],
            "sat:orbit_state": "ascending",
            "sar:observation_direction": "right",
            "view:incidence_angle": inc,
            "view:azimuth": az,
            "umbra:squint_angle_degrees_off_broadside": 5.0,
            "sar:instrument_mode": "SPOTLIGHT",
            "sar:resolution_range": 0.5,
            "sar:resolution_azimuth": 0.5,
            "sar:looks_range": 1,
            "sar:looks_azimuth": 2,
            "processing:version": "4.31.2",
        },
    }


def test_parse_umbra_stac_takes_polarization_and_geometry_from_the_sidecar():
    out = vs.parse_umbra_stac(_umbra_item("c", "2024-07-01T10:00:00Z", pol="vv", az=370.0))
    assert out["polarization"] == "VV" and out["view_azimuth_deg"] == pytest.approx(10.0)
    assert out["datetime"] == "2024-07-01T10:00:00.000000Z"
    assert out["reference_height_m"] == pytest.approx(-30.0)
    assert len(out["footprint"]) == 5 and len(out["footprint"][0]) == 2
    bad = _umbra_item("c", "2024-07-01T10:00:00Z")
    bad["properties"]["sar:polarizations"] = ["HH", "VV"]
    with pytest.raises(ValueError, match="one expected"):
        vs.parse_umbra_stac(bad)


def _umbra_objects(entries):
    """Bucket objects for (folder, collect_id, datetime, pol, gec bytes, with_sidecar)."""
    objects, modified = {}, {}
    for folder, cid, when, pol, gec, with_stac, uploaded in entries:
        name = folder.rsplit("/", 1)[-1]
        objects[f"{folder}/{name}_GEC.tif"] = gec
        modified[f"{folder}/{name}_GEC.tif"] = uploaded
        objects[f"{folder}/{name}_SICD.nitf"] = b"never downloaded"
        if with_stac:
            item = _umbra_item(cid, when, pol=pol)
            objects[f"{folder}/{name}.stac.v2.json"] = json.dumps(item).encode()
        else:
            meta = {"collects": [{"id": cid, "polarizations": ["VV"]}]}
            objects[f"{folder}/{name}_METADATA.json"] = json.dumps(meta).encode()
    return objects, modified


def test_umbra_index_dedupes_copies_keeps_distinct_collects_and_drops_future_dates(tmp_path):
    objects, modified = _umbra_objects(
        [
            # Two copies of one collect: the newer upload (task-data) wins.
            ("sar-data/tasks/Site A/t1/2024-07-01-10-00-00_UMBRA-05", "c1",
             "2024-07-01T10:00:00Z", "HH", b"old" * 10, True, "2024-07-02T00:00:00.000Z"),
            ("sar-data/task-data/t9/2024-07-01-10-00-00_UMBRA-05", "c1",
             "2024-07-01T10:00:00Z", "HH", b"new" * 11, True, "2025-01-01T00:00:00.000Z"),
            # Distinct collects 10 s apart stay separate.
            ("sar-data/tasks/Site A/t2/2024-07-01-10-00-10_UMBRA-06", "c2",
             "2024-07-01T10:00:10Z", "HH", b"x" * 20, True, "2024-07-02T00:00:00.000Z"),
            # Impossible year: dropped.
            ("open-data/u1/2035-03-10-12-45-00_UMBRA-07", "c3",
             "2035-03-10T12:45:00Z", "VV", b"y" * 20, True, "2025-01-01T00:00:00.000Z"),
            # No sidecar (METADATA.json only): not used.
            ("open-data/u2/2024-08-01-10-00-00_UMBRA-07", "c4",
             "2024-08-01T10:00:00Z", "VV", b"z" * 20, False, "2025-01-01T00:00:00.000Z"),
        ]
    )  # fmt: skip
    bucket = FakeBucket("umbra-open-data-catalog", objects, modified)
    index = vs.refresh_index("umbra", tmp_path, bucket=bucket, workers=4)
    by_id = index.by_id()
    assert sorted(by_id) == ["c1", "c2"]
    assert by_id["c1"].n_copies == 2 and by_id["c1"].product_key.startswith("sar-data/task-data/")
    assert by_id["c1"].product_bytes == 33
    assert index.stats["products"] == 5 and index.stats["products_without_stac"] == 1
    assert index.stats["future_dropped"] == 1
    assert not any("SICD" in m[1] for m in bucket.requests)  # complex products never read
    # The cache file holds the snapshot; reloading gives the same records.
    assert vs.index_path("umbra", tmp_path).exists()
    again = vs.load_index("umbra", tmp_path, bucket=bucket)
    assert again.snapshot == index.snapshot and again.records == index.records
    # A refresh reads only new or changed sidecars.
    n_gets = sum(1 for m in bucket.requests if m[0] == "GET" and not m[1].startswith("/?"))
    vs.refresh_index("umbra", tmp_path, bucket=bucket, workers=4)
    n_gets_after = sum(1 for m in bucket.requests if m[0] == "GET" and not m[1].startswith("/?"))
    assert n_gets_after == n_gets


def test_load_index_refreshes_only_for_windows_after_an_old_snapshot(tmp_path, monkeypatch):
    import datetime as dt

    calls = []
    index = vs.ArchiveIndex("umbra", "2026-01-01T00:00:00Z", [])
    vs.index_path("umbra", tmp_path).parent.mkdir(parents=True)
    monkeypatch.setattr(vs, "_read_index", lambda path: index)
    monkeypatch.setattr(vs, "refresh_index", lambda *a, **k: calls.append(1) or index)
    vs.index_path("umbra", tmp_path).write_bytes(b"x")
    now = dt.datetime(2026, 3, 1, tzinfo=dt.UTC)
    vs.load_index("umbra", tmp_path, window_end="2025-12-31", now=now)
    assert calls == []  # the window ended before the snapshot
    vs.load_index("umbra", tmp_path, window_end="2026-02-01", now=now)
    assert calls == [1]  # the window ends after an old snapshot
    vs.load_index(
        "umbra", tmp_path, window_end="2026-02-01", now=dt.datetime(2026, 1, 1, 12, tzinfo=dt.UTC)
    )
    assert calls == [1]  # snapshot younger than INDEX_REFRESH_MIN_AGE


def test_capella_extended_head_parsing():
    head = json.dumps(
        {
            "software_version": "2.58.0",
            "collect": {
                "collect_id": "abc",
                "image": {
                    "scale_factor": 0.0004,
                    "calibration": "limited",
                    "radiometry": "sigma_nought",
                    "center_pixel": {"incidence_angle": 36.572649739},
                },
                "radar": {"pointing": "left", "time_varying_parameters": [1] * 100},
            },
        }
    )
    out = vs.parse_capella_extended(head[:-40])  # truncated after the image object
    assert out["incidence_deg"] == pytest.approx(36.572649739)
    assert out["calibration"] == "limited" and out["scale_factor"] == 0.0004
    assert out["collect_id"] == "abc"
    with pytest.raises(ValueError):
        vs.parse_capella_extended('{"collect": {"image": {"scale_factor": 1')


def test_bucket_list_follows_continuation_tokens():
    pages = {
        None: "<Contents><Key>a/1</Key><LastModified>t</LastModified><ETag>&quot;e1&quot;</ETag>"
        "<Size>1</Size></Contents><NextContinuationToken>tok</NextContinuationToken>"
        "<IsTruncated>true</IsTruncated>",
        "tok": "<Contents><Key>a/2</Key><LastModified>t</LastModified><ETag>&quot;e2&quot;</ETag>"
        "<Size>2</Size></Contents><IsTruncated>false</IsTruncated>",
    }

    class Paged(vs.PublicBucket):
        def request(self, method, path, headers=None):
            token = dict(urllib.parse.parse_qsl(path[2:])).get("continuation-token")
            body = (
                f'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">{pages[token]}'
            )
            return 200, {}, (body + "</ListBucketResult>").encode()

    objects, _ = Paged("b").list("a/")
    assert [(o.key, o.size, o.etag) for o in objects] == [("a/1", 1, "e1"), ("a/2", 2, "e2")]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def test_azimuth_spread_wraps_at_north():
    assert vs.azimuth_spread([355.0, 5.0]) == pytest.approx(10.0)
    assert vs.azimuth_spread([350.0, 10.0, 20.0]) == pytest.approx(30.0)
    assert vs.azimuth_spread([100.0]) == 0.0


def test_consistent_subset_respects_both_tolerances_across_the_wrap():
    records = [
        rec("a", "2024-07-01T10:00:00Z", view_azimuth_deg=355.0, incidence_deg=30.0),
        rec("b", "2024-07-05T10:00:00Z", view_azimuth_deg=5.0, incidence_deg=31.0),
        rec("c", "2024-07-09T10:00:00Z", view_azimuth_deg=15.0, incidence_deg=34.9),
        rec("d", "2024-07-13T10:00:00Z", view_azimuth_deg=200.0, incidence_deg=30.5),
        rec("e", "2024-07-17T10:00:00Z", view_azimuth_deg=0.0, incidence_deg=40.0),
    ]
    sel = vs.select_collects(records, AOI, WINDOW, source="umbra")
    assert [r.collect_id for r in sel.chosen] == ["a", "b", "c"]
    assert sel.group["view_azimuth_spread_deg"] == pytest.approx(20.0)
    assert sel.group["incidence_spread_deg"] <= vs.INCIDENCE_TOLERANCE_DEG


def test_selection_never_mixes_polarizations_or_look_sides():
    records = [rec(f"h{i}", f"2024-07-{i + 1:02d}T10:00:00Z") for i in range(3)]
    records += [rec(f"v{i}", f"2024-08-{i + 1:02d}T10:00:00Z", polarization="VV") for i in range(4)]
    records += [rec("l1", "2024-09-01T10:00:00Z", polarization="VV", look_side="left")]
    sel = vs.select_collects(records, AOI, WINDOW, source="umbra")
    assert {r.polarization for r in sel.chosen} == {"VV"} and len(sel.chosen) == 4
    assert {r.look_side for r in sel.chosen} == {"right"}
    sel = vs.select_collects(records, AOI, WINDOW, source="umbra", polarization="HH")
    assert [r.collect_id for r in sel.chosen] == ["h0", "h1", "h2"]
    with pytest.raises(ValueError, match=r"mixed polarizations \['HH', 'VV'\]"):
        vs.select_collects(records, AOI, WINDOW, source="umbra", pinned_ids=["h0", "v0"])


def test_selection_spreads_up_to_max_collects_and_is_deterministic():
    records = [rec(f"c{i:02d}", f"2024-06-{i + 1:02d}T10:00:00Z") for i in range(10)]
    sel = vs.select_collects(records, AOI, WINDOW, source="umbra", max_collects=4)
    assert [r.collect_id for r in sel.chosen] == ["c00", "c03", "c06", "c09"]
    shuffled = list(reversed(records))
    again = vs.select_collects(shuffled, AOI, WINDOW, source="umbra", max_collects=4)
    assert [r.collect_id for r in again.chosen] == [r.collect_id for r in sel.chosen]
    assert again.reference == sel.reference


def test_selection_errors_and_single_date_warning():
    far = rec("far", footprint=_square(LON0 + 1, LAT0, 2500))
    with pytest.raises(NoDataError, match="covers the study area"):
        vs.select_collects([far], AOI, WINDOW, source="umbra")
    # Only part of the study area inside the footprints, and a study area bigger than a scene.
    edge = rec("edge", footprint=_square(LON0 + 0.006, LAT0, 700))
    with pytest.raises(NoDataError, match="covers the whole study area"):
        vs.select_collects([edge], AOI, WINDOW, source="umbra")
    huge = box(LON0 - 0.2, LAT0 - 0.2, LON0 + 0.2, LAT0 + 0.2)
    with pytest.raises(ValueError, match="larger than one umbra scene"):
        vs.select_collects([rec("a")], huge, WINDOW, source="umbra")
    # Outside the window: the error lists the nearest dates.
    with pytest.raises(NoDataError, match="2024-07-01"):
        vs.select_collects([rec("a")], AOI, ("2025-01-01", "2026-01-01"), source="umbra")
    # One consistent date: an error unless a single date is asked for.
    with pytest.raises(NoDataError, match="vhr_sar_max_collects=1"):
        vs.select_collects([rec("a")], AOI, WINDOW, source="umbra")
    sel = vs.select_collects([rec("a")], AOI, WINDOW, source="umbra", max_collects=1)
    assert len(sel.chosen) == 1 and "0.00-0.08" in sel.warnings[0]


def test_capella_calibration_filter_and_pinned_rules():
    records = [
        rec("f1", "2023-01-06T11:00:00Z", source="capella", calibration="full"),
        rec("f2", "2023-01-12T11:00:00Z", source="capella", calibration="full"),
        rec("p1", "2023-01-09T11:00:00Z", source="capella", calibration="partial"),
    ]
    window = ("2023-01-01", "2023-02-01")
    sel = vs.select_collects(records, AOI, window, source="capella")
    assert len(sel.chosen) == 3
    sel = vs.select_collects(records, AOI, window, source="capella", calibration="full")
    assert [r.collect_id for r in sel.chosen] == ["f1", "f2"]
    with pytest.raises(ValueError, match="not fully calibrated"):
        vs.select_collects(
            records, AOI, window, source="capella", calibration="full", pinned_ids=["f1", "p1"]
        )
    with pytest.raises(ValueError, match="not in the capella index"):
        vs.select_collects(records, AOI, window, source="capella", pinned_ids=["nope"])
    with pytest.raises(ValueError, match="outside the date window"):
        vs.select_collects(records, AOI, ("2023-01-10", "2023-02-01"), source="capella",
                          pinned_ids=["f1", "f2"])  # fmt: skip
    tilted = rec("t", "2023-01-20T11:00:00Z", source="capella", view_azimuth_deg=140.0)
    with pytest.raises(ValueError, match="view-azimuth spread 40.0"):
        vs.select_collects(
            [*records, tilted], AOI, window, source="capella", pinned_ids=["f1", "t"]
        )


# ---------------------------------------------------------------------------
# Radiometry and compositing
# ---------------------------------------------------------------------------


def test_capella_db_conversion():
    dn = np.array([0, 1000, 2500], dtype=np.float32)
    out = vs.capella_db(dn, 0.0004)
    assert out[0] == 0  # nodata stays 0
    assert out[1] == pytest.approx(20 * math.log10(0.4) + vs.CAPELLA_DB_OFFSET, abs=1e-4)
    assert out[2] == pytest.approx(20 * math.log10(1.0) + vs.CAPELLA_DB_OFFSET, abs=1e-4)
    assert vs.capella_db(np.array([1e-9], np.float32), 1e-6)[0] == pytest.approx(0.01)


def test_level_normalisation_and_median():
    rng = np.random.default_rng(1)
    base = rng.uniform(5, 60, (64, 64)).astype(np.float32)
    a, b, c = base + 10, base - 4, base + 1  # Umbra DN: levels differ additively
    b[:, :4] = 0  # nodata strip in one date
    out, info = vs.level_normalise([a, b, c])
    assert info["basis"] == "common" and len(info["offsets"]) == 3
    assert np.isnan(out[1][:, :4]).all()
    target = info["target"]
    for arr in out:
        assert np.nanmedian(arr[:, 4:]) == pytest.approx(target, abs=1e-3)
    med = vs.median_composite(out, lower=1.0, upper=255.0)
    assert med.shape == base.shape and (med[:, 4:] > 0).all()
    # Pixels with one valid date (of three) are nodata only when fewer than 2 are valid.
    one = [a, np.zeros_like(a), np.zeros_like(a)]
    assert (vs.median_composite(vs.level_normalise(one)[0], lower=1.0) == 0).all()
    # A value pushed to or below 0 by the shift stays valid (the floor clips the median).
    dark = np.full((40, 40), 81.0, np.float32)
    dark[:, :20] = 1.0  # median 41: shifted down by 15 below
    flat = np.full((40, 40), 11.0, np.float32)
    normalised, info = vs.level_normalise([flat, dark])
    assert info["target"] == pytest.approx(26.0)
    assert np.isfinite(normalised[1]).all() and (normalised[1][:, :20] < 0).all()
    med = vs.median_composite(normalised, lower=1.0)
    assert (med >= 1).all() and med[0, 0] == pytest.approx(6.0)


def test_pair_mean_and_closest_pair():
    records = [
        rec("a", "2024-07-01T10:00:00Z"),
        rec("b", "2024-07-20T10:00:00Z"),
        rec("c", "2024-07-22T10:00:00Z"),
    ]
    assert vs.closest_pair(records) == (1, 2)
    a = np.array([[2.0, np.nan]], np.float32)
    b = np.array([[4.0, 5.0]], np.float32)
    assert vs.pair_mean(a, b, lower=1.0).tolist() == [[3.0, 0.0]]


# ---------------------------------------------------------------------------
# Geometry: warps, DEM and co-registration
# ---------------------------------------------------------------------------


def _grid(half_m=300.0, res=1.0):
    import pyproj

    x, y = pyproj.Transformer.from_crs(4326, 32614, always_xy=True).transform(LON0, LAT0)
    x0, y1 = math.floor(x - half_m), math.ceil(y + half_m)
    n = int(2 * half_m / res)
    return ExportGrid("EPSG:32614", from_origin(x0, y1, res, res), n, n)


def _dem(path, height):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=200,
        height=200,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(LON0 - 0.05, LAT0 + 0.05, 0.0005, 0.0005),
    ) as dst:
        dst.write(np.full((200, 200), height, np.float32), 1)
    return path


def test_rpc_warp_moves_with_dem_height_by_h_over_tan_incidence(tmp_path):
    gec = write_gec(tmp_path / "gec.tif")
    grid = _grid()
    at0 = vs.warp_product(gec, grid, source="umbra", method="rpc_dem",
                         dem_path=_dem(tmp_path / "d0.tif", HEIGHT_OFF))  # fmt: skip
    up = vs.warp_product(gec, grid, source="umbra", method="rpc_dem",
                        dem_path=_dem(tmp_path / "d50.tif", HEIGHT_OFF + 50.0))  # fmt: skip
    assert (at0 > 0).mean() > 0.99 and (up > 0).mean() > 0.99
    shift = vs.register_adaptive(at0, up, 1.0, kind="sar")
    assert shift["ok"]
    # 50 m too high at 45 degrees incidence (looking east): content 50 m to the west.
    assert shift["dx"] == pytest.approx(-50.0, abs=1.0)
    assert shift["dy"] == pytest.approx(0.0, abs=1.5)  # UTM grid convergence: about 0.6 deg


def test_rotated_geotransform_fallback_and_rpc_agree_at_the_reference_height(tmp_path):
    gec = write_gec(tmp_path / "gec.tif")
    grid = _grid()
    rpc = vs.warp_product(gec, grid, source="umbra", method="rpc_dem",
                         dem_path=_dem(tmp_path / "d0.tif", HEIGHT_OFF))  # fmt: skip
    affine = vs.warp_product(gec, grid, source="umbra", method="geotransform")
    valid = (rpc > 0) & (affine > 0)
    assert valid.mean() > 0.95
    shift = vs.register_adaptive(rpc, affine, 1.0, kind="sar")  # same geometry at HEIGHT_OFF
    assert shift["ok"] and math.hypot(shift["dx"], shift["dy"]) < 0.3
    rotated = write_gec(tmp_path / "rot.tif", rotation_deg=5.0)
    with rasterio.open(rotated) as src:
        assert src.transform.b != 0 and src.rpcs is not None


def test_build_dem_hae_adds_the_geoid(tmp_path):
    info = vs.build_dem_hae(
        (LON0 - 0.01, LAT0 - 0.01, LON0 + 0.01, LAT0 + 0.01),
        tmp_path / "dem.tif",
        read_dem=lambda b, shape: np.full(shape, 100.0, np.float32),
        read_geoid=lambda b, shape: np.full(shape, -31.0, np.float32),
    )
    with rasterio.open(info["path"]) as src:
        assert src.crs.to_epsg() == 4326
        assert np.allclose(src.read(1), 69.0)
        assert abs(src.res[0] - 1 / 3600) < 1e-9
    assert info["geoid_undulation_m"] == [-31.0, -31.0]
    assert vs.glo30_tile_name(38, -122) == "Copernicus_DSM_COG_10_N38_00_W122_00_DEM"
    assert vs.glo30_tile_name(-1, 5) == "Copernicus_DSM_COG_10_S01_00_E005_00_DEM"


def test_cached_dem_info_points_at_the_moved_cache(tmp_path):
    """A DEM record written in another cache directory resolves to where the DEM is now."""
    built = tmp_path / "old_cache" / "dem" / "dem.tif"
    built.parent.mkdir(parents=True)
    info = vs.build_dem_hae(
        (LON0 - 0.01, LAT0 - 0.01, LON0 + 0.01, LAT0 + 0.01),
        built,
        read_dem=lambda b, shape: np.full(shape, 100.0, np.float32),
        read_geoid=lambda b, shape: np.full(shape, -31.0, np.float32),
    )
    built.with_suffix(".json").write_text(json.dumps(info))
    moved = tmp_path / "new_cache" / "dem"
    moved.parent.mkdir()
    built.parent.rename(moved)

    cached = vs._cached_dem_info(moved / "dem.tif")

    assert cached["path"] == str(moved / "dem.tif")
    assert Path(cached["path"]).is_file()
    assert cached["geoid_undulation_m"] == info["geoid_undulation_m"]


def test_coregistration_sign_and_shift_back():
    rng = np.random.default_rng(3)
    img = _fields_texture(900, seed=5).astype(np.float32)
    # Content displaced 7 px east and 4 px north: column +7, row -4.
    moved = np.zeros_like(img)
    moved[:-4, 7:] = img[4:, :-7]
    result = vs.register_adaptive(img, moved, 1.0, kind="sar")
    assert result["ok"] and result["scale"] == 1
    assert result["dx"] == pytest.approx(7.0, abs=0.3)
    assert result["dy"] == pytest.approx(4.0, abs=0.3)
    back = vs.shift_image(moved, result["dx"], result["dy"], 1.0)
    core = (slice(50, -50), slice(50, -50))
    assert np.corrcoef(back[core].ravel(), img[core].ravel())[0, 1] > 0.95
    # An optical base (three bands) gives the same sign.
    base = np.stack([img, img * 0.8 + 5, img * 0.5 + 10]) + rng.normal(0, 1, (3, 900, 900))
    opt = vs.register_adaptive(np.clip(base, 1, None), moved, 1.0, kind="optical")
    assert opt["ok"] and opt["dx"] == pytest.approx(7.0, abs=0.5)


def test_warp_shift_lands_displaced_content_at_the_truth(tmp_path):
    """warp_product(shift=(dx, dy)) undoes content displaced by (dx, dy)."""
    grid = _grid(200.0)
    texture = _fields_texture(500, seed=9)
    path = tmp_path / "north_up.tif"
    t = grid.transform
    with rasterio.open(
        path, "w", driver="GTiff", width=500, height=500, count=1, dtype="uint8",
        crs=grid.crs, transform=Affine(1, 0, t.c - 50, 0, -1, t.f + 50), nodata=0,
    ) as dst:  # fmt: skip
        dst.write(texture, 1)
    truth = vs.warp_product(path, grid, source="umbra", method="geotransform")
    # The same image, its content displaced 6 m east and 3 m south.
    shifted_path = tmp_path / "shifted.tif"
    with rasterio.open(
        shifted_path, "w", driver="GTiff", width=500, height=500, count=1, dtype="uint8",
        crs=grid.crs, transform=Affine(1, 0, t.c - 50 + 6, 0, -1, t.f + 50 - 3), nodata=0,
    ) as dst:  # fmt: skip
        dst.write(texture, 1)
    moved = vs.warp_product(shifted_path, grid, source="umbra", method="geotransform")
    est = vs.register_adaptive(truth, moved, 1.0, kind="sar")
    assert est["dx"] == pytest.approx(6.0, abs=0.3) and est["dy"] == pytest.approx(-3.0, abs=0.3)
    fixed = vs.warp_product(
        shifted_path, grid, source="umbra", method="geotransform", shift=(est["dx"], est["dy"])
    )
    valid = (fixed > 0) & (truth > 0)
    assert np.abs(fixed[valid] - truth[valid]).mean() < 1.0


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def test_downloads_resume_check_etag_and_refuse_complex_products(tmp_path):
    payload = bytes(range(256)) * 40
    key = "sar-data/tasks/x/2024-07-01-10-00-00_UMBRA-05/2024-07-01-10-00-00_UMBRA-05_GEC.tif"
    bucket = FakeBucket("umbra-open-data-catalog", {key: payload})
    record = rec("c1", product_key=key, product_bytes=len(payload), product_etag=bucket.etags[key])
    dest = vs.product_path(tmp_path, record)
    dest.parent.mkdir(parents=True)
    # An interrupted download of this version (partial files carry their ETag).
    dest.with_name(f"{dest.name}.{bucket.etags[key]}.part").write_bytes(payload[:1000])
    path, info = vs.ensure_product(bucket, record, tmp_path)
    assert path.read_bytes() == payload and info["etag"] == bucket.etags[key]
    streams = [m for m in bucket.requests if m[0] == "STREAM"]
    assert streams[-1][2]["Range"] == "bytes=1000-"
    assert streams[-1][2]["If-Match"] == f'"{bucket.etags[key]}"'
    # Cached: no new request; a changed ETag in the index means the object changed.
    n = len(bucket.requests)
    assert vs.ensure_product(bucket, record, tmp_path)[0] == path and len(bucket.requests) == n
    changed = rec("c1", product_key=key, product_bytes=len(payload), product_etag="other")
    assert vs.cached_product(tmp_path, changed) is None
    with pytest.raises(vs.ProductChangedError):
        vs.ensure_product(bucket, changed, tmp_path)
    for bad in ("x/y_CPHD.cphd", "x/y_SICD.nitf", "x/y_GEC.tiff"):
        with pytest.raises(ValueError, match="Refusing"):
            vs.check_product_key("umbra", bad)
    with pytest.raises(ValueError, match="Refusing"):
        vs.check_product_key(
            "capella", "data/CAPELLA_C08_SP_SLC_VV_20230106110004_20230106110028.tif"
        )
    vs.check_product_key("capella", "data/CAPELLA_C08_SP_GEO_VV_20230106110004_20230106110028.tif")


# ---------------------------------------------------------------------------
# Provenance: tags, attribution and cache key
# ---------------------------------------------------------------------------


def test_providers_carry_licence_credit_and_citation():
    umbra, capella = vs.get_provider("umbra"), vs.get_provider("capella")
    assert umbra.attribution == "SAR data: Umbra Lab, Inc., CC BY 4.0"
    assert capella.attribution == "© Capella Space Corp, All Rights Reserved."
    assert umbra.citation("2026-10-09") == (
        "Umbra Synthetic Aperture Radar (SAR) Open Data was accessed on 2026-10-09 from "
        "https://registry.opendata.aws/umbra-open-data."
    )
    assert capella.citation("2026-10-09").startswith(
        "Capella Space Synthetic Aperture Radar (SAR) Open Dataset was accessed on 2026-10-09"
    )
    assert umbra.product == "GEC" and capella.product == "GEO"
    assert "Modified by agribound" in capella.changes
    with pytest.raises(ValueError, match="not an open VHR SAR source"):
        vs.get_provider("sentinel1")


def test_cache_key_parts_hold_sorted_ids_etags_settings_and_snapshot(tmp_path):
    a, b = rec("a"), rec("b", "2024-07-09T10:00:00Z")
    sel = vs.Selection([a, b], a, {}, [], False, {})
    sel_rev = vs.Selection([b, a], a, {}, [], False, {})
    cfg = _cfg()
    snapshot = "2026-01-01T00:00:00Z"

    def key(config):
        return vs._cache_key_parts(config, sel, snapshot)

    parts = key(cfg)
    assert parts == vs._cache_key_parts(cfg, sel_rev, snapshot)
    assert "a:etag-a,b:etag-b" in parts and vs.VHR_SAR_RECIPE_TAG in parts
    assert parts != vs._cache_key_parts(cfg, sel, "2026-02-01T00:00:00Z")
    # Every setting that changes the composite enters the key (they enter no other key).
    for change in (
        {"vhr_sar_resolution_m": 2.0},
        {"vhr_sar_bands": "median+dates"},
        {"vhr_sar_coregister": "sar"},
        {"vhr_sar_polarization": "VV"},
        {"vhr_sar_max_collects": 3},
        {"vhr_sar_collect_ids": "a,b"},
        {"vhr_sar_collect_ids": "b,a"},
    ):
        assert key(cfg.merged(**change)) != parts, change
    capella = _cfg("capella")
    assert key(capella.merged(vhr_sar_calibration="full")) != key(capella)
    # The base file itself: an orthophoto replaced at the same path gives another key.
    base = tmp_path / "ortho.tif"
    base.write_bytes(b"1")
    with_base = cfg.merged(vhr_sar_coregister_base=str(base))
    first = key(with_base)
    assert first != parts and str(base.resolve()) in " ".join(first)
    base.write_bytes(b"22")
    assert key(with_base) != first
    assert "coregister_base_file" not in " ".join(parts)  # no base: the key is as before


# ---------------------------------------------------------------------------
# Builders with mocked HTTP
# ---------------------------------------------------------------------------


def _umbra_bucket(tmp_path, shifts):
    objects, modified = {}, {}
    for i, shift in enumerate(shifts):
        cid = f"umbra-{i}"
        when = f"2024-07-{1 + 7 * i:02d}T10:00:00Z"
        folder = f"sar-data/task-data/t{i}/2024-07-{1 + 7 * i:02d}-10-00-00_UMBRA-05"
        name = folder.rsplit("/", 1)[-1]
        gec = write_gec(tmp_path / f"{cid}.tif", shift_px=shift)
        objects[f"{folder}/{name}_GEC.tif"] = Path(gec).read_bytes()
        objects[f"{folder}/{name}.stac.v2.json"] = json.dumps(
            _umbra_item(cid, when, inc=45.0, az=90.0)
        ).encode()
    return FakeBucket("umbra-open-data-catalog", objects, modified)


def test_umbra_builder_end_to_end_with_mocked_http(tmp_path, monkeypatch):
    _stub_dem(monkeypatch)
    # Date 2 is displaced 6 px east (9 m) and 2 px south (3 m) inside its image.
    bucket = _umbra_bucket(tmp_path, [(0, 0), (2, 6)])
    monkeypatch.setattr(vs.VHRSARCompositeBuilder, "bucket_factory", staticmethod(lambda p: bucket))
    config = _cfg(
        date_range=("2024-06-01", "2024-08-31"),
        study_area="bbox:-100.002,39.998,-99.998,40.002",
        cache_dir=str(tmp_path / "cache"),
        vhr_sar_cache_dir=str(tmp_path / "vhr"),
        vhr_sar_coregister="sar",
        output_path=str(tmp_path / "fields.gpkg"),
    )
    builder = vs.VHRSARCompositeBuilder()
    path = builder.build(config)
    tags = builder.last_metadata
    with rasterio.open(path) as src:
        assert src.count == 1 and src.dtypes[0] == "float32" and src.nodata == 0
        assert src.descriptions == ("MEDIAN",)
        data = src.read(1)
        file_tags = src.tags()
    assert (data > 0).mean() > 0.99 and data[data > 0].min() >= 1 and data.max() <= 255
    assert file_tags["AGRIBOUND_ATTRIBUTION"] == "SAR data: Umbra Lab, Inc., CC BY 4.0"
    assert tags["AGRIBOUND_SOURCE"] == "umbra" and tags["AGRIBOUND_VALUE_SCALE"] == "dn"
    assert tags["AGRIBOUND_SAR_RECIPE"] == vs.VHR_SAR_RECIPE_TAG
    assert tags["AGRIBOUND_LICENSE"] == "CC BY 4.0"
    assert "registry.opendata.aws/umbra-open-data" in tags["AGRIBOUND_DATA_CITATION"]
    assert tags["AGRIBOUND_INDEX_SNAPSHOT"].endswith("Z")
    assert float(tags["AGRIBOUND_VALID_FRACTION"]) > 0.99
    collects = json.loads(tags["AGRIBOUND_SAR_COLLECTS"])
    assert [c["id"] for c in collects] == ["umbra-0", "umbra-1"]
    assert collects[0]["reference"] and collects[0]["orthorectification"] == "rpc_dem"
    assert collects[0]["etag"] and collects[0]["key"].endswith("_GEC.tif")
    # The co-registration found date 2's displacement (east +, north -).
    dx, dy = collects[1]["shift_m"]
    assert dx == pytest.approx(6 * GEC_PX_M, abs=0.6) and dy == pytest.approx(
        -2 * GEC_PX_M, abs=0.6
    )
    manifest = json.loads(Path(path).with_suffix(".json").read_text())
    assert manifest["selection"]["reference"] == "umbra-0"
    assert manifest["orthorectification"]["dem"]["geoid_undulation_m"] == [-30.0, -30.0]
    assert manifest["orthorectification"]["dem"]["dem_tiles"] == {"N40_W100": "GLO-30"}
    # A second build reuses the cached composite without contacting the bucket again.
    n = len(bucket.requests)
    again = vs.VHRSARCompositeBuilder()
    assert again.build(config) == path and len(bucket.requests) == n
    assert again.last_metadata["AGRIBOUND_SAR_COLLECTS"] == tags["AGRIBOUND_SAR_COLLECTS"]


def test_builder_enforces_the_download_budget(tmp_path, monkeypatch):
    _stub_dem(monkeypatch)
    bucket = _umbra_bucket(tmp_path, [(0, 0), (0, 0)])
    monkeypatch.setattr(vs.VHRSARCompositeBuilder, "bucket_factory", staticmethod(lambda p: bucket))
    config = _cfg(
        study_area="bbox:-100.002,39.998,-99.998,40.002",
        cache_dir=str(tmp_path / "cache"),
        vhr_sar_cache_dir=str(tmp_path / "vhr"),
        vhr_sar_coregister="sar",
        vhr_sar_max_download_gb=1e-6,
    )
    with pytest.raises(ValueError, match="vhr_sar_max_download_gb"):
        vs.VHRSARCompositeBuilder().build(config)
    assert not any(m[0] == "STREAM" for m in bucket.requests)
    # Cached products do not count: with one of the two cached, a cap between the size
    # of the other and the total suffices.
    index = vs.load_index("umbra", tmp_path / "vhr", window_end="2025-01-01", bucket=bucket)
    first, second = sorted(index.records, key=lambda r: r.datetime)
    vs.ensure_product(bucket, first, tmp_path / "vhr")
    cap = (second.product_bytes + first.product_bytes / 2) / 1e9
    vs.VHRSARCompositeBuilder().build(config.merged(vhr_sar_max_download_gb=cap))
    streams = [m[1] for m in bucket.requests if m[0] == "STREAM"]
    assert len(streams) == 2 and streams[0] != streams[1]  # each product once


def _capella_objects(tmp_path, dates, scale=0.0004, calibrations=None):
    import pyproj

    x, y = pyproj.Transformer.from_crs(4326, 32614, always_xy=True).transform(LON0, LAT0)
    objects = {}
    for i, day in enumerate(dates):
        pid = f"CAPELLA_C08_SP_GEO_VV_202301{day:02d}110004_202301{day:02d}110028"
        tif = tmp_path / f"{pid}.tif"
        dn = np.full((1600, 1600), 1000, np.uint16)
        dn[:, :800] = 2500  # two "fields"
        dn[:2, :] = 0
        with rasterio.open(
            tif, "w", driver="GTiff", width=1600, height=1600, count=1, dtype="uint16",
            crs="EPSG:32614", transform=from_origin(x - 400, y + 400, 0.5, 0.5), nodata=0,
        ) as dst:  # fmt: skip
            dst.write(dn, 1)
        data_dir = f"data/2023/1/{day}/{pid}"
        objects[f"{data_dir}/{pid}.tif"] = tif.read_bytes()
        objects[f"{data_dir}/{pid}_preview.tif"] = b"preview"
        objects[f"{data_dir}/{pid}_extended.json"] = json.dumps(
            {
                "software_version": "2.58.0",
                "collect": {
                    "collect_id": f"cap-{i}",
                    "image": {
                        "scale_factor": scale,
                        "calibration": (calibrations or {}).get(i, "full"),
                        "center_pixel": {"incidence_angle": 36.5 + 0.01 * i},
                    },
                },
            }
        ).encode()
        item = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [list(map(list, _square(LON0, LAT0, 390)))],
            },
            "properties": {
                "capella:collect_id": f"cap-{i}",
                "datetime": f"2023-01-{day:02d}T11:00:16Z",
                "platform": "capella-8",
                "sar:polarizations": ["VV"],
                "sat:orbit_state": "descending",
                "sar:observation_direction": "left",
                "view:incidence_angle": 36.5,
                "view:azimuth": 283.3,
                "sar:instrument_mode": "spotlight",
                "processing:software": {"Capella SAR Processor": "2.58.0"},
            },
        }
        stac_dir = (
            "stac/capella-open-data-by-datetime/capella-open-data-2023/capella-open-data-2023-01/"
            f"capella-open-data-2023-01-{day:02d}/{pid}"
        )
        objects[f"{stac_dir}/{pid}.json"] = json.dumps(item).encode()
    return objects


def test_capella_builder_with_mocked_http_writes_sigma0_db(tmp_path, monkeypatch):
    bucket = FakeBucket(
        "capella-open-data", _capella_objects(tmp_path, [6, 12], calibrations={1: "limited"})
    )
    monkeypatch.setattr(vs.VHRSARCompositeBuilder, "bucket_factory", staticmethod(lambda p: bucket))
    config = _cfg(
        "capella",
        year=2023,
        date_range=("2023-01-01", "2023-01-31"),
        study_area="bbox:-100.0015,39.9985,-99.9985,40.0015",
        cache_dir=str(tmp_path / "cache"),
        vhr_sar_cache_dir=str(tmp_path / "vhr"),
        vhr_sar_coregister="none",
        vhr_sar_bands="median+dates",
    )
    builder = vs.VHRSARCompositeBuilder()
    path = builder.build(config)
    tags = builder.last_metadata
    with rasterio.open(path) as src:
        assert src.descriptions[:2] == ("MEDIAN", "PAIR_MEAN") and src.count == 4
        median = src.read(1)
        assert src.tags()["AGRIBOUND_ATTRIBUTION"] == "© Capella Space Corp, All Rights Reserved."
    low = 20 * math.log10(0.4) + vs.CAPELLA_DB_OFFSET
    high = 20 * math.log10(1.0) + vs.CAPELLA_DB_OFFSET
    values = np.unique(np.round(median[median > 0], 2))
    assert set(values) <= {round(low, 2), round(high, 2)} or np.allclose(
        np.percentile(median[median > 0], [5, 95]), [low, high], atol=0.05
    )
    assert tags["AGRIBOUND_VALUE_SCALE"] == "db" and float(tags["AGRIBOUND_DB_OFFSET"]) == 50.0
    assert tags["AGRIBOUND_SAR_CALIBRATION"] == "full,limited"
    assert (
        tags["AGRIBOUND_SAR_PRODUCT"] == "GEO"
        and "Modified by agribound" in tags["AGRIBOUND_CHANGES"]
    )
    collects = json.loads(tags["AGRIBOUND_SAR_COLLECTS"])
    assert collects[1]["incidence_deg"] == pytest.approx(36.51)  # full precision from _extended
    assert not any("preview" in m[1] for m in bucket.requests)
    # vhr_sar_calibration="full" leaves one consistent date, which is not enough.
    with pytest.raises(NoDataError, match="Only one consistent"):
        vs.VHRSARCompositeBuilder().build(config.merged(vhr_sar_calibration="full"))


# ---------------------------------------------------------------------------
# Real buckets (network)
# ---------------------------------------------------------------------------

UMBRA_GEC = (
    "sar-data/task-data/baf284d7-84b7-49f6-8735-1a2a992dd2ae/2024-08-11-05-07-04_UMBRA-05/"
    "2024-08-11-05-07-04_UMBRA-05_GEC.tif"
)
CAPELLA_GEO = (
    "data/2023/1/6/CAPELLA_C08_SP_GEO_VV_20230106110004_20230106110028/"
    "CAPELLA_C08_SP_GEO_VV_20230106110004_20230106110028.tif"
)


@pytest.mark.network
def test_umbra_gec_header_over_vsicurl():
    bucket = vs.PublicBucket("umbra-open-data-catalog")
    url = f"/vsicurl/{bucket.url(UMBRA_GEC)}"
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(url) as src:
        assert src.dtypes[0] == "uint8" and src.crs.to_epsg() == 4326
        assert src.rpcs is not None and src.nodata == 0
    head = bucket.head(UMBRA_GEC)
    assert head.size > 1e8 and head.etag
    _, prefixes = bucket.list("sar-data/", delimiter="/")
    assert "sar-data/task-data/" in prefixes


@pytest.mark.network
def test_capella_geo_header_over_vsicurl():
    bucket = vs.PublicBucket("capella-open-data")
    url = f"/vsicurl/{bucket.url(CAPELLA_GEO)}"
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(url) as src:
        assert src.dtypes[0] == "uint16" and src.crs.to_epsg() == 32629
    text = bucket.get(
        CAPELLA_GEO.replace(".tif", "_extended.json"),
        byte_range=(0, vs.CAPELLA_EXTENDED_HEAD_BYTES - 1),
    ).decode("utf-8", "replace")
    head = vs.parse_capella_extended(text)
    assert head["calibration"] == "full" and head["scale_factor"] > 0
    assert head["incidence_deg"] == pytest.approx(36.57, abs=0.01)


def test_builder_registers_the_stack_to_a_user_base(tmp_path, monkeypatch):
    from agribound.composites.gee import compute_export_grid

    _stub_dem(monkeypatch)
    bucket = _umbra_bucket(tmp_path, [(0, 0), (2, 6)])
    monkeypatch.setattr(vs.VHRSARCompositeBuilder, "bucket_factory", staticmethod(lambda p: bucket))
    study_area = "bbox:-100.002,39.998,-99.998,40.002"
    margin = vs._expand_grid(
        compute_export_grid(box(-100.002, 39.998, -99.998, 40.002), "EPSG:32614", 1.0),
        vs.REGISTRATION_MARGIN_M,
    )
    # The base: date 1 orthorectified, then written 4 m east and 2 m south of its place.
    truth = vs.warp_product(tmp_path / "umbra-0.tif", margin, source="umbra", method="rpc_dem",
                            dem_path=_dem(tmp_path / "d.tif", HEIGHT_OFF))  # fmt: skip
    t = margin.transform
    base = tmp_path / "ortho.tif"
    with rasterio.open(
        base, "w", driver="GTiff", width=margin.width, height=margin.height, count=3,
        dtype="float32", crs=margin.crs, nodata=0,
        transform=Affine(t.a, 0, t.c + 4.0, 0, t.e, t.f - 2.0),
    ) as dst:  # fmt: skip
        dst.write(np.stack([truth, truth * 0.9, truth * 0.8]))
    config = _cfg(
        date_range=("2024-06-01", "2024-08-31"),
        study_area=study_area,
        cache_dir=str(tmp_path / "cache"),
        vhr_sar_cache_dir=str(tmp_path / "vhr"),
        vhr_sar_coregister="optical",
        vhr_sar_coregister_base=str(base),
    )
    builder = vs.VHRSARCompositeBuilder()
    builder.build(config)
    tags = builder.last_metadata
    coreg = json.loads(tags["AGRIBOUND_SAR_COREGISTRATION"])
    assert coreg["base"] == "user" and coreg["mode"] == "optical"
    assert coreg["base_shift_m"] == pytest.approx([-4.0, 2.0], abs=0.6)
    shifts = [c["shift_m"] for c in json.loads(tags["AGRIBOUND_SAR_COLLECTS"])]
    assert shifts[0] == pytest.approx([-4.0, 2.0], abs=0.6)
    assert shifts[1] == pytest.approx([6 * GEC_PX_M - 4.0, -2 * GEC_PX_M + 2.0], abs=0.8)


def test_optical_coregistration_needs_a_base(tmp_path, monkeypatch):
    _stub_dem(monkeypatch)
    bucket = _umbra_bucket(tmp_path, [(0, 0), (0, 0)])
    monkeypatch.setattr(vs.VHRSARCompositeBuilder, "bucket_factory", staticmethod(lambda p: bucket))
    monkeypatch.setattr(vs, "_naip_plus_base", lambda config, grid: (None, {"reason": "none here"}))
    config = _cfg(
        study_area="bbox:-100.002,39.998,-99.998,40.002",
        cache_dir=str(tmp_path / "cache"),
        vhr_sar_cache_dir=str(tmp_path / "vhr"),
        vhr_sar_coregister="optical",
    )
    with pytest.raises(RuntimeError, match="no base is available"):
        vs.VHRSARCompositeBuilder().build(config)
    with pytest.raises(ValueError, match="1 m or finer"):
        coarse = tmp_path / "coarse.tif"
        with rasterio.open(
            coarse, "w", driver="GTiff", width=10, height=10, count=1, dtype="uint8",
            crs="EPSG:32614", transform=from_origin(0, 0, 10, 10),
        ) as dst:  # fmt: skip
            dst.write(np.ones((1, 10, 10), np.uint8))
        vs._read_base_on_grid(coarse, _grid())


def test_builder_refuses_a_geographic_export_crs(tmp_path):
    config = _cfg(export_crs="EPSG:4326", cache_dir=str(tmp_path / "cache"))
    with pytest.raises(ValueError, match="projected CRS in metres"):
        vs.VHRSARCompositeBuilder().build(config)


# ---------------------------------------------------------------------------
# Review fixes (2026-10-09): consensus, DEM gaps, base units, downloads, notes
# ---------------------------------------------------------------------------


def test_coregistration_of_unrelated_images_fails_at_every_level():
    """Unrelated textures have no consensus shift: every level fails.

    Before the inlier radius was capped, their scattered tile estimates (spread
    25-45 m) all counted as inliers and the registration was accepted.
    """
    a = _fields_texture(900, seed=5).astype(np.float32)
    b = _fields_texture(900, seed=11).astype(np.float32)
    optical = np.stack([a, a * 0.8 + 5, a * 0.5 + 10])
    sar = vs.register_adaptive(a, b, 1.0, kind="sar")
    opt = vs.register_adaptive(optical, b, 1.0, kind="optical")
    for result in (sar, opt):
        assert not result["ok"] and result["scale"] == 4
        assert result.get("inlier_fraction", 0.0) < vs.MIN_INLIER_FRACTION
    # At full resolution every optical tile votes, but the votes scatter: the inlier
    # radius stops at the cap and the inliers are a small minority.
    level1 = vs.estimate_shift(
        vs.optical_edge_map(optical), vs.edge_map(b), 1.0,
        tile_px=256, max_shift_m=150.0, tol_m=6.0, min_peak=0.2,
    )  # fmt: skip
    assert level1["n_peak_ok"] == level1["n_tiles"] == 36
    assert level1["inlier_radius_m"] == vs.INLIER_RADIUS_CAP * 6.0
    assert level1["inlier_fraction"] < 0.25
    # Related images keep their result, with nearly every voting tile an inlier.
    moved = np.zeros_like(a)
    moved[:-4, 7:] = a[4:, :-7]
    good = vs.register_adaptive(a, moved, 1.0, kind="sar")
    assert good["ok"] and good["scale"] == 1 and good["inlier_fraction"] > 0.9


def test_builder_refuses_a_base_that_does_not_match(tmp_path, monkeypatch):
    _stub_dem(monkeypatch)
    bucket = _umbra_bucket(tmp_path, [(0, 0), (2, 6)])
    monkeypatch.setattr(vs.VHRSARCompositeBuilder, "bucket_factory", staticmethod(lambda p: bucket))
    margin = vs._expand_grid(
        compute_export_grid(box(-100.002, 39.998, -99.998, 40.002), "EPSG:32614", 1.0),
        vs.REGISTRATION_MARGIN_M,
    )
    side = max(margin.width, margin.height)
    other = _fields_texture(side, seed=21).astype(np.float32)[: margin.height, : margin.width]
    base = tmp_path / "elsewhere.tif"
    with rasterio.open(
        base, "w", driver="GTiff", width=margin.width, height=margin.height, count=3,
        dtype="float32", crs=margin.crs, nodata=0, transform=margin.transform,
    ) as dst:  # fmt: skip
        dst.write(np.stack([other, other * 0.9, other * 0.8]))
    config = _cfg(
        date_range=("2024-06-01", "2024-08-31"),
        study_area="bbox:-100.002,39.998,-99.998,40.002",
        cache_dir=str(tmp_path / "cache"),
        vhr_sar_cache_dir=str(tmp_path / "vhr"),
        vhr_sar_coregister="optical",
        vhr_sar_coregister_base=str(base),
    )
    with pytest.raises(RuntimeError, match="tile estimates do not agree"):
        vs.VHRSARCompositeBuilder().build(config)


def test_dem_falls_back_to_glo90_and_warns_where_there_is_no_tile(tmp_path, monkeypatch, caplog):
    """GLO-30 has no public tiles over Armenia and Azerbaijan: GLO-90 is read there."""
    bounds = (44.4, 40.1, 45.2, 40.3)  # tiles N40_E044 (GLO-90 only) and N40_E045 (none)

    def read(url, b, shape):
        if "copernicus-dem-90m" in url and "N40_00_E044_00" in url:
            heights = np.full(shape, np.nan, np.float32)
            heights[:, : round(shape[1] * 0.6 / 0.8)] = 900.0  # 44.4-45.0 of 44.4-45.2
            return heights
        return None  # every GLO-30 tile, and GLO-90 N40_E045

    monkeypatch.setattr(vs, "_read_url_window", read)
    with caplog.at_level(logging.WARNING, logger=vs.logger.name):
        heights, tiles = vs.read_copernicus_dem(bounds, (20, 80))
    assert tiles == {"N40_E044": "GLO-90", "N40_E045": "none"}
    assert np.all(heights[:, :60] == 900.0) and np.all(heights[:, 60:] == 0.0)
    assert "no public tile N40_E044; using GLO-90" in caplog.text
    assert "No Copernicus DEM tile N40_E045" in caplog.text
    info = vs.build_dem_hae(
        bounds, tmp_path / "dem.tif", read_geoid=lambda b, shape: np.zeros(shape, np.float32)
    )
    assert info["dem_tiles"] == tiles and info["orthometric_m"][2] == 900.0
    with rasterio.open(info["path"]) as src:
        assert "N40_E044 GLO-90" in src.tags()["AGRIBOUND_DEM"]


def test_user_base_pixel_size_is_read_in_the_crs_unit(tmp_path):
    """A State Plane orthophoto in US survey feet: 2 ft pixels are 0.61 m, not 2 m."""
    import pyproj

    x, y = pyproj.Transformer.from_crs(4326, 2226, always_xy=True).transform(-121.78, 38.53)
    grid = compute_export_grid(box(-121.782, 38.528, -121.778, 38.532), "EPSG:32610", 1.0)

    def write(name, feet):
        path = tmp_path / name
        with rasterio.open(
            path, "w", driver="GTiff", width=1200, height=1200, count=3, dtype="uint8",
            crs="EPSG:2226", transform=from_origin(x - 600 * feet, y + 600 * feet, feet, feet),
        ) as dst:  # fmt: skip
            dst.write(np.full((3, 1200, 1200), 7, np.uint8))
        return path

    bands, info = vs._read_base_on_grid(write("two_feet.tif", 2.0), grid)
    assert info["resolution_m"] == pytest.approx(0.61, abs=0.005)
    assert bands.shape == (3, grid.height, grid.width) and np.all(bands == 7)
    with pytest.raises(ValueError, match="1.22 m pixels"):
        vs._read_base_on_grid(write("four_feet.tif", 4.0), grid)


def test_a_partial_download_of_another_version_is_never_resumed(tmp_path):
    """Bytes of an older version of the object (another ETag) are discarded."""
    old, new = bytes(range(256)) * 8, bytes(reversed(range(256))) * 12
    key = "sar-data/tasks/x/2024-07-01-10-00-00_UMBRA-05/2024-07-01-10-00-00_UMBRA-05_GEC.tif"
    bucket = FakeBucket("umbra-open-data-catalog", {key: new})
    record = rec("c1", product_key=key, product_bytes=len(new), product_etag=bucket.etags[key])
    dest = vs.product_path(tmp_path, record)
    dest.parent.mkdir(parents=True)
    old_etag = hashlib.md5(old).hexdigest()
    dest.with_name(f"{dest.name}.{old_etag}.part").write_bytes(old[:1000])  # interrupted
    dest.with_name(dest.name + ".part").write_bytes(old[:500])  # first-version partial name
    path, info = vs.ensure_product(bucket, record, tmp_path)
    assert path.read_bytes() == new and info["etag"] == bucket.etags[key]
    (stream,) = [m for m in bucket.requests if m[0] == "STREAM"]
    assert "Range" not in stream[2]  # nothing of the old version was kept
    assert not list(dest.parent.glob("*.part"))


def test_copies_of_one_collect_have_their_own_files(tmp_path):
    """Umbra copies of one collect share the GEC file name; their local files differ."""
    a = rec("c1", product_key="sar-data/tasks/A/x/x_GEC.tif", product_etag="e1")
    b = rec("c1", product_key="sar-data/tasks/B/x/x_GEC.tif", product_etag="e2")
    assert vs.product_path(tmp_path, a) != vs.product_path(tmp_path, b)
    assert vs.product_path(tmp_path, a).name == "x_GEC.tif"
    # A product cached by the first version (<collect_id>/<name>) is reused when its
    # metadata names the same key, ETag and size, and only then.
    legacy = tmp_path / "umbra" / "products" / "c1" / "x_GEC.tif"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"\0" * a.product_bytes)
    meta = {"key": a.product_key, "etag": "e1", "size": a.product_bytes}
    legacy.with_name(legacy.name + ".json").write_text(json.dumps(meta))
    assert vs.cached_product(tmp_path, a) == legacy
    assert vs.cached_product(tmp_path, b) is None


def test_concurrent_downloads_of_one_product_make_one_transfer(tmp_path):
    """Two workers that need the same collect: one downloads, the other waits and reuses it."""
    payload = bytes(range(256)) * 200
    key = "sar-data/tasks/x/a/a_GEC.tif"

    class SlowBucket(FakeBucket):
        def _stream(self, path, headers):
            status, head, chunks, close = super()._stream(path, headers)
            data = b"".join(chunks)

            def slow():
                for i in range(0, len(data), 2048):
                    time.sleep(0.002)
                    yield data[i : i + 2048]

            return status, head, slow(), close

    bucket = SlowBucket("umbra-open-data-catalog", {key: payload})
    record = rec("a", product_key=key, product_bytes=len(payload), product_etag=bucket.etags[key])
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda _: vs.ensure_product(bucket, record, tmp_path), range(2)))
    assert [p for p, _ in results] == [results[0][0]] * 2
    assert results[0][0].read_bytes() == payload
    assert sum(m[0] == "STREAM" for m in bucket.requests) == 1


def test_single_date_messages_compare_like_with_like():
    """0.00-0.08 was a single date with the engine's default settings, not the recipe's."""
    note = vs.SINGLE_DATE_NOTE
    assert "0.00-0.08 with Delineate-Anything's default settings" in note
    assert "read at 2 m without tile merging" in note and "0.36-0.40" in note
    sel = vs.select_collects(
        [rec("a"), rec("b", "2024-07-09T10:00:00Z")], AOI, WINDOW, source="umbra", max_collects=1
    )
    assert any(note in w for w in sel.warnings)


def test_agent_proposes_the_builder_s_delineate_anything_settings():
    from agribound.agent.tools import VHR_SAR_DELINEATE_ANYTHING_PARAMS

    assert VHR_SAR_DELINEATE_ANYTHING_PARAMS == vs.DELINEATE_ANYTHING_PARAMS


def test_cached_index_is_read_only(tmp_path, monkeypatch):
    def no_refresh(*args, **kwargs):
        raise AssertionError("cached_index listed the archive")

    monkeypatch.setattr(vs, "refresh_index", no_refresh)
    assert vs.cached_index("umbra", tmp_path) is None  # no cache file
    assert not any(tmp_path.iterdir())  # and nothing written
    path = vs.index_path("umbra", tmp_path)
    old = {"format": "0", "source": "umbra", "snapshot": "2026-01-01T00:00:00Z", "copies": {}}
    vs._write_json_gz(path, old)
    before = path.read_bytes()
    assert vs.cached_index("umbra", tmp_path) is None  # another format: load_index rebuilds it
    path.write_bytes(b"not gzip")
    assert vs.cached_index("umbra", tmp_path) is None  # unreadable
    assert path.read_bytes() == b"not gzip" and before != b"not gzip"
    vs._write_json_gz(path, {**old, "format": vs.INDEX_FORMAT})
    index = vs.cached_index("umbra", tmp_path)
    assert index.snapshot == "2026-01-01T00:00:00Z" and index.records == []
    assert index.path == path


def test_damaged_cached_index_is_unreadable_not_a_crash(tmp_path, monkeypatch):
    import gzip

    monkeypatch.setattr(vs, "refresh_index", lambda *a, **k: pytest.fail("listed the archive"))
    path = vs.index_path("umbra", tmp_path)
    good = {"format": vs.INDEX_FORMAT, "source": "umbra", "snapshot": "2026-01-01T00:00:00Z"}
    vs._write_json_gz(path, {**good, "copies": {}})
    whole = path.read_bytes()
    damaged = {
        "truncated gzip": whole[: len(whole) // 2],
        "damaged deflate data": whole[:12] + bytes(len(whole) - 20) + whole[-8:],
        "JSON that is not an object": gzip.compress(b"[1, 2, 3]"),
        "missing keys": gzip.compress(json.dumps({"format": vs.INDEX_FORMAT}).encode()),
        "copies of the wrong type": gzip.compress(json.dumps({**good, "copies": 7}).encode()),
    }
    for label, data in damaged.items():
        path.write_bytes(data)
        assert vs.cached_index("umbra", tmp_path) is None, label
        assert path.read_bytes() == data, label  # never rewritten
