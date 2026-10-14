"""TESSERA store stamp (agribound.composites.tessera_stamp): object keys, HEAD, fingerprint.

The store's metadata (four UTM zones of the v1 store, fetched on 2026-10-09) is a
fixture (``tests/data/tessera_v1_metadata.json``); HTTP is mocked. The expected
shard keys are those geotessera 0.10.2's ``read_region`` requested for the same
boxes (recorded by wrapping its Zarr store); the ETags and dates are the HEAD
answers of 2026-10-09 (the 1.1 review's ``store_meta/heads.txt`` and this test's
own HEADs of the ``scales`` shards and of the ``x``/``y`` coordinate chunks). One
test, marked ``network``, sends real HEAD requests.

Decoding a zone's ``time`` array needs zarr >= 3 (a dependency of geotessera, the
``tessera`` extra); the tests that compute keys or stamps from the fixture are
skipped without it (CI's core job installs only ``.[dev]``).
"""

from __future__ import annotations

import base64
import errno
import hashlib
import http.server
import io
import json
import logging
import socket
import ssl
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import pyproj
import pytest
from shapely.geometry import Polygon, box
from shapely.ops import transform

from agribound.composites import tessera_stamp as ts
from agribound.composites.gee import resolve_export_crs
from agribound.composites.local import read_bbox_for_grid

DATA = Path(__file__).resolve().parents[1] / "data" / "tessera_v1_metadata.json"
STORE = "https://data.source.coop/tessera/tessera/zarr/v1"


def _has_zarr3() -> bool:
    try:
        from zarr.core.buffer import default_buffer_prototype  # noqa: F401
        from zarr.storage import MemoryStore  # noqa: F401
    except ImportError:
        return False
    return True


#: The test decodes a zone's ``time`` array (zarr >= 3, geotessera's dependency).
requires_zarr = pytest.mark.skipif(
    not _has_zarr3(), reason="zarr >= 3 (the tessera extra) decodes the zones' time arrays"
)

#: HEAD answers of 2026-10-09 (ETag, Last-Modified).
HEADS = {
    "utm20/embeddings/c/7/0/315/13": (
        '"08f30a2efe2dca18f508435770eea74e-33"',
        "Wed, 07 Oct 2026 12:39:10 GMT",
    ),
    "utm20/embeddings/c/7/0/316/13": (
        '"ccea24039a34be632380f384113b3f49-33"',
        "Wed, 07 Oct 2026 12:40:56 GMT",
    ),
    "utm20/embeddings/c/7/0/316/14": (
        '"b50eb30a87210912a7392b908c7a9d46-32"',
        "Wed, 07 Oct 2026 12:40:58 GMT",
    ),
    "utm20/embeddings/c/7/0/317/13": (
        '"6fd4b5e225f27f94a7930fde5e65dcfa-33"',
        "Wed, 07 Oct 2026 12:42:49 GMT",
    ),
    "utm20/embeddings/c/7/0/317/14": (
        '"7a82947cec289f9a51de583ca31280e8-31"',
        "Wed, 07 Oct 2026 12:42:42 GMT",
    ),
    "utm20/scales/c/7/315/13": (
        '"c690205fcfc50edc769c315397831fa1"',
        "Wed, 07 Oct 2026 12:38:48 GMT",
    ),
    "utm20/scales/c/7/316/13": (
        '"f17a6cf10a7ca2b1d303fb80c026b7dc"',
        "Wed, 07 Oct 2026 12:40:33 GMT",
    ),
    "utm20/scales/c/7/316/14": (
        '"65680da3d958deffbeccdf345c1ce994"',
        "Wed, 07 Oct 2026 12:40:36 GMT",
    ),
    "utm20/scales/c/7/317/13": (
        '"623b8313a3d20785ff29fc1f5a3c848a"',
        "Wed, 07 Oct 2026 12:42:29 GMT",
    ),
    "utm20/scales/c/7/317/14": (
        '"5aca9724ac1da4e264652b42ba9f0953"',
        "Wed, 07 Oct 2026 12:42:20 GMT",
    ),
    "utm45/embeddings/c/7/0/156/11": (
        '"64719e30eeb824f58d764676816531f4-37"',
        "Wed, 07 Oct 2026 17:28:07 GMT",
    ),
    "utm45/embeddings/c/7/0/157/11": (
        '"15589a3064f2763e09145fb091b25e8d-36"',
        "Wed, 07 Oct 2026 17:30:01 GMT",
    ),
    "utm45/scales/c/7/156/11": (
        '"a4abf342c2003eb414d31c850b4e57ac"',
        "Wed, 07 Oct 2026 17:27:55 GMT",
    ),
    "utm45/scales/c/7/157/11": (
        '"b8e2943cde7b466199b77716e8125fb2"',
        "Wed, 07 Oct 2026 17:29:49 GMT",
    ),
    # Coordinate chunks under the windows of examples 05, 15 and 02 (HEAD, 2026-10-09).
    "utm20/x/c/1": ('"9375e33511abd4e972d66743039c15f9"', "Fri, 31 Jul 2026 18:11:51 GMT"),
    "utm20/y/c/13": ('"d39e6f8cf0a0f5dc3dfef33d6e364330"', "Fri, 31 Jul 2026 18:11:54 GMT"),
    "utm45/x/c/1": ('"2a4fc6e1f952bd6d6bbc22bf0a4e9a00"', "Fri, 31 Jul 2026 18:17:42 GMT"),
    "utm45/y/c/9": ('"3ea54beb7b26862aaae8f8d4128b1c6d"', "Fri, 31 Jul 2026 18:17:44 GMT"),
}

#: Study areas of examples 05, 15 and 02 (2024).
EXAMPLE_05 = box(-60.65, -33.85, -60.45, -33.65)
EXAMPLE_15 = Polygon(
    [
        (-60.552058, -33.789222),
        (-60.274083, -33.734778),
        (-60.242050, -33.903368),
        (-60.363950, -33.995060),
        (-60.505737, -34.015376),
    ]
)
EXAMPLE_02 = box(88.35, 23.35, 88.50, 23.50)
#: Example 23's TESSERA sites: centre, side of the UTM square, year.
EXAMPLE_23 = {
    "twifo": (-1.51801, 5.53834, 5000, 2020),
    "madera": (-120.46685, 36.989886, 5000, 2022),
    "jaen": (-3.347141, 37.947077, 4000, 2023),
}


def _objects() -> dict[str, bytes]:
    doc = json.loads(DATA.read_text())
    out = {}
    for key, value in doc["objects"].items():
        if isinstance(value, dict) and set(value) == {"base64"}:
            out[key] = base64.b64decode(value["base64"])
        else:
            out[key] = json.dumps(value).encode("utf-8")
    return out


class FixtureClient(ts.StoreClient):
    """Serves the fixture's metadata to GET and *heads* to HEAD; records the requests."""

    def __init__(self, heads=None, objects=None, **kwargs):
        super().__init__(STORE, **kwargs)
        self.objects = _objects() if objects is None else objects
        self.heads = dict(HEADS if heads is None else heads)
        self.requests: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def _request(self, method, key):
        with self._lock:
            self.requests.append((method, key))
        if method == "GET":
            body = self.objects.get(key)
            return (404, {}, b"") if body is None else (200, {}, body)
        if key not in self.heads:
            return 404, {}, b""
        etag, modified = self.heads[key]
        return 200, {"etag": etag, "last-modified": modified}, b""


def _read_bbox(geometry):
    """The box build_tessera_embedding reads (export CRS "utm", the default)."""
    return read_bbox_for_grid(geometry, resolve_export_crs("utm", geometry))


def _square(lon, lat, side_m):
    """Example 23's study area: a square of *side_m* metres in the site's UTM zone."""
    from agribound.io import get_utm_crs

    utm = get_utm_crs(lon, lat)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_ll = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    x, y = to_utm.transform(lon, lat)
    half = side_m / 2
    return transform(to_ll.transform, box(x - half, y - half, x + half, y + half))


# ---------------------------------------------------------------------------
# Chunk keys from the arrays' metadata
# ---------------------------------------------------------------------------


class TestChunkKeys:
    def test_default_and_v2_encodings(self):
        assert ts.chunk_key({"name": "default"}, (7, 0, 316, 13)) == "c/7/0/316/13"
        dotted = {"name": "default", "configuration": {"separator": "."}}
        assert ts.chunk_key(dotted, (7, 316, 13)) == "c.7.316.13"
        assert ts.chunk_key({"name": "v2"}, (7, 316, 13)) == "7.316.13"
        assert ts.chunk_key({"name": "v2", "configuration": {"separator": "/"}}, (1, 2)) == "1/2"
        assert ts.chunk_key({"name": "v2"}, ()) == "0"
        assert ts.chunk_key({"name": "default"}, ()) == "c"
        with pytest.raises(ts.StoreCheckError, match="unknown chunk key encoding"):
            ts.chunk_key({"name": "other"}, (1,))

    def test_shards_of_a_window(self):
        meta = json.loads(_objects()["utm20/embeddings/zarr.json"])
        # One row into the next shard (4096 px) gives two shard rows; all 128 bands are read.
        keys = ts.chunk_keys(meta, {"time": (7, 7), "y": (4000, 4096), "x": (10, 20)})
        assert keys == ["c/7/0/0/0", "c/7/0/1/0"]
        scales = json.loads(_objects()["utm20/scales/zarr.json"])
        assert ts.chunk_keys(scales, {"time": (0, 0), "y": (0, 0), "x": (8191, 8192)}) == [
            "c/0/0/1",
            "c/0/0/2",
        ]
        time_meta = json.loads(_objects()["utm20/time/zarr.json"])
        assert ts.chunk_keys(time_meta, {}) == ["c/0"]

    def test_unusable_metadata(self):
        meta = json.loads(_objects()["utm20/scales/zarr.json"])
        with pytest.raises(ts.StoreCheckError, match="outside dimension"):
            ts.chunk_keys(meta, {"time": (9, 9)})
        with pytest.raises(ts.StoreCheckError, match="no dimension"):
            ts.chunk_keys(meta, {"band": (0, 0)})
        irregular = {**meta, "chunk_grid": {"name": "rectilinear", "configuration": {}}}
        with pytest.raises(ts.StoreCheckError, match="unsupported chunk grid"):
            ts.chunk_keys(irregular, {})
        with pytest.raises(ts.StoreCheckError, match="chunk grid"):
            ts.chunk_keys({"shape": [3]}, {})

    @requires_zarr
    def test_time_values_are_decoded_with_the_arrays_codecs(self):
        client = FixtureClient()
        meta = ts.zone_metadata(client, 20)
        assert meta.times == tuple(range(2017, 2026))
        assert meta.attrs["proj:code"] == "EPSG:32620"
        assert ts.zone_metadata(client, 31) is None  # not in the fixture: no such zone


# ---------------------------------------------------------------------------
# Keys of the examples' reads
# ---------------------------------------------------------------------------


def _keys(geometry, year):
    keys, markers, zones = ts.read_object_keys(FixtureClient(), _read_bbox(geometry), year)
    assert not markers
    return keys, zones


class TestExampleReads:
    @requires_zarr
    def test_example_05(self):
        keys, zones = _keys(EXAMPLE_05, 2024)
        assert zones == [20]
        assert keys == [
            "utm20/embeddings/c/7/0/315/13",
            "utm20/embeddings/c/7/0/316/13",
            "utm20/scales/c/7/315/13",
            "utm20/scales/c/7/316/13",
            # x chunks of 34816 px, y chunks of 96000 px: the window (columns 55079-57111,
            # rows 1294240-1296616, padded by one pixel) lies in x chunk 1 and y chunk 13.
            "utm20/x/c/1",
            "utm20/y/c/13",
        ]

    @requires_zarr
    def test_example_15(self):
        # The review listed 316/13 and 316/14; the read box also reaches 32 rows into
        # shard row 317, and geotessera requests those shards too.
        keys, _ = _keys(EXAMPLE_15, 2024)
        assert keys == [
            "utm20/embeddings/c/7/0/316/13",
            "utm20/embeddings/c/7/0/316/14",
            "utm20/embeddings/c/7/0/317/13",
            "utm20/embeddings/c/7/0/317/14",
            "utm20/scales/c/7/316/13",
            "utm20/scales/c/7/316/14",
            "utm20/scales/c/7/317/13",
            "utm20/scales/c/7/317/14",
            "utm20/x/c/1",
            "utm20/y/c/13",
        ]

    @requires_zarr
    def test_example_02(self):
        keys, zones = _keys(EXAMPLE_02, 2024)
        assert zones == [45]
        assert keys == [
            "utm45/embeddings/c/7/0/156/11",
            "utm45/embeddings/c/7/0/157/11",
            "utm45/scales/c/7/156/11",
            "utm45/scales/c/7/157/11",
            "utm45/x/c/1",
            "utm45/y/c/9",
        ]

    @requires_zarr
    @pytest.mark.parametrize(
        ("site", "expected"),
        [
            (
                "twifo",
                [
                    "utm30/embeddings/c/3/0/179/12",
                    "utm30/scales/c/3/179/12",
                    "utm30/x/c/1",
                    "utm30/y/c/11",
                ],
            ),
            (
                "madera",
                [
                    "utm10/embeddings/c/5/0/109/13",
                    "utm10/scales/c/5/109/13",
                    "utm10/x/c/1",
                    "utm10/y/c/6",
                ],
            ),
            (
                "jaen",
                [
                    "utm30/embeddings/c/6/0/91/7",
                    "utm30/scales/c/6/91/7",
                    "utm30/x/c/0",
                    "utm30/y/c/5",
                ],
            ),
        ],
    )
    def test_example_23(self, site, expected):
        lon, lat, side, year = EXAMPLE_23[site]
        keys, _ = _keys(_square(lon, lat, side), year)
        assert keys == expected

    @requires_zarr
    @pytest.mark.parametrize(
        ("geometry", "zone", "shape", "origin"),
        [
            # (rows, cols) and the window's upper-left corner of geotessera's read_region
            (EXAMPLE_05, 20, (2377, 2033), (716810.0, -3725260.0)),
            (EXAMPLE_15, 20, (3226, 3012), (725820.0, -3735250.0)),
            (EXAMPLE_02, 45, (1731, 1604), (637580.0, 2599950.0)),
        ],
    )
    def test_unpadded_window_is_geotesseras(self, geometry, zone, shape, origin):
        meta = ts.zone_metadata(FixtureClient(), zone)
        ((part_zone, sub),) = ts.zone_parts(_read_bbox(geometry))
        assert part_zone == zone
        col0, col1, row0, row1 = ts.pixel_window(meta.attrs, sub, pad=0)
        assert (row1 - row0 + 1, col1 - col0 + 1) == shape
        a, _, x0, _, e, y0 = meta.attrs["spatial:transform"]
        assert (x0 + col0 * a, y0 + row0 * e) == origin
        padded = ts.pixel_window(meta.attrs, sub)
        assert padded == (col0 - 1, col1 + 1, row0 - 1, row1 + 1)

    @requires_zarr
    def test_missing_zone_and_year_are_markers(self):
        # A box across 0 E: zone 30 is in the fixture, zone 31 is not.
        client = FixtureClient()
        keys, markers, zones = ts.read_object_keys(client, (-0.02, 5.5, 0.02, 5.52), 2020)
        assert zones == [30]
        assert [m.key for m in markers] == ["utm31/zarr.json"]
        assert all(
            k.startswith(("utm30/embeddings/c/3/", "utm30/scales/c/3/", "utm30/x/", "utm30/y/"))
            for k in keys
        )
        keys, markers, zones = ts.read_object_keys(client, (-60.6, -33.8, -60.5, -33.7), 2016)
        assert (keys, zones) == ([], [])
        assert [m.key for m in markers] == ["utm20/time[2016]"]

    def test_window_outside_the_grid(self):
        attrs = {
            "proj:code": "EPSG:32620",
            "spatial:transform": [10.0, 0.0, 600000.0, 0.0, -10.0, 4000000.0],
            "spatial:shape": [100, 100],
        }
        assert ts.pixel_window(attrs, (-60.65, -33.85, -60.45, -33.65)) is None
        with pytest.raises(ts.StoreCheckError, match="spatial:transform"):
            ts.pixel_window({**attrs, "spatial:transform": [10, 1, 0, 0, -10, 0]}, (0, 0, 1, 1))
        with pytest.raises(ts.StoreCheckError, match="usable grid"):
            ts.pixel_window({"proj:code": "EPSG:32620"}, (0, 0, 1, 1))


# ---------------------------------------------------------------------------
# Stamp
# ---------------------------------------------------------------------------


def _stamp(geometry=EXAMPLE_05, year=2024, heads=None):
    client = FixtureClient(heads=heads)
    return ts.compute_stamp(STORE, _read_bbox(geometry), year, client=client), client


@requires_zarr
class TestStamp:
    def test_example_05(self):
        stamp, client = _stamp()
        assert stamp.n_objects == 6 and stamp.zones == (20,)
        assert stamp.last_modified == "2026-10-07T12:40:56Z"
        assert [o.etag for o in stamp.objects][1] == "ccea24039a34be632380f384113b3f49-33"
        grid = "utm20 EPSG:32620 10.0,0.0,166020.0,0.0,-10.0,9217140.0"
        assert stamp.grids == (grid,)
        lines = [f"agribound-tessera-store-stamp {ts.STAMP_VERSION}", STORE, grid] + [
            f"{key} {HEADS[key][0].strip(chr(34))}" for key in sorted(o.key for o in stamp.objects)
        ]
        assert stamp.fingerprint == hashlib.sha256("\n".join(lines).encode()).hexdigest()
        assert stamp.tags() == {
            "TESSERA_STORE_FINGERPRINT": stamp.fingerprint,
            "TESSERA_STORE_LAST_MODIFIED": "2026-10-07T12:40:56Z",
            "TESSERA_STORE_OBJECTS": "6",
        }
        heads = sorted(key for method, key in client.requests if method == "HEAD")
        assert heads == sorted(o.key for o in stamp.objects)  # nothing else is HEADed
        assert "6 objects" in stamp.describe()

    def test_fingerprint_is_stable(self):
        first, _ = _stamp()
        ts.clear_memo()
        reordered = dict(reversed(list(HEADS.items())))
        second, _ = _stamp(heads=reordered)
        assert second.fingerprint == first.fingerprint
        assert second.objects == first.objects

    def test_rewritten_or_missing_shard_changes_the_fingerprint(self):
        before, _ = _stamp()
        ts.clear_memo()
        heads = dict(HEADS)
        heads["utm20/embeddings/c/7/0/316/13"] = (
            '"0123456789abcdef0123456789abcdef-33"',
            "Fri, 09 Oct 2026 08:00:00 GMT",
        )
        after, _ = _stamp(heads=heads)
        assert after.fingerprint != before.fingerprint
        assert after.last_modified == "2026-10-09T08:00:00Z"
        ts.clear_memo()
        del heads["utm20/scales/c/7/315/13"]
        missing, _ = _stamp(heads=heads)
        assert missing.n_objects == 5 and len(missing.objects) == 6
        assert missing.fingerprint not in (before.fingerprint, after.fingerprint)

    def test_coordinates_or_grid_that_move_the_pixels_change_the_fingerprint(self):
        """geotessera places the window by the x/y values and the zone's grid attributes."""
        before, _ = _stamp()
        ts.clear_memo()
        heads = dict(HEADS)
        heads["utm20/x/c/1"] = (
            '"0123456789abcdef0123456789abcdef"',
            "Fri, 09 Oct 2026 08:00:00 GMT",
        )
        moved, _ = _stamp(heads=heads)
        assert moved.fingerprint != before.fingerprint
        assert [o.key for o in moved.objects] == [o.key for o in before.objects]
        # Half a pixel east in spatial:transform: the same objects, another grid.
        ts.clear_memo()
        objects = _objects()
        zone = json.loads(objects["utm20/zarr.json"])
        zone["attributes"]["spatial:transform"][2] += 5.0
        objects["utm20/zarr.json"] = json.dumps(zone).encode()
        shifted = ts.compute_stamp(STORE, before.bbox, 2024, client=FixtureClient(objects=objects))
        assert [o.key for o in shifted.objects] == [o.key for o in before.objects]
        assert shifted.grids != before.grids
        assert shifted.fingerprint != before.fingerprint

    def test_other_store_or_year_other_fingerprint(self):
        stamp, _ = _stamp()
        other = ts.compute_stamp(STORE + "/", stamp.bbox, 2024, client=FixtureClient())
        assert other.fingerprint == stamp.fingerprint  # a trailing slash is the same store
        assert other.store_url == STORE
        assert ts.compute_stamp(STORE, stamp.bbox, 2023, client=FixtureClient()).fingerprint != (
            stamp.fingerprint
        )

    def test_metadata_without_a_grid_is_a_store_check_error(self):
        bbox = _read_bbox(EXAMPLE_05)
        objects = _objects()
        objects["utm20/zarr.json"] = json.dumps({"attributes": {}}).encode()
        with pytest.raises(ts.StoreCheckError, match="usable grid"):
            ts.compute_stamp(STORE, bbox, 2024, client=FixtureClient(objects=objects))
        ts.clear_memo()  # the answers above are remembered for the store URL
        objects = _objects()
        del objects["utm20/scales/zarr.json"]
        with pytest.raises(ts.StoreCheckError, match="no array 'scales'"):
            ts.compute_stamp(STORE, bbox, 2024, client=FixtureClient(objects=objects))


class TestStoreStampEntry:
    @requires_zarr
    def test_check_turned_off(self, monkeypatch):
        monkeypatch.setenv(ts.CHECK_ENV, "0")
        client = FixtureClient()
        with pytest.raises(ts.StoreCheckError, match=ts.CHECK_ENV):
            ts.store_stamp(STORE, _read_bbox(EXAMPLE_05), 2024, client=client)
        assert client.requests == []
        monkeypatch.setenv(ts.CHECK_ENV, "1")
        assert ts.store_stamp(STORE, _read_bbox(EXAMPLE_05), 2024, client=client).n_objects == 6

    def test_no_connection_is_remembered_for_the_store(self, monkeypatch):
        """A failed name lookup fails every area alike: later checks fail at once."""

        class Offline(FixtureClient):
            def _request(self, method, key):
                self.requests.append((method, key))
                raise ts.StoreCheckError(f"{method} {key} failed: gaierror", unreachable=True)

        client = Offline()
        with pytest.raises(ts.StoreCheckError, match="gaierror") as info:
            ts.store_stamp(STORE, _read_bbox(EXAMPLE_05), 2024, client=client)
        assert info.value.unreachable
        n = len(client.requests)
        # Another area of the same store: no request within FAILURE_TTL_S.
        with pytest.raises(ts.StoreCheckError, match="no connection to the store 0 s ago"):
            ts.store_stamp(STORE, _read_bbox(EXAMPLE_02), 2024, client=client)
        assert len(client.requests) == n
        monkeypatch.setattr(ts, "FAILURE_TTL_S", 0.0)
        with pytest.raises(ts.StoreCheckError):
            ts.store_stamp(STORE, _read_bbox(EXAMPLE_02), 2024, client=client)
        assert len(client.requests) > n

    @requires_zarr
    def test_an_http_failure_does_not_fail_other_areas(self):
        """Zone 45 answers HTTP 503 (after the retries); example 05 (zone 20) is checked."""

        class Throttled(FixtureClient):
            def _request(self, method, key):
                if method == "HEAD" and key.startswith("utm45/"):
                    with self._lock:
                        self.requests.append((method, key))
                    raise ts.StoreCheckError(f"HEAD {self.url(key)} failed: HTTP 503")
                return super()._request(method, key)

        client = Throttled()
        with pytest.raises(ts.StoreCheckError, match="HTTP 503") as info:
            ts.store_stamp(STORE, _read_bbox(EXAMPLE_02), 2024, client=client)
        assert not info.value.unreachable
        assert ts.store_stamp(STORE, _read_bbox(EXAMPLE_05), 2024, client=client).n_objects == 6
        # The failed requests are not repeated within FAILURE_TTL_S. (HEADs that had not
        # started when the first one failed were cancelled; they may be sent now.)
        failed = [r for r in client.requests if r[0] == "HEAD" and r[1].startswith("utm45/")]
        assert failed
        with pytest.raises(ts.StoreCheckError, match="not repeated within"):
            ts.store_stamp(STORE, _read_bbox(EXAMPLE_02), 2024, client=client)
        heads = [r for r in client.requests if r[0] == "HEAD" and r[1].startswith("utm45/")]
        assert len(heads) == len(set(heads))  # no HEAD request was sent twice

    @requires_zarr
    def test_answers_are_reused_for_a_minute_and_fresh_asks_again(self, monkeypatch):
        assert ts.MEMO_TTL_S == 60.0
        bbox = _read_bbox(EXAMPLE_05)
        first = ts.store_stamp(STORE, bbox, 2024, client=FixtureClient())
        rewritten = dict(HEADS)
        rewritten["utm20/embeddings/c/7/0/316/13"] = (
            '"rewritten-33"',
            "Fri, 09 Oct 2026 10:00:00 GMT",
        )
        # Within MEMO_TTL_S the process answers from the memo ...
        memo = FixtureClient(heads=rewritten)
        assert ts.store_stamp(STORE, bbox, 2024, client=memo).fingerprint == first.fingerprint
        assert memo.requests == []
        # ... unless the stamp is computed fresh (as before a read) ...
        fresh = FixtureClient(heads=rewritten, fresh=True)
        changed = ts.store_stamp(STORE, bbox, 2024, client=fresh)
        assert changed.fingerprint != first.fingerprint
        assert ("HEAD", "utm20/embeddings/c/7/0/316/13") in fresh.requests
        # ... and after MEMO_TTL_S it asks again.
        ts.clear_memo()
        ts.store_stamp(STORE, bbox, 2024, client=FixtureClient())
        monkeypatch.setattr(ts, "MEMO_TTL_S", 0.0)
        later = ts.store_stamp(STORE, bbox, 2024, client=FixtureClient(heads=rewritten))
        assert later.fingerprint == changed.fingerprint


# ---------------------------------------------------------------------------
# HTTP (urlopen mocked)
# ---------------------------------------------------------------------------


class _Response:
    def __init__(self, status=200, headers=None, body=b""):
        self.status = status
        self.headers = headers or {}
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def urlopen(monkeypatch):
    """Replace urlopen in tessera_stamp with a queue of answers; undo the conftest guard."""
    monkeypatch.undo()  # the conftest guard replaced StoreClient._request
    calls, answers = [], []

    def fake(request, timeout):
        calls.append(
            (request.get_method(), request.full_url, dict(request.header_items()), timeout)
        )
        answer = answers.pop(0) if answers else _Response()
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(ts, "urlopen", fake)
    monkeypatch.setattr(ts, "_retry_wait", lambda retry: 0.0)
    return calls, answers


def _http_error(code):
    return HTTPError("https://x", code, f"HTTP {code}", {}, io.BytesIO(b""))


class TestHttp:
    def test_head_parses_the_answer_and_sends_a_user_agent(self, urlopen):
        from agribound._version import __version__

        calls, answers = urlopen
        answers.append(
            _Response(
                headers={"ETag": '"ccea-33"', "Last-Modified": "Wed, 07 Oct 2026 12:40:56 GMT"}
            )
        )
        state = ts.StoreClient(STORE + "/").head("utm20/embeddings/c/7/0/316/13")
        assert state == ts.ObjectState(
            "utm20/embeddings/c/7/0/316/13", "ccea-33", "2026-10-07T12:40:56Z"
        )
        method, url, headers, timeout = calls[0]
        assert (method, url) == ("HEAD", f"{STORE}/utm20/embeddings/c/7/0/316/13")
        assert headers["User-agent"] == f"agribound/{__version__}"
        assert timeout == ts.HTTP_TIMEOUT_S

    def test_missing_object_and_weak_or_no_etag(self, urlopen):
        calls, answers = urlopen
        answers += [_http_error(404), _Response(headers={"ETag": 'W/"abc"'}), _Response()]
        client = ts.StoreClient(STORE)
        assert client.head("a").etag is None and not client.head("a").exists
        assert client.head("b").etag == "abc"
        assert client.head("c").etag == "no-etag;;"
        assert len(calls) == 3  # the second head("a") came from the memo

    def test_transient_failures_are_retried(self, urlopen):
        calls, answers = urlopen
        answers += [
            _http_error(503),
            TimeoutError("read timed out"),
            _Response(headers={"ETag": '"e"'}),
        ]
        assert ts.StoreClient(STORE).head("k").etag == "e"
        assert len(calls) == 3

    @pytest.mark.parametrize(
        ("error", "no_connection"),
        [
            (_http_error(403), False),
            (URLError(socket.gaierror(8, "nodename nor servname provided")), True),
            (URLError(ConnectionRefusedError(61, "Connection refused")), True),
            (URLError(TimeoutError("timed out")), True),
            # Compute nodes with internal DNS but no route out, a local firewall, a
            # TLS-intercepting proxy: no connection can be made.
            (URLError(OSError(errno.ENETUNREACH, "Network is unreachable")), True),
            (URLError(OSError(errno.EHOSTUNREACH, "No route to host")), True),
            (URLError(PermissionError(errno.EACCES, "Permission denied")), True),
            (URLError(ssl.SSLCertVerificationError(1, "certificate verify failed")), True),
        ],
    )
    def test_no_access_fails_at_once(self, urlopen, error, no_connection):
        calls, answers = urlopen
        answers.append(error)
        with pytest.raises(ts.StoreCheckError, match="HEAD .* failed") as info:
            ts.StoreClient(STORE).head("k")
        assert len(calls) == 1
        assert info.value.unreachable is no_connection

    def test_reset_connection_is_retried(self, urlopen):
        calls, answers = urlopen
        answers += [URLError(ConnectionResetError(errno.ECONNRESET, "reset")), _Response()]
        assert ts.StoreClient(STORE).head("k").exists
        assert len(calls) == 2

    def test_redirected_head_stays_head(self, monkeypatch):
        """urllib before Python 3.13 sends a redirected HEAD as GET (a whole shard)."""
        monkeypatch.undo()  # the conftest guard replaced StoreClient._request
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def _answer(self, body):
                seen.append((self.command, self.path))
                if self.path.endswith("/a"):
                    self.send_response(302)
                    self.send_header("Location", self.path[:-1] + "b")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                payload = b"x" * 1024
                self.send_response(200)
                self.send_header("ETag", '"abc-2"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if body:
                    self.wfile.write(payload)

            def do_HEAD(self):  # noqa: N802
                self._answer(body=False)

            def do_GET(self):  # noqa: N802
                self._answer(body=True)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/store"
            state = ts.StoreClient(url, retries=0).head("a")
        finally:
            server.shutdown()
            server.server_close()
        assert seen == [("HEAD", "/store/a"), ("HEAD", "/store/b")]
        assert state.etag == "abc-2"

    def test_retries_are_bounded(self, urlopen):
        calls, answers = urlopen
        answers += [_http_error(502)] * 10
        with pytest.raises(ts.StoreCheckError, match="HTTP 502"):
            ts.StoreClient(STORE, retries=2).head("k")
        assert len(calls) == 3

    def test_get_json(self, urlopen):
        calls, answers = urlopen
        answers += [_Response(body=b'{"a": 1}'), _Response(body=b"[1]"), _http_error(404)]
        client = ts.StoreClient(STORE)
        assert client.get_json("x/zarr.json") == {"a": 1}
        with pytest.raises(ts.StoreCheckError, match="not a JSON object"):
            client.get_json("y/zarr.json")
        assert client.get_json("z/zarr.json") is None
        assert calls[0][0] == "GET"

    def test_head_requests_run_a_bounded_number_at_a_time(self, urlopen, monkeypatch):
        lock, state = threading.Lock(), {"now": 0, "max": 0}

        def slow(request, timeout):
            with lock:
                state["now"] += 1
                state["max"] = max(state["max"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return _Response(headers={"ETag": '"e"'})

        monkeypatch.setattr(ts, "urlopen", slow)
        out = ts.StoreClient(STORE, workers=3).head_many([f"k{i}" for i in range(12)])
        assert len(out) == 12 and 1 < state["max"] <= 3

    def test_fresh_client_asks_again_once(self, urlopen):
        calls, answers = urlopen
        answers += [_Response(headers={"ETag": '"e1"'}), _Response(headers={"ETag": '"e2"'})]
        assert ts.StoreClient(STORE).head("k").etag == "e1"
        fresh = ts.StoreClient(STORE, fresh=True)
        assert fresh.head("k").etag == "e2" and fresh.head("k").etag == "e2"
        assert ts.StoreClient(STORE).head("k").etag == "e2"  # remembered for later clients
        assert len(calls) == 2


# ---------------------------------------------------------------------------
# geotessera's on-disk cache
# ---------------------------------------------------------------------------


@pytest.fixture
def cached_read(tmp_path):
    """A stamp for example 05 and a geotessera cache with copies of its objects."""
    pytest.importorskip("geotessera")
    stamp, _ = _stamp()
    heads = dict(HEADS)
    heads["zarr.json"] = ('"e5b7bc701710fd2f61a829b29f0503d7"', "Mon, 17 Aug 2026 09:00:43 GMT")
    for name, n in (("time", 1), ("band", 1), ("x", 2), ("y", 16)):
        for i in range(n):
            heads[f"utm20/{name}/c/{i}"] = (f'"{name}{i}"', "Fri, 31 Jul 2026 18:12:00 GMT")
    cache = tmp_path / "emb"
    root = ts.local_cache_root(cache, STORE)
    files = {}
    for key in ("zarr.json", "utm20/y/c/3", "utm20/embeddings/c/7/0/316/13", "utm45/x/c/0"):
        path = root.joinpath(*key.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"old copy")
        files[key] = path
    return stamp, cache, files, heads


class TestLocalCache:
    def test_root_is_geotesseras_cache_directory(self, tmp_path):
        store = pytest.importorskip("geotessera.store")
        store.zarr_store(STORE, cache_dir=tmp_path)  # creates its directory, no request
        assert ts.local_cache_root(tmp_path, STORE).is_dir()
        assert ts.local_cache_root(tmp_path, STORE) == tmp_path / "v1"

    def test_disk_cached_keys(self):
        assert ts.disk_cached_key("zarr.json") and ts.disk_cached_key("utm20/zarr.json")
        assert ts.disk_cached_key("utm20/embeddings/zarr.json")
        assert ts.disk_cached_key("utm20/x/c/1") and ts.disk_cached_key("utm01/time/c/0")
        assert not ts.disk_cached_key("utm20/embeddings/c/7/0/316/13")
        assert not ts.disk_cached_key("utm20/scales/c/7/316/13")
        assert not ts.disk_cached_key("utm20/embeddings_d64/c/7/0/316/13")

    @requires_zarr
    def test_unrecorded_copies_are_deleted_then_recorded(self, cached_read):
        stamp, cache, files, heads = cached_read
        current = ts.refresh_local_cache(cache, stamp, client=FixtureClient(heads=heads))
        # Copies without a recorded ETag go; the other zone's copy is not part of this read.
        assert not files["zarr.json"].exists()
        assert not files["utm20/y/c/3"].exists()
        assert files["utm45/x/c/0"].exists()
        # A shard copy (from geotessera's own cache) is not read by agribound: left alone.
        assert files["utm20/embeddings/c/7/0/316/13"].exists()
        assert "utm20/embeddings/c/7/0/316/13" not in current
        assert current["utm20/y/c/15"] == "y15" and current["zarr.json"].startswith("e5b7")
        assert "utm20/x/c/2" not in current  # x has 2 chunks

        # geotessera writes the copies again during the read; their ETags are recorded.
        for key in ("zarr.json", "utm20/y/c/3"):
            files[key].write_bytes(b"new copy")
        ts.record_local_cache(cache, STORE, {**current, "utm20/embeddings/c/7/0/316/13": "e"})
        index = json.loads((cache / "v1.agribound-etags.json").read_text())
        assert index["store"] == STORE
        assert index["objects"] == {
            "utm20/y/c/3": "y3",
            "zarr.json": "e5b7bc701710fd2f61a829b29f0503d7",
        }

        # The next read keeps them; a rewritten and a removed chunk are deleted.
        ts.clear_memo()
        assert ts.refresh_local_cache(cache, stamp, client=FixtureClient(heads=heads))
        assert all(files[k].exists() for k in index["objects"])
        ts.clear_memo()
        heads["zarr.json"] = ('"rewritten"', "Fri, 09 Oct 2026 08:00:00 GMT")
        del heads["utm20/y/c/3"]
        current = ts.refresh_local_cache(cache, stamp, client=FixtureClient(heads=heads))
        assert not files["zarr.json"].exists()
        assert not files["utm20/y/c/3"].exists()
        ts.record_local_cache(cache, STORE, current)
        index = json.loads((cache / "v1.agribound-etags.json").read_text())
        assert index["objects"] == {}


# A small TESSERA-like Zarr v3 store on disk: one zone, sharded embeddings and scales,
# x/y/time/band coordinates, consolidated metadata (as the v1 store is laid out).
MINI_ZONE, MINI_CRS = 20, "EPSG:32620"
MINI_X0, MINI_Y0 = 716800.0, -3725000.0
MINI_SIZE, MINI_SHARD, MINI_INNER = 128, 64, 32


def _write_mini_store(root: Path) -> None:
    zarr = pytest.importorskip("zarr")
    store = zarr.storage.LocalStore(root)
    group = zarr.open_group(store, mode="w", zarr_format=3)
    group.attrs.update({"geoemb:dimensions": 128})
    zone = group.create_group(f"utm{MINI_ZONE:02d}")
    zone.attrs.update(
        {
            "proj:code": MINI_CRS,
            "spatial:transform": [10.0, 0.0, MINI_X0, 0.0, -10.0, MINI_Y0],
            "spatial:shape": [MINI_SIZE, MINI_SIZE],
            "spatial:registration": "pixel",
        }
    )
    n = MINI_SIZE
    time_ = zone.create_array(
        "time", shape=(2,), chunks=(2,), dtype="int32", dimension_names=["time"]
    )
    time_[:] = np.array([2023, 2024], dtype="int32")
    band = zone.create_array(
        "band", shape=(128,), chunks=(128,), dtype="int32", dimension_names=["band"]
    )
    band[:] = np.arange(128, dtype="int32")
    x = zone.create_array("x", shape=(n,), chunks=(n // 2,), dtype="float64", dimension_names=["x"])
    x[:] = MINI_X0 + (np.arange(n) + 0.5) * 10.0
    y = zone.create_array("y", shape=(n,), chunks=(n // 2,), dtype="float64", dimension_names=["y"])
    y[:] = MINI_Y0 - (np.arange(n) + 0.5) * 10.0
    emb = zone.create_array(
        "embeddings",
        shape=(2, 128, n, n),
        shards=(1, 128, MINI_SHARD, MINI_SHARD),
        chunks=(1, 128, MINI_INNER, MINI_INNER),
        dtype="int8",
        dimension_names=["time", "band", "y", "x"],
    )
    rows, cols = np.mgrid[0:n, 0:n]
    emb[:] = np.broadcast_to(((rows + cols) % 120).astype("int8"), (2, 128, n, n))
    scales = zone.create_array(
        "scales",
        shape=(2, n, n),
        shards=(1, MINI_SHARD, MINI_SHARD),
        chunks=(1, MINI_INNER, MINI_INNER),
        dtype="float32",
        dimension_names=["time", "y", "x"],
    )
    scales[:] = 1.0
    zarr.consolidate_metadata(store)


def _mini_bbox(margin_m: float = 5.0):
    """EPSG:4326 box of the mini store's whole grid (every shard is read completely)."""
    to_ll = pyproj.Transformer.from_crs(MINI_CRS, "EPSG:4326", always_xy=True)
    extent = MINI_SIZE * 10.0
    corners = [
        to_ll.transform(MINI_X0 + dx, MINI_Y0 - dy)
        for dx in (-margin_m, extent + margin_m)
        for dy in (-margin_m, extent + margin_m)
    ]
    lons, lats = zip(*corners, strict=True)
    return (min(lons), min(lats), max(lons), max(lats))


class LocalStoreClient(ts.StoreClient):
    """Serves the files of a store on disk; ETag = MD5 of the bytes (S3, single part)."""

    def __init__(self, root: Path):
        super().__init__("https://example.invalid/zarr/mini")
        self.root = root

    def _request(self, method, key):
        path = self.root.joinpath(*key.split("/"))
        if not path.is_file():
            return 404, {}, b""
        data = path.read_bytes()
        etag = f'"{hashlib.md5(data).hexdigest()}"'
        return 200, {"etag": etag}, data if method == "GET" else b""


@requires_zarr
class TestMiniStore:
    """A real Zarr store, read with geotessera's GeoTesseraZarr (no network)."""

    def test_geotessera_store_keeps_only_metadata_and_coordinates(self, tmp_path):
        pytest.importorskip("geotessera")
        from geotessera import GeoTesseraZarr

        root = tmp_path / "mini"
        _write_mini_store(root)
        bbox = _mini_bbox()
        direct, transform, crs = GeoTesseraZarr(str(root)).read_region(bbox, 2024)
        assert direct.shape == (MINI_SIZE, MINI_SIZE, 128)  # every shard read completely

        def cached_files(cache):
            base = ts.local_cache_root(cache, str(root))
            return {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}

        # geotessera's own cache keeps the shards a read covers completely ...
        GeoTesseraZarr(str(root), cache_dir=tmp_path / "plain").read_region(bbox, 2024)
        plain = cached_files(tmp_path / "plain")
        assert any(f.startswith("utm20/embeddings/c/") for f in plain)
        # ... agribound's keeps only the metadata documents and coordinate chunks.
        cache = tmp_path / "agribound"
        store = ts.geotessera_store(str(root), cache)
        arr, transform2, crs2 = GeoTesseraZarr(store).read_region(bbox, 2024)
        assert np.array_equal(arr, direct, equal_nan=True)
        assert (transform2, crs2) == (transform, crs)
        kept = cached_files(cache)
        assert "zarr.json" in kept and "utm20/x/c/0" in kept and "utm20/y/c/1" in kept
        assert all(ts.disk_cached_key(f) for f in kept)
        assert not any(f.startswith(("utm20/embeddings/c/", "utm20/scales/c/")) for f in kept)
        # A second read uses the cached metadata and gives the same data.
        again, _, _ = GeoTesseraZarr(ts.geotessera_store(str(root), cache)).read_region(bbox, 2024)
        assert np.array_equal(again, direct, equal_nan=True)

    def test_a_coordinate_rewrite_changes_the_stamp(self, tmp_path):
        """A rewrite of a zone's x values moves geotessera's read; the stamp follows it."""
        pytest.importorskip("geotessera")
        import zarr
        from geotessera import GeoTesseraZarr

        root = tmp_path / "mini"
        _write_mini_store(root)
        bbox = _mini_bbox(margin_m=-200.0)  # inside the grid
        url = "https://example.invalid/zarr/mini"
        before = ts.compute_stamp(url, bbox, 2024, client=LocalStoreClient(root))
        assert any(o.key.startswith("utm20/x/c/") for o in before.objects)
        first, transform, _ = GeoTesseraZarr(str(root)).read_region(bbox, 2024)

        x = zarr.open_array(zarr.storage.LocalStore(root), path="utm20/x", mode="r+")
        x[:] = MINI_X0 + (np.arange(MINI_SIZE) + 0.5) * 10.0 + 10.0  # one pixel east
        moved, transform_moved, _ = GeoTesseraZarr(str(root)).read_region(bbox, 2024)
        assert transform_moved != transform or not np.array_equal(moved, first, equal_nan=True)
        ts.clear_memo()
        after = ts.compute_stamp(url, bbox, 2024, client=LocalStoreClient(root))
        assert after.fingerprint != before.fingerprint


# ---------------------------------------------------------------------------
# The real store (network)
# ---------------------------------------------------------------------------


@pytest.mark.network
def test_real_store_head_and_example_05_stamp():
    """HEAD one real shard, then the stamp of example 05's read (v1, 2024)."""
    pytest.importorskip("zarr")
    ts.clear_memo()
    client = ts.StoreClient(STORE)
    state = client.head("utm20/embeddings/c/7/0/316/13")
    assert state.exists and state.etag
    assert state.last_modified and state.last_modified.endswith("Z")
    assert not client.head("utm20/embeddings/c/7/0/999/99").exists
    stamp = ts.store_stamp(STORE, _read_bbox(EXAMPLE_05), 2024)
    assert [o.key for o in stamp.objects] == [
        "utm20/embeddings/c/7/0/315/13",
        "utm20/embeddings/c/7/0/316/13",
        "utm20/scales/c/7/315/13",
        "utm20/scales/c/7/316/13",
        "utm20/x/c/1",
        "utm20/y/c/13",
    ]
    assert stamp.n_objects == 6 and len(stamp.fingerprint) == 64
    assert stamp.grids == ("utm20 EPSG:32620 10.0,0.0,166020.0,0.0,-10.0,9217140.0",)
    logging.getLogger(__name__).info("example 05 store stamp: %s", stamp.describe())
