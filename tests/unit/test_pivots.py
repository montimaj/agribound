"""Tests for the centre-pivot post-process (agribound.postprocess.pivots).

Synthetic shapes in UTM zone 11N (metres), and the pipeline with a stubbed
engine on the local test raster; no data, network or ML dependencies.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely import affinity
from shapely.geometry import Point, Polygon, box

from agribound.config import AgriboundConfig
from agribound.postprocess import pivots
from agribound.postprocess.pivots import (
    PIVOT_COLUMN,
    circle_tolerance_m,
    fit_circle,
    resolve_pivot_mode,
    snap_pivots,
)

UTM = "EPSG:32611"
X0, Y0 = 500_000.0, 4_400_000.0


def _sector(r, a0, a1, cx=X0, cy=Y0, n=200):
    t = np.linspace(np.radians(a0), np.radians(a1), n)
    return Polygon([(cx, cy), *zip(cx + r * np.cos(t), cy + r * np.sin(t), strict=True)])


def _circle(r=400.0, cx=X0, cy=Y0):
    return Point(cx, cy).buffer(r, quad_segs=64)


def _ellipse(a, b, n=720):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(list(zip(X0 + a * np.cos(t), Y0 + b * np.sin(t), strict=True)))


def _layer(geoms, crs=UTM, **columns):
    data = {"fid": list(range(len(geoms))), **columns}
    return gpd.GeoDataFrame(data, geometry=list(geoms), crs=crs)


def _iou(a, b):
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter)


def _same(a, b):
    return len(a) == len(b) and all(
        x.equals_exact(y, 0.0) for x, y in zip(a.geometry, b.geometry, strict=True)
    )


# ---------------------------------------------------------------------------
# Circle fit
# ---------------------------------------------------------------------------


class TestCircleFit:
    def _points(self, r, n=400, noise=0.0, seed=1):
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rr = r + rng.normal(0, noise, n) if noise else np.full(n, r)
        return np.column_stack([X0 + rr * np.cos(t), Y0 + rr * np.sin(t)])

    def test_exact_and_noisy_circles(self):
        cx, cy, r = fit_circle(self._points(400.0), np.random.default_rng(0))
        # the algebraic fit on UTM coordinates (no centring) is good to about 1e-5 m
        assert (cx, cy, r) == pytest.approx((X0, Y0, 400.0), abs=1e-4)
        cx, cy, r = fit_circle(self._points(250.0, noise=1.0), np.random.default_rng(0))
        assert np.hypot(cx - X0, cy - Y0) < 0.5 and r == pytest.approx(250.0, abs=0.5)

    def test_no_circle(self):
        rng = np.random.default_rng(0)
        assert fit_circle(self._points(20.0), rng) is None  # below the 30 m radius
        assert fit_circle(self._points(400.0)[:7], rng) is None  # fewer than 8 points
        line = np.column_stack([np.linspace(0, 100, 50), np.zeros(50)])
        assert fit_circle(line, rng) is None  # collinear

    def test_deterministic_and_one_draw_per_fit(self):
        pts = self._points(300.0, noise=2.0)
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        assert fit_circle(pts, a) == fit_circle(pts, b)
        ref = np.random.default_rng(7)
        ref.integers(0, len(pts), size=(pivots.RANSAC_ITERATIONS, 3))
        assert a.integers(0, 10**9) == ref.integers(0, 10**9)

    def test_tolerance(self):
        assert circle_tolerance_m(100.0) == 5.0
        assert circle_tolerance_m(400.0) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Merging and snapping
# ---------------------------------------------------------------------------


class TestMergeAndSnap:
    @pytest.mark.parametrize(
        "pieces",
        [
            pytest.param([_sector(400, a, a + 90) for a in (0, 90, 180, 270)], id="4-quarters"),
            pytest.param(
                [
                    _sector(400, 0, 180, cy=Y0 + 0.5),
                    _sector(400, 180, 360, cy=Y0 - 0.5),
                ],
                id="halves-1m-gap",
            ),
            pytest.param(
                [
                    _circle().intersection(box(X0 - 500, Y0 - 500, X0 + 100, Y0 + 500)),
                    _circle().intersection(box(X0 + 101, Y0 - 500, X0 + 500, Y0 + 500)),
                ],
                id="tile-cut",
            ),
        ],
    )
    def test_cut_pivot_is_rejoined(self, pieces):
        layer = _layer(pieces)
        snapped = snap_pivots(layer, "snap")
        assert len(snapped) == 1 and list(snapped[PIVOT_COLUMN]) == ["merged"]
        assert _iou(snapped.geometry.iloc[0], _circle()) > 0.99
        assert len(shapely.get_coordinates(snapped.geometry.iloc[0])) == 4 * pivots.QUAD_SEGS + 1
        stats = snapped.attrs["pivot_stats"]
        assert (stats["n_groups"], stats["n_groups_multi"]) == (1, 1)
        assert stats["n_pieces_merged"] == len(pieces) and stats["n_in"] == len(pieces)
        merged = snap_pivots(layer, "merge")
        assert len(merged) == 1 and list(merged[PIVOT_COLUMN]) == ["merged"]
        # the union, closed by the 1 m buffer (which fills the cut): not a fitted circle
        assert _iou(merged.geometry.iloc[0], shapely.union_all(pieces)) > 0.99
        assert len(shapely.get_coordinates(merged.geometry.iloc[0])) != 4 * pivots.QUAD_SEGS + 1

    def test_group_takes_the_largest_polygons_attributes(self):
        pieces = [_sector(400, 0, 120), _sector(400, 120, 360)]
        out = snap_pivots(_layer(pieces, name=["small", "large"]), "snap")
        assert list(out["name"]) == ["large"] and list(out["fid"]) == [1]

    def test_single_round_polygon(self):
        wobbly = Polygon(
            [
                (X0 + 400 * (1 + 0.03 * np.sin(7 * t)) * np.cos(t),
                 Y0 + 400 * (1 + 0.03 * np.sin(7 * t)) * np.sin(t))
                for t in np.linspace(0, 2 * np.pi, 400, endpoint=False)
            ]
        )  # fmt: skip
        for geom in (_circle(), wobbly, _ellipse(410, 390)):
            out = snap_pivots(_layer([geom]), "snap")
            assert list(out[PIVOT_COLUMN]) == ["snapped"]
            assert _iou(out.geometry.iloc[0], geom) > 0.95
            kept = snap_pivots(_layer([geom]), "merge")
            # merge mode: a group of one is closed by the 1 m buffer and labelled "kept"
            assert list(kept[PIVOT_COLUMN]) == ["kept"]
            assert kept.geometry.iloc[0].equals_exact(geom.buffer(1.0).buffer(-1.0), 0.0)

    def test_snapped_circle_is_cut_out_of_other_polygons(self):
        rect = box(X0 + 300, Y0 - 100, X0 + 700, Y0 + 100)
        quarters = [_sector(400, a, a + 90).difference(rect) for a in (0, 90, 180, 270)]
        out = snap_pivots(_layer([*quarters, rect]), "snap")
        assert list(out[PIVOT_COLUMN]) == ["merged", "kept"]
        stats = out.attrs["pivot_stats"]
        assert (stats["n_clipped"], stats["n_dropped"]) == (1, 0)
        circle, cut = out.geometry
        assert cut.intersection(circle).area < 1e-6
        assert cut.area == pytest.approx(rect.difference(circle).area)
        # the attributes of the group's largest piece (the first of equal areas), then the rest
        largest = int(np.argsort(-np.array([q.area for q in quarters]), kind="stable")[0])
        assert list(out["fid"]) == [largest, 4]

    def test_residue_below_the_minimum_area_is_dropped(self):
        small = box(X0 + 390, Y0 - 30, X0 + 430, Y0 + 30)  # 1,824 m2 outside the circle
        out = snap_pivots(_layer([_circle(), small]), "snap")
        assert list(out[PIVOT_COLUMN]) == ["snapped"]
        stats = out.attrs["pivot_stats"]
        assert (stats["n_clipped"], stats["n_dropped"]) == (1, 1)
        kept = snap_pivots(_layer([_circle(), small]), "snap", min_area_m2=1000.0)
        assert list(kept[PIVOT_COLUMN]) == ["snapped", "kept"]

    def test_two_pivots_and_a_small_one(self):
        far = 1500.0
        layer = _layer(
            [
                _circle(),
                _sector(60, 0, 180, cx=X0 + far),
                _sector(60, 180, 360, cx=X0 + far),
            ]
        )
        out = snap_pivots(layer, "snap", min_area_m2=0)
        assert sorted(out[PIVOT_COLUMN]) == ["merged", "snapped"]
        assert out.attrs["pivot_stats"]["n_groups"] == 2


# ---------------------------------------------------------------------------
# Shapes left alone
# ---------------------------------------------------------------------------


class TestShapesLeftAlone:
    @pytest.mark.parametrize(
        "geoms",
        [
            pytest.param([box(X0 - 280, Y0 - 280, X0 + 280, Y0 + 280)], id="square"),
            pytest.param([box(X0 - 300, Y0 - 100, X0 + 300, Y0 + 100)], id="rectangle"),
            pytest.param([box(X0 - 220, Y0 - 220, X0 + 220, Y0 + 220).buffer(60)], id="rounded"),
            pytest.param([_sector(400, 0, 270)], id="three-quarter-pivot"),
            pytest.param([_sector(400, a, a + 90) for a in (0, 90, 180)], id="three-quarters-3"),
            pytest.param(
                [_circle().union(affinity.translate(_circle(), 800, 0))],
                id="two-pivots-one-polygon",
            ),
            pytest.param([_ellipse(420, 380), _ellipse(480, 333)], id="ellipses"),
            pytest.param([Point(X0, Y0).buffer(25, quad_segs=64)], id="radius-below-30m"),
        ],
    )
    def test_unchanged(self, geoms):
        layer = _layer(geoms)
        for mode in ("snap", "merge"):
            out = snap_pivots(layer, mode, min_area_m2=0)
            assert _same(out, layer)
            assert set(out[PIVOT_COLUMN]) == {"kept"}
            assert out.attrs["pivot_stats"]["n_groups"] == 0
            assert list(out["fid"]) == list(layer["fid"])

    def test_piece_mostly_outside_the_circle_is_not_a_member(self):
        left = _circle().intersection(box(X0 - 500, Y0 - 500, X0, Y0 + 500))
        right = _circle().intersection(box(X0 + 1, Y0 - 500, X0 + 500, Y0 + 500))
        right = right.union(box(X0 + 1, Y0 + 300, X0 + 450, Y0 + 500))  # 21 % outside
        out = snap_pivots(_layer([left, right]), "snap")
        assert _same(out, _layer([left, right]))


# ---------------------------------------------------------------------------
# Frames, modes and CRS
# ---------------------------------------------------------------------------


class TestFrames:
    def test_modes(self):
        assert resolve_pivot_mode(None) == "off"
        assert resolve_pivot_mode(" Snap ") == "snap" and resolve_pivot_mode("merge") == "merge"
        for bad in ("circle", True, 1, ""):
            with pytest.raises(ValueError, match="pivots"):
                resolve_pivot_mode(bad)
        layer = _layer([_circle()])
        assert snap_pivots(layer, "off") is layer
        with pytest.raises(ValueError, match="pivots"):
            snap_pivots(layer, "circles")

    def test_empty_and_missing_geometries(self):
        empty = snap_pivots(_layer([]), "snap")
        assert len(empty) == 0 and PIVOT_COLUMN in empty.columns
        assert empty.attrs["pivot_stats"]["n_in"] == 0
        layer = gpd.GeoDataFrame({"fid": [0, 1, 2]}, geometry=[None, Polygon(), _circle()], crs=UTM)
        layer.attrs["engine_meta"] = {"backend": "x"}
        out = snap_pivots(layer, "snap")
        assert list(out["fid"]) == [2] and list(out[PIVOT_COLUMN]) == ["snapped"]
        assert out.attrs["engine_meta"] == {"backend": "x"}
        assert out.attrs["pivot_stats"]["n_in"] == 1
        with pytest.raises(ValueError, match="CRS"):
            snap_pivots(gpd.GeoDataFrame(geometry=[_circle()]), "snap")

    def test_deterministic(self):
        pieces = [_sector(400, a, a + 90) for a in (0, 90, 180, 270)]
        pieces += [box(X0 + 900, Y0, X0 + 1200, Y0 + 300), _sector(300, 10, 200, cx=X0 - 1500)]
        a = snap_pivots(_layer(pieces), "snap")
        b = snap_pivots(_layer(pieces), "snap")
        assert _same(a, b) and a.attrs["pivot_stats"] == b.attrs["pivot_stats"]

    def test_geographic_input_is_processed_in_utm_and_returned_as_it_came(self):
        pieces = [_sector(400, a, a + 90) for a in (0, 90, 180, 270)] + [
            box(X0 + 900, Y0, X0 + 1200, Y0 + 300)
        ]
        lonlat = _layer(pieces).to_crs("EPSG:4326")
        out = snap_pivots(lonlat, "snap")
        assert out.crs == lonlat.crs and out.attrs["pivot_stats"]["crs"] == "EPSG:32611"
        # the same as reprojecting first
        expected = snap_pivots(lonlat.to_crs(UTM), "snap").to_crs("EPSG:4326")
        assert _same(out, expected)
        # the square is the input polygon, back in its own CRS
        square = out[out["fid"] == 4].geometry.iloc[0]
        assert square.equals_exact(lonlat.to_crs(UTM).to_crs(4326).geometry.iloc[4], 1e-9)

    def test_other_geometry_column_name(self):
        layer = _layer([_sector(400, 0, 180), _sector(400, 180, 360)]).rename_geometry("geom")
        out = snap_pivots(layer, "snap")
        assert out.geometry.name == "geom" and len(out) == 1


# ---------------------------------------------------------------------------
# Pipeline: engine_params["pivots"]
# ---------------------------------------------------------------------------

RX0, RY0 = 500000.0, 4000000.0  # lower-left corner of the 640 m test raster (EPSG:32611)
RCX, RCY = RX0 + 320, RY0 + 320  # its centre


class _StubEngine:
    """Returns the given polygons (EPSG:32611) with a ``score`` column."""

    def __init__(self, geoms):
        self.geoms = geoms

    def delineate(self, raster_path, config):
        scores = [0.9, 0.8, 0.7][: len(self.geoms)]
        gdf = gpd.GeoDataFrame({"score": scores}, geometry=list(self.geoms), crs=UTM)
        gdf.attrs["engine_meta"] = {"backend": "stub"}
        return gdf


#: A pivot of radius 200 m cut in two pieces of 200 and 160 degrees, and a 150 m square.
CUT_PIVOT = [
    _sector(200, 80, 280, cx=RCX, cy=RCY),
    _sector(200, -80, 80, cx=RCX, cy=RCY),
    box(RX0 + 5, RY0 + 5, RX0 + 155, RY0 + 155),
]


@pytest.fixture
def stub_run(monkeypatch, sample_rgb_tif, tmp_path):
    def run(
        name, geoms=CUT_PIVOT, outline_edits=False, aoi_box=None, settings=None, **engine_params
    ):
        from agribound.pipeline import delineate
        from agribound.provenance import read_provenance

        aoi = tmp_path / f"aoi_{name}.geojson"
        aoi_box = aoi_box or (RX0 - 1000, RY0 - 1000, RX0 + 1640, RY0 + 1640)
        gpd.GeoDataFrame(geometry=[box(*aoi_box)], crs=UTM).to_crs(4326).to_file(aoi)
        monkeypatch.setattr("agribound.engines.get_engine", lambda engine: _StubEngine(geoms))
        settings = dict(settings or {})
        params = {}
        if not outline_edits:
            settings["simplify_tolerance"] = 0.0
            params["smooth_iterations"] = 0
        out = tmp_path / name / "fields.gpkg"
        gdf = delineate(
            source="local",
            local_tif_path=sample_rgb_tif,
            study_area=str(aoi),
            engine="geoai",
            output_path=str(out),
            device="cpu",
            lulc_filter=False,
            engine_params={**params, **engine_params},
            **settings,
        )
        return gdf, read_provenance(out)

    return run


class TestPipelineHook:
    def test_default_is_off(self, stub_run):
        gdf, record = stub_run("default")
        assert len(gdf) == 3 and PIVOT_COLUMN not in gdf.columns
        assert "pivots" not in record["facts"] and "pivot_stats" not in gdf.attrs
        assert "pivots" not in record["config"]["engine_params"]

    def test_snap(self, stub_run):
        gdf, record = stub_run("snap", pivots="snap")
        assert len(gdf) == 2 and list(gdf[PIVOT_COLUMN]) == ["merged", "kept"]
        assert list(gdf["score"]) == [0.9, 0.7]  # the larger piece's attributes
        facts = record["facts"]["pivots"]
        assert facts["mode"] == "snap" and facts["n_groups_multi"] == 1
        assert facts["n_in"] == 3 and facts["n_out"] == 2
        assert facts["settings"]["min_area_m2"] == 2500.0
        assert gdf.attrs["pivot_stats"]["n_pieces_merged"] == 2
        circle = Point(RCX, RCY).buffer(200)
        assert _iou(gdf.to_crs(UTM).geometry.iloc[0], circle) > 0.99
        # the square is untouched, and the metadata columns describe the edited polygons
        assert gdf.to_crs(UTM).geometry.iloc[1].equals_exact(CUT_PIVOT[2], 1e-6)
        assert gdf["metrics:area"].iloc[0] == pytest.approx(np.pi * 200**2, rel=0.01)

    def test_merge_and_explicit_off(self, stub_run):
        gdf, record = stub_run("merge", pivots="merge")
        assert list(gdf[PIVOT_COLUMN]) == ["merged", "kept"]
        assert record["facts"]["pivots"]["mode"] == "merge"
        gdf, record = stub_run("off", pivots="off")
        assert len(gdf) == 3 and "pivots" not in record["facts"]

    def test_after_the_default_outline_edits(self, stub_run):
        """With the default smoothing and simplification, a whole pivot is snapped."""
        pivot = Point(RCX, RCY).buffer(200, quad_segs=16)
        gdf, record = stub_run("edits", geoms=[pivot], outline_edits=True, pivots="snap")
        assert list(gdf[PIVOT_COLUMN]) == ["snapped"]
        assert record["facts"]["postprocess"]["smooth_iterations"] == 3
        assert _iou(gdf.to_crs(UTM).geometry.iloc[0], pivot) > 0.98

    def test_clipped_study_area_is_kept(self, stub_run):
        """With aoi_selection="clip" the snapped circle is clipped to the study area again."""
        pivot = Point(RCX, RCY).buffer(200, quad_segs=64)
        east = RCX + 190.0  # the study area cuts 10 m off the pivot
        aoi = (RX0 - 1000, RY0 - 1000, east, RY0 + 1640)
        settings = {"aoi_selection": "clip"}
        gdf, record = stub_run("clip", geoms=[pivot], aoi_box=aoi, settings=settings, pivots="snap")
        assert list(gdf[PIVOT_COLUMN]) == ["snapped"]
        assert gdf.to_crs(UTM).total_bounds[2] <= east + 0.01
        facts = record["facts"]["pivots"]
        assert (facts["n_before_clip"], facts["n_after_clip"]) == (1, 1)
        # without the option the clipped polygon is not snapped back
        plain, record = stub_run("clip-off", geoms=[pivot], aoi_box=aoi, settings=settings)
        assert "pivots" not in record["facts"]
        assert plain.to_crs(UTM).total_bounds[2] <= east + 0.01

    def test_an_ensemble_output(self, monkeypatch, sample_rgb_tif, tmp_path):
        """The ensemble passes ``pivots`` to the pipeline, which snaps the ensemble's output."""
        from agribound.engines import ensemble
        from agribound.pipeline import delineate
        from agribound.provenance import read_provenance

        # Both members return the cut pivot and the square; the union keeps one of each.
        monkeypatch.setattr(ensemble, "get_engine", lambda name: _StubEngine(CUT_PIVOT))
        aoi = tmp_path / "aoi_ensemble.geojson"
        box_ = (RX0 - 1000, RY0 - 1000, RX0 + 1640, RY0 + 1640)
        gpd.GeoDataFrame(geometry=[box(*box_)], crs=UTM).to_crs(4326).to_file(aoi)
        out = tmp_path / "ensemble" / "fields.gpkg"
        gdf = delineate(
            source="local",
            local_tif_path=sample_rgb_tif,
            study_area=str(aoi),
            engine="ensemble",
            output_path=str(out),
            device="cpu",
            lulc_filter=False,
            simplify_tolerance=0.0,
            engine_params={
                "engines": ["delineate-anything", "ftw"],
                "merge_strategy": "union",
                "smooth_iterations": 0,
                "pivots": "snap",
            },
        )
        assert list(gdf[PIVOT_COLUMN]) == ["merged", "kept"]
        facts = read_provenance(out)["facts"]["pivots"]
        assert facts["mode"] == "snap" and facts["n_in"] == 3 and facts["n_out"] == 2
        assert _iou(gdf.to_crs(UTM).geometry.iloc[0], Point(RCX, RCY).buffer(200)) > 0.99

    def test_unknown_mode_is_refused_by_the_configuration(self, tmp_path):
        with pytest.raises(ValueError, match="pivots"):
            AgriboundConfig(
                source="local",
                local_tif_path=str(tmp_path / "x.tif"),
                engine="delineate-anything",
                output_path=str(tmp_path / "o.gpkg"),
                engine_params={"pivots": "circles"},
            )

    def test_a_member_pivots_key_is_refused_by_the_configuration(self, tmp_path):
        """The step runs on the ensemble's output, so a member's ``pivots`` would do nothing."""
        member = {"engine": "delineate-anything", "engine_params": {"pivots": "snap"}}
        with pytest.raises(ValueError, match="would not be used"):
            AgriboundConfig(
                source="local",
                local_tif_path=str(tmp_path / "x.tif"),
                engine="ensemble",
                output_path=str(tmp_path / "o.gpkg"),
                engine_params={"engines": ["ftw", member]},
            )

    def test_config_hash_unchanged_without_the_option(self, tmp_path):
        from agribound.provenance import config_hash

        kwargs = dict(
            source="local",
            local_tif_path=str(tmp_path / "x.tif"),
            engine="delineate-anything",
            output_path=str(tmp_path / "o.gpkg"),
        )
        base = AgriboundConfig(**kwargs)
        assert config_hash(base) == config_hash(AgriboundConfig(**kwargs, engine_params={}))
        assert config_hash(base) != config_hash(
            AgriboundConfig(**kwargs, engine_params={"pivots": "snap"})
        )
