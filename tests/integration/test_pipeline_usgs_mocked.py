from __future__ import annotations

import shutil
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from agribound.pipeline import delineate
from agribound.provenance import read_provenance


class StubEngine:
    """One field inside the study area (usgs_aoi.geojson) and one far outside it."""

    def delineate(self, raster_path, config):
        assert Path(raster_path).exists()
        return gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[box(-116.998, 42.002, -116.997, 42.003), box(0.0, 0.0, 0.001, 0.001)],
            crs="EPSG:4326",
        )


def test_pipeline_usgs_mocked_returns_vector_output(
    monkeypatch,
    tmp_path,
    sample_usgs_service_metadata,
    sample_usgs_query_features,
    sample_export_tif,
    sample_usgs_aoi_geojson,
):
    from agribound.clients.usgs_naip_plus import USGSNAIPPlusClient

    def fake_get_service_metadata(self):
        return sample_usgs_service_metadata

    def fake_query_candidates(self, bounds_3857, where, out_sr=3857, batch_size=50):
        return [
            self._feature_to_candidate(feature)
            for feature in sample_usgs_query_features["features"]
        ]

    def fake_export_image(
        self,
        *,
        bbox_3857,
        width,
        height,
        lock_raster_ids,
        output_path,
        compression="LZ77",
    ):
        shutil.copyfile(sample_export_tif, output_path)
        return {"href": "mock://export.tif"}

    monkeypatch.setattr(USGSNAIPPlusClient, "get_service_metadata", fake_get_service_metadata)
    monkeypatch.setattr(USGSNAIPPlusClient, "query_candidates", fake_query_candidates)
    monkeypatch.setattr(USGSNAIPPlusClient, "export_image", fake_export_image)
    monkeypatch.setattr("agribound.engines.get_engine", lambda engine: StubEngine())

    output_path = tmp_path / "fields.gpkg"

    result = delineate(
        source="usgs-naip-plus",
        engine="delineate-anything",
        study_area=sample_usgs_aoi_geojson,
        year=2023,
        output_path=str(output_path),
        device="cpu",
        usgs_state="MI",
        min_field_area_m2=0,
        simplify_tolerance=0,
        lulc_filter=False,
    )

    assert output_path.exists()
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 1
    assert (result["agribound:source"] == "usgs-naip-plus").all()
    assert result["id"].iloc[0].startswith(result.attrs["run_id"])

    record = read_provenance(output_path)
    assert record["status"] == "success"
    assert record["config"]["source"] == "usgs-naip-plus"
    assert record["config"]["usgs_state"] == "MI"
    assert record["facts"]["n_output"] == 1
    # The field outside the study area is removed by the default aoi_selection.
    assert record["facts"]["aoi_selection"] == {
        "rule": "representative_point",
        "n_before": 2,
        "n_after": 1,
    }
    assert Path(record["facts"]["raster_path"]).exists()

    # Same configuration again: the provenance hash matches, nothing is re-exported.
    def fail_export(self, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("export_image must not run for a reused output")

    monkeypatch.setattr(USGSNAIPPlusClient, "export_image", fail_export)
    again = delineate(
        source="usgs-naip-plus",
        engine="delineate-anything",
        study_area=sample_usgs_aoi_geojson,
        year=2023,
        output_path=str(output_path),
        device="cpu",
        usgs_state="MI",
        min_field_area_m2=0,
        simplify_tolerance=0,
        lulc_filter=False,
    )
    assert again.attrs["reused"] is True
    assert len(again) == 1
