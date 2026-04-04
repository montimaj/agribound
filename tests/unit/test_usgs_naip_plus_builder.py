from __future__ import annotations

from shapely.geometry import box

from agribound.composites.base import get_composite_builder
from agribound.config import AgriboundConfig


class TestUSGSNAIPPlusBuilder:
    def test_config_accepts_usgs_source(self, tmp_path):
        cfg = AgriboundConfig(
            source="usgs-naip-plus",
            engine="delineate-anything",
            year=2023,
            study_area=str(tmp_path / "aoi.geojson"),
            output_path=str(tmp_path / "out.gpkg"),
        )
        assert cfg.source == "usgs-naip-plus"
        assert cfg.is_gee_source() is False

    def test_factory_returns_usgs_builder(self):
        builder = get_composite_builder("usgs-naip-plus")
        assert builder.__class__.__name__ == "USGSNAIPPlusCompositeBuilder"

    def test_select_lock_raster_ids_returns_nonempty_list(
        self,
        sample_usgs_query_features,
    ):
        from agribound.clients.usgs_naip_plus import USGSNAIPPlusClient
        from agribound.composites.usgs import USGSNAIPPlusCompositeBuilder

        client = USGSNAIPPlusClient("https://example.com/ImageServer")
        builder = USGSNAIPPlusCompositeBuilder()

        candidates = [
            client._feature_to_candidate(feature)
            for feature in sample_usgs_query_features["features"]
        ]
        aoi = box(-13024380.0, 5265000.0, -13022380.0, 5266000.0)

        selected_ids, ranked = builder._select_lock_raster_ids(
            candidates,
            aoi,
            target_year=2023,
            max_ids=50,
        )

        assert selected_ids
        assert selected_ids == sorted(selected_ids)
        assert len(ranked) == 2
