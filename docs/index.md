# Agribound

**Unified agricultural field boundary delineation toolkit** combining satellite foundation models, embeddings, and global training data.

Agribound provides a single interface to multiple delineation engines and satellite sources, handling the full pipeline from satellite composite generation through post-processing and export. It supports Google Earth Engine-based imagery (Landsat, Sentinel-2, HLS, NAIP, SPOT), local GeoTIFFs, and pre-computed embedding datasets (Google Satellite Embedding, TESSERA).

The pipeline runs: **composite building** &rarr; **optional fine-tuning** &rarr; **delineation engine** &rarr; **post-processing** (smooth, simplify, filter) &rarr; **LULC crop filtering** &rarr; **export**. For ensembles, SAM2 boundary refinement is applied per source for pixel-accurate boundaries.

### Automatic LULC Crop Filtering

Unlike other field boundary packages that detect *all* visual boundaries (including roads, water, forests, and buildings), agribound **automatically removes non-agricultural polygons** using land-use/land-cover data. This is enabled by default and requires no user configuration.

The best available LULC dataset is selected automatically based on your study area:

- **US:** USGS Annual NLCD (1985&ndash;2024, 30 m) &mdash; classes 81/82 (Pasture, Cultivated Crops)
- **Global (&ge;2015):** Google Dynamic World (10 m, nearest year) &mdash; crop probability band
- **Global, pre-2015:** Copernicus C3S Land Cover (1992&ndash;2022, 300 m) &mdash; cropland classes

Disable with `lulc_filter=False` for local files without GEE access or unsupervised embedding workflows.

---

## Quick Install

```bash
pip install agribound
```

For GPU-accelerated engines and GEE support, install optional extras:

```bash
pip install agribound[gee,delineate-anything]
```

See the [Installation guide](installation.md) for all available extras.

---

## Quickstart

```python
import agribound

gdf = agribound.delineate(
    study_area="area.geojson",
    source="sentinel2",
    year=2024,
    engine="delineate-anything",
    gee_project="my-gee-project",
)
```

The returned `GeoDataFrame` contains field boundary polygons with area, perimeter, and provenance metadata. See the [Quickstart tutorial](user-guide/quickstart.md) for a complete walkthrough.

---

## Key Sections

| Section | Description |
|---|---|
| [Installation](installation.md) | Install agribound and optional dependencies |
| [Quickstart](user-guide/quickstart.md) | 5-minute tutorial covering Python and CLI usage |
| [Satellite Sources](user-guide/satellite-sources.md) | Available imagery sources and compositing options |
| [Engines](user-guide/engines.md) | Comparison of all seven delineation engines |
| [Configuration](user-guide/configuration.md) | Full reference for `AgriboundConfig` |
| [CLI Usage](user-guide/cli.md) | Command-line interface reference |
| [API Reference](api/pipeline.md) | Python API documentation |
| [Contributing](contributing.md) | Developer guide for adding engines and sources |
| [Citation & References](citation.md) | How to cite agribound, funding sources, and disclaimer |

---

## License

Agribound is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
