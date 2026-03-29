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

### Example Results

**Supervised: DINOv3 + SAM2 on NAIP (Eastern Lea County, New Mexico, USA)** — Fine-tuned on NMOSE reference boundaries, LULC-filtered (NLCD), SAM2-refined on 1 m NAIP. Blue = predicted, yellow = reference. The study area bbox extends slightly into Texas, so fields along the NM–TX border are also delineated.

![DINOv3 + SAM2 on NAIP](https://raw.githubusercontent.com/montimaj/agribound/main/assets/NM_example.png)

**Unsupervised: TESSERA + LULC Filter + SAM2 (Pampas, Argentina)** — No training, no reference data. TESSERA embedding clustering + LULC crop filter (Dynamic World) + SAM2 on Sentinel-2.

![TESSERA + LULC + SAM2](https://raw.githubusercontent.com/montimaj/agribound/main/assets/Pampas_example.png)

*Note: The satellite basemap shown in these screenshots may not correspond to the same acquisition date as the imagery used for delineation.*

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
