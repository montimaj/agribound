# Agribound

**Unified agricultural field boundary delineation toolkit** combining satellite foundation models, embeddings, and global training data.

Agribound provides a single interface to multiple delineation engines and satellite sources, handling the full pipeline from satellite composite generation through post-processing and export. It supports Google Earth Engine-based imagery (Landsat, Sentinel-2, HLS, NAIP, SPOT), local GeoTIFFs, and pre-computed embedding datasets (Google Satellite Embedding, TESSERA).

The pipeline runs: **composite building** &rarr; **optional fine-tuning** &rarr; **delineation engine** &rarr; **optional SAM2 boundary refinement** &rarr; **post-processing** (smooth, simplify, filter) &rarr; **export**. When running ensembles, each engine's output is independently refined by SAM2 before vote-based merging.

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
| [Engines](user-guide/engines.md) | Comparison of all six delineation engines |
| [Configuration](user-guide/configuration.md) | Full reference for `AgriboundConfig` |
| [CLI Usage](user-guide/cli.md) | Command-line interface reference |
| [API Reference](api/pipeline.md) | Python API documentation |
| [Contributing](contributing.md) | Developer guide for adding engines and sources |
| [Citation & References](citation.md) | How to cite agribound, funding sources, and disclaimer |

---

## License

Agribound is released under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).
