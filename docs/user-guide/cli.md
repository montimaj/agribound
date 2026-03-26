# CLI Usage

Agribound provides a command-line interface built with Click. The entry point is the `agribound` command.

## Global Options

```
agribound [OPTIONS] COMMAND [ARGS]...
```

| Option | Description |
|---|---|
| `--version` | Show the agribound version and exit. |
| `-v`, `--verbose` | Enable verbose (DEBUG-level) logging. |
| `--help` | Show help message and exit. |

## Commands

### delineate

Run the full field boundary delineation pipeline.

```bash
agribound delineate [OPTIONS]
```

**Options:**

| Option | Type | Default | Description |
|---|---|---|---|
| `--study-area` | TEXT | (required) | Path to GeoJSON/Shapefile or GEE asset ID. |
| `--source` | TEXT | `sentinel2` | Satellite source. |
| `--year` | INT | `2024` | Target year. |
| `--engine` | TEXT | `delineate-anything` | Delineation engine. |
| `--output`, `-o` | TEXT | Auto-generated | Output file path. Defaults to `fields_{source}_{year}.{format}`. |
| `--output-format` | CHOICE | `gpkg` | Output format: `gpkg`, `geojson`, `parquet`. |
| `--gee-project` | TEXT | None | GEE project ID. |
| `--export-method` | CHOICE | `local` | GEE export: `local`, `gdrive`, `gcs`. |
| `--gcs-bucket` | TEXT | None | GCS bucket name (required for `gcs` export). |
| `--composite-method` | CHOICE | `median` | Compositing: `median`, `greenest`, `max_ndvi`. |
| `--date-range` | TEXT TEXT | None | Start and end dates (YYYY-MM-DD). |
| `--cloud-cover-max` | INT | `20` | Maximum cloud cover percentage. |
| `--local-tif` | TEXT | None | Path to local GeoTIFF (for `source=local`). |
| `--min-area` | FLOAT | `2500.0` | Minimum field area in square meters. |
| `--simplify` | FLOAT | `2.0` | Simplification tolerance in pixels. |
| `--device` | CHOICE | `auto` | Compute device: `auto`, `cuda`, `cpu`, `mps`. |
| `--n-workers` | INT | `4` | Number of parallel workers. |
| `--reference` | TEXT | None | Reference boundaries for evaluation or fine-tuning. |
| `--fine-tune` | FLAG | False | Fine-tune engine on reference boundaries. |
| `--config` | TEXT | None | YAML config file (overrides other options). |

**Examples:**

Basic delineation with Sentinel-2:

```bash
agribound delineate \
    --study-area area.geojson \
    --source sentinel2 \
    --year 2024 \
    --engine delineate-anything \
    --gee-project my-project \
    --output fields.gpkg
```

Using a local GeoTIFF:

```bash
agribound delineate \
    --study-area area.geojson \
    --source local \
    --engine ftw \
    --local-tif composite.tif \
    --output fields.geojson \
    --output-format geojson
```

Restrict to growing season with stricter cloud filtering:

```bash
agribound delineate \
    --study-area area.geojson \
    --source sentinel2 \
    --year 2024 \
    --engine delineate-anything \
    --gee-project my-project \
    --composite-method greenest \
    --date-range 2024-04-01 2024-09-30 \
    --cloud-cover-max 10
```

Using a YAML config file:

```bash
agribound delineate --config run_config.yaml
```

With evaluation against reference boundaries:

```bash
agribound delineate \
    --study-area area.geojson \
    --source sentinel2 \
    --year 2024 \
    --engine delineate-anything \
    --gee-project my-project \
    --reference ground_truth.gpkg
```

With fine-tuning:

```bash
agribound delineate \
    --study-area area.geojson \
    --source sentinel2 \
    --year 2024 \
    --engine ftw \
    --gee-project my-project \
    --reference training_fields.gpkg \
    --fine-tune
```

### list-engines

List all available delineation engines with their approach and GPU requirements.

```bash
agribound list-engines
```

### list-sources

List all available satellite sources with resolution and access mode.

```bash
agribound list-sources
```

### auth

Authenticate with Google Earth Engine.

```bash
agribound auth [OPTIONS]
```

| Option | Type | Description |
|---|---|---|
| `--project` | TEXT | GEE project ID. |
| `--service-account-key` | TEXT | Path to a service account JSON key file. |

**Examples:**

Interactive browser authentication:

```bash
agribound auth --project my-gee-project
```

Service account authentication (CI/server):

```bash
agribound auth --project my-gee-project --service-account-key credentials.json
```

See [GEE Setup](gee-setup.md) for detailed authentication instructions.
