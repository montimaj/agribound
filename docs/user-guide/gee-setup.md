# GEE Setup

Google Earth Engine (GEE) is required for all satellite sources except `local`, `google-embedding`, and `tessera-embedding`. This guide covers project creation and authentication.

## Prerequisites

1. A Google account.
2. The GEE extra installed: `pip install agribound[gee]`.

## Creating a GEE Project

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.
3. Enable the Earth Engine API for your project:
    - Navigate to **APIs & Services > Library**.
    - Search for "Earth Engine API".
    - Click **Enable**.
4. Register your project for Earth Engine at [code.earthengine.google.com](https://code.earthengine.google.com/) if you have not already.

Note your **project ID** (e.g., `my-gee-project`). You will need it for agribound.

## Authentication Methods

### Interactive Browser Authentication

The simplest approach for local development. Run:

```bash
agribound auth --project my-gee-project
```

This opens a browser window for Google OAuth. After authorizing, credentials are cached locally for future sessions.

Alternatively, authenticate directly with the Earth Engine CLI:

```bash
earthengine authenticate
```

### Service Account Authentication

For CI pipelines, servers, or headless environments, use a GEE service account:

1. In the [Google Cloud Console](https://console.cloud.google.com/), go to **IAM & Admin > Service Accounts**.
2. Create a new service account.
3. Grant it the **Earth Engine Resource Writer** role (or a custom role with `earthengine.computations.create`).
4. Create a JSON key file and download it.
5. Register the service account email at [code.earthengine.google.com](https://code.earthengine.google.com/).

Then authenticate:

```bash
agribound auth --project my-gee-project --service-account-key /path/to/key.json
```

Or in Python:

```python
from agribound.auth import setup_gee

setup_gee(
    project="my-gee-project",
    service_account_key="/path/to/key.json",
)
```

### Environment Variable

You can set the `GEE_PROJECT` environment variable to avoid passing `--gee-project` on every command:

```bash
export GEE_PROJECT=my-gee-project
```

The `setup_gee()` function and CLI will read this variable automatically when no project is specified explicitly.

## Using GEE in the Pipeline

Once authenticated, pass the project ID to any agribound operation:

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

Or via CLI:

```bash
agribound delineate \
    --study-area area.geojson \
    --source sentinel2 \
    --gee-project my-gee-project
```

## Troubleshooting

| Problem | Solution |
|---|---|
| `earthengine-api` not installed | `pip install agribound[gee]` |
| "No project ID" error | Pass `--gee-project` or set `GEE_PROJECT` env var |
| Authentication expired | Re-run `agribound auth --project <id>` |
| Service account not authorized | Register the SA email at code.earthengine.google.com |
| API not enabled | Enable Earth Engine API in Google Cloud Console |
