# Google Earth Engine Setup

Earth Engine is used for:

- the imagery composites of `landsat`, `sentinel2`, `hls`, `naip`, `spot` and
  `spot-pan`;
- `google-embedding` with the default `google_embedding_backend="gee"`;
- the **LULC crop filter**, which is on by default and reads its datasets
  from Earth Engine for every source, including `local`, `usgs-naip-plus` and
  `tessera-embedding`;
- a study area given as a GEE vector asset ID.

A run needs no Earth Engine access only when all of these are avoided, for
example `source="local"` (or `usgs-naip-plus`, `tessera-embedding`, or
`google-embedding` with `google_embedding_backend="source_coop"`) with
`lulc_filter=False` and a local study-area file.
`AgriboundConfig.requires_gee()` reports whether the source or the LULC
filter needs Earth Engine (it does not look at the study area).

## One-time setup

1. Install the Earth Engine client: `pip install "agribound[gee]"` (included
   in `agribound[all]`, `agribound[all-gfm]` and `agribound[embedding]`).
2. Create or choose a Google Cloud project registered for Earth Engine
   (<https://console.cloud.google.com/>), with the Earth Engine API enabled.
3. Authenticate once:

    ```bash
    earthengine authenticate
    # or Application Default Credentials:
    gcloud auth application-default login \
        --scopes=https://www.googleapis.com/auth/earthengine,https://www.googleapis.com/auth/cloud-platform
    ```

4. Check it with agribound:

    ```bash
    agribound auth --project my-gee-project
    ```

For the GEE imagery sources (`landsat`, `sentinel2`, `hls`, `naip`, `spot`,
`spot-pan`) the project is resolved when the configuration is created, from
`gee_project` (`--gee-project`), then the `GEE_PROJECT` environment variable,
then `gcloud config get-value project`, then the `project_id` of the
credentials file: `gee_service_account_key` (`--gee-service-account-key`),
else `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`, else `GOOGLE_APPLICATION_CREDENTIALS`
(only the first of these that is set is read; user credentials from
`gcloud auth application-default login` have no `project_id`). Without any of
these the configuration raises `ValueError`. `setup_gee` resolves a missing
project in the same order, for example for the LULC filter on a non-GEE
source or for `agribound auth --service-account-key PATH` without
`--project`. An earlier source wins over the key: a batch job whose
environment sets `GEE_PROJECT` or has a gcloud project uses that project, not
the key's `project_id`.

`agribound tiles gee-project` prints the project this lookup finds, without
contacting Earth Engine. It takes `--project`, `--service-account-key`,
`--sources` and `--no-lulc-filter`, or `--config BASE.yaml`. The HPC and
region scripts call it before they run or submit anything (see
[HPC](hpc.md#earth-engine-project)).

## How agribound authenticates

Every Earth Engine call site goes through `agribound.auth.ensure_gee(config)`,
which calls `setup_gee` with the configuration's `gee_project`,
`gee_service_account_key`, `gee_high_volume` and `gee_workload_tag`.
Credentials are tried in this order:

1. the service-account key in `gee_service_account_key`
   (`--gee-service-account-key`);
2. the key named by `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY`;
3. stored credentials from `earthengine authenticate`;
4. Application Default Credentials (`google.auth.default` with the Earth
   Engine scopes; honours `GOOGLE_APPLICATION_CREDENTIALS`);
5. interactive `ee.Authenticate()`, **only in interactive sessions**. Inside a
   Slurm job, without a TTY, or with `interactive=False`, agribound raises a
   `RuntimeError` that lists the options above instead of waiting for a
   browser.

Initialisation is idempotent: calling it again with the same project, key and
endpoint does not re-initialise the client.

## Service accounts (CI, HPC)

1. Create a service account in the Earth Engine project and give it
   `roles/serviceusage.serviceUsageConsumer` and an Earth Engine role
   (`roles/earthengine.writer` if you use Drive/GCS exports).
2. Download a JSON key into storage only you can read (`chmod 600`).
3. Pass it with `gee_service_account_key` / `--gee-service-account-key`, or set
   `AGRIBOUND_GEE_SERVICE_ACCOUNT_KEY=/path/key.json`.
4. Test it: `agribound auth --project my-gee-project --service-account-key /path/key.json`.

## Request tuning

- `gee_max_requests` (default 8) caps the concurrent download requests of one
  process. Earth Engine allows 40 concurrent requests per project by default,
  so the sum over concurrent jobs should stay within that; a WARNING is logged
  when a single configuration asks for more than 40.
- `gee_high_volume=True` uses the high-volume endpoint, meant for many
  parallel small requests.
- `gee_workload_tag` labels the run's EECU usage for accounting (1-63
  characters, letters or digits at both ends).
- When Earth Engine reports that a project is in "restricted mode" (over its
  quota), agribound halves a download's concurrency.

Quota tiers, restricted mode and throttling for large runs are described in
[HPC and large areas](hpc.md#earth-engine-quotas-and-throttling).

## Restricted SPOT access

`AIRBUS/SPOT6_7` (`spot`, `spot-pan`) is not in the public catalogue; access
is limited to select Earth Engine users. In agribound it is for internal DRI
use; external users who need SPOT-based field boundaries should contact the
package author.

## References

- Gorelick, N., et al. (2017). Google Earth Engine: Planetary-scale geospatial
  analysis for everyone. *Remote Sensing of Environment* 202, 18-27.
  <https://doi.org/10.1016/j.rse.2017.06.031>
- Earth Engine Python installation guide:
  <https://developers.google.com/earth-engine/guides/python_install>
