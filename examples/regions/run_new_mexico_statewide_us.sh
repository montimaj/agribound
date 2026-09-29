#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_new_mexico_statewide_us.sh -- New Mexico statewide, USA
#
# Runs examples/run_region_delineation.sh with the defaults of new_mexico_statewide_us.yaml
# (years, sources, engines, tile size, halo, reference). Extra arguments are
# passed through, e.g.
#   examples/regions/run_new_mexico_statewide_us.sh --test --gee-project ID              # small test box, here
#   examples/regions/run_new_mexico_statewide_us.sh --mode slurm --profile delta --gee-project ID \
#       --gee-service-account-key ~/keys/ee.json --dry-run               # print the Slurm jobs
# See examples/run_region_delineation.sh --help and examples/hpc/README.md.
# ------------------------------------------------------------------------------
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${here}/../run_region_delineation.sh" --region-file "${here}/new_mexico_statewide_us.yaml" "$@"
