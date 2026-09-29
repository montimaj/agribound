#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# run_mato_grosso_br.sh -- Central Mato Grosso soybean belt, Brazil
#
# Runs examples/run_region_delineation.sh with the defaults of mato_grosso_br.yaml
# (years, sources, engines, tile size, halo, reference). Extra arguments are
# passed through, e.g.
#   examples/regions/run_mato_grosso_br.sh --test --gee-project ID              # small test box, here
#   examples/regions/run_mato_grosso_br.sh --mode slurm --profile delta --gee-project ID \
#       --gee-service-account-key ~/keys/ee.json --dry-run               # print the Slurm jobs
# See examples/run_region_delineation.sh --help and examples/hpc/README.md.
# ------------------------------------------------------------------------------
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${here}/../run_region_delineation.sh" --region-file "${here}/mato_grosso_br.yaml" "$@"
