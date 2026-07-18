#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_external_backends.sh [--force] [--skip-thermal] [--skip-noc] [--skip-smoke]

Set up all external TSIM backend dependencies:
  - HotSpot and 3D-ICE thermal simulators
  - DSENT and ORION NoC power models

This script delegates to:
  scripts/setup_thermal_backends.sh
  scripts/setup_noc_power_backends.sh
EOF
}

FORCE=0
SETUP_THERMAL=1
SETUP_NOC=1
SMOKE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --skip-thermal) SETUP_THERMAL=0 ;;
    --skip-noc) SETUP_NOC=0 ;;
    --skip-smoke) SMOKE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

args=()
if [[ "${FORCE}" -eq 1 ]]; then
  args+=(--force)
fi
if [[ "${SMOKE}" -eq 0 ]]; then
  args+=(--skip-smoke)
fi

if [[ "${SETUP_THERMAL}" -eq 1 ]]; then
  "${SCRIPT_DIR}/setup_thermal_backends.sh" "${args[@]}"
fi
if [[ "${SETUP_NOC}" -eq 1 ]]; then
  "${SCRIPT_DIR}/setup_noc_power_backends.sh" "${args[@]}"
fi

printf '[setup-all] external backends are ready\n'
