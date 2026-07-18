#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_noc_power_backends.sh [--force] [--skip-smoke]

Build the external NoC power backends used by TSIM:
  - DSENT 0.91 submodule at external/dsent0.91
  - VNoC/ORION submodule at external/vnoc20
  - TSIM ORION link probe at src/tools/tsim_orion_probe

Environment overrides:
  MAKE_JOBS      parallel make jobs, default: nproc
  CC             C compiler for the ORION probe, default: gcc
EOF
}

FORCE=0
SMOKE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --skip-smoke) SMOKE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAKE_JOBS="${MAKE_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
CC_BIN="${CC:-gcc}"

log() {
  printf '[setup-noc] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

require_cmd git
require_cmd make
require_cmd "${CC_BIN}"

cd "${REPO_DIR}"

log "initializing DSENT and VNoC/ORION submodules"
git submodule update --init external/dsent0.91 external/vnoc20

DSENT_DIR="${REPO_DIR}/external/dsent0.91/OENOC/dsent0.91"
ORION_DIR="${REPO_DIR}/external/vnoc20/orion3"
PROBE="${REPO_DIR}/src/tools/tsim_orion_probe"

if [[ ! -f "${DSENT_DIR}/Makefile" ]]; then
  echo "DSENT Makefile not found under ${DSENT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${ORION_DIR}/Makefile" ]]; then
  echo "ORION Makefile not found under ${ORION_DIR}" >&2
  exit 1
fi

if [[ "${FORCE}" -eq 1 ]]; then
  log "cleaning DSENT and ORION build products"
  make -C "${DSENT_DIR}" clean >/dev/null || true
  make -C "${ORION_DIR}" clean >/dev/null || true
  rm -f "${PROBE}"
fi

if [[ ! -x "${DSENT_DIR}/dsent" || "${FORCE}" -eq 1 ]]; then
  log "building DSENT"
  make -C "${DSENT_DIR}" -j"${MAKE_JOBS}"
else
  log "DSENT already built"
fi

if [[ ! -x "${ORION_DIR}/orion_router" || ! -f "${ORION_DIR}/libpower.a" || "${FORCE}" -eq 1 ]]; then
  log "building VNoC/ORION"
  make -C "${ORION_DIR}" -j"${MAKE_JOBS}"
else
  log "ORION already built"
fi
chmod u+x "${ORION_DIR}/orion_router"

if [[ ! -x "${PROBE}" || "${FORCE}" -eq 1 ]]; then
  log "building TSIM ORION link probe"
  "${CC_BIN}" -no-pie -I "${ORION_DIR}" -DTECHNEW \
    "${REPO_DIR}/src/tools/tsim_orion_probe.c" "${ORION_DIR}/libpower.a" -lm \
    -o "${PROBE}"
else
  log "TSIM ORION link probe already built"
fi

if [[ "${SMOKE}" -eq 1 ]]; then
  log "running NoC backend smoke test"
  PYTHONPATH="${REPO_DIR}/src" python3 - <<'PY'
from tsim_components.noc_power import NoCPowerConfig, describe
for backend in ("tsim_simple", "dsent", "orion"):
    meta = describe(NoCPowerConfig(backend=backend, frequency_hz=1.5e9))
    print(f"{backend}: energy={meta['total_dynamic_energy_j_per_flit']:.6e} scale={meta['scale_vs_tsim_simple']:.6f}")
PY
fi

log "NoC power backends are ready"
