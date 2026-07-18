#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/setup_thermal_backends.sh [--force] [--skip-hotspot] [--skip-3dice] [--skip-smoke]

Install/build external thermal simulators used by TSIM:
  - HotSpot under external/hotspot-7.0
  - 3D-ICE under external/3d-ice-src

The script downloads upstream GitHub archives unless the matching local archive
already exists under external/.

Environment overrides:
  MAKE_JOBS      parallel make jobs, default: nproc
  HOTSPOT_URL    default: https://github.com/uvahotspot/HotSpot/archive/9f92256.tar.gz
  THREEDICE_URL  default: https://github.com/esl-epfl/3d-ice/archive/refs/heads/master.tar.gz
EOF
}

FORCE=0
INSTALL_HOTSPOT=1
INSTALL_THREEDICE=1
SMOKE=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --skip-hotspot) INSTALL_HOTSPOT=0 ;;
    --skip-3dice) INSTALL_THREEDICE=0 ;;
    --skip-smoke) SMOKE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${REPO_DIR}/external"
MAKE_JOBS="${MAKE_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)}"
HOTSPOT_URL="${HOTSPOT_URL:-https://github.com/uvahotspot/HotSpot/archive/9f92256.tar.gz}"
THREEDICE_URL="${THREEDICE_URL:-https://github.com/esl-epfl/3d-ice/archive/refs/heads/master.tar.gz}"
HOTSPOT_ARCHIVE="${EXTERNAL_DIR}/hotspot-v7.0.tar.gz"
THREEDICE_ARCHIVE="${EXTERNAL_DIR}/3d-ice.tar.gz"
HOTSPOT_DIR="${EXTERNAL_DIR}/hotspot-7.0"
THREEDICE_DIR="${EXTERNAL_DIR}/3d-ice-src"

log() {
  printf '[setup-thermal] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

download_archive() {
  local url="$1"
  local out="$2"
  if [[ -f "${out}" && "${FORCE}" -eq 0 ]]; then
    log "using existing archive ${out}"
    return
  fi
  log "downloading ${url}"
  curl -L --fail --retry 3 --output "${out}.tmp" "${url}"
  mv "${out}.tmp" "${out}"
}

extract_single_root_archive() {
  local archive="$1"
  local dest="$2"
  local tmp
  tmp="$(mktemp -d "${EXTERNAL_DIR}/extract.XXXXXX")"
  tar -xzf "${archive}" -C "${tmp}"
  local roots=("${tmp}"/*)
  if [[ "${#roots[@]}" -ne 1 || ! -d "${roots[0]}" ]]; then
    echo "archive ${archive} did not contain one top-level directory" >&2
    rm -rf "${tmp}"
    exit 1
  fi
  if [[ -e "${dest}" ]]; then
    if [[ "${FORCE}" -ne 1 ]]; then
      echo "${dest} already exists; use --force to replace it" >&2
      rm -rf "${tmp}"
      exit 1
    fi
    rm -rf "${dest}"
  fi
  mv "${roots[0]}" "${dest}"
  rmdir "${tmp}"
}

require_cmd curl
require_cmd make
require_cmd tar
require_cmd python3

mkdir -p "${EXTERNAL_DIR}"

if [[ "${INSTALL_HOTSPOT}" -eq 1 ]]; then
  if [[ ! -x "${HOTSPOT_DIR}/hotspot" || "${FORCE}" -eq 1 ]]; then
    download_archive "${HOTSPOT_URL}" "${HOTSPOT_ARCHIVE}"
    extract_single_root_archive "${HOTSPOT_ARCHIVE}" "${HOTSPOT_DIR}"
    log "building HotSpot"
    make -C "${HOTSPOT_DIR}" -j"${MAKE_JOBS}"
  else
    log "HotSpot already built"
  fi
fi

if [[ "${INSTALL_THREEDICE}" -eq 1 ]]; then
  if [[ ! -x "${THREEDICE_DIR}/bin/3D-ICE-Emulator" || "${FORCE}" -eq 1 ]]; then
    download_archive "${THREEDICE_URL}" "${THREEDICE_ARCHIVE}"
    extract_single_root_archive "${THREEDICE_ARCHIVE}" "${THREEDICE_DIR}"
    log "building 3D-ICE"
    make -C "${THREEDICE_DIR}" -j"${MAKE_JOBS}"
  else
    log "3D-ICE already built"
  fi
fi

if [[ "${SMOKE}" -eq 1 ]]; then
  if [[ "${INSTALL_HOTSPOT}" -eq 1 ]]; then
    test -x "${HOTSPOT_DIR}/hotspot"
    log "HotSpot binary: ${HOTSPOT_DIR}/hotspot"
  fi
  if [[ "${INSTALL_THREEDICE}" -eq 1 ]]; then
    test -x "${THREEDICE_DIR}/bin/3D-ICE-Emulator"
    log "3D-ICE binary: ${THREEDICE_DIR}/bin/3D-ICE-Emulator"
  fi
fi

log "thermal backends are ready"
