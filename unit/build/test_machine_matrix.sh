#!/bin/bash

set -uo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 <machine-name>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MACHINE_NAME="$1"
MACHINE_TITLE="${MACHINE_NAME^}"
MACHINE_DIR="${SCRIPT_DIR}/machines/${MACHINE_NAME}"
MATRIX_BUILD_ROOT="${YAKL_MACHINE_MATRIX_BUILD_ROOT:-${SCRIPT_DIR}/kokkos-matrix/${MACHINE_NAME}}"
FAILED_ENVIRONMENTS=()

shopt -s nullglob
ENVIRONMENT_FILES=("${MACHINE_DIR}"/"${MACHINE_NAME}"_*.env)
shopt -u nullglob

if (( ${#ENVIRONMENT_FILES[@]} == 0 )); then
  echo "ERROR: No ${MACHINE_NAME} environments found under ${MACHINE_DIR}" >&2
  exit 1
fi

echo "${MACHINE_TITLE} test matrix"
echo "  Environments: ${#ENVIRONMENT_FILES[@]}"
echo "  Kokkos versions: 4.7.00 5.2.0"
echo "  Index widths: 64 32"
echo "  Build root: ${MATRIX_BUILD_ROOT}"

for ENVIRONMENT_FILE in "${ENVIRONMENT_FILES[@]}"; do
  ENVIRONMENT_NAME="$(basename "${ENVIRONMENT_FILE}" .env)"
  ENVIRONMENT_BUILD_ROOT="${MATRIX_BUILD_ROOT}/${ENVIRONMENT_NAME}"

  echo
  echo "======================================================================"
  echo "Testing ${ENVIRONMENT_NAME} with both Kokkos versions"
  echo "======================================================================"

  if (
    source "${ENVIRONMENT_FILE}" || exit 1
    export YAKL_KOKKOS_MATRIX_BUILD_ROOT="${ENVIRONMENT_BUILD_ROOT}"
    "${SCRIPT_DIR}/test_kokkos_versions.sh"
  ); then
    echo "${ENVIRONMENT_NAME}: PASS"
  else
    echo "${ENVIRONMENT_NAME}: FAIL" >&2
    FAILED_ENVIRONMENTS+=("${ENVIRONMENT_NAME}")
  fi
done

echo
echo "======================================================================"
if (( ${#FAILED_ENVIRONMENTS[@]} != 0 )); then
  echo "${MACHINE_TITLE} matrix failed for: ${FAILED_ENVIRONMENTS[*]}" >&2
  exit 1
fi

echo "All ${MACHINE_NAME} environments passed with both Kokkos versions and index widths"
