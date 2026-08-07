#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAKL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MATRIX_BUILD_ROOT="${YAKL_KOKKOS_MATRIX_BUILD_ROOT:-${SCRIPT_DIR}/kokkos-matrix}"
VERSIONS=(4.7.00 5.2.0)
FAILED_VERSIONS=()

for VERSION in "${VERSIONS[@]}"; do
  KOKKOS_SOURCE="${YAKL_ROOT}/external/kokkos-${VERSION}"
  if [[ ! -f "${KOKKOS_SOURCE}/CMakeLists.txt" ]]; then
    echo "ERROR: Missing Kokkos ${VERSION} submodule at ${KOKKOS_SOURCE}" >&2
    echo "Run: git submodule update --init --checkout external/kokkos-${VERSION}" >&2
    exit 1
  fi
done

for VERSION in "${VERSIONS[@]}"; do
  KOKKOS_SOURCE="${YAKL_ROOT}/external/kokkos-${VERSION}"
  BUILD_DIR="${MATRIX_BUILD_ROOT}/${VERSION}"
  mkdir -p "${BUILD_DIR}"

  echo
  echo "======================================================================"
  echo "Building and testing YAKL with Kokkos ${VERSION}"
  echo "======================================================================"

  if (
    cd "${BUILD_DIR}" || exit 1
    if [[ "${YAKL_BACKEND:-}" == "Kokkos_ENABLE_CUDA" ]]; then
      VERSION_CUDA_ROOT=""
      case "${VERSION}" in
        4.7.00) VERSION_CUDA_ROOT="${YAKL_KOKKOS_4_7_CUDA_ROOT:-}" ;;
        5.2.0)  VERSION_CUDA_ROOT="${YAKL_KOKKOS_5_2_CUDA_ROOT:-}" ;;
      esac
      if [[ -n "${VERSION_CUDA_ROOT}" ]]; then
        export CUDAToolkit_ROOT="${VERSION_CUDA_ROOT}"
        export CUDA_ROOT="${VERSION_CUDA_ROOT}"
        export PATH="${VERSION_CUDA_ROOT}/bin:${PATH}"
        echo "Using CUDA Toolkit at ${VERSION_CUDA_ROOT}"
      fi
    fi
    Kokkos_HOME="${KOKKOS_SOURCE}" "${SCRIPT_DIR}/cmakescript.sh" || exit 1
    BUILD_COMMAND=(cmake --build . --parallel)
    if [[ -n "${YAKL_BUILD_JOBS:-}" ]]; then BUILD_COMMAND+=("${YAKL_BUILD_JOBS}"); fi
    "${BUILD_COMMAND[@]}" || exit 1
    ctest -V || exit 1
  ); then
    echo "Kokkos ${VERSION}: PASS"
  else
    echo "Kokkos ${VERSION}: FAIL" >&2
    FAILED_VERSIONS+=("${VERSION}")
  fi
done

if (( ${#FAILED_VERSIONS[@]} != 0 )); then
  echo "Failed Kokkos versions: ${FAILED_VERSIONS[*]}" >&2
  exit 1
fi

echo "All Kokkos versions passed: ${VERSIONS[*]}"
