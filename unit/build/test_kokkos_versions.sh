#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAKL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MATRIX_BUILD_ROOT="${YAKL_KOKKOS_MATRIX_BUILD_ROOT:-${SCRIPT_DIR}/kokkos-matrix}"
VERSIONS=(4.7.00 5.2.0)
INDEX_WIDTHS=(64 32)
FAILED_CONFIGURATIONS=()

for VERSION in "${VERSIONS[@]}"; do
  KOKKOS_SOURCE="${YAKL_ROOT}/external/kokkos-${VERSION}"
  if [[ ! -f "${KOKKOS_SOURCE}/CMakeLists.txt" ]]; then
    echo "ERROR: Missing Kokkos ${VERSION} submodule at ${KOKKOS_SOURCE}" >&2
    echo "Run: git submodule update --init --checkout external/kokkos-${VERSION}" >&2
    exit 1
  fi
done

for VERSION in "${VERSIONS[@]}"; do
  for INDEX_BITS in "${INDEX_WIDTHS[@]}"; do
    KOKKOS_SOURCE="${YAKL_ROOT}/external/kokkos-${VERSION}"
    BUILD_DIR="${MATRIX_BUILD_ROOT}/${VERSION}/index${INDEX_BITS}"
    mkdir -p "${BUILD_DIR}"

    echo
    echo "======================================================================"
    echo "Building and testing YAKL with Kokkos ${VERSION} and ${INDEX_BITS}-bit indices"
    echo "======================================================================"

    if (
      cd "${BUILD_DIR}" || exit 1
      export YAKL_INDEX_BITS="${INDEX_BITS}"
      if [[ "${INDEX_BITS}" == "32" ]]; then export YAKL_UNIT_LARGE_MEMORY=OFF; fi
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
      echo "Kokkos ${VERSION}, ${INDEX_BITS}-bit indices: PASS"
    else
      echo "Kokkos ${VERSION}, ${INDEX_BITS}-bit indices: FAIL" >&2
      FAILED_CONFIGURATIONS+=("${VERSION}/index${INDEX_BITS}")
    fi
  done
done

if (( ${#FAILED_CONFIGURATIONS[@]} != 0 )); then
  echo "Failed configurations: ${FAILED_CONFIGURATIONS[*]}" >&2
  exit 1
fi

echo "All Kokkos/index-width configurations passed: Kokkos ${VERSIONS[*]}; indices ${INDEX_WIDTHS[*]}"
