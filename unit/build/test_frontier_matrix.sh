#!/bin/bash

set -u

ulimit -c 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAKL_MACHINE_MATRIX_BUILD_ROOT="${YAKL_FRONTIER_MATRIX_BUILD_ROOT:-${SCRIPT_DIR}/kokkos-matrix/frontier}" \
  exec "${SCRIPT_DIR}/test_machine_matrix.sh" frontier
