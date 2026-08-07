#!/bin/bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAKL_MACHINE_MATRIX_BUILD_ROOT="${YAKL_THATCHROOF_MATRIX_BUILD_ROOT:-${SCRIPT_DIR}/kokkos-matrix/thatchroof}" \
  exec "${SCRIPT_DIR}/test_machine_matrix.sh" thatchroof
