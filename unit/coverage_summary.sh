#!/bin/bash

set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <source-root> <build-root> <gcovr> <gcov>" >&2
  exit 2
fi

source_root=$(cd "$1" && pwd)
build_root=$(cd "$2" && pwd)
gcovr_executable=$3
gcov_executable=$4
coverage_work_dir=$(mktemp -d "${TMPDIR:-/tmp}/yakl-gcov.XXXXXX")
trap 'rm -rf "${coverage_work_dir}"' EXIT

# gcovr 5 cannot infer a valid working directory for NVCC-generated coverage objects. Generate each
# .gcov set separately to prevent duplicate header names from overwriting one another, then merge them.
while IFS= read -r -d '' coverage_file; do
  coverage_object_dir=$(mktemp -d "${coverage_work_dir}/object.XXXXXX")
  (cd "${coverage_object_dir}" && "${gcov_executable}" -pb "${coverage_file}" >/dev/null 2>&1)
done < <(find "${build_root}" -name '*.gcda' -print0)

echo
echo "YAKL src coverage"
cd "${source_root}"
"${gcovr_executable}" --root "${source_root}" \
                      --filter 'src/' \
                      --print-summary \
                      --use-gcov-files \
                      "${coverage_work_dir}"
