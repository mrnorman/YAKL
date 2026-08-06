#!/usr/bin/env bash
"$1" "$2"
status=$?
if [ "$status" -eq 0 ]; then
  echo "Expected validation scenario '$2' to fail, but it succeeded"
  exit 1
fi
exit 0
