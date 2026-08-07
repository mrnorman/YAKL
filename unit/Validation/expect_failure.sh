#!/usr/bin/env bash
"$@"
status=$?
if [ "$status" -eq 0 ]; then
  echo "Expected validation command '$*' to fail, but it succeeded"
  exit 1
fi
exit 0
