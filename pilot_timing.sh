#!/bin/bash
cd /Users/admin/dev/company/ai-claims/tests/kvcache/harness
PY=/opt/homebrew/bin/python3.12
for step in "B 16k" "D 16k" "C 16k,48k"; do
  set -- $step
  echo "=== arm $1 contexts $2 $(date -u +%FT%TZ)"
  $PY run.py --model glimmer --arm $1 --set calibration --run calibration --contexts $2
done
echo "=== done $(date -u +%FT%TZ)"
