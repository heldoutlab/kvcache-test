#!/bin/bash
cd /Users/admin/dev/company/ai-claims/tests/kvcache/harness
PY=/opt/homebrew/bin/python3.12
echo "=== gemma4 A calibration $(date -u +%FT%TZ)"
$PY run.py --model gemma4 --arm A --set calibration --run calibration
echo "=== gemma4 B 16k,48k $(date -u +%FT%TZ)"
$PY run.py --model gemma4 --arm B --set calibration --run calibration --contexts 16k,48k
echo "=== done $(date -u +%FT%TZ)"
