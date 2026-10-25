#!/bin/bash
# KV-cache test, unattended main run (NOT run yet). Resumable: each step skips finished questions.
# Order: arm A first (reference), then B, C, D, then the noise re-run; optional arms last.
# Usage:  ./queue.sh            # 6 documents per context
#         ./queue.sh 4          # time-rule fallback: first 4 documents per context (PREREG.md)
# Run under: caffeinate -dimsu ./queue.sh   (the Mac hibernated mid-run in the Bonsai 2 test)
cd "$(dirname "$0")/harness"
PY=/opt/homebrew/bin/python3.12
LOG=../results/queue.log
mkdir -p ../results
DOCS="${1:-6}"
say() { echo "$(date '+%m-%d %H:%M') $*" | tee -a $LOG; }
step() {
  local arm=$1 run=$2; shift 2
  for try in 1 2 3; do
    say "start $run $arm (try $try) docs=$DOCS $*"
    $PY run.py --model $MODEL --arm $arm --set public --run $run --docs $DOCS "$@" 2>&1 | tee -a ../results/${MODEL}_${run}_$arm.log
    if $PY - "$arm" "$run" "$DOCS" "$@" <<'PYEOF'
import sys, json, os
sys.path.insert(0, ".")
import run, config as cfg
arm, runname, docs = sys.argv[1], sys.argv[2], int(sys.argv[3])
ctxs = cfg.NOISE_CONTEXTS if runname == "noise" else cfg.CONTEXTS
sel = run.select_docs(run.load_set("public")[0], ctxs, docs)
want = {q["qid"] for d in sel for q in d["questions"]}
path = os.path.join(cfg.RESULTS_DIR, os.environ["MODEL"], "public", runname, arm + ".jsonl")
done = {json.loads(l)["qid"] for l in open(path)} if os.path.exists(path) else set()
sys.exit(0 if want <= done else 1)
PYEOF
    then say "done $run $arm"; return 0; fi
    sleep 30
  done
  say "FAILED $run $arm after 3 tries"; return 1
}
export MODEL=glimmer
step A main
step B main
step C main
step D main
step A noise
export MODEL=gemma4
step A main
step B main
# optional arms (author decides in PREREG.md before freezing); uncomment if chosen
# step E main
# step P main
$PY analyze.py --model glimmer > ../results/report_glimmer.txt 2>&1
$PY analyze.py --model gemma4 > ../results/report_gemma4.txt 2>&1
say "ALL DONE - reports in results/report_glimmer.txt and results/report_gemma4.txt"
