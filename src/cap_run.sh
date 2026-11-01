#!/bin/sh
# Cap a running training job at N epochs (decision 2026-10-30: df_e08_res_dinomac runs ~13 h/epoch).
# Waits until epoch N has started (so epoch N-1's val AUC and best.pt are saved), stops the job with
# SIGINT, then evaluates best.pt with train.py --eval-only (test metrics + df_e09 robustness).
# Usage: sh cap_run.sh <exp> <N> <train.py args of the run...>
cd "$(dirname "$0")"
exp=$1; n=$2; shift 2
L=../experiments/$exp/run.log
until grep -q "epoch $n step 0/" $L || grep -q "RUN COMPLETE\|early stop" $L; do sleep 60; done
if grep -q "RUN COMPLETE\|early stop" $L; then echo "$(date '+%F %T') $exp finished on its own"; exit 0; fi
pid=$(pgrep -f "train.py --exp $exp ")
echo "$(date '+%F %T') stopping $exp (pid $pid) after epoch $((n - 1))"
kill -INT $pid
for i in $(seq 1 12); do kill -0 $pid 2>/dev/null || break; sleep 10; done
kill -0 $pid 2>/dev/null && { echo "still alive -> SIGTERM"; kill -TERM $pid; sleep 30; }
echo "$(date '+%F %T') eval-only start"
../.venv/bin/python train.py --exp $exp "$@" --eval-only > ../experiments/$exp/eval.log 2>&1
echo "$(date '+%F %T') eval-only end exit=$?"
