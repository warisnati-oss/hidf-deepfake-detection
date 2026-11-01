#!/bin/sh
# Phase 4 training queue: runs every remaining model one after another on the GPU.
# Waits for the M1 baseline that is already running, then starts the next run only if
# experiments/STOP_QUEUE does not exist (touch it to stop after the current run).
# A failed run is logged and the queue moves on. Each run writes experiments/<exp>/run.log.
cd "$(dirname "$0")"
EXP=../experiments
PY=../.venv/bin/python
LOG=$EXP/queue.log

# wait for any training run that is already on the GPU
while pgrep -f "train.py --exp" >/dev/null; do sleep 60; done

run() {
  exp=$1; shift
  if [ -f $EXP/STOP_QUEUE ]; then echo "$(date '+%F %T') STOP_QUEUE found, not starting $exp" >> $LOG; exit 0; fi
  if [ -f $EXP/$exp/results.json ]; then echo "$(date '+%F %T') skip $exp (already done)" >> $LOG; return; fi
  mkdir -p $EXP/$exp
  echo "$(date '+%F %T') start $exp $*" >> $LOG
  $PY train.py --exp $exp "$@" > $EXP/$exp/run.log 2>&1
  rc=$?  # capture before $(date) resets it
  echo "$(date '+%F %T') end $exp exit=$rc" >> $LOG
}

# baselines (full image, 224 px, augmentation on)
run df_e05_gend          --model gend --backbone dinov3_l16 --epochs 5 --batch 32
run df_e07_crop_effnet   --model effnet_b4 --view crop --epochs 8 --batch 32
run df_e05b_dinov3_lora  --model dinov3_lora --warmup 0.3 --size-range 224 320 --epochs 5 --batch 16
# df_e07: P3 face crop
run df_e07_crop_gend     --model gend --backbone dinov3_l16 --view crop --epochs 5 --batch 32
run df_e07_crop_dinomac  --model dinov3_lora --warmup 0.3 --view crop --size-range 224 320 --epochs 5 --batch 16
# df_e06: augmentation off
run df_e06_noaug_effnet  --model effnet_b4 --no-augment --epochs 8 --batch 32
run df_e06_noaug_gend    --model gend --backbone dinov3_l16 --no-augment --epochs 5 --batch 32
run df_e06_noaug_dinomac --model dinov3_lora --warmup 0.3 --no-augment --size-range 224 320 --epochs 5 --batch 16
# df_e08: higher resolution
run df_e08_res384_effnet --model effnet_b4 --size 384 --epochs 8 --batch 16
run df_e08_res384_gend   --model gend --backbone dinov3_l16 --size 384 --epochs 5 --batch 16
run df_e08_res_dinomac   --model dinov3_lora --warmup 0.3 --size 384 --size-range 320 448 --epochs 5 --batch 8
echo "$(date '+%F %T') queue finished" >> $LOG
