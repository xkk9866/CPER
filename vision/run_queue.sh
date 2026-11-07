#!/bin/bash
# Full experiment queue: two concurrent GPU lanes, each serial. Logs -> $EMO_ROOT/runs/logs
cd "$(dirname "$0")" || exit 1
export EMO_ROOT="${EMO_ROOT:-$HOME/.cper}"
L=$EMO_ROOT/runs/logs; mkdir -p $L
run(){ tag=$1; shift; echo "[$(date +%H:%M)] start $tag $*" >> $L/queue.txt
       python train.py --tag $tag "$@" > $L/${tag}_$(echo "$@" | tr ' -' '__').log 2>&1
       grep RESULT $L/${tag}_$(echo "$@" | tr ' -' '__').log >> $L/results.txt; }
EP="--epochs 20"; EC="--epochs 6"
laneA(){
  for f in 0 1 2 3 4 5; do run ours --data ravdess --fold $f $EP; done
  for f in 0 1 2 3 4; do run ours --data cremad --fold $f $EC; done
  run ours --data ferplus
  for f in 0 1 2 3 4 5; do run abl_noE --data ravdess --fold $f --estep 0 $EP; done
  for f in 0 1 2 3 4 5; do run abl_eb --data ravdess --fold $f --re eb $EP; done
  run abl_noE --data cremad --fold 0 --estep 0 $EC
  run abl_eb --data cremad --fold 0 --re eb $EC
}
laneB(){
  for f in 0 1 2 3 4 5; do run ft_base --data ravdess --fold $f --re none --ctx 0 --reasoner tiny $EP; done
  for f in 0 1 2 3 4 5; do run mllm_base --data ravdess --fold $f --re none --ctx 0 $EP; done
  for f in 0 1 2 3 4 5; do run ctxmean --data ravdess --fold $f --re mean $EP; done
  for f in 0 1 2 3 4; do run ft_base --data cremad --fold $f --re none --ctx 0 --reasoner tiny $EC; done
  for f in 0 1 2 3 4; do run mllm_base --data cremad --fold $f --re none --ctx 0 $EC; done
  for f in 0 1 2 3 4; do run ctxmean --data cremad --fold $f --re mean $EC; done
  run ft_base --data ferplus --re none --reasoner tiny
  run mllm_base --data ferplus --re none
  for f in 0 1 2 3 4 5; do run abl_tiny --data ravdess --fold $f --reasoner tiny $EP; done
  run abl_tiny --data cremad --fold 0 --reasoner tiny $EC
  python zeroshot.py --model clip > $L/zs_clip.log 2>&1
  python zeroshot.py --model qwen2vl > $L/zs_qwen2vl.log 2>&1
  grep ZS $L/zs_*.log >> $L/results.txt
}
laneA & laneB & wait
echo "ALL EXPERIMENTS DONE $(date)" >> $L/queue.txt
