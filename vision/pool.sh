#!/bin/bash
# Job pool: N workers pull jobs from jobs.txt in priority order; a worker launches only when GPU memory allows.
# usage: bash pool.sh [n_workers]      -> prints "POOL DONE" when every job is finished
cd "$(dirname "$0")" || exit 1
export EMO_ROOT="${EMO_ROOT:-$HOME/.cper}"
L=$EMO_ROOT/runs/logs; LOCK=$EMO_ROOT/runs/locks_${JOBS:-jobs.txt}; mkdir -p $L $LOCK
N=${1:-2}; NEED_MB=${NEED_MB:-11000}

wait_gpu(){ while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ $((24564 - used)) -ge $NEED_MB ] && return; sleep 45; done; }

worker(){
  local w=$1 i=0
  grep -v "^#" ${JOBS:-jobs.txt} | grep -v '^\s*$' | while IFS='|' read -r tag args; do
    args=${args//@DATA_ROOT@/$EMO_ROOT}
    i=$((i+1)); tag=$(echo $tag | xargs); args=$(echo $args | xargs)
    mkdir $LOCK/job_$i 2>/dev/null || continue           # claimed by another worker
    if [ "$tag" = "ZS" ]; then
      wait_gpu; python zeroshot.py --model $args > $L/zs_$args.log 2>&1; grep ZS $L/zs_$args.log >> $L/results.txt; continue
    fi
    data=$(echo $args | sed -E 's/.*--data ([a-z]+).*/\1/'); fold=$(echo $args | grep -oE -- '--fold [0-9]' | awk '{print $2}'); fold=${fold:-0}
    seed=$(echo $args | grep -oE -- "--seed [0-9]+" | awk '{print $2}'); seed=${seed:-0}
    [ -f $EMO_ROOT/runs/$tag/${data}_f${fold}_s${seed}.json ] && continue
    wait_gpu
    echo "[$(date +%m-%d_%H:%M)] w$w start $tag $args" >> $L/queue.txt
    log=$L/${tag}_${data}_f${fold}_s${seed}.log
    python train.py --tag $tag $args < /dev/null > $log 2>&1
    grep RESULT $log >> $L/results.txt || echo "FAILED $tag $args" >> $L/results.txt
    sleep 20   # let the next worker see the freed memory before launching
  done
}
for w in $(seq 1 $N); do worker $w & sleep 90; done
wait
echo "POOL DONE $(date)" >> $L/queue.txt
echo "POOL DONE"
