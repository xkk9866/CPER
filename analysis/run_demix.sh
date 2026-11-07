#!/bin/bash
# runs every line of jobs_demix.txt (8 in parallel; tiny models on cached features, <1.5 GB GPU each)
source $EMO_ROOT/env.sh; cd "$(dirname "$0")" || exit 1
export EMO_ROOT="${EMO_ROOT:-$HOME/.cper}"; mkdir -p $EMO_ROOT/runs/demix/logs
cat jobs_demix.txt | xargs -P 8 -I{} sh -c 'python demix.py {} 2>&1 | grep -E "RESULT|Error" >> $EMO_ROOT/runs/demix/logs/results.txt'
echo ALLDONE >> $EMO_ROOT/runs/demix/logs/results.txt
