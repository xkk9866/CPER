#!/bin/bash
source $EMO_ROOT/env.sh; cd "$(dirname "$0")" || exit 1
export EMO_ROOT="${EMO_ROOT:-$HOME/.cper}"; mkdir -p $EMO_ROOT/runs/semix/logs
cat jobs_semix.txt | xargs -P 8 -I{} sh -c 'python semix.py {} 2>&1 | grep -E "RESULT|Error" >> $EMO_ROOT/runs/semix/logs/results.txt'
echo ALLDONE >> $EMO_ROOT/runs/semix/logs/results.txt
