#!/bin/sh
# usage: ./run_cmn.sh JOBS_FILE PARALLEL       (Git Bash on the RTX 4090 machine)
# Results: $CMN_OUT (default $EMO_ROOT/runs/cmn), one JSON per run + per-fold predictions in pred/.
# Runs whose JSON already exists are skipped, so the script can be restarted at any time.
# Each worker caps its GPU memory with MEM_FRAC (WDDM pages instead of failing: never exceed ~0.9 in total).
cd "$(dirname "$0")" || exit 1
export EMO_ROOT="${EMO_ROOT:-$HOME/.cper}"
export CMN_OUT=${CMN_OUT:-$EMO_ROOT/runs/cmn}
export MEM_FRAC=${MEM_FRAC:-0.13}
mkdir -p "$CMN_OUT/pred" "$CMN_OUT/logs"
LOG="$CMN_OUT/logs/results.txt"
mkdir -p "$(dirname "$LOG")"
tr -d '\r' < "$1" | xargs -P "$2" -L 1 sh -c '
  tag=$(python -c "import sys; from cmn import get_args; a=get_args(sys.argv[1:]); print(a.data+\"_\"+a.tag+\"_s\"+str(a.seed))" $0 "$@")
  [ -f "$CMN_OUT/$tag.json" ] && exit 0
  python cmn.py $0 "$@" > "$CMN_OUT/logs/$tag.log" 2>&1
  grep -E "RESULT|Error|Traceback" "$CMN_OUT/logs/$tag.log" | tail -n 3' >> "$LOG"
echo "DONE $1" >> "$LOG"
