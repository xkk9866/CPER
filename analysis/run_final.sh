#!/bin/sh
# usage: ./run_final.sh JOBS_FILE PARALLEL ; results in $EMO_ROOT/runs/final, existing results are skipped
cd "$(dirname "$0")" || exit 1
export EMO_ROOT="${EMO_ROOT:-$HOME/.cper}"
export COMPASS_OUT=$EMO_ROOT/runs/final
mkdir -p $EMO_ROOT/runs/final/pred $EMO_ROOT/runs/final/logs
LOG=$EMO_ROOT/runs/final/logs/results.txt
mkdir -p "$(dirname "$LOG")"
cat $1 | xargs -P $2 -L 1 sh -c '
  tag=$(python -c "import sys; from compass import get_args; a=get_args(sys.argv[1:]); print(a.data+\"_\"+a.tag+\"_s\"+str(a.seed))" $0 "$@")
  [ -f $EMO_ROOT/runs/final/$tag.json ] && exit 0
  python compass.py $0 "$@" 2>&1 | grep -E "RESULT|Error|Traceback"' >> $LOG
echo "DONE $1" >> $LOG
