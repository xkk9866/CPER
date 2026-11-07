#!/bin/sh
# 140 runs of compass.py, 8 in parallel on the local GPU; existing results are skipped
cd "$(dirname "$0")" || exit 1
export EMO_ROOT="${EMO_ROOT:-$HOME/.cper}"
LOG=$EMO_ROOT/runs/compass/logs/results.txt
mkdir -p "$(dirname "$LOG")"
cat jobs_compass.txt | xargs -P 8 -L 1 sh -c '
  tag=$(python -c "import sys; from compass import get_args; a=get_args(sys.argv[1:]); print(a.data+\"_\"+a.tag+\"_s\"+str(a.seed))" $0 "$@")
  [ -f $EMO_ROOT/runs/compass/$tag.json ] && exit 0
  python compass.py $0 "$@" 2>&1 | grep -E "RESULT|Error|Traceback"' >> $LOG
echo ALLDONE >> $LOG
