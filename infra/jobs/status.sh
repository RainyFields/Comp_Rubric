#!/bin/bash
# One-shot status of the fix-campaign jobs (no loop). Usage: bash infra/jobs/status.sh [sid ...]
# Default sids = the two training arms + the newest infra sid in JOBS.tsv.
CP=i18n-tt
H=/mnt/hdfs/mlsys/xiaoxuan/fold_replication
J=$(dirname "$0")/JOBS.tsv
if [ $# -gt 0 ]; then SIDS="$*"; else
  SIDS="ed81a06ae2477596 078762e6527e2b8d $(grep -P '\tfold-infra-fix-' "$J" | tail -1 | cut -f2)"
fi
echo "now: $(date '+%F %T %Z')"
for s in $SIDS; do
  merlin-cli --control-plane $CP job-v2 runs get --json "{\"sid\":\"$s\"}" 2>/dev/null | python3 -c '
import sys,json
try: d=json.load(sys.stdin)["data"]
except Exception as e: print(sys.argv[1], "ERR", e); sys.exit()
print(sys.argv[1], d.get("name"), d.get("trial_status") or d.get("status"), "created", d.get("created_at"))' "$s"
done
echo "INFRA_READY: $(cat $H/markers_fix/INFRA_READY 2>/dev/null || echo none)  heartbeat: $(stat -c %y $H/markers_fix/heartbeat.txt 2>/dev/null | cut -c1-19)"
for a in foldgrpo grpo; do
  L=$H/train_logs_fix/$a/train_$a.log
  printf "%-9s step %s  (log mtime %s)  val: %s\n" $a "$(grep -o 'training/global_step:[0-9]*' $L | tail -1 | cut -d: -f2)" \
    "$(stat -c %y $L | cut -c1-16)" "$(grep -o 'val-core/unknown/reward/mean@1:[0-9.]*' $L | cut -d: -f2 | tr '\n' ' ')"
done
ls $H/markers_fix/ | grep -E 'TRAIN_|FAILED|DEGRADED' || true
