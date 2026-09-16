#!/bin/bash
# Marker-gated infra keep-alive for ONE training arm (user-authorised 2026-09-16; see CLAUDE.md babysitter rule).
# Resubmits the search+judge infra job only when the current one is no longer running (queue reclamation on the
# 4-h grid) or its search endpoint is dead with a stale heartbeat; never more than once per 35 min; at most
# MAX_RESUBMITS times; stops the infra job and exits when the arm's TRAIN_<arm>_DONE/FAILED marker appears or
# the arm job leaves the running state.
#   usage: ARM=compactiongrpo ARM_SID=<sid> INFRA_SID=<current infra sid> SPEC=fold-infra-fix-h100.json bash infra/infra_resubmit_loop.sh
set -u
ARM=${ARM:?}; ARM_SID=${ARM_SID:?}; INFRA_SID=${INFRA_SID:?}; SPEC=${SPEC:-fold-infra-fix-h100.json}
MAX_RESUBMITS=${MAX_RESUBMITS:-40}; MIN_GAP=${MIN_GAP:-2100}; STALE=${STALE:-1200}
M=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/markers_fix
J=/home/tiger/xiaoxuan/FoldAgent/infra/jobs
CP=i18n-tt
log() { echo "[infra-loop $(date '+%F %T')] $*"; }
status() { merlin-cli --control-plane $CP job-v2 runs get --json "{\"sid\":\"$1\"}" 2>/dev/null | python3 -c 'import sys,json
try: d=json.load(sys.stdin)["data"]; print(d.get("runtime_context",{}).get("trial_status") or d.get("status") or "unknown")
except Exception: print("unknown")'; }
search_ok() { IP=$(cat $M/INFRA_READY 2>/dev/null) || return 1; case "$IP" in *:*) IPX="[$IP]";; *) IPX=$IP;; esac
  curl -sf -m 8 "http://$IPX:8000/search" -H 'Content-Type: application/json' -d '{"query":"t","k":1}' >/dev/null 2>&1; }
last_submit=$(date +%s); n=0; T_START=$(date +%s)
# only honour a DONE/FAILED marker written after this loop started (a stale one from a shakeout of the same arm
# would otherwise stop the infra before the training worker gets to clear it — seen 2026-09-16)
fresh() { [ -f "$1" ] && [ "$(stat -c %Y "$1" 2>/dev/null || echo 0)" -ge "$T_START" ]; }
log "start arm=$ARM arm_sid=$ARM_SID infra_sid=$INFRA_SID spec=$SPEC"
while true; do
  if fresh $M/TRAIN_${ARM}_DONE || fresh $M/TRAIN_${ARM}_FAILED; then
    log "arm marker present ($(ls $M | grep TRAIN_${ARM}_ | tr '\n' ' ')); stopping infra $INFRA_SID and exiting"
    merlin-cli --control-plane $CP job-v2 runs stop --json "{\"sid\":\"$INFRA_SID\"}" >/dev/null 2>&1; exit 0
  fi
  ast=$(status $ARM_SID)
  case "$ast" in failed|stopped|killed|error|done|finished)
    log "arm job $ast; stopping infra $INFRA_SID and exiting"
    merlin-cli --control-plane $CP job-v2 runs stop --json "{\"sid\":\"$INFRA_SID\"}" >/dev/null 2>&1; exit 0;; esac
  ist=$(status $INFRA_SID)
  hb=$(stat -c %Y $M/heartbeat.txt 2>/dev/null || echo 0); now=$(date +%s)
  dead=0
  case "$ist" in failed|stopped|killed|error|done|finished) dead=1;; esac
  if [ $dead -eq 0 ] && ! search_ok && [ $((now - hb)) -gt $STALE ] && [ $((now - last_submit)) -gt $MIN_GAP ]; then dead=1; fi
  if [ $dead -eq 1 ]; then
    if [ $((now - last_submit)) -le $MIN_GAP ]; then log "infra $INFRA_SID $ist but last submit $((now - last_submit))s ago; waiting"; sleep 120; continue; fi
    if [ $n -ge $MAX_RESUBMITS ]; then log "MAX_RESUBMITS reached; exiting"; exit 3; fi
    n=$((n+1))
    out=$(cd $J && merlin-cli --control-plane $CP job-v2 runs create --from-file $SPEC 2>/dev/null)
    sid=$(echo "$out" | python3 -c "import sys,json
try: d=json.load(sys.stdin); x=d.get('data',d); print(x.get('sid') or x.get('run_sid') or '')
except Exception: print('')")
    if [ -n "$sid" ]; then
      log "infra $INFRA_SID was $ist -> resubmitted #$n as $sid"
      printf "%s\t%s\t%s\t%s\tinfra keep-alive loop resubmit #%s for arm %s (previous %s was %s)\n" "$(date '+%F %H:%M')" "$sid" "${SPEC%.json}" "$SPEC" "$n" "$ARM" "$INFRA_SID" "$ist" >> $J/JOBS.tsv
      INFRA_SID=$sid; last_submit=$(date +%s)
    else
      log "resubmit failed (empty sid); retry in 5 min"; sleep 300; continue
    fi
  fi
  sleep 120
done
