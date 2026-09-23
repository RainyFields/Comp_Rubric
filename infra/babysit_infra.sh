#!/bin/bash
# DEPRECATED (2026-09-23): persistent auto-relaunch loops are against the working rules (no babysitters unless
# explicitly authorised). Kept for the August mlx-worker era only; the batch-job era uses
# infra/infra_resubmit_loop.sh, started by hand with user authorisation for one arm at a time.
# Infra babysitter: keeps search+judge node alive; emits one line per state change.
M=/home/tiger/xiaoxuan/FoldAgent/infra/markers
relaunches=0
while [ $relaunches -le 40 ]; do
  IP=$(cat "$M/INFRA_READY" 2>/dev/null)
  if [ -n "$IP" ]; then
    case "$IP" in *:*) IPX="[$IP]";; *) IPX=$IP;; esac
    if curl -sf -m 8 "http://$IPX:8000/search" -H 'Content-Type: application/json' -d '{"query":"t","k":1}' >/dev/null 2>&1; then
      sleep 120; continue
    fi
  fi
  relaunches=$((relaunches+1))
  echo "INFRA DEAD (relaunch #$relaunches)"
  pkill -f "mlxc worker launch.*fold-infr[a]" 2>/dev/null
  rm -f "$M"/INFRA_*
  # alternate queues: odd attempts H100, even attempts A100 (both cloudnative-maliva)
  if [ $((relaunches % 2)) -eq 0 ]; then
    QUEUE=compute-89-aliyun.va-cloudnative-aigcp-ark.eng.algorithm-guarantee; GTYPE=H100-SXM-80GB
  else
    QUEUE=compute-89-aliyun.va-cloudnative-ai-ark.eng.algorithm-guarantee; GTYPE=A100-SXM-80GB
  fi
  echo "attempt on $GTYPE"
  setsid nohup mlx worker launch --resourcetype arnold --usergroup ark-eng-algorithm \
    --cluster cloudnative-maliva --queuename "$QUEUE" \
    --gpu 8 --type "$GTYPE" --alias fold-infra --no-input \
    -- bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_infra.sh \
    > /home/tiger/xiaoxuan/FoldAgent/infra/launch_infra.log 2>&1 < /dev/null &
  for i in $(seq 1 240); do
    [ -f "$M/INFRA_READY" ] && { echo "INFRA READY: $(cat $M/INFRA_READY)"; break; }
    [ -f "$M/INFRA_FAILED" ] && { echo "INFRA RELAUNCH FAILED (see markers logs)"; break; }
    sleep 30
  done
done
echo "BABYSITTER EXHAUSTED (10 relaunches)"
