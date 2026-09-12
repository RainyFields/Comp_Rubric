#!/bin/bash
# Merlin/Arnold batch-job entrypoint: no-RL Qwen3-8B val-only contrast with the chat-template fix.
# Pod bootstrap (venvs + repo from HDFS tarballs at the devbox's absolute paths), judge model from
# HDFS, then worker_val_selfcontained.sh once per MODE (greedy, t1n4), results mirrored to HDFS.
# Env (job env_map): FOLD_MODES (default "greedy t1n4"), FOLD_TAGSUF (default "_fix"), FOLD_ARM/FOLD_STEP.
set -uo pipefail
ASSETS=/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets
HDFS_OUT=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/results
XD=/home/tiger/xiaoxuan
SRC=$XD/FoldAgent
MODES=${FOLD_MODES:-"greedy t1n4"}
export TAGSUF=${FOLD_TAGSUF:-_fix}
ARM=${FOLD_ARM:-norl}; STEP=${FOLD_STEP:-base}
export CKPT_ROOT=${FOLD_CKPT_ROOT:-/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt}
log() { echo "[foldjob $(date '+%m-%d %H:%M:%S')] $*"; }

log "node=$(hostname) arm=$ARM step=$STEP modes='$MODES' tagsuf=$TAGSUF"
ls "$ASSETS" >/dev/null 2>&1 || { log "FATAL: HDFS fuse not available"; exit 41; }
df -h /tmp | tail -1; nproc; free -g | head -2
mkdir -p "$XD/envs" /tmp/fold_tmp /tmp/models

restore_venv() {  # $1 = venv name
  [ -x "$XD/envs/$1/bin/python" ] && { log "venv $1 present"; return 0; }
  log "restoring venv $1 ..."
  local t=/tmp/$1.tar.gz
  cp "$ASSETS/$1.tar.gz" "$t" || return 1
  [ "$(md5sum "$t" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$ASSETS/$1.tar.gz.md5")" ] || { log "md5 mismatch $1"; return 1; }
  tar xzf "$t" -C "$XD/envs" && rm -f "$t"
}
restore_venv fold_infra || exit 43
restore_venv fold_train || exit 43
log "restoring repo ..."
rm -rf "$SRC"; tar xzf "$ASSETS/foldagent-repo.tar.gz" -C "$XD" || exit 44
( cd "$SRC" && git log --oneline -1 )
mkdir -p "$SRC/infra/markers" "$SRC/results"

# judge model: HDFS snapshot -> pod-local (avoids the 62 GB hf snapshot_download hang), parallel cp
JUDGE=/tmp/models/gpt-oss-120b
if [ ! -f "$JUDGE/config.json" ]; then
  log "copying gpt-oss-120b from HDFS ..."
  mkdir -p "$JUDGE"; t0=$(date +%s)
  # top-level files only: the snapshot also holds original/ (raw 60 GB weights) and .cache/, which vLLM does not need
  SRCJ=/mnt/hdfs/mlsys/users/xiaoxuan/models/gpt-oss-120b
  find "$SRCJ" -maxdepth 1 -type f -printf '%f\n' | xargs -P 8 -I{} cp "$SRCJ/{}" "$JUDGE"/ || exit 45
  for f in $(find "$SRCJ" -maxdepth 1 -type f -printf '%f\n'); do
    [ "$(stat -c %s "$JUDGE/$f")" = "$(stat -c %s "$SRCJ/$f")" ] || { log "judge file size mismatch: $f"; exit 45; }
  done
  log "judge copy done in $(( $(date +%s) - t0 )) s ($(du -sh "$JUDGE" | cut -f1), $(ls "$JUDGE" | wc -l) files)"
fi
export JUDGE_MODEL=$JUDGE

"$XD/envs/fold_train/bin/python" - <<'PYEOF' || { echo "[foldjob] preflight failed, exit 42"; exit 42; }
import sys, socket, torch
n = torch.cuda.device_count()
if n < 8: sys.exit(f"preflight FAILED: only {n} GPUs")
torch.zeros(1, device="cuda:0")
print(f"[foldjob] preflight OK: {n} GPUs on {socket.gethostname()}, torch {torch.__version__}", flush=True)
PYEOF

mkdir -p "$HDFS_OUT"
rc_all=0
for MODE in $MODES; do
  TAG=${ARM}_${STEP}_${MODE}${TAGSUF}_sc
  log "=== run $TAG ==="
  env ARM=$ARM STEP=$STEP MODE=$MODE bash "$SRC/infra/worker_val_selfcontained.sh"
  rc=$?; [ $rc -ne 0 ] && rc_all=$rc
  log "run $TAG rc=$rc; mirroring results to HDFS"
  # keep shim/search/judge logs small in the mirror; dump + val_metrics + valonly.log are what matter
  rsync -a --exclude 'shim.log' --exclude 'search.log' --exclude 'judge.log' "$SRC/results/valonly_$TAG/" "$HDFS_OUT/valonly_$TAG/" 2>/dev/null \
    || cp -r "$SRC/results/valonly_$TAG" "$HDFS_OUT/" 2>/dev/null
  tail -c 200000 "$SRC/results/valonly_$TAG/shim.log" > "$HDFS_OUT/valonly_$TAG/shim.tail.log" 2>/dev/null
  cp "$SRC/infra/markers/VALONLY_${TAG}"_* "$HDFS_OUT/valonly_$TAG/" 2>/dev/null
  cat "$SRC/results/valonly_$TAG/val_metrics.txt" 2>/dev/null
done
log "all done rc=$rc_all"
exit $rc_all
