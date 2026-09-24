#!/bin/bash
# Batch-job entrypoint: one RL arm (FOLD_ARM=foldgrpo|grpo|compactionrl|compactiongrpo) with the chat-template fix (repo commit in tarball).
# Waits for the shared infra pod via $MARK/INFRA_READY on HDFS, restores the latest complete HDFS checkpoint
# (verl resume_mode=auto), mirrors the train log to HDFS every 5 min, uploads checkpoints via worker_train.sh.
set -uo pipefail
source /mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/fold_common_bootstrap.sh
ARM=${FOLD_ARM:?}; STEPS=${FOLD_STEPS:-100}
[ -n "${FOLD_MODEL_PATH:-}" ] && export MODEL_PATH=$FOLD_MODEL_PATH   # policy family (default Qwen/Qwen3-8B in worker_train.sh)
[ -n "${FOLD_PROTOCOL:-}" ] && export PROTOCOL=$FOLD_PROTOCOL           # rollout protocol (agents/protocol.py): v2 default | legacy
# Qwen3.5 / Experiment I knobs (scripts/train_bc.sh reads them from the environment; unset = launcher defaults)
for kv in TRAIN_FILE:TRAIN_DATA_PATH VAL_FILE:TEST_DATA_PATH PROMPT_LENGTH:PROMPT_LENGTH RESPONSE_LENGTH:RESPONSE_LENGTH \
          TURN_MAX_NEW_TOKENS:TURN_MAX_NEW_TOKENS MAX_TURN:MAX_TURN MAX_TOOL_CALLS:MAX_TOOL_CALLS MAX_CALLS_PER_TURN:MAX_CALLS_PER_TURN \
          SESSION_TIMEOUT:SESSION_TIMEOUT USE_FUSED_KERNELS:USE_FUSED_KERNELS USE_REMOVE_PADDING:USE_REMOVE_PADDING \
          MAX_SUMMARIES:MAX_SUMMARIES SUMMARY_RATIO:SUMMARY_RATIO MAX_COMPACTIONS:MAX_COMPACTIONS COMPACTION_THRESHOLD:COMPACTION_THRESHOLD \
          TAIL_STEPS:TAIL_STEPS SUMMARY_MAX_TOKENS:SUMMARY_MAX_TOKENS ROLLOUT_N:ROLLOUT_N TRAIN_BATCH:TRAIN_BATCH LR:LR TP:TP; do
  src=FOLD_${kv%%:*}; dst=${kv##*:}; [ -n "${!src:-}" ] && export "$dst=${!src}"
done
OUT=${FOLD_OUT_TAG:-$ARM}   # HDFS output subdir (ckpt/val dumps/logs); shakeouts set e.g. <arm>_shakeout to keep the arm dir clean
export MARK=${FOLD_MARK:-/mnt/hdfs/mlsys/xiaoxuan/fold_replication/markers_fix}
export HDFS_CKPT=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt_fix/$OUT
export RUN_SUFFIX=${FOLD_RUN_SUFFIX:-_fix}
export KEEP_LOCAL=1
export INFRA_WAIT_ITERS=720   # 6 h: the shared infra pod may still be queued
export VAL_DUMP_DIR=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/val_dump_fix/$OUT
[ "${FOLD_ROLLOUT_DUMP:-0}" = 1 ] && export ROLLOUT_DUMP_DIR=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/rollout_dump_fix/$OUT
HDFS_LOGS=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/train_logs_fix/$OUT
LOCAL_CKPT=/tmp/fold_ckpt/$ARM
mkdir -p "$MARK" "$HDFS_CKPT" "$VAL_DUMP_DIR" "$HDFS_LOGS" "$LOCAL_CKPT"
log "train job arm=$ARM steps=$STEPS venv=$TRAIN_VENV model=${MODEL_PATH:-Qwen/Qwen3-8B} node=$(hostname) MARK=$MARK HDFS_CKPT=$HDFS_CKPT"
restore_venv fold_infra || exit 43
restore_venv "$TRAIN_VENV" || exit 43   # FOLD_VENV (fold_train | fold_train_q35), see fold_common_bootstrap.sh
restore_repo || exit 44
gpu_preflight || { log "preflight failed, exit 42"; exit 42; }
[ -f /mnt/hdfs/mlsys/users/xiaoxuan/arco-job-assets/wandb.key ] && export WANDB_API_KEY=$(cat /mnt/hdfs/mlsys/users/xiaoxuan/arco-job-assets/wandb.key)
export WANDB_RUN_ID=${FOLD_WANDB_RUN_ID:-fold_fix_${ARM}_2026-09-10} WANDB_RESUME=allow   # new arms set FOLD_WANDB_RUN_ID in the spec

# --- resume: newest verified HDFS checkpoint -> /tmp (resume_mode=auto reads latest_checkpointed_iteration.txt) ---
latest=$(ls -d "$HDFS_CKPT"/global_step_* 2>/dev/null | sort -V | while read d; do [ -f "$d/.upload_done" ] && echo "$d"; done | tail -1)
if [ -n "$latest" ]; then
  b=$(basename "$latest"); log "restoring $b from HDFS for resume ..."; t0=$(date +%s)
  cp -r "$latest" "$LOCAL_CKPT/" || exit 46
  [ "$(du -sb "$LOCAL_CKPT/$b" | cut -f1)" = "$(du -sb "$latest" | cut -f1)" ] || { log "restore size mismatch"; exit 46; }
  echo "${b#global_step_}" > "$LOCAL_CKPT/latest_checkpointed_iteration.txt"
  log "restored $b in $(( $(date +%s) - t0 )) s"
else
  log "no HDFS checkpoint yet — fresh start"
fi

# --- log mirror (the pod's /tmp and checkout die with the pod) ---
CHECKOUT=$XD/fold_arms/$ARM
( while true; do sleep 300; for f in "$CHECKOUT"/logs/train_${ARM}.log "$CHECKOUT"/logs/shim.log "$CHECKOUT"/logs/judge_calls.jsonl; do
    [ -f "$f" ] && tail -c 50000000 "$f" > "$HDFS_LOGS/$(basename "$f")" 2>/dev/null; done; done ) &
MIRROR_PID=$!

env ARM=$ARM STEPS=$STEPS bash "$SRC/infra/worker_train.sh"
rc=$?
kill $MIRROR_PID 2>/dev/null
for f in "$CHECKOUT"/logs/train_${ARM}.log "$CHECKOUT"/logs/shim.log "$CHECKOUT"/logs/judge_calls.jsonl; do
  [ -f "$f" ] && cp "$f" "$HDFS_LOGS/" 2>/dev/null; done
log "arm $ARM finished rc=$rc"
exit $rc
