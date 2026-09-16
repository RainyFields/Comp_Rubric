#!/bin/bash
# FoldAgent replication — training worker (one arm per worker)
# Usage via per-job wrapper (--envs does not reach scripts):
#   exec env ARM=foldgrpo STEPS=100 bash worker_train.sh   # or ARM=grpo; STEPS=5 for shakeout
# Arms differ by exactly two flags (locked decision):
#   foldgrpo: algorithm.adv_estimator=foldgrpo  process_reward='[flat,scope]'  (script defaults)
#   grpo:     algorithm.adv_estimator=grpo      process_reward=none
#   compactionrl / compactiongrpo: scripts/train_bc_compaction{rl,grpo}.sh (CompactionRL baseline, docs/baselines/)
set -uo pipefail

ARM=${ARM:?foldgrpo|grpo}; STEPS=${STEPS:?}
SRC=/home/tiger/xiaoxuan/FoldAgent
CHECKOUT=/home/tiger/xiaoxuan/fold_arms/$ARM
VENV=/home/tiger/xiaoxuan/envs/fold_train
MARK=${MARK:-$SRC/infra/markers}                                            # batch jobs: shared HDFS dir
HDFS_CKPT=${HDFS_CKPT:-/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt/$ARM}
KEEP_LOCAL=${KEEP_LOCAL:-2}                                                 # verified-uploaded ckpts to keep on /tmp
LOCAL_CKPT=/tmp/fold_ckpt/$ARM
case "$ARM" in
  foldgrpo)       RUN_NAME=foldgrpo_qwen3-8b_bcplus ;;
  grpo)           RUN_NAME=grpo_qwen3-8b_bcplus_baseline ;;
  compactionrl)   RUN_NAME=compactionrl_qwen3-8b_bcplus ;;        # CompactionRL, PPO + critic (paper-faithful)
  compactiongrpo) RUN_NAME=compactiongrpo_qwen3-8b_bcplus ;;      # CompactionRL rollout + group-relative advantage
  *) echo "unknown ARM=$ARM (foldgrpo|grpo|compactionrl|compactiongrpo)" >&2; exit 2 ;;
esac
RUN_NAME=${RUN_NAME}${RUN_SUFFIX:-}

log() { echo "[train:$ARM $(date +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; touch "$MARK/TRAIN_${ARM}_FAILED"; sleep 60; exit 1; }
rm -f "$MARK/TRAIN_${ARM}_FAILED" "$MARK/TRAIN_${ARM}_DONE"
mkdir -p "$LOCAL_CKPT" "$HDFS_CKPT"

wait_infra() {
  for i in $(seq 1 ${INFRA_WAIT_ITERS:-180}); do   # x30s; batch jobs raise this (infra pod may queue for hours)
    IP=$(cat "$MARK/INFRA_READY" 2>/dev/null) && { case "$IP" in *:*) IPX="[$IP]";; *) IPX=$IP;; esac
      curl -sf -m 8 "http://$IPX:8000/search" -H 'Content-Type: application/json' -d '{"query":"t","k":1}' >/dev/null 2>&1 && return 0; }
    sleep 30
  done; return 1
}
wait_infra || fail "no live infra after $(( ${INFRA_WAIT_ITERS:-180} / 2 ))min"
df -h /tmp | tail -1; nvidia-smi -L | grep -q "^GPU 0" || fail "NVML broken"

# --- per-arm checkout (isolation from crash-resume commit pickup) ---
if [ ! -d "$CHECKOUT/.git" ]; then
  git clone "$SRC" "$CHECKOUT" || fail "clone"
fi
mkdir -p "$CHECKOUT/data"
cp -n "$SRC/data/bc_train.parquet" "$SRC/data/bc_test.parquet" "$CHECKOUT/data/" 2>/dev/null || true

# --- training venv (vllm 0.10.2 + prebuilt flash-attn; NEVER source-build) ---
export HF_HOME=/tmp/fold_train_hf
export TMPDIR=/tmp/fold_tmp && mkdir -p "$TMPDIR"   # short, existing tmp for vLLM zmq ipc sockets
if [ ! -f "$VENV/ok" ]; then
  uv venv "$VENV" --python 3.11 || fail "uv venv"
  source "$VENV/bin/activate"
  uv pip install "vllm==0.10.2" || fail "vllm install"
  ABI=$(python -c 'import torch;print(torch._C._GLIBCXX_USE_CXX11_ABI)') || fail "torch abi probe"
  TV=$(python -c 'import torch;print(".".join(torch.__version__.split("+")[0].split(".")[:2]))')
  WHL="flash_attn-2.8.3+cu12torch${TV}cxx11abi${ABI^^}-cp311-cp311-linux_x86_64.whl"
  uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/$WHL" || fail "flash-attn wheel $WHL"
  grep -v "^# vllm\|^vllm" "$SRC/requirements.txt" > /tmp/req_no_vllm.txt
  uv pip install -r /tmp/req_no_vllm.txt || fail "repo requirements"
  uv pip install wandb httpx openai || fail "extras"
  touch "$VENV/ok"
else
  source "$VENV/bin/activate"
fi
python -c "import torch,vllm,flash_attn,ray; print('torch',torch.__version__,'vllm',vllm.__version__)" || fail "import gate"
hf download Qwen/Qwen3-8B >/dev/null 2>&1 || huggingface-cli download Qwen/Qwen3-8B >/dev/null || fail "dl Qwen3-8B"

# --- local judge shim (scope judge via OPENAI_URL custom dialect; grader via OPENAI_BASE_URL) ---
mkdir -p "$CHECKOUT/logs"
nohup python -u "$SRC/infra/judge_shim.py" \
  --upstream-marker "$MARK/INFRA_READY" \
  --port 8002 --log-dir "$CHECKOUT/logs" > "$CHECKOUT/logs/shim.log" 2>&1 &
SHIM_PID=$!
shim_ok=0
for i in $(seq 1 24); do  # up to 2 min for cold imports
  curl -sf -m 5 http://127.0.0.1:8002/v1/models >/dev/null 2>&1 && { shim_ok=1; break; }
  kill -0 $SHIM_PID 2>/dev/null || break
  sleep 5
done
[ $shim_ok -eq 1 ] || { tail -20 "$CHECKOUT/logs/shim.log"; fail "shim not up"; }


export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
export OPENAI_URL=http://127.0.0.1:8002/chat
export LOCAL_SEARCH_URL=http://127.0.0.1:8002

# --- W&B: use existing auth if present, else offline (never trigger auth) ---
if python -c "import wandb,sys; sys.exit(0 if wandb.Api().api_key else 1)" 2>/dev/null; then
  log "wandb authenticated — online"
else
  export WANDB_MODE=offline; log "wandb offline mode"
fi
export WANDB_NAME=$RUN_NAME

# --- checkpoint uploader: mirror to HDFS, verify size, keep newest 2 local ---
(
  while true; do
    sleep 600
    for d in "$LOCAL_CKPT"/global_step_*; do
      [ -d "$d" ] || continue
      base=$(basename "$d")
      newest=$(ls -d "$LOCAL_CKPT"/global_step_* 2>/dev/null | sort -V | tail -1)
      [ "$d" = "$newest" ] && continue          # never touch the in-progress/newest
      if [ ! -f "$HDFS_CKPT/$base/.upload_done" ]; then
        cp -r "$d" "$HDFS_CKPT/" && \
        [ "$(du -sb "$d" | cut -f1)" = "$(du -sb "$HDFS_CKPT/$base" | cut -f1)" ] && \
        touch "$HDFS_CKPT/$base/.upload_done"
      fi
      # delete local only if HDFS-verified and not among newest 2
      keep=$(ls -d "$LOCAL_CKPT"/global_step_* | sort -V | tail -$KEEP_LOCAL)
      if [ -f "$HDFS_CKPT/$base/.upload_done" ] && ! echo "$keep" | grep -q "$base$"; then
        rm -rf "$d"
      fi
    done
  done
) &
UPLOADER_PID=$!

# --- arm-specific flags ---
EXTRA=""
TRAIN_SCRIPT=scripts/train_bc_qwen3_8b.sh
case "$ARM" in
  grpo)           EXTRA="algorithm.adv_estimator=grpo ++actor_rollout_ref.rollout.plugin.process_reward=none" ;;
  compactionrl)   TRAIN_SCRIPT=scripts/train_bc_compactionrl.sh ;;      # flags live in the script
  compactiongrpo) TRAIN_SCRIPT=scripts/train_bc_compactiongrpo.sh ;;
esac

cd "$CHECKOUT"
mkdir -p logs
export PYTHONPATH=$CHECKOUT${PYTHONPATH:+:$PYTHONPATH}   # ray AgentLoopWorkers must import scripts.train_fold
sed -e "s#trainer.total_training_steps=100#trainer.total_training_steps=${STEPS}#" \
    -e "s#trainer.experiment_name=test_run#trainer.experiment_name=${RUN_NAME}#" \
    "$TRAIN_SCRIPT" > logs/launch_${ARM}.sh
# append overrides: ckpt dir + custom agent loop registration (workers load it) + arm flags
[ -n "${VAL_DUMP_DIR:-}" ] && EXTRA="$EXTRA trainer.validation_data_dir=${VAL_DUMP_DIR}"   # optional: per-val trajectory dumps
[ -n "${ROLLOUT_DUMP_DIR:-}" ] && EXTRA="$EXTRA trainer.rollout_data_dir=${ROLLOUT_DUMP_DIR}"   # optional: per-step training-rollout dumps (shakeouts)
sed -i "s#trainer.project_name=context_folding#trainer.project_name=context_folding trainer.default_local_dir=${LOCAL_CKPT} actor_rollout_ref.rollout.agent.agent_loop_config_path=${SRC}/infra/agent_loop_config.yaml ${EXTRA}#" logs/launch_${ARM}.sh
log "launching: STEPS=$STEPS EXTRA='$EXTRA'"
bash logs/launch_${ARM}.sh 2>&1 | tee "logs/train_${ARM}.log"
rc=${PIPESTATUS[0]}
kill $UPLOADER_PID 2>/dev/null
# final sync: upload ALL checkpoints (incl. newest) before declaring done — /tmp dies with the worker
for d in "$LOCAL_CKPT"/global_step_*; do
  [ -d "$d" ] || continue
  base=$(basename "$d")
  if [ ! -f "$HDFS_CKPT/$base/.upload_done" ]; then
    log "final upload: $base"
    rm -rf "$HDFS_CKPT/$base"
    cp -r "$d" "$HDFS_CKPT/" && \
    [ "$(du -sb "$d" | cut -f1)" = "$(du -sb "$HDFS_CKPT/$base" | cut -f1)" ] && \
    touch "$HDFS_CKPT/$base/.upload_done" || log "WARN: final upload failed for $base"
  fi
done
[ $rc -eq 0 ] || fail "training exit $rc (see $CHECKOUT/logs/train_${ARM}.log)"
touch "$MARK/TRAIN_${ARM}_DONE"
log "DONE"
