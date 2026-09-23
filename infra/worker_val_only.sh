#!/bin/bash
# FoldAgent replication — faithful checkpoint eval via the TRAINING stack's validation path.
# Rationale: standalone /chat/completions serving re-templates each turn (drops prior <think>),
# collapsing RL'd policies (0.027 standalone vs 0.26 training-val for the same ckpt). val_only
# reproduces training-time serving (token-in-token-out CallLLM) — identical protocol for all arms.
# Params: ARM=foldgrpo|grpo|norl  STEP=<n|base>  MODE=greedy|t1n4
set -uo pipefail

ARM=${ARM:?}; STEP=${STEP:?}; MODE=${MODE:?}
SRC=/home/tiger/xiaoxuan/Comp_Rubric
CHECKOUT=/home/tiger/xiaoxuan/fold_arms/valonly_${ARM}_${STEP}_${MODE}
VENV=/home/tiger/xiaoxuan/envs/fold_train
MARK=$SRC/infra/markers
HDFS_CKPT=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt/$ARM
LOCAL_CKPT=/tmp/fold_valonly/$ARM
TAG=${ARM}_${STEP}_${MODE}
OUTDIR=$SRC/results/valonly_$TAG

log() { echo "[valonly:$TAG $(date +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; touch "$MARK/VALONLY_${TAG}_FAILED"; sleep 60; exit 1; }
rm -f "$MARK/VALONLY_${TAG}_FAILED" "$MARK/VALONLY_${TAG}_DONE"
mkdir -p "$OUTDIR"

export TMPDIR=/tmp/fold_tmp && mkdir -p "$TMPDIR"
export HF_HOME=/tmp/fold_train_hf
df -h /tmp | tail -1
nvidia-smi -L | grep -q "^GPU 0" || { hostname -I > "$MARK/BAD_NODE_${TAG}.txt"; fail "NVML broken on $(hostname -I | awk '{print $1}')"; }
wait_infra() {  # wait up to 3h for a live infra (marker may be stale after a death)
  for i in $(seq 1 360); do
    IP=$(cat "$MARK/INFRA_READY" 2>/dev/null) && { case "$IP" in *:*) IPX="[$IP]";; *) IPX=$IP;; esac
      curl -sf -m 8 "http://$IPX:8000/search" -H 'Content-Type: application/json' -d '{"query":"t","k":1}' >/dev/null 2>&1 && return 0; }
    sleep 30
  done; return 1
}
wait_infra || fail "no live infra after 3h"

# --- fresh checkout (patches committed in $SRC) ---
rm -rf "$CHECKOUT"; git clone -q "$SRC" "$CHECKOUT" || fail "clone"
mkdir -p "$CHECKOUT/data" "$CHECKOUT/logs"
cp -n "$SRC/data/bc_train.parquet" "$SRC/data/bc_test.parquet" "$CHECKOUT/data/" 2>/dev/null || true

source "$VENV/bin/activate" || fail "venv"
hf download Qwen/Qwen3-8B >/dev/null 2>&1 || huggingface-cli download Qwen/Qwen3-8B >/dev/null || fail "dl base"

# --- ckpt restore for RL arms ---
RESUME_FLAGS="trainer.resume_mode=disable"
if [ "$ARM" != norl ]; then
  [ -d "$HDFS_CKPT/global_step_$STEP" ] || fail "no HDFS ckpt step $STEP"
  mkdir -p "$LOCAL_CKPT"
  if [ ! -d "$LOCAL_CKPT/global_step_$STEP" ]; then
    log "restoring ckpt step $STEP from HDFS"
    cp -r "$HDFS_CKPT/global_step_$STEP" "$LOCAL_CKPT/" || fail "ckpt restore"
  fi
  echo "$STEP" > "$LOCAL_CKPT/latest_checkpointed_iteration.txt"
  RESUME_FLAGS="trainer.resume_mode=auto trainer.default_local_dir=$LOCAL_CKPT"
fi

# --- local judge shim ---
nohup python -u "$SRC/infra/judge_shim.py" \
  --upstream-marker "$MARK/INFRA_READY" \
  --port 8002 --log-dir "$OUTDIR" > "$OUTDIR/shim.log" 2>&1 &
SHIM_PID=$!
shim_ok=0
for i in $(seq 1 24); do curl -sf -m 5 http://127.0.0.1:8002/v1/models >/dev/null 2>&1 && { shim_ok=1; break; }; kill -0 $SHIM_PID 2>/dev/null || break; sleep 5; done
[ $shim_ok -eq 1 ] || { tail -20 "$OUTDIR/shim.log"; fail "shim not up"; }


export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
export OPENAI_URL=http://127.0.0.1:8002/chat
export LOCAL_SEARCH_URL=http://127.0.0.1:8002
export WANDB_MODE=offline
export PYTHONPATH=$CHECKOUT${PYTHONPATH:+:$PYTHONPATH}

VAL_FLAGS="actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.n=1"
[ "$MODE" = t1n4 ] && VAL_FLAGS="actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.val_kwargs.n=4"
ARMFLAGS=""
[ "$ARM" = grpo ] && ARMFLAGS="algorithm.adv_estimator=grpo ++actor_rollout_ref.rollout.plugin.process_reward=none"

cd "$CHECKOUT"
sed -e "s#trainer.experiment_name=test_run#trainer.experiment_name=valonly_${TAG}#" \
    -e "s#trainer.val_before_train=False#trainer.val_before_train=True#" \
    -e "s#trainer.val_only=False#trainer.val_only=True#" \
    -e "s#trainer.project_name=context_folding#trainer.project_name=context_folding trainer.validation_data_dir=${OUTDIR}/dump ${RESUME_FLAGS} ${VAL_FLAGS} ${ARMFLAGS} actor_rollout_ref.rollout.agent.agent_loop_config_path=${SRC}/infra/agent_loop_config.yaml#" \
    scripts/train_bc_qwen3_8b.sh > logs/launch_valonly.sh
log "launching val_only: $TAG"
bash logs/launch_valonly.sh 2>&1 | tee "$OUTDIR/valonly.log"
rc=${PIPESTATUS[0]}
kill $SHIM_PID 2>/dev/null
grep -q "val/avg_score" "$OUTDIR/valonly.log" || fail "no val metrics produced (exit $rc)"
grep -oE "'val[^']*': [0-9.]+" "$OUTDIR/valonly.log" | tail -20 > "$OUTDIR/val_metrics.txt"
cat "$OUTDIR/val_metrics.txt"
touch "$MARK/VALONLY_${TAG}_DONE"
log "DONE"
