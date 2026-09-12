#!/bin/bash
# Self-contained checkpoint eval: infra (search+judge) AND val_only on ONE pod.
# No cross-worker dependency — survives queue churn; each run skips if already DONE.
# GPUs: 0-3 trainer (FSDP world 4, rollout TP4) | 4 embed search | 5-6 judge TP2 | 7 spare
# Models load via model_merger HF export + resume_mode=disable (world-8 ckpts can't resume world-4).
# Params: ARM=foldgrpo|grpo|norl  STEP=<n|base>  MODE=greedy|t1n4  [TAGSUF=<suffix>] [JUDGE_MODEL=<hf id|local dir>]
set -uo pipefail

ARM=${ARM:?}; STEP=${STEP:?}; MODE=${MODE:?}
SRC=/home/tiger/xiaoxuan/FoldAgent
VENVT=/home/tiger/xiaoxuan/envs/fold_train
VENVI=/home/tiger/xiaoxuan/envs/fold_infra
MARK=$SRC/infra/markers
TAG=${ARM}_${STEP}_${MODE}${TAGSUF:-}_sc
OUTDIR=$SRC/results/valonly_$TAG
CHECKOUT=/home/tiger/xiaoxuan/fold_arms/valsc_${TAG}
log() { echo "[valsc:$TAG $(date +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; touch "$MARK/VALONLY_${TAG}_FAILED"; exit 1; }
[ -f "$MARK/VALONLY_${TAG}_DONE" ] && { log "already done, skipping"; exit 0; }
rm -f "$MARK/VALONLY_${TAG}_FAILED"; mkdir -p "$OUTDIR"

export TMPDIR=/tmp/fold_tmp && mkdir -p "$TMPDIR"
export HF_HUB_ENABLE_HF_TRANSFER=1
df -h /tmp | tail -1
# NVML gate with loader-path recovery
if ! "$VENVI/bin/python" -c "import pynvml; pynvml.nvmlInit()" 2>/dev/null; then
  NVML_DIR=$(find /usr/lib /usr/lib64 /lib /opt -name "libnvidia-ml.so.1" 2>/dev/null | head -1 | xargs -r dirname)
  [ -n "$NVML_DIR" ] && export LD_LIBRARY_PATH="$NVML_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  "$VENVI/bin/python" -c "import pynvml; pynvml.nvmlInit()" || { hostname -I > "$MARK/BAD_NODE_${TAG}.txt"; fail "NVML unrecoverable"; }
fi
nvidia-smi -L | head -1

# ---------- local infra: downloads in parallel ----------
export HF_HOME=/tmp/fold_hf
source "$VENVI/bin/activate"
( hf download Qwen/Qwen3-Embedding-8B >/dev/null 2>&1 || huggingface-cli download Qwen/Qwen3-Embedding-8B >/dev/null ) & D1=$!
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}   # HF id, or a local snapshot dir (skips the download)
if [ -f "$JUDGE_MODEL/config.json" ]; then true & D2=$!; else
( hf download openai/gpt-oss-120b >/dev/null 2>&1 || huggingface-cli download openai/gpt-oss-120b >/dev/null ) & D2=$!; fi
( hf download Qwen/Qwen3-8B >/dev/null 2>&1 || huggingface-cli download Qwen/Qwen3-8B >/dev/null ) & D3=$!
wait $D1 || fail "dl embed"; wait $D2 || fail "dl judge"; wait $D3 || fail "dl base"

# gpt-oss judge: openai_harmony must load the o200k/cl100k tiktoken vocab; some pods (n214 pool) cannot reach
# openaipublic.blob.core.windows.net, so serve the vocab from the HDFS-staged copy via TIKTOKEN_ENCODINGS_BASE.
TIK_SRC=${TIKTOKEN_HDFS:-/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/tiktoken}
if [ -f "$TIK_SRC/o200k_base.tiktoken" ]; then
  mkdir -p /tmp/tiktoken && cp -n "$TIK_SRC"/*.tiktoken /tmp/tiktoken/ && export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken && log "tiktoken vocab staged locally"
fi
cd "$SRC/envs"
CUDA_VISIBLE_DEVICES=4 nohup python search_server.py \
  --model Qwen/Qwen3-Embedding-8B --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 127.0.0.1 --port 8000 > "$OUTDIR/search.log" 2>&1 &
SEARCH_PID=$!
CUDA_VISIBLE_DEVICES=5,6 nohup vllm serve "$JUDGE_MODEL" --enforce-eager \
  --tensor-parallel-size 2 --max-model-len 16384 \
  --served-model-name gpt-oss-120b gpt-5-nano \
  --host 127.0.0.1 --port 8001 > "$OUTDIR/judge.log" 2>&1 &
JUDGE_PID=$!

# ---------- model prep (parallel with infra warmup) ----------
if [ "$ARM" = norl ]; then
  MODEL=Qwen/Qwen3-8B
else
  CKPT_ROOT=${CKPT_ROOT:-/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt}   # retrain campaign: .../ckpt_fix
  HDFS_ACTOR=$CKPT_ROOT/$ARM/global_step_$STEP/actor
  [ -d "$HDFS_ACTOR" ] || fail "no HDFS ckpt"
  MODEL=/tmp/hf_${ARM}_${STEP}
  if [ ! -f "$MODEL/config.json" ]; then
    cp -r "$HDFS_ACTOR" /tmp/actor_${ARM}_${STEP} || fail "hdfs copy"
    ( cd "$SRC" && PYTHONPATH=$SRC "$VENVT/bin/python" -m verl.model_merger merge \
        --backend fsdp --local_dir /tmp/actor_${ARM}_${STEP} --target_dir "$MODEL" ) > "$OUTDIR/merge.log" 2>&1 \
      || { tail -5 "$OUTDIR/merge.log"; fail "merge"; }
    cp -n /tmp/actor_${ARM}_${STEP}/huggingface/* "$MODEL/" 2>/dev/null || true
    rm -rf /tmp/actor_${ARM}_${STEP}
  fi
fi

# ---------- wait local infra healthy ----------
ok_s=0; ok_j=0
for i in $(seq 1 90); do
  [ $ok_s -eq 0 ] && curl -sf -m 5 "http://127.0.0.1:8000/search" -H 'Content-Type: application/json' -d '{"query":"t","k":1}' >/dev/null 2>&1 && ok_s=1 && log "search up"
  [ $ok_j -eq 0 ] && curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null 2>&1 && ok_j=1 && log "judge up"
  [ $ok_s -eq 1 ] && [ $ok_j -eq 1 ] && break
  kill -0 $SEARCH_PID 2>/dev/null || { tail -5 "$OUTDIR/search.log"; fail "search died"; }
  kill -0 $JUDGE_PID 2>/dev/null || { tail -5 "$OUTDIR/judge.log"; fail "judge died"; }
  sleep 20
done
[ $ok_s -eq 1 ] && [ $ok_j -eq 1 ] || fail "local infra not up in 30min"

# ---------- shim (local upstreams, no marker) ----------
source "$VENVT/bin/activate"
nohup python -u "$SRC/infra/judge_shim.py" --upstream "http://127.0.0.1:8001/v1" \
  --port 8002 --log-dir "$OUTDIR" > "$OUTDIR/shim.log" 2>&1 &
SHIM_PID=$!
for i in $(seq 1 24); do curl -sf -m 5 http://127.0.0.1:8002/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -sf -m 5 http://127.0.0.1:8002/v1/models >/dev/null || fail "shim not up"

# ---------- val_only via training stack, world 4 ----------
rm -rf "$CHECKOUT"; git clone -q "$SRC" "$CHECKOUT" || fail "clone"
mkdir -p "$CHECKOUT/data" "$CHECKOUT/logs"
cp -n "$SRC/data/"*.parquet "$CHECKOUT/data/" 2>/dev/null || true
export OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://127.0.0.1:8002/v1 OPENAI_URL=http://127.0.0.1:8002/chat
export LOCAL_SEARCH_URL=http://127.0.0.1:8000
export WANDB_MODE=offline PYTHONPATH=$CHECKOUT
export CUDA_VISIBLE_DEVICES=0,1,2,3

VAL_FLAGS="actor_rollout_ref.rollout.val_kwargs.do_sample=False actor_rollout_ref.rollout.val_kwargs.n=1"
[ "$MODE" = t1n4 ] && VAL_FLAGS="actor_rollout_ref.rollout.val_kwargs.do_sample=True actor_rollout_ref.rollout.val_kwargs.temperature=1.0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 actor_rollout_ref.rollout.val_kwargs.n=4"
ARMFLAGS=""
[ "$ARM" = grpo ] && ARMFLAGS="algorithm.adv_estimator=grpo ++actor_rollout_ref.rollout.plugin.process_reward=none"

cd "$CHECKOUT"
sed -e "s#MODEL_PATH=Qwen/Qwen3-8B#MODEL_PATH=$MODEL#" \
    -e "s#actor_rollout_ref.rollout.tensor_model_parallel_size=8#actor_rollout_ref.rollout.tensor_model_parallel_size=4#" \
    -e "s#trainer.n_gpus_per_node=8#trainer.n_gpus_per_node=4#" \
    -e "s#trainer.experiment_name=test_run#trainer.experiment_name=valsc_${TAG}#" \
    -e "s#trainer.val_before_train=False#trainer.val_before_train=True#" \
    -e "s#trainer.val_only=False#trainer.val_only=True#" \
    -e "s#trainer.project_name=context_folding#trainer.project_name=context_folding trainer.resume_mode=disable trainer.validation_data_dir=${OUTDIR}/dump ${VAL_FLAGS} ${ARMFLAGS} actor_rollout_ref.rollout.agent.agent_loop_config_path=${SRC}/infra/agent_loop_config.yaml#" \
    scripts/train_bc_qwen3_8b.sh > logs/launch_valsc.sh
log "launching self-contained val_only: $TAG (model=$MODEL)"
bash logs/launch_valsc.sh 2>&1 | tee "$OUTDIR/valonly.log"
kill $SHIM_PID 2>/dev/null
grep -q "val/avg_score" "$OUTDIR/valonly.log" || fail "no val metrics"
grep -oE "'val[^']*': [0-9.]+" "$OUTDIR/valonly.log" | tail -20 > "$OUTDIR/val_metrics.txt"
cat "$OUTDIR/val_metrics.txt"
touch "$MARK/VALONLY_${TAG}_DONE"
kill $SEARCH_PID $JUDGE_PID 2>/dev/null
log "DONE"
