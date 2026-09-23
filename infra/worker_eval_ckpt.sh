#!/bin/bash
# FoldAgent replication — checkpoint eval battery (one worker per checkpoint)
# Params (bake into wrapper): ARM=foldgrpo|grpo|norl  STEP=50|100|base  [SKIP_GREEDY=1]
# Flow: HDFS ckpt -> model_merger (FSDP->HF) -> vLLM TP4 -> greedy + 4x T1.0 evals -> results/
set -uo pipefail

ARM=${ARM:?}; STEP=${STEP:?}; SKIP_GREEDY=${SKIP_GREEDY:-0}
REPO=/home/tiger/xiaoxuan/Comp_Rubric
VENV_INFRA=/home/tiger/xiaoxuan/envs/fold_infra
VENV_TRAIN=/home/tiger/xiaoxuan/envs/fold_train
MARK=$REPO/infra/markers
TAG=${ARM}_step${STEP}
log() { echo "[evalckpt:$TAG $(date +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; touch "$MARK/EVALCKPT_${TAG}_FAILED"; sleep 30; exit 1; }
rm -f "$MARK/EVALCKPT_${TAG}_FAILED" "$MARK/EVALCKPT_${TAG}_DONE"

export TMPDIR=/tmp/fold_tmp && mkdir -p "$TMPDIR"
export HF_HOME=/tmp/fold_eval_hf
df -h /tmp | tail -1; nvidia-smi -L | grep -q "^GPU 0" || fail "NVML broken"
INFRA_IP=$(cat "$MARK/INFRA_READY" 2>/dev/null) || fail "no INFRA_READY marker"
case "$INFRA_IP" in *:*) INFRA_IP="[$INFRA_IP]";; esac

# --- model prep ---
if [ "$ARM" = norl ]; then
  MODEL=Qwen/Qwen3-8B
  source "$VENV_INFRA/bin/activate" || fail "venv"
  hf download "$MODEL" >/dev/null 2>&1 || huggingface-cli download "$MODEL" >/dev/null || fail "dl base model"
else
  HDFS_ACTOR=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt/$ARM/global_step_$STEP/actor
  [ -d "$HDFS_ACTOR" ] || fail "missing HDFS ckpt $HDFS_ACTOR"
  LOCAL_ACTOR=/tmp/ckpt_${TAG}/actor
  MODEL=/tmp/hf_${TAG}
  if [ ! -f "$MODEL/config.json" ]; then
    log "copying ckpt from HDFS"
    mkdir -p "$(dirname "$LOCAL_ACTOR")"
    cp -r "$HDFS_ACTOR" "$LOCAL_ACTOR" || fail "hdfs copy"
    log "merging FSDP -> HF"
    cd "$REPO" && PYTHONPATH=$REPO "$VENV_TRAIN/bin/python" -m verl.model_merger merge \
      --backend fsdp --local_dir "$LOCAL_ACTOR" --target_dir "$MODEL" > /tmp/merge_${TAG}.log 2>&1 \
      || { tail -20 /tmp/merge_${TAG}.log; fail "model merge"; }
    # merger emits weights; tokenizer/config come from ckpt huggingface dir
    cp -n "$LOCAL_ACTOR/huggingface/"* "$MODEL/" 2>/dev/null || true
    rm -rf "/tmp/ckpt_${TAG}"   # free disk
  fi
  source "$VENV_INFRA/bin/activate" || fail "venv"
fi

# --- serve agent model once ---
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve "$MODEL" \
  --tensor-parallel-size 4 --max-model-len 40960 \
  --host 127.0.0.1 --port 8003 > /tmp/vllm_${TAG}.log 2>&1 &
AGENT_PID=$!
for i in $(seq 1 60); do
  curl -sf -m 5 http://127.0.0.1:8003/v1/models >/dev/null 2>&1 && break
  kill -0 $AGENT_PID 2>/dev/null || { tail -30 /tmp/vllm_${TAG}.log; fail "agent vllm died"; }
  sleep 30
done
curl -sf -m 5 http://127.0.0.1:8003/v1/models >/dev/null || fail "agent vllm not up in 30min"
curl -sf -m 10 "http://$INFRA_IP:8000/search" -H 'Content-Type: application/json' -d '{"query":"t","k":1}' >/dev/null || fail "infra search unreachable"

export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
export OPENAI_URL=http://127.0.0.1:8002/chat
export LOCAL_SEARCH_URL=http://$INFRA_IP:8000

run_eval() {  # $1=run_name $2=temperature
  local OUT=$REPO/results/${TAG}_$1
  [ -f "$OUT/EVAL_DONE" ] && { log "skip $1 (already done)"; return 0; }
  mkdir -p "$OUT"
  nohup python "$REPO/infra/judge_shim.py" \
    --upstream "http://$INFRA_IP:8001/v1" \
    --agent-upstream "http://127.0.0.1:8003/v1" \
    --force-agent-temperature "$2" \
    --port 8002 --log-dir "$OUT" > "$OUT/shim.log" 2>&1 &
  local SHIM_PID=$!
  for i in $(seq 1 24); do curl -sf -m 5 http://127.0.0.1:8002/v1/models >/dev/null 2>&1 && break; sleep 5; done
  curl -sf -m 5 http://127.0.0.1:8002/v1/models >/dev/null || { kill $SHIM_PID 2>/dev/null; fail "shim not up for $1"; }
  log "eval $1 (temp=$2) starting"
  ( cd "$REPO" && python scripts/eval_bc.py \
      --data_path data/bc_test.parquet \
      --model_name "$MODEL" \
      --num_workers 32 \
      --workflow search_branch \
      --prompt_length 8192 --response_length 32768 \
      --max_turn 100 --val_max_turn 100 \
      --max_session 10 --val_max_session 10 \
      --local_search_url "http://$INFRA_IP:8000" \
      --output_dir "$OUT" 2>&1 | tee "$OUT/eval.log" )
  local rc=${PIPESTATUS[0]}
  kill $SHIM_PID 2>/dev/null; sleep 2
  [ $rc -eq 0 ] || fail "eval $1 exit $rc"
  touch "$OUT/EVAL_DONE"
  grep -E "Avg Score|bc_test" "$OUT/eval.log" | tail -4
}

[ "$SKIP_GREEDY" = 1 ] || run_eval greedy 0.0
run_eval t1_1 1.0
run_eval t1_2 1.0
run_eval t1_3 1.0
run_eval t1_4 1.0

touch "$MARK/EVALCKPT_${TAG}_DONE"
kill $AGENT_PID 2>/dev/null; sleep 10; pkill -f "vllm serve" 2>/dev/null; sleep 5
log "ALL EVALS DONE for $TAG"
