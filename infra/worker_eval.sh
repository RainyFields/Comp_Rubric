#!/bin/bash
# FoldAgent replication — eval worker (no-RL arm and later checkpoint evals)
# Usage (baked into per-job wrapper; --envs does not reach this script):
#   exec env RUN_TAG=norl_greedy TEMP=0.0 MODEL_PATH=Qwen/Qwen3-8B bash worker_eval.sh
# Serves MODEL_PATH on local GPUs 0-3 (TP4), runs judge shim, runs eval_bc.py
# against the infra node's search server + gpt-oss-120b judge.
set -uo pipefail

RUN_TAG=${RUN_TAG:?}; TEMP=${TEMP:?}; MODEL_PATH=${MODEL_PATH:?}
REPO=/home/tiger/xiaoxuan/FoldAgent
VENV=/home/tiger/xiaoxuan/envs/fold_infra
MARK=$REPO/infra/markers
OUT=$REPO/results/$RUN_TAG
mkdir -p "$OUT"
log() { echo "[eval:$RUN_TAG $(date +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; touch "$OUT/EVAL_FAILED"; sleep 30; exit 1; }
rm -f "$OUT/EVAL_FAILED" "$OUT/EVAL_DONE"

INFRA_IP=$(cat "$MARK/INFRA_READY" 2>/dev/null) || fail "no INFRA_READY marker"
case "$INFRA_IP" in *:*) INFRA_IP="[$INFRA_IP]";; esac  # bracket bare IPv6 for URLs
df -h /tmp | tail -1; nvidia-smi -L | grep -q "^GPU 0" || fail "NVML broken"

export HF_HOME=/tmp/fold_eval_hf
export TMPDIR=/tmp/fold_tmp && mkdir -p "$TMPDIR"   # short, existing tmp for vLLM zmq ipc sockets
source "$VENV/bin/activate" || fail "venv activate"
# eval-side extras (idempotent, tiny)
uv pip install -q openai httpx omegaconf hydra-core pandas pyarrow tqdm || fail "pip extras"

# agent model serving (skip download if MODEL_PATH is a local dir, e.g. an HDFS-pulled ckpt)
[ -d "$MODEL_PATH" ] || hf download "$MODEL_PATH" >/dev/null 2>&1 || huggingface-cli download "$MODEL_PATH" >/dev/null || fail "dl $MODEL_PATH"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup vllm serve "$MODEL_PATH" \
  --tensor-parallel-size 4 --max-model-len 40960 \
  --host 127.0.0.1 --port 8003 > "$OUT/vllm_agent.log" 2>&1 &
AGENT_PID=$!

nohup python "$REPO/infra/judge_shim.py" \
  --upstream "http://$INFRA_IP:8001/v1" \
  --agent-upstream "http://127.0.0.1:8003/v1" \
  --force-agent-temperature "$TEMP" \
  --port 8002 --log-dir "$OUT" > "$OUT/shim.log" 2>&1 &
SHIM_PID=$!

for i in $(seq 1 60); do
  curl -sf -m 5 http://127.0.0.1:8003/v1/models >/dev/null 2>&1 && break
  kill -0 $AGENT_PID 2>/dev/null || { tail -30 "$OUT/vllm_agent.log"; fail "agent vllm died"; }
  sleep 30
done
curl -sf -m 5 http://127.0.0.1:8003/v1/models >/dev/null || fail "agent vllm not up after 30min"
curl -sf -m 5 http://127.0.0.1:8002/v1/models >/dev/null || { tail -20 "$OUT/shim.log"; fail "shim not up"; }
curl -sf -m 10 "http://$INFRA_IP:8000/search" -H 'Content-Type: application/json' -d '{"query":"test","k":1}' >/dev/null || fail "infra search unreachable"
log "serving up: agent=$MODEL_PATH temp=$TEMP infra=$INFRA_IP"

cd "$REPO"
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
export OPENAI_URL=http://127.0.0.1:8002/chat
export LOCAL_SEARCH_URL=http://$INFRA_IP:8000

python scripts/eval_bc.py \
  --data_path data/bc_test.parquet \
  --model_name "$MODEL_PATH" \
  --num_workers 32 \
  --workflow search_branch \
  --prompt_length 8192 \
  --response_length 32768 \
  --max_turn 100 \
  --val_max_turn 100 \
  --max_session 10 \
  --val_max_session 10 \
  --local_search_url "http://$INFRA_IP:8000" \
  --output_dir "$OUT" 2>&1 | tee "$OUT/eval.log"
rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] || fail "eval_bc.py exit $rc"
touch "$OUT/EVAL_DONE"
log "DONE — results in $OUT"
