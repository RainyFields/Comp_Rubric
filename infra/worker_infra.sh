#!/bin/bash
# FoldAgent replication — infra worker bootstrap (search server + gpt-oss-120b judge)
# Runs ATTACHED as the worker's foreground command; keeps worker alive with sleep loop.
# GPU plan: 0,1 = Qwen3-Embedding-8B search server | 2-5 = gpt-oss-120b vLLM TP4 | 6,7 = spare (Qwen3-8B eval serving)
set -uo pipefail

REPO=/home/tiger/xiaoxuan/Comp_Rubric
VENV=/home/tiger/xiaoxuan/envs/fold_infra
LOCAL=/tmp/fold_infra
MARK=${MARK:-$REPO/infra/markers}          # batch jobs: shared HDFS dir
JUDGE_MODEL=${JUDGE_MODEL:-openai/gpt-oss-120b}   # HF id, or a local snapshot dir (skips the download)
mkdir -p "$MARK" "$LOCAL"
rm -f "$MARK"/INFRA_FAILED "$MARK"/INFRA_DEGRADED_*   # keep a live INFRA_READY from a previous pod until this one is ready (consumers re-resolve per request)

log() { echo "[infra $(date +%H:%M:%S)] $*"; }
fail() { log "FAILED: $*"; touch "$MARK/INFRA_FAILED"; sleep 60; exit 1; }

# --- disk gate (memory: check disk before moving data) ---
df -h /tmp /home/tiger | tee "$MARK/df_at_start.txt"
AVAIL_GB=$(df -BG --output=avail /tmp | tail -1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge 150 ] || fail "only ${AVAIL_GB}G free on /tmp, need >=150G"

# --- GPU health gate ---
nvidia-smi -L | grep -q "^GPU 0" || fail "NVML broken on this node"
nvidia-smi -L | tee "$MARK/gpus.txt"

# --- env (uv venv; never python -m venv; never source-build flash-attn — not needed here) ---
export HF_HOME=$LOCAL/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_LOGGING_LEVEL=DEBUG
if [ ! -f "$VENV/ok" ]; then
  uv venv "$VENV" --python 3.11 || fail "uv venv"
  source "$VENV/bin/activate"
  uv pip install "vllm==0.11.0" datasets fastapi uvicorn "numpy<2.0.0" accelerate || fail "pip install"
  touch "$VENV/ok"
else
  source "$VENV/bin/activate"
fi
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)" || fail "import check"

# --- NVML loader-path gate: some nodes have libnvidia-ml.so.1 off the ld path (nvidia-smi still works) ---
if ! python -c "import pynvml; pynvml.nvmlInit()" 2>/dev/null; then
  log "NVML not loadable — searching for libnvidia-ml.so.1"
  NVML_DIR=$(find /usr/lib /usr/lib64 /lib /opt -name "libnvidia-ml.so.1" 2>/dev/null | head -1 | xargs -r dirname)
  if [ -n "$NVML_DIR" ]; then
    export LD_LIBRARY_PATH="$NVML_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    log "exported LD_LIBRARY_PATH=$NVML_DIR"
  fi
  python -c "import pynvml; pynvml.nvmlInit(); print('NVML recovered')" || { hostname -I > "$MARK/BAD_NODE_infra.txt"; fail "NVML unrecoverable on $(hostname -I | awk '{print \$1}')"; }
fi

# --- model + data downloads (idempotent via hf cache) ---
# Batch pods: seed the hf cache from HDFS (HF_HUB_SEED, e.g. fold-job-assets/hf_hub) so the search server never
# depends on huggingface.co reachability from the pod; anything downloaded fresh is copied back for the next pod.
HF_HUB_SEED=${HF_HUB_SEED:-}
if [ -n "$HF_HUB_SEED" ] && [ -d "$HF_HUB_SEED" ]; then
  mkdir -p "$HF_HOME/hub"; for d in "$HF_HUB_SEED"/*; do [ -d "$d" ] && [ ! -d "$HF_HOME/hub/$(basename "$d")" ] && { log "seeding $(basename "$d") from HDFS"; cp -r "$d" "$HF_HOME/hub/"; }; done
fi
log "downloading models/corpus in parallel (cached under $HF_HOME)"
t0=$(date +%s)
( hf download Tevatron/browsecomp-plus-corpus --repo-type dataset >/dev/null 2>&1 || huggingface-cli download Tevatron/browsecomp-plus-corpus --repo-type dataset >/dev/null ) & DL3=$!
( hf download miaolu3/browsecomp-plus corpus_embeddings.pkl --repo-type dataset >/dev/null 2>&1 || huggingface-cli download miaolu3/browsecomp-plus corpus_embeddings.pkl --repo-type dataset >/dev/null ) & DL4=$!
( hf download Qwen/Qwen3-Embedding-8B >/dev/null 2>&1 || huggingface-cli download Qwen/Qwen3-Embedding-8B >/dev/null ) & DL1=$!
if [ -f "$JUDGE_MODEL/config.json" ]; then true & DL2=$!; else
( hf download openai/gpt-oss-120b >/dev/null 2>&1 || huggingface-cli download openai/gpt-oss-120b >/dev/null ) & DL2=$!; fi
wait $DL1 || fail "dl embed model"
wait $DL2 || fail "dl gpt-oss-120b"
wait $DL3 || fail "dl corpus dataset"
wait $DL4 || fail "dl corpus embeddings"
log "downloads done in $(( $(date +%s) - t0 )) s: $(ls "$HF_HOME/hub" | tr '\n' ' ')"
if [ -n "$HF_HUB_SEED" ]; then mkdir -p "$HF_HUB_SEED"; for d in "$HF_HOME"/hub/datasets--* "$HF_HOME"/hub/models--Qwen--Qwen3-Embedding-8B; do
  [ -d "$d" ] && [ ! -d "$HF_HUB_SEED/$(basename "$d")" ] && { log "caching $(basename "$d") to HDFS"; cp -r "$d" "$HF_HUB_SEED/" || rm -rf "$HF_HUB_SEED/$(basename "$d")"; }; done; fi
df -h /tmp | tail -1

# gpt-oss judge: openai_harmony must load the o200k/cl100k tiktoken vocab; some pods (n214 pool) cannot reach
# openaipublic.blob.core.windows.net, so serve the vocab from the HDFS-staged copy via TIKTOKEN_ENCODINGS_BASE.
TIK_SRC=${TIKTOKEN_HDFS:-/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/tiktoken}
if [ -f "$TIK_SRC/o200k_base.tiktoken" ]; then
  mkdir -p /tmp/tiktoken && cp -n "$TIK_SRC"/*.tiktoken /tmp/tiktoken/ && export TIKTOKEN_ENCODINGS_BASE=/tmp/tiktoken && log "tiktoken vocab staged locally"
fi
# --- bind family: prefer IPv4 if the pod has one; else IPv6 dual-stack bind ---
IPV4=$(hostname -I | tr ' ' '\n' | grep -m1 '\.' || true)
if [ -n "$IPV4" ]; then HOST_BIND=0.0.0.0; MY_IP=$IPV4; else HOST_BIND="::"; MY_IP=$(hostname -I | awk '{print $1}'); fi
log "binding on $HOST_BIND, advertising $MY_IP"
# Health checks go through the ADVERTISED address (what consumers use). On some pods (dc61 A100 pool) a `::`
# uvicorn socket does not accept 127.0.0.1, so a loopback probe fails forever while the server is fine.
case "$MY_IP" in *:*) PROBE="[$MY_IP]";; *) PROBE=$MY_IP;; esac

# --- search server (downloads corpus+embeddings itself via datasets) ---
cd "$REPO/envs"
CUDA_VISIBLE_DEVICES=0,1 nohup python -u search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host "$HOST_BIND" --port 8000 > "$MARK/search_server.log" 2>&1 &
SEARCH_PID=$!

# --- judge: gpt-oss-120b on :8001 (eager on A100: skip cudagraph capture, faster init) ---
EXTRA_VLLM=""
nvidia-smi -L | grep -q A100 && EXTRA_VLLM="--enforce-eager"
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=2,3,4,5 nohup vllm serve "$JUDGE_MODEL" $EXTRA_VLLM \
  --tensor-parallel-size 4 --max-model-len 32768 \
  --served-model-name gpt-oss-120b gpt-5-nano \
  --host "$HOST_BIND" --port 8001 > "$MARK/vllm_judge.log" 2>&1 &
JUDGE_PID=$!

# --- health checks (search server must index corpus; allow long warmup) ---
ok_search=0; ok_judge=0
for i in $(seq 1 240); do  # up to 2h
  if [ $ok_search -eq 0 ] && curl -sf -m 20 "http://$PROBE:8000/search" -H 'Content-Type: application/json' -d '{"query":"test","k":1}' >/dev/null 2>&1; then
    ok_search=1; log "search server READY (t=${i}x30s) via $PROBE"; fi
  if [ $ok_judge -eq 0 ] && curl -sf -m 10 "http://$PROBE:8001/v1/models" >/dev/null 2>&1; then
    ok_judge=1; log "judge vLLM READY (t=${i}x30s) via $PROBE"; fi
  [ $ok_search -eq 1 ] && [ $ok_judge -eq 1 ] && break
  [ $((i % 10)) -eq 0 ] && log "waiting: search=$ok_search judge=$ok_judge | last search log: $(tail -c 300 "$MARK/search_server.log" | tr '\n' ' ' | cut -c1-200)"
  kill -0 $SEARCH_PID 2>/dev/null || { tail -30 "$MARK/search_server.log"; fail "search server died"; }
  kill -0 $JUDGE_PID  2>/dev/null || { tail -30 "$MARK/vllm_judge.log";  fail "judge vllm died"; }
  sleep 30
done
[ $ok_search -eq 1 ] && [ $ok_judge -eq 1 ] || fail "health checks timed out after 2h"

echo "$MY_IP" > "$MARK/INFRA_READY"
log "READY — search http://$MY_IP:8000  judge http://$MY_IP:8001/v1"

# keep worker alive; heartbeat every 10 min
while true; do
  date > "$MARK/heartbeat.txt"
  kill -0 $SEARCH_PID 2>/dev/null || { touch "$MARK/INFRA_DEGRADED_search"; }
  kill -0 $JUDGE_PID  2>/dev/null || { touch "$MARK/INFRA_DEGRADED_judge"; }
  sleep 600
done
