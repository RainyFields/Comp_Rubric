#!/bin/bash
# 1-GPU smoke of the Qwen3.5 stack (venv fold_train_q35, image modelchef-gpu:1.0.0.54 = CUDA 13.0 compat on R535).
# Gate items from docs/handoff/QWEN35_HANDOFF.md §3 step 3: torch cu130 on the pod, flash-attn 2.8.3, fla GDN kernel,
# HF load of Qwen3_5ForCausalLM (bf16) + forward with the fast path, vllm serve TP=1 + one enable_thinking completion
# (no <think> opener expected). TP=8 needs an 8-GPU pod (not part of this job). Log mirrored to $HDFS_OUT.
set -uo pipefail
source /mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/fold_common_bootstrap.sh   # TRAIN_VENV from FOLD_VENV
MODEL_SRC=${FOLD_MODEL_PATH:-/mnt/hdfs/mlsys/models/Qwen3.5-9B}
HDFS_OUT=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/results/smoke_q35_venv_$(date +%Y%m%d_%H%M)
mkdir -p "$HDFS_OUT"; LOG=/tmp/smoke_q35.log; exec > >(tee -a "$LOG") 2>&1
finish() { rc=$1; log "smoke finished rc=$rc"; cp "$LOG" "$HDFS_OUT/smoke.log"; ls /tmp/vllm_serve.log >/dev/null 2>&1 && tail -c 300000 /tmp/vllm_serve.log > "$HDFS_OUT/vllm_serve.tail.log"; echo "$rc" > "$HDFS_OUT/RC"; exit $rc; }
log "smoke node=$(hostname) image=${ARNOLD_BASE_IMAGE:-?} venv=$TRAIN_VENV model=$MODEL_SRC out=$HDFS_OUT"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; ls -d /usr/local/cuda* 2>/dev/null; ls /usr/local/cuda/compat 2>/dev/null | grep -o 'libcuda.so.[0-9.]*' | head -1
restore_venv "$TRAIN_VENV" || finish 43
PY=$XD/envs/$TRAIN_VENV/bin/python; export TMPDIR=/tmp/fold_tmp HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
log "staging model to /tmp/models ..."; t0=$(date +%s); mkdir -p /tmp/models && cp -r "$MODEL_SRC" /tmp/models/ || finish 45
MODEL=/tmp/models/$(basename "$MODEL_SRC"); log "model staged in $(( $(date +%s) - t0 )) s ($(du -sh $MODEL | cut -f1))"

log "===== A. torch / cuDNN / flash-attn / fla GDN kernel ====="
$PY -u - <<'PYEOF' || finish 51
import time, torch, sys
print("torch", torch.__version__, "cuda build", torch.version.cuda, "device", torch.cuda.get_device_name(0), "driver_api", getattr(torch.cuda, "driver_version", lambda: "?")(), flush=True)
a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16); torch.cuda.synchronize(); t = time.time()
for _ in range(20): c = a @ a
torch.cuda.synchronize(); dt = time.time() - t; print(f"matmul bf16 8192^2 x20: {dt:.2f}s = {20*2*8192**3/dt/1e12:.0f} TFLOP/s", flush=True)
print("cudnn conv:", torch.nn.functional.conv2d(torch.randn(1,3,64,64,device="cuda"), torch.randn(8,3,3,3,device="cuda")).shape, flush=True)
from flash_attn import flash_attn_func, flash_attn_varlen_func
q = torch.randn(2, 1024, 16, 128, device="cuda", dtype=torch.bfloat16); o = flash_attn_func(q, q, q, causal=True); torch.cuda.synchronize()
qv = torch.randn(2048, 16, 128, device="cuda", dtype=torch.bfloat16); cu = torch.tensor([0, 1024, 2048], device="cuda", dtype=torch.int32)
ov = flash_attn_varlen_func(qv, qv, qv, cu, cu, 1024, 1024, causal=True); torch.cuda.synchronize(); print("flash_attn func+varlen OK", o.shape, ov.shape, flush=True)
import fla; from fla.ops.gated_delta_rule import chunk_gated_delta_rule
B, T, H, D = 1, 2048, 16, 128
q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16); k = torch.randn_like(q); v = torch.randn_like(q)
g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda", dtype=torch.float32)); beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16)
out = chunk_gated_delta_rule(q, k, v, g, beta); out = out[0] if isinstance(out, tuple) else out; torch.cuda.synchronize()
print("fla", fla.__version__, "chunk_gated_delta_rule OK", tuple(out.shape), "finite", bool(torch.isfinite(out).all()), flush=True)
from fla.modules.convolution import causal_conv1d as fla_causal_conv1d; print("fla causal_conv1d import OK (verl causal_conv1d_implementation=fla)", flush=True)
try:
    import causal_conv1d; print("causal_conv1d package present", causal_conv1d.__version__)
except Exception as e: print("causal_conv1d package absent (expected):", type(e).__name__)
print("A OK", flush=True)
PYEOF

log "===== B. NCCL world-1 init ====="
$PY -u - <<'PYEOF' || log "B: NCCL check failed (non-fatal for this smoke)"
import os, socket, torch, torch.distributed as dist
sk = socket.socket(); sk.bind(("127.0.0.1", 0)); port = sk.getsockname()[1]; sk.close()
os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK="0", WORLD_SIZE="1")
dist.init_process_group("nccl"); t = torch.ones(1024, device="cuda"); dist.all_reduce(t); torch.cuda.synchronize(); print("nccl all_reduce OK", float(t[0]), flush=True)
print("B OK", flush=True)
PYEOF

log "===== C. HF Qwen3_5ForCausalLM load + forward (fast path?) ====="
MODEL=$MODEL $PY -u - <<'PYEOF' || finish 53
import os, time, logging, io, torch, transformers
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
buf = io.StringIO(); h = logging.StreamHandler(buf); logging.getLogger("transformers").addHandler(h); logging.getLogger("transformers").setLevel(logging.WARNING)
transformers.logging.set_verbosity_warning()
M = os.environ["MODEL"]; tok = AutoTokenizer.from_pretrained(M)
gp = tok.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=False)
print("generation prompt tail:", repr(gp[-40:]), "| prefilled <think>:", gp.endswith("<think>\n"), flush=True)
t = time.time(); model = Qwen3_5ForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda"); torch.cuda.synchronize()
print(f"loaded {type(model).__name__} in {time.time()-t:.0f}s; params {sum(p.numel() for p in model.parameters())/1e9:.2f}B; gpu mem {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
lin = [m for m in model.modules() if type(m).__name__ == "Qwen3_5GatedDeltaNet"]
print("GatedDeltaNet layers:", len(lin), "| causal_conv1d_fn:", getattr(lin[0], "causal_conv1d_fn", None) is not None, "| chunk_gated_delta_rule:", getattr(lin[0], "chunk_gated_delta_rule", None), flush=True)
msgs = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in ("Name three prime numbers.", "What is the capital of France? Think briefly.")]
enc = tok(msgs, return_tensors="pt", padding=True, padding_side="left").to("cuda")
with torch.no_grad():
    t = time.time(); out = model(**enc); torch.cuda.synchronize(); print(f"forward {tuple(enc.input_ids.shape)} in {time.time()-t:.2f}s; logits finite {bool(torch.isfinite(out.logits).all())}; vocab {out.logits.shape[-1]}", flush=True)
    t = time.time(); gen = model.generate(**enc, max_new_tokens=64, do_sample=False); torch.cuda.synchronize()
txt = tok.decode(gen[0, enc.input_ids.shape[1]:], skip_special_tokens=False)
print(f"greedy 64 tokens in {time.time()-t:.1f}s; sample starts: {txt[:160]!r}", flush=True)
print("starts with <think>:", txt.lstrip().startswith("<think>"), "| contains </think>:", "</think>" in txt, flush=True)
w = buf.getvalue(); print("transformers warnings during load/forward:", repr(w[:600]) if w else "none", flush=True)
print("FAST PATH WARNING PRESENT" if "fast path" in w.lower() or "fallback" in w.lower() else "no fast-path fallback warning", flush=True)
print("C OK", flush=True)
PYEOF

log "===== D. vllm serve TP=1 + chat completion with enable_thinking ====="
source "$XD/envs/$TRAIN_VENV/bin/activate"
nohup vllm serve "$MODEL" --served-model-name qwen35 --tensor-parallel-size 1 --max-model-len 32768 \
  --gpu-memory-utilization 0.85 --additional-config '{"gdn_prefill_backend":"triton"}' --host 127.0.0.1 --port 8010 > /tmp/vllm_serve.log 2>&1 &
VPID=$!; up=0
for i in $(seq 1 60); do curl -sf -m 5 http://127.0.0.1:8010/v1/models >/dev/null 2>&1 && { up=1; break; }; kill -0 $VPID 2>/dev/null || break; sleep 10; done
[ $up -eq 1 ] || { log "vllm serve did not come up"; tail -40 /tmp/vllm_serve.log; finish 54; }
log "vllm up after ~$((i*10))s; version: $(vllm --version 2>/dev/null | tail -1)"
$PY -u - <<'PYEOF' || { tail -40 /tmp/vllm_serve.log; finish 55; }
import json, time, urllib.request
def chat(body):
    req = urllib.request.Request("http://127.0.0.1:8010/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time(); r = json.load(urllib.request.urlopen(req, timeout=600)); return r, time.time() - t
body = {"model": "qwen35", "messages": [{"role": "user", "content": "Which is larger, 9.11 or 9.9? Answer briefly."}], "max_tokens": 1024, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}}
r, dt = chat(body); m = r["choices"][0]["message"]; c = m.get("content") or ""; rc = m.get("reasoning_content") or m.get("reasoning") or ""
print(f"thinking=True: {r['usage']} in {dt:.1f}s; finish={r['choices'][0]['finish_reason']}", flush=True)
print("content starts:", repr(c[:200]), flush=True); print("reasoning field:", repr(rc[:120]) if rc else "none", flush=True)
print("no <think> opener in content:", not c.lstrip().startswith("<think>"), "| </think> in content:", "</think>" in c, flush=True)
body["chat_template_kwargs"] = {"enable_thinking": False}; body["max_tokens"] = 256
r, dt = chat(body); c2 = r["choices"][0]["message"].get("content") or ""; print(f"thinking=False: {r['usage']} in {dt:.1f}s; content starts: {c2[:120]!r}", flush=True)
# FoldAgent-style bare function-call prompt: does the model emit <function=...> blocks?
body = {"model": "qwen35", "messages": [{"role": "system", "content": "You are a search agent. To search, output exactly one call: <function=search>\n<parameter=query>YOUR QUERY</parameter>\n</function>"}, {"role": "user", "content": "Q: Which theater was built by a local during the Depression?"}], "max_tokens": 2048, "temperature": 0, "chat_template_kwargs": {"enable_thinking": True}}
r, dt = chat(body); c3 = r["choices"][0]["message"].get("content") or ""
print(f"agent prompt: {r['usage']} in {dt:.1f}s; has <function=search>: {'<function=search>' in c3}; has <tool_call>: {'<tool_call>' in c3}; tail: {c3[-200:]!r}", flush=True)
print("D OK", flush=True)
PYEOF
kill $VPID 2>/dev/null; sleep 3
log "ALL SMOKE STEPS PASSED"; finish 0
