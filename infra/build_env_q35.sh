#!/bin/bash
# Build the Qwen3.5 training venv  ~/xiaoxuan/envs/fold_train_q35   (migration plan WP2, NEXT_STEPS step 3)
#
# Stack = verl 0.9.1's canonical [vllm] extra (its pyproject / docker/Dockerfile.uv.cu130):
#   torch 2.11.0+cu130, vLLM 0.24.0 (PyPI wheel is already a cu130/torch-2.11 build), transformers 5.9.0,
#   verl 0.9.1 (pip, native qwen3_5 packed forward), flash-attn 2.8.3 (prebuilt cu130/torch2.11 from the verl
#   wheelhouse), flash-linear-attention 0.5.2 (GDN chunk kernels + fla causal conv, both used by verl's
#   qwen3_5.py; the separate `causal-conv1d` package is optional: verl's engine config
#   `causal_conv1d_implementation=fla`), tensordict 0.10, cupy-cuda13x 14.0.1 (ray collective / NCCL).
# Python 3.12 = the interpreter verl 0.9.1 and the wheelhouse are built/tested for (also what the pod-proven
#   supo-cu130 venv uses).
# Pods: the ark H100/A100 pods run driver R535; a cu130 stack needs the CUDA 13.0 forward-compat libcuda, which
#   image aliyun-va-hub.byted.org/arnold/modelchef-gpu:1.0.0.54 ships (verified 2026-09-08, SUPO probes:
#   matmul/cuDNN/flash-attn/NCCL/vLLM all pass on R535). The Qwen3-8B specs use 1.0.0.38 (CUDA 12.9 compat only)
#   -> every Qwen3.5 job spec must set image 1.0.0.54.
#
# Usage: bash infra/build_env_q35.sh [--force]     idempotent; writes $VENV/ok + infra/fold_train_q35.freeze.txt
set -uo pipefail
XD=/home/tiger/xiaoxuan; VENV=$XD/envs/fold_train_q35; SRC=$XD/Comp_Rubric
WHEELHOUSE=https://verl-project.github.io/verl-wheelhouse/simple
TORCH_INDEX=https://download.pytorch.org/whl/cu130
log() { echo "[build_q35 $(date '+%m-%d %H:%M:%S')] $*"; }
[ "${1:-}" = --force ] && rm -rf "$VENV"
[ -f "$VENV/ok" ] && { log "venv present ($VENV/ok) - nothing to do"; exit 0; }
df -h "$XD" | tail -1
[ -x "$VENV/bin/python" ] || uv venv "$VENV" --python 3.12 || exit 1
PY=$VENV/bin/python
UVPIP="uv pip install --python $PY"

log "core stack: torch 2.11.0+cu130, vllm 0.24.0, transformers 5.9.0, verl 0.9.1, fla 0.5.2 ..."
$UVPIP --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match \
  "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" \
  "vllm==0.24.0" "transformers==5.9.0" "verl[verl-core]==0.9.1" \
  "flash-linear-attention==0.5.2" "tensordict==0.10.0" "cupy-cuda13x==14.0.1" \
  "ray[default]>=2.41.0" wandb httpx openai pyzmq msgspec fastapi uvicorn \
  huggingface_hub hf_transfer pytest pandas "pyarrow>=19" datasets codetiming hydra-core \
  accelerate dill peft pybind11 pylatexenc torchdata tensorboard packaging liger-kernel unidiff flask \
  || { log "FATAL: core install failed"; exit 2; }

log "flash-attn 2.8.3 prebuilt (verl wheelhouse, no source build) ..."
$UVPIP --no-deps --no-build --index "$WHEELHOUSE" --index-strategy unsafe-best-match "flash-attn==2.8.3" \
  || { log "FATAL: flash-attn wheel install failed"; exit 3; }

log "verifying imports (CPU box: CUDA-only modules are reported, not required) ..."
# NOTE: run from a neutral cwd - inside the repo the vendored verl/ (0.7.0.dev) shadows pip verl 0.9.1 (sys.path[0]=cwd).
cd "$XD/envs" && $PY - <<'PYEOF' || { echo "[build_q35] FATAL: import check failed"; exit 4; }
import importlib, sys
import torch, transformers, verl, fla, tensordict, ray, numpy
print(f"torch {torch.__version__} (cuda build {torch.version.cuda}) | transformers {transformers.__version__} | verl {verl.__version__} | fla {fla.__version__} | tensordict {tensordict.__version__} | ray {ray.__version__} | numpy {numpy.__version__} | python {sys.version.split()[0]}")
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration  # noqa
import verl.models.transformers.qwen3_5  # verl's packed GDN forward
from verl.trainer.ppo.core_algos import register_adv_est  # estimator registry used by our plugin
from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa
print("required imports OK")
for m in ("vllm", "flash_attn", "cupy", "flashinfer"):
    try:
        mod = importlib.import_module(m); print(f"  optional-on-CPU import {m}: OK {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  optional-on-CPU import {m}: {type(e).__name__}: {str(e)[:120]}  (expected without a GPU; re-check on the pod)")
PYEOF

uv pip freeze --python "$PY" > "$SRC/infra/fold_train_q35.freeze.txt" && log "freeze -> infra/fold_train_q35.freeze.txt ($(wc -l < "$SRC/infra/fold_train_q35.freeze.txt") packages)"
touch "$VENV/ok"; du -sh "$VENV" | tail -1; df -h "$XD" | tail -1
log "DONE: $VENV"
