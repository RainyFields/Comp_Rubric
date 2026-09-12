#!/bin/bash
# Batch-job entrypoint: shared infra pod (Qwen3-Embedding-8B search server :8000 + gpt-oss-120b judge :8001).
# Publishes its IP in $MARK/INFRA_READY on HDFS; training pods' judge_shim re-resolves it per request.
# This job never exits on its own (worker_infra.sh heartbeats forever) — STOP IT when both arms are done.
set -uo pipefail
source /mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/fold_common_bootstrap.sh
export MARK=${FOLD_MARK:-/mnt/hdfs/mlsys/xiaoxuan/fold_replication/markers_fix}
export HF_HUB_SEED=$ASSETS/hf_hub   # HDFS-cached hf hub entries (corpus, embeddings, embed model)
mkdir -p "$MARK"
log "infra job node=$(hostname) MARK=$MARK"
restore_venv fold_infra || exit 43
restore_venv fold_train || exit 43     # preflight uses it; cheap (2 min)
restore_repo || exit 44
copy_judge || exit 45
gpu_preflight || { log "preflight failed, exit 42"; exit 42; }
# worker_infra.sh: search on GPUs 0,1; judge TP4 on 2-5 (enforce-eager on A100); INFRA_READY = advertised IP
exec bash "$SRC/infra/worker_infra.sh"
