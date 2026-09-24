#!/bin/bash
# Shared pod bootstrap for the FoldAgent batch jobs (sourced by the entrypoints).
# Restores the devbox venvs + repo from md5-verified HDFS tarballs at the same absolute paths.
ASSETS=/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets
XD=/home/tiger/xiaoxuan
# Training/rollout venv: FOLD_VENV in the job env_map (fold_train = Qwen3-8B stack, default; fold_train_q35 = Qwen3.5 stack,
# needs image modelchef-gpu:1.0.0.54). Exported as TRAIN_VENV for worker_train.sh / worker_val_selfcontained.sh.
export TRAIN_VENV=${FOLD_VENV:-fold_train}
SRC=$XD/Comp_Rubric   # repo dir inside foldagent-repo.tar.gz (renamed from FoldAgent 2026-09-23)
log() { echo "[foldjob $(date '+%m-%d %H:%M:%S')] $*"; }
ls "$ASSETS" >/dev/null 2>&1 || { log "FATAL: HDFS fuse not available"; exit 41; }
df -h /tmp | tail -1; nproc; free -g | head -2
mkdir -p "$XD/envs" /tmp/fold_tmp
restore_venv() {
  [ -x "$XD/envs/$1/bin/python" ] && { log "venv $1 present"; return 0; }
  log "restoring venv $1 ..."; local t=/tmp/$1.tar.gz
  cp "$ASSETS/$1.tar.gz" "$t" || return 1
  [ "$(md5sum "$t" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$ASSETS/$1.tar.gz.md5")" ] || { log "md5 mismatch $1"; return 1; }
  # tarballs rooted at / (first member home/...) also carry the uv-managed interpreter the venv's bin/python links to -> extract at /
  case "$(tar tzf "$t" 2>/dev/null | head -1)" in home/*) tar xzf "$t" -C / ;; *) tar xzf "$t" -C "$XD/envs" ;; esac && rm -f "$t"
  [ -x "$XD/envs/$1/bin/python" ] || { log "venv $1: bin/python missing after restore (dangling interpreter link?)"; return 1; }
}
restore_repo() {
  log "restoring repo ..."; rm -rf "$SRC"; tar xzf "$ASSETS/foldagent-repo.tar.gz" -C "$XD" || return 1
  ( cd "$SRC" && git log --oneline -1 ); mkdir -p "$SRC/infra/markers" "$SRC/results"
}
copy_judge() {  # HDFS snapshot -> /tmp/models/gpt-oss-120b (top-level files only; original/ + .cache/ skipped)
  JUDGE=/tmp/models/gpt-oss-120b; SRCJ=/mnt/hdfs/mlsys/users/xiaoxuan/models/gpt-oss-120b
  [ -f "$JUDGE/config.json" ] && { export JUDGE_MODEL=$JUDGE; return 0; }
  log "copying gpt-oss-120b from HDFS ..."; mkdir -p "$JUDGE"; local t0=$(date +%s)
  find "$SRCJ" -maxdepth 1 -type f -printf '%f\n' | xargs -P 8 -I{} cp "$SRCJ/{}" "$JUDGE"/ || return 1
  for f in $(find "$SRCJ" -maxdepth 1 -type f -printf '%f\n'); do
    [ "$(stat -c %s "$JUDGE/$f")" = "$(stat -c %s "$SRCJ/$f")" ] || { log "judge size mismatch: $f"; return 1; }
  done
  log "judge copy done in $(( $(date +%s) - t0 )) s"; export JUDGE_MODEL=$JUDGE
}
gpu_preflight() {
  "$XD/envs/${TRAIN_VENV:-fold_train}/bin/python" - <<'PYEOF'
import sys, socket, torch
n = torch.cuda.device_count()
if n < 8: sys.exit(f"preflight FAILED: only {n} GPUs")
torch.zeros(1, device="cuda:0")
print(f"[foldjob] preflight OK: {n} GPUs on {socket.gethostname()}, torch {torch.__version__} (cuda {torch.version.cuda}) from {sys.executable}", flush=True)
PYEOF
}
