#!/bin/bash
# Usage: bash infra/stage_env_q35.sh   (after infra/build_env_q35.sh; overwrites fold_train_q35.tar.gz/.md5/.commit on HDFS)
# tarball ~/xiaoxuan/envs/fold_train_q35 -> HDFS fold-job-assets/fold_train_q35.tar.gz (+ .md5), same layout as fold_train.tar.gz
set -uo pipefail
XD=/home/tiger/xiaoxuan; ASSETS=/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets; T=/tmp/fold_train_q35.tar.gz
log() { echo "[tar_q35 $(date '+%m-%d %H:%M:%S')] $*"; }
df -h /tmp | tail -1
log "compressing ..."
if command -v pigz >/dev/null; then tar -C "$XD/envs" -cf - fold_train_q35 | pigz -p 48 > "$T"; else tar -C "$XD/envs" -czf "$T" fold_train_q35; fi || { log "FATAL tar"; exit 1; }
ls -la "$T"; log "md5 ..."; (cd /tmp && md5sum fold_train_q35.tar.gz > fold_train_q35.tar.gz.md5); cat /tmp/fold_train_q35.tar.gz.md5
log "copy to HDFS ..."; cp "$T" "$ASSETS/fold_train_q35.tar.gz" && cp /tmp/fold_train_q35.tar.gz.md5 "$ASSETS/fold_train_q35.tar.gz.md5" || { log "FATAL cp"; exit 2; }
log "verify HDFS md5 ..."; h=$(md5sum "$ASSETS/fold_train_q35.tar.gz" | cut -d' ' -f1); l=$(cut -d' ' -f1 "$ASSETS/fold_train_q35.tar.gz.md5")
[ "$h" = "$l" ] && { log "HDFS md5 OK $h"; (cd "$XD/Comp_Rubric" && git log -1 --format=%H) > "$ASSETS/fold_train_q35.tar.gz.commit"; ls -la "$ASSETS" | grep q35; rm -f "$T"; log "DONE"; } || { log "FATAL md5 mismatch hdfs=$h local=$l"; exit 3; }
