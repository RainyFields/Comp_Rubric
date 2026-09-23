#!/bin/bash
# GRPO tail-redo: steps 91-100 with live infra (steps 91-100 of original run were zero-reward no-ops)
mkdir -p /tmp/fold_ckpt/grpo
if [ ! -d /tmp/fold_ckpt/grpo/global_step_90 ]; then
  cp -r /mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt/grpo/global_step_90 /tmp/fold_ckpt/grpo/ || exit 1
fi
echo 90 > /tmp/fold_ckpt/grpo/latest_checkpointed_iteration.txt
exec env ARM=grpo STEPS=100 bash /home/tiger/xiaoxuan/Comp_Rubric/infra/worker_train.sh
