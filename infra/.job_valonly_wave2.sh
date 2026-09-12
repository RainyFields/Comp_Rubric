#!/bin/bash
env ARM=grpo STEP=100 MODE=greedy bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_val_only.sh
env ARM=grpo STEP=100 MODE=t1n4   bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_val_only.sh
env ARM=grpo STEP=50  MODE=greedy bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_val_only.sh
env ARM=grpo STEP=50  MODE=t1n4   bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_val_only.sh
