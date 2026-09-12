#!/bin/bash
env ARM=norl STEP=base MODE=greedy bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_val_only.sh
env ARM=foldgrpo STEP=50 MODE=t1n4 bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_val_only.sh
