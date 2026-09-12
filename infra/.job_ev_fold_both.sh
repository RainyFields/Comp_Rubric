#!/bin/bash
env ARM=foldgrpo STEP=50 bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_eval_ckpt.sh
env ARM=foldgrpo STEP=100 bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_eval_ckpt.sh
