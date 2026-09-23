#!/bin/bash
# Self-contained eval chain: validation run first, then all pending configs. DONE-tagged runs skip.
S=/home/tiger/xiaoxuan/Comp_Rubric/infra/worker_val_selfcontained.sh
env ARM=foldgrpo STEP=100 MODE=greedy bash $S   # merge-validation vs trusted 0.267
env ARM=grpo STEP=100 MODE=greedy bash $S
env ARM=grpo STEP=100 MODE=t1n4   bash $S
env ARM=grpo STEP=50  MODE=greedy bash $S
env ARM=grpo STEP=50  MODE=t1n4   bash $S
env ARM=norl STEP=base MODE=greedy bash $S
env ARM=foldgrpo STEP=50 MODE=t1n4 bash $S
