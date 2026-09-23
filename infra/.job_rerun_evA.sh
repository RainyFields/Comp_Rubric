#!/bin/bash
S=/home/tiger/xiaoxuan/Comp_Rubric/infra/worker_val_selfcontained.sh
export TAGSUF=_r2
env ARM=foldgrpo STEP=100 MODE=greedy bash $S
env ARM=foldgrpo STEP=100 MODE=t1n4   bash $S
env ARM=foldgrpo STEP=50  MODE=greedy bash $S
env ARM=foldgrpo STEP=50  MODE=t1n4   bash $S
env ARM=norl STEP=base MODE=greedy bash $S
