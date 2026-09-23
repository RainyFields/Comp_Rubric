#!/bin/bash
env ARM=foldgrpo STEP=100 MODE=greedy bash /home/tiger/xiaoxuan/Comp_Rubric/infra/worker_val_only.sh
env ARM=foldgrpo STEP=100 MODE=t1n4   bash /home/tiger/xiaoxuan/Comp_Rubric/infra/worker_val_only.sh
env ARM=foldgrpo STEP=50  MODE=greedy bash /home/tiger/xiaoxuan/Comp_Rubric/infra/worker_val_only.sh
env ARM=foldgrpo STEP=50  MODE=t1n4   bash /home/tiger/xiaoxuan/Comp_Rubric/infra/worker_val_only.sh
env ARM=norl STEP=base MODE=greedy bash /home/tiger/xiaoxuan/Comp_Rubric/infra/worker_val_only.sh
env ARM=norl STEP=base MODE=t1n4   bash /home/tiger/xiaoxuan/Comp_Rubric/infra/worker_val_only.sh
