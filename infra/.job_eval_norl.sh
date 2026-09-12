#!/bin/bash
exec env RUN_TAG=norl_greedy TEMP=0.0 MODEL_PATH=Qwen/Qwen3-8B bash /home/tiger/xiaoxuan/FoldAgent/infra/worker_eval.sh
