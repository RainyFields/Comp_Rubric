#!/bin/bash
# CompactionGRPO baseline: the CompactionRL rollout (agents/compaction_agent.py, trainable summaries,
# token-mean loss) with the FoldAgent arms' critic-free group-relative advantage (rollout-level
# normalisation broadcast to every compaction segment; position correction is the identity at
# gamma = lam = 1). Protocol-matched to GRPO/FoldGRPO. See docs/baselines/README.md.

PROMPT_LENGTH=8192
RESPONSE_LENGTH=32768
MAX_LENGTH=40960
MODEL_PATH=Qwen/Qwen3-8B
TRAIN_DATA_PATH=data/bc_train.parquet
TEST_DATA_PATH=data/bc_test.parquet

# --- CompactionRL rollout knobs (paper defaults where the FoldAgent stack allows; see docs/baselines/README.md) ---
MAX_COMPACTIONS=${MAX_COMPACTIONS:-3}            # at most three compactions per rollout -> 4x effective budget
VAL_MAX_COMPACTIONS=${VAL_MAX_COMPACTIONS:-3}    # 0 = single-window (x1) evaluation
COMPACTION_THRESHOLD=${COMPACTION_THRESHOLD:-8192}   # T_comp: remaining generated-token budget that triggers compaction (shakeout: 6144 rolled back a step before ~every summary)
TAIL_STEPS=${TAIL_STEPS:-2}                      # k recent (assistant, observation) steps kept verbatim
SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-2048}
TRAIN_SUMMARY=${TRAIN_SUMMARY:-True}             # False = "w/o summary training" ablation
MASK_UNFINISHED=${MASK_UNFINISHED:-False}        # paper: budget-exhausted rollouts train with reward 0 (FoldAgent arms mask them; shakeout: 63% of rollouts)


python -m scripts.train_fold \
  algorithm.adv_estimator=compaction_grpo \
  actor_rollout_ref.rollout.agent.default_agent_loop=compaction_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_LENGTH} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  data.train_files=${TRAIN_DATA_PATH} \
  data.val_files=${TEST_DATA_PATH} \
  data.train_batch_size=32 \
  data.max_prompt_length=${PROMPT_LENGTH} \
  data.max_response_length=${RESPONSE_LENGTH} \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_LENGTH} \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${MAX_LENGTH} \
  +actor_rollout_ref.rollout.plugin.workflow=search \
  +actor_rollout_ref.rollout.plugin.max_turn=100 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
  +actor_rollout_ref.rollout.plugin.session_timeout=3600 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=True \
  +actor_rollout_ref.rollout.plugin.val_max_turn=100 \
  +actor_rollout_ref.rollout.plugin.val_response_length=${RESPONSE_LENGTH} \
  +actor_rollout_ref.rollout.plugin.process_reward=none \
  +actor_rollout_ref.rollout.plugin.max_compactions=${MAX_COMPACTIONS} \
  +actor_rollout_ref.rollout.plugin.val_max_compactions=${VAL_MAX_COMPACTIONS} \
  +actor_rollout_ref.rollout.plugin.compaction_threshold=${COMPACTION_THRESHOLD} \
  +actor_rollout_ref.rollout.plugin.compaction_tail_steps=${TAIL_STEPS} \
  +actor_rollout_ref.rollout.plugin.summary_max_tokens=${SUMMARY_MAX_TOKENS} \
  +actor_rollout_ref.rollout.plugin.train_summary=${TRAIN_SUMMARY} \
  +actor_rollout_ref.rollout.plugin.mask_unfinished=${MASK_UNFINISHED} \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_training_steps=100 \
  trainer.test_freq=10 \
  trainer.save_freq=10 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  trainer.project_name=context_folding \
  trainer.experiment_name=test_run
