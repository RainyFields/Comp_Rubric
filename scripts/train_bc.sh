#!/bin/bash
# Model-agnostic launcher for every BrowseComp-Plus arm (one script, ARM + MODEL_PATH knobs).
#
#   ARM=foldgrpo|grpo|compactiongrpo|compactionrl  MODEL_PATH=Qwen/Qwen3-8B  bash scripts/train_bc.sh [extra hydra overrides]
#
# Defaults reproduce the frozen per-arm scripts exactly (scripts/train_bc_qwen3_8b.sh = authors' script, used for the
# Qwen3-8B campaigns; scripts/train_bc_compaction{grpo,rl}.sh): same rollout budget (32 prompts x 8 samples, 8k prompt /
# 32k response, lr 1e-6, 100 steps), same plugin flags per arm. Only MODEL_PATH / MODEL_TAG / STEPS and the compaction
# knobs are environment variables; everything else can be appended as hydra overrides ("$@").
# See docs/baselines/README.md (arms) and docs/plans/2026-09-23_qwen3.5_migration_plan.md (model switch).
set -uo pipefail

ARM=${ARM:?foldgrpo|grpo|compactiongrpo|compactionrl}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
MODEL_TAG=${MODEL_TAG:-$(basename "$MODEL_PATH" | tr 'A-Z' 'a-z')}      # run names / output dirs: qwen3-8b, qwen3.5-9b, ...
STEPS=${STEPS:-100}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${ARM}_${MODEL_TAG}_bcplus}
PROMPT_LENGTH=${PROMPT_LENGTH:-8192}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-32768}
MAX_LENGTH=$((PROMPT_LENGTH + RESPONSE_LENGTH))
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-data/bc_train.parquet}
TEST_DATA_PATH=${TEST_DATA_PATH:-data/bc_test.parquet}
N_GPUS=${N_GPUS:-8}
TP=${TP:-8}

# --- CompactionRL rollout knobs (docs/baselines/README.md) ---
MAX_COMPACTIONS=${MAX_COMPACTIONS:-3}
VAL_MAX_COMPACTIONS=${VAL_MAX_COMPACTIONS:-3}
COMPACTION_THRESHOLD=${COMPACTION_THRESHOLD:-8192}
TAIL_STEPS=${TAIL_STEPS:-2}
SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-2048}
TRAIN_SUMMARY=${TRAIN_SUMMARY:-True}
MASK_UNFINISHED=${MASK_UNFINISHED:-False}
# --- critic (compactionrl only) ---
CRITIC_LR=${CRITIC_LR:-3e-6}; CRITIC_EPOCHS=${CRITIC_EPOCHS:-2}; CRITIC_WARMUP=${CRITIC_WARMUP:-50}; LAM_ALPHA=${LAM_ALPHA:-1.5}

case "$ARM" in
  foldgrpo)
    ARM_FLAGS="algorithm.adv_estimator=foldgrpo actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent
      +actor_rollout_ref.rollout.plugin.workflow=search_branch
      +actor_rollout_ref.rollout.plugin.max_session=10 +actor_rollout_ref.rollout.plugin.val_max_session=10
      +actor_rollout_ref.rollout.plugin.enable_summary=False +actor_rollout_ref.rollout.plugin.branch_len=${RESPONSE_LENGTH}
      +actor_rollout_ref.rollout.plugin.process_reward=[flat,scope] +actor_rollout_ref.rollout.plugin.max_traj=11" ;;
  grpo)
    ARM_FLAGS="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent
      +actor_rollout_ref.rollout.plugin.workflow=search_branch
      +actor_rollout_ref.rollout.plugin.max_session=10 +actor_rollout_ref.rollout.plugin.val_max_session=10
      +actor_rollout_ref.rollout.plugin.enable_summary=False +actor_rollout_ref.rollout.plugin.branch_len=${RESPONSE_LENGTH}
      +actor_rollout_ref.rollout.plugin.process_reward=none +actor_rollout_ref.rollout.plugin.max_traj=11" ;;
  compactiongrpo|compactionrl)
    EST=compaction_grpo; [ "$ARM" = compactionrl ] && EST=compaction_gae
    ARM_FLAGS="algorithm.adv_estimator=${EST} actor_rollout_ref.rollout.agent.default_agent_loop=compaction_agent
      +actor_rollout_ref.rollout.plugin.workflow=search +actor_rollout_ref.rollout.plugin.process_reward=none
      +actor_rollout_ref.rollout.plugin.max_compactions=${MAX_COMPACTIONS}
      +actor_rollout_ref.rollout.plugin.val_max_compactions=${VAL_MAX_COMPACTIONS}
      +actor_rollout_ref.rollout.plugin.compaction_threshold=${COMPACTION_THRESHOLD}
      +actor_rollout_ref.rollout.plugin.compaction_tail_steps=${TAIL_STEPS}
      +actor_rollout_ref.rollout.plugin.summary_max_tokens=${SUMMARY_MAX_TOKENS}
      +actor_rollout_ref.rollout.plugin.train_summary=${TRAIN_SUMMARY}
      +actor_rollout_ref.rollout.plugin.mask_unfinished=${MASK_UNFINISHED}
      actor_rollout_ref.actor.loss_agg_mode=token-mean"
    if [ "$ARM" = compactionrl ]; then
      ARM_FLAGS="$ARM_FLAGS critic.model.path=${MODEL_PATH} critic.optim.lr=${CRITIC_LR} critic.ppo_epochs=${CRITIC_EPOCHS}
        critic.ppo_micro_batch_size_per_gpu=1 critic.ppo_max_token_len_per_gpu=${MAX_LENGTH}
        critic.forward_max_token_len_per_gpu=${MAX_LENGTH} critic.model.fsdp_config.param_offload=True
        critic.model.fsdp_config.optimizer_offload=True critic.model.enable_gradient_checkpointing=True
        trainer.critic_warmup=${CRITIC_WARMUP} algorithm.gamma=1.0 algorithm.lam=1.0 ++algorithm.compaction_lam_alpha=${LAM_ALPHA}"
    fi ;;
  *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac

# shellcheck disable=SC2086
python -m scripts.train_fold \
  $ARM_FLAGS \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_LENGTH} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP} \
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
  +actor_rollout_ref.rollout.plugin.max_turn=100 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
  +actor_rollout_ref.rollout.plugin.session_timeout=3600 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=True \
  +actor_rollout_ref.rollout.plugin.val_max_turn=100 \
  +actor_rollout_ref.rollout.plugin.val_response_length=${RESPONSE_LENGTH} \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=${N_GPUS} \
  trainer.nnodes=1 \
  trainer.total_training_steps=${STEPS} \
  trainer.test_freq=10 \
  trainer.save_freq=10 \
  trainer.project_name=context_folding \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  "$@"
