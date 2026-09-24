#!/bin/bash
# Model-agnostic launcher for every BrowseComp-Plus arm (one script, ARM + MODEL_PATH knobs).
#
#   ARM=foldgrpo|grpo|compactiongrpo|compactionrl  PROTOCOL=${PROTOCOL:-v2}   # agents/protocol.py: v2 (corrected, default) | legacy (pre-3f697bf training format)
MODEL_PATH=Qwen/Qwen3-8B  bash scripts/train_bc.sh [extra hydra overrides]
#
# Defaults reproduce the frozen per-arm scripts exactly (scripts/train_bc_qwen3_8b.sh = authors' script, used for the
# Qwen3-8B campaigns; scripts/train_bc_compaction{grpo,rl}.sh): same rollout budget (32 prompts x 8 samples, 8k prompt /
# 32k response, lr 1e-6, 100 steps), same plugin flags per arm. Only MODEL_PATH / MODEL_TAG / STEPS and the compaction
# knobs are environment variables; everything else can be appended as hydra overrides ("$@").
# See docs/baselines/README.md (arms) and docs/plans/2026-09-23_qwen3.5_migration_plan.md (model switch).
set -uo pipefail

ARM=${ARM:?grpo_no_compaction|grpo|foldgrpo|compactiongrpo|compactionrl|supo}
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
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
# --- verl 0.9.1 / Qwen3.5 model flags (docs/plans/2026-09-24_verl091_port_design.md) ---
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}     # packed forward (verl's own Qwen3.5 GatedDeltaNet path)
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-False}      # fused log-prob/entropy kernels (248k vocab); enable after the GPU shakeout
# --- rollout caps (decision D5, 2026-09-24; docs/EVAL_CONTRACT.md) ---
#   occupied-context ceiling per call = PROMPT_LENGTH + RESPONSE_LENGTH (CallLLM caps every call at that total; the split
#   only sets verl's tensor layout, the initial BC-Plus prompt is ~3k tokens)
TURN_MAX_NEW_TOKENS=${TURN_MAX_NEW_TOKENS:-8192}   # assistant generation cap per call
MAX_TURN=${MAX_TURN:-100}                          # assistant turns per rollout (main thread; branches count separately)
MAX_TOOL_CALLS=${MAX_TOOL_CALLS:-100}              # tool invocations per rollout
MAX_CALLS_PER_TURN=${MAX_CALLS_PER_TURN:-1}        # parallel tool invocations per assistant turn
SESSION_TIMEOUT=${SESSION_TIMEOUT:-7200}           # wall-clock seconds per rollout

# --- CompactionRL rollout knobs (docs/baselines/README.md) ---
MAX_COMPACTIONS=${MAX_COMPACTIONS:-3}
VAL_MAX_COMPACTIONS=${VAL_MAX_COMPACTIONS:-3}
COMPACTION_THRESHOLD=${COMPACTION_THRESHOLD:-8192}
TAIL_STEPS=${TAIL_STEPS:-2}
SUMMARY_MAX_TOKENS=${SUMMARY_MAX_TOKENS:-2048}
TRAIN_SUMMARY=${TRAIN_SUMMARY:-True}
MASK_UNFINISHED=${MASK_UNFINISHED:-False}
RESUME_KEEP_TASK_PROMPT=${RESUME_KEEP_TASK_PROMPT:-True}   # False = paper Eq. 9 resume context
# GLOBAL_TOKEN_MEAN: no longer a knob — verl 0.9.1 token-mean loss is normalised by the global (DP all-reduced) mini-batch token count
if [ "${PAPER_PROTOCOL:-0}" = 1 ]; then ROLLOUT_N=${ROLLOUT_N:-1}; TRAIN_BATCH=${TRAIN_BATCH:-128}; LR=${LR:-2e-6}
else ROLLOUT_N=${ROLLOUT_N:-8}; TRAIN_BATCH=${TRAIN_BATCH:-32}; LR=${LR:-1e-6}; fi
PPO_MINI=${PPO_MINI:-128}
# --- SUPO rollout knobs (arm supo; agents/compaction_agent.py summary_protocol=supo, decision D1) ---
MAX_SUMMARIES=${MAX_SUMMARIES:-2}                  # paper: 2 (BC-Plus); unified track: 3
SUMMARY_RATIO=${SUMMARY_RATIO:-0.95}               # trigger when occupied context >= ratio * ceiling
# --- critic (compactionrl only) ---
CRITIC_LR=${CRITIC_LR:-3e-6}; CRITIC_EPOCHS=${CRITIC_EPOCHS:-2}; CRITIC_WARMUP=${CRITIC_WARMUP:-50}; LAM_ALPHA=${LAM_ALPHA:-1.5}

case "$ARM" in
  foldgrpo)
    ARM_FLAGS="algorithm.adv_estimator=foldgrpo actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent
      +actor_rollout_ref.rollout.custom.plugin.workflow=search_branch
      +actor_rollout_ref.rollout.custom.plugin.max_session=10 +actor_rollout_ref.rollout.custom.plugin.val_max_session=10
      +actor_rollout_ref.rollout.custom.plugin.enable_summary=False +actor_rollout_ref.rollout.custom.plugin.branch_len=${RESPONSE_LENGTH}
      +actor_rollout_ref.rollout.custom.plugin.process_reward=[flat,scope] +actor_rollout_ref.rollout.custom.plugin.max_traj=11" ;;
  grpo_no_compaction)   # decision D2 baseline: no branch / summary / compaction tool in training (workflow=search)
    ARM_FLAGS="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent
      +actor_rollout_ref.rollout.custom.plugin.workflow=search
      +actor_rollout_ref.rollout.custom.plugin.max_session=1 +actor_rollout_ref.rollout.custom.plugin.val_max_session=1
      +actor_rollout_ref.rollout.custom.plugin.enable_summary=False +actor_rollout_ref.rollout.custom.plugin.branch_len=${RESPONSE_LENGTH}
      +actor_rollout_ref.rollout.custom.plugin.process_reward=none +actor_rollout_ref.rollout.custom.plugin.max_traj=1" ;;
  grpo)                 # ablation: branch tool exposed, outcome reward only
    ARM_FLAGS="algorithm.adv_estimator=grpo actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent
      +actor_rollout_ref.rollout.custom.plugin.workflow=search_branch
      +actor_rollout_ref.rollout.custom.plugin.max_session=10 +actor_rollout_ref.rollout.custom.plugin.val_max_session=10
      +actor_rollout_ref.rollout.custom.plugin.enable_summary=False +actor_rollout_ref.rollout.custom.plugin.branch_len=${RESPONSE_LENGTH}
      +actor_rollout_ref.rollout.custom.plugin.process_reward=none +actor_rollout_ref.rollout.custom.plugin.max_traj=11" ;;
  compactiongrpo|compactionrl|supo)
    EST=compaction_grpo; [ "$ARM" = compactionrl ] && EST=compaction_gae
    if [ "$ARM" = supo ]; then   # SUPO presets (decision D1): summaries = max_summaries, no verbatim tail, overlong rollouts masked
      EST=supo; MAX_COMPACTIONS=$MAX_SUMMARIES; VAL_MAX_COMPACTIONS=${VAL_MAX_SUMMARIES:-$MAX_SUMMARIES}; TAIL_STEPS=0; MASK_UNFINISHED=True; RESUME_KEEP_TASK_PROMPT=True
    fi
    ARM_FLAGS="algorithm.adv_estimator=${EST} actor_rollout_ref.rollout.agent.default_agent_loop=compaction_agent
      +actor_rollout_ref.rollout.custom.plugin.workflow=search +actor_rollout_ref.rollout.custom.plugin.process_reward=none
      +actor_rollout_ref.rollout.custom.plugin.max_compactions=${MAX_COMPACTIONS}
      +actor_rollout_ref.rollout.custom.plugin.val_max_compactions=${VAL_MAX_COMPACTIONS}
      +actor_rollout_ref.rollout.custom.plugin.compaction_threshold=${COMPACTION_THRESHOLD}
      +actor_rollout_ref.rollout.custom.plugin.compaction_tail_steps=${TAIL_STEPS}
      +actor_rollout_ref.rollout.custom.plugin.summary_max_tokens=${SUMMARY_MAX_TOKENS}
      +actor_rollout_ref.rollout.custom.plugin.train_summary=${TRAIN_SUMMARY}
      +actor_rollout_ref.rollout.custom.plugin.mask_unfinished=${MASK_UNFINISHED}
      +actor_rollout_ref.rollout.custom.plugin.resume_keep_task_prompt=${RESUME_KEEP_TASK_PROMPT}
      actor_rollout_ref.actor.loss_agg_mode=token-mean"
    if [ "$ARM" = supo ]; then   # SUPO semantics on the compaction rollout (D1): 95 % trigger, drop last pair, no tail, overlong mask
      ARM_FLAGS="$ARM_FLAGS +actor_rollout_ref.rollout.custom.plugin.summary_protocol=supo
        +actor_rollout_ref.rollout.custom.plugin.max_summaries=${MAX_SUMMARIES} +actor_rollout_ref.rollout.custom.plugin.summary_ratio=${SUMMARY_RATIO}"
    fi
    if [ "$ARM" = compactionrl ]; then
      ARM_FLAGS="$ARM_FLAGS critic.model.path=${MODEL_PATH} critic.optim.lr=${CRITIC_LR} critic.ppo_epochs=${CRITIC_EPOCHS}
        critic.ppo_micro_batch_size_per_gpu=1 critic.ppo_max_token_len_per_gpu=${MAX_LENGTH}
        critic.forward_max_token_len_per_gpu=${MAX_LENGTH} critic.fsdp.param_offload=True
        critic.fsdp.optimizer_offload=True critic.model.enable_gradient_checkpointing=True critic.enable=True
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
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  data.train_files=${TRAIN_DATA_PATH} \
  data.val_files=${TEST_DATA_PATH} \
  data.train_batch_size=${TRAIN_BATCH} \
  data.max_prompt_length=${PROMPT_LENGTH} \
  data.max_response_length=${RESPONSE_LENGTH} \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_LENGTH} \
  +actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${MAX_LENGTH} \
  +actor_rollout_ref.rollout.custom.plugin.max_turn=${MAX_TURN} \
  +actor_rollout_ref.rollout.custom.plugin.protocol=${PROTOCOL:-v2} \
  +actor_rollout_ref.rollout.custom.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.custom.plugin.turn_max_new_tokens=${TURN_MAX_NEW_TOKENS} \
  +actor_rollout_ref.rollout.custom.plugin.max_tool_calls=${MAX_TOOL_CALLS} \
  +actor_rollout_ref.rollout.custom.plugin.max_calls_per_turn=${MAX_CALLS_PER_TURN} \
  +actor_rollout_ref.rollout.custom.plugin.session_timeout=${SESSION_TIMEOUT} \
  +actor_rollout_ref.rollout.custom.plugin.must_finish=False \
  +actor_rollout_ref.rollout.custom.plugin.double_check=False \
  +actor_rollout_ref.rollout.custom.plugin.must_search=True \
  +actor_rollout_ref.rollout.custom.plugin.val_max_turn=${MAX_TURN} \
  +actor_rollout_ref.rollout.custom.plugin.val_response_length=${RESPONSE_LENGTH} \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=fold_sync \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=agents.verl_plugin.agent_loop.FoldAgentLoopManagerTQ \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=${SRC}/infra/agent_loop_config.yaml \
  actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING} \
  actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS} \
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
