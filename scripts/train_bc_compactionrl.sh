#!/bin/bash
# CompactionRL baseline (Li et al., arXiv:2607.05378) on BrowseComp-Plus with Qwen3-8B: PPO + critic,
# compaction rollouts (agents/compaction_agent.py), token-mean loss, cross-trajectory GAE with the paper's
# length-adaptive lambda. Rollout budget (32 prompts x 8 samples, 32k response, lr 1e-6) matches the
# FoldGRPO/GRPO arms; the paper's own values (group 1, batch 128, lr 2e-6, critic lr 3e-6) are the
# CRITIC_* / LR env vars below. See docs/baselines/README.md.

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
RESUME_KEEP_TASK_PROMPT=${RESUME_KEEP_TASK_PROMPT:-True}   # False = paper Eq. 9 (system + u_resume + tail; task only via the summary)
GLOBAL_TOKEN_MEAN=${GLOBAL_TOKEN_MEAN:-True}     # paper's token-level loss: equal weight per optimised token across the mini-batch
                                                 # (verl's plain token-mean with micro-batch 1 weights every SEGMENT equally; the finished
                                                 # compactiongrpo run af713ba2 used that)

# --- sampling protocol: FoldAgent parity by default (32 prompts x 8 samples, lr 1e-6); PAPER_PROTOCOL=1 = paper §5.1
#     (group size 1, global batch 128, policy lr 2e-6; critic lr 3e-6 below is the paper's in both cases)
if [ "${PAPER_PROTOCOL:-0}" = 1 ]; then
  ROLLOUT_N=${ROLLOUT_N:-1}; TRAIN_BATCH=${TRAIN_BATCH:-128}; LR=${LR:-2e-6}
else
  ROLLOUT_N=${ROLLOUT_N:-8}; TRAIN_BATCH=${TRAIN_BATCH:-32}; LR=${LR:-1e-6}
fi
PPO_MINI=${PPO_MINI:-128}                        # one policy update per batch (paper); 32x8 = 256 rollouts -> 2 mini-batches of segments

# --- critic (paper: critic initialised from the policy, lr 3e-6, two critic updates per policy update, value pre-training) ---
CRITIC_LR=${CRITIC_LR:-3e-6}
CRITIC_EPOCHS=${CRITIC_EPOCHS:-2}
CRITIC_WARMUP=${CRITIC_WARMUP:-50}               # value-pretraining steps before the first policy update (paper: 50)
LAM_ALPHA=${LAM_ALPHA:-1.5}                      # length-adaptive GAE lambda = 1 - 1/(alpha*l)


python -m scripts.train_fold \
  algorithm.adv_estimator=compaction_gae \
  actor_rollout_ref.rollout.agent.default_agent_loop=compaction_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_LENGTH} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  actor_rollout_ref.rollout.n=${ROLLOUT_N} \
  actor_rollout_ref.actor.optim.lr=${LR} \
  ++actor_rollout_ref.actor.global_token_mean=${GLOBAL_TOKEN_MEAN} \
  ++critic.global_token_mean=${GLOBAL_TOKEN_MEAN} \
  +actor_rollout_ref.rollout.plugin.resume_keep_task_prompt=${RESUME_KEEP_TASK_PROMPT} \
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
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${MAX_LENGTH} \
  +actor_rollout_ref.rollout.plugin.workflow=search \
  +actor_rollout_ref.rollout.plugin.max_turn=100 \
  +actor_rollout_ref.rollout.plugin.protocol=${PROTOCOL:-v2} \
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
  critic.model.path=${MODEL_PATH} \
  critic.optim.lr=${CRITIC_LR} \
  critic.ppo_epochs=${CRITIC_EPOCHS} \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.ppo_max_token_len_per_gpu=${MAX_LENGTH} \
  critic.forward_max_token_len_per_gpu=${MAX_LENGTH} \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  critic.model.enable_gradient_checkpointing=True \
  trainer.critic_warmup=${CRITIC_WARMUP} \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  ++algorithm.compaction_lam_alpha=${LAM_ALPHA} \
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
