# Baseline arms in this repository

All arms share the BrowseComp-Plus track, `Qwen3-8B` (thinking mode on), the gpt-oss-120b judge for every
judge role, the fixed observation tokenisation (commit b94b34f), and the rollout budget of the authors'
script (32 prompts × 8 samples per step, 32 768 response tokens per context, lr 1e-6, 100 steps). Every arm
is one `ARM` value of `infra/worker_train.sh` (batch job: `infra/jobs/fold-train-<arm>-*.json`) and of the
val-only path `infra/worker_val_selfcontained.sh` / `infra/jobs/fold_val_*.json`.

| ARM | Rollout | Advantage / loss | Script | Reference |
|---|---|---|---|---|
| `norl` | single-thread ReAct, no training | – | val-only | base model |
| `grpo` | fold agent scaffold (branch tool available), outcome reward only | GRPO (group of 8), token-mean | `scripts/train_bc_qwen3_8b.sh` ± 2 flags | Shao et al. 2024 |
| `foldgrpo` | fold agent, process rewards (flat + scope) | FoldGRPO | `scripts/train_bc_qwen3_8b.sh` | Sun et al. 2025 (arXiv:2510.11967) |
| `compactionrl` | single-thread ReAct with trainable context compaction (`agents/compaction_agent.py`) | PPO + critic, cross-trajectory GAE with length-adaptive λ, token-mean | `scripts/train_bc_compactionrl.sh` | Li et al. 2026 (arXiv:2607.05378), paper-faithful optimiser |
| `compactiongrpo` | same compaction rollout | rollout-level group-relative advantage broadcast to every segment, token-mean | `scripts/train_bc_compactiongrpo.sh` | protocol-matched (critic-free) variant |

## CompactionRL (`compactionrl`, `compactiongrpo`)

Paper: `refs/2607.05378_compactionrl.md` (Markdown extracted from the arXiv PDF; see `refs/README.md`).
No official code was released at the time of writing (the paper trains with THUDM/slime); this is an
independent implementation on the FoldAgent verl stack.

### What is implemented (paper section → code)

| Paper | Here |
|---|---|
| Compaction trigger `C − |h_t| < T_comp` (§4.1, Eq. 7) | `plugin.compaction_threshold` on the remaining *generated-token* budget of the current context segment (`response_length`); default 6144 (paper: 10 240 of a 64k window) |
| Summary `S_t ~ π_θ(· | h_t ⊕ q_sum)` (Eq. 8), trainable | `COMPACTION_SUMMARY_PROMPT` appended as a plain user turn; the policy's response is a normal trainable assistant turn; `plugin.train_summary=False` masks its loss (ablation "w/o sum.") |
| Resume context `(s) ⊕ u_resume(S_t) ⊕ (z_{t−k+1..t})`, k = 2 (Eq. 9) | new `Agent` with the task prompt, `COMPACTION_RESUME_TEMPLATE`, and the last `plugin.compaction_tail_steps` (assistant, observation) steps re-rendered as non-trainable turns; k shrinks until `T_comp` of budget remains |
| At most three compactions per rollout; ×1 vs ×4 evaluation (§5.1) | `plugin.max_compactions` (train) and `plugin.val_max_compactions` (eval; 0 = single window) |
| One shared task reward for all segments, no summary reward (§4.1) | every segment is one training sample with `reward_score = R`; `extra_fields.tokens_after` = optimised tokens generated after the segment in the same rollout |
| Token-level loss normalisation (Eq. 12) | `actor_rollout_ref.actor.loss_agg_mode=token-mean` (ablation: `seq-mean-token-mean`) |
| Cross-trajectory GAE (Eqs. 13–15), critic, length-adaptive λ = 1 − 1/(αl), α = 1.5 | `algorithm.adv_estimator=compaction_gae` (`core_algos.compute_compaction_gae_advantage_return`): local GAE per segment, policy advantage × (γλ_i)^{tokens_after}, critic targets keep the local return; `algorithm.compaction_lam_alpha`; `critic.*` from the policy checkpoint, `critic.ppo_epochs=2`, `trainer.critic_warmup` (paper: 50 value-pretraining steps) |
| PPO with group size 1, batch 128, lr 2e-6 / 3e-6 (§5.1) | **deviation:** the arms keep the FoldAgent rollout budget (32 × 8, lr 1e-6) so that every arm sees the same trajectories per step; the paper's values are exposed as env vars in the script |
| Coding benchmarks, Terminus scaffold | **deviation:** BrowseComp-Plus search agent; observations are search results and opened pages |

`compactiongrpo` replaces the critic with the arms' group-relative estimator
(`core_algos.compute_compaction_grpo_advantage`): rewards are normalised once per rollout inside the
prompt group (segments of one rollout are de-duplicated, as FoldGRPO does), the rollout advantage is
broadcast to all of its segments, and the position correction is applied (identity at γ = λ = 1).

### Budget accounting

Each segment's generated side (resume turn, tail, new turns, summary) is bounded by `response_length`.
The summary is guaranteed to fit: if `q_sum` plus `summary_max_tokens` would overrun, the most recent
steps are rolled back out of the segment before summarising (they stay verbatim in the tail; only their
loss is forgone; counted in `env_stats.rollback_before_summary`). Per-rollout statistics logged to W&B via
`env_stats`: `compactions`, `summary_tokens` (trained), `summary_tokens_generated`, `tail_steps`,
`budget_exhausted`, `turns`, `traj_num` (= segments).

### Commands

```bash
# CPU tests (rollout with a scripted LLM + real Qwen3 tokenizer; estimators)
~/xiaoxuan/envs/fold_train/bin/python -m unittest tests.test_compaction_agent tests.test_compaction_advantage -v

# local launch (8 GPUs, same layout as the other arms)
bash scripts/train_bc_compactionrl.sh            # PPO + critic (paper-faithful optimiser)
bash scripts/train_bc_compactiongrpo.sh          # critic-free, protocol-matched
TRAIN_SUMMARY=False bash scripts/train_bc_compactiongrpo.sh   # "w/o summary training" ablation
# batch jobs (needs a live infra node; log the sid in infra/jobs/JOBS.tsv)
cd infra/jobs && merlin-cli --control-plane i18n-tt job-v2 runs create --from-file fold-train-compactionrl-h100.json
# evaluation of a checkpoint: FOLD_ARM=compactionrl FOLD_STEP=<n> in a copy of fold_val_grpo_fix_step20_h100.json
# (VAL_MAX_COMPACTIONS=3 -> paper's x4 setting; 0 -> single window)
```

Knobs (env vars of the scripts): `MAX_COMPACTIONS`, `VAL_MAX_COMPACTIONS`, `COMPACTION_THRESHOLD`, `TAIL_STEPS`,
`SUMMARY_MAX_TOKENS`, `TRAIN_SUMMARY`; `compactionrl` only: `CRITIC_LR`, `CRITIC_EPOCHS`, `CRITIC_WARMUP`, `LAM_ALPHA`.

### Before launching a real run

1. Shakeout: `FOLD_STEPS=3` on one batch job and check `env_stats.compactions > 0`, segment counts in the
   W&B metric `reward/avg_trajs_per_gen_uid`, and that `critic/vf_loss` decreases (compactionrl).
2. Data compliance (mandatory on every job): BrowseComp-Plus is a public benchmark →
   `HAS_TT_DATA=False`, `MERLIN_JOB_INSTANCE_TYPE=Train`, `ML_FRAMEWORK=pytorch`, `STORAGE_TYPE=HDFS`; the
   env vars are already in every `infra/jobs/*.json`.
3. Cost: the critic doubles the model memory of the training pod (actor + critic, both offloaded FSDP) and
   adds a critic forward/backward per step; expect step time ≈ 1.5–2× the GRPO arm. `CRITIC_WARMUP=50`
   spends 50 rollout steps on value pre-training before the first policy update.

## Adding another baseline arm

1. Agent loop: new module under `agents/` returning one `AgentLoopOutput` per training sample; register it in
   `scripts/train_fold.py` and `infra/agent_loop_config.yaml`.
2. Advantage estimator (if any): add to `AdvantageEstimator` and `register_adv_est` in
   `verl/trainer/ppo/core_algos.py`, dispatch in `ray_trainer.compute_advantage`, and gate the critic in
   `verl/trainer/ppo/utils.need_critic`.
3. Script `scripts/train_bc_<arm>.sh`, `ARM` cases in `infra/worker_train.sh` and
   `infra/worker_val_selfcontained.sh`, a job spec in `infra/jobs/` (with the compliance env vars).
4. CPU tests under `tests/`, a row in the table above, and the analysis hooks in
   `scripts/analyze_fix_campaign.py` if the arm adds per-trajectory statistics.
