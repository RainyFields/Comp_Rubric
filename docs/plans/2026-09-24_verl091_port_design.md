# verl 0.9.1 port — design (WP3, decided 2026-09-24)

Facts established by two read-only audits of the vendored `verl/` (0.7.0.dev, a fork of a fork: the authors' FoldAgent
layer + our +208/−3 lines in 9 files) against pip verl 0.9.1 (`~/xiaoxuan/envs/fold_train_q35`).

## Route: V1 trainer (`trainer.use_v1=True`, TransferQueue) + out-of-tree plugin `agents/verl_plugin/`

Why not the legacy `RayPPOTrainer`: it is deprecated and its `batch.union(gen_batch_output)` / `_postprocess` assume exactly
one output per prompt, so the multi-output rollouts (Fold: main + branches; compaction/SUPO: segments) cannot pass through it.
The V1 path stores every agent-loop output as its own TransferQueue row keyed `{uid}_{session_id}_{index}` (= our `gen_uid` +
segment index), pads variable row counts natively (`upsample_batch_to_divisible_size`), and exposes clean override points.

| fork feature | 0.9.1 status | port |
|---|---|---|
| estimators `foldgrpo`, `compaction_gae`, `compaction_grpo` (+ new `supo`) | registry `register_adv_est` unchanged, but generic dispatch passes only `token_level_rewards, response_mask, config, index` | `agents/verl_plugin/estimators.py`: fork bodies verbatim, registered by string name (idempotent); called from our trainer override with `gen_uid`, `process_reward_mask`, `tokens_after`, `values`, `gamma`, `lam` |
| trainer dispatch (`compute_advantage` branches, `gen_uid`, overlong mask, dummy padding) | V1 `PPOTrainer._compute_advantage` fetches fixed fields; padding native; no overlong mask | `agents/verl_plugin/trainer.py`: `@register_trainer("fold_sync") class FoldSyncTrainer(SyncTrainer)` overriding `_compute_advantage`: also fetch `extra_fields`; `gen_uid = key.rsplit("_",1)[0]`; right-pad `process_reward_mask`; read `tokens_after`; dispatch our estimators; `mask_rollout`/overlong handled by zeroing `response_mask` in the worker (row keeps its `rm_scores`, so it still counts in the group statistics, as in the fork and in SUPO) |
| `actor.global_token_mean` / `critic.global_token_mean` | native: `agg_loss` token-mean normalises by the DP-all-reduced mini-batch token count | drop the knob (`GLOBAL_TOKEN_MEAN` → documented as always on) |
| `need_critic` for compaction_gae | imported by name, cannot patch | `critic.enable=True` for `compactionrl` |
| `AlgoConfig` extras | `config.algorithm` stays a DictConfig | `+algorithm.compaction_lam_alpha=1.5 +algorithm.compaction_whiten=True`, read via `config.get` |
| `rollout.plugin.*` | `RolloutConfig` dataclass rejects unknown keys | `+actor_rollout_ref.rollout.custom.plugin.*`; the loop exposes it to `agents/*` as `rollout.plugin` (shallow DictConfig copy) |
| vLLM `max_tokens` cap | native (clamped to `max_model_len − len(prompt)`) | drop; `CallLLM` keeps its own per-call cap (D5: 8 192 new tokens, occupied ≤ 65 536) |
| agent loop ctor / kwargs | `AgentLoopBase.__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config, hf_model_type, **kwargs)`; `run(sampling_params, **kwargs)` gets dataset fields + `uid`, `index`, `global_steps`, `session_id`, `raw_prompt` — **not** `validate` | loops accept `**kwargs`, drop `init_class`; custom worker injects `validate`; `is_train = not validate`; `gen_uid = f"{uid}_{session_id}"` |
| list outputs | V1 `AgentLoopWorkerTQ._agent_loop_postprocess` accepts lists but (a) broadcasts the final output's reward to all rows, (b) requires `extra_fields["reward_extra_info"]` | `agents/verl_plugin/agent_loop.py`: `FoldAgentLoopWorkerTQ` overrides `_agent_loop_postprocess` (per-row rewards kept, `reward_extra_info` guaranteed, `response_mask` zeroed for `mask_rollout` rows); `FoldAgentLoopManagerTQ` sets `agent_loop_workers_class`; selected with `+actor_rollout_ref.rollout.agent.agent_loop_manager_class=agents.verl_plugin.agent_loop.FoldAgentLoopManagerTQ` |
| sampling params | worker passes `temperature/top_p/top_k` (val: from `val_kwargs`; greedy via `apply_greedy_sampling_params`) but `CallLLM` rebuilt its own (T=1 always — pre-existing bug: "greedy" evals sampled at T=1) | `CallLLM` honours the `sampling_params` given to `run()` (temperature, top_p, top_k); greedy evals become truly greedy — **a protocol change to note in PROTOCOL.md / EVAL_CONTRACT** |
| registration in the right process | trainer class must be registered inside the Ray TaskRunner before `get_trainer_cls`; loops are imported by `_target_` in the workers | `scripts/train_fold.py main()` runs `run_ppo(config, task_runner_class=FoldTaskRunner)` where `FoldTaskRunner.run` imports the plugin first; `agent_loop_config_path` stays the import hook for the loops |
| validation | `_validate` scores the **last** row of each session, val metrics = `val-core/<data_source>/reward/mean@n` (+ every numeric key of `extra_fields["reward_extra_info"]` under `val-aux/...`), dumps `validation_data_dir/<step>.jsonl` with prompts/responses/scores/reward_extra_info per row | eval loops already return only the last segment / main; put per-rollout stats (finished, branches, compactions, stop_reason code, ledger totals) into `reward_extra_info`; the D9 raw-trajectory archive is written by the loop itself (capture hook), not by verl's dump; `worker_val_selfcontained.sh` greps the new metric names |
| hydra keys | `critic.model.fsdp_config.*` → `critic.fsdp.*`; `actor.ppo_infer_max_token_len_per_gpu` needs `+`; `trainer.use_v1`, `trainer.v1.trainer_mode`; `model.use_remove_padding=True`, `model.use_fused_kernels=True` for Qwen3.5 | `scripts/train_bc.sh` rewritten; frozen Qwen3-8B scripts left as historical (they need the vendored verl → branch) |
| vendored `verl/` | unusable under transformers 5 | removed from `main` (kept on branch `qwen3-8b-vendored-verl` for the legacy campaign); job tarball shrinks |

## Order of work
1. plugin package (estimators, trainer, worker/manager, runner) + `scripts/train_fold.py` adaptation + `CallLLM` sampling params;
2. `scripts/train_bc.sh` keys; `infra/worker_*.sh` sed templates; new arm `grpo_no_compaction` (and `supo` knobs on the compaction agent, D1);
3. tests: `tests/test_training_batch.py` worker shell against `FoldAgentLoopWorkerTQ._agent_loop_postprocess`, `test_compaction_advantage.py` importing the plugin; all tests on both tokenizers;
4. remove `verl/` from main; retar; 3-step GPU shakeout on Qwen3.5-9B (`grpo_no_compaction`) with rollout dumps — verification row E11 and the D10 checks.
