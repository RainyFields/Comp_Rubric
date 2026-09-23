# NEXT_STEPS — ordered checklist for the Qwen3.5 devbox

Tick in order. Commands assume the paths in `QWEN35_HANDOFF.md` §3.

0. [x] **GitHub**: `RainyFields/Comp_Rubric` created and `main` pushed 2026-09-23 (`a25f8a4`; see `docs/PROTOCOL_VERIFICATION.md` §0).
       Note: the old box's clone was shallow (grafted at the authors' commit); it was unshallowed with `git fetch --unshallow origin`
       before the push. Fresh clones from `github` are full-history and need nothing.
1. [x] **Clone + bootstrap** on the new box — DONE 2026-09-23 (box `mlxlab5hqy1qoh6a9ed9e6…`, CPU-only devbox): repo at
       `~/xiaoxuan/Comp_Rubric`; `data/*.parquet` extracted from the HDFS repo tarball; tokenizers copied to
       `~/xiaoxuan/tokenizers/{Qwen3-8B,Qwen3.5-9B}` (Qwen3-8B from `/mnt/hdfs/mlsys/models/Qwen3-8B`); `/mnt/hdfs/mlsys` RW;
       merlin-cli authenticated on `i18n-tt` (TLS retries, see gotchas). Remote naming on this clone: `origin` = RainyFields/Comp_Rubric
       (what the docs call `github`); the authors' upstream is NOT configured (rename/add was blocked by the tool policy in that session).
       Original text: `git clone git@github.com:RainyFields/Comp_Rubric.git ~/xiaoxuan/Comp_Rubric`;
       copy `data/bc_train.parquet`, `data/bc_test.parquet`, `data/bc_test_shakeout.parquet` (git-ignored; from the old box or
       regenerate per README); copy `~/xiaoxuan/tokenizers/Qwen3.5-9B` from HDFS `fold-job-assets/tokenizers/`; make sure
       `/mnt/hdfs/mlsys` is mounted RW; merlin-cli authenticated (`--control-plane i18n-tt`).
2. [~] **Tests on the old stack** — SKIPPED locally (23 GB free on `/`; pods restore `fold_train` from HDFS anyway). Clone integrity
       was checked in the q35 venv instead (step 3/4 notes). One module is missing from the GitHub clone: `scripts/e2e_metrics.py`
       (`tests/test_e2e_ledger.py` → `ModuleNotFoundError`); it was never committed — recover it from the old box and commit.
       Original text: restore `~/xiaoxuan/envs/fold_train` from
       `fold-job-assets/fold_train.tar.gz`; run the 91 tests with both tokenizers (`QWEN35_HANDOFF.md` §5 step 0). Expect
       `OK` (Qwen3-8B) and `OK (skipped=3)` (Qwen3.5: recorded-id replays are Qwen3-8B ids).
3. [x] **Build the Qwen3.5 stack** — BUILT 2026-09-23 by `infra/build_env_q35.sh` (idempotent, ~5 min with a warm uv cache):
       Python 3.12, torch 2.11.0+cu130, vLLM 0.24.0, transformers 5.9.0, verl 0.9.1 (pip), flash-attn 2.8.3 (prebuilt, verl
       wheelhouse), flash-linear-attention 0.5.2, tensordict 0.10.0, cupy-cuda13x 14.0.1, unidiff, flask = verl 0.9.1's own
       `[vllm]` extra (its uv lock), the same stack as the pod-proven `supo-cu130` venv. Freeze: `infra/fold_train_q35.freeze.txt`.
       Tarball: `fold-job-assets/fold_train_q35.tar.gz` (+ `.md5`, `.commit`). `causal-conv1d` is NOT installed (no torch-2.11 wheel;
       verl's packed Qwen3.5 forward uses fla for the conv: set `causal_conv1d_implementation=fla` in the engine config).
       **Pods**: cu130 needs image `aliyun-va-hub.byted.org/arnold/modelchef-gpu:1.0.0.54` (ships the CUDA 13.0 compat libcuda for the
       R535 driver; the Qwen3-8B specs use 1.0.0.38 which only has 12.9 compat) — every Qwen3.5 spec sets it.
       STILL OPEN from this step: the GPU smoke (no GPU on the devbox) — `Qwen3_5ForCausalLM` load, `vllm serve` TP=1/8, one
       completion with `enable_thinking`. DONE (same day, pushed): `FOLD_VENV` job env (default `fold_train`) → `TRAIN_VENV` in
       `fold_common_bootstrap.sh` / `fold_val_entrypoint.sh` / `fold_train_entrypoint.sh` (restore_venv + preflight python) →
       `VENV`/`VENVT` in `worker_train.sh` / `worker_val_selfcontained.sh` (a non-default venv is never built on the pod; it must
       come from `fold-job-assets/<name>.tar.gz`). Entrypoints + bootstrap re-copied to HDFS and the repo tarball re-tarred.
       Original text: verl 0.9.1, transformers 5.10.x, vLLM ≥ 0.18,
       flash-linear-attention 0.5.2, causal-conv1d 1.7.0, flash-attn wheel. Smoke: load `Qwen3_5ForCausalLM` 9B bf16; `vllm serve` 9B
       TP=1 and TP=8; one chat completion with `enable_thinking=true`; confirm the completion has no `<think>` opener. Tarball it to
       `fold-job-assets/fold_train_q35.tar.gz` (+ `.md5`) and add a `restore_venv fold_train_q35` line to the job entrypoints.
4. [ ] **Port to verl 0.9.1** (migration plan WP3) — measured 2026-09-23 in the q35 venv (73 tests collected here):
       * with the vendored `verl/` on `sys.path` (running from the repo root) 33/73 error: `verl/utils/model.py` imports
         `AutoModelForVision2Seq`, removed in transformers 5 → the vendored copy is unusable in this venv, as predicted;
       * with the vendored `verl/` removed (pip verl 0.9.1) **62/73 pass on both tokenizers**; the 11 errors are exactly:
         7× fork estimators missing upstream (`compute_compaction_gae_advantage_return`, `compute_compaction_grpo_advantage`,
         `global_token_mean_scale` → re-add as the plugin), 3× `AgentLoopWorker._agent_loop_postprocess()` now takes `validate`
         (`tests/test_training_batch.py` worker shell), 1× `scripts.e2e_metrics` missing from the clone (step 2).
       * **Code fix already applied (commit after 7578352)**: transformers 5 returns a `BatchEncoding` from
         `apply_chat_template(..., tokenize=True)` (`return_dict` default flipped); every such call in `agents/utils.py`,
         `scripts/audit_rollout_trace.py` and the tests now passes `return_dict=False` (byte-identical ids; harmless on 4.57).
       * verl 0.9.1 facts: `AgentLoopBase.__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, data_config,
         hf_model_type=None, **kwargs)`; `AgentLoopOutput` fields = prompt_ids, response_ids, response_mask, response_logprobs,
         routed_experts, multi_modal_data, reward_score, num_turns, metrics, extra_fields, mm_processor_kwargs, mm_processor_output;
         `register_adv_est` and `verl.utils.model.compute_position_id_with_mask` exist unchanged.
       Original text: pip `verl`, re-apply the 155-line estimator patch as a plugin, adapt
       `scripts/train_fold.py` agent loops to the 0.9 `AgentLoopBase` API, keep `agents/utils.Agent` tokenisation; check
       `global_token_mean` is redundant with 0.9's global token normalisation; `use_remove_padding=True`, `use_fused_kernels=True`.
       Re-run the 91 tests in the q35 venv (they exercise `verl.experimental.agent_loop._agent_loop_postprocess` — adapt the
       worker shell in `tests/test_training_batch.py` if the 0.9 signature differs).
5. [ ] **Qwen3.5 protocol smoke (inference)** — spec DRAFTED (not submitted): `infra/jobs/fold_smoke_qwen35_norl_v2_h100.json`
       (image 1.0.0.54, `FOLD_MODEL_PATH=/mnt/hdfs/mlsys/models/Qwen3.5-9B` — the shared HDFS copy, no HF download needed;
       `FOLD_VENV=fold_train_q35`, `FOLD_ARM=norl FOLD_STEP=base FOLD_PROTOCOL=v2 FOLD_PROMPT_CAPTURE=1`, 18-task shakeout file).
       Blocked on step 3(b) (entrypoint/worker must honour `FOLD_VENV`) and on checking the val worker's vLLM 0.24 server flags. Then: derive `infra/jobs/fold_smoke_qwen35_norl_v2_h100.json` from
       `fold_shk3_grpo50_A_branch_v2_h100.json` (`FOLD_ARM=norl FOLD_STEP=base FOLD_MODEL_PATH=Qwen/Qwen3.5-9B FOLD_PROTOCOL=v2
       FOLD_VAL_FILE=data/bc_test_shakeout.parquet FOLD_PROMPT_CAPTURE=1`), submit, then
       `scripts/shakeout_audit.py audit … --tokenizer ~/xiaoxuan/tokenizers/Qwen3.5-9B`. Gate: every check in
       `docs/PROTOCOL_VERIFICATION.md` rows E1–E9 PASS on the Qwen3.5 capture (think blocks: `find_think` handles the pre-filled opener).
6. [ ] **Training shakeout (3 steps)** on Qwen3.5-4B then 9B: `fold-train-compactiongrpo-shakeout-h100.json` with
       `FOLD_MODEL_PATH`, `FOLD_PROTOCOL=v2`, `FOLD_ROLLOUT_DUMP=1`; gates in `QWEN35_HANDOFF.md` §9 (finite loss, grad_norm > 0,
       rollout/actor log-prob corr ≥ 0.95, GDN kernels active, traces all green: `scripts/render_rollout_traces.py`). Dump one batch's
       `response_mask` statistics from the trainer (closes the NOT-YET-VERIFIED-on-device row).
7. [ ] **Baseline eval** `norl` Qwen3.5-9B greedy + T1 n=4 (150 tasks).
8. [ ] **Arms**: `grpo` (no compaction) → `compactiongrpo` → `compactionrl` on 9B, 100 steps each, `PROTOCOL=v2`; evaluate each with
       the protocol/knobs it was trained with; keep `infra/jobs/JOBS.tsv` updated; no babysitting loops without user authorisation.
9. [ ] Report: extend `scripts/analyze_fix_campaign.py` with `--model-tag`; one table Qwen3-8B vs Qwen3.5-9B per arm; push.

Do not start step 8 before steps 5 and 6 pass.
