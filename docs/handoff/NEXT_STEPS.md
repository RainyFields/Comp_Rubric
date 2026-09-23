# NEXT_STEPS — ordered checklist for the Qwen3.5 devbox

Tick in order. Commands assume the paths in `QWEN35_HANDOFF.md` §3.

0. [x] **GitHub**: `RainyFields/Comp_Rubric` created and `main` pushed 2026-09-23 (`a25f8a4`; see `docs/PROTOCOL_VERIFICATION.md` §0).
       Note: the old box's clone was shallow (grafted at the authors' commit); it was unshallowed with `git fetch --unshallow origin`
       before the push. Fresh clones from `github` are full-history and need nothing.
1. [ ] **Clone + bootstrap** on the new box: `git clone git@github.com:RainyFields/Comp_Rubric.git ~/xiaoxuan/Comp_Rubric`;
       copy `data/bc_train.parquet`, `data/bc_test.parquet`, `data/bc_test_shakeout.parquet` (git-ignored; from the old box or
       regenerate per README); copy `~/xiaoxuan/tokenizers/Qwen3.5-9B` from HDFS `fold-job-assets/tokenizers/`; make sure
       `/mnt/hdfs/mlsys` is mounted RW; merlin-cli authenticated (`--control-plane i18n-tt`).
2. [ ] **Tests on the old stack** (sanity that the clone is intact): restore `~/xiaoxuan/envs/fold_train` from
       `fold-job-assets/fold_train.tar.gz`; run the 91 tests with both tokenizers (`QWEN35_HANDOFF.md` §5 step 0). Expect
       `OK` (Qwen3-8B) and `OK (skipped=3)` (Qwen3.5: recorded-id replays are Qwen3-8B ids).
3. [ ] **Build the Qwen3.5 stack** `~/xiaoxuan/envs/fold_train_q35` (migration plan WP2): verl 0.9.1, transformers 5.10.x, vLLM ≥ 0.18,
       flash-linear-attention 0.5.2, causal-conv1d 1.7.0, flash-attn wheel. Smoke: load `Qwen3_5ForCausalLM` 9B bf16; `vllm serve` 9B
       TP=1 and TP=8; one chat completion with `enable_thinking=true`; confirm the completion has no `<think>` opener. Tarball it to
       `fold-job-assets/fold_train_q35.tar.gz` (+ `.md5`) and add a `restore_venv fold_train_q35` line to the job entrypoints.
4. [ ] **Port to verl 0.9.1** (migration plan WP3): pip `verl`, re-apply the 155-line estimator patch as a plugin, adapt
       `scripts/train_fold.py` agent loops to the 0.9 `AgentLoopBase` API, keep `agents/utils.Agent` tokenisation; check
       `global_token_mean` is redundant with 0.9's global token normalisation; `use_remove_padding=True`, `use_fused_kernels=True`.
       Re-run the 91 tests in the q35 venv (they exercise `verl.experimental.agent_loop._agent_loop_postprocess` — adapt the
       worker shell in `tests/test_training_batch.py` if the 0.9 signature differs).
5. [ ] **Qwen3.5 protocol smoke (inference)**: derive `infra/jobs/fold_smoke_qwen35_norl_v2_h100.json` from
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
