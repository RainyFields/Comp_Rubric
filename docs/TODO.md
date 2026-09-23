# TODO (written 2026-09-23; owner: Xiaoxuan; check boxes as they land)

Context: `docs/STATUS.md` (what exists), `docs/baselines/README.md` → "Fidelity audit" (why A is needed),
`docs/plans/2026-09-23_qwen3.5_migration_plan.md` (details behind C). Job recipes: `infra/jobs/`, ledger `infra/jobs/JOBS.tsv`.
Everything below that launches a job needs the compliance env vars (already in every spec: `HAS_TT_DATA=False`, public BC-Plus data).

## A. Replicate CompactionRL for real (the paper is PPO; `compactiongrpo` was a GRPO ablation)

- [ ] **Decide two knobs** (both exposed in `scripts/train_bc_compactionrl.sh` / `scripts/train_bc.sh`):
  - [ ] `PAPER_PROTOCOL=1` (group size 1, batch 128, lr 2e-6, critic lr 3e-6 — paper §5.1) **or** `0` (32×8, lr 1e-6, parity with the Fold/GRPO arms)
  - [ ] `RESUME_KEEP_TASK_PROMPT=False` (paper Eq. 9: system + u_resume + tail) **or** `True` (task prompt re-inserted, what ran so far)
- [ ] Infra pod: `cd infra/jobs && merlin-cli --control-plane i18n-tt job-v2 runs create --from-file fold-infra-fix-h100.json`; wait for
      `markers_fix/INFRA_READY` (never submit an arm before it); log the sid in `JOBS.tsv`.
- [ ] **Shakeout** `fold-train-compactionrl-shakeout-h100.json` (3 steps, `CRITIC_WARMUP=1`, `FOLD_ROLLOUT_DUMP=1`, `GLOBAL_TOKEN_MEAN=True`).
      Pass criteria: `critic/vf_loss` finite and falling on step 1 (critic-only), `actor/grad_norm` > 0 on steps 2–3, rollout-vs-actor
      log-prob corr ≥ 0.95, `[COMPACTION]` lines show segments 2–4 / rollout, then
      `python scripts/render_rollout_traces.py --dump …/rollout_dump_fix/compactionrl_shakeout/2.jsonl --out docs/traces/<date>_compactionrl_shakeout.md`
      → checklist all green. Watch memory: actor + critic (both offloaded FSDP) — if OOM, lower `ppo_max_token_len_per_gpu` or enable
      critic gradient checkpointing (already on) / `use_dynamic_bsz`.
- [ ] **Full run** `fold-train-compactionrl-h100.json` (100 steps, `CRITIC_WARMUP=50` → first policy update at step 51; budget ≈ 1.5–2× the
      GRPO step time). Needs the infra keep-alive loop (`infra/infra_resubmit_loop.sh`, one arm, user-authorised — see the 09-16 hand-off).
- [ ] Evals of the final ckpts (val-only specs, pattern `fold_val_grpo_fix_step20_h100.json`, `FOLD_ARM=compactionrl`): greedy + T1n4,
      Compacted ×4 (`VAL_MAX_COMPACTIONS=3`) **and** Single ×1 (`=0`) — the paper's Table 2 contrast.
- [ ] Ablations if time: `TRAIN_SUMMARY=False` (paper Table 3 "w/o sum."), `GLOBAL_TOKEN_MEAN=False` (paper Table 4 "w/o token-level loss"),
      `LAM_ALPHA=` constant λ / `compaction_whiten=False`.
- [ ] Report: add the arm to `scripts/analyze_fix_campaign.py` (compaction stats: finish rate, compactions/rollout, summary tokens from the
      `[COMPACTION]` lines) and a results section; label `compactiongrpo` as the critic-free ablation everywhere (W&B run
      `fold_compactiongrpo_2026-09` keeps its id).

## B. `compactiongrpo` wrap-up (finished run af713ba2, val .353 @100)

- [ ] Re-eval ckpt 20 and 40 (in-gap vals read 0.02) with val-only jobs.
- [ ] Greedy + T1n4 for ckpt 70 (best, .360) and 100; ×4 and ×1.
- [ ] Optional: rerun with `GLOBAL_TOKEN_MEAN=True` to measure the loss-weighting effect on the critic-free variant.

## C. Qwen3.5 (new dev box) — what is necessary, in order

Facts, decisions D1–D5 and work packages: `docs/plans/2026-09-23_qwen3.5_migration_plan.md`. Summary of what must happen:

### C0. Decide
- [ ] D1 size: **Qwen3.5-9B** (Qwen3-8B replacement) + **4B** for shakeouts; 27B only as a scale point.
- [ ] D2 stack: **port onto upstream verl 0.9.1** (native `qwen3_5` incl. packed GatedDeltaNet forward; transformers 5.5.3–5.10; vLLM ≥ 0.18)
      rather than patching the vendored 0.7.0.dev fork (transformers pinned 4.57.6, vLLM 0.10.2, no Qwen3.5).
- [ ] D3 window 32k/32k (parity) now; a 64k / T_comp 10 240 arm (paper setting) later.
- [ ] D4 thinking on (default); non-thinking only as an ablation.
- [ ] D5 no-RL Qwen3.5-9B greedy + T1n4 baseline before any training.

### C1. Environment (`~/xiaoxuan/envs/fold_train_q35`, then tarball → `fold-job-assets/fold_train_q35.tar.gz` + `.md5`)
- [ ] Python 3.11 uv venv; `verl==0.9.1`; torch per verl's pin; `vllm` ≥ 0.18 (the version verl 0.9.1's docker uses); prebuilt `flash-attn`;
      **`flash-linear-attention==0.5.2` + `causal-conv1d==1.7.0`** (GDN kernels; without them transformers silently falls back to a slow fp32 path —
      check the trainer log for the fallback warning); `tensordict<=0.10`; `wandb httpx openai ray`.
- [ ] Smoke on one GPU worker: `Qwen3_5ForCausalLM.from_pretrained("Qwen/Qwen3.5-9B", dtype=bf16)` forward; `vllm serve` 9B TP=1 and TP=8,
      one chat completion with `chat_template_kwargs={"enable_thinking": true}`; check the response has no opening `<think>` (pre-filled).
- [ ] Download `Qwen/Qwen3.5-9B` (+ `4B`) into the box's HF cache and seed `fold-job-assets/hf_hub/` (`cp -rL`, fuse cannot make symlinks).
- [ ] `fold_infra` venv unchanged (search server + gpt-oss-120b judge); `fold_infra.tar.gz` is reusable as is.

### C2. Port the training stack to verl 0.9.1 (WP3)
- [ ] Take `verl` from pip; keep the vendored copy on a branch for Qwen3-8B reproduction.
- [ ] Re-apply the fork patch as a plugin (no in-tree edits): `compaction_gae`, `compaction_grpo`, `foldgrpo` estimators via `register_adv_est`;
      `need_critic` hook; `AlgoConfig` extras (`compaction_lam_alpha`, `compaction_whiten`) via `++algorithm.*`; **`global_token_mean`** —
      check whether 0.9.1's engine already normalises token-mean over the global batch (`batch_num_tokens` in `verl/workers/utils/losses.py`);
      if so drop our patch; the `vllm_async_server` max_tokens cap (eabf814) is probably obsolete.
- [ ] Adapt `FoldAgentLoop` / `CompactionAgentLoop` (`scripts/train_fold.py`) to the 0.9 `AgentLoopBase.__init__(trainer_config, server_manager,
      tokenizer, processor, dataset_cls, data_config, hf_model_type)`; keep our own tokenisation (`agents/utils.Agent`), bypass the
      Continuous-Token builder; confirm a rollout may return a **list** of `AgentLoopOutput` (segments) — else wrap + split in `_postprocess`.
- [ ] `actor_rollout_ref.model.use_remove_padding=True` (0.9 has the packed GDN forward), `use_fused_kernels=True` (248k vocab × 32k tokens
      logits), `trust_remote_code=False`; verify FSDP→vLLM weight sync for the hybrid layers and the `model.language_model.*` prefix.
- [ ] `verl.model_merger` for `Qwen3_5ForConditionalGeneration` checkpoints (val-only path merges FSDP shards → HF).

### C3. Code that must change for Qwen3.5 (most already done 2026-09-23; verify on the new box)
- [x] `agents/parsing.py`: think opener optional (Qwen3.5 pre-fills `<think>\n`); `<tool_call>` wrapper tolerated.
- [x] `agents/model_profile.py`: template-detected profile; `Agent._render_prefix` / `get_generation_prompt` / `truncate_prompt` handle
      templates that raise "No user query found" (Qwen3.5) for system-only prefixes.
- [x] Tests parameterised over tokenizers (`tests/tokenizers.py`, `FOLD_TOKENIZER_PATH`); 36/36 pass on Qwen3-8B and Qwen3.5-9B.
- [x] `MODEL_PATH` / `MODEL_TAG` / `FOLD_MODEL_PATH` through workers, entrypoints, `scripts/train_bc.sh`.
- [ ] `analyze_fix_campaign.py` / `render_rollout_traces.py`: add a `--model-tag` and read the model-level HDFS layout
      (`fold_replication/<model_tag>/{ckpt,val_dump,train_logs,rollout_dump}/<arm>`); Qwen3-8B stays under `*_fix/`.
- [ ] Sampling: RL rollouts T=1.0 top_p 1 (unchanged); the model card recommends `presence_penalty=1.5` in thinking mode for chat —
      do **not** add it to RL rollouts (off-policy vs logprobs), but watch repetition (`is_weird`, `retry_cjk`) in the shakeout traces.
- [ ] Memory check for 9B: vocab 248 320 → logits/log-prob memory ≈ 1.6× Qwen3-8B; keep `ppo_micro_batch_size_per_gpu=1`, chunked logits.
- [ ] Judge shim / grader untouched (gpt-oss-120b); `infra/agent_loop_config.yaml` unchanged.

### C4. Runs on Qwen3.5-9B (WP4)
- [ ] `norl` greedy + T1n4 (base level; expect above Qwen3-8B's 0.207 / 0.170).
- [ ] 3-step `compactiongrpo` shakeout on 4B (`FOLD_STEPS=3`, `FOLD_ROLLOUT_DUMP=1`) → render traces, checklist all green (observations intact,
      think present with pre-filled opener, summary ↔ resume, budget).
- [ ] Arms on 9B: `compactionrl` (paper) → `grpo` → `foldgrpo`; vals every 10 steps; re-eval in-gap vals; final greedy + T1n4, ×4 / ×1.
- [ ] One table Qwen3-8B vs Qwen3.5-9B per arm in the report.

## D. Moving the repo to a new dev box — what does NOT travel with `git clone`

- [ ] **`data/bc_train.parquet` (3.9 MB) and `data/bc_test.parquet` (0.9 MB)** are git-ignored — copy them (or regenerate with the authors'
      recipe in `README.md`); jobs copy them from `$SRC/data/`.
- [ ] Venvs: rebuild from `fold-job-assets/fold_train.tar.gz` / `fold_infra.tar.gz` (`tar xzf -C ~/xiaoxuan/envs`) — or build `fold_train_q35` (C1).
      Path `~/xiaoxuan/envs/<name>` is hard-coded in `infra/worker_*.sh` and `fold_common_bootstrap.sh` (`XD=/home/tiger/xiaoxuan`).
- [ ] Repo location: scripts assume `/home/tiger/xiaoxuan/Comp_Rubric` (`SRC=`) — keep it, or change `SRC`/`XD` in `infra/worker_train.sh`,
      `infra/worker_val_selfcontained.sh`, `infra/jobs/fold_common_bootstrap.sh`, `infra/infra_resubmit_loop.sh`, `infra/jobs/status.sh`.
- [ ] Tokenizers for the CPU tests: `~/xiaoxuan/tokenizers/Qwen3.5-9B` (copy from `fold-job-assets/tokenizers/`) and `Qwen/Qwen3-8B` in the HF cache.
- [ ] HDFS mounts: `/mnt/hdfs/mlsys` must be mounted (RW) on the box — assets `/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/`, outputs
      `/mnt/hdfs/mlsys/xiaoxuan/fold_replication/`, W&B key `/mnt/hdfs/mlsys/users/xiaoxuan/arco-job-assets/wandb.key`.
- [ ] After **any** code change that a job must see: commit, retar to `fold-job-assets/foldagent-repo.tar.gz` (+ `.md5`, verify with `md5sum -c`);
      recipe in the 09-16 hand-off; currently at commit f22260d (= this TODO's commit minus docs).
- [ ] Tooling per box (see memory notes): merlin-cli (`--control-plane i18n-tt`, IPv6 pin for `ml.tiktok-row.net` if TLS times out), `mlx`,
      `uv`, `hf` CLI, GitHub SSH key registered, lark-cli re-auth; put PATH in `.profile` (non-interactive shells).
- [ ] `infra/markers/`, `infra/launch_*.log`, `results/` are runtime state (git-ignored) — nothing to carry.
- [ ] Push the repo to a **new** GitHub remote (origin = authors' repo) once you say go; `docs/reports/paper/` has uncommitted LaTeX edits —
      commit or stash them on the old box first.

## D2. Push / repository (BLOCKER for the handoff)
- [ ] Create the GitHub repository `RainyFields/Comp_Rubric` (private) — no `gh` CLI or API token on this box, so it could not be
      created here; remote `github` is already configured. Then `git push -u github main` (SSH works).
- [ ] Optional: rename remotes so `origin` = RainyFields/Comp_Rubric and `upstream` = sunnweiwei/FoldAgent.

## D3. Protocol follow-ups (nonblocking)
- [ ] Observation token budget per turn (`max_calls_per_turn=8` only bounds the number of calls).
- [ ] Consider `max_consecutive_no_call=5` for thinking policies; duplicate-action guard.
- [ ] Dump one live GPU training batch (response_mask stats) during the next training shakeout to close the "NOT YET VERIFIED on device" row.

## E. Housekeeping
- [ ] Delete the superseded `report/` (August draft) or leave it; `infra/babysit_infra.sh` is deprecated (no babysitters).
- [ ] `docs/reports/paper_fix/` and the E2E report need the compaction arms added once A is done.
