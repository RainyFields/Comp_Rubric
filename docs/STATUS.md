# Compaction-RL project — status (2026-09-23, SF time)

Repo `~/xiaoxuan/Comp_Rubric` (fork of github.com/sunnweiwei/FoldAgent, arXiv:2510.11967; NOT pushed — origin is the
authors' repo, push to a new remote after user OK). One track so far: **BrowseComp-Plus (BC-Plus) deep-research search
agent, Qwen3-8B policy (thinking mode on), gpt-oss-120b judge**, 8×H100 batch jobs on the ark-eng-algorithm queue.
Latest hand-off: `~/xiaoxuan/handoffs/2026-09-16_compactionrl-baseline_HANDOFF.md`. Method details: `docs/baselines/README.md`.

## 1. Methods implemented (training arms)

| arm | rollout scaffold | context management | advantage / loss | status |
|---|---|---|---|---|
| `norl` | single-thread ReAct (search / open_page / finish) | none (32k window) | – | evaluated (base model) |
| `grpo` | FoldAgent scaffold **with** the branch tool | folding available at inference, not rewarded | GRPO (group of 8), outcome reward only, seq-mean-token-mean | trained (fix campaign, 59 real steps) |
| `foldgrpo` | FoldAgent scaffold | learned folding (branch/return), process rewards (flat + scope) | FoldGRPO (paper) | trained (fix campaign, 46 real steps) |
| `compactiongrpo` | single-thread ReAct, no branch tool (`agents/compaction_agent.py`) | **trainable context compaction** (rollout side of CompactionRL, Li et al. arXiv:2607.05378): `q_sum` → policy writes `<summary>` → new segment = task prompt + resume template + last 2 steps verbatim; ≤3 compactions (×4 budget) | rollout-level GRPO advantage broadcast to every segment (critic-free; position factor identity at γ=λ=1; per-segment loss weighting) | **trained, 100/100 steps DONE 2026-09-18 — an ABLATION, not the paper's method** (see audit in `docs/baselines/README.md`) |
| `compactionrl` | same compaction rollout | same (+ paper Eq. 9 resume context via `RESUME_KEEP_TASK_PROMPT=False`) | **= CompactionRL**: PPO + critic, cross-trajectory GAE with length-adaptive λ (`compaction_gae`), critic warm-up 50, 2 critic epochs, **global token-level loss** (`global_token_mean`, added 2026-09-23) | implemented + CPU-tested, **never launched**; shakeout spec ready |

Shared setup (all arms): authors' `train_bc_qwen3_8b.sh` budget — 32 prompts × 8 samples per step, prompt 8 192 / response 32 768
tokens per context, lr 1e-6, 100 steps, val every 10 steps (150 test tasks, greedy, in-process), ckpt every 10 steps to HDFS.
Compaction knobs used in the full run: `COMPACTION_THRESHOLD=8192`, `TAIL_STEPS=2`, `SUMMARY_MAX_TOKENS=2048`, `TRAIN_SUMMARY=True`,
`MASK_UNFINISHED=False` (paper-faithful: budget-exhausted rollouts train with reward 0; the Fold/GRPO arms mask them).

**Rollout-code fixes 2026-09-23** (after the GRPO trace audit, `docs/traces/grpo_step50_trace_audit.md`): per-turn cap now applied,
branches/compaction tails inherit exact token ids, one call grammar for agent and environment, canonical `<|im_end|>\n` turn ends.
All numbers in §3 were produced before these fixes.

Infrastructure that is part of the "method": the **Qwen3 chat-template fix** (commit b94b34f) — observations wrapped in
`<tool_response>` and post-prompt turns rendered standalone, so `<think>` blocks are never stripped and observations never lose
their header (before the fix 13–21 % of tool calls got no observation and ~80 % of thinks were empty).

## 2. Benchmarks trained / evaluated on

| benchmark | role | data | evaluation |
|---|---|---|---|
| **BrowseComp-Plus** (Tevatron corpus, Qwen3-Embedding-8B retriever, local search server) | the only training + eval track | `data/bc_train.parquet` (680) / `data/bc_test.parquet` (150) | greedy (in-training val and val-only jobs) and T=1.0 n=4 val-only jobs; gpt-oss-120b judge (`infra/judge_shim.py`) |

No other benchmark has been trained or evaluated in this repo (SWE / `eval_swe.py` is untouched upstream code).

## 3. Results (BC-Plus test, 150 tasks; greedy unless noted; ±≈0.075 CI at n=150)

### 3.1 Fix campaign (post-template-fix retrain, Sep 10–14) — Fold vs GRPO

| ckpt | greedy | T1 n=4 (600 traj) | notes |
|---|---|---|---|
| no-RL base | 0.207 | 0.170 | |
| GRPO @40 / @50 / @60 | 0.307 / 0.300 / 0.320 | 0.308 / 0.323 / 0.342 | 59 reward-bearing steps (infra reclaimed after) |
| FoldGRPO @40 / @50 | 0.353 / 0.307 | 0.320 / 0.335 | 46 reward-bearing steps |

Statistical tie; Fold branches 3.6/traj vs 2.5, overlong@T1 24–32 % vs 13–18 %, empty-think 78–80 % vs 57–64 %. Reports:
`docs/reports/2026-09-14_foldagent_bcplus_fix_campaign_report.pdf`, paper version `docs/reports/paper_fix/main.pdf`.
E2E token study (Sep 17, `docs/reports/2026-09-17_foldagent_e2e_token_usage_report.pdf`): folding does not reduce peak active
context for RL'd policies; FoldGRPO spends +60 % end-to-end tokens vs GRPO for a non-significant gain.

### 3.2 CompactionGRPO full run (arm `af713ba2cbb770f9`, Sep 16 17:08 → Sep 18 18:12 PDT) — critic-free ablation, NOT CompactionRL

Greedy in-training val (150 tasks):

| step | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| val | 0.247 | 0.020* | 0.287 | 0.027* | 0.307 | 0.293 | **0.360** | 0.340 | 0.340 | 0.353 |

\* in-gap (infra reclaimed on the 4 h grid while val ran) → **re-evaluate ckpt 20 and 40 with val-only jobs** (spec pattern
`infra/jobs/fold_val_grpo_fix_step20_h100.json`, set `FOLD_ARM=compactiongrpo`, `VAL_MAX_COMPACTIONS=3`).
Training: 100 steps, 2 full gap steps (39, 40) + 7 partial-gap steps, train reward 0.09 (steps 1–10) → 0.13–0.14 (30–100),
2.1–2.4 segments per rollout late in training, `overlong_rate` 0 (nothing masked), step time 26–36 min, no zero-grad steps
(the user-authorised infra keep-alive loop resubmitted 13 infra pods and stopped itself when the arm finished).
Step-100 val dump: 59/150 tasks solved in a single window (acc 0.525), 79 finished after ≥1 compaction (acc 0.278),
12 exhausted the ×4 budget (acc 0). Empty-think rate 0.2 % (Fold/GRPO arms: 55–80 %); observations intact 99.3 %.
Checkpoints: `fold_replication/ckpt_fix/compactiongrpo/global_step_{10..100}`; val dumps `val_dump_fix/compactiongrpo/`;
log `train_logs_fix/compactiongrpo/train_compactiongrpo.log`; W&B `context_folding` / `fold_compactiongrpo_2026-09`.

Not yet done for this arm: T1 n=4 evals, single-window (×1) eval, re-eval of steps 20/40, results section in the report.

### 3.3 Sanity of the setup — rollout traces

`docs/traces/` (rendered by `scripts/render_rollout_traces.py`, see `docs/traces/README.md`): full compaction traces
(val step 100 and complete multi-segment training chains from shakeout2) with a per-rollout checklist (observation integrity,
think blocks, summary ↔ resume match, tail verbatim, budget), plus FoldGRPO/GRPO traces for comparison.
Findings from the traces: mechanics are as designed (q_sum → `<summary>` → resume + 2 tail steps; rollback before summary
occurs and is visible as a tail step missing from the previous segment's trained text); ~1 % of observations are clipped at a
segment end (the observation overflowed `response_length`; mask 0; the agent rolls that step back); 17/640 summaries in shakeout2
lacked `<summary>` tags (agent falls back to the think-stripped text); the step-100 policy writes a suffix after the finish call in
some turns ("NO suffix" rule) — harmless for scoring.

## 4. Code map

| area | files |
|---|---|
| rollouts | `agents/fold_agent.py` (Fold/GRPO), `agents/compaction_agent.py` (compaction), `agents/utils.py` (Agent context, tokenisation, `wrap_tool_response`, `render_single_turn`, `CallLLM`), `agents/prompts.py` (system prompts, `COMPACTION_SUMMARY_PROMPT`, `COMPACTION_RESUME_TEMPLATE`), `agents/e2e_ledger.py` |
| trainer | vendored verl 0.7.0.dev under `verl/` — fork drift is only 155 lines in 5 files (`trainer/ppo/core_algos.py` estimators, `ray_trainer.py` dispatch, `ppo/utils.py` need_critic, `trainer/config/algorithm.py`, `vllm_async_server.py` max_tokens cap); agent loops registered in `scripts/train_fold.py` + `infra/agent_loop_config.yaml` |
| launch | `scripts/train_bc_qwen3_8b.sh` (fold/grpo), `scripts/train_bc_compaction{grpo,rl}.sh`; `infra/worker_train.sh`, `infra/worker_val_selfcontained.sh`; batch specs + entrypoints + ledger in `infra/jobs/` (`JOBS.tsv`, `status.sh`); `infra/infra_resubmit_loop.sh` (keep-alive, user-authorised only) |
| analysis / reports | `scripts/analyze_fix_campaign.py` (per-step tables from HDFS mirrors; works for any arm: `--arms compactiongrpo`), `scripts/render_rollout_traces.py`, `scripts/e2e_metrics.py`, `docs/reports/build_*.py` |
| tests (25, CPU, pass 2026-09-23) | `tests/test_compaction_agent.py`, `test_compaction_advantage.py`, `test_qwen3_tool_response_wrapping.py`, `test_e2e_ledger.py`, `test_e2e_metrics.py` |

Job assets on HDFS: `/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/` (venv tarballs `fold_train` = vllm 0.10.2 / transformers 4.57.6 /
torch 2.8, `fold_infra` = vllm 0.11.0; repo tarball = commit b1f445c — **retar after any code change that must run in a job**).

## 5. Open items (tracked in `docs/TODO.md`)

0. **Replicate the paper's actual method**: shakeout `compactionrl` (`infra/jobs/fold-train-compactionrl-shakeout-h100.json`, 3 steps,
   critic warm-up 1, rollout dumps) → full 100-step run (`fold-train-compactionrl-h100.json`, warm-up 50). Decide `PAPER_PROTOCOL`
   (group size 1 / batch 128 / lr 2e-6 vs FoldAgent parity 32×8) and `RESUME_KEEP_TASK_PROMPT` (paper Eq. 9 = False). Report
   compactiongrpo as the GRPO ablation.
1. compactiongrpo: re-eval steps 20/40; T1 n=4 + single-window evals of ckpt 70 and 100 (val-only jobs); extend
   `analyze_fix_campaign.py` with compaction stats (finish rate, compactions/rollout, summary length) and add the arm to the report.
2. Optional arms: `compactionrl` (PPO + critic, ~1.5–2× step time), `TRAIN_SUMMARY=False` ablation, single-window eval of GRPO.
3. **Qwen3.5 migration** — plan in `docs/plans/2026-09-23_qwen3.5_migration_plan.md` (decisions D1–D5 need the user).
4. Push to a new GitHub remote (user OK pending); LaTeX `docs/reports/paper/` has the user's uncommitted edits — never touch.
