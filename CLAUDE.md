# Comp_Rubric — instructions for coding agents

Context folding / compaction RL for long-horizon search agents on BrowseComp-Plus. Read this file, then the
documents below in order, before starting any task. Do not act on assumptions the docs contradict.

## Read in this order
1. `docs/README.md` — index of all docs
2. `docs/handoff/QWEN35_HANDOFF.md` — what the project is, environment, paths, commands, gates
3. `docs/handoff/NEXT_STEPS.md` — ordered checklist; the current position is the first unticked box
4. `docs/STATUS.md` — methods implemented, arms, results so far, open items
5. `docs/TODO.md` — to-do list (A replicate CompactionRL, C Qwen3.5 migration, D new devbox, E housekeeping)
6. `docs/PROTOCOL.md` + `docs/PROTOCOL_VERIFICATION.md` — training–inference contract; what is / is not verified
7. `docs/baselines/README.md` — arm table, CompactionRL paper→code mapping, knobs
8. `docs/plans/2026-09-23_qwen3.5_migration_plan.md` — Qwen3.5 work packages, decisions D1–D5

Skim only when needed: `docs/traces/` (rendered rollouts + audits), `docs/reports/` (finished reports).
**Never edit `docs/reports/paper/`** (the user's uncommitted LaTeX review edits).

## Facts to internalise
- Two scaffolds in one codebase: FoldAgent (branch/return sub-contexts, arms `grpo`/`foldgrpo`) and CompactionRL
  (summarise-and-resume). `compactiongrpo` is a critic-free **ablation**; the paper-faithful PPO + critic arm is
  `compactionrl` and has never been launched on GPU.
- All Qwen3-8B campaigns are finished and written up. Nothing has been trained on Qwen3.5 yet; that is the goal.
- `plugin.protocol`: `legacy` only to evaluate pre-3f697bf Qwen3-8B checkpoints; `v2` for every new run (train AND
  eval). Never mix protocols within a checkpoint's lifetime. Set `PROTOCOL=` (scripts) / `FOLD_PROTOCOL` (job env_map)
  explicitly even though `v2` is the default.
- Rollout/training code verified at commit `d2c169e` (captured shakeouts, 91 CPU tests on both tokenizers). The vendored
  `verl/` (0.7.0.dev) cannot load Qwen3.5: the Qwen3.5 stack needs the migration plan's WP2/WP3 (new venv, verl 0.9.1)
  before any Qwen3.5 GPU run.
- Not yet verified: any Qwen3.5 inference or training; a live GPU training-batch dump (verification row E11).
- Remotes: `github` = git@github.com:RainyFields/Comp_Rubric.git (ours). `origin` = the paper authors' upstream
  (github.com/sunnweiwei/FoldAgent): **read-only, never push there**.

## Working rules
- GPU work runs as Merlin/Arnold **batch jobs** (job-v2, control plane `i18n-tt`, group 765 `ark-eng-algorithm`,
  H100 first then A100), never as interactive mlx workers for anything over ~1 h. Every job spec carries the data
  compliance env vars (`HAS_TT_DATA=False`, `MERLIN_JOB_INSTANCE_TYPE`, `ML_FRAMEWORK`, `STORAGE_TYPE`) and is recorded
  in `infra/jobs/JOBS.tsv` (sid, spec, purpose, outcome). Every training run is tracked in W&B (project `context_folding`).
- Before launching anything on GPU: state the spec, model, protocol, data classification and expected cost, and wait
  for the user's OK. Do not launch full Qwen3.5 training unless the user asks for it in that session.
- No babysitter / poll / auto-relaunch loops without explicit user approval. Launch, record the sid, check when asked.
- After any change under `agents/`, `envs/`, `verl/`, `scripts/` run the tests with BOTH tokenizers
  (`QWEN35_HANDOFF.md` §5 step 0); expect `OK` (Qwen3-8B) and `OK (skipped=3)` (Qwen3.5-9B). Add a test for every
  protocol-relevant change; recorded-output fixtures live in `tests/fixtures/`.
- Training–inference consistency is a property of the rollout code: any change to tokenisation, parsing, branch or
  compaction handling must be reflected in `docs/PROTOCOL.md`, gated by `plugin.protocol` if it changes behaviour for
  existing checkpoints, and re-verified (`docs/PROTOCOL_VERIFICATION.md`).
- Keep the source-of-truth docs current for the next agent: `docs/STATUS.md`, `docs/TODO.md`,
  `docs/handoff/NEXT_STEPS.md`, `infra/jobs/JOBS.tsv`. Write a handoff to `~/xiaoxuan/handoffs/<date>_<topic>_HANDOFF.md`
  at ~60 % context or at the end of a task, and check the newest handoff there (`ls -t`) before resuming work.
- Report times in San Francisco time. Convert PDFs / Office docs to Markdown (markitdown) before reading them.
- Commit small with clear messages; push to `github` only; never force-push; never commit `data/`, prompt captures,
  checkpoints, W&B keys or other secrets (`.gitignore` already covers the large trace `.jsonl` files).
- Reports follow the house standard: PDF + Markdown + assets folder with data and a rerunnable builder script.

## Before starting a task, report back (≤ 15 lines)
(a) current position in `NEXT_STEPS.md` and what blocks it; (b) environment pieces missing on this box (venvs, data
parquet files, tokenizers, HDFS mount, merlin-cli auth); (c) the exact first three commands you would run; (d) anything
in the docs that contradicts the code.
