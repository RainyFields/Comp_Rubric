# GRPO rollout shakeout and trajectory audit (fixed code) — setup and methodology

## Exact setup

| item | value |
|---|---|
| repo / commit | `~/xiaoxuan/Comp_Rubric`. Shakeout #1 ran **83a3070** (shared grammar + raw-id capture on top of 3f697bf, the four audit fixes); it exposed the branch-loop regression (summary §2). Shakeout #2 and the full runs ran **7c04e37** (malformed-terminal handling + loop protection). Audit tooling: `scripts/shakeout_audit.py`. After the full runs: `plugin.max_calls_per_turn` (default 8) was added (finding F-full-1) — not active in any run reported here |
| checkpoint | `/mnt/hdfs/mlsys/xiaoxuan/fold_replication/ckpt_fix/grpo/global_step_50/actor` — GRPO arm of the fix campaign, **Qwen3-8B** (`Qwen3ForCausalLM`, `model_type=qwen3`, chat template sha1 `b066ba71c1b5` == base Qwen/Qwen3-8B), thinking mode on, no `<think>` prefill; merged to HF with `verl.model_merger` inside the job |
| dataset | BrowseComp-Plus test split, `data/bc_test.parquet` (150 tasks); shakeout subset `data/bc_test_shakeout.parquet` (18 tasks: the rollouts with unclosed think blocks, `return`/`verify` in the main context, no-call turns, 11-branch rollouts, the 5 longest main contexts and 2 clipped-observation cases of the step-50 dump, plus 2 previously-correct controls; row map in `data/bc_test_shakeout_picks.json`) |
| generation | greedy (`val_kwargs.do_sample=False`, n=1), per-turn cap `plugin.turn_max_new_tokens=2048` (now enforced), `max_turn=100`, session timeout 7 200 s, judge gpt-oss-120b via the shim, retriever Qwen3-Embedding-8B, one 8×H100 pod per arm (`infra/worker_val_selfcontained.sh`) |
| active context | **40 960 tokens** (8 192 prompt + 32 768 response = the checkpoint's `max_position_embeddings`; vLLM `max_model_len` = 40 960). A 64k budget is not supported by this checkpoint without RoPE scaling and was therefore not used; both arms get the same 40k window |
| arm A — compaction OFF (`branch`) | fold agent, `workflow=search_branch` (the training-time prompt: search, open_page, branch, finish), `enable_summary=False` (no q_sum, no session reset, no history truncation: `truncate_prompt` only applies to the 8k task prompt, observations are never cut because `process_reward=none`), up to 10 branches of 32k each; the rollout ends at finish, `max_turn`, timeout, or when the main window is exhausted (`llm_none`) |
| arm A' — compaction OFF, no branch (`nobranch`) | same agent with `workflow=search` (search, open_page, finish): isolates the branch tool from compaction |
| arm B — compaction ON (`compact`) | `agents/compaction_agent.py` on the same checkpoint: `workflow=search`, q_sum when the remaining response budget < 8 192, summary ≤ 2 048 tokens (trainable in RL, irrelevant here), resume = task prompt + resume template + last 2 steps as **raw token ids** (`Agent.append_tokens`), ≤ 3 compactions (×4 budget), `mask_unfinished=False` |
| budgets | identical per-turn cap, window, `max_turn`, timeout and tools per arm; total generated tokens are **not** capped identically by construction (A: ≤ 11 windows via branches, B: ≤ 4 windows via compaction) — they are measured and reported per category instead |
| job specs / sids | shakeout #1 (83a3070, tag `_fx_shk_*`): 757bb88155557617 / edf3bc00392bf55a / 0b7b668cc6b37ef7; shakeout #2 (7c04e37, tag `_fx_shk2_*`): ef10ac6d0546ccea / 62dd772e78a7dc6c / 82fbf97e04779fe4; full 150 tasks (7c04e37, tag `_fx_full_*`): abec53a537251edc / a2339ece197b992f / 7f8f41600dda264f (A / A' / B). Old-code reference (same checkpoint, code before 3f697bf): 222c7dd69460d364 (`_capture`). All in `infra/jobs/JOBS.tsv` |
| raw artefacts (HDFS) | `/mnt/hdfs/mlsys/xiaoxuan/fold_replication/results/valonly_grpo_50_greedy<TAG>_sc/{capture/capture_<pid>.jsonl,dump/0.jsonl,judge_calls.jsonl,valonly.log,val_metrics.txt}` with TAG ∈ `_fx_shk_*` (#1), `_fx_shk2_*` (#2), `_fx_full_*` (150 tasks), `_capture` (old code). Capture files hold the raw prompt ids (suffix-of-previous or full, sha1-verified), completion ids, stored-turn ids + masks, per-call results, fork/append_tokens/rollout_end events (23–120 MB each) |

Compliance: public benchmark, `HAS_TT_DATA=False`, `MERLIN_JOB_INSTANCE_TYPE=Evaluation`, `ML_FRAMEWORK=pytorch`, `STORAGE_TYPE=HDFS` in every spec.

## Legal tools and grammar (as the checkpoint was prompted; `agents/tool_spec.py`, `agents/prompts.py`)

| context | tools | multiplicity |
|---|---|---|
| MAIN (`search_branch`) | `search(query, topk)`, `open_page(docid|url)`, `branch(description, prompt)`, `finish(answer, explanation, confidence)` | search/open_page may repeat in one turn ("you can search multiple queries in one turn by including multiple `<function=search>` actions"); **branch one task at a time** ("do not issue multiple simultaneous branches"); finish is terminal |
| BRANCH | `search`, `open_page`, `return(message)` | search/open_page may repeat; return is terminal; finish/branch are refused by the branch guard ("You are in branch mode…") |
| compaction agent (`search`) | `search`, `open_page`, `finish` (+ the fixed q_sum instruction as a user turn) | as above; no branch tool |

Prompt grammar (`PARALLEL_TOOL_PROMPT`): "ONLY reply in the following format with NO suffix … start with `<function=` and end with `</function>` … reasoning BEFORE the function call, NOT after". The shared parser (`agents.parsing.parse_actions`, used by the environment **and** the agents since 83a3070) implements: think blocks removed first (nothing inside `<think>` is ever an action); a call must start a line and be closed; all remaining calls execute in order with ids `<turn>.<k>`; `finish`/`return` must be the only call of the turn; `branch` must be the only call of the turn (single branch per turn — multiple simultaneous branches are **not** supported by the agent, so they are rejected with an `[Error]` observation rather than silently executed). Prompt and parser were kept consistent without changing the prompt text (the checkpoint was trained on it).

## Audit methodology

Capture (`agents/utils._capture_step/capture_action/capture_event`, env `FOLD_PROMPT_CAPTURE=1`): at every policy call the exact prompt ids (suffix-of-previous or full, sha1-verified on replay), the sampled completion ids, the stored turn ids + training mask, the per-call execution results, `fork` (branch inheritance: inherited length + sha1) and `append_tokens` (compaction tail) events, and `rollout_end` (stop reason, score). `scripts/shakeout_audit.py audit` replays the ids and evaluates, per turn: cap, prefix continuity, contiguous presence of every earlier completion in every later prompt (history preservation on ids), fork/tail exactness (sha1), mask validity, parsed-vs-executed consistency, observation = concatenation of per-call observations, template boundaries (`<|im_end|>\n`), template-inserted empty thinks; per rollout: outcome/finish/stop reason, turns, branches, compactions, generated tokens by category (main / branch / summary), prefill tokens, peak context, wall-clock, calls by type, multi-call and malformed turns, duplicate queries/branch prompts, long observations, citation grounding (docids cited in the finish explanation vs docids ever observed), judge verdict. `compare` builds the OFF/ON tables and paired outcome changes.

## Commands

```bash
# tests (59; both tokenizers)
for T in Qwen/Qwen3-8B ~/xiaoxuan/tokenizers/Qwen3.5-9B; do FOLD_TOKENIZER_PATH=$T ~/xiaoxuan/envs/fold_train/bin/python -m unittest discover -s tests -t .; done
# jobs (from infra/jobs; log sids in JOBS.tsv)
merlin-cli --control-plane i18n-tt job-v2 runs create --from-file fold_shk_grpo50_A_branch_h100.json     # + A_nobranch, B_compact; fold_full_* after the shakeout
# audit one arm
P=~/xiaoxuan/envs/fold_train/bin/python; R=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/results
$P scripts/shakeout_audit.py audit --results $R/valonly_grpo_50_greedy_fx_shk_A_branch_sc --arm A_branch --out docs/traces/grpo_fixed_shakeout/work
$P scripts/shakeout_audit.py compare --metrics docs/traces/grpo_fixed_shakeout/work/*_metrics.json --rollouts docs/traces/grpo_fixed_shakeout/work/*_rollouts.json --out docs/traces/grpo_fixed_shakeout/work
```

## Deliverables in this directory

| file | content |
|---|---|
| `summary.md` | completion criteria with evidence, confirmed bugs, OFF/ON comparison, trajectory-quality audit, costs, uncertainties, recommendations, full-set section |
| `readable_traces.md` | complete traces (model text in full, observations truncated with a marker, captured input tails) for the selected rollouts of each arm + the shakeout-#1 loop evidence |
| `trace_audit.jsonl` (git-ignored, 20 MB; copy at `/mnt/hdfs/mlsys/xiaoxuan/fold_replication/results/trace_audits/grpo_fixed_shakeout/`) | per-rollout and per-turn records (raw token references = capture file:line + sha1, model input tail, output, think checks, parsed/executed calls with ids, observation, masks, compaction/fork/tail checks, costs); `run` ∈ shakeout2, shakeout1_regression_evidence |
| `work/` | per-arm `_metrics.json`, `_rollouts.json`, `_readable.md`, `_trace_audit.jsonl`, `comparison.md`, `extra_stats.json`; `work/shakeout1/` the pre-fix run; `work/OLD_branch_18_*` the old-code run restricted to the 18 tasks |
