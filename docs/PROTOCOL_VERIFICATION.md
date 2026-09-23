# Protocol verification report (2026-09-23)

Status legend: **PASS** / **FAIL** / **NOT YET VERIFIED**. Every row names its reproduction command and the raw evidence.
Verified commit: see §0. Code under test for the captured shakeouts: `d2c169e` (shakeout #3) and `7c04e37` (shakeout #2 / full runs).

## 0. Commit and push

| item | value |
|---|---|
| repository | `git@github.com:RainyFields/Comp_Rubric.git` (remote `github`), branch `main` |
| verified commit | rollout/training code under test = **d2c169e** (shakeout #3 ran this tarball; `git diff d2c169e..HEAD -- agents envs verl scripts infra` touches only the audit tool); final repository commit = the commit that adds this line (see `git log -1`, recorded below in §4) |
| push | **NOT YET VERIFIED — BLOCKED**: the GitHub repository does not exist and this devbox has no `gh` CLI or API token to create it (SSH auth to GitHub works). Create the repo, then `git push -u github main`. |
| HDFS tarball / entrypoints | `fold-job-assets/foldagent-repo.tar.gz` (md5-verified) with `.commit` = the tarball's commit; `fold_common_bootstrap.sh`, `fold_val_entrypoint.sh`, `fold_train_entrypoint.sh` copied from the same commit |

## 1. Unit / deterministic checks (CPU, recorded outputs; no vLLM)

Command: `for T in Qwen/Qwen3-8B ~/xiaoxuan/tokenizers/Qwen3.5-9B; do FOLD_TOKENIZER_PATH=$T ~/xiaoxuan/envs/fold_train/bin/python -m unittest discover -s tests -t .; done`
Result: **91 tests — Qwen3-8B: OK; Qwen3.5-9B: OK (skipped=3, the Qwen3-8B id-replay cases)**.

| id | check | test | Qwen3-8B | Qwen3.5-9B |
|---|---|---|---|---|
| U1 | template: plain user turn strips history think, `<tool_response>` keeps it; generation prompt (`<|im_start|>assistant\n` vs `…<think>\n`); enable_thinking=False prefill; system-only prefix (Qwen3.5 raises) | `test_qwen35_template.py`, `test_qwen3_tool_response_wrapping.py` | PASS | PASS (template-level only: no Qwen3.5 checkpoint here) |
| U2 | sampled turn stored as gen prompt + raw ids + eos + `\n` (v2) / no newline (legacy); context == canonical render for canonical text | `test_trace_audit_fixes.py`, `test_protocol_compat.py` | PASS | PASS |
| U3 | turn cap sent to the engine (v2) / ignored (legacy) | `test_trace_audit_fixes.TestTurnCap`, `test_protocol_compat` | PASS | PASS |
| U4 | branch inherits exact ids (v2) / re-renders (legacy, template-normalised empty think) | `test_trace_audit_fixes`, `test_protocol_compat` | PASS | PASS |
| U5 | compaction tail = previous segment's raw ids (v2) / re-render (legacy); rolled-back step only in the tail | `test_compaction_agent`, `test_protocol_compat`, `test_training_batch` | PASS | PASS |
| U6 | action grammar: think never acts; multi search/open_page in order with ids; finish/return/branch alone; unclosed = not executed; per-turn cap 8; legacy grammar reproduces the old env (last adjacent group, think-anchored call) | `test_action_grammar.py`, `test_protocol_compat.TestLegacyGrammar` | PASS | PASS |
| U7 | env: per-call results 1:1, errors do not clobber earlier calls, terminal-with-others rejected, `must_search`, accepted finish recorded | `test_action_grammar.TestEnvironmentExecution` | PASS | PASS |
| U8 | malformed `return` (closed with `<function>`) ends a branch as flagged malformed return; no-call loop protection (3) in branches; can be disabled | `test_loop_protection.py` | PASS | PASS |
| U9 | training batch: input_ids layout, attention_mask, position_ids = cumsum−1, response_mask = sampled ids only (gen prompt, newline, observations, resume, tail, inherited history, padding = 0), `optimized_tokens` = mask sum, rollback boundary — via the trainer's `_agent_loop_postprocess` | `test_training_batch.py` | PASS | PASS |
| U10 | recorded real outputs (multi search / multi open_page / malformed + valid return / empty + non-empty think / cap-cut think / finish / call inside think / compaction summary / 92-call turn): parse + replay through Agent under both protocols; recorded ids are contiguous in the context; masks = sampled ids | `test_protocol_recorded.py` | PASS | PASS (parse); replay SKIPPED (Qwen3-8B ids) |
| U11 | old vs new parser on identical recorded outputs (3 539 old-run + 3 168 new-run completions): env execution differs in 12 + 3 turns (think-anchored calls, the old bug), branch/return decisions identical except 1 | ad-hoc script in this session (numbers in `docs/PROTOCOL.md` §4) | PASS | n/a |

## 2. Captured end-to-end checks on the Qwen3-8B GRPO step-50 checkpoint

Shakeout #2 (`7c04e37`, 18 tasks × 3 arms) and full runs (150 × 3): `docs/traces/grpo_fixed_shakeout/summary.md` §1 and §7.
Shakeout #3 (`d2c169e`): protocol `v2` (A branch, B compaction) and `legacy` (A branch) — rows below.

| id | check | evidence (shakeout #2 / full) | shakeout #3 (v2 A / v2 B / legacy A) |
|---|---|---|---|
| E1 | actual model-input ids == expected serialised prompt (sha1 replay of suffix-chained prompt ids) | 681/681; 6 308/6 308 | PASS 336/336 / PASS 289/289 / PASS 377/377 |
| E2 | history preserved: every earlier completion contiguous in every later prompt; prefix continuity except designed rollbacks | 0 unexpected misses / 0 breaks (both) | PASS 0 unexpected (22 designed rollbacks) / PASS 0 unexpected (1 designed rollback) / PASS 0 unexpected (30 designed) |
| E3 | branch inheritance exact (fork sha1) | 25/25; 353/353 | PASS 39/39 / n/a / n/a (legacy re-renders: 196 prompts with template-normalised empty thinks, as designed) |
| E4 | compaction tail exact (sha1 at the computed offset), resume prompt, next input | 190/190; 1 190/1 190 | n/a / PASS 188/188 tails exact at the computed offsets; resume prompt + next input replayed sha1-exact / n/a |
| E5 | parsed = executed = observation mapping (per-call results, concatenation) | 584/584; 5 497/5 500 (3 = guard/branch-limit artefacts) | PASS 291/291, obs 271/271 / PASS 241/241, obs 234/234 / PASS 327/328 (1 = branch-guard rejection of a `branch` call inside a branch, expected) |
| E6 | no think-block call executed | 0 (calls in think seen 26 / 161) | PASS (3 think-internal calls seen, 0 executed) / PASS (15 think-internal calls seen, 0 executed) / n/a in legacy (0 seen) |
| E7 | 2 048 cap real (`cap_over` 0), cap hits are think cuts | 0 over; 43–51 hits/arm | PASS `cap_over` 0, 6 cap hits / PASS `cap_over` 0, 2 cap hits / by design NOT enforced: 6 completions > 2 048 (max 3 505) |
| E8 | masks valid on every stored turn | 658/658; 6 165/6 165 | PASS 330/330 / PASS 289/289 / PASS 374/374 (stored turn ends at the trained eos) |
| E9 | termination + accounting: `rollout_end` per task, stop reasons, tokens by category | 54/54; 450/450 | PASS 18/18 rollout_end (finish 12, window 6), tokens by category / PASS 18/18 rollout_end (finish 4, budget exhausted 14), main/summary tokens 3 297 / 2 152 per task / PASS 18/18 (finish 15, window 3) |
| E10 | legacy protocol reproduces the training-time format: glued `<|im_end|><|im_start|>`, no cap, re-rendered branch history | — | — / — / **PASS**: 310/377 prompts carry the glued `<|im_end|><|im_start|>` boundary, cap not applied, no fork events (history re-tokenised), all 46 branch ends via the substring `return` rule |
| E11 | live GPU training batch (response_mask statistics from a real training step) | not dumped in any run | **NOT YET VERIFIED** (do in the next training shakeout; `docs/handoff/NEXT_STEPS.md` step 6) |

## 3. Reproduction

```bash
# audits of the captured runs (numbers above)
P=~/xiaoxuan/envs/fold_train/bin/python; R=/mnt/hdfs/mlsys/xiaoxuan/fold_replication/results
$P scripts/shakeout_audit.py audit --results $R/valonly_grpo_50_greedy_fx_shk3_A_branch_v2_sc     --arm shk3_A_v2     --out docs/traces/grpo_fixed_shakeout/work
$P scripts/shakeout_audit.py audit --results $R/valonly_grpo_50_greedy_fx_shk3_A_branch_legacy_sc --arm shk3_A_legacy --out docs/traces/grpo_fixed_shakeout/work
$P scripts/shakeout_audit.py audit --results $R/valonly_grpo_50_greedy_fx_shk3_B_compact_v2_sc    --arm shk3_B_v2     --out docs/traces/grpo_fixed_shakeout/work
# raw traces: $R/<run>/capture/capture_<pid>.jsonl (prompt/completion ids, masks, action/fork/append_tokens/rollout_end events)
```

## 4. Final state

- All E-rows PASS for protocol v2 on both scaffolds (branch, compaction) and the legacy protocol reproduces the training-time
  format (E10). Open: E11 (live GPU batch dump) — NOT YET VERIFIED, nonblocking for inference, to be closed in the first Qwen3.5
  training shakeout. Push: BLOCKED on the GitHub repository creation (§0).
- Final commit SHA: written by the last commit of the session (`git log -1 --format=%H`); the HDFS tarball `.commit` file names
  the same SHA.
