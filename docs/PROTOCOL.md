# Training–inference interaction protocol (Comp_Rubric)

This document is the contract between rollout collection (training) and evaluation (inference) for every arm in this
repository: what token stream the policy sees, what it is trained on, and how tool calls are parsed and executed. It records
the audit of 2026-09-23 (`docs/traces/grpo_step50_trace_audit.md`, `docs/traces/grpo_fixed_shakeout/`) and the resulting
explicit compatibility setting. Verification status per check is in `docs/PROTOCOL_VERIFICATION.md`.

## 1. One code path for training and inference

Rollouts for training (`scripts/train_fold.py` → `agents/fold_agent.py` / `agents/compaction_agent.py`) and for evaluation
(`infra/worker_val_selfcontained.sh` → the same agent loops with `trainer.val_only=True`) run the **same** `Agent` class
(`agents/utils.py`), the same environment (`envs/local_search.py`) and the same parser (`agents/parsing.py`). The training sample
of a context is `AgentContext.get_data()`: `prompt_ids = sum(chat_ids[:prompt_turn])`, `response_ids = sum(chat_ids[prompt_turn:])`,
`response_mask = token_mask` — i.e. the exact ids the policy received, with the mask marking its own sampled tokens. The trainer's
`AgentLoopWorker._agent_loop_postprocess` pads these (prompt left, response right), builds `attention_mask` and
`position_ids = cumsum(attention_mask) − 1`, and the actor's loss mask is `response_mask` (× `overlong_mask`). Therefore
*training–inference consistency is a property of the rollout code*, and the audit compares the rollout code that trained a
checkpoint with the rollout code that evaluates it.

## 2. Token stream (Qwen3 family templates)

| element | how it is produced | trained? |
|---|---|---|
| task prompt (system + user) | `apply_chat_template` of the first `prompt_turn` messages (prefix diff; anchor query for templates that refuse a system-only prefix, e.g. Qwen3.5) | no |
| generation prompt | template diff with `add_generation_prompt=True` (`<|im_start|>assistant\n`; Qwen3.5 appends `<think>\n`) | no |
| sampled assistant turn | **raw sampled ids** from vLLM + eos if missing (+ terminator newline, see protocol). Raw whitespace is preserved: a model-written `<think>\n</think>` stays as sampled, whereas a template re-render would normalise it to `<think>\n\n</think>\n\n`; so `Agent.context()` equals the canonical template render only when the sampled text is itself canonical | yes: sampled ids incl. eos; not the generation prompt / newline |
| tool observation | `<tool_response>\n…\n</tool_response>` rendered standalone by `render_single_turn` (keeps earlier `<think>` blocks: the template only strips reasoning before the latest *plain* user query) | no |
| instruction turns (branch prompt, q_sum, resume template) | plain user turns rendered standalone (a query boundary by design) | no |
| branch history | v2: parent's exact ids (`Agent.fork`); legacy: re-tokenised from text | no |
| compaction tail | v2: previous segment's exact ids (`Agent.append_tokens`); legacy: re-rendered | no |
| compaction summary | sampled turn after q_sum | yes (`train_summary=True`) |

Thinking mode is on (no `enable_thinking` anywhere); the policy writes its own `<think>…</think>` (Qwen3) or continues the
pre-filled `<think>\n` (Qwen3.5). `agents/model_profile.py` detects the family from the template; `agents/parsing.find_think`
accepts both forms.

## 3. Action grammar (`agents/parsing.parse_actions`, used by env and agents)

Think blocks are removed first — nothing inside `<think>` is ever an action. A call must start a line and be closed
(`<function=NAME>…</function>`); all remaining calls execute **in order** with ids `<turn>.<k>`, up to `plugin.max_calls_per_turn`
(default 8, 0 = off; extra calls are reported in the observation, not executed). `finish`/`return` are terminal and must be the
only call of the turn; `branch` must be the only call of the turn (the prompt says one branch at a time; simultaneous branches
are not supported by the agent). Violations reject the whole turn with an `[Error]` observation. Every executed call has exactly
one result (`env.last_action_results`: call_id, function, status ok/error/rejected, observation); the observation the policy sees
is the concatenation of the per-call observations plus the fixed reminder. Legal tools: MAIN = search, open_page, branch, finish;
BRANCH = search, open_page, return (finish/branch refused by the guard); compaction agent = search, open_page, finish.

A `return`/`finish` opener without `</function>` is a **malformed terminal**: not executable, but a branch ends on it with the
message extracted and `branch_end=malformed_return` recorded (this checkpoint closes ~54 % of its returns with `<function>`;
the old harness accepted them via a substring rule, so this preserves the training-time outcome).

## 4. The compatibility setting: `plugin.protocol`

Checkpoints trained before commit 3f697bf (Qwen3-8B campaigns Aug–Sep 2026: `norl`, `grpo`, `foldgrpo`, `compactiongrpo`) were
collected with a harness that differs from the corrected one. `plugin.protocol` selects which one is reproduced; per-switch
overrides exist (`plugin.<switch>`). Defaults: **`v2`** for every new run. **Evaluate an old checkpoint with `legacy`; train and
evaluate new checkpoints with `v2`; never mix within a checkpoint's lifetime.**

| switch | `legacy` (pre-3f697bf) | `v2` (corrected) | why it matters |
|---|---|---|---|
| `turn_end_newline` | sampled turn ends at `<|im_end|>`, next `<|im_start|>` glued | `<|im_end|>\n` (template-canonical), newline untrained | the token after eos differs; old checkpoints only ever saw the glued form |
| `enforce_turn_cap` | `turn_max_new_tokens` configured but never sent (turns up to 6k) | cap sent to vLLM | cap-cut thinks produce "No function call" turns (≈2–5 % of steps) |
| `branch_inherit` | history re-tokenised from text (`<think>\n</think>` → `<think>\n\n</think>`, empty think inserted for turns without one) | exact ids | branch view of history |
| `tail_inherit` | compaction tail re-rendered | exact ids | segment view of history |
| `action_grammar` | env: line-anchored closed calls on raw text, last adjacent group (<4 newlines), think calls can execute; agent: last closed call anywhere decides `branch`, substring `<function=return>` ends a branch | §3 | measured on 3 539 recorded old-run completions: env execution differs in 12 turns (think-anchored calls), branch/return decisions identical |

Safety nets in both modes: `max_consecutive_no_call` (3: branch forced return / rollout stop `no_call_loop`), `max_calls_per_turn` (8).
Where it is set: `PROTOCOL=` in `scripts/train_bc*.sh`, `FOLD_PROTOCOL` in job `env_map` (train entrypoint and val worker).

## 5. Training masks and batches

Verified on CPU with the trainer's own `_agent_loop_postprocess` (`tests/test_training_batch.py`): `input_ids` = left-padded prompt +
right-padded response; `attention_mask` 1 on real tokens; `position_ids` contiguous across the prompt/response boundary;
`response_mask` = 1 exactly on sampled ids (incl. eos), 0 on generation prompts, terminator newlines, observations, resume turns,
copied tails, inherited branch history and padding; a step rolled back before a compaction summary is absent from the sample that
rolled it back and present (untrained) in the next segment's tail. `optimized_tokens` in `extra_fields` equals the mask sum.
The GPU-side batch of a live training step has not been dumped (see verification report: NOT YET VERIFIED on device).

## 6. Known limitations

- Greedy decoding on this vLLM stack is not bit-reproducible across runs (same prompt → same first completion in 1–3 of 18 tasks);
  compare distributions, not paired rollouts.
- A single observation is not size-bounded (search topk ≤ 50 × 512 words, open_page 4 096 words per page); with several calls per
  turn the main context can overshoot the window and the next step returns `None` (`llm_none`). `max_calls_per_turn` limits the
  worst case; an observation token budget is still recommended.
- Cap-cut thinks and `max_consecutive_no_call=3` interact: three consecutive cut thinks end a rollout (`no_call_loop`, 3 of 450 rollouts).
- Legacy mode reproduces the training-time *format*; it cannot reproduce the old harness's accidental behaviours that depended on
  nondeterministic sampling.
- The Qwen3.5 checks are tokenizer/template-level on this devbox (no Qwen3.5 checkpoint here): see `tests/test_qwen35_template.py`.
