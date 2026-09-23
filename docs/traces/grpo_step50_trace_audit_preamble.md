## 0. Verified configuration (not assumed)

| item | value | how verified |
|---|---|---|
| checkpoint | `fold_replication/ckpt_fix/grpo/global_step_50/actor` (fix campaign, GRPO arm, in-training greedy val = 0.300) | `huggingface/config.json`: `Qwen3ForCausalLM`, `model_type=qwen3`, vocab 151 936, `transformers 4.57.6` |
| base model | **Qwen3-8B** (not Qwen3.5 — the "Qwen3.5 pre-fills the opener" wording in the older checklist was a generic note; no Qwen3.5 run exists) | checkpoint `chat_template.jinja` sha1 `b066ba71c1b5` == `Qwen/Qwen3-8B` (local HF cache) |
| thinking mode | on: generation prompt is `<|im_start|>assistant\n` (no pre-filled `<think>`); `enable_thinking` never set anywhere in the repo; policy writes its own `<think>…</think>` | `Agent.get_generation_prompt()` on the checkpoint tokenizer; `grep enable_thinking` over agents/scripts/infra |
| sampling for this dump | greedy (`val_kwargs.do_sample=False`, in-process val at step 50), 150 BC-Plus test tasks, `plugin.workflow=search_branch`, `max_session=10` branches, `enable_summary=False` (no main-context summaries), `turn_max_new_tokens=2048` (see finding F4: not applied) | `scripts/train_bc_qwen3_8b.sh`, `infra/worker_train.sh` |
| observation wrapping | every environment observation appended as `<tool_response>\n…\n</tool_response>` (commit b94b34f fix) | `agents/fold_agent.py` line "agent['main'].append({'role': 'user', 'content': wrap_tool_response(observation)})"; verified below on all 605 user turns |

### 0.1 What the Qwen3-8B template actually does (executed with the checkpoint tokenizer, `agents/utils.Agent`)

```
[1] plain user turn after an assistant think turn  -> the earlier <think> block is STRIPPED:
    ...<|im_start|>assistant\n<function=search>…</function><|im_end|>\n<|im_start|>user\nPLAIN INSTRUCTION<|im_end|>\n<|im_start|>assistant\n
[2] <tool_response>-wrapped user turn after the same assistant turn -> <think> block KEPT:
    ...<|im_start|>assistant\n<think>\nreason-1\n</think>\n\n<function=search>…</function><|im_end|>\n<|im_start|>user\n<tool_response>\nOBS\n</tool_response><|im_end|>\n<|im_start|>assistant\n
[3] generation prompt suffix: '<|im_start|>assistant\n'      (enable_thinking=False would be '<|im_start|>assistant\n<think>\n\n</think>\n\n')
[4] render_single_turn(assistant WITH think)    -> '<|im_start|>assistant\n<think>\nreason-1\n</think>\n\n<function=…><|im_end|>\n'   (kept)
    render_single_turn(assistant WITHOUT think) -> '<|im_start|>assistant\n<think>\n\n</think>\n\n<function=…><|im_end|>\n'         (EMPTY think INSERTED by the template)
    render_single_turn(assistant '<think>\n</think>' empty) -> '<|im_start|>assistant\n<think>\n\n</think>\n\n…'              (one newline -> two)
    render_single_turn(wrapped obs)             -> '<|im_start|>user\n<tool_response>\nOBS\n</tool_response><|im_end|>\n'
[5] a sampled completion as STORED (raw ids + eos): '<|im_start|>assistant\n<function=…></function><|im_end|>'   (no trailing newline)
```

### 0.2 Evidence chain: the dump IS the main-context model input

`Agent.step()` sends `self.context()` = `sum(chat_ids) + generation_prompt` token ids to the policy (`agents/utils.py`, `CallLLM` →
`server_manager.generate(prompt_ids=…)`). Assistant turns are stored as the **raw sampled ids + eos** (`Agent.append`, mask 1),
user turns as `render_single_turn` ids (mask 0). `AgentContext.get_data()` returns `prompt_ids = sum(chat_ids[:prompt_turn])`,
`response_ids = sum(chat_ids[prompt_turn:])`; the trainer decodes those (special tokens stripped) into the dump's `input`/`output`.
Therefore the model input before assistant turn *t* is exactly the prefix of `input + output` up to turn *t*, plus
`<|im_start|>assistant\n`. The only information lost in the dump is the special tokens; the audit re-renders every prompt
and every user turn with the checkpoint template and compares: **150/150 prompts and 605/605 user turns match byte-for-byte**
(after stripping the same special tokens). Assistant turns cannot be re-verified from text (they are raw ids), which is why
the capture job exists (§0.3).

What the dump does **not** contain: branch sub-contexts (`agent[branch]` is a separate `Agent`; only its return message reaches the
main context), and the special tokens.

### 0.3 Ground-truth capture (diagnostic evaluation)

`agents/utils._capture_step` (env `FOLD_PROMPT_CAPTURE_DIR`, added 2026-09-23) writes, at **every** `Agent.step`, the exact prompt
ids decoded **with** special tokens plus the sampled completion, for main and branch agents, tagged with the rollout uid.
Diagnostic job `222c7dd69460d364` (`infra/jobs/fold_val_grpo_fix_step50_capture_h100.json`: same checkpoint, greedy, 150 tasks,
`FOLD_PROMPT_CAPTURE=1`) was submitted 2026-09-23 10:55 PDT and mirrors to
`fold_replication/results/valonly_grpo_50_greedy_capture_sc/capture/`. Re-run this audit with `--capture <that dir>` to replace the
reconstructed prompts by captured ones (columns `captured_*`, `status_captured`, branch token accounting). Until then, everything
about branch *contents* below is derived from the code path, not from observed prompts.

## 1. Findings

**Primary question — does the policy receive its previous `<think>` blocks?** In the main context: **yes, verbatim.** All 931
non-empty history think blocks and all 831 empty ones are present in every later model input (they are raw sampled ids, never
re-rendered; observations are `<tool_response>`-wrapped so the template's query-boundary stripping never triggers — template
behaviour [1]/[2] above). Tool calls do not alter the history. There is no compaction in this arm (`enable_summary=False`; the
CompactionRL arms are separate, see `docs/traces/2026-09-23_compactiongrpo_*`).

**Branch boundaries DO alter the history the branch sees (code-derived, capture pending):**

- F1 — `fold_agent.py` builds a branch as `Agent(llm_client, agent['main'].messages(), …)`: the main history is **re-tokenised
  from text**, so every inherited assistant turn goes through `render_single_turn` (template [4]). Content of non-empty thinks is
  preserved (300/732 turns render identically), but all 428 model-generated empty thinks — the model always writes
  `<think>\n</think>` — are rewritten to `<think>\n\n</think>\n\n` (one newline becomes two), and the 4 turns whose think block was
  never closed get a template-**inserted** empty `<think>\n\n</think>` in front of their text. Location: `agents/fold_agent.py`
  (branch creation) + `agents/utils.AgentContext.get_turn_context` / `render_single_turn`. Impact: token-level differences in
  mask-0 context only; the branch's own generations are raw ids. Not a loss of reasoning, but the branch never sees the main
  context byte-identically, and "empty think" in a branch's inherited history is template-normalised, not model-emitted.
- F2 — When the branch returns, the main context only receives `Branch has finished its task, the returned message is: …`
  (wrapped); the branch's thinks, tool calls and observations are never in the main context (by design of folding). 333 such
  returns + 3 branch-limit messages in this dump; 336 branch calls in 115/150 rollouts.

**Tool-call / parser integrity (executed = what actually ran):**

- F3 — Two different parsers act on one turn: `fold_agent` uses the lenient `agents.parsing.extract_fn_call` (last
  `<function=…>…</function>` anywhere, incl. inside `<think>`) **only to detect `branch`**; everything else goes to the environment's
  `envs/local_search.extract_fn_call` (line-anchored regex, last *adjacent group*, **every call of that group executes**). Measured:
  18 turns had 2–4 calls executed by the env in one turn (observations concatenated); 33 turns contained an unclosed
  `<function=return>` (model-emitted, short turns of 140–1 059 tokens, so not a length cut) that the lenient parser accepts but the
  env rejects → observation "No function call was detected" (40 such observations in total, 7 for turns with no call at all).
  12 turns carry tool-call XML inside the think block; **0** of them were executed from inside the think (the executed call was
  always a later, line-anchored one). 3 observations are `[Error] The function "return"/"verify" is not supported` (the policy calls
  branch-mode tools in the main context). Observation kind matched the executed call in 605/605 user turns.
- F4 — **Bug:** `agents/utils.CallLLM._create_completion` computes `max_tokens = min(max_new_tokens, plugin.turn_max_new_tokens)` but
  sends `max_new_tokens` — the per-turn cap of 2 048 is never applied (dead variable). 35 assistant turns in this dump exceed 2 048
  tokens (max 6 099). Same code path for all arms and for training.
- F5 — Sampled assistant turns are stored as raw ids ending in `<|im_end|>` with **no newline**; the next user turn starts with
  `<|im_start|>user`. The template's canonical rendering has `<|im_end|>\n<|im_start|>`. So every sampled turn boundary in the model
  input lacks one `\n` relative to the training-data convention (visible in the dump as `</function>user\n<tool_response>`; 731/732
  turns). Consistent between rollout and update (same ids), so no rollout/actor mismatch — a deviation from the instruct format only.
  Location: `agents/utils.Agent.append` (completion branch: appends eos, no newline).
- F6 — 23 observations are cut at the end of the dump: the rollout's last observation overflowed `response_length` (32 768) and the
  training sample is clipped (mask 0). The policy did see the full observation in rollout (the context is not clipped there); the
  next step then failed the budget and the rollout ended (`stop_reason` max/timeout), 23/150 rollouts not finished.
- F7 — 4 turns with an unclosed `<think>` (rollouts 17, 52, 54, 62; 3–1 230 tokens; one is just `<think>\n` + eos): genuinely
  model-emitted malformed thinking, not template artefacts.

## 2. Unresolved (needs the capture job or a code change)

1. Byte-level confirmation of the reconstructed serialised prompts (assistant raw ids cannot be re-derived from text): capture job
   `222c7dd69460d364`, then `--capture`.
2. Branch sub-context contents and token usage at step 50 (not dumped): same capture. Supplementary numbers from the step-60
   E2E ledger (`results/valonly_grpo_60_greedy_e2e_branch_sc/ledger`, same scaffold): branches inherit a median 7 528 tokens
   (p90 20 983) of re-rendered history, end at a median 33 635-token context, generate a median 776 tokens each; main context ends at
   a median 10 709 tokens (p90 38 275); total generated per rollout median 3 800 (main 2 171 + branches), 2.41 branches/rollout.
3. Whether the whitespace normalisation of F1 measurably changes branch behaviour — would need an A/B with raw-id inheritance
   (`Agent` accepting pre-tokenised turns from the parent).

## 3. Suspected bugs and where to fix them

| # | location | fix |
|---|---|---|
| F4 | `agents/utils.py`, `CallLLM._create_completion`: `max_tokens = min(...)` unused | use it: `max_new_tokens = min(max_new_tokens, turn_max_new_tokens)` (changes rollout behaviour → decide before the next campaign) |
| F1 | `agents/fold_agent.py` branch creation (`Agent(llm_client, history, …)`) re-tokenises text | inherit the parent's `chat_ids`/`token_mask` for the inherited turns instead of re-rendering (keeps thinks byte-identical) |
| F3 | `agents/parsing.extract_fn_call` vs `envs/local_search.extract_fn_call` | use one parser for branch detection and execution (the env's), so a lenient `branch` match inside think cannot fork a branch the env would reject |
| F5 | `agents/utils.Agent.append` | append `\n` after eos for sampled turns if canonical formatting is wanted (must be applied in training and rollout together) |
