# Frozen evaluation contract (Experiment I, decision D8, 2026-09-24)

Everything below is held fixed for the main Qwen3.5 comparison (all methods, both model sizes, both budget tracks). Any
change needs a new contract version and invalidates cross-run comparisons. Values are read from the code at commit
`94ac367` (files named per row); the job assets pin the data revisions.

## Retrieval

| item | frozen value | where |
|---|---|---|
| corpus | `Tevatron/browsecomp-plus-corpus`, HF snapshot `b27b02bc3e45511b8b82a13e6f90ce761df726f6` (job cache `fold-job-assets/hf_hub/`) | `envs/search_server.py` `CORPUS_DATASET` |
| corpus embeddings | `miaolu3/browsecomp-plus` `corpus_embeddings.pkl`, snapshot `9f600f47c5ee9a6251ec5521eb279d8dc5df2966` | `envs/search_server.py` |
| retriever | `Qwen/Qwen3-Embedding-8B`, snapshot `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`, last-token pooling, query max length 8 192 | `envs/search_server.py` |
| search results per call | `topk` argument, default **10**; several `query` values in one call split the budget `max(10 // (n_queries + 1), 2)` each | `envs/local_search.py` (tool spec `agents/tool_spec.py`) |
| result snippet | first **512 words** of the document; a document already shown in an earlier search of the same rollout gets a one-line "already seen" stub and counts 0.25 towards `topk` | `envs/local_search.py` ~L426–432 |
| `open_page` | document by docid/url from prior results, truncated to the first **1 000 words** + `[Document is truncated.]` | `envs/local_search.py` `keep_first_n_words` |
| tool set | `search`, `open_page`, `finish` (+ `branch`/`return` only for branch-scaffold arms; summary/compaction never a tool) | `agents/tool_spec.py`, `agents/prompts.py` |
| parallel tool calls | **1 per assistant turn** (decision D5; `plugin.max_calls_per_turn=1`; extra calls are ignored with a notice) | `agents/parsing.py`, `envs/local_search.py` |

## Judge

| item | frozen value | where |
|---|---|---|
| judge model | `openai/gpt-oss-120b`, HDFS snapshot `/mnt/hdfs/mlsys/users/xiaoxuan/models/gpt-oss-120b`, served by vLLM (TP2 on the val pod / TP4 on the infra pod, `--enforce-eager`, max model len 16 384); every judge-style request (`gpt-5-nano`, `gpt-4o-mini`, `gpt-4.1`, `gpt-*`) is coerced to it by `infra/judge_shim.py` | `infra/worker_val_selfcontained.sh`, `infra/worker_infra.sh`, `infra/judge_shim.py` |
| judge prompt | `GRADER_TEMPLATE` (BrowseComp grader: extracted_final_answer / reasoning / correct yes-no / confidence), version = the text in `envs/local_search.py` at this commit | `envs/local_search.py` L16–34 |
| scoring rule | 1. exact match after normalisation (`em_score`: NFKD de-accent, lower-case, punctuation/whitespace collapse; also head-before-colon match) → 1; 2. empty prediction → 0; 3. LLM judge, up to 3 attempts until the report parses, `correct: yes` → 1; 4. if 0 and `relaxed_em` (parenthetical removal, quote/punctuation stripping) → one more judge call | `envs/local_search.py` `judge()`, `em_score`, `relaxed_em` |
| answer normalisation | the three BrowseComp label typo patches in `judge()` (kept) | `envs/local_search.py` L213–216 |
| judge sampling | vLLM defaults from the shim (no temperature injected for judge calls); judge outputs logged to `judge_calls.jsonl` per run | `infra/judge_shim.py` |

## Data

| item | frozen value |
|---|---|
| train | `data/bc_train_580.parquet` (580 tasks = the 680-task file minus the validation split) |
| validation | `data/bc_val.parquet` (100 tasks; seed 20260924; ids in `docs/splits/bc_val_split.json`; `scripts/make_val_split.py --check`) |
| held-out test | `data/bc_test.parquet` (150 tasks) — not touched until the protocol and checkpoint-selection rules are frozen |
| shakeout subset | `data/bc_test_shakeout.parquet` (18 test tasks; preflight/protocol audits only, never for model selection) |

## Rollout-side constants that are part of the contract (decision D5, initial values, to be calibrated on pilots)

| item | value |
|---|---|
| per-call occupied-context ceiling | 65 536 tokens (prompt + generation of one model call) |
| assistant generation cap per call | 8 192 new tokens (initial) |
| assistant turns per rollout | 100 |
| tool invocations per rollout | 100 |
| consecutive no-call turns | 3 (`plugin.max_consecutive_no_call`) |
| policy sampling (RL rollouts and T=1 evals) | temperature 1.0, top-p 1.0, no penalties; greedy evals temperature 0 |
| thinking | on; historical thinking preserved verbatim in the main context (`plugin.protocol=v2`) |
| evaluation rollouts | greedy ×1 + T=1 ×5 per task (all-attempt metrics primary) |

Changes to this file: bump the version line at the top, note the date and the commit, and record which runs used which version in `infra/jobs/JOBS.tsv`.
