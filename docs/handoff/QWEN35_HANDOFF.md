# Qwen3.5 handoff — Comp_Rubric (context folding / compaction RL on BrowseComp-Plus)

Written 2026-09-23 for a developer starting on a fresh devbox with no knowledge of the previous sessions. Read this, then
`NEXT_STEPS.md`, then `docs/PROTOCOL.md`. Everything else in `docs/` is reference material.

## 1. What this project is

We study **long-horizon search agents trained with RL under a fixed context window** on BrowseComp-Plus (BC-Plus, 680 train /
150 test tasks, local retriever + judge). Two ways of surviving the window are implemented as training arms in one codebase:

| arm | idea | optimiser |
|---|---|---|
| `grpo` | the FoldAgent scaffold (search, open_page, **branch**, finish): sub-tasks run in branches whose contexts are folded away; trained with plain GRPO on the outcome reward | GRPO |
| `foldgrpo` | same scaffold with FoldGRPO's process rewards (Sun et al. 2025) | FoldGRPO |
| `compactiongrpo` | **compaction-aware rollout** (CompactionRL, Li et al. 2026): when the window nearly fills, the policy writes a summary and continues from task prompt + summary + last 2 steps; up to 3 compactions; summaries are trainable; critic-free group-relative advantage | GRPO-style (an ablation, not the paper) |
| `compactionrl` | same rollout with the paper's PPO + critic + cross-trajectory GAE + token-level loss | PPO (paper-faithful; never launched on GPU) |

"GRPO without compaction" = `grpo` (or `norl` evaluation); "compaction-aware training" = `compactiongrpo` / `compactionrl`.
The Qwen3-8B campaigns are finished and written up (`docs/STATUS.md`, `docs/reports/`). **The next step is to move the whole
stack to Qwen3.5** (plan: `docs/plans/2026-09-23_qwen3.5_migration_plan.md`) and rerun the arms there. Nothing has been trained on
Qwen3.5 yet.

## 2. Repository, branch, commit, verified status

- Repository: `git@github.com:RainyFields/Comp_Rubric.git` (remote `github`, pushed 2026-09-23; the authors' upstream `origin` = github.com/sunnweiwei/FoldAgent is read-only for us). Branch `main`.
- Commits: rollout/training code verified at **d2c169e** (shakeout #3 tarball; identical code in every later commit);
  deliverables/docs finalised at **77350e6** (+ this line's commit). The HDFS tarball `foldagent-repo.tar.gz.commit` names the
  commit it was built from.
- Verified: the training–inference interaction protocol (`docs/PROTOCOL.md`) — token stream, action grammar, branch inheritance,
  compaction boundaries, training masks — with 91 CPU tests on both the Qwen3-8B and Qwen3.5-9B tokenizers and captured
  end-to-end shakeouts on the Qwen3-8B GRPO checkpoint (`docs/PROTOCOL_VERIFICATION.md`, `docs/traces/grpo_fixed_shakeout/`).
- Not verified on this devbox: any Qwen3.5 *inference or training* (no Qwen3.5 checkpoint runs here; the Qwen3.5 checks are
  tokenizer/template-level), and the GPU-side dump of a live training batch (the tensor construction is verified on CPU).

## 3. Environment and assets

Paths on the devbox (the scripts hard-code them; keep them or edit `SRC`/`XD` in `infra/worker_train.sh`,
`infra/worker_val_selfcontained.sh`, `infra/jobs/fold_common_bootstrap.sh`, `infra/infra_resubmit_loop.sh`, `infra/jobs/status.sh`):

| what | where |
|---|---|
| repo | `/home/tiger/xiaoxuan/Comp_Rubric` |
| venvs (Qwen3-8B stack) | `~/xiaoxuan/envs/fold_train` (vLLM 0.10.2, transformers 4.57.6, torch 2.8, verl 0.7.0.dev vendored under `verl/`), `~/xiaoxuan/envs/fold_infra` (vLLM 0.11 for search + judge); tarballs `fold-job-assets/fold_train.tar.gz`, `fold_infra.tar.gz` |
| **Qwen3.5 stack (BUILT 2026-09-23)** | `~/xiaoxuan/envs/fold_train_q35` = `bash infra/build_env_q35.sh`: Python 3.12, torch 2.11.0+cu130, vLLM 0.24.0, transformers 5.9.0, verl 0.9.1 (pip), flash-attn 2.8.3 (verl wheelhouse), flash-linear-attention 0.5.2, tensordict 0.10.0, cupy-cuda13x (= verl 0.9.1's `[vllm]` lock; no `causal-conv1d`: verl's Qwen3.5 forward uses fla, `causal_conv1d_implementation=fla`). Freeze `infra/fold_train_q35.freeze.txt`; tarball `fold-job-assets/fold_train_q35.tar.gz` (+`.md5`, `.commit`). Pods need image `modelchef-gpu:1.0.0.54` (CUDA 13.0 compat for driver R535). The vendored `verl/` 0.7 breaks under transformers 5 — never put the repo root on `sys.path` with this venv until WP3 replaces it |
| data | `data/bc_train.parquet`, `data/bc_test.parquet` (git-ignored; copy them), `data/bc_test_shakeout.parquet` (18-task subset) |
| HDFS assets | `/mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/`: repo tarball `foldagent-repo.tar.gz` (+`.md5`, `.commit`), venv tarballs, `fold_common_bootstrap.sh`, `fold_*_entrypoint.sh`, `hf_hub/` (Qwen3-8B, Qwen3-Embedding-8B, BC-Plus corpus), `tokenizers/Qwen3.5-9B/`, `tiktoken/` |
| HDFS outputs | `/mnt/hdfs/mlsys/xiaoxuan/fold_replication/{ckpt_fix,val_dump_fix,train_logs_fix,rollout_dump_fix,results}/` |
| judge | gpt-oss-120b served on the pod (`/mnt/hdfs/mlsys/users/xiaoxuan/models/gpt-oss-120b`), reached through `infra/judge_shim.py` |
| W&B | key at `fold-job-assets/../arco-job-assets/wandb.key`; project `context_folding` |
| jobs | Merlin/Arnold i18n-tt, group 765 ark-eng-algorithm, 8×H100 pods; `merlin-cli --control-plane i18n-tt job-v2 runs create --from-file <spec>`; ledger `infra/jobs/JOBS.tsv`; compliance env vars in every spec (`HAS_TT_DATA=False` etc.) |
| Qwen3.5 tokenizer (offline) | `~/xiaoxuan/tokenizers/Qwen3.5-9B` (copy from HDFS `fold-job-assets/tokenizers/`); Qwen3-8B tokenizer at `~/xiaoxuan/tokenizers/Qwen3-8B` (from `/mnt/hdfs/mlsys/models/Qwen3-8B`) |
| Qwen3.5 weights | shared HDFS `/mnt/hdfs/mlsys/models/Qwen3.5-9B` (19 GB, 4 safetensors shards) and `Qwen3.5-4B` (8.8 GB) — no HF download needed; stage to `/tmp/models/` on the pod (SUPO pattern) |

Per-box tooling: merlin-cli (auth with `--bytecloud-auth`), `uv`, `hf` CLI, SSH key registered on GitHub, PATH in `.profile`.

## 4. Protocol facts you must keep consistent between training and evaluation

`plugin.protocol` (`agents/protocol.py`): **`v2` for every Qwen3.5 run** (training and evaluation). `legacy` exists only to
evaluate the pre-3f697bf Qwen3-8B checkpoints. The five switches it sets (terminator newline, turn cap, branch/tail inheritance
by exact ids, strict action grammar) are documented in `docs/PROTOCOL.md` §4. Safety nets: `max_consecutive_no_call=3`,
`max_calls_per_turn=8`. Set `PROTOCOL=v2` (scripts) / `FOLD_PROTOCOL=v2` (job env_map) explicitly even though it is the default.

Qwen3.5 specifics (verified on the real `Qwen/Qwen3.5-9B` template, `tests/test_qwen35_template.py`):
- the template keeps `<think>` blocks only for assistant turns after the latest *plain* user query, exactly like Qwen3 — our
  `<tool_response>` wrapping and standalone rendering carry over unchanged;
- the generation prompt is `<|im_start|>assistant\n<think>\n` (opener **pre-filled**): sampled text has no `<think>` opener;
  `agents/parsing.find_think` handles it; every text regex must go through `agents/parsing`;
- the template raises `No user query found` for a chat without a plain user message (system-only prefix, resumed compaction
  segment with `resume_keep_task_prompt=False`): `Agent._render_prefix`/`get_generation_prompt`/`truncate_prompt` handle it;
- `<think>`, `</think>`, `<tool_response>`, `<tool_call>` are single tokens; eos `<|im_end|>`; terminator newline `\n` (id 198);
- vocab 248 320 (Qwen3-8B: 151 936) → chunked log-probs (`use_fused_kernels`) at 32k responses; native context 262k.
- **transformers 5 (the q35 venv)**: `tokenizer.apply_chat_template(..., tokenize=True)` returns a `BatchEncoding` unless
  `return_dict=False` is passed — every call in `agents/utils.py` / `scripts/audit_rollout_trace.py` / tests does so since 2026-09-23
  (ids are byte-identical to 4.57). Any new call site must do the same, or use `tokenize=False` + `encode`.

Training-mask semantics (`docs/PROTOCOL.md` §5) are model-independent and tested with both tokenizers.

## 5. Commands

```bash
# 0. tests (91; run with BOTH tokenizers — Qwen3.5 id-replay cases skip by design)
cd ~/xiaoxuan/Comp_Rubric
for T in ~/xiaoxuan/tokenizers/Qwen3-8B ~/xiaoxuan/tokenizers/Qwen3.5-9B; do FOLD_TOKENIZER_PATH=$T ~/xiaoxuan/envs/fold_train_q35/bin/python -m unittest discover -s tests -t .; done
# 2026-09-23 status in the q35 venv: the vendored verl/ makes 33/73 error (transformers 5); with verl/ removed from the tree
# (pip verl 0.9.1) 62/73 pass on both tokenizers — the 11 left are the WP3 items (NEXT_STEPS step 4).

# 1. Qwen3.5 model prep (once the q35 venv exists): download, then seed the HDFS hub cache used by pods
hf download Qwen/Qwen3.5-9B; hf download Qwen/Qwen3.5-4B
cp -rL ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B /mnt/hdfs/mlsys/users/xiaoxuan/fold-job-assets/hf_hub/

# 2. protocol smoke on the Qwen3.5 checkpoint = a captured val-only run of the untrained model (no-RL baseline), greedy, 18 tasks
#    (spec to derive from infra/jobs/fold_shk3_grpo50_A_branch_v2_h100.json: FOLD_ARM=norl FOLD_STEP=base FOLD_MODEL_PATH=Qwen/Qwen3.5-9B
#     FOLD_PROTOCOL=v2 FOLD_VAL_FILE=data/bc_test_shakeout.parquet FOLD_PROMPT_CAPTURE=1; the pod must use the q35 venv tarball)
cd infra/jobs && merlin-cli --control-plane i18n-tt job-v2 runs create --from-file fold_smoke_qwen35_norl_v2_h100.json
# audit the capture (all checks must PASS; see docs/PROTOCOL_VERIFICATION.md for the expected rows)
python scripts/shakeout_audit.py audit --results /mnt/hdfs/mlsys/xiaoxuan/fold_replication/results/valonly_norl_base_greedy_q35_smoke_sc \
    --arm q35_norl --out docs/traces/qwen35_smoke --tokenizer ~/xiaoxuan/tokenizers/Qwen3.5-9B

# 3. training shakeout (3 steps) then the arms
FOLD_STEPS=3 …  fold-train-compactiongrpo-shakeout-h100.json with FOLD_MODEL_PATH=Qwen/Qwen3.5-9B FOLD_PROTOCOL=v2 FOLD_ROLLOUT_DUMP=1
ARM=grpo MODEL_PATH=Qwen/Qwen3.5-9B PROTOCOL=v2 bash scripts/train_bc.sh          # local 8-GPU form; batch form = fold-train-*-h100.json + FOLD_MODEL_PATH
```

## 6. Intended experiments on Qwen3.5 (from `docs/TODO.md` §C, decisions D1–D5)

1. `norl` baseline (greedy + T1 n=4) on Qwen3.5-9B — also the protocol smoke.
2. `grpo` (branch scaffold, no compaction) 100 steps, 32×8, lr 1e-6, window 8k+32k — the "GRPO without compaction" reference.
3. `compactiongrpo` 100 steps (same budget; `COMPACTION_THRESHOLD=8192`, `TAIL_STEPS=2`, `SUMMARY_MAX_TOKENS=2048`,
   `MASK_UNFINISHED=False`, `RESUME_KEEP_TASK_PROMPT` as decided) and `compactionrl` (paper's PPO) — "compaction-aware training".
4. Evaluate every checkpoint with the **same** protocol/knobs it was trained with: greedy + T1 n=4; compaction arms ×4 and ×1.
Consistency rules: same `protocol`, `turn_max_new_tokens`, `max_calls_per_turn`, `max_consecutive_no_call`, window, prompts
(`workflow`), and tool set for training and evaluation of one checkpoint; never evaluate a compaction-trained checkpoint with
the branch prompt or vice versa.

## 7. Existing results (Qwen3-8B, for calibration)

Greedy BC-Plus test (150): no-RL 0.207; GRPO@50 0.300 in-training (0.34–0.37 on the fixed harness); FoldGRPO@40 0.353;
compactiongrpo@100 0.353 (critic-free ablation). Same GRPO checkpoint, fixed harness: branch scaffold 55/150, single window
13/150, inference-time compaction 36/150 (`docs/traces/grpo_fixed_shakeout/summary.md` §7).

## 8. Limitations and troubleshooting

- Greedy is not bit-reproducible on vLLM (batching); compare distributions.
- Observations are unbounded per call; `max_calls_per_turn=8` limits the blow-up; an observation token budget is still open.
- Cap-cut thinks (2 048) + `max_consecutive_no_call=3` can end a rollout early (3/450 rollouts).
- Legacy-mode evaluation of old checkpoints reproduces the *format*, not the old harness's accidental behaviours.
- The 4-hour reclamation grid on the ark H100 queue kills pods; training jobs resume from HDFS checkpoints (entrypoint restores the
  newest `.upload_done` ckpt); the shared infra pod needs the keep-alive loop (user authorisation).
- If a job fails at bootstrap: check `foldagent-repo.tar.gz.commit` matches your intended commit; retar after every code change.
- `merlin-cli` TLS timeouts: pin the IPv6 address of `ml.tiktok-row.net` in `/etc/hosts`.

## 9. Gate before any full-scale Qwen3.5 training

All of: (1) 91 tests pass with the Qwen3.5 tokenizer in the q35 venv; (2) the Qwen3.5 no-RL smoke capture audits clean
(prompt sha1 replay 100 %, masks valid 100 %, fork/tail exact, `cap_over` 0, no unexpected history loss, `rollout_end` for every
task, executed = parsed); (3) a 3-step training shakeout on Qwen3.5-4B or 9B with `FOLD_ROLLOUT_DUMP=1` shows finite loss,
`actor/grad_norm > 0`, rollout-vs-actor log-prob correlation ≥ 0.95 and rendered traces with an all-green checklist;
(4) GDN kernels active (no fp32 fallback warning in the trainer log).
