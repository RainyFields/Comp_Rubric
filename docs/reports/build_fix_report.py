#!/usr/bin/env python3
"""Build the FoldAgent BC-Plus *fix-campaign* report (house experiment-report standard,
COMMON_SETUP.md §6): Qwen3 think-stripping bug -> fix -> base contrast -> RL retrain (FoldGRPO vs
GRPO) -> final evals. Rerunnable end-to-end; supersedes the 2026-08-12 report's absolute numbers.

Usage:
  python3 build_fix_report.py                       # stage 1: data -> assets + figures + markdown, then PDF
  <fold_infra python> build_fix_report.py --render-pdf   # stage 2 only (weasyprint)

Inputs (read-only):
  /mnt/hdfs/mlsys/xiaoxuan/fold_replication/train_logs_fix/<arm>/train_<arm>.log   trainer logs (fixed code)
  /mnt/hdfs/mlsys/xiaoxuan/fold_replication/val_dump_fix/<arm>/<step>.jsonl        in-training greedy val dumps
  /mnt/hdfs/mlsys/xiaoxuan/fold_replication/results/valonly_<tag>_fix_sc/           val-only jobs (dump, judge log, metrics)
  results/valonly_<tag>/dump/0.jsonl                                                  August (unpatched) dumps, for the contrast
Outputs:
  docs/reports/2026-09-14_foldagent_bcplus_fix_campaign_report.{md,html,pdf}
  docs/reports/2026-09-14_foldagent_bcplus_fix_campaign_report_assets/  (figures paired with CSV/JSON + README.md)
"""
import csv, glob, json, os, re, statistics, subprocess, sys

ROOT = "/home/tiger/xiaoxuan/FoldAgent"
H = "/mnt/hdfs/mlsys/xiaoxuan/fold_replication"
RD = f"{ROOT}/docs/reports"
STEM = "2026-09-14_foldagent_bcplus_fix_campaign_report"
ASSETS = f"{RD}/{STEM}_assets"
MD, HTML, PDF = f"{RD}/{STEM}.md", f"{RD}/{STEM}.html", f"{RD}/{STEM}.pdf"
FOLD_INFRA_PY = "/home/tiger/xiaoxuan/envs/fold_infra/bin/python"
C_BASE, C_GRPO, C_FOLD = "#7a7a7a", "#0F4D92", "#B64342"
LAST_REAL = {"foldgrpo": 46, "grpo": 59}     # derived below and asserted


def render_pdf():
    import markdown
    from weasyprint import HTML as WHTML
    body = markdown.markdown(open(MD).read(), extensions=["tables", "fenced_code"])
    html = f"""<html><head><meta charset='utf-8'><style>
@page {{ size: A4; margin: 1.9cm 2.0cm; @bottom-center {{ content: counter(page); font-size: 8pt; color: #888; }} }}
body {{ font-family: 'DejaVu Serif', Georgia, serif; font-size: 10pt; line-height: 1.45; color: #1a1a1a; }}
h1 {{ font-size: 15.5pt; }} h2 {{ font-size: 12.5pt; margin-top: 1.2em; border-bottom: 1px solid #ccc; }}
h3 {{ font-size: 11pt; margin-top: 1em; }}
table {{ border-collapse: collapse; margin: 0.8em 0; font-size: 8.6pt; }}
th, td {{ border: 1px solid #999; padding: 3px 6px; }} th {{ background: #f0f0f0; }}
code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8.2pt; background: #f5f5f5; }}
pre {{ background: #f7f7f7; border: 1px solid #ddd; padding: 6px 8px; font-size: 7.6pt; line-height: 1.3;
       white-space: pre-wrap; word-wrap: break-word; }}
pre code {{ background: none; }} img {{ max-width: 100%; }}
blockquote {{ margin: 0.5em 0 1.1em 0; padding: 2px 12px; border-left: 3px solid #bbb; color: #333;
              font-size: 8.8pt; background: #fafafa; }}
</style></head><body>{body}</body></html>"""
    open(HTML, "w").write(html)
    WHTML(HTML, base_url=RD).write_pdf(PDF)
    print(f"PDF written: {PDF}")


if "--render-pdf" in sys.argv:
    render_pdf(); sys.exit(0)

# ============================================================== stage 1
sys.path.insert(0, RD); sys.path.insert(0, f"{ROOT}/scripts")
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from figstyle import apply_publication_style, finalize_figure  # noqa: E402
from analyze_fix_campaign import analyze_dump, parse_train_log, read_val_metrics  # noqa: E402

os.makedirs(ASSETS, exist_ok=True)
apply_publication_style(font_size=12, axes_linewidth=1.4)
readme_entries = []


def save_asset_csv(name, header, rows, desc):
    with open(f"{ASSETS}/{name}", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    readme_entries.append((name, desc))


def save_fig(fig, stem, desc):
    finalize_figure(fig, f"{ASSETS}/{stem}.png", dpi=200)
    readme_entries.append((f"{stem}.png/.pdf", f"Figure image. {desc} Backing data: {stem}.csv"))


def ci(p, n):
    return 1.96 * (p * (1 - p) / n) ** 0.5


# ------------------------------------------------------ training logs
train = {arm: parse_train_log(f"{H}/train_logs_fix/{arm}/train_{arm}.log") for arm in ("foldgrpo", "grpo")}
for arm, rows in train.items():
    assert [r["step"] for r in rows] == list(range(1, 101)), f"{arm} log incomplete"
    last_real = max(r["step"] for r in rows if not r["gap"] and float(r["grad_norm"]) > 0.1)
    assert last_real == LAST_REAL[arm], (arm, last_real)
fold_zero_grad = [r["step"] for r in train["foldgrpo"] if r["step"] > 46]
assert all(float(r["grad_norm"]) == 0.0 for r in train["foldgrpo"] if r["step"] > 46), "fold tail not zero-grad"
grpo_tainted = [r["step"] for r in train["grpo"] if r["step"] > 59 and float(r["grad_norm"]) > 0]
grpo_gaps_pre = [r["step"] for r in train["grpo"] if r["step"] <= 59 and (r["gap"] or r["partial"])]
fold_gaps_pre = [r["step"] for r in train["foldgrpo"] if r["step"] <= 46 and (r["gap"] or r["partial"])]

curve_rows = []
for arm, rows in train.items():
    for r in rows:
        if r["step"] <= LAST_REAL[arm]:
            seg = "real"
            if r["gap"]: seg = "gap_full"
            elif r["partial"]: seg = "gap_partial"
        else:
            seg = "no_reward_server_zero_grad" if float(r["grad_norm"]) == 0.0 else "no_reward_server_exact_match_only"
        curve_rows.append([arm, r["step"], seg, r["reward_mean"], r["overlong_rate"], r["turns_mean"],
                           r["resp_len_mean"], r["entropy"], r["grad_norm"], r["step_s"], r["val_acc"]])
save_asset_csv(
    "fig2_training_curves.csv",
    ["arm", "step", "segment", "reward_mean", "overlong_rate", "num_turns_mean", "response_length_mean",
     "actor_entropy", "actor_grad_norm", "step_seconds", "val_greedy_in_training"],
    curve_rows,
    "Per-step training metrics of the two post-fix RL arms, parsed from the per-step metric lines "
    "('step:N - key:value - ...') of the HDFS-mirrored trainer logs train_logs_fix/<arm>/train_<arm>.log "
    "by scripts/analyze_fix_campaign.py:parse_train_log (last occurrence of a step wins). Columns: "
    "reward_mean = critic/score/mean (batch-mean trajectory reward of the 256-rollout batch, 0-1); "
    "overlong_rate = reward/overlong_rate; num_turns_mean (assistant turns incl. branch-internal); "
    "response_length_mean (tokens); actor_entropy (nats); actor_grad_norm; step_seconds = timing_s/step; "
    "val_greedy_in_training = the in-process 150-item greedy validation (val-core/unknown/reward/mean@1), "
    "every 10 steps, RAW (values inside an infra gap read 0.0 and are superseded by val-only re-evals, "
    "see fig1). segment: 'real' = reward server up; 'gap_full' = rollout entirely without reward server "
    "(all scores 0, zero advantage => zero gradient); 'gap_partial' = server died mid-rollout (short "
    "responses, near-zero reward); after the final infra loss (step 47 Fold / 60 GRPO): "
    "'no_reward_server_zero_grad' (grad_norm exactly 0, weights unchanged) or "
    "'no_reward_server_exact_match_only' (GRPO only: tiny rewards from the exact-match shortcut in "
    "envs/local_search.py:217 without search or judge; grad_norm 0.03-0.07 => weights drifted). Backs fig2.")

# ------------------------------------------------------ validation dumps (greedy) per step
def val_series(arm):
    rows = {0: {"step": 0, "src": "no-RL base (val-only job)", "path": f"{H}/results/valonly_norl_base_greedy_fix_sc/dump/0.jsonl"}}
    for p in glob.glob(f"{H}/val_dump_fix/{arm}/*.jsonl"):
        s = int(os.path.basename(p).split(".")[0])
        if s <= LAST_REAL[arm]:
            rows[s] = {"step": s, "src": "in-training", "path": p}
    for p in glob.glob(f"{H}/results/valonly_{arm}_*_greedy_fix_sc/dump/0.jsonl"):
        s = int(p.split(f"valonly_{arm}_")[1].split("_")[0])
        rows[s] = {"step": s, "src": "val-only re-eval from uploaded checkpoint", "path": p}
    out = []
    for s in sorted(rows):
        a = analyze_dump(rows[s]["path"]); a.update(rows[s]); out.append(a)
    return out


val = {arm: val_series(arm) for arm in ("foldgrpo", "grpo")}
assert [r["step"] for r in val["foldgrpo"]] == [0, 10, 20, 30, 40, 50], [r["step"] for r in val["foldgrpo"]]
assert [r["step"] for r in val["grpo"]] == [0, 10, 20, 30, 40, 50, 60], [r["step"] for r in val["grpo"]]
assert val["foldgrpo"][-1]["src"].startswith("val-only") and val["grpo"][-1]["src"].startswith("val-only")
assert val["foldgrpo"][2]["src"].startswith("val-only") and val["grpo"][2]["src"].startswith("val-only")


# ------------------------------------------------------ T=1.0 n=4 val-only results
def t1n4(arm):
    out = {}
    for d in [f"{H}/results/valonly_norl_base_t1n4_fix_sc"] + glob.glob(f"{H}/results/valonly_{arm}_*_t1n4_fix_sc"):
        s = 0 if "norl_base" in d else int(d.split(f"valonly_{arm}_")[1].split("_")[0])
        m = read_val_metrics(f"{d}/val_metrics.txt")
        a = analyze_dump(f"{d}/dump/0.jsonl")
        out[s] = {"step": s, "mean4": m["val/avg_score"], "best4": m["val-aux/unknown/reward_score/best@4/mean"],
                  "worst4": m["val-aux/unknown/reward_score/worst@4/mean"], "overlong": m["val/overlong_rate"],
                  "turns": m.get("val/avg_num_turns", m.get("val-aux/num_turns/mean")), "n": int(m["val/num_unique_gen_uids"]),
                  "empty_think": a["empty_think_turn_pct"], "obs_intact": a["obs_intact_pct"],
                  "branch": a["branch_per_traj"], "finish": a["finish_pct"], "dir": d}
    return [out[k] for k in sorted(out)]


T4 = {arm: t1n4(arm) for arm in ("foldgrpo", "grpo")}
assert [r["step"] for r in T4["foldgrpo"]] == [0, 40, 50] and [r["step"] for r in T4["grpo"]] == [0, 40, 50, 60]

score_rows = []
for arm in ("foldgrpo", "grpo"):
    for r in val[arm]:
        score_rows.append([arm, "greedy_pass@1", r["step"], LAST_REAL[arm] if r["step"] == LAST_REAL[arm] // 10 * 10 + 10 else min(r["step"], LAST_REAL[arm]),
                           round(r["acc"], 6), r["n"], round(ci(r["acc"], r["n"]), 6), r["src"], os.path.relpath(r["path"], H)])
    for r in T4[arm]:
        score_rows.append([arm, "t1.0_mean@4", r["step"], min(r["step"], LAST_REAL[arm]),
                           round(r["mean4"], 6), r["n"], round(ci(r["mean4"], r["n"]), 6), "val-only job", os.path.relpath(r["dir"], H)])
save_asset_csv(
    "fig1_scores_vs_step.csv",
    ["arm", "operating_point", "checkpoint_step", "effective_trained_steps", "score", "n_graded", "ci95_binomial", "source", "source_path_under_fold_replication"],
    score_rows,
    "BC-Plus pass@1 (judge gpt-oss-120b, 150-item test split) per checkpoint for both post-fix arms. "
    "greedy rows: n=150 (1 rollout/item); t1.0 rows: n=600 (4 rollouts/item, mean@4). checkpoint_step is the "
    "saved checkpoint; effective_trained_steps = real reward-bearing updates it contains (Fold ckpt 50 = 46, "
    "GRPO ckpt 60 = 59; all earlier checkpoints are fully real). score for greedy rows = mean of the per-"
    "trajectory 'score' field of the dump (in-training dumps val_dump_fix/<arm>/<step>.jsonl; step 0 and "
    "any validation that ran inside an infra gap come from val-only jobs results/valonly_<tag>_fix_sc/dump/0.jsonl); "
    "t1.0 rows = 'val/avg_score' in that job's val_metrics.txt. ci95_binomial = 1.96*sqrt(p(1-p)/n) "
    "(i.i.d. approximation; the 4 T=1.0 rollouts per item are clustered, so those CIs are optimistic). Backs fig1.")

beh_rows = []
for arm in ("foldgrpo", "grpo"):
    for r in val[arm]:
        beh_rows.append([arm, "greedy", r["step"], r["empty_think_turn_pct"], r["traj_any_empty_pct"], r["obs_intact_pct"],
                         r["obs_partial"], r["obs_missing"], r["branch_per_traj"], r["search_per_traj"], r["open_per_traj"],
                         r["main_turns_per_traj"], r["chars_per_traj"], r["finish_pct"], r["all_turns_per_traj(log)"], r["overlong_pct(log)"]])
    for r in T4[arm]:
        beh_rows.append([arm, "t1.0", r["step"], r["empty_think"], "", r["obs_intact"], "", "", r["branch"], "", "", "", "", r["finish"], r["turns"], round(100 * r["overlong"], 1)])
save_asset_csv(
    "fig3_behavior_vs_step.csv",
    ["arm", "operating_point", "checkpoint_step", "empty_think_turn_pct", "traj_with_any_empty_think_pct", "obs_intact_pct",
     "obs_truncated", "obs_missing", "branch_calls_per_traj", "search_calls_per_traj", "open_page_calls_per_traj",
     "main_thread_turns_per_traj", "main_thread_chars_per_traj", "finish_pct", "all_turns_per_traj_logged", "overlong_pct_logged"],
    beh_rows,
    "Behavioural statistics per checkpoint, computed by scripts/analyze_fix_campaign.py:analyze_dump over every "
    "dumped trajectory (same dumps as fig1). The dump 'output' is the detokenised MAIN thread; turns are split on "
    "'\\nassistant\\n'. empty_think_turn_pct = share of assistant turns whose <think>...</think> block is empty; "
    "traj_with_any_empty_think_pct = trajectories with >=1 such turn; obs_intact_pct = share of non-finish tool "
    "calls that are followed by a complete <tool_response>...</tool_response> block (obs_truncated = block opened "
    "but cut by the 32k response limit; obs_missing = no block at all - the failure mode of the unpatched code); "
    "branch/search/open_page calls per trajectory = '<function=NAME>' counts (branch-internal calls are folded "
    "away and NOT counted); main_thread_chars = len(output); finish_pct = trajectories ending in a finish call; "
    "all_turns_per_traj_logged and overlong_pct_logged come from the run's val metrics (count all assistant turns "
    "incl. branch-internal). Backs fig3.")

# ------------------------------------------------------ bug contrast: unpatched (August) vs fixed
OLD = {"No-RL base": "norl_base_greedy_sc", "GRPO@100": "grpo_100_greedy_sc", "FoldGRPO@100": "foldgrpo_100_greedy_sc"}
old = {k: analyze_dump(f"{ROOT}/results/valonly_{t}/dump/0.jsonl") for k, t in OLD.items()}
for k, a in old.items():
    assert a["obs_intact_pct"] == 0.0, f"unpatched dump {k} unexpectedly has intact observations"
new = {"No-RL base": val["foldgrpo"][0], "GRPO@59": val["grpo"][-1], "FoldGRPO@46": val["foldgrpo"][-1]}
contrast_rows = []
for (ko, ao), (kn, an) in zip(old.items(), new.items()):
    contrast_rows.append(["unpatched (2026-08)", ko, OLD[ko], ao["acc"], ao["empty_think_turn_pct"], ao["obs_intact_pct"], ao["obs_missing"], ao["main_turns_per_traj"], ao["branch_per_traj"], ao["finish_pct"], ao["chars_per_traj"]])
    contrast_rows.append(["fixed (2026-09)", kn, os.path.relpath(an["path"], H), an["acc"], an["empty_think_turn_pct"], an["obs_intact_pct"], an["obs_missing"], an["main_turns_per_traj"], an["branch_per_traj"], an["finish_pct"], an["chars_per_traj"]])
save_asset_csv(
    "fig4_bug_contrast.csv",
    ["code", "model", "dump_source", "greedy_pass@1", "empty_think_turn_pct", "obs_intact_pct", "obs_missing", "main_thread_turns_per_traj", "branch_calls_per_traj", "finish_pct", "main_thread_chars"],
    contrast_rows,
    "Same parser (analyze_dump) applied to the August unpatched greedy dumps (results/valonly_<tag>/dump/0.jsonl, "
    "self-contained eval path of the 2026-08-12 report) and to the post-fix dumps. Under the unpatched code no "
    "observation carries a <tool_response> wrapper or its <|im_start|>user header (the template-diff slicing bug), "
    "so obs_intact_pct is 0 by construction and obs_missing counts every tool call; under the fix >=95% are intact "
    "(the remainder are cut by the 32k response limit). Note the RL rows compare different training budgets "
    "(unpatched: 100 steps; fixed: 46 Fold / 59 GRPO effective) - the contrast is about context integrity and "
    "thinking behaviour, not a matched-budget accuracy comparison. Backs fig4.")

# ============================================================== figures
# ---- fig1: scores vs step
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.0))
for ax, op, key, n, title in ((axes[0], "greedy", "acc", 150, "Greedy pass@1 (n=150 per point)"),
                              (axes[1], "t1", "mean4", 600, "T=1.0 mean@4 (n=600 per point)")):
    for arm, col, mk, lab in (("foldgrpo", C_FOLD, "o", "FoldGRPO"), ("grpo", C_GRPO, "s", "GRPO")):
        rows = val[arm] if op == "greedy" else T4[arm]
        xs = [min(r["step"], LAST_REAL[arm]) for r in rows]
        ys = [r[key] for r in rows]
        es = [ci(y, n) for y in ys]
        ax.errorbar(xs, ys, yerr=es, color=col, marker=mk, markersize=6.5, lw=2.0, capsize=3, label=lab)
        if op == "greedy":
            for r, x, y in zip(rows, xs, ys):
                if r["src"].startswith("val-only") and r["step"] > 0:
                    ax.plot([x], [y], marker=mk, markersize=11, markerfacecolor="none", markeredgecolor=col, ls="none")
    ax.axhline(val["foldgrpo"][0][key] if op == "greedy" else T4["foldgrpo"][0]["mean4"], color=C_BASE, ls=":", lw=1.4)
    ax.text(1, (val["foldgrpo"][0][key] if op == "greedy" else T4["foldgrpo"][0]["mean4"]) + 0.006, "no-RL base", color=C_BASE, fontsize=9)
    ax.set_xlabel("effective training steps (reward-bearing updates)")
    ax.set_title(title, fontsize=12)
    ax.set_xlim(-2, 64); ax.set_ylim(0.1, 0.45)
    ax.legend(fontsize=9.5, loc="upper left")
axes[0].set_ylabel("BC-Plus pass@1")
axes[0].text(46, 0.115, "ckpt 50\n= 46 real", color=C_FOLD, fontsize=8, ha="center")
axes[0].text(59, 0.115, "ckpt 60\n= 59 real", color=C_GRPO, fontsize=8, ha="center")
save_fig(fig, "fig1_scores_vs_step",
         "Validation score vs effective training steps, both arms, both operating points; hollow rings = val-only re-evaluations.")

# ---- fig2: training curves with gap shading
fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))
for arm, col, lab in (("foldgrpo", C_FOLD, "FoldGRPO"), ("grpo", C_GRPO, "GRPO")):
    rows = train[arm]; xs = [r["step"] for r in rows]
    real = [r for r in rows if r["step"] <= LAST_REAL[arm]]
    tail = [r for r in rows if r["step"] >= LAST_REAL[arm]]
    for ax, key in zip(axes, ("reward_mean", "overlong_rate", "resp_len_mean")):
        ax.plot([r["step"] for r in real], [float(r[key]) for r in real], color=col, lw=1.9, label=lab)
        ax.plot([r["step"] for r in tail], [float(r[key]) for r in tail], color=col, lw=1.3, ls=":", alpha=0.6)
        gx = [r["step"] for r in real if r["gap"] or r["partial"]]
        ax.plot(gx, [float(r[key]) for r in real if r["gap"] or r["partial"]], ls="none", marker="x", markersize=7, markeredgewidth=1.8, color=col)
axes[0].axvspan(46.5, 100.5, color=C_FOLD, alpha=0.07); axes[0].axvspan(59.5, 100.5, color=C_GRPO, alpha=0.07)
axes[0].text(73, 0.27, "no reward server\n(infra reclaimed 09-12 00:10)\nFold: grad = 0 from 47\nGRPO: exact-match-only from 60", fontsize=8, ha="center", color="#333")
axes[0].set_ylabel("batch-mean reward"); axes[0].set_title("Training reward", fontsize=12)
axes[1].set_ylabel("overlong rate"); axes[1].set_title("Context blow-out rate (train)", fontsize=12)
axes[2].set_ylabel("tokens"); axes[2].set_title("Mean response length", fontsize=12)
for ax in axes:
    ax.set_xlabel("training step"); ax.set_xlim(0, 101)
axes[0].legend(fontsize=9.5, loc="upper left"); axes[0].set_ylim(0, 0.32)
save_fig(fig, "fig2_training_curves",
         "Training reward, train overlong rate and response length per step; × = steps hit by an infra gap; dotted = after the final infra loss.")

# ---- fig3: behaviour vs step (greedy)
fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.5))
panels = [("empty_think_turn_pct", "empty <think> turns (%)"), ("branch_per_traj", "branch calls / trajectory"),
          ("obs_intact_pct", "intact observations (%)"), ("chars_per_traj", "main-thread chars")]
for ax, (key, ttl) in zip(axes, panels):
    for arm, col, mk, lab in (("foldgrpo", C_FOLD, "o", "FoldGRPO"), ("grpo", C_GRPO, "s", "GRPO")):
        rows = val[arm]
        ax.plot([min(r["step"], LAST_REAL[arm]) for r in rows], [r[key] / (1000 if key == "chars_per_traj" else 1) for r in rows],
                color=col, marker=mk, markersize=6, lw=2.0, label=lab)
    ax.set_title(ttl + (" (k)" if key == "chars_per_traj" else ""), fontsize=11)
    ax.set_xlabel("effective steps"); ax.set_xlim(-2, 64)
    if key == "obs_intact_pct": ax.set_ylim(90, 100.5)
axes[0].axhline(old["No-RL base"]["empty_think_turn_pct"], color=C_BASE, ls=":", lw=1.4)
axes[0].text(2, old["No-RL base"]["empty_think_turn_pct"] + 1, "unpatched base", color=C_BASE, fontsize=8.5)
axes[0].legend(fontsize=9, loc="lower right")
save_fig(fig, "fig3_behavior_vs_step",
         "Greedy-dump behaviour per checkpoint: empty-think share, branching, observation integrity, main-thread length.")

# ---- fig4: bug contrast bars
fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.6))
cats = ["No-RL base", "GRPO", "FoldGRPO"]
for ax, key, ttl, fmt in zip(axes, ("obs_intact_pct", "empty_think_turn_pct", "greedy_pass@1"),
                             ("Intact observations (%)", "Empty <think> turns (%)", "Greedy pass@1"), ("{:.0f}", "{:.0f}", "{:.3f}")):
    idx = {"obs_intact_pct": 5, "empty_think_turn_pct": 4, "greedy_pass@1": 3}[key]
    o = [r[idx] for r in contrast_rows if r[0].startswith("unpatched")]
    nn = [r[idx] for r in contrast_rows if r[0].startswith("fixed")]
    x = np.arange(3); w = 0.38
    b1 = ax.bar(x - w / 2, o, w, color="#cfcece", edgecolor="black", linewidth=1.1, label="unpatched (Aug, 100 steps)")
    b2 = ax.bar(x + w / 2, nn, w, color=[C_BASE, C_GRPO, C_FOLD], edgecolor="black", linewidth=1.1, label="fixed (Sep; 59 / 46 steps)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.004 if key == "greedy_pass@1" else 1), fmt.format(b.get_height()), ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=10); ax.set_title(ttl, fontsize=11.5)
    ax.set_ylim(0, (0.42 if key == "greedy_pass@1" else 112))
axes[2].legend(fontsize=8.5, loc="upper left")
save_fig(fig, "fig4_bug_contrast",
         "Unpatched (August) vs fixed code: observation integrity, empty-think share and greedy accuracy per arm.")

# ============================================================== config + timeline tables
CONFIG = [
    ("base model", "Seed-OSS-36B", "Qwen3-8B (thinking mode on)"),
    ("judge (reward, scope penalty, eval grading)", "GPT-5-nano / gpt-4o-mini / gpt-4.1", "gpt-oss-120b via vLLM behind `infra/judge_shim.py` (all roles)"),
    ("rollout batch / group size / PPO mini-batch", "32 / 8 / 128", "32 / 8 / 128 (authors' script verbatim)"),
    ("learning rate; response length; max branches", "1e-6; 32768; 10", "1e-6; 32768; 10"),
    ("training steps", "50", "100 planned; **effective 46 (FoldGRPO) / 59 (GRPO)** (§6)"),
    ("observation rendering (this campaign's change)", "template renders history verbatim (Seed-OSS)", "post-prompt turns rendered standalone + `<tool_response>` wrapping (commit b94b34f)"),
    ("compute", "-", "one 8×H100 Merlin batch job per arm + one 8×A100 infra job (search index + judge)"),
    ("validation", "-", "150-item BC-Plus test split, greedy, every 10 steps; final evals via self-contained val-only jobs (greedy + T=1.0 n=4)"),
]
save_asset_csv("table1_config.csv", ["setting", "paper", "this campaign"], [list(r) for r in CONFIG],
               "Configuration of the retrain campaign vs the paper's BC-Plus track; values from scripts/train_bc_qwen3_8b.sh, "
               "infra/jobs/fold-train-*.json and the trainer logs (actor/lr:1e-06). Backs Table 1.")

TIMELINE = [
    ("09-08", "Bug found while auditing val dumps: 0 % of observations carried their `<|im_start|>user` header, 13–16 % dropped entirely, ~80 % of `<think>` blocks empty in every arm (§2)."),
    ("09-09", "Fix committed (b94b34f) with 11 tokenizer-level tests; no-RL base re-measured on fixed code by a self-contained val-only batch job: greedy 0.207 (was 0.167), T=1.0 mean@4 0.170 (was 0.168)."),
    ("09-10 12:20 → 16:36", "Retrain campaign launch as Merlin batch jobs. Three infra-job attempts failed (search health-check on loopback vs a `::` socket; wrong embeddings filename; silent server); the first arm pair was stopped by the platform after 2.7 h idling for infra. Arms relaunched 16:36 once INFRA_READY was published (foldgrpo `ed81a06ae2477596`, grpo `078762e6527e2b8d`)."),
    ("09-10 20:10 → 09-12 00:10", "The infra pod (low GPU utilisation) is reclaimed by the queue on a fixed 4-hour UTC grid — ten times (#4 … #13), each costing ~20 min until the resubmitted pod was READY; training pods (busy) survived. Steps rolled out inside a gap are zero-reward no-ops (fold 1 step, grpo 5 steps before the final loss); both step-20 validations landed inside a gap (read 0.0) and were re-evaluated from the uploaded checkpoints by val-only jobs."),
    ("09-11 20:55", "Hands-off decision: no automatic infra resubmission (no babysitter loops)."),
    ("09-12 00:10", "Infra #13 reclaimed and not resubmitted. Both arms keep training without a reward server: FoldGRPO steps 47–100 have all-zero scores ⇒ zero advantage ⇒ **grad-norm exactly 0** (weights frozen at the post-step-46 policy); GRPO steps 60–100 receive sporadic tiny rewards (17 steps, 0.001–0.03) from the exact-match shortcut in `envs/local_search.py` (answers from memory, no search, no judge) ⇒ **grad-norm 0.03–0.07, weights drift** (checkpoints 70–100 discarded)."),
    ("09-12 19:05 / 09-14 03:04", "GRPO / FoldGRPO jobs finish at step 100."),
    ("09-14 03:56 → 05:44", "Final evals: five self-contained 8×H100 val-only jobs on the last clean checkpoints (Fold 40, 50; GRPO 40, 50, 60), greedy + T=1.0 n=4. This report built 09-14."),
]
save_asset_csv("incident_timeline.csv", ["date_sf_2026", "event"], [list(r) for r in TIMELINE],
               "Incident/timeline log of the campaign (SF time), from infra/jobs/JOBS.tsv, HDFS marker timestamps and trainer logs. Backs §6.")

# ============================================================== example trajectories
def norm(s): return re.sub(r"\s+", " ", (s or "")).strip().lower()


def user_question(d):
    inp = d["input"]; i, j = inp.find("\nuser\n"), inp.rfind("\nassistant")
    u = inp[i + 6:j] if i >= 0 and j > i else ""
    m = re.search(r"Question: (.*?)\n\nYour response should contain:", u, re.S)
    return (m.group(1) if m else u).strip()


def load_judge(d):
    calls = []
    for l in open(f"{d}/judge_calls.jsonl", errors="ignore"):
        try: c = json.loads(l)
        except Exception: continue
        if c.get("kind") != "openai": continue
        content = c["messages"][0]["content"]
        mq = re.search(r"\[question\]: (.*?)\n\n\[response\]", content, re.S)
        mr = re.search(r"\[response\]: (.*?)\n\nYour judgement", content, re.S)
        mg = re.search(r"\n\[correct_answer\]: (.*?)\n\nreasoning", content, re.S)
        mv = re.search(r"^correct: *(\S+)", c["response"], re.M)
        me = re.search(r"extracted_final_answer: *(.*?)\n", c["response"] + "\n")
        mreas = re.search(r"reasoning: (.*?)\n\ncorrect:", c["response"], re.S) or re.search(r"reasoning: (.*?)\ncorrect:", c["response"], re.S)
        if mq and mg:
            calls.append({"q": mq.group(1).strip(), "gold": mg.group(1).strip(), "response": (mr.group(1).strip() if mr else ""),
                          "verdict": (mv.group(1).strip() if mv else "?"), "extracted": (me.group(1).strip() if me else ""),
                          "reasoning": (mreas.group(1).strip() if mreas else "")})
    return calls


def match_judge(q, calls):
    key = norm(q)[:120]
    for c in calls:
        if key and (key in norm(c["q"]) or norm(c["q"])[:120] in norm(q)): return c
    return None


def skeleton(output, max_calls=24, arg_len=110):
    calls = re.findall(r"<function=(\w+)>(.*?)</function>", output, re.S)
    lines = []
    for i, (fn, a) in enumerate(calls[:max_calls], 1):
        a = re.sub(r"\s+", " ", a).strip()
        lines.append(f"{i:2d}. {fn}({a[:arg_len]}{'...' if len(a) > arg_len else ''})")
    if len(calls) > max_calls: lines.append(f"... ({len(calls) - max_calls} more calls)")
    return "\n".join(lines)


def think_excerpt(output, n=2, width=420):
    outs = []
    for m in re.finditer(r"<think>(.*?)</think>", output, re.S):
        t = m.group(1).strip()
        outs.append(("(empty)" if not t else t[:width] + ("..." if len(t) > width else "")))
        if len(outs) >= n: break
    return outs


def load_dump(path): return [json.loads(l) for l in open(path, errors="ignore")]


BASE_DIR = f"{H}/results/valonly_norl_base_greedy_fix_sc"
FOLD_DIR = f"{H}/results/valonly_foldgrpo_50_greedy_fix_sc"
EX = {}
base_items, base_j = load_dump(f"{BASE_DIR}/dump/0.jsonl"), load_judge(BASE_DIR)
for i, d in enumerate(base_items):
    if d["score"] == 0.0:
        j = match_judge(user_question(d), base_j)
        if j and j["verdict"] == "no": EX["base_fail"] = dict(idx=i, dump=f"{BASE_DIR}/dump/0.jsonl", d=d, j=j); break
fold_items, fold_j = load_dump(f"{FOLD_DIR}/dump/0.jsonl"), load_judge(FOLD_DIR)
for i, d in enumerate(fold_items):
    o = d["output"]
    if d["score"] == 1.0 and o.count("<function=branch>") >= 2 and "<tool_response>" in o:
        j = match_judge(user_question(d), fold_j)
        if j and j["verdict"] == "yes": EX["fold_success"] = dict(idx=i, dump=f"{FOLD_DIR}/dump/0.jsonl", d=d, j=j); break
for i, d in enumerate(fold_items):
    if d["score"] == 0.0 and norm(user_question(d)) != norm(EX["base_fail"]["d"] and user_question(EX["base_fail"]["d"])):
        j = match_judge(user_question(d), fold_j)
        if j and j["verdict"] == "no": EX["fold_fail"] = dict(idx=i, dump=f"{FOLD_DIR}/dump/0.jsonl", d=d, j=j); break
assert set(EX) == {"base_fail", "fold_success", "fold_fail"}, set(EX)

traj_json = {}
for name, e in EX.items():
    d, j = e["d"], e["j"]
    traj_json[name] = {"source_dump": e["dump"], "traj_index": e["idx"], "score": d["score"], "question": user_question(d),
                       "gold_answer": j["gold"], "model_response": j["response"], "judge_verdict": j["verdict"],
                       "judge_extracted_answer": j["extracted"], "judge_reasoning": j["reasoning"],
                       "n_main_thread_tool_calls": d["output"].count("<function="), "n_branch_calls": d["output"].count("<function=branch>"),
                       "n_tool_responses": d["output"].count("<tool_response>"), "n_empty_think": len(re.findall(r"<think>\s*</think>", d["output"])),
                       "n_think": d["output"].count("<think>"), "output_chars": len(d["output"]), "full_output": d["output"]}
json.dump(traj_json, open(f"{ASSETS}/trajectory_examples.json", "w"), indent=1)
readme_entries.append(("trajectory_examples.json",
    "The three example trajectories of §7 with their complete main-thread output ('full_output'). Selected deterministically "
    "(first dump entry meeting each criterion): base_fail = first fixed no-RL greedy trajectory with score 0 and judge verdict "
    "'no'; fold_success = first FoldGRPO ckpt-50 greedy trajectory with score 1, >=2 branch calls, a <tool_response> block and "
    "judge 'yes'; fold_fail = first FoldGRPO ckpt-50 greedy trajectory with score 0, a different question from base_fail, and judge 'no'. Question parsed from the "
    "dump 'input'; gold/response/judge fields from the question-matched entry of that job's judge_calls.jsonl."))


def fence(s, limit=None):
    s = (s or "").replace("```", "'''")
    if limit and len(s) > limit: s = s[:limit] + f"\n... [truncated; {len(s)} chars total - full text in trajectory_examples.json]"
    return f"```\n{s}\n```"


def render_example(title, name, note=""):
    e, t = EX[name], traj_json[name]
    th = think_excerpt(e["d"]["output"])
    parts = [f"### {title}",
             f"*Source: `{os.path.relpath(t['source_dump'], H)}` entry {t['traj_index']}; score {t['score']:.0f}; "
             f"{t['n_main_thread_tool_calls']} main-thread tool calls ({t['n_branch_calls']} branch), {t['n_tool_responses']} "
             f"observations, {t['n_empty_think']}/{t['n_think']} empty think blocks; main-thread length {t['output_chars']:,} chars.*" + (f" {note}" if note else ""),
             f"**Question** (BC-Plus):\n\n{fence(t['question'], 900)}",
             f"**Main-thread tool-call sequence** (arguments truncated):\n\n{fence(skeleton(e['d']['output']))}",
             "**First two `<think>` blocks** (verbatim, truncated):\n\n" + fence("\n---\n".join(th)),
             f"**Model final answer** (as graded):\n\n{fence(t['model_response'], 600)}",
             f"**Gold answer:** {t['gold_answer']}",
             f"**Judge verdict: `{t['judge_verdict']}`** — extracted answer: “{t['judge_extracted_answer']}”. Judge reasoning: "
             f"{t['judge_reasoning'][:400]}{'...' if len(t['judge_reasoning']) > 400 else ''}"]
    return "\n\n".join(parts)


# ============================================================== numbers for prose
def V(arm, step, key="acc"):
    return next(r[key] for r in val[arm] if r["step"] == step)


def T(arm, step, key="mean4"):
    return next(r[key] for r in T4[arm] if r["step"] == step)


S = dict(
    fold_g40=V("foldgrpo", 40), fold_g50=V("foldgrpo", 50), grpo_g40=V("grpo", 40), grpo_g50=V("grpo", 50), grpo_g60=V("grpo", 60),
    base_g=V("foldgrpo", 0), base_t=T("foldgrpo", 0), fold_t40=T("foldgrpo", 40), fold_t50=T("foldgrpo", 50),
    grpo_t40=T("grpo", 40), grpo_t50=T("grpo", 50), grpo_t60=T("grpo", 60),
    fold_b50=T("foldgrpo", 50, "best4"), grpo_b60=T("grpo", 60, "best4"),
    fold_et50=V("foldgrpo", 50, "empty_think_turn_pct"), grpo_et60=V("grpo", 60, "empty_think_turn_pct"), base_et=V("foldgrpo", 0, "empty_think_turn_pct"),
    fold_br50=V("foldgrpo", 50, "branch_per_traj"), grpo_br60=V("grpo", 60, "branch_per_traj"), base_br=V("foldgrpo", 0, "branch_per_traj"),
    fold_ov50=T("foldgrpo", 50, "overlong"), grpo_ov60=T("grpo", 60, "overlong"), base_ov=T("foldgrpo", 0, "overlong"),
    fold_ch50=V("foldgrpo", 50, "chars_per_traj"), grpo_ch60=V("grpo", 60, "chars_per_traj"),
    old_base_et=old["No-RL base"]["empty_think_turn_pct"], old_fold_et=old["FoldGRPO@100"]["empty_think_turn_pct"], old_grpo_et=old["GRPO@100"]["empty_think_turn_pct"],
    old_base_miss=old["No-RL base"]["obs_missing"], old_base_turns=old["No-RL base"]["main_turns_per_traj"], new_base_turns=val["foldgrpo"][0]["main_turns_per_traj"],
    n_grpo_tainted=len(grpo_tainted), fold_gaps_pre=fold_gaps_pre, grpo_gaps_pre=grpo_gaps_pre,
    fold_step_s=statistics.mean(int(r["step_s"]) for r in train["foldgrpo"][30:46]) / 60, grpo_step_s=statistics.mean(int(r["step_s"]) for r in train["grpo"][40:59]) / 60,
)
ci150, ci600 = ci(0.33, 150), ci(0.33, 600)

# ============================================================== markdown
A = f"{STEM}_assets"
def cap(t): return "> " + t.replace("\n", "\n> ")
config_md = "| Setting | Paper (Sun et al.) | This campaign |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a, b, c in CONFIG)
timeline_md = "| Date (SF, 2026) | Event |\n|---|---|\n" + "\n".join(f"| {d} | {e} |" for d, e in TIMELINE)
head_rows = [("No-RL base", 0, S["base_g"], S["base_t"], T("foldgrpo", 0, "best4")),
             ("GRPO ckpt 40", 40, S["grpo_g40"], S["grpo_t40"], T("grpo", 40, "best4")),
             ("GRPO ckpt 50", 50, S["grpo_g50"], S["grpo_t50"], T("grpo", 50, "best4")),
             ("GRPO ckpt 60 (final clean)", 59, S["grpo_g60"], S["grpo_t60"], S["grpo_b60"]),
             ("FoldGRPO ckpt 40", 40, S["fold_g40"], S["fold_t40"], T("foldgrpo", 40, "best4")),
             ("FoldGRPO ckpt 50 (final clean)", 46, S["fold_g50"], S["fold_t50"], S["fold_b50"])]
headline_md = "| Model | Effective steps | Greedy pass@1 (±CI) | T=1.0 mean@4 (±CI) | T=1.0 best@4 |\n|---|---|---|---|---|\n" + "\n".join(
    f"| {m} | {s} | {g:.3f} (±{ci(g,150):.3f}) | {t:.3f} (±{ci(t,600):.3f}) | {b:.3f} |" for m, s, g, t, b in head_rows)
save_asset_csv("table2_headline_scores.csv", ["model", "effective_steps", "greedy_pass@1", "t1.0_mean@4", "t1.0_best@4"],
               [[m, s, round(g, 4), round(t, 4), round(b, 4)] for m, s, g, t, b in head_rows], "Headline table (§4), same sources as fig1_scores_vs_step.csv. Backs Table 2.")
beh_md = ("| Model | Empty `<think>` turns | Branch calls/traj (main) | Overlong rate @T=1 | Main-thread chars | Finish rate |\n|---|---|---|---|---|---|\n"
          f"| No-RL base | {S['base_et']:.0f} % | {S['base_br']:.2f} | {100*S['base_ov']:.0f} % | {val['foldgrpo'][0]['chars_per_traj']/1000:.0f}k | {val['foldgrpo'][0]['finish_pct']:.0f} % |\n"
          f"| GRPO ckpt 60 (59 steps) | {S['grpo_et60']:.0f} % | {S['grpo_br60']:.2f} | {100*S['grpo_ov60']:.0f} % | {S['grpo_ch60']/1000:.0f}k | {val['grpo'][-1]['finish_pct']:.0f} % |\n"
          f"| FoldGRPO ckpt 50 (46 steps) | {S['fold_et50']:.0f} % | {S['fold_br50']:.2f} | {100*S['fold_ov50']:.0f} % | {S['fold_ch50']/1000:.0f}k | {val['foldgrpo'][-1]['finish_pct']:.0f} % |")

report_md = f"""# FoldAgent at 8B, take two: a Qwen3 chat-template bug, its fix, and a re-trained FoldGRPO-vs-GRPO comparison

**Project:** Replication of Sun et al., *Scaling Long-Horizon LLM Agent via Context-Folding* (arXiv:2510.11967) with the authors' re-implementation (github.com/sunnweiwei/FoldAgent), Qwen3-8B, BrowseComp-Plus track.
**Dates:** bug found 2026-09-08; fix 09-09; retrain 09-10 → 09-14 (SF time); report built 2026-09-14. **Author:** Xiaoxuan Lei (with Claude Code).
**Supersedes** the absolute numbers of `docs/reports/2026-08-12_foldagent_bcplus_replication_report.pdf` (all of which were produced by the unpatched code, §2).
**Assets:** every figure ships as PNG+PDF paired with its backing CSV/JSON in `docs/reports/{A}/` (schema + provenance in its `README.md`); rebuild end-to-end with `docs/reports/build_fix_report.py` (data extraction shared with `scripts/analyze_fix_campaign.py`).

## 1. Summary

1. **Every observation the August agents saw was corrupted.** Qwen3's chat template strips earlier assistant `<think>` blocks whenever the last message is a plain user turn; the agent code tokenised each observation as the *difference* between two template renders, so the slice was offset by the length of the preceding think block: every observation lost its `<|im_start|>user` header and its first tokens, 13–16 % were dropped entirely, and branch prompts were sliced the same way. The RL'd policies learned the workaround "do not think, or you lose the observation" (~80 % empty `<think>` blocks). Seed-OSS-36B (the paper's model) renders history verbatim, so the paper's setup is unaffected.
2. **The fix** (commit b94b34f; post-prompt turns rendered standalone behind a system + dummy-user anchor, observations wrapped in `<tool_response>`) makes 95–99 % of observations intact (the rest are cut by the 32 k response limit). On the no-RL base it changes behaviour (empty-think {S['old_base_et']:.0f} % → {S['base_et']:.0f} %, main-thread turns {S['old_base_turns']:.1f} → {S['new_base_turns']:.1f}) but not accuracy within CI (greedy 0.167 → {S['base_g']:.3f}, T=1.0 mean@4 0.168 → {S['base_t']:.3f}).
3. **Re-trained on fixed code, FoldGRPO and GRPO tie again.** At the step-matched checkpoint 40: greedy {S['fold_g40']:.3f} vs {S['grpo_g40']:.3f} (±{ci150:.3f}), T=1.0 mean@4 {S['fold_t40']:.3f} vs {S['grpo_t40']:.3f} (±{ci600:.3f}). At the last clean checkpoints (FoldGRPO 46 effective steps, GRPO 59): T=1.0 mean@4 {S['fold_t50']:.3f} vs {S['grpo_t60']:.3f}. Both arms clear the base by +16–17 points at T=1.0 — the RL gain replicates, the paper's FoldGRPO-over-GRPO margin does not (as in August).
4. **The campaign was truncated by infrastructure, not by the method.** The reward-serving pod was reclaimed by the shared queue on a 4-hour grid; after the last reclamation was (deliberately) not resubmitted, FoldGRPO's remaining 54 steps were exact no-ops (grad-norm 0) and GRPO's remaining 41 steps drifted on exact-match-only rewards and are discarded. Effective budgets: 46 vs 59 of 100 planned steps (§6).
5. **New behavioural finding:** with intact observations, the empty-think rate still *rises* under RL in both arms (base {S['base_et']:.0f} % → FoldGRPO {S['fold_et50']:.0f} %, GRPO {S['grpo_et60']:.0f} % of turns), so at 8B the outcome reward itself favours skipping visible reasoning; it is not an artefact of the bug. FoldGRPO branches more ({S['fold_br50']:.1f} vs {S['grpo_br60']:.1f} main-thread branch calls per trajectory) and — unlike August — no longer has the lower overlong rate ({100*S['fold_ov50']:.0f} % vs {100*S['grpo_ov60']:.0f} % at T=1.0).

## 2. The bug: template-diff tokenisation of observations under Qwen3

**Mechanism.** `AgentContext.get_turn_context` produced the token ids of a new user/observation turn as `render(chat[:i+1]) − render(chat[:i])`, i.e. the suffix of the longer template render. Qwen3's Jinja template keeps `<think>…</think>` only for assistant messages *after* the last plain user query; when an observation (a user-role message) is appended, every earlier assistant think block is removed from the render, so the longer render is *shorter* by the think length and the suffix slice is misaligned by exactly that amount. Consequences, measured on all August validation dumps with the same parser used below (`{A}/fig4_bug_contrast.csv`): 0 of {S['old_base_miss']:,} base-model observations carried a `<|im_start|>user` header; when the think block was longer than the observation, the observation vanished entirely (13–16 % of tool calls in the RL'd arms); branch prompts (also user-role) were sliced the same way. The policy still saw *something* (the tail of the observation), which is why August accuracies were plausible rather than zero.

**Why the paper is unaffected.** Seed-OSS-36B-Instruct's template renders a separate `reasoning_content` field and never parses or strips think tags from `content`, and has no "last query" logic, so the identical code yields an intact `<seed:bos>user\\n…` turn. The bug is Qwen3-template-specific — i.e. specific to our substitution of the base model.

**Fix** (`agents/utils.py::render_single_turn`, commit b94b34f): every post-prompt turn is rendered standalone behind a fixed system + dummy-user anchor whose prefix is stripped, so the slice no longer depends on history; tool observations are wrapped in `<tool_response>…</tool_response>`, which also keeps server-side re-templating (the deployment-serving caveat of the August report, §6 there) consistent with training. Eleven tests against the real Qwen3 tokenizer (`tests/test_qwen3_tool_response_wrapping.py`) check header presence, exact token equality with a direct encode, and think preservation across multi-turn histories.

![]({A}/fig4_bug_contrast.png)

{cap(f"**Figure 4 — unpatched vs fixed code, per arm (greedy, n=150 trajectories per bar).** Grey bars: the August self-contained greedy runs (unpatched code, 100 training steps); coloured bars: the post-fix runs (no-RL base; GRPO checkpoint 60 = 59 effective steps; FoldGRPO checkpoint 50 = 46 effective steps). Left: share of non-finish tool calls that are followed by a complete `<tool_response>` block — 0 by construction under the unpatched code (no observation had a header or wrapper) vs 96–99 % after the fix (the remainder are cut by the 32 k-token response limit). Middle: share of assistant turns whose `<think>` block is empty. Right: greedy pass@1 with the same judge. What it shows: the fix restores context integrity and halves the base model's empty-think rate ({S['old_base_et']:.0f} → {S['base_et']:.0f} %) without changing base accuracy beyond noise; the RL'd arms' empty-think rate is *not* restored (see §5) and their accuracies are not budget-matched across the two code versions (100 vs 46/59 steps), so the right panel is descriptive only. Data: {A}/fig4_bug_contrast.csv.")}

## 3. Setup

{config_md}

**Infrastructure (changed since August).** Everything ran as Merlin batch jobs on the `ark-eng-algorithm` queue: one 8×H100 training job per arm (verl + vLLM async rollouts, authors' script verbatim, GRPO = the same script ± `adv_estimator=grpo`, `process_reward=none`) and one 8×A100 *infra* job serving the BC-Plus search index (`:8000`) and the gpt-oss-120b judge (`:8001`); the arms reach the judge through `infra/judge_shim.py`, which re-resolves the infra pod's address from an HDFS marker on every request so that a replaced infra pod is picked up without restarting training. Checkpoints (every 10 steps), validation dumps and 5-minute log mirrors are written to HDFS; final evaluations run as *self-contained* val-only jobs (search + judge + rollout on one 8×H100 pod, `infra/jobs/fold_val_*.json`). Job specs, entrypoints and the submission ledger are under `infra/jobs/` (`JOBS.tsv`).

## 4. Results

{headline_md}

**Table 2** — pass@1 on the 150-item BC-Plus test split (judge gpt-oss-120b, training-stack validation path). "Effective steps" = reward-bearing updates contained in the checkpoint (§6). CIs are binomial 95 % (the T=1.0 CIs treat the 4 rollouts per item as independent and are optimistic); the paired CI for an arm difference at T=1.0 is about ±3.5 points. Data: `{A}/table2_headline_scores.csv`.

![]({A}/fig1_scores_vs_step.png)

{cap(f"**Figure 1 — validation score vs effective training steps.** Left: greedy pass@1 (n=150 per point); right: T=1.0 mean@4 (n=600 per point); red circles FoldGRPO, blue squares GRPO, dotted grey line the fixed no-RL base; error bars 95 % binomial CIs. x is the number of reward-bearing updates in the checkpoint (FoldGRPO checkpoint 50 sits at 46, GRPO checkpoint 60 at 59; all earlier checkpoints are fully real). Greedy points come from the in-process validation dumps except step 0 and the hollow-ringed points, which are val-only re-evaluations of the uploaded checkpoint (both step-20 validations and the final ones ran inside an infra gap and read 0.0 in-process). T=1.0 points are val-only jobs. What it shows: both arms leave the base within 10 steps and keep improving to ~0.31–0.35; FoldGRPO leads at greedy from step 30 (0.333/0.353 vs 0.247/0.307) but the lead is within the ±0.075 greedy CI and vanishes at T=1.0 (0.320 vs 0.308 at step 40; 0.335 at 46 vs 0.342 at 59). Data: {A}/fig1_scores_vs_step.csv.")}

**Findings.**

1. **RL gain replicates, larger than in August.** T=1.0 mean@4 rises from {S['base_t']:.3f} (base) to {S['fold_t50']:.3f} (FoldGRPO, 46 steps) and {S['grpo_t60']:.3f} (GRPO, 59 steps): +16–17 points, ≫ CI. August's +10–11 points came from 100 steps of the unpatched code.
2. **FoldGRPO vs GRPO: statistical tie, as in August.** Step-matched at 40 (both arms fully real): greedy {S['fold_g40']:.3f} vs {S['grpo_g40']:.3f}, T=1.0 {S['fold_t40']:.3f} vs {S['grpo_t40']:.3f}, best@4 {T('foldgrpo',40,'best4'):.3f} vs {T('grpo',40,'best4'):.3f}. FoldGRPO's nominal greedy lead (+4.6 at step 40, +0.7 at 46 vs 50) is inside the greedy CI and is not present at the powered T=1.0 tier. With 13 fewer effective steps FoldGRPO matches GRPO's final T=1.0 score ({S['fold_t50']:.3f} vs {S['grpo_t60']:.3f}) — a mild sample-efficiency hint, not a significant one.
3. **Greedy over-states FoldGRPO at step 40–50.** FoldGRPO's greedy score *drops* from 0.353 (40) to 0.307 (46) while its T=1.0 mean@4 rises (0.320 → 0.335); greedy at n=150 is simply too noisy to rank checkpoints, which is why the T=1.0 tier was pre-registered as the decisive one.

![]({A}/fig2_training_curves.png)

{cap(f"**Figure 2 — training dynamics.** Batch-mean trajectory reward (`critic/score/mean`, 256 rollouts per step), train overlong (context blow-out) rate, and mean response length (tokens) per step, parsed from the mirrored trainer logs. Solid = steps with a live reward server; × = steps hit by a 4-hour-grid infra reclamation before the final loss (FoldGRPO {S['fold_gaps_pre']}, GRPO {S['grpo_gaps_pre']}: scores zeroed or rollouts cut short, zero or near-zero gradient); shaded + dotted = after the final infra loss on 09-12 00:10 (FoldGRPO from step 47: grad-norm exactly 0 on every step; GRPO from step 60: {S['n_grpo_tainted']} steps with tiny exact-match-only rewards and grad-norm 0.03–0.07, hence discarded). What it shows: up to the truncation FoldGRPO's reward climbs 0.08 → ~0.2 and GRPO's 0.07 → ~0.2 (noisy, one seed); GRPO's train overlong rate stays at 0.2–0.5 while FoldGRPO's is 0.0 throughout (the fold penalty masks overlong trajectories in its reward accounting, so this train metric is not comparable across arms — use the eval overlong rate in §5); response length grows 10 k → 20 k tokens in both arms, which is why the step time grew from 13 to {S['fold_step_s']:.0f} (FoldGRPO) / {S['grpo_step_s']:.0f} (GRPO) minutes. Data: {A}/fig2_training_curves.csv.")}

## 5. Behaviour with intact contexts

{beh_md}

Greedy dumps (n=150) except the overlong column (T=1.0 val metrics, n=600). Branch calls are counted on the main thread only.

![]({A}/fig3_behavior_vs_step.png)

{cap(f"**Figure 3 — behaviour per checkpoint (greedy dumps, n=150 each; x = effective steps).** Panels: share of assistant turns with an empty `<think>` block (dotted grey = the unpatched August base at {S['old_base_et']:.0f} %); main-thread `branch` calls per trajectory; share of tool calls followed by a complete `<tool_response>` block; main-thread length in thousands of characters. Computed by `scripts/analyze_fix_campaign.py` from the same dumps as Figure 1. What it shows: (i) observations stay 95–99 % intact throughout training — the fix holds under RL; (ii) the empty-think rate nevertheless climbs from {S['base_et']:.0f} % to ~80 % (FoldGRPO) and ~58 % (GRPO) within 20–30 steps, i.e. the reward favours skipping visible reasoning at 8B even when thinking no longer costs the observation; (iii) FoldGRPO's branching grows steadily to {S['fold_br50']:.1f} calls per trajectory while GRPO plateaus at ~2.3, and FoldGRPO's main thread shrinks to {S['fold_ch50']/1000:.0f} k chars vs GRPO's {S['grpo_ch60']/1000:.0f} k — the folding mechanism is learned, as in August. Data: {A}/fig3_behavior_vs_step.csv.")}

- **Empty thinking is a property of the reward at 8B, not of the bug.** Both arms converge to mostly-empty `<think>` blocks with fully intact observations; FoldGRPO does so faster and further ({S['fold_et50']:.0f} % vs {S['grpo_et60']:.0f} %). The August report's "policies adapted to a corrupted format" explanation was therefore incomplete: the format bug explained the *base* model's high rate ({S['old_base_et']:.0f} % → {S['base_et']:.0f} % after the fix) but not the RL'd rate.
- **Folding replicates; the blow-out advantage does not.** FoldGRPO uses {S['fold_br50']:.1f} main-thread branches per trajectory vs GRPO's {S['grpo_br60']:.1f} and keeps a ~2× shorter main thread. But at T=1.0 its overlong rate ({100*S['fold_ov50']:.0f} %) is now *higher* than GRPO's ({100*S['grpo_ov60']:.0f} %) — the reverse of August (0.04 vs 0.25). With intact observations FoldGRPO's trajectories carry more real content per turn, and its sub-trajectories are truncated more often; the August "25× lower blow-out" claim was partly a product of the bug (dropped observations kept contexts short).
- **Tool-use profile.** RL shifts both arms from search-heavy to branch-heavy main threads (search calls per trajectory 1.2 → 0.3–1.3); finish rates stay 77–97 %.

## 6. Incident timeline, effective budgets, and what the truncation means

{timeline_md}

**How the effective budgets were established.** Each step's `actor/grad_norm` is logged. FoldGRPO steps 47–100 all have `critic/score/max = 0` (no trajectory scored) ⇒ every advantage is 0 ⇒ `grad_norm = 0.0` exactly (the recipe has no KL or entropy term), so checkpoints 50–100 are byte-for-byte the post-step-46 policy and checkpoint 50 is used as "final". GRPO's step 60 was likewise a zero-gradient no-op (checkpoint 60 = the post-step-59 policy), but steps 61–97 include {S['n_grpo_tainted']} steps with a handful of trajectories scored 1 by the exact-match shortcut in `envs/local_search.py` (`em_score` runs before any judge call; a memorised answer matches the label without search): grad-norms of 0.03–0.07 were applied to no-search trajectories, so checkpoints 70–100 are discarded. Steps that fell inside an earlier 4-hour-grid gap ({S['fold_gaps_pre']} for FoldGRPO, {S['grpo_gaps_pre']} for GRPO) were zero- or near-zero-gradient no-ops as well: the arms therefore contain **~45 (FoldGRPO) and ~54 (GRPO) genuinely reward-bearing updates** of 100 planned.

**Cost.** ~30 % of the campaign's GPU time (both arms × ~2 days after 09-12 00:10) produced no learning. The root cause is a queue policy (reclaiming low-utilisation pods every 4 h) interacting with a design that keeps the reward server on a separate, mostly idle pod; the remedies are either an automatic resubmit loop for the infra job or co-locating search + judge inside each training pod. Neither was adopted during this campaign (deliberate hands-off decision); a resumption from checkpoints 50/60 to 100 real steps remains possible (~58 h + ~31 h of 8×H100) and is the open decision.

## 7. Example trajectories (fixed code)

Deterministic selection (first dump entry matching each rule; rules and complete texts in `{A}/trajectory_examples.json`). Observations now appear as `<tool_response>` blocks; note the empty `<think>` blocks in the RL'd trajectory.

{render_example("7.1 No-RL base — failure (graded wrong)", "base_fail", "Base trajectories reason at length (think blocks non-empty) but often stop short of the answer.")}

{render_example("7.2 FoldGRPO ckpt 50 — success with branching", "fold_success")}

{render_example("7.3 FoldGRPO ckpt 50 — failure", "fold_fail")}

## 8. Deviations and threats to validity

- **Budget truncation and asymmetry** (§6): 46 vs 59 effective steps, both short of the planned 100 and of the August 100-step runs; the arms are compared step-matched (40/40) and at their last clean checkpoints. One seed per arm.
- **Judge substitution** (gpt-oss-120b for the paper's OpenAI judges) — absolute numbers are not comparable to the paper's Table 1; all cross-arm comparisons use one fixed judge.
- **Greedy tier is under-powered** (±7.5 points at n=150); conclusions rest on the T=1.0 n=4 tier (±3.7 nominal, ±3.5 paired).
- **Behavioural counts are main-thread only** (branch-internal turns are folded away in the dumps); logged `num_turns` (all turns) is reported alongside.
- The unpatched-vs-fixed accuracy contrast (Fig. 4 right) is not budget-matched and is descriptive only.
- The August report's serving-path finding (folded policies collapse under per-turn re-templating) was itself a symptom of this bug; with `<tool_response>` wrapping, server-side templating and training tokenisation agree, but we have not re-run the standalone-serving comparison on the fixed checkpoints.

## 9. Conclusion

With the Qwen3 observation-tokenisation bug fixed, the replication verdict is unchanged in substance and cleaner in evidence: RL on BC-Plus lifts an 8B agent by +16 points at T=1.0 within 46–59 steps, the folding mechanism (branching, main-thread compression) is learned, and FoldGRPO and GRPO remain a statistical tie ({S['fold_t50']:.3f} vs {S['grpo_t60']:.3f} mean@4). Two of the August behavioural claims are revised: the RL'd policies' near-empty thinking is a reward effect at 8B rather than a format artefact, and FoldGRPO's dramatically lower context blow-out rate does not survive intact observations. The campaign was cut to roughly half its budget by shared-queue reclamation of the reward-serving pod; the checkpoints and job specs allow resuming to 100 real steps if the comparison at full budget is wanted.

## Appendix A: Reproduction assets

- Code: `~/xiaoxuan/FoldAgent` (fix b94b34f; infra + analysis in later commits). Tests: `tests/test_qwen3_tool_response_wrapping.py`.
- Jobs: `infra/jobs/*.json` (specs), `infra/jobs/JOBS.tsv` (ledger with every sid), `infra/jobs/status.sh`.
- Data on HDFS `/mnt/hdfs/mlsys/xiaoxuan/fold_replication/`: `ckpt_fix/<arm>/global_step_N`, `val_dump_fix/<arm>/N.jsonl`, `train_logs_fix/<arm>/`, `results/valonly_<tag>_fix_sc/` (dump, val_metrics.txt, judge_calls.jsonl).
- Analysis: `python3 scripts/analyze_fix_campaign.py` → `results/fix_campaign/summary.md`; this report: `python3 docs/reports/build_fix_report.py`.
"""
open(MD, "w").write(report_md)
print(f"markdown written: {MD}")

readme = f"""# FoldAgent BC-Plus fix campaign — report assets

Every figure in `{STEM}.pdf` ships here as PNG+PDF paired with its machine-readable backing data.
Rebuild everything with `python3 docs/reports/build_fix_report.py` (stage 1 needs matplotlib/numpy; stage 2
renders the PDF with weasyprint from the fold_infra venv). Data extraction is shared with
`scripts/analyze_fix_campaign.py` (parse_train_log / analyze_dump / read_val_metrics).

Campaign: Qwen3-8B, BC-Plus, FoldGRPO vs GRPO re-trained on fixed observation tokenisation (commit b94b34f),
Merlin batch jobs; effective budgets 46 (FoldGRPO) / 59 (GRPO) steps of 100 planned (see report §6).
Raw sources: HDFS `/mnt/hdfs/mlsys/xiaoxuan/fold_replication/` (train_logs_fix, val_dump_fix, results/valonly_*_fix_sc)
and, for the unpatched contrast, `results/valonly_<tag>/dump/0.jsonl` of the August campaign.

## Files

""" + "\n".join(f"- **{n}** — {d}" for n, d in readme_entries) + "\n"
open(f"{ASSETS}/README.md", "w").write(readme)
print(f"assets written: {ASSETS}")
subprocess.run([FOLD_INFRA_PY, os.path.abspath(__file__), "--render-pdf"], check=True)
