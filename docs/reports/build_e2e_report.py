#!/usr/bin/env python3
"""Build the FoldAgent end-to-end token-usage report (context efficiency vs trajectory efficiency vs
end-to-end inference efficiency). 2x2(+base) = {GRPO ckpt 60, FoldGRPO ckpt 50, no-RL base} x
{no folding (workflow=search), folding available (workflow=search_branch)}, greedy + T=1.0 n=4,
150 BC-Plus test tasks, per-rollout exact token ledgers (agents/e2e_ledger.py).

Usage:
  python3 docs/reports/build_e2e_report.py                 # stage 1 (system python3: numpy+matplotlib) -> assets, md, then PDF
  <fold_infra python> docs/reports/build_e2e_report.py --render-pdf
Inputs (read-only): /mnt/hdfs/mlsys/xiaoxuan/fold_replication/results/valonly_<arm>_<step>_<mode>_e2e_<wf>_sc/{ledger,val_metrics.txt}
Outputs: docs/reports/2026-09-17_foldagent_e2e_token_usage_report.{md,html,pdf} + _assets/ (figures + CSV/JSON + README)
"""
import csv, glob, json, math, os, statistics, subprocess, sys
ROOT = "/home/tiger/xiaoxuan/FoldAgent"
sys.path.insert(0, ROOT)
H = os.environ.get("E2E_RESULTS_ROOT", "/mnt/hdfs/mlsys/xiaoxuan/fold_replication/results")
RD = f"{ROOT}/docs/reports"
STEM = "2026-09-17_foldagent_e2e_token_usage_report"
ASSETS = os.environ.get("E2E_ASSETS_DIR", f"{RD}/{STEM}_assets")
MD, HTML, PDF = (os.environ.get("E2E_MD", f"{RD}/{STEM}.md"), f"{RD}/{STEM}.html", f"{RD}/{STEM}.pdf")
FOLD_INFRA_PY = "/home/tiger/xiaoxuan/envs/fold_infra/bin/python"

CELLS = [  # (key, arm, step, wf, label, color)
    ("base_nofold", "norl", "base", "nobranch", "base / no fold", "#b5b5b5"),
    ("base_fold",   "norl", "base", "branch",   "base / fold",    "#6e6e6e"),
    ("grpo_nofold", "grpo", "60",   "nobranch", "GRPO / no fold", "#8fb3e0"),
    ("grpo_fold",   "grpo", "60",   "branch",   "GRPO / fold",    "#0F4D92"),
    ("fold_nofold", "foldgrpo", "50", "nobranch", "FoldGRPO / no fold", "#e8a09c"),
    ("fold_fold",   "foldgrpo", "50", "branch",   "FoldGRPO / fold",    "#B64342"),
]
MODES = ["greedy", "t1n4"]
KEYM = ["success", "total_e2e_tokens", "gen_tokens", "obs_tokens", "comp_in_tokens", "comp_gen_tokens", "turns",
        "tool_calls", "branch_calls", "forward_passes", "cum_prompt_tokens", "peak_ctx", "mean_ctx", "main_growth",
        "overlong", "post_total", "post_gen", "post_turns", "post_tool_calls", "has_fold", "empty_think_turns"]


def render_pdf():
    import markdown
    from weasyprint import HTML as WHTML
    body = markdown.markdown(open(MD).read(), extensions=["tables", "fenced_code"])
    html = f"""<html><head><meta charset='utf-8'><style>
@page {{ size: A4; margin: 1.9cm 2.0cm; @bottom-center {{ content: counter(page); font-size: 8pt; color: #888; }} }}
body {{ font-family: 'DejaVu Serif', Georgia, serif; font-size: 9.5pt; line-height: 1.42; color: #1a1a1a; }}
h1 {{ font-size: 15pt; }} h2 {{ font-size: 12.5pt; margin-top: 1.2em; border-bottom: 1px solid #ccc; }}
h3 {{ font-size: 10.5pt; margin-top: 1em; }}
table {{ border-collapse: collapse; font-size: 7.6pt; margin: 0.5em 0; width: 100%; }}
th, td {{ border: 1px solid #bbb; padding: 2px 4px; text-align: right; }} th {{ background: #eee; }} td:first-child, th:first-child {{ text-align: left; }}
img {{ max-width: 100%; }} code {{ font-family: 'DejaVu Sans Mono', monospace; font-size: 8pt; }}
pre {{ background: #f4f4f4; padding: 6px; font-size: 7.6pt; white-space: pre-wrap; }}
</style></head><body>{body}</body></html>"""
    open(HTML, "w").write(html)
    WHTML(string=html, base_url=RD).write_pdf(PDF)
    print("PDF ->", PDF)


if "--render-pdf" in sys.argv:
    render_pdf(); sys.exit(0)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.e2e_metrics import load_ledgers, rollout_metrics
sys.path.insert(0, RD)
from figstyle import apply_publication_style
apply_publication_style(font_size=11, axes_linewidth=1.4)
os.makedirs(ASSETS, exist_ok=True)
rng = np.random.default_rng(0)


def boot_ci(x, n=4000, f=np.mean):
    x = np.asarray([v for v in x if v == v], float)
    if len(x) == 0: return (float('nan'), float('nan'), float('nan'))
    idx = rng.integers(0, len(x), (n, len(x)))
    s = np.sort(f(x[idx], axis=1))
    return (float(f(x)), float(s[int(0.025 * n)]), float(s[int(0.975 * n)]))


def sign_test_p(d):
    d = [v for v in d if v == v and v != 0]
    n, k = len(d), sum(1 for v in d if v > 0)
    if n == 0: return float('nan')
    p = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n * 2
    return min(1.0, p)


def fmt(v, d=0):
    if v is None or (isinstance(v, float) and v != v): return "–"
    return f"{v:,.{d}f}"


# ---------------- load ----------------
rows, cellinfo = [], {}
for key, arm, step, wf, label, color in CELLS:
    for mode in MODES:
        tag = f"{arm}_{step}_{mode}_e2e_{wf}_sc"
        d = f"{H}/valonly_{tag}"
        recs = load_ledgers(f"{d}/ledger") if os.path.isdir(f"{d}/ledger") else []
        vm = {}
        if os.path.exists(f"{d}/val_metrics.txt"):
            for line in open(f"{d}/val_metrics.txt"):
                if "'val/avg_score'" in line or "reward/mean@1" in line:
                    try: vm['val_score'] = float(line.split(':')[-1])
                    except ValueError: pass
        done = os.path.exists(f"{d}/VALONLY_{tag}_DONE")
        cellinfo[(key, mode)] = dict(tag=tag, n=len(recs), done=done, val_score=vm.get('val_score'))
        for r in recs:
            m = rollout_metrics(r); m.update(cell=key, mode=mode, arm=arm, wf=wf); rows.append(m)
print({k: (v['n'], v['done']) for k, v in cellinfo.items()})
if not rows:
    sys.exit("no ledgers found yet")
with open(f"{ASSETS}/rollouts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def sel(key, mode, succ=None):
    return [r for r in rows if r['cell'] == key and r['mode'] == mode and (succ is None or r['success'] == succ)]


def task_means(key, mode, metric):
    """instance_id -> mean over samples (1 for greedy, 4 for t1n4)."""
    acc = {}
    for r in sel(key, mode):
        v = r[metric]
        if v == v: acc.setdefault(r['instance_id'], []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------- cell summaries ----------------
summary = []
for key, arm, step, wf, label, color in CELLS:
    for mode in MODES:
        for cond in ("all", "success"):
            rs = sel(key, mode, None if cond == "all" else 1)
            if not rs: continue
            rec = dict(cell=key, label=label, mode=mode, cond=cond, n=len(rs))
            for mtr in KEYM:
                mu, lo, hi = boot_ci([r[mtr] for r in rs])
                rec[mtr] = mu; rec[f"{mtr}_lo"] = lo; rec[f"{mtr}_hi"] = hi
                rec[f"{mtr}_median"] = float(np.nanmedian([r[mtr] for r in rs]))
            sr = {}
            for r in rs: sr[r['stop_reason']] = sr.get(r['stop_reason'], 0) + 1
            rec['stop_reasons'] = json.dumps(sr); rec['val_score_trainer'] = cellinfo[(key, mode)]['val_score']
            summary.append(rec)
with open(f"{ASSETS}/cell_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

# ---------------- paired task-level diffs ----------------
PAIRS = [("fold_fold", "grpo_fold", "FoldGRPO/fold − GRPO/fold (same tasks)"),
         ("fold_fold", "fold_nofold", "FoldGRPO: fold − no fold"),
         ("grpo_fold", "grpo_nofold", "GRPO: fold − no fold"),
         ("fold_nofold", "grpo_nofold", "FoldGRPO/no fold − GRPO/no fold"),
         ("base_fold", "base_nofold", "base: fold − no fold"),
         ("fold_fold", "grpo_nofold", "FoldGRPO/fold − GRPO/no fold")]
PAIRM = ["success", "total_e2e_tokens", "gen_tokens", "obs_tokens", "turns", "tool_calls", "peak_ctx", "mean_ctx", "cum_prompt_tokens", "forward_passes"]
paired = []
for a, b, lab in PAIRS:
    for mode in MODES:
        for mtr in PAIRM:
            ta, tb = task_means(a, mode, mtr), task_means(b, mode, mtr)
            common = sorted(set(ta) & set(tb))
            if len(common) < 5: continue
            d = [ta[t] - tb[t] for t in common]
            mu, lo, hi = boot_ci(d)
            paired.append(dict(pair=lab, a=a, b=b, mode=mode, metric=mtr, n_tasks=len(common), mean_diff=mu, ci_lo=lo, ci_hi=hi,
                               median_diff=float(np.median(d)), frac_positive=float(np.mean([v > 0 for v in d])),
                               sign_p=sign_test_p(d), mean_a=float(np.mean([ta[t] for t in common])), mean_b=float(np.mean([tb[t] for t in common]))))
with open(f"{ASSETS}/paired_diffs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(paired[0].keys())); w.writeheader(); w.writerows(paired)

# ---------------- figures ----------------
present = [c for c in CELLS if sel(c[0], "greedy") or sel(c[0], "t1n4")]


def savefig(fig, name):
    fig.savefig(f"{ASSETS}/{name}.png", dpi=170, bbox_inches="tight"); fig.savefig(f"{ASSETS}/{name}.pdf", bbox_inches="tight"); plt.close(fig)


# F1: success vs total e2e tokens (cell means with CIs) + budget curves
fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), constrained_layout=True)
f1 = []
for j, mode in enumerate(MODES):
    ax = axes[0, j]
    for key, arm, step, wf, label, color in present:
        rs = sel(key, mode)
        if not rs: continue
        sx, sy = boot_ci([r['total_e2e_tokens'] for r in rs]), boot_ci([r['success'] for r in rs])
        ax.errorbar(sx[0] / 1e3, sy[0], xerr=[[(sx[0] - sx[1]) / 1e3], [(sx[2] - sx[0]) / 1e3]], yerr=[[sy[0] - sy[1]], [sy[2] - sy[0]]],
                    fmt='o' if wf == 'branch' else 's', color=color, ms=8, capsize=3, label=label)
        f1.append(dict(mode=mode, cell=key, tokens_mean=sx[0], tokens_lo=sx[1], tokens_hi=sx[2], success=sy[0], success_lo=sy[1], success_hi=sy[2]))
    ax.set_xlabel("total end-to-end tokens per rollout (k)"); ax.set_ylabel("success rate"); ax.set_title(f"{mode}: success vs e2e tokens")
    if j == 0: ax.legend(fontsize=8)
    ax = axes[1, j]
    grid = np.linspace(0, 400e3, 200)
    for key, arm, step, wf, label, color in present:
        rs = sel(key, mode)
        if not rs: continue
        tok = np.array([r['total_e2e_tokens'] for r in rs]); suc = np.array([r['success'] for r in rs])
        ax.plot(grid / 1e3, [np.mean(suc * (tok <= g)) for g in grid], color=color, lw=2, ls='-' if wf == 'branch' else '--', label=label)
    ax.set_xlabel("end-to-end token budget B (k)"); ax.set_ylabel("P(solved and total tokens ≤ B)"); ax.set_title(f"{mode}: solved-within-budget")
savefig(fig, "fig1_success_vs_e2e_tokens")
json.dump(f1, open(f"{ASSETS}/fig1_success_vs_e2e_tokens.json", "w"), indent=1)

# F2: peak active context vs total e2e tokens (per rollout)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
f2 = []
for j, mode in enumerate(MODES):
    ax = axes[j]
    for key, arm, step, wf, label, color in present:
        rs = sel(key, mode)
        if not rs: continue
        x = np.array([r['total_e2e_tokens'] for r in rs]) / 1e3; y = np.array([r['peak_ctx'] for r in rs]) / 1e3
        ax.scatter(x, y, s=9, alpha=0.35, color=color, marker='o' if wf == 'branch' else 's')
        ax.scatter([x.mean()], [y.mean()], s=140, color=color, edgecolor='k', marker='o' if wf == 'branch' else 's', zorder=5, label=label)
        f2.append(dict(mode=mode, cell=key, tokens_mean=float(x.mean() * 1e3), peak_ctx_mean=float(y.mean() * 1e3)))
    lim = max(ax.get_xlim()[1], 60)
    ax.plot([0, lim], [0, lim], color='k', lw=0.8, ls=':', label='peak = total (no discard)')
    ax.axhline(40, color='grey', lw=0.8, ls='--'); ax.text(lim * 0.98, 41, 'window (8k prompt + 32k)', ha='right', fontsize=8, color='grey')
    ax.set_xlabel("total end-to-end tokens (k)"); ax.set_ylabel("peak active context (k tokens)"); ax.set_title(f"{mode}")
    if j == 0: ax.legend(fontsize=7.5, loc='upper left')
savefig(fig, "fig2_peak_ctx_vs_e2e_tokens")
json.dump(f2, open(f"{ASSETS}/fig2_peak_ctx_vs_e2e_tokens.json", "w"), indent=1)

# F3: post-first-fold usage (fold cells only)
foldcells = [c for c in present if c[3] == 'branch']
fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
f3 = []
for k, (mtr, lab) in enumerate([("post_total", "tokens after first fold (k)"), ("post_gen", "generated tokens after first fold (k)"),
                                ("post_turns", "turns after first fold"), ("post_tool_calls", "tool calls after first fold")]):
    ax = axes[k]; data, labels, colors = [], [], []
    for key, arm, step, wf, label, color in foldcells:
        for mode in MODES:
            v = [r[mtr] for r in sel(key, mode) if r['has_fold'] == 1]
            if not v: continue
            v = np.array(v) / (1e3 if 'tokens' in lab else 1)
            data.append(v); labels.append(f"{label.split(' / ')[0]}\n{mode}"); colors.append(color)
            f3.append(dict(metric=mtr, cell=key, mode=mode, n_folded=len(v), mean=float(v.mean()), median=float(np.median(v))))
    if data:
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, c in zip(bp['boxes'], colors): patch.set_facecolor(c); patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(lab, fontsize=9)
fig.suptitle("Interaction after the first fold (rollouts that folded at least once)", fontsize=11)
savefig(fig, "fig3_post_first_fold_usage")
json.dump(f3, open(f"{ASSETS}/fig3_post_first_fold_usage.json", "w"), indent=1)

# F4: paired per-task differences
fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
f4 = []
for k, (mtr, lab, sc) in enumerate([("success", "Δ success", 1), ("total_e2e_tokens", "Δ total e2e tokens (k)", 1e3),
                                    ("peak_ctx", "Δ peak active context (k)", 1e3), ("turns", "Δ turns", 1)]):
    ax = axes[k]; pos = 0; ticks, tlabels = [], []
    for a, b, lab_p in PAIRS[:4]:
        for mode in MODES:
            ta, tb = task_means(a, mode, mtr), task_means(b, mode, mtr)
            common = sorted(set(ta) & set(tb))
            if len(common) < 5: continue
            d = np.array([ta[t] - tb[t] for t in common]) / sc
            mu, lo, hi = boot_ci(d)
            jitter = rng.normal(0, 0.08, len(d))
            ax.scatter(pos + jitter, d, s=6, alpha=0.3, color='grey')
            ax.errorbar([pos], [mu], yerr=[[mu - lo], [hi - mu]], fmt='D', color='#B64342' if 'FoldGRPO/fold − GRPO' in lab_p else '#0F4D92', ms=7, capsize=4, zorder=5)
            ticks.append(pos); tlabels.append(f"{lab_p.replace(' (same tasks)', '')}\n{mode}"); pos += 1
            f4.append(dict(metric=mtr, pair=lab_p, mode=mode, n=len(common), mean=mu * sc, lo=lo * sc, hi=hi * sc))
    ax.axhline(0, color='k', lw=0.8); ax.set_xticks(ticks); ax.set_xticklabels(tlabels, fontsize=5.5, rotation=90); ax.set_ylabel(lab, fontsize=9)
fig.suptitle("Task-level paired differences (mean ± 95% bootstrap CI; grey = tasks)", fontsize=11)
savefig(fig, "fig4_paired_task_diffs")
json.dump(f4, open(f"{ASSETS}/fig4_paired_task_diffs.json", "w"), indent=1)

# F5: token composition per cell (stacked)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
comp_keys = [("prompt_tokens", "task prompt", "#dddddd"), ("gen_tokens", "generated (all threads)", "#0F4D92"), ("obs_tokens", "tool observations", "#8BCF8B"),
             ("comp_in_tokens", "fold prompts + returns", "#B64342"), ("frame_tokens", "template framing", "#999999")]
f5 = []
for j, mode in enumerate(MODES):
    ax = axes[j]; bottom = np.zeros(len(present)); xs = np.arange(len(present))
    for mtr, lab, c in comp_keys:
        vals = np.array([np.mean([r[mtr] for r in sel(key, mode)]) if sel(key, mode) else 0 for key, *_ in present]) / 1e3
        ax.bar(xs, vals, bottom=bottom, color=c, label=lab, width=0.65); bottom += vals
        for (key, *_), v in zip(present, vals): f5.append(dict(mode=mode, cell=key, component=mtr, mean_tokens=float(v * 1e3)))
    ax.set_xticks(xs); ax.set_xticklabels([c[4] for c in present], rotation=30, ha='right', fontsize=8); ax.set_ylabel("mean tokens per rollout (k)"); ax.set_title(mode)
    if j == 0: ax.legend(fontsize=7.5)
savefig(fig, "fig5_token_composition")
json.dump(f5, open(f"{ASSETS}/fig5_token_composition.json", "w"), indent=1)


# ---------------- markdown ----------------
def S(key, mode, cond="all"):
    for r in summary:
        if r['cell'] == key and r['mode'] == mode and r['cond'] == cond: return r
    return None


def main_table(mode, cond):
    cols = [("success", "success", 3), ("total_e2e_tokens", "e2e tokens", 0), ("gen_tokens", "generated", 0), ("obs_tokens", "observations", 0),
            ("comp_in_tokens", "fold in", 0), ("comp_gen_tokens", "fold gen", 0), ("turns", "turns", 1), ("tool_calls", "tool calls", 1),
            ("branch_calls", "branches", 2), ("forward_passes", "fwd passes", 1), ("cum_prompt_tokens", "cum. input", 0), ("peak_ctx", "peak ctx", 0),
            ("mean_ctx", "mean ctx", 0), ("overlong", "overlong", 2), ("post_total", "post-fold tokens", 0), ("post_turns", "post-fold turns", 1)]
    out = ["| cell (train / inference) | n | " + " | ".join(c[1] for c in cols) + " |", "|---|---|" + "|".join("---" for _ in cols) + "|"]
    for key, arm, step, wf, label, color in present:
        r = S(key, mode, cond)
        if not r: continue
        out.append(f"| {label} | {r['n']} | " + " | ".join(fmt(r[c[0]], c[2]) for c in cols) + " |")
    return "\n".join(out)


def paired_table(mode):
    out = ["| comparison | metric | n tasks | mean Δ [95% CI] | median Δ | % tasks Δ>0 | sign-test p |", "|---|---|---|---|---|---|---|"]
    for p in paired:
        if p['mode'] != mode or p['metric'] not in ("success", "total_e2e_tokens", "peak_ctx", "cum_prompt_tokens", "turns", "tool_calls"): continue
        d = 3 if p['metric'] == 'success' else (1 if p['metric'] in ('turns', 'tool_calls') else 0)
        out.append(f"| {p['pair']} | {p['metric']} | {p['n_tasks']} | {fmt(p['mean_diff'], d)} [{fmt(p['ci_lo'], d)}, {fmt(p['ci_hi'], d)}] | {fmt(p['median_diff'], d)} | {p['frac_positive']*100:.0f}% | {p['sign_p']:.3f} |")
    return "\n".join(out)


def stopreasons(mode):
    out = ["| cell | n | finish | max_turn | timeout | llm_none | overlong (main ≥ 32k) | folded ≥1 | branch limit |", "|---|---|---|---|---|---|---|---|---|"]
    for key, arm, step, wf, label, color in present:
        rs = sel(key, mode)
        if not rs: continue
        sr = {}
        for r in rs: sr[r['stop_reason']] = sr.get(r['stop_reason'], 0) + 1
        n = len(rs)
        out.append(f"| {label} | {n} | " + " | ".join(f"{sr.get(k, 0)/n*100:.0f}%" for k in ('finish', 'max_turn', 'timeout', 'llm_none')) +
                   f" | {np.mean([r['overlong'] for r in rs])*100:.0f}% | {np.mean([r['has_fold'] for r in rs])*100:.0f}% | {np.mean([r['branch_limit_hit'] for r in rs])*100:.0f}% |")
    return "\n".join(out)


cellstat = "\n".join(f"| {k[0]} | {k[1]} | {v['tag']} | {v['n']} | {'DONE' if v['done'] else 'partial'} | {fmt(v['val_score'], 3)} |" for k, v in cellinfo.items() if v['n'])
md = f"""# FoldAgent on BrowseComp-Plus: context efficiency vs end-to-end token usage

*Build: `python3 docs/reports/build_e2e_report.py` (this file, assets and figures are regenerated from the HDFS ledgers). Repo commit at build time: `{subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}`.*

## 1. Question and design

Does FoldAgent's folding reduce **active context** while *increasing* the total interaction needed to solve the same task?
Three efficiencies are separated: **context efficiency** (peak / mean active context per forward pass), **trajectory efficiency**
(turns, tool calls, tokens generated), and **end-to-end inference efficiency** (every token generated or returned before anything was
folded away, and the cumulative model input over all forward passes).

**Cells.** Training ∈ {{no-RL Qwen3-8B base, GRPO checkpoint 60 (59 real steps), FoldGRPO checkpoint 50 (46 real steps)}} ×
inference ∈ {{**no fold**: `plugin.workflow=search` (the branch tool is removed from the prompt; single-thread ReAct with search/open_page/finish),
**fold**: `plugin.workflow=search_branch` (the training-time prompt with the branch/return tools; the policy decides when to branch, a branch
inherits the main context, runs its own sub-trajectory and is folded to its return message)}}. Both RL checkpoints were trained with the
branch tool available: "vanilla GRPO" is the authors' baseline (same agent, plain GRPO, no fold-specific process reward), not a branch-free
ReAct-GRPO. The no-fold cells are therefore an inference-time ablation for every model.

**Evaluation.** Identical to the fix-campaign final evals: 150-item BC-Plus test split, two decodings — greedy (n=1) and T=1.0 top-p=1.0 n=4 —
prompt 8k + response window 32k per thread, `max_turn` 100 across threads, up to 10 branches, gpt-oss-120b judge, self-contained 8×H100 val-only
batch jobs (`infra/worker_val_selfcontained.sh`, `FOLD_WORKFLOW`, `FOLD_E2E_LEDGER=1`). Same tasks and same sampling settings across cells; task-level
pairing by `instance_id` (t1n4 pairs the per-task mean of 4 samples).

**Ledger.** `agents/e2e_ledger.py` records, for every thread of every rollout, each turn's exact token ids the policy generated or was fed
(`scripts/e2e_metrics.py` defines every metric). Folded-away branch tokens are counted in full; a branch's inherited context is counted as
model input of its forward passes, not as new tokens.

| cell | mode | results dir tag | rollouts | status | trainer val score |
|---|---|---|---|---|---|
{cellstat}

## 2. Main tables

### 2.1 All rollouts — greedy
{main_table('greedy', 'all')}

### 2.2 All rollouts — T=1.0, n=4
{main_table('t1n4', 'all')}

### 2.3 Successful rollouts only — greedy
{main_table('greedy', 'success')}

### 2.4 Successful rollouts only — T=1.0, n=4
{main_table('t1n4', 'success')}

Columns: *e2e tokens* = prompt + generated + framing + observations + fold prompts/returns (everything before discard); *fold in* = branch prompts,
overflow-summary prompts and folded return messages injected into contexts; *fold gen* = policy tokens spent on branch-call / return / summary turns;
*cum. input* = Σ input length over all forward passes; *peak ctx* = max input+output length of any forward pass; *post-fold* = after the first fold
(rollouts that folded; NaN otherwise). Means with bootstrap CIs are in `assets/cell_summary.csv`.

### 2.5 Termination
Greedy:

{stopreasons('greedy')}

T=1.0 n=4:

{stopreasons('t1n4')}

## 3. Paired task-level comparisons

Greedy (one rollout per task):

{paired_table('greedy')}

T=1.0 n=4 (per-task mean of 4 samples):

{paired_table('t1n4')}

## 4. Figures

![](./{STEM}_assets/fig1_success_vs_e2e_tokens.png)
*Fig. 1 — Success vs total end-to-end tokens (top: cell means ± 95% CI; bottom: fraction of rollouts solved within an end-to-end token budget).*

![](./{STEM}_assets/fig2_peak_ctx_vs_e2e_tokens.png)
*Fig. 2 — Peak active context vs total end-to-end tokens per rollout (large markers = cell means). Points below the dotted diagonal discarded tokens.*

![](./{STEM}_assets/fig3_post_first_fold_usage.png)
*Fig. 3 — Interaction after the first fold, for rollouts that folded.*

![](./{STEM}_assets/fig4_paired_task_diffs.png)
*Fig. 4 — Paired per-task differences.*

![](./{STEM}_assets/fig5_token_composition.png)
*Fig. 5 — Where the end-to-end tokens go.*

## 5. Findings

FINDINGS_PLACEHOLDER

## 6. Caveats

- Both RL checkpoints are truncated runs (46 / 59 real steps) from the fix campaign and were trained *with* the branch tool; the no-fold cells
  are an inference-time ablation with a different system prompt (`search` vs `search_branch`), so they measure "the same policy without the
  ability to fold", not a branch-free-trained baseline.
- "Compaction" here is FoldAgent's branch/fold; the CompactionRL-style summary compaction (`compactiongrpo` arm) was still training and is not included.
- Judge = gpt-oss-120b (not the paper's GPT models); success = judge-graded correctness of the final answer (0/1).
- The bottom of the 32k window: a thread that fills its window keeps its partial context; `overlong` marks main threads that reached the window.
- Token counts are exact ids from the ledger; the branch's inherited context is re-rendered by the chat template (±1 token per inherited turn).
- Greedy decoding on vLLM is not bit-reproducible across nodes; t1n4 CIs reflect sampling noise over 4 samples × 150 tasks.

## 7. Reproduce

```
# 1. jobs (already submitted; specs in infra/jobs/fold_e2e_*_h100.json, ledger in infra/jobs/JOBS.tsv)
cd ~/xiaoxuan/FoldAgent/infra/jobs && merlin-cli --control-plane i18n-tt job-v2 runs create --from-file fold_e2e_grpo60_branch_h100.json   # ... x6
# 2. report (reads /mnt/hdfs/mlsys/xiaoxuan/fold_replication/results/valonly_*_e2e_*_sc/ledger)
python3 docs/reports/build_e2e_report.py
# tests: ~/xiaoxuan/envs/fold_train/bin/python -m unittest tests.test_e2e_ledger tests.test_e2e_metrics
```
Artifacts: `docs/reports/{STEM}_assets/` (rollouts.csv = one row per rollout with every metric; cell_summary.csv; paired_diffs.csv; figures as PNG+PDF with JSON data).
"""
findings_file = f"{RD}/{STEM}_findings.md"
findings = open(findings_file).read() if os.path.exists(findings_file) else "_(findings are written after the runs complete: see `" + os.path.basename(findings_file) + "`)_"
md = md.replace("FINDINGS_PLACEHOLDER", findings)
open(MD, "w").write(md)
open(f"{ASSETS}/README.md", "w").write(f"""# {STEM} — assets
Regenerate with `python3 docs/reports/build_e2e_report.py`. Inputs: HDFS ledgers listed in the report §1.
- rollouts.csv — one row per rollout (cell, mode, instance_id, every metric of scripts/e2e_metrics.py)
- cell_summary.csv — per cell × mode × (all|success): mean, 95% bootstrap CI, median of each metric; stop-reason counts
- paired_diffs.csv — task-level paired differences with bootstrap CI and sign-test p
- fig1..fig5 .png/.pdf with the plotted numbers in the matching .json
""")
print("MD ->", MD)
if "--no-pdf" not in sys.argv:
    subprocess.run([FOLD_INFRA_PY, os.path.abspath(__file__), "--render-pdf"], check=True)
