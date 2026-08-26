# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib",
#   "seaborn",
#   "pandas",
#   "numpy",
# ]
# ///
"""Figures for the FoldAgent replication report (paper format).
Data provenance: CSVs in ../2026-08-12_foldagent_bcplus_replication_report_assets/
(themselves derived from results/valonly_*/ logs and dumps — see that folder's README)."""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/tiger/.claude/skills/generate-report/scripts')
from plots import bar, line, save_svg, PALETTE, CATEGORICAL

A = '/home/tiger/xiaoxuan/FoldAgent/docs/reports/2026-08-12_foldagent_bcplus_replication_report_assets'
OUT = '/home/tiger/xiaoxuan/FoldAgent/docs/reports/paper_format'

# --- Fig 1: scores by arm, both operating points, 95% binomial CIs ---
d1 = pd.read_csv(f'{A}/fig1_scores_by_arm.csv')
d1['operating point'] = d1['operating_point'].map({
    'greedy_pass@1': 'greedy pass@1 (n=150)',
    't1n4_mean@4': 'T=1.0 mean@4 (n=600)'}).fillna(d1['operating_point'])
fig = bar(d1, x='arm', y='score', hue='operating point', err='ci95_binomial',
          title='BC-Plus pass@1 by arm and decoding regime',
          ylabel='pass@1', xlabel='')
fig.set_size_inches(8.6, 4.6)
save_svg(fig, f'{OUT}/fig1_scores.svg')

# --- Fig 2: training curves (reward + val overlay; entropy) ---
d2 = pd.read_csv(f'{A}/fig2_training_curves.csv')
d2 = d2.sort_values(['arm', 'step']).groupby(['arm', 'step'], as_index=False).last()  # last occurrence per step
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
colors = {'foldgrpo': PALETTE.get('accent', '#b6274e'), 'grpo': PALETTE.get('primary', '#225573')}
for arm, g in d2.groupby('arm'):
    c = colors.get(arm, CATEGORICAL[0])
    axes[0].plot(g['step'], g['reward_avg_score'], color=c, lw=1.4, alpha=.85, label=f'{arm} (train reward)')
    v = g.dropna(subset=['val_avg_score'])
    axes[0].scatter(v['step'], v['val_avg_score'], color=c, marker='o', s=28, zorder=5,
                    edgecolor='white', linewidth=.6, label=f'{arm} (val, greedy)')
    axes[1].plot(g['step'], g['actor_entropy'], color=c, lw=1.4, alpha=.85, label=arm)
dead = d2[(d2['arm'] == 'grpo') & (d2['step'].isin([95, 96]))]
axes[0].scatter(dead['step'], dead['reward_avg_score'], marker='x', color='#d4880f', s=45, zorder=6,
                label='grpo dead steps (infra outage)')
axes[0].set_title('Training reward and validation score')
axes[0].set_xlabel('training step'); axes[0].set_ylabel('score')
axes[0].legend(fontsize=7.5, frameon=False)
axes[1].set_title('Policy entropy')
axes[1].set_xlabel('training step'); axes[1].set_ylabel('actor entropy')
axes[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
save_svg(fig, f'{OUT}/fig2_training.svg')

# --- Fig 3: behavior signatures, 3 panels ---
d3 = pd.read_csv(f'{A}/fig3_behavior_signatures.csv')
panels = [('overlong_rate', 'Context-blowout rate', None),
          ('avg_branches_per_traj', 'Branches per trajectory', None),
          ('avg_main_thread_chars', 'Main-thread length', 1e-3)]
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
armcols = [PALETTE.get('gray', '#7d99b1'), PALETTE.get('primary', '#225573'), PALETTE.get('accent', '#b6274e')]
for ax, (col, title, scale) in zip(axes, panels):
    vals = d3[col] * (scale or 1)
    ax.bar(d3['arm'], vals, color=armcols, width=.55)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f'{v:.2f}', ha='center', fontsize=8)
    ax.set_title(title + (' (k chars)' if scale else ''), fontsize=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_ylim(0, max(vals) * 1.18)
fig.suptitle('Behavioral signatures, greedy decoding (150 trajectories/arm)', fontsize=11)
fig.tight_layout()
save_svg(fig, f'{OUT}/fig3_behavior.svg')

# --- Fig 4: ECDF of main-thread tool calls ---
d4 = pd.read_csv(f'{A}/fig4_turns_ecdf.csv')
rows = []
for arm, g in d4.groupby('arm'):
    x = np.sort(g['n_main_thread_tool_calls'].values)
    yv = np.arange(1, len(x) + 1) / len(x)
    rows.append(pd.DataFrame({'arm': arm, 'main-thread tool calls': x, 'ECDF': yv}))
ecdf = pd.concat(rows)
fig = line(ecdf, x='main-thread tool calls', y='ECDF', hue='arm',
           title='ECDF of main-thread tool calls per trajectory (greedy, n=150/arm)',
           ylabel='fraction of trajectories')
fig.set_size_inches(8.2, 4.6)
save_svg(fig, f'{OUT}/fig4_ecdf.svg')
print('all figures written')
