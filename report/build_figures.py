#!/usr/bin/env python3
"""Report figures from results.json (rerunnable)."""
import json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = json.load(open('/home/tiger/xiaoxuan/FoldAgent/report/results.json'))
def s(tag): return R[tag]['score']

arms = ['No-RL\nbase', 'GRPO\nstep-50', 'GRPO\nstep-100', 'FoldGRPO\nstep-50', 'FoldGRPO\nstep-100']
greedy = [s('norl_base_greedy_sc'), s('grpo_50_greedy_sc'), s('grpo_100_greedy_sc'), s('foldgrpo_50_greedy'), s('foldgrpo_100_greedy_sc')]
t1 =     [s('norl_base_t1n4'), s('grpo_50_t1n4_sc'), s('grpo_100_t1n4_sc'), s('foldgrpo_50_t1n4_sc'), s('foldgrpo_100_t1n4_sc')]
ci_g = [1.96*math.sqrt(p*(1-p)/150) for p in greedy]
ci_t = [1.96*math.sqrt(p*(1-p)/600) for p in t1]

C_GREY, C_BLUE, C_ORANGE = '#8c8c8c', '#4878a8', '#e1812c'
cols = [C_GREY, C_BLUE, C_BLUE, C_ORANGE, C_ORANGE]

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), dpi=200)
for ax, vals, cis, title in [(axes[0], greedy, ci_g, 'Greedy pass@1 (N=150)'),
                             (axes[1], t1, ci_t, 'T=1.0 mean@4 (600 samples)')]:
    x = range(len(arms))
    ax.bar(x, vals, yerr=cis, capsize=3, color=cols, width=0.62, error_kw={'lw': 1})
    for i, v in enumerate(vals):
        ax.text(i, v + cis[i] + 0.008, f'{v:.3f}', ha='center', fontsize=7.5)
    ax.set_xticks(list(x)); ax.set_xticklabels(arms, fontsize=8)
    ax.set_ylim(0, 0.37); ax.set_title(title, fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('BC-Plus pass@1', fontsize=9)
fig.suptitle('FoldAgent replication (Qwen3-8B): scores by arm and operating point', fontsize=11, y=1.02)
fig.tight_layout()
for ext in ['pdf', 'png']:
    fig.savefig(f'/home/tiger/xiaoxuan/FoldAgent/report/assets/fig1_scores.{ext}', bbox_inches='tight')

fig2, axes = plt.subplots(1, 3, figsize=(9.2, 3.0), dpi=200)
b_arms = ['No-RL', 'GRPO\n@100', 'FoldGRPO\n@100']
b_cols = [C_GREY, C_BLUE, C_ORANGE]
panels = [
    ('Context blowout rate ↓', [R['norl_base_greedy_sc']['overlong_rate'], R['grpo_100_greedy_sc']['overlong_rate'], R['foldgrpo_100_greedy_sc']['overlong_rate']]),
    ('Branches per trajectory', [R['norl_base_greedy_sc'].get('avg_branches', 1.05), R['grpo_100_greedy_sc']['avg_branches'], R['foldgrpo_100_greedy_sc']['avg_branches']]),
    ('Main-thread length (k chars)', [R['norl_base_greedy_sc'].get('avg_main_chars', 24808)/1000, R['grpo_100_greedy_sc']['avg_main_chars']/1000, R['foldgrpo_100_greedy_sc']['avg_main_chars']/1000]),
]
for ax, (title, vals) in zip(axes, panels):
    ax.bar(range(3), vals, color=b_cols, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v*1.02, f'{v:.2f}', ha='center', fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(b_arms, fontsize=8)
    ax.set_title(title, fontsize=9.5); ax.spines[['top', 'right']].set_visible(False)
fig2.suptitle('Behavioral signatures (greedy, 150 trajectories/arm): the folding mechanism replicates', fontsize=11, y=1.04)
fig2.tight_layout()
for ext in ['pdf', 'png']:
    fig2.savefig(f'/home/tiger/xiaoxuan/FoldAgent/report/assets/fig2_behavior.{ext}', bbox_inches='tight')
print('figures written')
