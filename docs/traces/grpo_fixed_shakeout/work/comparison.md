| metric | A_branch | A_nobranch | B_compact | OLD_branch_18 |
|---|---|---|---|---|
| rollouts | 18 | 18 | 18 | 18 |
| accuracy | 0.278 | 0.111 | 0.056 | 0.278 |
| correct | 5 | 2 | 1 | 5 |
| finished | 11 | 2 | 4 | 14 |
| unfinished | 7 | 16 | 14 | 4 |
| stop_reasons | {'finish': 11, 'llm_none': 7} | {'llm_none': 16, 'finish': 2} | {'budget_exhausted': 14, 'finish': 4} | {'finish': 14, 'context_exhausted(llm_none)': 4} |
| n_main_turns | {'mean': 5.1, 'median': 5, 'max': 8} | {'mean': 7.7, 'median': 8, 'max': 14} | {'mean': 5.8, 'median': 6, 'max': 10} | {'mean': 6.9, 'median': 7, 'max': 13} |
| n_branches | {'mean': 1.4, 'median': 1, 'max': 4} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 4.1, 'median': 3, 'max': 10} |
| n_compactions | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 2.7, 'median': 3, 'max': 3} | {'mean': 0.0, 'median': 0, 'max': 0} |
| gen_main | {'mean': 3244.5, 'median': 3183, 'max': 7573} | {'mean': 3120.7, 'median': 2911, 'max': 5832} | {'mean': 3054.0, 'median': 3136, 'max': 5794} | {'mean': 2832.4, 'median': 2546, 'max': 6092} |
| gen_branch | {'mean': 824.5, 'median': 749, 'max': 2241} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 2907.9, 'median': 2878, 'max': 6530} |
| gen_summary | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 2227.0, 'median': 2329, 'max': 3198} | {'mean': 0.0, 'median': 0, 'max': 0} |
| gen_total | {'mean': 4069.0, 'median': 4003, 'max': 7573} | {'mean': 3120.7, 'median': 2911, 'max': 5832} | {'mean': 5281.0, 'median': 5511, 'max': 8797} | {'mean': 5740.3, 'median': 5517, 'max': 12077} |
| prefill_tokens | {'mean': 266808.8, 'median': 219758, 'max': 888928} | {'mean': 174193.2, 'median': 180904, 'max': 333876} | {'mean': 355246.1, 'median': 408469, 'max': 611992} | {'mean': 822298.5, 'median': 687888, 'max': 2012012} |
| peak_context | {'mean': 34864.1, 'median': 37176, 'max': 44451} | {'mean': 39594.1, 'median': 40610, 'max': 47668} | {'mean': 33989.1, 'median': 34210, 'max': 37292} | {'mean': 36595.9, 'median': 37294, 'max': 43410} |
| peak_context_main | {'mean': 24614.7, 'median': 22830, 'max': 44451} | {'mean': 39594.1, 'median': 40610, 'max': 47668} | {'mean': 31429.8, 'median': 33064, 'max': 37292} | {'mean': 17889.2, 'median': 12958, 'max': 43410} |
| wall_s | {'mean': 43.0, 'median': 43.6, 'max': 74.8} | {'mean': 24.4, 'median': 27.2, 'max': 43.6} | {'mean': 71.0, 'median': 72.7, 'max': 102.9} | {'mean': 584.1, 'median': 672.9, 'max': 900.7} |
| calls_by_type | {'search': 106, 'branch': 25, 'finish': 11, 'open_page': 67, 'return': 11} | {'search': 78, 'open_page': 45, 'finish': 2} | {'search': 158, 'open_page': 105, 'finish': 4} | {'search': 268, 'branch': 74, 'finish': 14, 'open_page': 274, 'return': 44} |
| multi_call_turns | 0 | 6 | 5 | 0 |
| malformed_turns | 26 | 8 | 9 | 33 |
| calls_in_think | 4 | 9 | 13 | 2 |
| unsupported_calls | 0 | 0 | 0 | 0 |
| duplicate_queries | 32 | 20 | 68 | 113 |
| duplicate_branch_prompts | 1 | 0 | 0 | 1 |
| long_observations | 1 | 2 | 9 | 0 |
| cap_hits | 4 | 8 | 4 | 4 |
| cap_over | 0 | 0 | 0 | 4 |
| history_missing_turns | 0 | 0 | 0 | 0 |
| prefix_breaks | 0 | 0 | 0 | 0 |
| fork_checks | {'exact': 25} | {} | {} | {} |
| tail_checks | {} | {} | {'exact': 190} | {} |
| other_checks | {'designed_rollback': 12, 'fork_ok': 25, 'history_missing_after_rollback': 4} | {} | {'tail_ok': 190, 'designed_rollback': 7, 'history_missing_after_rollback': 3} | {'glued_boundary_prompts': 603} |
| turns_with_glued_boundary | 0 | 0 | 0 | 603 |
| prompts_with_double_newline_empty_think | 12 | 0 | 0 | 466 |
| think | {'present': 43, 'empty': 186, 'none': 17} | {'present': 40, 'empty': 74, 'none': 24} | {'present': 47, 'empty': 242, 'none': 8} | {'present': 50, 'empty': 636, 'none': 8} |
| mask_ok | 239 | 122 | 297 | 0 |
| mask_checked | 239 | 122 | 297 | 0 |
| prompt_sha1_ok | [246, 246] | [138, 138] | [297, 297] | [0, 0] |
| exec_subset_ok | [214, 214] | [122, 122] | [248, 248] | [0, 0] |
| obs_concat_ok | [198, 198] | [113, 113] | [241, 241] | [0, 0] |
| citations | {'rollouts_with_citations': 5, 'unseen_citation_rollouts': 0} | {'rollouts_with_citations': 2, 'unseen_citation_rollouts': 0} | {'rollouts_with_citations': 4, 'unseen_citation_rollouts': 0} | {'rollouts_with_citations': 3, 'unseen_citation_rollouts': 0} |

### Paired outcomes (same task, same checkpoint; not a training effect)

**A_branch vs A_nobranch** on 18 paired tasks: both correct 0, only A_branch 5, only A_nobranch 2, both wrong 11

| task (question head) | A_branch score/stop/turns/gen | A_nobranch score/stop/turns/gen |
|---|---|---|
| There is a company that is said to produce savory sweets, plain, chocolate-filled, and fla… | 0.0 / finish / 4 / 2580 | 1.0 / finish / 11 / 1229 |
| Give me the name of the scholar and associate professor that is affiliated with one of the… | 1.0 / finish / 6 / 4003 | 0.0 / llm_none / 7 / 3092 |
| In 2006, an EU-funded project with a budget of Є1.30 million was launched and completed wi… | 1.0 / finish / 3 / 2215 | 0.0 / llm_none / 9 / 1234 |
| I’m thinking of two distinct blog posts, written by the same author, in the same year, aft… | 1.0 / finish / 4 / 2382 | 0.0 / llm_none / 8 / 3258 |
| The artist’s father was a Protestant minister. They had a brief marriage to a colloquial p… | 0.0 / llm_none / 5 / 3855 | 1.0 / finish / 5 / 2389 |
| Provide the name of the person who was born in the 1940s. This individual was born in a co… | 1.0 / finish / 3 / 3655 | 0.0 / llm_none / 14 / 5832 |
| A specific sports team was founded between 2010 and 2015, both years inclusive, by a speci… | 1.0 / finish / 4 / 5062 | 0.0 / llm_none / 3 / 3471 |
**A_branch vs B_compact** on 18 paired tasks: both correct 1, only A_branch 4, only B_compact 0, both wrong 13

| task (question head) | A_branch score/stop/turns/gen | B_compact score/stop/turns/gen |
|---|---|---|
| Give me the name of the scholar and associate professor that is affiliated with one of the… | 1.0 / finish / 6 / 4003 | 0.0 / budget_exhausted / 6 / 3941 |
| In 2006, an EU-funded project with a budget of Є1.30 million was launched and completed wi… | 1.0 / finish / 3 / 2215 | 0.0 / budget_exhausted / 10 / 4138 |
| I’m thinking of two distinct blog posts, written by the same author, in the same year, aft… | 1.0 / finish / 4 / 2382 | 0.0 / budget_exhausted / 5 / 2729 |
| A specific sports team was founded between 2010 and 2015, both years inclusive, by a speci… | 1.0 / finish / 4 / 5062 | 0.0 / budget_exhausted / 6 / 3668 |
**A_branch vs OLD_branch_18** on 18 paired tasks: both correct 3, only A_branch 2, only OLD_branch_18 2, both wrong 11

| task (question head) | A_branch score/stop/turns/gen | OLD_branch_18 score/stop/turns/gen |
|---|---|---|
| There is a company that is said to produce savory sweets, plain, chocolate-filled, and fla… | 0.0 / finish / 4 / 2580 | 1.0 / finish / 3 / 1928 |
| I’m thinking of two distinct blog posts, written by the same author, in the same year, aft… | 1.0 / finish / 4 / 2382 | 0.0 / context_exhausted(llm_none) / 13 / 2031 |
| Provide the name of the person who was born in the 1940s. This individual was born in a co… | 1.0 / finish / 3 / 3655 | 0.0 / finish / 7 / 6809 |
| A volleyball game took place after 2010 but before 2023, where the winning team scored 25,… | 0.0 / finish / 5 / 4727 | 1.0 / finish / 6 / 6524 |
**A_nobranch vs B_compact** on 18 paired tasks: both correct 0, only A_nobranch 2, only B_compact 1, both wrong 15

| task (question head) | A_nobranch score/stop/turns/gen | B_compact score/stop/turns/gen |
|---|---|---|
| The artist’s father was a Protestant minister. They had a brief marriage to a colloquial p… | 1.0 / finish / 5 / 2389 | 0.0 / budget_exhausted / 5 / 5511 |
| There is a company that is said to produce savory sweets, plain, chocolate-filled, and fla… | 1.0 / finish / 11 / 1229 | 0.0 / finish / 6 / 3163 |
| Provide the name of the person who was born in the 1940s. This individual was born in a co… | 0.0 / llm_none / 14 / 5832 | 1.0 / finish / 3 / 2966 |
**A_nobranch vs OLD_branch_18** on 18 paired tasks: both correct 1, only A_nobranch 1, only OLD_branch_18 4, both wrong 12

| task (question head) | A_nobranch score/stop/turns/gen | OLD_branch_18 score/stop/turns/gen |
|---|---|---|
| In 2006, an EU-funded project with a budget of Є1.30 million was launched and completed wi… | 0.0 / llm_none / 9 / 1234 | 1.0 / finish / 4 / 4536 |
| The artist’s father was a Protestant minister. They had a brief marriage to a colloquial p… | 1.0 / finish / 5 / 2389 | 0.0 / context_exhausted(llm_none) / 5 / 4129 |
| Give me the name of the scholar and associate professor that is affiliated with one of the… | 0.0 / llm_none / 7 / 3092 | 1.0 / finish / 9 / 7544 |
| A volleyball game took place after 2010 but before 2023, where the winning team scored 25,… | 0.0 / llm_none / 8 / 2867 | 1.0 / finish / 6 / 6524 |
| A specific sports team was founded between 2010 and 2015, both years inclusive, by a speci… | 0.0 / llm_none / 3 / 3471 | 1.0 / finish / 7 / 10490 |
**B_compact vs OLD_branch_18** on 18 paired tasks: both correct 0, only B_compact 1, only OLD_branch_18 5, both wrong 12

| task (question head) | B_compact score/stop/turns/gen | OLD_branch_18 score/stop/turns/gen |
|---|---|---|
| In 2006, an EU-funded project with a budget of Є1.30 million was launched and completed wi… | 0.0 / budget_exhausted / 10 / 4138 | 1.0 / finish / 4 / 4536 |
| Give me the name of the scholar and associate professor that is affiliated with one of the… | 0.0 / budget_exhausted / 6 / 3941 | 1.0 / finish / 9 / 7544 |
| A specific sports team was founded between 2010 and 2015, both years inclusive, by a speci… | 0.0 / budget_exhausted / 6 / 3668 | 1.0 / finish / 7 / 10490 |
| Provide the name of the person who was born in the 1940s. This individual was born in a co… | 1.0 / finish / 3 / 2966 | 0.0 / finish / 7 / 6809 |
| There is a company that is said to produce savory sweets, plain, chocolate-filled, and fla… | 0.0 / finish / 6 / 3163 | 1.0 / finish / 3 / 1928 |
| A volleyball game took place after 2010 but before 2023, where the winning team scored 25,… | 0.0 / finish / 8 / 6297 | 1.0 / finish / 6 / 6524 |
