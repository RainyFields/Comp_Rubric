| metric | full_A_branch | full_A_nobranch | full_B_compact | OLD_branch_150 |
|---|---|---|---|---|
| rollouts | 150 | 150 | 150 | 150 |
| accuracy | 0.367 | 0.087 | 0.24 | 0.0 |
| correct | 55 | 13 | 36 | 0 |
| finished | 132 | 24 | 65 | 0 |
| unfinished | 18 | 126 | 85 | 150 |
| stop_reasons | {'llm_none': 18, 'finish': 132} | {'llm_none': 125, 'finish': 24, 'no_call_loop': 1} | {'budget_exhausted': 83, 'finish': 65, 'no_call_loop': 2} | {'context_exhausted(llm_none)': 30, 'unknown(timeout_or_budget)': 120} |
| n_main_turns | {'mean': 5.3, 'median': 5, 'max': 12} | {'mean': 7.1, 'median': 7, 'max': 18} | {'mean': 5.7, 'median': 5, 'max': 22} | {'mean': 5.9, 'median': 5, 'max': 16} |
| n_branches | {'mean': 2.4, 'median': 1, 'max': 10} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 2.7, 'median': 2, 'max': 10} |
| n_compactions | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 2.1, 'median': 3, 'max': 3} | {'mean': 0.0, 'median': 0, 'max': 0} |
| gen_main | {'mean': 2756.6, 'median': 2391, 'max': 7719} | {'mean': 2847.5, 'median': 2410, 'max': 7460} | {'mean': 3419.5, 'median': 3237, 'max': 8223} | {'mean': 2694.4, 'median': 2448, 'max': 9005} |
| gen_branch | {'mean': 1536.7, 'median': 896, 'max': 7484} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 1709.3, 'median': 1088, 'max': 8031} |
| gen_summary | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 0.0, 'median': 0, 'max': 0} | {'mean': 1690.2, 'median': 1965, 'max': 4209} | {'mean': 0.0, 'median': 0, 'max': 0} |
| gen_total | {'mean': 4293.3, 'median': 4085, 'max': 10683} | {'mean': 2847.5, 'median': 2410, 'max': 7460} | {'mean': 5109.7, 'median': 5012, 'max': 11060} | {'mean': 4403.7, 'median': 3833, 'max': 12563} |
| prefill_tokens | {'mean': 424435.3, 'median': 226286, 'max': 2202157} | {'mean': 163802.6, 'median': 157812, 'max': 371860} | {'mean': 285483.1, 'median': 281323, 'max': 905721} | {'mean': 498628.3, 'median': 315483, 'max': 2266277} |
| peak_context | {'mean': 31458.6, 'median': 35783, 'max': 45787} | {'mean': 41956.4, 'median': 40563, 'max': 234309} | {'mean': 31224.3, 'median': 33756, 'max': 36915} | {'mean': 33150.0, 'median': 36244, 'max': 46570} |
| peak_context_main | {'mean': 17328.0, 'median': 13621, 'max': 45787} | {'mean': 41956.4, 'median': 40563, 'max': 234309} | {'mean': 29862.0, 'median': 32227, 'max': 36915} | {'mean': 19656.5, 'median': 15719, 'max': 46570} |
| wall_s | {'mean': 396.4, 'median': 376.6, 'max': 766.4} | {'mean': 112.0, 'median': 130.4, 'max': 230.5} | {'mean': 476.4, 'median': 568.6, 'max': 745.7} | {'mean': 469.6, 'median': 441.3, 'max': 920.3} |
| calls_by_type | {'search': 1165, 'branch': 355, 'finish': 133, 'open_page': 1105, 'return': 160} | {'search': 597, 'open_page': 570, 'finish': 25} | {'search': 1034, 'open_page': 744, 'finish': 66} | {'search': 1425, 'open_page': 1130, 'finish': 120, 'branch': 423, 'return': 208} |
| multi_call_turns | 35 | 45 | 51 | 0 |
| malformed_turns | 294 | 62 | 60 | 285 |
| calls_in_think | 16 | 84 | 61 | 26 |
| unsupported_calls | 0 | 0 | 0 | 0 |
| duplicate_queries | 364 | 131 | 391 | 486 |
| duplicate_branch_prompts | 10 | 0 | 0 | 15 |
| long_observations | 43 | 42 | 55 | 0 |
| cap_hits | 43 | 51 | 50 | 30 |
| cap_over | 0 | 0 | 0 | 30 |
| history_missing_turns | 0 | 0 | 0 | 0 |
| prefix_breaks | 0 | 0 | 0 | 0 |
| fork_checks | {'exact': 353} | {} | {} | {} |
| tail_checks | {} | {} | {'exact': 1190} | {} |
| other_checks | {'fork_ok': 353, 'designed_rollback': 228, 'history_missing_after_rollback': 83, 'exec_not_subset': 3} | {} | {'tail_ok': 1190, 'designed_rollback': 37, 'history_missing_after_rollback': 20} | {'glued_boundary_prompts': 2994} |
| turns_with_glued_boundary | 0 | 0 | 0 | 2994 |
| prompts_with_double_newline_empty_think | 228 | 0 | 0 |  |
| think | {'present': 351, 'empty': 2759, 'none': 76} | {'present': 333, 'empty': 547, 'none': 181} | {'present': 485, 'none': 57, 'empty': 1519} | {'present': 390, 'none': 62, 'empty': 3117} |
| mask_ok | 3168 | 936 | 2061 | 0 |
| mask_checked | 3168 | 936 | 2061 | 0 |
| prompt_sha1_ok | [3186, 3186] | [1061, 1061] | [2061, 2061] | [0, 0] |
| exec_subset_ok | [2813, 2816] | [936, 936] | [1748, 1748] | [0, 0] |
| obs_concat_ok | [2575, 2575] | [865, 866] | [1631, 1631] | [0, 0] |
| citations | {'rollouts_with_citations': 67, 'unseen_citation_rollouts': 2} | {'rollouts_with_citations': 21, 'unseen_citation_rollouts': 1} | {'rollouts_with_citations': 56, 'unseen_citation_rollouts': 0} | {'rollouts_with_citations': 58, 'unseen_citation_rollouts': 58} |

### Paired outcomes (same task, same checkpoint; not a training effect)

**full_A_branch vs full_A_nobranch** on 150 paired tasks: both correct 12, only full_A_branch 43, only full_A_nobranch 1, both wrong 94

| task (question head) | full_A_branch score/stop/turns/gen | full_A_nobranch score/stop/turns/gen |
|---|---|---|
| The following details describe an individual:  An African artist Began their career by pai… | 1.0 / finish / 3 / 1410 | 0.0 / llm_none / 8 / 1671 |
| Identify the company that meets the following criteria:  - An independent non-executive di… | 1.0 / finish / 4 / 2896 | 0.0 / llm_none / 11 / 2494 |
| A 2021 review was written for a historical strategy game that was released earlier in the … | 1.0 / finish / 6 / 2951 | 0.0 / llm_none / 12 / 1602 |
| I’m thinking of two distinct blog posts, written by the same author, in the same year, aft… | 1.0 / finish / 5 / 1334 | 0.0 / llm_none / 10 / 737 |
| I am looking for the name of a group of companies that satisfy the following conditions: 1… | 1.0 / finish / 4 / 2602 | 0.0 / llm_none / 10 / 1653 |
| I belong to a fantasy world and became aware of a prophecy involving twin keys, created by… | 1.0 / finish / 3 / 1328 | 0.0 / llm_none / 7 / 1514 |
| A genus can be found in the Northern Hemisphere, largely in temperate latitudes, among oth… | 1.0 / finish / 5 / 1876 | 0.0 / llm_none / 8 / 1961 |
| An individual was encouraged to use a different tool for their work during the year betwee… | 1.0 / finish / 3 / 1965 | 0.0 / finish / 5 / 5486 |
| I want you to find the name of the series that I am talking about. Here are various plot p… | 1.0 / finish / 3 / 1567 | 0.0 / llm_none / 9 / 754 |
| There is a short film, produced in 2017, with a runtime of 19 minutes, that revolves aroun… | 1.0 / finish / 5 / 3594 | 0.0 / llm_none / 13 / 1965 |
| I'm looking for the name of a 19th-century magazine that had fewer than eight published vo… | 1.0 / finish / 4 / 2030 | 0.0 / llm_none / 3 / 1292 |
| In 2006, an EU-funded project with a budget of Є1.30 million was launched and completed wi… | 1.0 / finish / 4 / 3127 | 0.0 / llm_none / 12 / 895 |
| In the mid-2000s, a specific pet was welcomed by a nursing home at six months old. This pe… | 1.0 / finish / 4 / 3831 | 0.0 / llm_none / 16 / 951 |
| As of December 2023, this individual: - This individual is a graduate with highest honors.… | 1.0 / finish / 3 / 2277 | 0.0 / llm_none / 14 / 1139 |
| According to a 2021 article, a certain individual expressed their love for poetry and shar… | 1.0 / finish / 2 / 1469 | 0.0 / llm_none / 10 / 2434 |
| On the Blantyre Telegraph, there's an article published between 2018 and 2020, inclusive. … | 1.0 / finish / 3 / 2352 | 0.0 / finish / 6 / 1782 |
| As of 2023 there is an individual who had previously held the same cabinet-level position … | 1.0 / finish / 5 / 3309 | 0.0 / llm_none / 5 / 1721 |
| A cleric was murdered after 2005 but before 2020. The incident happened as they were about… | 1.0 / finish / 4 / 2585 | 0.0 / llm_none / 18 / 1052 |
| I need help finding the name of the art exhibition I've heard about. My only details are a… | 1.0 / finish / 6 / 5074 | 0.0 / llm_none / 10 / 1898 |
| I am looking for a specialty food shop that has at least one location in both Mexico and C… | 1.0 / finish / 4 / 2059 | 0.0 / llm_none / 15 / 2272 |
| There is this band, and it is difficult to remember their name. Can you help?   - They are… | 1.0 / finish / 5 / 3962 | 0.0 / llm_none / 11 / 1867 |
| I am looking for the title of a book first published in 1898 by an author born in the 1860… | 1.0 / finish / 5 / 3497 | 0.0 / llm_none / 8 / 1288 |
| Prior to 2010, four poems by the same writer were published in the same issue of a literar… | 1.0 / finish / 3 / 1872 | 0.0 / llm_none / 11 / 658 |
| Give me the name of the scholar and associate professor that is affiliated with one of the… | 1.0 / finish / 4 / 3586 | 0.0 / llm_none / 9 / 2460 |
| As of December 2023, what is the name of the movie based on the details below:  It was rel… | 1.0 / finish / 4 / 1927 | 0.0 / llm_none / 8 / 973 |
| Identify a rare or uncommon fungi that typically appears in clusters after rainfall, chara… | 0.0 / llm_none / 7 / 3211 | 1.0 / finish / 5 / 5087 |
| In this MMA fight that took place before 2023, the statistics for the loser included 10 si… | 1.0 / finish / 5 / 1926 | 0.0 / llm_none / 8 / 667 |
| There is an insect that is loved and enjoyed by many people in African countries and is pa… | 1.0 / finish / 3 / 1742 | 0.0 / llm_none / 8 / 1569 |
| This building originated in the 12th century. It had a tower added in the 14th century. An… | 1.0 / finish / 3 / 3035 | 0.0 / llm_none / 9 / 2819 |
| There is a software developer who, sometime after 2010 but before 2023, claimed to have de… | 1.0 / finish / 5 / 2964 | 0.0 / llm_none / 6 / 2392 |
| Please tell me the aircraft's registration number that fits the following details as of 20… | 1.0 / finish / 3 / 2253 | 0.0 / finish / 5 / 2220 |
| A student of architecture who was born in the 19th Century but died in the 20th had a sibl… | 1.0 / finish / 3 / 2642 | 0.0 / llm_none / 11 / 2249 |
| There is a TV show episode in which one of the main characters undergoes a shift in their … | 1.0 / finish / 3 / 1999 | 0.0 / llm_none / 7 / 2315 |
| As of 2021, an artist with a bachelor's degree in biochemistry and a master's degree in bu… | 1.0 / finish / 2 / 1458 | 0.0 / llm_none / 8 / 2089 |
| A specific sports team was founded between 2010 and 2015, both years inclusive, by a speci… | 1.0 / finish / 7 / 6891 | 0.0 / llm_none / 10 / 5257 |
| This historic landmark was founded in the 15th century and underwent a name change followi… | 1.0 / finish / 7 / 4469 | 0.0 / llm_none / 3 / 4096 |
| I was born in the 20th century (outside the United States) and launched my music career wi… | 1.0 / finish / 4 / 2817 | 0.0 / llm_none / 3 / 3574 |
| What is the full name of a person known for investing in technology businesses? This indiv… | 1.0 / finish / 5 / 2507 | 0.0 / llm_none / 6 / 3317 |
| Please provide the month and year of birth for this individual. They served as the headmas… | 1.0 / finish / 3 / 3099 | 0.0 / llm_none / 7 / 2858 |
| There is a company that is said to produce savory sweets, plain, chocolate-filled, and fla… | 1.0 / finish / 3 / 4170 | 0.0 / finish / 2 / 1660 |
| The university was established between 2000 and 2003, inclusive. Prior to December 2023, t… | 1.0 / finish / 5 / 6387 | 0.0 / llm_none / 9 / 4702 |
| Who founded the company in the European nation, whose watch was worn by the first person t… | 1.0 / finish / 4 / 4476 | 0.0 / finish / 5 / 1329 |
| Provide the name of the person who was born in the 1940s. This individual was born in a co… | 1.0 / finish / 3 / 3908 | 0.0 / llm_none / 9 / 1457 |
| There is a book published about one of the deadliest serial killers in American history. T… | 1.0 / finish / 4 / 5107 | 0.0 / llm_none / 7 / 1509 |
**full_A_branch vs full_B_compact** on 150 paired tasks: both correct 29, only full_A_branch 26, only full_B_compact 7, both wrong 88

| task (question head) | full_A_branch score/stop/turns/gen | full_B_compact score/stop/turns/gen |
|---|---|---|
| Identify the company that meets the following criteria:  - An independent non-executive di… | 1.0 / finish / 4 / 2896 | 0.0 / finish / 2 / 3334 |
| A 2021 review was written for a historical strategy game that was released earlier in the … | 1.0 / finish / 6 / 2951 | 0.0 / budget_exhausted / 22 / 5012 |
| I’m thinking of two distinct blog posts, written by the same author, in the same year, aft… | 1.0 / finish / 5 / 1334 | 0.0 / budget_exhausted / 5 / 3593 |
| I am looking for the name of a group of companies that satisfy the following conditions: 1… | 1.0 / finish / 4 / 2602 | 0.0 / budget_exhausted / 8 / 4476 |
| I belong to a fantasy world and became aware of a prophecy involving twin keys, created by… | 1.0 / finish / 3 / 1328 | 0.0 / budget_exhausted / 5 / 4699 |
| A genus can be found in the Northern Hemisphere, largely in temperate latitudes, among oth… | 1.0 / finish / 5 / 1876 | 0.0 / budget_exhausted / 6 / 4076 |
| An individual was encouraged to use a different tool for their work during the year betwee… | 1.0 / finish / 3 / 1965 | 0.0 / finish / 5 / 3738 |
| I’m looking for the name and the release year of a TV series released before 2005 and afte… | 0.0 / finish / 4 / 4657 | 1.0 / finish / 2 / 3806 |
| I want you to find the name of the series that I am talking about. Here are various plot p… | 1.0 / finish / 3 / 1567 | 0.0 / budget_exhausted / 5 / 2758 |
| A pack of downloadable content was released for a strategy video game over three years aft… | 0.0 / finish / 4 / 2981 | 1.0 / finish / 7 / 5105 |
| There is a short film, produced in 2017, with a runtime of 19 minutes, that revolves aroun… | 1.0 / finish / 5 / 3594 | 0.0 / budget_exhausted / 7 / 4455 |
| In a late 20th-century game, my journey began when I emerged from a fictional world, voice… | 0.0 / finish / 5 / 2571 | 1.0 / finish / 10 / 1930 |
| In 2006, an EU-funded project with a budget of Є1.30 million was launched and completed wi… | 1.0 / finish / 4 / 3127 | 0.0 / budget_exhausted / 8 / 4754 |
| In the mid-2000s, a specific pet was welcomed by a nursing home at six months old. This pe… | 1.0 / finish / 4 / 3831 | 0.0 / finish / 5 / 1107 |
| As of December 2023, this individual: - This individual is a graduate with highest honors.… | 1.0 / finish / 3 / 2277 | 0.0 / budget_exhausted / 9 / 5442 |
| Between 2015 and 2022, a private hospital experienced a fatal power outage. Approximately … | 1.0 / finish / 4 / 1686 | 0.0 / budget_exhausted / 4 / 5976 |
| According to a 2021 article, a certain individual expressed their love for poetry and shar… | 1.0 / finish / 2 / 1469 | 0.0 / budget_exhausted / 5 / 3619 |
| As of 2023 there is an individual who had previously held the same cabinet-level position … | 1.0 / finish / 5 / 3309 | 0.0 / budget_exhausted / 4 / 2648 |
| I am looking for the title of a book first published in 1898 by an author born in the 1860… | 1.0 / finish / 5 / 3497 | 0.0 / budget_exhausted / 7 / 3808 |
| Give me the name of the scholar and associate professor that is affiliated with one of the… | 1.0 / finish / 4 / 3586 | 0.0 / budget_exhausted / 5 / 4684 |
| An actress who studied musical theatre and graduated in 2002 was cast in a soap opera crea… | 0.0 / finish / 5 / 2770 | 1.0 / finish / 4 / 5002 |
| A student of architecture who was born in the 19th Century but died in the 20th had a sibl… | 1.0 / finish / 3 / 2642 | 0.0 / budget_exhausted / 5 / 5512 |
| There is a TV show episode in which one of the main characters undergoes a shift in their … | 1.0 / finish / 3 / 1999 | 0.0 / budget_exhausted / 5 / 2959 |
| According to a biography published in a biographical dictionary in 1993, the subject of th… | 0.0 / finish / 3 / 2388 | 1.0 / finish / 6 / 4625 |
| This historic landmark was founded in the 15th century and underwent a name change followi… | 1.0 / finish / 7 / 4469 | 0.0 / budget_exhausted / 8 / 10384 |
| As of December 2022, I am looking for the name of a historical landmark that was erected i… | 0.0 / finish / 7 / 6216 | 1.0 / finish / 5 / 2007 |
| Identify the English name of structure originally constructed during the 16th century. It … | 1.0 / finish / 5 / 5931 | 0.0 / finish / 8 / 3831 |
| There is a company that is said to produce savory sweets, plain, chocolate-filled, and fla… | 1.0 / finish / 3 / 4170 | 0.0 / finish / 5 / 819 |
| Give the name of the game that was released exclusively between 2001 and 2007, in which th… | 1.0 / finish / 5 / 5086 | 0.0 / budget_exhausted / 8 / 5637 |
| The information I have about an artist is as follows: This artist was born in the early 19… | 0.0 / finish / 3 / 4613 | 1.0 / finish / 4 / 4460 |
| Who founded the company in the European nation, whose watch was worn by the first person t… | 1.0 / finish / 4 / 4476 | 0.0 / budget_exhausted / 2 / 7937 |
| A band was formed a year before a United States presidential election, and it has three fu… | 1.0 / finish / 4 / 4409 | 0.0 / budget_exhausted / 4 / 5772 |
| Provide the name of the person who was born in the 1940s. This individual was born in a co… | 1.0 / finish / 3 / 3908 | 0.0 / budget_exhausted / 9 / 8645 |
**full_A_branch vs OLD_branch_150** on 0 paired tasks: both correct 0, only full_A_branch 0, only OLD_branch_150 0, both wrong 0

| task (question head) | full_A_branch score/stop/turns/gen | OLD_branch_150 score/stop/turns/gen |
|---|---|---|
**full_A_nobranch vs full_B_compact** on 150 paired tasks: both correct 8, only full_A_nobranch 5, only full_B_compact 28, both wrong 109

| task (question head) | full_A_nobranch score/stop/turns/gen | full_B_compact score/stop/turns/gen |
|---|---|---|
| I'm looking for the name of a 19th-century magazine that had fewer than eight published vo… | 0.0 / llm_none / 3 / 1292 | 1.0 / finish / 4 / 1725 |
| In this MMA fight that took place before 2023, the statistics for the loser included 10 si… | 0.0 / llm_none / 8 / 667 | 1.0 / finish / 5 / 1127 |
| On the Blantyre Telegraph, there's an article published between 2018 and 2020, inclusive. … | 0.0 / finish / 6 / 1782 | 1.0 / finish / 12 / 4573 |
| Prior to 2010, four poems by the same writer were published in the same issue of a literar… | 0.0 / llm_none / 11 / 658 | 1.0 / finish / 6 / 1740 |
| A pack of downloadable content was released for a strategy video game over three years aft… | 0.0 / llm_none / 10 / 2952 | 1.0 / finish / 7 / 5105 |
| I’m looking for the name and the release year of a TV series released before 2005 and afte… | 0.0 / llm_none / 4 / 2906 | 1.0 / finish / 2 / 3806 |
| A cleric was murdered after 2005 but before 2020. The incident happened as they were about… | 0.0 / llm_none / 18 / 1052 | 1.0 / finish / 4 / 1046 |
| According to a biography published in a biographical dictionary in 1993, the subject of th… | 0.0 / finish / 4 / 4109 | 1.0 / finish / 6 / 4625 |
| The following details describe an individual:  An African artist Began their career by pai… | 0.0 / llm_none / 8 / 1671 | 1.0 / finish / 2 / 3048 |
| As of 2021, an artist with a bachelor's degree in biochemistry and a master's degree in bu… | 0.0 / llm_none / 8 / 2089 | 1.0 / finish / 5 / 741 |
| There is an insect that is loved and enjoyed by many people in African countries and is pa… | 0.0 / llm_none / 8 / 1569 | 1.0 / finish / 6 / 2770 |
| Between 2015 and 2022, a private hospital experienced a fatal power outage. Approximately … | 1.0 / finish / 4 / 3959 | 0.0 / budget_exhausted / 4 / 5976 |
| As of December 2023, what is the name of the movie based on the details below:  It was rel… | 0.0 / llm_none / 8 / 973 | 1.0 / finish / 6 / 3342 |
| As of December 2022, I am looking for the name of a historical landmark that was erected i… | 0.0 / llm_none / 3 / 1431 | 1.0 / finish / 5 / 2007 |
| I need help finding the name of the art exhibition I've heard about. My only details are a… | 0.0 / llm_none / 10 / 1898 | 1.0 / finish / 4 / 2566 |
| I am looking for a specialty food shop that has at least one location in both Mexico and C… | 0.0 / llm_none / 15 / 2272 | 1.0 / finish / 4 / 2408 |
| In a late 20th-century game, my journey began when I emerged from a fictional world, voice… | 0.0 / llm_none / 8 / 1974 | 1.0 / finish / 10 / 1930 |
| There is this band, and it is difficult to remember their name. Can you help?   - They are… | 0.0 / llm_none / 11 / 1867 | 1.0 / finish / 5 / 4772 |
| This building originated in the 12th century. It had a tower added in the 14th century. An… | 0.0 / llm_none / 9 / 2819 | 1.0 / finish / 5 / 2573 |
| There is a software developer who, sometime after 2010 but before 2023, claimed to have de… | 0.0 / llm_none / 6 / 2392 | 1.0 / finish / 5 / 1924 |
| Identify the English name of structure originally constructed during the 16th century. It … | 1.0 / finish / 3 / 2280 | 0.0 / finish / 8 / 3831 |
| An actress who studied musical theatre and graduated in 2002 was cast in a soap opera crea… | 0.0 / llm_none / 12 / 2149 | 1.0 / finish / 4 / 5002 |
| A specific sports team was founded between 2010 and 2015, both years inclusive, by a speci… | 0.0 / llm_none / 10 / 5257 | 1.0 / finish / 5 / 6741 |
| There is a book published about one of the deadliest serial killers in American history. T… | 0.0 / llm_none / 7 / 1509 | 1.0 / finish / 7 / 3351 |
| I was born in the 20th century (outside the United States) and launched my music career wi… | 0.0 / llm_none / 3 / 3574 | 1.0 / finish / 3 / 1700 |
| The university was established between 2000 and 2003, inclusive. Prior to December 2023, t… | 0.0 / llm_none / 9 / 4702 | 1.0 / finish / 10 / 8999 |
| A band was formed a year before a United States presidential election, and it has three fu… | 1.0 / finish / 8 / 6182 | 0.0 / budget_exhausted / 4 / 5772 |
| Please tell me the aircraft's registration number that fits the following details as of 20… | 0.0 / finish / 5 / 2220 | 1.0 / finish / 10 / 4162 |
| Give the name of the game that was released exclusively between 2001 and 2007, in which th… | 1.0 / finish / 6 / 3270 | 0.0 / budget_exhausted / 8 / 5637 |
| Identify a rare or uncommon fungi that typically appears in clusters after rainfall, chara… | 1.0 / finish / 5 / 5087 | 0.0 / budget_exhausted / 7 / 9728 |
| What is the full name of a person known for investing in technology businesses? This indiv… | 0.0 / llm_none / 6 / 3317 | 1.0 / finish / 9 / 2112 |
| The information I have about an artist is as follows: This artist was born in the early 19… | 0.0 / llm_none / 4 / 5315 | 1.0 / finish / 4 / 4460 |
| Please provide the month and year of birth for this individual. They served as the headmas… | 0.0 / llm_none / 7 / 2858 | 1.0 / finish / 3 / 3641 |
**full_A_nobranch vs OLD_branch_150** on 0 paired tasks: both correct 0, only full_A_nobranch 0, only OLD_branch_150 0, both wrong 0

| task (question head) | full_A_nobranch score/stop/turns/gen | OLD_branch_150 score/stop/turns/gen |
|---|---|---|
**full_B_compact vs OLD_branch_150** on 0 paired tasks: both correct 0, only full_B_compact 0, only OLD_branch_150 0, both wrong 0

| task (question head) | full_B_compact score/stop/turns/gen | OLD_branch_150 score/stop/turns/gen |
|---|---|---|
