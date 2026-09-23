# Per-task paired results — GRPO step 50, compaction OFF (A branch, A′ no-branch) vs ON (B), 150 tasks

Columns per arm: correct / finished / stop reason / compactions / branches / generated tokens / duplicate searches (of searches). Pattern = correctness A A′ B. Source: `work/full_*_rollouts.json` (captured runs abec53a5 / a2339ece / 7f8f4160). CSV with more columns: `paired_results.csv`.

| # | id | pattern | A: corr/fin/stop/br/gen/dup | A′: corr/fin/stop/gen/dup | B: corr/fin/stop/comp/gen/dup | question |
|---|---|---|---|---|---|---|
| 0 | 787 | 000 | 0 / 0 / window / 1 / 4874 / 0/5 | 0 / 0 / window / 2984 / 0/3 | 0 / 0 / budget / 3 / 5225 / 6/14 | Give me the full birth name of the artiste based on the following hint… |
| 1 | 801 | 001 | 0 / 1 / finish / 1 / 2770 / 0/3 | 0 / 0 / window / 2149 / 1/6 | 1 / 1 / finish / 3 / 5002 / 2/9 | An actress who studied musical theatre and graduated in 2002 was cast … |
| 2 | 810 | 100 | 1 / 1 / finish / 2 / 2642 / 7/14 | 0 / 0 / window / 2249 / 3/6 | 0 / 0 / budget / 3 / 5512 / 6/9 | A student of architecture who was born in the 19th Century but died in… |
| 3 | 815 | 100 | 1 / 1 / finish / 4 / 3497 / 3/9 | 0 / 0 / window / 1288 / 1/4 | 0 / 0 / budget / 3 / 3808 / 1/10 | I am looking for the title of a book first published in 1898 by an aut… |
| 4 | 833 | 000 | 0 / 1 / finish / 6 / 8484 / 8/16 | 0 / 0 / window / 6032 / 1/3 | 0 / 0 / budget / 3 / 7859 / 7/13 | The information below is about an individual who - is an alumnus of a … |
| 5 | 836 | 000 | 0 / 0 / window / 0 / 4290 / 0/4 | 0 / 0 / window / 4579 / 0/2 | 0 / 0 / budget / 3 / 9666 / 2/5 | What's the full name of the person who played the role of a Nurse in a… |
| 6 | 843 | 001 | 0 / 1 / finish / 3 / 2981 / 5/12 | 0 / 0 / window / 2952 / 0/6 | 1 / 1 / finish / 3 / 5105 / 2/6 | A pack of downloadable content was released for a strategy video game … |
| 7 | 863 | 111 | 1 / 1 / finish / 1 / 1689 / 0/5 | 1 / 1 / finish / 1664 / 0/1 | 1 / 1 / finish / 3 / 4860 / 6/12 | She holds an MBA and a Master’s degree, and as of 2023, she was a PhD … |
| 8 | 875 | 000 | 0 / 1 / finish / 8 / 6941 / 10/22 | 0 / 0 / window / 2682 / 1/5 | 0 / 0 / budget / 3 / 6754 / 1/5 | Two individuals from different industries share the same first name an… |
| 9 | 223 | 000 | 0 / 1 / finish / 6 / 5723 / 7/17 | 0 / 0 / window / 3372 / 2/5 | 0 / 0 / budget / 3 / 7584 / 6/15 | There is an author who as of 2023 was interested in gaming, had battle… |
| 10 | 226 | 000 | 0 / 1 / finish / 3 / 4592 / 2/7 | 0 / 0 / window / 1737 / 1/6 | 0 / 0 / budget / 3 / 4127 / 4/7 | As of December 2023, can you name the historical European landmark unv… |
| 11 | 235 | 100 | 1 / 1 / finish / 0 / 3309 / 0/2 | 0 / 0 / window / 1721 / 0/4 | 0 / 0 / budget / 3 / 2648 / 4/6 | As of 2023 there is an individual who had previously held the same cab… |
| 12 | 238 | 000 | 0 / 1 / finish / 4 / 6283 / 2/9 | 0 / 0 / window / 1985 / 0/4 | 0 / 0 / budget / 3 / 7212 / 5/11 | I'm looking for a character that appears in a game made before the rel… |
| 13 | 240 | 000 | 0 / 1 / finish / 1 / 4469 / 0/3 | 0 / 0 / window / 3654 / 4/7 | 0 / 1 / finish / 1 / 2977 / 0/4 | This individual interviewed one of the greatest athletes in history be… |
| 14 | 244 | 000 | 0 / 1 / finish / 3 / 3933 / 5/12 | 0 / 0 / window / 2330 / 0/4 | 0 / 0 / budget / 3 / 5442 / 4/9 | On April 5, 2022, a post with the title phrased as a question and ment… |
| 15 | 246 | 101 | 1 / 1 / finish / 1 / 2030 / 2/5 | 0 / 0 / window / 1292 / 0/3 | 1 / 1 / finish / 0 / 1725 / 0/1 | I'm looking for the name of a 19th-century magazine that had fewer tha… |
| 16 | 249 | 000 | 0 / 0 / window / 0 / 2323 / 1/6 | 0 / 0 / window / 2728 / 0/4 | 0 / 0 / budget / 3 / 7801 / 3/7 | An article detailing the history of a theatre that opened in 1930 was … |
| 17 | 254 | 000 | 0 / 1 / finish / 0 / 4977 / 0/3 | 0 / 0 / window / 5756 / 2/4 | 0 / 0 / budget / 3 / 9002 / 1/6 | What is the name of the band and their third full-length album, which … |
| 18 | 980 | 000 | 0 / 1 / finish / 2 / 4085 / 2/7 | 0 / 0 / window / 853 / 1/4 | 0 / 0 / budget / 3 / 6306 / 2/6 | Can you tell me the name of a movie which was released in year 2000's.… |
| 19 | 983 | 000 | 0 / 1 / finish / 6 / 7152 / 7/22 | 0 / 0 / window / 2043 / 1/4 | 0 / 0 / budget / 3 / 7126 / 4/7 | Three people wrote an article published between 2020 and 2023 listing … |
| 20 | 1002 | 000 | 0 / 1 / finish / 1 / 4385 / 0/4 | 0 / 0 / window / 2624 / 2/4 | 0 / 0 / budget / 3 / 4746 / 5/11 | There’s a man who used to work at a famous broadcasting network, befor… |
| 21 | 1015 | 000 | 0 / 1 / finish / 2 / 1946 / 0/5 | 0 / 0 / window / 1206 / 1/4 | 0 / 0 / budget / 3 / 5390 / 4/6 | As of December 2022, I am looking for the name of a museum that is nam… |
| 22 | 1016 | 010 | 0 / 0 / window / 0 / 3211 / 0/6 | 1 / 1 / finish / 5087 / 0/2 | 0 / 0 / budget / 3 / 9728 / 1/13 | Identify a rare or uncommon fungi that typically appears in clusters a… |
| 23 | 1025 | 111 | 1 / 1 / finish / 2 / 2004 / 1/5 | 1 / 1 / finish / 928 / 0/2 | 1 / 1 / finish / 1 / 4747 / 0/3 | There is a science-fiction novel that was published in 2010. It was pr… |
| 24 | 1028 | 000 | 0 / 1 / finish / 6 / 6614 / 9/21 | 0 / 0 / window / 4804 / 1/3 | 0 / 0 / budget / 3 / 6603 / 1/6 | Person A was born into a family of merchants in the 1900s and graduate… |
| 25 | 1036 | 000 | 0 / 1 / finish / 2 / 3146 / 3/6 | 0 / 0 / window / 698 / 3/5 | 0 / 0 / budget / 3 / 5759 / 6/10 | In an interview, this individual revealed that their love for music wa… |
| 26 | 1046 | 000 | 0 / 1 / finish / 1 / 4393 / 0/3 | 0 / 0 / window / 2491 / 1/3 | 0 / 0 / budget / 3 / 8465 / 4/12 | A specific company was founded in a specific year between 1990 and 201… |
| 27 | 1048 | 100 | 1 / 1 / finish / 3 / 3586 / 6/16 | 0 / 0 / window / 2460 / 1/3 | 0 / 0 / budget / 3 / 4684 / 1/8 | Give me the name of the scholar and associate professor that is affili… |
| 28 | 1052 | 000 | 0 / 1 / finish / 5 / 9479 / 3/12 | 0 / 0 / loop / 6144 / 0/0 | 0 / 0 / loop / 0 / 6144 / 0/0 | A film critic, who reviewed movies for over 40 years until their death… |
| 29 | 1055 | 000 | 0 / 1 / finish / 1 / 5421 / 1/3 | 0 / 0 / window / 3608 / 0/5 | 0 / 1 / finish / 1 / 6206 / 0/8 | A short story featuring small forest creatures was written by an autho… |
| 30 | 1057 | 000 | 0 / 0 / window / 0 / 853 / 3/8 | 0 / 0 / window / 2284 / 0/5 | 0 / 0 / budget / 3 / 5853 / 3/12 | In a botany blog post in 2020, an amateur botanist explains the use of… |
| 31 | 1061 | 000 | 0 / 1 / finish / 1 / 3796 / 0/4 | 0 / 0 / window / 4160 / 1/5 | 0 / 0 / budget / 3 / 6959 / 8/12 | In late 2020, a series of articles were published on a digital news pl… |
| 32 | 10 | 101 | 1 / 1 / finish / 1 / 1410 / 0/1 | 0 / 0 / window / 1671 / 1/4 | 1 / 1 / finish / 1 / 3048 / 0/3 | The following details describe an individual:  An African artist Began… |
| 33 | 478 | 101 | 1 / 1 / finish / 2 / 2352 / 4/8 | 0 / 1 / finish / 1782 / 0/3 | 1 / 1 / finish / 3 / 4573 / 10/15 | On the Blantyre Telegraph, there's an article published between 2018 a… |
| 34 | 495 | 100 | 1 / 1 / finish / 1 / 1469 / 0/2 | 0 / 0 / window / 2434 / 2/6 | 0 / 0 / budget / 3 / 3619 / 5/7 | According to a 2021 article, a certain individual expressed their love… |
| 35 | 500 | 110 | 1 / 1 / finish / 1 / 5931 / 0/1 | 1 / 1 / finish / 2280 / 0/1 | 0 / 1 / finish / 2 / 3831 / 3/7 | Identify the English name of structure originally constructed during t… |
| 36 | 506 | 111 | 1 / 1 / finish / 1 / 2361 / 0/2 | 1 / 1 / finish / 2550 / 0/3 | 1 / 1 / finish / 2 / 5236 / 1/6 | What is the name of this short film:  1. As of 2022, Its director is a… |
| 37 | 507 | 000 | 0 / 1 / finish / 3 / 5022 / 3/9 | 0 / 0 / window / 1683 / 4/9 | 0 / 0 / budget / 3 / 6050 / 6/13 | As of December 2023, name the band formed between 1960 and 1980, by a … |
| 38 | 513 | 000 | 0 / 0 / window / 0 / 5017 / 2/4 | 0 / 0 / window / 6460 / 1/3 | 0 / 0 / budget / 3 / 7733 / 5/6 | There are two different blog posts, both written between 2015 and Dece… |
| 39 | 517 | 000 | 0 / 1 / finish / 7 / 9474 / 1/7 | 0 / 0 / window / 4304 / 1/3 | 0 / 0 / budget / 3 / 9377 / 2/7 | Hey Chat, I'm trying to remember someone's popular name. This person w… |
| 40 | 519 | 000 | 0 / 1 / finish / 5 / 4915 / 7/14 | 0 / 0 / window / 2606 / 2/6 | 0 / 0 / budget / 3 / 4495 / 2/9 | Sometime before 1960, a couple met for the first time in London at the… |
| 41 | 544 | 000 | 0 / 1 / finish / 10 / 10368 / 17/29 | 0 / 0 / window / 3460 / 0/4 | 0 / 0 / budget / 3 / 8551 / 4/13 | Before 2023, an individual portrayed a village leader in a film writte… |
| 42 | 550 | 000 | 0 / 1 / finish / 5 / 5255 / 7/19 | 0 / 0 / window / 1904 / 0/3 | 0 / 0 / budget / 3 / 4365 / 3/6 | A sports owner from the 80s funded his team at the time from the money… |
| 43 | 562 | 001 | 0 / 1 / finish / 2 / 2388 / 1/5 | 0 / 1 / finish / 4109 / 0/3 | 1 / 1 / finish / 2 / 4625 / 1/5 | According to a biography published in a biographical dictionary in 199… |
| 44 | 257 | 000 | 0 / 1 / finish / 2 / 6215 / 0/4 | 0 / 0 / window / 4301 / 0/3 | 0 / 1 / finish / 1 / 7659 / 1/2 | In the 1990s, a graphic novel was published which was a dark tale abou… |
| 45 | 262 | 000 | 0 / 1 / finish / 4 / 4817 / 8/16 | 0 / 0 / window / 1903 / 1/4 | 0 / 0 / budget / 3 / 5310 / 6/10 | A well-known landmark in this university was donated by a class of stu… |
| 46 | 267 | 101 | 1 / 1 / finish / 5 / 5074 / 3/16 | 0 / 0 / window / 1898 / 4/7 | 1 / 1 / finish / 0 / 2566 / 0/2 | I need help finding the name of the art exhibition I've heard about. M… |
| 47 | 270 | 000 | 0 / 1 / finish / 4 / 6898 / 6/20 | 0 / 1 / finish / 5745 / 0/2 | 0 / 0 / loop / 0 / 6144 / 0/0 | There is a food dish that was discovered after the 16th century in a c… |
| 48 | 276 | 000 | 0 / 1 / finish / 1 / 3815 / 0/2 | 0 / 0 / window / 2388 / 2/4 | 0 / 0 / budget / 3 / 4031 / 2/7 | There was an entrepreneur who was known to exaggerate facts about a pa… |
| 49 | 286 | 000 | 0 / 1 / finish / 4 / 4474 / 5/14 | 0 / 0 / window / 4957 / 0/4 | 0 / 0 / budget / 3 / 5607 / 3/8 | A certain artiste born in England made their debut album between 2001 … |
| 50 | 287 | 000 | 0 / 0 / window / 0 / 6150 / 0/3 | 0 / 0 / window / 5180 / 2/4 | 0 / 1 / finish / 3 / 7316 / 4/9 | Please tell me the date when the author of the book described below wa… |
| 51 | 289 | 001 | 0 / 1 / finish / 3 / 4657 / 3/9 | 0 / 0 / window / 2906 / 0/4 | 1 / 1 / finish / 1 / 3806 / 0/6 | I’m looking for the name and the release year of a TV series released … |
| 52 | 293 | 000 | 0 / 1 / finish / 1 / 3107 / 1/5 | 0 / 0 / window / 2048 / 0/6 | 0 / 1 / finish / 2 / 7059 / 0/4 | I'm trying to recall the name of a movie released before Dec 31, 2023,… |
| 53 | 297 | 000 | 0 / 1 / finish / 7 / 6886 / 10/21 | 0 / 0 / window / 2409 / 0/4 | 0 / 0 / budget / 3 / 7402 / 5/9 | There was a fictional character who was a godfather in a movie. The ac… |
| 54 | 304 | 000 | 0 / 1 / finish / 8 / 8725 / 8/22 | 0 / 0 / window / 1960 / 3/6 | 0 / 1 / finish / 1 / 5889 / 0/3 | The artist’s father was a Protestant minister. They had a brief marria… |
| 55 | 351 | 000 | 0 / 0 / window / 4 / 5511 / 14/23 | 0 / 0 / window / 4096 / 0/5 | 0 / 0 / budget / 3 / 5051 / 2/8 | Given the following hints that are all true as of 2023, there was an i… |
| 56 | 354 | 000 | 0 / 0 / window / 0 / 3648 / 2/8 | 0 / 0 / window / 1119 / 3/8 | 0 / 1 / finish / 2 / 3830 / 4/9 | It was stated in a 2023 article that a certain individual ventured int… |
| 57 | 362 | 000 | 0 / 1 / finish / 7 / 7238 / 7/20 | 0 / 0 / window / 6114 / 1/3 | 0 / 0 / budget / 3 / 6378 / 1/8 | I'm looking for the name of a cartoon based on the following details, … |
| 58 | 41 | 111 | 1 / 1 / finish / 1 / 1418 / 0/2 | 1 / 1 / finish / 4316 / 0/2 | 1 / 1 / finish / 3 / 3277 / 3/5 | Born in the 1970s, this three-time World Half-Marathon champion, who w… |
| 59 | 46 | 101 | 1 / 1 / finish / 0 / 2507 / 0/3 | 0 / 0 / window / 3317 / 0/4 | 1 / 1 / finish / 1 / 2112 / 1/6 | What is the full name of a person known for investing in technology bu… |
| 60 | 52 | 100 | 1 / 1 / finish / 2 / 2896 / 1/7 | 0 / 0 / window / 2494 / 0/2 | 0 / 1 / finish / 1 / 3334 / 0/4 | Identify the company that meets the following criteria:  - An independ… |
| 61 | 54 | 000 | 0 / 1 / finish / 5 / 6699 / 3/13 | 0 / 1 / finish / 4456 / 1/3 | 0 / 0 / budget / 3 / 5849 / 4/12 | What is the English title of the novel that was first published in Fre… |
| 62 | 59 | 000 | 0 / 0 / window / 1 / 2412 / 1/7 | 0 / 0 / window / 1710 / 1/3 | 0 / 1 / finish / 1 / 4684 / 0/3 | Seven people are cited as authoring an article written about a clinica… |
| 63 | 68 | 000 | 0 / 1 / finish / 0 / 2552 / 0/2 | 0 / 0 / window / 1020 / 0/1 | 0 / 1 / finish / 0 / 2275 / 0/2 | In a tribute published in the early 2010s in Southern Africa, an accom… |
| 64 | 72 | 101 | 1 / 1 / finish / 1 / 2059 / 0/2 | 0 / 0 / window / 2272 / 0/8 | 1 / 1 / finish / 0 / 2408 / 0/3 | I am looking for a specialty food shop that has at least one location … |
| 65 | 569 | 100 | 1 / 1 / finish / 3 / 3831 / 4/8 | 0 / 0 / window / 951 / 1/6 | 0 / 1 / finish / 0 / 1107 / 0/1 | In the mid-2000s, a specific pet was welcomed by a nursing home at six… |
| 66 | 576 | 100 | 1 / 1 / finish / 0 / 4469 / 0/3 | 0 / 0 / window / 4096 / 1/5 | 0 / 0 / budget / 3 / 10384 / 5/8 | This historic landmark was founded in the 15th century and underwent a… |
| 67 | 580 | 100 | 1 / 1 / finish / 1 / 1567 / 0/3 | 0 / 0 / window / 754 / 0/5 | 0 / 0 / budget / 3 / 2758 / 2/6 | I want you to find the name of the series that I am talking about. Her… |
| 68 | 581 | 100 | 1 / 1 / finish / 1 / 4170 / 0/2 | 0 / 1 / finish / 1660 / 0/2 | 0 / 1 / finish / 0 / 819 / 0/2 | There is a company that is said to produce savory sweets, plain, choco… |
| 69 | 582 | 000 | 0 / 1 / finish / 8 / 7516 / 14/26 | 0 / 0 / window / 2410 / 4/6 | 0 / 0 / budget / 3 / 7456 / 5/13 | Researcher and author [Person A], born in 1930,  was the editorial suc… |
| 70 | 587 | 000 | 0 / 1 / finish / 0 / 3829 / 0/2 | 0 / 1 / finish / 4800 / 0/2 | 0 / 1 / finish / 0 / 3777 / 0/2 | A Pokemon was used by a competitor in a Pokemon VGC tournament between… |
| 71 | 591 | 101 | 1 / 1 / finish / 1 / 6387 / 1/6 | 0 / 0 / window / 4702 / 0/6 | 1 / 1 / finish / 1 / 8999 / 2/4 | The university was established between 2000 and 2003, inclusive. Prior… |
| 72 | 602 | 110 | 1 / 1 / finish / 0 / 1686 / 0/2 | 1 / 1 / finish / 3959 / 0/3 | 0 / 0 / budget / 3 / 5976 / 3/8 | Between 2015 and 2022, a private hospital experienced a fatal power ou… |
| 73 | 625 | 000 | 0 / 1 / finish / 5 / 6202 / 2/11 | 0 / 0 / window / 770 / 1/3 | 0 / 0 / budget / 3 / 4207 / 3/7 | There's a Japanese anime that was first released in the USA between 20… |
| 74 | 628 | 100 | 1 / 1 / finish / 4 / 3594 / 1/7 | 0 / 0 / window / 1965 / 0/7 | 0 / 0 / budget / 3 / 4455 / 11/16 | There is a short film, produced in 2017, with a runtime of 19 minutes,… |
| 75 | 629 | 001 | 0 / 1 / finish / 6 / 6216 / 12/23 | 0 / 0 / window / 1431 / 1/6 | 1 / 1 / finish / 0 / 2007 / 0/5 | As of December 2022, I am looking for the name of a historical landmar… |
| 76 | 644 | 101 | 1 / 1 / finish / 0 / 1926 / 0/2 | 0 / 0 / window / 667 / 1/4 | 1 / 1 / finish / 0 / 1127 / 0/2 | In this MMA fight that took place before 2023, the statistics for the … |
| 77 | 645 | 100 | 1 / 1 / finish / 0 / 1876 / 0/3 | 0 / 0 / window / 1961 / 3/6 | 0 / 0 / budget / 3 / 4076 / 2/5 | A genus can be found in the Northern Hemisphere, largely in temperate … |
| 78 | 661 | 100 | 1 / 1 / finish / 1 / 1965 / 0/2 | 0 / 1 / finish / 5486 / 0/2 | 0 / 1 / finish / 1 / 3738 / 0/4 | An individual was encouraged to use a different tool for their work du… |
| 79 | 1176 | 000 | 0 / 1 / finish / 2 / 5424 / 1/6 | 0 / 0 / window / 2119 / 1/5 | 0 / 1 / finish / 1 / 7248 / 0/8 | Please provide the name of the film with these characteristics:  - Sta… |
| 80 | 1182 | 000 | 0 / 1 / finish / 10 / 10683 / 14/37 | 0 / 0 / window / 2048 / 0/5 | 0 / 1 / finish / 1 / 5728 / 1/3 | This individual was born before 1951 and was the youngest of three sib… |
| 81 | 1188 | 000 | 0 / 1 / finish / 3 / 4087 / 1/10 | 0 / 0 / window / 1989 / 0/4 | 0 / 0 / budget / 3 / 6978 / 1/6 | A few years ago, I watched a TV series that I absolutely loved, but I … |
| 82 | 1195 | 101 | 1 / 1 / finish / 1 / 2817 / 0/2 | 0 / 0 / window / 3574 / 0/1 | 1 / 1 / finish / 0 / 1700 / 0/1 | I was born in the 20th century (outside the United States) and launche… |
| 83 | 1204 | 001 | 0 / 1 / finish / 0 / 2571 / 0/3 | 0 / 0 / window / 1974 / 1/3 | 1 / 1 / finish / 1 / 1930 / 1/5 | In a late 20th-century game, my journey began when I emerged from a fi… |
| 84 | 1209 | 000 | 0 / 1 / finish / 1 / 4122 / 1/4 | 0 / 0 / window / 2722 / 0/3 | 0 / 0 / budget / 3 / 5182 / 7/10 | This series, released prior to 2015, is known by four other titles. Th… |
| 85 | 1221 | 100 | 1 / 1 / finish / 1 / 1328 / 0/1 | 0 / 0 / window / 1514 / 1/3 | 0 / 0 / budget / 3 / 4699 / 1/6 | I belong to a fantasy world and became aware of a prophecy involving t… |
| 86 | 1222 | 000 | 0 / 1 / finish / 1 / 4311 / 0/2 | 0 / 0 / window / 1563 / 3/5 | 0 / 1 / finish / 3 / 4275 / 3/10 | Identify the title of a research publication published before June 202… |
| 87 | 1228 | 000 | 0 / 1 / finish / 7 / 9225 / 5/17 | 0 / 0 / window / 5076 / 0/3 | 0 / 1 / finish / 3 / 4774 / 2/6 | Can you identify a movie based on the following details:  •  This movi… |
| 88 | 1231 | 000 | 0 / 0 / window / 0 / 2371 / 0/5 | 0 / 0 / window / 2177 / 0/4 | 0 / 0 / budget / 3 / 4252 / 3/11 | As of 2023, identify this movie theater, built and owned by a local in… |
| 89 | 1236 | 000 | 0 / 1 / finish / 3 / 8810 / 5/8 | 0 / 0 / window / 6063 / 0/3 | 0 / 0 / budget / 3 / 8031 / 6/10 | This series was released before 2018, with each episode running for ov… |
| 90 | 1246 | 110 | 1 / 1 / finish / 2 / 5086 / 0/3 | 1 / 1 / finish / 3270 / 0/3 | 0 / 0 / budget / 3 / 5637 / 7/14 | Give the name of the game that was released exclusively between 2001 a… |
| 91 | 1263 | 100 | 1 / 1 / finish / 1 / 2602 / 0/2 | 0 / 0 / window / 1653 / 1/5 | 0 / 0 / budget / 3 / 4476 / 6/12 | I am looking for the name of a group of companies that satisfy the fol… |
| 92 | 1266 | 000 | 0 / 1 / finish / 1 / 8628 / 0/4 | 0 / 0 / window / 5425 / 3/5 | 0 / 0 / budget / 3 / 11060 / 3/5 | This foreign individual was born to a businessman in the late 20th cen… |
| 93 | 882 | 100 | 1 / 1 / finish / 0 / 1334 / 0/2 | 0 / 0 / window / 737 / 0/3 | 0 / 0 / budget / 3 / 3593 / 2/6 | I’m thinking of two distinct blog posts, written by the same author, i… |
| 94 | 885 | 001 | 0 / 1 / finish / 1 / 4613 / 0/3 | 0 / 0 / window / 5315 / 0/2 | 1 / 1 / finish / 0 / 4460 / 0/1 | The information I have about an artist is as follows: This artist was … |
| 95 | 886 | 100 | 1 / 1 / finish / 1 / 1999 / 0/1 | 0 / 0 / window / 2315 / 1/4 | 0 / 0 / budget / 3 / 2959 / 3/6 | There is a TV show episode in which one of the main characters undergo… |
| 96 | 897 | 000 | 0 / 0 / window / 2 / 3349 / 1/5 | 0 / 0 / window / 3404 / 0/5 | 0 / 1 / finish / 0 / 2723 / 0/2 | An African political leader and doctor was born in the late 1940s. In … |
| 97 | 904 | 000 | 0 / 1 / finish / 9 / 9540 / 10/30 | 0 / 0 / window / 3029 / 1/3 | 0 / 0 / budget / 3 / 3343 / 0/8 | There was a movie that came before 2010 directed by a person who was b… |
| 98 | 906 | 100 | 1 / 1 / finish / 1 / 4476 / 0/1 | 0 / 1 / finish / 1329 / 0/3 | 0 / 0 / budget / 3 / 7937 / 0/14 | Who founded the company in the European nation, whose watch was worn b… |
| 99 | 910 | 101 | 1 / 1 / finish / 3 / 2585 / 0/3 | 0 / 0 / window / 1052 / 1/3 | 1 / 1 / finish / 0 / 1046 / 0/2 | A cleric was murdered after 2005 but before 2020. The incident happene… |
| 100 | 925 | 111 | 1 / 1 / finish / 1 / 1965 / 0/1 | 1 / 1 / finish / 1593 / 0/1 | 1 / 1 / finish / 0 / 1276 / 0/1 | A very influential figure in the formation of a well-known club was fa… |
| 101 | 926 | 000 | 0 / 0 / window / 0 / 5532 / 0/4 | 0 / 0 / window / 4861 / 1/4 | 0 / 0 / budget / 3 / 9033 / 1/6 | I want the full birth name of this person. This person was born in the… |
| 102 | 946 | 110 | 1 / 1 / finish / 0 / 4409 / 0/2 | 1 / 1 / finish / 6182 / 1/3 | 0 / 0 / budget / 3 / 5772 / 2/6 | A band was formed a year before a United States presidential election,… |
| 103 | 966 | 101 | 1 / 1 / finish / 1 / 3035 / 0/1 | 0 / 0 / window / 2819 / 2/5 | 1 / 1 / finish / 0 / 2573 / 0/2 | This building originated in the 12th century. It had a tower added in … |
| 104 | 968 | 101 | 1 / 1 / finish / 2 / 1742 / 0/2 | 0 / 0 / window / 1569 / 0/1 | 1 / 1 / finish / 1 / 2770 / 0/2 | There is an insect that is loved and enjoyed by many people in African… |
| 105 | 976 | 101 | 1 / 1 / finish / 2 / 3099 / 0/5 | 0 / 0 / window / 2858 / 0/7 | 1 / 1 / finish / 0 / 3641 / 0/1 | Please provide the month and year of birth for this individual. They s… |
| 106 | 673 | 000 | 0 / 1 / finish / 5 / 4616 / 8/18 | 0 / 0 / window / 3327 / 0/4 | 0 / 0 / budget / 3 / 3910 / 11/14 | After the death of which poetess, her recordings of large number of po… |
| 107 | 714 | 000 | 0 / 1 / finish / 6 / 7628 / 6/18 | 0 / 0 / window / 3609 / 0/6 | 0 / 1 / finish / 1 / 4672 / 1/6 | Can you identify the last name of the author of a thesis that focused … |
| 108 | 718 | 000 | 0 / 1 / finish / 7 / 9623 / 6/17 | 0 / 0 / window / 4072 / 3/6 | 0 / 0 / budget / 3 / 5649 / 2/8 | A writer and producer was born in the late 1920s under the zodiac sign… |
| 109 | 722 | 100 | 1 / 1 / finish / 1 / 2277 / 0/1 | 0 / 0 / window / 1139 / 1/2 | 0 / 0 / budget / 3 / 5442 / 2/6 | As of December 2023, this individual: - This individual is a graduate … |
| 110 | 724 | 000 | 0 / 1 / finish / 7 / 6699 / 7/17 | 0 / 0 / window / 4374 / 1/3 | 0 / 1 / finish / 1 / 5757 / 1/3 | Provide the full name (do not include any academic or professional tit… |
| 111 | 728 | 100 | 1 / 1 / finish / 0 / 2951 / 0/3 | 0 / 0 / window / 1602 / 4/7 | 0 / 0 / budget / 3 / 5012 / 8/17 | A 2021 review was written for a historical strategy game that was rele… |
| 112 | 731 | 000 | 0 / 0 / window / 0 / 4451 / 1/4 | 0 / 0 / window / 3518 / 0/3 | 0 / 1 / finish / 0 / 3908 / 0/2 | A species was described and named for the first time in the 1780s by a… |
| 113 | 734 | 000 | 0 / 1 / finish / 3 / 5423 / 2/10 | 0 / 0 / window / 4694 / 2/7 | 0 / 0 / budget / 3 / 8700 / 2/5 | An actress born in a city whose flag bears a legendary creature was se… |
| 114 | 735 | 111 | 1 / 1 / finish / 1 / 2224 / 0/1 | 1 / 1 / finish / 962 / 0/1 | 1 / 1 / finish / 0 / 1069 / 0/1 | Several individuals in their pre-teen and teenage years were arrested … |
| 115 | 738 | 000 | 0 / 0 / window / 0 / 5777 / 0/4 | 0 / 0 / window / 1132 / 0/4 | 0 / 0 / budget / 3 / 6820 / 4/10 | A volleyball game took place after 2010 but before 2023, where the win… |
| 116 | 741 | 000 | 0 / 0 / window / 0 / 2743 / 0/4 | 0 / 0 / window / 2349 / 4/7 | 0 / 0 / budget / 3 / 4191 / 1/7 | A specific person, who will be referred to as person 1 from now on, pa… |
| 117 | 753 | 101 | 1 / 1 / finish / 4 / 6891 / 2/7 | 0 / 0 / window / 5257 / 1/4 | 1 / 1 / finish / 1 / 6741 / 0/2 | A specific sports team was founded between 2010 and 2015, both years i… |
| 118 | 768 | 101 | 1 / 1 / finish / 4 / 3962 / 0/7 | 0 / 0 / window / 1867 / 4/7 | 1 / 1 / finish / 2 / 4772 / 0/3 | There is this band, and it is difficult to remember their name. Can yo… |
| 119 | 178 | 100 | 1 / 1 / finish / 1 / 3127 / 3/6 | 0 / 0 / window / 895 / 4/6 | 0 / 0 / budget / 3 / 4754 / 9/14 | In 2006, an EU-funded project with a budget of Є1.30 million was launc… |
| 120 | 192 | 000 | 0 / 1 / finish / 4 / 5917 / 6/13 | 0 / 0 / window / 3164 / 0/6 | 0 / 1 / finish / 2 / 9697 / 0/6 | There was an artist who passed away after 2010 but before 2023. Some o… |
| 121 | 196 | 000 | 0 / 1 / finish / 4 / 4169 / 10/15 | 0 / 0 / window / 2575 / 0/4 | 0 / 0 / budget / 3 / 6008 / 8/15 | Hey Chat, I'm trying to remember someone's stage name. Can you tell me… |
| 122 | 203 | 000 | 0 / 1 / finish / 1 / 1865 / 0/5 | 0 / 0 / window / 2084 / 0/5 | 0 / 0 / budget / 3 / 3232 / 8/13 | A publication, prior to 2023, featured accounts of several unique indi… |
| 123 | 205 | 000 | 0 / 1 / finish / 1 / 2155 / 0/1 | 0 / 0 / window / 2639 / 0/5 | 0 / 0 / budget / 3 / 5187 / 9/11 | A new school was founded in the '90s by combining a girls' and boys' s… |
| 124 | 122 | 000 | 0 / 0 / window / 1 / 2128 / 4/6 | 0 / 0 / window / 1886 / 2/4 | 0 / 1 / finish / 1 / 5114 / 0/4 | I'm looking for the year of death of a person who was born between 194… |
| 125 | 128 | 000 | 0 / 1 / finish / 0 / 5685 / 0/2 | 0 / 0 / window / 3531 / 1/4 | 0 / 1 / finish / 2 / 6421 / 1/7 | An article was published in November of 2019, by a media company found… |
| 126 | 138 | 000 | 0 / 1 / finish / 4 / 8911 / 3/11 | 0 / 0 / window / 7460 / 0/3 | 0 / 0 / budget / 3 / 8184 / 4/6 | The was a movie released before 2022 that fits in the below criteria: … |
| 127 | 1077 | 000 | 0 / 1 / finish / 0 / 2127 / 0/2 | 0 / 0 / window / 1518 / 1/3 | 0 / 0 / budget / 3 / 5744 / 1/7 | An author wrote the first part in a two part article about depression … |
| 128 | 1095 | 100 | 1 / 1 / finish / 1 / 3908 / 0/1 | 0 / 0 / window / 1457 / 0/1 | 0 / 0 / budget / 3 / 8645 / 10/13 | Provide the name of the person who was born in the 1940s. This individ… |
| 129 | 1101 | 111 | 1 / 1 / finish / 2 / 2451 / 5/7 | 1 / 1 / finish / 1264 / 0/1 | 1 / 1 / finish / 0 / 1444 / 0/1 | An ocean conservation organization brought attention to a global crime… |
| 130 | 1108 | 101 | 1 / 1 / finish / 2 / 2253 / 0/2 | 0 / 1 / finish / 2220 / 0/3 | 1 / 1 / finish / 1 / 4162 / 1/5 | Please tell me the aircraft's registration number that fits the follow… |
| 131 | 1124 | 000 | 0 / 1 / finish / 2 / 4016 / 0/7 | 0 / 0 / window / 2308 / 0/5 | 0 / 0 / budget / 3 / 3862 / 2/8 | There is a band formed in the 70s that sang in several languages. The … |
| 132 | 1148 | 101 | 1 / 1 / finish / 1 / 1872 / 0/2 | 0 / 0 / window / 658 / 2/4 | 1 / 1 / finish / 0 / 1740 / 0/3 | Prior to 2010, four poems by the same writer were published in the sam… |
| 133 | 1150 | 000 | 0 / 1 / finish / 5 / 4947 / 6/14 | 0 / 0 / window / 672 / 2/3 | 0 / 0 / budget / 3 / 6280 / 0/7 | In a blog post submitted online sometime after 2010 and before 2015, t… |
| 134 | 1161 | 000 | 0 / 1 / finish / 0 / 4155 / 0/3 | 0 / 1 / finish / 3218 / 0/3 | 0 / 0 / budget / 3 / 7481 / 11/13 | Can you tell me the first and last name of this retired athlete who me… |
| 135 | 89 | 101 | 1 / 1 / finish / 2 / 5107 / 3/9 | 0 / 0 / window / 1509 / 2/4 | 1 / 1 / finish / 2 / 3351 / 0/6 | There is a book published about one of the deadliest serial killers in… |
| 136 | 96 | 000 | 0 / 1 / finish / 3 / 2744 / 3/10 | 0 / 0 / window / 1556 / 2/4 | 0 / 0 / budget / 3 / 3096 / 1/7 | According to a bulletin published in 2012, a syndicate of Scots establ… |
| 137 | 97 | 101 | 1 / 1 / finish / 1 / 1458 / 0/4 | 0 / 0 / window / 2089 / 2/4 | 1 / 1 / finish / 0 / 741 / 0/2 | As of 2021, an artist with a bachelor's degree in biochemistry and a m… |
| 138 | 109 | 000 | 0 / 1 / finish / 0 / 6613 / 0/3 | 0 / 0 / window / 7387 / 2/4 | 0 / 0 / budget / 3 / 9734 / 4/9 | Can you tell me the first and last birth name of this person based on … |
| 139 | 376 | 000 | 0 / 1 / finish / 5 / 7346 / 1/16 | 0 / 0 / window / 4986 / 0/4 | 0 / 0 / budget / 3 / 6538 / 3/7 | There is a country that has been colonized and had different independe… |
| 140 | 397 | 111 | 1 / 1 / finish / 2 / 2530 / 0/2 | 1 / 1 / finish / 1533 / 0/1 | 1 / 1 / finish / 0 / 2811 / 0/2 | A letter originated from a specific Air Force Base, which was also the… |
| 141 | 403 | 000 | 0 / 1 / finish / 0 / 3062 / 0/2 | 0 / 0 / window / 3127 / 1/4 | 0 / 1 / finish / 0 / 3339 / 0/2 | I'm looking for the name of the place where a person was hospitalized … |
| 142 | 408 | 101 | 1 / 1 / finish / 3 / 2964 / 0/7 | 0 / 0 / window / 2392 / 0/5 | 1 / 1 / finish / 0 / 1924 / 0/2 | There is a software developer who, sometime after 2010 but before 2023… |
| 143 | 417 | 000 | 0 / 1 / finish / 1 / 1937 / 0/3 | 0 / 0 / window / 601 / 1/4 | 0 / 0 / budget / 3 / 3868 / 3/14 | There is a person who used an animal-related nickname on the internet … |
| 144 | 418 | 000 | 0 / 1 / finish / 1 / 2732 / 0/3 | 0 / 0 / window / 3020 / 1/5 | 0 / 0 / budget / 3 / 5126 / 3/4 | A beauty company unveiled in the 2000s was founded by someone who had … |
| 145 | 422 | 000 | 0 / 1 / finish / 2 / 4760 / 1/5 | 0 / 0 / window / 4915 / 0/4 | 0 / 0 / budget / 3 / 5649 / 10/14 | A paper was published well into the 20th century, and by December 2023… |
| 146 | 438 | 000 | 0 / 1 / finish / 2 / 2711 / 4/6 | 0 / 1 / finish / 2889 / 0/4 | 0 / 1 / finish / 0 / 2878 / 0/2 | In 2014, a couple celebrated their wedding and worked with several ven… |
| 147 | 446 | 000 | 0 / 0 / window / 0 / 4232 / 3/7 | 0 / 0 / window / 5353 / 0/1 | 0 / 0 / budget / 3 / 6775 / 12/15 | What's the name of a 12-time best-selling author whose first novel was… |
| 148 | 165 | 101 | 1 / 1 / finish / 0 / 1927 / 0/3 | 0 / 0 / window / 973 / 2/4 | 1 / 1 / finish / 1 / 3342 / 0/2 | As of December 2023, what is the name of the movie based on the detail… |
| 149 | 176 | 000 | 0 / 1 / finish / 4 / 4349 / 7/17 | 0 / 0 / window / 3101 / 0/4 | 0 / 1 / finish / 0 / 6188 / 0/2 | A Harvard award-winning author wrote an article less than 5 years befo… |

## Pattern counts (A A′ B)

| pattern | tasks | diagnostic value |
|---|---|---|
| 000 | 87 | unsolved everywhere |
| 100 | 22 | branch-only success |
| 101 | 21 | branch + compaction, single window fails |
| 111 | 8 | solved everywhere |
| 001 | 7 | compaction-only success |
| 110 | 4 | OFF both, ON fails |
| 010 | 1 | single-window-only success |
