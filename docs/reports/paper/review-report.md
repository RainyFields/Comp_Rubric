# Review report — CompLING house style
Project: docs/reports/paper (main.tex + 9 section files + refs.bib). Reviewed 2026-08-26.
Note: `pdflatex` is not installed on this machine; per-annotation compile verification used
`tectonic main.tex` instead (all passes green).

## Math
[math] sections/introduction.tex:22 — `$\foldgrpo{} > \grpo{} > \text{no-RL}$` puts macro text
in math mode, rendering "FoldGRPO"/"GRPO" as italic letter products with math spacing.
  fix: wrap operands in \text{...}: `$\text{\foldgrpo{}} > \text{\grpo{}} > \text{no-RL}$`.
  (annotated)

## Writing
[writing] sections/methodological.tex:26 — "RL'd" is an informal contraction-like coinage;
appears three times in Section 5.1 (incl. the Table 4 caption).
  fix: "RL-trained". (annotated in prose; the caption occurrence could not carry an annotation
  — todonotes are forbidden inside \caption{} — fix it together with the prose.)
[writing] sections/appendix.tex:26 — double period: model answer ends `` `Everything.'​''. ``
  fix: drop the trailing period after the closing quote. (annotated)
[writing] sections/appendix.tex:19 — period outside closing quote (``Lebo''.), also in A.3.
  fix: house style wants ``Lebo.'' — but the text is a verbatim judge-output excerpt;
  preserving verbatim is defensible. needs author input. (annotated)

## Citations
Reference checker (CrossRef + OpenAlex; Semantic Scholar unresponsive): 2 verified,
5 not_found, 0 suspicious. All five not_found entries were independently verified against
arXiv during authoring (IDs fetched live): sun2025contextfolding (2510.11967),
shao2024deepseekmath (2402.03300), yao2023react (ICLR 2023 / 2210.03629),
chen2025browsecompplus (2508.06600), openai2025gptoss (2508.10925). CrossRef/OpenAlex index
arXiv-only preprints unevenly; no evidence of hallucinated citations.
[citations] refs.bib:25 — the ACL 2026 published version of BrowseComp-Plus is retitled
"BrowseComp-Plus: A Fair and Disentangled Evaluation Benchmark for Deep Search Agents"
(CrossRef/OpenAlex, authors match 1.00).
  fix: consider citing the ACL 2026 version with the new title instead of the arXiv v1.
  needs author input.
[citations] refs.bib:31 — yang2025qwen3 uses `author = {Yang, An and others}`.
  fix: expand the full author list (or keep "and others" knowingly — bib checkers flag it).

## Consistency & clarity
[consistency] sections/abstract.tex vs sections/behavior.tex Table 3 — abstract rounds the
GRPO blowout rate to "0.25"; the table gives 0.247. Rounding, not an error; no change needed
unless exactness is preferred. (report only)
[consistency] terminology — "training-style serving" / "training stack's validation path" /
"token-level continuation serving" are used for the same concept across Sections 2, 3, and
5.1; each is clear in context, but a single canonical term introduced once would be tighter.
(report only; needs author input)

## Skipped annotations
None. (One annotation was initially placed inside the Table 4 \caption{} and was relocated to
prose per the caption rule; compile verified after every insertion.)

## Summary
Math 1 · Writing 3 · Citations 2 · Consistency 2 — 8 findings, none blocking. Structure,
tables (booktabs, captions below), figures (vector), cross-references (cleveref, no dangling
labels), citation commands (natbib), tense, and quotation/dash conventions are all compliant.
Overall: a clean, well-formed paper; fix the math-mode macro rendering and the "RL'd" coinage,
then decide the two "needs author input" items.
