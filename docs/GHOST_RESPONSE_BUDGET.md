# Complete factual contrasts and whole-reply budgets

V13 identified two application failures distinct from source entailment. Brief
factual mode always kept only the first selected claim, so the Bessedovsky
verification limit survived but the separately selected reported-information
claim disappeared. In a short mixed reply, the opinion paragraph received its
own length target after a forty-word factual block, label and citations; the
whole reply reached 97 words for a request of about fifty.

`plan_response_budget` now packs complete selected factual statements in priority
order. Brief replies have a default sixty-word/two-sentence target; explicit
word targets override the default. A conservative sentence estimate is used,
not a language parser. A second card that fits is preserved; omitted IDs are
logged separately. At least the first selected factual card stays whole even
if its qualification makes the target impossible. The source text is never
cut, rewritten or stripped of uncertainty to fit a budget. This packing does
not establish that the model selected the most relevant claims or ordered them
well. Semantic relevance and necessary contrasts still require review.

For normal factual/mixed/view tasks, the app plans the exact canonical factual
display before drafting. The draft sees that display and the remaining word
allowance after accounting for facts, interpretation label and citation token.
The residual allowance overrides paragraph-only ranges. The auditor receives
the actual complete candidate as before. Exact quotation requests keep their
original structured-anchor path. No extra model call is added.

The final `response_budget` logs the requested target, planned factual words,
interpretation overhead, draft allowance, omitted factual IDs, an impossible-
minimum flag, actual displayed words and over-target status. A target is not a
hard guarantee that the model will comply. Overlong opinions are not silently
truncated, and a short refusal is not counted as a successful answer. Requests
that cannot fit a complete reviewed claim are reported in diagnostics rather
than satisfied by removing the claim's qualification. All transient planning
fields stay out of durable fact memory.

Local regressions cover the observed Bessedovsky contrast, the ILO two-sentence
finding, an exact fifty-word synthetic mixed display including labels/citations,
and an impossible ten-word target that leaves the full qualified claim intact.
Synthetic approval fixtures test application mechanics, not semantic quality.
Forty-nine focused tests pass. The broader suite retains its two documented
historical snapshot errors; historical locks are not rewritten.

V14 (`config/ghost_rendering_regression_v1.json`) preserves V13's exact ten
questions, profile, source packets and reviewed ledger. Same pinned untouched
base and generation policy; ordinary retrieval, not a fixed-context override.
It also includes the post-V13 task-enum recovery. Some later conversation
histories may change because actual replies change. This is a combined
application regression, not a component ablation, new semantic validation or
independent gold. At most thirty evaluation calls plus one memory probe on the
same private notebook under standing approval; no training/adapters/paid compute.

Review actual Bessedovsky contrast coverage, dated ILO scope, final total words,
draft compliance and honest diagnostics. Continue checking source-inference
errors and unnecessary refusals; length compliance alone is not usefulness.
Do not tune political judgments to the coding agent's preferences. Frozen V13
outputs remain unchanged. Independent source/conversation review is still the
next quality evidence described in [the review plan](GHOST_NEXT_REVIEW.md).


## Verified V14 result — 26 October 2026

Private saved version 16 completed ten replies/23 generations, with zero input
budget rejection or output truncation and an 11.32 GiB probe peak. Frozen replay,
all token reconstruction and archived artifact hashes pass. Control profile,
collection, ledger and questions match V13 exactly.

Bessedovsky now retains both complete cards in 42 words instead of twenty.
The dated ILO finding stays two sentences/33 words and the study-only limit
stays two sentences/40 words. The post-V13 enum normalization now also has a
GPU result: valid ILO evidence renders instead of blocking.

The budget instruction is insufficient. The final display has a forty-word
fact block, four words of label/citation overhead and a six-word draft allowance.
The normalized opinion is forty words, giving 84 total instead of about fifty
requested (previously 97). `over_target` correctly flags it. This is measured
instrumentation and partial shortening, not a solved word limit. Do not claim
that passing synthetic fixtures establishes model compliance.

Semantic problems remain separate: the deeper view is longer but repetitive,
and an unsupported assertion about every previous technological revolution is
approved. Preserve political preferences as permissible new views while
checking historical/descriptive claims. Detailed local review and a neutral
worksheet are under `evaluation/runs/ghost_smoke_v14/`. Independent review has
not happened. Future brevity designs may need reviewed compact fact variants
or a separately evaluated bounded rewrite workflow; neither is implemented.
No more prompt tuning on these inspected cases or training is queued.


The subsequent V18 [compact-form trial](GHOST_BRIEF_FACT_FORMS.md) adds reviewed
short alternatives without cutting source qualifications. It achieves the
requested one-sentence limit and retains complete assumptions in the tested
opening. V14 remains an immutable record of the earlier failed length strategy.
