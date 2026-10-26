# Next review after the fresh end-to-end study

V13 completes the requested small grounded prototype experiment: bounded
retrieval, source metadata/certainty, insufficient-evidence questions and a
sustained conversation using the untouched base. It does not establish a
reliable finished chatbot. The important result is that the failure stages can
now be separated using exact saved requests, candidates, audits and replies.

| Stage | Evidence from V13 | Remaining work |
|---|---|---|
| Retrieval | Relevant packets present; definite-study references retain ILO; unrelated election abstains | Broader independently authored source-recall cases; this small set is not a general recall estimate |
| Task interface | Factual premise correction returns an invalid enum; prediction question is routed to opinion | Observed enum synonym now logged/normalized locally; other unknown modes still block; test misclassification separately |
| Factual rendering | Correct dated findings and uncertainty; brief Bessedovsky reply drops a needed selected card | Preserve the necessary contrast while meeting sentence/word budgets |
| Interpretation | Exposure gradients are confused with adoption barriers | Review descriptive premises inside opinions; normative preference itself needs no institutional endorsement |
| Auditor | Clean designed calibration, but approves the exposure/barrier error in a real conversation | Independently review entailment, false acceptance and unnecessary refusal before trusting a replacement reviewer |
| Conversation | Tested mode changes work; deeper reply repeats earlier language; final reply misses overall word target | Judge argument development and whole displayed-answer length, including canonical facts |

The next useful evidence is independent review, followed by a preregistered
study with new questions. Do not keep adjusting prompts against these same
inspected replies. The local review packet is
`evaluation/runs/ghost_smoke_v13/REVIEW_WORKSHEET.md`; it includes frozen source
blocks, actual displayed replies and generated candidates, without our case
labels or the model's approval reasons. Reviewers should record:

1. Whether descriptive claims follow from the supplied source and preserve its
   dates, speaker, certainty and context.
2. Whether an opinion is a permissible new normative/conditional argument or
   contains an unsupported assertion; record ambiguous rhetoric explicitly.
3. Whether the actual reply answers the request, handles disagreement and
   follows depth, shortening and mode changes.
4. Whether filtered/rejected content deserved rejection, and whether relevant
   evidence was unavailable, missed, selected incorrectly or rendered poorly.

No independent review has yet happened. The coding agent's existing judgments
are unblinded development evidence, not gold. Resolve disagreements explicitly;
do not manufacture a unique correct modern Trotsky opinion. Keep factual
entailment and conversational taste separate.

Before another quality comparison, freeze new reviewer-authored questions and
criteria, include supported and unsupported premises, and retain matched
untouched-base controls. Compare any proposed reviewer or simpler presentation
against both false acceptance and unnecessary refusal; do not select on model
approval counts. Introduce changes separately where practical. Existing case
sets remain regressions, excluded from claims of fresh performance.

The post-V13 `premise_corrections` → `facts` normalization has a local regression
but no new GPU result. It preserves validated claim IDs, generates no opinion,
logs its action and still requires auditing. Do not attribute improvement to it
in the frozen V13 output. Other semantic and rendering failures remain open.

Keep both adapters experimental and training paused. Free Kaggle already fits
these bounded batches; no paid GPU has been provisioned. Source review and
application design are the next bottlenecks, not demonstrated GPU capacity.
There is no persistent inference service or production deployment.


On the next user continuation, application work also addresses the known
brief-card and whole-reply budget defects. V14 is an exact-question rendering
regression, not renewed semantic prompt tuning or fresh validation. Whole cards
and uncertainty remain intact, and actual model compliance is measured rather
than assumed. See [the response-budget design](GHOST_RESPONSE_BUDGET.md).
