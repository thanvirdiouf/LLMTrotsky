# What the ghost experiments establish

The purpose of the prototype is to distinguish failures of evidence retrieval,
source interpretation and conversation before deciding on more training.
Execution success and checker approval are separate from usefulness and source
faithfulness. These are small same-agent development reviews, not independent
historian validation. V1–v4 share inspected questions; v5 introduces a new set
that becomes inspected regressions after review. Most interventions combine
changes, so aggregate differences are not causal estimates.

A forceful new political judgment is not an error merely because a reviewer
disagrees with it. Separate normative views, conditional arguments and
descriptive assertions; there is no unique authentic reference answer for a
fictional modern opinion. Ambiguous rhetorical claims about personal
circumstances need that ambiguity recorded, not an invented independent gold
label. The conservative v5 review flags missing conditionality in its job-loss
line; that line can also be read rhetorically. The exact transcript is retained
so a later reader can reassess it.

## Observed failure boundaries

| Observation | Located failure | Evidence and response |
|---|---|---|
| A follow-up loses its relevant source | Retrieval/conversation state | Persist explicit topic and approved supporting source IDs; retain actual displayed history |
| Technology opinions retell unrelated organizational letters | Coverage/basis selection | Add a reviewed technology speech; log application-added historical basis IDs |
| Useful historical cards are selected but the answer is blocked for an absent empty field | Application schema | An omitted empty quote list adds no evidence; accept it and retain valid factual cards |
| A negative question produces a false answerable boolean despite useful supporting IDs | Selection interface | Select evidence IDs without asking for the truth value of the question |
| The checker approves the wrong vendor observation date | Model interpretation/audit | Render reviewed factual statements canonically; do not call this improved model factual reasoning |
| A worried worker is described as already having lost a job | Unsupported interpretation | User concern does not authenticate an event; audit descriptive premises inside opinions |
| A study-only answer includes an unrelated historical argument | Selection/render scope | Source membership is insufficient; explicitly exclude unrelated context from that displayed mode |
| A fuller-view request has no draft stage and repeats facts | Conversation routing | The model never got the interpretive task; repair the intent transition before considering training |
| A checker approves irrelevant canonical facts | Audit relevance | Check the current request, not only textual entailment |
| A 2025 exposure estimate cannot establish a 2026 outcome | Expected evidence limit | A useful scoped limitation is correct behavior; inventing zero or a precise job-loss count is not |

The v5 trace identifies a further audit-design concern: the checker inherited
the character's roleplay instructions. The v7 experiment gives it a neutral
auditor role, retains full original context and permits new normative reasoning
while requiring support for descriptive premises. V7 confirms a neutral instruction alone is insufficient: unsupported inferences
still pass, and the auditor partly applies an earlier facts-only constraint to
a current request for a fictional view. V8 supplies the current request/mode
and prospective display explicitly. Neither change guarantees accurate auditing.

## What structural guarantees cover

The application checks source membership, anchor/speaker identity, immutable
factual wording, quotation spans, citation IDs, date provenance and explicit
context budgets. It separates accepted factual memory from fictional opinion
basis. Those checks cannot establish that every generated interpretation is
sound, that all assumptions were identified, or that every supported statement
is relevant. Even a labeled fictional view can contain unsupported assertions.

Selection, drafting and auditing now see full bounded source blocks and reviewed
propositions. Earlier summary-only drafting did not resolve interpretation. The conversation log retains all
displayed replies, while model input contains the most recent four messages
and an explicit omitted-message count. Durable topic/basis state is not a
semantic summary of every earlier turn. Six evaluated turns therefore do not
prove unrestricted long-conversation memory.

## Compute and training decisions

The compact 5,000-token T4 probe peaks at approximately 11.32 GiB. V5's actual
evaluation inputs range from 1,021 to 3,521 tokens, with twenty generations
taking about 302 seconds excluding startup and the probe. This establishes
capacity for this bounded batch, not an arbitrary larger corpus, longer context
or production latency. The historical 6,546-token trial still failed with OOM.
No paid GPU was provisioned.

Keep both adapters experimental and training paused. Schema failures, missing
source coverage, lost topic state and skipped draft stages need application or
retrieval fixes. Unsupported generated premises and false semantic approvals
remain model-quality concerns after those boundaries are controlled. Before
training, test genuinely new questions and longer conversations, inspect
unnecessary refusals as well as false acceptances, and retain a matched
untouched-base reference. Do not train on these inspected tests and then report
them as fresh evidence.

Detailed local records, source hashes and response-bound judgments live in
`evaluation/runs/ghost_smoke_vN/`. Research payloads and transcripts remain
excluded from the public source repository.

V8 displays the fuller view after explicit current-task/display metadata, but
still approves adoption/displacement overreach. A shortening draft is filtered
for model-authored citation markup before any semantic audit. These inspected
regressions have reached diminishing diagnostic value. The next fresh
[matched-source direct/checked comparison](GHOST_COMPARISON.md) keeps base and
available context fixed while testing whole-pipeline usefulness and faithfulness.
Do not assume the more elaborate pipeline is better without that comparison.

The fresh v9 comparison confirms both categories on new robustness wording:
direct engages with the dialogue but invents certainty, while checked skips a
viewpoint task or blocks format/coverage despite available sources. V10 tests
current-task classification in the existing select call, full-context drafting
and narrow structural recovery without training. An additional local natural-
retrieval miss on the definite study reference is corrected separately; the
fixed-context comparison cannot validate retrieval recall. Keep this distinction
when interpreting any improvement on the now-inspected questions.

V10 confirms task classification can repair the tested skipped-stage problem
without extra calls or training. It still fails an ordinary paragraph because
of malformed JSON, so V11 removes that paragraph's serialization contract.
This preserves structured evidence control while avoiding a format failure
that should not be confused with source absence or interpretive ability.


V11 removes ordinary draft JSON failures but exposes reviewer role confusion:
fictional first-person lifetime references are treated as human-user claims, and
new normative conclusions are incorrectly attributed to institutional sources.
This is distinct from valid objections to unsupported causal premises. V12
isolates the auditor with five valid and five deliberately false fixed
candidates, expected labels withheld. Speaker and source-support roles are
explicit; political preference need not be endorsed by ILO, while descriptive
premises still require evidence. See [calibration](GHOST_AUDIT_CALIBRATION.md).


V12 makes all ten preregistered designed audit decisions correctly, but V13's
fresh full pipeline still approves a source-concept error (exposure gradients
versus adoption barriers). It also exposes an invalid task enum, a source-status
question routed to opinion, brief rendering that drops a necessary card, and
whole-answer length/depth failures. Relevant natural retrieval and definite-
study retention work in these cases. The observed enum synonym is now recovered
locally with logging; no post-fix GPU result is claimed. Stop adapting to these
inspected questions. [Independent review and next evidence](GHOST_NEXT_REVIEW.md)
are prepared; no independent review has yet occurred. Training stays paused.


V14 regresses the exact V13 questions/evidence with complete-card packing and
whole-reply budget planning. It fixes the brief report/witness contrast and
recovers the observed factual task enum, while preserving ILO dated scope.
The final model draft ignores a six-word allowance and yields 84 displayed words
for about fifty requested. Budget diagnostics are correct; compliance is not
established. A deeper but repetitive opinion adds an unsupported universal
claim about past technological revolutions and is approved. Keep the measured
application fixes, leave those failures open, and obtain independent review
before further semantic prompt tuning or training.
