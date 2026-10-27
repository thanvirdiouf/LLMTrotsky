# Five-question candidate synthesis trial

The 27B neutral-source checker matched all ten fixed expected decisions in
saved39. Unblinded source review supported the decisions but found an extra
political characterization and a wrong block locator in two reasons. This is
enough to test experimental synthesis, not to enable the checker in production.
See [the audit results](GHOST_CANDIDATE_AUDIT_COMPARISON.md).

## Fixed questions and controls

`ghost_candidate_synthesis_v1` preserves the five factual-writer messages from
`ghost_smoke_v27` / saved29 exactly. They cover hearing chronology and household
roles, Eastman's condemnation and theoretical disagreement, two distinct
Blumkin inferences, the Rakovsky/Bessedovsky report routes and certainty, and
the dated ILO percentage denominator. Original questions and local evaluation
expectations are unchanged. The fresh mapped-transfer tasks remain unrun.

The complete source text, speaker roles, authorship, dates and context in each
writer request are unchanged. Candidate input IDs are reconstructed locally.
Original evaluator expectations remain local and hash-bound. No evaluation
answer is supplied to the writer or checker.

Each independent case has one factual draft and, if structurally valid, one
neutral-source audit. Drafts retain the existing one-to-four-paragraph,
2,000-character and literal-quotation constraints. Unsupported or unmatched
quotation delimiters block the case before checking. Truncated stages block
the case. The checker receives the same source context and actual generated
paragraphs; missing verdicts reject those paragraphs. Only accepted paragraphs
appear in the displayed research reply, with a limitation if others were
rejected. A model-approved paragraph is not a verified historical fact.

This trial differs from the old 9B source-reading pipeline in backbone,
inference preparation/backend and checker prompt. The writer input is matched,
but displayed answers form a whole-pipeline comparison, not a single-variable
capacity experiment. No political-persona opinion or sustained dialogue is
being tested yet.

## Runtime and checks

The separate candidate runtime retains the pinned 27B model, NF4/FP16,
fixed34/30 decoder-layer split and efficient-attention policy validated in
saved38/39. It checks six synthetic CUDA cases before loading weights.
The two successful GPU placement/attention helpers are copied exactly.
The runtime now reserves generation calls centrally, so even dependent
evaluation stages cannot exceed the frozen limits.

The limits are five independent displayed research replies, at most ten
evaluation calls with 1,024-output-token caps, and one separate unscored
5,000-input/16-output probe. Installation is capped at600seconds and the
workload at2,700seconds. Finite logits, input tokens, output stops and
full-attention coverage are recorded per generation. Existing results or
journals prevent automatic rerun/resume. No CPU/disk weight offload, adapters,
training, paid compute, public upload or active9B/v3 change is permitted.

Build and verify with:

```bash
.venv-tokenizer/bin/python scripts/build_ghost_candidate_synthesis.py --version ghost_candidate_synthesis_v1
.venv-tokenizer/bin/python scripts/verify_ghost_candidate_synthesis.py --version ghost_candidate_synthesis_v1 --tokenizer evaluation/ghost_candidate_synthesis_v1/tokenizer_assets
```

After completion download the exact saved version's
`ghost_candidate_synthesis_bundle/` and `ghost_candidate_synthesis_results/`
into `evaluation/runs/ghost_candidate_synthesis_v1/download/`.
The dedicated verifier checks frozen inputs/expectations, source integrity,
tokens, scope, placement, memory, finite logits, backend coverage and journal
hashes. It replays the frozen dependent writer/checker pipeline in an isolated
CPU process, requiring the displayed answer and call journals to reproduce
exactly. Replay checks behavior, not truth.

Four new tests exercise exact dependent request replay, source-context
preservation, literal-quote/truncation blocks, missing/negative verdicts,
no trusted-memory promotion, and actual call/output caps. All120 focused
tests pass. Full393 tests retain the same two historical README snapshot
errors; original locks are unchanged. Submission state is in the handoff.
The old saved39 audit continues to verify after the runtime extension;
its original frozen helper is unchanged in the archived notebook.

## Result-dependent review

Review every actual displayed answer against its source and original rubric.
Record requested-task coverage, answerable versus genuinely missing evidence,
dates, roles, hypothesis/prediction scope, unsupported assertions, inappropriate
refusals and checker errors separately. The implementing coding agent's review
is unblinded development review, not independent or human validation. Keep any
unexpected intermediate reasoning private.

Usable synthesis can justify the fixed fresh transfer questions, followed by
sustained conversations and voice. Checker approval alone cannot. Failed or
incomplete synthesis requires diagnosis; it does not justify automatic
activation, training or a blind rerun with larger budgets.

The frozen trial was submitted as private saved40 after commit `f92b314` was
pushed. The fetched notebook's three normalized cell sources/types match the
local submission exactly. Status at confirmation: QUEUED. The heartbeat uses
this saved version and dedicated verifier; no synthesis quality result yet.
CPU preflight confirmed all five writer input token sequences match the old
9B control and the full generation configuration round-trips unchanged.
