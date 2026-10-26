# Present-day Trotsky ghost — active project

The active product is an **experimental, clearly fictional Leon Trotsky ghost
living in the present day**, with no 1940 knowledge cutoff. Historical records,
dated contemporary evidence and newly imagined views remain distinct. Both
adapters stay experimental and training stays paused.

## Current result — 26 October 2026

Thirteen ghost runs are complete, downloaded and verified. The latest rendering
regression (V14, saved Kaggle version 16) makes ten replies in 23 generations
on the exact V13 questions and evidence. It fixes brief reported-information
coverage and recovers the observed invalid task label while preserving dated
ILO scope. **Whole-answer brevity and semantic reliability remain unresolved.**
The final reply is 84 words for about fifty requested; the auditor approves an
unsupported universal claim about past technological revolutions. Neither
pipeline is promoted. Both adapters stay experimental and training stays paused.
[Failure boundaries](GHOST_FAILURE_ANALYSIS.md),
[concrete next review](GHOST_NEXT_REVIEW.md).

| Ghost version / Kaggle version | Evaluation calls | Main finding |
|---|---:|---|
| v1 / 4 | 16 | Eight structural blocks; wrong observation date incorrectly approved |
| v2 / 5 | 19 | Canonical facts fix the date; retained sources improve continuity |
| v3 / 6 | 19 | Five drafts omit an empty field; useful conversation still fails |
| v4 / 7 | 19 | Useful factual/identity answers; unsupported opinion premises still pass |
| v5 / 8 | 20 | New factual cases preserve source limits; conversation fails modes and assumptions |
| v6 / none | 0 | Prepared routing-only bundle preserved, never uploaded or run |
| v7 / 9 | 22 | Study-only scope and draft routing fixed; audit confuses prior constraints |
| v8 / 10 | 21 | Fuller view displays; source overreach passes; shortening filtered for raw citations |
| v9 / 11 | 16 | Fresh comparison: direct source overreach; checked task/format failures |
| v10 / 12 | 18 | Valid task labels and dated two-sentence finding; final opinion JSON is incomplete |
| v11 / 13 | 19 | Prose completes; reviewer confuses speaker and source-support roles |
| v12 / 14 | 10 | All five valid accepted/five traps rejected; explanations still imperfect |
| v13 / 15 | 22 | Natural retrieval works; task/schema, source inference, coverage and depth/length still fail |
| v14 / 16 | 23 | Brief contrast and task recovery fixed; total length and semantic audit still fail |

V1–v4 use the same inspected ten-turn development set. V5 introduces four
single questions and a six-turn conversation; v7/v8 reuse those as regressions.
V9 introduces five newly authored turns in each of two arms with identical
available ILO/radio context and separate actual histories. It compares whole
pipelines, not individual components. Source-exposed development and same-agent
unblinded review are not independent historian validation or a representative
benchmark. Detailed local reports/transcripts: `evaluation/runs/ghost_smoke_vN/`.

V12's designed calibration accepts five valid candidates and rejects five
traps, with expected labels withheld. V13 demonstrates why that is not a
production success rate. The model judge remains fallible on actual generated
reasoning. Exact local reports/transcripts and source-bound judgments are in
`evaluation/runs/ghost_smoke_v12/` through `evaluation/runs/ghost_smoke_v14/`.

## Run and reproduce

The complete local research archive is required; public Git contains code,
profiles and design, not source excerpts or model artifacts.

```bash
python3 scripts/build_ghost_chat.py
python3 scripts/ghost_chat.py --dry-run \
  --question "How would you assess renting an A40 on Runpod today?"
python3 -m unittest discover -s tests -p 'test_ghost*.py' -v
```

With an operator-configured service already serving the untouched base:

```bash
python3 scripts/ghost_chat.py --base-url https://YOUR_SERVER/v1 \
  --log /tmp/trotsky-ghost-chat.jsonl
```

The CLI takes `TROTSKY_MODEL_API_KEY` from the environment without logging it.
No persistent remote ghost service has been deployed. HTTP model names are
operator assertions, not proof of revision or absence of adapters. The Kaggle
workflow separately records the pinned revision, stack and frozen parameters.
No weights or GPU inference run locally.

`config/active_chat.json` selects `config/persona_ghost_expansion.json`,
`config/ghost_source_plan_v2.json` and `evaluation/ghost_chat_v2/`. Its reference
date is **26 October 2026**. Frozen initial smokes retain 25 October. A reference
date is not live news; refresh sources and version artifacts explicitly.
An explicit source plan requires a new explicit output path, and building never
changes the active pointer. Existing generated evidence cannot be overwritten
with different content.

## Evidence and conversation contracts

| Content | Allowed behavior | Constraint |
|---|---|---|
| Historical record | Explain documented arguments, reports and predictions | Preserve author, speaker, original date, context and uncertainty |
| Contemporary evidence | Use supported facts after 1940 | Retain publication/update kind, separate observation date and limited scope |
| Fictional interpretation | Form new values, arguments and conditional views | Label the new view; support descriptive premises; never imply authentic posthumous writing |

The active collection has ten packets and 39 reviewed propositions. The
original seven historical excerpts and Runpod listing are unchanged. The vendor
listing was observed 25 October 2026 and separately labels an 8 October update.
Evidence v2 adds a complete bounded section of the speech delivered 1 March
1926, excluding its later publisher introduction, and three complete paragraphs
from an ILO article published 20 May 2025, observed 26 October 2026. Predictions,
potential exposure and observed outcomes remain distinct. Broad current-affairs
coverage and automatic live-news retrieval are not implemented.
[Source provenance and expansion](GHOST_SOURCE_EXPANSION.md).

The pipeline selects evidence, drafts, audits and renders. Factual prose comes
from reviewed immutable statements; exact quotes come from original anchors
with recorded speaker. Citation markup is inserted by the renderer. Model
prose cannot invent quotation/citation markup. Newly imagined reasoning is
labeled **“My present-day interpretation”** and excluded from verified-fact
memory. Retrieved reviewed historical basis is required for new persona views;
application-added basis IDs and excluded unrelated context are logged.

Selection, drafting and audit now retain full bounded source blocks. Earlier
versions used a summary-only draft view; that did not establish reliable
interpretation and could hide necessary surrounding limits. Audit input includes the current
request/mode and prospective rendered display, including the fictional label.
The auditor has a neutral role, but is the same fallible base model. Matching
source membership, format and citation anchors is not semantic entailment.
A passing review gives no factual guarantee.

Topic/basis state and actual displayed replies carry forward. Stored history
allows ten previous turns; model input includes only the most recent four
messages and an explicit omitted-message count. This is not unrestricted
semantic memory. The selection call now classifies the current request as facts, view or mixed,
with brief/normal/expanded detail. It adds no model calls. Legacy outputs fall
back to the earlier rule and are explicitly logged. New wording is not gated
solely by a keyword list. Fuller drafting permits 110–180 words, shorter
requests 35–60, subject to actual model behavior. Exact greetings, thanks,
farewells and direct persona-identity questions use application replies without
model calls, preserving source state.

## Validation and limits

Forty-nine focused ghost/source/harness/calibration tests pass. The full suite
runs 322 tests with two pre-existing historical pilot snapshot errors for the
already changed `evaluation/README.md`; historical locks remain preserved.
Kaggle runs verify submitted/downloaded payloads, exact frozen-code replay,
all input/output token reconstructions, zero optimizer updates, no adapters and
unchanged archived artifacts. Source curation and quality judgments are by the
continuing coding agent, not an independent historian.

The separate 5,000-token probe peaks at approximately 11.32 GiB on the T4.
This supports the compact batch, not arbitrary larger contexts or production
latency. The superseded historical 1940 v2 trial still failed at 6,546 input
tokens after fourteen direct answers. It is a separate partial archive, not a
completed quality comparison. No A40 or other paid GPU was provisioned.

Use [the smoke protocol](GHOST_SMOKE.md) for exact-version downloads and
verification, and [the fresh comparison design](GHOST_COMPARISON.md) for controls.
`python3 scripts/export_ghost_transcript.py --version ghost_smoke_vN` exports
verified exact displayed replies and source links without assigning scores.
Do not resume training or promote an adapter from execution success or scores
on inspected questions. Fresh source-based conversation evidence is required.

### Current application and evaluation boundary

Ordinary opinions use plain prose wrapped into internal segments. Selector and
auditor remain structured, factual wording stays canonical and exact quotation
requests use source anchors. Narrow known citation/style normalization is logged;
unknown references and invented quotes remain guarded. Missing verdicts reject
only uncovered segments and never become approvals. The current user request
is last in every stage. Candidate speaker and historical/modern support roles
are explicit; a new normative conclusion need not be endorsed by ILO.

Natural definite-study retention is tested locally and exercised by V13. The
older fixed-context comparison does not measure retrieval. V13 also reveals
that brief first-card rendering can omit a necessary contrast and canonical
fact blocks can make the overall answer too long even with a short opinion.
V14 fixes that brief-card contrast. Whole-answer length remains open, separately
from source inference errors.

After V13, only the observed task label `premise_corrections` is normalized to
`facts`, with a logged original value. Other unknown enums still fail closed.
The regression confirms that valid source IDs reach canonical factual rendering
without drafting an opinion. V14 verifies that recovery on the GPU; frozen
V13 outputs retain the original failure. Do not rebuild earlier versions with
current helpers or count this as a measured semantic improvement.

The next step is independent source/conversation review of the latest local
V14 worksheet, then new preregistered questions. No independent review has yet
occurred. Stop prompt micro-tuning on inspected cases and retain the untouched
base reference. The larger corpus, unlimited memory, live news and production
service remain outside the demonstrated prototype.


V14 verifies the rendering intervention on the exact V13 controls. The brief
Bessedovsky reply now includes both selected claims in 42 words; the ILO finding
stays 33 words/two sentences and its scope-reset limit stays 40 words/two
sentences. The observed enum recovery displays supported source limits instead
of blocking. The whole-reply planner correctly logs a six-word draft allowance,
but the model writes forty opinion words; the final display is 84 words instead
of the requested about fifty. It is not a solved length constraint.

The deeper view is longer but repetitive and includes an unsupported historical
universal that the auditor approves. Further semantic prompt tuning on these
inspected cases is not justified. Independent review remains pending, using the
source/context worksheet and exact transcript in the local V14 run directory.
See [response-budget results](GHOST_RESPONSE_BUDGET.md). No new run is queued.
