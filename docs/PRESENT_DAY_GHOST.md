# Present-day Trotsky ghost — active project

The active product is an **experimental, clearly fictional Leon Trotsky ghost
living in the present day**, with no 1940 knowledge cutoff. Historical records,
dated contemporary evidence and newly imagined views remain distinct. Both
adapters stay experimental and training stays paused.

## Current result — 26 October 2026

Ten ghost runs are complete, downloaded and verified. They establish bounded
T4 memory capacity and several application fixes, but **do not establish reliable
interpretation or conversation**. V11 completes ordinary opinion prose without
JSON failures, yet its auditor confuses the fictional assistant with the user
and sometimes treats new opinions as institutional source claims. Canonical
facts survive. Some rejected premises still have legitimate source concerns.
Neither pipeline is promoted. V12 prepares a balanced checker-only diagnostic;
see [speaker/source-role calibration](GHOST_AUDIT_CALIBRATION.md).

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
| v12 / pending | at most 10 | Five valid/five invalid fixed audit candidates; expected labels withheld |

V1–v4 use the same inspected ten-turn development set. V5 introduces four
single questions and a six-turn conversation; v7/v8 reuse those as regressions.
V9 introduces five newly authored turns in each of two arms with identical
available ILO/radio context and separate actual histories. It compares whole
pipelines, not individual components. Source-exposed development and same-agent
unblinded review are not independent historian validation or a representative
benchmark. Detailed local reports/transcripts: `evaluation/runs/ghost_smoke_vN/`.

The last ordinary smoke has six clearly useful factual/scope/identity
replies. Its fuller-view mode works, but interpretation still misreads adoption
barriers as a statement about displacement and invents causal explanations for
a research metric. The shorter draft is filtered before model audit for writing
raw citations in prose. Model status is not a quality score.
[Failure boundaries and training decision](GHOST_FAILURE_ANALYSIS.md).

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

Forty-six focused ghost/source/harness/calibration tests pass. The full suite
runs 319 tests with two pre-existing historical pilot snapshot errors for the
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

### Application fixes following the fresh comparison

For mixed replies, reviewed contemporary facts are materialized independently
of model-written packet IDs. Interpretation-only requests do not dump canonical
facts by default. Narrow single-term quote styling and trailing known source
references are normalized and logged; unknown references, multiword quotes and
requested exact quotation remain guarded. Missing audit verdicts explicitly
reject only the uncovered segments; they never become approvals or erase a
separately valid reviewed segment. Semantic judgment remains fallible.

A separate local test found that a definite reference to the study lost its prior
ILO topic. The natural retriever now retains that source reference while explicit
new subjects still win. This patch was made after v10 was frozen and is recorded
separately: fixed-context GPU comparisons do not measure natural retrieval.
The same inspected five-turn comparison is prepared as v10 to test application
changes, not to claim fresh generalization.

### Latest generation contract

V10 returns task/detail labels on all five checked turns without legacy fallback.
Its dated study-only reply meets two sentences, and its viewpoint turn reaches
drafting. The final turn still blocks because the model emits incomplete JSON
for an ordinary paragraph. V11 therefore uses plain prose for ordinary opinion
drafting; the application wraps that paragraph internally. Selection/audit stay
structured, facts stay canonical, and exact quotation requests keep the structured
source-anchor path. The current request is last in every stage. Known selected
citation markers, whitespace and short source-matching concept styling are
normalized/logged; unknown references and invented quotes remain guarded.
This is a same-question regression, not fresh proof of reliable conversation.


V11 is verified as saved version 13: nineteen requests, no budget rejection or
length truncation, direct messages/replies byte-identical to v10. It displays
three partial replies, one factual reply and one refusal. Its source-safe
study-only answer is two sentences (33 words); rejected opinions reveal the
speaker/support-role errors described above. V12 explicitly identifies those
roles and tests ten fixed candidates before any further conversation study.
Candidates are never trusted replies or memory. The same base judge remains
fallible; a successful calibration would not establish production reliability.
