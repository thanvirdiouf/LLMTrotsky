# Ghost inference protocol and continuation

This workflow tests the clearly fictional present-day persona with the pinned
untouched Qwen base. It makes no training updates and attaches neither adapter.
A successful run, structural check or model approval is not a quality score.

## Current state — 26 October 2026

Ghost v1–v5, v7 and v8 are complete, downloaded, verified and source-reviewed.
V6 is preserved as a prepared routing-only bundle, **never uploaded or run**.
V9 is the completed fresh matched-source direct/checked comparison, saved
Kaggle version 11. V10 completed as saved version 12; V11 completed as saved version 13; V12 completed as saved version 14; V13 completed as saved version 15 with fresh end-to-end questions; V14 completed as saved version 16 with the same questions. [Active guide](PRESENT_DAY_GHOST.md),
[comparison design](GHOST_COMPARISON.md), [failure analysis](GHOST_FAILURE_ANALYSIS.md).

| Local version | Saved Kaggle version | Evaluation calls | Source/case scope |
|---|---:|---:|---|
| ghost_smoke_v1 | 4 | 16 | Initial 8 packets/33 claims, initial ten turns |
| ghost_smoke_v2 | 5 | 19 | Same inspected cases |
| ghost_smoke_v3 | 6 | 19 | Same inspected cases |
| ghost_smoke_v4 | 7 | 19 | Same inspected cases |
| ghost_smoke_v5 | 8 | 20 | 10 packets/39 claims; four new singles and six-turn conversation |
| ghost_smoke_v6 | none | 0 | Prepared, unrun |
| ghost_smoke_v7 | 9 | 22 | Inspected v5 regressions |
| ghost_smoke_v8 | 10 | 21 | Inspected v5 regressions |
| ghost_smoke_v9 | 11 | 16 | Five new dialogue turns in each of two matched-source arms |
| ghost_smoke_v10 | 12 | 18 | Valid task labels; final paragraph has malformed JSON |
| ghost_smoke_v11 | 13 | 19 | Prose completes; auditor role confusion causes rejections |
| ghost_smoke_v12 | 14 | 10 | Five valid accepted, five false rejected; no blocks |
| ghost_smoke_v13 | 15 | 22 | Fresh ordinary retrieval; task/schema, inference and conversation limits |
| ghost_smoke_v14 | 16 | 23 | Exact V13 controls; contrast/task recovery fixed, length/audit still fail |
| ghost_smoke_v15 | 17 | 22 | New reviewer-authored controlled-source baseline |
| ghost_smoke_v16 | 18 | 22 | Same ten replies; historical fallback unexercised |
| ghost_smoke_v17 | 19 | 24 | Finite selection shape succeeds, mixed usefulness |
| ghost_smoke_v18 | 20 | 24 | Reviewed compact forms; improved coverage/sentence compliance |
| ghost_smoke_v19 | 21 | 25 | New two-plus-eight transfer study; context omissions and parser failures |
| ghost_smoke_v20 | 22 | 25 | Missing user context supplied; delayed displayed answer still incomplete |
| ghost_smoke_v21 | 23 | 27 | Two parser blocks removed; partial answers still omit task details |
| ghost_smoke_v22 | 24 | 10 | Five valid accepted; two false corroboration claims also accepted |
| ghost_smoke_v23 | 25 | 10 | Two corroboration errors fixed; one new false theory-agreement approval |
| ghost_smoke_v24 | 26 | 27 | New claim rendered; packing still drops selected requested facts |
| ghost_smoke_v25 | 27 | 28 | Fresh tasks: context retained, factual completeness/shortening still fail |
| ghost_smoke_v26 | 28 | 20 | Five independent factual questions in canonical and source-reading arms |

V1–v4 retain a frozen reference date of 25 October; later versions use 26 October.
Execution occurred on 26 October. Do not silently change dates, helpers or
sources in a frozen experiment. Source/card/profile files, journals, raw outputs,
verification and response-bound judgments remain local under
`evaluation/runs/ghost_smoke_vN/`.

## Standing authorization and bounds

After the initial automatic-review upload block, the user explicitly approved
the ghost smoke and bounded follow-up iterations on 26 October 2026. Scope:
the same private notebook `thanvirdiouf/trotsky-grounded-base-chatbot`, numeric
ID 140792344; project helpers/persona prompts, reviewed public historical and
contemporary excerpts/propositions, and evaluation questions. At most ten
answers/thirty evaluation requests plus one short memory probe per run. No
training, adapters or paid compute. Do not repeat the resolved permission
question within that scope. Preserve the original notebook title/URL.

The ordinary smoke uses at most three stages per answer. The comparison has
five direct and five checked turns, at most twenty evaluation requests. Both
include one separate unscored memory probe. The builder rejects excessive case
sets. Expected checks/rubrics are excluded from model cases.

Before evaluation, a synthetic 5,000-token request generates at most sixteen
tokens to test prefill capacity. Its reply never enters conversation or a score.
Probe errors are recorded. Each real request has a 5,120-token input cap and
1,024-token generation cap. Over-budget requests are journaled without calling
the model. Incomplete stages fail closed. No source paragraph is silently
truncated; CUDA cache is released between calls. The measured probe peaks near
11.32 GiB on Kaggle GPU0 Tesla T4.

The superseded historical 1940 v2 trial is separate: fourteen direct answers,
no checked-arm answers, and OOM at 6,546 input tokens. Its partial archive is
not a completed comparison. Saved version 3 restored the original notebook
title with no execution. No paid GPU was rented.

## Build and inspect

Use the complete local research archive and cached tokenizer assets:

```bash
.venv-tokenizer/bin/python scripts/build_ghost_smoke.py \
  --version ghost_smoke_vN \
  --evaluation-spec config/ghost_chat_evaluation_v2.json
python3 -m unittest discover -s tests -p 'test_ghost*.py' -v
```

For the fresh direct/checked experiment, use
`--evaluation-spec config/ghost_comparison_evaluation_v1.json`. The model,
revision, packages, greedy/nonthinking policy and no-adapter checks remain pinned.
An existing frozen version cannot be rebuilt with changed inputs. Use a new
version rather than replacing its notebook, lock, protocol or payloads.

Local artifacts include `evaluation/ghost_smoke_vN/{protocol,evaluation_spec,bundle_lock}.json`
and `training/kaggle_cli/ghost_smoke_vN/`. The generic notebook copy is a working
convenience; the per-version staged notebook is the verification source.

- `evaluate_ghost_smoke.py` journals stages and exact displayed history.
- `evaluate_ghost_comparison.py` keeps matched source context and separate arms.
- `kaggle_ghost_smoke.py` verifies the pinned stack/base and runs on Kaggle only.
- `verify_ghost_smoke.py` reconstructs tokens and replays frozen helpers.

The user authorized milestone commits/pushes. Publish helpers, profiles, case
design and documentation; retain excerpts, ledgers, notebooks, outputs and
weights in the ignored research archive. Origin remains SSH. The working gh
credential helper can push through an explicit HTTPS URL without exposing
credentials; direct SSH signing still needs an unlocked agent. Never request
or print a passphrase/token.

## Download and verify

Download the exact saved Kaggle version into the corresponding local run
folder; do not rely on whichever version becomes latest later. Then:

```bash
.venv-tokenizer/bin/python scripts/verify_ghost_smoke.py \
  --version ghost_smoke_vN \
  --tokenizer training/data_revision_v2/tokenizer_assets
python3 scripts/export_ghost_transcript.py --version ghost_smoke_vN
```

Verification checks submitted/downloaded bytes, preserved historical helpers and
adapter hashes, model identity and frozen parameters, journal coverage/hashes,
probe/generation bounds, and exact prompt/output tokenization. Replay executes
the frozen ordinary or comparison evaluator in a separate CPU process with no
inference. Later active code changes cannot redefine an old run.

The transcript preserves exact displayed answers and links source metadata.
Direct-arm citation references are labeled as model-written and unchecked;
references outside supplied context or unknown IDs are flagged. The exporter
verifies the answer journal and frozen source/case bytes, assigns no scores,
and leaves the research transcript excluded from public Git.

## Review and result-dependent decisions

Inspect every displayed answer and candidate against the source. Count factual
errors, source certainty, speaker/date preservation, unsupported opinion
premises, rejected supported content, unnecessary refusal, relevance, voice,
depth, actual shortening and continuity separately. An opinion has no unique
historically correct reference answer. Strong normative views are allowed;
new descriptive claims still need evidence or an explicit hypothetical.
Record ambiguous rhetoric as ambiguity. Same-agent review is not independent
historian validation.

Inspect stage presence and filter reasons: a `review_rejected` status can arise
from an application format filter before any model audit. Missing draft stages
are routing failures, not evidence that the model failed a task it never saw.
A source-faithful but irrelevant fact dump does not satisfy a conversation.
Correct refusals of unavailable evidence differ from refusing answerable turns.

V1–v4 locate schema, date rendering, source-state and drafting problems. V5
shows that relevant technology sources alone do not fix conversational control
or unsupported assumptions. V7 fixes study-only scope and skipped drafting,
but the audit partly applies old constraints and still accepts bad inferences.
V8 displays a fuller view with the correct fictional label, yet the same source
overreach passes and shortening is filtered for raw citation markup. Further
tuning of these inspected questions cannot establish general reliability.
The fresh comparison tests whether the complete pipeline improves faithfulness
at the expense of useful conversation, with the same available evidence.

Keep training paused and both adapters experimental until that distinction is
clearer on fresh evidence. Memory feasibility does not justify a training run
or deployment, and a checker approval is never a semantic guarantee.

V9 separates base interpretation errors from application failures: direct
misstates adoption/displacement limits and predicts inevitability; checked
skips drafting for new viewpoint wording, filters a stylistic term and blocks
valid content when one verdict is missing. V10 classifies facts/view/mixed and
detail in the existing select call, restores full context to drafting, normalizes
narrow format mistakes, and fails closed only on unreviewed segments. Same
questions, sources, profile and model; direct control is re-generated.
Natural definite-source retention is separately unit-tested outside the frozen
fixed-context experiment. No source-data or training changes are involved.

V10 is verified with eighteen generations and no input-budget or length-cap
failure. Valid task labels fix the tested transitions and the study-only reply
is dated and two sentences. The last paragraph ends with invalid JSON despite
a stop token. V11 removes that serialization requirement for ordinary opinions,
while retaining structured selection, audit and requested source quotations.
The application supplies factual blocks/labels/citations and logs narrow styling
normalization. Same source/model/case controls; maximum twenty requests plus
one probe. Interpretive faithfulness remains a separate quality question.


V11 is complete and verified, including frozen replay and token reconstruction.
V12 selects `evaluate_ghost_audit.py` only for its calibration purpose: ten fixed
candidates, ten checker requests, one separate probe. No candidates become
trusted chat output or verified memory. The model receives no expected labels.
See [calibration protocol](GHOST_AUDIT_CALIBRATION.md).


V13 is complete and verified: ten replies, 22 generations, no input budget
rejection or truncation, 11.32 GiB probe peak. Ordinary retrieval retains the
needed evidence, but classification/schema, brief coverage, source inference,
depth and overall brevity still fail. After freezing, a narrow local enum
normalization is unit-tested but not GPU-retested. Preserve V13 as executed.
See [next review and decision](GHOST_NEXT_REVIEW.md). That review queued no run; a subsequent user continuation prepares V14 for rendering fixes.


V14 is complete/verified: ten replies, 23 generations, no budget rejection or
length truncation, 11.32 GiB probe peak. Brief source contrast and the observed
enum recovery now work on GPU. Whole-reply allowance is logged correctly but
not obeyed (84 total words for about fifty); the auditor still approves an
unsupported historical universal. Local report/transcript/neutral worksheet are
in `evaluation/runs/ghost_smoke_v14/`. No further run is queued. Training and
adapter promotion remain paused.

V27 was submitted as private saved version29 with revised reading-writer instructions; same ten replies and
at-most25-call limit, plus one probe. See GHOST_SOURCE_READING_COMPARISON.md.

V27/saved29 verified22calls. V28 stages reader versus mapped reader, ten replies
and at most25calls+probe1; see GHOST_EVIDENCE_MAP.md.

V28/saved30 verified13calls: all five mapped outputs block before composition.
V29 stages a block-reference map revision, same ten tasks and25call/probe limits.
See GHOST_EVIDENCE_MAP.md for causes and the unchanged validation boundary.

V29 submitted as private saved31; results pending.

V29/saved31 verified18calls with two displayed mapped answers containing
unsupported details. V30 stages ten neutral factual-auditor candidates, at most
ten evaluation calls plus probe1; see GHOST_NEUTRAL_SOURCE_AUDIT.md.

V30 submitted as private saved32: ten neutral-source audit candidates, results
pending.

V30/saved32 verified10calls with one false approval and one false rejection,
three format blocks. V31 stages the identical ten candidates with the same
pinned base, thinking enabled and4,096 combined output tokens; at most10calls
plus probe1. See GHOST_THINKING_AUDIT.md.
