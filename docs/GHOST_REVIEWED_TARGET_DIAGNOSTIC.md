# Focused literal-target diagnostic

Saved44 recovered one omission but increased irrelevant selections, lost an
Eastman detail and still missed Blumkin's reform position. This diagnostic tests
whether selection behaves differently when a single literal span is the target
and the rest of the original question is explicitly context. It is a familiar,
post-hoc development diagnostic, not a fresh benchmark or model-quality proof.

The five target spans are fixed in
`config/ghost_reviewed_target_design_v1.json`. Three set source scope or a
hypothetical setting; two ask about factual source content. All are unique exact
substrings of the unchanged original questions. Full source packets and every
available reviewed summary remain supplied. No model-written reasons, factual
rewriting, self-audit gate or trusted-memory updates are introduced.

The model receives the literal target, original question and sources, with a
generic ID-selection instruction. It does not receive target category labels,
expected outputs, required record IDs or saved43/44 selections. The instruction
explains that scope/background alone calls for no factual records. This tests the
combined focused-target instruction and input intervention; it does not isolate
instruction from attention or establish a general intent parser.

The **outputs answer only their target spans**, not the five complete original
questions. Original questions remain journal context for provenance. An expected
empty selection for a scope/background span is not a refusal or a claim of absent
corpus evidence. The two factual controls prevent always-empty behavior from
passing all checks. The model/runtime/parser/renderer/source records/generation/
probe are unchanged from saved43; only the frozen request messages differ.

Local expectations are frozen separately at
`evaluation/runs/ghost_reviewed_target_v1/expectations.json`. They check empty
selection for the three controls and required-record membership for the two
factual probes. Only their SHA256 is in the uploaded protocol. Membership can
pass despite irrelevant additional records or other semantic defects; inspect
actual selections and source wording separately. These coder-authored expectations
and their review are not independent or human validation.

## Scope and verification

Local version: `ghost_reviewed_target_v1`; frozen inputs:
`evaluation/ghost_reviewed_target_inputs_v1`. Five ID-only calls/five diagnostic
displays plus one 5000-input/16-output probe. Inputs:
2669/3774/3763/2529/1497tokens, no truncation. Output cap1024, installation600s,
workload2700s, same private notebook/pinned27B/two free T4s. No training, adapters,
paid compute, local weights, public upload or active9B/v3 change.

The dedicated builder/verifier bind the exact parent/source/runtime controls,
target spans, frozen requests, protocol and local expectations hash. After the
exact saved run completes, download `ghost_reviewed_target_bundle/*` and
`ghost_reviewed_target_results/*` to the new run's `download/` directory, then:

```sh
.venv-tokenizer/bin/python scripts/verify_reviewed_target.py \
  --version ghost_reviewed_target_v1 \
  --tokenizer evaluation/ghost_reviewed_target_inputs_v1/tokenizer_assets
```

The verifier checks real tokens, budgets, finite logits, GPU/backend coverage and
exact isolated archived parser/display replay, then reports the limited local
diagnostic checks separately from semantic quality. Inspect metadata before final
JSON; intermediate reasoning stays private. Put new source/selection review in
the new run directory, never in frozen curation/input reviews.

Four new tests cover exact target spans, unchanged complete context without
expectation leakage, ambiguity/rewriting/input-budget rejection and local scoring
that distinguishes blocked from valid empty outputs and does not call membership
semantic validation. The CPU preflight uses synthetic arbitrary IDs, real local
tokenizer and temporary archived-metadata fixtures only, rejecting changed display
text and changed expectation bytes. It is not GPU execution or quality evidence.

## Decisions after results

Review the three controls for irrelevant selection and the two factual probes
for actually requested advice/context, including unwanted records and curation
limitations. Success would justify a bounded experiment composing focused
selections on the original tasks; it would not validate a chatbot or authorize
activation. If factual omissions persist even when isolated, document a
selection/representation limitation rather than another broad writer prompt or
automatic training. In either case, source-form precision remains separate.
Fixed fresh transfer and sustained conversation remain gated on useful behavior.

## Submission

The frozen diagnostic was submitted once as private Kaggle saved45 after commit
`17d133f` was pushed. Fetched normalized saved45 cells match the local notebook.
All152 focused ghost tests and the final CPU verifier preflight passed; actual
GPU selection quality is pending. The source curation, active9B/v3, paused
training and fresh-transfer gate remain unchanged.

## Saved45 result

Saved45 completed with five calls plus the probe in657.56seconds. All outputs
stopped normally; no thinking markers or truncations occurred. Exact frozen
source/runtime/expectation/token/logit/backend and parser/display replay passed.
Result SHA256:
`e51f7e657fb6608e1f0dc21ab4d8a14c8d24eacd296fef107eef86aca26ff101`.

The limited local checks passed2/5: the Eastman factual membership probe and the
hypothetical-setting control. Hearing and ILO scope-only targets instead selected
three factual records each, addressing the original full questions. This is a
target-selection failure despite source-supported wording. Composition is not
justified. The Eastman response recovered initiative/tactics; its separate
immutable-summary precision limits remain.

Blumkin selected the rejected allegation and direct-conversation duty advice but
missed the broader reform-context record required by the unchanged development
rubric. Source blocks5-6 describe the conversation; block7 separately describes
an informative letter carried through Blumkin to friends. The frozen rubric
failure must not be presented as proof that the selected direct advice is false,
or that the letter's reform passage was spoken directly to Blumkin. This task/
expectation-scope caveat is recorded in the unblinded run review; expectations
remain unchanged. Neither membership scoring nor this coding-agent review is
independent validation.

The next bounded diagnostic tests a concrete input-label hypothesis: the full
original question is named `current_user_request` inside the data despite the
target-only task. Renaming that key and its single system reference to
`original_question_context`, with every value/control/expectation unchanged,
tests this ambiguity. It is not yet a proven cause. Only one new run is planned;
if scope failures persist, stop selector-prompt retries on these familiar cases
and review routing architecture and task expectations offline.
