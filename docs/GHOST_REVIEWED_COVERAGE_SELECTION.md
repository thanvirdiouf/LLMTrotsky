# Question-part source selection comparison

Saved43's initial unblinded implementation review identified relevant available
records that the selector omitted, alongside separate source-summary precision
limits. This trial tests a selection intervention while preserving every source
record and the original five questions/expectations. It does not address missing
precision in the immutable records, activate a route, train a model or validate
the source curation independently.

The tracked design is `config/ghost_reviewed_coverage_design_v1.json`. Generic
punctuation boundaries split each original question into literal ordered parts;
concatenating them reproduces every original character. Decimal points remain
inside numbers. These are mechanical segments, not independently reviewed task
decompositions. Quoted errors, background and constraints remain user data.

Each call receives the same full controlled source packets and every available
reviewed summary as saved43, plus the literal question parts. The output must be
JSON containing one `part_claim_ids` array per part. Each array contains distinct
known IDs or is empty. The same ID may serve multiple parts; the displayed union
preserves first-selection order and is bounded at 12 distinct records. It never
silently drops selected records. An invalid array, unknown ID, extra factual text,
truncation or over-limit union blocks the reply without retry. Empty arrays do
not establish that the corpus lacks evidence. Presence of arrays does not prove
semantic relevance or complete task coverage.

This is one five-call comparison of the combined instruction, literal-part input
and output-schema intervention. It cannot isolate the effect of segmentation
alone. No old selections, oracle IDs, expected answers or evaluator-authored task
requirements are supplied to the model. Original source-summary omissions and
source errors cannot be repaired by changing IDs alone. Fresh transfer remains
gated until actual useful, source-faithful behavior justifies it.

## Frozen scope

- Parent: `ghost_reviewed_selection_v1`, private saved43.
- New local version: `ghost_reviewed_coverage_v1`.
- Inputs: `evaluation/ghost_reviewed_coverage_inputs_v1/`; parent inputs unchanged.
- Original question part counts: 3/4/5/5/2.
- Input tokens: 2816/3865/3887/2656/1623; no truncation.
- Five selector calls/five research replies, 1024-token output cap, plus one
  5000-input/16-output memory probe. Non-thinking greedy generation unchanged.
- Same pinned Qwen3.5-27B, private notebook, two free T4s, GPU-only parameters,
  600-second installation and 2700-second workload caps.
- Loader, attention helper, generation implementation, renderer, profile,
  collection, ledger, memory probe and common grounding helpers are byte-identical
  to the archived parent. Free-form factual rewriting and model self-audit are
  absent; trusted-memory updates and active-model changes remain disabled.

The builder and verifier enforce the unchanged shared-runtime/source comparison.
The dedicated verifier checks frozen notebook/payload/input provenance, parent
requests, real tokenizer reconstruction, finite logits, GPU placement/backend
coverage, output budgets and isolated byte-exact parser/display replay.

```sh
.venv-tokenizer/bin/python scripts/verify_reviewed_coverage.py \
  --version ghost_reviewed_coverage_v1 \
  --tokenizer evaluation/ghost_reviewed_coverage_inputs_v1/tokenizer_assets
```

After completion download the exact saved version's `ghost_reviewed_coverage_bundle/*`
and `ghost_reviewed_coverage_results/*` into
`evaluation/runs/ghost_reviewed_coverage_v1/download/`. Use this verifier, not the
older selector, audit, synthesis or memory-probe verifier. Inspect reasoning
metadata before reading final JSON; keep intermediate model reasoning private.

## Validation and result-dependent decisions

Six new tests cover character/decimal-preserving decomposition, unchanged full
source inputs without review-answer leakage, duplicate/unknown/over-limit IDs,
cross-part union retention, fail-closed parsing without retries, exact frozen
pipeline replay and the actual five-call bound. All 148 focused ghost tests pass.
A CPU-only temporary-fixture preflight uses synthetic arbitrary IDs with the
real tokenizer and archived runtime metadata to exercise the complete verifier;
it rejects modified displayed text even with recomputed journal hashes. It makes
zero model calls and is neither GPU execution nor quality evidence. Frozen source
curation and original selector reviews are untouched; the new design review and
preflight are ignored under the new run directory.

Review selected records against actual source wording and original tasks. Report
requested detail coverage, missed available evidence, irrelevant selections,
certainty/speaker/date fidelity, unsupported curated assertions, inappropriate
refusals and usability separately from schema validity. Compare against actual
saved43 selections without claiming this familiar five-case set is fresh or
independent. The review is unblinded implementing-agent development review.

If selection omissions improve without introducing serious errors, address any
remaining source-representation gaps through a separately reviewed opt-in ledger
extension before considering the fixed fresh questions. If they persist, document
the selection limitation and choose a targeted diagnostic rather than another
broad writer prompt, self-audit activation or training. No automatic promotion
follows a successful run or complete-looking arrays.

## Submission

The frozen trial was submitted once as private Kaggle saved44 after preparation
commit `8a2cd8d` was pushed. Exact fetched saved44 normalized notebook cells match
the frozen source. Receipts/source remain ignored under the run directory. No
actual model-quality result is claimed at submission; active9B/v3 and all fresh
transfer/training gates remain unchanged.

## Saved44 result

Exact saved44 completed and passed the dedicated frozen/token/runtime/replay
verifier. Five calls plus the probe took658.28seconds; all five outputs stopped
normally, with9/3/5/7/5selected records and no truncations or thinking markers.
The result hash is
`282ff2d396202a31e7c6f0d189d39abaa9e626471a567062c01a47a294c2e400`.

The initial unblinded implementing-agent source review finds mixed task quality.
The hearing response recovers the corrected March date, but selects all nine
available summaries under a source-scope constraint, adding irrelevant material.
Eastman loses the previously selected initiative/tactics context, while its
immutable-form precision gap persists. Blumkin still omits the reform position;
its added prospective card retains prediction status. Report/hypothesis limits
and ILO distinctions survive, with redundancy and unrequested context remaining.
The total rises from22 to29records, without consistent improvement in completeness
or presentation. No unnecessary refusals occurred. Curation interpretations and
negative source assessments are not independently established by record status.

The next justified step is a focused target-span selectivity diagnostic: pure
scope/background spans versus factual request spans, with full original context
and source menus unchanged. It must preserve local expectations outside model
messages, and cannot become a factual self-audit or justify activation. Fresh
transfer remains gated. No broad writer retry or training is justified.
