# Reviewed-record selector experiment

Saved42 again approved both incorrect hearing candidates. This experiment
removes free-form factual rewriting and the self-checker from the factual display
path. The model selects reviewed summary IDs; the application reproduces the
selected statements and provenance. Selection can still be irrelevant or
incomplete, and the reviewed statements can still contain curation errors.
This is an opt-in research route, with no active9B/v3 replacement.

## Source representation

The new plan `config/ghost_reviewed_ledger_plan_v1.json` starts from the unchanged
v4 collection and 42-claim ledger. Nine additions cover details identified in
the offline audit: the corrected March date, Eastman's initiative/tactics,
Blumkin's qualified disclosure purpose, Rakovsky's reported transport confidence,
bounded fate/informant limits, and ILO category shares, arithmetic and local
application limits. All original records and source packets remain unchanged.

Five additions paraphrase sources, including an author's qualified inference.
Three are application assessments of what entire bounded excerpts do not
establish. One is application arithmetic. Review kind describes the processing;
certainty remains separately represented by status and attribution. Bounded
assessments are labelled inferences, not invented source admissions. The ILO
sum is explicitly derived, not a separately reported source number. Decimal
metadata checks verify 5.7 + 4.7 = 10.4 and source-token occurrence; population,
category and wording semantics still require source review. A stipulated
hypothetical calculation remains distinct from an empirical office estimate.

The new builder accepts mixed-domain supplements only for existing hash-bound
packets, assigns dates/domain from those packets and rejects metadata overrides,
old-ID reuse, partial source scopes for absence claims, wrong arithmetic and
arithmetic presented as an explicit source statement. The old historical-only
supplement guard is unchanged and still rejects modern packets. Changed output
files require a new directory. Statements, anchors and source examples remain
ignored locally.

```bash
python3 scripts/build_reviewed_source_ledger.py --output evaluation/ghost_reviewed_coverage_v2
python3 scripts/audit_reviewed_coverage_extension.py --output evaluation/runs/ghost_reviewed_coverage_v2
.venv-tokenizer/bin/python scripts/prepare_reviewed_selector.py --output evaluation/ghost_reviewed_selector_inputs_v1
```

The result has 51 claims. Coding-agent source review is unblinded development
review, not independent or human validation. The local `SOURCE_REVIEW.json`
binds the supplements, ledger and input lock. Oracle displays demonstrate
representation and preservation only:4/4/7/9/6 inspected statements, with no
selected facts dropped. These are not measured model choices, minimal necessary
selections, or task-accuracy scores. Oracle IDs and evaluation expectations
are excluded from the model messages and executable bundle.

## Frozen inference trial

`ghost_reviewed_selection_v1` uses the same five original questions and controlled
source packets as the earlier factual trials. The model receives full bounded
passages and every available reviewed summary for those packets, including
status and attribution. It outputs only `{"claim_ids":[...]}` with at most 12
distinct known IDs. Source text and user premises remain data, not instructions.
Unknown IDs, extra model text, duplicate IDs or incomplete output block the case
without retry. Empty selection says summaries were not selected; it does not
claim the source corpus lacks evidence.

The renderer preserves every selected statement, source date, author, status,
attribution, anchor speaker/role and context reference. No model-written factual
sentence is displayed and no model audit approves it. Relevance, completeness
and semantic assurance remain explicitly unverified. No selected fact is added
to trusted memory. The displayed format is a source-summary panel rather than
the earlier free-form answer; usefulness must be assessed on that actual format.

This is a whole-pipeline development comparison: proposition coverage and output
architecture both change. It is not a model-size ablation or fresh validation.
Retrieval is controlled, not measured. The fixed fresh-transfer questions remain
unrun; conversation and voice come after usable factual behavior.

The loader and efficient-attention helpers are byte-identical to saved41.
The same pinned Qwen3.5-27B revision, NF4/FP16, fixed 34/30 split, two free T4s,
private notebook, no CPU/disk parameter offload, and600s/2700s caps apply.
There are exactly five evaluation calls capped at 1,024 output tokens and one
separate 5,000-input/16-output memory probe. Input lengths are
2,612/3,686/3,697/2,470/1,440tokens. No training, adapters, paid compute,
public upload or active model change is permitted.

```bash
python3 scripts/build_reviewed_selection.py --version ghost_reviewed_selection_v1
.venv-tokenizer/bin/python scripts/verify_reviewed_selection.py --version ghost_reviewed_selection_v1 --tokenizer evaluation/ghost_reviewed_selector_inputs_v1/tokenizer_assets
```

After completion download the exact saved version's
`ghost_reviewed_selection_bundle/` and `ghost_reviewed_selection_results/` into
`evaluation/runs/ghost_reviewed_selection_v1/download/`. Use the dedicated
reviewed-selection verifier, not the audit, synthesis or memory-probe verifiers.
It checks frozen input/ledger provenance, tokens, scope, numerical health,
placement and backend coverage, then replays the exact archived selection and
rendering code in isolation. It requires every selected ID to render, and
rejects changed displayed text even if journal hashes were recomputed.

Ten new tests across ledger, input and runtime modules cover scope/metadata
preservation, arithmetic and absence typing, the unchanged historical guard,
withheld oracle/expected answers, strict ID-only selection, no factual rewriting,
blocked output, call limits and frozen replay. All 138 focused tests pass.
CPU preflight used synthetic ID outputs and the real tokenizer to exercise the
verifier; it did not perform model inference or establish model quality/GPU
execution. Synthetic artifacts were confined to temporary test directories.
The full suite of 411 tests retains the same two historical README snapshot errors; original
locks were not changed. Actual submission state is recorded in the handoff.

## Review after completion

First inspect metadata for unexpected reasoning channels and keep intermediate
reasoning private. Review actual selected summaries against the source and
original task, separating source wording, selection relevance, complete requested
coverage, unnecessary refusals and readability. Do not call immutable wording
proof of relevance or truth. Put inference-output review in the run directory;
do not modify the frozen curation/input review. Source-record quality remains
unblinded implementing-agent review unless a separate reviewer actually reviews it.
Neither a successful run nor complete selection permits automatic activation,
training or a claim of independent/human validation.

The frozen trial was submitted as private Kaggle saved43 after commit
`32dfc5e` was pushed. Fetched normalized notebook cell sources/types match
local input exactly. Status at confirmation: RUNNING. The heartbeat uses this
saved version and dedicated verifier. No actual selector-quality result is
claimed yet; active9B/v3, training and fresh-transfer gates are unchanged.
