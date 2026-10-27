# Reviewed-proposition rendering and coverage

The saved42 checker diagnostic falsely approved both new incorrect hearing
candidates even under focused questions. The three familiar pairs remained
correct. Free-form factual writing plus this self-checker is therefore not a
sufficient factual safeguard. This offline step inspects an alternative using
the existing opt-in v4 ledger's 42 reviewed propositions.

## Opt-in renderer

`scripts/reviewed_source_display.py` accepts source-record IDs only. It rejects
unknown or duplicate IDs and extra generated text. Up to12 selected records are
allowed; exceeding the bound raises an explicit error rather than silently
dropping records. Every selected statement is rendered verbatim from the
ledger, including its uncertainty wording. No factual rephrasing or model
approval is involved.

Records preserve source author/title/URL, composition-date evidence and interval,
observation date separately where applicable, attribution, status, anchor speaker
and role, exact anchor text/offset, full-packet fingerprint and context boundary.
The original bounded packets remain the surrounding context; anchors alone are
not a replacement for them. Changed statement or record fingerprints fail.
The displayed source-summary label distinguishes these reviewed paraphrases
from authentic source quotations.

The result keeps relevance, requested-task coverage and semantic assurance
explicitly unverified and makes no accepted-claim or memory updates. A reviewed
ledger can contain errors; these development reviews are not independent or human
gold. Selection can still be irrelevant or incomplete. This new module is an
opt-in inspection component, not an activated chat route or a fix to activev3.

## Offline audit

```bash
python3 scripts/audit_reviewed_source_coverage.py --output evaluation/runs/ghost_reviewed_coverage_v1
```

The audit reads the unchanged opt-in v4 source plan and the original five-question
evaluation specification. It uses manually inspected claim-ID lists to expose
representation and packing limits. These oracle lists are development analysis,
not measured model selection, automatic task-completeness scores, or a runtime
question router. They must not be supplied as answers to a later selector.

Two inspected six-record selections lose records after the old packing function's
first-four cap, before its omitted-fact accounting. In the Blumkin sequence this
drops reform and prediction records; in the two-letter sequence it drops the
Bessedovsky records. The new renderer preserves all six in each sequence.
This proves a packing limit for these selections, not that no alternative shorter
selection could answer a task. Active code and frozen controls were not changed.

Four of the five questions also have details absent from the selected immutable
statement forms despite being available in full source context: the precise
corrected March date, Eastman's own initiative/risk, two-source transport
confidence and bounded unknowns, and the ILO category values/arithmetic and local
information requirements. These are proposition-representation gaps, not proof
that retrieval failed or the corpus lacks evidence. Full task coverage has not
been established by the oracle displays.

The ignored report binds the plan, collection, ledger, original evaluation,
renderer and active-pointer hashes. Rendered examples, source records and source
anchors stay ignored. There are zero model calls, no uploads and no training.
Four new tests cover preservation of six statements/provenance, strict ID-only
selection, changed source/statement/date/speaker rejection, no trusted-memory
promotion, and separate observation/publication dates. All128 focused tests pass;
full401 retains the two known historical README snapshot errors; original
locks were not changed.

## Next gate

Before another GPU trial, fill only the justified representation gaps in a new
opt-in ledger. Preserve all original statements, packets, questions, expectations
and active pointers. Review new propositions directly against their source anchors
and distinguish arithmetic derivation from what the source explicitly states.
Then freeze a selector experiment that receives full passages and reviewed
propositions, but never the oracle ID lists or evaluation expectations. Measure
selection relevance and displayed task coverage independently of source wording.
No free-form factual rewriting or self-checker approval should promote a fact.
Fresh transfer, sustained conversation and voice remain later gates; training,
adapters and active9B/v3 remain unchanged.
