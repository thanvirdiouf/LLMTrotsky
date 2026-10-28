# Original-question context label counterfactual

Saved45 obeyed only one of three scope/background controls. Two source-scope
targets produced factual records addressing the full original question. Its
nested data calls that full question `current_user_request`, although the system
asks for a single `selection_target_span`. This diagnostic tests that specific
input ambiguity. It does not establish in advance that the label caused failure.

Tracked design: `config/ghost_reviewed_target_context_v1.json`. New local version:
`ghost_reviewed_target_v2`; inputs: `evaluation/ghost_reviewed_target_inputs_v2`.
The intervention changes exactly one JSON key name and its single system-prompt
reference: `current_user_request` becomes `original_question_context`. Key order,
all data values, source/menu text, five original questions, target spans and
output schema remain unchanged. The runtime/parser/renderer/loader/attention/
generation/probe are byte-identical to the earlier runs. CPU preparation and
verification reconstruct the transformation and reject any additional change.

The five target-span diagnostic outputs are not five complete original-question
answers. Three controls expect no records for scope/background; two factual
probes check limited required-record membership. Local expectation bytes are
identical to saved45's frozen expectations, and only their hash is uploaded.
Model messages contain no expected labels, required-record lists or old outputs.
Passing membership does not establish completeness or exclude irrelevant cards.
The Blumkin broad-context versus direct-conversation caveat in the saved45 review
remains; the counterfactual does not weaken or silently rewrite its expectation.

Scope: five selector calls/five diagnostic displays plus one5000-input/16-output
probe; same input counts2669/3774/3763/2529/1497,1024output,600s installation,
2700s workload, private same notebook, pinned27B and two free T4s. No training,
adapters, paid compute, local model weights, public upload, source changes or
active9B/v3 replacement. This is one planned run, without automatic retry or
budget increase. If the scope controls still fail, stop selector-prompt retries
on these familiar cases and review architecture/expectation scope offline.

After completion download the exact saved job's `ghost_reviewed_target_bundle/*`
and `ghost_reviewed_target_results/*` to the new version's run `download/`, then:

```sh
.venv-tokenizer/bin/python scripts/verify_reviewed_target_context.py \
  --version ghost_reviewed_target_v2 \
  --tokenizer evaluation/ghost_reviewed_target_inputs_v2/tokenizer_assets
```

Use this dedicated counterfactual verifier, not the older target verifier. It
checks frozen source/runtime/input identity, the exact rename, identical local
expectation bytes, real tokens/scope/logits/backend and isolated exact replay.
Inspect metadata before final JSON, keep intermediate reasoning private, and
review actual source/target relevance separately from the local checks. Put the
new unblinded implementing-agent review in the new run directory; all earlier
frozen curation and input reviews remain unchanged. No result is independent or
human validation. Fresh transfer and activation remain gated.

Three new tests check the exact key/reference-only transformation, preserved
values/order/context without expectation leakage, rejection of ambiguous or
changed targets, and no silent source truncation. The CPU verifier preflight uses
synthetic IDs and a real tokenizer with temporary archived-metadata fixtures;
it makes no model calls and establishes neither GPU execution nor model quality.
