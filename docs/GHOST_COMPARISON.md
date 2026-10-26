# Fresh matched-source direct/checked conversation

This inference-only comparison follows the inspected ghost regressions. Five
newly authored dialogue turns run in each of two arms: ten displayed replies,
at most twenty evaluation requests, plus one separate 5,000-token memory probe.
Same private Kaggle notebook, pinned untouched base, packages, generation policy
and reference date; no adapters, training or paid compute.

`config/ghost_comparison_evaluation_v1.json` contains the new questions and local
review checks. The model payload omits those checks. Both arms receive the same
ILO and radio packets, in the same order, and the same six available reviewed
propositions. The context is fixed deliberately: this is not a test of natural
retrieval recall. Sources were already exposed during development and are not
claimed absent from pretraining.

The direct arm answers in plain prose with instructions to preserve certainty,
cite sources and distinguish new views. It has no semantic audit or canonical
factual renderer. Its status is explicitly `direct_unchecked`; generated claims
never enter verified-fact memory. The checked arm uses the current application,
including selection, canonical facts, interpretation and fallible auditing.
Those whole pipelines differ; individual component effects are not isolated.

Each arm receives its own actual displayed replies as history. Both model views
contain the most recent four messages with an omitted-message count. Source
context stays fixed, while checked selection may choose a subset of the six
available propositions. Direct runs first, then checked, with a fresh
conversation state for each. Do not claim randomized order, blinded scoring or
independent historian validation.

The five turns test separating findings from judgment, an invalid inference
from adoption barriers, a two-sentence study-only request, a return to imagined
reasoning about a conditional scenario, and shortening. New wording is retained
even where static routing suggests a weakness; those pre-run hypotheses are
recorded in the local review protocol. This is a purposeful robustness test,
not a representative benchmark. Once inspected, these questions become
development regressions.

## Reproduce and review

```bash
.venv-tokenizer/bin/python scripts/build_ghost_smoke.py \
  --version ghost_smoke_v9 \
  --evaluation-spec config/ghost_comparison_evaluation_v1.json
python3 -m unittest discover -s tests -p 'test_ghost*.py' -v
```

The existing builder preserves earlier notebooks and checks the ten-answer
limit. The runtime chooses `evaluate_ghost_comparison.py` only for the explicit
comparison purpose. It journals every request/reply and checks the twenty-call
maximum. The verifier replays the frozen comparison helper rather than the
ordinary smoke evaluator. Legacy smoke verification is retained and tested.

After downloading the exact saved version:

```bash
.venv-tokenizer/bin/python scripts/verify_ghost_smoke.py \
  --version ghost_smoke_v9 \
  --tokenizer training/data_revision_v2/tokenizer_assets
python3 scripts/export_ghost_transcript.py --version ghost_smoke_v9
```

Review every answer against the source. Compare source faithfulness,
unsupported premises, useful reasoning, refusals, two-sentence scope, mode
changes, depth, actual shortening and voice separately. Inspect rejected drafts
and audit reasons, not just approval counts. A direct answer can be useful but
unfaithful; a checked answer can be source-safe but fail the request. Do not
interpret either arm's status as a quality score. Keep all research outputs
local under the repository's publication boundary.

## Completed first comparison and regression

Saved Kaggle version 11 (`ghost_smoke_v9`) is verified: ten answers, sixteen
requests, no budget rejection/truncation, 11.32 GiB probe peak. Direct engages
with mode changes and shortening but adds unsupported certainty and omits the
requested date. Checked preserves canonical source limits but fails mode,
styling or review coverage. Neither whole pipeline is promoted. Detailed local
review: `evaluation/runs/ghost_smoke_v9/QUALITY_REVIEW.md`.

These questions are now inspected regressions. Prepared v10 keeps the exact
profile, source collection, ledger, model cases and direct evaluator. Its
checked select call returns facts/view/mixed plus detail; all stages see full
bounded context. It also normalizes narrow styling, preserves valid reviewed
segments when others lack verdicts, and materializes selected modern facts for
mixed replies. The direct control is generated again rather than copied. This
is a combined application intervention, not a component ablation. Use a new
version when building; preserve the earlier frozen comparison.
