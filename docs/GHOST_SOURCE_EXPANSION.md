# Source expansion and six-turn evaluation

Prepared and activated as development evidence on 26 October 2026 after v4
review. The original eight-packet collection and all frozen smokes are preserved.
V4 improves factual answers but still approves unsupported opinion premises;
this expansion targets relevant basis coverage, not a quality promotion.
Building an explicit plan does not change the active pointer.

`config/ghost_source_plan_v2.json` selects a reference-date profile and local
reviewed source inputs. The original eight packets and 33 claim records are
unchanged; the candidate collection has ten packets and 39 propositions.

| Addition | Source date | Observation/provenance | Boundary |
|---|---|---|---|
| Trotsky, *Radio, Science, Technique and Society* | Delivered 1 March 1926 | Existing raw HTML hash and exact extracted block fingerprint | Complete contiguous speech section; later publisher introduction excluded |
| ILO, *How might generative AI impact different occupations?* | Published 20 May 2025 | Observed 26 October 2026; bounded transcription hash | Three complete paragraphs; omitted regional comparison and charts recorded |

The [1926 speech](https://www.marxists.org/archive/trotsky/1926/03/science.htm)
provides a more relevant basis for technology questions than the initial
organizational letters. Its political arguments remain attributed arguments;
its forecast remains a prediction. Marconi's view is a report within Trotsky's
speech, not an independently verified transcript. The later publisher's
introduction cannot become Trotsky's own retrospective knowledge.

The [ILO article](https://www.ilo.org/resource/article/how-might-generative-ai-impact-different-occupations)
supplies a modern research example with explicit limits. Publication,
observation, potential exposure and prospective assessment stay distinct.
It does not supply verified 2026 outcomes. The research context is also available
in [ILO Working Paper 140](https://www.ilo.org/publications/generative-ai-and-jobs-refined-global-index-occupational-exposure).

These are same-agent curated sources, not independent historian review. New to
this collection does not mean absent from the base model's pretraining.
Source transcriptions, anchors and reviewed statements remain ignored local
research inputs. The public repository contains the plan, tooling and case
design, without publishing those excerpts.

## Reproduce without activating

```bash
python3 scripts/build_ghost_chat.py \
  --source-plan config/ghost_source_plan_v2.json \
  --output evaluation/ghost_chat_v2
python3 -m unittest discover -s tests -p 'test_ghost*.py' -v
```

The builder checks raw historical bytes, contiguous blocks, extracted-context
fingerprints, dates, claim anchors/speakers and historical basis membership.
Modern inputs require exactly one publication or update date, plus a separate
observation date; both must be within the selected reference date. Existing
generated evidence is immutable: changes require a new output version.

`config/ghost_chat_evaluation_v2.json` prepares four single questions and a
six-turn conversation. It tests misleading exposure claims, absent recent
outcomes, prediction versus later fact, vendor scope, disagreement, facts-only
mode, greater depth, identity and shortening. Expected checks remain outside
the model case payload. Once responses are inspected, these become development
regressions. Report source faithfulness and usefulness separately.

The smoke builder accepts `--evaluation-spec config/ghost_chat_evaluation_v2.json`
with a new `--version ghost_smoke_vN`. It still reads the explicitly active
profile/collection; choosing a case file does not activate new evidence.
The deliberate update to `config/active_chat.json` selects evidence v2 and
the 26 October profile. A synthetic preflight peaks at 3,464 tokens, below
the 5,120 runtime cap; actual histories still need runtime checks. The new
smoke must be reviewed under the existing inference-only limits.

Follow-up preparation also corrects a deterministic routing issue: an explicit
facts-only request suppresses the prior opinion mode and keeps the prior source
topic. Selection and audit receive date evidence and source boundaries;
summary-only drafting is labeled. These changes are subsequent to frozen smoke
v4, so its results cannot validate them.

For new opinions, retrieved approved basis IDs are explicitly added when the
model omits them, logged as `application_added_basis_claim_ids`; interpretation
segments must reference a historical basis. This enforces provenance membership,
not semantic entailment. The checker still needs source-based review.

V5 completed as saved Kaggle version 8. Its four single cases preserve source
limits, but conversation still fails. The questions are now inspected
regressions. V6 is preserved unrun; the next combined v7 test uses a neutral
auditor and removes unrelated historical context from study-only answers.
The source expansion does not by itself establish reliable interpretation.
