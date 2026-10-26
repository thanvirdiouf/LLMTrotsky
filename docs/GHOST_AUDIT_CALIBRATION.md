# Speaker and source-role calibration

The verified v11 comparison removed ordinary opinion JSON failures, but exposed
reviewer role confusion. Its neutral auditor sometimes treated the fictional
assistant's documented lifetime references as claims by the human user. It also
attributed new political judgments to the ILO merely because the application
cited the ILO as background evidence. This does not make every rejected draft
valid: causal claims, broad assertions and unsupported premises still require
inspection.

The correction explicitly identifies the candidate speaker, historical starting
points and modern descriptive premises. A normative conclusion need not be
endorsed by the source that supplies its factual premises. A conditional scenario
can govern its paragraph without repeating “if” in every sentence. Extra real
outcomes, invented motives, false authorship and source misattribution remain
errors. The auditor is still the same fallible base model.

`config/ghost_audit_calibration_v1.json` preregisters ten candidates: five valid
and five invalid. Expected decisions and rationales are withheld from all model
requests. Valid examples cover fictional first person, conditional reasoning,
normative preferences, dated findings and publication versus observation dates.
Traps cover false measured job loss, false institutional endorsement, invented
user events/motives, authentic posthumous authorship and impossible chronology.
The authoring agent specifies the contract; this is not independent gold scoring
or a claim that one modern political opinion is historically correct.

V12 uses the same private notebook, pinned untouched base, source profile and two
fixed ILO/radio packets. It makes ten checker-only requests plus one separate
5,000-token memory probe. No selection or drafting occurs. Test candidates,
including deliberately false statements, bypass factual canonicalization only
inside this isolated diagnostic. They are never trusted chatbot replies or
verified memory. The displayed diagnostic output is the audit JSON. No training,
adapters or paid compute. This fits the user's existing bounded-run approval.

The evaluator uses the production check request and validation code. Every call
is journaled immediately; exact replay forbids new inference. The frozen builder,
runtime and verifier select this evaluator only for `ghost_audit_calibration`.
Report false acceptances, false rejections and structural blocks separately;
inspect reasons and do not count a block as a correct semantic rejection.
Candidates become inspected development cases once executed.

```bash
.venv-tokenizer/bin/python scripts/build_ghost_smoke.py \
  --version ghost_smoke_v12 \
  --evaluation-spec config/ghost_audit_calibration_v1.json
python3 -m unittest discover -s tests -p 'test_ghost*.py'
```

After exact saved-version download, run the ordinary frozen/token verifier with
`--version ghost_smoke_v12`. Keep raw answers, candidate displays, local scores
and source-bound reasons under `evaluation/runs/ghost_smoke_v12/`. Compare
structural execution success separately from semantic decisions. Ten designed
examples cannot establish a reliable production auditor. A clean calibration
would justify a fresh conversation study, not deployment or resumed training;
continued errors would justify independent human review or a separately tested
reviewer before more prompt tuning.


## Completed result and next study

V12 completed as private saved version 14: ten checker generations, no input
budget rejection or truncation, 11.32 GiB probe peak. Frozen replay and exact
local input/output token reconstruction pass. It accepts all five valid
candidates and rejects all five traps, with zero blocks under preregistered
contract labels. Some rejection explanations still add loose secondary
reasoning; correct decisions are not proof that every rationale is sound.
Detailed local review/scores/transcript are in `evaluation/runs/ghost_smoke_v12/`.

This result justifies a fresh complete-pipeline study. V13 uses
`config/ghost_chat_evaluation_v3.json`: four singles and a six-turn conversation,
ordinary natural retrieval rather than fixed source context, ten replies and
at most thirty calls plus one probe. Singles test a historical prediction,
reported information versus firsthand witness, missing measured modern outcomes
and an unrelated election premise. Dialogue tests dated evidence, fear versus
actual events, conditional political reasoning, deeper disagreement, study-only
scope and shortening. Review checks remain local and withheld; inspect actual
source routing, selected claims, drafts, audits and displayed answers separately.
These questions are new wording over exposed sources, not independent gold.
Training and adapter promotion remain paused.
