# Present-day ghost smoke: protocol and continuation

This is the next bounded test of the active fictional present-day persona.
It makes no training updates and attaches neither experimental adapter.

## Frozen experiment

The bundle contains six single questions and one four-turn conversation from
`config/ghost_chat_evaluation.json`: ten displayed answers, at most thirty
evaluation-stage requests, and one separate memory probe. Source checks and
rubrics are withheld from model cases. The same seven historical packets,
dated Runpod vendor excerpt and 33 reviewed propositions are used. The model,
revision, NF4 stack and nonthinking greedy policy remain pinned.

Before evaluation, an unscored 5,000-token synthetic request generates at most
16 tokens to test prefill memory. It never becomes a conversation turn or a
quality score. Real stage inputs have a 5,120-token cap. Over-budget requests
are recorded without calling the model; incomplete stage output is blocked
before display. These are operational outcomes, not successful semantic checks.
The generation cap is 1,024 tokens per evaluation call. CUDA cache is released
between calls; no source paragraphs or model inputs are silently truncated.

The failed 1940 v2 trial is a separate archive. It had 14 direct-control
answers, no checked answers and a T4 OOM at 6,546 input tokens. The ghost smoke
tests compact context in the actual runtime rather than assuming the tokenizer
budget establishes GPU memory capacity.

## Rebuild and inspect

```bash
.venv-tokenizer/bin/python scripts/build_ghost_smoke.py
python3 -m unittest discover -s tests -p test_ghost_smoke.py -v
```

The complete local research archive and tokenizer assets are required.
The builder preserves frozen files and refuses a changed rebuild. Its initial
preflight draft is archived separately; the final bundle adds explicit probe
failure reporting and separate counts for inference versus budget rejections.
It preserves the existing Kaggle notebook title and numeric identity to avoid
renaming its URL again.

Prepared local artifacts:

- `evaluation/ghost_smoke_v1/protocol.json`, `evaluation_spec.json`, `bundle_lock.json`.
- `training/kaggle_cli/ghost_smoke_v1/` and `notebooks/kaggle_ghost_smoke.ipynb`.
- `scripts/evaluate_ghost_smoke.py`: immediate per-stage journaling and replay.
- `scripts/kaggle_ghost_smoke.py`: pinned GPU runtime and memory probe.
- `scripts/verify_ghost_smoke.py`: frozen-code local replay and tokenizer checks.

After the initial automatic-review block, the user explicitly approved this
smoke and bounded follow-up iterations on 26 October 2026. This covers the
same private notebook, project helpers/persona prompts, reviewed public-source
excerpts/propositions and evaluation questions, with at most thirty evaluation
calls plus one short memory probe per run. It excludes training, adapters and
paid compute. The builder rejects case sets exceeding ten answers.

Saved Kaggle version 4 ran at the original private notebook URL.
The frozen persona reference date is 25 October; execution is on 26 October.
Keep that distinction in reports rather than silently changing a running
experiment. Use `--version ghost_smoke_v2` on the builder and verifier for a
subsequent changed experiment; previous bundles stay immutable.

### Results and bounded follow-up

Version 4 completed: the 5,000-token probe passed at 11.32 GiB peak and all
journals replay and tokenize exactly. Its sixteen evaluation generations
produced eight structural blocks, one appropriate missing-evidence refusal and
one inaccurate substantive answer. Four selection errors and four draft
structural failures are separated from semantic quality; no memory or token
budget failure occurred. The model checker incorrectly approved the wrong
observation date in the substantive answer.

The local reports and records are in `evaluation/runs/ghost_smoke_vN/`.
V2 (saved version 5) completed with nineteen calls: three partial answers, five
refusals, one approved and one rejected. It fixes the canonical observation date
and source retention but exposes an ambiguous answerable boolean and complex
drafting failures. V3 (saved version 6) also completed with nineteen calls:
two approved, one identity clarification, one rejected, five schema blocks and
one refusal. Five drafts omit only the empty quotes field; opinions still
retell old disputes and add unsupported premises. Only three responses fulfill
their request. One approved vendor-fact answer is irrelevant to endorsement.

V4 targets these observed failures with optional empty quotes, direct canonical
fact materialization, identity clarification without a prior-success dependency,
and explicitly summary-only drafting. Selection/audit retain full blocks and
all traces preserve the original retrieval. The same ten inspected cases remain
development regressions. These are combined interventions, not ablations or
fresh generalization measurements. New source coverage remains deferred.

## Verification and decisions after the run

Download the specific saved version into `evaluation/runs/ghost_smoke_v1/download/`
and verify it without loading model weights locally:

```bash
.venv-tokenizer/bin/python scripts/verify_ghost_smoke.py \
  --tokenizer training/data_revision_v2/tokenizer_assets
```

The verifier checks submitted/downloaded payloads, preserved artifacts,
generation bounds, the memory probe and both journals. It replays the frozen
application in a separate local process, so a later active-code change cannot
silently redefine the old run. All prompts and decoded model replies can be
reconstructed through the tokenizer.

Review every displayed answer and candidate segment against the source. Count
real factual failures, rejected supported content, missing modern evidence,
fiction/history confusion, citation/quotation errors, refusals of answerable
questions, input-budget blocks and malformed/truncated stages separately.
Inspect checker approvals and rejections instead of treating them as gold
judgments. Review usefulness, voice, explanation depth and continuity alongside
source faithfulness. Any review by the authoring agent is unblinded and not
independent historian validation.

If the memory probe fails, record it and adjust context/runtime or reassess
hardware before another bounded run. If structured output fails, fix the
specific schema interaction and measure it again. If safe answers mostly
refuse, improve evidence selection and coverage rather than claiming grounding
success. If supported answers are useful and the checker handles the observed
traps, expand to additional reviewed contemporary source families and longer
conversations. Preserve all inspected questions as development regressions.
Do not resume training merely because the application executes successfully.

## Git milestone

Implementation commit `14576a3` and smoke-workflow commit `cfbce01` are published
on GitHub main. HTTPS pushing had no usable credentials. At the user's request,
`origin` now uses SSH. GitHub accepts the existing public key, but signing fails
without an unlocked agent in this session; `gh` reports an invalid token.

The connected GitHub account provided a working publication route: the reviewed
file trees were uploaded and checked against their exact local tree SHAs, then
committed with expected-head leases. Local unpublished counterparts `e3ffbce`
and `033c720` were rebased onto the byte-identical published equivalents. Main
is synchronized and the SSH remote remains configured. An SSH unlock is not
needed for this completed publication, though future direct SSH pushes still
need a usable agent. Do not ask for or record passphrases or tokens. Research
source excerpts and model artifacts remain excluded.

On 26 October the user re-authenticated `gh`. A network-enabled check confirmed
the account is valid, and documentation commit `af18185` was pushed using the
`gh` credential helper with an explicit HTTPS push URL. Origin remains SSH.
Future publishing can use this working helper without copying credentials into
code, arguments or logs.
