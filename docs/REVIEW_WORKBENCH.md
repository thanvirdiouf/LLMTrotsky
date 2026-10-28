# Experimental chat review workbench

The user explicitly requested request-to-retrieval integration and a chat prototype
for their review. This is a new end-to-end review workflow, with the earlier
familiar selector-prompt stop rule preserved. Active9B/v3, both experimental
adapters, frozen cases and paused training remain unchanged.

The local interface has three clearly distinguished capabilities:

- **Review conversations:** real, verified Kaggle-generated replies, source
  passages and a local feedback form. Mechanical verification is not factual
  approval. Until the new run completes, samples are shown as pending.
- **Source lookup:** interactive BM25 retrieval over the bounded collection. It
  runs locally without a model and labels its results as candidate passages.
- **Live chat:** an adapter for an explicitly configured external endpoint. It is
  unavailable by default; Kaggle batch evaluation is not an always-on server.
  No model weights are downloaded or run locally.

Run the local-only server:

```sh
.venv-tokenizer/bin/python scripts/review_chat_server.py --port 8765
```

Open `http://127.0.0.1:8765`. The server binds only loopback, checks Host/Origin,
serves a fixed asset allowlist and renders source/model text as text. API keys stay
server-side. The default source snapshot is the opt-in53-record ledger; the
active application configuration is untouched. Feedback is append-only JSONL at
`evaluation/review_workbench/feedback.jsonl`, ignored by Git, stored on this
computer and never automatically uploaded. Review notes are human-submitted
feedback; they do not become training data or verified source facts.

Optional live operation requires an operator-provided URL and matching local
public tokenizer assets, for example `--base-url URL --model Qwen/Qwen3.5-9B
--tokenizer PATH`. The UI identifies the configured destination before a live
request. Endpoint model identity is operator asserted, not remotely attested.
Use `TROTSKY_MODEL_API_KEY` only if that endpoint requires it. No endpoint is
configured or contacted in the default build; no paid service is provisioned.

## Evidence and conversation flow

`grounded_request_boundary.py` supplies an explicit current question and opaque
prior turns. Source/detail UI controls are separate application events. The new
`review_chat.py` scores complete source packets with the existing BM25 engine,
adding bibliographic author/ID text only to its search index, then restores the
unchanged original packets. It filters the ranked candidates by configured source
scope before choosing at most two packets. Global corpus statistics are retained
so selecting one source does not collapse BM25 scores. Prior model replies never
enter the retrieval query. Lexical candidates are not proof of answerability.

The pinned Qwen template requires one initial system message. Persona, policy and
explicitly labelled documentary JSON occupy it; original prior turns and the
actual current user question remain separate conversation messages. Sources
retain author, dates, observed date, speaker role and full bounded context.
Reviewed summaries retain status/attribution. This prompt boundary does not
establish prompt-injection resistance or factual support by itself.

Token budgeting preserves complete packets and the latest conversation pair.
Older whole pairs or a lower-ranked whole packet can be omitted with an explicit
trace. It fails rather than crop the current question, a source caveat or the
latest pair. Model output contains source-based `reply` and separately labelled
`fictional_reflection`. Source-only mode requires empty reflection. Invalid JSON,
extra fields, truncation or reasoning-channel markers block display; no automatic
retry occurs. Citation checks are mechanical diagnostics, not an entailment gate.
No self-audit approval or trusted factual memory is used.

## Bounded sample run

Design: `config/ghost_review_workbench_v1.json`. Final prepared local version:
`ghost_review_chat_v3`. Earlier unsubmittedv1/v2 preparations are preserved; neither
was pushed to Kaggle. The design has five conversations and ten replies total:
technology/ILO/disagreement, Eastman/payment/hypothetical press offer/shorter answer,
misleading private quotation, fictional identity and concise source-only facts.
It uses real retrieval with no oracle source IDs or supplied expected answers.
This is a development review set, not an independently held-out benchmark. The
fixed `ghost_mapped_transfer_v1` questions remain unrun.

The review candidate is the approved pinned Qwen3.5-27B on two free T4s, separate
from the active9B baseline. At most ten evaluation calls/ten model replies plus
one5000-input/16-output probe;1024output,600s setup and2700s workload caps. Common
loader/placement/attention/generation code is unchanged. No training, adapters,
paid compute, local weights, public upload or active-model replacement.

After completion download the exact saved version's `ghost_review_chat_bundle/*`
and `ghost_review_chat_results/*` into the run's `download/` directory, then:

```sh
.venv-tokenizer/bin/python scripts/verify_review_chat.py \
  --version ghost_review_chat_v3 \
  --tokenizer evaluation/ghost_review_chat_v3/tokenizer_assets
```

The dedicated verifier checks frozen scope/source identity, token/logit/backend
coverage and exact dependent retrieval/history/parser replay from archived code.
The UI exposes only final parsed replies and sources after that verification;
raw model journals, reasoning and system prompts are not served. A blocked
format is shown as a failure, not replaced with a fabricated reply. Independent
source support, relevance, completeness and voice still require review.

Six new focused tests cover current-question/context separation, scoped retrieval,
whole-context budgets, reflection/format privacy, dependent ten-turn execution,
verified sample import and local-only feedback. CPU preflight uses synthetic
outputs in temporary fixtures and a real tokenizer, never a live model; it tests
replay and tamper rejection only. Actual generated samples are never replaced
by the synthetic preflight.

Browser verification at the normal in-app viewport exercised source lookup,
publication versus observation date, full speaker-labelled context and a native
source dialog. HTTP smoke tests reject cross-origin POSTs, foreign Host headers,
arbitrary file paths and an unconfigured live model call. No fake human review
was saved. All170 focused tests pass (158ghost,6boundary,6new review tests).

The final frozen sample was submitted once as private Kaggle saved47 after
commit `0042729`; fetched normalized source matches. At submission the run is
pending, so no model-quality result is claimed. The interface reads actual
verified samples dynamically when verification is available.

## First sample run and technical continuation

Saved47 completed and passed frozen token/runtime/dependent replay verification.
It made7calls plus the probe, with5parsed first replies,2plain-prose follow-ups,
and3dependent turns not generated. The model history had converted prior JSON
outputs to prose. The technical correction preserves the JSON field boundary in
model history while the browser still renders prose. No source wording, question,
prompt, model, decoding or budget increase is introduced.

A new local version `ghost_review_chat_v4` reuses the five structurally valid
first replies from exact saved47, bound by original call/result hashes and exact
initial requests. It permits only5new model calls for failed/missing follow-ups,
plus the usual probe. It never reruns/resumes saved47 or regenerates the five
preserved replies. The verifier replays both archived seeds and new calls, and
reports their origins separately. Facts in the archived replies remain
unvalidated; structural validity is not semantic support. Existing source review
notes flag overclaims for human inspection, and no answers are silently edited.

Feedback binds each human submission to the exact run, answer journal hash and
reply fingerprint, so reviews remain attributable when a later technical run
preserves a reply. The local browser submitter is not authenticated as an
independent expert; ratings do not automatically become validated training data.
