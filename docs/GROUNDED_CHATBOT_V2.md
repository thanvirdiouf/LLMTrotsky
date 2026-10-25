# Grounded chatbot v2 — 25 October 2026

This historical-persona version is preserved as an incomplete experiment.
Its approved Kaggle trial exhausted T4 memory after 14 direct-control answers,
before the checked arm. The active project has moved to
[the present-day Trotsky ghost](PRESENT_DAY_GHOST.md). No v2 quality improvement
has been established by the partial run.

V2 implements the requested fixes in separate files. V1 code, source collection,
notebook, scores and both saved adapters remain unchanged. No further training
is selected. This version runs the untouched base through evidence selection,
drafting, a model-assisted support audit and deterministic answer rendering.

## Run and inspect

```bash
python3 scripts/grounded_chat_v2.py --dry-run \
  --question "Explain the Rakovsky hypothesis briefly."
python3 -m unittest discover -s tests -p 'test_grounded_chat*.py' -v
python3 scripts/build_kaggle_grounded_chat_v2.py
```

For interactive use with a configured remote endpoint serving the untouched
base model:

```bash
python3 scripts/grounded_chat_v2.py --base-url https://YOUR_SERVER/v1 \
  --log /tmp/trotsky-chat-v2.jsonl
```

The endpoint uses the existing `TROTSKY_MODEL_API_KEY` environment variable.
It is not logged. Endpoint model identity is operator-asserted; the HTTP client
cannot attest model revision or adapter absence. The Kaggle runner verifies
the pinned snapshot and frozen base parameters. No weights run locally.

## Source state and evidence records

`scripts/grounded_chat_v2.py` retains active source IDs across follow-ups, rather
than depending on the last two user turns. Explicit names and topic switches
reset the active source. A source packet is kept complete or dropped as a
whole when the budget cannot fit it. The rule-based topic resolver is intended
for this seven-packet scope; ambiguous references and unseen subject names
remain evaluation cases rather than solved language understanding.

`config/grounded_chat_claims_v2.json` contains **31 reviewed propositions** with
source packet, composition date, attribution, certainty status and exact source
anchors. The builder checks their anchor text, speaker and date against the
original collection. This mechanical validation does not prove the propositions
are faithful paraphrases: they have single coding-agent source curation and
need independent historical review before stronger claims of reliability.
These are evidence records, not training examples or model-generated memories.

The hypothesis remains a hypothesis, a reported allegation remains attributed,
and a prediction remains dated. Only known claim IDs can be selected; a model
cannot rewrite their canonical metadata. Conversation state records only the
claim IDs surviving the current answer's checks. Earlier assistant prose and
user disagreement never become new evidence. Source state is retained within a
chat; traces serialize it for inspection. The CLI does not automatically resume
an old chat after restarting.

## Checks before display

Each answer uses at most three base-model calls:

1. Select relevant existing claim IDs, or return insufficient evidence.
2. Draft prose segments tied to those claims, or request exact source anchors
   for quotations.
3. Audit every complete segment for support, certainty, attribution, temporal
   limits, relevance and unsupported additions, against both propositions and
   original source context.

Strict parsing rejects duplicate JSON keys, malformed output, unknown claims,
missing check coverage and incomplete generations. Model-authored citation
markup and direct quotation delimiters are rejected in prose. Quotations are
inserted from the original anchor bytes, with the recorded speaker; citation
IDs are inserted from the referenced source packets. Thus a model cannot type
`[eastman]` as the citation for a hearing claim.

Only segments passing every check are rendered. Rejected segments are kept in
the trace, omitted from display, and excluded from accepted-claim memory. If no
segment survives, the application gives a scoped refusal without an invented
explanation of why an event could never have occurred. Malformed/truncated
stages fail closed without a retry. Operational/network failures propagate
instead of being disguised as a valid insufficient-evidence judgment.

**The support checker is fallible.** It uses the same base model and can approve
an unsupported sentence or reject a good one. Structural validity and exact
citations are enforced in code; semantic faithfulness and answer usefulness
still require source-based evaluation. A claim ID accompanying prose is not
an entailment proof. The renderer makes no semantic-guarantee claim.

## Verified local results

All **21 grounded-chat tests pass**, including 12 new v2 tests. They cover long
follow-ups and explicit topic changes, exact quotes/citations, proposition
integrity, rejected additions excluded from display and memory, incomplete
stage handling, no automatic network retries, matched retrieval between arms,
and journal replay without repeating generation calls.

The full project suite runs **273 tests with the same two pre-existing errors**:
historical notebook fixtures reject the unchanged `evaluation/README.md` against
the old pilot snapshot lock. That file and the historical locks were preserved.

Offline source-recall regression improves from **20/21 to 21/21** on v1's
source-backed single and conversation turns. The former lost Rakovsky packet
is now retained. This is retrieval availability, not a model-quality result.
The tokenizer preflight reaches at most **6,635 input tokens** with synthetic
360-word prior replies, below the 8,192-token budget. Runtime still validates
the actual tokenized input and never silently truncates it.

## Prepared controlled comparison

`config/grounded_chat_evaluation_v2.json` freezes eight new single questions and
two four-turn conversations: **16 turns per arm, 32 displayed answers total**.
Both arms receive identical persistent-source retrieval, original packets and
reviewed evidence propositions. Compare direct single-pass generation with the
select/draft/check/render pipeline. Retrieval changes are assessed separately
through the offline v1 regressions. The new comparison uses the same pinned
base and packages, with a matched 1,024-token per-call cap; it is not a rerun of
the old 512-token experiment.

The pipeline can use up to three calls per turn, so the whole comparison has a
maximum of **64 model calls**. It records every stage, blocked draft, model
token list, rendered answer, source trace and state transition. Completed
model calls are journaled immediately and replayed on exact-identity resume;
answer journals are separately checked. No model check can silently become an
independent quality score.

The questions are newly written before v2 inference, but all seven source
packets were previously exposed and claim curation followed v1 failures. Treat
this as development evidence, not unseen-source validation. The review must
measure supported-answer usefulness, refusal rate, explanation depth and voice,
as well as factual errors, citation/quote fidelity, blocked stages and latency.
Record genuine failures of the semantic checker, not only failures it catches.

Frozen artifacts and submission folder:

- `evaluation/grounded_chat_v2/protocol.json` and `bundle_lock.json`.
- `evaluation/grounded_chat_v2/preserved_v1_and_adapters.json`.
- `evaluation/grounded_chat_v2/retrieval_regression.json` and
  `tokenizer_preflight.json`.
- `notebooks/kaggle_grounded_chat_v2.ipynb`.
- `training/kaggle_cli/grounded_chat_v2/`.

After an initial automatic approval block, the user explicitly approved the
new source/claim/evaluation payload and bounded comparison. **Saved version 2
failed with CUDA OOM** on the previously used private notebook, preserving version 1.
Kaggle renamed the notebook URL when the submission title gained a v2 suffix:
the current slug is `thanvirdiouf/trotsky-grounded-base-chatbot-v2`. Both saved
versions' embedded payloads were downloaded and verified against their frozen
local bundles; the notebook remains private, with numeric identity 140792344.
The original title/URL was restored using a no-run save, version 3, without
extra inference. The partial run and exact failure are archived in
`evaluation/runs/kaggle_grounded_chat_v2/`.

`scripts/verify_grounded_chat_v2.py` checks the downloaded bundle, frozen model
identity, both journals and all state transitions through inference-free local
replay. It can also reconstruct every prompt and decode every model stage with
the local tokenizer. This complete-run verifier cannot certify the incomplete
trial. No checked-arm drafts exist, so no full pipeline quality review is
possible. No independent or blinded review is claimed. The convention was
recorded before answer inspection in
`evaluation/runs/kaggle_grounded_chat_v2/review/review_protocol.json`.

V1 peaked at 11.2 GiB and recorded 13.6 minutes of generation for 32 replies;
that did not establish capacity for v2's longer prompts. V2's 6,546-token input
exhausted the T4's 14.56 GiB during prefill despite passing the tokenizer budget.
The active ghost version reduces duplicated context, but needs a real memory
smoke test. An A40 could help capacity and interactive iteration; no paid
compute has been rented or provisioned here.
