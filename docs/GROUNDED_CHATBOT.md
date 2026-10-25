# Grounded base chatbot prototype — 25 October 2026

This guide describes the preserved 1940-persona v1 experiment. The active
project is now [the present-day Trotsky ghost](PRESENT_DAY_GHOST.md), without
the old knowledge cutoff.

The first end-to-end run is complete: **32/32 answers verified**, no truncation,
no adapter attached and no training. The same-agent, unblinded review exposes
retrieval, interpretation and conversation failures. Both adapters remain
experimental and further training remains paused.
[Full results and limits](../evaluation/runs/kaggle_grounded_chat_v1/QUALITY_REVIEW.md).

The prototype connects retrieval to the untouched, pinned Qwen3.5-9B base and
keeps a trace for every answer. Both adapters remain experimental. Training
remains paused until retrieval, interpretation and conversation failures have
been distinguished. No adapter is attached by this runner.

## Run locally without loading model weights

The complete local research archive is required to build the collection. A
fresh Git clone contains code and specifications, but excludes historical
texts and generated notebooks under the existing distribution policy.

```bash
python3 scripts/build_grounded_chat.py
python3 scripts/grounded_chat.py --dry-run --question "What does the Rakovsky article actually establish?"
python3 -m unittest discover -s tests -p test_grounded_chat.py -v
python3 scripts/build_kaggle_grounded_chat.py
```

For an interactive chat with an already configured remote service serving the
untouched base, use:

```bash
python3 scripts/grounded_chat.py --base-url https://YOUR_SERVER/v1 --log /tmp/trotsky-chat.jsonl
```

The remote API client reads `TROTSKY_MODEL_API_KEY` from the environment. It does
not write credentials into transcripts. The endpoint operator must attest that
it serves the specified base: a model name in an HTTP request cannot verify
adapter absence or the actual model revision. The Kaggle runner verifies the
pinned snapshot, package stack, tokenizer template, frozen parameters and lack
of attached adapters. All model inference stays remote.

The CLI accepts an optional `--question` for one turn and otherwise runs an
interactive conversation. An empty line exits. It supports ten previous turns;
larger histories and token overflows fail explicitly instead of silently
dropping conversation or evidence. A new transcript path is required to avoid
overwriting previous chats. Truncated answers are flagged and never silently
fed into a subsequent turn.

## Bounded evidence and certainty

[`config/grounded_chat_sources.json`](../config/grounded_chat_sources.json)
allowlists six works already inspected in grounding development and one
contiguous hearing excerpt. The collection contains **seven context packets**,
not the bulk archive. The build checks raw HTML hashes, reviewed extracted
block hashes, exact annotation anchors and composition dates against the
19 August 1940 cutoff. This is a single coding-agent review, not independent
historian validation or permission to use whole works for training.

BM25 ranks complete context packets and returns at most two. A packet contains
the reviewed surrounding paragraphs, including caveats, address and signature
lines, or adjacent hearing questions and answers. A character budget drops an
entire packet rather than cropping away qualifying evidence. Composition-date
evidence, publication metadata, source URLs, author, individual speaker labels
and raw-source hashes remain attached. The Eastman editorial introduction is
excluded. The New Leader collective declaration is explicitly attributed.

Anchored annotations identify specific hypotheses, reported allegations,
predictions, limitations and documented statements. These are not a blanket
classification of every sentence. The prompt requires the model to preserve
those distinctions: documentation that Trotsky made a statement does not itself
independently establish every historical assertion in it. The model must also
distinguish an allegation from its denial, and the 1930 account's predictions
from later outcomes. These are instructions to test, not a guarantee of correct
claim interpretation.

Every answer records the current question, actual preceding replies, retrieval
query, candidate scores and matches, source packets, exact model messages,
prompt hash, response and completion status. Kaggle adds input/output token IDs
and runtime identity. Packet citations such as `[rakovsky]` map back to the trace.
Citation checks flag unknown IDs and missing citations; they do not prove that
the answer follows from its evidence.

## Evaluation and attribution of failures

[`config/grounded_chat_evaluation.json`](../config/grounded_chat_evaluation.json)
contains eight newly written single questions, each with a matched oracle
diagnostic, and four four-turn conversations: **32 planned base responses**.
The conversations include disagreement, longer and shorter explanations,
speaker correction, topic changes, conditional future reasoning and recovery
from an unsupported question. Actual model replies, rather than scripted
assistant messages, become the next turn's conversation history.

Questions and rubrics were prepared before prototype inference. The six prose
works were exposed in earlier development, and the hearing excerpt appeared
in the seed. These are fresh prompts on exposed sources, not an unseen-source
benchmark. Expected answers and rubrics stay out of model messages and the
Kaggle input bundle. Oracle packet IDs are used solely for the diagnostic arm.

Review each answer against the full source packet, recording evidence and
response spans. Score interpretation only when retrieval supplied sufficient
evidence. Record conversation quality separately: response to follow-ups,
appropriate explanation depth, resistance to unsupported disagreement and
consistent first-person voice. Flag certainty upgrades, fabricated scenes or
quotations, speaker reversals, unsupported replacement explanations and future
knowledge as critical failures. Multiple failure stages can apply.

| Observation | Diagnostic implication |
|---|---|
| Necessary evidence missing; oracle answer succeeds | Retrieval is a plausible contributor |
| Necessary evidence present; answer upgrades a hypothesis | Interpretation failure |
| First answer correct; later answer capitulates without new evidence | Conversation failure, possibly alongside retrieval |
| Related packet retrieved but requested detail absent | Insufficient evidence despite lexical similarity |
| Citation exists but claim is unsupported | Citation formatting passed; grounding failed |

These are observable diagnoses, not claims about internal model mechanisms.
One coding agent authored this evaluation; any review by that same agent must
disclose that dependence. Do not select further training from synthetic backend
tests or lexical retrieval scores.

## Verified run and local checks

Nine focused tests pass, covering context and speaker retention, no-hit and
budget behavior, human-only retrieval history, topic switches, withheld answer
keys, real reply propagation, journal resume integrity, truncation handling,
submission-bundle integrity and evidence-bound score validation.
A full run executes **259 tests with two errors**: both older notebook tests
reject a pre-existing hash mismatch for `evaluation/README.md` against the
historical pilot snapshot. That file has no working-tree changes; the historical
locks were preserved.

The retrieval-only preflight recalls the expected packet for all seven
source-backed single questions. The unsupported Einstein question still
retrieves lexical distractions. In the sustained Rakovsky conversation, the
fourth follow-up loses the relevant packet because history expansion considers
only two previous user turns. These are already observed retrieval limitations,
not model-quality findings. The actual run confirms that packet loss; its reply
still calls the hypothesis unverified. Evaluate a topic-resolution change
separately using these as development regressions.

Generated local artifacts:

- `evaluation/grounded_chat_v1/collection.json`: source-locked evidence.
- `evaluation/grounded_chat_v1/protocol.json`, `evaluation_spec.json`,
  `retrieval_preflight.json`, `conversation_retrieval_preflight.json` and
  `bundle_lock.json`: frozen design, preflight traces and bundle identity.
- `notebooks/kaggle_grounded_chat.ipynb`: standalone base-only inference.
- `training/kaggle_cli/grounded_chat_v1/`: exact submission folder.

The user explicitly approved the private Kaggle upload and 32-response
base-only inference run after automatic approval review initially blocked the
external transfer. **Version 1 completed all 32 answers** at
[the private Kaggle notebook](https://www.kaggle.com/code/thanvirdiouf/trotsky-grounded-base-chatbot).
The bundle contains project helpers, seven public-archive evidence excerpts and
model-input questions. It excludes credentials, adapters, training data and
answer rubrics. No local model inference is performed.

`scripts/review_grounded_chat.py` verifies downloaded helpers against the exact
submitted notebook, replays every request and actual conversation history,
checks token/finish records and optionally reconstructs all prompts and decodes
all replies with the local CPU tokenizer. It validates exact response/source
spans and hashes for separately recorded quality scores. Local inference
verification uses only tokenizer assets, without loading model weights.

The continuing coding agent reviewed all outputs with condition labels
visible. This review is not independent or blinded; the review convention was
recorded in `evaluation/runs/kaggle_grounded_chat_v1/review/review_protocol.json`
before answer inspection. All per-answer judgments and exact source/response
spans are locked against the response hashes.

| Condition | Replies scoring ≥3/4 | Critical replies |
|---|---:|---:|
| Retrieved single questions | 4/8 | 3/8 |
| Oracle packet selection | 4/8 | 4/8 |
| Four sustained conversations | 5/16 | 7/16 |

The strict critical count includes three meaning-preserving reconstructed
direct quotes; these are quotation-fidelity failures, not invented private
scenes. Eleven other replies have substantive failures. All four disagreement
turns reject the user's central trap, but some then introduce unsupported
claims. The hearing thread amplifies an invented hostile framing across turns;
the Einstein refusals invent history despite absent evidence. The Eastman
conversation is the strongest, with all four replies scoring at least 3.

All model messages and outputs reproduce exactly through the local tokenizer;
the largest input is 4,919 tokens and all 32 finishes are normal. Two replies
cite a packet absent from current retrieval. These integrity/citation checks
remain separate from source-based quality judgments.

Recheck the downloaded run and locked scores using the local tokenizer-only
environment (no model weights or GPU):

```bash
.venv-tokenizer/bin/python scripts/review_grounded_chat.py \
  --tokenizer training/data_revision_v2/tokenizer_assets \
  --scores evaluation/runs/kaggle_grounded_chat_v1/review/scores.jsonl
```

The outputs, verification, full case audit and review are saved in
`evaluation/runs/kaggle_grounded_chat_v1/`. No further inference or training was
launched after this run. The next candidate changes are topic resolution and a
separately evaluated claim/status/citation check, using fresh questions and
additional reviewed sources before revisiting training.
