# Present-day Trotsky ghost — active project direction

The active goal is now a **clearly fictional ghost of Leon Trotsky conversing
in the present day**, without a 1940 knowledge cutoff. This replaces the active
product goal of reproducing exactly what he could know on 19 August 1940.
The old experiments remain archived and reproducible.

## Measured progress — 26 October 2026

Five ghost smokes are complete and verified. V1–v4 use the same inspected
ten-turn set; v5 evaluates four new questions and a six-turn conversation,
which are now inspected regressions. All passed the separate 5,000-token probe at 11.32 GiB, without
truncation or input-budget rejection. These establish compact T4 feasibility,
not conversational readiness.

| Smoke / Kaggle version | Evaluation calls | Main finding |
|---|---:|---|
| v1 / 4 | 16 | Eight structural blocks; checker approved a wrong observation date |
| v2 / 5 | 19 | Canonical facts fix the date; retained sources fix continuity; excessive refusal remains |
| v3 / 6 | 19 | Selection ambiguity fixed; five drafts omit an empty quote field; useful conversation still fails |
| v4 / 7 | 19 | No schema blocks; useful factual/identity answers; unsupported opinion premises still pass the checker |
| v5 / 8 | 20 | Four new factual cases preserve source limits; conversation invents personal circumstances and misses mode transitions |

Local source-based reviews are in `evaluation/runs/ghost_smoke_vN/QUALITY_REVIEW.md`.
V3 fulfills only the modern factual question, identity clarification and scoped
missing-news refusal. One checker-approved vendor fact answer is irrelevant to
an endorsement question. Same-agent unblinded review is not independent
historian validation. No interpretation or conversation reliability is claimed.

V4 gives five clearly useful answers and preserves the historical article's
uncertainty. Its opinion audit still approves unsupported historical and price
comparisons. The next test therefore changes relevant source coverage and basis
selection: the active evidence version adds the 1926 radio speech and a dated
ILO AI/work article. Opinions require a retrieved historical basis; application
basis additions are logged. Facts-only follow-ups retain their topic and
suppress opinion drafting. Date kinds and context boundaries enter model input.

Twenty-four ghost tests, four source-provenance tests and four harness tests
pass. V5's four single questions and identity clarification are useful, but
its views make unsupported personal assumptions; facts-only mode includes an
unrequested historical argument, and later turns repeat facts. All model checks
pass despite these failures. No conversation reliability is claimed.

The next combined regression gives auditing a neutral role, tightens study-only
scope, discourages personal assumptions, and fixes return-to-opinion routing.
The prepared routing-only v6 is preserved **unrun**. V7 uses the exact v5
questions as inspected regressions; model approval remains fallible.
[Evidence and case design](GHOST_SOURCE_EXPANSION.md).

The persona can discuss later history, contemporary technology and current
politics. New views are fictional interpretations, not rediscovered Trotsky
writings. It keeps conceptual precision and a documented political starting
point while allowing curiosity, disagreement, humor and ordinary conversation.
It should not turn every reply into a signed public address or repeat a fiction
disclaimer at every turn. The interface identifies the persona as fictional.

## Run the active version

```bash
python3 scripts/build_ghost_chat.py
python3 scripts/ghost_chat.py --dry-run \
  --question "How would you assess renting an A40 on Runpod today?"
python3 -m unittest discover -s tests -p test_ghost_chat.py -v
```

With an already configured remote service serving the untouched base model:

```bash
python3 scripts/ghost_chat.py --base-url https://YOUR_SERVER/v1 \
  --log /tmp/trotsky-ghost-chat.jsonl
```

The endpoint uses `TROTSKY_MODEL_API_KEY` from the environment and never logs it.
No model weights or GPU inference run locally. Endpoint identity remains an
operator assertion; an HTTP model name cannot attest revision or adapter
absence. No remote ghost service has been deployed or tested yet.

`config/active_chat.json` selects `config/persona_ghost_expansion.json`, the
versioned source plan and generated ghost evidence files. The active reference date is **26 October 2026**. Frozen initial smokes retain
25 October for reproducibility. It is a configurable reference date,
not a fixed historical knowledge boundary. Update it and version the evidence
artifacts when refreshing sources. A reference date alone does not supply news.

## Three kinds of content

| Kind | What the bot may do | What remains constrained |
|---|---|---|
| Historical record | Explain what Trotsky or another speaker documented | Original author, date, roles, order, quotation and uncertainty |
| Contemporary evidence | Use facts documented after 1940 | Source author, observation/update dates, scope and freshness |
| Fictional interpretation | Form new opinions and conditional reasoning in this persona | Label as a present-day interpretation; support factual premises; never attribute it to the historical author |

For example, a vendor's modern price is attributed to that vendor. A new
political judgment about access to technology is the fictional persona's view.
It must not become an authentic Trotsky endorsement of a modern company.
Likewise, knowing later history does not turn an old prediction into a proven
claim within the old document. A later outcome needs its own source.

The renderer labels newly imagined reasoning **“My present-day interpretation”**
once in an answer containing such reasoning. Historical and modern facts keep
their source citations. Exact quotations still come from original source
anchors with their recorded speaker. Model-written quotation and citation
markup are rejected. A new interpretation is not entered into verified-fact
memory; its historical/contemporary basis IDs are recorded separately.

## Evidence coverage and pipeline

The original collection contains seven bounded historical packets and one
dated modern example: Runpod's official A40 Pods listing, observed on
25 October 2026. The page separately labels its update date as 8 October 2026.
Its author is Runpod, and the snapshot is explicitly a bounded text
transcription, not a raw HTML hash or full-page archive.

That preserved version has 33 reviewed propositions. Active evidence v2 adds
a complete bounded section of the 1 March 1926 radio speech and three paragraphs
from an official ILO article published 20 May 2025, observed 26 October 2026.
It contains ten packets and 39 propositions; prediction, estimate and observed
outcome remain distinct. This is limited reviewed coverage; **broad current-affairs
coverage and automatic live news retrieval have not been implemented**.
Additional reviewed contemporary sources belong in a new versioned local
source input selected by `config/ghost_source_plan_v2.json` or its successor.
Retain the publication or update date, separate observation date, authorship,
exact blocks and supported propositions. Rebuild into a new
evidence version and update the active pointer rather than silently overwriting
an existing source snapshot. User-supplied modern scenarios can be discussed
conditionally without being authenticated as facts.

The select/draft/check/render pipeline supports separate fact,
quotation and interpretation segments. The current checker has a neutral auditing role and permits clearly fictional
normative reasoning while requiring support for descriptive premises. Earlier
checks inherited the character prompt and accepted unsupported premises; the
neutral-role change requires runtime evaluation. It is the same base model and remains
fallible. A passing verdict is not proof of semantic faithfulness. Source
curation is by the continuing coding agent, not an independent historian.

Topic state persists over follow-ups and resets on explicit topic changes.
For a present-day interpretive question, retrieval can include a reviewed
historical basis alongside the relevant modern source. These are starting
principles with their source-era context, not a claim that every earlier
position was Trotsky's final view.

Prompt construction avoids repeating source text in every proposition. The
original context blocks stay intact; claim anchors become block references
and character offsets in the model request. Full source/proposition data stay
in the trace for audit. JSON is compact, and omitted older conversation messages
are explicitly counted. No qualifying source paragraph is silently cropped.

## Verification and next evaluation

Twenty-three ghost-specific local tests cover the active profile, use of post-1940
evidence, source domains/dates, fictional-interpretation labeling, exact modern
quotation rendering, compact-context fidelity, insufficient evidence and
rejection of unsupported factual premises hidden inside an opinion. The
grounded-chat tests remain applicable to the archived v1/v2 components.

`config/ghost_chat_evaluation.json` prepares new checks for historical fidelity,
modern facts, false posthumous authorship, conditional reasoning, missing news
and multi-turn opinion/source distinctions. A new opinion has no single
historically correct reference answer; assess its coherence, documentary basis,
transparency and usefulness separately. Do not score a post-1940 date as a
failure merely because of its date.

The five real ghost smokes are verified, with the limitations described
above. Local synthetic-backend tests establish implementation behavior, not
real model quality; the first model results show the checker can be wrong.
The new source-expansion smoke must be reviewed before claiming practical reliability.
Both adapters remain experimental; training remains paused.

The [next smoke-test protocol](GHOST_SMOKE.md) describes the prepared
memory-first, ten-answer runs and result-dependent continuation. Its helper
tests pass, and the user has explicitly approved bounded follow-up iterations
on the same private notebook and payload categories.

## What happened to the 1940 v2 trial

The approved Kaggle v2 comparison stopped after **14 direct-control answers**,
before its checked arm. A 6,546-token prompt exhausted the T4's 14.56 GiB device
memory during attention prefill. The tokenizer budget had passed; token limits
alone were not a sufficient memory test. The partial journals and exact error
are preserved in `evaluation/runs/kaggle_grounded_chat_v2/` and are not a complete
quality comparison. The old experiment was not rerun after the goal changed.

Kaggle's original notebook title and URL were restored with saved version 3
using `--no-run`. Version 1's completed run and version 2's failed run remain
preserved. No extra inference was launched by the title restoration.

The compact ghost inputs fit the measured 5,000-token probe; semantic quality
still fails the development smoke. An A40's additional memory could help longer
contexts and interactive work; the earlier claim that Kaggle had ample capacity
applied to v1's smaller prompts. No paid GPU has been rented or provisioned.

The historical `config/persona.json`, original seed data, training locks and
saved adapters remain unchanged. This is an active application/persona change,
not a silent reinterpretation of the old training or evaluation results.

### Current routing follow-up

After v5 was frozen/submitted, static inspection found that its fourth
conversation turn requests a fictional view using wording the intent rule
misses after facts-only mode. That finding is recorded before answer inspection
in `evaluation/runs/ghost_smoke_v5/routing_preinspection.json`. The local follow-up
recognizes fictional views, own reasoning and explicit disagreement; a full
pipeline regression passes after basis reset. Frozen v5 does not contain this
patch, so its model run cannot validate it.

For a readable transcript of a verified run, use
`python3 scripts/export_ghost_transcript.py --version ghost_smoke_vN`.
This exports exact displayed replies and source metadata after checking the
answer hash and frozen source/case payloads. It assigns no quality score.
