# Conversation source retention experiment

On29October2026 the user delegated further review and project decisions: “Go
ahead. I don't need to review anymore. I trust your judgement.” Implementing-agent
review may therefore guide the next steps without another human review gate. It
is still unblinded development review, not independent validation. Training,
paid compute, adapters, public uploads and automatic active-model replacement
remain outside this work.

## Problem and separate opt-in policy

Saved48's last press follow-up retrieved hearing/Bessedovsky instead of Eastman.
The old query builder added prior human questions only for short queries or a
limited anaphora expression. The eight-content-term follow-up missed that gate.
The model then cited an absent source using prior conversation context.

`conversation_retrieval.py` introduces `human_context_decay_v1`: current query
terms retain weight1; the last four human questions contribute distinct terms
at weights0.5,0.25,0.125,0.0625. Explicit generic topic-reset phrases clear older
context, including on subsequent turns. Otherwise a new topic competes through
its current terms. There is no semantic intent classifier and no claim that
implicit changes always eliminate older context. No assistant text, generated
answer, expected source ID or model self-audit enters the query.

The existing full-corpus BM25 statistics, configured source scope, at most two
whole packets,18,000-character packet bound and5,120-token input ceiling remain.
The request preserves the original current question and complete latest pair.
Only whole older pairs or a lower-ranked whole packet may be removed. Default
`prepare_turn` and the current workbench retain the old policy. Old9B helpers,
frozen notebook payloads and active configuration are unchanged.

## Offline evidence and limits

Before running the audit, ten development controls were fixed in
`config/ghost_conversation_retrieval_v1.json`. These cover five continuations,
two explicit resets, two implicit named-subject changes and a continuation after
a reset. Expected IDs are local audit checks, never model-message routing.
The audit reports8/10 for the old policy and10/10 for the new one. It repairs a
missing hearing context and prevents pre-reset Eastman leakage. On the actual
archived failing press question it recovers Eastman as the first packet.

The implicit-change checks require the new subject to lead, not complete
elimination of the old source. Both implicit changes still include the old
subject as secondary context. That is a measured noise limitation, not a
complete topic-routing solution. Ten local checks are not independent holdout
or semantic answer validation. Audit output is ignored at
`evaluation/runs/ghost_conversation_retrieval_v1/offline.json`.

Four new tests cover user-only context, bounded history, reset persistence,
source scope, exact current-question preservation and whole-packet budgeting.
All176 focused tests pass (158ghost,6boundary,8review,4conversation retrieval).
The CPU preflight for the new inference bundle uses ten synthetic outputs and
real tokenizer assets, isolated archived-code replay and tamper rejection.
It makes zero model calls and supplies no GPU or quality evidence.

## Fresh conversation run

`config/ghost_conversation_context_v1.json` fixes five new two-turn development
conversations: payment/personal enrichment, total-versus-resident secretaries,
explicit switch to ILO, implicit switch to an informant report, and a request
for missing original testimony. These are new questions on existing bounded
sources, not independent source-family transfer. The original fixed transfer
set remains unrun. No expected answers or oracle IDs enter the runtime.

Local version `ghost_review_chat_v5` is prepared separately with ten new calls
and at most ten replies, plus one5,000-input/16-output probe. Initial inputs
have4,317/4,546/4,294/1,991/4,109tokens. Actual follow-ups will use actual earlier
outputs with the same5,120-token guard. Model, persona/policy, source53snapshot,
greedy non-thinking decoding, loader, attention, GPU placement and output cap
remain unchanged. Questions and retrieval are both new, so this is a candidate
conversation test, not a causal generated-answer comparison with saved48.

It uses the existing private Kaggle notebook and pinned Qwen3.5-27B on two free
T4s,600s setup/2700s workload caps,1024output. No local weights, training,
adapters, paid compute, public upload or active9B/v3 replacement. A local
pre-output source rubric is hash-locked separately and is not uploaded.

After an actual submission record its exact saved version. Download that
version's ghost_review_chat_bundle/* and ghost_review_chat_results/* into
`evaluation/runs/ghost_review_chat_v5/download/`, then run:

```sh
.venv-tokenizer/bin/python scripts/verify_review_chat.py \
  --version ghost_review_chat_v5 \
  --tokenizer evaluation/ghost_review_chat_v5/tokenizer_assets
```

Verify metadata before inspecting final JSON; keep intermediate reasoning
private. Dedicated verification checks frozen inputs/tokens/GPU/backend and
isolated dependent retrieval/history/parser replay. Then separately review
source support, missing information, roles, certainty, relevance, completeness
and voice against the frozen local rubric. A format pass is not source approval.
No rerun, larger budget, self-audit gate or automatic activation follows failure.

## Submission

The preparedv5 was submitted once after commit6cb6d6b as private saved49.
The fetched exact saved49 notebook's normalized cells match the frozen source;
initial status RUNNING. Never resubmit it. No semantic result is claimed yet.
The local source rubric and lock remain outside the uploaded payload.

## Version 49 results reviewed on 10 November 2026

All ten turns passed frozen request, tokenizer, output, GPU and dependent
conversation replay. The run made ten answer calls and one memory probe.
Peak GPU allocation was 10.59 and 9.77 GiB. The pre-run design and rubric hashes
matched. The implementing agent reviewed each final reply against its actual
packets and the frozen rubric. Detailed notes and the verifier report are stored
locally under `evaluation/runs/ghost_review_chat_v5/`.

Every reply received and cited its relevant packet. Eight ranked it first.
Both the Eastman and hearing follow-ups retained their source. The explicit
switch removed Eastman and correctly used the ILO article. The old ILO topic
ranked ahead of Bessedovsky on the implicit switch, and the hearing ranked
ahead of Bessedovsky on the request for missing testimony. Both answers still
used Bessedovsky for their central conclusions.

The core distinctions survived, including payment versus personal enrichment,
total versus resident secretaries, potential exposure versus observed job loss,
and reported information versus firsthand testimony. Supporting prose
sometimes exceeded the sources. Examples include turning wide publicity into
the only possible route, treating adoption barriers as preventing adoption,
and making the ILO forecast contingent on social dialogue. The final answer
gave a bounded statement that the informant's name and original testimony were
absent from the supplied material. It also cited `bessedovsky_report`, an
internal summary ID included in the request. The application correctly flags
that ID because its citation targets are source packets.

## Packet citation control

The local `ghost_review_chat_v6` bundle uses the same five conversations,
retrieval, source packets, summaries, attribution, model and budgets. It omits
only the internal IDs on the summaries given to the model. Each summary keeps
its parent packet ID. The archived v5 bundle and the workbench default preserve
their original behavior.

The exact-history CPU comparison reconstructed all ten v5 requests with this
setting. Source selection, the system instructions, conversation messages, and
all other documentary fields were identical. Input savings ranged from 44 to
111 tokens. A synthetic ten-turn preflight passed the normal replay verifier and
rejected a tampered answer. These are structural checks. The controlled GPU
comparison measures whether the model uses the packet citation while retaining
the bounded answer, and checks all turns for citation and source-quality
regressions. Generated follow-up histories may diverge when earlier replies
change. The existing rubric and added citation criterion are locked locally in
`evaluation/ghost_review_chat_v6/local_comparison_lock.json`.

Version 50 completed on the same private notebook. Its saved source matched the
frozen bundle. All ten replies and the probe passed exact replay. The two runs
selected the same packets in the same order on every turn. Replies with an
unresolved citation fell from one to zero. The missing-testimony reply cited
Bessedovsky and preserved the bounded answer. Each reply cited its relevant
packet. Detailed findings are in the local v6 `SOURCE_REVIEW.json`.

The source review found clearer qualifications in the explicit ILO switch and
the payment follow-up. It also found persistent overstatement of the Eastman
publicity rationale and of how Trotsky checked the memoirs. The first press
reflection adds unsupported contemporary generalizations. The citation
improvement is narrow and the evidence comes from an unblinded development
comparison on the same ten turns.

Use packet-only IDs in the next opt-in evaluation with the prewritten transfer
questions in `config/ghost_mapped_transfer_v1.json`. The saved 49 and 50 results
make speaker attribution, evidential limits, and source certainty the main
checks. Keep the prewritten questions and expected checks fixed. The transfer
set is still unused at this checkpoint.
