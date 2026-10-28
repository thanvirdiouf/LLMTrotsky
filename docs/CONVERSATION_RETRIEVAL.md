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
