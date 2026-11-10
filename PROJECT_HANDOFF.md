# Project handoff

**Current objective.** Make a useful fictional present-day Trotsky conversation
with traceable evidence. The active route is the pinned Qwen3.5-9B base with
reviewed source packets, selected by `config/active_chat.json`. The local
workbench offers source lookup and verified examples. Live inference needs a
configured endpoint.

## What the experiments established

| Stage | Finding |
| --- | --- |
| Corpus and early pilots | Extracted 1,913 source pages and reviewed 500 seed examples. QLoRA pilots produced two experimental adapters with mixed quality gains. |
| Supplied evidence | On 24 development cases, passages improved 19 base answers and 22 adapter answers. Uncertainty errors remained. Grounded base became the reference. |
| Grounded chat and selector studies | Prototype v1 verified 32 replies. The v2 comparison stopped after a memory failure. Later selection, mapping, and checker trials exposed omissions and false approvals. |
| Conversation v4–v5 | The local workbench verified ten displayed replies from saved47/48. Private saved49 tested ten new replies. The relevant packet was present and cited every time, ranking first in eight. Old-topic material could still outrank a new subject. |
| Packet citations, v6 | Private saved50 removed internal summary IDs from the model prompt. Exact replay passed. Replies with unresolved citations fell from one to zero on the same ten questions. Some answers still overstated source certainty or added unsupported modern generalizations. |

Saved49 and saved50 are local `ghost_review_chat_v5` and `ghost_review_chat_v6`.
Their verifier reports and turn-by-turn development reviews are in
`evaluation/runs/`. These are implementing-agent findings from a small
development comparison.

## Transfer evaluation

Private Kaggle saved51 (local `ghost_review_chat_v7`) completed. All five
replies passed exact replay with no truncations or unresolved citations.
The supplied passages helped preserve the main distinctions, including
reported information versus eyewitness testimony and exposure versus observed
job loss. The source review found omissions and overstatements even with the
right passage. See the local `evaluation/runs/ghost_review_chat_v7/SOURCE_REVIEW.md`.

Private saved52 (v8) completed and replayed all five ordinary retrieval replies.
Every expected packet ranked first alongside a second packet. The radio
omission persisted. Bessedovsky regressed from overstating verification to
accepting the false premise that Trotsky knew none of the facts firsthand.

Private saved53 (v9) replayed five calls with the same evidence and added
guidance. Four replies were blocked for including a fictional reflection
in source-only mode. The remaining Bessedovsky reply still accepted the
false premise. The guidance variant failed.

## Next work

The local opt-in `evaluation/ghost_reviewed_coverage_v4/` snapshot now includes
a source-anchored statement about Trotsky’s personal acquaintance with at least
a dozen facts and his limits on verifying the rest. The radio class-control
statement already exists. Both statements passed deterministic rendering with
their source citations when explicitly selected. This fixes a gap in the
reviewed evidence. Selection and generated answers still require evaluation.

Private saved54 (v10) tests five fresh, independent questions with one supplied
packet each. The model selects reviewed statement IDs. The renderer preserves
their wording and citations. The fixed questions are in
`config/ghost_reviewed_transfer_v1.json`. Expected IDs and the checksum are local
under `evaluation/runs/ghost_review_chat_v10/`. Synthetic replay and all 94
local tests passed. The actual five-call result is pending.

Verify saved54 with `scripts/verify_review_chat.py` and the v10 tokenizer after
downloading its bundle and outputs into the run’s `download/` directory.
Review coverage, relevance, and the missing-evidence control separately.
Use the result to decide whether to test retrieval and sustained dialogue.

Continue based on measured results. Training is paused and both adapters are
experimental. Authorized 27B trials use the same private Kaggle notebook, two
free T4s, and at most ten displayed replies, thirty evaluation calls and one
short probe per run. Review findings guide decisions without another user
review round. Promotion, paid compute, and publication require separate scope.

For commands, preserved protocols, and the full history, see
[the research archive](docs/RESEARCH_ARCHIVE.md).
