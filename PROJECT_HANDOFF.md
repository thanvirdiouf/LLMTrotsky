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

Private saved54 (v10) tested five fresh questions with one supplied packet
each. All five calls replayed. Hearing dates selected all three required
records. The missing-detail control selected none. The other three replies
returned prose and were blocked. The renderer preserved the accepted record
wording and citations. All 94 local tests passed.

Questions are in `config/ghost_reviewed_transfer_v1.json`. The local v10 run
directory holds the frozen checks, verifier report, and source review.
| Evidence selection experiment | Result |
| --- | --- |
| Saved54, unconstrained ID output | 2/5 passed. Three returned prose. |
| Saved55, constrained ID output | 3/5 passed. All outputs valid. Memoir and ILO coverage missed a requested point. |
| Saved56, include-or-exclude for each record | 2/5 passed. Memoir improved, with regressions elsewhere. |
| Saved57, separate selections for question parts | 4/5 passed. Omitted Goldman's acknowledgment of his 1902 mistake. Nine calls for five answers. |
| Saved58, same method on five fresh questions | 4/5 passed. All required records selected, with one unrelated Blumkin letter record. Eight calls for five answers. |

Both question-part runs passed exact replay and both missing-detail controls.
The parts were authored in advance in `config/ghost_reviewed_question_parts.json`.
The experiment uses supplied passages and renders the selected records verbatim.
Fresh questions are in `config/ghost_reviewed_transfer_v2.json`. Local run
directories retain the locked rubrics, verification, and per-case scores.
All 94 local tests passed.

Private saved59 (v16) passed exact replay for ten turns and fifteen calls.
Automatic splitting and retrieval ranked the expected packet first for
all fifteen parts. Fourteen parts and nine turns met the locked coverage
and relevance checks. Every required record was selected. The first Eastman
answer added two tangential records about Opposition tactics. Both topic
switches passed, and the hearing follow-up retained the supported answer
while flagging the missing spouse occupation.

The test uses earlier user questions for references. Removing earlier-turn
questions left the rankings unchanged on these ten turns. A separate
four-case diagnostic of shorter references ranked the intended packet first
in four cases with history and one without it. Selection conditioned on the
assistant's earlier reply and sustained dialogue remain open.

The design is in `config/ghost_reviewed_followups_v1.json`. The local v16
directory holds the verified output, locked rubric, and per-part results.
All 94 local tests passed. Next test longer follow-ups that explicitly refer
to what the assistant just said, including a correction and a disagreement.

Continue based on measured results. Training is paused and both adapters are
experimental. Authorized 27B trials use the same private Kaggle notebook, two
free T4s, and at most ten displayed replies, thirty evaluation calls and one
short probe per run. Review findings guide decisions without another user
review round. Promotion, paid compute, and publication require separate scope.

For commands, preserved protocols, and the full history, see
[the research archive](docs/RESEARCH_ARCHIVE.md).
