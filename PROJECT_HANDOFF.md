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

Private Kaggle saved52 (local `ghost_review_chat_v8`) is the matched ordinary
retrieval run. The frozen prompts rank the expected packet first on all five
questions and include a second packet in each. The opt-in v9 comparison is
prepared with the same questions and packets plus guidance on completeness,
attribution, certainty, and exact hypothetical arithmetic. The prompts were
compared locally to confirm that only this guidance changed.

## Next work

1. Use the completed supplied-passage run as the baseline for the five
   [fixed transfer questions](config/ghost_mapped_transfer_v1.json). Carry its
   interpretation errors into the comparison.
2. Run the same questions with normal retrieval. Compare the actual packets
   and answers to separate retrieval misses from interpretation errors.
3. After those checks pass, test sustained disagreement, follow-ups, topic
   changes, and the distinction between evidence and the fictional voice.

Continue based on measured results. Training is paused and both adapters are
experimental. Authorized 27B trials use the same private Kaggle notebook, two
free T4s, and at most ten displayed replies, thirty evaluation calls and one
short probe per run. Review findings guide decisions without another user
review round. Promotion, paid compute, and publication require separate scope.

For commands, preserved protocols, and the full history, see
[the research archive](docs/RESEARCH_ARCHIVE.md).
