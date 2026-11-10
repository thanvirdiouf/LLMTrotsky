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

Private Kaggle saved51 (local `ghost_review_chat_v7`) was submitted once.
It supplies exactly the specified packet for each of the five fixed questions
and uses packet-only citations. Scoring checks stay local. The run uses five
answer calls and one memory probe. Results are pending.

Download saved51's bundle and results under
`evaluation/runs/ghost_review_chat_v7/download/`. Verify with
`scripts/verify_review_chat.py --version ghost_review_chat_v7 --tokenizer evaluation/ghost_review_chat_v7/tokenizer_assets`
in the local tokenizer environment, then score the final replies against the
fixed transfer design before deciding whether to run ordinary retrieval.

## Next work

1. Use the five untouched questions and scoring checks in
   [the transfer design](config/ghost_mapped_transfer_v1.json). Start with the
   supplied passage and the packet-only citation format. Check speaker
   attribution, uncertainty, completeness, arithmetic, and bounded refusals.
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
