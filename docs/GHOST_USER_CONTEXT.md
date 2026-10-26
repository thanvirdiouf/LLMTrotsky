# Original user context across a longer conversation

V19/saved Kaggle21 is the no-retention transfer baseline. V20/saved22 adds a
bounded store of original hypothetical user messages. Sources, questions, base
revision, quantization, generation settings and call limits stay fixed. The
only changed frozen payloads are the ghost helper and protocol hash. Each arm
uses its own actual displayed history, so downstream changes can reflect both
the added context and changed earlier replies.

The store retains at most three whole messages and 2,500 characters. Records
preserve user role, unverified status, chronological turn, exact text and content
fingerprint. Whole-message evictions and over-budget omissions are explicit;
qualifications are not cut off. Hypothetical/revision cues select messages, with
later revisions superseding earlier assumptions and the current request
superseding old task instructions. These records never become verified source
claims or assistant summaries. Fingerprints check content consistency, not
independent truth or cryptographic authentication of a user.

The model ordinarily sees the last four messages. Only original user context
that has left that window is supplied separately. In this study the store
retains the original offer and revised offer (turns1/4); it does not retain the
earlier disagreement (turn2). This is bounded hypothesis retention, not general
semantic conversation memory or a claim to remember everything.

Both runs passed exact frozen-code replay and local CPU-tokenizer reconstruction:
ten replies, 25 generation calls, no input rejection or truncation, 11.32 GiB
peak. The first two independent questions and first three conversation turns
have identical model messages and displayed replies. Retention first changes
input at turn4. At delayed turn7, both original and revised offers reach all
three model stages; the largest V20 input is 4,868 tokens under the 5,120 cap.
This establishes the application supplied the missing context, not that the
model used it well.

The delayed displayed answer remains byte-identical and incomplete. Its draft
mentions the revised fund, but still overstates what the historical account
proves and is removed by the quotation guard. The final reply changes from a
105-word answer to only a repeated source statement because its interpretation
also contains raw quotation marks. Other changed turns show differing reasoning
and source selections. Pipeline dispositions move from six model-approved/two
partial/two blocked to five/three/two; these are not independent quality scores.
No overall quality gain is claimed from context availability.

The second AI reviewer saved standalone neutral A judgments before B was
provided. The saved source-based comparison finds seven byte-identical replies and mixed
changes, with no overall improvement established. The safer final source
attribution removes nearly all requested task coverage. Exact judgments, neutral
packets and the implementation-only `application_input_audit.json` remain in
`evaluation/runs/ghost_user_context_comparison/`. Pipeline
statuses, rejected drafts and model verdicts are withheld from that reviewer.
This is not human historical validation. The transfer questions were authored
by that reviewer after its previous study judgments had been saved.

The completed separate V21/saved23 trial changes only stage-output formatting, using
the same V20 inputs and memory controls. The persistent parser failures are
unchanged between V19/V20; fixing them does not address source certainty or
coverage. See [stage formatting](GHOST_STAGE_OUTPUT_FORMAT.md). Historical
helpers, frozen trials and both experimental adapters remain preserved.
