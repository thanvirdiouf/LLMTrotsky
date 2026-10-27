# Thinking-mode factual audit diagnostic

V30/saved32 completed and passed frozen replay/token verification: ten calls,
ten diagnostic judgments, no input rejection or truncation,11.32GiB peak. Against
the frozen primary-coder expectations it has four correct approvals, one correct
rejection, one false approval, one false rejection and three format blocks.
Zero of five pairs have both formal decisions correct. These are development
outcomes, not human gold or a representative accuracy estimate.

The valid surname-only observation is rejected with an invented implication
about deliberate omission. The false named-informant claim is approved by
reversing subject and informant. Three outputs end in a stray backtick and remain
formally blocked. Their raw JSON intentions are not counted as successful
semantic rejection; one would approve unsupported Opposition membership, and
some rationales also mix the two source packets. The neutral checker is not
enabled in synthesis, trusted memory or training.

Before changing the backbone or buying hardware, the next bounded inference
diagnostic tests an unused capability of the same pinned base. Qwen documents
thinking as its default mode and direct responses as explicitly disabling it.
See the [official Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B).
All earlier runs explicitly disabled thinking.

V31 reuses exactly the ten V30 candidates, questions, sources, expectations and
auditor messages. It enables thinking and raises the combined reasoning/final
output cap from1,024 to4,096 tokens. Greedy generation, source context, base pin,
NF4, free Kaggle GPU0 and input cap5,120 stay unchanged. This compares inference
policies, not thinking in isolation from its larger output budget. It is not
fresh transfer and involves no training or adapter.

The raw model output/tokens stay in the private generation journal for replay.
Only content after exactly one completed Qwen thinking separator is parsed as
final judgment JSON. Intermediate text cannot approve a candidate. Missing or
ambiguous separators, truncated generations and malformed final JSON block.
No chain of reasoning is displayed as a chatbot answer or used as evidence.
The usual six-Boolean schema and strict final parser remain in force; this does
not repair semantic errors or relax the three earlier backtick failures.

The builder permits this thinking flag only for the neutral factual-audit
calibration. Runtime and verifier bind the flag to the frozen protocol and use it
for both memory-probe and evaluation chat-template reconstruction. Defaults for
old protocols remain non-thinking. A separate5,000-input/16-output memory probe
is unscored; each of ten candidate judgments has a single4,096-token cap.
No retry cascade, paid compute or backbone download is added.

Four new tests cover final-channel separation, preservation of raw output and
final fingerprints, malformed/incomplete/truncated rejection, identity mismatch,
unchanged candidates, scope restrictions and exact cached replay. Old V30 remains
verifiable with the updated verifier. If this inference policy still fails the
simple factual contrasts, do not keep treating self-review as verification or
train from its approvals. Use the inspectable source view while planning a
separately evaluated synthesis/evaluator change. If it improves, test it on
actual generated candidates before fresh questions and sustained conversations.

All97 focused tests pass, including source inspection and separate raw/final
judgment hashes. The full370-test run retains only the same two historical README
snapshot errors. V30 passes frozen/token verification with the updated verifier.
CPU preflight constructs ten thinking requests, maximum1,911 input tokens;
auditor message lists are identical to actual V30 calls. Candidate strings and
expectations match exactly. Synthetic intermediate text is excluded from the
scored response. None of this establishes factual model quality.

V31 submitted as private Kaggle saved33 after f1e4f00 was committed and pushed.
Results are pending; the default chatbot and audit policies are unchanged.

## Verified outcome

V31/saved33 verified ten calls with nine output-cap truncations and one complete
correct rejection;0of5pairs both formally correct. No input rejection, peak
11.32GiB. Truncated calls are not semantic rejections and raw intermediate text
was not displayed. The tested greedy thinking policy is not integrated. Next
preparation is [hardware inventory and a backbone review](GHOST_HARDWARE_AND_BACKBONE_REVIEW.md),
not an automatic increase to reasoning budget.
