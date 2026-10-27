# Task-to-evidence mapping diagnostic

The source-reading comparison exposed two bottlenecks: discarded answers with
nonliteral quotations, and approved answers that blur source attribution or
hypothesis scope. V27/saved29 improved displayed reading replies from one to
three of five, but two still blocked. Frozen replay/token verification passed
22 calls, ten replies, no input rejection or truncation,11.32GiB peak. This is
execution evidence, not a quality score or a reason to promote the reader.

The next experimental path is `MappedSourceReadingChat`. It uses three calls:

1. Map up to five requested parts to proposed answers, a support classification,
   certainty category and at most two exact source substrings per part.
2. Compose one prose paragraph per mapped ID, with the full passages available.
3. Audit each paragraph against the full sources, with the map explicitly
   labelled unverified.

The application checks source packet/block membership and literal anchors, and
fills author, speaker and date from the original metadata. A generated map cannot
overwrite those fields. Missing-evidence units have no anchors; supported/partial
units require them. Composition must include each mapped ID exactly once in order;
it cannot silently discard a mapped task. Existing quotation/prose checks and
missing-verdict rejection remain in force. No generated content becomes a verified
claim or persistent interpretation. The route accepts independent cases only and
is not installed in ordinary conversation or the active chatbot.

These checks cannot establish that the model found every requested part, that an
anchor entails its proposed answer, or that a certainty label is correct. Those
are explicit false flags in the map, and both composer and auditor receive the
complete source context. Self-audit remains fallible. Application-generated source
references show supplied context, not sentence-level proof. Structured JSON is
strict; malformed/truncated stages block and raw outputs remain in the journal.

V28 compares the existing two-call reader against the three-call mapped reader
on the same five inspected questions and identical at-most-two source packets.
This is a development comparison of whole pipelines, not fresh validation or a
single-component ablation. Five replies per arm give at most25 calls plus the
separate memory probe, within the authorized private/free Kaggle limits. Base,
quantization, generation limits, dates and evidence plan remain pinned. The active
pointer stays v3; sourceplanv4 is explicitly experimental.

Five new tests cover anchor tampering and metadata, missing support, dropped or
duplicated task IDs, missing verdicts, malformed/truncated maps, empty verified
memory, fixed paired context, call bounds, hidden expectations and exact replay.
All86 focused tests pass; full359 retains the same two historical README snapshot
errors, with old locks unchanged. Synthetic CPU preflight constructs25 requests with five
map units and identical paired retrieval; maximum4,690 input tokens below5,120.
This does not bound actual generated maps/drafts; runtime checks every request.

After exact download/replay/token verification, assess displayed completeness,
source roles, uncertainty, unsupported details and unnecessary refusal separately.
Inspect rejected stages to locate failure without counting hidden drafts as
successful answers. Compare unchanged control outputs for drift. If the map helps
without new substantive errors, use fresh paraphrased tasks and missing/misleading
source cases next, followed by sustained follow-ups and disagreement. If it merely
moves errors into the map or fails the format budget, fix the measured bottleneck
before conversation integration. Keep both adapters experimental and training
paused; a paid GPU does not repair interpretation or application policy.

A separate five-task transfer design, ghost_mapped_transfer_v1.json, was authored
from full bounded sources before inspecting V28 results. It tests procedural
speaker roles, a missing confession transcript, reported speech versus a later
outcome, explicitly assumed numerical application, and partial firsthand
familiarity versus an unnamed informant. The primary coder knows earlier
failures; this is source-authored development transfer, not independent heldout
validation. Its expectations are removed from model payloads. Run it only after
V28 establishes whether the new path is technically usable; preserve original
questions and expectations if later implementation fixes are needed.

CPU retrieval preflight on the five transfer questions includes the intended
packet in every case without budget omission. Three cases also retrieve an
extra packet. This is a small source-specific retrieval diagnostic, not broad
recall validation. The paired inference test still fixes identical context so
retrieval differences cannot explain a reader/mapped-reader difference. Natural
retrieval plus dialogue remains a later end-to-end check. Local traces are in
evaluation/ghost_mapped_transfer_v1/retrieval_preflight.json.
