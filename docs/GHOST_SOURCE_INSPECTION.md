# Inspect source evidence without generation

The factual self-auditor has not earned trust as an automatic approval gate.
The next usable application feature is direct access to the actual bounded
passages, with original dates, authors, speakers and surrounding context.

```bash
python3 scripts/ghost_chat.py --evidence-only --question 'What does the letter say about Bessedovsky and the purge?'
python3 scripts/ghost_chat.py --evidence-only --source-id hearing --source-id ilo_genai_2025
python3 scripts/ghost_chat.py --evidence-only --source-id hearing --source-format json
```

These commands need no model endpoint or GPU and make no inference calls. They
print original source packets or a readable literal-text view. The JSON preserves
all packet metadata and source blocks. Markdown includes source links, the
composition date evidence, observation date where supplied, context boundaries
and speaker labels. Source text is fenced literally so embedded markup cannot
become executable-looking display content. It is not rewritten into the ghost's
voice, authenticated as an event or labelled as a verified answer.

Question mode uses the existing active-v3 retriever, which can miss evidence or
return irrelevant packets. Exact-ID mode inspects one or two known packets.
Neither mode changes conversation state or trusted claim memory. Unsupported
IDs and empty requests fail rather than inventing a passage. This source view
is an inspection tool, not an automatic substitute for answering every subtask.
The default chat route, active pointer and endpoint requirements are unchanged.

If saving a view, keep it in the ignored research archive, for example:

```bash
python3 scripts/ghost_chat.py --evidence-only --source-id hearing > evaluation/hearing_source_view.md
```

Do not add source text to public commits. The existing rights/publication boundary
still applies. `--log` is reserved for answer journals; source inspection writes
to stdout instead of mixing evidence views into conversation answer logs.

Four tests check complete source/role/qualification preservation, zero generated
answers and unchanged memory, retrieved-candidate limits and invalid selection,
literal-fence safety, and endpoint-free CLI JSON. A CPU-only display check on the
five fixed transfer questions includes every intended packet without budget
omissions. Three cases also retrieve an extra packet. These are retrieval/display
checks, not fresh synthesized-answer evaluation or broad recall measurements.
Private JSON/Markdown examples and packet fingerprints are under
`evaluation/ghost_source_inspection/`.

The combined current suite passes97 focused tests; full370 retains the same two
historical README snapshot errors. No source bytes or protected locks changed.
