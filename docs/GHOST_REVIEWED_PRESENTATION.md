# Offline presentation trial

Saved43 produced five valid ID selections but its repeated inline provenance made
the responses cumbersome. `scripts/preview_reviewed_selection.py` replays those
actual selections in a separate static HTML artifact. It makes zero model calls
and does not change the selected IDs, factual statements, attribution, record
status, original outputs or active application route.

Statements appear in their selected order. Status and attribution remain visible
alongside each statement. Source titles, authors and source dates appear once
per source within each case; observation dates remain separately labelled. An
expandable provenance section retains every complete source record, including
anchor text, speaker roles, boundaries, review labels and derivations. The page
labels statements as curated summaries rather than verbatim source quotations.
Source claim status is not presented as independent verification.

The script checks every original display field against deterministic rendering
of the supplied collection/ledger, checks the original question, packet scope,
available-record fingerprints and case order against frozen requests, and
refuses blocked replies or overwriting an existing trial. HTML text and links
are escaped; scripts, external resources and forms are disabled by the page's
content security policy. It is a presentation replay, not a substitute for the
dedicated model-run verifier. Verify the original run first.

Reproduce the new artifact in a new output directory:

```sh
python scripts/preview_reviewed_selection.py \
  --bundle evaluation/runs/ghost_reviewed_selection_v1/download/ghost_reviewed_selection_bundle \
  --answers evaluation/runs/ghost_reviewed_selection_v1/download/ghost_reviewed_selection_results/answers.jsonl \
  --output evaluation/runs/ghost_reviewed_selection_v1/presentation_v3
```

The latest artifact is ignored at
`evaluation/runs/ghost_reviewed_selection_v1/presentation_v2/preview.html` with a
hash-bound `replay.json`; the first offline rendering in `presentation_v1/` is
also preserved. It preserves all 22 selected statements across the
five original questions. Four new focused tests check exact displayed statement
text, HTML escaping, retention of provenance and dates, rejection of modified
display/question/memory metadata, refusal to overwrite, and unchanged input
bytes. All 18 reviewed-source tests passed. The browser security policy rejected
the local-file preview; no alternate browser route was attempted. Consequently
visual layout, usability and comparative readability remain unvalidated. HTML
structure/text checks do not establish those outcomes.

## Remaining coverage work

The preview deliberately leaves saved43's omissions visible. Three different
issues require separate treatment:

| Issue | Development evidence | Appropriate next experiment |
| --- | --- | --- |
| Available summaries omitted | Hearing March date and Blumkin reform position were available but unselected | Task-coverage ID selection with the same original questions, sources and menu |
| Immutable summary lacks precision | Eastman seasonal timing, actor and group-decision detail remain compressed | A new source-reviewed opt-in ledger extension; preserve existing records |
| Unnecessary material or response format | Unrequested context, long source panels, no literal replacement-field formatting | Presentation review and relevance assessment, without inventing factual wording |

These examples are post-hoc development findings, not runtime routing rules or
answer IDs to supply to a model. A prospective coverage pass must receive full
controlled sources and all available reviewed summaries, preserve the original
questions/expectations, and be frozen/tested as a new experiment. It may select
IDs but must not generate new factual text or become a factual self-audit gate.
All remaining semantic review is unblinded implementation review. Fresh transfer
and conversation/voice evaluation remain gated; neither this preview nor record
preservation demonstrates complete or independently validated answers.
