# Trotsky’s ghost

A fictional present-day Trotsky chatbot. It separates the historical record, dated
modern evidence, and imagined interpretation. Source packets retain the speaker,
date, context, and uncertainty.

## Run locally

The workbench uses Python 3.10+ and the private source snapshot at
`evaluation/ghost_reviewed_coverage_v3/`.

```sh
python3 scripts/review_chat_server.py --port 8765
```

Open [the workbench](http://127.0.0.1:8765). Browse verified sample conversations
and inspect sources through their numbered citations. Source lookup makes zero
model calls. To display the latest experimental samples, add
`--review-run evaluation/runs/ghost_review_chat_v6`.

Inspect the active route’s evidence from the terminal.

```sh
python3 scripts/ghost_chat.py --evidence-only --source-id hearing
```

Live chat requires an operator-configured model endpoint and matching local
tokenizer. Run `python3 scripts/review_chat_server.py --help` for options.
A Git checkout contains the code and configurations. The source snapshot,
evaluation journals, model files, and credentials are local.

## Develop

```sh
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests
```

The test suite uses the local research inputs. [Current status and next steps](PROJECT_HANDOFF.md)
and [archive recovery](docs/RESEARCH_ARCHIVE.md) cover the research workflow.

Original code and documentation are [GPL-3.0-only](LICENSE). Sources, datasets
and model artifacts have separate terms. [Attribution and licensing](NOTICE.md)
