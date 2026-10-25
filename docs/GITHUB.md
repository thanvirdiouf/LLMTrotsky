# GitHub source distribution

This repository publishes the original project software and documentation.
The working directory also contains a separate local research archive, which
is intentionally excluded from Git. See [NOTICE.md](../NOTICE.md) for the
license boundary and source attribution.

## What a fresh clone contains

- Corpus fetching, extraction, curation, validation, and dataset builders.
- Kaggle notebook builders, training helpers, evaluation and review tools.
- Tests, persona configuration, source URL lists, prompts, and project notes.
- The complete GPLv3 license and third-party attribution notice.

It does not contain credentials, virtual environments, the supplied or crawled
corpus, curated data or review locks, generated notebooks, saved evaluation
responses, adapters, or checkpoints. Paths and results in historical project
notes describe the local research archive; links to those artifacts will only
work in a workspace that has them. A clone alone cannot reproduce the archived
experiment results or regenerate the reviewed seed and its evidence locks.
Downloading source HTML does not recreate the curated reviews.

The grounded-chat and ghost implementations also require the local
`config/grounded_chat_*.json` review/source specifications,
`config/ghost_contemporary_sources.json`, and generated evidence under
`evaluation/`. These contain source excerpts, reviewed propositions or reference
checks and are excluded along with the research archive. The active persona,
application pointers, original ghost evaluation design, code and guides are
published. Restore the reviewed inputs before running the evidence builders.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the fixture-based checks that work without the research archive:

```bash
python3 -m unittest discover -s tests -p test_corpus_dates.py -v
python3 -m unittest discover -s tests -p test_extract_mia.py -v
```

The full test suite and research workflows described in the main README also
need the excluded local artifacts. Restore those from your own verified archive
before running them. Review the rights for each material before publishing any
additional datasets, source passages, or bundled notebooks.

## Publish the repository

With GitHub CLI installed and authenticated, run from the project root:

```bash
gh auth login --hostname github.com
gh repo create LLMTrotsky --public --source=. --remote=origin --push
```

Alternatively, create an empty repository named `LLMTrotsky` on GitHub without
adding a README, license, or gitignore, then add its SSH or HTTPS clone URL as
the `origin` remote and push `main`.

Before publishing, inspect `git ls-files` to confirm the intended file set.
Ignored files stay on disk. Adding a file with `git add -f` bypasses the
publication exclusions and requires a fresh review.
