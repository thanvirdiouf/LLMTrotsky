#!/usr/bin/env python3
"""Rebuild offline artifacts and record each verified stage in the handoff."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-update-note", help="Explanation for an inspected revision to frozen seed evidence, persona or messages")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    stages = [
        ("scripts/extract_mia.py", "Rebuilt paragraph extraction from the cached archive."),
        ("scripts/build_catalog.py", "Rebuilt source families, duplicate links, chronology assessments and review queue."),
        ("scripts/build_seed_dataset.py", "Rebuilt the source-locked reviewed seed dataset."),
        ("scripts/build_evaluation.py", "Verified and rebuilt evaluation cases with whole-family reservations."),
        ("scripts/clean_candidates.py", "Rebuilt cleaned candidates and exact repair logs."),
        ("scripts/validate_project.py", "Passed end-to-end integrity, provenance, date and source-family separation checks."),
    ]
    for script, note in stages:
        command = [sys.executable, script]
        if script == "scripts/build_seed_dataset.py" and args.review_update_note:
            command.extend(["--review-update-note", args.review_update_note])
        try:
            subprocess.run(command, cwd=project, check=True)
        except subprocess.CalledProcessError:
            subprocess.run([sys.executable, "scripts/update_handoff.py", "--note", "Preparation stopped with a failed stage: " + script + ". Inspect the error before continuing."], cwd=project, check=False)
            raise
        subprocess.run([sys.executable, "scripts/update_handoff.py", "--note", note], cwd=project, check=True)


if __name__ == "__main__":
    main()
