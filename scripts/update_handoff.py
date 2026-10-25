#!/usr/bin/env python3
"""Refresh the managed live-status section without replacing handoff prose."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

START = "<!-- LIVE_STATUS_START -->"
END = "<!-- LIVE_STATUS_END -->"


def update(project, note=None):
    path = project / "PROJECT_HANDOFF.md"
    text = path.read_text()
    root = project / "corpus_mia"
    state = json.loads((root / "crawl_state.json").read_text())
    audit = json.loads((root / "audit.json").read_text()) if (root / "audit.json").exists() else {}
    now = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M %Z")
    lines = [START, "## Latest status snapshot", "", f"Snapshot: {now}. This section supersedes older numeric snapshots below.", "",
             f"- Crawl: {len(state['done']):,} visited URLs; {len(state['queue']):,} pending; {len(state['failures']):,} recorded failures.",
             f"- Crawl state last updated (UTC): `{state.get('updated_utc', 'unknown')}`.",
             "- Crawl queue exhausted; no downloader is needed unless adding sources." if not state["queue"] and not state.get("in_flight") else "- Confirm the downloader is active before starting another instance; do not use concurrent writers."]
    if audit:
        lines.extend([
            f"- Last extraction: {audit['downloaded_pages']:,} cached pages; generated (UTC): `{audit.get('generated_utc', 'not recorded')}`.",
            f"- Unreviewed body-candidate words, including duplicates: {audit['body_candidate_words_including_duplicates_and_unreviewed_material']:,}.",
            f"- Authentic-dialogue Q&A candidates: {audit['qa_candidates_unreviewed']:,}; automatic bulk-training approvals: {audit['approved_training_examples']}.",
        ])
    seed_path = project / "datasets" / "seed_audit.json"
    if seed_path.exists():
        seed = json.loads(seed_path.read_text())
        lines.append(f"- Separately reviewed training seed: {seed['examples']} examples from {seed['source_documents']} source documents (coding-agent review, not independent historian validation).")
        lines.append(f"- Reviewed multi-turn conversations: {seed.get('multi_turn_conversations', 0)}.")
    catalog_path = root / "catalog_audit.json"
    if catalog_path.exists():
        temporal = json.loads(catalog_path.read_text())["cutoff_assessments"]
        lines.append(f"- Composition-date evidence: {temporal.get('on_or_before_cutoff', 0):,} on/before cutoff; {temporal.get('composition_date_unverified', 0):,} unverified; {temporal.get('date_interval_overlaps_cutoff', 0)} overlapping cutoff; {temporal.get('after_cutoff', 0)} after cutoff.")
    evaluation_path = project / "evaluation" / "audit.json"
    if evaluation_path.exists():
        evaluation = json.loads(evaluation_path.read_text())
        lines.append(f"- Initial evaluation: {evaluation['cases']} cases; seed/evaluation source-family overlap: {evaluation['seed_source_family_overlap']}.")
        lines.append("- Remote model-run status and reviews: `evaluation/runs/` and `docs/REMOTE_MODEL_WORKFLOW.md`. The evaluation-build audit does not count later Kaggle runs.")
    cleaning_path = root / "cleaning_audit.json"
    if cleaning_path.exists():
        cleaning = json.loads(cleaning_path.read_text())
        lines.append(f"- Cleaned candidate corpus: {cleaning['documents']:,} text pages; {cleaning['candidate_words_including_duplicates_and_unreviewed_material']:,} words including duplicates and unreviewed content.")
    lines.extend(["- Local preparation is CPU-only. Remote inference and training status are tracked in `docs/REMOTE_MODEL_WORKFLOW.md`.", END])
    section = "\n".join(lines)
    if START in text and END in text:
        before, rest = text.split(START, 1)
        _, after = rest.split(END, 1)
        text = before + section + after
    else:
        first, rest = text.split("\n", 1)
        text = first + "\n\n" + section + "\n" + rest
    if note:
        if "## Work log" not in text:
            text += "\n## Work log\n\n"
        text += f"- {now}: {note}\n"
    path.write_text(text)
    print(section)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--note")
    args = parser.parse_args()
    update(args.project, args.note)
