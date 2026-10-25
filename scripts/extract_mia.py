#!/usr/bin/env python3
"""Extract auditable paragraph records from cached MIA HTML; no model calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlsplit

from bs4 import BeautifulSoup, Comment, NavigableString, Tag
try:
    from .corpus_dates import composition_evidence, interval_from_evidence, parse_date_range, MONTH_PATTERN, YEAR_PATTERN
except ImportError:  # Direct script invocation from the project directory.
    from corpus_dates import composition_evidence, interval_from_evidence, parse_date_range, MONTH_PATTERN, YEAR_PATTERN

BLOCK = {"body", "center", "address", "div", "section", "article", "main", "header", "footer", "nav", "table", "tr", "td", "th", "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "blockquote", "pre", "dl", "dt", "dd", "hr"}
SKIP = {"script", "style", "noscript", "template"}
SPEAKER = re.compile(r"^((?:(?:(?:DR|MR)\.?\s+)?(?:LEON\s+(?:D\.?\s+)?)?TROTSKY|(?:(?:DR|MR)\.?\s+)?(?:GOLDMAN|DEWEY|FINERTY|STOLBERG|BEALS|RÜHLE|RUHLE|RUEHLE|LA\s*FOLLETTE|INTERPRETER|REPORTER))(?:\s*\([^\n)]{1,100}\))?)\s*:\s*(.*)", re.I | re.S)
PERSON_WORD = r"[A-ZÀ-Þ][A-Za-zÀ-ž’'\-]*\.?"
UNKNOWN_SPEAKER = re.compile(r"^(" + PERSON_WORD + r"(?:\s+(?:(?:van|von|de|der|la)\s+)?" + PERSON_WORD + r"){0,5}(?:\s*\([^\n)]{1,100}\))?)\s*:\s*(.*)", re.S)
NOTE_HEADING = re.compile(r"^(?:(?:editor[’']?s?|translator[’']?s?|author[’']?s?)\s+)?(?:footnotes?|endnotes?|notes|references|bibliography)\s*[:.]?$", re.I)
TROTSKY_SIGNATURE = re.compile(r"\b(?:Leon\s+(?:D\.?\s+)?Trotsky|L\.\s*(?:D\.\s*)?Trotsky|L\.\s*T\.)", re.I)
# Reviewed against the cached source: the first two paragraphs are the 1941
# editors' introduction; the third begins Trotsky's 7 August 1939 article.
# There is no structural HTML boundary, so this exception must stay hash-bound.
REVIEWED_EDITORIAL_ENDS = {
    "a57054f27c1109c82b6400861621dcbe1fe28782bdb255aca8d784cf16bcce04": (
        "It is well that there is art in the world as well as politics. It is well that the potentialities of art are as inexhaustible as life itself.",
    ),
}


def parse_speaker(text):
    match = SPEAKER.match(text) or UNKNOWN_SPEAKER.match(text)
    if not match:
        return None
    label = match[1].strip().upper()
    canonical = re.sub(r"\s*\([^)]*\)$", "", label)
    canonical = re.sub(r"^(?:DR|MR)\.?\s+", "", canonical)
    if canonical in {"TROTSKY", "LEON TROTSKY", "LEON D. TROTSKY", "LEON D TROTSKY"}:
        canonical = "TROTSKY"
    return canonical, label, match[2]


def inline_text(node):
    if isinstance(node, Comment):
        return ""
    if isinstance(node, NavigableString):
        # Source-code wrapping is whitespace; only actual BRs create line breaks.
        return re.sub(r"\s+", " ", str(node))
    if node.name in SKIP:
        return ""
    if node.name == "br":
        return "\n"
    return "".join(inline_text(child) for child in node.children)


def blocks_from_html(soup, reviewed_editorial_ends=()):
    records = []
    section = 0
    containers = {}

    def visit(node, ancestry):
        nonlocal section
        if not isinstance(node, Tag) or node.name in SKIP:
            return
        if node.name == "hr":
            section += 1
            return
        if node.name in {"table", "blockquote", "div", "section", "aside"}:
            containers[id(node)] = f"{node.name}_{len(containers)}"
        ancestry = ancestry + [node]
        pending = []

        def flush():
            nonlocal section
            text = "".join(inline_text(n) for n in pending)
            text = re.sub(r" *\n *", "\n", text).strip()
            pending.clear()
            if not text:
                return
            classes = sorted({c for n in ancestry for c in n.get("class", [])})
            own_classes = node.get("class", [])
            role = "body_candidate"
            reasons = []
            if re.fullmatch(r"(?:\*\s*){3,}", text):
                role = "separator"
                section += 1
            elif any(n.name in {"nav", "footer"} for n in ancestry) or set(classes) & {"rtn", "navigation", "navbar", "nav", "toc", "skip"}:
                role = "navigation"
            elif set(classes) & {"information", "info"}:
                role = "metadata"
            elif re.match(r"^Last (?:updated|Updated)", text):
                role = "web_update"
            elif set(classes) & {"note", "notes", "footnote", "footnotes", "endnote", "endnotes"}:
                role = "note_ownership_unverified"
            elif "head" in classes or re.search(r"(?:[—–-]\s*(?:The )?Ed(?:s|itors)?\.?\s*$|Transcriber[’']?s? (?:note|comment)|Translator[’']?s? (?:note|comment))", text, re.I):
                role = "editorial_candidate"
                reasons.append("explicit editorial/translator/transcriber marker")
            elif node.name in {"h1", "h2", "h3", "h4", "h5", "h6"} or set(own_classes) & {"sub", "subtitle"}:
                role = "heading"
            if "head" in own_classes:
                reasons.append("head class: inspect for editorial introduction")
            if re.search(r"\[\d+\]|\((?:cid|sic):?", text):
                reasons.append("inline note or encoding marker")
            records.append({
                "block_id": len(records), "tag": node.name, "classes": own_classes,
                "ancestor_classes": classes, "html_anchor": node.get("id") or node.get("name"),
                "ancestor_tags": [n.name for n in ancestry],
                "layout_containers": [containers[id(n)] for n in ancestry if id(n) in containers],
                "section_index": section,
                "role": role, "text": text, "review_reasons": reasons,
            })

        for child in node.children:
            if isinstance(child, Tag) and (child.name in BLOCK or child.find(list(BLOCK)) is not None):
                flush()
                visit(child, ancestry)
            else:
                pending.append(child)
        flush()

    visit(soup.body or soup, [])
    current_section = None
    mode = None
    editorial_containers = set()
    for block in records:
        if block["section_index"] != current_section:
            current_section = block["section_index"]
            mode = None
        if mode == "editorial_candidate" and any(block["text"].startswith(prefix) for prefix in reviewed_editorial_ends):
            mode = None
            block["review_reasons"].append("reviewed editorial boundary tied to verified source SHA256")
        if re.match(r"^(?:Transcriber[’']?s?\s+comments?|Introduction\s+by\s+(?:the\s+)?Editors|Ed\.\s*Note|Editorial\s+Note|Editor[’']?s?\s+Note)\b", block["text"], re.I):
            if block["layout_containers"]:
                editorial_containers.add(block["layout_containers"][-1])
            else:
                mode = "editorial_candidate"
        elif NOTE_HEADING.match(block["text"]):
            mode = "note_ownership_unverified"
        if set(block["layout_containers"]) & editorial_containers and block["role"] not in {"metadata", "navigation", "web_update"}:
            block["role"] = "editorial_candidate"
            block["review_reasons"].append("inside explicitly marked editorial layout container")
        elif mode and block["role"] not in {"metadata", "navigation", "web_update", "separator"}:
            block["role"] = mode
            block["review_reasons"].append("following explicit editorial/note marker until next HTML section boundary")
    return records


def dateline_interval(text):
    """Recognize a date with only signature/place text around it."""
    interval = parse_date_range(text)
    if not interval or interval["latest"] > "1940-08-21":
        return None
    places = re.compile(r"\b(?:Coyoac[aá]n|Moscow|Prinkipo|Constantinople|Kadikoi|Oslo|Community|Hospital|Brest[ -]Litovsk|The Kremlin|Alma[ -]Ata|Mexico|D\.?\s*F\.?)\b", re.I)
    residue = TROTSKY_SIGNATURE.sub("", text)
    residue = re.sub(MONTH_PATTERN, "", residue, flags=re.I)
    residue = re.sub(YEAR_PATTERN, "", residue)
    residue = re.sub(r"\b\d{1,2}(?:st|nd|rd|th)?\b", "", residue, flags=re.I)
    residue = places.sub("", residue)
    return None if re.search(r"[^\s.,;:()&/–—-]", residue) else interval


def signed_dateline_evidence(blocks):
    """Accept short Trotsky signatures with dates, never generic date classes.

    MIA also uses date/author classes for publication citations, other writers,
    and quoted letters. A complete signature and a tightly bounded residue are
    required here; unsigned dates remain hints requiring manual review.
    """
    evidence = []
    for block in blocks:
        text = block["text"]
        classes = {c.casefold() for c in block["ancestor_classes"]}
        if block["role"] != "body_candidate" or len(text) > 200 or "blockquote" in block["ancestor_tags"]:
            continue
        if classes & {"quote", "quoteb", "title", "footer", "keyw"} or not TROTSKY_SIGNATURE.search(text):
            continue
        if not dateline_interval(text):
            continue
        evidence.append({"kind": "signed_trotsky_dateline", "block_ids": [block["block_id"]], "text": text})
    if evidence:
        # A page may collect several prefaces or letters. One signed component
        # must not supply a falsely narrow date for other, unsigned datelines.
        signed_bounds = interval_from_evidence(evidence)
        signed_ids = {block_id for item in evidence for block_id in item["block_ids"]}
        dateline_start = rf"^(?:[^\d\n:;]{{1,60}}[,\n]\s*)?(?:{MONTH_PATTERN}\s+|\d{{1,2}}\s+{MONTH_PATTERN})"
        for block in blocks:
            if block["block_id"] in signed_ids or block["role"] != "body_candidate" or len(block["text"]) > 120:
                continue
            if not re.match(dateline_start, block["text"], re.I):
                continue
            other = parse_date_range(block["text"])
            if other and (other["earliest"] < signed_bounds["earliest"] or other["latest"] > signed_bounds["latest"]):
                return []
    return evidence


def signed_letter_evidence(blocks):
    """Bind an opening dateline to one complete, explicitly signed letter."""
    body = [block for block in blocks if block["role"] == "body_candidate"]
    if len(body) < 3 or not dateline_interval(body[0]["text"]):
        return []
    if not re.match(r"^Dear\s+[^\n:]{1,100}[:,]", body[1]["text"]):
        return []
    # Multiple salutations/signatures or intervening headings may indicate a
    # collection of letters. A single header must not date that whole page.
    if sum(bool(re.match(r"^Dear\s", block["text"])) for block in body) != 1:
        return []
    if sum(bool(TROTSKY_SIGNATURE.search(block["text"])) for block in body) != 1:
        return []
    first, last = body[0], body[-1]
    if any(block["role"] == "heading" and first["block_id"] < block["block_id"] < last["block_id"] for block in blocks):
        return []
    if any("blockquote" in block["ancestor_tags"] or {c.casefold() for c in block["ancestor_classes"]} & {"quote", "quoteb"} for block in (first, last)):
        return []
    signature = last["text"]
    signature = re.sub(r"^(?:With(?:\s+\w+){0,8}\s+(?:greetings|wishes)|(?:Yours(?:\s+\w+){0,4}|Fraternally(?:\s+yours)?|Comradely greetings|Sincerely yours))[,.]?\s*", "", signature, flags=re.I)
    if not re.fullmatch(rf"(?:{TROTSKY_SIGNATURE.pattern})[.,]?", signature.strip(), re.I):
        return []
    return [{"kind": "signed_letter_dateline", "block_ids": [first["block_id"], last["block_id"]], "text": first["text"] + "\n" + last["text"]}]


def normalized_hash(text):
    return hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest()


def document(record, root, links):
    raw = (root / record["raw_path"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"Source hash mismatch for {record['url']}")
    soup = BeautifulSoup(raw, "lxml")
    blocks = blocks_from_html(soup, REVIEWED_EDITORIAL_ENDS.get(record["sha256"], ()))
    title = record["html_title"]
    metadata = {}
    for block in blocks:
        if block["role"] == "metadata":
            for line in block["text"].splitlines():
                match = re.match(r"([^:;.]{2,70})[:;.]\s*(.*)", line)
                if match:
                    metadata.setdefault(match[1].strip().lower(), []).append(match[2].strip())
    text = "\n\n".join(b["text"] for b in blocks)
    body = "\n\n".join(b["text"] for b in blocks if b["role"] in {"body_candidate", "heading"})
    path = urlsplit(record["url"]).path
    flags = []
    date_evidence = composition_evidence(metadata) + signed_dateline_evidence(blocks) + signed_letter_evidence(blocks)
    composition_raw = "; ".join(item["text"] for item in date_evidence)
    date_interval = interval_from_evidence(date_evidence)
    if re.search(r"\b(?:18|19|20)\d{2}\b", composition_raw):
        written_years = sorted({int(y) for y in re.findall(r"\b(?:18|19|20)\d{2}\b", composition_raw)})
    else:
        written_years = []
        flags.append("composition_date_unverified")
    if any(y > 1940 for y in written_years):
        flags.append("written_field_contains_post_1940_date")
    metadata_interval = interval_from_evidence([item for item in date_evidence if item["kind"] == "explicit_composition_metadata"])
    signed_interval = interval_from_evidence([item for item in date_evidence if item["kind"] != "explicit_composition_metadata"])
    if metadata_interval and signed_interval and (metadata_interval["latest"] < signed_interval["earliest"] or signed_interval["latest"] < metadata_interval["earliest"]):
        flags.append("composition_date_evidence_conflict")
    if any(b["role"] == "editorial_candidate" or "head" in b["classes"] for b in blocks):
        flags.append("editorial_material_requires_review")
    if "\ufffd" in text or "(cid:" in text:
        flags.append("encoding_artifacts")
    if re.search(r"[A-Za-z]{35,}", body):
        flags.append("possible_missing_spaces")
    speakers = {p[0] for b in blocks if b["role"] in {"body_candidate", "heading"} for line in b["text"].splitlines() if (p := parse_speaker(line))}
    if len(speakers) >= 2:
        flags.append("multiple_speakers")
    nlinks = len(soup.find_all("a", href=True))
    index_hint = bool(re.search(r"/(?:index|contents|toc|sw|rp-index)\.html?$", path, re.I))
    kind = "index_candidate" if index_hint else "text_candidate"
    if "RESTRICTED DIRECTORY" in title.upper():
        kind = "directory_placeholder"
    if not body.strip():
        flags.append("no_body_candidate")
    doc_id = "mia_" + hashlib.sha256(record["url"].encode()).hexdigest()[:20]
    # Hints, never substituted for a verified composition date or authorship.
    match = re.search(r"/((?:18|19)\d{2})/", path)
    return {
        "document_id": doc_id, "url": record["url"], "final_url": record["final_url"],
        "title": title, "page_kind": kind, "language_scope": "English archive (not automatically language-verified)",
        "source_html": record["raw_path"], "source_sha256": record["sha256"],
        "source_encoding": soup.original_encoding, "fetched_utc": record["fetched_utc"],
        "metadata_fields_raw": metadata, "composition_year_candidates": written_years,
        "composition_date_interval": date_interval, "composition_date_evidence": date_evidence,
        "heading_date_hints": [{"text": b["text"], "interval": parse_date_range(b["text"])} for b in blocks if b["role"] == "heading" and re.match(rf"^\(?\s*(?:{MONTH_PATTERN}\s+\d|\d{{1,2}}\s+{MONTH_PATTERN})", b["text"], re.I)],
        "url_year_hint": int(match[1]) if match else None,
        "headings": [b["text"] for b in blocks if b["role"] == "heading"],
        "incoming_links": links.get(record["url"], []), "outgoing_link_count": nlinks,
        "text": text, "body_candidate_text": body, "body_candidate_words": len(body.split()),
        "body_candidate_sha256": normalized_hash(body), "blocks": blocks,
        "quality_flags": flags, "authorship_status": "unverified", "review_status": "pending",
        "approved_for_training": False,
    }


def qa_candidates(doc):
    turns = []
    active = False
    section = 0
    for b in doc["blocks"]:
        if b["role"] == "separator":
            active = False
            section += 1
            continue
        if NOTE_HEADING.match(b["text"].strip()):
            active = False
            section += 1
            continue
        if b["role"] not in {"body_candidate", "heading"}:
            # A note or editorial insertion must never be appended as speech.
            continue
        # Explicit BR-separated turns are supported as well as one turn per P.
        for line in b["text"].splitlines():
            parsed = parse_speaker(line)
            if parsed:
                speaker, label, text = parsed
                turns.append({"speaker": speaker, "source_label": label, "text": text, "block_ids": [b["block_id"]], "review_reasons": [], "section": section})
                if "(" in label:
                    turns[-1]["review_reasons"].append("speaker_label_has_parenthetical_qualifier")
                active = True
            elif b["role"] == "heading":
                if active and turns:
                    turns[-1]["review_reasons"].append("intervening_heading_or_stage_direction")
                # Subheadings/stage directions are not spoken words.
                if b["tag"] in {"h1", "h2", "h3"}:
                    active = False
                    section += 1
            elif active and turns:
                turns[-1]["text"] += ("\n\n" if turns[-1]["text"] else "") + line
                turns[-1]["block_ids"].append(b["block_id"])
            if active and turns and "blockquote" in b.get("ancestor_tags", []):
                turns[-1]["review_reasons"].append("quoted_material_requires_review")
    output = []
    for i, turn in enumerate(turns):
        if i == 0 or turn["speaker"] != "TROTSKY" or turns[i - 1]["speaker"] == "TROTSKY":
            continue
        previous = turns[i - 1]
        if previous["section"] != turn["section"]:
            continue
        if "?" not in previous["text"] or not turn["text"].strip():
            continue
        flags = ["speaker_attribution_and_context_require_review", "inline_notes_may_remain"]
        flags.extend(sorted(set(turn["review_reasons"] + previous["review_reasons"])))
        if len(turn["text"].split()) < 10:
            flags.append("short_answer_requires_context")
        if len(previous["text"].split()) > 400 or len(turn["text"].split()) > 1200:
            flags.append("long_turn_requires_review")
        output.append({
            "candidate_id": f"{doc['document_id']}_turn_{i}", "document_id": doc["document_id"],
            "source_url": doc["url"], "source_sha256": doc["source_sha256"],
            "question_speaker": previous["speaker"], "question": previous["text"].strip(),
            "answer_speaker": "TROTSKY", "answer": turn["text"].strip(),
            "question_block_ids": sorted(set(previous["block_ids"])),
            "answer_block_ids": sorted(set(turn["block_ids"])),
            "context_turns": turns[max(0, i - 4):i - 1],
            "answer_origin": "source_transcript_extraction_unreviewed",
            "quality_flags": flags, "review_status": "pending", "approved_for_training": False,
        })
    return output


def extract(root):
    snapshot = (root / "downloads.jsonl").read_text()
    # The crawler can append while extraction runs. Ignore only a partial final
    # write, never a malformed complete record in the middle of the manifest.
    lines = snapshot.splitlines()
    if snapshot and not snapshot.endswith("\n"):
        lines = lines[:-1]
    downloads = [json.loads(l) for l in lines]
    # If a page was retried in a later run, use its most recent saved record.
    downloads = list({r["url"]: r for r in downloads}.values())
    state = json.loads((root / "crawl_state.json").read_text())
    counts = Counter()
    flags = Counter()
    duplicates = defaultdict(list)
    total_words = 0
    qa_count = 0
    samples = []
    documents_tmp = root / "documents.jsonl.tmp"
    qa_tmp = root / "qa_candidates.jsonl.tmp"
    with documents_tmp.open("w") as docs, qa_tmp.open("w") as qa:
        for r in downloads:
            doc = document(r, root, state["links"])
            docs.write(json.dumps(doc, ensure_ascii=False) + "\n")
            counts[doc["page_kind"]] += 1
            flags.update(doc["quality_flags"])
            total_words += doc["body_candidate_words"]
            if doc["body_candidate_text"]:
                duplicates[doc["body_candidate_sha256"]].append(doc["document_id"])
            for item in qa_candidates(doc) if doc["page_kind"] == "text_candidate" else []:
                qa.write(json.dumps(item, ensure_ascii=False) + "\n")
                qa_count += 1
            if len(samples) < 12 and doc["page_kind"] == "text_candidate":
                samples.append({"document_id": doc["document_id"], "url": doc["url"], "title": doc["title"], "metadata_fields_raw": doc["metadata_fields_raw"], "first_blocks": doc["blocks"][:12]})
    documents_tmp.replace(root / "documents.jsonl")
    qa_tmp.replace(root / "qa_candidates.jsonl")
    dup_groups = [ids for ids in duplicates.values() if len(ids) > 1]
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "crawl_snapshot_utc": state.get("updated_utc"),
        "downloaded_pages": len(downloads), "pending_downloads": len(state["queue"]),
        "download_failures": state["failures"], "page_kinds": counts,
        "body_candidate_words_including_duplicates_and_unreviewed_material": total_words,
        "quality_flag_counts": flags, "exact_body_duplicate_groups": dup_groups,
        "qa_candidates_unreviewed": qa_count, "approved_training_examples": 0,
        "scope": "Linked HTML pages within /archive/trotsky/, excluding bio/, photo/, quotes.htm, plus the explicit supplemental MIA works listed in crawl_state.json. Other external pages and PDF-only material are logged but not fetched.",
        "supplemental_urls": state.get("supplemental_urls", []),
        "limitations": [
            "A downloaded page is not proof of authorship, completeness, or historical availability.",
            "Block roles and page kinds are heuristics; editorial introductions, quotations, notes, and foreign-language pages require review.",
            "Publication, composition, web-update, and URL dates are distinct. URL years are only hints.",
            "No OCR conjectures, LLM rewriting, or training/validation split has been performed.",
            "Duplicate detection currently covers identical normalized body text; near duplicates and book excerpts need grouping before splitting.",
            "Transcript extraction is provisional and may require surrounding dialogue; no candidates are training-approved.",
        ],
    }
    (root / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (root / "extraction_samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"exact_body_duplicate_groups", "limitations", "download_failures"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("corpus_mia"))
    extract(parser.parse_args().input)
