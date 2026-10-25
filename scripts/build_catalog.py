#!/usr/bin/env python3
"""Build review queues and conservative source families before dataset splitting."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import heapq
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlsplit
try:
    from .corpus_dates import cutoff_status
except ImportError:
    from corpus_dates import cutoff_status


def words(text):
    return re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold())


def read_rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def book_hint(url, title=""):
    path = urlsplit(url).path
    families = [
        (r"/(?:1930/mylife|works/mylife)/", "my_life"),
        (r"/193[026]/(?:history|hrr)/", "history_of_the_russian_revolution"),
        (r"/1936/revbet/", "the_revolution_betrayed"),
        (r"/1937/dewey/", "dewey_commission_hearings"),
        (r"/1937/ssf/", "stalin_school_of_falsification"),
        (r"/1940/xx/stalin/", "stalin_biography"),
        (r"/(?:idom|1939/idom)/", "in_defense_of_marxism"),
        (r"/(?:1907/1905|works/1905)/", "1905"),
        (r"/1924/ffyci-[12]/", "first_five_years_of_the_comintern"),
        (r"/1932/pcr/", "problems_of_the_chinese_revolution"),
        (r"/1931/tpr/", "permanent_revolution_and_results_prospects"),
        (r"/1904/tasks/", "our_political_tasks"),
        (r"/1914/war/", "war_and_the_international"),
        (r"/1938/tp/", "transitional_program_collection"),
        (r"/1918/ourrevo/", "our_revolution_collection"),
    ]
    for pattern, label in families:
        if re.search(pattern, path):
            return label
    return None


class Groups:
    def __init__(self, keys):
        self.parent = {key: key for key in keys}

    def find(self, key):
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def join(self, left, right):
        a, b = sorted((self.find(left), self.find(right)))
        self.parent[b] = a


def build(root):
    persona_path = root.parent / "config" / "persona.json"
    cutoff = json.loads(persona_path.read_text())["historical_as_of"] if persona_path.exists() else "1940-08-19"
    docs = {}
    tokens = {}
    exact = defaultdict(list)
    books = defaultdict(list)
    signatures = defaultdict(list)
    queue = []
    for doc in read_rows(root / "documents.jsonl"):
        key = doc["document_id"]
        body = "\n\n".join(b["text"] for b in doc["blocks"] if b["role"] == "body_candidate")
        wt = words(body)
        fingerprint = hashlib.sha256(" ".join(wt).encode()).hexdigest()
        family = book_hint(doc["url"], doc["title"])
        docs[key] = {k: doc[k] for k in ["document_id", "url", "title", "page_kind", "source_sha256", "composition_year_candidates", "url_year_hint", "quality_flags", "body_candidate_words"]}
        docs[key].update({"comparison_words": len(wt), "book_family_hint": family, "comparison_sha256": fingerprint})
        docs[key]["composition_date_interval"] = doc.get("composition_date_interval")
        docs[key]["persona_cutoff_status"] = cutoff_status(doc.get("composition_date_interval"), cutoff)
        if doc["page_kind"] != "directory_placeholder" and len(wt) >= 30:
            tokens[key] = wt
            exact[fingerprint].append(key)
            if family:
                books[family].append(key)
            # Bottom-k stable shingle hashes are only a candidate search. Actual
            # Jaccard/containment is checked on full normalized word shingles.
            hashes = {int.from_bytes(hashlib.blake2b(" ".join(wt[i:i + 5]).encode(), digest_size=8).digest(), "big") for i in range(max(0, len(wt) - 4))}
            for value in heapq.nsmallest(64, hashes):
                signatures[value].append(key)
        priority = 0
        reasons = list(doc["quality_flags"])
        if docs[key]["persona_cutoff_status"] != "on_or_before_cutoff":
            reasons.append(docs[key]["persona_cutoff_status"])
        if doc["page_kind"] == "text_candidate":
            priority += 10
        if doc["url_year_hint"] and doc["url_year_hint"] >= 1937:
            priority += 20
        if "multiple_speakers" in reasons:
            priority += 15
        if family in {"stalin_biography", "stalin_school_of_falsification"}:
            reasons.append("mixed_authorship_or_edition_requires_review")
        if not reasons:
            reasons.append("verify_authorship_date_and_text")
        queue.append({"document_id": key, "url": doc["url"], "title": doc["title"], "review_priority": priority, "review_reasons": reasons, "review_status": "pending", "approved_for_training": False})

    groups = Groups(docs)
    edges = []
    for label, members in books.items():
        for member in members[1:]:
            groups.join(members[0], member)
    for members in exact.values():
        for member in members[1:]:
            groups.join(members[0], member)
            edges.append({"left": members[0], "right": member, "reason": "identical_normalized_body"})
    pair_hits = Counter()
    for members in signatures.values():
        if len(members) > 60:
            continue
        members = sorted(set(members))
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                pair_hits[left, right] += 1

    @lru_cache(maxsize=64)
    def shingles(key):
        wt = tokens[key]
        return {tuple(wt[i:i + 5]) for i in range(len(wt) - 4)}

    for (left, right), hits in pair_hits.items():
        if hits < 10 or groups.find(left) == groups.find(right):
            continue
        if min(len(tokens[left]), len(tokens[right])) < 150:
            continue
        a, b = shingles(left), shingles(right)
        overlap = len(a & b)
        containment = overlap / max(1, min(len(a), len(b)))
        jaccard = overlap / max(1, len(a | b))
        if jaccard >= 0.72 or containment >= 0.85:
            groups.join(left, right)
            edges.append({"left": left, "right": right, "reason": "near_duplicate_or_excerpt", "shingle_jaccard": round(jaccard, 4), "shorter_containment": round(containment, 4)})

    components = defaultdict(list)
    for key in docs:
        components[groups.find(key)].append(key)
    group_rows = []
    for members in components.values():
        members.sort()
        gid = "family_" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        labels = sorted({docs[m]["book_family_hint"] for m in members if docs[m]["book_family_hint"]})
        group_rows.append({"source_family_id": gid, "document_ids": members, "book_family_hints": labels, "grouping_status": "provisional_until_crawl_and_review_complete", "split": None})
        for member in members:
            docs[member]["source_family_id"] = gid
    for name, rows in [("catalog.jsonl", docs.values()), ("source_families.jsonl", group_rows), ("duplicate_edges.jsonl", edges), ("review_queue.jsonl", sorted(queue, key=lambda r: (-r["review_priority"], r["url"])) )]:
        temp = root / (name + ".tmp")
        with temp.open("w") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temp.replace(root / name)
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "documents": len(docs), "historical_cutoff": cutoff, "cutoff_assessments": Counter(d["persona_cutoff_status"] for d in docs.values()), "source_families": len(group_rows), "duplicate_edges": len(edges), "edge_reasons": Counter(e["reason"] for e in edges), "largest_family_sizes": sorted([len(g["document_ids"]) for g in group_rows], reverse=True)[:12], "splits_assigned": False, "limitations": ["Automatic book-family hints and shingle matching need review.", "Detection is not exhaustive: low-overlap alternate translations and small paraphrased excerpts may be missed.", "Family IDs may change as additional pages are downloaded; freeze only after the crawl is complete."]}
    (root / "catalog_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("corpus_mia"))
    build(parser.parse_args().input)
