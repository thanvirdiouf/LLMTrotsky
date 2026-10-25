import hashlib
import json
from copy import deepcopy
import unittest
from unittest.mock import Mock

from scripts.remote_model import call_chat, chat_endpoint, generation_sources, review_key, validate_generated


def source():
    evidence = "A revolution is not an automatic machine. It is made by living people."
    return {"source_document_id": "doc", "source_url": "https://www.marxists.org/source.htm", "source_sha256": "raw", "source_block_ids": [1], "source_family_id": "family", "source_evidence": evidence, "evidence_sha256": hashlib.sha256(evidence.encode()).hexdigest(), "source_date_evidence": {"basis": "explicit_composition_metadata", "interval": {"earliest": "1937-01-01", "latest": "1937-12-31"}}, "approved_for_seed_training": True}


def corpus():
    item = source()
    doc = {"document_id": "doc", "url": item["source_url"], "source_sha256": "raw", "composition_date_interval": item["source_date_evidence"]["interval"], "blocks": [{"block_id": 1, "role": "body_candidate", "text": item["source_evidence"]}, {"block_id": 2, "role": "editorial_candidate", "text": "A later editor wrote this commentary."}, {"block_id": 3, "role": "body_candidate", "text": "This is a real but unreviewed passage from the same document."}]}
    catalog = {"doc": {"source_family_id": "family", "source_sha256": "raw"}}
    assignment = {"doc": {"split": "train", "source_family_id": "family", "url": item["source_url"]}}
    return {"doc": doc}, catalog, assignment


def example(**updates):
    item = {"question": "Does revolution happen automatically?", "answer": "A revolution is not an automatic machine.", "answer_origin": "verbatim_source", "evidence_spans": ["A revolution is not an automatic machine."], "approved_for_training": True}
    item.update(updates)
    return item


class RemoteModelTests(unittest.TestCase):
    def test_validation_never_approves_teacher_outputs(self):
        records = validate_generated(json.dumps({"examples": [example()]}), source(), 2)
        self.assertFalse(records[0]["approved_for_training"])
        self.assertEqual(records[0]["review_status"], "pending")

    def test_invented_evidence_and_fabricated_verbatim_are_rejected(self):
        for update in [{"evidence_spans": ["This sentence is not in the source passage."]}, {"answer": "A completely invented historical quotation."}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_generated(json.dumps({"examples": [example(**update)]}), source(), 2)

    def test_invalid_shape_duplicates_and_provenance_changes_are_rejected(self):
        for obj in [[], {"examples": [1]}, {"examples": [example(), example()]}, {"examples": [example(source_url="changed")]}]:
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                validate_generated(json.dumps(obj), source(), 2)

    def test_heldout_unassigned_and_corrupted_sources_are_rejected(self):
        documents, catalog, assignments = corpus()
        for split in ["test", "validation", "unassigned_pending_review"]:
            assignments["doc"]["split"] = split
            with self.subTest(split=split), self.assertRaises(ValueError):
                generation_sources([source()], assignments, "1940-08-19", documents, catalog, {review_key(source())}, set())
        bad = source()
        bad["source_evidence"] += " Edited silently."
        assignments["doc"]["split"] = "train"
        with self.assertRaises(ValueError):
            generation_sources([bad], assignments, "1940-08-19", documents, catalog, {review_key(source())}, set())

    def test_valid_sources_are_deduplicated(self):
        documents, catalog, assignments = corpus()
        self.assertEqual(len(generation_sources([source(), source()], assignments, "1940-08-19", documents, catalog, {review_key(source())}, set())), 1)

    def test_current_holdout_overrides_a_stale_training_manifest(self):
        documents, catalog, assignments = corpus()
        with self.assertRaisesRegex(ValueError, "Current evaluation specifications"):
            generation_sources([source()], assignments, "1940-08-19", documents, catalog, {review_key(source())}, {"family"})

    def test_changed_authoritative_date_still_requires_a_new_frozen_review(self):
        documents, catalog, assignments = corpus()
        old_approval = {review_key(source())}
        updated = source()
        updated["source_date_evidence"] = {"basis": "explicit_composition_metadata", "interval": {"earliest": "1939-01-01", "latest": "1939-12-31"}}
        documents["doc"]["composition_date_interval"] = updated["source_date_evidence"]["interval"]
        with self.assertRaisesRegex(ValueError, "frozen evidence review"):
            generation_sources([updated], assignments, "1940-08-19", documents, catalog, old_approval, set())

    def test_self_consistent_forged_evidence_and_dates_are_rejected(self):
        documents, catalog, assignments = corpus()
        for updates in [
            {"source_evidence": "Invented source passage with a freshly computed checksum."},
            {"source_date_evidence": {"basis": "explicit_composition_metadata", "interval": {"earliest": "1936-01-01", "latest": "1936-12-31"}}},
            {"source_url": "https://www.marxists.org/different.htm"},
            {"source_sha256": "fabricated"}, {"source_family_id": "different_family"},
            {"approved_for_seed_training": "false"},
            {"source_block_ids": []}, {"source_block_ids": [1, 1]},
            {"source_block_ids": [True]}, {"source_block_ids": [999]},
            {"source_block_ids": [2], "source_evidence": documents["doc"]["blocks"][1]["text"]},
            {"source_block_ids": [3], "source_evidence": documents["doc"]["blocks"][2]["text"]},
        ]:
            bad = {**source(), **updates}
            bad["evidence_sha256"] = hashlib.sha256(bad["source_evidence"].encode()).hexdigest()
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                generation_sources([bad], assignments, "1940-08-19", documents, catalog, {review_key(source())}, set())

    def test_stale_split_and_catalog_are_rejected(self):
        for target in ["split", "catalog"]:
            documents, catalog, assignments = deepcopy(corpus())
            if target == "split":
                assignments["doc"]["source_family_id"] = "old_family"
            else:
                catalog["doc"]["source_sha256"] = "old_html"
            with self.subTest(target=target), self.assertRaises(ValueError):
                generation_sources([source()], assignments, "1940-08-19", documents, catalog, {review_key(source())}, set())

    def test_endpoint_and_redirect_handling(self):
        self.assertEqual(chat_endpoint("https://server.example/v1/"), "https://server.example/v1/chat/completions")
        for url in ["file:///tmp/x", "https://name:password@server.example/v1", "https://server.example/v1?key=value"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                chat_endpoint(url)
        session = Mock()
        session.post.return_value.status_code = 302
        with self.assertRaises(ValueError):
            call_chat(session, "https://server.example/v1", {}, "")
        self.assertFalse(session.post.call_args.kwargs["allow_redirects"])

    def test_response_parsing_does_not_export_reasoning(self):
        session = Mock()
        response = session.post.return_value
        response.status_code = 200
        response.json.return_value = {"choices": [{"message": {"content": "Answer", "reasoning_content": "Internal trace"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 4}}
        content, finish, usage = call_chat(session, "https://server.example/v1", {}, "")
        self.assertEqual((content, finish, usage), ("Answer", "stop", {"completion_tokens": 4}))


if __name__ == "__main__":
    unittest.main()
