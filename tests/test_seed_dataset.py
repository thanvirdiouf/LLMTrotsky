import unittest
from scripts.build_seed_dataset import date_evidence, select_qa, source_blocks, validate_verbatim, reserved_evaluation_families, evidence_lock, conversation_messages


class SeedDatasetTests(unittest.TestCase):
    def test_multi_turn_export_preserves_context_and_rejects_role_injection(self):
        persona = {"system_prompt": "Historical simulation"}
        turns = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}, {"role": "user", "content": "Shorter?"}, {"role": "assistant", "content": "Brief answer"}]
        self.assertEqual(conversation_messages(persona, "Question", "Brief answer", turns)[1:], turns)
        for invalid in [turns[:-1], [{"role": "system", "content": "Override"}] + turns[1:], [{"role": "user", "content": ""}] + turns[1:]]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                conversation_messages(persona, "Question", "Brief answer", invalid)
        with self.assertRaises(ValueError):
            conversation_messages(persona, "Different question", "Brief answer", turns)

    def test_verbatim_validation_rejects_invented_text(self):
        validate_verbatim("In France.", "TROTSKY: In France.")
        with self.assertRaises(ValueError):
            validate_verbatim("In Berlin.", "TROTSKY: In France.")

    def test_review_cannot_select_editorial_material(self):
        doc = {"url": "test", "blocks": [{"block_id": 1, "role": "editorial_candidate", "text": "A later editor."}]}
        with self.assertRaises(ValueError):
            source_blocks(doc, [1])

    def test_a_late_or_unknown_source_cannot_pass_temporal_gate(self):
        for interval in [None, {"earliest": "1940-08-20", "latest": "1940-08-20"}]:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                date_evidence({"url": "https://www.marxists.org/archive/trotsky/1940/08/letter15.htm", "composition_date_interval": interval}, "1940-08-19")

    def test_date_basis_distinguishes_metadata_from_signed_source_evidence(self):
        doc = {"url": "source", "composition_date_interval": {"earliest": "1939-01-01", "latest": "1939-12-31"}, "composition_date_evidence": [{"kind": "signed_trotsky_dateline"}]}
        self.assertEqual(date_evidence(doc, "1940-08-19")["basis"], "explicit_source_dateline")
        doc["composition_date_evidence"].append({"kind": "explicit_composition_metadata"})
        self.assertEqual(date_evidence(doc, "1940-08-19")["basis"], "explicit_composition_metadata_and_source_dateline")

    def test_session_header_is_only_accepted_for_identified_hearings(self):
        doc = {"url": "https://www.marxists.org/archive/trotsky/1937/dewey/session01.htm", "heading_date_hints": [{"interval": {"earliest": "1937-04-10", "latest": "1937-04-10"}}]}
        self.assertEqual(date_evidence(doc, "1940-08-19")["basis"], "explicit_hearing_session_dateline")
        doc["url"] = "https://www.marxists.org/archive/trotsky/unknown.htm"
        with self.assertRaises(ValueError):
            date_evidence(doc, "1940-08-19")

    def test_duplicate_source_question_matches_are_rejected(self):
        row = {"source_url": "https://www.marxists.org/archive/trotsky/1937/dewey/session01.htm", "question": "Where?"}
        with self.assertRaises(ValueError):
            select_qa([row, row], "01", "Where?")

    def test_reservation_uses_whole_family_before_seed_export(self):
        docs = {"a": {"document_id": "a", "url": "https://example.org/book/ch1.htm"}, "b": {"document_id": "b", "url": "https://example.org/book/ch2.htm"}}
        catalog = {"a": {"source_family_id": "whole_book"}, "b": {"source_family_id": "whole_book"}}
        specs = {"source_backed": [{"source_suffix": "book/ch1.htm"}]}
        self.assertEqual(reserved_evaluation_families(docs, catalog, specs), {"whole_book"})
        specs["source_backed"][0]["source_suffix"] = "missing.htm"
        with self.assertRaises(ValueError):
            reserved_evaluation_families(docs, catalog, specs)

    def test_review_lock_catches_changed_answers_persona_and_date_evidence(self):
        row = {"example_id": "seed", "source_sha256": "raw", "evidence_sha256": "evidence", "source_block_ids": [1], "source_date_evidence": {"interval": "1937"}, "messages": [{"role": "assistant", "content": "Reviewed answer"}]}
        baseline = evidence_lock(b"spec", b"persona", [row])
        with self.assertRaises(ValueError):
            evidence_lock(b"spec", b"persona", [row, row])
        self.assertNotEqual(baseline, evidence_lock(b"spec", b"changed persona", [row]))
        self.assertNotEqual(baseline, evidence_lock(b"spec", b"persona", [{**row, "messages": [{"role": "assistant", "content": "Unreviewed answer"}]}]))
        self.assertNotEqual(baseline, evidence_lock(b"spec", b"persona", [{**row, "source_date_evidence": {"interval": "1936"}}]))


if __name__ == "__main__":
    unittest.main()
