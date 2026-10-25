import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup
from scripts.extract_mia import blocks_from_html, document, qa_candidates, parse_speaker, signed_dateline_evidence, signed_letter_evidence
from scripts.fetch_mia import allowed, canonical


def blocks(html):
    return blocks_from_html(BeautifulSoup(html, "lxml"))


def transcript(html):
    return {
        "document_id": "test", "url": "https://www.marxists.org/archive/trotsky/test.htm",
        "source_sha256": "test", "blocks": blocks(html),
    }


class ExtractionTests(unittest.TestCase):
    def test_inline_markup_does_not_split_words_or_add_punctuation_spaces(self):
        result = blocks('<p><span>U</span>ndoubtedly the <em>Iskra</em>, and <a href="#x">Lenin</a>.</p>')
        self.assertEqual(result[0]["text"], "Undoubtedly the Iskra, and Lenin.")

    def test_source_wrapping_differs_from_explicit_br(self):
        result = blocks("<p>one\n two<br>three</p>")
        self.assertEqual(result[0]["text"], "one two\nthree")

    def test_nested_layout_and_inline_containers_keep_paragraphs(self):
        result = blocks('<center><font><p>First.</p><p>Second <i>paragraph</i>.</p></font></center>')
        self.assertEqual([b["text"] for b in result], ["First.", "Second paragraph."])

    def test_comments_scripts_and_styles_are_not_text(self):
        result = blocks('<p>Text<!--hidden--><script>bad()</script><style>x{}</style>.</p>')
        self.assertEqual(result[0]["text"], "Text.")

    def test_editorial_intro_metadata_and_footer_are_separate(self):
        result = blocks('<p class="information">Written: 1937</p><p class="head">A later introduction.</p><p>Speech.</p><footer><p>Navigation</p></footer>')
        self.assertEqual([b["role"] for b in result], ["metadata", "editorial_candidate", "body_candidate", "navigation"])

    def test_transcriber_comment_region_stops_at_hr(self):
        result = blocks('<h5>Transcriber’s comments:</h5><h5>A note written in 2002.</h5><hr><p>Trotsky’s letter.</p>')
        self.assertEqual([b["role"] for b in result], ["editorial_candidate", "editorial_candidate", "body_candidate"])

    def test_notes_marker_does_not_remove_preceding_body(self):
        result = blocks('<p>Authored paragraph.</p><h4>Footnotes</h4><p>Whose note?</p>')
        self.assertEqual([b["role"] for b in result], ["body_candidate", "note_ownership_unverified", "note_ownership_unverified"])

    def test_editorial_table_does_not_consume_article_outside_it(self):
        result = blocks('<table><tr><td><p>Ed. Note: Written later.</p><p>Another editor paragraph.</p></td></tr></table><h4>* * *</h4><p>Author’s original words.</p>')
        self.assertEqual([b["role"] for b in result], ["editorial_candidate", "editorial_candidate", "separator", "body_candidate"])

    def test_unknown_speaker_ends_trotsky_turn(self):
        result = qa_candidates(transcript('<p>GOLDMAN: Where did you live?</p><p>TROTSKY: In France.</p><p>FRANKEL: My own testimony.</p><p>Further testimony.</p>'))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["answer"], "In France.")

    def test_proper_case_and_interpreter_labels_end_speech(self):
        for other in ["Cannon", "RUEHLE (through interpreter)", "GOLDMAN (reading)"]:
            with self.subTest(other=other):
                result = qa_candidates(transcript(f'<p>Goldman: Where did you live?</p><p>Trotsky: In France.</p><p>{other}: Other words.</p>'))
                self.assertEqual(result[0]["answer"], "In France.")

    def test_prose_before_a_colon_is_not_a_speaker(self):
        self.assertIsNone(parse_speaker("In the first place: we must examine the facts."))
        self.assertIsNone(parse_speaker("For example: consider this case."))

    def test_mrs_trotsky_is_not_leon_trotsky(self):
        self.assertEqual(parse_speaker("MRS. TROTSKY: My words.")[0], "MRS. TROTSKY")
        result = qa_candidates(transcript('<p>GOLDMAN: Where?</p><p>MRS. TROTSKY: At home.</p>'))
        self.assertEqual(result, [])

    def test_honorific_and_continuation_paragraphs(self):
        result = qa_candidates(transcript('<p>MR. GOLDMAN: When did you arrive?</p><p>MR. TROTSKY: In 1937.</p><p>I travelled by ship.</p>'))
        self.assertEqual(result[0]["answer"], "In 1937.\n\nI travelled by ship.")
        self.assertFalse(result[0]["approved_for_training"])

    def test_notes_and_stage_directions_are_not_speech(self):
        result = qa_candidates(transcript('<p>GOLDMAN: Where?</p><p>TROTSKY: In France.</p><h5>(Laughter.)</h5><h4>Footnotes</h4><p>An editor explains events after 1940.</p>'))
        self.assertEqual(result[0]["answer"], "In France.")

    def test_br_separated_turns(self):
        result = qa_candidates(transcript('<p>GOLDMAN: Where?<br>TROTSKY: In France.<br>FRANKEL: Other words.</p>'))
        self.assertEqual(result[0]["answer"], "In France.")

    def test_do_not_pair_question_and_answer_across_sections(self):
        result = qa_candidates(transcript('<p>GOLDMAN: Where?</p><h2>Another session</h2><p>TROTSKY: A new statement.</p>'))
        self.assertEqual(result, [])

    def test_quoted_passages_are_marked_for_review(self):
        result = qa_candidates(transcript('<p>GOLDMAN: What did he say?</p><p>TROTSKY: I quote:</p><blockquote><p>Another person’s words.</p></blockquote>'))
        self.assertIn("quoted_material_requires_review", result[0]["quality_flags"])

    def test_source_hash_is_checked_and_dates_are_distinct(self):
        raw = b'<title>Journal</title><p class="information">Written; December 1936-January 1937<br>First published: 1941<br>Translated. By a translator.</p><p>Original words.</p>'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'page.html').write_bytes(raw)
            record = {"raw_path": "page.html", "sha256": hashlib.sha256(raw).hexdigest(), "html_title": "Journal", "url": "https://www.marxists.org/archive/trotsky/1937/01/journal.htm", "final_url": "test", "fetched_utc": "test"}
            result = document(record, root, {})
            self.assertEqual(result["composition_year_candidates"], [1936, 1937])
            self.assertEqual(result["metadata_fields_raw"]["first published"], ["1941"])
            self.assertNotIn("written_field_contains_post_1940_date", result["quality_flags"])
            self.assertNotIn("1941", result["body_candidate_text"])
            record["sha256"] = "bad"
            with self.assertRaises(ValueError):
                document(record, root, {})

    def test_short_signed_datelines_are_explicit_evidence(self):
        result = signed_dateline_evidence(blocks('<p class="author">L. Trotsky<br>Moscow, February 26, 1922</p>'))
        self.assertEqual(result[0]["kind"], "signed_trotsky_dateline")
        self.assertEqual(result[0]["block_ids"], [0])
        self.assertIn("February 26, 1922", result[0]["text"])

    def test_one_signed_preface_does_not_date_a_collection_of_prefaces(self):
        html = '<p>New preface.</p><p>L. Trotsky<br>Moscow<br>12 January 1922</p><h4>Earlier preface</h4><p>Earlier words.</p><p>Vienna,<br>October 1909</p>'
        self.assertEqual(signed_dateline_evidence(blocks(html)), [])

    def test_unsigned_dates_other_authors_and_publication_citations_are_not_evidence(self):
        for html in [
            '<p class="date">December 25, 1922</p><p class="date">LENIN<br>January 4, 1923</p>',
            '<p class="author">George Novack<br>March 1, 1968</p>',
            '<p class="author">L. Trotsky<br>Pravda, February 26, 1922</p>',
            '<p class="author">L. Trotsky<br>March 1, 1968</p>',
            '<blockquote><p>L. Trotsky<br>March 1, 1938</p></blockquote>',
            '<p class="quoteb">L. Trotsky<br>March 1, 1938</p>',
            '<p class="title">Leon Trotsky 1900</p>',
            '<p>Leon Trotsky wrote his reply on March 1, 1938.</p>',
            '<p class="head">L. Trotsky<br>March 1, 1938</p>',
        ]:
            with self.subTest(html=html):
                self.assertEqual(signed_dateline_evidence(blocks(html)), [])

    def test_complete_letter_binds_header_date_to_its_signature(self):
        html = '<p>August 12, 1940</p><p>Dear Friend:</p><p>The Texas story is important.</p><p class="author">Fraternally,<br>L. TROTSKY</p>'
        result = signed_letter_evidence(blocks(html))
        self.assertEqual(result[0]["block_ids"], [0, 3])
        self.assertIn("August 12, 1940", result[0]["text"])

    def test_a_single_header_does_not_date_mixed_letters_or_quoted_correspondence(self):
        for html in [
            '<p>August 12, 1940</p><p>Dear Friend:</p><p>First letter.</p><h4>Another letter</h4><p>Second letter.</p><p>Fraternally,<br>L. Trotsky</p>',
            '<p>August 12, 1940</p><p>Dear Friend:</p><p>First letter.</p><p>Dear Other Friend:</p><p>Fraternally,<br>L. Trotsky</p>',
            '<blockquote><p>August 12, 1940</p><p>Dear Friend:</p><p>Words.</p><p>Fraternally,<br>L. Trotsky</p></blockquote>',
            '<p>August 12, 1940</p><p>Dear Friend:</p><p>Words.</p><p>George Novack</p>',
            '<p>Published August 12, 1940</p><p>Dear Friend:</p><p>Words.</p><p>Fraternally,<br>L. Trotsky</p>',
        ]:
            with self.subTest(html=html):
                self.assertEqual(signed_letter_evidence(blocks(html)), [])

    def test_conflicting_metadata_and_signed_letter_dates_keep_both_bounds(self):
        raw = b'<p class="information">Written: 2 August 1940</p><p>August 12, 1940</p><p>Dear Friend:</p><p>The Texas story is important.</p><p>Fraternally,<br>L. TROTSKY</p>'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'page.html').write_bytes(raw)
            record = {"raw_path": "page.html", "sha256": hashlib.sha256(raw).hexdigest(), "html_title": "test", "url": "https://www.marxists.org/archive/trotsky/1940/test.htm", "final_url": "test", "fetched_utc": "test"}
            result = document(record, root, {})
            interval = result["composition_date_interval"]
            self.assertEqual((interval["earliest"], interval["latest"]), ("1940-08-02", "1940-08-12"))
            self.assertIn("composition_date_evidence_conflict", result["quality_flags"])
            self.assertEqual(len(result["composition_date_evidence"]), 2)

    def test_document_retains_auditable_composition_evidence(self):
        raw = b'<p class="information">Written on April 3, 1927.<br>Published: 1938</p><p>Words.</p>'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'page.html').write_bytes(raw)
            record = {"raw_path": "page.html", "sha256": hashlib.sha256(raw).hexdigest(), "html_title": "test", "url": "https://www.marxists.org/archive/trotsky/1927/test.htm", "final_url": "test", "fetched_utc": "test"}
            result = document(record, root, {})
            self.assertEqual(result["composition_date_interval"]["latest"], "1927-04-03")
            self.assertEqual(result["composition_year_candidates"], [1927])
            self.assertEqual(result["composition_date_evidence"][0]["field"], "written on april 3, 1927")
            self.assertNotIn("composition_date_unverified", result["quality_flags"])

    def test_reviewed_editorial_boundary_requires_both_source_hash_and_prefix(self):
        raw = b'<p>Editorial Note: Published in 1941.</p><p>It appears here for the first time.</p><p>It is well that there is art in the world.</p><p>Author body.</p>'
        prefix = "It is well that there is art in the world."
        sha = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'page.html').write_bytes(raw)
            record = {"raw_path": "page.html", "sha256": sha, "html_title": "test", "url": "https://www.marxists.org/archive/trotsky/test.htm", "final_url": "test", "fetched_utc": "test"}
            with patch('scripts.extract_mia.REVIEWED_EDITORIAL_ENDS', {sha: (prefix,)}):
                result = document(record, root, {})
                self.assertEqual([b["role"] for b in result["blocks"]], ["editorial_candidate", "editorial_candidate", "body_candidate", "body_candidate"])
                self.assertNotIn("1941", result["body_candidate_text"])
                self.assertIn("Author body.", result["body_candidate_text"])
            for exceptions in [{"different-source-hash": (prefix,)}, {sha: ("Different authored paragraph.",)}]:
                with patch('scripts.extract_mia.REVIEWED_EDITORIAL_ENDS', exceptions):
                    result = document(record, root, {})
                    self.assertEqual(result["body_candidate_text"], "")


class CrawlScopeTests(unittest.TestCase):
    def test_canonical_and_excluded_urls(self):
        self.assertEqual(canonical("http://marxists.org/archive/trotsky/1937/x.htm#note"), "https://www.marxists.org/archive/trotsky/1937/x.htm")
        self.assertIsNone(canonical("https://example.com/archive/trotsky/x.htm"))
        self.assertFalse(allowed("https://www.marxists.org/archive/trotsky/photo/index.htm"))
        self.assertFalse(allowed("https://www.marxists.org/archive/trotsky/bio/index.htm"))
        self.assertFalse(allowed("https://www.marxists.org/archive/trotsky/book.pdf"))
        self.assertTrue(allowed("https://www.marxists.org/archive/trotsky/1937/dewey/session01.htm"))


if __name__ == "__main__":
    unittest.main()
