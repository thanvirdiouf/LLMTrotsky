import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import test_ghost_reviewed_display as fixtures
from grounded_chat import fingerprint
from grounded_chat_v2 import EvidenceError
from preview_reviewed_selection import compact_case, build_preview
from reviewed_source_display import records_for_packets, render_selection


class StatementParser(HTMLParser):
    def __init__(self):
        super().__init__(); self.in_statement = False; self.statements = []

    def handle_starttag(self, tag, attrs):
        if tag == 'p' and ('class', 'statement') in attrs:
            self.in_statement = True; self.statements.append('')

    def handle_endtag(self, tag):
        if tag == 'p': self.in_statement = False

    def handle_data(self, data):
        if self.in_statement: self.statements[-1] += data


class ReviewedPreviewTests(unittest.TestCase):
    def fixture(self):
        collection, ledger = fixtures.ReviewedDisplayTests().fixture()
        collection['packets'][0]['observed_date'] = '2026-10-26'
        collection['packets_sha256'] = fingerprint(collection['packets'])
        ledger['collection_sha256'] = collection['packets_sha256']
        ledger['claims'][0]['statement'] = '<script>alert("x")</script> & uncertain testimony.'
        records = records_for_packets(collection, ledger, ['source'])
        row = {**render_selection(list(records), records), 'case_id': 'case',
               'question': 'Explain <all> relevant details.',
               'source_context_packet_ids': ['source'], 'status': 'reviewed_summaries_selected'}
        return collection, ledger, records, row

    def test_all_six_statements_remain_exact_and_provenance_accessible(self):
        _, _, records, row = self.fixture()
        fragment, report = compact_case(row, records, 0)
        parser = StatementParser(); parser.feed(fragment)
        self.assertEqual(parser.statements, [r['statement'] for r in records.values()])
        self.assertIn('source date 1930-01-05; observed 2026-10-26', fragment)
        for value in ('Actual speaker', 'body_candidate', 'Bounded context', 'reported_allegation', 'anchor_text'):
            self.assertIn(value, fragment)
        self.assertEqual(report['selected_claim_ids'], row['selected_claim_ids'])
        self.assertFalse(report['readability_validated'])
        self.assertFalse(report['requested_task_coverage_verified'])

    def test_source_and_question_markup_are_escaped(self):
        _, _, records, row = self.fixture()
        fragment, _ = compact_case(row, records, 0)
        self.assertNotIn('<script>', fragment)
        self.assertIn('&lt;script&gt;', fragment)
        self.assertIn('&lt;all&gt;', fragment)

    def test_altered_response_record_and_blocked_output_are_rejected(self):
        _, _, records, row = self.fixture()
        for field, value in [('response', 'Rewritten facts'), ('source_records', []),
                             ('status', 'blocked'), ('memory_updates', ['promoted'])]:
            altered = {**row, field: value}
            with self.subTest(field=field), self.assertRaises(EvidenceError):
                compact_case(altered, records, 0)

    def test_replay_checks_original_question_before_creating_output(self):
        collection, ledger, records, row = self.fixture()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); bundle = root / 'bundle'; bundle.mkdir()
            for name, value in [('collection.json', collection), ('ledger.json', ledger),
                                ('audit_inputs.json', [{'case_id': 'case', 'question': row['question'],
                                 'source_packet_ids': ['source'], 'available_record_fingerprints':
                                 {cid: r['record_fingerprint'] for cid, r in records.items()}}])]:
                (bundle / name).write_text(json.dumps(value))
            answers = root / 'answers.jsonl'; answers.write_text(json.dumps(row) + '\n')
            original = answers.read_bytes()
            report = build_preview(bundle, answers, root / 'valid')
            self.assertEqual(report['model_calls'], 0)
            self.assertEqual(answers.read_bytes(), original)
            self.assertTrue((root / 'valid' / 'preview.html').is_file())
            with self.assertRaises(FileExistsError): build_preview(bundle, answers, root / 'valid')
            altered = copy.deepcopy(row); altered['question'] = 'Changed question'
            answers.write_text(json.dumps(altered) + '\n')
            with self.assertRaises(EvidenceError): build_preview(bundle, answers, root / 'invalid')
            self.assertFalse((root / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
