"""Execute the notebook's data/resume path offline; no model weights or GPU needed."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts import kaggle_parallel as runtime


PROJECT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((PROJECT / 'notebooks/kaggle_prompt_baseline.ipynb').read_text())
CELLS = {c['id']: ''.join(c['source']) for c in NOTEBOOK['cells']}


class KaggleNotebookTests(unittest.TestCase):
    def setUp(self):
        self.ns = vars(runtime)
        self.cases = self.ns['load_cases'](PROJECT / 'evaluation/cases.jsonl')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def row(self, case=None, run_id='run'):
        case = case or self.cases[0]
        return dict(run_id=run_id, model_id='model', case_id=case['case_id'], response='Answer, with\na newline.',
                    prompt_sha256=self.ns['fingerprint'](self.ns['model_messages'](case)), finish_reason='stop')

    def test_python_cells_compile_and_ids_are_unique(self):
        self.assertEqual(len(CELLS), len(NOTEBOOK['cells']))
        for c in NOTEBOOK['cells']:
            if c['cell_type'] == 'code' and c['id'] != 'install':
                source = CELLS[c['id']]
                if c['id'] == 'worker-source':
                    source = source.split('\n', 1)[1]
                    self.assertEqual(source, (PROJECT / 'scripts/kaggle_parallel.py').read_text())
                compile(source, c['id'], 'exec')

    def test_two_outputs_are_real_jsonl_and_resume_round_trips(self):
        path = self.root / 'responses.jsonl'
        rows = [self.row(c) for c in self.cases[:2]]
        with path.open('w') as stream:
            for row in rows:
                self.ns['append_jsonl'](stream, row)
        self.assertEqual(len(path.read_text().splitlines()), 2)
        self.assertEqual(self.ns['read_completed'](path, 'run', self.cases), rows)

    def test_resume_rejects_mixed_runs_duplicate_cases_and_changed_prompts(self):
        path = self.root / 'responses.jsonl'
        for rows in [[self.row(run_id='different')], [self.row(), self.row()],
                     [{**self.row(), 'prompt_sha256': 'bad'}]]:
            with path.open('w') as stream:
                for row in rows:
                    self.ns['append_jsonl'](stream, row)
            with self.assertRaises(ValueError):
                self.ns['read_completed'](path, 'run', self.cases)

    def test_dataset_discovery_rejects_ambiguity_and_modified_data(self):
        for folder in ['a', 'b']:
            (self.root / folder).mkdir()
            (self.root / folder / 'cases.jsonl').write_text('{}\n')
        with self.assertRaisesRegex(ValueError, 'Set CASES_PATH'):
            self.ns['load_cases'](input_root=self.root)
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.ns['load_cases'](self.root / 'a/cases.jsonl')

    def test_model_input_excludes_references_but_keeps_conversation(self):
        case = next(c for c in self.cases if c['category'] == 'multi_turn')
        self.assertEqual(self.ns['model_messages'](case), case['messages'])
        self.assertNotIn('reference_checks', self.ns['model_messages'](case)[-1])
        forged = {**case, 'messages': [{'role': 'user', 'content': 'question', 'reference_checks': 'secret'}]}
        self.assertEqual(self.ns['model_messages'](forged), [{'role': 'user', 'content': 'question'}])

    def test_review_contains_question_evidence_and_no_model_label(self):
        self.ns['export_review']([self.row()], self.cases, self.root)
        with (self.root / 'human_review.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]['source_evidence'], self.cases[0]['reference_evidence'])
        self.assertEqual(json.loads(rows[0]['messages']), self.cases[0]['messages'])
        self.assertNotIn('model_id', rows[0])
        self.assertEqual(rows[0]['temporal_consistency_0_4'], '')
        self.assertTrue((self.root / 'review_key.csv').exists())

    def test_truncation_including_multiple_eos_ids(self):
        finish = self.ns['finish_reason']
        self.assertEqual(finish([1, 2], [2, 3]), 'stop')
        self.assertEqual(finish([1, 2], 3), 'length')
        self.assertEqual(finish([], 3), 'length')

    def test_worker_runs_on_its_single_gpu_without_optional_generation_file(self):
        class TokenList(list):
            def tolist(self):
                return list(self)
            def __getitem__(self, key):
                result = super().__getitem__(key)
                return TokenList(result) if isinstance(key, slice) else result

        class Inputs(dict):
            def to(self, device):
                self.test_device = device
                return self

        captured = []
        def apply_template(messages, **kwargs):
            captured.append(messages)
            return Inputs(input_ids=SimpleNamespace(shape=(1, 5)))

        tok = SimpleNamespace(chat_template='enable_thinking', pad_token_id=0, eos_token_id=2,
                              apply_chat_template=apply_template, decode=lambda *a, **k: 'Response')
        text_config = SimpleNamespace(eos_token_id=[2, 3])
        config = SimpleNamespace(get_text_config=Mock(return_value=text_config))
        gen = SimpleNamespace(eos_token_id=[2, 3], to_dict=lambda: {'max_new_tokens': 512})
        generation_factory = SimpleNamespace(
            from_pretrained=Mock(side_effect=OSError('generation_config.json missing')),
            from_model_config=Mock(return_value=gen))
        model = Mock()
        model.generate.return_value = [TokenList([1, 1, 1, 1, 1, 42, 2])]
        loader = Mock(return_value=model)
        fake_torch = SimpleNamespace(float16='float16', set_num_threads=Mock(),
            inference_mode=contextlib.nullcontext, cuda=SimpleNamespace(is_available=lambda: True,
            device_count=lambda: 1, set_device=Mock(), synchronize=Mock()))
        fake_transformers = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tok),
            AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=loader),
            GenerationConfig=generation_factory, BitsAndBytesConfig=lambda **k: k, set_seed=Mock())
        output = self.root / 'responses.gpu-1.jsonl'
        jobs = runtime.make_jobs(self.cases, set(), [1], 2)[0][1]
        job = dict(snapshot_path='/cached/model', output_path=str(output), gpu_id=1, model_id='model',
            run_id='run', jobs=jobs, max_new_tokens=512, max_input_tokens=8192, seed=42,
            load_in_4bit=True, dtype_name='float16', expected_generation=gen.to_dict(),
            expected_template_sha256=runtime.digest(tok.chat_template.encode()))
        path = self.root / 'job.json'
        path.write_text(json.dumps(job))
        import sys
        with patch.dict(sys.modules, {'torch': fake_torch, 'transformers': fake_transformers}), \
             contextlib.redirect_stdout(io.StringIO()):
            runtime.run_worker(path)
        loader.assert_called_once()
        self.assertEqual(loader.call_args.kwargs['device_map'], {'': 0})
        self.assertTrue(loader.call_args.kwargs['local_files_only'])
        self.assertEqual(loader.call_args.kwargs['quantization_config']['bnb_4bit_quant_type'], 'nf4')
        self.assertFalse(gen.do_sample)
        self.assertEqual(gen.max_new_tokens, 512)
        generation_factory.from_pretrained.assert_not_called()
        generation_factory.from_model_config.assert_called_with(text_config)
        config.get_text_config.assert_called_with(decoder=True)
        rows = runtime.read_completed(output, 'run', self.cases)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['gpu_id'] == 1 and r['finish_reason'] == 'stop' for r in rows))
        self.assertTrue(all(r['generation_seconds'] >= 0 for r in rows))
        self.assertEqual(captured, [runtime.model_messages(c) for c in self.cases[:2]])


if __name__ == '__main__':
    unittest.main()
