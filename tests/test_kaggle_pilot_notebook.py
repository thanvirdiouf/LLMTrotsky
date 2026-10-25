"""Verify the standalone pilot notebook and its payloads without a GPU."""
import ast
import base64
import builtins
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import symtable
import tempfile
import unittest
import zlib

from scripts import build_kaggle_pilot as builder
from scripts.build_kaggle_training_smoke import PACKAGES
from scripts.build_training_pilot import validate_training_payload, validate_lock
from pilot_snapshot import materialize_pilot_snapshot


PROJECT = Path(__file__).resolve().parents[1]


def cells_by_id(notebook):
    return {cell['id']: ''.join(cell['source']) for cell in notebook['cells']}


def unpack(notebook):
    source = cells_by_id(notebook)['bundle']
    literal = source.split('BUNDLED_FILES = ', 1)[1].split('\n', 1)[0]
    bundle = ast.literal_eval(literal)
    decoded = {name: zlib.decompress(base64.b64decode(item['zlib_base64']))
               for name, item in bundle.items()}
    return bundle, decoded


class KagglePilotNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.snapshot = materialize_pilot_snapshot(PROJECT, temporary.name, builder.BUNDLED_PATHS.values())
        cls.notebook = builder.make_notebook(cls.snapshot)
        cls.cells = cells_by_id(cls.notebook)
        cls.bundle, cls.decoded = unpack(cls.notebook)

    def test_regular_python_cells_compile_and_ids_are_unique(self):
        self.assertEqual(len(self.cells), len(self.notebook['cells']))
        for cell in self.notebook['cells']:
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                if cell['id'] != 'install':
                    compile(''.join(cell['source']), cell['id'], 'exec')
        install = self.cells['install']
        self.assertTrue(install.startswith('%pip install -q '))
        for package, version in PACKAGES.items():
            self.assertIn(f'{package}=={version}', install)

    def test_every_bundle_file_has_exact_source_bytes_and_sha256(self):
        self.assertEqual(set(self.bundle), set(builder.BUNDLED_PATHS))
        for name, relative in builder.BUNDLED_PATHS.items():
            with self.subTest(name=name):
                payload = self.decoded[name]
                self.assertEqual(hashlib.sha256(payload).hexdigest(), self.bundle[name]['sha256'])
                self.assertEqual(payload, (PROJECT / relative).read_bytes())
                if name.endswith('.py'):
                    compile(payload, name, 'exec')

    def test_bundled_python_has_no_unbound_global_names(self):
        # Compilation alone misses a runtime NameError behind the Kaggle guard.
        for name, payload in self.decoded.items():
            if not name.endswith('.py'):
                continue
            root = symtable.symtable(payload.decode(), name, 'exec')
            defined = {symbol.get_name() for symbol in root.get_symbols()
                       if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}
            pending = [root]
            while pending:
                scope = pending.pop()
                pending.extend(scope.get_children())
                for symbol in scope.get_symbols():
                    if symbol.is_referenced() and symbol.is_global():
                        with self.subTest(file=name, scope=scope.get_name(), symbol=symbol.get_name()):
                            self.assertTrue(symbol.get_name() in defined
                                            or hasattr(builtins, symbol.get_name())
                                            or symbol.get_name() == '__file__')

    def test_all_500_reviewed_rows_are_exact_and_contamination_guarded(self):
        lock = json.loads(self.decoded['pilot_lock.json'])
        rows = validate_training_payload(self.decoded['train_seed.jsonl'], lock)
        self.assertEqual(len(rows), 500)
        self.assertEqual(len({row['example_id'] for row in rows}), 500)
        self.assertTrue(all(set(row) == {'example_id', 'messages'} for row in rows))
        self.assertTrue(all(set(message) == {'role', 'content'} for row in rows for message in row['messages']))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            validate_training_payload(self.decoded['train_seed.jsonl'] + b'\n', lock)

    def test_evaluation_contains_only_the_original_model_inputs(self):
        original = {row['case_id']: row for row in
                    (json.loads(line) for line in (PROJECT / 'evaluation/cases.jsonl').read_text().splitlines())}
        rows = [json.loads(line) for line in self.decoded['pilot_evaluation_inputs.jsonl'].splitlines()]
        self.assertEqual(len(rows), 36)
        self.assertEqual({row['case_id'] for row in rows}, set(original))
        forbidden = {'reference_evidence', 'reference_checks', 'evidence_sha256',
                     'response', 'answer', 'generated_text', 'source_evidence'}
        for row in rows:
            with self.subTest(case_id=row['case_id']):
                self.assertFalse(set(row) & forbidden)
                self.assertEqual(row['messages'], [
                    {'role': message['role'], 'content': message['content']}
                    for message in original[row['case_id']]['messages']])
                self.assertTrue(all(set(message) == {'role', 'content'} for message in row['messages']))

    def test_unpack_and_portable_semantic_validation_run_in_clean_cpu_process(self):
        # This executes the actual notebook cell using only stdlib/helper code.
        # A fresh interpreter also proves standalone conditional imports work.
        with tempfile.TemporaryDirectory() as temporary:
            bundle_path = Path(temporary) / 'bundle'
            source = self.cells['bundle'].replace(
                "Path('/kaggle/working/trotsky_pilot_bundle')", f'Path({str(bundle_path)!r})')
            completed = subprocess.run([sys.executable, '-'], input=source, cwd=temporary,
                                       capture_output=True, text=True, check=True)
            self.assertIn('Verified reviewed training rows: 500', completed.stdout)
            self.assertIn('Verified model-input-only evaluation cases: 36', completed.stdout)
            self.assertNotIn('torch', completed.stderr)
            self.assertEqual({path.name for path in bundle_path.iterdir() if path.is_file()},
                             set(builder.BUNDLED_PATHS))

    def test_entry_point_uses_pilot_api_and_resume_is_explicit(self):
        self.assertIn('from pilot_protocol import DEFAULT_SETTINGS', self.cells['settings'])
        self.assertIn('SETTINGS = deepcopy(DEFAULT_SETTINGS)', self.cells['settings'])
        self.assertIn('RESUME_CHECKPOINT = None', self.cells['settings'])
        self.assertIn("OUT = Path('/kaggle/working/trotsky_qlora_pilot')", self.cells['settings'])
        self.assertEqual(self.cells['train-save-reload-evaluate'],
                         'RUN_DIR = run_pilot(BUNDLE, OUT, SETTINGS, resume_checkpoint=RESUME_CHECKPOINT)\n')
        self.assertNotIn('run_smoke(', ''.join(self.cells.values()))

    def test_live_inventory_rejects_new_evaluation_without_rewriting_old_lock(self):
        frozen = json.loads((PROJECT / 'training/pilot_lock.json').read_bytes())
        live_inventory = sorted(p.relative_to(PROJECT).as_posix()
                                for p in (PROJECT / 'evaluation').rglob('*') if p.is_file())
        self.assertNotEqual(live_inventory, frozen['evaluation_artifact_inventory'])
        with self.assertRaisesRegex(ValueError, 'Pilot lock is stale|Excluded evaluation/generated passage'):
            validate_lock(PROJECT, frozen)
        self.assertEqual(validate_lock(self.snapshot, frozen), frozen)

    def test_generation_is_deterministic_and_saved_notebook_matches_frozen_snapshot(self):
        self.assertEqual(self.notebook, builder.make_notebook(self.snapshot))
        path = PROJECT / 'notebooks/kaggle_qlora_pilot.ipynb'
        self.assertEqual(json.loads(path.read_text()), self.notebook)


if __name__ == '__main__':
    unittest.main()
