"""Offline safety/data-path tests; GPU compatibility is established only on Kaggle."""
import base64
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import zlib

from scripts import build_kaggle_training_smoke as builder
from scripts.build_training_pilot import validate_training_payload
from pilot_snapshot import materialize_pilot_snapshot
from scripts.kaggle_training import (DEFAULT_SETTINGS, PROJECTIONS, configure_probe_stop,
                                     run_smoke, select_targets, validate_settings)

PROJECT = Path(__file__).resolve().parents[1]


class KaggleTrainingTests(unittest.TestCase):
    def test_operational_probe_uses_actual_training_boundary_and_records_historical_eos(self):
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: 248046,
                                    convert_ids_to_tokens=lambda token_id: "<|im_end|>")
        generation = SimpleNamespace(eos_token_id=248044)
        report = configure_probe_stop(tokenizer, generation)
        self.assertEqual(generation.eos_token_id, 248046)
        self.assertEqual(report["original_config_eos_token_id"], 248044)
        self.assertTrue(report["differs_from_original_config"])

    def test_operational_probe_rejects_missing_tokenizer_answer_boundary(self):
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: None)
        with self.assertRaisesRegex(RuntimeError, "assistant boundary"):
            configure_probe_stop(tokenizer, SimpleNamespace(eos_token_id=248044))

    def test_target_discovery_covers_hybrid_blocks_and_excludes_other_components(self):
        layers = ["linear_attention", "full_attention"]
        expected = [f"model.layers.{i}.{group}.{leaf}" for i, attention in enumerate(["linear_attn", "self_attn"])
                    for group in [attention, "mlp"] for leaf in PROJECTIONS[group]]
        names = expected + ["model.visual.blocks.0.attn.q_proj", "lm_head", "model.embed_tokens"]
        self.assertEqual(select_targets(names, layers), sorted(expected))
        with self.assertRaisesRegex(ValueError, "missing"):
            select_targets(expected[1:], layers)
        with self.assertRaisesRegex(ValueError, "Unexpected decoder"):
            select_targets(expected + ["model.layers.0.new_branch.proj"], layers)

    def test_preflight_cannot_silently_become_long_or_change_label_policy(self):
        validate_settings(DEFAULT_SETTINGS)
        for update in [{"steps": 100}, {"packing": True}, {"micro_batch_size": 2},
                       {"truncation": "right"}, {"learning_rate": float("nan")},
                       {"revision": "main"}, {"amp_initial_scale": 0},
                       {"amp_initial_scale": float("inf")}, {"amp_initial_scale": True}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_settings({**DEFAULT_SETTINGS, **update})

    def test_local_training_stops_before_importing_gpu_libraries(self):
        if not Path('/kaggle/working').exists():
            with self.assertRaisesRegex(RuntimeError, "Kaggle"):
                run_smoke('/missing', '/missing')

    def test_generated_notebook_compiles_and_embeds_exact_reviewed_payload_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = materialize_pilot_snapshot(PROJECT, temporary, builder.BUNDLED_PATHS.values())
            notebook = builder.make_notebook(snapshot)
        self.assertEqual(len({c['id'] for c in notebook['cells']}), len(notebook['cells']))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code' and cell['id'] != 'install':
                compile(''.join(cell['source']), cell['id'], 'exec')
        bundle_cell = next(c for c in notebook['cells'] if c['id'] == 'bundle')
        source = ''.join(bundle_cell['source'])
        expression = source.split('BUNDLED_FILES = ', 1)[1].split('\n', 1)[0]
        import ast
        bundle = ast.literal_eval(expression)
        decoded = {k: zlib.decompress(base64.b64decode(v['zlib_base64'])) for k, v in bundle.items()}
        self.assertEqual(set(decoded), set(builder.BUNDLED_PATHS))
        for name, relative in builder.BUNDLED_PATHS.items():
            self.assertEqual(decoded[name], (PROJECT / relative).read_bytes())
        rows = validate_training_payload(decoded['train_seed.jsonl'], json.loads(decoded['pilot_lock.json']))
        self.assertEqual(len(rows), 500)
        self.assertTrue(all(set(row) == {'example_id', 'messages'} for row in rows))


if __name__ == '__main__':
    unittest.main()
