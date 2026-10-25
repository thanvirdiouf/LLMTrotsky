"""Exercise checkpoint publication, corruption rejection and state restoration on CPU."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from scripts.pilot_checkpoints import (find_latest_checkpoint, load_checkpoint,
                                       save_checkpoint, verify_checkpoint)


class FakeModel:
    def __init__(self):
        self.weights = [1.0, 2.0]
        self.saves = []

    def save_pretrained(self, path, **kwargs):
        self.saves.append(kwargs)
        (path / 'adapter_config.json').write_text('{}')
        (path / 'adapter_model.safetensors').write_text(json.dumps(self.weights))


class StateObject:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return copy.deepcopy(self.state)

    def load_state_dict(self, state):
        self.state = copy.deepcopy(state)


class FakeTorch:
    def __init__(self):
        self.rng = [1, 2, 3]
        self.cuda_rng = [[4, 5, 6]]
        self.loads = []
        self.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                    get_rng_state_all=lambda: copy.deepcopy(self.cuda_rng),
                                    set_rng_state_all=self.set_cuda_rng)

    def set_cuda_rng(self, value):
        self.cuda_rng = copy.deepcopy(value)

    def get_rng_state(self):
        return copy.deepcopy(self.rng)

    def set_rng_state(self, value):
        self.rng = copy.deepcopy(value)

    def save(self, state, path):
        Path(path).write_text(json.dumps(state))

    def load(self, path, **kwargs):
        self.loads.append(kwargs)
        return json.loads(Path(path).read_text())


class PilotCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = FakeModel()
        self.optimizer = StateObject({'step': 1, 'momentum': [0.2, 0.3]})
        self.scaler = StateObject({'scale': 1.0})
        self.torch = FakeTorch()
        self.identity = {'dataset': 'frozen', 'settings': {'epochs': 1, 'seed': 42}}
        self.order = ['a', 'b', 'c']
        self.rng_before = random.getstate()
        self.addCleanup(random.setstate, self.rng_before)

    def metrics(self, step):
        path = self.root / 'metrics.jsonl'
        path.write_text(''.join(json.dumps({'example_id': self.order[i], 'step': i + 1,
                                           'verified_optimizer_step': i + 1}) + '\n' for i in range(step)))
        return path

    def save(self, step, **kwargs):
        return save_checkpoint(self.root / 'checkpoints', step, self.identity, self.order,
                               self.model, self.optimizer, self.scaler, self.torch,
                               metrics_path=self.metrics(step), **kwargs)

    def restore(self, path, **kwargs):
        def load_adapter(model, directory):
            model.weights = json.loads((directory / 'adapter_model.safetensors').read_text())
        return load_checkpoint(path, self.identity, self.order, self.model, self.optimizer,
                               self.scaler, self.torch, load_adapter=load_adapter, **kwargs)

    def test_round_trip_restores_weights_momenta_scale_rng_and_next_position(self):
        random.seed(98)
        saved_python = random.getstate()
        reference = random.Random()
        reference.setstate(saved_python)
        expected_draw = reference.random()
        path = self.save(1)
        self.model.weights = [99]
        self.optimizer.state = {'step': 20}
        self.scaler.state = {'scale': 65536}
        self.torch.rng = [99]
        self.torch.cuda_rng = [[99]]
        random.seed(900)
        self.assertEqual(self.restore(path), 1)
        self.assertEqual(self.model.weights, [1.0, 2.0])
        self.assertEqual(self.optimizer.state, {'step': 1, 'momentum': [0.2, 0.3]})
        self.assertEqual(self.scaler.state, {'scale': 1.0})
        self.assertEqual(self.torch.rng, [1, 2, 3])
        self.assertEqual(self.torch.cuda_rng, [[4, 5, 6]])
        self.assertEqual(random.random(), expected_draw)
        self.assertEqual(self.torch.loads, [{'map_location': 'cpu', 'weights_only': True}])
        self.assertFalse(self.model.saves[0]['save_embedding_layers'])
        manifest = verify_checkpoint(path, self.identity, self.order)
        self.assertEqual(manifest['next_index'], 1)
        self.assertEqual((path / 'training_metrics.jsonl').read_bytes(), self.metrics(1).read_bytes())

    def test_interrupted_save_keeps_previous_published_checkpoint(self):
        first = self.save(0)
        def fail(state, path):
            Path(path).write_bytes(b'partial')
            raise OSError('simulated interrupted serialization')
        with self.assertRaisesRegex(OSError, 'interrupted'):
            self.save(1, save_state=fail)
        self.assertEqual(find_latest_checkpoint(first.parent, self.identity, self.order), first)
        self.assertEqual(list(first.parent.glob('*.partial')), [])
        verify_checkpoint(first)

    def test_corrupt_latest_is_rejected_without_silent_fallback(self):
        self.save(0)
        latest = self.save(1)
        (latest / 'training_state.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'inventory/hash'):
            find_latest_checkpoint(latest.parent, self.identity, self.order)

    def test_partial_unpublished_directory_is_ignored_and_completed_immutable(self):
        first = self.save(0)
        (first.parent / '.checkpoint-000100.partial').mkdir()
        self.assertEqual(find_latest_checkpoint(first.parent), first)
        with self.assertRaises(FileExistsError):
            self.save(0)

    def test_changed_identity_order_or_missing_completion_marker_fail_before_restore(self):
        path = self.save(1)
        original = copy.deepcopy(self.model.weights)
        for identity, order in [({'dataset': 'other'}, self.order), (self.identity, list(reversed(self.order)))]:
            with self.subTest(identity=identity, order=order), self.assertRaises(ValueError):
                load_checkpoint(path, identity, order, self.model, self.optimizer, self.scaler,
                                self.torch, load_adapter=lambda *args: self.fail('must not mutate model'))
        self.assertEqual(self.model.weights, original)
        (path / 'COMPLETE').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            verify_checkpoint(path)

    def test_incoherent_metric_progress_is_rejected_before_publication(self):
        metric_path = self.metrics(0)
        with self.assertRaisesRegex(ValueError, 'exactly the completed'):
            save_checkpoint(self.root / 'checkpoints', 1, self.identity, self.order,
                            self.model, self.optimizer, self.scaler, self.torch, metrics_path=metric_path)
        self.assertIsNone(find_latest_checkpoint(self.root / 'checkpoints'))


if __name__ == '__main__':
    unittest.main()
