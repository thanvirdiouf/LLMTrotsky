"""Preparation-order invariants without model weights or GPU libraries."""
from types import SimpleNamespace
import unittest

from scripts.kbit_preparation import prepare_frozen_quantized_base


class Parameter:
    def __init__(self, device="cuda:0", dtype="float16", large=False):
        self.device, self.dtype, self.large = device, dtype, large
        self.requires_grad = True


class Module:
    def __init__(self, parameter, events, name):
        self.parameter, self.events, self.name = parameter, events, name

    def parameters(self):
        return iter([self.parameter])

    def to(self, *, device=None, dtype=None):
        self.events.append((self.name, device, dtype, self.parameter.device))
        if device is not None:
            self.parameter.device = device
        if dtype is not None:
            self.parameter.dtype = dtype
        return self


class Model:
    def __init__(self, tied=False):
        self.events = []
        self.input = Module(Parameter(large=True), self.events, "input")
        self.output = self.input if tied else Module(Parameter(large=True), self.events, "output")
        self.norm = Parameter()
        self.config = SimpleNamespace(use_cache=True)

    def get_input_embeddings(self):
        return self.input

    def get_output_embeddings(self):
        return self.output

    def named_parameters(self):
        yield "embed_tokens.weight", self.input.parameter
        if self.output is not self.input:
            yield "lm_head.weight", self.output.parameter
        yield "norm.weight", self.norm

    def parameters(self):
        return (parameter for _, parameter in self.named_parameters())


class KbitPreparationTests(unittest.TestCase):
    def prepare(self, model, **kwargs):
        self.assertEqual(kwargs, {"use_gradient_checkpointing": self.training,
                                "gradient_checkpointing_kwargs": {"use_reentrant": False}})
        for parameter in model.parameters():
            # Model the observed failure: a large FP32 cast on GPU would OOM.
            if parameter.large and parameter.device != "cpu":
                raise MemoryError("transient FP32 GPU embedding allocation")
            parameter.requires_grad = False
            parameter.dtype = "float32"
        return model

    def run_preparation(self, model, training=True, prepare=None):
        self.training = training
        return prepare_frozen_quantized_base(model, dtype="float16", device="cuda:0",
            prepare_kbit=prepare or self.prepare, training=training)

    def test_large_casts_happen_on_cpu_and_norms_keep_prepared_precision(self):
        model = Model()
        self.assertIs(self.run_preparation(model), model)
        self.assertEqual(model.events, [
            ("input", "cpu", None, "cuda:0"),
            ("output", "cpu", None, "cuda:0"),
            ("input", None, "float16", "cpu"),
            ("input", "cuda:0", None, "cpu"),
            ("output", None, "float16", "cpu"),
            ("output", "cuda:0", None, "cpu"),
        ])
        self.assertTrue(all(parameter.device == "cuda:0" for parameter in model.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(model.norm.dtype, "float32")
        self.assertEqual(model.input.parameter.dtype, "float16")
        self.assertEqual(model.output.parameter.dtype, "float16")
        self.assertFalse(model.config.use_cache)

    def test_reload_uses_the_same_staging_and_precision_without_checkpointing(self):
        model = Model()
        self.run_preparation(model, training=False)
        self.assertEqual(model.norm.dtype, "float32")
        self.assertTrue(all(parameter.device == "cuda:0" for parameter in model.parameters()))

    def test_same_embedding_module_is_staged_only_once(self):
        model = Model(tied=True)
        self.run_preparation(model)
        self.assertEqual(len(model.events), 3)

    def test_failed_preparation_propagates_original_error_before_gpu_restoration(self):
        model = Model()

        def fail(model, **kwargs):
            raise FloatingPointError("original preparation failure")

        with self.assertRaisesRegex(FloatingPointError, "original preparation failure"):
            self.run_preparation(model, prepare=fail)
        self.assertTrue(all(event[1] == "cpu" for event in model.events))

    def test_no_cpu_parameter_may_escape_successful_preparation(self):
        model = Model()
        model.norm.device = "cpu"
        with self.assertRaisesRegex(RuntimeError, "Base parameters remain outside"):
            self.run_preparation(model)


if __name__ == "__main__":
    unittest.main()
