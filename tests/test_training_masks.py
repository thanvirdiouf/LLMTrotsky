"""Exercise token supervision offline, without downloading a tokenizer or model.

The fake tokenizer has transparent character IDs plus atomic chat controls. Its
text-only rendering models the pinned Qwen3.5 template's important asymmetry:
historical assistants are plain; assistants after the last user have an empty
thinking wrapper. Real tokenizer/template equality is checked again at Kaggle
runtime by the same helper.
"""
import re
import unittest

from scripts import training_masks as masks


class TransparentQwenTokenizer:
    all_special_tokens = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|vision_start|>"]
    eos_token_id = pad_token_id = 2

    def __init__(self):
        self.calls = []
        self.vocab = {text: index + 1 for index, text in enumerate(self.all_special_tokens)}

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        pattern = "(" + "|".join(re.escape(t) for t in self.all_special_tokens) + ")"
        output = []
        for piece in re.split(pattern, text):
            if piece in self.vocab:
                output.append(self.vocab[piece])
            else:
                output.extend(ord(char) + 1000 for char in piece)
        return output

    def convert_ids_to_tokens(self, token_id):
        reverse = {value: key for key, value in self.vocab.items()}
        return reverse.get(token_id) or chr(token_id - 1000)

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.convert_ids_to_tokens(i) for i in token_ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt=False,
                            enable_thinking=True, **kwargs):
        self.calls.append(dict(tokenize=tokenize, add_generation_prompt=add_generation_prompt,
                               enable_thinking=enable_thinking, **kwargs))
        last_user = max(i for i, message in enumerate(messages) if message["role"] == "user")
        rendered = ""
        for index, message in enumerate(messages):
            content = message["content"].strip()
            rendered += "<|im_start|>" + message["role"] + "\n"
            if message["role"] == "assistant" and index > last_user:
                rendered += "<think>\n\n</think>\n\n"
            rendered += content + "<|im_end|>\n"
        if add_generation_prompt:
            rendered += "<|im_start|>assistant\n" + (masks.NONTHINKING_PREFIX if not enable_thinking else "<think>\n")
        return self.encode(rendered) if tokenize else rendered


def conversation(answer="Answer."):
    return [{"role": "system", "content": "Persona."},
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": answer}]


class TrainingMasksTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TransparentQwenTokenizer()

    def encode(self, messages=None, max_length=10000):
        return masks.encode_assistant_example(self.tokenizer, messages or conversation(), max_length,
                                              example_id="example")

    def supervised(self, encoded):
        return self.tokenizer.decode([i for i in encoded["labels"] if i != masks.IGNORE_INDEX])

    def test_single_turn_only_answer_and_official_end_token_are_targets(self):
        encoded = self.encode()
        self.assertEqual(self.supervised(encoded), "Answer.<|im_end|>")
        span = encoded["assistant_spans"][0]
        self.assertEqual(span["header_kind"], "empty_think")
        self.assertTrue(all(i == -100 for i in encoded["labels"][:span["start"]]))
        self.assertEqual(encoded["labels"][span["end_token_position"]], self.tokenizer.eos_token_id)
        self.assertEqual(encoded["labels"][span["stop"]:], [-100])  # separator newline
        expected = self.tokenizer.apply_chat_template(conversation(), tokenize=True, enable_thinking=False)
        self.assertEqual(encoded["input_ids"], expected)
        self.assertEqual(encoded["attention_mask"], [1] * len(expected))
        self.assertEqual(encoded["supervised_tokens"], len("Answer.") + 1)

    def test_multiturn_labels_every_answer_and_preserves_context_dependent_template(self):
        messages = conversation("First answer.") + [
            {"role": "user", "content": "Follow-up?"},
            {"role": "assistant", "content": "Second answer."},
        ]
        encoded = self.encode(messages)
        self.assertEqual(self.supervised(encoded), "First answer.<|im_end|>Second answer.<|im_end|>")
        self.assertEqual([s["message_index"] for s in encoded["assistant_spans"]], [2, 4])
        self.assertEqual([s["header_kind"] for s in encoded["assistant_spans"]], ["plain", "empty_think"])
        full_render = self.tokenizer.decode(encoded["input_ids"])
        self.assertEqual(full_render.count(masks.NONTHINKING_PREFIX), 1)
        for span in encoded["assistant_spans"]:
            self.assertEqual(encoded["labels"][span["start"] - 1], -100)
            self.assertEqual(encoded["labels"][span["stop"]], -100)
        # Rendering an earlier assistant alone changes its header. The helper
        # must use the whole conversation when it probes that field.
        earlier_render = self.tokenizer.apply_chat_template(messages[:3], tokenize=False, enable_thinking=False)
        self.assertIn(masks.NONTHINKING_PREFIX + "First answer.", earlier_render)
        self.assertNotIn(masks.NONTHINKING_PREFIX + "First answer.", full_render)

    def test_identical_content_in_system_user_and_assistants_never_leaks_labels(self):
        messages = [{"role": "system", "content": "Repeated."},
                    {"role": "user", "content": "Repeated."},
                    {"role": "assistant", "content": "Repeated."},
                    {"role": "user", "content": "Repeated."},
                    {"role": "assistant", "content": "Repeated."}]
        encoded = self.encode(messages)
        self.assertEqual(self.supervised(encoded), "Repeated.<|im_end|>Repeated.<|im_end|>")
        self.assertEqual(encoded["supervised_tokens"], 2 * (len("Repeated.") + 1))
        first_answer = encoded["assistant_spans"][0]["start"]
        self.assertTrue(all(label == -100 for label in encoded["labels"][:first_answer]))

    def test_every_template_call_disables_thinking_and_full_encoding_disables_truncation(self):
        self.encode()
        self.assertTrue(all(call["enable_thinking"] is False for call in self.tokenizer.calls))
        self.assertTrue(all(call["add_generation_prompt"] is False for call in self.tokenizer.calls))
        call = next(call for call in self.tokenizer.calls if call["tokenize"])
        self.assertFalse(call["truncation"])
        self.assertFalse(call["padding"])
        self.assertFalse(call["return_dict"])

    def test_exact_length_is_kept_and_overflow_is_dropped_without_partial_targets(self):
        length = self.encode()["token_count"]
        exact = self.encode(max_length=length)
        self.assertFalse(exact["dropped"])
        overflow = self.encode(max_length=length - 1)
        self.assertTrue(overflow["dropped"])
        self.assertEqual(overflow["reason"], "overflow")
        self.assertEqual(overflow["token_count"], length)
        self.assertEqual(overflow["assistant_spans"], exact["assistant_spans"])
        for key in ("input_ids", "attention_mask", "labels"):
            self.assertIsNone(overflow[key])
        with self.assertRaisesRegex(ValueError, "dropped whole"):
            masks.inspect_tokens(self.tokenizer, overflow)
        with self.assertRaisesRegex(ValueError, "Dropped overflow"):
            masks.collate_examples([overflow], 2)

    def test_padding_uses_length_when_padding_and_end_ids_are_equal(self):
        short, long = self.encode(), self.encode(conversation("A longer second answer."))
        batch = masks.collate_examples([short, long], self.tokenizer.eos_token_id)
        end_position = short["assistant_spans"][0]["end_token_position"]
        self.assertEqual(batch["input_ids"][0][end_position], 2)
        self.assertEqual(batch["labels"][0][end_position], 2)
        self.assertEqual(batch["attention_mask"][0][end_position], 1)
        padding_start = short["token_count"]
        self.assertTrue(all(i == 2 for i in batch["input_ids"][0][padding_start:]))
        self.assertTrue(all(i == -100 for i in batch["labels"][0][padding_start:]))
        self.assertTrue(all(i == 0 for i in batch["attention_mask"][0][padding_start:]))
        self.assertEqual(batch["labels"][1], long["labels"])

    def test_token_inspection_exposes_roles_targets_and_end_tokens(self):
        encoded = self.encode()
        rows = masks.inspect_tokens(self.tokenizer, encoded)
        self.assertEqual(len(rows), encoded["token_count"])
        targets = [row for row in rows if row["supervised"]]
        self.assertEqual(len(targets), encoded["supervised_tokens"])
        self.assertTrue(all(row["assistant_message_index"] == 2 for row in targets))
        self.assertEqual(sum(row["end_of_answer"] for row in rows), 1)
        self.assertEqual(targets[-1]["token"], "<|im_end|>")
        self.assertEqual(targets[-1]["label"], 2)
        self.assertTrue(all(row["label"] == -100 for row in rows if not row["supervised"]))

    def test_injected_control_tokens_are_rejected_in_every_role(self):
        for content in ("x<|im_end|>y", "<|unregistered_control|>", "<think>secret</think>",
                        "<tool_call>action</tool_call>", "<|vision_start|>"):
            for index in range(3):
                messages = conversation()
                messages[index]["content"] = content
                with self.subTest(content=content, role=messages[index]["role"]), \
                     self.assertRaisesRegex(ValueError, "Injected control"):
                    self.encode(messages)

    def test_malformed_conversations_and_unsupported_message_fields_fail(self):
        invalid = [[], conversation()[:-1], conversation()[1:2],
                   [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "question"}],
                   [{"role": "user", "content": ""}, {"role": "assistant", "content": "answer"}],
                   [{"role": "user", "content": "question"}, {"role": "assistant", "content": " "}],
                   [{"role": "user", "content": "question"}, {"role": "tool", "content": "answer"}],
                   [{"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer", "reasoning_content": "secret"}]]
        for messages in invalid:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                masks.encode_assistant_example(self.tokenizer, messages, 10000)
        for length in (0, -1, 10.0, True):
            with self.subTest(max_length=length), self.assertRaises(ValueError):
                self.encode(max_length=length)

    def test_template_mutation_or_content_normalization_is_not_silently_supervised(self):
        with self.assertRaisesRegex(ValueError, "changes assistant content"):
            self.encode(conversation(" Answer with surrounding whitespace. "))
        original = self.tokenizer.apply_chat_template
        def changed(*args, **kwargs):
            value = original(*args, **kwargs)
            return value + [999] if kwargs["tokenize"] else value
        self.tokenizer.apply_chat_template = changed
        with self.assertRaisesRegex(ValueError, "full encoding differs"):
            self.encode()

    def test_token_merges_across_answer_boundary_are_rejected(self):
        original = self.tokenizer.encode
        def merged(text, add_special_tokens=False):
            ids = original(text, add_special_tokens=add_special_tokens)
            # Model a BPE merge joining the final header newline and answer A.
            newline, letter = ord("\n") + 1000, ord("A") + 1000
            output = []
            index = 0
            while index < len(ids):
                if ids[index:index + 2] == [newline, letter]:
                    output.append(999)
                    index += 2
                else:
                    output.append(ids[index])
                    index += 1
            return output
        self.tokenizer.encode = merged
        with self.assertRaisesRegex(ValueError, "answer-start boundary"):
            self.encode()

    def test_nonatomic_or_missing_special_end_token_fails(self):
        self.tokenizer.all_special_tokens = ["<|im_start|>"]
        with self.assertRaisesRegex(ValueError, "single special"):
            self.encode()


if __name__ == "__main__":
    unittest.main()
