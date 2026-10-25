"""Blinded pilot export validates provenance and leaves human scoring blank."""
import copy
import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

from scripts import export_pilot_review as exporter


PROJECT = Path(__file__).resolve().parents[1]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class PilotReviewExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        self.inputs = [json.loads(line) for line in
                       (PROJECT / "training/pilot_evaluation_inputs.jsonl").read_text().splitlines()]
        self.comparison = {"identity_sha256": "a" * 64, "conditions": ["baseline", "adapter"],
                           "generation": {**exporter.GENERATION, "top_k": 0, "top_p": 1.0},
                           "enable_thinking": False, "evaluation_inputs_sha256": exporter.INPUTS_SHA256,
                           "chat_template_sha256": exporter.TEMPLATE_SHA256}
        self.comparison["generation_sha256"] = exporter.fingerprint(self.comparison["generation"])
        self.conditions = {}
        for number, condition in enumerate(["baseline", "adapter"]):
            self.conditions[condition] = [{"case_id": case["case_id"], "category": case["category"],
                "split": case["split"], "condition": condition, "identity_sha256": "a" * 64,
                "generation_sha256": self.comparison["generation_sha256"],
                "prompt_sha256": exporter.fingerprint(case["messages"]),
                "input_sha256": exporter.fingerprint(case["messages"]),
                "response": f"Recorded answer {number} for {case['case_id']}.\nLiteral ``` and # text.",
                "token_ids": [100 + number, 248046], "token_count": 2, "finish_reason": "stop"}
                for case in self.inputs]
        self.write_run()

    def write_run(self):
        (self.run / "comparison_manifest.json").write_text(json.dumps(self.comparison))
        for condition, rows in self.conditions.items():
            write_jsonl(self.run / f"{condition}_responses.jsonl", rows)

    def read_csv(self, path):
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def assert_rejected(self, message=None):
        with self.assertRaisesRegex(ValueError, message or ".+"):
            exporter.export_review(self.run)
        self.assertFalse((self.run / "review").exists())

    def test_complete_export_is_blinded_paired_and_unscored_with_review_evidence(self):
        state = random.getstate()
        output = exporter.export_review(self.run)
        self.assertEqual(state, random.getstate())
        reviews = self.read_csv(output / "pilot_review.csv")
        keys = self.read_csv(output / "pilot_review_key.csv")
        self.assertEqual(len(reviews), 72)
        self.assertEqual(len({row["pair_id"] for row in reviews}), 36)
        self.assertEqual([row["answer_label"] for row in reviews], ["A", "B"] * 36)
        self.assertTrue(all("condition" not in row and "identity_sha256" not in row for row in reviews))
        self.assertTrue(all(row[column] == "" for row in reviews for column in
                            [*exporter.SCORE_COLUMNS, "critical_failure", *exporter.FLAG_COLUMNS, "notes"]))
        by_key = {row["review_id"]: row for row in keys}
        for review in reviews:
            key = by_key[review["review_id"]]
            original = next(row for row in self.conditions[key["condition"]] if row["case_id"] == key["case_id"])
            self.assertEqual(review["response"], original["response"])
            self.assertEqual(key["response_sha256"], exporter.digest(review["response"].encode()))
        for index in range(0, 72, 2):
            left, right = reviews[index:index + 2]
            self.assertEqual(left["messages"], right["messages"])
            self.assertEqual(left["source_evidence"], right["source_evidence"])
            self.assertEqual(left["reference_checks"], right["reference_checks"])
            self.assertEqual({by_key[left["review_id"]]["condition"], by_key[right["review_id"]]["condition"]},
                             {"baseline", "adapter"})
        self.assertEqual(sum(bool(row["source_evidence"]) for row in reviews), 24)
        markdown = (output / "pilot_review.md").read_text()
        self.assertIn("Reference evidence and checklists below were withheld", markdown)
        self.assertIn("previously inspected cases are development evidence", markdown)
        self.assertIn("    Literal ``` and # text.", markdown)
        self.assertNotIn("baseline", markdown)
        self.assertNotIn("adapter", markdown)
        manifest = json.loads((output / "review_manifest.json").read_text())
        self.assertFalse(manifest["quality_scores_assigned"])
        for name, expected in manifest["output_sha256"].items():
            self.assertEqual(exporter.digest((output / name).read_bytes()), expected)

    def test_assignment_is_deterministic_per_case_and_independent_of_journal_order(self):
        first = exporter.export_review(self.run, output_dir=self.run / "first", seed=7)
        for rows in self.conditions.values():
            rows.reverse()
        self.write_run()
        second = exporter.export_review(self.run, output_dir=self.run / "second", seed=7)
        for name in ["pilot_review.csv", "pilot_review.md", "pilot_review_key.csv"]:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        third = exporter.export_review(self.run, output_dir=self.run / "third", seed=8)
        assignments = lambda out: {row["case_id"]: row["condition"] for row in
                                   self.read_csv(out / "pilot_review_key.csv") if row["answer_label"] == "A"}
        self.assertNotEqual(assignments(first), assignments(third))
        self.assertEqual(set(assignments(first).values()), {"baseline", "adapter"})

    def test_missing_duplicate_unknown_and_cross_condition_rows_are_rejected(self):
        original = copy.deepcopy(self.conditions)
        for change in [lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
                       lambda rows: rows[0].update(case_id="unknown"),
                       lambda rows: rows[0].update(condition="baseline")]:
            with self.subTest(change=change):
                self.conditions = copy.deepcopy(original)
                change(self.conditions["adapter"])
                self.write_run()
                self.assert_rejected()

    def test_mixed_run_prompt_generation_and_tokenized_input_hashes_are_rejected(self):
        original = copy.deepcopy(self.conditions)
        for update in [{"identity_sha256": "b" * 64}, {"prompt_sha256": "c" * 64},
                       {"generation_sha256": "d" * 64}, {"input_sha256": "e" * 64},
                       {"category": "changed"}, {"split": "changed"}]:
            with self.subTest(update=update):
                self.conditions = copy.deepcopy(original)
                self.conditions["adapter"][0].update(update)
                self.write_run()
                self.assert_rejected()

    def test_manifest_drift_and_nonfinite_json_are_rejected(self):
        original = copy.deepcopy(self.comparison)
        for update in [{"generation_sha256": "0" * 64}, {"enable_thinking": True},
                       {"evaluation_inputs_sha256": "1" * 64}, {"chat_template_sha256": "2" * 64},
                       {"conditions": ["adapter", "baseline"]}, {"identity_sha256": "bad"}]:
            with self.subTest(update=update):
                self.comparison = {**original, **update}
                self.write_run()
                self.assert_rejected("Comparison")
        self.comparison = copy.deepcopy(original)
        self.comparison["generation"]["eos_token_id"] = 248044
        self.comparison["generation_sha256"] = exporter.fingerprint(self.comparison["generation"])
        self.write_run()
        self.assert_rejected("Comparison")
        (self.run / "comparison_manifest.json").write_text('{"generation": NaN}')
        self.assert_rejected("Non-finite")

    def test_frozen_inputs_and_reviewer_evidence_cannot_be_substituted(self):
        for filename, parameter in [("training/pilot_evaluation_inputs.jsonl", "inputs_path"),
                                    ("evaluation/cases.jsonl", "cases_path")]:
            altered = self.run / Path(filename).name
            altered.write_bytes((PROJECT / filename).read_bytes() + b"\n")
            with self.subTest(parameter=parameter), self.assertRaisesRegex(ValueError, "checksum"):
                exporter.export_review(self.run, **{parameter: altered})
        self.assertFalse((self.run / "review").exists())

    def test_stop_token_counts_and_truncation_evidence_are_checked(self):
        original = copy.deepcopy(self.conditions)
        for update in [{"token_count": 99}, {"token_count": True}, {"token_ids": [True, 248046]},
                       {"token_ids": [-1, 248046]}, {"finish_reason": "length"},
                       {"token_ids": [101, 102], "finish_reason": "length"},
                       {"token_ids": [248046, 248046]}, {"token_ids": [], "token_count": 0}]:
            with self.subTest(update=update):
                self.conditions = copy.deepcopy(original)
                self.conditions["baseline"][0].update(update)
                self.write_run()
                self.assert_rejected("token count or stop")
        self.conditions = copy.deepcopy(original)
        self.conditions["baseline"][0].update(token_ids=[101] * 512, token_count=512, finish_reason="length")
        self.write_run()
        output = exporter.export_review(self.run)
        limited = [row for row in self.read_csv(output / "pilot_review.csv") if row["finish_reason"] == "length"]
        self.assertEqual(len(limited), 1)
        self.assertEqual(limited[0]["token_count"], "512")

    def test_existing_scored_export_is_never_overwritten(self):
        output = exporter.export_review(self.run)
        scored = output / "pilot_review.csv"
        scored.write_text("Human scored copy\n")
        with self.assertRaisesRegex(FileExistsError, "preserve scored"):
            exporter.export_review(self.run)
        self.assertEqual(scored.read_text(), "Human scored copy\n")
        with self.assertRaisesRegex(ValueError, "seed"):
            exporter.export_review(self.run, output_dir=self.run / "invalid", seed=True)


if __name__ == "__main__":
    unittest.main()
