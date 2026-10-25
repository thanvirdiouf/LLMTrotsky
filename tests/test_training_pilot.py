import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_training_pilot import (
    BASELINE_PATH, MODEL_ID, MODEL_REVISION, build_lock, canonical_sha256,
    sha256_bytes, validate_contamination, validate_lock, validate_training_payload,
    validate_training_rows,
)


REFERENCE = "This reserved evaluation passage contains unique historical evidence that must never become any part of the supervised pilot dataset."
GENERATED = "This generated benchmark answer has distinct invented details and is forbidden even when it appears inside a longer training response."


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def fixture(project, copied_answer=None):
    persona = {"persona_id": "test_persona", "historical_as_of": "1940-08-19", "system_prompt": "Persona for this test."}
    write_json(project / "config/persona.json", persona)
    write_json(project / "reviews/seed_specifications.json", {"reviewed": True})
    write_json(project / "reviews/evaluation_specifications.json", {"source_backed": [], "behavioral": []})
    rows, records = [], []
    for number in range(2):
        messages = [{"role": "system", "content": persona["system_prompt"]},
                    {"role": "user", "content": f"Question {number}?"},
                    {"role": "assistant", "content": copied_answer if number == 0 and copied_answer else f"Reviewed answer {number}."}]
        if number == 1:
            messages.extend([{"role": "user", "content": "A follow up question?"},
                             {"role": "assistant", "content": "A reviewed follow up answer."}])
        row = {"example_id": f"example_{number}", "messages": messages}
        rows.append(row)
        evidence = "\n".join(message["content"] for message in messages if message["role"] == "assistant")
        records.append(dict(row, approved_for_seed_training=True, review_status="agent_reviewed_seed",
                            persona_id=persona["persona_id"], historical_as_of=persona["historical_as_of"],
                            source_document_id="train_doc", source_family_id="train_family",
                            source_url="https://source.test/train", source_sha256="train_source_sha",
                            source_evidence=evidence, evidence_sha256=sha256_bytes(evidence.encode()),
                            source_block_ids=[number], source_date_evidence={"date": "1937-04-10"},
                            answer_origin="verbatim_source"))
    write_rows(project / "datasets/train_seed.jsonl", rows)
    write_rows(project / "datasets/seed_records.jsonl", records)
    examples = {}
    for record in records:
        examples[record["example_id"]] = {key: record[key] for key in
                                          ("source_sha256", "evidence_sha256", "source_block_ids", "source_date_evidence")}
        examples[record["example_id"]]["messages_sha256"] = sha256_bytes(json.dumps(
            record["messages"], ensure_ascii=False, sort_keys=True).encode())
    write_json(project / "reviews/seed_evidence_lock.json", {
        "specifications_sha256": sha256_bytes((project / "reviews/seed_specifications.json").read_bytes()),
        "persona_sha256": sha256_bytes((project / "config/persona.json").read_bytes()), "examples": examples})
    catalog = [{"document_id": "train_doc", "source_family_id": "train_family", "source_sha256": "train_source_sha", "url": "https://source.test/train"},
               {"document_id": "eval_doc", "source_family_id": "eval_family", "source_sha256": "eval_source_sha", "url": "https://source.test/eval"}]
    write_rows(project / "corpus_mia/catalog.jsonl", catalog)
    write_rows(project / "evaluation/source_split_manifest.jsonl", [
        dict(row, split="train_seed_source" if row["document_id"] == "train_doc" else "validation") for row in catalog])
    write_rows(project / "evaluation/cases.jsonl", [{"case_id": "heldout", "split": "validation",
        "source_document_id": "eval_doc", "source_family_id": "eval_family", "source_sha256": "eval_source_sha",
        "messages": [{"role": "system", "content": persona["system_prompt"]}, {"role": "user", "content": "What is the reserved evaluation question?"}],
        "reference_evidence": REFERENCE, "evidence_sha256": sha256_bytes(REFERENCE.encode())}])
    write_json(project / BASELINE_PATH, {"run_id": "baseline_id", "identity": {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "quantization": "bnb_nf4_double",
        "enable_thinking": False, "generation": {"do_sample": False, "num_beams": 1, "max_new_tokens": 512},
        "dtype": "torch.float16", "seed": 42, "max_input_tokens": 8192,
        "experiment": {"original_case_ids": ["heldout"], "answer_instruction": "Be concise."},
        "chat_template_sha256": "template_hash", "packages": {"transformers": "test"}}})
    write_rows(project / "evaluation/runs/example/responses.jsonl", [{"case_id": "heldout", "response": GENERATED}])
    with (project / "evaluation/runs/example/paired_answers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["case_id", "prompt_only_response"])
        writer.writeheader()
        writer.writerow({"case_id": "heldout", "prompt_only_response": GENERATED})
    return rows


class TrainingPilotTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.rows = fixture(self.project)
        self.lock = build_lock(self.project, expected_records=2)

    def test_deterministic_lock_and_portable_exact_payload_validation(self):
        self.assertEqual(self.lock, build_lock(self.project, expected_records=2))
        self.assertEqual(self.lock, validate_lock(self.project, self.lock))
        payload = (self.project / "datasets/train_seed.jsonl").read_bytes()
        self.assertEqual(self.rows, validate_training_payload(payload, self.lock))
        self.assertEqual(self.lock["dataset"]["multi_turn_records"], 1)
        self.assertEqual(self.lock["evaluation"]["locally_archived_unique_generated_answers"], 1)
        encoded = json.dumps(self.lock)
        self.assertNotIn(REFERENCE, encoded)
        self.assertNotIn(GENERATED, encoded)
        self.assertEqual(self.lock["provenance"]["source_family_overlap"], 0)

    def test_even_byte_only_dataset_drift_is_rejected(self):
        payload = (self.project / "datasets/train_seed.jsonl").read_bytes()
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_training_payload(payload + b"\n", self.lock)

    def test_copy_guards_reject_reference_or_generated_answer_inside_response(self):
        for copied in [REFERENCE, GENERATED, GENERATED.upper().replace(" ", "  ")]:
            with self.subTest(copied=copied):
                modified = copy.deepcopy(self.rows)
                modified[0]["messages"][-1]["content"] = "A preamble. " + copied + " A closing sentence."
                with self.assertRaisesRegex(ValueError, "Excluded evaluation/generated passage"):
                    validate_contamination(modified, self.lock["contamination_guard"])

    def test_rebuilding_cannot_admit_copied_evaluation_passages(self):
        fixture(self.project, copied_answer=REFERENCE)
        with self.assertRaisesRegex(ValueError, "Excluded evaluation/generated"):
            build_lock(self.project, expected_records=2)

    def test_unreviewed_examples_cannot_enter_the_pilot(self):
        path = self.project / "datasets/seed_records.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["approved_for_seed_training"] = False
        write_rows(path, records)
        with self.assertRaisesRegex(ValueError, "reviewed approval"):
            build_lock(self.project, expected_records=2)

    def test_source_family_conflict_is_rejected(self):
        path = self.project / "evaluation/source_split_manifest.jsonl"
        assignments = [json.loads(line) for line in path.read_text().splitlines()]
        assignments[0]["split"] = "validation"
        write_rows(path, assignments)
        with self.assertRaisesRegex(ValueError, "reserved training family"):
            build_lock(self.project, expected_records=2)

    def test_stale_review_evidence_is_rejected(self):
        path = self.project / "reviews/seed_evidence_lock.json"
        reviewed = json.loads(path.read_text())
        reviewed["examples"]["example_0"]["source_block_ids"] = [999]
        write_json(path, reviewed)
        with self.assertRaisesRegex(ValueError, "evidence lock differs"):
            build_lock(self.project, expected_records=2)

    def test_new_benchmark_artifact_invalidates_existing_lock(self):
        write_rows(self.project / "evaluation/runs/new/responses.jsonl", [{"response": "A newly generated answer."}])
        with self.assertRaisesRegex(ValueError, "lock is stale"):
            validate_lock(self.project, self.lock)

    def test_complete_turns_are_required_even_with_updated_row_hash(self):
        modified = copy.deepcopy(self.rows)
        modified[1]["messages"].pop()
        lock = copy.deepcopy(self.lock)
        lock["dataset"]["example_sha256"]["example_1"] = canonical_sha256(modified[1])
        with self.assertRaisesRegex(ValueError, "Incomplete training conversation"):
            validate_training_rows(modified, lock)

    def test_portable_helper_has_no_external_import_dependencies(self):
        # Execution with a non-main module name mirrors standalone notebook bundling.
        source = (Path(__file__).resolve().parents[1] / "scripts/build_training_pilot.py").read_text()
        namespace = {"__name__": "bundled_pilot_integrity"}
        exec(compile(source, "pilot_integrity.py", "exec"), namespace)
        payload = (self.project / "datasets/train_seed.jsonl").read_bytes()
        self.assertEqual(self.rows, namespace["validate_training_payload"](payload, self.lock))


if __name__ == "__main__":
    unittest.main()
