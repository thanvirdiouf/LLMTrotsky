from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.build_evaluation import compile_evaluation


def fixture(project, same_family=False):
    for name in ["corpus_mia", "config", "reviews", "datasets"]:
        (project / name).mkdir()
    docs = [{"document_id": key, "url": "https://www.marxists.org/archive/trotsky/1930/mylife/" + key + ".htm", "source_sha256": "test", "blocks": [{"block_id": 1, "role": "body_candidate", "text": "Evidence."}]} for key in ["eval1", "eval2", "train"]]
    catalog = [{"document_id": d["document_id"], "url": d["url"], "source_family_id": "book" if d["document_id"].startswith("eval") or same_family else "training"} for d in docs]
    (project / "corpus_mia/crawl_state.json").write_text(json.dumps({"queue": []}))
    for name, data in [("documents.jsonl", docs), ("catalog.jsonl", catalog)]:
        (project / "corpus_mia" / name).write_text("".join(json.dumps(r) + "\n" for r in data))
    (project / "config/persona.json").write_text(json.dumps({"system_prompt": "Persona", "historical_as_of": "1940-08-19"}))
    (project / "datasets/seed_records.jsonl").write_text(json.dumps({"source_document_id": "train", "source_family_id": "book" if same_family else "training"}) + "\n")
    (project / "reviews/evaluation_specifications.json").write_text(json.dumps({"source_backed": [{"id": "case", "split": "test", "source_suffix": "eval1.htm", "block_ids": [1], "question": "Question?", "checks": ["Check"]}], "behavioral": []}))


class EvaluationTests(unittest.TestCase):
    def test_entire_family_is_reserved_and_csv_is_well_formed(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            fixture(project)
            with redirect_stdout(io.StringIO()):
                compile_evaluation(project)
            assignments = [json.loads(l) for l in (project / "evaluation/source_split_manifest.jsonl").read_text().splitlines()]
            self.assertEqual([a["split"] for a in assignments], ["test", "test", "train_seed_source"])
            with (project / "evaluation/score_sheet.csv").open() as stream:
                sheet = list(csv.reader(stream))
            self.assertEqual(len(sheet[0]), len(sheet[1]))

    def test_training_family_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            fixture(project, same_family=True)
            with self.assertRaisesRegex(ValueError, "overlaps training"):
                compile_evaluation(project)

    def test_incomplete_crawl_cannot_freeze_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            fixture(project)
            (project / "corpus_mia/crawl_state.json").write_text(json.dumps({"queue": ["pending"]}))
            with self.assertRaisesRegex(ValueError, "Finish the current crawl"):
                compile_evaluation(project)


if __name__ == "__main__":
    unittest.main()
