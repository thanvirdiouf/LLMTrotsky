import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from scripts.build_catalog import book_hint, build, words


class CatalogTests(unittest.TestCase):
    def test_normalization_and_book_hints(self):
        self.assertEqual(words("Workers’ DEMOCRACY!"), ["workers", "democracy"])
        self.assertEqual(book_hint("https://www.marxists.org/archive/trotsky/1930/mylife/ch12.htm"), "my_life")
        self.assertIsNone(book_hint("https://www.marxists.org/archive/trotsky/1939/07/letter.htm"))

    def test_book_and_duplicate_families_remain_unsplit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            bodies = [" ".join(f"word{i}" for i in range(250)), " ".join(f"other{i}" for i in range(250)), " ".join(f"word{i}" for i in range(250))]
            paths = ["1930/mylife/ch01.htm", "1930/mylife/ch02.htm", "1939/01/excerpt.htm"]
            for i, (body, path) in enumerate(zip(bodies, paths)):
                rows.append({"document_id": str(i), "url": "https://www.marxists.org/archive/trotsky/" + path, "title": "Test", "page_kind": "text_candidate", "source_sha256": "test", "composition_year_candidates": [1930], "url_year_hint": 1930, "quality_flags": [], "body_candidate_words": 250, "blocks": [{"role": "body_candidate", "text": body}]})
            (root / "documents.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            with redirect_stdout(io.StringIO()):
                build(root)
            families = [json.loads(l) for l in (root / "source_families.jsonl").read_text().splitlines()]
            self.assertEqual(len(families), 1)
            self.assertEqual(set(families[0]["document_ids"]), {"0", "1", "2"})
            self.assertIsNone(families[0]["split"])
            audit = json.loads((root / "catalog_audit.json").read_text())
            self.assertFalse(audit["splits_assigned"])


if __name__ == "__main__":
    unittest.main()
