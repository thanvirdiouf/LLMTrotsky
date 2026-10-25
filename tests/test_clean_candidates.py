import unittest
from scripts.clean_candidates import repair_text


class ConservativeRepairTests(unittest.TestCase):
    def test_known_entities_are_repaired_but_unknown_numbers_remain(self):
        text, changes = repair_text("#8220;word#8221; #9999;", "test", [])
        self.assertEqual(text, "“word” #9999;")
        self.assertEqual(sum(c["count"] for c in changes), 2)

    def test_lexical_repairs_are_source_bound(self):
        rules = [{"id": "one", "source_url": "allowed", "before": "cqmplained", "after": "complained", "reason": "Reviewed"}]
        self.assertEqual(repair_text("cqmplained", "other", rules), ("cqmplained", []))
        self.assertEqual(repair_text("cqmplained", "allowed", rules)[0], "complained")


if __name__ == "__main__":
    unittest.main()
