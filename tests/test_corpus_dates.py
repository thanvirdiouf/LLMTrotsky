import unittest
from scripts.corpus_dates import composition_interval, cutoff_status, parse_date_range, interval_from_evidence


class SourceDateTests(unittest.TestCase):
    def test_two_exact_date_formats_and_ordinals(self):
        self.assertEqual(parse_date_range("18th August 1940.")["latest"], "1940-08-18")
        self.assertEqual(parse_date_range("August 18, 1940")["earliest"], "1940-08-18")

    def test_month_interval_cannot_pass_a_midmonth_cutoff(self):
        interval = parse_date_range("August 1940")
        self.assertEqual(interval["latest"], "1940-08-31")
        self.assertEqual(cutoff_status(interval, "1940-08-19"), "date_interval_overlaps_cutoff")

    def test_range_across_years(self):
        interval = parse_date_range("December 1936-January 1937")
        self.assertEqual((interval["earliest"], interval["latest"]), ("1936-12-01", "1937-01-31"))

    def test_shared_year_ranges_keep_every_explicit_endpoint(self):
        for text, first, last in [
            ("10 & 12 March 1938", "1938-03-10", "1938-03-12"),
            ("3/4 August 1938", "1938-08-03", "1938-08-04"),
            ("May 22 & 24, 1932", "1932-05-22", "1932-05-24"),
            ("13, 22 & 31 October 1932", "1932-10-13", "1932-10-31"),
            ("April 23 through July 20, 1938", "1938-04-23", "1938-07-20"),
            ("22 November–5 December 1921", "1921-11-22", "1921-12-05"),
            ("August–September 1932", "1932-08-01", "1932-09-30"),
            ("Nov-Dec 1932", "1932-11-01", "1932-12-31"),
            ("1921-23", "1921-01-01", "1923-12-31"),
        ]:
            with self.subTest(text=text):
                interval = parse_date_range(text)
                self.assertEqual((interval["earliest"], interval["latest"]), (first, last))
                self.assertEqual(interval["precision"], "range")

    def test_punctuation_and_abbreviations_in_explicit_dates(self):
        for text, expected in [("July 6,1938", "1938-07-06"), ("Jan. 5, 1940", "1940-01-05"), ("21st April, 1924", "1924-04-21")]:
            with self.subTest(text=text):
                self.assertEqual(parse_date_range(text)["earliest"], expected)

    def test_approximate_and_ambiguous_dates_do_not_create_verified_bounds(self):
        self.assertIsNone(parse_date_range("circa February 1918"))
        self.assertIsNone(parse_date_range("December-January 1937"))
        self.assertIsNone(parse_date_range("22 & 26 September"))

    def test_composition_date_in_metadata_key_is_retained(self):
        self.assertEqual(composition_interval({"written february 22, 1937": [""]})["latest"], "1937-02-22")
        self.assertEqual(composition_interval({"written in exile in turkey, june 10, 1933": [""]})["latest"], "1933-06-10")

    def test_explicit_composition_field_aliases(self):
        for key in ["writtten", "originally written", "discussed", "interviewed", "recorded stenographically"]:
            with self.subTest(key=key):
                self.assertEqual(composition_interval({key: ["18 March 1939"]})["latest"], "1939-03-18")
        self.assertIsNone(composition_interval({"interview source": ["1942"], "discussed by the editors": ["undated"]}))

    def test_publication_clause_within_written_field_is_separate(self):
        interval = composition_interval({"written": ["6 March 1937 and published in July 1941"]})
        self.assertEqual(interval["latest"], "1937-03-06")

    def test_explicit_added_postscript_extends_composition_interval(self):
        interval = composition_interval({"written in exile in turkey, june 10, 1933": [""], "last two paragraphs added as postscript november 2, 1933": [""]})
        self.assertEqual((interval["earliest"], interval["latest"]), ("1933-06-10", "1933-11-02"))

    def test_partial_signature_does_not_narrow_a_whole_work_year(self):
        interval = interval_from_evidence([{"text": "Written: 1935"}, {"text": "Leon Trotsky\nSeptember 14, 1935"}])
        self.assertEqual((interval["earliest"], interval["latest"]), ("1935-01-01", "1935-12-31"))

    def test_publication_date_is_not_composition(self):
        interval = composition_interval({"written": ["1937"], "first published": ["1941"], "copyleft": ["2002"]})
        self.assertEqual(interval["latest"], "1937-12-31")

    def test_dictaphone_date_is_composition_evidence(self):
        interval = composition_interval({"delivered via dictaphone": ["20 August 1940."]})
        self.assertEqual(cutoff_status(interval, "1940-08-19"), "after_cutoff")

    def test_unknown_and_invalid_dates_are_not_fabricated(self):
        self.assertIsNone(parse_date_range("undated"))
        self.assertEqual(cutoff_status(None, "1940-08-19"), "composition_date_unverified")
        self.assertEqual(parse_date_range("31 February 1940")["precision"], "year")


if __name__ == "__main__":
    unittest.main()
