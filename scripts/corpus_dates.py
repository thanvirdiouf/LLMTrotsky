"""Conservative English source-date intervals; publication dates stay separate."""
import calendar
from datetime import date
import re

MONTHS = {name.casefold(): i for i, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.casefold(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTHS["sept"] = 9
MONTH_PATTERN = "(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\.?"
YEAR_PATTERN = r"(?:18|19|20)\d{2}"
RANGE_SEPARATOR = r"(?:[-–—/&]|\band\b|\bthrough\b|\bto\b)"
DAY_LIST = rf"\d{{1,2}}(?:\s*(?:{RANGE_SEPARATOR}|,)\s*\d{{1,2}})+"
COMPOSITION_FIELD = re.compile(r"^(?:(?:originally\s+)?(?:written|writtten|delivered|dictated|discussed|interviewed|recorded stenographically)\b|last\s+\w+\s+paragraphs?\s+added\s+as\s+(?:a\s+)?postscript\b)", re.I)
PUBLICATION_CLAUSE = re.compile(r"(?:;|\band\b)?\s*\b(?:(?:first|originally)\s+)?(?:published|printed|appeared|translated|transcribed|reprinted|proofread)\b", re.I)


def _month(text):
    return MONTHS[text.casefold().rstrip(".")]


def parse_date_range(text):
    raw = text
    text = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", text, flags=re.I)
    # An approximate date is evidence of an estimate, not a verified interval.
    if re.search(rf"\b(?:circa|around|about|probably|possibly|presumably)\s+(?:{MONTH_PATTERN}|\d)", text, re.I):
        return None
    spans = []
    covered_years = set()
    exact_regions = []

    def add_region(match, first, last, precision):
        exact_regions.append(match.span())
        spans.append((first, last, precision))
        covered_years.update(range(first.year, last.year + 1))

    # A shared year belongs to every explicit endpoint: do not silently narrow
    # "10 & 12 March 1938" to March 12 or "August-September" to September.
    for pattern, order in [
        (rf"\b({DAY_LIST})\s+({MONTH_PATTERN})[,.]?\s*({YEAR_PATTERN})\b", "day_first"),
        (rf"\b({MONTH_PATTERN})\s+({DAY_LIST})[,.]?\s*({YEAR_PATTERN})\b", "month_first"),
    ]:
        for match in re.finditer(pattern, text, re.I):
            exact_regions.append(match.span())
            try:
                days = [int(day) for day in re.findall(r"\d+", match[1 if order == "day_first" else 2])]
                month = _month(match[2 if order == "day_first" else 1])
                year = int(match[3])
                dates = [date(year, month, day) for day in days]
            except ValueError:
                continue
            spans.append((min(dates), max(dates), "range"))
            covered_years.add(year)
    for pattern, order in [
        (rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})\s*{RANGE_SEPARATOR}\s*({MONTH_PATTERN})\s+(\d{{1,2}})[,.]?\s*({YEAR_PATTERN})\b", "month_first"),
        (rf"\b(\d{{1,2}})\s+({MONTH_PATTERN})\s*{RANGE_SEPARATOR}\s*(\d{{1,2}})\s+({MONTH_PATTERN})[,.]?\s*({YEAR_PATTERN})\b", "day_first"),
    ]:
        for match in re.finditer(pattern, text, re.I):
            exact_regions.append(match.span())
            try:
                year = int(match[5])
                if order == "month_first":
                    first, last = date(year, _month(match[1]), int(match[2])), date(year, _month(match[3]), int(match[4]))
                else:
                    first, last = date(year, _month(match[2]), int(match[1])), date(year, _month(match[4]), int(match[3]))
            except ValueError:
                continue
            if first > last:
                return None  # A missing earlier year cannot be supplied by inference.
            spans.append((first, last, "range"))
            covered_years.add(year)
    for match in re.finditer(rf"\b({MONTH_PATTERN})\s*{RANGE_SEPARATOR}\s*({MONTH_PATTERN})[,.]?\s*({YEAR_PATTERN})\b", text, re.I):
        first_month, last_month, year = _month(match[1]), _month(match[2]), int(match[3])
        if first_month > last_month:
            return None
        add_region(match, date(year, first_month, 1), date(year, last_month, calendar.monthrange(year, last_month)[1]), "range")
    for match in re.finditer(rf"\b({YEAR_PATTERN})\s*[-–—/]\s*(\d{{2}})\b", text):
        first_year = int(match[1])
        last_year = first_year // 100 * 100 + int(match[2])
        if first_year > last_year:
            return None
        add_region(match, date(first_year, 1, 1), date(last_year, 12, 31), "range")
    for pattern, order in [
        (rf"\b(\d{{1,2}})\s+({MONTH_PATTERN})[,.]?\s*({YEAR_PATTERN})\b", "day_first"),
        (rf"\b({MONTH_PATTERN})\s+(\d{{1,2}})[,.]?\s*({YEAR_PATTERN})\b", "month_first"),
    ]:
        for match in re.finditer(pattern, text, re.I):
            if any(start <= match.start() and match.end() <= end for start, end in exact_regions):
                continue
            exact_regions.append(match.span())
            try:
                month = _month(match[2 if order == "day_first" else 1])
                day = int(match[1 if order == "day_first" else 2])
                year = int(match[3])
                d = date(year, month, day)
            except ValueError:
                continue
            spans.append((d, d, "day"))
            covered_years.add(year)
    for match in re.finditer(rf"\b({MONTH_PATTERN})[,.]?\s+({YEAR_PATTERN})\b", text, re.I):
        if any(start <= match.start() and match.end() <= end for start, end in exact_regions):
            continue
        month, year = _month(match[1]), int(match[2])
        spans.append((date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]), "month"))
        covered_years.add(year)
    for year in {int(y) for y in re.findall(rf"\b{YEAR_PATTERN}\b", text)} - covered_years:
        spans.append((date(year, 1, 1), date(year, 12, 31), "year"))
    if not spans:
        return None
    spans = sorted(set(spans))
    return {"earliest": min(s[0] for s in spans).isoformat(), "latest": max(s[1] for s in spans).isoformat(), "precision": spans[0][2] if len(spans) == 1 else "range", "raw": raw}


def composition_evidence(metadata):
    evidence = []
    for key, values in metadata.items():
        if not COMPOSITION_FIELD.match(key):
            continue
        for value in values:
            # Some MIA information paragraphs have no colon: the complete
            # "Written on ..." statement is then retained as the metadata key.
            text = f"{key}: {value}" if value else key
            text = PUBLICATION_CLAUSE.split(text, maxsplit=1)[0].strip(" ;,.")
            if parse_date_range(text):
                evidence.append({"kind": "explicit_composition_metadata", "field": key, "text": text})
    return evidence


def composition_interval(metadata):
    return interval_from_evidence(composition_evidence(metadata))


def interval_from_evidence(evidence):
    intervals = [interval for item in evidence if (interval := parse_date_range(item["text"]))]
    if not intervals:
        return None
    first, last = min(item["earliest"] for item in intervals), max(item["latest"] for item in intervals)
    bounds = {(item["earliest"], item["latest"]) for item in intervals}
    return {"earliest": first, "latest": last, "precision": intervals[0]["precision"] if len(bounds) == 1 else "range", "raw": "; ".join(item["text"] for item in evidence)}


def cutoff_status(interval, cutoff):
    if interval is None:
        return "composition_date_unverified"
    if interval["latest"] <= cutoff:
        return "on_or_before_cutoff"
    if interval["earliest"] > cutoff:
        return "after_cutoff"
    return "date_interval_overlaps_cutoff"
