#!/usr/bin/env python3
"""Resumable, single-connection crawl of the public English Trotsky archive."""
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import fcntl
import json
from pathlib import Path
import re
import time
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

ROOT = "https://www.marxists.org/archive/trotsky/"
SEED = ROOT + "works/index.htm"
AGENT = "TrotskyCorpusResearch/0.1 (personal archival research; single connection)"


def canonical(url):
    parts = urlsplit(urldefrag(url)[0])
    if parts.hostname not in {"marxists.org", "www.marxists.org"}:
        return None
    if parts.scheme not in {"http", "https"}:
        return None
    path = parts.path
    if path.endswith("/"):
        path += "index.htm"
    return urlunsplit(("https", "www.marxists.org", path, "", ""))


def allowed(url, supplemental=()):
    if url in supplemental:
        return True
    if not url or not url.startswith(ROOT):
        return False
    relative = url[len(ROOT):]
    if relative.startswith(("bio/", "photo/")) or relative == "quotes.htm":
        return False
    return bool(re.search(r"\.html?$", relative, re.I))


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def crawl(output, limit, supplemental_file=None):
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw_html").mkdir(exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = AGENT
    # The saved raw robots file makes the crawl policy auditable.
    response = session.get("https://www.marxists.org/robots.txt", timeout=(15, 45))
    response.raise_for_status()
    (output / "robots.txt").write_bytes(response.content)
    robots = RobotFileParser()
    robots.parse(response.text.splitlines())
    delay = max(1.1, float(robots.crawl_delay(AGENT) or robots.crawl_delay("*") or 1))
    last_finished = time.monotonic()
    state_path = output / "crawl_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
    else:
        state = {"queue": [SEED], "done": [], "links": {}, "failures": {}, "excluded": {}}
    extra_records = json.loads(supplemental_file.read_text()) if supplemental_file else []
    supplemental = set(state.get("supplemental_urls", []))
    for row in extra_records:
        url = canonical(row["url"])
        if not url or not re.search(r"\.html?$", url, re.I):
            raise ValueError("Supplemental sources must be explicit public MIA HTML URLs")
        supplemental.add(url)
        ref = {"from": row["linked_from"], "text": row["title"]}
        refs = state["links"].setdefault(url, [])
        if ref not in refs:
            refs.append(ref)
    state["supplemental_urls"] = sorted(supplemental)
    queue = deque(state["queue"])
    done = set(state["done"])
    queued = set(queue)
    in_flight = state.get("in_flight")
    if in_flight and in_flight not in done and in_flight not in queued:
        queue.appendleft(in_flight)
        queued.add(in_flight)
    for url in sorted(supplemental):
        if url not in done and url not in queued:
            queue.append(url)
            queued.add(url)
            state["excluded"].pop(url, None)
    in_flight = None
    count = 0

    def save():
        state["queue"] = list(queue)
        state["done"] = sorted(done)
        state["in_flight"] = in_flight
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_path, state)

    try:
        while queue and (limit == 0 or count < limit):
            url = queue.popleft()
            queued.discard(url)
            if url in done:
                continue
            in_flight = url
            save()
            if not robots.can_fetch(AGENT, url):
                state["excluded"][url] = "robots.txt"
                done.add(url)
                in_flight = None
                continue
            record = None
            for attempt in range(3):
                time.sleep(max(0, delay - (time.monotonic() - last_finished)))
                try:
                    # Check every redirect before requesting its target.
                    target = url
                    for _ in range(6):
                        response = session.get(target, timeout=(15, 45), allow_redirects=False)
                        last_finished = time.monotonic()
                        if response.is_redirect:
                            nxt = canonical(urljoin(target, response.headers.get("Location", "")))
                            if not allowed(nxt, supplemental) or not robots.can_fetch(AGENT, nxt):
                                raise ValueError("redirect outside allowed crawl scope")
                            time.sleep(delay)
                            target = nxt
                        else:
                            break
                    if response.is_redirect:
                        raise ValueError("too many redirects")
                    if response.status_code in {429, 503}:
                        time.sleep(min(30, max(5, int(response.headers.get("Retry-After", "5")))))
                    response.raise_for_status()
                    if "html" not in response.headers.get("Content-Type", "").lower():
                        raise ValueError("non-HTML response")
                    raw = response.content
                    key = hashlib.sha256(url.encode()).hexdigest()[:20]
                    path = output / "raw_html" / (key + ".html")
                    path.write_bytes(raw)
                    soup = BeautifulSoup(raw, "lxml")
                    title = soup.title.get_text(" ", strip=True) if soup.title else ""
                    record = {
                        "url": url, "final_url": target, "status": response.status_code,
                        "fetched_utc": datetime.now(timezone.utc).isoformat(),
                        "raw_path": str(path.relative_to(output)), "sha256": hashlib.sha256(raw).hexdigest(),
                        "bytes": len(raw), "content_type": response.headers.get("Content-Type"),
                        "html_title": title, "detected_encoding": soup.original_encoding,
                    }
                    if "RESTRICTED DIRECTORY" not in title.upper():
                        found = []
                        for a in soup.find_all(["a", "area"], href=True):
                            linked = canonical(urljoin(target, a["href"]))
                            if not linked or linked == url:
                                continue
                            if allowed(linked, supplemental):
                                refs = state["links"].setdefault(linked, [])
                                ref = {"from": url, "text": a.get_text(" ", strip=True)}
                                if ref not in refs:
                                    refs.append(ref)
                                if linked not in done and linked not in queued:
                                    found.append(linked)
                                    queued.add(linked)
                            else:
                                state["excluded"].setdefault(linked, "outside Trotsky HTML scope or biography/photo/quotation collection")
                        # Visit late-life material early, while still completing the entire queue.
                        found.sort(key=lambda u: (not bool(re.search(r"/(?:193[7-9]|1940)/", u)), u))
                        queue.extend(found)
                    break
                except (requests.RequestException, ValueError) as exc:
                    last_finished = time.monotonic()
                    error = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, requests.HTTPError) and response.status_code in {401, 403, 404, 410}:
                        break
                    if attempt < 2:
                        time.sleep(2 ** (attempt + 1))
            done.add(url)
            in_flight = None
            count += 1
            if record:
                with (output / "downloads.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                state["failures"].pop(url, None)
            else:
                state["failures"][url] = error
            save()
            if count % 10 == 0 or not record or not queue:
                print(json.dumps({"visited": len(done), "pending": len(queue), "failures": len(state["failures"]), "last": url}), flush=True)
    finally:
        save()
        session.close()
    print(json.dumps({"finished": not queue, "visited": len(done), "pending": len(queue), "failures": len(state["failures"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("corpus_mia"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum requests this run; 0 means exhaust the discovered queue")
    parser.add_argument("--supplemental", type=Path, help="Explicit additional MIA HTML works linked from the main index; no general off-archive crawl")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".crawl.lock").open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another downloader owns this corpus directory")
        crawl(args.output, args.limit, args.supplemental)
