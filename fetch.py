"""Fetch the SEP contents page and every entry page from the pinned archive edition.

Raw HTML is cached under data/raw/<edition>/. Resume is by file existence: every
file is written atomically, so a file on disk is a complete fetch. Honors the
5-second crawl-delay in SEP's robots.txt.

    uv run fetch.py            # everything
    uv run fetch.py --limit 5  # first 5 missing entries, for testing
"""

import argparse
import json
import random
import re
import time

import requests

from common import ARCHIVE_BASE, RAW, RAW_ENTRIES, write_bytes_atomic

CRAWL_DELAY = 5.0
RETRY_STATUS = {429, 500, 502, 503, 504}
USER_AGENT = "sep-datamap/0.1 (personal research project; polite crawler)"
FAILURES_FILE = RAW / "fetch_failures.json"

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT


def fetch_with_retry(url: str, max_retries: int = 4, timeout: int = 30) -> bytes:
    for attempt in range(max_retries):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code in RETRY_STATUS and attempt < max_retries - 1:
                retry_after = resp.headers.get("Retry-After", "")
                backoff = min(2**attempt * 10, 120)
                wait = float(retry_after) if retry_after.isdigit() else backoff
                wait += random.uniform(0, wait * 0.1)
                print(f"  {resp.status_code} on {url}, retrying in {wait:.0f}s")
            else:
                resp.raise_for_status()
                return resp.content
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 10, 120)
            print(f"  attempt {attempt + 1}: {e}, retrying in {wait:.0f}s")
        time.sleep(wait)
    raise RuntimeError(f"exhausted retries for {url}")


def entry_slugs(contents_html: str) -> list[str]:
    # The contents page lists many entries more than once ("X — see Y"), so dedupe.
    return sorted(set(re.findall(r'href="entries/([^"/]+)/"', contents_html)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    contents_path = RAW / "contents.html"
    if not contents_path.exists():
        write_bytes_atomic(
            fetch_with_retry(ARCHIVE_BASE + "contents.html"), contents_path
        )
        time.sleep(CRAWL_DELAY)
    slugs = entry_slugs(contents_path.read_text(encoding="utf-8"))

    todo = [s for s in slugs if not (RAW_ENTRIES / f"{s}.html").exists()]
    print(
        f"{len(slugs)} entries in contents; {len(slugs) - len(todo)} cached, {len(todo)} to fetch"
    )
    if args.limit is not None:
        todo = todo[: args.limit]
    if todo:
        print(
            f"~{len(todo) * CRAWL_DELAY / 3600:.1f} h at {CRAWL_DELAY:.0f}s crawl delay"
        )

    failures = {}
    for n, slug in enumerate(todo, start=1):
        try:
            html = fetch_with_retry(f"{ARCHIVE_BASE}entries/{slug}/")
            write_bytes_atomic(html, RAW_ENTRIES / f"{slug}.html")
        except Exception as e:  # one bad entry shouldn't stop the crawl
            failures[slug] = repr(e)
            print(f"  FAILED {slug}: {e!r}")
        if n % 50 == 0 or n == len(todo):
            print(
                f"{n}/{len(todo)} fetched this run ({len(failures)} failed)", flush=True
            )
        if n < len(todo):
            time.sleep(CRAWL_DELAY)

    if failures:
        write_bytes_atomic(json.dumps(failures, indent=2).encode(), FAILURES_FILE)
        print(
            f"{len(failures)} failures written to {FAILURES_FILE}; re-run to retry them"
        )
    elif FAILURES_FILE.exists():
        FAILURES_FILE.unlink()


if __name__ == "__main__":
    main()
