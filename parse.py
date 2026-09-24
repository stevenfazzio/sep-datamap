"""Parse cached SEP entry pages into data/entries.parquet.

One row per entry: title, authors, dates, the lead section ("preamble"), the
top-level table of contents, and the Related Entries cross-links.

    uv run parse.py
"""

import html
import re
from datetime import datetime

import pandas as pd
from bs4 import BeautifulSoup

from common import (
    ARCHIVE_BASE,
    ENTRIES_PARQUET,
    LIVE_BASE,
    RAW,
    RAW_ENTRIES,
    write_parquet_safely,
)

# These pages ship with a blank <h1> and no citation_author, in the Fall 2026
# archive and on the live site alike. The contents page only has index-style link
# text ("Weyl, Hermann"), so the display titles are given here.
TITLE_OVERRIDES = {
    "weyl": "Hermann Weyl",
    "skepticism-latin-america": "Skepticism in Latin America",
}

# Boilerplate sections at the end of every entry's table of contents.
TOC_BOILERPLATE = {
    "Bibliography",
    "Academic Tools",
    "Other Internet Resources",
    "Related Entries",
}
DATE_RE = r"[A-Z][a-z]{2} ([A-Z][a-z]{2} \d{1,2}, \d{4})"


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_date(match: re.Match | None) -> pd.Timestamp | None:
    if match is None:
        return None
    return pd.Timestamp(datetime.strptime(match.group(1), "%b %d, %Y"))


def display_name(citation_author: str) -> str:
    # citation_author is "Last, First" (or "Last, Jr., First"); the hovercard wants
    # "First Last".
    parts = [p.strip() for p in citation_author.split(",")]
    if len(parts) == 3:
        last, suffix, first = parts
        return f"{first} {last}, {suffix}"
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return citation_author.strip()


def contents_authors(contents_html: str) -> dict[str, list[str]]:
    """Authors per slug from the contents page: `<strong>..</strong></a> (A and B)`."""
    out = {}
    pattern = r'<a href="entries/([^"/]+)/"><strong>.*?</strong></a>\s*\(([^)]*)\)'
    for slug, names in re.findall(pattern, contents_html):
        names = html.unescape(clean(names))
        out[slug] = [n for n in re.split(r",\s*(?:and\s+)?|\s+and\s+", names) if n]
    return out


def parse_entry(slug: str, page: str) -> dict:
    soup = BeautifulSoup(page, "lxml")
    body = soup.find(id="aueditable")
    preamble = soup.find(id="preamble")
    pubinfo = (
        clean(soup.find(id="pubinfo").get_text(" ")) if soup.find(id="pubinfo") else ""
    )

    toc = soup.find(id="toc")
    toc_sections = []
    if toc is not None and toc.find("ul") is not None:
        for li in toc.find("ul").find_all("li", recursive=False):
            a = li.find("a")
            heading = clean(a.get_text(" ")) if a else ""
            if heading and heading not in TOC_BOILERPLATE:
                toc_sections.append(heading)

    related = []
    rel = soup.find(id="related-entries")
    if rel is not None:
        for a in rel.find_all("a", href=True):
            m = re.fullmatch(r"\.\./([^/]+)/", a["href"])
            if m:
                related.append(m.group(1))

    authors = [
        m["content"] for m in soup.find_all("meta", attrs={"name": "citation_author"})
    ]
    preamble_text = clean(preamble.get_text(" ")) if preamble is not None else ""
    h1 = body.find("h1") if body else None
    title = (clean(h1.get_text(" ")) if h1 else "") or TITLE_OVERRIDES.get(slug, slug)

    return {
        "slug": slug,
        "title": title,
        "authors": [display_name(a) for a in authors if a.strip()],
        "first_published": parse_date(
            re.search(rf"First published {DATE_RE}", pubinfo)
        ),
        "last_revised": parse_date(
            re.search(rf"substantive revision {DATE_RE}", pubinfo)
        ),
        "preamble": preamble_text,
        "preamble_words": len(preamble_text.split()),
        "toc_sections": toc_sections,
        "related": sorted(set(related)),
        "url": f"{LIVE_BASE}entries/{slug}/",
        "archive_url": f"{ARCHIVE_BASE}entries/{slug}/",
    }


def main():
    paths = sorted(RAW_ENTRIES.glob("*.html"))
    rows, failures = [], {}
    for path in paths:
        try:
            rows.append(parse_entry(path.stem, path.read_text(encoding="utf-8")))
        except Exception as e:
            failures[path.stem] = repr(e)
    df = pd.DataFrame(rows)
    print(f"parse: {len(paths)} pages in → {len(df)} rows out ({len(failures)} failed)")
    for slug, err in failures.items():
        print(f"  FAILED {slug}: {err}")

    # Pages without citation_author metadata: fall back to the contents listing.
    fallback = contents_authors((RAW / "contents.html").read_text(encoding="utf-8"))
    no_authors = df.authors.str.len() == 0
    df.loc[no_authors, "authors"] = df.loc[no_authors, "slug"].map(
        lambda s: fallback.get(s, [])
    )
    print(f"  authors from contents page: {sorted(df.slug[no_authors])}")
    print(f"  title from overrides: {sorted(set(df.slug) & set(TITLE_OVERRIDES))}")
    print(f"  title still missing: {sorted(df.slug[df.title == df.slug])}")

    # Report, don't drop: a thin or missing preamble is something to look at.
    print(f"  no preamble: {(df.preamble_words == 0).sum()}")
    print(f"  preamble < 40 words: {(df.preamble_words.between(1, 39)).sum()}")
    print(f"  no authors: {(df.authors.str.len() == 0).sum()}")
    print(f"  no first_published: {df.first_published.isna().sum()}")
    print(f"  preamble words: {df.preamble_words.describe().round(0).to_dict()}")

    write_parquet_safely(df, ENTRIES_PARQUET)
    print(f"wrote {ENTRIES_PARQUET}")


if __name__ == "__main__":
    main()
