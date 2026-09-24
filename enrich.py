"""LLM-extracted fields for each SEP entry, via the Claude Message Batches API.

One call per entry returns: a one-line summary (in the model's own words, so the
public map carries no SEP prose), entry type, primary subfield, and tradition.
The categoricals are schema enums so the colormaps stay clean; each has an
`_other` free-text escape hatch for the hovercard and search.

    uv run enrich.py --sample 60   # validation run on a fixed random sample
    uv run enrich.py               # everything not yet enriched

Resumable: the batch id is checkpointed to disk right after submission, and a
re-run polls that batch instead of submitting a new one. Refusals are recorded
as terminal; errored or expired requests are left out so the next run retries them.
"""

import argparse
import json
import os
import random
import time
from enum import StrEnum

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from pydantic import BaseModel, ValidationError

from common import DATA, ENTRIES_PARQUET, write_bytes_atomic, write_parquet_safely

MODEL = "claude-opus-5"
EFFORT = "medium"
ENRICHMENT_PARQUET = DATA / "enrichment.parquet"
PENDING_FILE = DATA / "enrich_pending.json"
# Batch API rates for claude-opus-5 (50% of $5 / $25 per MTok).
USD_PER_MTOK_IN, USD_PER_MTOK_OUT = 2.50, 12.50


class EntryType(StrEnum):
    philosopher = "philosopher"
    philosophers_views = "philosophers_views"
    text = "text"
    tradition = "tradition"
    topic = "topic"
    other = "other"


class Subfield(StrEnum):
    metaphysics = "metaphysics"
    epistemology = "epistemology"
    logic = "logic"
    language = "language"
    mind = "mind"
    ethics = "ethics"
    political_social = "political_social"
    aesthetics = "aesthetics"
    science = "science"
    mathematics = "mathematics"
    religion = "religion"
    law = "law"
    action_decision = "action_decision"
    several = "several"
    other = "other"


class Tradition(StrEnum):
    ancient = "ancient"
    medieval = "medieval"
    renaissance = "renaissance"
    early_modern = "early_modern"
    nineteenth_century = "nineteenth_century"
    analytic = "analytic"
    continental = "continental"
    chinese = "chinese"
    indian = "indian"
    buddhist = "buddhist"
    japanese = "japanese"
    islamic = "islamic"
    jewish = "jewish"
    african = "african"
    latin_american = "latin_american"
    other = "other"


class Enrichment(BaseModel):
    summary: str
    entry_type: EntryType
    entry_type_other: str
    subfield: Subfield
    subfield_other: str
    tradition: Tradition
    tradition_other: str


def enum_prop(enum: type[StrEnum]) -> dict:
    return {"type": "string", "enum": [e.value for e in enum]}


# Written out rather than taken from Enrichment.model_json_schema(), which routes
# enums through $defs/$ref; a flat schema is simplest for structured outputs.
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "entry_type": enum_prop(EntryType),
        "entry_type_other": {"type": "string"},
        "subfield": enum_prop(Subfield),
        "subfield_other": {"type": "string"},
        "tradition": enum_prop(Tradition),
        "tradition_other": {"type": "string"},
    },
    "required": list(Enrichment.model_fields),
    "additionalProperties": False,
}

SYSTEM = """\
You are labelling entries of the Stanford Encyclopedia of Philosophy for an \
interactive map of the whole encyclopedia. For each entry you get its title, its \
top-level section headings, and its lead section. Return the fields below.

summary
One sentence of at most 25 words saying what the entry's subject is: the concept, \
problem, person, work, or tradition itself, not what the article does. Write it in \
your own words. Do not quote or closely paraphrase the lead, use no quotation marks, \
and do not begin with "This entry" or restate the title. For a person, say who they \
were (era, place, role) and what they are known for. It appears in a hover card \
directly under the title.

entry_type
- philosopher: a person as a whole, their life and thought (e.g. "Peter Abelard").
- philosophers_views: one person's views on a specific topic or one area of their \
work (e.g. "Kant's Aesthetics", "Aristotle's Logic").
- text: a single work (e.g. a particular treatise or sutra).
- tradition: a school, movement, tradition, or historical period as such (e.g. \
"Abhidharma", "18th Century German Philosophy Prior to Kant").
- topic: a concept, problem, theory, argument, or field (e.g. "Abduction", "Moral Luck").
- other: none of these; name the kind in entry_type_other.

subfield
The main area of philosophy the subject belongs to: metaphysics, epistemology, logic, \
language (philosophy of language), mind (including cognitive science and perception), \
ethics (normative, meta-, and applied), political_social, aesthetics, science \
(philosophy of science, physics, biology, etc.), mathematics (philosophy of \
mathematics), religion, law, action_decision (action theory, agency, decision and \
game theory). Use several only when the subject genuinely spans areas with no primary \
one, which is typical for major philosophers and broad traditions; a topic entry \
nearly always has one primary area. Use other and name it in subfield_other when \
none fits.

tradition
The historical or cultural tradition the subject belongs to: ancient (Greco-Roman), \
medieval (Latin West and Byzantium), renaissance, early_modern (c. 1600 to Kant), \
nineteenth_century (post-Kantian Western thought), analytic (from Frege onward, \
including contemporary Anglophone philosophy), continental (phenomenology, \
existentialism, hermeneutics, critical theory, post-structuralism, and so on), \
chinese, indian (non-Buddhist Indian traditions), buddhist (any region), japanese, \
islamic (Arabic, Persian, and other Islamic-world philosophy), jewish, african, \
latin_american. A thematic topic entry framed by contemporary debate is analytic even \
if it surveys historical antecedents; use a historical tradition only when the entry \
is mainly about that period's or culture's thought. Use other and name it in \
tradition_other when none fits.

Leave each *_other field as an empty string unless its field is other."""


def user_message(row) -> str:
    headings = "\n".join(f"- {h}" for h in row.toc_sections) or "(none)"
    return (
        f"Title: {row.title}\n\nSection headings:\n{headings}\n\n"
        f"Lead section:\n{row.preamble}"
    )


def build_request(row) -> Request:
    return Request(
        custom_id=row.slug,
        params=MessageCreateParamsNonStreaming(
            model=MODEL,
            max_tokens=4000,
            # Identical across every request, and over Opus 5's 512-token minimum.
            system=[
                {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}
            ],
            thinking={"type": "adaptive"},
            output_config={
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            messages=[{"role": "user", "content": user_message(row)}],
        ),
    )


def load_done() -> pd.DataFrame:
    if ENRICHMENT_PARQUET.exists():
        return pd.read_parquet(ENRICHMENT_PARQUET)
    return pd.DataFrame(columns=["slug"])


def submit(client, rows: list) -> dict:
    batch = client.messages.batches.create(requests=[build_request(r) for r in rows])
    # Checkpoint before polling: a re-run resumes this batch rather than paying twice.
    pending = {
        "batch_id": batch.id,
        "slugs": [r.slug for r in rows],
        "model": MODEL,
        "effort": EFFORT,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_bytes_atomic(json.dumps(pending, indent=2).encode(), PENDING_FILE)
    print(f"submitted batch {batch.id} with {len(rows)} requests")
    return pending


def wait(client, batch_id: str) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(
            f"  {batch.processing_status}: processing={c.processing} "
            f"succeeded={c.succeeded} errored={c.errored} expired={c.expired}",
            flush=True,
        )
        if batch.processing_status == "ended":
            return
        time.sleep(60)


def collect(client, pending: dict) -> tuple[list[dict], dict]:
    rows, failures = [], {}
    for result in client.messages.batches.results(pending["batch_id"]):
        slug = result.custom_id
        if result.result.type != "succeeded":
            # errored / expired / canceled: not recorded, so the next run retries it
            failures[slug] = result.result.type
            continue
        msg = result.result.message
        base = {
            "slug": slug,
            "model": msg.model,
            "input_tokens": msg.usage.input_tokens,
            "cache_write_tokens": msg.usage.cache_creation_input_tokens or 0,
            "cache_read_tokens": msg.usage.cache_read_input_tokens or 0,
            "output_tokens": msg.usage.output_tokens,
            "batch_id": pending["batch_id"],
        }
        # A refusal is HTTP 200 with no usable content, so check before reading it.
        # Terminal: an identical retry is declined identically.
        if msg.stop_reason == "refusal":
            rows.append({**base, "status": "refusal"})
            continue
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if msg.stop_reason == "max_tokens" or text is None:
            failures[slug] = f"stop_reason={msg.stop_reason}"
            continue
        try:
            fields = Enrichment.model_validate_json(text).model_dump(mode="json")
        except ValidationError as e:
            failures[slug] = f"validation: {e.errors()[0]['msg']}"
            continue
        rows.append({**base, "status": "ok", **fields})
    return rows, failures


def report(df: pd.DataFrame) -> None:
    ok = df[df.status == "ok"]
    print(f"\n{len(ok)} ok, {(df.status == 'refusal').sum()} refusals")
    for col in ["entry_type", "subfield", "tradition"]:
        print(f"\n{col}:\n{ok[col].value_counts().to_string()}")
        others = ok.loc[ok[col] == "other", f"{col}_other"]
        if len(others):
            print(f"  {col}_other: {sorted(others)}")
    words = ok.summary.str.split().str.len()
    print(
        f"\nsummary words: min={words.min()} median={words.median():.0f} max={words.max()}"
    )
    # Cache writes bill at 1.25x the input rate, cache reads at 0.1x.
    cost = (
        df.input_tokens * USD_PER_MTOK_IN
        + df.cache_write_tokens * USD_PER_MTOK_IN * 1.25
        + df.cache_read_tokens * USD_PER_MTOK_IN * 0.1
        + df.output_tokens * USD_PER_MTOK_OUT
    ) / 1e6
    print(
        f"tokens/entry: {df.input_tokens.mean():.0f} in, "
        f"{df.cache_read_tokens.mean():.0f} cache-read, "
        f"{df.cache_write_tokens.mean():.0f} cache-write, "
        f"{df.output_tokens.mean():.0f} out → ${cost.mean():.4f}/entry, "
        f"${cost.sum():.2f} total at batch rates"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    client = anthropic.Anthropic()
    entries = pd.read_parquet(ENTRIES_PARQUET)
    done = load_done()

    if PENDING_FILE.exists():
        pending = json.loads(PENDING_FILE.read_text())
        print(
            f"resuming batch {pending['batch_id']} ({len(pending['slugs'])} requests)"
        )
    else:
        todo = entries[~entries.slug.isin(done.slug)].sort_values("slug")
        if args.sample is not None:
            # Sample from the whole corpus with a fixed seed, then drop what's done,
            # so the validation set is stable across re-runs.
            sample = set(random.Random(0).sample(sorted(entries.slug), args.sample))
            todo = todo[todo.slug.isin(sample)]
        if todo.empty:
            print("nothing to enrich")
            report(done)
            return
        print(f"{len(done)} already enriched; submitting {len(todo)}")
        pending = submit(client, list(todo.itertuples()))

    wait(client, pending["batch_id"])
    rows, failures = collect(client, pending)
    print(
        f"enrich: {len(pending['slugs'])} submitted → {len(rows)} recorded "
        f"({len(failures)} to retry)"
    )
    for slug, why in failures.items():
        print(f"  RETRY {slug}: {why}")

    new = pd.DataFrame(rows, columns=None if rows else ["slug"])
    merged = pd.concat([done[~done.slug.isin(new.slug)], new], ignore_index=True)
    write_parquet_safely(merged.sort_values("slug"), ENRICHMENT_PARQUET)
    os.remove(PENDING_FILE)
    report(merged)


if __name__ == "__main__":
    main()
