"""Render the interactive datamap to docs/index.html, which GitHub Pages serves.

Everything shown publicly is metadata or model-written text: title, authors,
dates, the generated one-line summary, and the categorical labels. SEP's own
prose stays out of the page (it's in the embeddings, not the HTML).

    uv run render.py
    uv run social_preview.py   # screenshot of the rendered page -> link-card image
"""

import colorsys
import html
import os
import re
import tempfile

import datamapplot
import glasbey
import pandas as pd
from matplotlib.colors import to_hex, to_rgb

from cluster import LABELS_PARQUET
from common import EDITION, ENTRIES_PARQUET, ROOT, write_bytes_atomic
from enrich import ENRICHMENT_PARQUET

DOCS = ROOT / "docs"
OUTPUT = DOCS / "index.html"
SOCIAL_PREVIEW = DOCS / "social-preview.png"
PUBLIC_URL = "https://stevenfazzio.com/sep-datamap/"
TITLE = "Stanford Encyclopedia of Philosophy Map"

# Type: a serif title reads as a reference work to this map's humanities audience,
# while labels and hovercards stay in a sans, which holds up small, bold, and
# outlined over points and is quicker to scan. The two are a designed pair.
# Source Sans *Pro*, the earlier release of Source Sans 3: DataMapPlot writes the
# family name unquoted into CSS and into deck.gl's canvas font string, and a name
# token starting with a digit is invalid there. "Source Sans 3" fell back to Times
# in the page and to the canvas default "10px sans-serif" for the region labels.
BODY_FONT = "Source Sans Pro"  # DataMapPlot loads it (labels, subtitle, hovercards)
TITLE_FONT_CSS = (
    "https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500"
    "&display=swap"
)
SEASONS = {"spr": "Spring", "sum": "Summer", "fall": "Fall", "win": "Winter"}

# DataMapPlot sets line-height 0.95 on its panels, so a wrapped title's descenders
# run into the subtitle. The pill styling matches huggingface-dataset-map's badge.
CUSTOM_CSS = """
#main-title {
  font-family: 'Source Serif 4', Georgia, serif !important;
  font-weight: 500 !important;
  line-height: 1.1 !important;
  letter-spacing: -0.01em;
  /* When it wraps on laptop-width screens, split evenly rather than strand "Map". */
  text-wrap: balance;
}
#main-title + br + span {
  display: inline-block; margin-top: 4px; line-height: 1.3; text-wrap: pretty;
}
.title-pill {
  display: inline-block; margin: 8px 6px 0 0; padding: 3px 10px;
  font-size: 11px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
  color: #636c76; background: rgba(99, 108, 118, 0.08);
  border: 1px solid rgba(99, 108, 118, 0.15); border-radius: 12px;
}
"""

ENTRY_TYPE_LABELS = {
    "philosopher": "Philosopher",
    "philosophers_views": "A philosopher's views",
    "text": "Single work",
    "tradition": "Tradition or period",
    "topic": "Topic",
    "other": "Other",
}
SUBFIELD_LABELS = {
    "metaphysics": "Metaphysics",
    "epistemology": "Epistemology",
    "logic": "Logic",
    "language": "Philosophy of language",
    "mind": "Philosophy of mind",
    "ethics": "Ethics",
    "political_social": "Social & political",
    "aesthetics": "Aesthetics",
    "science": "Philosophy of science",
    "mathematics": "Philosophy of mathematics",
    "religion": "Philosophy of religion",
    "law": "Philosophy of law",
    "action_decision": "Action & decision",
    "several": "Several subfields",
    "other": "Other",
}
TRADITION_LABELS = {
    "ancient": "Ancient",
    "medieval": "Medieval",
    "renaissance": "Renaissance",
    "early_modern": "Early modern",
    "nineteenth_century": "19th century",
    # The enrichment prompt files thematic topic entries framed by current debate
    # here, so for topics this means "contemporary", not membership of a school.
    "analytic": "Analytic / contemporary",
    "continental": "Continental",
    "chinese": "Chinese",
    "indian": "Indian",
    "buddhist": "Buddhist",
    "japanese": "Japanese",
    "islamic": "Islamic world",
    "jewish": "Jewish",
    "african": "Africana",
    "latin_american": "Latin American",
    "other": "Other",
}

# Non-categories are pinned to greys so the named categories carry the colour.
# "Analytic / contemporary" is 61% of entries and mostly re-encodes entry type, so
# as a pale background it lets the historical and non-Western traditions stand out.
NEUTRAL_COLORS = {
    "Analytic / contemporary": "#cbcbcb",
    "Several subfields": "#cbcbcb",
    "Other": "#8c8c8c",
    "Unknown": "#e2e2e2",
}

# Cards are read in quick succession while sweeping the mouse, so every card has
# the same layout: type pill, title, labelled fields, then the summary and the
# credit line. The fields go above the summary because summaries run 2-5 lines:
# below it, the fields jumped by up to 55 px from card to card; under the title
# they move only when the title wraps. The dots use the Subfield and Tradition
# colormap colours so a card reads against whichever legend is showing.
DOT = (
    "display: inline-block; width: 8px; height: 8px; border-radius: 50%; "
    "margin-right: 6px; box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.15); "
    "background: {color};"
)
FIELD_NAME = "color: #6b7280; font-size: 11px; padding-top: 1px;"
HOVER_TEMPLATE = f"""
<div style="max-width: 340px; white-space: normal; font-weight: 400; color: #1f2328;">
  <span style="display: inline-block; padding: 1px 8px; margin-bottom: 5px;
    border-radius: 999px; font-size: 11px; font-weight: 600;
    background: {{pill_bg}}; color: {{pill_fg}};">{{entry_type}}</span>
  <div style="font-weight: 600; font-size: 15px; line-height: 1.25;">{{title}}</div>
  <div style="display: grid; grid-template-columns: 62px minmax(0, 1fr);
    gap: 3px 10px; margin-top: 6px; font-size: 12px; line-height: 1.35;">
    <div style="{FIELD_NAME}">Subfield</div>
    <div><span style="{DOT.format(color="{subfield_dot}")}"></span>{{subfield}}</div>
    <div style="{FIELD_NAME}">Tradition</div>
    <div><span style="{DOT.format(color="{tradition_dot}")}"></span>{{tradition}}</div>
  </div>
  <div style="font-size: 13px; margin-top: 8px; padding-top: 7px;
    border-top: 1px solid #e5e7eb; line-height: 1.4;">{{summary}}</div>
  <div style="color: #6b7280; font-size: 11px; margin-top: 7px;">{{credit}}</div>
</div>
"""


def categorical_palette(n_colors: int, avoid: list[str]) -> list[str]:
    """n_colors glasbey colours, each also kept distinct from the `avoid` colours."""
    # Seeding with the greys already on the map stops glasbey handing out a muted
    # colour that reads as one of them (a sage green did, before seeding).
    palette = glasbey.extend_palette(
        avoid,
        palette_size=len(avoid) + n_colors,
        colorblind_safe=True,
        # glasbey's default, moderate red-green CVD.
        cvd_severity=50.0,
        # A floor of 35 rather than 25: near-black colours are distinct as swatches
        # but read as the same colour as small dots (Logic vs. science did).
        lightness_bounds=(35, 75),
        chroma_bounds=(20, 90),
    )
    return palette[len(avoid) :]


def color_mapping(values: pd.Series) -> dict[str, str]:
    """Category -> colour, assigned most frequent first.

    glasbey's palette is greedy, so its earliest colours are the most distinct;
    handing them out by frequency gives the largest categories the clearest colours.
    """
    order = values.value_counts().index.tolist()
    named = [c for c in order if c not in NEUTRAL_COLORS]
    neutrals = {c: NEUTRAL_COLORS[c] for c in order if c in NEUTRAL_COLORS}
    mapping = dict(zip(named, categorical_palette(len(named), list(neutrals.values()))))
    mapping.update(neutrals)
    return mapping


def categorical_colormap(field: str, description: str, values: pd.Series):
    # Explicit "Unknown" rather than dropping points, so gaps in the metadata don't
    # read as holes in the map.
    values = values.fillna("Unknown").astype(str)
    mapping = color_mapping(values)
    meta = {
        "field": field,
        "description": description,
        "kind": "categorical",
        # An explicit mapping, rather than a palette DataMapPlot assigns itself, so
        # the hovercard pill can use exactly the colours on the map.
        "color_mapping": mapping,
    }
    return values.to_numpy(), meta, mapping


def relative_luminance(rgb) -> float:
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b) -> float:
    la, lb = sorted([relative_luminance(a), relative_luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def pill_colors(hex_color: str) -> tuple[str, str]:
    """A pale tint for the pill background and a deep shade of the same hue for text."""
    h, _, s = colorsys.rgb_to_hls(*to_rgb(hex_color))
    bg = colorsys.hls_to_rgb(h, 0.92, min(s, 0.65))
    lightness = 0.35
    fg = colorsys.hls_to_rgb(h, lightness, s)
    while contrast(fg, bg) < 4.5:  # WCAG AA for small text
        lightness -= 0.03
        fg = colorsys.hls_to_rgb(h, lightness, s)
    return to_hex(bg), to_hex(fg)


def display(values: pd.Series, others: pd.Series, labels: dict) -> pd.Series:
    """Human label for an enum value, using the free-text escape hatch for `other`."""
    shown = values.map(labels)
    use_other = (values == "other") & others.fillna("").str.len().gt(0)
    shown[use_other] = others[use_other].str.strip().str.capitalize()
    return shown.fillna("Unknown")


def credit(row) -> str:
    authors = list(row.authors)
    names = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    dates = []
    if pd.notna(row.first_published):
        dates.append(f"Published {row.first_published.year}")
    if pd.notna(row.last_revised):
        dates.append(f"revised {row.last_revised.year}")
    return " · ".join(p for p in [names, ", ".join(dates)] if p)


def edition_label(edition: str) -> str:
    """'fall2026' -> 'Fall 2026 edition'."""
    season, year = edition[:-4], edition[-4:]
    return f"{SEASONS[season]} {year} edition"


def title_pills() -> str:
    # The edition, not a fetch date: the corpus is pinned to that archive snapshot.
    pills = [edition_label(EDITION), "Unofficial"]
    spans = "".join(f'<span class="title-pill">{html.escape(p)}</span>' for p in pills)
    return f"<div>{spans}</div>"


def open_graph_tags(n_entries: int) -> str:
    og = {
        "og:title": TITLE,
        "og:description": (
            f"{n_entries:,} Stanford Encyclopedia of Philosophy entries, laid out by the "
            "meaning of each entry's lead section and named at four zoom levels. "
            "Pan, zoom, hover and search."
        ),
        "og:type": "website",
        "og:url": PUBLIC_URL,
        # Declared unconditionally: social_preview.py screenshots the rendered page,
        # so the image is always produced after the HTML that names it.
        "og:image": PUBLIC_URL + SOCIAL_PREVIEW.name,
    }
    tags = [
        f'<meta property="{k}" content="{html.escape(v, quote=True)}">'
        for k, v in og.items()
    ]
    tags.append(
        f'<meta name="description" content="{html.escape(og["og:description"])}">'
    )
    tags.append('<meta name="twitter:card" content="summary_large_image">')
    return "\n".join(tags)


def main():
    entries = pd.read_parquet(ENTRIES_PARQUET)
    enrichment = pd.read_parquet(ENRICHMENT_PARQUET)
    labels = pd.read_parquet(LABELS_PARQUET)

    df = entries.merge(labels, on="slug", how="left", validate="1:1")
    df = df.merge(enrichment, on="slug", how="left", validate="1:1")
    assert len(df) == len(entries), "merge changed the row count"
    missing_xy = df.x.isna().sum()
    assert missing_xy == 0, f"{missing_xy} entries have no coordinates"
    print(f"render: {len(df)} entries; {df.summary.isna().sum()} without enrichment")

    type_label = display(df.entry_type, df.entry_type_other, ENTRY_TYPE_LABELS)
    subfield_label = display(df.subfield, df.subfield_other, SUBFIELD_LABELS)
    tradition_label = display(df.tradition, df.tradition_other, TRADITION_LABELS)
    summary = df.summary.fillna("").str.strip()
    # About 5% of generated summaries omit the final period.
    unterminated = summary.ne("") & ~summary.str.endswith((".", "?", "!"))
    summary[unterminated] = summary[unterminated] + "."
    authors = df.authors.map(lambda a: ", ".join(a))

    # Colormap values collapse free-text "other" answers into one "Other" category;
    # the hovercard and search keep the specific text.
    type_values, type_meta, type_colors = categorical_colormap(
        "type", "Entry type", type_label.where(df.entry_type != "other", "Other")
    )
    subfield_values, subfield_meta, subfield_colors = categorical_colormap(
        "subfield", "Subfield", subfield_label.where(df.subfield != "other", "Other")
    )
    tradition_values, tradition_meta, tradition_colors = categorical_colormap(
        "tradition",
        "Tradition",
        tradition_label.where(df.tradition != "other", "Other"),
    )
    pills = {cat: pill_colors(color) for cat, color in type_colors.items()}

    extra = pd.DataFrame(
        {
            "title": df.title.map(html.escape),
            "summary": summary.map(html.escape),
            "entry_type": type_label.map(html.escape),
            "pill_bg": [pills[v][0] for v in type_values],
            "pill_fg": [pills[v][1] for v in type_values],
            # Values show the free-text answer for "other"; dots use the colormap
            # category the point is actually coloured by.
            "subfield": subfield_label.map(html.escape),
            "subfield_dot": [subfield_colors[v] for v in subfield_values],
            "tradition": tradition_label.map(html.escape),
            "tradition_dot": [tradition_colors[v] for v in tradition_values],
            "credit": [html.escape(credit(r)) for r in df.itertuples()],
            "url": df.url,
            # Search the composed text, not just titles: the summary and facets are
            # where an entry's content shows up.
            "search_text": (
                df.title
                + " "
                + authors
                + " "
                + summary
                + " "
                + type_label
                + " "
                + subfield_label
                + " "
                + tradition_label
            ),
        }
    )

    label_cols = sorted(
        (c for c in df.columns if c.startswith("label_layer_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),  # numeric: _10 must sort after _2
    )
    label_layers = [df[c].fillna("Unlabelled").to_numpy() for c in label_cols]
    n_finest = pd.Series(label_layers[0]).nunique()
    n_coarsest = pd.Series(label_layers[-1]).nunique()
    assert n_finest >= n_coarsest, "label layers must be finest-first for DataMapPlot"

    year = df.first_published.dt.year.to_numpy().astype(float)
    # Entry type first: it is the split the layout is organised around.
    colormap_rawdata = [type_values, subfield_values, tradition_values, year]
    colormap_metadata = [
        type_meta,
        subfield_meta,
        tradition_meta,
        {
            "field": "year",
            "description": "Year first published",
            "kind": "continuous",
            "cmap": "viridis",
        },
    ]

    fig = datamapplot.create_interactive_plot(
        df[["x", "y"]].to_numpy(),
        *label_layers,
        hover_text=df.title.to_numpy(),
        extra_point_data=extra,
        hover_text_html_template=HOVER_TEMPLATE,
        on_click="window.open(`{url}`)",
        enable_search=True,
        search_field="search_text",
        colormap_rawdata=colormap_rawdata,
        colormap_metadata=colormap_metadata,
        cvd_safer=True,
        title=TITLE,
        sub_title=(
            f"{len(df):,} entries, placed by the meaning of their lead sections. "
            # Nothing else says clicking works: deck.gl shows a grab cursor on points.
            # Kept short enough to stay on one line at 1280 px, where a second line
            # pushes the panel over the top region label.
            "Click one to open it."
        ),
        # The defaults (36/18) make the title panel cover the top of the map and
        # collide with the colormap legend on laptop-width screens.
        title_font_size=24,
        sub_title_font_size=13,
        font_family=BODY_FONT,
        tooltip_font_family=BODY_FONT,
        tooltip_font_weight=400,
        custom_css=CUSTOM_CSS,
        noise_label="Unlabelled",
    )

    DOCS.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=DOCS, suffix=".html")
    os.close(fd)
    try:
        fig.save(tmp)
        text = open(tmp, encoding="utf-8").read()
        # After the charset declaration, which browsers only honour in the first
        # 1024 bytes of the document.
        anchor = re.search(r"<meta[^>]*charset[^>]*>", text) or re.search(
            "<head>", text
        )
        assert anchor is not None, "no <head> in the rendered page"
        at = anchor.end()
        head = (
            open_graph_tags(len(df))
            + '\n<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            + f'\n<link rel="stylesheet" href="{TITLE_FONT_CSS}">'
        )
        text = text[:at] + "\n" + head + text[at:]
        # Pills go inside the title panel, after the subtitle; the panel holds only
        # spans, so its first closing </div> is its own.
        text, n = re.subn(
            r'(<div\s+id="title-container"[^>]*>.*?)(</div>)',
            lambda m: m.group(1) + title_pills() + m.group(2),
            text,
            count=1,
            flags=re.DOTALL,
        )
        assert n == 1, "title panel not found for the pills"
        write_bytes_atomic(text.encode("utf-8"), OUTPUT)
    finally:
        os.unlink(tmp)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB)")
    print(f"  next: uv run social_preview.py to refresh {SOCIAL_PREVIEW.name}")


if __name__ == "__main__":
    main()
