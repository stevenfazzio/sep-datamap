# A Map of the Stanford Encyclopedia of Philosophy

An interactive map of the 1,866 entries in the [Stanford Encyclopedia of
Philosophy](https://plato.stanford.edu/) (Fall 2026 edition), placed by the meaning
of each entry's lead section and named at four zoom levels.

**Live map:** https://stevenfazzio.com/sep-datamap/

This is an unofficial project and is not affiliated with the SEP. The published
page carries only entry metadata (titles, authors, dates) and model-written text
(one-line summaries, labels, region names); none of the SEP's own prose is
redistributed. Clicking a point opens the entry on the SEP site.

## How it's built

| Stage | Script | What it does |
|---|---|---|
| Fetch | `fetch.py` | Crawls the pinned Fall 2026 archive edition, honouring the 5 s crawl delay in SEP's robots.txt |
| Parse | `parse.py` | Title, authors, dates, lead section, section headings, and Related Entries per entry |
| Enrich | `enrich.py` | Claude (Message Batches API) writes a one-line summary and labels entry type, subfield, and tradition |
| Embed | `embed.py` | Qwen3-Embedding-4B, served by vLLM on a RunPod serverless endpoint, embeds title + lead |
| Cluster | `cluster.py` | UMAP to 2-d, then [Toponymy](https://github.com/TutteInstitute/toponymy) finds and names regions with Claude |
| Render | `render.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) builds `docs/index.html` |
| Card | `social_preview.py` | Screenshots the rendered map in Chrome for the link-preview image |

Regions are found on the same 2-d layout that is plotted, so every named region
matches something visible on the map.

## Reproducing

Requires [uv](https://docs.astral.sh/uv/), `ANTHROPIC_API_KEY`, and, for the
embedding steps, `RUNPOD_API_KEY` plus an endpoint serving Qwen3-Embedding-4B
(its ID is set in `embedder.py`). The social card needs a local Chrome.

```bash
uv run fetch.py          # ~2.6 h at the required crawl delay
uv run parse.py
uv run enrich.py --sample 60 && uv run enrich.py
uv run embed.py
uv run cluster.py --explore
uv run cluster.py --min-cluster-size 8
uv run render.py
uv run social_preview.py
```

Intermediate data lives in `data/` and is not committed. Embeddings and LLM
responses are cached on disk, so re-running a stage does not repeat paid calls.

## License

The code is released under the [MIT License](LICENSE). The encyclopedia entries
themselves belong to the SEP and their authors.
