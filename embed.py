"""Embed each entry's title + lead section with Qwen3-Embedding-4B.

Writes data/embeddings.npz (slugs, vectors), row-aligned with entries.parquet.
Vectors come through embedder.py's disk cache, so re-running is free for any
entry whose text hasn't changed.

    uv run embed.py
"""

import numpy as np
import pandas as pd

from common import DATA, ENTRIES_PARQUET, write_npz_atomic
from embedder import QwenEndpointEmbedder

EMBEDDINGS = DATA / "embeddings.npz"


def document_text(row) -> str:
    # The title is short and strongly topical; it anchors leads that open obliquely.
    return f"{row.title}\n\n{row.preamble}"


def main():
    entries = pd.read_parquet(ENTRIES_PARQUET)
    empty = entries.preamble_words == 0
    if empty.any():
        # Title-only text still places the entry; flag it rather than dropping it.
        print(f"  {empty.sum()} entries have no lead; embedding title only")

    texts = [document_text(r) for r in entries.itertuples()]
    vectors = QwenEndpointEmbedder().encode(texts, show_progress_bar=True)
    assert vectors.shape[0] == len(entries), (vectors.shape, len(entries))
    assert np.isfinite(vectors).all()
    norms = np.linalg.norm(vectors, axis=1)
    print(
        f"embed: {len(entries)} entries → {vectors.shape}, norms {norms.min():.3f}–{norms.max():.3f}"
    )

    def check(loaded):
        assert loaded["vectors"].shape == vectors.shape
        assert list(loaded["slugs"]) == list(entries.slug)

    write_npz_atomic(
        EMBEDDINGS,
        check=check,
        slugs=entries.slug.to_numpy(dtype=str),
        vectors=vectors,
    )
    print(f"wrote {EMBEDDINGS}")


if __name__ == "__main__":
    main()
