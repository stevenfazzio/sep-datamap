"""Reduce embeddings to 2-d, then find and name regions with Toponymy.

Clustering runs on the same 2-d coordinates that get plotted, so named regions
match what a viewer sees. That makes UMAP's min_dist a clustering parameter:
it stays low so regions come out dense enough for density clustering.

    uv run cluster.py --explore            # free: layer counts at a few settings
    uv run cluster.py --min-cluster-size N # paid: fit + LLM naming at one setting

Writes data/labels.parquet with x, y and label_layer_0..k, where layer 0 is the
FINEST layer (Toponymy's in-memory order, which DataMapPlot also expects).
"""

import argparse
import json

import numpy as np
import pandas as pd
import umap
from toponymy import Toponymy, ToponymyClusterer

from common import (
    DATA,
    ENTRIES_PARQUET,
    write_bytes_atomic,
    write_npz_atomic,
    write_parquet_safely,
)
from embed import EMBEDDINGS, document_text
from embedder import QwenEndpointEmbedder
from namer import AsyncClaudeNamer

COORDS = DATA / "umap_coords.npz"
LABELS_PARQUET = DATA / "labels.parquet"
TOPIC_NAMES_JSON = DATA / "topic_names.json"
UMAP_PARAMS = dict(
    n_neighbors=15, min_dist=0.05, metric="cosine", n_components=2, random_state=42
)


def load_embeddings(entries: pd.DataFrame) -> np.ndarray:
    data = np.load(EMBEDDINGS)
    assert list(data["slugs"]) == list(entries.slug), (
        "embeddings out of sync with entries"
    )
    return data["vectors"]


def get_coords(slugs: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    params_key = json.dumps(UMAP_PARAMS, sort_keys=True)
    if COORDS.exists():
        cached = np.load(COORDS)
        if list(cached["slugs"]) == list(slugs) and str(cached["params"]) == params_key:
            return cached["coords"]
    print(f"running UMAP {UMAP_PARAMS}")
    coords = umap.UMAP(**UMAP_PARAMS).fit_transform(vectors).astype(np.float32)
    write_npz_atomic(COORDS, slugs=slugs, coords=coords, params=params_key)
    return coords


def make_clusterer(min_cluster_size: int) -> ToponymyClusterer:
    return ToponymyClusterer(min_clusters=4, base_min_cluster_size=min_cluster_size)


def explore(coords: np.ndarray, vectors: np.ndarray, sizes: list[int]) -> None:
    for size in sizes:
        clusterer = make_clusterer(size)
        clusterer.fit(clusterable_vectors=coords, embedding_vectors=vectors)
        counts = []
        for layer in clusterer.cluster_layers_:  # finest first
            labels = layer.cluster_labels
            counts.append(
                f"{labels.max() + 1} ({(labels == -1).mean():.0%} unlabelled)"
            )
        print(
            f"base_min_cluster_size={size}: {len(counts)} layers, finest→coarsest: "
            + " | ".join(counts)
        )


def fit_and_name(
    entries, coords, vectors, min_cluster_size: int, detail: tuple[float, float]
) -> None:
    namer = AsyncClaudeNamer()
    embedder = QwenEndpointEmbedder()
    topic_model = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=make_clusterer(min_cluster_size),
        object_description="Stanford Encyclopedia of Philosophy entries",
        corpus_description="the Stanford Encyclopedia of Philosophy",
        # Spread evenly across layers and rounded onto Toponymy's seven name-length
        # tiers: with four layers, 0.4-0.8 asks for 4-8 words at the finest layer
        # and 1-4 at the coarsest (0.2-0.8 gave 6-12-word finest names).
        lowest_detail_level=detail[0],
        highest_detail_level=detail[1],
    )
    texts = [document_text(r) for r in entries.itertuples()]
    # Toponymy.fit takes (objects, high-D embeddings, low-D map): the reverse of
    # the clusterer's argument order, so pass by keyword.
    topic_model.fit(texts, embedding_vectors=vectors, clusterable_vectors=coords)

    names = topic_model.topic_names_
    assert len(names[0]) >= len(names[-1]), "expected layer 0 to be the finest"
    out = pd.DataFrame({"slug": entries.slug, "x": coords[:, 0], "y": coords[:, 1]})
    for i, vec in enumerate(topic_model.topic_name_vectors_):
        out[f"label_layer_{i}"] = vec
    write_parquet_safely(out, LABELS_PARQUET)
    write_bytes_atomic(json.dumps(names, indent=2).encode(), TOPIC_NAMES_JSON)

    for i, layer_names in enumerate(names):
        share = (out[f"label_layer_{i}"] == "Unlabelled").mean()
        print(f"layer {i}: {len(layer_names)} names, {share:.0%} unlabelled")
    u = namer.usage
    cost = (u["input_tokens"] * 5 + u["output_tokens"] * 25) / 1e6
    print(
        f"naming: {u['calls']} calls ({u['cached']} more from cache), "
        f"{u['input_tokens']} in / {u['output_tokens']} out tokens ≈ ${cost:.2f}"
    )
    print(f"wrote {LABELS_PARQUET} and {TOPIC_NAMES_JSON}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--sizes", type=int, nargs="+", default=[5, 8, 12, 20])
    parser.add_argument("--min-cluster-size", type=int, default=None)
    parser.add_argument("--detail", type=float, nargs=2, default=[0.4, 0.8])
    args = parser.parse_args()

    entries = pd.read_parquet(ENTRIES_PARQUET)
    vectors = load_embeddings(entries)
    coords = get_coords(entries.slug.to_numpy(dtype=str), vectors)

    if args.explore:
        explore(coords, vectors, args.sizes)
    elif args.min_cluster_size is not None:
        fit_and_name(
            entries, coords, vectors, args.min_cluster_size, tuple(args.detail)
        )
    else:
        parser.error("pass --explore or --min-cluster-size N")


if __name__ == "__main__":
    main()
