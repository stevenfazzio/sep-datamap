"""Qwen3-Embedding-4B via a RunPod serverless (load-balancer) vLLM endpoint.

Implements Toponymy's TextEmbedderProtocol (`encode(texts, show_progress_bar)`), so
the same object embeds the documents and, later, Toponymy's keyphrases and topic
names. Every vector is cached on disk by text hash: re-running any stage never
re-embeds a string, and the endpoint only wakes for genuinely new text.

The endpoint scales to zero and a load balancer has no queue, so the first request
after idle fails until a worker is up; `wait_until_ready` absorbs that.
"""

import hashlib
import os
import random
import sqlite3
import time

import numpy as np
import requests
from tqdm import tqdm

from common import DATA

# The endpoint that built the published map (a RunPod load-balancer endpoint on
# vllm/vllm-openai:v0.28.0 with `--runner pooling --convert embed`) was deleted
# once the map was done. Anything already embedded comes from the disk cache;
# embedding new text needs a new endpoint and its ID here.
ENDPOINT_ID = "jxrvvepxi1lwkx"
BASE_URL = f"https://{ENDPOINT_ID}.api.runpod.ai"
HEALTH_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health"
MODEL = "Qwen/Qwen3-Embedding-4B"
CACHE_PATH = DATA / "embedding_cache.sqlite"
BATCH_SIZE = 64


class QwenEndpointEmbedder:
    def __init__(self, cache_path=CACHE_PATH, batch_size=BATCH_SIZE):
        self.batch_size = batch_size
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {os.environ['RUNPOD_API_KEY']}"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, model TEXT, vec BLOB)"
        )
        self._ready = False

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(f"{MODEL}\x00{text}".encode()).hexdigest()

    def wait_until_ready(self, max_wait_s: int = 1200) -> None:
        if self._ready:
            return
        start = time.time()
        while time.time() - start < max_wait_s:
            try:
                # A request to the LB is what triggers a scale-up from zero.
                r = self.session.get(f"{BASE_URL}/ping", timeout=30)
                if r.status_code == 200:
                    print(f"endpoint ready after {time.time() - start:.0f}s")
                    self._ready = True
                    return
                status = r.status_code
            except requests.RequestException as e:
                status = type(e).__name__
            try:
                workers = self.session.get(HEALTH_URL, timeout=30).json().get("workers")
            except (requests.RequestException, ValueError):
                workers = "?"
            print(
                f"  waiting for worker ({time.time() - start:.0f}s): ping={status} workers={workers}"
            )
            time.sleep(15)
        raise TimeoutError(f"endpoint not ready after {max_wait_s}s")

    def _embed_remote(self, texts: list[str], max_retries: int = 5) -> np.ndarray:
        for attempt in range(max_retries):
            try:
                r = self.session.post(
                    f"{BASE_URL}/v1/embeddings",
                    json={"model": MODEL, "input": texts, "encoding_format": "float"},
                    timeout=300,
                )
                # 400 here is the LB's "timed out waiting for worker" after a scale-down.
                if (
                    r.status_code in {400, 429, 500, 502, 503, 504}
                    and attempt < max_retries - 1
                ):
                    wait = min(2**attempt * 10, 120) + random.uniform(0, 5)
                    print(
                        f"  {r.status_code}: {r.text[:200]!r}; retrying in {wait:.0f}s"
                    )
                    self._ready = False
                    time.sleep(wait)
                    self.wait_until_ready()
                    continue
                r.raise_for_status()
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                return np.asarray([d["embedding"] for d in data], dtype=np.float32)
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = min(2**attempt * 10, 120)
                print(f"  attempt {attempt + 1}: {e!r}; retrying in {wait:.0f}s")
                time.sleep(wait)
        raise RuntimeError("exhausted retries")

    def encode(
        self, texts, show_progress_bar=None, verbose=None, **kwargs
    ) -> np.ndarray:
        texts = list(texts)
        keys = [self._key(t) for t in texts]
        cached = {}
        for i in range(0, len(keys), 900):  # stay under SQLite's variable limit
            chunk = keys[i : i + 900]
            rows = self.db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})",
                chunk,
            )
            cached.update({k: np.frombuffer(v, dtype=np.float32) for k, v in rows})

        # Dedupe so repeated strings (common among keyphrases) are embedded once.
        todo = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in cached))
        if todo:
            self.wait_until_ready()
            batches = range(0, len(todo), self.batch_size)
            for i in tqdm(
                batches, desc="embedding", disable=not (show_progress_bar or verbose)
            ):
                batch = todo[i : i + self.batch_size]
                vecs = self._embed_remote(batch)
                rows = [(self._key(t), MODEL, v.tobytes()) for t, v in zip(batch, vecs)]
                with self.db:  # committed per batch, so a crash loses at most one batch
                    self.db.executemany(
                        "INSERT OR REPLACE INTO emb VALUES (?, ?, ?)", rows
                    )
                cached.update({row[0]: vec for row, vec in zip(rows, vecs)})

        return np.vstack([cached[k] for k in keys])
