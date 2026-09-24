"""A Toponymy LLM wrapper that calls Claude through the Anthropic SDK.

Toponymy's own AnthropicNamer goes through LiteLLM and always sends `temperature`
with a small `max_tokens` (128 for a topic name). Claude Opus 5 rejects sampling
parameters, and its adaptive thinking needs headroom beyond the visible answer,
so this wrapper drops temperature and gives each call room to think.

Toponymy drives async wrappers with a fresh `asyncio.run()` per stage, so the
client and semaphore are rebuilt whenever the running event loop changes; both
bind to the loop they were first used on. Successful responses are cached on disk
by prompt, so re-running a fit never pays twice for the same prompt.
"""

import asyncio
import hashlib
import sqlite3

import anthropic
from toponymy.llm_wrappers import AsyncLLMWrapper

from common import DATA

CACHE_PATH = DATA / "llm_cache.sqlite"


class AsyncClaudeNamer(AsyncLLMWrapper):
    # Errors an identical retry can't fix; Toponymy's base class fails fast on these.
    FAIL_FAST_EXCEPTIONS = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.BadRequestError,
        anthropic.NotFoundError,
    )

    def __init__(
        self,
        model: str = "claude-opus-5",
        effort: str = "medium",
        max_concurrent_requests: int = 8,
        max_tokens: int = 16000,
        cache_path=CACHE_PATH,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.max_concurrent_requests = max_concurrent_requests
        self.extra_prompting = ""
        self.callback = None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "cached": 0}
        self._loop = None
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(cache_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, text TEXT)"
        )

    def _resources(self):
        loop = asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop = loop
            self._client = anthropic.AsyncAnthropic(max_retries=4)
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self._client, self._semaphore

    def _key(self, system: str | None, user: str) -> str:
        raw = "\x00".join([self.model, self.effort, system or "", user])
        return hashlib.sha256(raw.encode()).hexdigest()

    async def _create(self, system: str | None, user: str) -> str:
        key = self._key(system, user)
        row = self.db.execute("SELECT text FROM llm WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.usage["cached"] += 1
            return row[0]

        client, semaphore = self._resources()
        kwargs = {"system": system} if system else {}
        async with semaphore:
            msg = await client.beta.messages.create(
                model=self.model,
                # Toponymy's per-call max_tokens sizes only the visible answer; it
                # would truncate thinking, so the wrapper sets its own ceiling.
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                messages=[{"role": "user", "content": user}],
                **kwargs,
            )
        self.usage["calls"] += 1
        self.usage["input_tokens"] += msg.usage.input_tokens
        self.usage["output_tokens"] += msg.usage.output_tokens
        if msg.stop_reason == "refusal":
            raise RuntimeError(f"refusal: {msg.stop_details}")
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if text is None:
            raise RuntimeError(f"no text block (stop_reason={msg.stop_reason})")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO llm VALUES (?, ?)", (key, text))
        return text

    async def _call_single_llm(self, prompt, temperature, max_tokens) -> str:
        return await self._create(None, prompt["combined"])

    async def _call_single_llm_with_system(
        self, prompt, temperature, max_tokens
    ) -> str:
        return await self._create(prompt["system"], prompt["user"])
