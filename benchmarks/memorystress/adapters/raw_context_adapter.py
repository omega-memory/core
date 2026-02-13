"""Raw context adapter — stuffs all sessions into the LLM context window.

Baseline measuring what pure context-window recall achieves without
any memory system. Demonstrates the cost ceiling that Mastra-style
compression-only architectures face.
"""

from __future__ import annotations

import time

from benchmarks.memorystress.adapters.base import (
    CostSnapshot,
    IngestResult,
    MemorySystemAdapter,
    QueryResult,
)
from benchmarks.memorystress.llm import call_llm

# Max characters to stuff into context (~100k tokens at ~4 chars/token)
_MAX_CONTEXT_CHARS = 400_000

_RAG_PROMPT = """\
Below are conversation logs from past sessions, ordered chronologically. \
Answer the question based on these logs. If the answer cannot be determined, say so.

When the same fact appears in multiple sessions with different values, \
use the value from the MOST RECENT session.

Conversation logs:

{sessions}

Question: {question}
Answer:"""


def _format_turns(turns: list[dict]) -> str:
    lines = []
    for turn in turns:
        lines.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(lines)


class RawContextAdapter(MemorySystemAdapter):
    """Adapter that keeps all sessions in memory and stuffs them into context."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        max_context_chars: int = _MAX_CONTEXT_CHARS,
    ):
        self.model = model
        self.api_key = api_key
        self.max_context_chars = max_context_chars
        self._sessions: list[dict] = []
        self._cost = CostSnapshot()

    def ingest(self, session: dict) -> IngestResult:
        self._sessions.append(session)
        return IngestResult(success=True)

    def query(self, question: dict) -> QueryResult:
        question_text = question.get("question", "")

        # Build context from most recent sessions that fit in the window
        blocks = []
        total_chars = 0
        for session in reversed(self._sessions):
            text = _format_turns(session.get("turns", []))
            header = f"[Session {session.get('session_id', '?')} | Date: {session.get('simulated_date', '?')}]"
            block = f"{header}\n{text}\n"
            if total_chars + len(block) > self.max_context_chars:
                break
            blocks.append(block)
            total_chars += len(block)

        # Reverse to chronological order
        blocks.reverse()
        sessions_str = "\n".join(blocks) if blocks else "(No sessions available)"

        prompt = _RAG_PROMPT.format(sessions=sessions_str, question=question_text)

        t0 = time.monotonic()
        try:
            answer = call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=512,
                api_key=self.api_key,
            )
        except Exception as e:
            answer = f"[ERROR: {e}]"
        generation_ms = (time.monotonic() - t0) * 1000

        tokens = len(prompt) // 4 + len(answer) // 4
        self._cost.query_tokens += tokens
        self._cost.total_tokens += tokens
        self._cost.total_api_calls += 1

        return QueryResult(
            answer=answer,
            retrieved_context=[b[:200] for b in blocks[:5]],
            tokens_used=tokens,
            retrieval_latency_ms=0.0,
            generation_latency_ms=generation_ms,
        )

    def reset(self) -> None:
        self._sessions.clear()
        self._cost = CostSnapshot()

    def get_cost(self) -> CostSnapshot:
        return self._cost
