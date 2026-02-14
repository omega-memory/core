"""OMEGA adapter — uses SQLiteStore for memory storage and retrieval."""

from __future__ import annotations

import os
import shutil
import tempfile
import time

from benchmarks.memorystress.adapters.base import (
    CostSnapshot,
    IngestResult,
    MemorySystemAdapter,
    QueryResult,
)
from benchmarks.memorystress.llm import call_llm

# RAG prompt for answering questions from retrieved context
_RAG_PROMPT = """\
I will give you several notes from past conversations. \
Please answer the question based on the relevant notes. \
If the question cannot be answered based on the provided notes, say so.

Important:
- Notes are in chronological order (oldest first). Higher note numbers are more recent.
- When the same fact appears in multiple notes with different values, always use \
the value from the MOST RECENT note. Earlier values are SUPERSEDED.
- If the question asks "how many" or for a count, enumerate all items and state the total.
- Give a direct, concise answer.

Notes from past conversations:

{notes}

Question: {question}
Answer:"""


def _format_turns(turns: list[dict]) -> str:
    """Format session turns as text for ingestion."""
    lines = []
    for turn in turns:
        lines.append(f"{turn['role']}: {turn['content']}")
    return "\n".join(lines)


def _format_note(content: str, date_str: str, index: int) -> str:
    """Format a retrieved memory as a numbered note block."""
    return (
        f"[Note {index} | Date: {date_str}]\n"
        f"{content}\n"
        f"[End Note {index}]"
    )


class OmegaAdapter(MemorySystemAdapter):
    """Adapter wrapping OMEGA's SQLiteStore."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        retrieval_limit: int = 15,
    ):
        self.model = model
        self.api_key = api_key
        self.retrieval_limit = retrieval_limit
        self._agent_id: str = "agent_a"
        self._tmpdir: str | None = None
        self._store = None
        self._cost = CostSnapshot()
        self._init_store()

    def _init_store(self) -> None:
        """Create a fresh temp directory and SQLiteStore."""
        self._tmpdir = tempfile.mkdtemp(prefix="memorystress_omega_")
        os.environ["OMEGA_HOME"] = self._tmpdir

        from omega.sqlite_store import SQLiteStore

        db_path = os.path.join(self._tmpdir, "bench.db")
        self._store = SQLiteStore(db_path=db_path)

    def set_agent_id(self, agent_id: str) -> None:
        self._agent_id = agent_id

    def ingest(self, session: dict) -> IngestResult:
        content = _format_turns(session.get("turns", []))
        if not content.strip():
            return IngestResult(success=True)

        try:
            self._store.store(
                content=content,
                session_id=session.get("session_id", ""),
                metadata={
                    "event_type": "session_summary",
                    "referenced_date": session.get("simulated_date", ""),
                    "priority": 3,
                    "agent_type": session.get("agent_id", self._agent_id),
                    "phase": session.get("phase", 0),
                },
                skip_inference=True,
            )
            # Rough token estimate: ~4 chars per token
            tokens = len(content) // 4
            self._cost.ingest_tokens += tokens
            self._cost.total_tokens += tokens
            self._cost.total_api_calls += 1
            return IngestResult(success=True, tokens_used=tokens, api_calls=1)
        except Exception as e:
            return IngestResult(success=False, error=str(e))

    def query(self, question: dict) -> QueryResult:
        question_text = question.get("question", "")
        agent_scope = question.get("agent_scope", self._agent_id)

        t0 = time.monotonic()
        results = self._store.query(
            question_text,
            limit=self.retrieval_limit,
            agent_type=agent_scope,
            include_infrastructure=True,
        )
        retrieval_ms = (time.monotonic() - t0) * 1000

        # Format retrieved notes
        note_blocks = []
        context_texts = []
        for i, r in enumerate(results, 1):
            date_str = "Unknown"
            if r.metadata:
                date_str = r.metadata.get("referenced_date", "Unknown")
            note_blocks.append(_format_note(r.content, date_str, i))
            context_texts.append(r.content[:200])

        notes_str = "\n\n".join(note_blocks) if note_blocks else "(No relevant notes found)"
        prompt = _RAG_PROMPT.format(notes=notes_str, question=question_text)

        t1 = time.monotonic()
        try:
            answer = call_llm(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=512,
                api_key=self.api_key,
            )
        except Exception as e:
            answer = f"[ERROR: {e}]"
        generation_ms = (time.monotonic() - t1) * 1000

        tokens = len(prompt) // 4 + len(answer) // 4
        self._cost.query_tokens += tokens
        self._cost.total_tokens += tokens
        self._cost.total_api_calls += 1

        return QueryResult(
            answer=answer,
            retrieved_context=context_texts,
            tokens_used=tokens,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=generation_ms,
        )

    def reset(self) -> None:
        if self._store:
            self._store.close()
            self._store = None
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
        self._cost = CostSnapshot()
        self._init_store()

    def get_cost(self) -> CostSnapshot:
        return self._cost

    def close(self) -> None:
        """Clean up resources."""
        if self._store:
            self._store.close()
            self._store = None
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
