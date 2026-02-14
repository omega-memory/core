"""Null adapter — always answers "I don't know." for grading sanity checks."""

from __future__ import annotations

from benchmarks.memorystress.adapters.base import (
    CostSnapshot,
    IngestResult,
    MemorySystemAdapter,
    QueryResult,
)


class NullAdapter(MemorySystemAdapter):
    """Baseline that never recalls anything. Should score ~0%."""

    def ingest(self, session: dict) -> IngestResult:
        return IngestResult(success=True)

    def query(self, question: dict) -> QueryResult:
        return QueryResult(answer="I don't know.")

    def reset(self) -> None:
        pass

    def get_cost(self) -> CostSnapshot:
        return CostSnapshot()
