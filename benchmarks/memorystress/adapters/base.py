"""MemorySystemAdapter — abstract interface for pluggable memory backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class IngestResult:
    """Result of ingesting a single session."""

    success: bool = True
    tokens_used: int = 0
    api_calls: int = 0
    error: str | None = None


@dataclass
class QueryResult:
    """Result of querying the memory system."""

    answer: str = ""
    retrieved_context: list[str] = field(default_factory=list)
    tokens_used: int = 0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0


@dataclass
class CostSnapshot:
    """Cumulative cost accounting snapshot."""

    total_tokens: int = 0
    total_api_calls: int = 0
    estimated_cost_usd: float = 0.0
    ingest_tokens: int = 0
    query_tokens: int = 0


class MemorySystemAdapter(ABC):
    """Abstract base for memory system adapters.

    Each adapter wraps a memory backend (OMEGA, raw context, etc.) and
    exposes a uniform interface for the benchmark harness.
    """

    @abstractmethod
    def ingest(self, session: dict) -> IngestResult:
        """Ingest a single session into the memory system.

        Args:
            session: A Session dict with keys: session_id, turns, simulated_date,
                     agent_id, phase, planted_fact_ids, etc.
        """
        ...

    @abstractmethod
    def query(self, question: dict) -> QueryResult:
        """Query the memory system with a benchmark question.

        Args:
            question: A Question dict with keys: question, agent_scope,
                      question_type, etc.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the adapter to a clean state (drop all stored data)."""
        ...

    @abstractmethod
    def get_cost(self) -> CostSnapshot:
        """Return cumulative cost accounting."""
        ...

    def set_agent_id(self, agent_id: str) -> None:
        """Set the current agent identity for multi-agent tests.

        Default implementation is a no-op; adapters that support
        multi-agent scoping should override.
        """
        pass
