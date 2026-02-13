"""MemoryStress dataset schema — dataclasses for the benchmark format.

The top-level JSON file (memorystress_v1.json) contains all facts, sessions,
questions, and contradiction chains needed to run the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Fact:
    """A single ground-truth fact planted in a session."""

    fact_id: str  # "F001"
    content: str  # "User's favorite language is Rust"
    category: str  # preference|decision|technical_fact|personal_info|event|relationship
    phase_planted: int  # 1, 2, or 3
    session_planted: str  # "S003"
    simulated_date: str  # ISO date
    importance: int  # 1-5
    mention_count: int  # how many sessions reference it
    mention_sessions: list[str] = field(default_factory=list)
    contradiction_chain: str | None = None  # chain_id or None
    recoverable: bool = True  # should this be answerable after eviction pressure?
    tags: list[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy|medium|hard


@dataclass
class ContradictionChain:
    """A fact that evolves across sessions (updated, reverted, accumulated)."""

    chain_id: str  # "C007"
    original_fact_id: str
    versions: list[dict] = field(default_factory=list)
    # [{version, content, session_id, simulated_date, phase}]
    final_ground_truth: str = ""
    pattern: str = "fact_update"
    # fact_update|fact_update_revert|fact_accumulate|fact_partial_update


@dataclass
class Session:
    """A single conversation session to be ingested."""

    session_id: str  # "S003"
    session_index: int
    phase: int  # 1, 2, or 3
    simulated_date: str  # ISO datetime
    topic: str
    agent_id: str  # "agent_a" or "agent_b"
    turns: list[dict] = field(default_factory=list)  # [{role, content}]
    planted_fact_ids: list[str] = field(default_factory=list)
    noise_level: str = "low"  # low|medium|high
    token_count_estimate: int = 0


@dataclass
class Question:
    """A benchmark question targeting specific facts."""

    question_id: str  # "Q001"
    question: str
    target_fact_ids: list[str] = field(default_factory=list)
    target_chain_id: str | None = None
    answer: str = ""
    answer_detail: str = ""  # context for grader
    phase_asked: int = 1  # checkpoint where this is asked (1-4)
    target_fact_phase: int = 1  # when the fact was planted
    age_sessions: int = 0  # distance in sessions
    age_days: int = 0  # distance in simulated days
    question_type: str = "fact_recall"
    # fact_recall|preference_recall|contradiction_resolution|
    # temporal_ordering|cross_agent_recall|cold_start_recall|
    # single_mention_recall
    scoring_dimensions: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    agent_scope: str = "agent_a"  # which agent to query as
    cold_start: bool = False  # if True, run with fresh agent instance


@dataclass
class BenchmarkData:
    """Top-level container for the full benchmark dataset."""

    version: str = "1.0"
    phases: dict = field(default_factory=dict)
    # phase boundaries + date ranges
    stats: dict = field(default_factory=dict)
    # total counts
    facts: list[Fact] = field(default_factory=list)
    contradiction_chains: list[ContradictionChain] = field(default_factory=list)
    sessions: list[Session] = field(default_factory=list)
    questions: list[Question] = field(default_factory=list)
