"""DatasetLoader — parse, validate, and filter MemoryStress benchmark data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from benchmarks.memorystress.schema import (
    BenchmarkData,
    ContradictionChain,
    Fact,
    Question,
    Session,
)


class ValidationError(Exception):
    """Raised when dataset validation fails."""


class DatasetLoader:
    """Load and validate a MemoryStress JSON dataset."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: BenchmarkData | None = None

    def load(self) -> BenchmarkData:
        """Parse the JSON file into typed dataclasses."""
        raw = json.loads(self.path.read_text())

        facts = [Fact(**f) for f in raw.get("facts", [])]
        chains = [ContradictionChain(**c) for c in raw.get("contradiction_chains", [])]
        sessions = [Session(**s) for s in raw.get("sessions", [])]
        questions = [Question(**q) for q in raw.get("questions", [])]

        self._data = BenchmarkData(
            version=raw.get("version", "1.0"),
            phases=raw.get("phases", {}),
            stats=raw.get("stats", {}),
            facts=facts,
            contradiction_chains=chains,
            sessions=sessions,
            questions=questions,
        )
        return self._data

    @property
    def data(self) -> BenchmarkData:
        if self._data is None:
            raise RuntimeError("Call load() first")
        return self._data

    def validate(self) -> list[str]:
        """Run referential integrity and coverage checks. Returns list of warnings."""
        data = self.data
        warnings: list[str] = []

        fact_ids = {f.fact_id for f in data.facts}
        session_ids = {s.session_id for s in data.sessions}
        chain_ids = {c.chain_id for c in data.contradiction_chains}

        # Check session references in facts
        for f in data.facts:
            if f.session_planted not in session_ids:
                warnings.append(
                    f"Fact {f.fact_id}: session_planted '{f.session_planted}' not in sessions"
                )
            for sid in f.mention_sessions:
                if sid not in session_ids:
                    warnings.append(
                        f"Fact {f.fact_id}: mention_session '{sid}' not in sessions"
                    )
            if f.contradiction_chain and f.contradiction_chain not in chain_ids:
                warnings.append(
                    f"Fact {f.fact_id}: contradiction_chain '{f.contradiction_chain}' not found"
                )

        # Check fact references in sessions
        for s in data.sessions:
            for fid in s.planted_fact_ids:
                if fid not in fact_ids:
                    warnings.append(
                        f"Session {s.session_id}: planted_fact_id '{fid}' not in facts"
                    )

        # Check fact references in questions
        for q in data.questions:
            for fid in q.target_fact_ids:
                if fid not in fact_ids:
                    warnings.append(
                        f"Question {q.question_id}: target_fact_id '{fid}' not in facts"
                    )
            if q.target_chain_id and q.target_chain_id not in chain_ids:
                warnings.append(
                    f"Question {q.question_id}: target_chain_id '{q.target_chain_id}' not found"
                )

        # Check chain references
        for c in data.contradiction_chains:
            if c.original_fact_id not in fact_ids:
                warnings.append(
                    f"Chain {c.chain_id}: original_fact_id '{c.original_fact_id}' not in facts"
                )

        # Coverage: every fact should be targeted by at least one question
        targeted_facts = set()
        for q in data.questions:
            targeted_facts.update(q.target_fact_ids)
        uncovered = fact_ids - targeted_facts
        if uncovered:
            warnings.append(
                f"{len(uncovered)} facts have no targeting question: "
                f"{sorted(uncovered)[:5]}..."
            )

        # Difficulty distribution
        difficulties = {}
        for q in data.questions:
            difficulties[q.difficulty] = difficulties.get(q.difficulty, 0) + 1
        if len(difficulties) < 2:
            warnings.append(f"Low difficulty diversity: {difficulties}")

        return warnings

    def sessions_for_phase(self, phase: int) -> list[Session]:
        """Return sessions belonging to a specific phase, sorted by index."""
        return sorted(
            [s for s in self.data.sessions if s.phase == phase],
            key=lambda s: s.session_index,
        )

    def questions_for_phase(self, phase: int) -> list[Question]:
        """Return questions to be asked at a specific checkpoint phase."""
        return [q for q in self.data.questions if q.phase_asked == phase]

    def cold_start_questions(self) -> list[Question]:
        """Return questions marked for cold-start testing."""
        return [q for q in self.data.questions if q.cold_start]

    def to_json(self) -> str:
        """Serialize the dataset back to JSON."""
        return json.dumps(asdict(self.data), indent=2)
