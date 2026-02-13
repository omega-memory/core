#!/usr/bin/env python3
"""OMEGA UserPromptSubmit hook — Auto-capture decisions and lessons from user prompts.

Fires on every user prompt. Detects decision and lesson patterns and stores them
as 'decision' or 'lesson_learned' event type in OMEGA memory. Uses conservative
matching to avoid noise.
"""
import json
import re
import sys


# Decision indicators (case-insensitive patterns)
DECISION_PATTERNS = [
    r"\blet'?s?\s+(?:go\s+with|use|switch\s+to|stick\s+with|move\s+to)\b",
    r"\bi\s+(?:decided?|chose|picked|went\s+with|prefer)\b",
    r"\bwe\s+(?:should|will|are\s+going\s+to)\s+(?:use|go\s+with|switch|adopt|implement)\b",
    r"\b(?:decision|approach|strategy):\s*\S",
    r"\binstead\s+of\s+\S+[,\s]+(?:use|let'?s|we'?ll)\b",
    r"\bfrom\s+now\s+on\b",
    r"\bremember\s+(?:that|this)\b",
]

# Lesson indicators (case-insensitive patterns)
LESSON_PATTERNS = [
    r"\bi\s+learned\s+that\b",
    r"\bturns?\s+out\b",
    r"\bthe\s+trick\s+is\b",
    r"\bnote\s+to\s+self\b",
    r"\btil\b|\btoday\s+i\s+learned\b",
    r"\bthe\s+fix\s+was\b",
    r"\bthe\s+problem\s+was\b",
    r"\bdon'?t\s+forget\b",
    r"\bimportant:\s*\S",
    r"\bkey\s+(?:insight|takeaway|learning)\b",
    r"\bnever\s+(?:again|do|use)\b",
    r"\balways\s+(?:make\s+sure|remember|check)\b",
]

# Minimum prompt length to avoid matching on short commands
MIN_PROMPT_LENGTH = 20

# Maximum prompts to process per session (avoid runaway storage)
_captured_count = 0
MAX_CAPTURES_PER_SESSION = 20


def _summarize_content(prompt: str, max_len: int = 60) -> str:
    """Extract a concise summary from the prompt for the echo line."""
    # Strip common prefixes like "Decision: " or "Lesson: "
    text = re.sub(r"^(Decision|Lesson):\s*", "", prompt, flags=re.IGNORECASE).strip()
    # Take first sentence or first max_len chars
    first_sentence = re.split(r"[.!?\n]", text)[0].strip()
    if len(first_sentence) <= max_len:
        return first_sentence
    return first_sentence[:max_len].rsplit(" ", 1)[0] + "..."


def _echo_capture(result: str, event_type: str, prompt: str):
    """Print a 1-line capture confirmation visible to the user.

    Parses bridge.auto_capture() return value to distinguish:
    - New capture → [OMEGA] Captured: decision about X
    - Evolution   → [OMEGA] Memory evolved: added insight to existing memory
    - Dedup/Block → silent (no output)
    """
    if not result:
        return

    summary = _summarize_content(prompt)

    if "Memory Evolved" in result:
        # Extract evolution number from "Evolution #N"
        evo_match = re.search(r"Evolution #(\d+)", result)
        evo_num = evo_match.group(1) if evo_match else "?"
        print(f"[OMEGA] Memory evolved: {event_type} updated (evolution #{evo_num}) — {summary}")
    elif "Memory Captured" in result:
        print(f"[OMEGA] Captured: {event_type} — {summary}")
    # Dedup/Blocked → stay silent


def _detect_decision(prompt: str) -> bool:
    """Check if prompt contains a decision pattern."""
    if len(prompt) < MIN_PROMPT_LENGTH:
        return False
    prompt_lower = prompt.lower()
    return any(re.search(pat, prompt_lower) for pat in DECISION_PATTERNS)


def _detect_lesson(prompt: str) -> bool:
    """Check if prompt contains a lesson/insight pattern."""
    if len(prompt) < MIN_PROMPT_LENGTH:
        return False
    prompt_lower = prompt.lower()
    return any(re.search(pat, prompt_lower) for pat in LESSON_PATTERNS)


def main():
    global _captured_count
    if _captured_count >= MAX_CAPTURES_PER_SESSION:
        return

    # Read hook input from stdin
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        data = json.loads(raw)
    except (json.JSONDecodeError, Exception):
        return

    prompt = data.get("prompt", "")
    session_id = data.get("session_id", "")
    cwd = data.get("cwd", "")

    if not prompt:
        return

    # Decision takes priority if both match
    if _detect_decision(prompt):
        try:
            from omega.bridge import auto_capture
            result = auto_capture(
                content=f"Decision: {prompt[:500]}",
                event_type="decision",
                metadata={"source": "auto_capture_hook", "project": cwd},
                session_id=session_id,
                project=cwd,
            )
            _captured_count += 1
            _echo_capture(result, "decision", prompt)
        except ImportError:
            pass
        except Exception:
            pass
        return

    if _detect_lesson(prompt):
        # Lesson quality gate: min 60 chars, >= 8 words, substance validation
        if len(prompt) < 60 or len(prompt.split()) < 8:
            return
        _tech_signals = ["/", "`", "Error", "error", ".py", ".js", ".ts", "import ", "def ", "class "]
        if len(prompt) < 100 and not any(s in prompt for s in _tech_signals):
            return

        try:
            from omega.bridge import auto_capture
            result = auto_capture(
                content=f"Lesson: {prompt[:500]}",
                event_type="lesson_learned",
                metadata={"source": "auto_capture_hook", "project": cwd},
                session_id=session_id,
                project=cwd,
            )
            _captured_count += 1
            _echo_capture(result, "lesson", prompt)
        except ImportError:
            pass
        except Exception:
            pass


if __name__ == "__main__":
    main()
