"""Shared task text utilities for hooks and status bar."""
import logging
import os
import re

logger = logging.getLogger(__name__)

# System-generated prefixes that aren't real tasks
_SKIP_PREFIXES = (
    "MEMORY HANDOFF",
    "[Request interrupted",
    "Implement the following plan",
    "Execute this plan",
    "Follow this plan",
    "Here is the plan",
)

# Short non-task messages
_SKIP_EXACT = frozenset({
    "proceed", "continue", "yes", "no", "ok", "go ahead", "lgtm",
})


def clean_task_text(prompt: str) -> str:
    """Extract a clean, short task description from a user prompt.

    Strips XML tags, system prefixes, markdown headers.  Splits at the first
    sentence boundary (letter followed by `. `) and caps at 60 chars.

    Returns empty string if the prompt isn't a real task.
    """
    text = prompt.strip()
    if not text:
        return ""

    # Strip XML-like tags
    text = re.sub(r"<[a-zA-Z_-]+>[^<]*</[a-zA-Z_-]+>", "", text)
    text = re.sub(r"</?[a-zA-Z_-]+>", "", text)
    text = text.strip()

    # Skip system-generated / non-task messages
    for pfx in _SKIP_PREFIXES:
        if text.startswith(pfx):
            return ""
    if text.lower() in _SKIP_EXACT:
        return ""

    # Strip markdown heading markers
    text = re.sub(r"^#{1,4}\s*", "", text)
    # Strip "Resume:" prefix
    text = re.sub(r"^[Rr]esume\s*:\s*", "", text)

    # First line only (multi-line prompts: first line is usually the task)
    text = text.split("\n")[0].strip()

    # Sentence split: only at letter + punctuation + space (not "v2." or "87.")
    text = re.sub(r"([a-zA-Z])[.!?]\s.*", r"\1", text)
    text = text.strip()

    # Cap at 60 chars, cut at last space if mid-word
    if len(text) > 60:
        text = text[:60]
        if " " in text:
            text = text[:text.rfind(" ")]

    return text if len(text) >= 8 else ""


def _basic_clean(prompt: str) -> str:
    """Strip XML tags and system prefixes, but don't truncate.

    Returns the full cleaned text for Haiku to summarize from.
    Returns empty string if the prompt isn't a real task.
    """
    text = prompt.strip()
    if not text:
        return ""

    # Strip XML-like tags
    text = re.sub(r"<[a-zA-Z_-]+>[^<]*</[a-zA-Z_-]+>", "", text)
    text = re.sub(r"</?[a-zA-Z_-]+>", "", text)
    text = text.strip()

    for pfx in _SKIP_PREFIXES:
        if text.startswith(pfx):
            return ""
    if text.lower() in _SKIP_EXACT:
        return ""

    text = re.sub(r"^#{1,4}\s*", "", text)
    text = re.sub(r"^[Rr]esume\s*:\s*", "", text)

    return text.strip()


def summarize_task_text(prompt: str) -> str:
    """Summarize a user prompt into a concise 3-8 word task title using Haiku.

    Falls back to clean_task_text() if the API call fails or the key is missing.
    """
    full_text = _basic_clean(prompt)
    if not full_text or len(full_text) < 8:
        return ""

    # Short prompts are already concise enough
    if len(full_text) <= 30:
        return clean_task_text(prompt)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return clean_task_text(prompt)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=2.0)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=30,
            cache_control={"type": "ephemeral"},
            messages=[{"role": "user", "content": full_text[:500]}],
            system=(
                "Summarize this developer task/question into a concise 3-8 word "
                "status bar title. Output ONLY the title, no quotes, no punctuation, "
                "lowercase. Focus on the ACTION and TARGET, not filler words. "
                "Examples: 'fix auth token refresh bug', 'add dark mode toggle', "
                "'refactor statusline task display', 'debug failing CI pipeline'."
            ),
        )
        summary = response.content[0].text.strip().rstrip(".")
        if 5 <= len(summary) <= 60:
            return summary
        return clean_task_text(prompt)
    except Exception as e:
        logger.debug("Haiku summarization failed: %s", e)
        return clean_task_text(prompt)
