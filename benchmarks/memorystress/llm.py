"""Shared LLM calling utilities for MemoryStress.

Extracted from scripts/longmemeval_official.py — multi-provider support
for OpenAI, Anthropic, Google, Groq, and Grok.
"""

from __future__ import annotations

import os

# Provider registry: model prefix → config for OpenAI-compatible or Anthropic
PROVIDER_REGISTRY = {
    "gpt-": {"base_url": None, "env_key": "OPENAI_API_KEY"},
    "o1": {"base_url": None, "env_key": "OPENAI_API_KEY"},
    "o3": {"base_url": None, "env_key": "OPENAI_API_KEY"},
    "o4": {"base_url": None, "env_key": "OPENAI_API_KEY"},
    "gemini-": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GOOGLE_API_KEY",
    },
    "grok-": {"base_url": "https://api.x.ai/v1", "env_key": "XAI_API_KEY"},
    "llama-": {"base_url": "https://api.groq.com/openai/v1", "env_key": "GROQ_API_KEY"},
    "meta-llama/": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
    },
    "qwen/": {"base_url": "https://api.groq.com/openai/v1", "env_key": "GROQ_API_KEY"},
    "claude-": {"provider": "anthropic", "env_key": "ANTHROPIC_API_KEY"},
}


def resolve_provider(model: str) -> dict:
    """Look up provider config from model name prefix."""
    for prefix, config in PROVIDER_REGISTRY.items():
        if model.startswith(prefix):
            return config
    raise ValueError(
        f"Unknown model '{model}'. Supported prefixes: "
        + ", ".join(PROVIDER_REGISTRY.keys())
    )


def call_llm(
    messages: list[dict],
    model: str,
    max_tokens: int = 256,
    temperature: float = 0,
    api_key: str | None = None,
) -> str:
    """Call an LLM via the appropriate provider, return response text."""
    config = resolve_provider(model)
    key = api_key or os.environ.get(config["env_key"])
    if not key:
        raise ValueError(
            f"No API key for model '{model}'. "
            f"Set ${config['env_key']} or pass --api-key."
        )

    if config.get("provider") == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        system = None
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)
        kwargs = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text.strip()
    else:
        import openai

        client = openai.OpenAI(base_url=config.get("base_url"), api_key=key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
