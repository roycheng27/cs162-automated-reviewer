"""Approximate API pricing, used for cost estimates and the budget guardrail.

Prices are USD per 1,000,000 tokens (input, output) and are APPROXIMATE — they
drift and vary by host. They are only used to estimate spend and to stop a run
before it blows a budget, so the estimate is deliberately conservative (it
ignores prompt-cache discounts, which makes the real bill come in lower).

Override per run with --price-in / --price-out if you need exact numbers.
"""

from __future__ import annotations

# key substring -> (price_in, price_out) per 1M tokens.
# Matching is longest-substring-wins, case-insensitive, so "claude-haiku-4-5"
# matches "claude-haiku-4-5" and an OpenRouter id like
# "meta-llama/llama-3.3-70b-instruct" matches "llama-3.3-70b".
PRICES: dict[str, tuple[float, float]] = {
    # --- Anthropic ---
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-sonnet-4.6": (3.00, 15.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4.7": (5.00, 25.00),
    # --- OpenAI ---
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # --- open-source (typical OpenRouter / Together / Groq pricing) ---
    "llama-3.1-8b": (0.02, 0.05),
    "llama-3.2-3b": (0.01, 0.02),
    "llama-3.3-70b": (0.12, 0.30),
    "llama-3.1-70b": (0.12, 0.30),
    "llama-3.1-405b": (0.80, 0.80),
    "qwen-2.5-7b": (0.04, 0.10),
    "qwen-2.5-72b": (0.13, 0.40),
    "qwen3-32b": (0.10, 0.30),
    "qwq-32b": (0.15, 0.20),
    "mistral-7b": (0.03, 0.05),
    "mixtral-8x7b": (0.24, 0.24),
    "mistral-small": (0.10, 0.30),
    "gemma-2-9b": (0.03, 0.06),
    "deepseek-chat": (0.30, 1.00),
    "deepseek-v3": (0.30, 1.00),
    "deepseek-r1": (0.55, 2.20),
    # --- Google ---
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-flash": (0.10, 0.40),
    # --- OpenAI reasoning ---
    "o1-mini": (1.10, 4.40),
    "o3-mini": (1.10, 4.40),
}


def lookup(model: str) -> tuple[float, float] | None:
    """Return (price_in, price_out) per 1M tokens for a model id, or None."""
    m = model.lower()
    best_len = -1
    best_price = None
    for key, price in PRICES.items():
        if key in m and len(key) > best_len:
            best_len = len(key)
            best_price = price
    return best_price


def estimate_cost(model: str, input_tokens: int, output_tokens: int,
                  price_in: float | None = None,
                  price_out: float | None = None) -> float | None:
    """Estimate USD cost. Returns None if the model price is unknown.

    Conservative: charges the full input rate on every prompt token, ignoring
    any prompt-cache discount, so the real bill should be lower.
    """
    if price_in is None or price_out is None:
        p = lookup(model)
        if p is None:
            return None
        price_in, price_out = p
    return input_tokens / 1e6 * price_in + output_tokens / 1e6 * price_out
