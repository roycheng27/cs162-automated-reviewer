"""Provider-agnostic LLM backend for the automated reviewer.

This lets you run the reviewer across many models — open-source through
mid-tier — to compare them (the Fig. 1b experiment in the paper).

Backends:
  anthropic   -- Claude via the official Anthropic SDK, with prompt caching.
  openai      -- OpenAI models (api.openai.com).
  openrouter  -- One key, hundreds of models: open-source (Llama, Qwen,
                 Mistral, DeepSeek, Gemma) and mid-tier closed. Recommended
                 for a multi-model comparison.
  together / groq / deepinfra -- other open-source model hosts.

All of openai/openrouter/together/groq/deepinfra speak the OpenAI-compatible
API, so they share one provider class that just varies base_url + API key.

API keys are read from the environment — see API_KEY_ENV below.
"""

from __future__ import annotations

import os

# provider name -> (base_url, api-key env var)
OPENAI_COMPATIBLE = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "deepinfra": ("https://api.deepinfra.com/v1/openai", "DEEPINFRA_API_KEY"),
}

# A cheap, sensible default model per provider (no expensive models).
DEFAULT_MODELS = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4o-mini",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "groq": "llama-3.3-70b-versatile",
    "deepinfra": "meta-llama/Llama-3.3-70B-Instruct",
}

# Providers whose models support OpenAI-style JSON mode by default.
_JSON_MODE_DEFAULT = {"openai": True, "openrouter": False, "together": False,
                      "groq": False, "deepinfra": False}


class Provider:
    """Common interface. `generate` returns (raw_text, usage_dict).

    usage_dict keys: input_tokens (total prompt tokens, cache included),
    output_tokens, cache_read_tokens, cache_write_tokens.
    """

    name = "base"
    model = ""

    def generate(self, system: str, guidelines: str, user_prompt: str):
        raise NotImplementedError


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, thinking: bool = True, max_tokens: int = 8000):
        import anthropic  # lazy import so other backends work without it

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.client = anthropic.Anthropic()
        self.model = model
        self.thinking = thinking
        self.max_tokens = max_tokens

    def generate(self, system: str, guidelines: str, user_prompt: str):
        # The guidelines block is identical for every paper, so cache_control
        # marks it as a reusable prefix — after request #1 it is served from
        # cache at ~10% of input cost. The per-paper user prompt changes every
        # request and must come after the cached prefix.
        system_blocks = [
            {"type": "text", "text": system},
            {"type": "text", "text": guidelines,
             "cache_control": {"type": "ephemeral"}},
        ]
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user_prompt}],
        )
        if self.thinking:
            kwargs["thinking"] = {"type": "adaptive"}

        resp = self.client.messages.create(**kwargs)

        text = "".join(b.text for b in resp.content if b.type == "text")
        cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
        usage = {
            # input_tokens = total prompt size (cached + uncached)
            "input_tokens": resp.usage.input_tokens + cache_read + cache_write,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
        }
        return text, usage


class OpenAICompatibleProvider(Provider):
    """Backend for any OpenAI-compatible endpoint (OpenAI, OpenRouter, ...)."""

    def __init__(self, name: str, model: str, base_url: str, api_key_env: str,
                 json_mode: bool, max_tokens: int = 8000):
        import openai  # lazy import

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"{api_key_env} is not set")
        self.client = openai.OpenAI(base_url=base_url, api_key=key)
        self.name = name
        self.model = model
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        # o-series reasoning models on the official OpenAI endpoint use a
        # different token parameter and reject `temperature`.
        self._is_openai = "api.openai.com" in base_url
        self._is_reasoning = self._is_openai and model.startswith(("o1", "o3", "o4"))

    def _call(self, messages: list[dict[str, str]], json_mode: bool):
        kwargs: dict[str, object] = {"model": self.model, "messages": messages}
        if self._is_reasoning:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens
            kwargs["temperature"] = 0
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return self.client.chat.completions.create(**kwargs)

    def generate(self, system: str, guidelines: str, user_prompt: str):
        # Stable guidelines first -> OpenAI-compatible hosts auto-cache long
        # repeated prefixes; nothing to configure.
        messages = [
            {"role": "system", "content": system + "\n\n" + guidelines},
            {"role": "user", "content": user_prompt},
        ]
        try:
            resp = self._call(messages, self.json_mode)
        except Exception:
            if not self.json_mode:
                raise
            # Some open-source models reject JSON mode — fall back to plain
            # generation; the response parser is robust to extra prose.
            resp = self._call(messages, False)

        if not getattr(resp, "choices", None):
            raise RuntimeError("provider returned an empty response (no choices) "
                               "-- possible content filter or upstream error")
        text = resp.choices[0].message.content or ""
        u = resp.usage
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        usage = {
            "input_tokens": u.prompt_tokens,   # already the full prompt size
            "output_tokens": u.completion_tokens,
            "cache_read_tokens": cached,
            "cache_write_tokens": 0,
        }
        return text, usage



class MockProvider(Provider):
    """Offline provider: deterministic fake reviews, no network or API key.

    For smoke-testing the pipeline (prompt assembly, parsing, evaluation,
    aggregation) without spending money. The score is derived from a hash of
    the user prompt so output is stable and varied across papers but carries
    no real signal.
    """

    name = "mock"

    def __init__(self, model: str = "mock"):
        self.model = model or "mock"

    def generate(self, system: str, guidelines: str, user_prompt: str):
        import hashlib, json as _json
        h = int(hashlib.sha256(user_prompt.encode()).hexdigest(), 16)
        overall = 3 + (h % 7)              # 3..9
        decision = "Accept" if overall >= 6 else "Reject"
        review = {
            "summary": "Mock review for offline pipeline testing.",
            "strengths": ["mock strength"],
            "weaknesses": ["mock weakness"],
            "questions": ["mock question"],
            "limitations": "mock limitations",
            "soundness": 1 + (h % 4),
            "presentation": 1 + ((h // 7) % 4),
            "contribution": 1 + ((h // 13) % 4),
            "overall": overall,
            "confidence": 1 + ((h // 17) % 5),
            "decision": decision,
        }
        text = _json.dumps(review)
        usage = {"input_tokens": len(user_prompt) // 4,
                 "output_tokens": len(text) // 4,
                 "cache_read_tokens": 0, "cache_write_tokens": 0}
        return text, usage


def get_provider(name: str, model: str | None = None, thinking: bool = True,
                 json_mode: bool | None = None,
                 base_url: str | None = None) -> Provider:
    """Build a provider.

    name      -- 'anthropic', 'openai', 'openrouter', 'together', 'groq',
                 'deepinfra'
    model     -- model id; defaults to a cheap model for the provider
    thinking  -- Anthropic adaptive thinking on/off
    json_mode -- force OpenAI JSON mode; default depends on the provider
    base_url  -- override the endpoint (for any other OpenAI-compatible host)
    """
    name = name.lower()
    if name == "mock":
        return MockProvider(model=model or "mock")
    model = model or DEFAULT_MODELS.get(name)
    if model is None:
        raise ValueError(f"No default model for provider '{name}' — pass --model")

    if name == "anthropic":
        return AnthropicProvider(model=model, thinking=thinking)

    if name in OPENAI_COMPATIBLE:
        default_url, key_env = OPENAI_COMPATIBLE[name]
        jm = _JSON_MODE_DEFAULT[name] if json_mode is None else json_mode
        return OpenAICompatibleProvider(
            name=name, model=model, base_url=base_url or default_url,
            api_key_env=key_env, json_mode=jm)

    raise ValueError(
        f"Unknown provider '{name}'. Expected one of: anthropic, "
        f"{', '.join(OPENAI_COMPATIBLE)}")
