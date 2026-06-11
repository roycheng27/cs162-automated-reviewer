"""In-context few-shot demonstrations for the automated reviewer.

This supports the *few-shot* experiment: instead of changing the model, we
prepend K worked (paper -> review) examples to the reviewer's instructions so
the model can imitate calibrated scoring and the exact JSON format. No weights
are updated -- this is pure in-context learning, the same mechanism the
original AI Scientist reviewer uses.

The demonstration block is appended to the *guidelines* (the per-paper-
invariant prefix), so it is identical for every paper and therefore cached
exactly like the rest of the prompt. Only the per-paper user prompt changes.

Leakage control
---------------
Few-shot exemplars MUST NOT be papers in the evaluation set. `fetch_examples.py`
draws exemplars from a different pool and excludes every id in data/papers.json.
The shipped `data/fewshot_examples.json` is a small curated set of illustrative
reviews not tied to any evaluation paper, so it is leakage-free by construction.

Each example is a dict with:
    title, abstract, excerpt        -- the demonstration paper (excerpt is a
                                       short slice of full text, kept small to
                                       control token cost)
    review                          -- the gold review JSON (same schema the
                                       model must emit)
"""

from __future__ import annotations

import json
from dataclasses import replace

from .prompt_variants import PromptSpec

# How many characters of an exemplar paper to show. Exemplars teach format and
# calibration, not content, so a short excerpt keeps the prompt cheap.
DEFAULT_EXCERPT_CHARS = 1500


def load_examples(path: str) -> list[dict]:
    """Load an exemplar file (a JSON list of example dicts)."""
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of examples")
    return data


def _render_one(i: int, ex: dict, excerpt_chars: int) -> str:
    title = ex.get("title", "").strip()
    abstract = ex.get("abstract", "").strip()
    excerpt = (ex.get("excerpt") or ex.get("full_text") or "").strip()
    if excerpt_chars and len(excerpt) > excerpt_chars:
        excerpt = excerpt[:excerpt_chars] + " [...]"
    review = ex.get("review", {})
    block = [f"## Example {i}", "### Paper", f"Title: {title}"]
    if abstract:
        block.append(f"Abstract: {abstract}")
    if excerpt:
        block.append(f"Excerpt: {excerpt}")
    block.append("### Calibrated reviewer JSON")
    block.append(json.dumps(review, indent=2))
    return "\n".join(block)


def render_block(examples: list[dict], k: int | None = None,
                 excerpt_chars: int = DEFAULT_EXCERPT_CHARS) -> str:
    """Render up to k examples into a demonstration block of text."""
    if k is not None:
        examples = examples[:k]
    if not examples:
        return ""
    parts = [
        "# Worked examples",
        "Below are example reviews that demonstrate the expected calibration "
        "and the exact JSON format. Note how scores use the full range and how "
        "weak papers receive overall < 6. Study them, then review the NEW "
        "paper the same way.",
        "",
    ]
    parts.extend(_render_one(i, ex, excerpt_chars)
                 for i, ex in enumerate(examples, 1))
    parts.append("\n# Now review the new paper")
    return "\n\n".join(parts)


def attach(spec: PromptSpec, examples: list[dict], k: int | None = None,
           excerpt_chars: int = DEFAULT_EXCERPT_CHARS) -> PromptSpec:
    """Return a copy of `spec` with a few-shot block appended to guidelines.

    The block goes at the END of the guidelines so the original instructions +
    output schema are read first and the examples immediately precede the new
    paper. The combined system+guidelines stays per-paper-invariant (cacheable).
    """
    block = render_block(examples, k=k, excerpt_chars=excerpt_chars)
    if not block:
        return spec
    new_guidelines = spec.guidelines.rstrip() + "\n\n" + block
    used = len(examples if k is None else examples[:k])
    return replace(spec, name=f"{spec.name}+fewshot{used}",
                   guidelines=new_guidelines)
