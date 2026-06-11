"""Prompt-format variants for the automated reviewer.

This module supports the *prompt-sensitivity* experiment: we hold the task,
the data, and the output schema fixed, and vary only how the reviewer
instructions are *formatted / framed*, then measure how each model's
accept/reject calibration responds.

Every variant returns a `PromptSpec`:

    spec.system        -- role statement (Anthropic system / OpenAI system head)
    spec.guidelines    -- the long, per-paper-invariant block (cached prefix)
    spec.build_user(title, abstract, full_text) -> str   -- per-paper content

`spec.system + spec.guidelines` is identical for every paper, so it caches
exactly like the original design (see providers.py). Only `build_user`
varies between papers.

The default variant reproduces the original `prompts.py` byte-for-byte, so
runs without `--prompt-style` are unchanged and remain comparable to the
existing results.

Variants are deliberately tied to the project's headline finding (most models
collapse to constant-Accept): several reframe the base rate or the reasoning
procedure to test whether *format alone* can break that collapse, independent
of model capability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import prompts

UserBuilder = Callable[[str, str, str], str]


@dataclass
class PromptSpec:
    name: str
    system: str
    guidelines: str
    build_user: UserBuilder


# Strict JSON schema shared by all variants so evaluate.py parses every run
# identically.
_SCHEMA_BLOCK = """\
{
  "summary": "<2-4 sentence neutral summary of what the paper does>",
  "strengths": ["<strength>", "..."],
  "weaknesses": ["<weakness>", "..."],
  "questions": ["<question for the authors>", "..."],
  "limitations": "<are limitations / ethical concerns adequately addressed?>",
  "soundness": <int 1-4>,
  "presentation": <int 1-4>,
  "contribution": <int 1-4>,
  "overall": <int 1-10>,
  "confidence": <int 1-5>,
  "decision": "Accept" | "Reject"
}"""


def _default() -> PromptSpec:
    # Identical to prompts.py (the existing baseline).
    return PromptSpec(
        name="default",
        system=prompts.REVIEWER_SYSTEM,
        guidelines=prompts.REVIEW_GUIDELINES,
        build_user=prompts.build_user_prompt,
    )


def _terse() -> PromptSpec:
    # Compress instructions to the minimum; same schema. Tests whether brevity
    # (less hand-holding) changes calibration.
    system = "You are a rigorous, calibrated ML conference reviewer (NeurIPS/ICLR)."
    guidelines = (
        "Review the paper on soundness, presentation, contribution. Score each "
        "1-4; overall 1-10; confidence 1-5. Accept iff overall >= 6. Top venues "
        "reject most submissions.\n\n"
        "Respond with ONLY this JSON object:\n" + _SCHEMA_BLOCK
    )
    return PromptSpec("terse", system, guidelines, prompts.build_user_prompt)


def _plain() -> PromptSpec:
    # Strip all markdown structure; same content as default, prose only. Tests
    # sensitivity to markup rather than wording.
    system = prompts.REVIEWER_SYSTEM
    guidelines = (
        "Follow the official NeurIPS reviewer guidelines and produce one "
        "structured review. Read the whole paper before scoring and be "
        "specific. Assess soundness (claims supported by correct theory and "
        "sufficient, well-designed experiments, with appropriate baselines, "
        "datasets, metrics and ablations), presentation (clarity, organisation, "
        "readable figures and tables, accurate citation of prior work), and "
        "contribution (novelty and significance relative to the literature, and "
        "honesty of scope). Also note acknowledged and unacknowledged "
        "limitations, reproducibility, and ethical concerns. Score soundness, "
        "presentation and contribution as integers from 1 (poor) to 4 "
        "(excellent). Score overall from 1 to 10, where 6 is marginally above "
        "the acceptance threshold and 5 is marginally below it. Score "
        "confidence from 1 to 5. Use the full range and do not cluster around "
        "the middle. Top-tier venues reject most submissions, so reject naive "
        "or underdeveloped ideas, missing or weak baselines, claims unsupported "
        "by experiments, methodological errors, and overstated contributions. "
        "Choose Accept only if overall is at least 6, otherwise Reject, and "
        "keep the decision consistent with the score. Respond with only a "
        "single JSON object using exactly these keys: " + _SCHEMA_BLOCK
    )
    return PromptSpec("plain", system, guidelines, prompts.build_user_prompt)


def _rubric_first() -> PromptSpec:
    # Put the rubric and the "venues reject most" prior BEFORE the role text.
    # Tests primacy / ordering effects.
    system = prompts.REVIEWER_SYSTEM
    guidelines = """\
# Scoring rubric (read this first)

Top-tier venues reject the majority of submissions. Calibrate accordingly and
use the FULL range -- do not cluster around the middle.

soundness, presentation, contribution -- integer 1-4 (4 excellent ... 1 poor).
overall -- integer 1-10:
  8 = top 50% / clear accept       7 = good paper / accept
  6 = marginally above threshold   5 = marginally below threshold
  4 = ok but not good enough       3 = clear reject ... 1 = trivial/wrong
confidence -- integer 1-5 (5 certain ... 1 educated guess).
decision -- "Accept" iff overall >= 6, else "Reject" (must match the score).

# What to assess

1. Soundness: claims supported by correct theory and/or sufficient, well-
   designed experiments; appropriate baselines, datasets, metrics, ablations.
2. Presentation: clarity, organisation, readable figures/tables, accurate
   citation of prior work.
3. Contribution: novelty and significance vs. the literature; honest scope.
Also note limitations, reproducibility, and ethical concerns.

# Output format

Respond with ONLY this JSON object -- no prose, no code fences:
""" + _SCHEMA_BLOCK
    return PromptSpec("rubric_first", system, guidelines,
                      prompts.build_user_prompt)


def _cot() -> PromptSpec:
    # Ask for explicit step-by-step reasoning in a dedicated field BEFORE the
    # scores (still strict JSON). Tests whether eliciting reasoning helps
    # non-reasoning models break the Accept-collapse.
    system = prompts.REVIEWER_SYSTEM
    schema = """\
{
  "reasoning": "<step-by-step analysis: weigh the strongest evidence FOR and AGAINST acceptance, then state which dominates and why>",
  "summary": "<2-4 sentence neutral summary>",
  "strengths": ["<strength>", "..."],
  "weaknesses": ["<weakness>", "..."],
  "questions": ["<question for the authors>", "..."],
  "limitations": "<are limitations / ethical concerns adequately addressed?>",
  "soundness": <int 1-4>,
  "presentation": <int 1-4>,
  "contribution": <int 1-4>,
  "overall": <int 1-10>,
  "confidence": <int 1-5>,
  "decision": "Accept" | "Reject"
}"""
    guidelines = """\
# Reviewer guidelines

Reason before you score. First, in the "reasoning" field, work through the
paper step by step: identify the central claim, the strongest supporting
evidence, and the most serious weakness or missing experiment. Explicitly
argue the case FOR rejection and the case FOR acceptance, then decide which is
stronger. Only after that reasoning, fill in the scores so they FOLLOW from it.

Assess soundness, presentation and contribution (each 1-4), overall (1-10,
where 6 is marginally above the acceptance threshold), and confidence (1-5).
Use the full range. Top venues reject most submissions: reject naive ideas,
weak or missing baselines, unsupported claims, and overstated contributions.
Accept iff overall >= 6; keep the decision consistent with the score.

# Output format

Respond with ONLY this JSON object -- no prose, no code fences:
""" + schema
    return PromptSpec("cot", system, guidelines, prompts.build_user_prompt)


def _strict_gatekeeper() -> PromptSpec:
    # Explicitly inject the real base rate and a Reject-by-default stance.
    # Directly probes whether stating the prior fixes the Accept-collapse that
    # capability alone does not.
    system = (
        "You are a selective Area Chair at a top ML venue. Roughly 70% of "
        "submissions are rejected. Your job is to protect the venue's bar: "
        "the DEFAULT outcome is Reject, and a paper must earn acceptance with "
        "clear, sufficient evidence. You are fair but you do not rubber-stamp."
    )
    guidelines = """\
# Reviewer guidelines

Start from the assumption that the paper will be rejected, and only move to
Accept if the evidence in the paper clearly justifies it. A paper is NOT
acceptable merely because it is competent, well written, or free of obvious
errors -- it must make a sufficiently novel and well-supported contribution to
displace the ~70% of submissions that are rejected.

Assess:
1. Soundness -- claims supported by correct theory and/or sufficient, well-
   designed experiments; appropriate baselines, metrics, ablations.
2. Presentation -- clarity, organisation, accurate citation.
3. Contribution -- novelty and significance vs. the literature; honest scope.

Score soundness/presentation/contribution 1-4, overall 1-10, confidence 1-5.
Use the FULL range; most papers should land at overall 3-5. Accept iff
overall >= 6; the decision must match the score.

# Output format

Respond with ONLY this JSON object -- no prose, no code fences:
""" + _SCHEMA_BLOCK
    return PromptSpec("strict_gatekeeper", system, guidelines,
                      prompts.build_user_prompt)


_REGISTRY = {
    "default": _default,
    "terse": _terse,
    "plain": _plain,
    "rubric_first": _rubric_first,
    "cot": _cot,
    "strict_gatekeeper": _strict_gatekeeper,
}

STYLES = tuple(_REGISTRY.keys())


def get_spec(style: str) -> PromptSpec:
    """Return the PromptSpec for a named style (raises on unknown style)."""
    try:
        return _REGISTRY[style]()
    except KeyError:
        raise ValueError(
            f"Unknown prompt style '{style}'. Choose one of: {', '.join(STYLES)}"
        )
