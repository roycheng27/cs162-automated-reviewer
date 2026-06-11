"""Run the automated reviewer over a set of papers.

For each paper this prompts an LLM to act as a conference reviewer and return
structured scores (soundness / presentation / contribution / overall /
confidence) plus an Accept/Reject decision. With --n-reviews > 1 it produces
an ensemble of independent reviews and (optionally) an area-chair meta-review
that reconciles them -- the design used in the paper.

Run (one model):
    python -m automated_reviewer.review --provider openrouter \\
        --model meta-llama/llama-3.3-70b-instruct --out results/llama70b.json

Compare 5 models -- give each its own output file, then evaluate each:
    for M in ... ; do python -m automated_reviewer.review --provider ... ; done

Every run prints an estimated cost and respects --budget: it stops before a
paper if the projected spend would exceed the budget. Output is resumable --
papers already in the output file are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from . import pricing, prompts, prompt_variants, fewshot
from .providers import get_provider


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def parse_json(text: str) -> dict:
    """Pull a JSON object out of a model response, tolerating code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found in response")
    return json.loads(text[start:end + 1])


def _normalize(review: dict) -> dict:
    """Coerce scores to ints and keep the decision consistent with `overall`."""
    out = dict(review)
    for k in ("soundness", "presentation", "contribution", "overall", "confidence"):
        try:
            out[k] = int(round(float(out.get(k))))
        except (TypeError, ValueError):
            out[k] = None
    decision = str(out.get("decision", "")).strip().capitalize()
    if decision not in ("Accept", "Reject"):
        decision = "Accept" if (out.get("overall") or 0) >= 6 else "Reject"
    out["decision"] = decision
    return out


def _generate_review(provider, system, guidelines, user_prompt, retries: int = 1):
    """Call the provider and parse the JSON.

    Retries once on either an API error or a parse error. Any failure is
    returned as an error string rather than raised, so one bad paper (a 400,
    an empty response, a malformed JSON) never aborts the whole run.
    """
    last_err = None
    usage: dict = {}
    for _ in range(retries + 1):
        try:
            raw, usage = provider.generate(system, guidelines, user_prompt)
        except Exception as e:  # noqa: BLE001 - record and continue, never crash
            last_err = f"API error: {e}"
            continue
        try:
            return _normalize(parse_json(raw)), usage, None
        except (ValueError, json.JSONDecodeError) as e:
            last_err = str(e)
    return None, usage, str(last_err)


def _totals(results: dict) -> tuple[int, int]:
    """Sum input/output tokens recorded across all reviewed papers."""
    tin = tout = 0
    for rec in results.values():
        u = rec.get("usage", {})
        tin += u.get("input_tokens", 0)
        tout += u.get("output_tokens", 0)
    return tin, tout


def review_papers(papers_path: str, out_path: str, provider_name: str,
                   model: str | None, n_reviews: int, meta_review: bool,
                   limit: int | None, thinking: bool, max_chars: int | None,
                   json_mode: bool | None, base_url: str | None,
                   budget: float | None, price_in: float | None,
                   price_out: float | None, prompt_style: str = "default",
                   fewshot_k: int = 0,
                   fewshot_file: str = "data/fewshot_examples.json") -> None:
    with open(papers_path) as fh:
        papers = json.load(fh)
    if limit is not None:
        papers = papers[:limit]

    provider = get_provider(provider_name, model, thinking=thinking,
                            json_mode=json_mode, base_url=base_url)
    _log(f"Provider: {provider.name} / {provider.model} | "
         f"n_reviews={n_reviews} meta_review={meta_review}")

    # Build the prompt spec: a format variant, optionally with few-shot
    # demonstrations attached. Both stay per-paper-invariant (cacheable).
    spec = prompt_variants.get_spec(prompt_style)
    if fewshot_k and fewshot_k > 0:
        examples = fewshot.load_examples(fewshot_file)
        spec = fewshot.attach(spec, examples, k=fewshot_k)
    _log(f"Prompt: style={prompt_style} few_shot={fewshot_k} -> spec={spec.name}")
    if budget is not None:
        _log(f"Budget: ${budget:.2f} (run stops if projected spend exceeds it)")

    # Resume: load any reviews already written.
    results: dict = {}
    if os.path.exists(out_path):
        with open(out_path) as fh:
            results = json.load(fh)
        _log(f"Resuming -- {len(results)} papers already reviewed")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    def cost_so_far() -> float | None:
        tin, tout = _totals(results)
        return pricing.estimate_cost(provider.model, tin, tout,
                                     price_in, price_out)

    for i, paper in enumerate(papers, 1):
        pid = paper["id"]
        if pid in results:
            continue

        # Budget guardrail: stop before spending past the budget.
        if budget is not None:
            spent = cost_so_far()
            if spent is not None and spent >= budget:
                _log(f"\nStopping: estimated spend ${spent:.2f} reached "
                     f"budget ${budget:.2f}. {len(results)} papers reviewed.")
                break

        full_text = paper["full_text"]
        if max_chars and len(full_text) > max_chars:
            _log(f"  note: {pid} text {len(full_text)} chars -> truncated to {max_chars}")
            full_text = full_text[:max_chars]
        user_prompt = spec.build_user(
            paper["title"], paper["abstract"], full_text)

        reviews, errors = [], []
        agg = {"input_tokens": 0, "output_tokens": 0,
               "cache_read_tokens": 0, "cache_write_tokens": 0}

        for _ in range(n_reviews):
            review, usage, err = _generate_review(
                provider, spec.system, spec.guidelines, user_prompt)
            for k in agg:
                agg[k] += usage.get(k, 0)
            if review is None:
                errors.append(err)
            else:
                reviews.append(review)

        if not reviews:
            _log(f"  [{i}/{len(papers)}] {pid}: FAILED ({errors})")
            results[pid] = {"reviews": [], "final": None,
                            "errors": errors, "usage": agg}
            with open(out_path, "w") as fh:
                json.dump(results, fh, indent=2)
            continue

        if meta_review and len(reviews) > 1:
            meta_prompt = prompts.build_meta_prompt(reviews)
            final, usage, err = _generate_review(
                provider, prompts.META_SYSTEM, prompts.META_GUIDELINES,
                meta_prompt)
            for k in agg:
                agg[k] += usage.get(k, 0)
            if final is None:
                final = reviews[0]
        else:
            final = reviews[0]

        results[pid] = {
            "reviews": reviews,
            "final": final,
            "errors": errors,
            "usage": agg,
        }
        _log(f"  [{i}/{len(papers)}] {pid}: {final['decision']:7s} "
             f"overall={final['overall']} (truth={paper['decision']})")

        # Incremental save -> the run is resumable if interrupted.
        with open(out_path, "w") as fh:
            json.dump(results, fh, indent=2)

    # Final cost summary.
    tin, tout = _totals(results)
    est = pricing.estimate_cost(provider.model, tin, tout, price_in, price_out)
    _log(f"\nDone. {len(results)} papers in {out_path}")
    _log(f"Tokens: input={tin:,} output={tout:,}")
    if est is not None:
        _log(f"Estimated cost (conservative): ${est:.2f}")
    else:
        _log(f"Estimated cost: unknown model price for '{provider.model}' -- "
             f"pass --price-in / --price-out to estimate.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the automated reviewer.")
    ap.add_argument("--papers", default="data/papers.json",
                    help="Input papers JSON (default: data/papers.json)")
    ap.add_argument("--out", default="results/reviews.json",
                    help="Output reviews JSON (default: results/reviews.json)")
    ap.add_argument("--provider", default="openrouter",
                    choices=["anthropic", "openai", "openrouter", "together",
                             "groq", "deepinfra", "mock"],
                    help="LLM provider (default: openrouter)")
    ap.add_argument("--model", default=None,
                    help="Model id (default: a cheap model for the provider)")
    ap.add_argument("--base-url", default=None,
                    help="Override the API endpoint (any OpenAI-compatible host)")
    ap.add_argument("--n-reviews", type=int, default=1,
                    help="Independent reviews per paper (default: 1)")
    ap.add_argument("--meta-review", action="store_true",
                    help="Add an area-chair meta-review (needs --n-reviews > 1)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only review the first N papers (for a quick test)")
    ap.add_argument("--no-thinking", action="store_true",
                    help="Disable Claude adaptive thinking (Anthropic only)")
    ap.add_argument("--max-chars", type=int, default=None,
                    help="Truncate paper text to this many chars (default: full)")
    ap.add_argument("--json-mode", dest="json_mode", action="store_true",
                    default=None, help="Force OpenAI-style JSON mode on")
    ap.add_argument("--no-json-mode", dest="json_mode", action="store_false",
                    help="Force JSON mode off (rely on prompt + parser)")
    ap.add_argument("--prompt-style", default="default",
                    choices=list(prompt_variants.STYLES),
                    help="Reviewer prompt format variant (default: default)")
    ap.add_argument("--few-shot", dest="few_shot", type=int, default=0,
                    help="Number of in-context exemplars to prepend (0=off)")
    ap.add_argument("--few-shot-file", dest="few_shot_file",
                    default="data/fewshot_examples.json",
                    help="JSON file of exemplars (see fetch_examples.py)")
    ap.add_argument("--budget", type=float, default=4.0,
                    help="Stop before projected spend exceeds this USD amount "
                         "(default: 4.0; use 0 to disable)")
    ap.add_argument("--price-in", type=float, default=None,
                    help="Override input price (USD per 1M tokens)")
    ap.add_argument("--price-out", type=float, default=None,
                    help="Override output price (USD per 1M tokens)")
    args = ap.parse_args()

    review_papers(
        papers_path=args.papers,
        out_path=args.out,
        provider_name=args.provider,
        model=args.model,
        n_reviews=args.n_reviews,
        meta_review=args.meta_review,
        limit=args.limit,
        thinking=not args.no_thinking,
        max_chars=args.max_chars,
        json_mode=args.json_mode,
        base_url=args.base_url,
        budget=None if args.budget == 0 else args.budget,
        price_in=args.price_in,
        price_out=args.price_out,
        prompt_style=args.prompt_style,
        fewshot_k=args.few_shot,
        fewshot_file=args.few_shot_file,
    )


if __name__ == "__main__":
    main()
