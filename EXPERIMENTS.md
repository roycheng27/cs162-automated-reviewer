# Extension experiments: few-shot demonstrations and prompt-format sensitivity

This document describes two extensions to the Automated Reviewer and the exact
steps to reproduce them. Both extensions hold the data (the 60 balanced ICLR
2024 papers), the output schema, and the decision rule (`Accept` iff
`overall >= 6`) fixed, and vary only the *prompt* — so every result remains
directly comparable to the existing single-model runs in the README.

All new code preserves the default behaviour: a run with no new flags is
byte-for-byte the original prompt and reproduces the existing results.

## 1. Motivation

The main study's headline finding is that reviewer quality does **not** track
raw model capability: seven of eleven models (including GPT-4o) collapse to
accepting *every* paper. Both extensions ask whether a cheaper intervention
than model capability — namely *what we put in the prompt* — can move a model
off that Accept-collapse.

- **Experiment A (few-shot).** Prepend `k` worked `(paper -> calibrated
  review)` demonstrations in context. This is the in-context priming the
  original *AI Scientist* reviewer uses. No weights are updated.
- **Experiment B (prompt format).** Hold the instructions' content roughly
  fixed but vary their *formatting and framing*, and measure how each model's
  calibration responds.

## 2. New components

| File | Role |
|---|---|
| `automated_reviewer/prompt_variants.py` | Six named prompt styles; `default` reproduces the original prompt. |
| `automated_reviewer/fewshot.py` | Loads exemplars and attaches a demonstration block to the (cacheable) guidelines. |
| `automated_reviewer/fetch_examples.py` | Pulls real ICLR reviews as exemplars, excluding the evaluation set (no leakage). |
| `data/fewshot_examples.json` | Curated, leakage-free fallback exemplars (one Accept, one Reject). |
| `automated_reviewer/aggregate.py` | Collapses many `metrics_*.json` runs into one comparison table + CSV. |
| `run_experiments.sh` | Runs the model x style x few-shot grid, evaluates, and aggregates. |

New `review.py` flags: `--prompt-style {default,terse,plain,rubric_first,cot,strict_gatekeeper}`,
`--few-shot K`, `--few-shot-file PATH`. New provider `--provider mock` runs the
whole pipeline offline (no API key, deterministic fake reviews) for smoke tests.

## 3. Prompt-format variants (Experiment B)

| Style | What it changes | Hypothesis it probes |
|---|---|---|
| `default` | original prompt | baseline |
| `terse` | minimal instructions, same schema | does less hand-holding change calibration? |
| `plain` | same content, all markdown stripped | sensitivity to markup vs. wording |
| `rubric_first` | rubric + "venues reject most" placed first | primacy / ordering effects |
| `cot` | adds a `reasoning` field filled in before the scores | does eliciting reasoning break the collapse in non-reasoning models? |
| `strict_gatekeeper` | injects the ~70% reject base rate + Reject-by-default stance | does stating the prior fix the collapse that capability does not? |

## 4. How to run

### 4.0 Prerequisites
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then add OPENROUTER_API_KEY
```
Confirm `data/papers.json` (the 60-paper set) is present.

### 4.1 Offline smoke test (no API key, ~seconds)
Verifies prompt assembly, parsing, evaluation, and aggregation before spending:
```bash
python -m automated_reviewer.review --provider mock --prompt-style cot \
    --few-shot 2 --limit 12 --budget 0 --out results/reviews_smoke.json
python -m automated_reviewer.evaluate --reviews results/reviews_smoke.json \
    --out results/metrics_smoke.json
```

### 4.2 (Optional) Fetch real exemplars instead of the curated fallback
```bash
python -m automated_reviewer.fetch_examples --k 2 --balanced \
    --papers data/papers.json --out data/fewshot_examples.json
```
This excludes every id in `data/papers.json`, so exemplars never leak into
evaluation. If you skip this step, the shipped curated exemplars are used.

### 4.3 A single real cell
```bash
python -m automated_reviewer.review --provider openrouter \
    --model openai/gpt-4o-mini --prompt-style strict_gatekeeper --few-shot 0 \
    --budget 1.0 --out results/reviews_gpt-4o-mini__strict_gatekeeper__fs0.json
python -m automated_reviewer.evaluate \
    --reviews results/reviews_gpt-4o-mini__strict_gatekeeper__fs0.json \
    --out   results/metrics_gpt-4o-mini__strict_gatekeeper__fs0.json
```

### 4.4 The full grid
```bash
bash run_experiments.sh                 # pilot: 3 models, 8 papers/cell (cheap)
FULL=1 bash run_experiments.sh          # all 11 models x 6 styles x {0,2}-shot, 60 papers
FULL=1 BUDGET=3 bash run_experiments.sh # raise the per-cell USD guardrail
```
The full grid includes the frontier/Anthropic models (the project's full
8+ model set): llama-3.1-8b, llama-3.3-70b, qwen-2.5-72b, gemini-2.0-flash,
gpt-4o-mini, deepseek-chat, gpt-4o, deepseek-r1, qwen3-235b, claude-sonnet-4.6,
claude-opus-4.7. The pilot uses a weak/mid/frontier trio (llama-3.1-8b,
gpt-4o-mini, claude-sonnet-4.6) so you see the contrast cheaply first.
Then read the comparison table:
```bash
python -m automated_reviewer.aggregate --glob 'results/metrics_*__*.json' \
    --out results/aggregate.csv
```

## 5. Output naming convention

`aggregate.py` reads the condition straight off the filename:
```
results/metrics_<model>__<style>__fs<k>.json
e.g.  results/metrics_gpt-4o-mini__strict_gatekeeper__fs2.json
```
Keep the double-underscore separators so the model / style / few-shot columns
populate correctly.

## 6. Cost and budget

A full 60-paper pass on a cheap model is ~$0.15 (README §5). The full grid is
`models x styles x few_shot` = **11 x 6 x 2 = 132 cells**. Cheap models are a
few dollars total, but the **frontier (gpt-4o, Anthropic) and reasoning
(deepseek-r1, qwen3-235b) models cost meaningfully more** — reasoning models in
particular emit very large outputs. Gate every cell with the `BUDGET` env var
(default `1.0`, per-cell USD guardrail; `0` disables). Few-shot adds ~5k cached
prompt tokens per cell. Run the pilot first to confirm the effect before
committing to the full 132-cell grid, and consider running the expensive models
on a reduced style subset.

## 7. What to verify for a successful run

1. **Smoke test passes** (Section 4.1) before any paid run.
2. **`OPENROUTER_API_KEY` is set** (`echo $OPENROUTER_API_KEY`); the `default`
   style on `gpt-4o-mini` should reproduce the README's constant-Accept result
   (balanced accuracy 0.500) — if it does, the harness is faithful.
3. **No leakage**: if you fetched exemplars, confirm none of their ids appear
   in `data/papers.json` (fetch_examples enforces this automatically).
4. **Per-cell logs** in `logs/` show no repeated API errors or JSON parse
   failures (a few skips are normal; see README §5).
5. **`results/aggregate.csv`** has one row per cell with sensible `n`.

## 8. Reading the results

- **Few-shot effect**: for each (model, style), compare `fs0` vs `fs2`. A drop
  in FPR / rise in balanced accuracy means demonstrations broke the collapse.
- **Format effect**: within a model at `fs0`, compare styles. If
  `strict_gatekeeper` or `cot` lifts a collapsed model above 0.500 while
  `default` does not, format alone is sufficient — a result complementary to
  the capability story.
- Headline on **balanced accuracy** and **AUC**, not F1 (README §F9).
