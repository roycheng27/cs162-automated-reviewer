#!/usr/bin/env bash
# Experiment grid for the two extensions:
#   (1) few-shot in-context demonstrations
#   (2) prompt-format / framing variants
#
# Holds data + schema fixed; varies model x prompt-style x few-shot, then
# evaluates and aggregates every cell into one comparison table.
#
# Requires OPENROUTER_API_KEY in your environment / .env.
#
# Cost: a full 60-paper pass is ~$0.15 on cheap models (README §5), but frontier
# and reasoning models cost substantially more and reasoning models emit very
# large outputs. The FULL grid is 11 models x 6 styles x 2 few-shot = 132 cells,
# so gate spend with BUDGET (per-cell USD guardrail) and start with the pilot.
#
# Usage:
#   bash run_experiments.sh                 # pilot: 3 models, 8 papers/cell
#   FULL=1 bash run_experiments.sh          # full grid, all 11 models, 60 papers
#   FULL=1 BUDGET=3 bash run_experiments.sh # raise per-cell budget guardrail
set -euo pipefail

PROVIDER=openrouter
RESULTS=results
BUDGET="${BUDGET:-1.0}"        # per-cell USD guardrail (0 disables)
mkdir -p "$RESULTS" logs

# Model rows: "openrouter_model_id|tag|extra per-model args"
# The tag becomes part of the output filename; keep it stable across runs.
if [[ "${FULL:-0}" == "1" ]]; then
  LIMIT_ARG=""                  # all 60 papers
  MODELS=(
    "meta-llama/llama-3.1-8b-instruct|llama-3.1-8b|"
    "meta-llama/llama-3.3-70b-instruct|llama-3.3-70b|"
    "qwen/qwen-2.5-72b-instruct|qwen-2.5-72b|--max-chars 80000"
    "google/gemini-2.0-flash-001|gemini-2.0-flash|"
    "openai/gpt-4o-mini|gpt-4o-mini|"
    "deepseek/deepseek-chat|deepseek-chat|"
    "openai/gpt-4o|gpt-4o|"
    "deepseek/deepseek-r1|deepseek-r1|"
    "qwen/qwen3-235b-a22b-thinking-2507|qwen3-235b|"
    "anthropic/claude-sonnet-4.6|claude-sonnet-4.6|"
    "anthropic/claude-opus-4.7|claude-opus-4.7|"
  )
else
  LIMIT_ARG="--limit ${PILOT_LIMIT:-8}"   # quick, cheap pilot
  MODELS=(                                # weak -> mid -> frontier(Anthropic)
    "meta-llama/llama-3.1-8b-instruct|llama-3.1-8b|"
    "openai/gpt-4o-mini|gpt-4o-mini|"
    "anthropic/claude-sonnet-4.6|claude-sonnet-4.6|"
  )
fi

# Prompt-format variants (see automated_reviewer/prompt_variants.py).
STYLES=(default terse plain rubric_first cot strict_gatekeeper)

# Few-shot settings: 0 = off, 2 = two in-context exemplars.
FEWSHOTS=(0 2)

run_cell () {
  local model_id="$1" tag="$2" extra="$3" style="$4" fs="$5"
  local name="${tag}__${style}__fs${fs}"
  local reviews="$RESULTS/reviews_${name}.json"
  local metrics="$RESULTS/metrics_${name}.json"
  echo ">>> $name"
  python -m automated_reviewer.review \
    --provider "$PROVIDER" --model "$model_id" \
    --prompt-style "$style" --few-shot "$fs" \
    $extra $LIMIT_ARG --budget "$BUDGET" \
    --out "$reviews" 2> "logs/${name}.log"
  python -m automated_reviewer.evaluate \
    --reviews "$reviews" --out "$metrics" >> "logs/${name}.log" 2>&1
}

for entry in "${MODELS[@]}"; do
  IFS="|" read -r model_id tag extra <<< "$entry"
  for style in "${STYLES[@]}"; do
    for fs in "${FEWSHOTS[@]}"; do
      run_cell "$model_id" "$tag" "$extra" "$style" "$fs"
    done
  done
done

echo "=== Aggregating ==="
python -m automated_reviewer.aggregate --glob "$RESULTS/metrics_*__*.json" \
  --out "$RESULTS/aggregate.csv"
echo "Done. See $RESULTS/aggregate.csv"
