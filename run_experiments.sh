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
#   TIER=1 bash run_experiments.sh          # cost-optimized plan (see below)
#
# TIER mode (TIER=1) -- the cost-minimizing plan for the final paper:
#   * Reuses the existing default/fs0 baselines (already in results/ for all
#     models) -- those cells are SKIPPED, never regenerated.
#   * Cheap open models (llama-3.1-8b, llama-3.3-70b, qwen-2.5-72b,
#     gemini-2.0-flash, gpt-4o-mini, deepseek-chat): the full style x few-shot
#     grid (minus default/fs0) -- they cost cents.
#   * Expensive models (gpt-4o, deepseek-r1, claude-sonnet-4.6,
#     claude-opus-4.7): ONLY the hypothesis-critical conditions --
#     strict_gatekeeper and cot, at fs0 and fs2.
#   * qwen3-235b (reasoning, huge outputs, no price guardrail) is SKIPPED;
#     set TIER_QWEN3=1 to include it minimally (strict_gatekeeper,cot @ fs0).
#   Per-cell BUDGET is set PER MODEL (above its measured 60-paper pass cost) so
#   the guardrail never truncates a paper cell below n=60. Override all cells
#   with the BUDGET env var if you need to.
set -euo pipefail

PROVIDER=openrouter
RESULTS=results
BUDGET_OVERRIDE="${BUDGET:-}"  # if set, forces this per-cell budget everywhere
BUDGET="${BUDGET:-1.0}"        # per-cell USD guardrail (0 disables); pilot/full
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
  local model_id="$1" tag="$2" extra="$3" style="$4" fs="$5" cell_budget="$6"
  local name="${tag}__${style}__fs${fs}"
  local reviews="$RESULTS/reviews_${name}.json"
  local metrics="$RESULTS/metrics_${name}.json"
  echo ">>> $name  (budget \$$cell_budget)"
  python -m automated_reviewer.review \
    --provider "$PROVIDER" --model "$model_id" \
    --prompt-style "$style" --few-shot "$fs" \
    $extra $LIMIT_ARG --budget "$cell_budget" \
    --out "$reviews" 2> "logs/${name}.log"
  python -m automated_reviewer.evaluate \
    --reviews "$reviews" --out "$metrics" >> "logs/${name}.log" 2>&1
}

if [[ "${TIER:-0}" == "1" ]]; then
  # --- Cost-optimized plan for the final paper -------------------------------
  LIMIT_ARG=""                  # all 60 papers for every paper cell
  # Per-model budgets sit above each model's measured 60-paper pass cost so the
  # guardrail never truncates below n=60 (estimates are conservative / no-cache).
  # Cheap models (full style x few-shot grid, default/fs0 skipped):
  #   "model_id|tag|extra|budget"
  # Note: qwen-2.5-72b dropped (its 80k-char input exceeds the current 32k
  # OpenRouter provider context) and gemini-2.0-flash retired on OpenRouter ->
  # substituted with gemini-2.5-flash (whole Google row, baseline included).
  CHEAP=(
    "meta-llama/llama-3.1-8b-instruct|llama-3.1-8b||0.5"
    "meta-llama/llama-3.3-70b-instruct|llama-3.3-70b||1.0"
    "google/gemini-2.5-flash|gemini-2.5-flash||1.0"
    "openai/gpt-4o-mini|gpt-4o-mini||1.0"
    "deepseek/deepseek-chat|deepseek-chat||1.5"
  )
  CHEAP_STYLES=(default terse plain rubric_first cot strict_gatekeeper)
  CHEAP_FS=(0 2)
  # Expensive models (hypothesis-critical styles only). Collapse-focused scope:
  # gpt-4o (collapsed frontier) + deepseek-r1 (partial). The Anthropic frontier
  # is already non-collapsed, so it is opt-in via TIER_ANTHROPIC=1.
  EXPENSIVE=(
    "openai/gpt-4o|gpt-4o||5.0"
    "deepseek/deepseek-r1|deepseek-r1||1.5"
  )
  if [[ "${TIER_ANTHROPIC:-0}" == "1" ]]; then
    EXPENSIVE+=(
      "anthropic/claude-sonnet-4.6|claude-sonnet-4.6||8.0"
      "anthropic/claude-opus-4.7|claude-opus-4.7||16.0"
    )
  fi
  EXP_STYLES=(strict_gatekeeper cot)
  EXP_FS=(0 2)

  # Staging: TIER_STAGE = cheap | expensive | all (default all). Lets us run the
  # cheap grid, inspect it, then come back for the expensive subset.
  STAGE="${TIER_STAGE:-all}"

  # bash 3.2 (macOS default) has no namerefs, so loops are inlined per group.
  if [[ "$STAGE" == "cheap" || "$STAGE" == "all" ]]; then
  for entry in "${CHEAP[@]}"; do
    IFS="|" read -r model_id tag extra budget <<< "$entry"
    # TIER_ONLY=<tag> restricts this run to one model (for parallel launches).
    [[ -n "${TIER_ONLY:-}" && "$tag" != "${TIER_ONLY}" ]] && continue
    [[ -n "$BUDGET_OVERRIDE" ]] && budget="$BUDGET_OVERRIDE"
    for style in "${CHEAP_STYLES[@]}"; do
      for fs in "${CHEAP_FS[@]}"; do
        [[ "$style" == "default" && "$fs" == "0" ]] && continue  # reuse baseline
        run_cell "$model_id" "$tag" "$extra" "$style" "$fs" "$budget"
      done
    done
  done
  fi

  if [[ "$STAGE" == "expensive" || "$STAGE" == "all" ]]; then
  for entry in "${EXPENSIVE[@]}"; do
    IFS="|" read -r model_id tag extra budget <<< "$entry"
    [[ -n "${TIER_ONLY:-}" && "$tag" != "${TIER_ONLY}" ]] && continue
    [[ -n "$BUDGET_OVERRIDE" ]] && budget="$BUDGET_OVERRIDE"
    for style in "${EXP_STYLES[@]}"; do
      for fs in "${EXP_FS[@]}"; do
        run_cell "$model_id" "$tag" "$extra" "$style" "$fs" "$budget"
      done
    done
  done

  if [[ "${TIER_QWEN3:-0}" == "1" ]]; then
    # Reasoning model: minimal, fs0 only, explicit price so the guardrail works.
    q_model="qwen/qwen3-235b-a22b-thinking-2507"; q_tag="qwen3-235b"
    q_extra="--price-in 0.30 --price-out 2.50"; q_budget="2.0"
    [[ -n "$BUDGET_OVERRIDE" ]] && q_budget="$BUDGET_OVERRIDE"
    for style in strict_gatekeeper cot; do
      run_cell "$q_model" "$q_tag" "$q_extra" "$style" "0" "$q_budget"
    done
  fi
  fi  # end expensive stage

  # Skip aggregation for single-model parallel launches (TIER_NOAGG=1); the
  # caller aggregates once after all parallel jobs finish to avoid races.
  if [[ -z "${TIER_NOAGG:-}" ]]; then
    echo "=== Aggregating ==="
    python -m automated_reviewer.aggregate --glob "$RESULTS/metrics_*__*.json" \
      --out "$RESULTS/aggregate.csv"
    echo "Done. See $RESULTS/aggregate.csv"
  fi
  exit 0
fi

for entry in "${MODELS[@]}"; do
  IFS="|" read -r model_id tag extra <<< "$entry"
  for style in "${STYLES[@]}"; do
    for fs in "${FEWSHOTS[@]}"; do
      run_cell "$model_id" "$tag" "$extra" "$style" "$fs" "$BUDGET"
    done
  done
done

echo "=== Aggregating ==="
python -m automated_reviewer.aggregate --glob "$RESULTS/metrics_*__*.json" \
  --out "$RESULTS/aggregate.csv"
echo "Done. See $RESULTS/aggregate.csv"
