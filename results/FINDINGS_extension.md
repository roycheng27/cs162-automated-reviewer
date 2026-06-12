# Extension findings: can prompt format or few-shot break the Accept-collapse?

This summarizes the prompt-only extension experiments (few-shot demonstrations
and prompt-format variants) run over the cheap/mid-tier model set. It answers
the question: **does the Accept-collapse (most models accepting every paper)
yield to prompt engineering, or only to model capability?**

Data, schema, and decision rule (`Accept` iff `overall >= 6`) are held fixed;
only the prompt varies. Metrics are balanced accuracy and AUC with bootstrap
95% CIs (never F1, which is ~0.667 by construction on the balanced set).

## Scope actually run

Cheap/mid-tier tier, 5 models x 6 styles x {fs0, fs2} (default/fs0 reused from
the original baselines):

- llama-3.1-8b, llama-3.3-70b, gpt-4o-mini, deepseek-chat, gemini-2.5-flash.

**Model-set notes (reproducibility):**
- `gemini-2.0-flash` was **retired on OpenRouter** mid-study; the Google slot
  uses **`gemini-2.5-flash`** instead, with its own freshly-run baseline so all
  gemini cells are internally comparable. The original 2.0-flash baseline
  (bal.acc 0.517) is kept only as a historical reference.
- `qwen-2.5-72b` was **dropped**: its 80k-char inputs exceed the current
  OpenRouter provider's 32,768-token context window.
- Expensive tier (gpt-4o, deepseek-r1, and the Anthropic frontier) was **not
  run** — the cheap-tier result already answers the core question.

## Per-model results (baseline vs. the two hypothesis-critical framings)

`*` marks a condition whose 95% CI lower bound is strictly above 0.500.

| Model | baseline (default fs0) | best format condition | FPR shift | AUC(best) |
|---|---|---|---|---|
| llama-3.1-8b | 0.500 (FPR 1.00) | strict_gk fs0 -> 0.600 [0.50, 0.70] | 1.00->0.70 | 0.565 |
| gpt-4o-mini | 0.500 (FPR 1.00) | cot fs2 -> 0.550 [0.50, 0.61] | 1.00->0.90 | 0.647 |
| deepseek-chat | 0.500 (FPR 1.00) | strict_gk fs2 -> 0.567 [0.49, 0.64] | 1.00->0.83 | 0.721 |
| llama-3.3-70b | 0.500 (FPR 1.00) | none — all 11 conditions = 0.500 / FPR 1.00 | none | — |
| gemini-2.5-flash | 0.533 (FPR 0.93) | strict_gk fs2 -> 0.617 [0.53, 0.70] `*` | 0.93->0.73 | 0.739 |

gemini-2.5-flash is the only model with conditions whose 95% CI is entirely
above 0.500: `default fs2`, `cot fs0`, and `strict_gatekeeper fs2`. Its
`strict_gatekeeper fs0` cuts FPR to 0.54 (from 0.93).

Full per-condition table (all 6 conditions/model with CIs, AUC, FPR/FNR):
`results/aggregate.csv`.

## Findings

1. **Prompt format alone does NOT break the collapse for collapsed models —
   with one capability-gated exception.** Of the four models that fully collapse
   at baseline, none reaches a statistically reliable lift above chance. The
   `strict_gatekeeper` framing (stating the ~70% reject base rate, Reject-by-
   default) can lower FPR, but it trades accepts for false rejects, so balanced
   accuracy stays near 0.500 with CIs straddling it (e.g. gpt-4o-mini
   `strict_gk fs0` drops FPR to 0.40 but bal.acc is still 0.517 — it merely
   flips from accept-all toward reject-all).

2. **llama-3.3-70b is completely rigid**: a 70B model that returns *exactly*
   0.500 / FPR 1.00 on all 11 prompt variants. No reframing, CoT, or few-shot
   moves it at all.

3. **The effect is capability-dependent.** Only `gemini-2.5-flash` — the most
   capable model in this tier, already marginally off-collapse at baseline —
   responds reliably: `strict_gatekeeper`/`cot` lift it to bal.acc ~0.60-0.62
   (CIs above 0.5), AUC up to 0.739, and `strict_gatekeeper` roughly halves its
   FPR. Caveat: it was already slightly off-collapse, so part of the
   "responsiveness" may be that it was never fully collapsed.

4. **Lever ranking:** `strict_gatekeeper` (inject the base rate) > `cot` >
   few-shot. **Few-shot alone is the weakest** lever — `default fs0` -> `fs2`
   barely moves any model (<= +0.03 balanced accuracy).

## Bottom line for the paper

Prompt engineering does **not substitute for capability** in breaking the
Accept-collapse; it only takes hold once a model is capable enough to act on the
instruction. This complements the headline finding that reviewer quality tracks
capability — prompt format is a **complement, not a replacement**.
