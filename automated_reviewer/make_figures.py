"""Generate result figures from the reviewer runs.

Produces three publication-style figures in figures/ and prints a combined
metrics table:

  fig_accuracy.png -- balanced accuracy + AUC per model (the capability gradient)
  fig_scores.png   -- overall-score distribution per model, split by true label
  fig_roc.png      -- ROC curve per model

Run (after the reviewer + evaluate steps):
    python -m automated_reviewer.make_figures
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt  # noqa: E402

from . import metrics  # noqa: E402

# Models in weak -> strong order. Skipped automatically if a file is missing.
MODELS = [
    ("llama-3.1-8b",     "results/reviews_meta-llama_llama-3.1-8b-instruct.json"),
    ("llama-3.3-70b",    "results/reviews_llama-3.3-70b.json"),
    ("qwen-2.5-72b",     "results/reviews_qwen_qwen-2.5-72b-instruct.json"),
    ("gemini-2.0-flash", "results/reviews_google_gemini-2.0-flash.json"),
    ("gpt-4o-mini",      "results/reviews_openai_gpt-4o-mini.json"),
    ("deepseek-chat",    "results/reviews_deepseek_deepseek-chat.json"),
    ("gpt-4o",           "results/reviews_openai_gpt-4o.json"),
    ("deepseek-r1",      "results/reviews_deepseek_deepseek-r1.json"),
    ("qwen3-235b*",       "results/reviews_qwen_qwen3-235b-thinking.json"),
    ("claude-opus-4.7",   "results/reviews_anthropic_claude-opus-4.7.json"),
    ("claude-sonnet-4.6", "results/reviews_anthropic_claude-sonnet-4.6.json"),
]

FIG_DIR = "figures"


def load(papers_path: str = "data/papers.json") -> list[dict]:
    """Join each model's reviews with ground truth -> y_true / y_pred / scores."""
    papers = {p["id"]: p for p in json.load(open(papers_path))}
    data = []
    for label, path in MODELS:
        if not os.path.exists(path):
            print(f"  (skip {label}: {path} not found)", file=sys.stderr)
            continue
        reviews = json.load(open(path))
        yt, yp, sc = [], [], []
        for pid, rec in reviews.items():
            final = rec.get("final")
            if pid not in papers or final is None:
                continue
            yt.append(1 if papers[pid]["decision"] == "Accept" else 0)
            yp.append(1 if final["decision"] == "Accept" else 0)
            ov = final.get("overall")
            sc.append(float(ov) if ov is not None else float(yp[-1]))
        if yt:
            data.append({"label": label, "y_true": yt, "y_pred": yp, "scores": sc})
    return data


def roc_points(y_true: list[int], scores: list[float]) -> list[tuple[float, float]]:
    """ROC curve points (fpr, tpr) by sweeping the score threshold."""
    pos = sum(y_true)
    neg = len(y_true) - pos
    pts = [(0.0, 0.0)]
    for t in sorted(set(scores), reverse=True):
        tp = sum(1 for yt, s in zip(y_true, scores) if yt == 1 and s >= t)
        fp = sum(1 for yt, s in zip(y_true, scores) if yt == 0 and s >= t)
        pts.append((fp / neg if neg else 0.0, tp / pos if pos else 0.0))
    pts.append((1.0, 1.0))
    return pts


def fig_accuracy(data: list[dict]) -> None:
    labels = [d["label"] for d in data]
    ba = [metrics.balanced_accuracy(d["y_true"], d["y_pred"]) for d in data]
    auc = [metrics.auc(d["y_true"], d["scores"]) for d in data]

    x = range(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar([i - w / 2 for i in x], ba, w, label="Balanced accuracy", color="#4C72B0")
    ax.bar([i + w / 2 for i in x], auc, w, label="AUC", color="#DD8452")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.text(len(labels) - 0.5, 0.515, "chance (0.5)", color="gray", fontsize=8,
            ha="right")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Reviewer quality vs. model capability")
    ax.legend()
    for i, (b, a) in enumerate(zip(ba, auc)):
        ax.text(i - w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_accuracy.png", dpi=150)
    plt.close(fig)


def fig_scores(data: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, d in enumerate(data):
        acc = [s for yt, s in zip(d["y_true"], d["scores"]) if yt == 1]
        rej = [s for yt, s in zip(d["y_true"], d["scores"]) if yt == 0]
        ax.boxplot([acc], positions=[i - 0.2], widths=0.32,
                   patch_artist=True, boxprops=dict(facecolor="#55A868"),
                   medianprops=dict(color="black"))
        ax.boxplot([rej], positions=[i + 0.2], widths=0.32,
                   patch_artist=True, boxprops=dict(facecolor="#C44E52"),
                   medianprops=dict(color="black"))
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d["label"] for d in data], rotation=20, ha="right")
    ax.set_ylabel("Reviewer overall score (1-10)")
    ax.set_title("Score distribution by true decision "
                 "(green = truly Accepted, red = truly Rejected)")
    ax.axhline(6, ls="--", c="gray", lw=1)
    ax.text(len(data) - 0.5, 6.1, "decision threshold", color="gray",
            fontsize=8, ha="right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_scores.png", dpi=150)
    plt.close(fig)


def fig_roc(data: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for d in data:
        pts = roc_points(d["y_true"], d["scores"])
        a = metrics.auc(d["y_true"], d["scores"])
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker=".", label=f"{d['label']} (AUC={a:.2f})")
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC: ranking accepts above rejects by score")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_roc.png", dpi=150)
    plt.close(fig)


def print_table(data: list[dict]) -> None:
    print(f"\n{'Model':<16}{'n':>4}{'bal.acc':>9}{'acc':>7}{'F1':>7}"
          f"{'AUC':>7}{'FPR':>7}{'FNR':>7}")
    print("-" * 64)
    for d in data:
        m = metrics.compute_all(d["y_true"], d["y_pred"], d["scores"])
        print(f"{d['label']:<16}{m['n']:>4}{m['balanced_accuracy']:>9.3f}"
              f"{m['accuracy']:>7.3f}{m['f1']:>7.3f}{m['auc']:>7.3f}"
              f"{m['fpr']:>7.3f}{m['fnr']:>7.3f}")


def main() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load()
    if not data:
        print("No review files found -- run the reviewer first.", file=sys.stderr)
        sys.exit(1)
    fig_accuracy(data)
    fig_scores(data)
    fig_roc(data)
    print_table(data)
    print(f"\nWrote 3 figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
