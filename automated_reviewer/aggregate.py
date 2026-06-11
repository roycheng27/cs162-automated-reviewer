"""Aggregate many metrics_*.json runs into one comparison table.

Use after running the experiment grid (model x prompt-style x few-shot). It
collects every metrics file you point it at, pulls the automated-reviewer row
from each, and emits a single sorted table to stdout plus a CSV.

A "condition" label is derived from the filename. The recommended naming
convention (see run_experiments.sh) is:

    results/metrics_<model>__<style>__fs<k>.json
        e.g. metrics_gpt-4o-mini__strict_gatekeeper__fs2.json

so the model / prompt-style / few-shot setting can be read straight off the
table and grouped.

Run:
    python -m automated_reviewer.aggregate results/metrics_*__*.json
    python -m automated_reviewer.aggregate --glob 'results/metrics_*.json' \
        --out results/aggregate.csv
"""

from __future__ import annotations

import argparse
import csv
import glob as globmod
import json
import os
import re
import sys


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _parse_label(path: str) -> tuple[str, str, str]:
    """Return (model, style, few_shot) parsed from the filename if possible."""
    base = os.path.basename(path)
    base = re.sub(r"^metrics_", "", base)
    base = re.sub(r"\.json$", "", base)
    model, style, fs = base, "default", "0"
    if "__" in base:
        parts = base.split("__")
        model = parts[0]
        for part in parts[1:]:
            if part.startswith("fs"):
                fs = part[2:]
            else:
                style = part
    return model, style, fs


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in sorted(paths):
        try:
            with open(path) as fh:
                report = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            _log(f"  skip {path}: {e}")
            continue
        r = report.get("automated_reviewer")
        if not r:
            _log(f"  skip {path}: no automated_reviewer block")
            continue
        model, style, fs = _parse_label(path)
        ci = r.get("balanced_accuracy_95ci", [None, None])
        rows.append({
            "model": model,
            "style": style,
            "few_shot": fs,
            "n": r.get("n"),
            "balanced_accuracy": r.get("balanced_accuracy"),
            "ci_low": ci[0],
            "ci_high": ci[1],
            "accuracy": r.get("accuracy"),
            "auc": r.get("auc"),
            "f1": r.get("f1"),
            "fpr": r.get("fpr"),
            "fnr": r.get("fnr"),
            "mean_overall": report.get("mean_predicted_overall"),
            "file": os.path.basename(path),
        })
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        _log("No rows to show.")
        return
    rows = sorted(rows, key=lambda r: (r["model"], r["style"], r["few_shot"]))
    hdr = (f"{'model':<26}{'style':<18}{'fs':>3} {'n':>4} "
           f"{'bal.acc':>8} {'95% CI':>16} {'acc':>7} {'AUC':>7} "
           f"{'F1':>7} {'FPR':>6} {'FNR':>6} {'mOvr':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        ci = (f"[{r['ci_low']:.2f},{r['ci_high']:.2f}]"
              if r["ci_low"] is not None else "n/a")
        def f(x, p=3):
            return f"{x:.{p}f}" if isinstance(x, (int, float)) else "n/a"
        print(f"{r['model']:<26}{r['style']:<18}{str(r['few_shot']):>3} "
              f"{str(r['n']):>4} {f(r['balanced_accuracy']):>8} {ci:>16} "
              f"{f(r['accuracy']):>7} {f(r['auc']):>7} {f(r['f1']):>7} "
              f"{f(r['fpr'],2):>6} {f(r['fnr'],2):>6} {f(r['mean_overall'],2):>6}")


def write_csv(rows: list[dict], out_path: str) -> None:
    fields = ["model", "style", "few_shot", "n", "balanced_accuracy",
              "ci_low", "ci_high", "accuracy", "auc", "f1", "fpr", "fnr",
              "mean_overall", "file"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    _log(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate metrics_*.json runs.")
    ap.add_argument("files", nargs="*", help="metrics_*.json files")
    ap.add_argument("--glob", default=None,
                    help="Glob pattern instead of listing files")
    ap.add_argument("--out", default="results/aggregate.csv",
                    help="CSV output path (default: results/aggregate.csv)")
    args = ap.parse_args()

    paths = list(args.files)
    if args.glob:
        paths += globmod.glob(args.glob)
    if not paths:
        paths = globmod.glob("results/metrics_*.json")
    if not paths:
        _log("No metrics files found.")
        sys.exit(1)

    rows = load_rows(paths)
    print_table(rows)
    write_csv(rows, args.out)


if __name__ == "__main__":
    main()
