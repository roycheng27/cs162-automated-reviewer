"""Fetch real ICLR reviews to use as few-shot exemplars (leakage-controlled).

The original AI Scientist reviewer is primed with real conference reviews. This
command pulls a handful of ICLR submissions *together with their official
reviewer notes*, distils each into the project's gold review schema, and writes
data/fewshot_examples.json for use by `--few-shot`.

Crucially it EXCLUDES every paper already in the evaluation set
(data/papers.json) so no exemplar can leak into evaluation.

Run:
    python -m automated_reviewer.fetch_examples --k 2 --balanced

OpenReview public data is readable anonymously (same as fetch.py).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

from . import fetch  # reuse client + helpers


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _excluded_ids(papers_path: str) -> set[str]:
    if not os.path.exists(papers_path):
        return set()
    with open(papers_path) as fh:
        return {p["id"] for p in json.load(fh)}


def _int_or_none(v):
    """Parse a leading integer out of an OpenReview rating like '6: marginal'."""
    if v is None:
        return None
    s = str(v).strip()
    num = ""
    for ch in s:
        if ch.isdigit():
            num += ch
        elif num:
            break
    return int(num) if num else None


def _extract_reviews(submission) -> list[dict]:
    """Pull official review notes from a submission's direct replies."""
    replies = (submission.details or {}).get("directReplies", []) or []
    out = []
    for reply in replies:
        invs = fetch._reply_invitations(reply)
        if not any(inv.endswith("/Official_Review") or inv.endswith("/Review")
                   for inv in invs):
            continue
        out.append(reply.get("content", {}) or {})
    return out


def _distil_review(content: dict, decision: str) -> dict | None:
    """Map a raw OpenReview review note into the project's gold schema."""
    def cv(key, default=""):
        return fetch._cval(content, key, default)

    rating = _int_or_none(cv("rating") or cv("overall_rating"))
    if rating is None:
        return None
    soundness = _int_or_none(cv("soundness")) or 3
    presentation = _int_or_none(cv("presentation")) or 3
    contribution = _int_or_none(cv("contribution")) or 3
    confidence = _int_or_none(cv("confidence")) or 3

    def as_list(text):
        text = (text or "").strip()
        return [text] if text else []

    return {
        "summary": (cv("summary") or cv("paper_summary") or "")[:600],
        "strengths": as_list(cv("strengths"))[:1],
        "weaknesses": as_list(cv("weaknesses"))[:1],
        "questions": as_list(cv("questions")),
        "limitations": (cv("limitations") or "")[:400],
        "soundness": max(1, min(4, soundness)),
        "presentation": max(1, min(4, presentation)),
        "contribution": max(1, min(4, contribution)),
        "overall": max(1, min(10, rating)),
        "confidence": max(1, min(5, confidence)),
        "decision": decision,
    }


def fetch_examples(venue: str, k: int, balanced: bool, papers_path: str,
                   out_path: str, seed: int, excerpt_chars: int) -> None:
    exclude = _excluded_ids(papers_path)
    _log(f"Excluding {len(exclude)} evaluation paper ids from the exemplar pool")

    client = fetch.get_client()
    _log(f"Fetching submissions for {venue} ...")
    submissions = client.get_all_notes(
        invitation=f"{venue}/-/Submission", details="directReplies")

    candidates = []
    for sub in submissions:
        if sub.id in exclude:
            continue
        raw = fetch._extract_decision(sub)
        if raw is None:
            continue
        candidates.append((sub, fetch._binarize(raw)))
    _log(f"  {len(candidates)} candidate exemplar papers (decision available)")

    rng = random.Random(seed)
    rng.shuffle(candidates)
    if balanced:
        accepts = [c for c in candidates if c[1] == "Accept"]
        rejects = [c for c in candidates if c[1] == "Reject"]
        order = []
        for a, r in zip(accepts, rejects):
            order.extend([a, r])
        candidates = order

    examples = []
    for sub, decision in candidates:
        if len(examples) >= k:
            break
        raw_reviews = _extract_reviews(sub)
        review = None
        for rc in raw_reviews:
            review = _distil_review(rc, decision)
            if review is not None:
                break
        if review is None:
            continue
        content = sub.content or {}
        try:
            pdf_bytes = client.get_attachment(id=sub.id, field_name="pdf")
            full_text = fetch._pdf_to_text(pdf_bytes)
        except Exception:  # noqa: BLE001 - excerpt is optional
            full_text = ""
        examples.append({
            "id": sub.id,
            "title": fetch._cval(content, "title"),
            "abstract": fetch._cval(content, "abstract"),
            "excerpt": full_text[:excerpt_chars],
            "review": review,
        })
        _log(f"  [{len(examples)}/{k}] {decision:7s} "
             f"overall={review['overall']} {fetch._cval(content, 'title')[:50]}")

    if not examples:
        _log("No exemplars with parseable reviews found.")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(examples, fh, indent=2)
    _log(f"\nWrote {len(examples)} exemplars to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fetch real ICLR reviews as few-shot exemplars.")
    ap.add_argument("--venue", default="ICLR.cc/2024/Conference")
    ap.add_argument("--k", type=int, default=2, help="Number of exemplars")
    ap.add_argument("--balanced", action="store_true",
                    help="Draw an even Accept/Reject mix")
    ap.add_argument("--papers", default="data/papers.json",
                    help="Evaluation set to EXCLUDE (no leakage)")
    ap.add_argument("--out", default="data/fewshot_examples.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--excerpt-chars", type=int, default=1500)
    args = ap.parse_args()
    fetch_examples(args.venue, args.k, args.balanced, args.papers, args.out,
                   args.seed, args.excerpt_chars)


if __name__ == "__main__":
    main()
