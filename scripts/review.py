#!/usr/bin/env python3
"""Tier 4: Human Review CLI.

Review annotations that need human confirmation (confidence: 0.60-0.85).

Usage:
    python scripts/review.py --input data/batch_results.json
    python scripts/review.py --input data/batch_results.json --approve-all
    python scripts/review.py --input data/batch_results.json --auto
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"


def load_results(input_file: str) -> list:
    """Load annotation results from JSON."""
    with open(input_file, "r", encoding="utf-8") as f:
        return json.load(f)


def get_needs_review(results: list) -> list:
    """Filter results that need human review."""
    return [r for r in results if r.get("tier3", {}).get("decision") == "review"]


def review_item(item: dict, idx: int, total: int) -> dict:
    """Review a single item and return updated result."""
    text = item["text"][:100] + "..." if len(item["text"]) > 100 else item["text"]
    tier2 = item.get("tier2", {})
    tier3 = item.get("tier3", {})

    print(f"\n{'=' * 70}")
    print(f"[{idx + 1}/{total}] REVIEW REQUIRED")
    print(f"{'=' * 70}")
    print(f"\nText: {text}\n")

    print("Agent Predictions:")
    if isinstance(tier2, dict):
        for agent, pred in tier2.items():
            if isinstance(pred, dict):
                label = pred.get("label", "?")
                conf = pred.get("confidence", 0)
                print(f"  - {agent}: label={label}, confidence={conf:.2f}")
    else:
        print(
            f"  - Label: {tier2.get('label', '?')}, Confidence: {tier2.get('confidence', 0):.2f}"
        )

    print(
        f"\nCurrent Decision: {tier3.get('decision', '?')} ({tier3.get('confidence', 0):.2f})"
    )

    print("\nOptions:")
    print("  [0] Approve (agree with current)")
    print("  [1] Override to Toxic")
    print("  [2] Override to Non-Toxic")
    print("  [3] Escalate to expert")
    print("  [s] Skip for now")
    print("  [q] Quit")

    while True:
        choice = input("\nYour choice: ").strip().lower()

        if choice == "0":
            return {
                **item,
                "review_action": "approve",
                "reviewed_at": datetime.now().isoformat(),
            }
        elif choice == "1":
            return {
                **item,
                "review_action": "override_toxic",
                "reviewed_label": "1",
                "reviewed_at": datetime.now().isoformat(),
            }
        elif choice == "2":
            return {
                **item,
                "review_action": "override_non_toxic",
                "reviewed_label": "0",
                "reviewed_at": datetime.now().isoformat(),
            }
        elif choice == "3":
            return {
                **item,
                "review_action": "escalate",
                "reviewed_at": datetime.now().isoformat(),
            }
        elif choice == "s":
            return None
        elif choice == "q":
            print("Quitting...")
            exit(0)
        else:
            print("Invalid choice. Try again.")


def review_all(results: list, input_file: str) -> list:
    """Review all items that need review."""
    needs_review = get_needs_review(results)
    total = len(needs_review)

    if total == 0:
        print("No items need review. All decisions are approved or escalated.")
        return results

    print(f"\nFound {total} items needing review")
    print(f"Input file: {input_file}\n")

    reviewed = []
    skipped = []

    for i, item in enumerate(needs_review):
        result = review_item(item, i, total)
        if result:
            reviewed.append(result)
        else:
            skipped.append(item)

    for r in reviewed:
        for j, orig in enumerate(results):
            if orig.get("task_id") == r.get("task_id"):
                results[j] = r
                break

    print(f"\n{'=' * 70}")
    print("REVIEW SUMMARY")
    print(f"{'=' * 70}")
    print(f"Reviewed: {len(reviewed)}")
    print(f"Skipped: {len(skipped)}")

    return results


def approve_all(results: list) -> list:
    """Approve all review items."""
    needs_review = get_needs_review(results)

    for item in needs_review:
        item["review_action"] = "bulk_approve"
        item["reviewed_at"] = datetime.now().isoformat()

        if "tier3" not in item:
            item["tier3"] = {}
        item["tier3"]["decision"] = "approve"

    print(f"Approved {len(needs_review)} items")
    return results


def auto_decide(results: list) -> list:
    """Auto-decide based on majority vote from agents."""
    for item in results:
        tier2 = item.get("tier2", {})

        if isinstance(tier2, dict):
            labels = []
            for pred in tier2.values():
                if isinstance(pred, dict) and "label" in pred:
                    labels.append(pred["label"])

            if labels:
                from collections import Counter

                vote = Counter(labels).most_common(1)[0][0]

                if "tier3" not in item:
                    item["tier3"] = {}
                item["tier3"]["label"] = vote
                item["tier3"]["decision"] = "auto_approved"
                item["tier3"]["auto_reason"] = "majority_vote"

    print(f"Auto-decided {len(results)} items based on majority vote")
    return results


def show_stats(results: list):
    """Show statistics."""
    total = len(results)

    decisions = {"approve": 0, "review": 0, "escalate": 0}
    for r in results:
        d = r.get("tier3", {}).get("decision", "unknown")
        decisions[d] = decisions.get(d, 0) + 1

    print(f"\n{'=' * 70}")
    print("STATISTICS")
    print(f"{'=' * 70}")
    print(f"Total: {total}")
    print(
        f"Approved: {decisions.get('approve', 0)} ({decisions.get('approve', 0) / total * 100:.1f}%)"
    )
    print(
        f"Review: {decisions.get('review', 0)} ({decisions.get('review', 0) / total * 100:.1f}%)"
    )
    print(
        f"Escalate: {decisions.get('escalate', 0)} ({decisions.get('escalate', 0) / total * 100:.1f}%)"
    )

    reviewed = sum(1 for r in results if r.get("review_action"))
    print(f"Reviewed: {reviewed}")


def main():
    parser = argparse.ArgumentParser(description="Human Review CLI for MAFA")
    parser.add_argument(
        "--input",
        "-i",
        default=str(DATA_DIR / "batch_mafa_results.json"),
        help="Input JSON file",
    )
    parser.add_argument(
        "--output", "-o", help="Output JSON file (default: overwrite input)"
    )
    parser.add_argument(
        "--approve-all", action="store_true", help="Approve all review items"
    )
    parser.add_argument(
        "--auto", action="store_true", help="Auto-decide based on majority vote"
    )
    parser.add_argument("--stats", action="store_true", help="Show statistics only")

    args = parser.parse_args()

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: File not found: {input_file}")
        return

    results = load_results(str(input_file))
    print(f"Loaded {len(results)} results from {input_file}")

    if args.stats:
        show_stats(results)
        return

    if args.auto:
        results = auto_decide(results)
        output_file = args.output or str(input_file)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")
        return

    if args.approve_all:
        results = approve_all(results)
        output_file = args.output or str(input_file)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_file}")
        return

    results = review_all(results, str(input_file))
    output_file = args.output or str(input_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {output_file}")
    show_stats(results)


if __name__ == "__main__":
    main()
