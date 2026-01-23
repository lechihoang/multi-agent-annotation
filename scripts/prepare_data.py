#!/usr/bin/env python3
"""Prepare data for Blind Annotation simulation.

Splits original dataset into:
1. seed.csv: First N examples (with labels) for RAG/Few-shot.
2. unlabeled.csv: Remaining examples (labels hidden/removed) for annotation.
"""

import csv
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import get_config

DATA_DIR = Path(__file__).parent.parent / "data"


def split_data(train_file: str, dev_file: str = "dev.csv", seed_size: int = 100):
    config = get_config()

    # Get column names from config
    text_col = getattr(config.task.columns, "text", "review")
    label_col = getattr(config.task.columns, "label", "label")

    print(f"Using columns from config: Text='{text_col}', Label='{label_col}'")

    train_path = DATA_DIR / train_file
    dev_path = DATA_DIR / dev_file
    seed_path = DATA_DIR / "seed.csv"
    unlabeled_path = DATA_DIR / "unlabeled.csv"

    if not train_path.exists():
        print(f"Error: Train file {train_path} not found.")
        return

    # 1. Process Train File
    print(f"Reading {train_path}...")
    train_rows = []
    fieldnames = None
    with open(train_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            train_rows.append(row)

    # 2. Extract Seed
    if len(train_rows) <= seed_size:
        print(f"Error: Train size ({len(train_rows)}) <= seed size ({seed_size}).")
        return

    seed_data = train_rows[:seed_size]
    train_remain = train_rows[seed_size:]

    # 3. Process Dev File (Optional)
    dev_rows = []
    if dev_path.exists():
        print(f"Reading {dev_path}...")
        with open(dev_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # Verify headers match
            if reader.fieldnames != fieldnames:
                print("Warning: Dev file headers differ from Train file.")
            for row in reader:
                dev_rows.append(row)
    else:
        print(f"Note: Dev file {dev_path} not found. Proceeding with Train only.")

    # 4. Combine
    combined_data = train_remain + dev_rows

    # 5. Write Outputs
    # Write Seed (Only text and label based on config)
    seed_cols = []

    # STRICT MODE: Columns MUST exist
    if text_col not in fieldnames:
        print(
            f"Error: Configured text column '{text_col}' not found in CSV. Available: {fieldnames}"
        )
        return
    seed_cols.append(text_col)

    if label_col not in fieldnames:
        print(
            f"Error: Configured label column '{label_col}' not found in CSV. Available: {fieldnames}"
        )
        return
    seed_cols.append(label_col)

    with open(seed_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=seed_cols)
        writer.writeheader()

        for row in seed_data:
            clean_row = {k: row[k] for k in seed_cols if k in row}
            writer.writerow(clean_row)

    # Write Unlabeled (Only text column)
    # Identify the text column we actually found
    found_text_col = seed_cols[0] if seed_cols else text_col
    unlabeled_cols = [found_text_col]

    with open(unlabeled_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=unlabeled_cols)
        writer.writeheader()

        for row in combined_data:
            clean_row = {k: row[k] for k in unlabeled_cols if k in row}
            writer.writerow(clean_row)

    # 6. Report Metadata for Splitting later
    print(f"\n✓ Created {seed_path} ({len(seed_data)} samples)")
    print(f"✓ Created {unlabeled_path} ({len(combined_data)} samples)")
    print("\nIMPORTANT METADATA (Save this for later splitting):")
    print(f"- Seed (from Train): 1st {len(seed_data)} rows of original train.")
    print(f"- Train Remain: Lines 1 to {len(train_remain)} in unlabeled.csv")
    if dev_rows:
        print(
            f"- Dev Set: Lines {len(train_remain) + 1} to {len(combined_data)} in unlabeled.csv"
        )

    print(
        f"\nReady! Run: ./venv/bin/python scripts/run_arq_batch.py --input data/unlabeled.csv"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="train.csv", help="Train filename")
    parser.add_argument("--dev", default="dev.csv", help="Dev filename (optional)")
    parser.add_argument("--seed", type=int, default=100, help="Seed size")
    args = parser.parse_args()

    split_data(args.train, args.dev, args.seed)
