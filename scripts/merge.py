#!/usr/bin/env python3
"""
Script tổng hợp dữ liệu từ seed.csv và batch_arq_results.csv
để tạo train_labeled.csv và val_labeled.csv

Logic mới:
- seed.csv: 100 samples đầu của train (đã có nhãn gốc)
- batch_arq_results.csv: Kết quả gán nhãn cho unlabeled (train[100:] + val)
  theo đúng thứ tự của unlabeled.csv

Cách tách:
- unlabeled[0:4287] -> tương ứng với train[100:] (4287 samples)
- unlabeled[4287:] -> tương ứng với val (548 samples)

Vậy:
- train_labeled = seed (100) + batch_arq_results[0:4287]
- val_labeled = batch_arq_results[4287:4835]
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"


def load_data():
    """Load các file cần thiết."""
    print("Loading data files...")
    
    # Load seed (100 samples đầu)
    seed_df = pd.read_csv(DATA_DIR / "seed.csv")
    print(f"  - seed.csv: {len(seed_df)} samples")
    
    # Load train.csv
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"  - train.csv: {len(train_df)} samples")
    
    # Load val.csv
    val_df = pd.read_csv(DATA_DIR / "val.csv")
    print(f"  - val.csv: {len(val_df)} samples")
    
    # Load batch_arq_results
    arq_df = pd.read_csv(DATA_DIR / "batch_arq_results.csv")
    print(f"  - batch_arq_results.csv: {len(arq_df)} samples")
    
    return seed_df, train_df, val_df, arq_df


def create_labeled_files(seed_df, train_df, val_df, arq_df):
    """
    Tạo train_labeled.csv và val_labeled.csv theo thứ tự:
    - train_labeled = seed (100 samples) + batch_arq_results[0:train_remainder]
    - val_labeled = batch_arq_results[train_remainder:]
    
    Trong đó train_remainder = len(train) - 100
    """
    print("\nCreating labeled files...")
    
    # Tính số lượng
    train_remainder = len(train_df) - 100  # 4287
    val_count = len(val_df)  # 548
    
    print(f"  - Train remainder (train[100:]): {train_remainder}")
    print(f"  - Val count: {val_count}")
    print(f"  - Expected total in batch_arq: {train_remainder + val_count}")
    print(f"  - Actual batch_arq size: {len(arq_df)}")
    
    # Kiểm tra số lượng
    if len(arq_df) != train_remainder + val_count:
        print(f"  - WARNING: batch_arq_results size mismatch!")
    
    # 1. Tạo train_labeled
    print("\nCreating train_labeled.csv...")
    
    # Seed part (100 samples)
    seed_part = seed_df.copy()
    seed_output = pd.DataFrame({
        'text': seed_part['review'],
        'label': seed_part['label'].astype(int),
        'source': ['seed'] * len(seed_part),
        'confidence': [1.0] * len(seed_part),
        'decision': ['ground_truth'] * len(seed_part)
    })
    
    # ARQ part cho train (train_remainder samples đầu tiên)
    arq_train = arq_df.iloc[:train_remainder].copy()
    arq_train_output = pd.DataFrame({
        'text': arq_train['text'],
        'label': arq_train['final_label'].astype(int),
        'source': ['arq_labeled'] * len(arq_train),
        'confidence': arq_train['confidence'],
        'decision': arq_train['decision']
    })
    
    # Kết hợp
    train_labeled = pd.concat([seed_output, arq_train_output], ignore_index=True)
    print(f"  - Seed: {len(seed_output)}")
    print(f"  - ARQ train: {len(arq_train_output)}")
    print(f"  - Total train_labeled: {len(train_labeled)}")
    
    # 2. Tạo val_labeled
    print("\nCreating val_labeled.csv...")
    
    # ARQ part cho val (các samples còn lại)
    arq_val = arq_df.iloc[train_remainder:].copy()
    val_labeled = pd.DataFrame({
        'text': arq_val['text'],
        'label': arq_val['final_label'].astype(int),
        'source': ['arq_labeled'] * len(arq_val),
        'confidence': arq_val['confidence'],
        'decision': arq_val['decision']
    })
    
    print(f"  - ARQ val: {len(val_labeled)}")
    
    return train_labeled, val_labeled


def main():
    """Main function."""
    print("=" * 60)
    print("Merging Labeled Data - MAFA ViOCD")
    print("=" * 60)
    
    # Load data
    seed_df, train_df, val_df, arq_df = load_data()
    
    # Tạo labeled files
    train_labeled, val_labeled = create_labeled_files(seed_df, train_df, val_df, arq_df)
    
    # Kiểm tra tổng số
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  - train_labeled: {len(train_labeled)} samples")
    print(f"    + Seed (ground truth): 100")
    print(f"    + ARQ labeled: {len(train_labeled) - 100}")
    print(f"  - val_labeled: {len(val_labeled)} samples")
    print(f"  - Total: {len(train_labeled) + len(val_labeled)} samples")
    print("=" * 60)
    
    # Thống kê label
    print("\nLabel Distribution:")
    train_dist = train_labeled['label'].value_counts().sort_index()
    val_dist = val_labeled['label'].value_counts().sort_index()
    print(f"  - Train: {dict(train_dist)}")
    print(f"  - Val: {dict(val_dist)}")
    
    # Thống kê decision
    print("\nDecision Distribution:")
    train_dec = train_labeled['decision'].value_counts()
    val_dec = val_labeled['decision'].value_counts()
    print(f"  - Train: {dict(train_dec)}")
    print(f"  - Val: {dict(val_dec)}")
    
    # Lưu file
    output_train = DATA_DIR / "train_labeled.csv"
    output_val = DATA_DIR / "val_labeled.csv"
    
    train_labeled.to_csv(output_train, index=False)
    val_labeled.to_csv(output_val, index=False)
    
    print(f"\nOutput files saved:")
    print(f"  - {output_train}")
    print(f"  - {output_val}")


if __name__ == "__main__":
    main()
