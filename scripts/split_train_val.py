#!/usr/bin/env python3
"""
Split annotated data into train and val sets based on original train.csv and val.csv.
Matches reviews from annotated.csv with train.csv and val.csv to get labels.
"""

import csv


def main():
    # Read annotated.csv and create a dict by review text
    annotated = {}
    with open('data/annotated.csv', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('final_label', '').strip() in ['0', '1']:
                annotated[row['text']] = row['final_label']

    print(f'Annotated unique reviews: {len(annotated)}')

    # Read train.csv and match -> train_labeled.csv
    train_labeled = []
    with open('data/train.csv', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row['review']
            if review in annotated:
                train_labeled.append({'review': review, 'label': annotated[review]})

    print(f'Train labeled: {len(train_labeled)}')

    # Read val.csv and match -> val_labeled.csv
    val_labeled = []
    with open('data/val.csv', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = row['review']
            if review in annotated:
                val_labeled.append({'review': review, 'label': annotated[review]})

    print(f'Val labeled: {len(val_labeled)}')

    # Write train_labeled.csv
    with open('data/train_labeled.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['review', 'label'])
        writer.writeheader()
        writer.writerows(train_labeled)

    # Write val_labeled.csv
    with open('data/val_labeled.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['review', 'label'])
        writer.writeheader()
        writer.writerows(val_labeled)

    # Stats
    train_0 = sum(1 for r in train_labeled if r['label'] == '0')
    train_1 = sum(1 for r in train_labeled if r['label'] == '1')
    val_0 = sum(1 for r in val_labeled if r['label'] == '0')
    val_1 = sum(1 for r in val_labeled if r['label'] == '1')

    print(f'\n=== Summary ===')
    print(f'Train: {len(train_labeled)} (Label 0: {train_0}, Label 1: {train_1})')
    print(f'Val: {len(val_labeled)} (Label 0: {val_0}, Label 1: {val_1})')
    print(f'\nSaved to: data/train_labeled.csv, data/val_labeled.csv')


if __name__ == '__main__':
    main()
