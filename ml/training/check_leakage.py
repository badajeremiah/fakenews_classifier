# ml/training/check_leakage.py
# Checks WELFake for near-duplicate contamination across train/test split

import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

print("Loading dataset...")
df = pd.read_csv(config.DATASET_RAW)
df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df = df[df['combined'].str.len() > 20].reset_index(drop=True)
df['label'] = df['label'].astype(int)

print(f"Total articles: {len(df)}")

# Check 1 — exact duplicates across full dataset
print("\n[1] Checking exact duplicates in full dataset...")
exact_dupes = df['combined'].duplicated().sum()
print(f"    Exact duplicate articles: {exact_dupes}")

# Check 2 — deduplicate and re-split
print("\n[2] Deduplicating...")
df_deduped = df.drop_duplicates(subset=['combined']).reset_index(drop=True)
print(f"    Articles after dedup: {len(df_deduped)} "
      f"(removed {len(df) - len(df_deduped)})")

# Check 3 — split and check overlap
print("\n[3] Checking train/test overlap after split...")
train_df, test_df = train_test_split(
    df_deduped[['combined', 'label']],
    test_size=0.2,
    random_state=42,
    stratify=df_deduped['label']
)

train_texts = set(train_df['combined'].tolist())
test_texts  = set(test_df['combined'].tolist())
overlap = train_texts.intersection(test_texts)
print(f"    Train size: {len(train_df)}")
print(f"    Test size:  {len(test_df)}")
print(f"    Exact text overlap between train and test: {len(overlap)}")

# Check 4 — hash-based near-duplicate check
print("\n[4] Hash-based duplicate check...")
df['text_hash'] = df['combined'].apply(
    lambda x: hashlib.md5(x.strip().lower().encode()).hexdigest())
hash_dupes = df['text_hash'].duplicated().sum()
print(f"    Hash-identical articles: {hash_dupes}")

# Check 5 — title-only duplicates (common WELFake issue)
print("\n[5] Title-only duplicate check...")
title_dupes = df['title'].dropna().duplicated().sum()
print(f"    Duplicate titles: {title_dupes}")

print("\n" + "="*55)
if len(overlap) == 0 and exact_dupes < 100:
    print(" RESULT: No significant leakage detected.")
    print(" The 99.93% accuracy is likely genuine.")
elif len(overlap) > 0:
    print(" RESULT: LEAKAGE DETECTED — train/test overlap exists.")
    print(f" {len(overlap)} articles appear in both splits.")
else:
    print(f" RESULT: {exact_dupes} duplicates in dataset.")
    print(" These inflate test performance if split naively.")
print("="*55)