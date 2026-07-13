# ml/training/check_bilstm_leakage.py
# Checks whether BiLSTM train/test split was contaminated by duplicates

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

print("Loading dataset...")
df = pd.read_csv(config.DATASET_RAW)
df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
df = df[df['combined'].str.len() > 10].reset_index(drop=True)
df['label'] = df['label'].astype(int)
print(f"Total articles (raw, as used in BiLSTM training): {len(df)}")

# Replicate exact split used in train_model.py
# train_test_split was called on X_seq and y — which correspond
# to this full dataframe row-for-row after cleaning
y = df['label'].values
indices = np.arange(len(df))

idx_train, idx_test = train_test_split(
    indices,
    test_size=config.TEST_SIZE,
    random_state=config.RANDOM_STATE,
    stratify=y
)

train_texts = set(df['combined'].iloc[idx_train].tolist())
test_texts  = set(df['combined'].iloc[idx_test].tolist())

overlap = train_texts.intersection(test_texts)

print(f"\nTrain size: {len(idx_train)}")
print(f"Test size:  {len(idx_test)}")
print(f"\nArticles appearing in BOTH train and test: {len(overlap)}")

if len(overlap) > 0:
    print("\nSample overlapping articles:")
    for i, text in enumerate(list(overlap)[:3]):
        print(f"\n  [{i+1}] {text[:150]}...")
    print(f"\n[!] LEAKAGE CONFIRMED — BiLSTM results are inflated")
    print(f"    Estimated inflation: {len(overlap)/len(idx_test)*100:.2f}% of test set contaminated")
else:
    print("\n[✓] No leakage detected — BiLSTM results are clean")

print("\n" + "="*55)