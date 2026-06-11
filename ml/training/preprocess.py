# ml/training/preprocess.py
# Text preprocessing pipeline for Hybrid Fake News Detection System
# Researcher: Bada Toluwani Jeremiah (SEN/20/5094)
# FUTA — Department of Software Engineering

import os
import re
import sys
import numpy as np
import pandas as pd
import joblib
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Add project root to path ───────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

# ── Download NLTK resources ────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


# ── 1. Load Dataset ────────────────────────────────────
def load_dataset():
    print("[1/5] Loading dataset...")
    df = pd.read_csv(config.DATASET_RAW)
    print(f"      Raw shape: {df.shape}")
    print(f"      Columns: {list(df.columns)}")
    print(f"      Label distribution:\n{df['label'].value_counts()}")
    return df


# ── 2. Clean Text ──────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)       # remove URLs
    text = re.sub(r'<.*?>', '', text)                  # remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)              # keep letters only
    text = re.sub(r'\s+', ' ', text).strip()           # normalize whitespace
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return ' '.join(tokens)


def preprocess_texts(df):
    print("[2/5] Cleaning text...")
    # Combine title and text if both exist
    if 'title' in df.columns and 'text' in df.columns:
        df['combined'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    elif 'text' in df.columns:
        df['combined'] = df['text'].fillna('')
    else:
        raise ValueError("Dataset must have a 'text' column")

    df['cleaned'] = df['combined'].apply(clean_text)
    df = df[df['cleaned'].str.len() > 10].reset_index(drop=True)
    print(f"      Cleaned shape: {df.shape}")
    return df


# ── 3. Train Word2Vec ──────────────────────────────────
def train_word2vec(df):
    print("[3/5] Training Word2Vec embeddings...")
    sentences = [text.split() for text in df['cleaned']]
    w2v_model = Word2Vec(
        sentences=sentences,
        vector_size=config.WORD2VEC_VECTOR_SIZE,
        window=config.WORD2VEC_WINDOW,
        min_count=config.WORD2VEC_MIN_COUNT,
        workers=config.WORD2VEC_WORKERS,
        seed=config.RANDOM_STATE
    )
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
    save_path = os.path.join(config.EMBEDDINGS_DIR, 'word2vec.model')
    w2v_model.save(save_path)
    print(f"      Vocabulary size: {len(w2v_model.wv)}")
    print(f"      Saved to: {save_path}")
    return w2v_model


# ── 4. Fit TF-IDF Vectorizer ───────────────────────────
def fit_tfidf(df):
    print("[4/5] Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    tfidf.fit(df['cleaned'])
    os.makedirs(config.VECTORIZERS_DIR, exist_ok=True)
    save_path = os.path.join(config.VECTORIZERS_DIR, 'tfidf_vectorizer.pkl')
    joblib.dump(tfidf, save_path)
    print(f"      Features: {config.TFIDF_MAX_FEATURES}")
    print(f"      Saved to: {save_path}")
    return tfidf


# ── 5. Tokenize, Pad and Save Arrays ──────────────────
def tokenize_and_save(df, w2v_model, tfidf):
    print("[5/5] Tokenizing and padding sequences...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['cleaned'])

    sequences = tokenizer.texts_to_sequences(df['cleaned'])
    X_seq = pad_sequences(
        sequences,
        maxlen=config.MAX_SEQUENCE_LENGTH,
        padding='post',
        truncating='post'
    )

    labels = df['label'].values

    os.makedirs(config.DATASET_PROCESSED, exist_ok=True)

    np.save(os.path.join(config.DATASET_PROCESSED, 'X_sequences.npy'), X_seq)
    np.save(os.path.join(config.DATASET_PROCESSED, 'y_labels.npy'), labels)
    joblib.dump(tokenizer, os.path.join(
        config.VECTORIZERS_DIR, 'tokenizer.pkl'))
    from scipy.sparse import save_npz
    X_tfidf_sparse = tfidf.transform(df['cleaned'])
    save_npz(os.path.join(config.DATASET_PROCESSED, 'X_tfidf.npz'), X_tfidf_sparse)
    print(f"      TF-IDF matrix shape: {X_tfidf_sparse.shape}")

    print(f"      Sequence shape: {X_seq.shape}")
    print(f"      Labels shape: {labels.shape}")
    print(f"      Saved to: {config.DATASET_PROCESSED}")
    return X_seq, labels, tokenizer


# ── Main Pipeline ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(" Fake News Detector — Preprocessing Pipeline")
    print("=" * 55)

    df = load_dataset()
    df = preprocess_texts(df)
    w2v_model = train_word2vec(df)
    tfidf = fit_tfidf(df)
    tokenize_and_save(df, w2v_model, tfidf)

    print("=" * 55)
    print(" Preprocessing complete. All artifacts saved.")
    print("=" * 55)