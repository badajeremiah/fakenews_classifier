# ml/training/train_model.py
# Hybrid BiLSTM + TF-IDF model training script
# Researcher: Bada Toluwani Jeremiah (SEN/20/5094)
# FUTA — Department of Software Engineering

import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Bidirectional,
                                     LSTM, Dense, Dropout, Concatenate)
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)

# ── Add project root to path ───────────────────────────
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# ── 1. Load Preprocessed Data ──────────────────────────
def load_data():
    print("[1/6] Loading preprocessed data...")
    X_seq = np.load(os.path.join(
        config.DATASET_PROCESSED, 'X_sequences.npy'))
    y = np.load(os.path.join(
        config.DATASET_PROCESSED, 'y_labels.npy'))
    tfidf = joblib.load(os.path.join(
        config.VECTORIZERS_DIR, 'tfidf_vectorizer.pkl'))
    tokenizer = joblib.load(os.path.join(
        config.VECTORIZERS_DIR, 'tokenizer.pkl'))

    print(f"      Sequences: {X_seq.shape}")
    print(f"      Labels: {y.shape}")
    return X_seq, y, tfidf, tokenizer


# ── 2. Build Embedding Matrix ──────────────────────────
def build_embedding_matrix(tokenizer):
    print("[2/6] Building embedding matrix from Word2Vec...")
    from gensim.models import Word2Vec
    w2v = Word2Vec.load(os.path.join(
        config.EMBEDDINGS_DIR, 'word2vec.model'))

    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros(
        (vocab_size, config.WORD2VEC_VECTOR_SIZE))

    found = 0
    for word, idx in tokenizer.word_index.items():
        if word in w2v.wv:
            embedding_matrix[idx] = w2v.wv[word]
            found += 1

    print(f"      Vocabulary size: {vocab_size}")
    print(f"      Words found in Word2Vec: {found}/{vocab_size}")
    return embedding_matrix, vocab_size


# ── 3. Build Hybrid Model ──────────────────────────────
def build_model(vocab_size, embedding_matrix):
    print("[3/6] Building hybrid BiLSTM + TF-IDF model...")

    # Path A — Sequence input (BiLSTM branch)
    seq_input = Input(shape=(config.MAX_SEQUENCE_LENGTH,),
                      name='sequence_input')
    embedding = Embedding(
        input_dim=vocab_size,
        output_dim=config.WORD2VEC_VECTOR_SIZE,
        weights=[embedding_matrix],
        input_length=config.MAX_SEQUENCE_LENGTH,
        trainable=True,
        name='word2vec_embedding'
    )(seq_input)
    bilstm = Bidirectional(
        LSTM(config.BILSTM_UNITS, return_sequences=False),
        name='bilstm'
    )(embedding)

    # Path B — TF-IDF input (statistical branch)
    tfidf_input = Input(shape=(config.TFIDF_MAX_FEATURES,),
                        name='tfidf_input')
    tfidf_dense = Dense(config.DENSE_REDUCTION_UNITS,
                        activation='relu',
                        name='tfidf_reduction')(tfidf_input)

    # Fusion — concatenate both branches
    merged = Concatenate(name='fusion')([bilstm, tfidf_dense])
    dense1 = Dense(config.FUSION_DENSE_UNITS,
                   activation='relu',
                   name='fusion_dense')(merged)
    dropout = Dropout(config.DROPOUT_RATE,
                      name='dropout')(dense1)
    output = Dense(1, activation='sigmoid',
                   name='output')(dropout)

    model = Model(
        inputs=[seq_input, tfidf_input],
        outputs=output,
        name='hybrid_fake_news_detector'
    )
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


# ── 4. Prepare TF-IDF Features ─────────────────────────
def prepare_tfidf_features(X_seq, tfidf, tokenizer):
    print("[4/6] Preparing TF-IDF features...")
    # Reconstruct cleaned texts from sequences for TF-IDF
    reverse_index = {v: k for k, v in tokenizer.word_index.items()}
    texts = []
    for seq in X_seq:
        words = [reverse_index.get(idx, '') for idx in seq if idx != 0]
        texts.append(' '.join(words))
    X_tfidf = tfidf.transform(texts).toarray().astype(np.float32)
    print(f"      TF-IDF shape: {X_tfidf.shape}")
    return X_tfidf


# ── 5. Train Model ─────────────────────────────────────
def train_model(model, X_seq, X_tfidf, y):
    print("[5/6] Training model...")
    X_seq_train, X_seq_test, X_tfidf_train, X_tfidf_test, y_train, y_test = \
        train_test_split(X_seq, X_tfidf, y,
                         test_size=config.TEST_SIZE,
                         random_state=config.RANDOM_STATE,
                         stratify=y)

    print(f"      Train: {len(y_train)} | Test: {len(y_test)}")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(
        config.MODELS_DIR, 'best_model.h5')

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=3,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=2, verbose=1)
    ]

    history = model.fit(
        [X_seq_train, X_tfidf_train], y_train,
        validation_data=([X_seq_test, X_tfidf_test], y_test),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    return history, X_seq_test, X_tfidf_test, y_test


# ── 6. Evaluate and Save Results ───────────────────────
def evaluate_model(model, X_seq_test, X_tfidf_test, y_test, history):
    print("[6/6] Evaluating model...")
    y_pred_prob = model.predict(
        [X_seq_test, X_tfidf_test], verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print(" HYBRID MODEL RESULTS")
    print("=" * 55)
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print("=" * 55)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Real', 'Fake']))

    # Save confusion matrix plot
    eval_dir = os.path.join(config.BASE_DIR, 'ml', 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title('Hybrid BiLSTM + TF-IDF — Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # Save training history plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, 'training_history.png'), dpi=150)
    plt.close()

    print(f"\n  Plots saved to: {eval_dir}")

    # Save metrics to file
    metrics = {
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1_score': f1
    }
    joblib.dump(metrics, os.path.join(eval_dir, 'metrics.pkl'))
    return metrics


# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(" Fake News Detector — Model Training")
    print("=" * 55)

    X_seq, y, tfidf, tokenizer = load_data()
    embedding_matrix, vocab_size = build_embedding_matrix(tokenizer)
    model = build_model(vocab_size, embedding_matrix)
    X_tfidf = prepare_tfidf_features(X_seq, tfidf, tokenizer)
    history, X_seq_test, X_tfidf_test, y_test = train_model(
        model, X_seq, X_tfidf, y)
    evaluate_model(model, X_seq_test, X_tfidf_test, y_test, history)

    print("\n Training complete. Model saved to ml/models/best_model.h5")