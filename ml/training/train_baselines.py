# ml/training/train_baselines.py
# Baseline classifier training for benchmarking
# Researcher: Bada Toluwani Jeremiah (SEN/20/5094)
# FUTA — Department of Software Engineering

import os
import sys
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

def load_data():
    print("[1/3] Loading data...")
    X_tfidf = load_npz(os.path.join(
        config.DATASET_PROCESSED, 'X_tfidf.npz'))
    y = np.load(os.path.join(
        config.DATASET_PROCESSED, 'y_labels.npy'))
    print(f"      TF-IDF shape: {X_tfidf.shape}")
    print(f"      Labels shape: {y.shape}")
    return X_tfidf, y

def evaluate(name, model, X_test, y_test, results):
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    results[name] = {
        'accuracy': acc, 'precision': prec,
        'recall': rec, 'f1_score': f1
    }
    print(f"\n  [{name}]")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(classification_report(y_test, y_pred,
                                target_names=['Real','Fake']))

def train_baselines():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y)

    print(f"      Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print("\n[2/3] Training baseline classifiers...")

    results = {}

    # Naive Bayes
    print("\n  Training Naive Bayes...")
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    evaluate('Naive Bayes', nb, X_test, y_test, results)

    # Logistic Regression
    print("\n  Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000,
                            random_state=config.RANDOM_STATE)
    lr.fit(X_train, y_train)
    evaluate('Logistic Regression', lr, X_test, y_test, results)

    # SVM
    print("\n  Training SVM (LinearSVC)...")
    svm = LinearSVC(max_iter=2000,
                    random_state=config.RANDOM_STATE)
    svm.fit(X_train, y_train)
    evaluate('SVM', svm, X_test, y_test, results)

    print("\n[3/3] Saving baseline results...")
    eval_dir = os.path.join(config.BASE_DIR, 'ml', 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    joblib.dump(results, os.path.join(
        eval_dir, 'baseline_metrics.pkl'))

    print("\n" + "="*55)
    print(" BASELINE SUMMARY")
    print("="*55)
    for name, m in results.items():
        print(f"  {name:<25} Acc: {m['accuracy']*100:.2f}%  "
              f"F1: {m['f1_score']:.4f}")
    print("="*55)
    print(f"\n  Results saved to: {eval_dir}")

if __name__ == "__main__":
    print("="*55)
    print(" Fake News Detector — Baseline Training")
    print("="*55)
    train_baselines()