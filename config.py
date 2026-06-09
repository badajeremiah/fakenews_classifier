# config.py
# Central configuration for Hybrid Fake News Detection System
# Researcher: Bada Toluwani Jeremiah (SEN/20/5094)
# FUTA — Department of Software Engineering

import os

# Detect environment
IS_COLAB = 'COLAB_GPU' in os.environ

if IS_COLAB:
    BASE_DIR = "/content/drive/MyDrive/fakenews_classifier"
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Paths ──────────────────────────────────────────────
DATASET_RAW      = os.path.join(BASE_DIR, "datasets", "raw", "WELFake_Dataset.csv")
DATASET_PROCESSED= os.path.join(BASE_DIR, "datasets", "processed")
MODELS_DIR       = os.path.join(BASE_DIR, "ml", "models")
EMBEDDINGS_DIR   = os.path.join(BASE_DIR, "ml", "embeddings")
VECTORIZERS_DIR  = os.path.join(BASE_DIR, "ml", "vectorizers")
NOTEBOOKS_DIR    = os.path.join(BASE_DIR, "notebooks")

# ── Model Parameters (locked per methodology) ──────────
MAX_SEQUENCE_LENGTH   = 500
TFIDF_MAX_FEATURES    = 10000
WORD2VEC_VECTOR_SIZE  = 100
WORD2VEC_WINDOW       = 5
WORD2VEC_MIN_COUNT    = 2
WORD2VEC_WORKERS      = 4
BILSTM_UNITS          = 128
DENSE_REDUCTION_UNITS = 128
FUSION_DENSE_UNITS    = 64
DROPOUT_RATE          = 0.5
BATCH_SIZE            = 64
EPOCHS                = 10
TEST_SIZE             = 0.2
RANDOM_STATE          = 42

# ── Labels ─────────────────────────────────────────────
LABEL_MAP = {0: "Real", 1: "Fake"}