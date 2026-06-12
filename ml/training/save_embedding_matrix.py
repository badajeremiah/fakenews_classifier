# ml/training/save_embedding_matrix.py
# Precomputes and saves embedding matrix to avoid gensim on Colab

import os
import sys
import json
import numpy as np
import joblib
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import tokenizer_from_json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

print("Loading tokenizer...")
with open(os.path.join(config.VECTORIZERS_DIR, 'tokenizer.json'),
          'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(json.load(f))

print("Loading Word2Vec...")
w2v = Word2Vec.load(os.path.join(config.EMBEDDINGS_DIR, 'word2vec.model'))

print("Building embedding matrix...")
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros(
    (vocab_size, config.WORD2VEC_VECTOR_SIZE), dtype=np.float32)

found = 0
for word, idx in tokenizer.word_index.items():
    if word in w2v.wv:
        embedding_matrix[idx] = w2v.wv[word].astype(np.float32)
        found += 1

del w2v
save_path = os.path.join(config.EMBEDDINGS_DIR, 'embedding_matrix.npy')
np.save(save_path, embedding_matrix)

print(f"Vocabulary size:         {vocab_size}")
print(f"Words found in Word2Vec: {found}/{vocab_size}")
print(f"Matrix shape:            {embedding_matrix.shape}")
print(f"Matrix size:             {embedding_matrix.nbytes / 1e6:.1f} MB")
print(f"Saved to:                {save_path}")