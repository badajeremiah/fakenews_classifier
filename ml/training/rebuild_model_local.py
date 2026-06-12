# ml/training/rebuild_model_local.py
# Rebuilds model architecture locally and loads weights from Colab-saved file

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Bidirectional,
                                     LSTM, Dense, Dropout, Concatenate)

print("Loading embedding matrix...")
embedding_matrix = np.load(os.path.join(
    config.EMBEDDINGS_DIR, 'embedding_matrix.npy'))
vocab_size = embedding_matrix.shape[0]
print(f"Embedding matrix: {embedding_matrix.shape}")

print("Building model architecture...")
seq_input = Input(shape=(config.MAX_SEQUENCE_LENGTH,), name='sequence_input')
embedding = Embedding(
    input_dim=vocab_size,
    output_dim=config.WORD2VEC_VECTOR_SIZE,
    weights=[embedding_matrix],
    trainable=True,
    name='word2vec_embedding'
)(seq_input)
bilstm = Bidirectional(
    LSTM(config.BILSTM_UNITS, return_sequences=False),
    name='bilstm'
)(embedding)

tfidf_input = Input(shape=(config.TFIDF_MAX_FEATURES,), name='tfidf_input')
tfidf_dense = Dense(config.DENSE_REDUCTION_UNITS,
                    activation='relu',
                    name='tfidf_reduction')(tfidf_input)

merged = Concatenate(name='fusion')([bilstm, tfidf_dense])
dense1 = Dense(config.FUSION_DENSE_UNITS,
               activation='relu',
               name='fusion_dense')(merged)
dropout = Dropout(config.DROPOUT_RATE, name='dropout')(dense1)
output = Dense(1, activation='sigmoid', name='output')(dropout)

model = Model(inputs=[seq_input, tfidf_input], outputs=output,
              name='hybrid_fake_news_detector')
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Loading weights from Colab-saved model...")
model.load_weights(os.path.join(config.MODELS_DIR, 'best_model.h5'))

print("Saving locally compatible model...")
local_save_path = os.path.join(config.MODELS_DIR, 'best_model_local.h5')
model.save(local_save_path)
print(f"Saved to: {local_save_path}")
print("Done.")