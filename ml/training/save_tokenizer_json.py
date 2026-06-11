# ml/training/save_tokenizer_json.py
# Re-saves Keras tokenizer as JSON for cross-version compatibility

import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
import config

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
import json

print("Loading tokenizer.pkl...")
tokenizer = joblib.load(os.path.join(
    config.VECTORIZERS_DIR, 'tokenizer.pkl'))

print("Saving as tokenizer.json...")
tokenizer_json = tokenizer.to_json()
save_path = os.path.join(config.VECTORIZERS_DIR, 'tokenizer.json')
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(tokenizer_json, f, ensure_ascii=False)

print(f"Saved to: {save_path}")
print(f"Vocabulary size: {len(tokenizer.word_index)}")