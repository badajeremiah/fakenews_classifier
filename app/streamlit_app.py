# app/streamlit_app.py
# Hybrid Fake News Detection System — Streamlit Interface
# Researcher: Bada Toluwani Jeremiah (SEN/20/5094)
# FUTA — Department of Software Engineering

import os
import sys
import re
import json
import numpy as np
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Page Configuration ─────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Load Artifacts ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    import joblib

    model = load_model(os.path.join(config.MODELS_DIR, 'best_model_local.h5'))
    
    with open(os.path.join(config.VECTORIZERS_DIR, 'tokenizer.json'),
              'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(json.load(f))

    tfidf = joblib.load(os.path.join(
        config.VECTORIZERS_DIR, 'tfidf_vectorizer.pkl'))

    return model, tokenizer, tfidf


# ── Text Cleaning ──────────────────────────────────────
def clean_text(text):
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


# ── Prediction ─────────────────────────────────────────
def predict(text, model, tokenizer, tfidf):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text(text)

    # Sequence path
    seq = tokenizer.texts_to_sequences([cleaned])
    seq_padded = pad_sequences(seq,
                               maxlen=config.MAX_SEQUENCE_LENGTH,
                               padding='post',
                               truncating='post')

    # TF-IDF path
    tfidf_vec = tfidf.transform([cleaned]).toarray().astype(np.float32)

    # Predict
    prob = model.predict([seq_padded, tfidf_vec], verbose=0)[0][0]
    label = 'Fake' if prob > 0.5 else 'Real'
    confidence = prob if prob > 0.5 else 1 - prob

    return label, float(confidence), float(prob)


# ── UI ─────────────────────────────────────────────────
st.title("🔍 Hybrid Fake News Detector")
st.markdown(
    "Paste a news article or headline below. "
    "The system will classify it as **Real** or **Fake** "
    "using a Hybrid Bi-LSTM + TF-IDF deep learning model."
)
st.divider()

# Load artifacts with spinner
with st.spinner("Loading model and artifacts..."):
    model, tokenizer, tfidf = load_artifacts()

st.success("Model loaded and ready.", icon="✅")
st.divider()

# Input
user_input = st.text_area(
    "📰 Enter News Article Text:",
    height=200,
    placeholder="Paste your news article or headline here..."
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_btn = st.button("🔍 Classify", use_container_width=True)

st.divider()

# Prediction output
if classify_btn:
    if not user_input.strip():
        st.warning("Please enter some text before classifying.")
    elif len(user_input.strip().split()) < 5:
        st.warning("Please enter at least a few sentences for accurate classification.")
    else:
        with st.spinner("Analysing article..."):
            label, confidence, raw_prob = predict(
                user_input, model, tokenizer, tfidf)

        # Result display
        if label == 'Fake':
            st.error(f"### 🚨 FAKE NEWS DETECTED", icon="🚨")
            st.markdown(f"The model classified this article as **Fake** "
                        f"with a confidence of **{confidence*100:.2f}%**.")
        else:
            st.success(f"### ✅ REAL NEWS", icon="✅")
            st.markdown(f"The model classified this article as **Real** "
                        f"with a confidence of **{confidence*100:.2f}%**.")

        st.divider()

        # Probability bar
        st.markdown("#### 📊 Classification Probability")
        col_real, col_fake = st.columns(2)
        with col_real:
            st.metric("Real Probability", f"{(1-raw_prob)*100:.2f}%")
        with col_fake:
            st.metric("Fake Probability", f"{raw_prob*100:.2f}%")

        st.progress(float(raw_prob), text=f"Fake probability: {raw_prob*100:.2f}%")

        st.divider()
        st.caption(
            "⚠️ This system provides probabilistic predictions based on "
            "learned patterns in the WELFake dataset. Results should be "
            "used as a guide, not as an absolute editorial judgment."
        )