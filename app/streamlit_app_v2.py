# app/streamlit_app_v2.py
# Hybrid Fake News Detection System — Enhanced Interface
# Researcher: Bada Toluwani Jeremiah (SEN/20/5094)
# FUTA — Department of Software Engineering

import os
import sys
import re
import json
import numpy as np
import streamlit as st
import requests
from bs4 import BeautifulSoup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Page Configuration ─────────────────────────────────
st.set_page_config(
    page_title="Hybrid Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #A0A0A0;
        margin-bottom: 1.5rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 0.5rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .badge-credible {
        background-color: #1E4D2B;
        color: #4ADE80;
    }
    .badge-unverified {
        background-color: #4D3319;
        color: #FBBF24;
    }
    .highlight-fake {
        background-color: rgba(248, 113, 113, 0.2);
        border-left: 3px solid #F87171;
        padding: 0.5rem 0.8rem;
        margin: 0.4rem 0;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Hugging Face Model Repo ────────────────────────────
HF_REPO = "TheBada/fakenews-bilstm-hybrid"

# ── Known Source Credibility Lists ─────────────────────
CREDIBLE_DOMAINS = {
    "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "aljazeera.com", "npr.org", "wsj.com", "channelstv.com",
    "punchng.com", "premiumtimesng.com", "vanguardngr.com"
}


# ── Load Artifacts from Hugging Face ───────────────────
@st.cache_resource
def load_artifacts():
    from huggingface_hub import hf_hub_download
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (Input, Embedding, Bidirectional,
                                         LSTM, Dense, Dropout, Concatenate)
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    import joblib

    # Download artifacts from Hugging Face
    model_path = hf_hub_download(repo_id=HF_REPO, filename="best_model_local.h5")
    tfidf_path = hf_hub_download(repo_id=HF_REPO, filename="tfidf_vectorizer.pkl")
    tokenizer_path = hf_hub_download(repo_id=HF_REPO, filename="tokenizer.json")
    embedding_path = hf_hub_download(repo_id=HF_REPO, filename="embedding_matrix.npy")

    # Load tokenizer
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer = tokenizer_from_json(json.load(f))

    # Load embedding matrix
    embedding_matrix = np.load(embedding_path)
    vocab_size = embedding_matrix.shape[0]

    # Rebuild architecture
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
    model.load_weights(model_path)

    tfidf = joblib.load(tfidf_path)

    return model, tokenizer, tfidf


# ── Text Cleaning ──────────────────────────────────────
@st.cache_resource
def get_stopwords():
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    return set(stopwords.words('english'))


def clean_text(text):
    from nltk.tokenize import word_tokenize
    stop_words = get_stopwords()

    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


# ── URL Article Extraction ─────────────────────────────
def extract_article_from_url(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/120.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove non-content elements
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()

    # Try common article containers
    article = soup.find('article')
    if article:
        paragraphs = article.find_all('p')
    else:
        paragraphs = soup.find_all('p')

    text = ' '.join(p.get_text() for p in paragraphs)
    text = re.sub(r'\s+', ' ', text).strip()

    title = soup.find('title')
    title_text = title.get_text() if title else ""

    from urllib.parse import urlparse
    domain = urlparse(url).netloc.replace('www.', '')

    return title_text, text, domain


# ── Sentence-Level Analysis ────────────────────────────
def analyze_sentences(text, model, tokenizer, tfidf):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    sentences = sent_tokenize(text)
    # Filter very short sentences
    sentences = [s for s in sentences if len(s.split()) >= 5]

    if len(sentences) < 2:
        return []

    results = []
    for sent in sentences:
        cleaned = clean_text(sent)
        if len(cleaned.split()) < 2:
            continue

        seq = tokenizer.texts_to_sequences([cleaned])
        seq_padded = pad_sequences(seq,
                                   maxlen=config.MAX_SEQUENCE_LENGTH,
                                   padding='post',
                                   truncating='post')
        tfidf_vec = tfidf.transform([cleaned]).toarray().astype(np.float32)

        prob = model.predict([seq_padded, tfidf_vec], verbose=0)[0][0]
        results.append((sent, float(prob)))

    # Sort by fake probability descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:3]


# ── Prediction ─────────────────────────────────────────
def predict(text, model, tokenizer, tfidf):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    seq_padded = pad_sequences(seq,
                               maxlen=config.MAX_SEQUENCE_LENGTH,
                               padding='post',
                               truncating='post')

    tfidf_vec = tfidf.transform([cleaned]).toarray().astype(np.float32)

    prob = model.predict([seq_padded, tfidf_vec], verbose=0)[0][0]
    label = 'Fake' if prob > 0.5 else 'Real'
    confidence = prob if prob > 0.5 else 1 - prob

    return label, float(confidence), float(prob)


# ── Session State Init ─────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []


# ── UI Header ───────────────────────────────────────────
st.markdown('<p class="main-header">🔍 Hybrid Fake News Detector</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Bi-Directional LSTM + TF-IDF Deep Learning '
    'Classification System</p>',
    unsafe_allow_html=True
)

with st.spinner("Loading model artifacts..."):
    model, tokenizer, tfidf = load_artifacts()

st.divider()

# ── Input Mode Selection ────────────────────────────────
input_mode = st.radio(
    "Choose input method:",
    ["📝 Paste Text", "🔗 Enter URL"],
    horizontal=True
)

article_text = ""
article_title = ""
domain = None

if input_mode == "📝 Paste Text":
    article_text = st.text_area(
        "Enter News Article Text:",
        height=200,
        placeholder="Paste your news article or headline here..."
    )
else:
    url_input = st.text_input(
        "Enter Article URL:",
        placeholder="https://example.com/news/article-title"
    )
    if url_input:
        with st.spinner("Fetching article..."):
            try:
                article_title, article_text, domain = extract_article_from_url(url_input)
                if article_text:
                    st.success(f"Article fetched: {len(article_text.split())} words")
                    with st.expander("📄 Preview fetched text"):
                        st.write(article_title)
                        st.write(article_text[:1000] + "..." if len(article_text) > 1000 else article_text)
                else:
                    st.warning("Could not extract article text from this URL. "
                              "The site may block automated access or use "
                              "JavaScript rendering.")
            except Exception as e:
                st.error(f"Failed to fetch URL: {e}")

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    classify_btn = st.button("🔍 Classify", use_container_width=True)

st.divider()

# ── Prediction Output ───────────────────────────────────
if classify_btn:
    if not article_text.strip():
        st.warning("Please enter text or a valid URL before classifying.")
    elif len(article_text.strip().split()) < 5:
        st.warning("Please provide more text for accurate classification.")
    else:
        with st.spinner("Analysing article..."):
            label, confidence, raw_prob = predict(
                article_text, model, tokenizer, tfidf)

        # Source credibility badge
        if domain:
            if domain in CREDIBLE_DOMAINS:
                st.markdown(
                    f'<span class="source-badge badge-credible">'
                    f'✅ Source: {domain} (recognized credible outlet)</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<span class="source-badge badge-unverified">'
                    f'⚠️ Source: {domain} (not in known credible list)</span>',
                    unsafe_allow_html=True
                )

        # Result display
        if label == 'Fake':
            st.error("### 🚨 FAKE NEWS DETECTED", icon="🚨")
            st.markdown(f"The model classified this article as **Fake** "
                        f"with a confidence of **{confidence*100:.2f}%**.")
        else:
            st.success("### ✅ REAL NEWS", icon="✅")
            st.markdown(f"The model classified this article as **Real** "
                        f"with a confidence of **{confidence*100:.2f}%**.")

        # Low confidence warning
        if 0.45 <= raw_prob <= 0.55:
            st.warning(
                "⚠️ **Low confidence classification.** The model's "
                "prediction is close to the decision boundary (50%). "
                "This result should be treated with caution and "
                "verified through additional sources.",
                icon="⚠️"
            )

        st.divider()

        # Probability bar
        st.markdown("#### 📊 Classification Probability")
        col_real, col_fake = st.columns(2)
        with col_real:
            st.metric("Real Probability", f"{(1-raw_prob)*100:.2f}%")
        with col_fake:
            st.metric("Fake Probability", f"{raw_prob*100:.2f}%")

        st.progress(float(raw_prob), text=f"Fake probability: {raw_prob*100:.2f}%")

        # Sentence-level analysis
        st.divider()
        st.markdown("#### 🔬 Most Suspicious Sentences")
        sentence_results = analyze_sentences(article_text, model, tokenizer, tfidf)

        if sentence_results:
            for sent, prob in sentence_results:
                if prob > 0.5:
                    st.markdown(
                        f'<div class="highlight-fake">'
                        f'<b>{prob*100:.1f}% Fake-leaning:</b><br>{sent}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"**{prob*100:.1f}% Fake-leaning:** {sent}")
        else:
            st.caption("Article too short for sentence-level analysis.")

        # Add to history
        st.session_state.history.insert(0, {
            'text_preview': article_text[:80] + "...",
            'label': label,
            'confidence': confidence,
            'domain': domain
        })
        st.session_state.history = st.session_state.history[:10]

        st.divider()
        st.caption(
            "⚠️ This system provides probabilistic predictions based on "
            "learned patterns in the WELFake dataset. Results should be "
            "used as a guide, not as an absolute editorial judgment. "
            "Source credibility badges reflect a limited known-domain "
            "list and are not a comprehensive verification system."
        )

# ── Classification History ──────────────────────────────
if st.session_state.history:
    st.divider()
    with st.expander(f"📜 Classification History ({len(st.session_state.history)})"):
        for i, item in enumerate(st.session_state.history):
            icon = "🚨" if item['label'] == 'Fake' else "✅"
            domain_text = f" | Source: {item['domain']}" if item['domain'] else ""
            st.markdown(
                f"{icon} **{item['label']}** ({item['confidence']*100:.1f}%) "
                f"— {item['text_preview']}{domain_text}"
            )