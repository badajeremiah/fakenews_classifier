# fakenews_classifier
# Hybrid Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![License](https://img.shields.io/badge/License-MIT-green)

**Researcher:** Bada Toluwani Jeremiah (SEN/20/5094)  
**Institution:** Federal University of Technology, Akure (FUTA)  
**Department:** Software Engineering  
**Supervisor:** Dr. O.O. Ogunbodede  
**Academic Year:** 2025/2026  

---

## Overview

A research-grade Hybrid Deep Learning system for automated fake news 
detection, combining Bi-Directional LSTM (semantic context) with 
TF-IDF (statistical features) to overcome the accuracy plateau of 
traditional classifiers stalling at 63–70%.

---

## System Architecture
Input Text
│
├──► Path A (Semantic)
│         Word2Vec Embeddings (100-dim)
│              │
│         Bi-Directional LSTM (128 units)
│              │
│         Output: (batch, 256)
│
├──► Path B (Statistical)
│         TF-IDF Vectorizer (10,000 features)
│              │
│         Dense Reduction Layer (128 units, ReLU)
│              │
│         Output: (batch, 128)
│
└──► Fusion Layer
Concatenation → (batch, 384)
Dense (64 units, ReLU)
Dropout (0.5)
Dense (1 unit, Sigmoid)
│
0 = Real News
1 = Fake News
---

## Dataset

**WELFake Dataset** — 72,134 balanced news articles  
Merged from: Kaggle, McIntire, Reuters, BuzzFeed  
Preprocessing: Tokenization, stopword removal, padding (500 tokens)

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.15 / Keras |
| Embeddings | Gensim Word2Vec |
| Feature Extraction | Scikit-learn TF-IDF |
| NLP Preprocessing | NLTK |
| Web Interface | Streamlit |
| API Backend | FastAPI |
| Training Environment | Google Colab Pro |

---

## Repository Structure
fakenews_classifier/
├── backend/              # FastAPI inference server
├── frontend/             # Streamlit web application
├── ml/
│   ├── models/           # Saved model weights (.h5)
│   ├── embeddings/       # Trained Word2Vec model
│   ├── vectorizers/      # Saved TF-IDF vectorizer
│   ├── training/         # Training scripts
│   └── evaluation/       # Metrics and confusion matrix
├── datasets/
│   ├── raw/              # Original WELFake CSV
│   └── processed/        # Tokenized and padded arrays
├── notebooks/            # Google Colab training notebooks
├── docs/
│   ├── architecture/     # System diagrams
│   └── report/           # FYP report chapters
├── tests/                # Unit tests
├── config.py             # Environment configuration
└── requirements.txt      # Dependencies
---

## Evaluation Targets

| Metric | Baseline (Naive Bayes) | Target (Hybrid BiLSTM) |
|--------|----------------------|----------------------|
| Accuracy | ~70% | >90% |
| Precision | ~70% | >90% |
| Recall | ~70% | >90% |
| F1-Score | ~70% | >90% |

---

## Baselines

- Naive Bayes + TF-IDF (10,000 features)
- Logistic Regression + TF-IDF (10,000 features)

---

## Research Gaps Addressed

1. **Context Blindness** — BiLSTM reads sentences forward and 
   backward, capturing sarcasm and semantic dependencies
2. **Accuracy Plateau** — Hybrid fusion breaks past the 70% ceiling
3. **Dataset Fragmentation** — WELFake consolidates 4 sources 
   into one balanced corpus

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/badajeremiah/fakenews_classifier.git
cd fakenews_classifier

# Create virtual environment (Python 3.11 required)
py -3.11 -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Place WELFake_Dataset.csv in:
# datasets/raw/WELFake_Dataset.csv
```

---

## References

- Adebayo, P. O., Oladipo, I. D., Abdulraheem, M., Balogun, G. B., & Tomori, A. R. 
  (2025). Fake news detection model using ensemble bi-directional LSTM and support 
  vector machines. *OAUSTECH Journal of Engineering and Intelligent Technology, 1*(1).

- Ahmed, A. A. A., Aljarbouh, A., Donepudi, P. K., & Choi, M. S. (2019). Detecting 
  fake news using machine learning: A systematic literature review. *Journal of 
  Information Systems and Informatics, 2*(1).

- Alameri, S. A., & Mohd, M. (2020). Comparison of fake news detection using machine 
  learning and deep learning techniques. *Semantic Scholar*.

- Alnabhan, M., & Branco, P. (2024). Fake news detection using deep learning: A 
  systematic literature review. *[Journal]*.

- Ghosh, S., & Shah, C. (2019). Toward automatic fake news classification. 
  *Proceedings of the 52nd Hawaii International Conference on System Sciences*.

- Ghosh, H., Sen, D., Shah, S. B., Oloruntoba, O., Yarramaneni, C. M., Manga, M. K., 
  & Agarwal, P. (2023). Hybrid fake news detection using machine learning and deep 
  learning. *ResearchGate*.

- Kwaknat, M. D., & Gurumdimma, N. (2025). Fake news detection: A machine learning 
  approach. *International Journal of Research and Innovation in Applied Science, 
  10*(1).

- Kumari, R., & Singh, M. K. (2024). A deep learning multimodal framework for fake 
  news detection. *[Journal]*.

- Ojha, R. P., et al. (2023). Controlling of fake information dissemination in online 
  social networks: An epidemiological approach. *[Journal]*.

- Roumeliotis, K. I., Tselikas, N. D., & Nasiopoulos, D. K. (2025). Fake news 
  detection and classification: A comparative study of convolutional neural networks, 
  large language models, and natural language processing models. *Future Internet, 
  17*(1), 28.
  