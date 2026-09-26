🇫🇷 [Version française](README_FR.md)

# 🎬 Sentiment Analysis — Classical NLP vs. Deep Learning

> Fine-tuning a Transformer model on French movie reviews and comparing it to classical NLP approaches.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-DL-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Context

Sentiment analysis is a fundamental NLP task in business: customer reviews, media monitoring, feedback analysis. This project compares classical approaches (TF-IDF + linear models) against pre-trained Transformers (CamemBERT) on French movie reviews — a domain where nuance, irony, and specialized vocabulary make the task particularly challenging.

## 🎯 Objectives

- Explore a corpus of ~200k French movie reviews (Allociné)
- Build a classical NLP baseline (TF-IDF + Logistic Regression / SVM)
- Fine-tune CamemBERT (BERT for French) on sentiment classification
- Compare performance and analyze errors
- Deploy a real-time prediction Streamlit demo

## 🔧 Tech Stack

| Tool | Usage |
|------|-------|
| **Python 3.10+** | Main language |
| **Scikit-learn** | TF-IDF baseline + classical models |
| **Hugging Face Transformers** | CamemBERT, tokenizer, Trainer |
| **PyTorch** | Deep Learning backend |
| **Datasets (HF)** | Allociné corpus loading |
| **Streamlit** | Interactive demo |

## 📁 Structure

```
06-sentiment-nlp/
├── README.md                          ← This file
├── README_FR.md                       ← French version
├── requirements.txt
├── .gitignore / LICENSE
├── data/
│   └── download_data.py
├── notebooks/
│   ├── 01_eda_baseline.ipynb          ← EDA + TF-IDF
│   └── 02_camembert_finetuning.ipynb  ← Fine-tuning (GPU recommended)
├── app/
│   └── streamlit_app.py
├── models/                            ← Gitignored
├── assets/
└── scripts/
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python data/download_data.py
jupyter notebook notebooks/01_eda_baseline.ipynb
```

### Fine-tuning CamemBERT (GPU recommended)

```bash
# Locally with GPU:
jupyter notebook notebooks/02_camembert_finetuning.ipynb

# On Google Colab (free):
# Upload notebook 02 → Runtime → Change runtime type → GPU
```

### Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## 📊 Live Demo

> [🔗 View the app on Streamlit Cloud](https://diogoa78-07-demonstrateur-hsrfdt8xqoiobofkvqmaqd.streamlit.app/)

## 📄 Data Source

- **Allociné Reviews — Hugging Face Datasets**
- ~200k French movie reviews
- Labels: positive / negative
- URL: [huggingface.co/datasets/allocine](https://huggingface.co/datasets/allocine)
- Model: [CamemBERT-base](https://huggingface.co/camembert-base)

## 📜 License

MIT — see [LICENSE](LICENSE).
