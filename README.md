# 🎬 Analyse de sentiment — NLP classique vs. Deep Learning

> Fine-tuner un modèle Transformer sur des critiques de films en français et le comparer aux approches NLP classiques.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-DL-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Contexte

L'analyse de sentiment est une tâche NLP fondamentale en entreprise : avis clients, veille média, analyse de feedback. Ce projet confronte les approches classiques (TF-IDF + modèles linéaires) aux Transformers pré-entraînés (CamemBERT) sur des critiques de films en français — un terrain où la nuance, l'ironie et le vocabulaire spécialisé rendent la tâche particulièrement exigeante.

## 🎯 Objectifs

- Explorer un corpus de ~200k critiques de films en français (Allociné)
- Construire un baseline NLP classique (TF-IDF + Logistic Regression / SVM)
- Fine-tuner CamemBERT (BERT pour le français) sur la classification de sentiment
- Comparer les performances et analyser les erreurs
- Déployer une démo Streamlit de prédiction en temps réel

## 🔧 Stack technique

| Outil | Usage |
|-------|-------|
| **Python 3.10+** | Langage principal |
| **Scikit-learn** | Baseline TF-IDF + modèles classiques |
| **Hugging Face Transformers** | CamemBERT, tokenizer, Trainer |
| **PyTorch** | Backend Deep Learning |
| **Datasets (HF)** | Chargement du corpus Allociné |
| **Streamlit** | Démo interactive |

## 📁 Structure

```
06-sentiment-nlp/
├── README.md
├── requirements.txt
├── .gitignore / LICENSE
├── data/
│   └── download_data.py
├── notebooks/
│   ├── 01_eda_baseline.ipynb          ← EDA + TF-IDF
│   └── 02_camembert_finetuning.ipynb  ← Fine-tuning (GPU recommandé)
├── app/
│   └── streamlit_app.py
├── models/                            ← Gitignored
├── assets/
└── scripts/
```

## 🚀 Démarrage

```bash
pip install -r requirements.txt
python data/download_data.py
jupyter notebook notebooks/01_eda_baseline.ipynb
```

### Fine-tuning CamemBERT (GPU recommandé)

```bash
# En local avec GPU :
jupyter notebook notebooks/02_camembert_finetuning.ipynb

# Sur Google Colab (gratuit) :
# Uploader le notebook 02 → Runtime → Change runtime type → GPU
```

### App Streamlit

```bash
streamlit run app/streamlit_app.py
```

## 📊 Démo live

> [🔗 Voir l'app sur Streamlit Cloud](URL_STREAMLIT)

## 📄 Source des données

- **Allociné Reviews — Hugging Face Datasets**
- ~200k critiques de films en français
- Labels : positif / négatif
- URL : [huggingface.co/datasets/allocine](https://huggingface.co/datasets/allocine)
- Modèle : [CamemBERT-base](https://huggingface.co/camembert-base)

## 📜 Licence

MIT — voir [LICENSE](LICENSE).
