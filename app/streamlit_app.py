"""
🎬 Analyse de sentiment — App Streamlit
Prédiction en temps réel avec Baseline TF-IDF et CamemBERT.

Usage : streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Sentiment Analysis", page_icon="🎬", layout="wide")

ROOT = Path(__file__).parent.parent
MODELS = ROOT / "models"

# ── Initialiser la clé du widget text_area ──
if "critique" not in st.session_state:
    st.session_state["critique"] = "Un film absolument magnifique, avec des acteurs talentueux et une histoire captivante. À voir !"

# ── Chargement des modèles ──

@st.cache_resource
def load_baseline():
    model_path = MODELS / "baseline_model.pkl"
    tfidf_path = MODELS / "tfidf_vectorizer.pkl"
    if not model_path.exists() or not tfidf_path.exists():
        return None, None
    return joblib.load(model_path), joblib.load(tfidf_path)


@st.cache_resource
def load_camembert():
    model_path = MODELS / "camembert_finetuned"
    if not model_path.exists():
        return None, None
    try:
        from transformers import CamembertTokenizer, CamembertForSequenceClassification
        tokenizer = CamembertTokenizer.from_pretrained(str(model_path))
        model = CamembertForSequenceClassification.from_pretrained(str(model_path))
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.warning(f"Erreur CamemBERT : {e}")
        return None, None


baseline_model, tfidf = load_baseline()
camembert_model, camembert_tokenizer = load_camembert()

# ── Exemples pré-remplis ──

examples = {
    "😍 Enthousiaste": "Un chef-d'œuvre ! Les acteurs sont incroyables et la réalisation est parfaite. Je recommande à 100%.",
    "🤔 Nuancé": "Pas mal dans l'ensemble, quelques longueurs mais une belle fin qui rattrape tout. Correct sans plus.",
    "😏 Ironique": "Ah oui, vraiment génial ce film. Si vous aimez vous ennuyer pendant 2h, foncez.",
    "👎 Négatif": "Nul à pleurer. Scénario vide, acteurs mauvais, une vraie perte de temps et d'argent.",
    "😐 Mitigé": "De bons moments mais aussi des passages incompréhensibles. Je suis très partagé sur ce film.",
}


def set_example(example_text):
    """Callback pour remplir la zone de texte."""
    st.session_state["critique"] = example_text


# ── Interface ──

st.title("🎬 Analyse de sentiment — Critiques de films")
st.markdown("Saisissez une critique de film en français et comparez les prédictions du **baseline TF-IDF** et de **CamemBERT**.")

# Exemples à tester
st.subheader("💡 Exemples à tester")
cols = st.columns(len(examples))
for i, (label, example) in enumerate(examples.items()):
    with cols[i]:
        st.button(label, on_click=set_example, args=(example,), use_container_width=True)

# Zone de texte — utilise key= pour que session_state la contrôle
text = st.text_area(
    "Votre critique :",
    height=150,
    key="critique",
)

col1, col2 = st.columns(2)

if st.button("🔮 Analyser le sentiment", type="primary", use_container_width=True):

    if not text.strip():
        st.warning("Entrez une critique d'abord.")
        st.stop()

    # ── Baseline TF-IDF ──
    with col1:
        st.subheader("📊 Baseline TF-IDF")
        if baseline_model and tfidf:
            X_tfidf = tfidf.transform([text])
            pred_baseline = baseline_model.predict(X_tfidf)[0]

            if hasattr(baseline_model, "predict_proba"):
                proba = baseline_model.predict_proba(X_tfidf)[0]
                conf_baseline = max(proba)
            elif hasattr(baseline_model, "decision_function"):
                score = baseline_model.decision_function(X_tfidf)[0]
                conf_baseline = abs(score) / (abs(score) + 1)
            else:
                conf_baseline = 0.5

            sentiment = "✅ Positif" if pred_baseline == 1 else "❌ Négatif"
            st.metric("Sentiment", sentiment)
            st.metric("Confiance", f"{conf_baseline*100:.0f}%")

            if hasattr(baseline_model, "coef_"):
                feature_names = tfidf.get_feature_names_out()
                tfidf_vector = X_tfidf.toarray()[0]
                nonzero = tfidf_vector > 0
                if nonzero.any():
                    coefs = baseline_model.coef_[0] if baseline_model.coef_.ndim > 1 else baseline_model.coef_
                    contributions = tfidf_vector * coefs
                    top_idx = np.abs(contributions).argsort()[-5:][::-1]
                    st.caption("Mots les plus influents :")
                    for idx in top_idx:
                        if tfidf_vector[idx] > 0:
                            direction = "↑" if contributions[idx] > 0 else "↓"
                            st.text(f"  {direction} {feature_names[idx]} ({contributions[idx]:+.3f})")
        else:
            st.warning("Baseline non disponible. Lance le notebook 1.")

    # ── CamemBERT ──
    with col2:
        st.subheader("🤖 CamemBERT")
        if camembert_model and camembert_tokenizer:
            import torch

            inputs = camembert_tokenizer(
                text, return_tensors="pt", truncation=True,
                padding="max_length", max_length=256,
            )

            with torch.no_grad():
                outputs = camembert_model(**inputs)
                logits = outputs.logits
                proba_camembert = torch.softmax(logits, dim=1)[0].numpy()

            pred_camembert = proba_camembert.argmax()
            conf_camembert = proba_camembert.max()

            sentiment = "✅ Positif" if pred_camembert == 1 else "❌ Négatif"
            st.metric("Sentiment", sentiment)
            st.metric("Confiance", f"{conf_camembert*100:.0f}%")

            st.caption(f"P(négatif) = {proba_camembert[0]:.3f}")
            st.caption(f"P(positif) = {proba_camembert[1]:.3f}")
        else:
            st.warning("CamemBERT non disponible. Lance le notebook 2.")
