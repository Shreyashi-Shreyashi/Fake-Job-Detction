
import streamlit as st
import joblib, re, shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix

nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)

@st.cache_resource
def load_all():
    tfidf     = joblib.load("models/tfidf_vectorizer.pkl")
    model     = joblib.load("models/xgboost_model.pkl")
    features  = joblib.load("models/structured_features.pkl")
    explainer = shap.TreeExplainer(model)
    return tfidf, model, features, explainer

tfidf, model, structured_features, explainer = load_all()

lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>",            " ", text)
    text = re.sub(r"&[a-z]+;",         " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]",        " ", text)
    text = re.sub(r"\s+",             " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words and len(w) > 2]
    return " ".join(words)

def get_shap_chart(X_input):
    vals  = explainer.shap_values(X_input)
    vals  = vals[0] if isinstance(vals, list) else vals.flatten()
    names = list(tfidf.get_feature_names_out()) + structured_features
    df    = pd.DataFrame({"feature": names, "shap_value": vals})
    df["abs"] = df["shap_value"].abs()
    return df.sort_values("abs", ascending=False).head(10)

st.set_page_config(page_title="Fake Job Detector",
                   page_icon="🔍", layout="centered")
st.title("🔍 Fake Job Detector")
st.markdown("Paste job details below to check if it is real or fake.")
st.divider()

title        = st.text_input("Job Title",
                              placeholder="e.g. Data Analyst")
description  = st.text_area("Job Description",  height=150,
                              placeholder="Paste the full description...")
requirements = st.text_area("Requirements",     height=100,
                              placeholder="Paste requirements...")
company_info = st.text_area("Company Profile",  height=100,
                              placeholder="Paste company info...")

c1, c2, c3 = st.columns(3)
has_logo      = c1.selectbox("Company Logo?", [1,0],
                              format_func=lambda x:"Yes" if x else "No")
has_questions = c2.selectbox("Screening Questions?", [1,0],
                              format_func=lambda x:"Yes" if x else "No")
telecommute   = c3.selectbox("Remote Job?", [0,1],
                              format_func=lambda x:"Yes" if x else "No")
c4, c5 = st.columns(2)
has_salary = c4.selectbox("Salary Listed?", [0,1],
                           format_func=lambda x:"Yes" if x else "No")
has_dept   = c5.selectbox("Department Listed?", [0,1],
                           format_func=lambda x:"Yes" if x else "No")
st.divider()

if st.button("🔍 Analyse Job Posting", use_container_width=True):
    if not description.strip():
        st.warning("Please paste at least a job description.")
    else:
        combined = " ".join([title, company_info,
                              description, requirements])
        cleaned  = clean_text(combined)
        X_text   = tfidf.transform([cleaned])
        struct   = np.array([[has_logo, has_questions, telecommute,
                              has_salary, has_dept,
                              len(title), len(cleaned)]], dtype=float)
        X_input  = hstack([X_text, csr_matrix(struct)])
        pred     = model.predict(X_input)[0]
        prob     = model.predict_proba(X_input)[0][1]
        risk     = round(float(prob)*100, 1)

        if pred == 1:
            st.error("⚠️ This job posting appears FRAUDULENT")
        else:
            st.success("✅ This job posting appears LEGITIMATE")

        m1, m2, m3 = st.columns(3)
        m1.metric("Fraud Risk",  f"{risk}%")
        m2.metric("Verdict",     "FAKE" if pred==1 else "REAL")
        m3.metric("Risk Level",
                  "🔴 High" if risk>=70 else
                  "🟡 Medium" if risk>=40 else "🟢 Low")
        st.progress(float(prob))

        st.divider()
        st.subheader("📋 Flags detected")
        flags = []
        if has_logo == 0:
            flags.append("❌ No company logo — strongest fraud signal")
        if telecommute == 1:
            flags.append("⚠️ Remote job — higher fraud rate in dataset")
        if has_salary == 0:
            flags.append("⚠️ No salary range listed")
        if has_questions == 0:
            flags.append("⚠️ No screening questions")
        if len(cleaned) < 300:
            flags.append("❌ Very short description — common in fake jobs")
        for f in flags:
            st.markdown(f)
        if not flags:
            st.markdown("✅ No structural red flags found")

        st.divider()
        st.subheader("🧠 Why did the model decide this?")
        st.caption("Red bars push toward FAKE. Blue bars push toward REAL.")

        shap_df = get_shap_chart(X_input)
        fig, ax = plt.subplots(figsize=(8,5))
        colors  = ["#FF5722" if x>0 else "#2196F3"
                   for x in shap_df["shap_value"]]
        ax.barh(shap_df["feature"], shap_df["shap_value"],
                color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("← Toward REAL          Toward FAKE →")
        ax.set_title("Top 10 factors influencing this prediction",
                     fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        fake_f = shap_df[shap_df["shap_value"]>0.1]["feature"].tolist()
        real_f = shap_df[shap_df["shap_value"]<-0.1]["feature"].tolist()
        if fake_f:
            st.markdown(f"🔴 **Fraud signals:** "
                        f"`{'`, `'.join(fake_f[:3])}`")
        if real_f:
            st.markdown(f"🟢 **Legitimacy signals:** "
                        f"`{'`, `'.join(real_f[:3])}`")

        st.caption(f"Model: XGBoost | F1: 0.85 | "
                   f"Trained on 17,880 job postings")
