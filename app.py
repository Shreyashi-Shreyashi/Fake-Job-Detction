
import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load models

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import nltk
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
@st.cache_resource
def load_models():
    tfidf   = joblib.load('models/tfidf_vectorizer.pkl')
    model   = joblib.load('models/xgboost_model.pkl')
    return tfidf, model

tfidf, model = load_models()

# Text cleaner — same as Phase 3
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>',           ' ', text)
    text = re.sub(r'&[a-z]+;',        ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]',       ' ', text)
    text = re.sub(r'\s+',            ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words and len(w) > 2]
    return ' '.join(words)

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title='Fake Job Detector',
    page_icon='🔍',
    layout='centered'
)

st.title('🔍 Fake Job Detector')
st.markdown('Paste job details below to check if it is real or fake.')
st.divider()

# ── Input fields ──────────────────────────────────────
title        = st.text_input('Job Title', placeholder='e.g. Data Analyst')
description  = st.text_area('Job Description', height=150,
                             placeholder='Paste the full job description...')
requirements = st.text_area('Requirements', height=100,
                             placeholder='Paste requirements...')
company_info = st.text_area('Company Profile', height=100,
                             placeholder='Paste company info...')

col1, col2, col3 = st.columns(3)
has_logo      = col1.selectbox('Company Logo?', [1, 0],
                               format_func=lambda x: 'Yes' if x else 'No')
has_questions = col2.selectbox('Screening Questions?', [1, 0],
                               format_func=lambda x: 'Yes' if x else 'No')
telecommute   = col3.selectbox('Remote Job?', [0, 1],
                               format_func=lambda x: 'Yes' if x else 'No')

col4, col5 = st.columns(2)
has_salary    = col4.selectbox('Salary Listed?', [0, 1],
                               format_func=lambda x: 'Yes' if x else 'No')
has_dept      = col5.selectbox('Department Listed?', [0, 1],
                               format_func=lambda x: 'Yes' if x else 'No')

st.divider()

# ── Predict button ────────────────────────────────────
if st.button('🔍 Analyse Job Posting', use_container_width=True):

    if not description.strip():
        st.warning('Please paste at least a job description.')

    else:
        # Combine and clean text
        combined = ' '.join([title, company_info, description, requirements])
        cleaned  = clean_text(combined)

        # TF-IDF transform
        X_text = tfidf.transform([cleaned])

        # Structured features — must match Phase 3 order exactly
        struct_vals = np.array([[
            has_logo,
            has_questions,
            telecommute,
            has_salary,
            has_dept,
            len(title),
            len(cleaned)
        ]], dtype=float)

        X_input = hstack([X_text, csr_matrix(struct_vals)])

        # Predict
        pred     = model.predict(X_input)[0]
        prob     = model.predict_proba(X_input)[0][1]
        risk_pct = round(prob * 100, 1)

        # ── Results ───────────────────────────────────
        if pred == 1:
            st.error('⚠️  This job posting appears FRAUDULENT')
        else:
            st.success('✅  This job posting appears LEGITIMATE')

        # Metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric('Fraud Risk', f'{risk_pct}%')
        c2.metric('Verdict',    'FAKE' if pred == 1 else 'REAL')

        if risk_pct >= 70:
            c3.metric('Risk Level', '🔴 High')
        elif risk_pct >= 40:
            c3.metric('Risk Level', '🟡 Medium')
        else:
            c3.metric('Risk Level', '🟢 Low')

        st.progress(float(prob))

        # ── Red flags ─────────────────────────────────
        st.divider()
        st.subheader('📋 Flags detected')
        flags = []
        if has_logo == 0:
            flags.append('❌ No company logo — strongest fraud signal')
        if telecommute == 1:
            flags.append('⚠️  Remote job — higher fraud rate in dataset')
        if has_salary == 0:
            flags.append('⚠️  No salary range listed')
        if has_questions == 0:
            flags.append('⚠️  No screening questions')
        if len(cleaned) < 300:
            flags.append('❌ Very short description — common in fake jobs')

        if flags:
            for flag in flags:
                st.markdown(flag)
        else:
            st.markdown('✅ No structural red flags found')

        st.caption('Model: XGBoost | F1: 0.85 | Trained on 17,880 job postings')
