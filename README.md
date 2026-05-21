# Fake Job Posting Detection

A machine learning project to detect fraudulent job postings using NLP, SQL analysis, and XGBoost — deployed as a live Streamlit web application.

🔗 **[Live Demo](https://fake-job-detction-cyivu6wugcfulmwyyjhadt.streamlit.app/)**

---

## Overview

Fake job postings are a growing problem, disproportionately targeting entry-level and part-time job seekers. This project builds an end-to-end fraud detection pipeline on 17,880 real-world job postings — from exploratory data analysis and SQL querying through to model deployment — using a TF-IDF + XGBoost classification approach.

**Key Result:** XGBoost achieved **F1-score of 0.85** and **98.6% accuracy** on the fraudulent class, selected as the production model powering the live app.

---

## Table of Contents

- [Dataset](#dataset)
- [Project Phases](#project-phases)
  - [Phase 1 — EDA](#phase-1--eda)
  - [Phase 2 — SQL Analysis](#phase-2--sql-analysis)
  - [Phase 3 — Text Preprocessing & TF-IDF](#phase-3--text-preprocessing--tf-idf)
  - [Phase 4 — Train/Test Split](#phase-4--traintest-split)
  - [Phase 5 — Model Training & Comparison](#phase-5--model-training--comparison)
  - [Phase 6 — Deployment](#phase-6--deployment)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [How to Run Locally](#how-to-run-locally)
- [Live App](#live-app)

---

## Dataset

- **Source:** [Fake Job Postings Dataset — Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- **Size:** 17,880 job postings
- **Class distribution:** 95.2% real, 4.8% fraudulent (heavily imbalanced)

---

## Project Phases

### Phase 1 — EDA

Explored the full dataset using Pandas, Matplotlib and Seaborn to understand class distribution, missing values, correlations and fraud patterns.

**Key findings:**
- Only 4.8% of postings are fraudulent — heavily imbalanced dataset
- `has_company_logo` is the strongest single fraud signal (correlation: -0.26)
- Fake job postings have ~600 fewer characters of combined text than real ones
- 84% of `salary_range` values are missing — converted to a binary flag
- Oil & Energy (38%) and Accounting (36%) industries have the highest fraud rates

EDA defined every downstream decision: which features to engineer, which metric to optimise (F1, not accuracy), and what patterns the model should learn.

---

### Phase 2 — SQL Analysis

Loaded the dataset into SQLite and wrote 5 analytical queries covering fraud rate by industry, experience level, employment type and country.

**Key findings:**
- Part-time roles have a 9.3% fraud rate vs 4.2% for full-time
- Entry-level workers are disproportionately targeted (6.6% fraud rate)
- Mid-senior level has the lowest fraud rate (2.97%) — experienced candidates verify companies more carefully
- Location data was inconsistent; country extraction flagged as a data quality issue

SQL findings directly informed feature engineering decisions in Phase 3.

---

### Phase 3 — Text Preprocessing & TF-IDF

Combined 5 text columns into one unified field. Cleaned text by removing HTML tags, punctuation and stopwords, then lemmatised words to root form. Applied TF-IDF vectorisation (10,000 features, unigrams + bigrams) and merged with 7 structured features into a final matrix of **17,880 × 10,007**.

**Key decisions:**
- `fit_transform` on training data only, `transform` on test — prevents data leakage
- Structured features included: `has_company_logo`, `has_questions`, `salary_flag`, `text_length` and employment/experience encodings

---

### Phase 4 — Train/Test Split

80/20 stratified split to preserve the 4.8% fraud ratio in both sets.

| Split | Rows | Fraudulent |
|-------|------|------------|
| Train | 14,304 | 693 (4.84%) |
| Test | 3,576 | 173 (4.84%) |

Stratification ensures the model is evaluated on a realistic fraud distribution.

---

### Phase 5 — Model Training & Comparison

Trained 3 models with `class_weight='balanced'` to handle class imbalance. Evaluated using Precision, Recall and F1 on the fraudulent class.

| Model | Precision | Recall | F1 | Accuracy |
|---|---|---|---|---|
| Logistic Regression | 0.571 | 0.925 | 0.706 | 96.3% |
| Random Forest | 1.000 | 0.607 | 0.755 | 98.1% |
| **XGBoost** | **0.913** | **0.792** | **0.848** | **98.6%** |

**XGBoost selected** as the production model — best balance of precision and recall (highest F1). Random Forest had perfect precision but missed 39% of fake jobs; Logistic Regression caught more frauds but flagged too many real jobs as fake.

---

### Phase 6 — Deployment

Built a Streamlit web app that replicates the full training pipeline at inference time:
1. User inputs job posting details
2. Text is combined, cleaned and TF-IDF transformed
3. Structured features are added
4. XGBoost predicts fraud probability
5. App displays: fraud risk score, verdict, risk level, and detected flags

Deployed on **Streamlit Cloud** via GitHub integration — auto-redeploys on every push.

**Validation:**
- Obvious fake posting → 99.9% fraud probability
- Legitimate internship listing → 1.5% fraud probability

---

## Results

| Metric | Value |
|---|---|
| Best Model | XGBoost |
| F1-Score (Fake class) | **0.85** |
| Accuracy | **98.6%** |
| Precision (Fake) | 0.913 |
| Recall (Fake) | 0.792 |

> **Note:** Accuracy alone is misleading on imbalanced datasets. A model predicting "real" for everything would score 95.2% — F1 on the fake class is the meaningful metric here.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Analysis | Pandas, NumPy, Matplotlib, Seaborn |
| SQL | SQLite3 |
| NLP | Scikit-learn (TF-IDF) |
| Modelling | Scikit-learn, XGBoost |
| Deployment | Streamlit, Streamlit Cloud |
| Version Control | Git, GitHub |

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Shreyashi-Shreyashi/Fake-Job-Detction.git
cd Fake-Job-Detction

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## Live App

🔗 [https://fake-job-detction-cyivu6wugcfulmwyyjhadt.streamlit.app/](https://fake-job-detction-cyivu6wugcfulmwyyjhadt.streamlit.app/)

Paste any job posting into the app and get an instant fraud probability score with detected risk flags.

---

*Built by [Shreyashi](https://github.com/Shreyashi-Shreyashi)*
