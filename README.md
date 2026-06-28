# 🏥 Health Prediction System
### AI/ML-Powered Diabetes Risk Assessment & Patient Management

A full-stack patient management web application that uses a trained **Random Forest ML model** to assess diabetes risk from blood test biomarkers. Built with Streamlit, scikit-learn, and SQLAlchemy.

---

## 🎯 Project Overview

This system allows healthcare users to register patients, input blood test results, and instantly receive a structured **Clinical Assessment Report** powered by machine learning. All records are stored in a local SQLite database with full CRUD support.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 Dashboard | View all patients, aggregate stats, and individual reports |
| ➕ Patient Registration | Add patients with personal info and blood test values |
| 🤖 AI Risk Prediction | Random Forest model classifies diabetes risk into 4 levels |
| 🩸 Biomarker Analysis | Colour-coded breakdown of Glucose, HbA1c, and Cholesterol |
| 📋 Clinical Report | Structured report with risk level, analysis, and recommendations |
| ✏️ Edit & Re-Analyse | Update patient values and regenerate the ML report |
| 🗑️ Delete Records | Remove patients with confirmation guard |
| ✅ Input Validation | Email, date of birth, and numeric range validation |

---

## 🧠 ML Model

| Detail | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Estimators | 100 trees, max depth 8, min samples leaf 5 |
| Class Balancing | `class_weight="balanced"` to handle imbalanced data |
| Features | Age, Fasting Glucose, HbA1c (%), Total Cholesterol |
| Training Samples | 5,000 synthetic samples (WHO / ADA / Kaggle distributions) |
| Train/Test Split | 80% / 20% stratified |
| Dataset Reference | [Diabetes Health Dataset — Rabie El Kharoua, Kaggle 2024](https://www.kaggle.com/datasets/rabieelkharoua/diabetes-health-dataset-analysis) |
| Output | Binary: Diabetic / No Diabetes + risk probability % |

### Risk Classification Levels

| Level | Condition |
|---|---|
| ✅ LOW RISK | No diabetes detected, all markers normal |
| ⚠️ MILD RISK | Borderline values, lifestyle changes advised |
| 🔶 MODERATE-HIGH RISK | Multiple indicators present, physician referral advised |
| 🔴 HIGH RISK | Critical levels, immediate consultation required |

---

## 🗂️ Project Structure

```
health-prediction/
│
├── HEALTH_PREDICTION.ipynb   # Main notebook — generates all source files
├── app.py                    # Streamlit frontend (UI, navigation, forms)
├── database.py               # SQLAlchemy ORM models & CRUD operations
├── validators.py             # Input validation (email, DOB, numeric ranges)
├── ml_engine.py              # ML prediction + clinical report generator
├── train_model.py            # Model training & evaluation script
├── model/
│   ├── rf_health_model.pkl   # Trained Random Forest model
│   └── scaler.pkl            # StandardScaler (fitted on training data)
└── health.db                 # SQLite database (auto-created, not committed)
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/health-prediction.git
cd health-prediction
```

### 2. Install dependencies

```bash
pip install streamlit sqlalchemy pandas scikit-learn numpy joblib
```

### 3. Run the notebook

Open `HEALTH_PREDICTION.ipynb` in **Jupyter** or **Google Colab** and run all cells. This will:
- Generate all `.py` source files automatically
- Train and save the ML model to the `model/` folder
- Launch the Streamlit app

### 4. Run the app locally

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## ☁️ Running on Google Colab

The notebook includes **Cloudflare Tunnel** support for instant public access:

1. Open the notebook in Google Colab
2. Run all cells in order
3. The final cell prints a live public URL (`*.trycloudflare.com`)
4. Open that URL in any browser — no local setup needed

---

## 📊 Clinical Reference Ranges

| Biomarker | Normal | Pre-Diabetic | Diabetic |
|---|---|---|---|
| Glucose (mg/dL) | 70–99 | 100–125 | ≥ 126 |
| HbA1c (%) | < 5.7 | 5.7–6.4 | ≥ 6.5 |
| Cholesterol (mg/dL) | < 200 | 200–239 | ≥ 240 |

*Based on WHO / American Diabetes Association (ADA) guidelines.*

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit (custom CSS, responsive layout) |
| Database | SQLite via SQLAlchemy ORM |
| ML | scikit-learn — Random Forest, StandardScaler |
| Data | NumPy, pandas, joblib |
| Deployment | Cloudflare Tunnel (Colab) |
| Language | Python 3.x |

---

## ⚠️ Disclaimer

This application is developed for **educational purposes only**. AI/ML predictions are not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
