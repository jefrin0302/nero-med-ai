# 🧬 Nero Med AI — Wilson's Disease Clinical AI & RAG Decision Support System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20PWA-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20LogReg%20%7C%20Random%20Forest%20%7C%20SVM-purple.svg)](https://scikit-learn.org/)
[![Preprocessing](https://img.shields.io/badge/Preprocessing-RobustScaler%20%7C%20Median%20Imputer-teal.svg)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20Explainable%20AI-orange.svg)](https://shap.readthedocs.io/)
[![Clinical Guidelines](https://img.shields.io/badge/Guidelines-AASLD%20%7C%20EASL%20%7C%20Leipzig-red.svg)](https://www.aasld.org/)

An enterprise-grade, clinical decision support system designed to assist hepatologists and neurologists in early, accurate diagnosis and risk stratification of **Wilson's Disease** (hepatolenticular degeneration).

---

## 🌟 Key Features

### 1. Multi-Model AI Stacking Ensemble (The Brain)
Combines distinct machine learning paradigms to detect subtle, multi-organ clinical patterns:
* **Preprocessing:** `SimpleImputer(median)` + outlier-resilient `RobustScaler(IQR)` across 19 continuous lab markers + `OneHotEncoder` across 4 categorical features.
* **Model 1 — XGBoost Classifier:** Gradient-boosted decision trees sensitive to non-linear clinical thresholds and subtle biomarker cutoffs.
* **Model 2 — Logistic Regression (LogReg):** Establishes balanced, calibrated statistical reference odds.
* **Model 3 — Random Forest Classifier:** Bagged ensemble of decorrelated decision trees delivering low-variance splits and high noise resilience.
* **Model 4 — Support Vector Machine (SVM):** RBF kernel captures complex non-linear boundary crossings across hepatic and neurological biomarkers.
* **Stacking Meta-Classifier & Isotonic Calibrator:** Meta logistic regression synthesizes individual model probabilities into a consensus risk probability (**100.0% Holdout Test Accuracy**, **1.0000 ROC-AUC**).

### 2. Explainable AI (SHAP XAI)
* **Local Feature Attributions:** Interactive SHAP Waterfall and Force plots showing which biomarkers drove the risk score for the specific patient.
* **Global Interpretability:** Beeswarm plots and Summary Bar charts visualizing biomarker impact across the population.

### 3. Clinical Retrieval-Augmented Generation (RAG)
* **Guideline-Grounded Decision Support:** Uses ChromaDB vector search and structured clinical question banks to generate evidence-based management plans (chelators, zinc salts, dietary copper restriction, Leipzig scoring).
* **Interactive Clinical Assistant:** Conversational Q&A modal grounded in AASLD and EASL clinical practice guidelines.

### 4. Progressive Web App (PWA)
* Offline-capable Service Worker, web app manifest, and responsive mobile-first clinical dashboard interface.

---

## 📁 Repository Structure

```text
├── app.py                     # Flask web server, REST API, & inference pipeline
├── model.py                   # Model training, 5-fold cross-validation, & evaluation
├── rag_system.py              # Clinical RAG retrieval engine & chatbot
├── build_vectorstore.py       # ChromaDB vectorstore embedding generation
├── test.py                    # Unit tests & pipeline validation
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git ignore configuration
├── LICENSE                    # Open-source license
├── Wilson_disease_dataset.csv # Primary clinical training dataset
├── medical_docs/              # Clinical guideline documents & question banks
├── static/                    # CSS, JavaScript, icons, and PWA assets
├── templates/                 # Jinja2 HTML templates & dashboards
└── wilson_artifacts/          # Serialized models, weights, scalers, and plots
    ├── svm.joblib             # Trained SVM model
    ├── logreg_base.joblib     # Trained Logistic Regression model
    ├── bilstm_model.keras     # Trained Bi-LSTM neural network
    ├── meta_model.joblib      # Stacking meta-classifier
    ├── preprocessor.joblib    # Feature preprocessing pipeline
    └── meta_info.json         # Evaluation metrics & feature schema
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10 or higher
* `pip` and virtual environment support

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jefrin0302/nero-med-ai.git
   cd nero-med-ai
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the web application:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 📊 Model Performance Summary

| Architecture | Model Paradigm | Test Accuracy | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Model 1: XGBoost** | Extreme Gradient Boosted Trees | 100.0% | 1.0000 |
| **Model 2: LogReg** | Calibrated Reference Odds | 100.0% | 1.0000 |
| **Model 3: Random Forest** | Bagged Ensemble of Trees | 100.0% | 1.0000 |
| **Model 4: SVM** | RBF Kernel Non-Linear | 99.99% | 1.0000 |
| **Stacking Meta-Classifier** | **4-Model Calibrated Consensus Ensemble** | **100.0%** | **1.0000** |

---

## 🛡️ License
This project is licensed under the terms of the [LICENSE](LICENSE) file.
