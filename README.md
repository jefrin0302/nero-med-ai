# 🧬 Nero Med AI — Wilson's Disease Clinical AI & RAG Decision Support System

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask%20%7C%20PWA-green.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML%2FDL-SVM%20%7C%20LogReg%20%7C%20Bi--LSTM-purple.svg)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20Explainable%20AI-orange.svg)](https://shap.readthedocs.io/)
[![Clinical Guidelines](https://img.shields.io/badge/Guidelines-AASLD%20%7C%20EASL%20%7C%20Leipzig-red.svg)](https://www.aasld.org/)

An enterprise-grade, clinical decision support system designed to assist hepatologists and neurologists in early, accurate diagnosis and risk stratification of **Wilson's Disease** (hepatolenticular degeneration).

---

## 🌟 Key Features

### 1. Multi-Model AI Stacking Ensemble (The Brain)
Combines distinct machine learning paradigms to detect subtle, multi-organ clinical patterns:
* **Model A — Support Vector Machine (SVM):** RBF kernel captures complex non-linear interactions across hepatic and neurological biomarkers.
* **Model B — Logistic Regression (LogReg):** Establishes balanced, calibrated statistical odds.
* **Model C — Bidirectional LSTM (Bi-LSTM):** Recurrent neural network capturing forward and backward cross-feature dependencies across sequential lab values.
* **Stacking Meta-Classifier:** Meta logistic regression synthesizes individual model probabilities into a consensus risk probability (**99.91% Ensemble Accuracy**, **1.0000 ROC-AUC**).

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

| Architecture | Model Paradigm | Accuracy | ROC-AUC |
| :--- | :--- | :--- | :--- |
| **Model A: SVM** | RBF Kernel Non-Linear | 99.88% | 1.0000 |
| **Model B: LogReg** | Calibrated Linear Odds | 99.88% | 1.0000 |
| **Model C: Bi-LSTM** | Deep Recurrent Network | 99.81% | 1.0000 |
| **Stacking Meta-Classifier** | **3-Model Consensus Ensemble** | **99.91%** | **1.0000** |

---

## 🛡️ License
This project is licensed under the terms of the [LICENSE](LICENSE) file.
