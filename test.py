#!/usr/bin/env python3
"""
test.py

Validation test harness for the 4-Model Stacking Ensemble:
- Preprocessor: Median Imputer + RobustScaler(IQR) + OneHotEncoder
- Base Models: XGBoost, Logistic Regression, Random Forest, Support Vector Machine (SVM)
- Stacking Meta-Classifier + Isotonic Probability Calibrator
"""

import os
import joblib
import numpy as np
import pandas as pd

ART_DIR = "wilson_artifacts"

# 1. Load saved production artifacts
print("Loading production artifacts from:", os.path.abspath(ART_DIR))
preprocessor = joblib.load(os.path.join(ART_DIR, "preprocessor.joblib"))
xgb_model = joblib.load(os.path.join(ART_DIR, "xgb.joblib"))
logreg_model = joblib.load(os.path.join(ART_DIR, "logreg_base.joblib"))
rf_model = joblib.load(os.path.join(ART_DIR, "rf.joblib"))
svm_model = joblib.load(os.path.join(ART_DIR, "svm.joblib"))
meta_model = joblib.load(os.path.join(ART_DIR, "meta_model.joblib"))
calibrator = joblib.load(os.path.join(ART_DIR, "calibrator.joblib"))

print("✓ Preprocessor loaded successfully.")
print("✓ XGBoost loaded successfully.")
print("✓ Logistic Regression loaded successfully.")
print("✓ Random Forest loaded successfully.")
print("✓ Support Vector Machine loaded successfully.")
print("✓ Stacking Meta-Classifier loaded successfully.")
print("✓ Isotonic Probability Calibrator loaded successfully.\n")

# 2. Benchmark Clinical Test Cohort (Known Positives and Negatives)
data = [
    ["Daniel Perry",14,"Female",9.34,229.03,18.5,144.98,73.71,72.04,1.31,3.93,110.55,1.05,127.93,1,5.07,1,80.1,1,1,"West","Low","FALSE",25.21,1],
    ["Lauren Luna",21,"Female",9.14,200.23,16.15,104.19,25.89,67.51,0.22,4.10,89.18,1.10,105.97,1,6.18,0,96.84,1,1,"South","Medium","FALSE",33.51,1],
    ["Andrew Lewis",31,"Male",9.76,267.34,14.37,153.17,45.91,76.68,3.53,5.49,145.40,1.18,85.50,1,4.43,1,69.16,1,1,"West","High","TRUE",26.23,1],
    ["John Nash",44,"Male",13.68,101.04,12.81,72.49,59.51,35.47,1.29,3.76,118.66,1.14,75.28,0,2.81,1,86.72,0,0,"South","High","FALSE",18.70,0],
    ["Elizabeth Blevins",53,"Female",23.50,121.82,20.61,110.77,44.47,24.84,2.65,5.45,161.10,1.22,92.20,0,0.54,1,90.49,0,0,"East","Low","TRUE",37.20,0],
    ["Stephen Davis",59,"Female",22.85,145.98,10.81,120.36,50.11,48.07,1.38,3.95,121.90,1.27,59.52,0,2.34,1,85.77,1,0,"North","Medium","FALSE",28.16,0],
    ["Stephen Wallace",43,"Male",14.53,173.64,8.77,106.67,32.74,60.62,1.20,4.21,132.30,1.04,78.33,0,2.75,0,85.34,0,0,"West","High","FALSE",38.48,0]
]

columns = [
    "Name","Age","Sex","Ceruloplasmin Level","Copper in Blood Serum",
    "Free Copper in Blood Serum","Copper in Urine","ALT","AST",
    "Total Bilirubin","Albumin","Alkaline Phosphatase (ALP)",
    "Prothrombin Time / INR","Gamma-Glutamyl Transferase (GGT)",
    "Kayser-Fleischer Rings","Neurological Symptoms Score","Psychiatric Symptoms",
    "Cognitive Function Score","Family History","ATB7B Gene Mutation","Region",
    "Socioeconomic Status","Alcohol Use","BMI","Is_Wilson_Disease"
]

df = pd.DataFrame(data, columns=columns)
df["Alcohol Use"] = df["Alcohol Use"].astype(str)
df_features = df.drop(columns=["Name", "Is_Wilson_Disease"])

# 3. Preprocess with RobustScaler Pipeline
X = preprocessor.transform(df_features)

# 4. Predict Individual Constituent Probabilities
p_xgb = xgb_model.predict_proba(X)[:, 1]
p_log = logreg_model.predict_proba(X)[:, 1]
p_rf = rf_model.predict_proba(X)[:, 1]
p_svm = svm_model.predict_proba(X)[:, 1]

# 5. Stacking Meta-Classifier + Isotonic Calibration
stacked = np.column_stack([p_xgb, p_log, p_rf, p_svm])
final_prob = calibrator.predict_proba(stacked)[:, 1]
final_pred = (final_prob >= 0.5).astype(int)

# 6. Evaluation and Verification Output
print("=" * 70)
print(f"{'PATIENT NAME':20s} | {'ACTUAL':6s} | {'PRED':8s} | {'FINAL %':8s} | {'XGB%':6s} | {'LOG%':6s} | {'RF%':6s} | {'SVM%':6s}")
print("=" * 70)

all_correct = True
for i in range(len(df)):
    name = df.iloc[i]["Name"]
    actual = df.iloc[i]["Is_Wilson_Disease"]
    pred = final_pred[i]
    prob = final_prob[i] * 100.0
    
    if pred != actual:
        all_correct = False
        
    print(f"{name:20s} | {actual:6d} | {('Positive' if pred==1 else 'Negative'):8s} | {prob:7.1f}% | {p_xgb[i]*100:5.1f}% | {p_log[i]*100:5.1f}% | {p_rf[i]*100:5.1f}% | {p_svm[i]*100:5.1f}%")

print("=" * 70)
if all_correct:
    print("✓ SUCCESS: All 7 clinical test patients matched their expected diagnosis with 100% consensus accuracy!")
else:
    print("❌ MISMATCH DETECTED: Review individual predictions above.")
