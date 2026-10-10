#!/usr/bin/env python3
"""
scripts/generate_dataset_v2.py

Generates synthetic dataset v2 (Wilson_disease_dataset_v2.csv) with realistic,
overlapping biomarker distributions, non-zero differential diagnoses,
and physiological boundary constraints.

DISCLAIMER:
This dataset is synthetically generated for computational development, machine learning
experimentation, and educational demonstration. It is NOT clinically validated real patient data
and must never be claimed or used as validated clinical evidence.

Key Remediation Improvements:
1. Removed deterministic label shortcuts:
   - Kayser-Fleischer Rings: non-zero prevalence (~3%) in controls (representing differential
     cholestatic disorders like PBC), ~65% in Wilson cases.
   - ATP7B Mutation: non-zero prevalence (~3.5%) in controls (representing heterozygous carrier
     frequency and benign VUS), ~80% in Wilson cases.
   - Age: Broad, realistic overlapping age spans across cohorts (Wilson: 5-55, Controls: 5-75).
2. Physiological boundary enforcement:
   - All laboratory values strictly strictly bounded to physiological biological minimums (>= 0).
   - Zero negative values for serum copper, urine copper, ceruloplasmin, transaminases, bilirubin.
3. Realistic Differential Cohorts:
   - Non-Wilson liver disease (NAFLD, viral hepatitis, autoimmune hepatitis) and neurological
     controls (essential tremor, parkinsonism) providing realistic overlap and challenging the model.
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42
TOTAL_SAMPLES = 60000
N_WILSON = 20000
N_CONTROL = 40000

def generate_synthetic_v2(output_path="Wilson_disease_dataset_v2.csv", seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    
    # -------------------------------------------------------------
    # 1. Cohort Subgroups
    # -------------------------------------------------------------
    # Wilson cohort subtypes:
    # - Hepatic presentation (~45%): Younger, high transaminases, KF rings often absent (45% KF)
    # - Neurological presentation (~40%): Older youth/adults, high neuro score, KF rings present (88% KF)
    # - Presymptomatic / Screening (~15%): Borderline labs, mild copper changes, ATP7B positive
    w_subtypes = rng.choice(["hepatic", "neuro", "presymptomatic"], size=N_WILSON, p=[0.45, 0.40, 0.15])
    
    # Control cohort subtypes:
    # - Healthy normal controls (~50%)
    # - Chronic liver disease / cholestasis differentials (~35%): PBC, AIH, NAFLD
    # - Movement disorder / neurological differentials (~15%): Tremor, dystonia, parkinsonism
    c_subtypes = rng.choice(["healthy", "liver_diff", "neuro_diff"], size=N_CONTROL, p=[0.50, 0.35, 0.15])

    # -------------------------------------------------------------
    # 2. Wilson Cohort Feature Generation
    # -------------------------------------------------------------
    w_age = np.zeros(N_WILSON)
    w_cerulo = np.zeros(N_WILSON)
    w_serum_cu = np.zeros(N_WILSON)
    w_free_cu = np.zeros(N_WILSON)
    w_urine_cu = np.zeros(N_WILSON)
    w_alt = np.zeros(N_WILSON)
    w_ast = np.zeros(N_WILSON)
    w_tbili = np.zeros(N_WILSON)
    w_alb = np.zeros(N_WILSON)
    w_alp = np.zeros(N_WILSON)
    w_inr = np.zeros(N_WILSON)
    w_ggt = np.zeros(N_WILSON)
    w_kf = np.zeros(N_WILSON, dtype=int)
    w_neuro = np.zeros(N_WILSON)
    w_psych = np.zeros(N_WILSON, dtype=int)
    w_cog = np.zeros(N_WILSON)
    w_fam_hx = np.zeros(N_WILSON, dtype=int)
    w_atp7b = np.zeros(N_WILSON, dtype=int)

    for i in range(N_WILSON):
        st = w_subtypes[i]
        if st == "hepatic":
            w_age[i] = rng.uniform(8, 35)
            w_cerulo[i] = rng.normal(11.5, 3.5)
            w_serum_cu[i] = rng.normal(160.0, 35.0)
            w_free_cu[i] = rng.normal(26.0, 6.0)
            w_urine_cu[i] = rng.normal(170.0, 45.0)
            w_alt[i] = rng.normal(95.0, 30.0)
            w_ast[i] = rng.normal(85.0, 25.0)
            w_tbili[i] = rng.normal(2.2, 0.8)
            w_alb[i] = rng.normal(3.8, 0.4)
            w_alp[i] = rng.normal(70.0, 20.0) # Wilson characteristically low ALP
            w_inr[i] = rng.normal(1.35, 0.25)
            w_ggt[i] = rng.normal(85.0, 25.0)
            w_kf[i] = rng.choice([0, 1], p=[0.55, 0.45]) # ~45% in hepatic
            w_neuro[i] = rng.exponential(1.5)
            w_psych[i] = rng.choice([0, 1], p=[0.65, 0.35])
            w_cog[i] = rng.normal(84.0, 8.0)
            w_fam_hx[i] = rng.choice([0, 1], p=[0.65, 0.35])
            w_atp7b[i] = rng.choice([0, 1], p=[0.20, 0.80])
        elif st == "neuro":
            w_age[i] = rng.uniform(16, 50)
            w_cerulo[i] = rng.normal(9.5, 3.0)
            w_serum_cu[i] = rng.normal(180.0, 40.0)
            w_free_cu[i] = rng.normal(32.0, 8.0)
            w_urine_cu[i] = rng.normal(195.0, 50.0)
            w_alt[i] = rng.normal(55.0, 18.0)
            w_ast[i] = rng.normal(52.0, 16.0)
            w_tbili[i] = rng.normal(1.6, 0.6)
            w_alb[i] = rng.normal(4.1, 0.4)
            w_alp[i] = rng.normal(80.0, 25.0)
            w_inr[i] = rng.normal(1.20, 0.18)
            w_ggt[i] = rng.normal(65.0, 20.0)
            w_kf[i] = rng.choice([0, 1], p=[0.12, 0.88]) # ~88% in neurological
            w_neuro[i] = rng.normal(6.5, 2.5)
            w_psych[i] = rng.choice([0, 1], p=[0.35, 0.65])
            w_cog[i] = rng.normal(74.0, 10.0)
            w_fam_hx[i] = rng.choice([0, 1], p=[0.60, 0.40])
            w_atp7b[i] = rng.choice([0, 1], p=[0.15, 0.85])
        else: # presymptomatic
            w_age[i] = rng.uniform(5, 40)
            w_cerulo[i] = rng.normal(14.0, 4.0)
            w_serum_cu[i] = rng.normal(130.0, 30.0)
            w_free_cu[i] = rng.normal(20.0, 5.0)
            w_urine_cu[i] = rng.normal(95.0, 25.0)
            w_alt[i] = rng.normal(45.0, 15.0)
            w_ast[i] = rng.normal(42.0, 14.0)
            w_tbili[i] = rng.normal(1.2, 0.4)
            w_alb[i] = rng.normal(4.4, 0.3)
            w_alp[i] = rng.normal(95.0, 25.0)
            w_inr[i] = rng.normal(1.10, 0.12)
            w_ggt[i] = rng.normal(48.0, 15.0)
            w_kf[i] = rng.choice([0, 1], p=[0.80, 0.20])
            w_neuro[i] = rng.exponential(0.8)
            w_psych[i] = rng.choice([0, 1], p=[0.80, 0.20])
            w_cog[i] = rng.normal(88.0, 6.0)
            w_fam_hx[i] = rng.choice([0, 1], p=[0.30, 0.70]) # high family history in screening
            w_atp7b[i] = rng.choice([0, 1], p=[0.10, 0.90])

    # -------------------------------------------------------------
    # 3. Control Cohort Feature Generation
    # -------------------------------------------------------------
    c_age = np.zeros(N_CONTROL)
    c_cerulo = np.zeros(N_CONTROL)
    c_serum_cu = np.zeros(N_CONTROL)
    c_free_cu = np.zeros(N_CONTROL)
    c_urine_cu = np.zeros(N_CONTROL)
    c_alt = np.zeros(N_CONTROL)
    c_ast = np.zeros(N_CONTROL)
    c_tbili = np.zeros(N_CONTROL)
    c_alb = np.zeros(N_CONTROL)
    c_alp = np.zeros(N_CONTROL)
    c_inr = np.zeros(N_CONTROL)
    c_ggt = np.zeros(N_CONTROL)
    c_kf = np.zeros(N_CONTROL, dtype=int)
    c_neuro = np.zeros(N_CONTROL)
    c_psych = np.zeros(N_CONTROL, dtype=int)
    c_cog = np.zeros(N_CONTROL)
    c_fam_hx = np.zeros(N_CONTROL, dtype=int)
    c_atp7b = np.zeros(N_CONTROL, dtype=int)

    for i in range(N_CONTROL):
        st = c_subtypes[i]
        if st == "healthy":
            c_age[i] = rng.uniform(15, 75)
            c_cerulo[i] = rng.normal(28.0, 5.0)
            c_serum_cu[i] = rng.normal(105.0, 18.0)
            c_free_cu[i] = rng.normal(11.0, 3.0)
            c_urine_cu[i] = rng.normal(25.0, 8.0)
            c_alt[i] = rng.normal(24.0, 7.0)
            c_ast[i] = rng.normal(22.0, 6.0)
            c_tbili[i] = rng.normal(0.8, 0.25)
            c_alb[i] = rng.normal(4.5, 0.3)
            c_alp[i] = rng.normal(90.0, 20.0)
            c_inr[i] = rng.normal(1.02, 0.08)
            c_ggt[i] = rng.normal(30.0, 10.0)
            c_kf[i] = 0
            c_neuro[i] = rng.exponential(0.4)
            c_psych[i] = rng.choice([0, 1], p=[0.82, 0.18])
            c_cog[i] = rng.normal(92.0, 5.0)
            c_fam_hx[i] = rng.choice([0, 1], p=[0.94, 0.06])
            c_atp7b[i] = rng.choice([0, 1], p=[0.98, 0.02]) # 2% carrier / VUS
        elif st == "liver_diff":
            # Liver disease mimics: NAFLD, PBC, Autoimmune Hepatitis
            c_age[i] = rng.uniform(20, 70)
            c_cerulo[i] = rng.normal(24.0, 6.0) # ceruloplasmin can drop to ~16-20 in liver failure
            c_serum_cu[i] = rng.normal(118.0, 25.0)
            c_free_cu[i] = rng.normal(14.0, 4.5)
            c_urine_cu[i] = rng.normal(52.0, 18.0) # urine Cu can rise in cholestasis / PBC
            c_alt[i] = rng.normal(78.0, 32.0) # elevated transaminases
            c_ast[i] = rng.normal(68.0, 28.0)
            c_tbili[i] = rng.normal(1.9, 0.75)
            c_alb[i] = rng.normal(3.9, 0.45)
            c_alp[i] = rng.normal(165.0, 45.0) # elevated ALP in cholestasis
            c_inr[i] = rng.normal(1.24, 0.20)
            c_ggt[i] = rng.normal(110.0, 40.0)
            c_kf[i] = rng.choice([0, 1], p=[0.92, 0.08]) # ~8% pseudo-KF copper deposits in PBC
            c_neuro[i] = rng.exponential(0.6)
            c_psych[i] = rng.choice([0, 1], p=[0.75, 0.25])
            c_cog[i] = rng.normal(88.0, 7.0)
            c_fam_hx[i] = rng.choice([0, 1], p=[0.92, 0.08])
            c_atp7b[i] = rng.choice([0, 1], p=[0.96, 0.04]) # 4% carrier / VUS
        else: # neuro_diff
            # Tremor / parkinsonism / dystonia differentials
            c_age[i] = rng.uniform(25, 75)
            c_cerulo[i] = rng.normal(27.0, 5.0)
            c_serum_cu[i] = rng.normal(108.0, 20.0)
            c_free_cu[i] = rng.normal(12.0, 3.5)
            c_urine_cu[i] = rng.normal(28.0, 10.0)
            c_alt[i] = rng.normal(26.0, 8.0)
            c_ast[i] = rng.normal(25.0, 8.0)
            c_tbili[i] = rng.normal(0.85, 0.25)
            c_alb[i] = rng.normal(4.4, 0.3)
            c_alp[i] = rng.normal(92.0, 22.0)
            c_inr[i] = rng.normal(1.04, 0.09)
            c_ggt[i] = rng.normal(32.0, 12.0)
            c_kf[i] = 0
            c_neuro[i] = rng.normal(5.5, 2.0) # prominent neurological symptoms
            c_psych[i] = rng.choice([0, 1], p=[0.55, 0.45])
            c_cog[i] = rng.normal(76.0, 9.0)
            c_fam_hx[i] = rng.choice([0, 1], p=[0.88, 0.12])
            c_atp7b[i] = rng.choice([0, 1], p=[0.97, 0.03])

    # -------------------------------------------------------------
    # 4. Assemble DataFrames & Apply Physical Range Constraints
    # -------------------------------------------------------------
    w_df = pd.DataFrame({
        "Age": np.clip(w_age, 5.0, 65.0).round(1),
        "Ceruloplasmin Level": np.clip(w_cerulo, 3.0, 35.0).round(2),
        "Copper in Blood Serum": np.clip(w_serum_cu, 40.0, 350.0).round(2),
        "Free Copper in Blood Serum": np.clip(w_free_cu, 5.0, 60.0).round(2),
        "Copper in Urine": np.clip(w_urine_cu, 25.0, 450.0).round(2),
        "ALT": np.clip(w_alt, 15.0, 300.0).round(2),
        "AST": np.clip(w_ast, 15.0, 280.0).round(2),
        "Total Bilirubin": np.clip(w_tbili, 0.3, 8.0).round(2),
        "Albumin": np.clip(w_alb, 2.0, 5.5).round(2),
        "Alkaline Phosphatase (ALP)": np.clip(w_alp, 20.0, 250.0).round(2),
        "Prothrombin Time / INR": np.clip(w_inr, 0.8, 3.0).round(2),
        "Gamma-Glutamyl Transferase (GGT)": np.clip(w_ggt, 15.0, 250.0).round(2),
        "Kayser-Fleischer Rings": w_kf,
        "Neurological Symptoms Score": np.clip(w_neuro, 0.0, 15.0).round(2),
        "Psychiatric Symptoms": w_psych,
        "Cognitive Function Score": np.clip(w_cog, 30.0, 100.0).round(1),
        "Family History": w_fam_hx,
        "ATB7B Gene Mutation": w_atp7b,
        "Is_Wilson_Disease": 1
    })

    c_df = pd.DataFrame({
        "Age": np.clip(c_age, 5.0, 80.0).round(1),
        "Ceruloplasmin Level": np.clip(c_cerulo, 8.0, 50.0).round(2),
        "Copper in Blood Serum": np.clip(c_serum_cu, 50.0, 220.0).round(2),
        "Free Copper in Blood Serum": np.clip(c_free_cu, 3.0, 35.0).round(2),
        "Copper in Urine": np.clip(c_urine_cu, 10.0, 150.0).round(2),
        "ALT": np.clip(c_alt, 10.0, 250.0).round(2),
        "AST": np.clip(c_ast, 10.0, 220.0).round(2),
        "Total Bilirubin": np.clip(c_tbili, 0.2, 5.5).round(2),
        "Albumin": np.clip(c_alb, 2.2, 5.5).round(2),
        "Alkaline Phosphatase (ALP)": np.clip(c_alp, 30.0, 320.0).round(2),
        "Prothrombin Time / INR": np.clip(c_inr, 0.8, 2.2).round(2),
        "Gamma-Glutamyl Transferase (GGT)": np.clip(c_ggt, 10.0, 250.0).round(2),
        "Kayser-Fleischer Rings": c_kf,
        "Neurological Symptoms Score": np.clip(c_neuro, 0.0, 15.0).round(2),
        "Psychiatric Symptoms": c_psych,
        "Cognitive Function Score": np.clip(c_cog, 40.0, 100.0).round(1),
        "Family History": c_fam_hx,
        "ATB7B Gene Mutation": c_atp7b,
        "Is_Wilson_Disease": 0
    })

    combined = pd.concat([w_df, c_df], ignore_index=True)
    
    # Demographics and context (independent of class)
    n_tot = len(combined)
    combined["Sex"] = rng.choice(["Male", "Female"], size=n_tot, p=[0.50, 0.50])
    combined["Region"] = rng.choice(["North", "South", "East", "West"], size=n_tot)
    combined["Socioeconomic Status"] = rng.choice(["Low", "Medium", "High"], size=n_tot)
    combined["Alcohol Use"] = rng.choice([False, True], size=n_tot, p=[0.65, 0.35])
    combined["BMI"] = np.clip(rng.normal(26.5, 4.5, size=n_tot), 18.0, 42.0).round(2)
    
    # Synthetic patient names
    first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Sam", "Chris", "Pat", "Robin", "Casey", "Drew", "Kelly", "Cameron"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Martinez", "Hernandez"]
    combined["Name"] = [f"{rng.choice(first_names)} {rng.choice(last_names)}" for _ in range(n_tot)]

    # Introduce realistic 8-10% missingness (MCAR) on selected laboratory biomarkers
    missing_cols = [
        "Free Copper in Blood Serum", "Copper in Urine", "Total Bilirubin",
        "Albumin", "Alkaline Phosphatase (ALP)", "Prothrombin Time / INR",
        "Gamma-Glutamyl Transferase (GGT)", "Cognitive Function Score"
    ]
    for col in missing_cols:
        mask = rng.random(size=n_tot) < 0.09
        combined.loc[mask, col] = np.nan

    # Order columns canonically matching Wilson_disease_dataset.csv
    cols_order = [
        "Name", "Age", "Sex", "Ceruloplasmin Level", "Copper in Blood Serum",
        "Free Copper in Blood Serum", "Copper in Urine", "ALT", "AST",
        "Total Bilirubin", "Albumin", "Alkaline Phosphatase (ALP)",
        "Prothrombin Time / INR", "Gamma-Glutamyl Transferase (GGT)",
        "Kayser-Fleischer Rings", "Neurological Symptoms Score",
        "Psychiatric Symptoms", "Cognitive Function Score", "Family History",
        "ATB7B Gene Mutation", "Region", "Socioeconomic Status",
        "Alcohol Use", "BMI", "Is_Wilson_Disease"
    ]
    combined = combined[cols_order]

    # Shuffle rows
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Save
    combined.to_csv(output_path, index=False)
    print(f"Generated synthetic v2 dataset saved to: {output_path} (Shape: {combined.shape})")
    return combined

if __name__ == "__main__":
    generate_synthetic_v2()
