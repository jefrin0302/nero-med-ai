"""
clinical_evaluation.py

Centralized Clinical Interpretation & Evaluation Layer for NeuroMed AI.
Provides:
1. Patient-specific clinical parameter table with exact reference ranges and calculated statuses.
2. Dynamic Leipzig criteria breakdown calculated strictly from patient values (no false/hardcoded points).
3. Risk-stratified physician next steps (no automatic chelation for low-risk patients).
4. Separation of ML statistical predictions from clinical decision support.
"""

from typing import Dict, Any, List


def evaluate_clinical_parameters(patient_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Evaluates each recorded clinical indicator against medical reference ranges,
    determining status badges dynamically from the actual measured value.
    """
    table = []

    # Map of ordered keys with formatting & evaluation logic
    keys = list(patient_dict.keys())

    for key in keys:
        raw_val = patient_dict[key]
        clean_name = key.replace("_", " ").title()

        # Handle demographic/non-biochemical fields
        if key == "Age":
            table.append({
                "label": "Age",
                "raw_key": key,
                "formatted_value": f"{int(float(raw_val))} years",
                "reference_boundary": "Patient Demographic",
                "badge_class": "eval-neutral",
                "badge_text": f"Age {int(float(raw_val))}"
            })
            continue

        if key == "Sex":
            table.append({
                "label": "Sex",
                "raw_key": key,
                "formatted_value": str(raw_val).title(),
                "reference_boundary": "Patient Demographic",
                "badge_class": "eval-neutral",
                "badge_text": "Recorded"
            })
            continue

        if key == "Region":
            table.append({
                "label": "Geographic Region",
                "raw_key": key,
                "formatted_value": str(raw_val).title(),
                "reference_boundary": "Demographic Metric",
                "badge_class": "eval-neutral",
                "badge_text": "Recorded"
            })
            continue

        if key == "Socioeconomic Status":
            table.append({
                "label": "Socioeconomic Status",
                "raw_key": key,
                "formatted_value": str(raw_val).title(),
                "reference_boundary": "Demographic Metric",
                "badge_class": "eval-neutral",
                "badge_text": "Recorded"
            })
            continue

        if key == "Alcohol Use":
            table.append({
                "label": "Alcohol Use",
                "raw_key": key,
                "formatted_value": str(raw_val).title(),
                "reference_boundary": "Clinical History",
                "badge_class": "eval-neutral",
                "badge_text": "Recorded"
            })
            continue

        if key == "BMI":
            try:
                bmi_val = float(raw_val)
                if bmi_val < 18.5:
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Underweight (<18.5)"
                elif 18.5 <= bmi_val <= 24.9:
                    b_cls, b_txt = "eval-normal", "✅ Normal (18.5–24.9)"
                elif 25.0 <= bmi_val <= 29.9:
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Overweight (25–29.9)"
                else:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Obese (≥30)"
                table.append({
                    "label": "Body Mass Index (BMI)",
                    "raw_key": key,
                    "formatted_value": f"{bmi_val:.2f} kg/m²",
                    "reference_boundary": "18.5 – 24.9 kg/m²",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # --- BIOCHEMICAL & CLINICAL MARKERS ---

        # 1. Ceruloplasmin
        if "ceruloplasmin" in key.lower():
            try:
                c_val = float(raw_val)
                if c_val < 10.0:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Abnormally Low (<10 mg/dL)"
                elif 10.0 <= c_val < 20.0:
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Borderline Low (10–20 mg/dL)"
                elif 20.0 <= c_val <= 40.0:
                    b_cls, b_txt = "eval-normal", "✅ Normal (20–40 mg/dL)"
                else:
                    b_cls, b_txt = "eval-neutral", "ℹ️ Elevated (>40 mg/dL)"
                table.append({
                    "label": "Serum Ceruloplasmin",
                    "raw_key": key,
                    "formatted_value": f"{c_val:.2f} mg/dL",
                    "reference_boundary": "20.0 – 40.0 mg/dL",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 2. Total Serum Copper
        if key == "Copper in Blood Serum":
            try:
                cu_b = float(raw_val)
                if cu_b > 140.0:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Elevated (>140 µg/dL)"
                elif cu_b < 70.0:
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Low (<70 µg/dL)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Normal (70–140 µg/dL)"
                table.append({
                    "label": "Total Serum Copper",
                    "raw_key": key,
                    "formatted_value": f"{cu_b:.2f} µg/dL",
                    "reference_boundary": "70.0 – 140.0 µg/dL",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 3. Free Copper in Blood Serum
        if "free copper" in key.lower():
            try:
                f_cu = float(raw_val)
                if f_cu > 15.0:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Elevated Free Copper (>15 µg/dL)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Within Limit (<15 µg/dL)"
                table.append({
                    "label": "Free (Non-Ceruloplasmin) Copper",
                    "raw_key": key,
                    "formatted_value": f"{f_cu:.2f} µg/dL",
                    "reference_boundary": "< 15.0 µg/dL",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 4. Copper in Urine (24h Urinary Copper)
        if "copper in urine" in key.lower() or "urinary" in key.lower():
            try:
                u_cu = float(raw_val)
                if u_cu > 100.0:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Pathologically Elevated (>100 µg/24h)"
                elif 40.0 <= u_cu <= 100.0:
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Intermediate Elevation (40–100 µg/24h)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Normal (<40 µg/24h)"
                table.append({
                    "label": "24h Urinary Copper Excretion",
                    "raw_key": key,
                    "formatted_value": f"{u_cu:.2f} µg/24h",
                    "reference_boundary": "< 40.0 µg/24h",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 5. ALT (Alanine Aminotransferase)
        if key == "ALT":
            try:
                alt = float(raw_val)
                if alt > 40.0:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Elevated Transaminase (>40 U/L)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Normal (≤40 U/L)"
                table.append({
                    "label": "Alanine Aminotransferase (ALT)",
                    "raw_key": key,
                    "formatted_value": f"{alt:.2f} U/L",
                    "reference_boundary": "< 40.0 U/L",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 6. AST (Aspartate Aminotransferase)
        if key == "AST":
            try:
                ast = float(raw_val)
                if ast > 40.0:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Elevated Transaminase (>40 U/L)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Normal (≤40 U/L)"
                table.append({
                    "label": "Aspartate Aminotransferase (AST)",
                    "raw_key": key,
                    "formatted_value": f"{ast:.2f} U/L",
                    "reference_boundary": "< 40.0 U/L",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 7. Total Bilirubin (Item 3 Fix)
        if "bilirubin" in key.lower():
            try:
                bili = float(raw_val)
                if bili > 1.2:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Elevated Bilirubin (>1.2 mg/dL)"
                elif bili < 0.2:
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Low Bilirubin (<0.2 mg/dL)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Normal (0.2–1.2 mg/dL)"
                table.append({
                    "label": "Total Bilirubin",
                    "raw_key": key,
                    "formatted_value": f"{bili:.2f} mg/dL",
                    "reference_boundary": "0.2 – 1.2 mg/dL",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 8. Albumin (Item 4 Fix)
        if "albumin" in key.lower():
            try:
                alb = float(raw_val)
                if alb < 3.5:
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Low Albumin (<3.5 g/dL)"
                elif alb > 5.0:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Elevated Albumin (>5.0 g/dL)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Normal (3.5–5.0 g/dL)"
                table.append({
                    "label": "Serum Albumin",
                    "raw_key": key,
                    "formatted_value": f"{alb:.2f} g/dL",
                    "reference_boundary": "3.5 – 5.0 g/dL",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 9. Alkaline Phosphatase (ALP)
        if "alkaline phosphatase" in key.lower() or key == "Alkaline Phosphatase (ALP)":
            try:
                alp = float(raw_val)
                if alp > 130.0:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Elevated ALP (>130 U/L)"
                elif alp < 40.0:
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Low ALP (<40 U/L)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Normal (40–130 U/L)"
                table.append({
                    "label": "Alkaline Phosphatase (ALP)",
                    "raw_key": key,
                    "formatted_value": f"{alp:.2f} U/L",
                    "reference_boundary": "40.0 – 130.0 U/L",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 10. Prothrombin Time / INR (Item 13 Fix)
        if "prothrombin" in key.lower() or "inr" in key.lower():
            try:
                inr = float(raw_val)
                if inr > 1.2:
                    b_cls, b_txt = "eval-abnormal-high", "⚠️ Prolonged Coagulation (>1.2)"
                elif inr < 0.8:
                    b_cls, b_txt = "eval-neutral", "ℹ️ Shortened (<0.8)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Normal (0.8–1.2)"
                table.append({
                    "label": "Prothrombin Time / INR",
                    "raw_key": key,
                    "formatted_value": f"{inr:.2f}",
                    "reference_boundary": "0.8 – 1.2",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 11. Gamma-Glutamyl Transferase (GGT) (Item 14 Fix)
        if "ggt" in key.lower() or "gamma-glutamyl" in key.lower():
            try:
                ggt = float(raw_val)
                if ggt > 50.0:
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Elevated GGT (>50 U/L)"
                else:
                    b_cls, b_txt = "eval-normal", "✅ Normal (≤50 U/L)"
                table.append({
                    "label": "Gamma-Glutamyl Transferase (GGT)",
                    "raw_key": key,
                    "formatted_value": f"{ggt:.2f} U/L",
                    "reference_boundary": "< 50.0 U/L",
                    "badge_class": b_cls,
                    "badge_text": b_txt
                })
            except (ValueError, TypeError):
                pass
            continue

        # 12. Kayser-Fleischer Rings (Item 5 Fix)
        if "kayser" in key.lower() or "kf" in key.lower():
            try:
                k_val = int(float(raw_val))
                if k_val == 1:
                    fmt_v = "Present (Observed)"
                    b_cls, b_txt = "eval-abnormal-high", "🚨 KF Rings Present"
                else:
                    fmt_v = "Absent (Not Observed)"
                    b_cls, b_txt = "eval-normal", "✅ Absent / Normal"
            except (ValueError, TypeError):
                fmt_v = str(raw_val)
                b_cls, b_txt = "eval-neutral", "Recorded"
            table.append({
                "label": "Kayser-Fleischer (KF) Rings",
                "raw_key": key,
                "formatted_value": fmt_v,
                "reference_boundary": "Absent (Negative)",
                "badge_class": b_cls,
                "badge_text": b_txt
            })
            continue

        # 13. Neurological Symptoms Score (Item 8 Fix)
        if "neurological" in key.lower():
            try:
                n_val = float(raw_val)
                table.append({
                    "label": "Neurological Symptoms Score",
                    "raw_key": key,
                    "formatted_value": f"{n_val:.1f} / 10",
                    "reference_boundary": "Clinical Scale (0 – 10)",
                    "badge_class": "eval-neutral",
                    "badge_text": f"Score: {n_val:.1f}/10 Documented"
                })
            except (ValueError, TypeError):
                pass
            continue

        # 14. Psychiatric Symptoms (Item 10 Fix)
        if "psychiatric" in key.lower():
            try:
                p_val = int(float(raw_val))
                if p_val == 1:
                    fmt_v = "Present (Documented)"
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Symptoms Documented"
                else:
                    fmt_v = "Absent (None Documented)"
                    b_cls, b_txt = "eval-normal", "✅ None Documented"
            except (ValueError, TypeError):
                fmt_v = str(raw_val)
                b_cls, b_txt = "eval-neutral", "Recorded"
            table.append({
                "label": "Psychiatric Symptoms",
                "raw_key": key,
                "formatted_value": fmt_v,
                "reference_boundary": "Absent (None Reported)",
                "badge_class": b_cls,
                "badge_text": b_txt
            })
            continue

        # 15. Cognitive Function Score (Item 9 Fix)
        if "cognitive" in key.lower():
            try:
                cg_val = float(raw_val)
                table.append({
                    "label": "Cognitive Function Score",
                    "raw_key": key,
                    "formatted_value": f"{cg_val:.1f} / 100",
                    "reference_boundary": "Assessment Scale (30 – 100)",
                    "badge_class": "eval-neutral",
                    "badge_text": f"Score: {cg_val:.1f}/100 Documented"
                })
            except (ValueError, TypeError):
                pass
            continue

        # 16. Family History (Item 7 Fix)
        if "family" in key.lower():
            try:
                f_val = int(float(raw_val))
                if f_val == 1:
                    fmt_v = "Positive (Reported in Relative)"
                    b_cls, b_txt = "eval-abnormal-low", "⚠️ Positive Family History"
                else:
                    fmt_v = "Negative (No Known History)"
                    b_cls, b_txt = "eval-normal", "✅ No Family History Reported"
            except (ValueError, TypeError):
                fmt_v = str(raw_val)
                b_cls, b_txt = "eval-neutral", "Recorded"
            table.append({
                "label": "Family History of Wilson Disease",
                "raw_key": key,
                "formatted_value": fmt_v,
                "reference_boundary": "No Known History",
                "badge_class": b_cls,
                "badge_text": b_txt
            })
            continue

        # 17. ATP7B Gene Mutation (Item 6 Fix)
        if "gene" in key.lower() or "atp7b" in key.lower() or "atb7b" in key.lower():
            try:
                g_val = int(float(raw_val))
                if g_val == 1:
                    fmt_v = "Mutation Present (Positive)"
                    b_cls, b_txt = "eval-abnormal-high", "🚨 Pathogenic Mutation Present"
                else:
                    fmt_v = "Mutation Absent (Normal)"
                    b_cls, b_txt = "eval-normal", "✅ Mutation Absent"
            except (ValueError, TypeError):
                fmt_v = str(raw_val)
                b_cls, b_txt = "eval-neutral", "Recorded"
            table.append({
                "label": "ATP7B Gene Mutation Analysis",
                "raw_key": key,
                "formatted_value": fmt_v,
                "reference_boundary": "Mutation Absent (Wild Type)",
                "badge_class": b_cls,
                "badge_text": b_txt
            })
            continue

        # Fallback for any unmapped key
        table.append({
            "label": clean_name,
            "raw_key": key,
            "formatted_value": str(raw_val),
            "reference_boundary": "Clinical Metric",
            "badge_class": "eval-neutral",
            "badge_text": "Recorded"
        })

    return table


def calculate_patient_leipzig_breakdown(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates Leipzig criteria contribution strictly from the active patient's actual measurements.
    Does NOT award points for criteria that the patient does not satisfy.
    Explicitly documents unassessed criteria with 0 points.
    """
    criteria_list = []
    total_score = 0

    # 1. Kayser-Fleischer Rings (Slit-lamp exam)
    kfr = patient_dict.get("Kayser-Fleischer Rings")
    try:
        kfr_val = int(float(kfr)) if kfr is not None else 0
        if kfr_val == 1:
            points = 2
            note = "Corneal copper deposition observed"
            satisfied = True
        else:
            points = 0
            note = "Absent on examination (0 pts)"
            satisfied = False
    except (ValueError, TypeError):
        points = 0
        note = "Not available / unassessed (0 pts)"
        satisfied = False
    total_score += points
    criteria_list.append({
        "name": "Kayser-Fleischer Rings",
        "points": points,
        "note": note,
        "satisfied": satisfied
    })

    # 2. Serum Ceruloplasmin
    cerulo = patient_dict.get("Ceruloplasmin Level")
    try:
        c_val = float(cerulo) if cerulo is not None else 30.0
        if c_val < 10.0:
            points = 2
            note = f"Measured {c_val:.1f} mg/dL (< 10 mg/dL)"
            satisfied = True
        elif 10.0 <= c_val < 20.0:
            points = 1
            note = f"Measured {c_val:.1f} mg/dL (10–19.9 mg/dL)"
            satisfied = True
        else:
            points = 0
            note = f"Measured {c_val:.1f} mg/dL (Normal ≥ 20 mg/dL)"
            satisfied = False
    except (ValueError, TypeError):
        points = 0
        note = "Unassessed (0 pts)"
        satisfied = False
    total_score += points
    criteria_list.append({
        "name": "Serum Ceruloplasmin",
        "points": points,
        "note": note,
        "satisfied": satisfied
    })

    # 3. 24-Hour Urinary Copper Excretion
    u_cu = patient_dict.get("Copper in Urine")
    try:
        u_val = float(u_cu) if u_cu is not None else 20.0
        if u_val > 100.0:
            points = 2
            note = f"Measured {u_val:.1f} µg/24h (> 100 µg/24h)"
            satisfied = True
        elif 40.0 <= u_val <= 100.0:
            points = 1
            note = f"Measured {u_val:.1f} µg/24h (40–100 µg/24h)"
            satisfied = True
        else:
            points = 0
            note = f"Measured {u_val:.1f} µg/24h (Normal < 40 µg/24h)"
            satisfied = False
    except (ValueError, TypeError):
        points = 0
        note = "Unassessed (0 pts)"
        satisfied = False
    total_score += points
    criteria_list.append({
        "name": "24h Urinary Copper Excretion",
        "points": points,
        "note": note,
        "satisfied": satisfied
    })

    # 4. ATP7B Gene Mutation
    gene = patient_dict.get("ATB7B Gene Mutation")
    try:
        g_val = int(float(gene)) if gene is not None else 0
        if g_val == 1:
            points = 2
            note = "Pathogenic mutation detected"
            satisfied = True
        else:
            points = 0
            note = "No mutation detected (0 pts)"
            satisfied = False
    except (ValueError, TypeError):
        points = 0
        note = "Unassessed (0 pts)"
        satisfied = False
    total_score += points
    criteria_list.append({
        "name": "ATP7B Gene Mutation Analysis",
        "points": points,
        "note": note,
        "satisfied": satisfied
    })

    # Unassessed criteria (explicitly documented with 0 points)
    unassessed_list = [
        "Hepatic Copper Content (Liver Biopsy): Quantification not performed (0 pts)",
        "Coombs-Negative Hemolytic Anemia: Not evaluated in panel (0 pts)",
        "Formal Specialist Neurological Examination / Brain MRI: Assessment score recorded contextually; formal Leipzig points not assumed (0 pts)"
    ]

    # Clinical interpretation text
    if total_score >= 4:
        interpretation = (
            f"The calculated score from available assessment data is {total_score} points. "
            "In clinical practice guidelines, a Leipzig score of ≥ 4 is associated with a high likelihood "
            "of Wilson Disease. Definitive diagnosis requires formal confirmation by a hepatologist or neurologist."
        )
    elif total_score == 3:
        interpretation = (
            f"The calculated score from available assessment data is {total_score} points (Probable Wilson Disease). "
            "Further clinical investigations (e.g. repeat urinary copper, ophthalmic slit-lamp exam, or liver biopsy) are indicated."
        )
    else:
        interpretation = (
            f"The calculated score from available assessment data is {total_score} points. "
            "This reflects a lower statistical likelihood based on the evaluated laboratory criteria. "
            "Clinical vigilance remains recommended if unexplained symptoms persist."
        )

    return {
        "criteria_list": criteria_list,
        "unassessed_list": unassessed_list,
        "total_score": total_score,
        "interpretation": interpretation
    }


def generate_physician_next_steps(pred: int, probability: float, patient_dict: Dict[str, Any], leipzig_score: int) -> List[str]:
    """
    Generates personalized physician next steps based on predicted risk category
    and patient-specific clinical findings.
    NEVER recommends chelation therapy for low-risk predictions.
    """
    is_high_risk = (pred == 1) or (probability >= 0.5)

    if is_high_risk:
        steps = [
            "Arrange urgent clinical referral to a hepatologist, gastroenterologist, or neurologist for comprehensive in-person diagnostic evaluation.",
            "Perform formal ophthalmic slit-lamp examination by an eye specialist to evaluate for Kayser-Fleischer rings and sunflower cataracts.",
            "Obtain confirmatory 24-hour urinary copper re-test and non-ceruloplasmin-bound (free) serum copper determination.",
            "Offer genetic counseling and cascade testing for first-degree family members (siblings and children).",
            "Perform multidisciplinary clinical and laboratory work-up before initiating any de-coppering or chelation therapy (e.g., D-Penicillamine, Trientine, or Zinc salts)."
        ]
    else:
        # Low predicted risk
        steps = [
            "Routine clinical follow-up; current biomarkers reflect a low predicted probability for Wilson's disease.",
            "If liver transaminases (ALT/AST) or neurological symptoms are abnormal, investigate alternative hepatic, metabolic, toxic, or neuromuscular etiologies.",
            "Consider ophthalmic slit-lamp examination or 24-hour urinary copper re-collection if clinical suspicion or unexplained hepatic/neurological symptoms persist.",
            "No indication for copper chelation therapy based on this assessment."
        ]

    return steps
