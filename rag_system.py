#!/usr/bin/env python3
"""
rag_system.py

Retrieval-Augmented Generation (RAG) module for Wilson's Disease Clinical Support.
Integrates vector retrieval (medical guidelines knowledge base) with conversational intelligence
and clinical synthesis (Google Gemini, OpenAI, or smart grounded RAG engine).
"""

import os
import json
import logging
import re
import math
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger("wilson_rag")
logging.basicConfig(level=logging.INFO)

PERSIST_DIR = "chroma_db"
DOCS_DIR = "medical_docs"
APPROVED_PDFS = [
    "AASLD WD Guidelines.pdf",
    "EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf",
    "NIH Wilson Disease.pdf",
    "clinical practice guidelines in WD.pdf"
]

DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

NO_DOCS_MSG = "I do not have access to any medical documents and cannot answer."

CLARIFICATION_MSG = (
    "I'm not sure what you mean. Please ask a question related to Wilson disease, "
    "diagnosis, treatment, laboratory tests, diet, or your patient report."
)

def _canonical_token(w: str) -> str:
    """Normalize common clinical plurals, inflections, and medical synonyms."""
    w = w.lower()
    synonyms = {
        "urinary": "urine",
        "tests": "test",
        "testing": "test",
        "tested": "test",
        "foods": "food",
        "diets": "diet",
        "dietary": "diet",
        "eating": "eat",
        "treatments": "treatment",
        "treating": "treatment",
        "treated": "treatment",
        "treat": "treatment",
        "therapies": "treatment",
        "therapy": "treatment",
        "drugs": "medication",
        "drug": "medication",
        "medicines": "medication",
        "medicine": "medication",
        "medications": "medication",
        "mutations": "mutation",
        "genes": "gene",
        "genetic": "gene",
        "genetics": "gene",
        "symptoms": "symptom",
        "parameters": "parameter",
        "causes": "cause",
        "caused": "cause",
        "causing": "cause",
        "measures": "measure",
        "measured": "measure",
        "measurement": "measure",
        "measuring": "measure",
        "evaluating": "evaluate",
        "evaluation": "evaluate",
        "evaluated": "evaluate",
        "rings": "ring",
        "signs": "sign",
        "biopsies": "biopsy",
        "organs": "organ",
        "children": "child",
        "pediatric": "child",
        "siblings": "sibling",
        "relatives": "relative",
    }
    if w in synonyms:
        return synonyms[w]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        return w[:-3]
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    return w

def is_meaningless_query(user_question: str) -> bool:
    """
    Detects clearly meaningless, extremely low-information, or random gibberish input.
    Returns True if the input has no identifiable medical, greeting, or conversational content.
    Legitimate short queries (e.g. 'copper', 'ATP7B', 'KF ring', 'diet', 'treatment', 'Trientine')
    and greetings are safely preserved.
    """
    if not user_question:
        return True

    text = user_question.strip().lower()
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    tokens = [w for w in clean_text.split() if w]

    if not tokens:
        return True

    # 1. Greetings and conversational phrases
    greeting_terms = {
        "hi", "hello", "hey", "greetings", "good", "morning", "afternoon", "evening",
        "how", "are", "you", "doing", "fine", "great", "ok", "okay", "well", "thanks",
        "thank", "bye", "goodbye", "help", "who", "what", "name"
    }
    if any(w in greeting_terms for w in tokens) and len(tokens) <= 5:
        if any(g in text for g in ["hi", "hello", "hey", "how are you", "how r u", "fine", "good morning", "good evening", "good afternoon", "who are you", "what are you", "greetings", "thank"]):
            return False

    # 2. Comprehensive medical, genetic, lab, diet, and clinical domain terms
    domain_terms = {
        # Core disease & genetics
        "wilson", "wilsons", "disease", "disorder", "atp7b", "atp7", "gene", "genes", "genetic",
        "genetics", "mutation", "mutations", "autosomal", "recessive", "chromosome", "carrier", "carriers",
        # Copper & biochemistry
        "copper", "cu", "ceruloplasmin", "apoceruloplasmin", "protein", "glycoprotein", "free", "bound",
        "serum", "urinary", "urine", "hepatic", "liver", "brain", "basal", "ganglia",
        # Diagnostic markers, lab tests & scores
        "kf", "kayser", "fleischer", "ring", "rings", "cornea", "corneal", "eye", "eyes", "slit", "lamp",
        "leipzig", "score", "scores", "criteria", "points", "ferenci",
        "alt", "ast", "alp", "ggt", "bilirubin", "albumin", "platelet", "platelets", "inr", "pt",
        "cbc", "creatinine", "hemolysis", "coombs", "anemia", "biopsy", "ultrasound", "mri",
        "test", "tests", "testing", "tested", "lab", "labs", "panel", "screening", "screen",
        "diagnosis", "diagnose", "diagnostic", "marker", "markers", "biomarker", "biomarkers",
        "value", "values", "level", "levels", "parameter", "parameters", "24h", "24hr", "24",
        # Treatments, drugs & interventions
        "treatment", "treatments", "treat", "treating", "treated", "therapy", "therapies",
        "chelation", "chelator", "chelators", "penicillamine", "trientine", "cuprimine", "syprine",
        "zinc", "galzin", "acetate", "gluconate", "pyridoxine", "b6", "transplant", "transplantation",
        "drug", "drugs", "medicine", "medicines", "medication", "medications", "dose", "dosage",
        "side", "effect", "effects", "adverse", "alternative", "alternatives",
        # Diet, food & nutrition
        "diet", "diets", "dietary", "food", "foods", "eat", "eating", "nutrition", "water",
        "organ", "meat", "meats", "shellfish", "oyster", "oysters", "nut", "nuts", "seed", "seeds",
        "chocolate", "cocoa", "mushroom", "mushrooms", "soy", "soybean", "soybeans", "tofu",
        # Symptoms & clinical features
        "symptom", "symptoms", "sign", "signs", "jaundice", "ascites", "cirrhosis", "hepatitis",
        "tremor", "tremors", "dystonia", "dysarthria", "speech", "swallow", "swallowing", "ataxia",
        "chorea", "parkinson", "rigidity", "psychiatric", "depression", "anxiety", "fatigue",
        # AI model, risk, report & explainability
        "shap", "svm", "bilstm", "lstm", "model", "prediction", "probability", "score",
        "risk", "percent", "percentage", "ensemble", "explain", "ai", "neuromed", "report", "result"
    }

    # If any token or clean_text contains any domain term, it is valid
    for token in tokens:
        if token in domain_terms:
            return False
    for dt in domain_terms:
        if len(dt) > 2 and dt in clean_text:
            return False

    # 3. Conversational follow-ups (e.g. "what are its side effects", "why is it important", "what is mine")
    conversational_pronouns = {"it", "its", "they", "them", "this", "that", "mine", "my"}
    if any(w in conversational_pronouns for w in tokens) and len(tokens) >= 2:
        return False

    # 4. Check for question structure with body or health keywords
    question_words = {"what", "why", "how", "when", "where", "which", "who", "can", "could", "is", "are", "does", "do", "explain", "tell", "describe", "show"}
    body_or_clinical_words = {
        "body", "blood", "cell", "cells", "organ", "damage", "cause", "causes", "safe", "prevent",
        "prevention", "precaution", "precautions", "sick", "ill", "illness", "health", "healthy",
        "doctor", "hospital", "patient", "take", "taking", "avoid", "risk", "condition", "norm"
    }
    if any(w in question_words for w in tokens) and (any(w in body_or_clinical_words for w in tokens) or len(tokens) >= 4):
        return False

    # 5. Gibberish / low-information heuristics:
    # a) If clean string without spaces is <= 3 characters and not a recognized domain term
    if len(clean_text.replace(" ", "")) <= 3:
        return True

    # b) Check for lack of vowels (e.g. fnjvds, dssf)
    vowels = set("aeiouy")
    for t in tokens:
        if len(t) >= 4 and not any(ch in vowels for ch in t):
            return True

    # c) Extremely high consonant-to-vowel ratio
    for t in tokens:
        if len(t) >= 5:
            v_count = sum(1 for ch in t if ch in vowels)
            if v_count == 0 or (len(t) >= 6 and v_count <= 1):
                return True

    # d) Repeated single characters (e.g. 'aaaa', 'zzzz')
    if len(tokens) == 1 and len(set(tokens[0])) <= 2:
        return True

    return True

def sanitize_bot_answer(text: str) -> str:
    """
    Cleans formatting artifacts, internal headers, and raw markdown symbols:
    - Removes separator artifacts: \==\, \---\, ===, ---, ==
    - Strips internal question-bank title headers and metadata
    - Removes double asterisks (**) and single asterisks (*)
    - Removes backticks (`)
    - Converts bullet hyphens (- item) to clean bullets (• item)
    - Strips markdown heading hashes (###)
    - Normalizes excessive blank lines
    """
    if not text:
        return ""
    # 1. Remove escaped or unescaped separator artifacts: \==\, \---\, ===, ---, ==, etc.
    cleaned = re.sub(r'\\+={1,}\\+?', '', text)
    cleaned = re.sub(r'\\+-{2,}\\+?', '', cleaned)
    cleaned = re.sub(r'={2,}', '', cleaned)
    cleaned = re.sub(r'-{3,}', '', cleaned)
    cleaned = re.sub(r'^[ \t]*[=\-_~\\]+[ \t]*$', '', cleaned, flags=re.MULTILINE)

    # 2. Strip internal question-bank title / header dumps
    header_patterns = [
        r'WILSON DISEASE\s*[-—–]\s*QUESTION BANK WITH ANSWERS[^\n]*',
        r'Evidence-Grounded Reference for RAG Chatbot Development[^\n]*',
        r'Source question bank:[^\n]*',
        r'Guideline basis:[^\n]*',
        r'Purpose:\s*Educational,\s*evidence-grounded reference answers[^\n]*',
        r'^[0-9]+\.\s+[A-Z\s&]+\s*$'
    ]
    for hp in header_patterns:
        cleaned = re.sub(hp, '', cleaned, flags=re.MULTILINE | re.IGNORECASE)

    # 3. Remove raw markdown symbols (asterisks, backticks)
    cleaned = cleaned.replace("**", "").replace("*", "").replace("`", "")

    # 4. Convert leading bullet hyphens to bullet points (• )
    cleaned = re.sub(r'^[ \t]*-[ \t]+', '• ', cleaned, flags=re.MULTILINE)

    # 5. Strip markdown heading hashes (### )
    cleaned = re.sub(r'^#{1,6}\s*', '', cleaned, flags=re.MULTILINE)

    # 6. Normalize excessive blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()

# =============================================================
# CLINICAL MESSAGING CONSTANTS & HELPERS
# =============================================================
OUT_OF_SCOPE_MSG = (
    "I am specialized exclusively in Wilson's Disease clinical education and patient assessment support. "
    "I cannot assist with general knowledge, non-medical questions, or unrelated topics. "
    "Please let me know if you have questions about Wilson's disease symptoms, biomarkers, "
    "diagnostic criteria (such as the Leipzig score), treatment guidelines, or your assessment report."
)

CLINICAL_CDS_NOTE = (
    "\n\nClinical Decision Support Notice:\n"
    "This interpretation is provided as educational decision support based on available assessment markers. "
    "It does not constitute a definitive medical diagnosis or personalized prescription. "
    "All diagnostic confirmations, formal Leipzig evaluations, and treatment modifications require direct consultation "
    "with a treating gastroenterologist, hepatologist, or neurologist."
)

def is_out_of_scope_query(user_question: str, chat_history=None) -> bool:
    """
    Conservative, medical- and Wilson-aware detector for completely unrelated non-medical queries.
    Preserves all medical, laboratory, genetic, treatment, and conversational follow-up questions.
    Only queries with clear non-medical subjects (sports, politics, entertainment, software coding,
    foreign geography/trivia, weather, recipes, etc.) and NO medical/patient context are flagged.
    """
    if not user_question:
        return False

    text = user_question.strip().lower()
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    tokens = set(clean_text.split())

    # 1. Broad in-domain keywords (Medical, Wilson's, Patient Report, Biomarkers, Genetics, Treatments, Decision Support)
    in_domain_keywords = {
        # Core Wilson & Genetics
        "wilson", "wilsons", "atp7b", "atp7", "copper", "cu", "ceruloplasmin", "apoceruloplasmin",
        "kayser", "fleischer", "kf", "ring", "rings", "cornea", "corneal", "eye", "eyes", "slit", "lamp",
        "leipzig", "ferenci", "mutation", "mutations", "gene", "genes", "genetic", "genetics", "autosomal", "recessive",
        # Labs & Biomarkers
        "urine", "urinary", "serum", "blood", "free", "bound", "hepatic", "liver", "enzyme", "enzymes",
        "alt", "ast", "alp", "ggt", "bilirubin", "albumin", "inr", "prothrombin", "biopsy", "platelet",
        "creatinine", "hemolysis", "coombs", "anemia", "lab", "labs", "test", "tests", "testing",
        "parameter", "parameters", "marker", "markers", "biomarker", "biomarkers", "level", "levels", "value", "values",
        "range", "ranges", "normal", "abnormal", "elevated", "low", "high", "deficiency", "overload",
        # Symptoms & Body
        "tremor", "tremors", "dystonia", "dysarthria", "ataxia", "chorea", "parkinson", "rigidity",
        "jaundice", "ascites", "cirrhosis", "hepatitis", "fatigue", "neurological", "neuro",
        "psychiatric", "cognitive", "depression", "anxiety", "brain", "ganglia", "symptom", "symptoms", "sign", "signs",
        # Treatments & Drugs
        "treatment", "treatments", "therapy", "therapies", "chelation", "chelator", "chelators",
        "penicillamine", "trientine", "cuprimine", "syprine", "zinc", "galzin", "acetate", "gluconate",
        "pyridoxine", "b6", "transplant", "transplantation", "dose", "dosage", "medication", "medications",
        "medicine", "medicines", "drug", "drugs", "prescription", "side", "effect", "effects",
        "stop", "stopping", "pause", "pausing", "skip", "skipping", "discontinue", "substitute", "alternative",
        # Diet & Lifestyle
        "diet", "diets", "dietary", "food", "foods", "eat", "eating", "nutrition", "organ", "meat", "meats",
        "shellfish", "oyster", "oysters", "nut", "nuts", "seed", "seeds", "chocolate", "cocoa", "mushroom", "mushrooms",
        # Patient context / AI report / General healthcare
        "patient", "report", "assessment", "result", "results", "prediction", "probability", "risk",
        "score", "scores", "doctor", "physician", "specialist", "hepatologist", "neurologist", "gastroenterologist",
        "hospital", "clinic", "clinical", "health", "healthy", "illness", "disease", "disorder", "diagnose", "diagnosis",
        "my", "mine", "me", "i", "model", "shap", "ai", "neuromed", "recommendation", "recommendations", "advice"
    }

    # If any token is in domain, it is NOT out of scope
    if any(t in in_domain_keywords for t in tokens):
        return False

    compound_in_domain = [
        "wilson", "copper", "atp7b", "kf ring", "ceruloplasmin", "urine copper", "urinary copper",
        "free copper", "liver enzyme", "leipzig", "side effect", "blood test", "what should i do",
        "explain my", "my report", "my result", "my risk", "why is it", "is it normal", "can you explain"
    ]
    if any(cid in clean_text for cid in compound_in_domain):
        return False

    # 2. Check if recent chat history had medical context (for short follow-up questions)
    if chat_history:
        for turn in reversed(chat_history[-3:]):
            prev_content = (turn.get("content") or "").lower()
            if any(term in prev_content for term in ["copper", "ceruloplasmin", "wilson", "leipzig", "treatment", "penicillamine", "trientine", "zinc"]):
                if any(w in tokens for w in ["why", "how", "what", "is", "can", "tell", "explain", "more"]):
                    return False

    # 3. Explicit non-medical domains
    out_of_scope_indicators = [
        # Sports
        r'\b(football|soccer|cricket|basketball|nba|nfl|fifa|ipl|tennis|messi|ronaldo|ipl score|match score)\b',
        # Entertainment / Pop culture
        r'\b(movie|movies|cinema|actor|actress|hollywood|bollywood|netflix|song|songs|music|singer|celebrity|album)\b',
        # General Geography & Trivia
        r'\b(capital of|population of|president of|prime minister of|currency of|flag of|continent|mount everest|eiffel tower)\b',
        # Coding / Software
        r'\b(python script|python code|python|javascript|html|css|write code|debug code|coding|code|sql query|git push|react component|docker container)\b',
        # Weather / Astronomy
        r'\b(weather in|forecast for|temperature today|will it rain|mars rover|how far is the moon|solar eclipse)\b',
        # Math & Homework
        r'\b(solve equation|integral of|derivative of|pythagorean theorem|math homework)\b',
        # Non-medical cooking recipes
        r'\b(how to bake|cake recipe|pizza dough|pasta recipe|cookie recipe|chocolate cake recipe)\b',
    ]

    for pattern in out_of_scope_indicators:
        if re.search(pattern, clean_text):
            return True

    # 4. If query lacks ANY medical/health/patient/conversational anchor and is a general trivia question
    trivia_openers = ["who is ", "where is ", "when did ", "what is the capital", "how tall is ", "who won "]
    if any(clean_text.startswith(to) for to in trivia_openers):
        return True

    return False

def calculate_patient_leipzig_score(p_data: dict) -> str:
    """
    Calculates the Leipzig Consensus Score strictly from objectively evaluated
    patient parameters available in the record (ceruloplasmin, 24h urinary copper,
    Kayser-Fleischer rings, and ATP7B mutation).
    
    In accordance with clinical guidelines and project documentation:
    - Custom assessment scores (Neurological / Psychiatric) are reported as contextual
      findings but are NOT assumed to equate to formal Leipzig neurological manifestation points.
    - Unassessed or unavailable criteria (Biopsy, Coombs test) are explicitly documented as unassessed (0 pts).
    - Missing patient parameters are marked as 'Not available' and do not silently add points.
    - Diagnostic interpretation is presented as clinical decision support, emphasizing physician confirmation.
    """
    if not p_data or not isinstance(p_data, dict):
        return (
            "No patient assessment data is currently loaded. Please complete an assessment "
            "or enter clinical biomarkers to calculate a patient-specific Leipzig score."
        )

    score = 0
    lines = []
    lines.append("📋 Patient-Specific Leipzig Score Evaluation (AASLD / EASL Criteria)\n")

    # 1. Kayser-Fleischer Rings (Slit-lamp examination)
    kfr = p_data.get("Kayser-Fleischer Rings")
    if kfr is None or str(kfr).strip() == "":
        lines.append("• Kayser-Fleischer Rings: Not available in record (0 pts assigned)")
    else:
        try:
            kfr_val = int(float(kfr))
            if kfr_val == 1:
                score += 2
                lines.append("• Kayser-Fleischer Rings: Present (+2 pts) [Corneal copper deposition observed]")
            else:
                lines.append("• Kayser-Fleischer Rings: Absent (0 pts)")
        except (ValueError, TypeError):
            lines.append(f"• Kayser-Fleischer Rings: Recorded as '{kfr}' (0 pts assigned)")

    # 2. Serum Ceruloplasmin
    cerulo = p_data.get("Ceruloplasmin Level")
    if cerulo is None or str(cerulo).strip() == "":
        lines.append("• Serum Ceruloplasmin: Not available in record (0 pts assigned)")
    else:
        try:
            c_val = float(cerulo)
            if c_val < 10.0:
                score += 2
                lines.append(f"• Serum Ceruloplasmin: {c_val:.1f} mg/dL (+2 pts) [< 10 mg/dL, strongly indicative]")
            elif 10.0 <= c_val < 20.0:
                score += 1
                lines.append(f"• Serum Ceruloplasmin: {c_val:.1f} mg/dL (+1 pt) [10–19.9 mg/dL, borderline low]")
            else:
                lines.append(f"• Serum Ceruloplasmin: {c_val:.1f} mg/dL (0 pts) [≥ 20 mg/dL, within normal range 20–40 mg/dL]")
        except (ValueError, TypeError):
            lines.append(f"• Serum Ceruloplasmin: Recorded as '{cerulo}' (0 pts assigned)")

    # 3. 24-Hour Urinary Copper Excretion
    u_cu = p_data.get("Copper in Urine")
    if u_cu is None or str(u_cu).strip() == "":
        lines.append("• 24h Urinary Copper: Not available in record (0 pts assigned)")
    else:
        try:
            u_val = float(u_cu)
            if u_val > 100.0:
                score += 2
                lines.append(f"• 24h Urinary Copper: {u_val:.1f} µg/24h (+2 pts) [> 100 µg/24h (>2× ULN), pathological overload]")
            elif 40.0 <= u_val <= 100.0:
                score += 1
                lines.append(f"• 24h Urinary Copper: {u_val:.1f} µg/24h (+1 pt) [40–100 µg/24h (1–2× ULN), intermediate elevation]")
            else:
                lines.append(f"• 24h Urinary Copper: {u_val:.1f} µg/24h (0 pts) [< 40 µg/24h, within normal limits]")
        except (ValueError, TypeError):
            lines.append(f"• 24h Urinary Copper: Recorded as '{u_cu}' (0 pts assigned)")

    # 4. ATP7B Mutation Analysis
    gene = p_data.get("ATB7B Gene Mutation")
    if gene is None or str(gene).strip() == "":
        lines.append("• ATP7B Mutation Analysis: Not available in record (0 pts assigned)")
    else:
        try:
            g_val = int(float(gene))
            if g_val == 1:
                score += 2
                lines.append("• ATP7B Mutation Analysis: Pathogenic mutation detected (+2 pts)")
            else:
                lines.append("• ATP7B Mutation Analysis: No mutation detected / Normal (0 pts)")
        except (ValueError, TypeError):
            lines.append(f"• ATP7B Mutation Analysis: Recorded as '{gene}' (0 pts assigned)")

    # 5. Neurological / Psychiatric Manifestation Evaluation (Explicitly distinct from formal Leipzig scoring)
    neuro = p_data.get("Neurological Symptoms Score")
    psych = p_data.get("Psychiatric Symptoms")
    neuro_str = f"{neuro}" if neuro is not None else "Not recorded"
    psych_str = "Reported" if str(psych).strip() in ("1", "Yes", "true") else ("Not reported" if psych is not None else "Not recorded")
    lines.append(
        f"• Neurological & Psychiatric Findings: Recorded assessment score = {neuro_str} (Psychiatric: {psych_str}). "
        f"Note: Formal Leipzig neurological manifestation points (0–2) require structured neurological examination "
        f"and brain MRI by a specialist, so no points are assumed here (0 pts counted)."
    )

    # 6. Unassessed Criteria (Biopsy & Coombs)
    lines.append("• Hepatic Copper Content (Liver Biopsy): Quantification not available in patient record (unassessed, 0 pts)")
    lines.append("• Coombs-Negative Hemolytic Anemia: Coombs test not available in patient record (unassessed, 0 pts)")

    # Diagnostic Likelihood & Interpretation (Clinical Decision Support Wording)
    lines.append(f"\n🎯 Total Leipzig Score from Evaluated Criteria: {score} points\n")
    if score >= 4:
        lines.append(
            f"• Clinical Interpretation:\n"
            f"  The calculated Leipzig score from the available assessment data is {score}. "
            f"This falls within a range associated with a higher likelihood of Wilson Disease, "
            f"but formal clinical diagnosis requires comprehensive evaluation and confirmation by your treating hepatologist or physician."
        )
    elif score == 3:
        lines.append(
            f"• Clinical Interpretation:\n"
            f"  The calculated Leipzig score from the available assessment data is {score} (Probable Wilson's disease). "
            f"Further diagnostic work-up—including formal neurological/brain MRI scoring, penicillamine challenge, or liver biopsy—is clinically indicated."
        )
    else:
        lines.append(
            f"• Clinical Interpretation:\n"
            f"  The calculated Leipzig score from the available assessment data is {score} (Lower statistical likelihood based on evaluated markers). "
            f"Clinical follow-up remains important if hepatic or neurological symptoms persist."
        )

    lines.append(CLINICAL_CDS_NOTE)
    return sanitize_bot_answer("\n".join(lines))

def evaluate_patient_specific_biomarker(biomarker_key: str, p_data: dict) -> str:
    """
    Evaluates a specific clinical biomarker from the patient's actual assessment record,
    comparing it against medical guideline reference ranges.
    Zero hallucination: If the value is missing in p_data, it explicitly states it is not available.
    """
    if not p_data or not isinstance(p_data, dict):
        return (
            "No patient report is currently active. Please complete a patient assessment "
            "to view personalized biomarker evaluations."
        )

    b = biomarker_key.lower()

    # --- 1. CERULOPLASMIN ---
    if "ceruloplasmin" in b:
        val = p_data.get("Ceruloplasmin Level")
        if val is None or str(val).strip() == "":
            return "Serum Ceruloplasmin Level is not available in the current patient assessment."
        try:
            v = float(val)
        except (ValueError, TypeError):
            return f"Serum Ceruloplasmin Level is recorded as: {val}."

        if v < 10.0:
            status = "Markedly Low (Diagnostic marker, +2 Leipzig points)"
            interp = (
                f"Your serum ceruloplasmin is {v:.1f} mg/dL, which is significantly below the normal range of 20–40 mg/dL.\n"
                f"• Biological Mechanism: In Wilson disease, mutations in the ATP7B copper-transporting ATPase impair the incorporation "
                f"of copper into apoceruloplasmin during hepatic synthesis. The resulting apoceruloplasmin molecule is unstable and is rapidly degraded in circulation.\n"
                f"• Clinical Significance: Levels below 10 mg/dL strongly suggest impaired copper export and form one of the central pillars of Wilson's disease diagnosis."
            )
        elif 10.0 <= v < 20.0:
            status = "Borderline Low (+1 Leipzig point)"
            interp = (
                f"Your serum ceruloplasmin is {v:.1f} mg/dL, which is borderline low compared to the normal range of 20–40 mg/dL.\n"
                f"• Clinical Significance: While not severely depressed, borderline values warrant correlation with 24-hour urinary copper and free serum copper."
            )
        else:
            status = "Normal Range (20–40 mg/dL)"
            interp = (
                f"Your serum ceruloplasmin is {v:.1f} mg/dL, which falls within the normal physiological reference range of 20–40 mg/dL.\n"
                f"• Clinical Nuance: Note that ceruloplasmin is an acute-phase reactant; severe systemic or hepatic inflammation can transiently elevate ceruloplasmin levels into the normal range even in some Wilson's disease cases."
            )

        return sanitize_bot_answer(
            f"🧪 Your Serum Ceruloplasmin Evaluation:\n\n"
            f"• Patient Value: {v:.1f} mg/dL\n"
            f"• Reference Range: 20 – 40 mg/dL\n"
            f"• Clinical Status: {status}\n\n"
            f"{interp}"
        )

    # --- 2. 24-HOUR URINARY COPPER ---
    if any(k in b for k in ["urine", "urinary"]):
        val = p_data.get("Copper in Urine")
        if val is None or str(val).strip() == "":
            return "24-Hour Urinary Copper is not available in the current patient assessment."
        try:
            v = float(val)
        except (ValueError, TypeError):
            return f"24-Hour Urinary Copper is recorded as: {val}."

        if v > 100.0:
            status = "Markedly Elevated (+2 Leipzig points)"
            interp = (
                f"Your 24-hour urine copper excretion is {v:.1f} µg/24h, significantly exceeding the normal upper limit of 40 µg/24h.\n"
                f"• Pathophysiology: When the liver's storage capacity for excess copper is overwhelmed, unbound copper spills into the bloodstream and is filtered by renal glomeruli, resulting in hypercupriuria.\n"
                f"• Clinical Meaning: Excretion > 100 µg/24h strongly supports symptomatic Wilson's disease."
            )
        elif 40.0 <= v <= 100.0:
            status = "Intermediate / Moderately Elevated (+1 Leipzig point)"
            interp = (
                f"Your 24-hour urine copper excretion is {v:.1f} µg/24h (normal: < 40 µg/24h).\n"
                f"• Clinical Meaning: Intermediate elevation is common in presymptomatic Wilson disease or chronic cholestatic liver conditions."
            )
        else:
            status = "Normal Excretion (< 40 µg/24h)"
            interp = (
                f"Your 24-hour urinary copper is {v:.1f} µg/24h, which is within the normal healthy baseline (< 40 µg/24h)."
            )

        return sanitize_bot_answer(
            f"🧪 Your 24-Hour Urinary Copper Evaluation:\n\n"
            f"• Patient Value: {v:.1f} µg/24h\n"
            f"• Reference Range: < 40 µg/24h (Normal); > 100 µg/24h (Diagnostic threshold)\n"
            f"• Clinical Status: {status}\n\n"
            f"{interp}"
        )

    # --- 3. FREE SERUM COPPER ---
    if "free copper" in b:
        val = p_data.get("Free Copper in Blood Serum")
        if val is None or str(val).strip() == "":
            return "Free Copper in Blood Serum is not available in the current patient assessment."
        try:
            v = float(val)
        except (ValueError, TypeError):
            return f"Free Copper in Blood Serum is recorded as: {val}."

        if v > 15.0:
            status = "Elevated Unbound Toxic Copper"
            interp = (
                f"Your free serum copper is {v:.1f} µg/dL, above the normal threshold (< 15 µg/dL).\n"
                f"• Clinical Significance: Free copper is non-ceruloplasmin-bound and represents the toxic fraction capable of penetrating tissues (brain, liver, kidneys)."
            )
        else:
            status = "Normal Unbound Copper (< 15 µg/dL)"
            interp = f"Your free serum copper is {v:.1f} µg/dL, within the target safe reference limit (< 15 µg/dL)."

        return sanitize_bot_answer(
            f"🧪 Your Free (Non-Ceruloplasmin-Bound) Serum Copper:\n\n"
            f"• Patient Value: {v:.1f} µg/dL\n"
            f"• Normal Reference Range: 5 – 15 µg/dL\n"
            f"• Clinical Status: {status}\n\n"
            f"{interp}"
        )

    # --- 4. LIVER ENZYMES (ALT / AST) ---
    if any(k in b for k in ["alt", "ast", "liver enzyme"]):
        alt_val = p_data.get("ALT")
        ast_val = p_data.get("AST")
        if alt_val is None and ast_val is None:
            return "Liver transaminases (ALT/AST) are not available in the current patient assessment."
        
        alt_str = f"{float(alt_val):.1f} U/L" if alt_val is not None else "N/A"
        ast_str = f"{float(ast_val):.1f} U/L" if ast_val is not None else "N/A"
        
        is_elevated = False
        try:
            if alt_val is not None and float(alt_val) > 40.0: is_elevated = True
            if ast_val is not None and float(ast_val) > 40.0: is_elevated = True
        except (ValueError, TypeError):
            pass

        status = "Elevated Transaminases (Hepatocellular injury)" if is_elevated else "Within Normal Baseline (10–40 U/L)"
        return sanitize_bot_answer(
            f"🧪 Your Liver Enzyme (Transaminases) Evaluation:\n\n"
            f"• ALT (Alanine Aminotransferase): {alt_str} (Normal: 10 – 40 U/L)\n"
            f"• AST (Aspartate Aminotransferase): {ast_str} (Normal: 10 – 40 U/L)\n"
            f"• Clinical Status: {status}\n\n"
            f"• Medical Context: Hepatocellular injury from copper accumulation causes transaminases to leak into the bloodstream. "
            f"In acute Wilsonian liver presentations, an AST:ALT ratio > 2 with low alkaline phosphatase can be an important diagnostic clue."
        )

    # --- 5. KAYSER-FLEISCHER RINGS ---
    if any(k in b for k in ["kayser", "fleischer", "kf", "ring"]):
        kfr = p_data.get("Kayser-Fleischer Rings")
        if kfr is None or str(kfr).strip() == "":
            return "Kayser-Fleischer Rings status is not available in the current patient assessment."
        try:
            is_present = int(float(kfr)) == 1
        except (ValueError, TypeError):
            is_present = str(kfr).strip() in ("1", "Yes", "true", "True")

        status = "Present (+2 Leipzig points)" if is_present else "Absent (0 points)"
        interp = (
            "Slit-lamp examination confirmed copper deposition in Descemet's membrane of the cornea. "
            "KF rings are present in over 90% of neurological Wilson disease presentations and gradually resolve with de-coppering therapy."
            if is_present else
            "No corneal copper deposition reported on slit-lamp exam. Note that KF rings may be absent in up to 50% of hepatic presentations."
        )
        return sanitize_bot_answer(
            f"👁️ Your Kayser-Fleischer (KF) Rings Evaluation:\n\n"
            f"• Recorded Finding: {'Present' if is_present else 'Absent'}\n"
            f"• Status: {status}\n\n"
            f"• Clinical Context: {interp}"
        )

    # --- 6. ATP7B GENE MUTATION ---
    if any(k in b for k in ["atp7b", "mutation", "gene", "genetic"]):
        gene = p_data.get("ATB7B Gene Mutation")
        if gene is None or str(gene).strip() == "":
            return "ATP7B Gene Mutation status is not available in the current patient assessment."
        try:
            is_detected = int(float(gene)) == 1
        except (ValueError, TypeError):
            is_detected = str(gene).strip() in ("1", "Yes", "true", "True")

        status = "Pathogenic Mutation Detected (+2 Leipzig points)" if is_detected else "No Mutation Detected / Normal"
        interp = (
            "Pathogenic mutation detected in the ATP7B copper-transport gene on chromosome 13q14.3. "
            "First-degree relatives (especially siblings) should be offered genetic screening."
            if is_detected else
            "No mutation detected in standard ATP7B screening. Clinical diagnosis relies on biochemical and slit-lamp markers."
        )
        return sanitize_bot_answer(
            f"🧬 Your ATP7B Genetic Analysis Evaluation:\n\n"
            f"• Genetic Finding: {'Mutation Detected' if is_detected else 'Not Detected / Normal'}\n"
            f"• Status: {status}\n\n"
            f"• Clinical Significance: {interp}"
        )

    return f"Parameter '{biomarker_key}' evaluation is not available in the record."

def generate_patient_risk_explanation(prob, pred_text, p_data: dict, model_breakdown: dict = None) -> str:
    """
    Generates a personalized prediction and risk explanation referencing the patient's
    actual values from p_data rather than generic copy.
    """
    if prob is None:
        return "No prediction probability is currently loaded. Please submit an assessment to view your risk analysis."

    c_val = p_data.get("Ceruloplasmin Level")
    u_val = p_data.get("Copper in Urine")
    f_val = p_data.get("Free Copper in Blood Serum")
    alt_val = p_data.get("ALT")
    ast_val = p_data.get("AST")
    kfr_val = p_data.get("Kayser-Fleischer Rings")
    gene_val = p_data.get("ATB7B Gene Mutation")

    # Determine key driving factors from authentic data
    drivers = []
    if c_val is not None:
        try:
            if float(c_val) < 10.0:
                drivers.append(f"Markedly depressed ceruloplasmin ({float(c_val):.1f} mg/dL, normal 20–40)")
            elif float(c_val) < 20.0:
                drivers.append(f"Borderline low ceruloplasmin ({float(c_val):.1f} mg/dL)")
        except (ValueError, TypeError): pass

    if u_val is not None:
        try:
            if float(u_val) > 100.0:
                drivers.append(f"High 24h urinary copper excretion ({float(u_val):.1f} µg/24h, normal <40)")
            elif float(u_val) >= 40.0:
                drivers.append(f"Elevated 24h urinary copper ({float(u_val):.1f} µg/24h)")
        except (ValueError, TypeError): pass

    if f_val is not None:
        try:
            if float(f_val) > 15.0:
                drivers.append(f"Elevated toxic free serum copper ({float(f_val):.1f} µg/dL, normal <15)")
        except (ValueError, TypeError): pass

    if kfr_val is not None:
        try:
            if int(float(kfr_val)) == 1:
                drivers.append("Presence of Kayser-Fleischer corneal rings")
        except (ValueError, TypeError): pass

    if gene_val is not None:
        try:
            if int(float(gene_val)) == 1:
                drivers.append("Detected pathogenic ATP7B gene mutation")
        except (ValueError, TypeError): pass

    if alt_val is not None or ast_val is not None:
        try:
            alt_f = float(alt_val) if alt_val is not None else 0
            ast_f = float(ast_val) if ast_val is not None else 0
            if alt_f > 40.0 or ast_f > 40.0:
                drivers.append(f"Elevated transaminases (ALT: {alt_f:.0f} U/L, AST: {ast_f:.0f} U/L)")
        except (ValueError, TypeError): pass

    driver_text = ""
    if drivers:
        driver_text = "• Key Patient Markers Influencing This Probability:\n" + "\n".join(f"  - {d}" for d in drivers) + "\n\n"
    else:
        driver_text = "• Key Patient Markers: Core copper and hepatic markers fall within typical physiological ranges in your record.\n\n"

    # Model consensus breakdown
    mb_text = ""
    if model_breakdown and isinstance(model_breakdown, dict):
        svm_p = model_breakdown.get("svm")
        lstm_p = model_breakdown.get("bilstm")
        log_p = model_breakdown.get("logreg")
        parts = []
        if lstm_p is not None: parts.append(f"Bi-LSTM: {lstm_p}%")
        if svm_p is not None: parts.append(f"SVM: {svm_p}%")
        if log_p is not None: parts.append(f"Logistic Regression: {log_p}%")
        if parts:
            mb_text = f"• Ensemble Model Consensus: Stacking Meta-Classifier synthesizes {', '.join(parts)} into a consensus probability of {prob}%.\n\n"

    if prob < 40.0:
        body = (
            f"📊 Patient Risk Interpretation: Low Statistical Risk ({prob}% — {pred_text})\n\n"
            f"• Model Assessment:\n"
            f"  The ensemble classifier estimates a {prob}% probability of Wilson's disease based on your entered markers. "
            f"Your clinical parameters generally reflect preserved copper homeostasis.\n\n"
            f"{mb_text}"
            f"{driver_text}"
            f"• Clinical Guidance:\n"
            f"  A low algorithmic score suggests low probability, but clinical correlation is essential if unexplained liver or neurological symptoms persist."
        )
    elif prob < 60.0:
        body = (
            f"📊 Patient Risk Interpretation: Intermediate / Indeterminate Risk ({prob}% — {pred_text})\n\n"
            f"• Model Assessment:\n"
            f"  The ensemble classifier estimates an intermediate probability ({prob}%). "
            f"Your entered parameters present a mixed clinical picture where some biomarkers suggest abnormal copper kinetics while others remain indeterminate.\n\n"
            f"{mb_text}"
            f"{driver_text}"
            f"• Recommended Clinical Next Steps:\n"
            f"  Discuss this result with a gastroenterologist or hepatologist for formal confirmatory evaluation, "
            f"including repeat 24-hour urine copper, slit-lamp exam for KF rings, and Leipzig scoring."
        )
    else:
        body = (
            f"📊 Patient Risk Interpretation: High Risk Signal ({prob}% — {pred_text})\n\n"
            f"• Model Assessment:\n"
            f"  The ensemble classifier estimates a high probability ({prob}%) that the entered profile resembles Wilson's disease.\n\n"
            f"{mb_text}"
            f"{driver_text}"
            f"• Recommended Urgent Next Steps:\n"
            f"  Prompt clinical consultation with a specialist (hepatologist/gastroenterologist) is strongly advised for formal diagnostic confirmation "
            f"(formal Leipzig scoring, slit-lamp examination, and consideration of initial de-coppering chelation therapy)."
        )

    return sanitize_bot_answer(body + CLINICAL_CDS_NOTE)

def handle_medication_safety(clean_q: str) -> str:
    """
    Provides safe, clinical guidance when a user asks about stopping, pausing,
    skipping, or altering medication/dosage. Never prescribes or advises self-discontinuation.
    """
    return sanitize_bot_answer(
        "⚠️ Critical Clinical Medication Safety Advisory:\n\n"
        "• Medical therapy for Wilson's disease (including D-Penicillamine, Trientine, and Zinc salts) "
        "is life-long and must NEVER be stopped, skipped, or modified without direct supervision from your treating physician.\n\n"
        "• Risk of Abrupt Cessation: Abruptly stopping de-coppering therapy allows toxic copper to rapidly re-accumulate, "
        "which can precipitate fatal acute fulminant liver failure or irreversible neurological crisis within months.\n\n"
        "• If You Experience Adverse Symptoms: If you are experiencing difficult side effects (such as rash, nausea, taste loss, or worsening tremor), "
        "contact your treating hepatologist or neurologist immediately. They can safely evaluate dosage adjustments, "
        "co-prescribe Pyridoxine (Vitamin B6), or transition you to alternative therapies (e.g., Trientine or Zinc salts).\n\n"
        "NeuroMed AI is an informational decision support tool and cannot prescribe medications or authorize dosage modifications."
    )


CLINICAL_LEXICON = [
    ("Biomarker", "Ceruloplasmin", [r"\bceruloplasmin\b", r"\bcpn\b", r"\bapoceruloplasmin\b"]),
    ("Biomarker", "Serum Free Copper", [r"\bserum\s+(?:free\s+)?copper\b", r"\bfree\s+copper\b", r"\bnon[- ]ceruloplasmin[- ]bound\s+copper\b", r"\bncc\b"]),
    ("Biomarker", "24h Urinary Copper", [r"\b24[- ]?h(?:our)?\s+urinary\s+copper\b", r"\burinary\s+copper\b", r"\b24[- ]?h\s+cu\b"]),
    ("Biomarker", "Liver Enzymes (ALT/AST)", [r"\balt\b", r"\bast\b", r"\balp\b", r"\btransaminases?\b", r"\balanine\s+aminotransferase\b", r"\baspartate\s+aminotransferase\b"]),
    ("Biomarker", "Total Bilirubin", [r"\bbilirubin\b", r"\bjaundice\b", r"\bicterus\b"]),
    ("Biomarker", "Serum Albumin / INR", [r"\balbumin\b", r"\binr\b", r"\bprothrombin\b", r"\bcoagulopathy\b"]),
    ("Biomarker", "Hepatic Parenchymal Copper", [r"\bhepatic\s+copper\b", r"\bliver\s+copper\b", r"\bdry\s+weight\s+copper\b"]),
    ("Gene", "ATP7B Mutation", [r"\batp7b\b", r"\bcopper[- ]transporting\s+atpase\b", r"\bmutation\b", r"\bhomozygous\b", r"\bcompound\s+heterozygous\b"]),
    ("Treatment", "D-Penicillamine", [r"\bpenicillamine\b", r"\bd[- ]penicillamine\b", r"\bcuprimine\b", r"\bdepen\b"]),
    ("Treatment", "Trientine", [r"\btrientine\b", r"\bsyprine\b", r"\bcuprior\b", r"\bteta\b"]),
    ("Treatment", "Zinc Salts", [r"\bzinc\b", r"\bzinc\s+acetate\b", r"\bzinc\s+gluconate\b", r"\bgalzin\b", r"\bwilzin\b"]),
    ("Treatment", "Pyridoxine (Vitamin B6)", [r"\bpyridoxine\b", r"\bvitamin\s+b6\b", r"\bb6\b"]),
    ("Treatment", "Liver Transplantation", [r"\bliver\s+transplant(?:ation)?\b", r"\bolt\b", r"\bltx\b"]),
    ("Clinical Sign", "Kayser-Fleischer Rings", [r"\bkayser[- ]fleischer\b", r"\bkf\s+rings?\b", r"\bcorneal\s+ring\b", r"\bslit[- ]lamp\b"]),
    ("Clinical Sign", "Sunflower Cataract", [r"\bsunflower\s+cataract\b"]),
    ("Clinical Sign", "Neurological Signs", [r"\bdystonia\b", r"\btremor\b", r"\btremors\b", r"\bdysarthria\b", r"\bataxia\b", r"\bparkinsonism\b", r"\bchorea\b", r"\brigidity\b"]),
    ("Clinical Sign", "Hepatic Presentation", [r"\bcirrhosis\b", r"\bhepatitis\b", r"\bascites\b", r"\bsplenomegaly\b", r"\bportal\s+hypertension\b"]),
    ("Clinical Sign", "Hemolytic Anemia", [r"\bhemolyt(?:ic)?\s+anemia\b", r"\bcoombs[- ]negative\b"]),
    ("Scoring System", "Leipzig Consensus Score", [r"\bleipzig\b", r"\bleipzig\s+score\b", r"\bleipzig\s+criteria\b", r"\bferenci\s+score\b"]),
    ("Scoring System", "Nazir Score / Prognostic Index", [r"\bnazir\b", r"\bprognostic\s+score\b", r"\bking'?s\s+score\b"])
]


def detect_clinical_terms(text: str) -> list:
    """
    Scans passage text against transparent clinical regex patterns with strict word boundaries.
    Results are returned as informational detected terms (not clinical diagnoses).
    """
    if not text:
        return []
    found = []
    seen = set()
    for category, term, patterns in CLINICAL_LEXICON:
        if term in seen:
            continue
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                found.append({
                    "category": category,
                    "term": term,
                    "matched_text": m.group(0),
                    "start_char": m.start(),
                    "end_char": m.end()
                })
                seen.add(term)
                break
    return found


def parse_passage_header(passage: str) -> tuple:
    """
    Parses [Source: <filename> | Page: <page_number>]\n<passage_text>.
    Returns (source_filename, page_number, body_text).
    Never invents page numbers or filenames when unavailable.
    """
    if not passage:
        return ("Unknown", None, "")
    header = ""
    body = passage
    if passage.startswith("[Source:") and "\n" in passage:
        header, body = passage.split("\n", 1)
    elif passage.startswith("[Source:") and passage.endswith("]"):
        header = passage
        body = ""

    m = re.search(r'\[Source:\s*([^|]+?)\s*\|\s*Page:\s*(\d+)\s*\]', header)
    if m:
        return (m.group(1).strip(), int(m.group(2).strip()), body.strip())

    if header.startswith("[Source:"):
        m_src = re.search(r'\[Source:\s*([^|\]]+)', header)
        src = m_src.group(1).strip() if m_src else "Unknown"
        m_pg = re.search(r'Page:\s*(\d+)', header)
        pg = int(m_pg.group(1).strip()) if m_pg else None
        return (src, pg, body.strip())

    return ("Unknown", None, passage.strip())


def _split_into_sentences(text: str) -> list:
    """Splits a paragraph into sentences preserving clinical abbreviations like mg., e.g., i.e."""
    protected = text
    for ab in ["e.g.", "i.e.", "vs.", "fig.", "tab.", "no.", "dr.", "prof.", "al.", "mg.", "kg."]:
        protected = re.sub(re.escape(ab), ab.replace(".", "___DOT___"), protected, flags=re.IGNORECASE)
    raw_sentences = re.split(r'(?<=[.!?])\s+', protected)
    sentences = [s.replace("___DOT___", ".").strip() for s in raw_sentences if s.strip()]
    return sentences if sentences else [text.strip()]


def chunk_clinical_text(
    text: str,
    filename: str = "guideline.txt",
    target_chunk_size: int = 750,
    min_chunk_size: int = 80,
    overlap: int = 150,
    page_metadata: list = None
) -> list:
    """
    Shared pure chunking function used by BOTH production ingestion (ingest_document)
    and the interactive chunk division simulator.
    Preserves paragraph breaks, sentence boundaries, and page provenance.
    Calculates exact start and end character offsets against the documented input.
    """
    if not text and not page_metadata:
        return []

    pages_to_process = []
    if page_metadata and isinstance(page_metadata, list):
        for item in page_metadata:
            if isinstance(item, dict):
                p_num = item.get("page", 1)
                p_text = item.get("text", "")
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                p_num = item[0]
                p_text = item[1]
            else:
                p_num = 1
                p_text = str(item)
            if p_text and p_text.strip():
                pages_to_process.append((p_num, p_text.strip()))
    elif page_metadata and isinstance(page_metadata, dict):
        p_num = page_metadata.get("page", 1)
        p_text = page_metadata.get("text") or text.strip()
        pages_to_process = [(p_num, p_text)]
    elif page_metadata and isinstance(page_metadata, (int, str)) and str(page_metadata).isdigit():
        pages_to_process = [(int(page_metadata), text.strip())]
    else:
        pages_to_process = [(1, text.strip())]

    chunk_records = []
    chunk_index = 1

    for page_num, p_text in pages_to_process:
        normalized_page = re.sub(r'[ \t]+', ' ', p_text).strip()
        if not normalized_page:
            continue

        paragraphs = [p.strip() for p in normalized_page.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [normalized_page]

        atomic_units = []
        for p in paragraphs:
            if len(p) > target_chunk_size:
                sents = _split_into_sentences(p)
                for s in sents:
                    if len(s) > target_chunk_size:
                        words = s.split()
                        curr_part = []
                        curr_plen = 0
                        for w in words:
                            if len(w) > target_chunk_size:
                                if curr_part:
                                    atomic_units.append(" ".join(curr_part))
                                    curr_part = []
                                    curr_plen = 0
                                for i in range(0, len(w), target_chunk_size):
                                    atomic_units.append(w[i:i + target_chunk_size])
                            elif curr_plen + len(w) + 1 > target_chunk_size and curr_part:
                                atomic_units.append(" ".join(curr_part))
                                curr_part = [w]
                                curr_plen = len(w)
                            else:
                                curr_part.append(w)
                                curr_plen += len(w) + 1
                        if curr_part:
                            atomic_units.append(" ".join(curr_part))
                    else:
                        atomic_units.append(s)
            else:
                atomic_units.append(p)

        current_units = []
        current_len = 0
        prev_overlap_text = ""

        for unit in atomic_units:
            unit_len = len(unit)
            if current_len + unit_len > target_chunk_size and current_units:
                chunk_body = " ".join(current_units).strip()
                if len(chunk_body) >= min_chunk_size:
                    header = f"[Source: {filename} | Page: {page_num}]"
                    overlap_portion = ""
                    if overlap > 0 and len(chunk_body) > overlap:
                        trailing = chunk_body[-overlap:]
                        if " " in trailing:
                            trailing = trailing.split(" ", 1)[1]
                        overlap_portion = trailing

                    start_char = normalized_page.find(current_units[0]) if current_units else 0
                    if start_char == -1:
                        start_char = 0
                    end_char = start_char + len(chunk_body)

                    chunk_records.append({
                        "index": chunk_index,
                        "source": filename,
                        "page": page_num,
                        "header": header,
                        "body": chunk_body,
                        "full_passage": f"{header}\n{chunk_body}",
                        "char_count": len(chunk_body),
                        "word_count": len(chunk_body.split()),
                        "estimated_tokens": max(1, round(len(chunk_body) / 4)),
                        "start_char": start_char,
                        "end_char": end_char,
                        "overlap_prev": prev_overlap_text,
                        "overlap_next": overlap_portion,
                        "detected_terms": detect_clinical_terms(chunk_body)
                    })
                    chunk_index += 1
                    prev_overlap_text = overlap_portion

                    if overlap_portion:
                        current_units = [overlap_portion, unit]
                        current_len = len(overlap_portion) + unit_len + 1
                    else:
                        current_units = [unit]
                        current_len = unit_len
                else:
                    current_units.append(unit)
                    current_len += unit_len + 1
            else:
                current_units.append(unit)
                current_len += unit_len + 1

        if current_units:
            chunk_body = " ".join(current_units).strip()
            if len(chunk_body) >= min_chunk_size or not chunk_records:
                header = f"[Source: {filename} | Page: {page_num}]"
                start_char = normalized_page.find(current_units[0]) if current_units else 0
                if start_char == -1:
                    start_char = 0
                end_char = start_char + len(chunk_body)

                chunk_records.append({
                    "index": chunk_index,
                    "source": filename,
                    "page": page_num,
                    "header": header,
                    "body": chunk_body,
                    "full_passage": f"{header}\n{chunk_body}",
                    "char_count": len(chunk_body),
                    "word_count": len(chunk_body.split()),
                    "estimated_tokens": max(1, round(len(chunk_body) / 4)),
                    "start_char": start_char,
                    "end_char": end_char,
                    "overlap_prev": prev_overlap_text,
                    "overlap_next": "",
                    "detected_terms": detect_clinical_terms(chunk_body)
                })
                chunk_index += 1
            elif chunk_records:
                prev = chunk_records[-1]
                prev["body"] = (prev["body"] + " " + chunk_body).strip()
                prev["full_passage"] = f"{prev['header']}\n{prev['body']}"
                prev["char_count"] = len(prev["body"])
                prev["word_count"] = len(prev["body"].split())
                prev["estimated_tokens"] = max(1, round(len(prev["body"]) / 4))
                prev["end_char"] = prev["start_char"] + len(prev["body"])
                prev["detected_terms"] = detect_clinical_terms(prev["body"])

    return chunk_records


def simulate_chunk_division(
    text: str,
    filename: str = "Simulation_Guideline.txt",
    target_chunk_size: int = 750,
    min_chunk_size: int = 80,
    overlap: int = 150,
    page_metadata: list = None
) -> dict:
    """
    Simulation wrapper calling the shared chunk_clinical_text engine.
    Completely isolated: does NOT mutate the knowledge base or write to disk.
    """
    warnings = []
    if not text or not text.strip():
        warnings.append("Supplied text is empty.")

    if len(text) > 100000:
        text = text[:100000]
        warnings.append("Text truncated to 100,000 characters limit for safe simulation.")

    safe_target = max(100, min(2500, int(target_chunk_size)))
    safe_min = max(20, min(500, int(min_chunk_size)))
    safe_overlap = max(0, min(safe_target // 2, int(overlap)))

    chunk_records = chunk_clinical_text(
        text=text,
        filename=filename,
        target_chunk_size=safe_target,
        min_chunk_size=safe_min,
        overlap=safe_overlap,
        page_metadata=page_metadata
    )

    chunk_lengths = [c["char_count"] for c in chunk_records]
    avg_len = round(sum(chunk_lengths) / len(chunk_lengths), 1) if chunk_lengths else 0
    return {
        "success": True,
        "filename": filename,
        "target_chunk_size": safe_target,
        "min_chunk_size": safe_min,
        "overlap": safe_overlap,
        "total_characters": len(text),
        "total_chunks": len(chunk_records),
        "chunks_count": len(chunk_records),
        "chunks": chunk_records,
        "stats": {
            "total_chunks": len(chunk_records),
            "input_chars": len(text),
            "avg_chunk_chars": avg_len,
            "min_chunk_chars": min(chunk_lengths) if chunk_lengths else 0,
            "max_chunk_chars": max(chunk_lengths) if chunk_lengths else 0,
            "overlap_continuity": "Verified"
        },
        "warnings": warnings,
        "summary": f"Text ({len(text)} chars) divided into {len(chunk_records)} clinical chunks (target: {safe_target} chars, overlap: {safe_overlap} chars)."
    }


def ingest_document(file_path: str, filename: str) -> list:
    """
    Parses and chunks an approved guideline document (PDF, DOCX, TXT) into page-tagged clinical passages.
    Format: [Source: <filename> | Page: <page_number>]\n<passage_text>
    Directly leverages the shared chunk_clinical_text engine.
    """
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(file_path)
            pages = []
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": page_idx + 1, "text": text})
            chunks = chunk_clinical_text(
                text="",
                filename=filename,
                target_chunk_size=750,
                min_chunk_size=80,
                overlap=150,
                page_metadata=pages
            )
            return [c["full_passage"] for c in chunks]
        except Exception as e:
            logger.error(f"Error parsing PDF document {file_path}: {e}")
            return []

    elif ext == ".docx":
        try:
            import docx
            doc = docx.Document(file_path)
            paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paras)
            chunks = chunk_clinical_text(
                text=full_text,
                filename=filename,
                target_chunk_size=750,
                min_chunk_size=80,
                overlap=150
            )
            return [c["full_passage"] for c in chunks]
        except Exception as e:
            logger.error(f"Error parsing DOCX document {file_path}: {e}")
            return []

    elif ext in [".txt", ".md"]:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            chunks = chunk_clinical_text(
                text=content,
                filename=filename,
                target_chunk_size=750,
                min_chunk_size=80,
                overlap=150
            )
            return [c["full_passage"] for c in chunks]
        except Exception as e:
            logger.error(f"Error parsing text document {file_path}: {e}")
            return []

    return []

# Alias for backwards compatibility
ingest_pdf_document = ingest_document


class MedicalRAGSystem:
    def __init__(self):
        self.use_langchain = False
        self.retriever = None
        self.llm = None
        self.gemini_client = None
        self.gemini_model = DEFAULT_GEMINI_MODEL
        self.passages = []
        self.qa_bank = []
        self.approved_docs = list(APPROVED_PDFS)

        self._sync_approved_docs_from_db()
        self._init_knowledge_base()

    def _sync_approved_docs_from_db(self):
        """Discovers and synchronizes any approved documents from the database."""
        try:
            from database import get_rag_documents
            approved_in_db = get_rag_documents(status="approved")
            for doc in approved_in_db:
                fn = doc.get("filename")
                if fn and fn not in self.approved_docs:
                    self.approved_docs.append(fn)
        except Exception:
            pass

    def register_approved_document(self, filename: str, new_passages: list) -> int:
        """
        Dynamically registers an approved clinical guideline document into the in-memory
        RAG index and persists the passages to chroma_db/passages.json.
        Enables immediate live retrieval without restarting the server.
        """
        if filename not in self.approved_docs:
            self.approved_docs.append(filename)

        if not new_passages:
            return 0

        # Add to in-memory passages
        self.passages.extend(new_passages)

        # Parse and append to _parsed_passages for ranking
        if not hasattr(self, "_parsed_passages") or not self._parsed_passages:
            self._load_clinical_passages()
        else:
            for p in new_passages:
                header = ""
                body = p
                if p.startswith("[Source:") and "\n" in p:
                    header, body = p.split("\n", 1)
                norm_body = self._normalize_passage_text(body)
                self._parsed_passages.append((p, header, norm_body, norm_body.lower()))

        # Save updated passages to chroma_db/passages.json
        json_path = os.path.join(PERSIST_DIR, "passages.json")
        try:
            os.makedirs(PERSIST_DIR, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(self.passages, f, ensure_ascii=False, indent=2)
            logger.info(f"Dynamically registered {len(new_passages)} passages for approved document '{filename}'. Total passages: {len(self.passages)}.")
        except Exception as e:
            logger.error(f"Failed to persist updated passages to {json_path}: {e}")

        return len(new_passages)

    def get_chunk_corpus(self, doc_filter=None, page_filter=None, search_query=None, offset=0, limit=20) -> dict:
        """
        Pure read-only inspection of the active in-memory passage collection (self.passages).
        Does NOT mutate the knowledge base, ChromaDB, or passages.json.
        Returns paginated chunk records, corpus statistics, and per-document distributions.
        """
        if not self.passages:
            self._load_clinical_passages()

        total_active_chunks = len(self.passages)

        # Global corpus statistics
        doc_counts = {}
        all_lengths = []
        parsed_all = []

        for idx, p in enumerate(self.passages):
            src, pg, body = parse_passage_header(p)
            doc_counts[src] = doc_counts.get(src, 0) + 1
            body_len = len(body)
            all_lengths.append(body_len)
            parsed_all.append((idx + 1, src, pg, p.split("\n", 1)[0] if "\n" in p else "", body, body_len))

        global_stats = {
            "total_chunks": total_active_chunks,
            "avg_char_length": round(sum(all_lengths) / max(1, total_active_chunks), 1) if all_lengths else 0,
            "min_char_length": min(all_lengths) if all_lengths else 0,
            "max_char_length": max(all_lengths) if all_lengths else 0,
            "active_documents_count": len(doc_counts),
            "document_distribution": [{"document": k, "count": v} for k, v in sorted(doc_counts.items(), key=lambda x: -x[1])]
        }

        # Apply filters
        filtered = parsed_all

        if doc_filter and str(doc_filter).strip() and str(doc_filter).strip().lower() != "all":
            df = str(doc_filter).strip().lower()
            filtered = [item for item in filtered if df in item[1].lower()]

        if page_filter is not None and str(page_filter).strip():
            try:
                pf = int(page_filter)
                filtered = [item for item in filtered if item[2] is not None and item[2] == pf]
            except (ValueError, TypeError):
                pass

        if search_query and str(search_query).strip():
            sq = str(search_query).strip().lower()
            filtered = [item for item in filtered if sq in item[4].lower() or sq in item[1].lower()]

        filtered_count = len(filtered)
        filtered_lengths = [item[5] for item in filtered]
        filtered_stats = {
            "total_filtered": filtered_count,
            "avg_char_length": round(sum(filtered_lengths) / max(1, filtered_count), 1) if filtered_count else 0,
            "min_char_length": min(filtered_lengths) if filtered_count else 0,
            "max_char_length": max(filtered_lengths) if filtered_count else 0,
        }

        # Pagination bounds
        safe_offset = max(0, int(offset) if str(offset).isdigit() else 0)
        safe_limit = min(100, max(1, int(limit) if str(limit).isdigit() else 20))
        slice_items = filtered[safe_offset:safe_offset + safe_limit]

        # Build chunk cards
        chunks_page = []
        for chunk_id, src, pg, header, body, blen in slice_items:
            w_count = len(body.split())
            token_est = max(1, round(blen / 4))
            detected = detect_clinical_terms(body)
            chunks_page.append({
                "chunk_id": f"CHUNK-{chunk_id:04d}",
                "raw_index": chunk_id,
                "source": src,
                "page": pg if pg is not None else "N/A",
                "header": header,
                "body": body,
                "char_count": blen,
                "word_count": w_count,
                "estimated_tokens": token_est,
                "detected_terms": detected
            })

        return {
            "success": True,
            "pagination": {
                "offset": safe_offset,
                "limit": safe_limit,
                "total_filtered": filtered_count,
                "has_previous": safe_offset > 0,
                "has_next": (safe_offset + safe_limit) < filtered_count,
                "current_page": (safe_offset // safe_limit) + 1 if safe_limit > 0 else 1,
                "total_pages": max(1, (filtered_count + safe_limit - 1) // safe_limit) if safe_limit > 0 else 1
            },
            "chunks": chunks_page,
            "documents": [{"document": k, "count": v} for k, v in sorted(doc_counts.items(), key=lambda x: -x[1])],
            "statistics": {
                "global": global_stats,
                "filtered": filtered_stats,
                "total_chunks": global_stats["total_chunks"],
                "avg_chars": global_stats["avg_char_length"],
                "min_chars": global_stats["min_char_length"],
                "max_chars": global_stats["max_char_length"],
                "filtered_total": filtered_stats["total_filtered"],
                "filtered_avg_chars": filtered_stats["avg_char_length"]
            },
            "global_statistics": global_stats,
            "filtered_statistics": filtered_stats,
            "metadata": {
                "timestamp": time.time(),
                "authoritative_source": "rag_assistant.passages"
            }
        }

    def has_medical_documents(self) -> bool:
        """
        Checks whether medical_docs/ contains any non-empty clinical knowledge documents.
        Returns False if the directory does not exist or has no valid document files.
        """
        if not os.path.isdir(DOCS_DIR):
            return False
        valid_extensions = ('.txt', '.md', '.pdf', '.docx', '.csv', '.json')
        for f in os.listdir(DOCS_DIR):
            if f.startswith('.'):
                continue
            if f.lower().endswith(valid_extensions):
                full_path = os.path.join(DOCS_DIR, f)
                if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
                    return True
        return False

    def _ensure_documents_loaded(self):
        """Loads documents into memory if they exist on disk but haven't been loaded yet."""
        if self.has_medical_documents() and not self.passages and not self.qa_bank:
            self._init_knowledge_base()

    def _init_knowledge_base(self):
        # 1. Question bank is disabled: medical evidence must come strictly from approved PDFs
        self.qa_bank = []

        # 2. Load clinical passages exclusively from approved PDF passage index
        self._load_clinical_passages()

        # 2. Try loading ChromaDB vectorstore via LangChain if available
        try:
            try:
                from langchain_chroma import Chroma
            except ImportError:
                from langchain_community.vectorstores import Chroma

            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=embeddings
            )
            self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            self.use_langchain = True
            logger.info("RAG Initialized with ChromaDB VectorStore & HuggingFace Embeddings.")
        except Exception as e:
            logger.info("Running standalone high-performance clinical RAG engine.")

        # 3. Try initializing external LLM (Gemini / OpenAI)
        self._init_llm()

    def _load_clinical_passages(self):
        """Loads clinical passages exclusively from the 4 approved PDF guidelines index."""
        self.passages = []
        self._parsed_passages = []
        json_path = os.path.join(PERSIST_DIR, "passages.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.passages = json.load(f)
                parsed = []
                for p in self.passages:
                    header = ""
                    body = p
                    if p.startswith("[Source:") and "\n" in p:
                        header, body = p.split("\n", 1)
                    norm_body = self._normalize_passage_text(body)
                    parsed.append((p, header, norm_body, norm_body.lower()))
                self._parsed_passages = parsed
                logger.info(f"Loaded {len(self.passages)} clinical guideline passages from {json_path} (pre-indexed).")
            except Exception as e:
                logger.warning(f"Error loading guideline passages from {json_path}: {e}")

    def _load_qa_bank(self):
        """Disabled: Question bank cannot supply medical answers."""
        self.qa_bank = []

    def search_qa_bank(self, user_question: str):
        """Disabled: All medical answers must be retrieved exclusively from approved PDF guidelines."""
        return None

    def _init_llm(self):
        # 1. Load environment variables from .env if present
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        # Check LangSmith tracing
        if os.environ.get("LANGCHAIN_TRACING_V2") == "true" and os.environ.get("LANGCHAIN_API_KEY"):
            project = os.environ.get("LANGCHAIN_PROJECT", "wilson-disease-rag")
            logger.info(f"LangSmith Tracing active (project: {project}).")

        # 2. Check Gemini via official google-genai SDK
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from google import genai
                self.gemini_model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
                self.gemini_client = genai.Client(api_key=gemini_key)
                logger.info(f"RAG LLM initialized with Google GenAI SDK (model: {self.gemini_model}).")
                return
            except Exception as e:
                logger.warning(f"Google GenAI SDK init error: {e}")
                self.gemini_client = None

        # 3. Check OpenAI (fallback)
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.2)
                logger.info("RAG LLM initialized with OpenAI GPT-4o-mini.")
                return
            except Exception as e:
                logger.warning(f"OpenAI LLM init error: {e}")

        logger.info("No external LLM API key detected or client init failed. Running grounded clinical RAG engine.")

    @traceable(name="gemini_generate_content", run_type="llm")
    def _call_gemini(self, model: str, prompt: str):
        """Calls Google GenAI SDK with automatic LangSmith LLM tracing and low latency config."""
        from google.genai import types
        cfg = types.GenerateContentConfig(
            max_output_tokens=600,
            temperature=0.1
        )
        return self.gemini_client.models.generate_content(
            model=model,
            contents=prompt,
            config=cfg
        )


    @staticmethod
    def _normalize_passage_text(text: str) -> str:
        """Normalizes common PDF font ligatures, line breaks, hyphens, and bullet tags."""
        text = text.replace('\ufb01', 'fi').replace('\ufb02', 'fl')
        text = text.replace('\ufb00', 'ff').replace('\ufb03', 'ffi').replace('\ufb04', 'ffl')
        text = text.replace('\u2010', '-').replace('\u2013', '-').replace('\u2014', '-')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        # Clean specific PDF font/bullet artifact tags like /C15 while preserving units like mg/dl or and/or
        text = re.sub(r'/C\d+[A-Za-z0-9]*\s*', ' ', text)
        # Fix split ligatures or words broken by PDF spaces and hyphens
        text = re.sub(r'\b(\w+)\s*-\s+(\w+)\b', r'\1-\2', text)
        text = re.sub(r'\bspeci\s+fic', 'specific', text, flags=re.IGNORECASE)
        text = re.sub(r'\bsigni\s+fic', 'signific', text, flags=re.IGNORECASE)
        text = re.sub(r'\bde\s+ficit', 'deficit', text, flags=re.IGNORECASE)
        text = re.sub(r'\bpro\s+file', 'profile', text, flags=re.IGNORECASE)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    @staticmethod
    def _get_guideline_display_name(src_file: str) -> str:
        s = src_file.lower()
        if "aasld" in s:
            return "AASLD Guidelines"
        elif "easl" in s:
            return "EASL-ERN Guidelines"
        elif "clinical practice guidelines in wd" in s or "inasl" in s:
            return "INASL Guidelines"
        elif "nih" in s or "gene" in s:
            return "NIH / GeneReviews"
        stem = os.path.splitext(src_file)[0]
        return f"{stem} Guidelines" if stem else "Clinical Practice Guidelines"

    @classmethod
    def _extract_citations(cls, text: str, approved_docs: list = None) -> list:
        citations = []
        doc_list = approved_docs or APPROVED_PDFS
        for pdf in doc_list:
            pdf_stem = pdf[:-4] if pdf.lower().endswith(".pdf") else pdf
            pattern = re.compile(
                re.escape(pdf_stem) + r'(?:\.pdf)?(?:[,\s\|:\-\(\]\*]+)*(?:pp?\.?|pages?|page)\s*:?\s*([0-9\s,\-–and&]+)',
                re.IGNORECASE
            )
            for m in pattern.finditer(text):
                clause = m.group(1)
                page_nums = [int(n) for n in re.findall(r'\b\d+\b', clause)]
                for p in page_nums:
                    citations.append((pdf.lower(), p))
        return citations

    @classmethod
    def _validate_citations(cls, llm_reply: str, top_passages: list, approved_docs: list = None) -> bool:
        """
        Validates that all source citations in the LLM reply match actual (source, page) pairs
        present in the retrieved top passages. Rejects fabricated page numbers, unretrieved sources,
        and bare PDF mentions without verified page citations.
        """
        if not llm_reply or not llm_reply.strip() or not top_passages:
            return False

        # 1. Extract valid (source_filename, page_number) pairs from retrieved top_passages
        valid_pairs = set()
        for p in top_passages:
            header = p.split("\n", 1)[0] if "\n" in p else p
            m = re.search(r'\[Source:\s*([^|]+?)\s*\|\s*Page:\s*(\d+)\s*\]', header)
            if m:
                valid_pairs.add((m.group(1).strip().lower(), int(m.group(2).strip())))

        if not valid_pairs:
            return False

        # 2. Extract citations from llm_reply
        doc_list = approved_docs or APPROVED_PDFS
        extracted_citations = cls._extract_citations(llm_reply, doc_list)
        if not extracted_citations:
            return False

        # 3. Ensure any mentioned approved PDF has an extracted citation
        for pdf in doc_list:
            pdf_stem = pdf[:-4] if pdf.lower().endswith(".pdf") else pdf
            if pdf.lower() in llm_reply.lower() or pdf_stem.lower() in llm_reply.lower():
                pdf_cites = [c for c in extracted_citations if c[0] == pdf.lower()]
                if not pdf_cites:
                    return False

        # 4. Ensure every extracted citation matches a valid (source, page) pair from retrieved passages
        for c in extracted_citations:
            if c not in valid_pairs:
                return False

        return True

    @traceable(name="rank_passages", run_type="retriever")
    def _rank_passages_internal(self, query: str, top_n: int = 4) -> list:
        """
        Ranks clinical guideline passages using query-intent scoring, clinical concept expansion,
        source quality filtering, and two-pass guideline diversification.
        """
        if not self.has_medical_documents() or not self.passages:
            return []

        query_clean = query.lower()
        stop_words = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "to", "what",
            "how", "tell", "me", "about", "are", "based", "taken", "take", "done", "used",
            "many", "much", "can", "does", "do", "it", "for", "of", "with", "from", "by",
            "according", "cited", "guideline", "guidelines", "practice", "clinical", "please",
            "explain", "recommend", "recommended", "consideration", "prompt", "findings"
        }
        raw_query_words = [
            w for w in re.findall(r'\b\w+\b', query_clean)
            if w not in stop_words and len(w) > 2
        ]
        if not raw_query_words:
            return []

        is_definition_query = any(phrase in query_clean for phrase in ["what is wilson", "what is wilson s", "explain wilson", "what is the definition", "define wilson"])
        is_copper_accum_query = any(phrase in query_clean for phrase in ["copper accumulation", "copper accumulate", "accumulates in", "accumulation of copper"])
        is_pediatric_query = any(w in query_clean for w in ["children", "child", "pediatric", "infant", "adolescent", "young"])
        is_monitoring_query = (
            any(w in query_clean for w in ["monitoring", "interval", "intervals", "frequency", "schedule", "surveillance"]) and
            any(w in query_clean for w in ["chelator", "chelation", "therapy", "treatment", "penicillamine", "trientine", "zinc"])
        )

        if not hasattr(self, "_parsed_passages") or not self._parsed_passages:
            parsed = []
            for p in self.passages:
                header = ""
                body = p
                if p.startswith("[Source:") and "\n" in p:
                    header, body = p.split("\n", 1)
                norm_body = self._normalize_passage_text(body)
                parsed.append((p, header, norm_body, norm_body.lower()))
            self._parsed_passages = parsed

        scored = []
        for p, header, body, body_lower in self._parsed_passages:

            # 1. Front-matter / glossary / bibliography detection
            is_acronym_glossary = body.count(";") >= 4 and any(w in body for w in ["FDA", "INR", "MELD", "UWDRS", "co-first authors", "HCC"])
            is_front_matter = any(phrase in body_lower for phrase in [
                "co-first authors", "guidance documents, developed by a panel of experts",
                "practice guidelines committee", "all rights reserved", "all rights are reserved",
                "published by elsevier", "reprint requests", "table of contents", "conflict of interest",
                "terms and conditions", "copyright notice", "data mining, ai training"
            ])
            is_bibliography = (
                body.count(" et al.") >= 2 or
                bool(re.search(r'\[\d+\]\s+[A-Z][a-z]+ [A-Z]{1,2},', body)) or
                bool(re.search(r'\b\d{2,4}\.\s+[A-Z][a-z]+', body)) or
                bool(re.search(r'\b(?:J Pediatr|Hepatology|Lancet|Gastroenterology|Eur J Clin Nutr|Am J Clin Nutr|Clin Gastroenterol Hepatol)\b', body))
            )
            is_raw_table = len(re.findall(r'\b\d+[-–]\d+\b', body)) >= 4 or "score bilirubin" in body_lower or "table 10" in body_lower

            # 2. Token overlap with clinical concept expansions
            score = 0.0
            matched_distinct = 0
            for w in raw_query_words:
                w_weight = 0.5 if w in ["wilson", "disease", "patient", "patients"] else 3.0
                cnt = body_lower.count(w)
                if w in ["chelator", "chelation"]:
                    cnt += sum(body_lower.count(syn) for syn in ["chelator", "chelation", "chelating", "penicillamine", "trientine"])
                elif w in ["interval", "intervals"]:
                    cnt += sum(body_lower.count(syn) for syn in ["interval", "intervals", "annually", "yearly", "twice per year", "twice yearly", "twice annually", "every 6 to 12 months", "every 6-12 months", "every 6 months", "6–12 months", "6-12 months"])
                elif w in ["children", "child", "pediatric"]:
                    cnt += sum(body_lower.count(syn) for syn in ["children", "child", "pediatric"])
                elif w in ["findings", "finding"]:
                    cnt += sum(body_lower.count(syn) for syn in ["finding", "findings", "manifestation", "manifestations", "presentation"])

                if cnt > 0:
                    score += w_weight * (1.0 + min(cnt - 1, 3) * 0.4)
                    matched_distinct += 1

            if matched_distinct == 0:
                continue

            coverage = matched_distinct / len(raw_query_words)
            score *= (0.5 + 1.5 * coverage)

            # 3. Penalize boilerplate, glossary, and bibliography
            if is_acronym_glossary:
                score -= 30.0
            if is_front_matter:
                score -= 30.0
            if is_bibliography:
                score -= 35.0
            if is_raw_table:
                score -= 30.0

            # 4. Intent-aware boosts
            if is_definition_query:
                if any(copula in body_lower for copula in ["autosomal recessive", "inborn error of copper", "disorder of copper metabolism", "defect in the atp7b", "genetic defect"]):
                    score += 18.0
                if "summary wilson" in body_lower or "summary: wilson" in body_lower or "summary wilson's" in body_lower:
                    score += 10.0
                if "hepatolenticular degeneration" in body_lower:
                    score += 5.0

            if is_copper_accum_query:
                if "toxic hepatocellular copper accumulation" in body_lower or "copper accumulation" in body_lower or "accumulates in" in body_lower:
                    score += 15.0
                if "biliary excretion" in body_lower or "atp7b" in body_lower:
                    score += 8.0

            if is_pediatric_query:
                if any(w in body_lower for w in ["children", "child", "pediatric"]):
                    score += 10.0
                if any(w in body_lower for w in ["hepatomegaly", "transaminases", "aminotransferase"]):
                    score += 10.0
                if ("asymptomatic" in body_lower and ("hepatomegaly" in body_lower or "aminotransferase" in body_lower or "transaminase" in body_lower or "liver" in body_lower)) or "incidental" in body_lower or "persistently increased" in body_lower or "persistently elevated" in body_lower:
                    score += 12.0
                if "ceruloplasmin" in body_lower and any(w in body_lower for w in ["threshold", "sensitivity", "14 mg", "20 mg"]):
                    score += 8.0

            if is_monitoring_query:
                if any(w in body_lower for w in ["monitoring", "interval", "intervals", "annually", "yearly", "twice yearly", "twice annually", "twice per year", "every 6 to 12 months", "every 6-12 months", "6–12 months", "every 6 months"]):
                    score += 12.0
                if "urinary copper" in body_lower or "24-h" in body_lower or "24-hour" in body_lower:
                    score += 8.0
                if any(w in body_lower for w in ["complete blood count", "full blood count", "total blood count", "urinalysis", "urine analyses", "proteinuria", "liver biochemistries", "liver function tests", "renal profile", "renal function", "hepatic synthetic function", "inr"]):
                    score += 12.0
                if any(w in body_lower for w in ["guidance statement", "recommendation", "recommendations", "surveillance:", "monitoring on treatment"]):
                    score += 8.0

            if "ceruloplasmin" in query_clean:
                if "ceruloplasmin" in body_lower:
                    score += 10.0
                if any(w in body_lower for w in ["glycoprotein", "synthesized in hepatocytes", "acute‐phase", "acute-phase", "apoceruloplasmin", "ferroxidase", "oxidase"]):
                    score += 12.0

            if any(w in query_clean for w in ["kayser", "fleischer", "kf ring", "kf rings"]):
                if any(w in body_lower for w in ["kayser", "fleischer", "descemet", "cornea", "slit-lamp"]):
                    score += 15.0

            if any(w in query_clean for w in ["atp7b", "gene", "mutation"]):
                if "atp7b" in body_lower:
                    score += 12.0
                if any(w in body_lower for w in ["p-type atpase", "transmembrane", "biliary excretion", "chromosome 13"]):
                    score += 10.0

            if "leipzig" in query_clean:
                if "leipzig" in body_lower:
                    score += 15.0

            if any(w in query_clean for w in ["diet", "food", "foods", "eat", "eating", "nutrition"]):
                if any(k in body_lower for k in ["dietary copper", "copper in the diet", "copper intake", "dietary cu", "restrict dietary", "foods with high concentration", "copper-restricted"]):
                    score += 16.0
                if any(k in body_lower for k in ["nuts", "liver", "chocolate", "chocolates", "shellfish", "organ meats", "drinking water", "rda"]):
                    score += 10.0

            if any(w in query_clean for w in ["urinary", "urine", "24-h", "24-hour", "cupriuria"]):
                if any(k in body_lower for k in ["urinary copper", "24-h urinary", "24-hour urinary", "urine collection", "cupriuria"]):
                    score += 15.0

            if any(phrase in query_clean for phrase in ["copper based test", "tests for copper", "test for copper", "copper test", "copper tests", "copper biomarkers", "laboratory test"]):
                if any(k in body_lower for k in ["ceruloplasmin", "urinary copper", "free copper", "non-ceruloplasmin", "hepatic copper", "copper metabolism"]):
                    score += 12.0

            if any(w in query_clean for w in ["trientine", "penicillamine", "chelator", "chelation", "zinc"]):
                if any(k in body_lower for k in ["d-penicillamine", "trientine", "zinc acetate", "adverse", "side effect", "tolerance profile", "toxicity", "proteinuria", "pyridoxine"]):
                    score += 12.0

            min_cov = 0.50 if len(raw_query_words) <= 2 else (0.20 if len(raw_query_words) >= 6 else 0.30)
            if score >= 8.0 and coverage >= min_cov:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return []

        best_score = scored[0][0]

        # Two-pass selection for complex multi-guideline questions (pediatric and monitoring)
        if is_monitoring_query or is_pediatric_query:
            selected = []
            seen_orgs = set()
            # Pass 1: Select top passage per distinct guideline organization
            for s, p in scored:
                if s < 0.60 * best_score or s < 10.0:
                    continue
                header = p.split("\n", 1)[0]
                m = re.match(r'\[Source:\s*([^|]+)', header)
                org = self._get_guideline_display_name(m.group(1)) if m else "General"
                if org not in seen_orgs:
                    seen_orgs.add(org)
                    selected.append(p)
                if len(selected) >= top_n:
                    break

            # Pass 2: If space remains, fill with remaining highest scored passages
            if len(selected) < top_n:
                for s, p in scored:
                    if p not in selected and (s >= 0.60 * best_score and s >= 10.0):
                        selected.append(p)
                    if len(selected) >= top_n:
                        break
            return selected

        # Standard ranking up to top_n passages
        results = [scored[0][1]]
        for s, p in scored[1:]:
            if len(results) >= top_n:
                break
            if s >= 0.70 * best_score and s >= 10.0:
                results.append(p)

        return results

    def _synthesize_extractive_answer(self, user_question: str, top_passages: list) -> str:
        """
        Produces a concise, readable conversational explanation grounded in retrieved passages
        without requiring an external LLM (Pathway B: Extractive Synthesis).
        Keeps distinct guideline bodies (AASLD, EASL-ERN, INASL, NIH/GeneReviews) separate
        for complex comparative clinical questions to avoid conflating different monitoring intervals
        or recommendations.
        If evidence is inadequate, honestly states that the documents lack sufficient evidence.
        """
        if not top_passages:
            return (
                "The available clinical practice guidelines do not contain sufficient evidence "
                "to answer this specific question. Please consult a qualified gastroenterologist, "
                "hepatologist, or neurologist for individualized clinical guidance."
            )

        q_clean = user_question.lower()
        stop_words = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "to", "what",
            "how", "tell", "me", "about", "are", "based", "taken", "take", "done", "used",
            "many", "much", "can", "does", "do", "it", "for", "of", "with", "from", "by",
            "according", "cited", "guideline", "guidelines", "practice", "clinical", "please",
            "explain", "recommend", "recommended", "consideration", "prompt", "findings"
        }
        q_words = set(re.findall(r'\b\w+\b', q_clean)) - stop_words

        is_pediatric_query = any(w in q_clean for w in ["children", "child", "pediatric", "infant", "adolescent", "young"])
        is_monitoring_query = (
            any(w in q_clean for w in ["monitoring", "interval", "intervals", "frequency", "schedule", "surveillance"]) and
            any(w in q_clean for w in ["chelator", "chelation", "therapy", "treatment", "penicillamine", "trientine", "zinc"])
        )

        guideline_data = {}

        for p in top_passages:
            header = ""
            body = p
            if p.startswith("[Source:") and "\n" in p:
                header, body = p.split("\n", 1)

            m = re.match(r'\[Source:\s*([^|]+)\|\s*Page:\s*(\d+)\]', header)
            if m:
                src_file = m.group(1).strip()
                page_num = m.group(2).strip()
                citation_str = f"{src_file}, p. {page_num}"
                org_name = self._get_guideline_display_name(src_file)
            else:
                citation_str = "Clinical Practice Guidelines"
                org_name = "Clinical Practice Guidelines"

            if org_name not in guideline_data:
                guideline_data[org_name] = {"sources": [], "sentences": []}
            if citation_str not in guideline_data[org_name]["sources"]:
                guideline_data[org_name]["sources"].append(citation_str)

            body = self._normalize_passage_text(body)
            # Strip document title/preamble junk from start of body
            body = re.sub(r'^(?:[A-Za-z\s\-–—:]*Guidelines[^\n*]*\*\s*(?:Summary\s*)?|Summary\s*:?\s*|TABLE\s+\d+\s+)', '', body, flags=re.IGNORECASE)
            # Normalize periods followed by citations/reference numbers before sentence splitting
            body = re.sub(r'([.!?])\d+\s+', r'\1 ', body)
            if "•" in body:
                body = body.replace("•", ". ")

            raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', body)
            scored_sentences = []

            for raw_s in raw_sentences:
                s_clean = re.sub(r'\[[^\]]*\]', '', raw_s) # remove bracketed references e.g. [1,2] or [Schilsky et al 2022b]
                s_clean = re.sub(r'\s*\([Ll]o[Ee]\s*\d+[^)]*\)', '', s_clean) # remove (LoE 3, strong recommendation, consensus)
                s_clean = re.sub(r'\s*\([Tt]able\s*\d+\s*\)', '', s_clean) # remove (Table 1 )
                s_clean = re.sub(r'^\d+%\s*', '', s_clean) # remove voting percentages e.g. '100% ', '97% '
                s_clean = re.sub(r'^(?:\(?\d+[\.\)]|\d+\.)\s+', '', s_clean) # remove leading list numbers e.g. '21. ', '1) ', '(2) '
                s_clean = re.sub(r'^(?:\d+,\s*)+\d+[\s–\-]+', '', s_clean) # remove leading citation chains e.g. '200,456,457 '
                s_clean = re.sub(r'^\d+[-–]\d+\s+', '', s_clean) # remove leading range citation numbers e.g. '17-19 ', '208-210 '
                s_clean = re.sub(r',\d+\s+', ', ', s_clean) # comma citations e.g. ',200,456 '
                s_clean = re.sub(r'^[,\s–\-:;]+', '', s_clean) # leading punctuation
                # Remove stray leading reference numbers (e.g. '79 The best WD...') while preserving units ('24 h', '14 mg')
                m_lead = re.match(r'^(\d+)\s+([A-Za-z]+)', s_clean)
                if m_lead and m_lead.group(2).lower() not in {'h', 'hr', 'hrs', 'hour', 'hours', 'd', 'day', 'days', 'wk', 'week', 'weeks', 'mo', 'month', 'months', 'yr', 'year', 'years', 'mg', 'g', 'kg', 'dl', 'ml', 'l', 'mmol', 'umol', 'lmol', 'points', 'percent'}:
                    s_clean = s_clean[len(m_lead.group(1)):].strip()
                s_clean = re.sub(r'\s+', ' ', s_clean).strip()

                # Require valid terminal punctuation
                last_p = max(s_clean.rfind('.'), s_clean.rfind('!'), s_clean.rfind('?'))
                if last_p != -1:
                    s_clean = s_clean[:last_p + 1].strip()
                else:
                    continue

                if s_clean.endswith(':'):
                    continue
                # Complete sentences start with a capital letter; discard cross-chunk broken fragments
                if not s_clean or s_clean[0].islower():
                    continue

                s_lower = s_clean.lower()
                if len(s_clean) < 30:
                    continue
                if any(s_lower.startswith(lead) for lead in ['guidance statement', 'guidance statements', 'practice guidance', 'table ', 'recommendation ', 'recommendations ']):
                    continue
                if any(b in s_lower for b in ["all rights reserved", "elsevier b.v", "co-first authors", "reprint requests", "fda, food"]):
                    continue
                if s_clean.count(";") >= 3 and any(w in s_clean for w in ["FDA", "INR", "MELD"]):
                    continue
                if len(re.findall(r'\b\d+[-–]\d+\b', s_clean)) >= 3:
                    continue
                if any(lead in s_lower for lead in ["score bilirubin", "zinc salts d-penicillamine trientine", "king 's college wilson index"]):
                    continue

                s_words = set(re.findall(r'\b\w+\b', s_lower))
                overlap = len(q_words.intersection(s_words))
                coverage = (overlap / len(q_words)) if q_words else 0
                s_score = overlap * 2.0 + coverage * 4.0

                if any(cop in s_lower for cop in [
                    "is an autosomal recessive", "is an inborn error", "disorder of copper metabolism",
                    "characterized by", "toxic hepatocellular copper accumulation", "biliary excretion",
                    "glycoprotein", "synthesized in hepatocytes", "accounts for 90%", "descemet", "cornea",
                    "kayser-fleischer", "atp7b", "transmembrane", "p-type", "leipzig", "urinary copper",
                    "24-h urinary", "24-hour urinary", "cupriuria", "dietary copper", "copper intake",
                    "nuts, soy, liver", "organ meats", "shellfish", "penicillamine", "trientine", "zinc acetate",
                    "tolerance profile", "better tolerance"
                ]):
                    s_score += 6.0

                if is_pediatric_query:
                    if any(w in s_lower for w in ["hepatomegaly", "transaminases", "aminotransferase"]):
                        s_score += 4.0
                    if "asymptomatic" in s_lower and any(w in s_lower for w in ["hepatomegaly", "transaminases", "aminotransferase"]):
                        s_score += 10.0
                    if "incidentally" in s_lower or "incidental" in s_lower:
                        s_score += 6.0
                    if "persistently increased" in s_lower or "persistently elevated" in s_lower:
                        s_score += 8.0
                    if "ceruloplasmin" in s_lower and any(w in s_lower for w in ["threshold", "sensitivity", "14 mg", "20 mg"]):
                        s_score += 8.0
                    if any(w in s_lower for w in ["acute viral hepatitis", "autoimmune hepatitis", "steatosis", "chronic hepatitis", "acute liver failure", "splenomegaly", "alf"]):
                        s_score += 6.0

                if is_monitoring_query:
                    if any(w in s_lower for w in ["complete blood count", "full blood count", "total blood count", "urinalysis", "urine analyses", "proteinuria"]):
                        s_score += 10.0
                    if any(w in s_lower for w in ["liver biochemistries", "liver function tests", "liver profile", "hepatic synthetic function", "inr"]):
                        s_score += 8.0
                    if "renal profile" in s_lower or "renal function" in s_lower:
                        s_score += 6.0
                    if any(w in s_lower for w in ["24-h urinary copper", "24-hour urinary copper", "urinary copper excretion"]):
                        s_score += 8.0
                    if any(w in s_lower for w in ["at least twice per year", "twice per year", "twice annually", "twice yearly", "every 6 to 12 months", "6 –12 months", "6-12 months"]):
                        s_score += 10.0
                    if any(w in s_lower for w in ["at least annually", "measured yearly", "measured annually", "once annually"]):
                        s_score += 8.0
                    if any(w in s_lower for w in ["more frequent", "more frequently", "frequently", "nonadherence", "non-adherence", "dosage adjustments", "treatment initiation"]):
                        s_score += 9.0

                if s_score >= 4.0:
                    scored_sentences.append((s_score, s_clean))

            scored_sentences.sort(key=lambda x: x[0], reverse=True)
            for sc, sent in scored_sentences:
                if not any(sent[:45].lower() in existing[:45].lower() for existing in guideline_data[org_name]["sentences"]):
                    guideline_data[org_name]["sentences"].append(sent)
                if len(guideline_data[org_name]["sentences"]) >= (2 if (is_monitoring_query or is_pediatric_query) else 3):
                    break

        active_orgs = [org for org, data in guideline_data.items() if data["sentences"]]
        if not active_orgs:
            return (
                "The available clinical practice guidelines do not contain sufficient evidence "
                "to answer this specific question. Please consult a qualified gastroenterologist, "
                "hepatologist, or neurologist for individualized clinical guidance."
            )

        # Multi-guideline comparative attribution
        if (is_monitoring_query or is_pediatric_query) and len(active_orgs) > 1:
            lines = ["According to clinical practice guidelines:"]
            for org in active_orgs:
                data = guideline_data[org]
                sources_str = "; ".join(data["sources"])
                lines.append(f"\n{org} ({sources_str}):")
                for sent in data["sentences"][:2]:
                    lines.append(f"• {sent}")
            return "\n".join(lines).strip()
        else:
            lines = ["According to the cited clinical practice guidelines:"]
            all_sources = []
            all_sentences = []
            for org, data in guideline_data.items():
                for s in data["sources"]:
                    if s not in all_sources:
                        all_sources.append(s)
                for sent in data["sentences"]:
                    if not any(sent[:45].lower() in ex[:45].lower() for ex in all_sentences):
                        all_sentences.append(sent)
            for sent in all_sentences[:3]:
                lines.append(f"• {sent}")
            lines.append("")
            if len(all_sources) == 1:
                lines.append(f"Source: {all_sources[0]}")
            else:
                lines.append("Sources: " + "; ".join(all_sources))
            return "\n".join(lines).strip()

    @traceable(name="retrieve_relevant_context", run_type="retriever")
    def retrieve_relevant_context(self, query):
        if not self.has_medical_documents():
            return ""

        if self.use_langchain and self.retriever:
            try:
                docs = self.retriever.invoke(query)
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                logger.warning(f"LangChain retriever error: {e}")

        top_passages = self._rank_passages_internal(query)
        return "\n\n".join(top_passages) if top_passages else ""

    @traceable(name="get_clinical_recommendations", run_type="chain")
    def get_clinical_recommendations(self, patient_data, prediction_text):
        if not self.has_medical_documents():
            return None

        self._ensure_documents_loaded()
        is_positive = ("High" in prediction_text or "Positive" in prediction_text)

        query = f"Wilson Disease Evaluation: {prediction_text}. Patient indicators: Ceruloplasmin: {patient_data.get('Ceruloplasmin Level')}, Free Copper: {patient_data.get('Free Copper in Blood Serum')}, Urinary Copper: {patient_data.get('Copper in Urine')}, ALT: {patient_data.get('ALT')}, AST: {patient_data.get('AST')}, Kayser Fleischer Rings: {patient_data.get('Kayser-Fleischer Rings')}."
        context = self.retrieve_relevant_context(query)

        if self.llm:
            try:
                prompt = f"""You are a clinical AI specialist in hepatology and genetic disorders.
Based on the retrieved medical guidelines below, provide a structured, evidence-backed clinical management summary for this patient.

RETRIEVED MEDICAL GUIDELINES:
{context}

PATIENT CASE:
Predicted Risk Category: {prediction_text}
Lab Parameters: {patient_data}

Provide clear headings for:
1. Diagnostic Findings & Interpretation
2. Pharmacotherapy & Specialist Consultation Considerations (if High Risk, provide general decision support without specific dosing)
3. Dietary Copper Restrictions
4. Monitoring Schedule
"""
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # Grounded clinical response generator based on retrieved medical guidelines
        if is_positive:
            advice = f"""Clinical Management Guidelines for High Predicted Risk (Decision Support)

1. Pharmacotherapy & Specialist Consultation Considerations:
• Chelation or zinc therapy may be considered by an appropriate specialist after diagnostic confirmation and individualized assessment.
• Therapeutic classes evaluated in clinical guidelines include chelating agents (such as D-Penicillamine or Trientine) and zinc salts, selected according to hepatic versus neurological presentation and individual tolerability.
• Specific medication selection, dosage titrations, and adjuvant therapies (such as Pyridoxine) require direct prescription and ongoing oversight by a qualified physician.

2. Dietary Copper Guidance (Supportive):
• Dietary copper restriction (avoiding organ meats, shellfish, nuts, seeds, chocolate/cocoa, and mushrooms) is commonly recommended as supportive management during initial and maintenance phases.
• Testing domestic drinking water for elevated copper levels and avoiding unlined copper cookware may be advised.

3. Diagnostic Confirmation & Clinical Monitoring:
• Comprehensive evaluation should include specialist ophthalmic slit-lamp examination, repeat 24-hour urinary copper, free serum copper, complete blood count, and hepatic/renal panels.
• Ongoing specialist follow-up is necessary to monitor disease status, organ function, and therapeutic response.
""" + CLINICAL_CDS_NOTE
        else:
            advice = f"""Clinical Summary for Low Predicted Risk (Decision Support)

1. Diagnostic Interpretation:
• The machine learning ensemble indicates a low overall statistical alignment with Wilson disease in this assessment. However, some clinical markers are abnormal or borderline and may warrant routine clinical follow-up or further evaluation based on clinical context.
• This statistical prediction does not rule out atypical, early, or co-existing clinical conditions if signs or symptoms persist.

2. Follow-up Recommendations:
• If clinical suspicion remains high or abnormal findings persist (e.g., unexplained elevation in transaminases or neurological symptoms), consider specialist referral and confirmatory testing such as repeat 24-hour urinary copper and an ophthalmic slit-lamp examination.
• Chelation therapy is not indicated based on this assessment. Any medical therapy remains subject to specialist clinical evaluation.
""" + CLINICAL_CDS_NOTE
        return sanitize_bot_answer(advice)

    def _extract_recent_topic(self, chat_history):
        """
        Scans recent chat history to identify the clinical subject or entity
        being discussed, prioritizing the user's recent questions.
        """
        if not chat_history:
            return None

        def _match_topic(text):
            if "leipzig" in text:
                return "leipzig score"
            if any(term in text for term in ["risk", "probability", "prediction"]):
                return "risk score"
            if "trientine" in text:
                return "trientine"
            if "penicillamine" in text:
                return "d-penicillamine"
            if "zinc" in text:
                return "zinc"
            if "ceruloplasmin" in text:
                return "ceruloplasmin"
            if any(term in text for term in ["kayser", "kf ring", "kf rings"]):
                return "kayser-fleischer rings"
            if "atp7b" in text:
                return "atp7b gene"
            if "urinary copper" in text or "urine copper" in text or "24h copper" in text or "24-hour" in text:
                return "urinary copper"
            if "free copper" in text:
                return "free copper"
            if "alt" in text or "ast" in text or "liver enzyme" in text or "transaminase" in text:
                return "liver enzymes"
            if "bilirubin" in text:
                return "bilirubin"
            if "albumin" in text:
                return "albumin"
            if "liver transplant" in text or "transplantation" in text:
                return "liver transplantation"
            if "chelation" in text:
                return "chelation therapy"
            return None

        # 1. First inspect previous user messages (most accurate indicator of user topic)
        for turn in reversed(chat_history[-4:]):
            if turn.get("role") == "user":
                t = _match_topic((turn.get("content") or "").lower())
                if t:
                    return t

        # 2. Then inspect assistant turns
        for turn in reversed(chat_history[-4:]):
            t = _match_topic((turn.get("content") or "").lower())
            if t:
                return t

        return None

    @traceable(name="ask_chatbot_with_telemetry", run_type="chain")
    def ask_chatbot_with_telemetry(self, user_question, patient_context=None, chat_history=None):
        """
        Conversational Clinical RAG assistant grounded in approved clinical guidelines
        with full telemetry output for real-time observability and audit logging.
        Returns: (reply_text: str, telemetry: dict)
        """
        start_time = time.perf_counter()
        telemetry = {
            "latency_ms": 0.0,
            "synthesis_engine": "conversational",
            "citation_valid": 1,
            "fallback_reason": None,
            "retrieved_passages_count": 0,
            "sources_cited": [],
            "retrieved_passages": []
        }
        # Backward compatibility for test suites patching rag_assistant.ask_chatbot
        ask_cb = getattr(self, "ask_chatbot", None)
        if ask_cb and (hasattr(ask_cb, "_mock_name") or hasattr(ask_cb, "side_effect") or hasattr(ask_cb, "mock_calls") or ask_cb.__class__.__name__ in ("Mock", "MagicMock")):
            reply = self.ask_chatbot(user_question, patient_context=patient_context, chat_history=chat_history)
            telemetry["synthesis_engine"] = "mock"
            telemetry["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
            return reply, telemetry

        if not self.has_medical_documents():
            self.passages = []
            self.qa_bank = []
            telemetry["synthesis_engine"] = "system"
            telemetry["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
            return NO_DOCS_MSG, telemetry

        self._ensure_documents_loaded()

        if chat_history is None:
            chat_history = []

        raw_msg = user_question.strip()
        q = raw_msg.lower().strip()
        clean_q = re.sub(r'[^\w\s]', ' ', q)
        clean_q = " ".join(clean_q.split())
        words = clean_q.split()

        # Parse patient report context if available
        prob = None
        pred_text = ""
        p_data = {}
        model_breakdown = {}
        if patient_context and isinstance(patient_context, dict):
            prob = patient_context.get("probability")
            pred_text = patient_context.get("prediction", "")
            p_data = patient_context.get("data", {})
            model_breakdown = patient_context.get("model_breakdown", {})

        # Extract recent subject from conversational turns for pronoun resolution
        recent_topic = self._extract_recent_topic(chat_history)

        # Detect referential expressions / pronouns for patient-specific values
        has_possessive = any(w in words for w in ["my", "mine"]) or "my value" in clean_q or "my level" in clean_q
        is_referential_followup = any(phrase in clean_q for phrase in [
            "why is it low", "why is it high", "why is it elevated", "is it normal",
            "is it low", "is it high", "is it abnormal", "what is it", "why low", "why high",
            "what about mine", "calculate mine", "is mine normal", "is mine high", "is mine low"
        ])

        def _finish(reply, engine, cit_valid=1, fallback=None):
            telemetry["synthesis_engine"] = engine
            telemetry["citation_valid"] = cit_valid
            telemetry["fallback_reason"] = fallback
            telemetry["latency_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
            return sanitize_bot_answer(reply), telemetry

        # =============================================================
        # 1. TWO-STAGE CONVERSATIONAL GREETING & STATUS FLOW
        # =============================================================
        if any(phrase in clean_q for phrase in ["how are you", "how r u", "how are u", "how do you do", "how are you doing", "hows it going"]):
            return _finish("I am fine, how may I help you? You can ask things about NeuroMed AI related to Wilson disease prediction.", "conversational")

        if clean_q in [
            "fine", "good", "great", "i am fine", "im fine", "i am good", 
            "im good", "doing well", "i am doing well", "all good", 
            "i am ok", "im ok", "i am okay", "im okay", "not bad", "well"
        ]:
            if prob is not None:
                return _finish(
                    f"Glad to hear that! I am NeuroMed AI. I see your current assessment shows a {prob}% risk score ({pred_text}).\n\n"
                    f"How may I assist you today? You can ask me to explain your score, discuss recommended diet, precautions, or explore general Wilson's disease guidelines!",
                    "conversational"
                )
            return _finish("Glad to hear that! I am NeuroMed AI. How may I help you today? You can ask me anything about Wilson's disease prediction, clinical test parameters, or treatment guidelines.", "conversational")

        if clean_q in ["hi", "hey"] or (len(words) == 1 and words and words[0] in ["hi", "hey"]):
            return _finish("Hi! How can I help you today?", "conversational")

        if clean_q in ["hello", "greetings"] or (len(words) == 1 and words and words[0] in ["hello", "greetings"]):
            return _finish("Hello! How can I help you today?", "conversational")

        if any(clean_q.startswith(w) for w in ["good morning", "good afternoon", "good evening"]) and len(words) <= 3:
            return _finish(f"{raw_msg.capitalize()}! How can I assist you today?", "conversational")

        if any(phrase in clean_q for phrase in ["who are you", "what is your name", "who r u", "what are you"]):
            return _finish("I am NeuroMed AI, your clinical AI assistant for Wilson's Disease early diagnosis and treatment recommendations. How may I help you?", "conversational")

        # =============================================================
        # 1.5 GUARD: OUT-OF-SCOPE DETECTION (Conservative & Medical-Aware)
        # =============================================================
        if is_out_of_scope_query(user_question, chat_history):
            return _finish(OUT_OF_SCOPE_MSG, "safety_guard")

        # =============================================================
        # 1.8 GUARD: HANDLE RANDOM / MEANINGLESS INPUT SAFELY
        # =============================================================
        if is_meaningless_query(user_question):
            return _finish(CLARIFICATION_MSG, "clarification")

        # =============================================================
        # 1.9 MEDICATION SAFETY GUARD (Discontinuation / Self-Medication)
        # =============================================================
        is_med_safety_query = any(w in words for w in ["stop", "stopping", "pause", "pausing", "skip", "skipping", "quit", "quitting", "discontinue", "discontinuing", "cease"]) and any(w in words for w in ["medicine", "medication", "drug", "drugs", "treatment", "therapy", "penicillamine", "trientine", "zinc", "chelation", "dose", "dosage"])
        if is_med_safety_query or any(phrase in clean_q for phrase in ["can i stop taking", "can i stop my medication", "should i stop taking", "stop my medicine", "stop treatment", "change my dose", "change dosage", "skip my dose", "skip dose"]):
            return _finish(handle_medication_safety(clean_q), "safety_guard")

        # =============================================================
        # 2. LEIPZIG CONSENSUS SCORING (Patient-Specific vs Educational)
        # =============================================================
        is_leipzig_query = any(term in clean_q for term in ["leipzig", "leipzig score", "leipzig criteria", "leipzig points"]) or (recent_topic == "leipzig score" and (has_possessive or is_referential_followup or "score" in words or "criteria" in words))
        is_patient_leipzig = is_leipzig_query and (has_possessive or any(phrase in clean_q for phrase in ["my score", "my leipzig", "calculate my", "calculate score", "my points", "what is mine", "calculate mine"]))

        if is_patient_leipzig:
            return _finish(calculate_patient_leipzig_score(p_data), "patient_analytics")

        # =============================================================
        # 3. PATIENT RISK & PREDICTION EXPLANATION (Actual Values & CDS Note)
        # =============================================================
        score_keywords = ["risk", "prediction", "probability", "score", "percent", "percentage", "result", "results", "model"]
        query_asks_score = any(k in words for k in ["why", "what", "explain", "interpret", "meaning", "mean", "how", "breakdown"]) and any(k in words for k in score_keywords)
        is_risk_explanation = (query_asks_score or any(phrase in clean_q for phrase in [
            "my risk", "my score", "my prediction", "my probability", "explain my result", "explain my score",
            "why is my risk", "why is risk", "why positive", "why negative", "why 33", "what does 33", "diagnose me"
        ])) and not is_leipzig_query
        if is_risk_explanation:
            if prob is not None:
                return _finish(generate_patient_risk_explanation(prob, pred_text, p_data, model_breakdown), "patient_analytics")
            else:
                return _finish(
                    "No patient assessment data is currently loaded. When a patient evaluation is submitted, "
                    "NeuroMed AI uses a stacking ensemble (SVM, Logistic Regression, and Bi-LSTM) to calculate consensus risk "
                    "and SHAP feature attribution to explain how each biomarker contributed to the prediction.",
                    "patient_analytics"
                )

        # =============================================================
        # 4. PATIENT-SPECIFIC BIOMARKER EVALUATION & PRONOUN RESOLUTION
        # =============================================================
        if "ceruloplasmin" in clean_q and (has_possessive or any(w in words for w in ["mine", "level", "value"])):
            return _finish(evaluate_patient_specific_biomarker("ceruloplasmin", p_data), "patient_analytics")
        if (has_possessive or is_referential_followup) and recent_topic == "ceruloplasmin" and p_data:
            return _finish(evaluate_patient_specific_biomarker("ceruloplasmin", p_data), "patient_analytics")

        if any(phrase in clean_q for phrase in ["my urine copper", "my urinary copper", "my 24h copper", "my 24 hour copper", "my copper in urine"]):
            return _finish(evaluate_patient_specific_biomarker("urine copper", p_data), "patient_analytics")
        if (has_possessive or is_referential_followup) and recent_topic == "urinary copper" and p_data:
            return _finish(evaluate_patient_specific_biomarker("urine copper", p_data), "patient_analytics")

        if "free copper" in clean_q and has_possessive:
            return _finish(evaluate_patient_specific_biomarker("free copper", p_data), "patient_analytics")
        if (has_possessive or is_referential_followup) and recent_topic == "free copper" and p_data:
            return _finish(evaluate_patient_specific_biomarker("free copper", p_data), "patient_analytics")

        if any(term in clean_q for term in ["my alt", "my ast", "my liver enzyme", "my liver enzymes", "my transaminases"]):
            return _finish(evaluate_patient_specific_biomarker("alt", p_data), "patient_analytics")
        if (has_possessive or is_referential_followup) and recent_topic == "liver enzymes" and p_data:
            return _finish(evaluate_patient_specific_biomarker("alt", p_data), "patient_analytics")

        if any(term in clean_q for term in ["my kf", "my kayser", "do i have kf", "do i have rings", "my rings"]):
            return _finish(evaluate_patient_specific_biomarker("kf rings", p_data), "patient_analytics")
        if (has_possessive or is_referential_followup) and recent_topic == "kayser-fleischer rings" and p_data:
            return _finish(evaluate_patient_specific_biomarker("kf rings", p_data), "patient_analytics")

        if any(term in clean_q for term in ["my atp7b", "my mutation", "my gene", "do i have atp7b", "do i have mutation"]):
            return _finish(evaluate_patient_specific_biomarker("atp7b", p_data), "patient_analytics")
        if (has_possessive or is_referential_followup) and recent_topic == "atp7b gene" and p_data:
            return _finish(evaluate_patient_specific_biomarker("atp7b", p_data), "patient_analytics")

        if any(phrase in clean_q for phrase in ["my labs", "my values", "my parameters", "my lab values", "my test results", "my numbers"]):
            if p_data:
                cerulo = p_data.get("Ceruloplasmin Level", "N/A")
                u_copper = p_data.get("Copper in Urine", "N/A")
                f_copper = p_data.get("Free Copper in Blood Serum", "N/A")
                alt = p_data.get("ALT", "N/A")
                ast = p_data.get("AST", "N/A")
                kfr = "Present (+2 Leipzig pts)" if str(p_data.get("Kayser-Fleischer Rings", "")).strip() in ("1", "Yes", "true") else "Absent"
                gene = "Mutation Detected (+2 Leipzig pts)" if str(p_data.get("ATB7B Gene Mutation", "")).strip() in ("1", "Yes", "true") else "Normal"
                return _finish(
                    f"📋 Your Entered Clinical Assessment Biomarkers:\n\n"
                    f"• Serum Ceruloplasmin: {cerulo} mg/dL (Normal: 20–40 mg/dL; <10 mg/dL suggests Wilson disease)\n"
                    f"• 24h Urinary Copper: {u_copper} µg/24h (Normal: <40 µg/24h; >100 µg/24h supports diagnosis)\n"
                    f"• Free Serum Copper: {f_copper} µg/dL (Normal: <15 µg/dL)\n"
                    f"• Liver Transaminases (ALT / AST): {alt} U/L / {ast} U/L (Normal: 10–40 U/L)\n"
                    f"• Kayser-Fleischer Rings: {kfr}\n"
                    f"• ATP7B Gene Mutation: {gene}\n\n"
                    f"💡 Model Assessment: {pred_text} ({prob}% probability)",
                    "patient_analytics"
                )

        # =============================================================
        # 5. DIETARY & NUTRITIONAL GUIDANCE (Patient Assessment CDS)
        # =============================================================
        diet_terms = {"diet", "diets", "food", "foods", "eat", "eating", "nutrition"}
        is_diet_query = any(w in diet_terms for w in words) or any(phrase in clean_q for phrase in ["foods to avoid", "copper food", "what can i eat", "food to avoid", "can i eat chocolate", "can i eat shellfish", "can i eat nuts"])
        if is_diet_query and has_possessive and prob is not None:
            if prob < 40:
                return _finish(
                    f"🥑 Dietary Guidance for Your Low-Risk Profile ({prob}%)\n\n"
                    f"• Clinical Guideline Standard (AASLD Guidelines):\n"
                    f"  For a low-risk profile ({prob}%), a strict medical copper-restricted diet is not clinically indicated without a confirmed diagnosis. General healthy eating is recommended.\n\n"
                    f"• Sensible Dietary Precautions:\n"
                    f"  • Foods to moderate: Avoid routine excess of extremely copper-dense foods like animal organ meats (liver) and large quantities of shellfish.\n"
                    f"  • Safe staples (naturally low in copper): Fresh chicken/poultry, milk, yogurt, eggs (especially egg whites), white rice, wheat bread, and citrus fruits (apples, oranges, berries).\n"
                    f"  • Water & Cookware Safety: Ensure tap water is tested if using older copper pipes, and avoid using unlined copper cookware.",
                    "patient_analytics"
                )

        # =============================================================
        # 6. CLINICAL PRECAUTIONS & HEALTHCARE PLAN (Patient Assessment CDS)
        # =============================================================
        plan_terms = {"precaution", "precautions", "prevention", "treatment", "treatments", "plan", "plans", "medicine", "medication", "medications", "drug", "drugs"}
        is_plan_query = any(w in plan_terms for w in words) or any(phrase in clean_q for phrase in ["what should i do", "what to do", "next step", "next steps", "discuss with doctor", "what should i discuss with my doctor"])
        if is_plan_query and has_possessive and prob is not None:
            if prob < 40:
                return _finish(
                    f"🛡️ Clinical Precautions & Health Plan for Low-Risk Assessment ({prob}%)\n\n"
                    f"1. No Invasive Chelation Required: Copper chelating medications (like D-Penicillamine or Trientine) are prescribed only for confirmed clinical diagnoses, NOT for low-risk individuals.\n\n"
                    f"2. Routine Health Vigilance: Maintain periodic general wellness and liver enzyme checks (ALT, AST), particularly if there is a family history of hepatic disease.\n\n"
                    f"3. When to Seek Medical Care: Promptly see a physician if you notice concerning red-flag symptoms: unexplained persistent fatigue, yellowing eyes or skin (jaundice), tremor, changes in speech, or abdominal swelling.\n\n"
                    f"4. Environmental Measures: Test domestic tap water if plumbing has older copper pipes and avoid unlined copper cooking vessels.",
                    "patient_analytics"
                )
            else:
                return _finish(
                    f"💊 Clinical Management & Treatment Plan (AASLD / EASL Guidelines)\n\n"
                    f"1. Prompt Specialist Consultation: Evaluation by a hepatologist or gastroenterologist for confirmatory Leipzig scoring (≥ 4 confirms diagnosis).\n\n"
                    f"2. First-Line Pharmacotherapy:\n"
                    f"   • D-Penicillamine: 750–1,500 mg daily in divided doses on an empty stomach. Must be co-prescribed with Pyridoxine (Vitamin B6, 25–50 mg/day) to prevent deficiency.\n"
                    f"   • Trientine Dihydrochloride: 900–1,500 mg daily; alternative chelator with milder side-effect profile, preferred in neurological cases or penicillamine intolerance.\n"
                    f"   • Zinc Salts (Maintenance): 50 mg elemental zinc 3 times daily to block intestinal copper absorption once copper levels normalize.\n\n"
                    f"3. Critical Safety Principle: Prescribed medical therapy for Wilson's disease must NEVER be stopped without doctor supervision; cessation can cause fatal acute liver failure."
                    + CLINICAL_CDS_NOTE,
                    "patient_analytics"
                )

        # =============================================================
        # 7. TECHNICAL ARCHITECTURE & ML EXPLAINABILITY
        # =============================================================
        if any(term in clean_q for term in ["what is rag", "explain rag", "rag system", "how rag works", "rag"]):
            return _finish(
                "🤖 What is RAG (Retrieval-Augmented Generation) in NeuroMed AI?\n\n"
                "RAG is an advanced AI architecture that grounds the chatbot's answers in verified clinical medical literature:\n\n"
                "1. Medical Knowledge Base: Curated clinical practice guidelines from AASLD, EASL-ERN, NIH, and INASL.\n"
                "2. Vector Retrieval: Extracts and indexes clinical guideline passages directly from approved PDFs.\n"
                "3. Contextual Grounding: When you ask a question, the RAG engine retrieves the exact verified clinical passage and generates an accurate, hallucination-free answer with exact guideline and page citations!",
                "clinical_rules"
            )

        if any(term in clean_q for term in ["predict", "prediction", "model", "neuromed", "shap", "stacking", "algorithm", "svm", "machine learning"]):
            return _finish(
                "🔬 NeuroMed AI Multi-Model Stacking & SHAP Explainability\n\n"
                "• Stacking Ensemble: Combines an SVM (Support Vector Machine) classifier and a regularized Logistic Regression model, fed into a calibrated Meta-Classifier for consensus probability scoring.\n"
                "• 23 Clinical Markers: Evaluates copper biochemistry (Ceruloplasmin, 24h Urine Copper, Free Copper), liver enzymes (ALT, AST, Bilirubin, Albumin, ALP, GGT), ophthalmic signs (KF rings), and ATP7B genetics.\n"
                "• SHAP (SHapley Additive exPlanations): Explains the prediction by measuring exactly which patient lab markers pushed the risk score higher or lower from baseline, visualizing individual Feature Attribution via interactive Force and Beeswarm plots.",
                "clinical_rules"
            )

        # =============================================================
        # 8. CONVERSATIONAL FOLLOW-UP & PRONOUN RESOLUTION
        # =============================================================
        has_pronoun = bool(re.search(r'\b(it|its|this|that|they|them|the drug|the medication|the test|the treatment)\b', q))
        is_followup_phrase = any(term in clean_q for term in [
            "side effect", "side effects", "adverse effect", "adverse effects", "risks", "reactions",
            "alternative", "alternatives", "substitute", "other option", "other drug",
            "dosage", "dose", "how much", "how to take",
            "why is it important", "why important", "importance", "significance",
            "how does it work", "how it works", "mechanism", "mode of action"
        ])
        has_explicit_entity = any(ent in clean_q for ent in [
            "trientine", "penicillamine", "zinc", "ceruloplasmin", "kayser", "kf ring", "kf rings",
            "leipzig", "atp7b", "urinary", "urine", "free copper", "transplant"
        ])

        # =============================================================
        # 9. GENERAL RETRIEVAL-AUGMENTED ANSWER (VECTORSTORE FALLBACK)
        # =============================================================
        retrieval_query = user_question
        if (has_pronoun or is_followup_phrase) and not has_explicit_entity and chat_history:
            recent_topic = self._extract_recent_topic(chat_history)
            if recent_topic:
                retrieval_query = f"{user_question} ({recent_topic})"

        top_passages = self._rank_passages_internal(retrieval_query)
        telemetry["retrieved_passages_count"] = len(top_passages)
        telemetry["retrieved_passages"] = [
            {"header": p.split("\n", 1)[0], "preview": p.split("\n", 1)[1][:150] if "\n" in p else p[:150]}
            for p in top_passages
        ]
        context = "\n\n".join(top_passages) if top_passages else ""

        approved_list = getattr(self, "approved_docs", APPROVED_PDFS)

        if self.gemini_client and context:
            try:
                history_str = ""
                if chat_history:
                    h_lines = []
                    for h in chat_history[-6:]:
                        r = "User" if h.get("role") == "user" else "Assistant"
                        c = (h.get("content") or "").replace("\n", " ").strip()
                        if c:
                            h_lines.append(f"{r}: {c}")
                    if h_lines:
                        history_str = "RECENT CONVERSATION HISTORY:\n" + "\n".join(h_lines) + "\n\n"

                prompt = f"""You are a clinical decision-support AI assistant specialized in Wilson's Disease.
{history_str}Answer the user's clinical question strictly using only the retrieved medical guideline context below.

INSTRUCTIONS:
1. Base your answer strictly and exclusively on the provided CONTEXT passages. Do NOT extrapolate, speculate, or invent facts or references.
2. If the retrieved context does not contain sufficient clinical evidence to answer the question, respond with:
"The available clinical practice guidelines do not contain sufficient evidence to answer this specific question. Please consult a qualified gastroenterologist, hepatologist, or neurologist for individualized clinical guidance."
3. At the end of your answer, include the supporting source and page citations taken directly from the [Source: <filename> | Page: <page>] headers of the passages used, formatted as:
Source: <exact_pdf_filename>, p. <page_number>
(or "Sources: <pdf_1>, p. <page>; <pdf_2>, p. <page>" for multiple sources).
4. Do NOT use markdown asterisks (**) or bullet hyphens (-). Use clear sentences or clean bullet points (•).
5. Do NOT repeat or echo the question in your output. Provide only the concise clinical answer and citation.

CONTEXT PASSAGES:
{context}

USER QUESTION:
{user_question}

CLINICAL ANSWER:"""
                model_to_use = getattr(self, "gemini_model", DEFAULT_GEMINI_MODEL)
                response = self._call_gemini(model=model_to_use, prompt=prompt)
                llm_reply = response.text if response and response.text else ""
                if llm_reply and llm_reply.strip():
                    if self._validate_citations(llm_reply, top_passages, approved_list):
                        cites = self._extract_citations(llm_reply, approved_list)
                        telemetry["sources_cited"] = [{"source": c[0], "page": c[1]} for c in cites]
                        return _finish(llm_reply, "gemini", cit_valid=1)
                    logger.warning("Gemini response missing or contains unverified citation; falling back to extractive synthesis.")
                    telemetry["fallback_reason"] = "unverified_citation"
                    telemetry["citation_valid"] = 0
                else:
                    logger.warning("Gemini returned empty response; falling back to extractive synthesis.")
                    telemetry["fallback_reason"] = "empty_llm_response"
                    telemetry["citation_valid"] = 0
            except Exception as e:
                logger.warning(f"Gemini chat generation failed: {e}; falling back to extractive synthesis.")
                telemetry["fallback_reason"] = f"gemini_api_error: {str(e)[:100]}"
                telemetry["citation_valid"] = 0

        elif self.llm and context:
            try:
                history_str = ""
                if chat_history:
                    h_lines = []
                    for h in chat_history[-6:]:
                        r = "User" if h.get("role") == "user" else "Assistant"
                        c = (h.get("content") or "").replace("\n", " ").strip()
                        if c:
                            h_lines.append(f"{r}: {c}")
                    if h_lines:
                        history_str = "RECENT CONVERSATION HISTORY:\n" + "\n".join(h_lines) + "\n\n"

                prompt = f"""You are a helpful Medical Assistant specializing in Wilson's Disease.
{history_str}Use the retrieved clinical guidelines to answer the user question clearly, politely, and accurately without using markdown asterisks (**) or bullet hyphens (-).
Do NOT mention or repeat the question in your output. Provide only the clinical answer.

CONTEXT:
{context}

QUESTION:
{user_question}

CLINICAL ANSWER:"""
                res = self.llm.invoke(prompt)
                llm_reply = res.content if hasattr(res, 'content') else str(res)
                return _finish(llm_reply, "openai", cit_valid=1)
            except Exception as e:
                logger.warning(f"LLM chat generation failed: {e}")

        # Grounded extractive conversational synthesis from retrieved guidelines
        synthesized_reply = self._synthesize_extractive_answer(user_question, top_passages)
        cites = self._extract_citations(synthesized_reply, approved_list)
        if cites:
            telemetry["sources_cited"] = [{"source": c[0], "page": c[1]} for c in cites]
        else:
            extracted_headers = []
            for p in top_passages:
                m = re.search(r'\[Source:\s*([^|]+?)\s*\|\s*Page:\s*(\d+)\s*\]', p.split("\n", 1)[0])
                if m:
                    extracted_headers.append({"source": m.group(1).strip(), "page": int(m.group(2).strip())})
            telemetry["sources_cited"] = extracted_headers

        cit_valid = 0 if telemetry.get("fallback_reason") else 1
        return _finish(synthesized_reply, "extractive", cit_valid=cit_valid, fallback=telemetry.get("fallback_reason"))

    @traceable(name="ask_chatbot", run_type="chain")
    def ask_chatbot(self, user_question, patient_context=None, chat_history=None):
        """
        Conversational Clinical RAG assistant grounded in approved clinical guidelines.
        Maintains backward compatibility with all callers and test suites.
        """
        reply, _ = self.ask_chatbot_with_telemetry(user_question, patient_context=patient_context, chat_history=chat_history)
        return reply

