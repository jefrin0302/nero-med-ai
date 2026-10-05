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

logger = logging.getLogger("wilson_rag")
logging.basicConfig(level=logging.INFO)

PERSIST_DIR = "chroma_db"
DOCS_DIR = "medical_docs"

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

    # 3. Conversational follow-ups (e.g. "what are its side effects", "why is it important")
    conversational_pronouns = {"it", "its", "they", "them", "this", "that"}
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

class MedicalRAGSystem:
    def __init__(self):
        self.use_langchain = False
        self.retriever = None
        self.llm = None
        self.passages = []
        self.qa_bank = []

        self._init_knowledge_base()

    def _init_knowledge_base(self):
        # 1. Load structured Q&A pairs from Question Bank for instant answer lookup
        self._load_qa_bank()

        # 2. Load clinical passages directly from guidelines
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
        # 1. Load Question Bank
        qb_path = os.path.join(DOCS_DIR, "wilson_disease_question_bank.txt")
        if os.path.exists(qb_path):
            try:
                with open(qb_path, "r", encoding="utf-8") as f:
                    text = f.read()
                sections = re.split(r'\n(?=[0-9]+\.\s+)', text)
                clean_sections = []
                for s in sections:
                    s_str = s.strip()
                    if len(s_str) < 30:
                        continue
                    # Skip document title/header sections
                    if any(h in s_str for h in ["QUESTION BANK WITH ANSWERS", "Evidence-Grounded Reference", "Guideline basis:"]):
                        continue
                    clean_sections.append(s_str)
                self.passages.extend(clean_sections)
            except Exception as e:
                logger.warning(f"Error loading question bank text: {e}")

        # 2. Load Guidelines
        guidelines_path = os.path.join(DOCS_DIR, "wilson_disease_guidelines.txt")
        if os.path.exists(guidelines_path):
            try:
                with open(guidelines_path, "r", encoding="utf-8") as f:
                    text = f.read()
                sections = re.split(r'\n(?=[0-9]+\.\s+[A-Z\s&]+)', text)
                clean_gl = []
                for s in sections:
                    s_str = s.strip()
                    if len(s_str) < 40:
                        continue
                    if any(h in s_str for h in ["QUESTION BANK WITH ANSWERS", "Evidence-Grounded Reference", "Guideline basis:"]):
                        continue
                    clean_gl.append(s_str)
                self.passages.extend(clean_gl)
            except Exception as e:
                logger.warning(f"Error loading guidelines text: {e}")

        # Fallback to json if needed
        if not self.passages:
            json_path = os.path.join(PERSIST_DIR, "passages.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        self.passages = json.load(f)
                except Exception:
                    pass

    def _load_qa_bank(self):
        """
        Parses the official question bank into structured (question, answer) pairs.
        Allows exact and semantic lookup so the chatbot returns pure answers
        without repeating the question.
        """
        self.qa_bank = []
        qb_path = os.path.join(DOCS_DIR, "wilson_disease_question_bank.txt")
        if not os.path.exists(qb_path):
            return

        try:
            with open(qb_path, "r", encoding="utf-8") as f:
                content = f.read()

            blocks = re.findall(
                r'(\d+)\.\s+([^\n]+(?:\?|\.))\n([^\n]+(?:\n(?!\d+\.|\d+\s+[A-Z=])[^\n]+)*)',
                content
            )
            for num, q_text, ans_text in blocks:
                q_clean = q_text.strip()
                q_norm = re.sub(r'[^\w\s]', ' ', q_clean.lower())
                q_words = set(q_norm.split()) - {"what", "is", "are", "the", "a", "an", "in", "of", "for", "to", "can", "how", "does", "do", "it"}
                self.qa_bank.append({
                    "num": int(num) if num.isdigit() else 0,
                    "question": q_clean,
                    "q_norm": " ".join(q_norm.split()),
                    "words": q_words,
                    "answer": ans_text.strip()
                })
            logger.info(f"Loaded {len(self.qa_bank)} structured clinical Q&A items from Question Bank.")
        except Exception as e:
            logger.warning(f"Error loading structured QA bank: {e}")

    def search_qa_bank(self, user_question: str):
        """
        Finds the best matching clinical answer in the question bank.
        Returns ONLY the authoritative answer(s), never echoing the question.
        Enforces confidence thresholds and intent checks to prevent poor or spurious matches.
        """
        if not getattr(self, "qa_bank", None):
            return None

        clean_user_q = re.sub(r'^\d+[\.\)]?\s*', '', user_question).strip()
        norm_user = " ".join(re.sub(r'[^\w\s]', ' ', clean_user_q.lower()).split())
        if not norm_user or len(norm_user) < 3:
            return None

        stop_words = {
            "what", "is", "are", "the", "a", "an", "in", "of", "for", "to", "can",
            "how", "does", "do", "it", "based", "taken", "take", "done", "used",
            "many", "much", "which", "get", "tell", "me", "about"
        }
        user_words = set(norm_user.split()) - stop_words
        if not user_words:
            return None

        # Special handler for "what is wilson disease" / "explain wilson disease":
        # Returns the 2-line definition, plus cause, plus treatment outlook
        is_what_is_wd = (
            norm_user in ["what is wilson disease", "what is wilsons disease", "explain wilson disease", "what is wilson s disease", "tell me about wilson disease", "wilson disease", "wilsons disease"]
            or (user_words == {"wilson", "disease"})
            or (clean_user_q.lower() in ["what is wilson disease", "what is wilson disease?"])
        )
        if is_what_is_wd:
            ans1 = self.qa_bank[0]["answer"] if len(self.qa_bank) > 0 else ""
            ans2 = self.qa_bank[1]["answer"] if len(self.qa_bank) > 1 else ""
            ans_treat = ""
            for item in self.qa_bank:
                if "treated" in item["q_norm"]:
                    ans_treat = item["answer"]
                    break
            parts = [ans1]
            if ans2:
                parts.append(ans2)
            if ans_treat:
                parts.append(ans_treat)
            return "\n\n".join(parts)

        # Canonicalize user stems
        user_stems = {_canonical_token(w) for w in user_words}
        is_user_test = any(w in user_stems for w in ["test", "lab", "measure", "biopsy", "screen", "diagnosis"])
        is_user_diet = any(w in user_stems for w in ["food", "eat", "diet", "nutrition", "soy", "wheat", "meat", "shellfish", "nut", "chocolate", "mushroom"])

        best_item = None
        best_score = 0.0

        for item in self.qa_bank:
            item_q = item["q_norm"]
            # 1. Exact normalized match
            if norm_user == item_q:
                best_item = item
                best_score = 100.0
                break

            # 2. Whole-word substring match (require min 6 chars to avoid short noise)
            if len(norm_user) >= 6 and (f" {norm_user} " in f" {item_q} " or f" {item_q} " in f" {norm_user} "):
                score = 60.0 + len(user_words & item["words"]) * 5.0
                if score > best_score:
                    best_score = score
                    best_item = item

            # 3. Canonical stem overlap
            item_stems = {_canonical_token(w) for w in item["words"]}
            overlap = len(user_stems & item_stems)
            if overlap > 0:
                # Intent compatibility check: avoid returning diet info for testing queries and vice versa
                is_item_diet = any(w in item_stems for w in ["food", "eat", "diet", "nutrition", "soy", "wheat", "meat", "shellfish", "nut", "chocolate", "mushroom"])
                is_item_test = any(w in item_stems for w in ["test", "lab", "measure", "biopsy", "screen", "diagnosis"])

                if is_user_test and not is_user_diet and is_item_diet:
                    continue
                if is_user_diet and not is_user_test and is_item_test:
                    continue

                user_cov = overlap / len(user_stems)
                item_cov = overlap / len(item_stems)
                f1 = 2 * (user_cov * item_cov) / (user_cov + item_cov + 1e-6)

                # Confidence scoring formula
                score = (50.0 * f1) + (30.0 * user_cov) + (overlap * 4.0)

                # Require high coverage or multiple matching key terms
                if (user_cov >= 0.6 or overlap >= 2) and score > best_score:
                    best_score = score
                    best_item = item

        # Acceptance threshold: match must have high confidence (>= 35.0)
        if best_item and best_score >= 35.0:
            primary_ans = best_item["answer"]
            if "cause" in norm_user or "causes" in norm_user:
                extra_accum = ""
                for it in self.qa_bank:
                    if "copper accumulate" in it["q_norm"] or "why does copper accumulate" in it["q_norm"]:
                        extra_accum = it["answer"]
                        break
                if extra_accum and extra_accum != primary_ans:
                    return f"{primary_ans}\n\n{extra_accum}"
            return primary_ans

        return None

    def _init_llm(self):
        # Check Gemini
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_key, temperature=0.2)
                logger.info("RAG LLM initialized with Google Gemini 1.5 Flash.")
                return
            except Exception as e:
                logger.warning(f"Gemini LLM init error: {e}")

        # Check OpenAI
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.2)
                logger.info("RAG LLM initialized with OpenAI GPT-4o-mini.")
                return
            except Exception as e:
                logger.warning(f"OpenAI LLM init error: {e}")

        logger.info("No external LLM API key detected. Running grounded clinical RAG engine.")

    def retrieve_relevant_context(self, query):
        if self.use_langchain and self.retriever:
            try:
                docs = self.retriever.invoke(query)
                return "\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                logger.warning(f"LangChain retriever error: {e}")

        # Keyword and semantic scoring over clean guideline sections
        query_lower = query.lower()
        words = set(re.findall(r'\w+', query_lower))
        stop_words = {
            "the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "to", "what",
            "how", "tell", "me", "about", "are", "based", "taken", "take", "done", "used",
            "many", "much", "can", "does", "do", "it", "for", "of", "with"
        }
        meaningful_words = words - stop_words
        if not meaningful_words:
            return ""

        # Check intent to avoid cross-domain false matches (e.g. food passages for testing queries)
        is_test_query = any(w in meaningful_words for w in [
            "test", "tests", "testing", "tested", "lab", "labs", "biopsy", "measure",
            "measures", "measured", "screen", "screening", "diagnosis", "diagnostic"
        ])
        is_diet_query = any(w in meaningful_words for w in [
            "food", "foods", "eat", "eating", "diet", "diets", "dietary", "nutrition",
            "meat", "shellfish", "nut", "nuts", "chocolate", "mushroom", "mushrooms", "soy"
        ])

        scored = []
        for p in self.passages:
            p_lower = p.lower()
            p_words = set(re.findall(r'\w+', p_lower))
            # Match score with extra weight for exact multi-word substring match
            score = len(meaningful_words.intersection(p_words)) * 2
            if any(term in p_lower for term in meaningful_words if len(term) > 3):
                score += 3

            # Intent penalty: do not return food/diet info for a testing/diagnostic query
            if is_test_query and not is_diet_query:
                if any(w in p_lower for w in ["soy", "soybean", "tofu", "wheat", "shellfish", "oyster", "recipes", "dietary copper restriction"]):
                    score -= 15
            elif is_diet_query and not is_test_query:
                if any(w in p_lower for w in ["biopsy", "leipzig score", "slit-lamp", "penicillamine challenge"]):
                    score -= 10

            if score > 0:
                scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_passages = [p for s, p in scored[:2] if s >= 4]

        return "\n\n".join(top_passages)

    def get_clinical_recommendations(self, patient_data, prediction_text):
        is_positive = "Positive" in prediction_text

        query = f"Wilson Disease Diagnosis: {prediction_text}. Patient indicators: Ceruloplasmin: {patient_data.get('Ceruloplasmin Level')}, Free Copper: {patient_data.get('Free Copper in Blood Serum')}, Urinary Copper: {patient_data.get('Copper in Urine')}, ALT: {patient_data.get('ALT')}, AST: {patient_data.get('AST')}, Kayser Fleischer Rings: {patient_data.get('Kayser-Fleischer Rings')}."
        context = self.retrieve_relevant_context(query)

        if self.llm:
            try:
                prompt = f"""You are a clinical AI specialist in hepatology and genetic disorders.
Based on the retrieved medical guidelines below, provide a structured, evidence-backed clinical management summary for this patient.

RETRIEVED MEDICAL GUIDELINES:
{context}

PATIENT CASE:
Diagnosis: {prediction_text}
Lab Parameters: {patient_data}

Provide clear headings for:
1. Diagnostic Findings & Interpretation
2. Pharmacotherapy & Chelation Recommendations (if Positive)
3. Dietary Copper Restrictions
4. Monitoring Schedule
"""
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # Grounded clinical response generator based on retrieved medical guidelines
        if is_positive:
            advice = f"""Clinical Management Guidelines for Positive Diagnosis

1. Pharmacotherapy & De-coppering Options:
• First-line Chelation Therapy: D-Penicillamine (750 mg – 1500 mg/day in divided doses, taken on an empty stomach). Co-administer Pyridoxine (Vitamin B6, 25–50 mg/day) to prevent deficiency.
• Alternative Chelator: Trientine Dihydrochloride (900 mg – 1500 mg/day) if D-Penicillamine is not tolerated or in patients with neurological involvement.
• Maintenance / Zinc Therapy: Zinc Acetate (50 mg elemental zinc 3 times daily) to block intestinal copper absorption once copper levels normalize.

2. Strictly Avoid High-Copper Foods:
• Organ meats (liver, kidney), shellfish (oysters, crab), nuts (cashews, almonds), seeds, chocolate/cocoa, and mushrooms.
• Test drinking water for elevated copper levels and avoid unlined copper cookware.

3. Required Diagnostic Monitoring:
• Monitor 24-hour urinary copper (target: 200–500 µg/day on chelation), free serum copper (target: 5–15 µg/dL), CBC, LFTs, and renal function.
• Life-long medication adherence is essential to prevent acute fulminant hepatic failure.
"""
        else:
            advice = f"""Clinical Summary for Negative Diagnosis

1. Diagnostic Interpretation:
• The machine learning model indicates a low probability of Wilson's disease based on current biochemical markers (Ceruloplasmin: {patient_data.get('Ceruloplasmin Level', 'N/A')}, Free Copper: {patient_data.get('Free Copper in Blood Serum', 'N/A')}).

2. Follow-up Recommendations:
• If clinical suspicion remains high (e.g., unexplained persistent elevation in ALT/AST or family history of liver disease), consider repeat 24-hour urinary copper screening and ophthalmologic slit-lamp exam for Kayser-Fleischer rings.
• Continue routine hepatic and neurological evaluation as clinically indicated.
"""
        return sanitize_bot_answer(advice)

    def _extract_recent_topic(self, chat_history):
        """
        Scans recent chat history (from newest to oldest) to identify the clinical subject
        or entity being discussed, enabling pronoun and follow-up reference resolution.
        """
        if not chat_history:
            return None

        # Inspect up to 4 recent messages in reverse order
        for turn in reversed(chat_history[-4:]):
            text = (turn.get("content") or "").lower()
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
            if "leipzig" in text:
                return "leipzig score"
            if "atp7b" in text:
                return "atp7b gene"
            if "urinary copper" in text or "urine copper" in text:
                return "urinary copper"
            if "free copper" in text:
                return "free copper"
            if "liver transplant" in text or "transplantation" in text:
                return "liver transplantation"
            if "chelation" in text:
                return "chelation therapy"
        return None

    def ask_chatbot(self, user_question, patient_context=None, chat_history=None):
        """
        Conversational Clinical RAG assistant grounded in the AASLD/EASL/INASL Question Bank
        and aware of the active Patient Dashboard / Prediction Report.
        All responses are cleanly sanitized to remove raw asterisks (**) and bullet hyphens (-).
        """
        if chat_history is None:
            chat_history = []

        raw_msg = user_question.strip()
        q = raw_msg.lower().strip()
        clean_q = re.sub(r'[^\w\s]', '', q).strip()
        words = clean_q.split()

        # Parse patient report context if available
        prob = None
        pred_text = ""
        p_data = {}
        if patient_context and isinstance(patient_context, dict):
            prob = patient_context.get("probability")
            pred_text = patient_context.get("prediction", "")
            p_data = patient_context.get("data", {})

        # =============================================================
        # 1. TWO-STAGE CONVERSATIONAL GREETING & STATUS FLOW
        # =============================================================
        if any(phrase in clean_q for phrase in ["how are you", "how r u", "how are u", "how do you do", "how are you doing", "hows it going"]):
            return sanitize_bot_answer("I am fine, how may I help you? You can ask things about NeuroMed AI related to Wilson disease prediction.")

        if clean_q in [
            "fine", "good", "great", "i am fine", "im fine", "i am good", 
            "im good", "doing well", "i am doing well", "all good", 
            "i am ok", "im ok", "i am okay", "im okay", "not bad", "well"
        ]:
            if prob is not None:
                return sanitize_bot_answer(
                    f"Glad to hear that! I am NeuroMed AI. I see your current assessment shows a {prob}% risk score ({pred_text}).\n\n"
                    f"How may I assist you today? You can ask me to explain your score, discuss recommended diet, precautions, or explore general Wilson's disease guidelines!"
                )
            return sanitize_bot_answer("Glad to hear that! I am NeuroMed AI. How may I help you today? You can ask me anything about Wilson's disease prediction, clinical test parameters, or treatment guidelines.")

        if clean_q in ["hi", "hey"] or (len(words) == 1 and words and words[0] in ["hi", "hey"]):
            return sanitize_bot_answer("Hi! How can I help you today?")

        if clean_q in ["hello", "greetings"] or (len(words) == 1 and words and words[0] in ["hello", "greetings"]):
            return sanitize_bot_answer("Hello! How can I help you today?")

        if any(clean_q.startswith(w) for w in ["good morning", "good afternoon", "good evening"]) and len(words) <= 3:
            return sanitize_bot_answer(f"{raw_msg.capitalize()}! How can I assist you today?")

        if any(phrase in clean_q for phrase in ["who are you", "what is your name", "who r u", "what are you"]):
            return sanitize_bot_answer("I am NeuroMed AI, your clinical AI assistant for Wilson's Disease early diagnosis and treatment recommendations. How may I help you?")

        # =============================================================
        # 1.5 GUARD: HANDLE RANDOM / MEANINGLESS INPUT SAFELY
        # =============================================================
        if is_meaningless_query(user_question):
            return sanitize_bot_answer(CLARIFICATION_MSG)

        # =============================================================
        # 2. PATIENT-REPORT-AWARE HANDLERS (AASLD / EASL Question Bank Grounded)
        # =============================================================

        # --- A. RISK SCORE & PREDICTION INTERPRETATION ---
        score_keywords = ["risk", "prediction", "probability", "score", "percent", "percentage", "result"]
        query_asks_score = any(k in clean_q for k in ["why", "what", "explain", "interpret", "meaning", "mean", "how"]) and any(k in clean_q for k in score_keywords)
        if query_asks_score or any(term in clean_q for term in ["my risk", "my score", "my prediction", "why 33", "what does 33"]):
            if prob is not None:
                if prob < 40:
                    return sanitize_bot_answer(
                        f"📊 Personalized Risk Interpretation for Your Assessment ({prob}% — Low Statistical Risk)\n\n"
                        f"• What this means (AASLD 2023 / Question Bank Sec. 8):\n"
                        f"  Based on your entered clinical and laboratory parameters, the stacking ensemble model estimates a low probability ({prob}%) that this pattern is consistent with Wilson's disease. This is a statistical estimate, not a clinical diagnosis.\n\n"
                        f"• Why your risk is {prob}%:\n"
                        f"  Core diagnostic markers—such as non-elevated free copper, absence of Kayser-Fleischer rings, and preserved liver synthetic function—pulled the prediction into the lower risk category.\n\n"
                        f"• Important Clinical Note:\n"
                        f"  A low-risk score suggests low likelihood but does not completely rule out atypical presentations. As recommended by AASLD guidelines, clinical judgment by a physician always remains necessary if any liver or neurological symptoms persist."
                    )
                elif prob < 60:
                    return sanitize_bot_answer(
                        f"📊 Personalized Risk Interpretation for Your Assessment ({prob}% — Moderate / Intermediate Risk)\n\n"
                        f"• What this means (EASL-ERN / Question Bank Sec. 9):\n"
                        f"  The model finds a mixed or intermediate pattern of features—some parameters resemble Wilson's disease while others do not. The result is uncertain and warrants closer clinical assessment rather than a firm conclusion.\n\n"
                        f"• Recommended Confirmatory Actions:\n"
                        f"  Discuss this result with a physician to perform targeted diagnostic testing: serum ceruloplasmin, 24-hour urinary copper, and a slit-lamp eye examination for Kayser-Fleischer rings (AASLD / Leipzig score)."
                    )
                else:
                    return sanitize_bot_answer(
                        f"📊 Personalized Risk Interpretation for Your Assessment ({prob}% — High Risk Signal)\n\n"
                        f"• What this means (AASLD / EASL / Question Bank Sec. 10):\n"
                        f"  Your entered biochemical and clinical indicators strongly resemble patterns associated with Wilson's disease in trained clinical cohorts.\n\n"
                        f"• Recommended Urgent Next Steps:\n"
                        f"  A high model probability is a strong signal warranting prompt clinical evaluation by a gastroenterologist or hepatologist for confirmatory Leipzig scoring and discussion of first-line de-coppering chelation therapy."
                    )

        # --- B. CONTEXTUAL DIET & FOOD RESTRICTIONS ---
        diet_terms = {"diet", "diets", "food", "foods", "eat", "eating", "nutrition"}
        is_diet_query = any(w in diet_terms for w in words) or any(phrase in clean_q for phrase in ["foods to avoid", "copper food", "what can i eat", "food to avoid"])
        if is_diet_query:
            if prob is not None and prob < 40:
                return sanitize_bot_answer(
                    f"🥑 Dietary Guidance for Your Low-Risk Profile ({prob}%)\n\n"
                    f"• Clinical Guideline Standard (AASLD / Question Bank Sec. 8.5–8.6):\n"
                    f"  For a low-risk profile ({prob}%), a strict medical copper-restricted diet is not clinically indicated without a confirmed diagnosis. General healthy eating is recommended.\n\n"
                    f"• Sensible Dietary Precautions:\n"
                    f"  • Foods to moderate: Avoid routine excess of extremely copper-dense foods like animal organ meats (liver) and large quantities of shellfish.\n"
                    f"  • Safe staples (naturally low in copper): Fresh chicken/poultry, milk, yogurt, eggs (especially egg whites), white rice, wheat bread, and citrus fruits (apples, oranges, berries).\n"
                    f"  • Water & Cookware Safety: Ensure tap water is tested if using older copper pipes, and avoid using unlined copper cookware."
                )
            else:
                return sanitize_bot_answer(
                    f"🥑 Strict Dietary Copper Restriction Protocol (AASLD / Question Bank Sec. 6)\n\n"
                    f"• Daily Target: Limit dietary copper intake to < 0.9 mg/day, especially during the first year of de-coppering therapy.\n\n"
                    f"🚫 Foods to Strictly Avoid:\n"
                    f"• Organ Meats: Liver, kidney, heart, brain (highest dietary copper concentration)\n"
                    f"• Shellfish: Oysters, crab, lobster, clams, shrimp\n"
                    f"• Nuts & Seeds: Cashews, almonds, walnuts, sunflower seeds, sesame\n"
                    f"• Chocolate & Cocoa: Dark chocolate, cocoa powder, chocolate-flavored drinks\n"
                    f"• Mushrooms & Soy: All mushroom varieties, soybeans, tofu\n"
                    f"• Dried Fruits: Raisins, prunes, dates\n\n"
                    f"💧 Water Precaution: Test household drinking water; copper levels >0.1 ppm require certified filtration. Never use unlined copper pots or cookware."
                )

        # --- C. CONTEXTUAL PRECAUTIONS & TREATMENT PLAN ---
        plan_terms = {"precaution", "precautions", "prevention", "treatment", "treatments", "plan", "plans", "medicine", "medication", "medications", "drug", "drugs"}
        is_plan_query = any(w in plan_terms for w in words) or any(phrase in clean_q for phrase in ["what should i do", "what to do", "next step", "next steps"])
        if is_plan_query:
            if prob is not None and prob < 40:
                return sanitize_bot_answer(
                    f"🛡️ Clinical Precautions & Health Plan for Low-Risk Assessment ({prob}%)\n\n"
                    f"1. No Invasive Chelation Required: Copper chelating medications (like D-Penicillamine or Trientine) are prescribed only for confirmed clinical diagnoses, NOT for low-risk individuals (Question Bank Sec. 8.4).\n\n"
                    f"2. Routine Health Vigilance: Maintain periodic general wellness and liver enzyme checks (ALT, AST), particularly if there is a family history of hepatic disease.\n\n"
                    f"3. When to Seek Medical Care: Promptly see a physician if you notice concerning red-flag symptoms: unexplained persistent fatigue, yellowing eyes or skin (jaundice), tremor, changes in speech, or abdominal swelling.\n\n"
                    f"4. Environmental Measures: Test domestic tap water if plumbing has older copper pipes and avoid unlined copper cooking vessels."
                )
            else:
                return sanitize_bot_answer(
                    f"💊 Clinical Management & Treatment Plan (AASLD / EASL / WHO Guidelines)\n\n"
                    f"1. Prompt Specialist Consultation: Evaluation by a hepatologist or gastroenterologist for confirmatory Leipzig scoring (≥ 4 confirms diagnosis).\n\n"
                    f"2. First-Line Pharmacotherapy (WHO Essential Medicines / Question Bank Sec. 11):\n"
                    f"   • D-Penicillamine: 750–1,500 mg daily in divided doses on an empty stomach. Must be co-prescribed with Pyridoxine (Vitamin B6, 25–50 mg/day) to prevent deficiency.\n"
                    f"   • Trientine Dihydrochloride: 900–1,500 mg daily; alternative chelator with milder side-effect profile, preferred in neurological cases or penicillamine intolerance.\n"
                    f"   • Zinc Salts (Maintenance): 50 mg elemental zinc 3 times daily to block intestinal copper absorption once copper levels normalize.\n\n"
                    f"3. Critical Rule: Prescribed medical therapy for Wilson's disease must NEVER be stopped without doctor supervision; cessation can cause fatal acute liver failure."
                )

        # --- D. PATIENT-SPECIFIC LAB VALUES QUERY ---
        if any(term in clean_q for term in ["my ceruloplasmin", "my urine copper", "my alt", "my ast", "my kf", "my rings", "my copper", "my bilirubin", "my labs", "my values"]):
            if p_data:
                cerulo = p_data.get("Ceruloplasmin Level", "N/A")
                u_copper = p_data.get("Copper in Urine", "N/A")
                f_copper = p_data.get("Free Copper in Blood Serum", "N/A")
                alt = p_data.get("ALT", "N/A")
                ast = p_data.get("AST", "N/A")
                kfr = p_data.get("Kayser-Fleischer Rings", "N/A")
                gene = p_data.get("ATB7B Gene Mutation", "N/A")
                return sanitize_bot_answer(
                    f"📋 Your Recorded Clinical Lab Parameters:\n\n"
                    f"• Serum Ceruloplasmin: {cerulo} mg/dL (Normal: 20–40 mg/dL; <10 mg/dL suggests Wilson disease)\n"
                    f"• 24h Urinary Copper: {u_copper} µg/24h (Normal: <40 µg/24h; >100 µg/24h supports diagnosis)\n"
                    f"• Free Serum Copper: {f_copper} µg/dL (Normal: <15 µg/dL)\n"
                    f"• Liver Enzymes (ALT / AST): {alt} U/L / {ast} U/L (Normal: 10–40 U/L)\n"
                    f"• Kayser-Fleischer Rings: {kfr} (Ophthalmic copper corneal deposition)\n"
                    f"• ATP7B Gene Mutation: {gene}\n\n"
                    f"💡 Overall Ensemble Prediction: {pred_text} ({prob}% probability)"
                )

        # =============================================================
        # 3. TECHNICAL & MEDICAL DOMAIN QUESTIONS (Question Bank Grounded)
        # =============================================================

        # --- WHAT IS RAG? ---
        if any(term in clean_q for term in ["what is rag", "explain rag", "rag system", "how rag works", "rag"]):
            return sanitize_bot_answer(
                "🤖 What is RAG (Retrieval-Augmented Generation) in NeuroMed AI?\n\n"
                "RAG is an advanced AI architecture that grounds the chatbot's answers in verified clinical medical literature:\n\n"
                "1. Medical Knowledge Base: Curated guidelines from the AASLD 2022/2023, EASL-ERN 2024, INASL Indian Guidelines, and the WHO Essential Medicines List.\n"
                "2. Vector Retrieval (ChromaDB): Converts authoritative clinical question banks into dense semantic embeddings.\n"
                "3. Contextual Grounding: When you ask a question, the RAG engine retrieves the exact verified clinical passage and generates an accurate, hallucination-free answer tailored to the active patient's report!"
            )

        # --- WHAT IS CERULOPLASMIN? ---
        if any(term in clean_q for term in ["what is ceruloplasmin", "explain ceruloplasmin", "ceruloplasmin is what", "ceruloplasmin"]):
            return sanitize_bot_answer(
                "🧪 Ceruloplasmin Overview (Question Bank Sec. 4.3–4.6)\n\n"
                "• Definition: Ceruloplasmin is the primary copper-carrying glycoprotein synthesized by hepatocytes in the liver. Under normal physiology, over 90% of circulating blood copper is bound to it.\n\n"
                "• Normal Reference Range: 20 – 40 mg/dL\n\n"
                "• Role in Wilson's Disease:\n"
                "  Mutations in the ATP7B gene impair the incorporation of copper into apoceruloplasmin, causing unstable molecules that are rapidly degraded. Consequently, serum ceruloplasmin is typically < 10 mg/dL in Wilson disease (+2 Leipzig points).\n\n"
                "• Clinical Nuance: In ~5–15% of patients (especially during acute liver inflammation), ceruloplasmin can be within normal limits because it is an acute-phase reactant."
            )

        # --- WHAT IS LEIPZIG SCORE? ---
        if any(term in clean_q for term in ["leipzig", "score", "criteria", "points"]):
            return sanitize_bot_answer(
                "📋 Leipzig Consensus Scoring System for Wilson's Disease (Question Bank Sec. 4.15–4.16)\n\n"
                "• Kayser-Fleischer Rings: Present = +2 | Absent = 0\n"
                "• Neurological Symptoms / Brain MRI: Severe = +2 | Mild = +1 | None = 0\n"
                "• Serum Ceruloplasmin: < 10 mg/dL = +2 | 10–20 mg/dL = +1 | > 20 mg/dL = 0\n"
                "• 24-Hour Urinary Copper: > 2× ULN (>100 µg/d) = +2 | 1–2× ULN = +1 | Normal = 0\n"
                "• Hepatic Copper (Biopsy): > 250 µg/g dry weight = +2 | 50–250 µg/g = +1\n"
                "• ATP7B Gene Mutation: Both chromosomes = +4 | 1 chromosome = +1\n\n"
                "🎯 Diagnostic Interpretation:\n"
                "• ≥ 4 Points: Diagnosis Established (Definite Wilson's Disease)\n"
                "• 3 Points: Diagnosis Probable (requires further clinical testing)\n"
                "• ≤ 2 Points: Diagnosis Unlikely"
            )

        # --- WHAT ARE KAYSER-FLEISCHER RINGS? ---
        if any(term in clean_q for term in ["kayser", "fleischer", "kf ring", "kf rings"]):
            return sanitize_bot_answer(
                "👁️ Kayser-Fleischer (KF) Rings (Question Bank Sec. 3.4–3.5)\n\n"
                "• Definition: Golden-brown to greenish copper deposits in Descemet's membrane of the peripheral corneal limbus.\n"
                "• Clinical Prevalence: Present in >90% of patients with neurological manifestations and ~50% of hepatic cases.\n"
                "• Diagnosis: Requires dedicated slit-lamp ophthalmologic microscopy.\n"
                "• Diagnostic Significance: Confirmed presence awards +2 points on the Leipzig scale. Rings gradually fade and resolve with sustained de-coppering chelation therapy."
            )

        # --- WHAT IS ATP7B? ---
        if any(term in clean_q for term in ["atp7b", "gene", "mutation", "genetic"]):
            return sanitize_bot_answer(
                "🧬 ATP7B Genetic Profile in Wilson's Disease\n\n"
                "• Gene Location: ATP7B gene is located on chromosome 13q14.3.\n"
                "• Function: Encodes a P-type copper-transporting ATPase responsible for excreting excess copper into bile and incorporating copper into apoceruloplasmin.\n"
                "• Inheritance: Autosomal recessive. Offspring must inherit two pathogenic variants (homozygous or compound heterozygous) to develop clinical disease.\n"
                "• Diagnostic Value: Detection of two disease-causing mutations confirms diagnosis (+4 Leipzig points). First-degree relatives (siblings) should undergo genetic screening."
            )

        # --- WHAT IS SHAP & HOW PREDICTION WORKS ---
        if any(term in clean_q for term in ["predict", "prediction", "model", "neuromed", "shap", "stacking", "algorithm", "svm", "machine learning"]):
            return sanitize_bot_answer(
                "🔬 NeuroMed AI Multi-Model Stacking & SHAP Explainability (Question Bank Sec. 15)\n\n"
                "• Stacking Ensemble: Combines an SVM (Support Vector Machine) classifier and a regularized Logistic Regression model, fed into a calibrated Meta-Classifier for consensus probability scoring.\n"
                "• 23 Clinical Markers: Evaluates copper biochemistry (Ceruloplasmin, 24h Urine Copper, Free Copper), liver enzymes (ALT, AST, Bilirubin, Albumin, ALP, GGT), ophthalmic signs (KF rings), and ATP7B genetics.\n"
                "• SHAP (SHapley Additive exPlanations): Explains the prediction by measuring exactly which patient lab markers pushed the risk score higher or lower from baseline, visualizing individual Feature Attribution via interactive Force and Beeswarm plots."
            )

        # --- WHAT ARE THE COPPER BASED TESTS / TESTS FOR COPPER ---
        is_copper_test_query = (
            any(phrase in clean_q for phrase in [
                "copper based test", "copper based tests", "tests for copper", "test for copper",
                "tests measure copper", "test measures copper", "copper blood test", "copper test",
                "copper tests", "copper laboratory test", "copper lab test", "copper biomarkers"
            ])
            or (("copper" in words or "cu" in words) and any(w in words for w in ["test", "tests", "testing", "taken", "measure", "measures", "measured"]) and not any(w in words for w in ["urine", "urinary", "diet", "food", "eat"]))
        )
        if is_copper_test_query:
            return sanitize_bot_answer(
                "🧪 Copper-Based Diagnostic Tests for Wilson's Disease (AASLD / EASL Guidelines)\n\n"
                "Comprehensive evaluation of copper metabolism is essential for confirming Wilson's disease. Key copper-related tests include:\n\n"
                "1. Serum Ceruloplasmin Test:\n"
                "• Measures the primary copper-binding glycoprotein in blood. Levels < 10 mg/dL strongly support Wilson disease (+2 Leipzig points; normal range: 20–40 mg/dL).\n\n"
                "2. 24-Hour Urinary Copper Excretion Test:\n"
                "• Measures total copper eliminated in urine over 24 hours. A diagnostic value > 100 µg/24h (>1.6 µmol/d) confirms pathological copper overload (+2 Leipzig points; normal: < 40 µg/24h).\n\n"
                "3. Free (Non-Ceruloplasmin-Bound) Serum Copper:\n"
                "• Calculated as Total Serum Copper (µg/dL) - [3.15 × Ceruloplasmin (mg/dL)]. Levels > 15–25 µg/dL indicate elevated toxic copper circulating to vital organs.\n\n"
                "4. Total Serum Copper Test:\n"
                "• Measures total circulating copper (bound and unbound). Often reduced due to low ceruloplasmin, but rises acutely in fulminant hepatic necrosis.\n\n"
                "5. Hepatic Copper Quantification (Liver Biopsy):\n"
                "• Gold-standard tissue measurement. Parenchymal copper concentration > 250 µg/g dry weight establishes definitive tissue accumulation (+2 Leipzig points).\n\n"
                "6. Slit-Lamp Ophthalmic Examination for Kayser-Fleischer (KF) Rings:\n"
                "• Identifies golden-brown copper deposits in Descemet's membrane of the peripheral cornea (+2 Leipzig points)."
            )

        # --- 24-HOUR URINARY COPPER / URINE COPPER TEST ---
        if any(phrase in clean_q for phrase in ["24 hour urine copper", "24 hour urinary copper", "24hr urine copper", "24 h urine copper", "urine copper test", "urinary copper test", "urine copper", "urinary copper"]):
            return sanitize_bot_answer(
                "🧪 24-Hour Urinary Copper Test (Question Bank Sec. 4.10–4.11 & AASLD Guidelines)\n\n"
                "• Clinical Definition: Measures the total amount of copper excreted in urine over a full 24-hour collection period. It is one of the most reliable initial screening, diagnostic, and therapy-monitoring tests in Wilson disease.\n\n"
                "• Diagnostic Reference Ranges:\n"
                "  • Normal Baseline: < 40 µg/24 hours (< 0.6 µmol/day)\n"
                "  • Symptomatic Wilson Disease: > 100 µg/24 hours (> 1.6 µmol/day) — provides +2 points on the Leipzig diagnostic scale\n"
                "  • Intermediate / Asymptomatic: 40–100 µg/24 hours (warrants further clinical evaluation or penicillamine challenge test)\n\n"
                "• Treatment Monitoring Targets:\n"
                "  • Chelation Therapy (D-Penicillamine / Trientine): Target excretion is 200–500 µg/24h during stable maintenance therapy.\n"
                "  • Zinc Maintenance Therapy: Target excretion decreases to < 75 µg/24h, confirming effective blockade of intestinal copper absorption."
            )

        # --- STANDALONE COPPER TERM ---
        if clean_q == "copper" or clean_q in ["what is copper", "explain copper", "about copper", "copper role"]:
            return sanitize_bot_answer(
                "🧪 Copper Metabolism & Pathology in Wilson's Disease (Question Bank Sec. 2.1–2.8)\n\n"
                "• Normal Physiological Role: Copper is an essential trace element required as an enzymatic cofactor for mitochondrial energy production (cytochrome c oxidase), iron metabolism (ceruloplasmin), and neurotransmitter synthesis.\n\n"
                "• Pathophysiology in Wilson's Disease:\n"
                "  • In healthy individuals, the ATP7B transporter in the liver pumps surplus copper into bile for excretion in stool.\n"
                "  • In Wilson disease, mutations in ATP7B impair biliary copper excretion and ceruloplasmin incorporation. Excess copper progressively accumulates in hepatocytes, spills into the circulation as toxic free copper, and deposits in the brain, corneas, and kidneys.\n\n"
                "• Clinical Diagnostic Markers: Low serum ceruloplasmin (<10 mg/dL), elevated 24-hour urine copper (>100 µg/day), elevated non-ceruloplasmin-bound copper (>15 µg/dL), and Kayser-Fleischer rings."
            )

        # =============================================================
        # 3.5 CONVERSATIONAL FOLLOW-UP & PRONOUN RESOLUTION
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

        if (has_pronoun or is_followup_phrase) and not has_explicit_entity and chat_history:
            recent_topic = self._extract_recent_topic(chat_history)
            if recent_topic:
                # 1. Follow-up: Side effects & adverse events
                if any(term in clean_q for term in ["side effect", "side effects", "adverse", "risk", "risks", "reaction", "reactions"]):
                    if recent_topic == "trientine":
                        return sanitize_bot_answer(
                            "💊 Trientine Adverse Effects & Safety Profile (AASLD / EASL Guidelines)\n\n"
                            "• General Tolerability: Trientine has a significantly milder side-effect profile than D-penicillamine and is better tolerated by most patients.\n\n"
                            "• Documented Side Effects (Question Bank Sec. 11.9 / AASLD 2023):\n"
                            "  • Dysgeusia (loss or disturbance of taste sensation)\n"
                            "  • Mild proteinuria or renal changes\n"
                            "  • Early neurological worsening: Paradoxical neurological decline upon starting therapy (less frequent than with D-penicillamine)\n"
                            "  • Sideroblastic anemia or bone marrow changes (rare, secondary to induced copper deficiency)\n"
                            "  • Gastrointestinal upset or nausea\n\n"
                            "• Clinical Monitoring: Routine CBC, urinalysis, 24-hour urinary copper, and neurological exams should be monitored regularly during therapy."
                        )
                    elif recent_topic == "d-penicillamine":
                        return sanitize_bot_answer(
                            "💊 D-Penicillamine Adverse Effects & Safety Profile (AASLD / EASL Guidelines)\n\n"
                            "• Documented Adverse Effects (Question Bank Sec. 11.9 / AASLD 2023):\n"
                            "  • Hypersensitivity: Early skin rash, fever, lymphadenopathy\n"
                            "  • Renal Toxicity: Proteinuria, membranous nephropathy, nephrotic syndrome\n"
                            "  • Hematologic Toxicity: Bone marrow suppression (leukopenia, thrombocytopenia, rare aplastic anemia)\n"
                            "  • Autoimmune Syndromes: Drug-induced lupus, myasthenia gravis, Goodpasture syndrome\n"
                            "  • Neurological Worsening: Paradoxical worsening of tremors/dysarthria in 10–20% of patients upon initiating therapy\n\n"
                            "• Essential Co-Medication: Pyridoxine (Vitamin B6, 25–50 mg/day) must always be co-prescribed to prevent drug-induced deficiency."
                        )
                    elif recent_topic == "zinc":
                        return sanitize_bot_answer(
                            "💊 Zinc Therapy Adverse Effects (AASLD / EASL Guidelines)\n\n"
                            "• Primary Side Effect: Gastrointestinal irritation (dyspepsia, gastric pain, nausea) in up to 10–20% of patients.\n"
                            "• Other Considerations: Headaches or microcytic anemia if copper levels drop excessively.\n"
                            "• Administration Tip: Zinc must be taken separately from food and chelators (at least 1 hour before or 2 hours after meals) for optimal absorption."
                        )

                # 2. Follow-up: Alternatives & substitutions
                if any(term in clean_q for term in ["alternative", "alternatives", "substitute", "other option", "other drug"]):
                    if recent_topic in ["d-penicillamine", "trientine"]:
                        return sanitize_bot_answer(
                            "💊 Treatment Alternatives for Wilson's Disease (AASLD / EASL / WHO Guidelines)\n\n"
                            "• 1. Trientine Dihydrochloride (Primary First-Line Chelator Alternative):\n"
                            "  If D-penicillamine is contraindicated or causes intolerance (severe rash, nephrotoxicity, cytopenias), Trientine is the recommended first-line alternative with fewer immunological side effects.\n\n"
                            "• 2. Zinc Salts (Maintenance or Presymptomatic Alternative):\n"
                            "  Zinc acetate or gluconate blocks intestinal copper absorption by inducing intestinal metallothionein. Used for maintenance after de-coppering or first-line in presymptomatic/neurological patients.\n\n"
                            "• 3. Liver Transplantation:\n"
                            "  Reserved for acute fulminant liver failure or decompensated cirrhosis unresponsive to chelation (corrects metabolic defect)."
                        )
                    elif recent_topic == "zinc":
                        return sanitize_bot_answer(
                            "💊 Alternatives to Zinc Therapy in Wilson's Disease\n\n"
                            "• Active Chelating Agents: D-Penicillamine or Trientine Dihydrochloride are the primary systemic chelators when active de-coppering (rather than intestinal absorption blockade) is required."
                        )

                # 3. Follow-up: Clinical Importance / Why is it important
                if any(w in clean_q for w in ["why", "importance", "important", "role", "significance"]):
                    if recent_topic == "ceruloplasmin":
                        return sanitize_bot_answer(
                            "🧪 Why Ceruloplasmin is Important in Wilson's Disease (Question Bank Sec. 4.3–4.6)\n\n"
                            "• 1. Primary Copper Carrier in Blood:\n"
                            "  Ceruloplasmin binds over 90% of circulating copper under normal physiology.\n\n"
                            "• 2. Cardinal Diagnostic Biomarker:\n"
                            "  ATP7B gene mutations prevent copper incorporation into apoceruloplasmin, causing rapid degradation. Consequently, serum ceruloplasmin is typically < 10 mg/dL in Wilson disease.\n\n"
                            "• 3. Leipzig Score Pillar:\n"
                            "  A level < 10 mg/dL awards +2 points toward the Leipzig diagnostic consensus score, making it a critical screening and diagnostic pillar.\n\n"
                            "• 4. Guiding Free Copper Calculation:\n"
                            "  Serum ceruloplasmin is used to calculate non-ceruloplasmin-bound (free) copper, which correlates with tissue toxicity."
                        )
                    elif recent_topic == "kayser-fleischer rings":
                        return sanitize_bot_answer(
                            "👁️ Clinical Importance of Kayser-Fleischer (KF) Rings\n\n"
                            "• Direct Ophthalmic Sign: Formed by copper deposition in Descemet's membrane of the cornea.\n"
                            "• High Diagnostic Value: Confirmed presence awards +2 Leipzig points. Present in >90% of neurological Wilson disease cases.\n"
                            "• Monitoring Response: Rings gradually fade and disappear with successful long-term de-coppering therapy."
                        )
                    elif recent_topic == "atp7b gene":
                        return sanitize_bot_answer(
                            "🧬 Clinical Importance of the ATP7B Gene\n\n"
                            "• Definitive Genetic Etiology: Pathogenic mutations on both chromosome 13 alleles confirm Wilson's disease (+4 Leipzig points).\n"
                            "• Family Screening: Identifies asymptomatic siblings early, enabling preventative therapy before irreversible organ damage occurs."
                        )

                # General follow-up lookup against Question Bank with enriched subject
                enriched_msg = f"{raw_msg} {recent_topic}"
                enriched_ans = self.search_qa_bank(enriched_msg)
                if enriched_ans:
                    return sanitize_bot_answer(enriched_ans)

        # =============================================================
        # 4. CUSTOM USER-DEFINED Q&A LIST
        # =============================================================
        custom_qa_list = [
            # Add any other specific questions and answers here!
        ]

        for qa in custom_qa_list:
            if any(trigger in clean_q for trigger in qa.get("triggers", [])):
                return sanitize_bot_answer(qa["answer"])

        # =============================================================
        # 4.5 OFFICIAL QUESTION BANK DIRECT LOOKUP (EXACT & SEMANTIC)
        # Returns ONLY authoritative clinical answers without mentioning the question
        # =============================================================
        qb_answer = self.search_qa_bank(raw_msg)
        if qb_answer:
            return sanitize_bot_answer(qb_answer)

        # =============================================================
        # 5. GENERAL RETRIEVAL-AUGMENTED ANSWER (VECTORSTORE FALLBACK)
        # =============================================================
        retrieval_query = user_question
        if (has_pronoun or is_followup_phrase) and not has_explicit_entity and chat_history:
            recent_topic = self._extract_recent_topic(chat_history)
            if recent_topic:
                retrieval_query = f"{user_question} ({recent_topic})"

        context = self.retrieve_relevant_context(retrieval_query)

        if self.llm:
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
                return sanitize_bot_answer(llm_reply)
            except Exception as e:
                logger.warning(f"LLM chat generation failed: {e}")

        # Grounded structured clinical summary from retrieved guidelines
        # Strip any questions or headers so only pure answers are returned
        if not context or not context.strip():
            return sanitize_bot_answer(CLARIFICATION_MSG)

        lines = context.replace("===", "").strip().splitlines()
        content_lines = []
        for l in lines:
            if re.match(r'^\s*\d+[\.\)]\s+.*[?]', l) or re.match(r'^[A-Z][^?]*\?', l):
                continue
            content_lines.append(l)
        clean_body = "\n".join(content_lines).strip()
        if not clean_body:
            return sanitize_bot_answer(CLARIFICATION_MSG)

        return sanitize_bot_answer(clean_body)

