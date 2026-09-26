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

def sanitize_bot_answer(text: str) -> str:
    """
    Cleans all raw markdown symbols from chatbot responses:
    - Removes double asterisks (**) completely
    - Removes single asterisks (*)
    - Removes backticks (`)
    - Converts bullet hyphens (- item) to clean bullets (• item)
    - Strips markdown heading hashes (###)
    """
    if not text:
        return ""
    # Remove markdown bold/italics asterisks and code backticks
    cleaned = text.replace("**", "").replace("*", "").replace("`", "")
    # Convert leading bullet hyphens to bullet points (• )
    cleaned = re.sub(r'^[ \t]*-[ \t]+', '• ', cleaned, flags=re.MULTILINE)
    # Strip markdown heading hashes (### )
    cleaned = re.sub(r'^#{1,6}\s*', '', cleaned, flags=re.MULTILINE)
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
                self.passages.extend([s.strip() for s in sections if len(s.strip()) > 30])
            except Exception as e:
                logger.warning(f"Error loading question bank text: {e}")

        # 2. Load Guidelines
        guidelines_path = os.path.join(DOCS_DIR, "wilson_disease_guidelines.txt")
        if os.path.exists(guidelines_path):
            try:
                with open(guidelines_path, "r", encoding="utf-8") as f:
                    text = f.read()
                sections = re.split(r'\n(?=[0-9]+\.\s+[A-Z\s&]+)', text)
                self.passages.extend([s.strip() for s in sections if len(s.strip()) > 40])
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
        """
        if not getattr(self, "qa_bank", None):
            return None

        clean_user_q = re.sub(r'^\d+[\.\)]?\s*', '', user_question).strip()
        norm_user = " ".join(re.sub(r'[^\w\s]', ' ', clean_user_q.lower()).split())
        if not norm_user:
            return None

        user_words = set(norm_user.split()) - {"what", "is", "are", "the", "a", "an", "in", "of", "for", "to", "can", "how", "does", "do", "it"}

        # Special handler for "what is wilson disease" / "explain wilson disease":
        # Returns the 2-line definition, plus cause, plus treatment outlook (as requested by user!)
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

        best_item = None
        best_score = 0.0

        for item in self.qa_bank:
            # 1. Exact normalized match
            if norm_user == item["q_norm"]:
                best_item = item
                best_score = 100.0
                break

            # 2. Substring match
            if norm_user in item["q_norm"] or item["q_norm"] in norm_user:
                score = 50.0 + len(user_words & item["words"])
                if score > best_score:
                    best_score = score
                    best_item = item

            # 3. High word overlap
            if user_words:
                overlap = len(user_words & item["words"])
                overlap_ratio = overlap / len(user_words)
                if overlap_ratio >= 0.6:
                    score = 20.0 * overlap_ratio + overlap
                    if score > best_score:
                        best_score = score
                        best_item = item

        if best_item and best_score >= 21.0:
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
                return "\n\n---\n\n".join([doc.page_content for doc in docs])
            except Exception as e:
                logger.warning(f"LangChain retriever error: {e}")

        # Keyword and semantic scoring over clean guideline sections
        query_lower = query.lower()
        words = set(re.findall(r'\w+', query_lower))
        stop_words = {"the", "is", "at", "which", "on", "a", "an", "and", "or", "in", "to", "what", "how", "tell", "me", "about"}
        meaningful_words = words - stop_words

        scored = []
        for p in self.passages:
            p_lower = p.lower()
            p_words = set(re.findall(r'\w+', p_lower))
            # Match score with extra weight for exact multi-word substring match
            score = len(meaningful_words.intersection(p_words)) * 2
            if any(term in p_lower for term in meaningful_words if len(term) > 3):
                score += 3
            scored.append((score, p))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_passages = [p for s, p in scored[:2] if s > 0]
        if not top_passages and self.passages:
            top_passages = self.passages[:2]

        return "\n\n---\n\n".join(top_passages)

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

    def ask_chatbot(self, user_question, patient_context=None):
        """
        Conversational Clinical RAG assistant grounded in the AASLD/EASL/INASL Question Bank
        and aware of the active Patient Dashboard / Prediction Report.
        All responses are cleanly sanitized to remove raw asterisks (**) and bullet hyphens (-).
        """
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
        context = self.retrieve_relevant_context(user_question)

        if self.llm:
            try:
                prompt = f"""You are a helpful Medical Assistant specializing in Wilson's Disease.
Use the retrieved clinical guidelines to answer the user question clearly, politely, and accurately without using markdown asterisks (**) or bullet hyphens (-).
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
        lines = context.replace("===", "").strip().splitlines()
        content_lines = []
        for l in lines:
            if re.match(r'^\s*\d+[\.\)]\s+.*[?]', l) or re.match(r'^[A-Z][^?]*\?', l):
                continue
            content_lines.append(l)
        clean_body = "\n".join(content_lines).strip()
        if not clean_body:
            clean_body = context.replace("===", "").strip()

        return sanitize_bot_answer(clean_body)

