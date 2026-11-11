# 🏥 MEDITRON: Comprehensive Technical Guide & Clinical Architecture Document

> **Document Type**: Architecture Specification & Technical Analysis  
> **Source Grounding**: Official EPFL LLM Team Repository (`epfLLM/meditron`), Paper (*arXiv:2311.16079*), and Technical Specifications  
> **Target Project**: Nero Med AI (Wilson's Disease Clinical Support & RAG Engine)  
> **Version**: 1.0.0  

---

## 📋 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Model Identification & Core Specifications](#2-model-identification--core-specifications)
3. [The GAP-Replay Pretraining Corpus](#3-the-gap-replay-pretraining-corpus)
4. [Distributed Training & Hardware Architecture](#4-distributed-training--hardware-architecture)
5. [Supervised Fine-Tuning (SFT) & Chain-of-Thought (CoT)](#5-supervised-fine-tuning-sft--chain-of-thought-cot)
6. [Clinical Capabilities & Benchmark Performance](#6-clinical-capabilities--benchmark-performance)
7. [The Official Clinical Advisory Notice (Crucial Guardrails)](#7-the-official-clinical-advisory-notice-crucial-guardrails)
8. [Deployment Options (vLLM, Transformers, Ollama)](#8-deployment-options-vllm-transformers-ollama)
9. [Integration Blueprint for Nero Med AI](#9-integration-blueprint-for-nero-med-ai)

---

## 1. Executive Summary

**MEDITRON** is a family of open-source, evidence-based medical Large Language Models (LLMs) developed by the **EPFL LLM Team** (École Polytechnique Fédérale de Lausanne, Switzerland) in collaboration with the **Yale School of Medicine** and international humanitarian stakeholders (including the International Committee of the Red Cross - ICRC).

Prior to Meditron, state-of-the-art clinical language models (such as Google’s Med-PaLM 2 or OpenAI’s GPT-4) were strictly closed-source, proprietary, and cloud-hosted. This created severe privacy barriers for hospitals subject to HIPAA/GDPR, as well as cost and accessibility barriers for low-resource healthcare institutions.

Meditron addresses this by performing **continual domain-adaptive pretraining** on Meta's open Llama-2 foundation models using **48.1 Billion tokens** of curated, peer-reviewed medical evidence (*GAP-Replay*). In academic evaluations, **Meditron-70B outperformed Llama-2-70B, GPT-3.5, and Flan-PaLM** across standard clinical reasoning benchmarks.

---

## 2. Model Identification & Core Specifications

```
                           Llama-2 Base Foundation
                           (7B and 70B Parameters)
                                      │
                                      ▼
                   GAP-Replay Continual Pre-Training
                       (48.1 Billion Medical Tokens)
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
             MEDITRON-7B                           MEDITRON-70B
       • 7 Billion parameters                • 70 Billion parameters
       • 2,048 sequence length               • 4,096 sequence length
       • Workstation GPU deployable          • Multi-GPU cluster deployable
```

| Parameter | Meditron-7B | Meditron-70B |
| :--- | :--- | :--- |
| **Developer** | EPFL LLM Team & Yale School of Medicine | EPFL LLM Team & Yale School of Medicine |
| **Architecture** | Auto-regressive Causal Decoder Transformer | Auto-regressive Causal Decoder Transformer |
| **Base Model** | Meta Llama-2-7B | Meta Llama-2-70B |
| **Context Length** | 2,048 tokens | 4,096 tokens (4k) |
| **Input / Output** | Text only $\rightarrow$ Text only | Text only $\rightarrow$ Text only |
| **Knowledge Cutoff** | August 2023 | August 2023 |
| **Training Framework**| Megatron-LLM (distributed) | Megatron-LLM (distributed) |
| **License** | Llama 2 Community License / Apache 2.0 Code | Llama 2 Community License / Apache 2.0 Code |
| **Hugging Face ID** | `epfl-llm/meditron-7b` | `epfl-llm/meditron-70b` |

---

## 3. The GAP-Replay Pretraining Corpus

The cornerstone of Meditron is its domain-adaptive pretraining dataset, dubbed **GAP-Replay**, comprising **48.1 Billion tokens** across four specialized corpora:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           GAP-Replay Corpus (48.1B Tokens)                              │
├────────────────────────────────┬────────────────────────────────────────────────────────┤
│ G - Clinical Guidelines        │ 46,000 guidelines from international health orgs &     │
│                                │ hospitals (36k open-access on HuggingFace datasets).   │
├────────────────────────────────┼────────────────────────────────────────────────────────┤
│ A - Paper Abstracts            │ 16.1 Million abstracts extracted from closed-access    │
│                                │ PubMed and PubMed Central (PMC) publications.          │
├────────────────────────────────┼────────────────────────────────────────────────────────┤
│ P - Medical Papers             │ 5.0 Million full-text, peer-reviewed biomedical papers │
│                                │ from open-access PubMed Central (PMC).                 │
├────────────────────────────────┼────────────────────────────────────────────────────────┤
│ Replay - General Knowledge     │ 400 Million general-domain tokens sampled from         │
│                                │ RedPajama-v1 to prevent "catastrophic forgetting".    │
└────────────────────────────────┴────────────────────────────────────────────────────────┘
```

### Why the "Replay" Component is Vital
When neural networks are trained exclusively on a new domain (e.g., pure medical pathology), they suffer from **catastrophic forgetting**—losing their ability to follow general English grammatical structures, multi-step logical deduction, and everyday context. By interleaving a **400M-token general replay buffer**, Meditron preserved Llama-2's reasoning capacity while acquiring deep medical specialization.

---

## 4. Distributed Training & Hardware Architecture

Training a 70-billion parameter model on tens of billions of tokens requires supercomputing-scale hardware. The EPFL team developed and utilized **Megatron-LLM** (a distributed training framework derived from NVIDIA Megatron-LM).

### Hardware Infrastructure
* **16 Compute Nodes**, each equipped with **8x NVIDIA A100 (80GB SXM) GPUs** (Total: **128x A100 GPUs**).
* Intra-node communication: **NVLink** and **NVSwitch**.
* Inter-node networking: Single **NVIDIA ConnectX-6 DX** with **RDMA over Converged Ethernet (RoCE)**.
* Host Compute: Dual **AMD EPYC 7543 32-Core Processors** and **512 GB RAM** per node.

### 3-Way 3D Parallelism Scheme

```
                        3D Megatron Parallelism Scheme
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
  Data Parallelism (DP)      Tensor Parallelism (TP)     Pipeline Parallelism (PP)
 Splits batches across        Splits matrix weight       Splits transformer layers
 different GPU workers        tensors across GPUs        across sequential GPU stages
```

* **Meditron-7B Configuration**: $\text{DP}=16, \text{TP}=4, \text{PP}=1$ (Micro-batch: 10, Global Batch: 1600, LR: $3\times 10^{-4}$).
* **Meditron-70B Configuration**: $\text{DP}=2, \text{TP}=8, \text{PP}=8$ (Micro-batch: 2, Global Batch: 512, LR: $1.5\times 10^{-4}$).

---

## 5. Supervised Fine-Tuning (SFT) & Chain-of-Thought (CoT)

Meditron-70B was released primarily as a **continued-pretrained foundation model**. To adapt it to interactive conversational question-answering, the authors created an automated fine-tuning pipeline (`sft.py`).

### Key SFT Characteristics
1. **Chain-of-Thought (CoT) Prompting**: Tuned on medical reasoning benchmarks (like `medqa_cot_train.jsonl`), teaching the model to articulate its clinical rationale step-by-step before concluding on a diagnosis or treatment.
2. **Chat Markup Tokenization**: Integrated dialogue delimiter tokens:
   * `<|im_start|>` (initiates system, user, or assistant turn)
   * `<|im_end|>` (closes the dialogue turn)
3. **In-Context Demonstrations**: In zero-shot or few-shot scenarios, Meditron employs $k=3$ to $k=5$ demonstration exemplars within the prompt context window to steer output formatting.

---

## 6. Clinical Capabilities & Benchmark Performance

Meditron-70B was evaluated against four primary clinical QA and reasoning benchmarks:
1. **MedQA (USMLE)**: Questions modeled after the United States Medical Licensing Examination. Meditron-70B scored **~70.2%**, achieving a decisive passing grade and outperforming baseline Llama-2-70B and GPT-3.5.
2. **PubMedQA**: Biomedical research question-answering requiring understanding of study methodology, statistical significance, and conclusions.
3. **MedMCQA**: Multi-subject Indian medical entrance exam questions covering 21 preclinical, paraclinical, and clinical specialties.
4. **MMLU Clinical Subjects**: Evaluating broad medical biochemistry, anatomy, clinical psychology, and medical genetics.

---

## 7. The Official Clinical Advisory Notice (Crucial Guardrails)

The EPFL team included a mandatory advisory notice that must govern any engineering adoption:

> [!CAUTION]
> **Official Advisory Notice from EPFL**:  
> *"While Meditron is designed to encode medical knowledge from sources of high-quality evidence, it is not yet adapted to deliver this knowledge appropriately, safely, or within professional actionable constraints. We recommend against using Meditron in medical applications without extensive use-case alignment, as well as additional testing, specifically including randomized controlled trials in real-world practice settings."*

### Why This Notice Matters for Our Project:
1. **Meditron is NOT an autonomous clinician**: It must never be allowed to prescribe drug dosages or confirm diagnoses autonomously.
2. **It must be constrained by RAG**: In our architecture, Meditron must **never answer from memory alone**. It must be strictly fed the exact retrieved passages from verified AASLD/EASL guideline PDFs stored in our ChromaDB vector database.
3. **Human-in-the-Loop**: All outputs are decision-support suggestions for licensed physicians, not direct-to-patient orders.

---

## 8. Deployment Options

### Option A: High-Throughput Production via `vLLM` & `FastChat` (Recommended)
vLLM utilizes PagedAttention to maximize memory efficiency and inference throughput:
```bash
# Start an OpenAI-compatible local vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model epfl-llm/meditron-7b \
    --port 8000 \
    --max-model-len 2048 \
    --dtype bfloat16
```

### Option B: Local Lightweight Workstation Deployment via `Ollama`
For clinical workstations without enterprise clusters, quantized Meditron (GGUF format) runs locally:
```bash
# Pull and start local Meditron instance
ollama run meditron:7b
# Exposes API at: http://localhost:11434/v1
```

### Option C: Native Hugging Face Transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "epfl-llm/meditron-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

prompt = "<|im_start|>user\nWhat are the primary clinical indications for D-penicillamine in Wilson's disease?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 9. Integration Blueprint for Nero Med AI

Here is how Meditron directly integrates into the Nero Med AI software stack:

```
                                  USER QUERY / PATIENT LAB REPORT
                                                 │
                                                 ▼
                                     rag_system.py (Retrieval)
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             ChromaDB Vector Store                           Biomarker & Leipzig Engine
         (AASLD, EASL, NIH Guideline PDFs)                   (clinical_evaluation.py)
                        │                                                 │
                        └────────────────────────┬────────────────────────┘
                                                 │
                                                 ▼
                                  Construct Guarded Clinical Prompt:
                        "Using ONLY the official guideline excerpt below,
                         evaluate the patient's chelation management..."
                                                 │
                                                 ▼
                                ┌─────────────────────────────────┐
                                │   RAG Provider (Configurable)   │
                                └────────────────┬────────────────┘
                                                 │
                   ┌─────────────────────────────┴─────────────────────────────┐
                   ▼                                                           ▼
       🏥 Mode A: Local Meditron                                    🌐 Mode B: Cloud Gemini
       • 100% On-Premise / HIPAA Compliant                          • High-speed cloud API
       • Connects to: http://localhost:11434                        • Zero GPU needed
       • Uses GAP-Replay medical pre-training                       • Default development mode
```

### Next Steps for Implementation
1. Add `MEDITRON_URL` and `RAG_LLM_PROVIDER` parameters to [.env](file:///d:/nero-med-ai%20-%20backup/.env).
2. Update [rag_system.py](file:///d:/nero-med-ai%20-%20backup/rag_system.py) to support local OpenAI-compatible endpoints (`http://localhost:11434/v1` or `http://localhost:8000/v1`).
3. Keep the numerical diagnostic models ([train_v3_quad_ensemble.py](file:///d:/nero-med-ai%20-%20backup/scripts/train_v3_quad_ensemble.py)) strictly separate from the language generation layer.

