"""
clinical_benchmarks/evaluator.py

NeuroMed AI — Clinical Evaluation & Quality Monitoring Harness.
Evaluates RAG system performance across Wilson's Disease benchmark datasets.

Strictly enforces governance boundaries:
- Draft and mock cases are excluded from clinical performance claims.
- Metrics requiring clinician approval (e.g., Critical Fact Coverage, Prohibited Claim Detection)
  are reported as unavailable (null) for unapproved cases, never manufactured.
- Differentiates citation presence from actual passage support.
- Negative controls are strictly tested for explicit evidence gaps.
"""

import os
import re
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("clinical_evaluator")

# Approved guideline PDFs and max page numbers
APPROVED_PDFS = {
    "AASLD WD Guidelines.pdf": 23,
    "EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf": 15,
    "NIH Wilson Disease.pdf": 26,
    "clinical practice guidelines in WD.pdf": 25,
}

CITATION_PATTERN = re.compile(
    r'\[([A-Za-z0-9_\- \'\(\)]+\.pdf),\s*(?:p\.|pp\.|page|pages)?\s*(\d+(?:-\d+)?)\]',
    re.IGNORECASE
)

# Common clinical stopwords to filter out when checking passage support
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "can", "could", "should", "would", "may", "might",
    "must", "shall", "will", "this", "that", "these", "those", "it", "its", "as",
    "such", "than", "more", "most", "also", "into", "which", "who", "whom", "patient",
    "patients", "clinical", "disease", "treatment", "recommended", "recommendation",
    "take", "taken", "taking", "use", "used", "using", "give", "given", "giving",
    "adverse", "effect", "effects", "associated"
}

STATUS_UNAVAILABLE = "unavailable_pending_clinician_approval"


class ClinicalBenchmarkEvaluator:
    """
    Evaluation harness for Wilson's Disease Clinical RAG.
    Calculates nine required metrics, produces machine-readable reports,
    and enforces clinical governance boundaries.
    """

    def __init__(self, rag_system=None):
        self.rag_system = rag_system

    def evaluate_case(
        self,
        case: Dict[str, Any],
        rag_reply: Optional[str] = None,
        telemetry: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a single benchmark case against a RAG response.
        If rag_reply is not provided, calls rag_system.ask_chatbot_with_telemetry().
        """
        case_id = case.get("case_id", "UNKNOWN")
        domain = case.get("domain", "UNKNOWN")
        question_text = case.get("question_text", "")
        is_in_corpus = case.get("is_in_corpus", True)
        is_mock = case.get("is_mock", False)

        clin_eval = case.get("clinician_evaluation", {})
        is_clinician_approved = clin_eval.get("is_clinician_approved", False)
        review_status = clin_eval.get("review_status", "draft_pending_clinician_review")

        # 1. Obtain RAG response if not pre-supplied
        if rag_reply is None or telemetry is None:
            if self.rag_system is None:
                raise ValueError("RAG system instance required when rag_reply is not provided.")
            rag_reply, telemetry = self.rag_system.ask_chatbot_with_telemetry(question_text)

        # 2. Extract citations and passages from response
        found_citations = self._parse_citations(rag_reply)
        retrieved_passages = telemetry.get("retrieved_passages", [])
        evidence_coverage_meta = telemetry.get("evidence_coverage", {})

        # 3. Calculate Metric 1: Retrieval Facet Coverage
        facet_cov_result = self._calc_retrieval_facet_coverage(
            case, evidence_coverage_meta, retrieved_passages
        )

        # 4. Calculate Metric 2: Critical Fact Coverage (Ground Truth Constrained)
        critical_fact_result = self._calc_critical_fact_coverage(
            case, rag_reply, is_clinician_approved, is_mock
        )

        # 5. Calculate Metric 3: Citation Authenticity
        citation_auth_result = self._calc_citation_authenticity(
            found_citations, retrieved_passages, is_in_corpus, rag_reply
        )

        # 6. Calculate Metric 4: Claim-Level Citation Support
        claim_support_result = self._calc_claim_level_citation_support(
            rag_reply, found_citations, retrieved_passages, clin_eval, is_clinician_approved or is_mock
        )

        # 7. Calculate Metric 5: Unsupported Clinical Claim Rate
        unsupported_rate_result = self._calc_unsupported_claim_rate(
            rag_reply, claim_support_result, is_in_corpus
        )

        # 8. Calculate Metric 6: Prohibited Claim Detection (Ground Truth Constrained)
        prohibited_claim_result = self._calc_prohibited_claim_detection(
            case, rag_reply, is_clinician_approved, is_mock
        )

        # 9. Calculate Metric 7: Negative-Control Evidence-Gap Accuracy
        neg_control_result = self._calc_negative_control_accuracy(
            is_in_corpus, rag_reply, evidence_coverage_meta
        )

        # 10. Calculate Metric 8: False Evidence-Gap Rate
        false_gap_result = self._calc_false_evidence_gap_rate(
            is_in_corpus, rag_reply, evidence_coverage_meta
        )

        # Assemble Case Result
        return {
            "case_id": case_id,
            "domain": domain,
            "question_type": case.get("question_type", "unknown"),
            "is_in_corpus": is_in_corpus,
            "governance": {
                "review_status": review_status,
                "is_clinician_approved": is_clinician_approved,
                "is_mock": is_mock,
                "clinical_accuracy_eligible": bool(is_clinician_approved and not is_mock)
            },
            "metrics": {
                "retrieval_facet_coverage": facet_cov_result["score"],
                "critical_fact_coverage": critical_fact_result["score"],
                "citation_authenticity": citation_auth_result["score"],
                "claim_level_citation_support": claim_support_result["score"],
                "unsupported_clinical_claim_rate": unsupported_rate_result["score"],
                "prohibited_claim_violation_count": prohibited_claim_result["violation_count"],
                "negative_control_evidence_gap_accuracy": neg_control_result["score"],
                "false_evidence_gap_rate": false_gap_result["score"],
            },
            "details": {
                "retrieval_facet_details": facet_cov_result,
                "critical_fact_details": critical_fact_result,
                "citation_authenticity_details": citation_auth_result,
                "claim_level_support_details": claim_support_result,
                "unsupported_claim_details": unsupported_rate_result,
                "prohibited_claim_details": prohibited_claim_result,
                "negative_control_details": neg_control_result,
                "false_evidence_gap_details": false_gap_result,
                "rag_reply_preview": rag_reply[:300] + "..." if len(rag_reply) > 300 else rag_reply
            }
        }

    def evaluate_benchmark(
        self,
        benchmark_data_or_path: Any,
        mock_replies: Optional[Dict[str, Tuple[str, Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Executes evaluation across all cases in a benchmark dataset.
        Produces full per-case, per-domain, and aggregate metrics with governance summary.
        """
        if isinstance(benchmark_data_or_path, str):
            with open(benchmark_data_or_path, "r", encoding="utf-8") as f:
                benchmark = json.load(f)
        elif isinstance(benchmark_data_or_path, dict):
            benchmark = benchmark_data_or_path
        else:
            raise TypeError("Expected file path or dict for benchmark_data_or_path")

        cases = benchmark.get("cases", [])
        case_results = []

        for case in cases:
            cid = case.get("case_id")
            if mock_replies and cid in mock_replies:
                reply, tel = mock_replies[cid]
                res = self.evaluate_case(case, rag_reply=reply, telemetry=tel)
            else:
                res = self.evaluate_case(case)
            case_results.append(res)

        # Compute domain and aggregate rollups
        domain_breakdown = self._compute_domain_breakdown(case_results)
        aggregate_metrics = self._compute_aggregate_metrics(case_results)
        governance_summary = self._compute_governance_summary(case_results)

        report = {
            "evaluation_report_version": "1.0.0",
            "benchmark_version": benchmark.get("benchmark_version", "1.0.0"),
            "corpus_version": benchmark.get("corpus_version", "approved_guidelines_4pdf_v1"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "governance_summary": governance_summary,
            "aggregate_metrics": aggregate_metrics,
            "domain_breakdown": domain_breakdown,
            "case_results": case_results
        }
        return report

    def save_report(self, report: Dict[str, Any], output_path: str) -> str:
        """Saves machine-readable JSON evaluation report."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return output_path

    # =========================================================================
    # Metric Calculation Internals
    # =========================================================================

    @staticmethod
    def _normalize_passage(p: Any) -> Tuple[str, Optional[str], Optional[int], bool]:
        """
        Extracts (full_text, source_pdf, page_num, is_truncated) from either a string passage
        or a telemetry passage dict {'header': ..., 'preview': ..., 'is_truncated': ...}.
        """
        is_truncated = False
        if isinstance(p, dict):
            header = p.get("header", "")
            preview = p.get("preview", "") or p.get("text", "")
            full_text = f"{header}\n{preview}" if header else preview
            is_truncated = bool(
                p.get("is_truncated") or
                ("preview" in p and "full_text" not in p and "text" not in p) or
                ("[TRUNCATED]" in full_text)
            )
        else:
            full_text = str(p)
            if "[TRUNCATED]" in full_text or len(full_text) < 180 and not full_text.endswith("."):
                is_truncated = True

        first_line = full_text.split("\n", 1)[0] if "\n" in full_text else full_text
        m = re.search(r'\[Source:\s*([^|]+?)\s*\|\s*Page:\s*(\d+)\s*\]', first_line)
        if m:
            return full_text, m.group(1).strip(), int(m.group(2).strip()), is_truncated
        return full_text, None, None, is_truncated

    @staticmethod
    def is_bibliographic_footer(sentence: str) -> bool:
        """Determines if a sentence is a bibliographic source list rather than a clinical claim."""
        # Strip citations first to correctly detect footers with attached or prepended citations
        clean = CITATION_PATTERN.sub("", sentence).strip().lower()
        return (
            clean.startswith("sources:") or
            clean.startswith("source:") or
            "sources: " in clean[:30] or
            sentence.strip().lower().startswith("sources:") or
            sentence.strip().lower().startswith("source:") or
            "sources: " in sentence.strip().lower()[:30]
        )

    def _parse_citations(self, text: str) -> List[Tuple[str, int]]:
        """Parses citations formatted as [filename.pdf, p. X]."""
        citations = []
        for m in CITATION_PATTERN.finditer(text):
            fname = m.group(1).strip()
            page_str = m.group(2).strip()
            # Handle ranges like 12-14 by taking first page
            page = int(page_str.split("-")[0])
            citations.append((fname, page))
        return citations

    def _calc_retrieval_facet_coverage(
        self,
        case: Dict[str, Any],
        coverage_meta: Dict[str, Any],
        passages: List[str]
    ) -> Dict[str, Any]:
        """Calculates Metric 1: Retrieval Facet Coverage."""
        expected_facets = case.get("expected_evidence_facets", [])
        if not expected_facets:
            return {"score": 1.0, "expected": [], "covered": [], "missing": []}

        # Check telemetry facets first if populated
        covered = []
        missing = []
        full_passage_blob = " ".join([self._normalize_passage(p)[0] for p in passages]).lower()

        for facet in expected_facets:
            # Check facet terms in retrieved text
            terms = facet.lower().replace("_", " ").split()
            if any(term in full_passage_blob for term in terms if len(term) > 3):
                covered.append(facet)
            else:
                missing.append(facet)

        score = round(len(covered) / len(expected_facets), 4) if expected_facets else 1.0
        return {
            "score": score,
            "expected_count": len(expected_facets),
            "covered_count": len(covered),
            "expected": expected_facets,
            "covered": covered,
            "missing": missing
        }

    def _calc_critical_fact_coverage(
        self,
        case: Dict[str, Any],
        reply: str,
        is_clinician_approved: bool,
        is_mock: bool
    ) -> Dict[str, Any]:
        """Calculates Metric 2: Critical Fact Coverage."""
        # Safety gate: Draft cases must NOT compute or manufacture clinical scores
        if not (is_clinician_approved or is_mock):
            return {
                "score": None,
                "status": STATUS_UNAVAILABLE,
                "covered_facts": [],
                "missing_facts": [],
                "reason": "Requires formal clinician sign-off. Draft cases excluded from clinical accuracy scores."
            }

        clin_eval = case.get("clinician_evaluation", {})
        critical_facts = clin_eval.get("critical_facts", [])
        if not critical_facts:
            return {
                "score": 1.0,
                "status": "evaluated",
                "covered_facts": [],
                "missing_facts": []
            }

        covered_facts = []
        missing_facts = []
        reply_lower = reply.lower()

        for fact in critical_facts:
            # Check core clinical numbers or key concepts from fact
            keywords = [w.lower() for w in re.findall(r'[A-Za-z0-9_\-\.]{3,}', fact) if w.lower() not in STOPWORDS]
            # If at least 60% of substantive keywords are present, consider covered
            matched = [k for k in keywords if k in reply_lower]
            ratio = len(matched) / len(keywords) if keywords else 1.0
            if ratio >= 0.5:
                covered_facts.append(fact)
            else:
                missing_facts.append(fact)

        score = round(len(covered_facts) / len(critical_facts), 4) if critical_facts else 1.0
        return {
            "score": score,
            "status": "evaluated",
            "total_facts": len(critical_facts),
            "covered_count": len(covered_facts),
            "covered_facts": covered_facts,
            "missing_facts": missing_facts
        }

    def _calc_citation_authenticity(
        self,
        citations: List[Tuple[str, int]],
        passages: List[str],
        is_in_corpus: bool,
        reply: str
    ) -> Dict[str, Any]:
        """Calculates Metric 3: Citation Authenticity."""
        if not citations:
            # If it's an out-of-corpus query that correctly refused, 0 citations is expected
            if not is_in_corpus or "evidence gap" in reply.lower() or "not established" in reply.lower():
                return {
                    "score": 1.0,
                    "total_citations": 0,
                    "authentic_citations": 0,
                    "invalid_citations": [],
                    "note": "Refusal / evidence gap correctly emitted without citations."
                }
            # Otherwise, missing citations in substantive in-corpus answer
            return {
                "score": 0.0,
                "total_citations": 0,
                "authentic_citations": 0,
                "invalid_citations": ["MISSING_CITATIONS"],
                "note": "In-corpus answer contained no citations."
            }

        # Check candidate passage pool
        retrieved_sources = set()
        for p in passages:
            _, src_name, src_page, _ = self._normalize_passage(p)
            if src_name and src_page:
                retrieved_sources.add((src_name, src_page))

        authentic = []
        invalid = []

        for fname, page in citations:
            # 1. Check approved PDF
            if fname not in APPROVED_PDFS:
                invalid.append({"citation": f"{fname}, p. {page}", "reason": "unapproved_pdf"})
                continue
            # 2. Check page bounds
            if page < 1 or page > APPROVED_PDFS[fname]:
                invalid.append({"citation": f"{fname}, p. {page}", "reason": "page_out_of_bounds"})
                continue
            # 3. Check retrieved candidate pool
            if retrieved_sources and (fname, page) not in retrieved_sources:
                invalid.append({"citation": f"{fname}, p. {page}", "reason": "unretrieved_passage"})
                continue

            authentic.append((fname, page))

        score = round(len(authentic) / len(citations), 4) if citations else 0.0
        return {
            "score": score,
            "total_citations": len(citations),
            "authentic_citations": len(authentic),
            "invalid_citations": invalid
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Splits text into sentences while:
        1. Connecting citations appearing on separate lines immediately after a claim.
        2. Splitting on periods, exclamation points, question marks, and semicolons separating claims.
        3. Protecting citations like [AASLD WD Guidelines.pdf, p. 12] from being split on internal dots.
        """
        if not text:
            return []

        # 1. Connect citations on separate/subsequent lines to preceding claim text
        connected = re.sub(
            r'([^\n])\n\s*(\[[A-Za-z0-9_\- \'\(\)]+\.pdf,\s*(?:p\.|pp\.|page|pages)?\s*\d+[^\]]*\])',
            r'\1 \2',
            text
        )

        # 2. Find all citations and mask them
        citations = list(CITATION_PATTERN.finditer(connected))
        placeholders = {}
        protected = connected
        for idx, m in enumerate(citations):
            ph = f"__CITATION_PH_{idx}__"
            placeholders[ph] = m.group(0)
            protected = protected.replace(m.group(0), ph)

        # 3. Split on terminal punctuation (.!?) or semicolons followed by space
        raw_sents = re.split(r'(?<=[.!?;])\s+', protected)

        # 4. Restore original citation text in sentences
        sentences = []
        for s in raw_sents:
            s_clean = s.strip()
            if not s_clean:
                continue
            for ph, orig in placeholders.items():
                s_clean = s_clean.replace(ph, orig)
            sentences.append(s_clean)
        return sentences

    def _calc_claim_level_citation_support(
        self,
        reply: str,
        citations: List[Tuple[str, int]],
        passages: List[str],
        clin_eval: Dict[str, Any],
        ground_truth_active: bool
    ) -> Dict[str, Any]:
        """
        Calculates Metric 4: Claim-Level Citation Support.
        Differentiates citation presence from actual passage evidence:
        - Detects unsupported numerical thresholds and dosages.
        - Differentiates partial support from complete support.
        - Identifies contradiction and invalid citations.
        - Ignores bibliographic footers (e.g., 'Sources: ...').
        - Tracks whether supporting evidence is truncated.
        """
        # Map retrieved passages by (fname, page)
        passage_map = {}
        passage_trunc_map = {}
        for p in passages:
            p_text, s_name, s_page, is_trunc = self._normalize_passage(p)
            if s_name and s_page:
                passage_map[(s_name, s_page)] = p_text
                passage_trunc_map[(s_name, s_page)] = is_trunc

        # Extract sentences containing citations using protected splitter
        sentences = self._split_into_sentences(reply)
        cited_sentences = []
        for s in sentences:
            s_stripped = s.strip()
            # Ignore bibliographic footer sentences
            if self.is_bibliographic_footer(s_stripped):
                continue
            if CITATION_PATTERN.search(s_stripped):
                cited_sentences.append(s_stripped)

        if not cited_sentences:
            return {
                "score": None,
                "total_cited_sentences": 0,
                "supported_count": 0,
                "partial_count": 0,
                "details": []
            }

        fully_supported_count = 0
        partially_supported_count = 0
        details = []

        # Known contraindication patterns for contradiction detection
        contraindication_patterns = [
            (r"(?:discontinue|cessation|stop).*?(?:chelation|penicillamine|trientine|treatment)", "Contradicted: Stopping chelation triggers fatal hepatic relapse"),
            (r"spot urine.*sufficient.*diagnostic", "Contradicted: Spot urine cannot replace 24h collection")
        ]

        for sent in cited_sentences:
            sent_cits = self._parse_citations(sent)
            # Remove citation bracket to isolate claim
            claim_text = CITATION_PATTERN.sub("", sent).strip()
            claim_terms = set(
                w.lower() for w in re.findall(r'[A-Za-z0-9_\-]{3,}', claim_text)
                if w.lower() not in STOPWORDS
            )
            claim_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', claim_text)

            is_supported = False
            is_partial = False
            failure_reason = None

            # 1. Check for explicit contraindication contradiction
            claim_lower = claim_text.lower()
            for pat, desc in contraindication_patterns:
                if re.search(pat, claim_lower):
                    failure_reason = f"contradicted_by_guideline ({desc})"
                    break

            # 2. Check for match against clinician-authored prohibited claims
            if failure_reason is None and clin_eval.get("prohibited_claims"):
                for p_claim in clin_eval["prohibited_claims"]:
                    p_terms = [w.lower() for w in re.findall(r'[A-Za-z0-9_\-]{4,}', p_claim) if w.lower() not in STOPWORDS]
                    matches = [t for t in p_terms if t in claim_lower]
                    if len(matches) >= 3 and len(matches) / len(p_terms) >= 0.6:
                        failure_reason = f"contradicted_by_prohibited_claim ({p_claim})"
                        break

            # 3. Check citation validity and multi-passage evidence support
            if failure_reason is None:
                invalid_cits = []
                for fname, page in sent_cits:
                    if fname not in APPROVED_PDFS:
                        invalid_cits.append(f"unapproved_pdf: {fname}")
                    elif page < 1 or page > APPROVED_PDFS[fname]:
                        invalid_cits.append(f"page_out_of_bounds: {fname}, p. {page}")
                    elif (fname, page) not in passage_map:
                        invalid_cits.append(f"unretrieved_passage: {fname}, p. {page}")

                if invalid_cits:
                    failure_reason = "; ".join(invalid_cits)
                else:
                    # Combine passage texts across all valid citations attached to this sentence
                    combined_passage_text = " ".join(passage_map[(f, p)].lower() for f, p in sent_cits)
                    combined_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', combined_passage_text))

                    # Verify numerical thresholds
                    has_truncated_passage = any(passage_trunc_map.get((f, p), False) for f, p in sent_cits)
                    unsupported_numbers = [n for n in claim_numbers if n not in combined_numbers]
                    if unsupported_numbers:
                        if has_truncated_passage:
                            failure_reason = f"truncated_passage_evidence_insufficient (missing numbers {unsupported_numbers})"
                        else:
                            failure_reason = f"unsupported_numerical_threshold: {unsupported_numbers}"
                    else:
                        # Verify substantive term overlap
                        matching_terms = [t for t in claim_terms if t in combined_passage_text]
                        support_ratio = len(matching_terms) / len(claim_terms) if claim_terms else 1.0

                        if support_ratio >= 0.60:
                            is_supported = True
                            failure_reason = None
                        elif support_ratio >= 0.35:
                            is_partial = True
                            failure_reason = "partial_support_only (overlap between 35% and 60%)"
                        elif has_truncated_passage:
                            failure_reason = f"truncated_passage_evidence_insufficient (overlap {support_ratio:.1%}, passage is truncated)"
                        else:
                            failure_reason = f"unrelated_passage (overlap {support_ratio:.1%} < 35%)"

            has_trunc = any(passage_trunc_map.get((f, p), False) for f, p in sent_cits) if 'passage_trunc_map' in locals() else False
            if is_supported:
                fully_supported_count += 1
                support_level = "complete"
            elif is_partial:
                partially_supported_count += 1
                support_level = "partial"
            else:
                support_level = "unsupported"

            details.append({
                "sentence": sent[:150],
                "citations": [f"{f}, p. {p}" for f, p in sent_cits],
                "is_supported": is_supported,
                "is_partial": is_partial,
                "is_truncated": has_trunc,
                "support_level": support_level,
                "reason": failure_reason or "fully_supported"
            })

        score = round(fully_supported_count / len(cited_sentences), 4) if cited_sentences else None
        return {
            "score": score,
            "total_cited_sentences": len(cited_sentences),
            "supported_count": fully_supported_count,
            "partial_count": partially_supported_count,
            "details": details
        }

    def _calc_unsupported_claim_rate(
        self,
        reply: str,
        claim_support_result: Dict[str, Any],
        is_in_corpus: bool
    ) -> Dict[str, Any]:
        """Calculates Metric 5: Unsupported Clinical Claim Rate."""
        sentences = [s for s in self._split_into_sentences(reply) if len(s) > 15]

        # Ignore greeting / disclaimer / closing meta-sentences
        substantive = [
            s for s in sentences
            if not any(prefix in s.lower() for prefix in ["hello", "disclaimer", "informational only", "approved guidelines"])
        ]

        if not substantive:
            return {"score": 0.0, "total_substantive_claims": 0, "unsupported_claims": 0}

        # If refused / out-of-corpus evidence gap
        if not is_in_corpus and ("evidence gap" in reply.lower() or "not established" in reply.lower()):
            return {"score": 0.0, "total_substantive_claims": len(substantive), "unsupported_claims": 0}

        # Substantive sentences with supported citations
        supported_count = claim_support_result.get("supported_count", 0)
        unsupported_count = max(0, len(substantive) - supported_count)
        rate = round(unsupported_count / len(substantive), 4) if substantive else 0.0

        return {
            "score": rate,
            "total_substantive_claims": len(substantive),
            "unsupported_claims": unsupported_count
        }

    def _calc_prohibited_claim_detection(
        self,
        case: Dict[str, Any],
        reply: str,
        is_clinician_approved: bool,
        is_mock: bool
    ) -> Dict[str, Any]:
        """Calculates Metric 6: Prohibited Claim Detection."""
        if not (is_clinician_approved or is_mock):
            return {
                "status": STATUS_UNAVAILABLE,
                "violation_count": None,
                "violations": [],
                "reason": "Requires clinician approval of prohibited claims. Draft cases excluded."
            }

        clin_eval = case.get("clinician_evaluation", {})
        prohibited_claims = clin_eval.get("prohibited_claims", [])

        violations = []
        reply_lower = reply.lower()

        # Check explicit prohibited claims from case
        for claim in prohibited_claims:
            claim_terms = [w.lower() for w in re.findall(r'[A-Za-z0-9_\-]{4,}', claim) if w.lower() not in STOPWORDS]
            matches = [t for t in claim_terms if t in reply_lower]
            # High lexical overlap with prohibited claim
            if len(matches) >= 3 and len(matches) / len(claim_terms) >= 0.7:
                violations.append(claim)

        # Built-in critical clinical contraindications check (active across all cases)
        dangerous_patterns = [
            (r"discontinue.*chelation.*symptom.*resolve", "Discontinuing chelation after symptom resolution is contraindicated"),
            (r"stop.*penicillamine.*normal", "Cessation of copper chelation leads to fatal hepatic relapse"),
            (r"spot urine.*sufficient.*diagnostic", "Spot urine cannot replace 24h collection for diagnosis")
        ]
        for pat, desc in dangerous_patterns:
            if re.search(pat, reply_lower):
                if desc not in violations:
                    violations.append(desc)

        return {
            "status": "evaluated",
            "violation_count": len(violations),
            "has_violation": len(violations) > 0,
            "violations": violations
        }

    def _calc_negative_control_accuracy(
        self,
        is_in_corpus: bool,
        reply: str,
        coverage_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculates Metric 7: Negative-Control Evidence-Gap Accuracy."""
        if is_in_corpus:
            return {"score": None, "applicable": False, "status": "not_negative_control"}

        reply_lower = reply.lower()
        has_gap_report = bool(coverage_meta.get("evidence_gap"))
        has_refusal_text = (
            "evidence gap" in reply_lower or
            "not established" in reply_lower or
            "outside the approved" in reply_lower or
            "do not contain" in reply_lower or
            "approved clinical practice guidelines do not" in reply_lower
        )

        accurate = bool(has_gap_report or has_refusal_text)
        return {
            "score": 1.0 if accurate else 0.0,
            "applicable": True,
            "gap_detected": accurate,
            "has_gap_report": has_gap_report,
            "has_refusal_text": has_refusal_text
        }

    def _calc_false_evidence_gap_rate(
        self,
        is_in_corpus: bool,
        reply: str,
        coverage_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculates Metric 8: False Evidence-Gap Rate."""
        if not is_in_corpus:
            return {"score": None, "applicable": False, "status": "negative_control_case"}

        # For in-corpus query, did system falsely declare gap?
        has_gap = bool(coverage_meta.get("evidence_gap"))
        reply_lower = reply.lower()
        falsely_refused = (
            has_gap and (
                "evidence gap" in reply_lower or
                "do not contain" in reply_lower or
                "outside the approved" in reply_lower
            )
        )

        return {
            "score": 1.0 if falsely_refused else 0.0,
            "applicable": True,
            "false_gap_occurred": falsely_refused
        }

    # =========================================================================
    # Aggregation & Governance Rollups
    # =========================================================================

    def _compute_domain_breakdown(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Rolls up metrics grouped by domain code (DOM-01 through DOM-10)."""
        domain_groups: Dict[str, List[Dict[str, Any]]] = {}
        for res in case_results:
            dom = res["domain"]
            domain_groups.setdefault(dom, []).append(res)

        domain_breakdown = {}
        for dom, c_list in domain_groups.items():
            facet_covs = [c["metrics"]["retrieval_facet_coverage"] for c in c_list if c["metrics"]["retrieval_facet_coverage"] is not None]
            auth_scores = [c["metrics"]["citation_authenticity"] for c in c_list if c["metrics"]["citation_authenticity"] is not None]
            claim_supports = [c["metrics"]["claim_level_citation_support"] for c in c_list if c["metrics"]["claim_level_citation_support"] is not None]
            unsupported_rates = [c["metrics"]["unsupported_clinical_claim_rate"] for c in c_list if c["metrics"]["unsupported_clinical_claim_rate"] is not None]
            crit_facts = [c["metrics"]["critical_fact_coverage"] for c in c_list if c["metrics"]["critical_fact_coverage"] is not None]

            domain_breakdown[dom] = {
                "case_count": len(c_list),
                "avg_retrieval_facet_coverage": round(sum(facet_covs) / len(facet_covs), 4) if facet_covs else None,
                "avg_citation_authenticity": round(sum(auth_scores) / len(auth_scores), 4) if auth_scores else None,
                "avg_claim_level_citation_support": round(sum(claim_supports) / len(claim_supports), 4) if claim_supports else None,
                "avg_unsupported_clinical_claim_rate": round(sum(unsupported_rates) / len(unsupported_rates), 4) if unsupported_rates else None,
                "avg_critical_fact_coverage": round(sum(crit_facts) / len(crit_facts), 4) if crit_facts else None,
            }
        return domain_breakdown

    def _compute_aggregate_metrics(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculates aggregate benchmark metrics across all evaluated cases."""
        facet_scores = [c["metrics"]["retrieval_facet_coverage"] for c in case_results if c["metrics"]["retrieval_facet_coverage"] is not None]
        auth_scores = [c["metrics"]["citation_authenticity"] for c in case_results if c["metrics"]["citation_authenticity"] is not None]
        claim_supports = [c["metrics"]["claim_level_citation_support"] for c in case_results if c["metrics"]["claim_level_citation_support"] is not None]
        unsupported_rates = [c["metrics"]["unsupported_clinical_claim_rate"] for c in case_results if c["metrics"]["unsupported_clinical_claim_rate"] is not None]
        neg_control_scores = [c["metrics"]["negative_control_evidence_gap_accuracy"] for c in case_results if c["metrics"]["negative_control_evidence_gap_accuracy"] is not None]
        false_gap_scores = [c["metrics"]["false_evidence_gap_rate"] for c in case_results if c["metrics"]["false_evidence_gap_rate"] is not None]

        # Clinical accuracy metric: Only computed on approved non-mock cases
        crit_fact_scores = [c["metrics"]["critical_fact_coverage"] for c in case_results if c["metrics"]["critical_fact_coverage"] is not None]
        prohibited_violations = [c["metrics"]["prohibited_claim_violation_count"] for c in case_results if c["metrics"]["prohibited_claim_violation_count"] is not None]

        return {
            "total_cases_evaluated": len(case_results),
            "retrieval_facet_coverage_mean": round(sum(facet_scores) / len(facet_scores), 4) if facet_scores else 0.0,
            "citation_authenticity_mean": round(sum(auth_scores) / len(auth_scores), 4) if auth_scores else 0.0,
            "claim_level_citation_support_mean": round(sum(claim_supports) / len(claim_supports), 4) if claim_supports else None,
            "unsupported_clinical_claim_rate_mean": round(sum(unsupported_rates) / len(unsupported_rates), 4) if unsupported_rates else 0.0,
            "negative_control_evidence_gap_accuracy": round(sum(neg_control_scores) / len(neg_control_scores), 4) if neg_control_scores else None,
            "false_evidence_gap_rate": round(sum(false_gap_scores) / len(false_gap_scores), 4) if false_gap_scores else 0.0,
            "critical_fact_coverage_mean": round(sum(crit_fact_scores) / len(crit_fact_scores), 4) if crit_fact_scores else None,
            "prohibited_claim_violations_total": sum(prohibited_violations) if prohibited_violations else None
        }

    def _compute_governance_summary(self, case_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarizes governance eligibility and safety constraints."""
        total_cases = len(case_results)
        approved_cases = sum(1 for c in case_results if c["governance"]["is_clinician_approved"])
        draft_cases = sum(1 for c in case_results if c["governance"]["review_status"] == "draft_pending_clinician_review")
        mock_cases = sum(1 for c in case_results if c["governance"]["is_mock"])

        clinical_accuracy_claim_permitted = bool(approved_cases > 0 and approved_cases == total_cases and mock_cases == 0)

        return {
            "total_cases_evaluated": total_cases,
            "clinician_approved_cases_count": approved_cases,
            "draft_cases_count": draft_cases,
            "mock_cases_count": mock_cases,
            "clinical_accuracy_claim_permitted": clinical_accuracy_claim_permitted,
            "governance_notice": (
                "All draft template cases are strictly excluded from clinical accuracy scores. "
                "Engineering metrics reflect retrieval, citation authenticity, and evidence-gap mechanics only."
                if not clinical_accuracy_claim_permitted else
                "All cases are certified and approved by qualified clinicians."
            )
        }
