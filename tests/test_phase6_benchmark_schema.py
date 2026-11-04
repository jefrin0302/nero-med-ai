"""
test_phase6_benchmark_schema.py

Verification Test Suite for Phase 6 Deliverable 1:
Benchmark Schema, Template Dataset, Draft-07 Validator, and Clinician Review Guide.

Verifies:
1. benchmark_schema.json is well-formed JSON and contains required meta-schema keys.
2. wilson_benchmark_v1.json is well-formed JSON and validates strictly against benchmark_schema.json using Draft-07 validation.
3. Draft-07 validator strictly catches schema violations (missing properties, invalid enums, pattern mismatches, invalid types).
4. All 10 clinical domains (DOM-01 through DOM-10) are represented.
5. All initial template cases carry review_status='draft_pending_clinician_review' and is_clinician_approved=False.
6. In-corpus flags are correctly assigned (True for DOM-01..09, False for DOM-10 negative control).
7. Root approved_guideline_documents matches rag_system.APPROVED_PDFS.
8. CLINICIAN_REVIEW_GUIDE.md exists and contains required instructions and safety boundaries.
"""

import os
import sys
import json
import re
import unittest

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_system import APPROVED_PDFS
from clinical_benchmarks.draft7_validator import (
    validate_draft7,
    Draft7ValidationError,
    UnsupportedSchemaKeywordError,
    check_schema_conformance
)

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clinical_benchmarks")
SCHEMA_PATH = os.path.join(BENCHMARK_DIR, "benchmark_schema.json")
DATASET_PATH = os.path.join(BENCHMARK_DIR, "wilson_benchmark_v1.json")
GUIDE_PATH = os.path.join(BENCHMARK_DIR, "CLINICIAN_REVIEW_GUIDE.md")


class TestPhase6BenchmarkSchemaAndIntegrity(unittest.TestCase):
    def test_schema_file_exists_and_is_valid_json(self):
        """Verifies benchmark_schema.json exists and parses as valid JSON."""
        self.assertTrue(os.path.exists(SCHEMA_PATH), f"Schema file not found at {SCHEMA_PATH}")
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.assertIsInstance(schema, dict)
        self.assertIn("properties", schema)
        self.assertIn("required", schema)
        self.assertEqual(schema.get("type"), "object")

    def test_dataset_file_exists_and_is_valid_json(self):
        """Verifies wilson_benchmark_v1.json exists and parses as valid JSON."""
        self.assertTrue(os.path.exists(DATASET_PATH), f"Dataset file not found at {DATASET_PATH}")
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("benchmark_version", data)
        self.assertIn("cases", data)
        self.assertIsInstance(data["cases"], list)

    def test_dataset_validates_against_draft7_schema(self):
        """Verifies wilson_benchmark_v1.json strictly passes Draft-07 schema validation."""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Full recursive Draft-07 schema validation
        try:
            validate_draft7(data, schema)
        except Draft7ValidationError as e:
            self.fail(f"Dataset failed Draft-07 schema validation: {e}")

        # Check approved guideline documents match APPROVED_PDFS
        self.assertEqual(
            sorted(data["approved_guideline_documents"]),
            sorted(APPROVED_PDFS),
            "Benchmark approved_guideline_documents must match rag_system.APPROVED_PDFS"
        )

    def test_draft7_validator_catches_violations(self):
        """Verifies Draft-07 validator rejects invalid data structures and values."""
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)

        # 1. Missing required root field
        invalid_data_1 = {"benchmark_version": "1.0.0"}
        with self.assertRaises(Draft7ValidationError) as cm1:
            validate_draft7(invalid_data_1, schema)
        self.assertEqual(cm1.exception.rule, "required")

        # 2. Invalid case_id regex pattern
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            valid_data = json.load(f)
        bad_id_data = json.loads(json.dumps(valid_data))
        bad_id_data["cases"][0]["case_id"] = "INVALID-ID"
        with self.assertRaises(Draft7ValidationError) as cm2:
            validate_draft7(bad_id_data, schema)
        self.assertEqual(cm2.exception.rule, "pattern")

        # 3. Invalid domain enum
        bad_domain_data = json.loads(json.dumps(valid_data))
        bad_domain_data["cases"][0]["domain"] = "DOM-99"
        with self.assertRaises(Draft7ValidationError) as cm3:
            validate_draft7(bad_domain_data, schema)
        self.assertEqual(cm3.exception.rule, "enum")

        # 4. Invalid date format
        bad_date_data = json.loads(json.dumps(valid_data))
        bad_date_data["last_updated"] = "November 3, 2026"
        with self.assertRaises(Draft7ValidationError) as cm4:
            validate_draft7(bad_date_data, schema)
        self.assertEqual(cm4.exception.rule, "format:date")

    def test_all_ten_clinical_domains_represented(self):
        """Verifies all ten clinical domains (DOM-01 through DOM-10) are present in the dataset."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        expected_domains = [f"DOM-{i:02d}" for i in range(1, 11)]
        present_domains = [c["domain"] for c in data["cases"]]

        for dom in expected_domains:
            self.assertIn(dom, present_domains, f"Domain {dom} missing from benchmark dataset")

        self.assertEqual(len(data["cases"]), 10, "Expected exactly 10 initial representative domain cases")

    def test_draft_status_and_safety_flags_enforced(self):
        """Verifies that all template cases are flagged as draft pending clinician review."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for case in data["cases"]:
            clin_eval = case["clinician_evaluation"]
            self.assertEqual(
                clin_eval["review_status"],
                "draft_pending_clinician_review",
                f"Case {case['case_id']} review_status must be 'draft_pending_clinician_review'"
            )
            self.assertFalse(
                clin_eval["is_clinician_approved"],
                f"Case {case['case_id']} is_clinician_approved must be False in template"
            )
            self.assertIsNone(clin_eval["reviewer_name"])
            self.assertIsNone(clin_eval["reviewed_at"])
            self.assertIsNone(clin_eval["clinician_reference_answer"])
            self.assertEqual(clin_eval["critical_facts"], [])
            self.assertEqual(clin_eval["acceptable_paraphrases"], [])
            self.assertEqual(clin_eval["prohibited_claims"], [])
            self.assertEqual(clin_eval["mandatory_source_page_references"], [])

    def test_negative_control_corpus_flagging(self):
        """Verifies DOM-10 negative control has is_in_corpus=False and DOM-01..09 have is_in_corpus=True."""
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for case in data["cases"]:
            if case["domain"] == "DOM-10":
                self.assertFalse(case["is_in_corpus"], "DOM-10 out-of-corpus query must have is_in_corpus=False")
                self.assertEqual(case["question_type"], "negative_control")
            else:
                self.assertTrue(case["is_in_corpus"], f"In-corpus case {case['case_id']} must have is_in_corpus=True")

    def test_clinician_review_guide_exists_and_covers_required_sections(self):
        """Verifies CLINICIAN_REVIEW_GUIDE.md exists and contains necessary instructional sections."""
        self.assertTrue(os.path.exists(GUIDE_PATH), f"Review guide not found at {GUIDE_PATH}")
        with open(GUIDE_PATH, "r", encoding="utf-8") as f:
            guide_text = f.read()

        required_phrases = [
            "Clinical Decision Support",
            "AASLD WD Guidelines.pdf",
            "EASL-ERN Clinical Practice Guidelines on Wilson's disease.pdf",
            "NIH Wilson Disease.pdf",
            "clinical practice guidelines in WD.pdf",
            "draft_pending_clinician_review",
            "clinician_approved",
            "critical_facts",
            "prohibited_claims",
            "mandatory_source_page_references"
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, guide_text, f"Guide missing required phrase/section: {phrase}")


    def test_draft7_subset_validator_catches_unsupported_keywords_and_malformed_schemas(self):
        """Verifies subset validator raises UnsupportedSchemaKeywordError and catches malformed schemas."""
        # 1. Unsupported keyword: $ref
        schema_with_ref = {
            "type": "object",
            "$ref": "#/definitions/Case"
        }
        with self.assertRaises(UnsupportedSchemaKeywordError) as cm_ref:
            validate_draft7({"a": 1}, schema_with_ref)
        self.assertEqual(cm_ref.exception.keyword, "$ref")

        # 2. Unsupported keyword: allOf
        schema_with_allof = {
            "allOf": [{"type": "object"}]
        }
        with self.assertRaises(UnsupportedSchemaKeywordError) as cm_allof:
            validate_draft7({}, schema_with_allof)
        self.assertEqual(cm_allof.exception.keyword, "allOf")

        # 3. Unsupported keyword: minimum
        schema_with_min = {
            "type": "integer",
            "minimum": 0
        }
        with self.assertRaises(UnsupportedSchemaKeywordError) as cm_min:
            validate_draft7(5, schema_with_min)
        self.assertEqual(cm_min.exception.keyword, "minimum")

        # 4. Malformed schema: invalid type string
        schema_bad_type = {
            "type": "nonexistent_type"
        }
        with self.assertRaises(Draft7ValidationError) as cm_bad_type:
            validate_draft7("hello", schema_bad_type)
        self.assertEqual(cm_bad_type.exception.rule, "malformed_type")

        # 5. Malformed schema: required is not a list
        schema_bad_req = {
            "type": "object",
            "required": "not_a_list"
        }
        with self.assertRaises(Draft7ValidationError) as cm_bad_req:
            validate_draft7({}, schema_bad_req)
        self.assertEqual(cm_bad_req.exception.rule, "malformed_required")

        # 6. Malformed schema: schema is not a dict
        with self.assertRaises(Draft7ValidationError) as cm_not_dict:
            validate_draft7({}, "not_a_dict")
        self.assertEqual(cm_not_dict.exception.rule, "malformed_schema")


if __name__ == "__main__":
    unittest.main()
