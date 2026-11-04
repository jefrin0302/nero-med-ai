"""
clinical_benchmarks/run_evaluation.py

Executable runner for NeuroMed AI Clinical RAG Benchmark Evaluation.
Loads the benchmark dataset, executes evaluation across all cases using MedicalRAGSystem,
computes per-domain and aggregate metrics, verifies governance constraints,
and outputs a machine-readable JSON report.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag_system import MedicalRAGSystem
from clinical_benchmarks.evaluator import ClinicalBenchmarkEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_benchmark")

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(BENCHMARK_DIR, "wilson_benchmark_v1.json")
DEFAULT_OUTPUT = os.path.join(BENCHMARK_DIR, "reports", "evaluation_report_v1.json")


def run_benchmark_suite(
    dataset_path: str = DEFAULT_DATASET,
    output_path: str = DEFAULT_OUTPUT
) -> dict:
    logger.info("Initializing MedicalRAGSystem...")
    rag = MedicalRAGSystem()

    logger.info("Initializing ClinicalBenchmarkEvaluator...")
    evaluator = ClinicalBenchmarkEvaluator(rag_system=rag)

    logger.info(f"Loading benchmark dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    logger.info(f"Executing evaluation across {len(benchmark_data.get('cases', []))} cases...")
    report = evaluator.evaluate_benchmark(benchmark_data)

    logger.info(f"Saving evaluation report to: {output_path}")
    evaluator.save_report(report, output_path)

    logger.info("Evaluation complete.")
    logger.info("--- AGGREGATE SUMMARY ---")
    for k, v in report.get("aggregate_metrics", {}).items():
        logger.info(f"  {k}: {v}")

    logger.info("--- GOVERNANCE SUMMARY ---")
    for k, v in report.get("governance_summary", {}).items():
        logger.info(f"  {k}: {v}")

    return report


if __name__ == "__main__":
    run_benchmark_suite()

