import csv
import os

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

import sys as _sys
import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from evaluators.base import BatchEvaluationResult
from core.config import RAG_METRIC_KEYS
from core.logging_utils import log_stage


class RagAdapter:
    evaluator_type = "RAG"
    valid_metric_keys = RAG_METRIC_KEYS

    def load_golden_set_csv(self, filepath: str) -> list:
        qa_pairs = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"question", "expected_answer", "retrieval_context", "context"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV must contain columns: {required}. Found: {reader.fieldnames}")

            for row in reader:
                retrieval_context = [doc.strip() for doc in row["retrieval_context"].split("|") if doc.strip()]
                context = [doc.strip() for doc in row["context"].split("|") if doc.strip()]
                qa_pairs.append({
                    "question": row["question"].strip(),
                    "expected_answer": row["expected_answer"].strip(),
                    "retrieval_context": retrieval_context,
                    "context": context,
                    "metadata": row.get("metadata", "").strip() if "metadata" in row else "",
                })

        log_stage(f"[OK] Loaded {len(qa_pairs)} Q&A pairs from {filepath}")
        return qa_pairs

    def run_batch_evaluation(self, qa_pairs: list, selected_metrics: list) -> BatchEvaluationResult:
        test_cases = []
        for qa_pair in qa_pairs:
            test_cases.append(
                LLMTestCase(
                    input=qa_pair["question"],
                    actual_output=qa_pair["generated_answer"],
                    expected_output=qa_pair["expected_answer"],
                    retrieval_context=qa_pair["retrieval_context"],
                    context=qa_pair["context"],
                )
            )

        answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o", include_reason=True)
        faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o", include_reason=True)
        contextual_precision = ContextualPrecisionMetric(threshold=0.7, model="gpt-4o", include_reason=True)
        contextual_recall = ContextualRecallMetric(threshold=0.7, model="gpt-4o", include_reason=True)
        contextual_relevancy = ContextualRelevancyMetric(threshold=0.7, model="gpt-4o")

        all_metrics = {
            "answer_relevancy": answer_relevancy,
            "faithfulness": faithfulness,
            "contextual_precision": contextual_precision,
            "contextual_recall": contextual_recall,
            "contextual_relevancy": contextual_relevancy,
        }

        metrics = [all_metrics[k] for k in selected_metrics]
        log_stage(f"[>>] Running RAG metrics: {selected_metrics}")
        return BatchEvaluationResult(test_cases=test_cases, metrics=metrics, evaluation_result=None)
