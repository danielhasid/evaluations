import csv
import json
import os

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase

import sys as _sys
import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

try:  # package import path (preferred for IDE/static analysis)
    from .base import BatchEvaluationResult
    from ..core.config import RAG_METRIC_KEYS
    from ..core.logging_utils import log_stage
except ImportError:  # script execution fallback
    from evaluators.base import BatchEvaluationResult
    from core.config import RAG_METRIC_KEYS
    from core.logging_utils import log_stage


class RagAdapter:
    evaluator_type = "RAG"
    valid_metric_keys = RAG_METRIC_KEYS

    @staticmethod
    def _parse_context_list(raw_value: str) -> list:
        """Parse context list from JSON array, pipe-separated text, or plain text."""
        text = (raw_value or "").strip()
        if not text:
            return []

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                cleaned = []
                for item in parsed:
                    item_text = str(item).strip()
                    if item_text:
                        cleaned.append(item_text)
                if cleaned:
                    return cleaned
        except json.JSONDecodeError:
            pass

        if "|" in text:
            return [doc.strip() for doc in text.split("|") if doc.strip()]

        return [text]

    def load_golden_set_csv(self, filepath: str) -> list:
        qa_pairs = []
        # Resolve path relative to LLM_judge root if it's a relative path
        if not os.path.isabs(filepath):
            filepath = os.path.join(_ROOT, filepath)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"question", "expected_answer", "retrieval_context", "context"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV must contain columns: {required}. Found: {reader.fieldnames}")

            for row in reader:
                question = (row.get("question") or "").strip()
                expected = (row.get("expected_answer") or "").strip()
                if not question and not expected:
                    continue
                if not question or not expected:
                    raise ValueError("Each row must include non-empty question and expected_answer.")

                retrieval_context = self._parse_context_list(row.get("retrieval_context") or "")
                context = self._parse_context_list(row.get("context") or "")
                if not retrieval_context:
                    raise ValueError(f"RAG row for question '{question}' is missing retrieval_context.")
                if not context:
                    raise ValueError(f"RAG row for question '{question}' is missing context.")

                qa_pairs.append({
                    "question": question,
                    "expected_answer": expected,
                    "retrieval_context": retrieval_context,
                    "context": context,
                    "metadata": (row.get("metadata") or "").strip(),
                })

        log_stage(f"[OK] Loaded {len(qa_pairs)} Q&A pairs from {filepath}")
        return qa_pairs

    def run_batch_evaluation(self, qa_pairs: list, selected_metrics: list, truths_limit: int = None) -> BatchEvaluationResult:
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

        gemini_model = GeminiModel(model="gemini-1.5-flash", use_vertexai=True)

        answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=gemini_model, include_reason=True)
        faithfulness = FaithfulnessMetric(
            threshold=0.7,
            model=gemini_model,
            include_reason=True,
            truths_extraction_limit=truths_limit
        )
        contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=gemini_model, include_reason=True)
        contextual_recall = ContextualRecallMetric(threshold=0.7, model=gemini_model, include_reason=True)
        contextual_relevancy = ContextualRelevancyMetric(threshold=0.7, model=gemini_model)

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
