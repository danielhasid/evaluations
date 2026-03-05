import csv
import json
import os

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

import sys as _sys
import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

try:  # package import path (preferred for IDE/static analysis)
    from .base import BatchEvaluationResult
    from ..core.config import GEVAL_METRIC_KEYS
    from ..core.logging_utils import log_stage
except ImportError:  # script execution fallback
    from evaluators.base import BatchEvaluationResult
    from core.config import GEVAL_METRIC_KEYS
    from core.logging_utils import log_stage


class GevalAdapter:
    evaluator_type = "GEval"
    valid_metric_keys = GEVAL_METRIC_KEYS

    @staticmethod
    def _parse_context_field(raw_value: str) -> list:
        """Parse context from JSON list, pipe-separated text, or plain text."""
        text = (raw_value or "").strip()
        if not text:
            return []

        # JSON list support: ["ctx1", "ctx2"]
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

        # Pipe-separated support: ctx1|ctx2|ctx3
        if "|" in text:
            return [part.strip() for part in text.split("|") if part.strip()]

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
            required = {"question", "expected_answer"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV must contain columns: {required}. Found: {reader.fieldnames}")

            for row in reader:
                question = (row.get("question") or "").strip()
                expected = (row.get("expected_answer") or "").strip()
                if not question and not expected:
                    # Ignore fully empty lines in CSV.
                    continue
                if not question or not expected:
                    raise ValueError("Each row must include non-empty question and expected_answer.")

                context = self._parse_context_field(row.get("context") or "")

                qa_pairs.append({
                    "question": question,
                    "expected_answer": expected,
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
                    context=qa_pair["context"] if qa_pair.get("context") else None,
                )
            )

        fluency = GEval(
            name="Fluency",
            criteria="Is the output grammatically correct and easy to understand?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
        )
        relevance = GEval(
            name="Relevance",
            criteria="Does the output appropriately and directly answer the input question?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            threshold=0.5,
        )
        correctness = GEval(
            name="Correctness",
            criteria="Is the actual output factually correct and consistent with the expected answer?",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK",
            ],
            rubric=[
                Rubric(score_range=(0, 2), expected_outcome="Factually incorrect."),
                Rubric(score_range=(3, 6), expected_outcome="Mostly correct."),
                Rubric(score_range=(7, 9), expected_outcome="Correct but missing minor details."),
                Rubric(score_range=(10, 10), expected_outcome="100% correct."),
            ],
            evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT],
            threshold=0.5,
        )
        hallucination = GEval(
            name="Hallucination",
            criteria="Does the output match the expected output without adding false claims?",
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=0.5,
        )

        all_metrics = {
            "fluency": fluency,
            "relevance": relevance,
            "correctness": correctness,
            "hallucination": hallucination,
        }
        metrics = [all_metrics[k] for k in selected_metrics]
        log_stage(f"[>>] Running GEval metrics: {selected_metrics}")
        evaluation_result = evaluate(test_cases=test_cases, metrics=metrics)
        return BatchEvaluationResult(test_cases=test_cases, metrics=metrics, evaluation_result=evaluation_result)
