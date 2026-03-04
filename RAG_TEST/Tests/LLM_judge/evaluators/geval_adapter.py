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

from evaluators.base import BatchEvaluationResult
from core.config import GEVAL_METRIC_KEYS
from core.logging_utils import log_stage


class GevalAdapter:
    evaluator_type = "GEval"
    valid_metric_keys = GEVAL_METRIC_KEYS

    def load_golden_set_csv(self, filepath: str) -> list:
        qa_pairs = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"question", "expected_answer"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"CSV must contain columns: {required}. Found: {reader.fieldnames}")

            for row in reader:
                raw_context = (row.get("context") or "").strip()
                try:
                    context = json.loads(raw_context) if raw_context else []
                except json.JSONDecodeError:
                    context = [raw_context] if raw_context else []

                qa_pairs.append({
                    "question": row["question"].strip(),
                    "expected_answer": row["expected_answer"].strip(),
                    "context": context,
                    "metadata": (row.get("metadata") or "").strip(),
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
