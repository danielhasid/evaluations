import json
import os
import sys
from dataclasses import dataclass


THIS_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.config import RunConfig
from core.pipeline import run_pipeline
from evaluators.base import BatchEvaluationResult


@dataclass
class DummyTestCase:
    input: str
    actual_output: str
    expected_output: str


class FakeMetric:
    def __init__(self, name: str, threshold: float = 0.5):
        self.name = name
        self.threshold = threshold
        self.score = None
        self.reason = ""

    def measure(self, test_case):
        self.score = 0.9 if test_case.actual_output else 0.0
        self.reason = "Dummy smoke metric evaluation."


class DummyAdapter:
    def __init__(self, evaluator_type: str, valid_metric_keys: list):
        self.evaluator_type = evaluator_type
        self.valid_metric_keys = valid_metric_keys

    def load_golden_set_csv(self, filepath):
        return [{
            "question": "What is Python?",
            "expected_answer": "A programming language.",
            "metadata": "smoke",
        }]

    def run_batch_evaluation(self, qa_pairs, selected_metrics):
        test_cases = [
            DummyTestCase(
                input=qa_pairs[0]["question"],
                actual_output=qa_pairs[0]["generated_answer"],
                expected_output=qa_pairs[0]["expected_answer"],
            )
        ]
        metrics = [FakeMetric(name=selected_metrics[0], threshold=0.5)]
        return BatchEvaluationResult(test_cases=test_cases, metrics=metrics, evaluation_result=None)


class FakeAnswerGenerator:
    def __init__(self):
        pass

    def generate(self, messages):
        return "Python is a programming language."


def _patch_answer_generator(monkeypatch):
    monkeypatch.setattr("core.pipeline.OpenAIAnswerGenerator", FakeAnswerGenerator)


def test_geval_smoke_pipeline(tmp_path, monkeypatch):
    _patch_answer_generator(monkeypatch)
    output_json = tmp_path / "geval_results.json"
    cfg = RunConfig(
        evaluator_type="GEval",
        input_csv="unused.csv",
        output_json=str(output_json),
        metrics=["fluency"],
    )
    adapter = DummyAdapter(evaluator_type="GEval", valid_metric_keys=["fluency"])

    qa_pairs, test_cases, metrics, run_json = run_pipeline(cfg, adapter)

    assert len(qa_pairs) == 1
    assert len(test_cases) == 1
    assert len(metrics) == 1
    assert run_json == str(output_json)

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["evaluator_type"] == "GEval"
    assert data["results"][0]["status"] == "pass"


def test_rag_smoke_pipeline(tmp_path, monkeypatch):
    _patch_answer_generator(monkeypatch)
    output_json = tmp_path / "rag_results.json"
    cfg = RunConfig(
        evaluator_type="RAG",
        input_csv="unused.csv",
        output_json=str(output_json),
        metrics=["answer_relevancy"],
    )
    adapter = DummyAdapter(evaluator_type="RAG", valid_metric_keys=["answer_relevancy"])

    qa_pairs, test_cases, metrics, run_json = run_pipeline(cfg, adapter)

    assert len(qa_pairs) == 1
    assert len(test_cases) == 1
    assert len(metrics) == 1
    assert run_json == str(output_json)

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["evaluator_type"] == "RAG"
    assert data["results"][0]["status"] == "pass"
