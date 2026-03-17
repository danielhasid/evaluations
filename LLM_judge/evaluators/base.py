from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Tuple


@dataclass
class BatchEvaluationResult:
    test_cases: List[Any]
    metrics: List[Any]
    evaluation_result: Optional[Any] = None


class EvaluatorAdapter(Protocol):
    evaluator_type: str
    valid_metric_keys: List[str]

    def load_golden_set_csv(self, filepath: str) -> list:
        ...

    def run_batch_evaluation(self, qa_pairs: list, selected_metrics: list, truths_limit: int = None) -> BatchEvaluationResult:
        ...
