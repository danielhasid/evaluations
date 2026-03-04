import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


GEVAL_METRIC_KEYS = ["fluency", "relevance", "correctness", "hallucination"]
RAG_METRIC_KEYS = [
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
]


@dataclass
class RunConfig:
    evaluator_type: str
    input_csv: str = "golden_set.csv"
    output_json: Optional[str] = None
    output_dir: str = "results"
    metrics: List[str] = field(default_factory=list)
    generate_summary: bool = True
    generate_dashboard: bool = True
    dashboard_filename: str = "confident_ai_dashboard.html"


def make_run_json_path(output_dir: str = ".", prefix: str = "evaluation_results") -> str:
    """Generate a unique timestamped JSON filename for one evaluation run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{prefix}_{ts}.json")
