import os
import re
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
    input_csv: str = "datasets/llm/LLM_goldenset.csv"
    output_json: Optional[str] = None
    output_dir: str = "results"
    metrics: List[str] = field(default_factory=list)
    generate_summary: bool = True
    generate_dashboard: bool = True
    dashboard_filename: str = "confident_ai_dashboard.html"
    truths_extraction_limit: Optional[int] = None


def _sanitize_filename_token(token: str) -> str:
    token = str(token or "").strip()
    if not token:
        return "Unknown"
    # Keep names filesystem-safe while preserving readable evaluator labels.
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", token)
    return safe.strip("_") or "Unknown"


def make_run_json_path(
    output_dir: str = ".",
    prefix: str = "evaluation_results",
    evaluator_type: Optional[str] = None,
) -> str:
    """Generate a unique timestamped JSON filename for one evaluation run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluator_token = _sanitize_filename_token(evaluator_type) if evaluator_type is not None else None
    if evaluator_token:
        filename = f"{evaluator_token}_{prefix}_{ts}.json"
    else:
        filename = f"{prefix}_{ts}.json"
    return os.path.join(output_dir, filename)
