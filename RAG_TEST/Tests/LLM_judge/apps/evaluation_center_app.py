import sys as _sys
import os as _os

# Ensure the LLM_judge root directory is on sys.path so all local packages
# (core/, evaluators/, etc.) resolve correctly regardless of how this module
# is invoked (script, package import, or pytest).
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from core.config import GEVAL_METRIC_KEYS, RAG_METRIC_KEYS, RunConfig
from core.env import load_environment, require_openai_api_key
from core.pipeline import run_pipeline
from core.logging_utils import log_stage
from evaluators.geval_adapter import GevalAdapter
from evaluators.rag_adapter import RagAdapter
from analyze_eval import generate_evaluation_summary, save_summary_to_json


class EvaluationCenterFacade:
    def __init__(self):
        load_environment()
        self.api_key = require_openai_api_key()

    def run_geval_evaluation(
        self,
        input_csv: str = "golden_set.csv",
        output_json: str = None,
        output_dir: str = "results",
        metrics=None,
    ):
        selected = metrics or GEVAL_METRIC_KEYS.copy()
        if metrics is None:
            log_stage(f"[INFO] No GEval metrics specified; using defaults: {selected}")

        config = RunConfig(
            evaluator_type="GEval",
            input_csv=input_csv,
            output_json=output_json,
            output_dir=output_dir,
            metrics=selected,
        )
        qa_pairs, test_cases, metrics_list, run_json = run_pipeline(config=config, adapter=GevalAdapter())
        self._post_process(run_json)
        return qa_pairs, test_cases, metrics_list

    def run_rag_evaluation(
        self,
        input_csv: str = "golden_set.csv",
        output_json: str = None,
        output_dir: str = "results",
        metrics=None,
    ):
        if not metrics:
            raise ValueError(
                "You must specify which RAG metrics to run. "
                f"Valid options: {RAG_METRIC_KEYS}"
            )

        config = RunConfig(
            evaluator_type="RAG",
            input_csv=input_csv,
            output_json=output_json,
            output_dir=output_dir,
            metrics=metrics,
        )
        qa_pairs, test_cases, metrics_list, run_json = run_pipeline(config=config, adapter=RagAdapter())
        self._post_process(run_json)
        return qa_pairs, test_cases, metrics_list

    def _post_process(self, output_json: str) -> None:
        log_stage("\n[OK] Evaluation complete!")
        log_stage("\n[LLM] Generating analysis summary with GPT-4o...")
        summary = generate_evaluation_summary(output_json, self.api_key)
        save_summary_to_json(output_json, summary)
        log_stage("\n--- Analysis Summary ---")
        log_stage(summary)
