import os
from typing import Tuple

import sys as _sys
import os as _os

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

try:  # package import path (preferred for IDE/static analysis)
    from .answering import OpenAIAnswerGenerator
    from .dataset_answering import generate_answers_for_dataset
    from .config import RunConfig, make_run_json_path
    from .logging_utils import log_stage
    from .results import display_results, save_initial_results, update_results_with_metrics
    from ..evaluators.base import BatchEvaluationResult, EvaluatorAdapter
except ImportError:  # script execution fallback
    from core.answering import OpenAIAnswerGenerator
    from core.dataset_answering import generate_answers_for_dataset
    from core.config import RunConfig, make_run_json_path
    from core.logging_utils import log_stage
    from core.results import display_results, save_initial_results, update_results_with_metrics
    from evaluators.base import BatchEvaluationResult, EvaluatorAdapter


def run_pipeline(config: RunConfig, adapter: EvaluatorAdapter) -> Tuple[list, list, list, str]:
    if not config.metrics:
        raise ValueError(
            f"You must specify which {adapter.evaluator_type} metrics to run. "
            f"Valid options: {adapter.valid_metric_keys}"
        )

    invalid = [key for key in config.metrics if key not in adapter.valid_metric_keys]
    if invalid:
        raise ValueError(
            f"Unknown {adapter.evaluator_type} metrics: {invalid}. "
            f"Valid options: {adapter.valid_metric_keys}"
        )

    if config.output_json is None:
        os.makedirs(config.output_dir, exist_ok=True)
        output_json = make_run_json_path(
            config.output_dir,
            prefix="evaluation_results",
            evaluator_type=adapter.evaluator_type,
        )
    else:
        output_json = config.output_json
        config.output_dir = os.path.dirname(os.path.abspath(output_json)) or "."

    log_stage(">> Loading golden set Q&A pairs from CSV...")
    qa_pairs = adapter.load_golden_set_csv(config.input_csv)

    answer_generator = OpenAIAnswerGenerator()
    qa_pairs = generate_answers_for_dataset(qa_pairs, answer_generator=answer_generator)

    save_initial_results(qa_pairs, evaluator_type=adapter.evaluator_type, output_filepath=output_json)

    batch: BatchEvaluationResult = adapter.run_batch_evaluation(
        qa_pairs, 
        selected_metrics=config.metrics,
        truths_limit=config.truths_extraction_limit
    )
    update_results_with_metrics(
        test_cases=batch.test_cases,
        metrics_list=batch.metrics,
        output_filepath=output_json,
        evaluation_result=batch.evaluation_result,
    )

    display_results(qa_pairs, output_filepath=output_json)
    return qa_pairs, batch.test_cases, batch.metrics, output_json
