"""Backward-compatible GEval module built on top of shared adapters."""

try:
    from .core.answering import OpenAIAnswerGenerator, generate_answers_for_dataset as shared_generate_answers
    from .core.config import GEVAL_METRIC_KEYS
    from .core.results import (
        display_results as shared_display_results,
        save_initial_results as shared_save_initial_results,
        update_results_with_metrics as shared_update_results_with_metrics,
    )
    from .evaluators.geval_adapter import GevalAdapter
except ImportError:  # pragma: no cover - script execution fallback
    from core.answering import OpenAIAnswerGenerator, generate_answers_for_dataset as shared_generate_answers
    from core.config import GEVAL_METRIC_KEYS
    from core.results import (
        display_results as shared_display_results,
        save_initial_results as shared_save_initial_results,
        update_results_with_metrics as shared_update_results_with_metrics,
    )
    from evaluators.geval_adapter import GevalAdapter


def load_golden_set_csv(filepath):
    return GevalAdapter().load_golden_set_csv(filepath)


def generate_answers_for_dataset(qa_pairs):
    return shared_generate_answers(qa_pairs, answer_generator=OpenAIAnswerGenerator())


def run_batch_evaluation(qa_pairs, selected_metrics: list):
    batch = GevalAdapter().run_batch_evaluation(qa_pairs, selected_metrics)
    return batch.test_cases, batch.metrics, batch.evaluation_result


def save_initial_results(qa_pairs, output_filepath="evaluation_results.json"):
    shared_save_initial_results(qa_pairs, evaluator_type="GEval", output_filepath=output_filepath)


def update_results_with_metrics(
    qa_pairs,
    test_cases,
    metrics_list,
    evaluation_result=None,
    output_filepath="evaluation_results.json",
):
    del qa_pairs
    shared_update_results_with_metrics(
        test_cases=test_cases,
        metrics_list=metrics_list,
        output_filepath=output_filepath,
        evaluation_result=evaluation_result,
    )


def display_results(qa_pairs, test_cases=None, output_filepath="evaluation_results.json"):
    del test_cases
    shared_display_results(qa_pairs, output_filepath=output_filepath)


def main():
    input_csv = "golden_set.csv"
    output_json = "evaluation_results.json"
    qa_pairs = load_golden_set_csv(input_csv)
    qa_pairs = generate_answers_for_dataset(qa_pairs)
    save_initial_results(qa_pairs, output_json)
    test_cases, metrics, evaluation_result = run_batch_evaluation(
        qa_pairs,
        selected_metrics=GEVAL_METRIC_KEYS,
    )
    update_results_with_metrics(qa_pairs, test_cases, metrics, evaluation_result, output_json)
    display_results(qa_pairs, test_cases, output_json)


if __name__ == "__main__":
    main()
