"""Backward-compatible RAG evaluator module built on top of shared adapters."""

try:
    from .core.answering import OpenAIAnswerGenerator, generate_answers_for_dataset as shared_generate_answers
    from .core.config import RAG_METRIC_KEYS
    from .core.results import (
        display_results as shared_display_results,
        save_initial_results as shared_save_initial_results,
        update_results_with_metrics as shared_update_results_with_metrics,
    )
    from .evaluators.rag_adapter import RagAdapter
except ImportError:  # pragma: no cover - script execution fallback
    from core.answering import OpenAIAnswerGenerator, generate_answers_for_dataset as shared_generate_answers
    from core.config import RAG_METRIC_KEYS
    from core.results import (
        display_results as shared_display_results,
        save_initial_results as shared_save_initial_results,
        update_results_with_metrics as shared_update_results_with_metrics,
    )
    from evaluators.rag_adapter import RagAdapter


def load_golden_set_csv(filepath):
    return RagAdapter().load_golden_set_csv(filepath)


def generate_answers_for_dataset(qa_pairs):
    return shared_generate_answers(qa_pairs, answer_generator=OpenAIAnswerGenerator())


def run_batch_evaluation(qa_pairs, selected_metrics: list):
    batch = RagAdapter().run_batch_evaluation(qa_pairs, selected_metrics)
    return batch.test_cases, batch.metrics


def save_initial_results(qa_pairs, output_filepath="evaluation_results.json"):
    shared_save_initial_results(qa_pairs, evaluator_type="RAG", output_filepath=output_filepath)


def update_results_with_metrics(
    qa_pairs,
    test_cases,
    metrics_list,
    output_filepath="evaluation_results.json",
):
    del qa_pairs
    shared_update_results_with_metrics(
        test_cases=test_cases,
        metrics_list=metrics_list,
        output_filepath=output_filepath,
        evaluation_result=None,
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
    test_cases, metrics = run_batch_evaluation(
        qa_pairs,
        selected_metrics=RAG_METRIC_KEYS,
    )
    update_results_with_metrics(qa_pairs, test_cases, metrics, output_json)
    display_results(qa_pairs, test_cases, output_json)


if __name__ == "__main__":
    main()
