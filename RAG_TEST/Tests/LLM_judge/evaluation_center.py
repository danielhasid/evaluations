import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Support both key names used across DeepEval/Confident AI setups.
confident_key = os.getenv("CONFIDENT_API_KEY") or os.getenv("DEEPEVAL_API_KEY")
if confident_key:
    os.environ.setdefault("CONFIDENT_API_KEY", confident_key)
    os.environ.setdefault("DEEPEVAL_API_KEY", confident_key)

from golden_set_geval_metrix import (
    load_golden_set_csv as geval_load_csv,
    generate_answers_for_dataset as geval_generate_answers,
    save_initial_results as geval_save_initial,
    run_batch_evaluation as geval_run_batch,
    update_results_with_metrics as geval_update_metrics,
    display_results as geval_display,
)
from golden_set_evaluator_rag_metrix import (
    load_golden_set_csv as rag_load_csv,
    generate_answers_for_dataset as rag_generate_answers,
    save_initial_results as rag_save_initial,
    run_batch_evaluation as rag_run_batch,
    update_results_with_metrics as rag_update_metrics,
    display_results as rag_display,
)
from analyze_eval import generate_evaluation_summary, save_summary_to_json
from generate_dashboard import create_dashboard

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")


def _make_run_json_path(output_dir: str = ".", prefix: str = "evaluation_results") -> str:
    """Generate a unique timestamped JSON filename for a single evaluation run."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(output_dir, f"{prefix}_{ts}.json")


class EvaluationCenter:
    def __init__(self):
        self.evaluation_metrics = []

    def add_evaluation_metric(self, metric):
        self.evaluation_metrics.append(metric)

    def evaluate(self, input_text, expected_output, actual_output):
        for metric in self.evaluation_metrics:
            metric.evaluate(input_text, expected_output, actual_output)

    def get_evaluation_results(self):
        return self.evaluation_metrics

    @staticmethod
    def run_geval_evaluation(
        input_csv="golden_set.csv",
        output_json=None,
        output_dir=".",
        metrics=None,
    ):
        """
        Run the full GEval golden set evaluation pipeline.

        Each run writes a new timestamped JSON file so historical results are
        preserved. The dashboard is regenerated from all files in output_dir.

        Args:
            input_csv: Path to the golden set CSV file.
            output_json: Explicit path for the result JSON. When None (default),
                         a timestamped filename is auto-generated inside output_dir.
            output_dir: Directory where per-run JSON files accumulate (default ".").
                        Also used as the source for the combined dashboard.
            metrics: Optional list of GEval metric keys to run.
                     Valid keys: fluency, relevance, correctness, hallucination

        Examples:
            # Run all GEval metrics
            center.run_geval_evaluation(metrics=["fluency", "relevance", "correctness", "hallucination"])

            # Run a subset
            center.run_geval_evaluation(metrics=["fluency", "correctness"])
        """
        if metrics is None:
            metrics = ["fluency", "relevance", "correctness", "hallucination"]
            print(f"ℹ️ No GEval metrics specified; using defaults: {metrics}")
        if not metrics:
            raise ValueError(
                "You must specify which GEval metrics to run. "
                "Valid options: fluency, relevance, correctness, hallucination"
            )

        if output_json is None:
            os.makedirs(output_dir, exist_ok=True)
            output_json = _make_run_json_path(output_dir, prefix="evaluation_results")
        else:
            output_dir = os.path.dirname(os.path.abspath(output_json)) or "."

        print("🔹 Loading golden set Q&A pairs from CSV...")
        qa_pairs = geval_load_csv(input_csv)

        qa_pairs = geval_generate_answers(qa_pairs)

        geval_save_initial(qa_pairs, output_json)

        batch_result = geval_run_batch(qa_pairs, selected_metrics=metrics)
        if len(batch_result) == 3:
            test_cases, metrics_list, evaluation_result = batch_result
        else:
            test_cases, metrics_list = batch_result
            evaluation_result = None

        geval_update_metrics(
            qa_pairs,
            test_cases,
            metrics_list,
            evaluation_result=evaluation_result,
            output_filepath=output_json,
        )

        geval_display(qa_pairs, test_cases)

        print("\n✅ Evaluation complete!")

        print("\n🤖 Generating analysis summary with GPT-4o...")
        summary = generate_evaluation_summary(output_json, api_key)
        save_summary_to_json(output_json, summary)
        print("\n--- Analysis Summary ---")
        print(summary)

        print("\nGenerating HTML dashboard from all runs in directory...")
        create_dashboard(output_dir, "confident_ai_dashboard.html")

        return qa_pairs, test_cases, metrics_list

    @staticmethod
    def run_rag_evaluation(
        input_csv="golden_set.csv",
        output_json=None,
        output_dir=".",
        metrics=None,
    ):
        """
        Run the full RAG metrics evaluation pipeline.

        Each run writes a new timestamped JSON file so historical results are
        preserved. The dashboard is regenerated from all files in output_dir.

        Args:
            input_csv: Path to the golden set CSV file.
            output_json: Explicit path for the result JSON. When None (default),
                         a timestamped filename is auto-generated inside output_dir.
            output_dir: Directory where per-run JSON files accumulate (default ".").
                        Also used as the source for the combined dashboard.
            metrics: Required list of RAG metric keys to run.
                     Valid keys: answer_relevancy, faithfulness,
                     contextual_precision, contextual_recall, contextual_relevancy

        Examples:
            # Response quality only (2 API calls per test case instead of 5)
            center.run_rag_evaluation(metrics=["answer_relevancy", "faithfulness"])

            # Retrieval quality only
            center.run_rag_evaluation(metrics=["contextual_precision", "contextual_recall", "contextual_relevancy"])

            # Full end-to-end RAG evaluation
            center.run_rag_evaluation(metrics=["answer_relevancy", "faithfulness", "contextual_precision", "contextual_recall", "contextual_relevancy"])
        """
        if not metrics:
            raise ValueError(
                "You must specify which RAG metrics to run. "
                "Valid options: answer_relevancy, faithfulness, "
                "contextual_precision, contextual_recall, contextual_relevancy"
            )

        if output_json is None:
            os.makedirs(output_dir, exist_ok=True)
            output_json = _make_run_json_path(output_dir, prefix="evaluation_results")
        else:
            output_dir = os.path.dirname(os.path.abspath(output_json)) or "."

        print("🔹 Loading golden set Q&A pairs from CSV...")
        qa_pairs = rag_load_csv(input_csv)

        qa_pairs = rag_generate_answers(qa_pairs)

        rag_save_initial(qa_pairs, output_json)

        test_cases, metrics_list = rag_run_batch(qa_pairs, selected_metrics=metrics)

        rag_update_metrics(qa_pairs, test_cases, metrics_list, output_json)

        rag_display(qa_pairs, test_cases)

        print("\n✅ RAG Evaluation complete!")

        print("\n🤖 Generating analysis summary with GPT-4o...")
        summary = generate_evaluation_summary(output_json, api_key)
        save_summary_to_json(output_json, summary)
        print("\n--- Analysis Summary ---")
        print(summary)

        print("\nGenerating HTML dashboard from all runs in directory...")
        create_dashboard(output_dir, "confident_ai_dashboard.html")

        return qa_pairs, test_cases, metrics_list


if __name__ == "__main__":
    center = EvaluationCenter()
    # center.run_geval_evaluation(metrics=["fluency"])

    center.run_rag_evaluation(metrics=["answer_relevancy", "faithfulness"])
