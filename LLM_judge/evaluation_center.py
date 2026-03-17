"""Backward-compatible entrypoint for LLM evaluation pipelines."""

try:
    from .apps.evaluation_center_app import EvaluationCenterFacade
except ImportError:  # pragma: no cover - script execution fallback
    from apps.evaluation_center_app import EvaluationCenterFacade


class EvaluationCenter:
    """Thin compatibility facade over the new modular pipeline."""

    def __init__(self):
        self._app = EvaluationCenterFacade()
        self.evaluation_metrics = []

    def add_evaluation_metric(self, metric):
        self.evaluation_metrics.append(metric)

    def evaluate(self, input_text, expected_output, actual_output):
        for metric in self.evaluation_metrics:
            metric.evaluate(input_text, expected_output, actual_output)

    def get_evaluation_results(self):
        return self.evaluation_metrics

    def run_geval_evaluation(self, input_csv="datasets/llm/LLM_goldenset.csv", output_json=None, output_dir="results", metrics=None):
        """Run GEval-based evaluation.

        Args:
            input_csv: GEval dataset CSV path.
            output_json: Optional explicit output JSON path.
            output_dir: Directory for generated artifacts.
            metrics: Optional GEval metric keys. If None, defaults are used.
        """
        return self._app.run_geval_evaluation(
            input_csv=input_csv,
            output_json=output_json,
            output_dir=output_dir,
            metrics=metrics,
        )

    def run_rag_evaluation(self, input_csv="datasets/rag/RAG_goldenset.csv", output_json=None, output_dir="results", metrics=None, truths_extraction_limit=None):
        """Run RAG-based evaluation.

        Args:
            input_csv: RAG dataset CSV path.
            output_json: Optional explicit output JSON path.
            output_dir: Directory for generated artifacts.
            metrics: Required RAG metric keys (e.g. answer_relevancy, faithfulness).
            truths_extraction_limit: Optional truth extraction limit for faithfulness.
                Use None to extract all truths.
        """
        return self._app.run_rag_evaluation(
            input_csv=input_csv,
            output_json=output_json,
            output_dir=output_dir,
            metrics=metrics,
            truths_extraction_limit=truths_extraction_limit,
        )


if __name__ == "__main__":
    center = EvaluationCenter()
    center.run_geval_evaluation(
        input_csv="datasets/llm/LLM_goldenset.csv",
        output_dir="results",
        # Set None or [] to run all default GEval metrics.
        # Available options:
        # - "fluency"
        # - "relevance"
        # - "correctness"
        # - "hallucination"
        metrics=["fluency", "hallucination"],
    )

    # center.run_rag_evaluation(
    #     input_csv="datasets/rag/RAG_goldenset.csv",
    #     output_dir="results",
    #     # Available options:
    #     # - "answer_relevancy"
    #     # - "faithfulness"
    #     # - "contextual_precision"
    #     # - "contextual_recall"
    #     # - "contextual_relevancy"
    #     metrics=["answer_relevancy", "faithfulness"],
    #     truths_extraction_limit=None,  # None = extract all truths (default), or set int (e.g., 10)
    # )
