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

    def run_geval_evaluation(self, input_csv="golden_set.csv", output_json=None, output_dir="results", metrics=None):
        return self._app.run_geval_evaluation(
            input_csv=input_csv,
            output_json=output_json,
            output_dir=output_dir,
            metrics=metrics,
        )

    def run_rag_evaluation(self, input_csv="golden_set.csv", output_json=None, output_dir="results", metrics=None):
        return self._app.run_rag_evaluation(
            input_csv=input_csv,
            output_json=output_json,
            output_dir=output_dir,
            metrics=metrics,
        )


if __name__ == "__main__":
    center = EvaluationCenter()
    center.run_geval_evaluation(
        input_csv="golden_set.csv",
        output_dir="results",
    )