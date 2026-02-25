from golden_set_evaluator import (
    load_golden_set_csv,
    generate_answers_for_dataset,
    save_initial_results,
    run_batch_evaluation,
    update_results_with_metrics,
    display_results,
)


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

    def run_golden_set_evaluation(self, input_csv="golden_set.csv", output_json="evaluation_results.json"):
        """Run the full golden set evaluation pipeline from golden_set_evaluator."""
        print("🔹 Loading golden set Q&A pairs from CSV...")
        qa_pairs = load_golden_set_csv(input_csv)

        qa_pairs = generate_answers_for_dataset(qa_pairs) #Change to the correct system

        save_initial_results(qa_pairs, output_json)

        test_cases, metrics = run_batch_evaluation(qa_pairs)

        update_results_with_metrics(qa_pairs, test_cases, metrics, output_json)

        display_results(qa_pairs, test_cases)

        print("\n✅ Evaluation complete!")

        return qa_pairs, test_cases, metrics
