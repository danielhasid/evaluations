import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .logging_utils import log_stage


def _extract_results_from_evaluate(evaluation_result: Any) -> List[Dict[str, Dict[str, Any]]]:
    """Best-effort extraction of per-test, per-metric data from DeepEval evaluate()."""
    extracted = []
    if evaluation_result is None:
        return extracted

    test_results = getattr(evaluation_result, "test_results", None)
    if test_results is None and isinstance(evaluation_result, dict):
        test_results = evaluation_result.get("test_results")
    if test_results is None:
        return extracted

    for test_result in test_results:
        row_metrics = {}
        metrics_data = getattr(test_result, "metrics_data", None)
        if metrics_data is None and isinstance(test_result, dict):
            metrics_data = test_result.get("metrics_data")

        if not metrics_data:
            extracted.append(row_metrics)
            continue

        for md in metrics_data:
            if isinstance(md, dict):
                metric_name = md.get("name") or md.get("metric")
                score = md.get("score")
                reason = md.get("reason")
                threshold = md.get("threshold")
                passed = md.get("success")
                if passed is None:
                    passed = md.get("passed")
            else:
                metric_name = getattr(md, "name", None) or getattr(md, "metric", None)
                score = getattr(md, "score", None)
                reason = getattr(md, "reason", None)
                threshold = getattr(md, "threshold", None)
                passed = getattr(md, "success", None)
                if passed is None:
                    passed = getattr(md, "passed", None)

            if metric_name:
                row_metrics[metric_name] = {
                    "score": score,
                    "reason": reason,
                    "threshold": threshold,
                    "passed": passed,
                }

        extracted.append(row_metrics)

    return extracted


def save_initial_results(qa_pairs: list, evaluator_type: str, output_filepath: str) -> None:
    """Save initial results without metric scores."""
    results = []
    for qa_pair in qa_pairs:
        results.append({
            "question": qa_pair.get("question", ""),
            "generated_answer": qa_pair.get("generated_answer", ""),
            "expected_answer": qa_pair.get("expected_answer", ""),
            "metadata": qa_pair.get("metadata", ""),
            "timestamp": datetime.now().isoformat(),
            "evaluation_metrics": None,
            "status": "pending",
        })

    data = {
        "evaluator_type": evaluator_type,
        "results": results,
        "analysis_summary": None,
    }

    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log_stage(f"\n[SAVED] Initial results saved to: {output_filepath}")


def update_results_with_metrics(
    test_cases: list,
    metrics_list: list,
    output_filepath: str,
    evaluation_result: Any = None,
) -> None:
    """Update JSON results with metric scores and reasons."""
    with open(output_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    extracted_by_case = _extract_results_from_evaluate(evaluation_result)
    log_stage("\n[>>] Extracting DeepEval metric data...")

    for i, (result, test_case) in enumerate(zip(results, test_cases)):
        metric_data = extracted_by_case[i] if i < len(extracted_by_case) else {}
        all_passed = True

        # Fallback for versions where evaluate() result has no detailed metrics.
        if not metric_data:
            for metric in metrics_list:
                try:
                    metric.measure(test_case)
                    if hasattr(metric, "score") and metric.score is not None:
                        metric_name = getattr(metric, "name", metric.__class__.__name__)
                        threshold = getattr(metric, "threshold", 0.5)
                        passed = metric.score >= threshold
                        metric_data[metric_name] = {
                            "score": metric.score,
                            "reason": getattr(metric, "reason", None),
                            "threshold": threshold,
                            "passed": passed,
                        }
                except Exception as exc:
                    log_stage(f"  Warning: Could not extract metric data for {metric} on test case {i}: {exc}")
                    all_passed = False

        for metric_name, row in metric_data.items():
            threshold = row.get("threshold")
            score = row.get("score")
            passed = row.get("passed")

            if threshold is None:
                threshold = next(
                    (getattr(m, "threshold", 0.5) for m in metrics_list if getattr(m, "name", "") == metric_name),
                    0.5,
                )
                row["threshold"] = threshold

            if passed is None and isinstance(score, (int, float)):
                passed = score >= threshold
                row["passed"] = passed

            if passed is False:
                all_passed = False

            if isinstance(score, (int, float)):
                log_stage(
                    f"  Test case {i + 1}, {metric_name}: {score:.4f} (threshold: {threshold}) "
                    f"- {'PASS' if passed else 'FAIL'}"
                )

        result["evaluation_metrics"] = metric_data if metric_data else None
        result["status"] = "pass" if all_passed and metric_data else "failed"

    data["results"] = results
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log_stage(f"\n[SAVED] Results updated with evaluation metrics and reasons in: {output_filepath}")


def display_results(qa_pairs: list, output_filepath: str) -> None:
    """Display detailed results for each Q&A pair from a specific output file."""
    log_stage("\n" + "=" * 80)
    log_stage("EVALUATION RESULTS")
    log_stage("=" * 80)

    try:
        with open(output_filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = data.get("results", [])
    except FileNotFoundError:
        results = []

    for i, qa_pair in enumerate(qa_pairs, 1):
        log_stage(f"\n[Question {i}]")
        log_stage(f"Q: {qa_pair.get('question', '')}")
        log_stage("\n[Generated Answer]")
        log_stage(f"A: {qa_pair.get('generated_answer', '')}")
        log_stage("\n[Expected Answer]")
        log_stage(f"E: {qa_pair.get('expected_answer', '')}")

        if qa_pair.get("metadata"):
            log_stage(f"\n[Metadata]: {qa_pair['metadata']}")

        if i <= len(results):
            status = results[i - 1].get("status", "unknown")
            status_label = "PASS" if status == "pass" else "FAIL" if status == "failed" else "PENDING"
            log_stage(f"\n[Overall Status]: {status_label}")

        metrics = results[i - 1].get("evaluation_metrics") if i <= len(results) else None
        if metrics:
            log_stage("\n[Evaluation Metrics]")
            for metric_name, metric in metrics.items():
                if isinstance(metric, dict):
                    score = metric.get("score", "N/A")
                    reason = metric.get("reason", "")
                    passed = metric.get("passed", None)
                    threshold = metric.get("threshold", None)

                    status_indicator = ""
                    if passed is not None:
                        status_indicator = " [PASS]" if passed else " [FAIL]"

                    score_text = f"{score:.4f}" if isinstance(score, float) else str(score)
                    threshold_text = f" (threshold: {threshold})" if threshold is not None else ""
                    log_stage(f"  {metric_name}: {score_text}{threshold_text}{status_indicator}")
                    if reason:
                        log_stage(f"    Reason: {reason}")
                else:
                    metric_text = f"{metric:.4f}" if isinstance(metric, float) else str(metric)
                    log_stage(f"  {metric_name}: {metric_text}")
        log_stage("-" * 80)
