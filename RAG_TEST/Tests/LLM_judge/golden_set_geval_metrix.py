import os
import csv
import json
from datetime import datetime
import openai
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics.g_eval import Rubric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from dotenv import load_dotenv

load_dotenv()

# --- OpenAI setup ---
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
client = openai.OpenAI(api_key=api_key)


class GPT4ChatModel:
    def __init__(self):
        self.model_name = "gpt-4"
        self.client = client
        self.temperature = 0.7

    def generate(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"An error occurred: {e}"


gpt4 = GPT4ChatModel()


# --- CSV Dataset Loading ---
def load_golden_set_csv(filepath):
    """
    Load Q&A pairs from a CSV file.
    Expected columns: question, expected_answer, and optional metadata
    Returns a list of dictionaries with question, expected_answer, and metadata
    """
    qa_pairs = []

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        required_columns = {'question', 'expected_answer'}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {required_columns}. Found: {reader.fieldnames}")

        for row in reader:
            raw_context = (row.get('context') or '').strip()
            try:
                context = json.loads(raw_context) if raw_context else []
            except json.JSONDecodeError:
                context = [raw_context] if raw_context else []

            qa_pair = {
                'question': row['question'].strip(),
                'expected_answer': row['expected_answer'].strip(),
                'context': context,
                'metadata': (row.get('metadata') or '').strip()
            }
            qa_pairs.append(qa_pair)

    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs from {filepath}")
    return qa_pairs


# --- Answer Generation ---
def generate_answer(question):
    """
    Generate an answer for a given question using GPT-4.
    Returns the generated answer as a string.
    """
    messages = [{
        "role": "user",
        "content": f"Answer the following question concisely and accurately:\n\n{question}"
    }]
    return gpt4.generate(messages)


def generate_answers_for_dataset(qa_pairs):
    """
    Generate answers for all questions in the dataset.
    Returns the qa_pairs list with an added 'generated_answer' field.
    """
    print(f"\n🤖 Generating answers for {len(qa_pairs)} questions...")

    for i, qa_pair in enumerate(qa_pairs, 1):
        print(f"  [{i}/{len(qa_pairs)}] Generating answer for: {qa_pair['question'][:60]}...")
        qa_pair['generated_answer'] = generate_answer(qa_pair['question'])

    print("✅ Answer generation complete!")
    return qa_pairs


GEVAL_METRIC_KEYS = ["fluency", "relevance", "correctness", "hallucination"]


def _build_evaluation_dataset(test_cases):
    """
    Build EvaluationDataset in a version-compatible way.
    Some DeepEval versions accept test_cases in __init__, others require add_* methods.
    """
    try:
        return EvaluationDataset(test_cases=test_cases)
    except TypeError:
        dataset = EvaluationDataset()
        if hasattr(dataset, "add_test_cases"):
            dataset.add_test_cases(test_cases)
            return dataset
        if hasattr(dataset, "add_test_case"):
            for tc in test_cases:
                dataset.add_test_case(tc)
            return dataset
        if hasattr(dataset, "test_cases"):
            existing = getattr(dataset, "test_cases")
            if isinstance(existing, list):
                existing.extend(test_cases)
            else:
                setattr(dataset, "test_cases", test_cases)
            return dataset
        raise TypeError(
            "Unsupported DeepEval EvaluationDataset API. "
            "Could not attach test cases to dataset."
        )


def _extract_results_from_evaluate(evaluation_result):
    """
    Best-effort extraction of per-test, per-metric data from DeepEval evaluate().
    Returns: list[dict[str, dict]] aligned by test case index.
    """
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


# --- DeepEval Evaluation ---
def run_batch_evaluation(qa_pairs, selected_metrics: list):
    """
    Evaluate all Q&A pairs using DeepEval GEval metrics.
    Creates test cases and runs only the caller-specified metrics.

    Args:
        qa_pairs: List of Q&A pair dicts with generated answers.
        selected_metrics: Required list of metric keys to run.
                          Valid keys: fluency, relevance, correctness, hallucination

    Returns:
        (test_cases, metrics, evaluation_result) tuple.

    Raises:
        ValueError: If selected_metrics is empty or contains unknown keys.
    """
    if not selected_metrics:
        raise ValueError(
            f"You must specify which GEval metrics to run. "
            f"Valid options: {GEVAL_METRIC_KEYS}"
        )

    invalid = [k for k in selected_metrics if k not in GEVAL_METRIC_KEYS]
    if invalid:
        raise ValueError(
            f"Unknown GEval metrics: {invalid}. "
            f"Valid options: {GEVAL_METRIC_KEYS}"
        )

    test_cases = []
    for qa_pair in qa_pairs:
        test_case = LLMTestCase(
            input=qa_pair['question'],
            actual_output=qa_pair['generated_answer'],
            expected_output=qa_pair['expected_answer'],
            context=qa_pair['context'] if qa_pair.get('context') else None
        )
        test_cases.append(test_case)

    # Build all available GEval metric objects
    fluency = GEval(
        name="Fluency",
        criteria="Is the output grammatically correct and easy to understand?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )

    relevance = GEval(
        name="Relevance",
        criteria="Does the output appropriately and directly answer the input question?",
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        threshold=0.5
    )

    correctness = GEval(
        name="Correctness",
        criteria="Is the actual output factually correct and consistent with the expected answer?",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
            "You should also heavily penalize omission of detail",
            "Vague language, or contradicting OPINIONS, are OK"
        ],
        rubric=[
            Rubric(score_range=(0, 2), expected_outcome="Factually incorrect."),
            Rubric(score_range=(3, 6), expected_outcome="Mostly correct."),
            Rubric(score_range=(7, 9), expected_outcome="Correct but missing minor details."),
            Rubric(score_range=(10, 10), expected_outcome="100% correct."),
        ],
        evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )

    hallucination = GEval(
        name="Hallucination",
        criteria="Does the output match the expected output without adding false claims?",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT
        ],
        threshold=0.5
    )

    all_metrics = {
        "fluency": fluency,
        "relevance": relevance,
        "correctness": correctness,
        "hallucination": hallucination,
    }

    metrics = [all_metrics[k] for k in selected_metrics]
    print(f"📐 Running GEval metrics: {selected_metrics}")

    # Run DeepEval in batch mode as requested.
    # In this DeepEval version, evaluate() expects an iterable of test cases.
    evaluation_result = evaluate(test_cases=test_cases, metrics=metrics)

    return test_cases, metrics, evaluation_result


# --- Results Output ---
def display_results(qa_pairs, test_cases):
    """
    Display detailed results for each Q&A pair.
    Shows metric scores and reasons from the JSON file.
    """
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)

    # Load the results from JSON to get the metrics with reasons
    try:
        with open("evaluation_results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = data['results']
    except FileNotFoundError:
        results = []

    for i, qa_pair in enumerate(qa_pairs, 1):
        print(f"\n[Question {i}]")
        print(f"Q: {qa_pair['question']}")
        print(f"\n[Generated Answer]")
        print(f"A: {qa_pair['generated_answer']}")
        print(f"\n[Expected Answer]")
        print(f"E: {qa_pair['expected_answer']}")

        if qa_pair.get('metadata'):
            print(f"\n[Metadata]: {qa_pair['metadata']}")

        # Display overall status
        if i <= len(results):
            status = results[i - 1].get('status', 'unknown')
            status_emoji = '✅' if status == 'pass' else '❌' if status == 'failed' else '⏳'
            print(f"\n[Overall Status]: {status_emoji} {status.upper()}")

        # Display metrics with scores and reasons from JSON
        if i <= len(results) and results[i - 1].get('evaluation_metrics'):
            print(f"\n[Evaluation Metrics]")
            for metric_name, metric_data in results[i - 1]['evaluation_metrics'].items():
                if isinstance(metric_data, dict):
                    score = metric_data.get('score', 'N/A')
                    reason = metric_data.get('reason', '')
                    passed = metric_data.get('passed', None)
                    threshold = metric_data.get('threshold', None)

                    status_indicator = ''
                    if passed is not None:
                        status_indicator = ' ✅' if passed else ' ❌'

                    score_text = f"{score:.4f}" if isinstance(score, float) else str(score)
                    threshold_text = f" (threshold: {threshold})" if threshold is not None else ""
                    print(f"  {metric_name}: {score_text}{threshold_text}{status_indicator}")

                    if reason:
                        print(f"    Reason: {reason}")
                else:
                    # Fallback for old format (just score)
                    print(f"  {metric_name}: {metric_data:.4f}" if isinstance(metric_data,
                                                                              float) else f"  {metric_name}: {metric_data}")

        print("-" * 80)


def save_initial_results(qa_pairs, output_filepath="evaluation_results.json"):
    """
    Save initial Q&A pairs with generated answers to JSON file before evaluation.
    This creates the results file without metric scores.
    """
    results = []

    for qa_pair in qa_pairs:
        result = {
            'question': qa_pair['question'],
            'generated_answer': qa_pair['generated_answer'],
            'expected_answer': qa_pair['expected_answer'],
            'metadata': qa_pair.get('metadata', ''),
            'timestamp': datetime.now().isoformat(),
            'evaluation_metrics': None,
            'status': 'pending'
        }
        results.append(result)

    data = {
        'evaluator_type': 'GEval',
        'results': results,
        'analysis_summary': None
    }

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Initial results saved to: {output_filepath}")


def update_results_with_metrics(
    qa_pairs,
    test_cases,
    metrics_list,
    evaluation_result=None,
    output_filepath="evaluation_results.json",
):
    """
    Run DeepEval metrics on each test case and update the JSON file with scores and reasons.
    Each metric is measured individually per test case to extract the score and reason attributes.
    """
    with open(output_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']

    print("\n🔎 Extracting DeepEval metric data...")

    extracted_by_case = _extract_results_from_evaluate(evaluation_result)

    for i, (result, test_case) in enumerate(zip(results, test_cases)):
        metric_data = extracted_by_case[i] if i < len(extracted_by_case) else {}
        all_passed = True

        # Fallback to measure() only if evaluate() output did not provide metric data.
        if not metric_data:
            for metric in metrics_list:
                try:
                    metric.measure(test_case)
                    if hasattr(metric, 'score') and metric.score is not None:
                        metric_name = getattr(metric, 'name', metric.__class__.__name__)
                        threshold = getattr(metric, 'threshold', 0.5)
                        passed = metric.score >= threshold
                        metric_data[metric_name] = {
                            'score': metric.score,
                            'reason': getattr(metric, 'reason', None),
                            'threshold': threshold,
                            'passed': passed
                        }
                except Exception as e:
                    print(f"  Warning: Could not extract metric data for {metric} on test case {i}: {e}")
                    all_passed = False

        for metric_name, data_row in metric_data.items():
            threshold = data_row.get("threshold")
            score = data_row.get("score")
            passed = data_row.get("passed")

            if threshold is None:
                threshold = next(
                    (getattr(m, "threshold", 0.5) for m in metrics_list if getattr(m, "name", "") == metric_name),
                    0.5
                )
                data_row["threshold"] = threshold

            if passed is None and isinstance(score, (int, float)):
                passed = score >= threshold
                data_row["passed"] = passed

            if passed is False:
                all_passed = False

            if isinstance(score, (int, float)):
                print(
                    f"  Test case {i + 1}, {metric_name}: {score:.4f} (threshold: {threshold}) - {'✅ PASS' if passed else '❌ FAIL'}"
                )

        result['evaluation_metrics'] = metric_data if metric_data else None
        result['status'] = 'pass' if all_passed and metric_data else 'failed'

    data['results'] = results

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results updated with evaluation metrics and reasons in: {output_filepath}")


# --- Main Pipeline ---
def main():
    INPUT_CSV = "golden_set.csv"
    OUTPUT_JSON = "evaluation_results.json"

    print("🔹 Loading golden set Q&A pairs from CSV...")
    qa_pairs = load_golden_set_csv(INPUT_CSV)

    # Generate answers for all questions
    qa_pairs = generate_answers_for_dataset(qa_pairs)

    # STEP 1: Save initial results (without metrics)
    save_initial_results(qa_pairs, OUTPUT_JSON)

    # STEP 2: Run DeepEval evaluation (specify which metrics to run)
    test_cases, metrics, evaluation_result = run_batch_evaluation(
        qa_pairs,
        selected_metrics=["fluency", "relevance", "correctness", "hallucination"]
    )

    # STEP 3: Update JSON with metrics
    update_results_with_metrics(qa_pairs, test_cases, metrics, evaluation_result, OUTPUT_JSON)

    # Display final results
    display_results(qa_pairs, test_cases)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
