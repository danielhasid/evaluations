import os
import csv
import json
from datetime import datetime
import openai
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

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
    Load Q&A pairs from a CSV file for RAG evaluation.
    Expected columns: question, expected_answer, retrieval_context, context, and optional metadata
    Returns a list of dictionaries with all required fields for RAG metrics.
    """
    qa_pairs = []

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        required_columns = {'question', 'expected_answer', 'retrieval_context', 'context'}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {required_columns}. Found: {reader.fieldnames}")

        for row in reader:
            # Parse pipe-separated context strings into lists
            retrieval_context = [doc.strip() for doc in row['retrieval_context'].split('|') if doc.strip()]
            context = [doc.strip() for doc in row['context'].split('|') if doc.strip()]
            
            qa_pair = {
                'question': row['question'].strip(),
                'expected_answer': row['expected_answer'].strip(),
                'retrieval_context': retrieval_context,
                'context': context,
                'metadata': row.get('metadata', '').strip() if 'metadata' in row else ''
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


RAG_METRIC_KEYS = [
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
]


# --- DeepEval Evaluation ---
def run_batch_evaluation(qa_pairs, selected_metrics: list):
    """
    Evaluate all Q&A pairs using DeepEval RAG metrics.
    Creates test cases and runs only the caller-specified metrics.

    Args:
        qa_pairs: List of Q&A pair dicts with generated answers.
        selected_metrics: Required list of metric keys to run.
                          Valid keys: answer_relevancy, faithfulness,
                          contextual_precision, contextual_recall, contextual_relevancy

    Common combinations:
        Response quality only:   ["answer_relevancy", "faithfulness"]
        Retrieval quality only:  ["contextual_precision", "contextual_recall", "contextual_relevancy"]
        End-to-end RAG:          all five keys

    Returns:
        (test_cases, metrics) tuple.

    Raises:
        ValueError: If selected_metrics is empty or contains unknown keys.
    """
    if not selected_metrics:
        raise ValueError(
            f"You must specify which RAG metrics to run. "
            f"Valid options: {RAG_METRIC_KEYS}"
        )

    invalid = [k for k in selected_metrics if k not in RAG_METRIC_KEYS]
    if invalid:
        raise ValueError(
            f"Unknown RAG metrics: {invalid}. "
            f"Valid options: {RAG_METRIC_KEYS}"
        )

    test_cases = []
    for qa_pair in qa_pairs:
        test_case = LLMTestCase(
            input=qa_pair['question'],
            actual_output=qa_pair['generated_answer'],
            expected_output=qa_pair['expected_answer'],
            retrieval_context=qa_pair['retrieval_context'],
            context=qa_pair['context']
        )
        test_cases.append(test_case)

    # Build all available RAG metric objects
    # Note: Using gpt-4o instead of gpt-4 because RAG metrics require structured outputs
    answer_relevancy = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )

    contextual_precision = ContextualPrecisionMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )

    contextual_recall = ContextualRecallMetric(
        threshold=0.7,
        model="gpt-4o",
        include_reason=True
    )

    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.7,
        model="gpt-4o"
    )

    all_metrics = {
        "answer_relevancy": answer_relevancy,
        "faithfulness": faithfulness,
        "contextual_precision": contextual_precision,
        "contextual_recall": contextual_recall,
        "contextual_relevancy": contextual_relevancy,
    }

    metrics = [all_metrics[k] for k in selected_metrics]
    print(f"📐 Running RAG metrics: {selected_metrics}")

    return test_cases, metrics


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
        'evaluator_type': 'RAG',
        'results': results,
        'analysis_summary': None
    }

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Initial results saved to: {output_filepath}")


def update_results_with_metrics(qa_pairs, test_cases, metrics_list, output_filepath="evaluation_results.json"):
    """
    Run DeepEval metrics on each test case and update the JSON file with scores and reasons.
    Each metric is measured individually per test case to extract the score and reason attributes.
    """
    with open(output_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    results = data['results']

    print("\n🔎 Evaluating with DeepEval (LLM-as-a-Judge)...")

    for i, (result, test_case) in enumerate(zip(results, test_cases)):
        metric_data = {}
        all_passed = True

        for metric in metrics_list:
            try:
                # Measure the metric on this specific test case
                metric.measure(test_case)

                # Access the score and reason from the metric object
                if hasattr(metric, 'score') and metric.score is not None:
                    # Get metric name
                    metric_name = getattr(metric, 'name', metric.__class__.__name__)
                    
                    # Get threshold for this metric
                    threshold = getattr(metric, 'threshold', 0.5)
                    passed = metric.score >= threshold

                    # Store both score and reason
                    metric_data[metric_name] = {
                        'score': metric.score,
                        'reason': getattr(metric, 'reason', None),
                        'threshold': threshold,
                        'passed': passed
                    }

                    # Track if any metric failed
                    if not passed:
                        all_passed = False

                    print(f"  Test case {i + 1}, {metric_name}: {metric.score:.4f} (threshold: {threshold}) - {'✅ PASS' if passed else '❌ FAIL'}")
                    if hasattr(metric, 'reason') and metric.reason:
                        reason_preview = metric.reason[:100] + "..." if len(metric.reason) > 100 else metric.reason
                        print(f"    Reason: {reason_preview}")
            except Exception as e:
                print(f"  Warning: Could not extract metric data for {metric} on test case {i}: {e}")
                all_passed = False

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
    test_cases, metrics = run_batch_evaluation(
        qa_pairs,
        selected_metrics=["answer_relevancy", "faithfulness", "contextual_precision", "contextual_recall", "contextual_relevancy"]
    )

    # STEP 3: Update JSON with metrics
    update_results_with_metrics(qa_pairs, test_cases, metrics, OUTPUT_JSON)

    # Display final results
    display_results(qa_pairs, test_cases)

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
