"""Demo script for translation-summary and GEval scoring."""

import os

import openai
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from dotenv import load_dotenv

load_dotenv()


class GPT4ChatModel:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.model_name = "gpt-4"
        self.client = openai.OpenAI(api_key=api_key)
        self.temperature = 0.7

    def generate(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()


def summarize_to_foreign(text, target_lang, model: GPT4ChatModel):
    messages = [{
        "role": "user",
        "content": f"Summarize the following English text in {target_lang}. Be concise and accurate.\n\n{text}",
    }]
    return model.generate(messages)


def back_translate_to_english(foreign_text, from_language, model: GPT4ChatModel):
    messages = [{
        "role": "user",
        "content": f"Translate the following {from_language} text back into English:\n\n{foreign_text}",
    }]
    return model.generate(messages)


def run_deepeval_metrics(input_text, ai_output):
    test_case = LLMTestCase(input=input_text, actual_output=ai_output, expected_output=None)
    dataset = EvaluationDataset(test_cases=[test_case])

    metrics = [
        GEval(
            name="Fluency",
            criteria="Is the output grammatically correct and easy to understand?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        ),
        GEval(
            name="Coherence",
            criteria="Is the output logically structured and cohesive?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        ),
        GEval(
            name="Relevance",
            criteria="Does the output appropriately and directly answer the input?",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        ),
    ]

    print("\n🔎 Evaluating with DeepEval (LLM-as-a-Judge)...")
    evaluate(dataset, metrics=metrics)


def main():
    model = GPT4ChatModel()
    source = load_text("input_text.txt")
    reference = load_text("reference_summary.txt")
    target_language = "Romanian"

    summary_foreign = summarize_to_foreign(source, target_language, model)
    backtranslated = back_translate_to_english(summary_foreign, target_language, model)

    print(reference)
    run_deepeval_metrics(source, backtranslated)


if __name__ == "__main__":
    main()
