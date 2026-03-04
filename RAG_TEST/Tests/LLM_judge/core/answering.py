from typing import Dict, List

import openai

from .env import require_openai_api_key
from .logging_utils import log_stage


class OpenAIAnswerGenerator:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7):
        api_key = require_openai_api_key()
        self.model_name = model_name
        self.temperature = temperature
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return response.choices[0].message.content.strip()


def generate_answer(question: str, answer_generator: OpenAIAnswerGenerator) -> str:
    messages = [{
        "role": "user",
        "content": f"Answer the following question concisely and accurately:\n\n{question}",
    }]
    return answer_generator.generate(messages)


def generate_answers_for_dataset(qa_pairs: list, answer_generator: OpenAIAnswerGenerator) -> list:
    log_stage(f"\n[LLM] Generating answers for {len(qa_pairs)} questions...")
    for i, qa_pair in enumerate(qa_pairs, 1):
        question = qa_pair.get("question", "")
        log_stage(f"  [{i}/{len(qa_pairs)}] Generating answer for: {question[:60]}...")
        qa_pair["generated_answer"] = generate_answer(question, answer_generator)
    log_stage("[OK] Answer generation complete!")
    return qa_pairs
