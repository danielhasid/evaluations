from typing import Dict, List

import openai

from .dataset_answering import generate_answers_for_dataset as shared_generate_answers_for_dataset
from .env import require_openai_api_key


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
    """Backward-compatible passthrough to shared dataset answer generation."""
    return shared_generate_answers_for_dataset(qa_pairs, answer_generator=answer_generator)
