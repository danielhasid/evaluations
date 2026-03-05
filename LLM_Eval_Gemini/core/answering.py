import os
from typing import Dict, List

import vertexai
from vertexai.generative_models import GenerativeModel

from .dataset_answering import generate_answers_for_dataset as shared_generate_answers_for_dataset
from .env import require_google_credentials


class GeminiAnswerGenerator:
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.7):
        credentials = require_google_credentials()
        vertexai.init(
            project=credentials["project"],
            location=credentials["location"],
        )
        self.model_name = model_name
        self.temperature = temperature
        self._model = GenerativeModel(model_name)

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages list to a single prompt and call Gemini."""
        prompt = "\n".join(m.get("content", "") for m in messages)
        response = self._model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature},
        )
        return response.text.strip()


def generate_answers_for_dataset(qa_pairs: list, answer_generator: GeminiAnswerGenerator) -> list:
    """Backward-compatible passthrough to shared dataset answer generation."""
    return shared_generate_answers_for_dataset(qa_pairs, answer_generator=answer_generator)
