import os

from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.evaluate import evaluate
from dotenv import load_dotenv

load_dotenv()

# --- OpenAI setup ---
api_key = os.getenv("OPENAI_API_KEY")
answer_relevancy_metric = AnswerRelevancyMetric()
test_case = LLMTestCase(
  input="Who is the current president of the United States of America?",
  actual_output="Joe Biden",
  retrieval_context=["Joe Biden serves as the current president of America."]
)

evaluate(test_cases=[test_case], metrics=[answer_relevancy_metric])