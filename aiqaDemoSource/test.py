from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

answer_relevancy_metric = AnswerRelevancyMetric()
test_case = LLMTestCase(
  input="Who is the current president of the United States of America?",
  actual_output="Joe Biden",
  retrieval_context=["Joe Biden serves as the current president of America."]
)

answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.score)