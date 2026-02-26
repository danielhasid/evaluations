# Golden Set Evaluator with RAG Metrics

A Python script for evaluating RAG (Retrieval-Augmented Generation) systems using DeepEval's RAG-specific metrics.

## Overview

This script evaluates RAG system outputs using five specialized metrics:
1. **AnswerRelevancyMetric** - Measures how relevant the answer is to the question
2. **FaithfulnessMetric** - Checks if the answer is faithful to the retrieved context
3. **ContextualPrecisionMetric** - Evaluates precision of retrieved contexts
4. **ContextualRecallMetric** - Measures recall of relevant contexts
5. **ContextualRelevancyMetric** - Assesses relevance of retrieved contexts to the question

## Installation

Make sure you have the required dependencies installed:

```bash
pip install openai deepeval python-dotenv
```

## CSV Format for RAG Evaluation

The CSV file must include the following columns:

- `question` (required): The question to ask
- `expected_answer` (required): The reference/expected answer
- `retrieval_context` (required): Retrieved documents (pipe-separated)
- `context` (required): Golden/ideal context documents (pipe-separated)
- `metadata` (optional): Additional metadata for categorization

### Example CSV:

```csv
question,expected_answer,retrieval_context,context,metadata
"What is Python?","Python is a high-level programming language...","Python is a programming language|Python supports OOP|Python has simple syntax","Python is a high-level language|Created by Guido van Rossum|Emphasizes readability","basic_knowledge"
```

### Understanding Context Fields:

- **retrieval_context**: Documents that your RAG system actually retrieved (simulated in this case)
- **context**: The ideal/golden documents that should have been retrieved

The pipe character `|` separates multiple documents in each field.

## Usage

1. Set up your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=your-api-key-here
```

2. Prepare your `golden_set.csv` file with questions, expected answers, and contexts

3. Run the script:

```bash
python golden_set_evaluator_rag_metrix.py
```

## Output

The script generates:

1. **Console output**: Shows evaluation progress, metric scores, and pass/fail status
2. **evaluation_results.json**: Contains all Q&A data, contexts, and detailed metric results

### Sample JSON Output:

```json
{
  "question": "What is Python?",
  "generated_answer": "...",
  "expected_answer": "...",
  "retrieval_context": ["...", "..."],
  "context": ["...", "..."],
  "metadata": "basic_knowledge",
  "timestamp": "2026-02-24T...",
  "status": "pass",
  "evaluation_metrics": {
    "AnswerRelevancyMetric": {
      "score": 0.85,
      "reason": "The answer directly addresses the question...",
      "threshold": 0.7,
      "passed": true
    },
    "FaithfulnessMetric": {
      "score": 0.92,
      "reason": "The answer is consistent with the retrieved context...",
      "threshold": 0.7,
      "passed": true
    }
  }
}
```

## RAG Metrics Explained

### 1. AnswerRelevancyMetric
- **Purpose**: Measures how relevant the generated answer is to the input question
- **Uses**: input, actual_output
- **Good for**: Detecting hallucinations or off-topic responses

### 2. FaithfulnessMetric
- **Purpose**: Checks if the answer is grounded in the retrieved context
- **Uses**: actual_output, retrieval_context
- **Good for**: Ensuring answers don't contradict or go beyond retrieved documents

### 3. ContextualPrecisionMetric
- **Purpose**: Evaluates if retrieved contexts are precise and relevant
- **Uses**: input, expected_output, retrieval_context
- **Good for**: Measuring retriever quality - are irrelevant docs being retrieved?

### 4. ContextualRecallMetric
- **Purpose**: Measures if all relevant information was retrieved
- **Uses**: expected_output, retrieval_context
- **Good for**: Detecting missing relevant documents in retrieval

### 5. ContextualRelevancyMetric
- **Purpose**: Assesses overall relevance of retrieved contexts to the question
- **Uses**: input, retrieval_context
- **Good for**: Evaluating retriever's ability to find relevant documents

## Configuration

You can modify the following in the script:

### Metric Thresholds (lines 133-156)
All metrics use a threshold of `0.7` (70%). Adjust as needed:
```python
answer_relevancy = AnswerRelevancyMetric(
    threshold=0.8,  # Increase for stricter evaluation
    model="gpt-4"
)
```

### Model Selection
All metrics use `gpt-4o` for evaluation (required for structured outputs). You can change to other compatible models:
- `"gpt-4o"` (default, supports structured outputs)
- `"gpt-4o-mini"` (faster, cheaper, supports structured outputs)
- **Note**: Regular `gpt-4` does NOT support structured outputs and will cause errors with RAG metrics

### File Paths (lines 324-325)
```python
INPUT_CSV = "golden_set.csv"
OUTPUT_JSON = "evaluation_results.json"
```

## Workflow

1. **Load CSV** - Reads questions, answers, and contexts
2. **Generate Answers** - Uses GPT-4 to answer questions
3. **Save Initial Results** - Creates JSON with pending status
4. **Run Evaluation** - Evaluates with all 5 RAG metrics
5. **Update Results** - Adds scores, reasons, and pass/fail status
6. **Display Results** - Shows comprehensive evaluation report

## Key Features

- Two-stage JSON creation (before and after evaluation)
- Detailed metric scores with LLM-generated reasoning
- Pass/fail status for each metric and overall test case
- Threshold-based evaluation
- Mock retrieval contexts for testing without a real RAG system

## Differences from golden_set_geval_metrix.py

1. **RAG-specific metrics** instead of general GEval metrics
2. **Context fields required** in CSV (retrieval_context and context)
3. **Evaluates retrieval quality** in addition to answer quality
4. **More comprehensive** for RAG system evaluation

## Next Steps

To integrate with a real RAG system:
1. Replace `generate_answer()` function to call your RAG system
2. Capture actual `retrieval_context` from your retriever
3. Update CSV with real golden contexts
4. Run evaluation to measure RAG system performance
