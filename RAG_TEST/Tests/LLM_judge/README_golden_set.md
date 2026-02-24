# Golden Set Evaluator

A Python script for evaluating LLM-generated answers against a golden set of question-answer pairs using DeepEval metrics.

## Overview

This script:
1. Loads questions and expected answers from a CSV file
2. Generates answers using GPT-4
3. Evaluates the generated answers using DeepEval's LLM-as-a-Judge metrics
4. Saves results to JSON for further analysis

## Installation

Make sure you have the required dependencies installed:

```bash
pip install openai deepeval python-dotenv
```

## CSV Format

Create a `golden_set.csv` file with the following columns:

- `question` (required): The question to ask
- `expected_answer` (required): The reference/expected answer
- `metadata` (optional): Any additional metadata for the question

### Example CSV:

```csv
question,expected_answer,metadata
"What is Python?","Python is a high-level, interpreted programming language known for its simplicity and readability.","basic_knowledge"
"How does async/await work?","Async/await is a pattern for writing asynchronous code in Python.","advanced_concepts"
```

## Usage

1. Set up your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=your-api-key-here
```

2. Prepare your `golden_set.csv` file with questions and expected answers

3. Run the script:

```bash
python golden_set_evaluator.py
```

## Output

The script generates:

1. **Console output**: Shows the evaluation progress and results
2. **evaluation_results.json**: Contains all questions, generated answers, expected answers, and metadata

## Metrics

The script evaluates answers using four GEval metrics:

- **Fluency**: Is the output grammatically correct and easy to understand?
- **Coherence**: Is the output logically structured and cohesive?
- **Relevance**: Does the output appropriately answer the input question?
- **Correctness**: Is the actual output factually correct compared to the expected answer?

## Configuration

You can modify the following in the `main()` function:

- `INPUT_CSV`: Path to your golden set CSV file (default: `"golden_set.csv"`)
- `OUTPUT_JSON`: Path for the results JSON file (default: `"evaluation_results.json"`)

You can also adjust the GPT-4 model settings in the `GPT4ChatModel` class:
- `model_name`: GPT model to use (default: `"gpt-4"`)
- `temperature`: Sampling temperature (default: `0.7`)

## Key Differences from llm_as_a_judge.py

1. **CSV-based input**: Loads questions from CSV instead of text files
2. **Batch processing**: Handles multiple Q&A pairs in one run
3. **Reference answers**: Compares generated answers to expected answers
4. **Bug fix**: Removed the unsupported `skip_on_missing_params` parameter
5. **Correctness metric**: Added a metric that compares against expected answers

## Notes

- The script uses OpenAI's GPT-4 for answer generation
- DeepEval metrics also use an LLM for evaluation (configurable in DeepEval settings)
- Results are saved with timestamps for tracking multiple evaluation runs
