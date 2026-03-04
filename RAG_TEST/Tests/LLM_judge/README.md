# LLM Judge - QA Evaluation Framework

A modular pipeline for evaluating LLM-generated answers against a golden set of Q&A pairs.
Supports two evaluation modes: **GEval** (custom criteria) and **RAG** (retrieval-augmented generation metrics).

---

## How It Works - The Full Flow

```
golden_set.csv
      |
      v
[1] Load Q&A pairs          (adapter: load_golden_set_csv)
      |
      v
[2] Generate answers        (GPT-4 via core/answering.py)
      |
      v
[3] Save initial results    (core/results.py -> results/<Evaluator>_evaluation_results_TIMESTAMP.json)
      |
      v
[4] Run evaluation metrics  (DeepEval via adapter: run_batch_evaluation)
      |
      v
[5] Update results with     (core/results.py - scores, reasons, pass/fail per metric)
    metric scores
      |
      v
[6] Display results         (printed to console)
      |
      v
[7] Generate GPT-4o summary (analyze_eval.py - written back into the same JSON)
```

---

## File Structure and Responsibilities

```
LLM_judge/
|
|-- evaluation_center.py          Entry point. Run this file directly.
|
|-- core/
|   |-- config.py                 RunConfig dataclass + metric key lists
|   |-- env.py                    .env loading + API key validation
|   |-- answering.py              GPT-4 answer generation (OpenAI client)
|   |-- pipeline.py               Main orchestrator: connects all steps
|   |-- results.py                Save/update/display JSON results
|   `-- logging_utils.py          log_stage() used across all modules
|
|-- evaluators/
|   |-- base.py                   EvaluatorAdapter interface + BatchEvaluationResult
|   |-- geval_adapter.py          GEval metric definitions + CSV loader
|   `-- rag_adapter.py            RAG metric definitions + CSV loader
|
|-- apps/
|   `-- evaluation_center_app.py  EvaluationCenterFacade - wires everything together
|
|-- analyze_eval.py               GPT-4o report generation from results JSON
|-- generate_dashboard.py         (Optional) HTML dashboard from results JSONs
|
|-- golden_set.csv                Your Q&A test dataset (you provide this)
`-- results/                      All output files land here (auto-created)
    |-- GEval_evaluation_results_TIMESTAMP.json
    |-- RAG_evaluation_results_TIMESTAMP.json
    `-- ...
```

---

## File-by-File Explanation

### `evaluation_center.py` - Entry Point

The file you run. Contains the `EvaluationCenter` class which is a thin wrapper over
`EvaluationCenterFacade`. Delegates all logic to `apps/evaluation_center_app.py`.

```python
center = EvaluationCenter()
center.run_geval_evaluation()          # GEval with all default metrics
center.run_rag_evaluation(metrics=["answer_relevancy", "faithfulness"])
```

---

### `apps/evaluation_center_app.py` - Facade

`EvaluationCenterFacade` is the main coordinator. It:
1. Loads environment variables and validates the OpenAI API key
2. Builds a `RunConfig` with the chosen metrics and paths
3. Calls `run_pipeline()` with the right adapter (GEval or RAG)
4. After the pipeline finishes, calls `analyze_eval.py` to generate the GPT-4o summary

---

### `core/config.py` - Configuration

Defines two things:

**`RunConfig`** - the single object passed through the entire pipeline:

| Field | Default | Description |
|---|---|---|
| `evaluator_type` | required | `"GEval"` or `"RAG"` |
| `input_csv` | `"golden_set.csv"` | Path to your Q&A dataset |
| `output_json` | `None` | Override output file path (auto-generated if None) |
| `output_dir` | `"results"` | Folder where JSON results are saved |
| `metrics` | `[]` | List of metric keys to run |

**Metric key lists** - the valid string keys for each mode:

```python
GEVAL_METRIC_KEYS = ["fluency", "relevance", "correctness", "hallucination"]

RAG_METRIC_KEYS   = ["answer_relevancy", "faithfulness",
                     "contextual_precision", "contextual_recall", "contextual_relevancy"]
```

---

### `core/env.py` - Environment Setup

Called once at startup. Loads `.env` from the project root and validates that
`OPENAI_API_KEY` is set. Also syncs `CONFIDENT_API_KEY` / `DEEPEVAL_API_KEY` aliases.

---

### `core/pipeline.py` - Pipeline Orchestrator

`run_pipeline(config, adapter)` is the heart of the system. It runs every step
in order and is completely agnostic to whether you are using GEval or RAG - the
adapter handles the differences.

Steps:
1. Validate metrics against `adapter.valid_metric_keys`
2. Create the `results/` folder and generate an evaluator-prefixed timestamped filename
3. Load Q&A pairs from CSV via `adapter.load_golden_set_csv()`
4. Generate answers via `OpenAIAnswerGenerator` (GPT-4)
5. Save initial results to JSON (no scores yet, status = "pending")
6. Run batch evaluation via `adapter.run_batch_evaluation()`
7. Update the JSON with scores, reasons, and pass/fail status
8. Print results to console

---

### `core/answering.py` - Answer Generation

Sends each question to GPT-4 and returns the generated answer.
The model and temperature are configurable (`gpt-4`, `temperature=0.7` by default).

---

### `core/results.py` - Result Persistence

Three functions used by the pipeline:

- `save_initial_results()` - writes the first version of the JSON (no scores yet)
- `update_results_with_metrics()` - reads the JSON, fills in scores + reasons + pass/fail, writes it back
- `display_results()` - prints each Q&A pair with its metric scores to the console

The output JSON structure:
```json
{
  "evaluator_type": "GEval",
  "analysis_summary": "... (added after GPT-4o summary step) ...",
  "results": [
    {
      "question": "...",
      "expected_answer": "...",
      "generated_answer": "...",
      "status": "pass",
      "timestamp": "...",
      "evaluation_metrics": {
        "Fluency": { "score": 0.85, "threshold": 0.5, "passed": true, "reason": "..." }
      }
    }
  ]
}
```

---

### `evaluators/base.py` - Adapter Interface

Defines the contract every evaluator adapter must follow:

```python
class EvaluatorAdapter(Protocol):
    evaluator_type: str          # "GEval" or "RAG"
    valid_metric_keys: list      # list of accepted metric key strings

    def load_golden_set_csv(self, filepath: str) -> list: ...
    def run_batch_evaluation(self, qa_pairs, selected_metrics) -> BatchEvaluationResult: ...
```

`BatchEvaluationResult` holds the test cases, metrics, and raw DeepEval result object.

---

### `evaluators/geval_adapter.py` - GEval Adapter

Implements the adapter for custom criteria evaluation using DeepEval's `GEval`.

**CSV format required:**
```
question,expected_answer,context,metadata
"What is Python?","A programming language.","","category_a"
```
- `context` - optional, JSON array or plain string
- `metadata` - optional label/category for your own grouping

**Metrics it can run:**

| Key | Criteria |
|---|---|
| `fluency` | Is the output grammatically correct and easy to understand? |
| `relevance` | Does the output directly answer the question? |
| `correctness` | Is the output factually consistent with the expected answer? |
| `hallucination` | Does the output add false claims not in the expected answer? |

All metrics use GPT-4o as the judge (via DeepEval) with a default threshold of 0.5.

---

### `evaluators/rag_adapter.py` - RAG Adapter

Implements the adapter for RAG-specific metrics using DeepEval's built-in RAG metrics.

**CSV format required:**
```
question,expected_answer,retrieval_context,context,metadata
"What is X?","X is ...","doc1|doc2","doc1|doc2","category_a"
```
- `retrieval_context` - pipe-separated list of retrieved documents
- `context` - pipe-separated list of ground-truth context documents
- Both are required columns for RAG evaluation

**Metrics it can run:**

| Key | What it measures |
|---|---|
| `answer_relevancy` | Is the generated answer relevant to the question? |
| `faithfulness` | Is the answer grounded in the retrieved context (no hallucination)? |
| `contextual_precision` | Are the retrieved docs ranked with the most relevant ones first? |
| `contextual_recall` | Did the retriever surface all the necessary information? |
| `contextual_relevancy` | How relevant is the retrieved context to the question? |

All RAG metrics use `gpt-4o` as the judge with a default threshold of 0.7.

---

### `analyze_eval.py` - GPT-4o Report Generator

After the pipeline finishes, this is called automatically to produce a written analysis report.
It sends the full results JSON to GPT-4o and asks for a report covering:

1. Overall pass rate and average metric scores
2. Weakness analysis (failed tests only)
3. Detailed breakdown of every failed test case
4. Actionable recommendations

The report text is saved back into the same results JSON under `"analysis_summary"`.

---

## Quick Start

### 1. Prerequisites

Create a `.env` file in `LLM_judge/`:
```
OPENAI_API_KEY=sk-...
```

### 2. Prepare your dataset

Create `golden_set.csv` in `LLM_judge/`:

For GEval:
```csv
question,expected_answer,context,metadata
"What is Python?","A high-level programming language.","","general"
```

For RAG:
```csv
question,expected_answer,retrieval_context,context,metadata
"What is Python?","A high-level programming language.","Python is...","Python is...","general"
```

### 3. Run

```powershell
cd C:\work\PythonProject\RAG_TEST\Tests\LLM_judge

# GEval - all 4 default metrics
python evaluation_center.py

# Or from Python code:
from evaluation_center import EvaluationCenter

# GEval - specific metrics
center = EvaluationCenter()
center.run_geval_evaluation(metrics=["correctness", "hallucination"])

# RAG - must specify metrics explicitly
center.run_rag_evaluation(metrics=["answer_relevancy", "faithfulness"])
```

### 4. Output

Results are saved to:
- `results/GEval_evaluation_results_YYYYMMDD_HHMMSS.json` for GEval runs
- `results/RAG_evaluation_results_YYYYMMDD_HHMMSS.json` for RAG runs
The GPT-4o analysis summary is appended to the same file automatically.

### Dashboard Notes

- The dashboard is regenerated automatically after each run at `results/confident_ai_dashboard.html`.
- The runs view supports filtering by evaluator (`All`, `GEval`, `RAG`), and the performance graph updates to match the selected filter.
- If two metrics have exactly the same values across runs, their lines can overlap visually (for example, `FaithfulnessMetric` and `AnswerRelevancyMetric` both at `1.0`).

---

## Adding a New Metric or Evaluator

To add a new evaluation mode:

1. Create `evaluators/my_adapter.py` implementing `load_golden_set_csv()` and `run_batch_evaluation()`
2. Add the metric keys to `core/config.py`
3. Add a `run_my_evaluation()` method in `apps/evaluation_center_app.py` and `evaluation_center.py`

The pipeline in `core/pipeline.py` requires no changes.

---

## Debugging Tips

- Add a `print()` or breakpoint in `core/pipeline.py` to inspect the state between steps
- Break on `evaluators/geval_adapter.py:run_batch_evaluation` to inspect metric objects before DeepEval runs
- The JSON in `results/` after step 3 (before metric scoring) shows exactly what was sent to the evaluator
- Set `metrics=["correctness"]` to run a single metric for faster debug cycles
