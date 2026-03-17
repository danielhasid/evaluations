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
|   |-- dataset_answering.py      Answer generation loop for dataset rows (ChromaRAG-ready)
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
|   |-- evaluation_center_app.py  EvaluationCenterFacade - wires everything together
|   `-- dataset_server.py         Flask server: dashboard UI + REST API for dataset management
|
|-- scripts/
|   |-- llm_as_a_judge_demo.py    Standalone demo: translate → summarize → GEval score
|   `-- rag_deep_eval_demo.py     Standalone demo: manually run RAG metrics on a test case
|
|-- tests/
|   `-- test_pipeline_smoke.py    Smoke tests for GEval and RAG pipelines (no real LLM calls)
|
|-- analyze_eval.py               GPT-4o report generation from results JSON
|-- generate_dashboard.py         HTML dashboard generated from all results JSONs
|
|-- datasets/
|   |-- llm/LLM_goldenset.csv     Default GEval dataset
|   `-- rag/RAG_goldenset.csv     Default RAG dataset
|
|-- golden_set.csv                Optional custom dataset if you pass input_csv explicitly
`-- results/                      All output files land here (auto-created)
    |-- GEval_evaluation_results_TIMESTAMP.json
    |-- RAG_evaluation_results_TIMESTAMP.json
    |-- Evaluation_dashbord.html
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

### `core/dataset_answering.py` - Dataset Answer Generation Loop

`generate_answers_for_dataset(qa_pairs, answer_generator)` iterates over all rows in the
loaded dataset and calls the provided `answer_generator.generate(messages)` for each question.
It logs progress as `[i/n]` and writes the result back into each `qa_pair["generated_answer"]`.

It also contains commented-out code for activating **ChromaRAG** answer generation — swap to
`ChromaRAGAnswerGenerator` and uncomment the block to automatically fill `retrieval_context`
from the Chroma retriever alongside the generated answer.

---

### `apps/dataset_server.py` - Flask Dataset Server

A Flask web server that exposes a browser UI and REST API for managing datasets and triggering
evaluation runs without touching the command line.

Run it with:
```powershell
python apps/dataset_server.py
```
Then open [http://localhost:5000](http://localhost:5000).

**REST endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the evaluation dashboard HTML |
| `GET` | `/browse?path=<dir>` | Lists folders and CSV files under the project root |
| `GET` | `/datasets/load?path=<csv>` | Returns `{columns, rows}` for a CSV file |
| `POST` | `/datasets/save` | Writes updated rows back to a CSV file |
| `DELETE` | `/datasets/row` | Deletes a single row by zero-based index |
| `GET` | `/metrics` | Returns valid metric keys for `llm` and `rag` modes |
| `POST` | `/run` | Starts a GEval or RAG evaluation in a background thread |
| `GET` | `/run/status` | Polls the running evaluation for log output and status |
| `POST` | `/run/stop` | Requests cancellation of the running evaluation |

File browsing is sandboxed to the `LLM_judge/` directory (path traversal is blocked).

---

### `scripts/llm_as_a_judge_demo.py` - LLM-as-a-Judge Demo

A self-contained script demonstrating the LLM-as-a-judge pattern:

1. Loads source text from `input_text.txt` and a reference summary from `reference_summary.txt`
2. Uses GPT-4 to summarize the source text into a target language (Romanian by default)
3. Back-translates the summary into English
4. Scores the back-translation with three GEval metrics: **Fluency**, **Coherence**, **Relevance**

Useful as a quick standalone proof-of-concept, independent of the main pipeline.

---

### `scripts/rag_deep_eval_demo.py` - RAG DeepEval Demo

A minimal script for manually testing individual RAG metrics (`AnswerRelevancyMetric`,
`FaithfulnessMetric`) on a single hardcoded test case. Useful for:

- Verifying DeepEval is installed and the API key works
- Experimenting with a single Q&A pair before building a full dataset

---

### `tests/test_pipeline_smoke.py` - Smoke Tests

Pytest tests that exercise the full pipeline without making any real OpenAI or DeepEval API
calls. Uses `DummyAdapter` (returns a single hardcoded Q&A pair) and `FakeAnswerGenerator`
(returns a fixed string) to verify the pipeline wiring end-to-end.

Two tests are included:

| Test | Evaluator | Metric |
|------|-----------|--------|
| `test_geval_smoke_pipeline` | GEval | `fluency` |
| `test_rag_smoke_pipeline` | RAG | `answer_relevancy` |

Run with:
```powershell
pytest tests/test_pipeline_smoke.py
```

---

## Quick Start

### 1. Prerequisites

Create a `.env` file in `LLM_judge/`:
```
OPENAI_API_KEY=sk-...
```

### 2. Prepare your dataset

You can either:

- use the default sample datasets:
  - `datasets/llm/LLM_goldenset.csv` (GEval)
  - `datasets/rag/RAG_goldenset.csv` (RAG)
- or provide your own CSV path via `input_csv=...`

If you use a custom CSV, expected formats are:

GEval:
```csv
question,expected_answer,context,metadata
"What is Python?","A high-level programming language.","","general"
```

RAG:
```csv
question,expected_answer,retrieval_context,context,metadata
"What is Python?","A high-level programming language.","Python is...","Python is...","general"
```

### 3. Run

**Option A — Direct execution (recommended)**

Run `evaluation_center.py` directly from the `LLM_judge/` directory:

```powershell
cd c:\Eval\evaluations\LLM_judge
python evaluation_center.py
```

To change which metrics or dataset are used, edit the `__main__` block at the bottom of `evaluation_center.py`:

```python
if __name__ == "__main__":
    center = EvaluationCenter()

    # GEval - pick any combination of: fluency, relevance, correctness, hallucination
    center.run_geval_evaluation(
        input_csv="datasets/llm/LLM_goldenset.csv",
        output_dir="results",
        metrics=["fluency", "relevance", "correctness", "hallucination"],
    )

    # RAG - pick any combination of: answer_relevancy, faithfulness,
    #        contextual_precision, contextual_recall, contextual_relevancy
    # center.run_rag_evaluation(
    #     input_csv="datasets/rag/RAG_goldenset.csv",
    #     output_dir="results",
    #     metrics=["answer_relevancy", "faithfulness"],
    # )
```

**Option B — From Python code**

Import and call programmatically from any script:

```python
from evaluation_center import EvaluationCenter

center = EvaluationCenter()

# GEval
center.run_geval_evaluation(
    input_csv="datasets/llm/LLM_goldenset.csv",
    metrics=["correctness", "hallucination"]
)

# RAG
center.run_rag_evaluation(
    input_csv="datasets/rag/RAG_goldenset.csv",
    metrics=["answer_relevancy", "faithfulness"]
)
```

**Option C — Flask UI**

Start the web server to manage datasets and trigger runs from a browser:

```powershell
python apps/dataset_server.py
```

Then open [http://localhost:5000](http://localhost:5000).

### 4. Output

Results are saved to:
- `results/GEval_evaluation_results_YYYYMMDD_HHMMSS.json` for GEval runs
- `results/RAG_evaluation_results_YYYYMMDD_HHMMSS.json` for RAG runs
The GPT-4o analysis summary is appended to the same file automatically.

### Dashboard Notes

- The dashboard is regenerated automatically after each run at `results/Evaluation_dashbord.html`.
- The runs view supports filtering by evaluator (`All`, `GEval`, `RAG`), and the performance graph updates to match the selected filter.
- If two metrics have exactly the same values across runs, their lines can overlap visually (for example, `FaithfulnessMetric` and `AnswerRelevancyMetric` both at `1.0`).

---

## Adding a New Metric or Evaluator

To add a new evaluation mode:

1. Create `evaluators/my_adapter.py` implementing `load_golden_set_csv()` and `run_batch_evaluation()`
2. Add the metric keys to `core/config.py`
3. Add a `run_my_evaluation()` method in `apps/evaluation_center_app.py` and `evaluation_center.py`
4. Expose it in `apps/dataset_server.py` by adding the new type to the `/run` endpoint's `eval_type` check

The pipeline in `core/pipeline.py` requires no changes.

---

## Debugging Tips

- Add a `print()` or breakpoint in `core/pipeline.py` to inspect the state between steps
- Break on `evaluators/geval_adapter.py:run_batch_evaluation` to inspect metric objects before DeepEval runs
- The JSON in `results/` after step 3 (before metric scoring) shows exactly what was sent to the evaluator
- Set `metrics=["correctness"]` to run a single metric for faster debug cycles
- Use `scripts/rag_deep_eval_demo.py` to test a single RAG metric in isolation before running the full pipeline
- Run `pytest tests/test_pipeline_smoke.py` to verify pipeline wiring after any structural change (no API calls required)
- For the Flask server, check the terminal output for per-request logs; the `/run/status` endpoint streams the same `log_stage` output that appears in the console
