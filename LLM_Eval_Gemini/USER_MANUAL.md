# LLM Eval Gemini — User Manual

## Overview

This pipeline evaluates a Q&A dataset using **DeepEval LLM-as-a-Judge** metrics and produces:

- A structured JSON file with per-question scores and pass/fail status.
- A **Gemini 1.5 Flash** generated analysis report written directly into the same JSON file under the `analysis_summary` key.

All LLM calls (answer generation, evaluation judging, and report generation) use **Gemini 1.5 Flash via Vertex AI**.

Two evaluation modes are available:

| Mode | EvaluationCenter Method | Metrics |
|---|---|---|
| **GEval** | `run_geval_evaluation()` | Fluency, Relevance, Correctness, Hallucination |
| **RAG** | `run_rag_evaluation()` | Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, Contextual Relevancy |

---

## Prerequisites

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up Google Cloud credentials

You need a GCP project with Vertex AI enabled and one of the following authentication methods:

**Option A — Application Default Credentials (recommended for local development):**
```bash
gcloud auth application-default login
```

**Option B — Service account key file:**
```bash
# Windows
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\service-account-key.json

# macOS / Linux
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### 3. Set your project configuration

Edit `.env.local` in the `LLM_Eval_Gemini/` directory:

```
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

Or set them as environment variables directly in your shell.

---

## Input File: `golden_set.csv`

Place your dataset CSV in the project directory or pass the path explicitly via `input_csv=`.

### GEval mode — required columns

| Column | Description |
|---|---|
| `question` | The question to evaluate |
| `expected_answer` | The ground-truth answer |
| `context` | *(optional)* Supporting context as a JSON list or plain string |
| `metadata` | *(optional)* Category label for the test case |

### RAG mode — required columns

| Column | Description |
|---|---|
| `question` | The question to evaluate |
| `expected_answer` | The ground-truth answer |
| `retrieval_context` | Pipe-separated (`\|`) list of retrieved documents |
| `context` | Pipe-separated (`\|`) list of ground-truth context documents |
| `metadata` | *(optional)* Category label for the test case |

### Example row (RAG mode)

```
question,expected_answer,retrieval_context,context,metadata
"What is Python?","Python is a high-level language.","Python supports OOP.|Python is readable.","Python is great.","basic_knowledge"
```

---

## Running the Evaluation

All scripts must be run from the `LLM_Eval_Gemini/` directory.

```bash
cd C:\work\PythonProject\LLM_Eval_Gemini
```

### Option A — Run via `EvaluationCenter` (recommended)

Create a small runner script or use a Python shell:

```python
from evaluation_center import EvaluationCenter

center = EvaluationCenter()

# Choose ONE of the two methods below:

# GEval pipeline (Fluency, Relevance, Correctness, Hallucination)
center.run_geval_evaluation(
    input_csv="datasets/llm/LLM_goldenset.csv",
    output_json="results/evaluation_results.json"
)

# RAG pipeline (Answer Relevancy, Faithfulness, Contextual metrics)
center.run_rag_evaluation(
    input_csv="datasets/rag/RAG_goldenset.csv",
    metrics=["answer_relevancy", "faithfulness"],
    output_json="results/evaluation_results.json"
)
```

Both methods:
1. Load Q&A pairs from the CSV.
2. Generate answers using **Gemini 1.5 Flash** (Vertex AI).
3. Save initial results to JSON.
4. Run DeepEval metrics judged by **Gemini 1.5 Flash**.
5. Update JSON with scores and pass/fail status.
6. Display a per-question results summary in the console.
7. Call **Gemini 1.5 Flash** to generate an analysis report and save it to the JSON under `analysis_summary`.

---

### Option B — Run via `evaluation_center.py`

Run the module directly:

```bash
python evaluation_center.py
```

Or call methods from Python for full control:

```python
from evaluation_center import EvaluationCenter

center = EvaluationCenter()
center.run_geval_evaluation(
    input_csv="datasets/llm/LLM_goldenset.csv",
    output_json="results/evaluation_results.json"
)

center.run_rag_evaluation(
    input_csv="datasets/rag/RAG_goldenset.csv",
    output_json="results/evaluation_results.json",
    metrics=["answer_relevancy", "faithfulness"]
)
```

---

### Option C — Re-run the analysis only

If a results JSON already exists and you only want to regenerate the analysis summary:

```bash
python analyze_eval.py
```

This reads the existing results JSON, sends it to **Gemini 1.5 Flash**, prints the report to the console, and writes it back under `analysis_summary`.

> **Note:** Edit `analyze_eval.py` and set `json_path` to point to your results file before running.

---

## Output File: `evaluation_results.json`

After a full run the file has the following structure:

```json
{
  "evaluator_type": "GEval",
  "results": [
    {
      "question": "What is Python?",
      "generated_answer": "Python is a high-level programming language...",
      "expected_answer": "Python is cool",
      "metadata": "basic_knowledge",
      "timestamp": "2026-03-05T10:00:00.000000",
      "evaluation_metrics": {
        "Fluency": { "score": 0.9, "reason": "...", "threshold": 0.5, "passed": true },
        "Relevance": { "score": 0.85, "reason": "...", "threshold": 0.5, "passed": true }
      },
      "status": "pass"
    }
  ],
  "analysis_summary": "## 1. Overall Score & Status\n\nTotal tests: 3 ..."
}
```

### Field reference

| Field | Description |
|---|---|
| `results` | Array of all evaluated test cases |
| `results[].status` | `"pass"` / `"failed"` / `"pending"` |
| `results[].evaluation_metrics` | Per-metric score, reason, threshold, and pass/fail |
| `analysis_summary` | Full Gemini-generated report (4 sections, see below) |

### Analysis summary sections

1. **Overall Score & Status** — total tests, pass rate, average metric scores, executive summary.
2. **Weaknesses Analysis** — patterns and root causes of failures.
3. **Detailed Breakdown of Failed Tests** — per-question failure explanation.
4. **Actionable Recommendations** — 2–3 concrete steps to improve prompts, dataset, or thresholds.

---

## Pipeline Flow

```
golden_set.csv
      │
      ▼
load_golden_set_csv()
      │
      ▼
generate_answers_for_dataset()   ← Gemini 1.5 Flash generates answers (Vertex AI)
      │
      ▼
save_initial_results()           → evaluation_results.json (status: pending)
      │
      ▼
run_batch_evaluation()           ← DeepEval runs metrics (judge: Gemini 1.5 Flash)
      │
      ▼
update_results_with_metrics()    → evaluation_results.json (status: pass/failed)
      │
      ▼
generate_evaluation_summary()    ← Gemini 1.5 Flash analyses results
      │
      ▼
save_summary_to_json()           → evaluation_results.json (analysis_summary added)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT not set` | Missing env var | Add it to `.env.local` or set in your shell |
| `GOOGLE_CLOUD_LOCATION not set` | Missing env var | Add `GOOGLE_CLOUD_LOCATION=us-central1` to `.env.local` |
| `google.auth.exceptions.DefaultCredentialsError` | No GCP credentials found | Run `gcloud auth application-default login` or set `GOOGLE_APPLICATION_CREDENTIALS` |
| `Permission denied on Vertex AI` | Project lacks Vertex AI API access | Enable the Vertex AI API in your GCP project console |
| `CSV file not found` | Wrong path or filename | Confirm the CSV path is correct relative to `LLM_Eval_Gemini/` |
| `CSV must contain columns: ...` | Missing required column | Add the missing column to your CSV (see Input File section) |
| `KeyError: 'results'` | Old flat-list JSON format | Delete the old JSON file and re-run |
| DeepEval metric errors | Vertex AI quota or model unavailability | Check GCP quotas; retry after a short wait |
