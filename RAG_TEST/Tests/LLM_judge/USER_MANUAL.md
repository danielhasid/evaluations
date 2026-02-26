# LLM Evaluation Pipeline — User Manual

## Overview

This pipeline evaluates a Q&A dataset using **DeepEval LLM-as-a-Judge** metrics and produces:

- A structured JSON file (`evaluation_results.json`) with per-question scores and pass/fail status.
- A GPT-4o generated analysis report written directly into the same JSON file under the `analysis_summary` key.

Two evaluation modes are available:

| Mode | Evaluator File | Metrics |
|---|---|---|
| **GEval** | `golden_set_geval_metrix.py` | Fluency, Relevance, Correctness, Hallucination |
| **RAG** | `golden_set_evaluator_rag_metrix.py` | Answer Relevancy, Faithfulness, Contextual Precision, Contextual Recall, Contextual Relevancy |

---

## Prerequisites

### 1. Install dependencies

```bash
pip install -r RAG_TEST/requirements.txt
```

### 2. Set your OpenAI API key

Create a `.env` file in the `RAG_TEST/Tests/LLM_judge/` directory:

```
OPENAI_API_KEY=sk-...your-key-here...
```

---

## Input File: `golden_set.csv`

Place your dataset CSV in the same directory as the scripts (`RAG_TEST/Tests/LLM_judge/`).

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

All scripts must be run from the `RAG_TEST/Tests/LLM_judge/` directory.

```bash
cd RAG_TEST/Tests/LLM_judge
```

### Option A — Run via `EvaluationCenter` (recommended)

Create a small runner script or use a Python shell:

```python
from evaluation_center import EvaluationCenter

center = EvaluationCenter()

# Choose ONE of the two methods below:

# GEval pipeline (Fluency, Relevance, Correctness, Hallucination)
center.run_geval_evaluation(
    input_csv="golden_set.csv",
    output_json="evaluation_results.json"
)

# RAG pipeline (Answer Relevancy, Faithfulness, Contextual metrics)
center.run_rag_evaluation(
    input_csv="golden_set.csv",
    output_json="evaluation_results.json"
)
```

Both methods:
1. Load Q&A pairs from the CSV.
2. Generate answers using GPT-4.
3. Save initial results to JSON.
4. Run DeepEval metrics.
5. Update JSON with scores and pass/fail status.
6. Display a per-question results summary in the console.
7. Call GPT-4o to generate an analysis report and save it to `evaluation_results.json` under `analysis_summary`.

---

### Option B — Run the evaluator directly

**GEval pipeline:**

```bash
python golden_set_geval_metrix.py
```

**RAG pipeline:**

```bash
python golden_set_evaluator_rag_metrix.py
```

Both scripts default to `golden_set.csv` as input and `evaluation_results.json` as output. To customise paths, edit the `INPUT_CSV` / `OUTPUT_JSON` constants at the bottom of each file inside the `main()` function.

> Note: Running the evaluators directly does **not** automatically generate the analysis summary. Use Option A or run `analyze_eval.py` afterwards (see below).

---

### Option C — Re-run the analysis only

If `evaluation_results.json` already exists and you only want to regenerate or update the analysis summary:

```bash
python analyze_eval.py
```

This reads the existing `evaluation_results.json`, sends it to GPT-4o, prints the report to the console, and writes it back to the file under `analysis_summary`.

---

## Output File: `evaluation_results.json`

After a full run the file has the following structure:

```json
{
  "results": [
    {
      "question": "What is Python?",
      "generated_answer": "Python is a high-level programming language...",
      "expected_answer": "Python is cool",
      "metadata": "basic_knowledge",
      "timestamp": "2026-02-26T10:00:00.000000",
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
| `analysis_summary` | Full GPT-4o generated report (4 sections, see below) |

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
generate_answers_for_dataset()   ← GPT-4 generates answers
      │
      ▼
save_initial_results()           → evaluation_results.json (status: pending)
      │
      ▼
run_batch_evaluation()           ← DeepEval runs metrics
      │
      ▼
update_results_with_metrics()    → evaluation_results.json (status: pass/failed)
      │
      ▼
generate_evaluation_summary()    ← GPT-4o analyses results
      │
      ▼
save_summary_to_json()           → evaluation_results.json (analysis_summary added)
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `OPENAI_API_KEY not set` | Missing `.env` file or env var | Create `.env` with your key in the script directory |
| `CSV file not found` | Wrong path or filename | Confirm `golden_set.csv` is in `RAG_TEST/Tests/LLM_judge/` |
| `CSV must contain columns: ...` | Missing required column | Add the missing column to your CSV (see Input File section) |
| `KeyError: 'results'` | Old flat-list JSON format | Delete the old `evaluation_results.json` and re-run |
| DeepEval metric errors | API quota or model unavailability | Check OpenAI account quota; retry after a short wait |
