# LLM Judge Architecture

This folder now uses a modular evaluation architecture focused on readability and debugability.

## Directory Layout

- `evaluation_center.py`: backward-compatible orchestration entrypoint.
- `apps/`: application-level facade (`EvaluationCenterFacade`).
- `core/`: shared pipeline modules (`config`, `env`, `answering`, `pipeline`, `results`).
- `evaluators/`: evaluator-specific adapters (`geval_adapter`, `rag_adapter`).
- `scripts/`: demo/experimental scripts moved out of production flow.
- `golden_set_geval_metrix.py` and `golden_set_evaluator_rag_metrix.py`: compatibility wrappers.

## Execution Flow

1. `EvaluationCenter` calls `EvaluationCenterFacade`.
2. Facade builds a `RunConfig`.
3. `core.pipeline.run_pipeline()`:
   - loads the dataset via adapter,
   - generates answers via shared OpenAI generator,
   - persists initial results,
   - runs selected metrics,
   - writes metric results and prints structured output.
4. Facade runs post-processing:
   - summary generation (`analyze_eval.py`),
   - dashboard generation (`generate_dashboard.py`).

## Debug Workflow

- Add breakpoints in:
  - `core/pipeline.py` for end-to-end stage transitions.
  - `evaluators/geval_adapter.py` or `evaluators/rag_adapter.py` for metric construction.
  - `core/results.py` for pass/fail and JSON output behavior.
- Each stage emits explicit console markers (`🔹`, `📐`, `💾`) to locate failures quickly.

## Run Examples

- GEval (default metrics):
  - `python evaluation_center.py`
- RAG (explicit metrics):
  - Use `EvaluationCenter().run_rag_evaluation(metrics=["answer_relevancy", "faithfulness"])` in Python.

## Backward Compatibility

- Existing public function names are preserved in:
  - `evaluation_center.py`
  - `golden_set_geval_metrix.py`
  - `golden_set_evaluator_rag_metrix.py`
- Old scripts remain callable, but now delegate to `scripts/` demo modules.
