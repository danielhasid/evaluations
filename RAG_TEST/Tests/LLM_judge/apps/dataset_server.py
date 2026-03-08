"""
Flask server for Dataset Management.

Serves the dashboard and exposes REST endpoints for:
  - Browsing the filesystem to locate CSV files
  - Loading, editing, adding, and deleting rows in any CSV
  - Triggering GEval or RAG evaluation runs

Run from the LLM_judge directory:
    python apps/dataset_server.py
Then open http://localhost:5000 in your browser.
"""

import os
import sys

# Force UTF-8 globally — must happen before any other imports.
# deepeval and other libraries print emoji (e.g. ✨) that crash on Windows cp1252.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import csv
import io
import threading

from flask import Flask, jsonify, request, send_file

# Ensure the LLM_judge root is on sys.path regardless of how this is invoked
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.config import GEVAL_METRIC_KEYS, RAG_METRIC_KEYS

app = Flask(__name__)

# The project root is the LLM_judge directory; file browsing is restricted here
PROJECT_ROOT = _ROOT
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
DASHBOARD_FILE = os.path.join(RESULTS_DIR, "confident_ai_dashboard.html")


# ─────────────────────────────────────
# HELPERS
# ─────────────────────────────────────

def _safe_path(path: str) -> str:
    """Resolve and validate that a path stays inside PROJECT_ROOT."""
    resolved = os.path.realpath(os.path.join(PROJECT_ROOT, path))
    if not resolved.startswith(os.path.realpath(PROJECT_ROOT)):
        raise ValueError(f"Access denied: path outside project root: {path}")
    return resolved


def _read_csv(filepath: str) -> dict:
    """Read a CSV file and return {columns, rows}."""
    rows = []
    columns = []
    with open(filepath, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
    return {"columns": columns, "rows": rows}


def _write_csv(filepath: str, columns: list, rows: list) -> None:
    """Write rows back to a CSV file preserving column order."""
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ─────────────────────────────────────
# SERVE DASHBOARD
# ─────────────────────────────────────

@app.route("/")
def index():
    if os.path.exists(DASHBOARD_FILE):
        return send_file(DASHBOARD_FILE)
    return "<h2>Dashboard not found. Run an evaluation first to generate it.</h2>", 404


# ─────────────────────────────────────
# FILE BROWSER
# ─────────────────────────────────────

@app.route("/browse")
def browse():
    """
    GET /browse?path=<relative_or_absolute_dir>

    Returns JSON with two lists:
      - folders: subdirectory names in the given directory
      - files:   .csv filenames in the given directory
      - current: the resolved absolute path being listed
    """
    raw = request.args.get("path", "")
    try:
        if raw:
            target = _safe_path(raw)
        else:
            target = os.path.realpath(PROJECT_ROOT)

        if not os.path.isdir(target):
            return jsonify({"error": f"Not a directory: {raw}"}), 400

        entries = os.listdir(target)
        folders = sorted(
            e for e in entries if os.path.isdir(os.path.join(target, e)) and not e.startswith(".")
        )
        files = sorted(
            e for e in entries if e.lower().endswith(".csv") and os.path.isfile(os.path.join(target, e))
        )
        return jsonify({
            "current": target,
            "parent": os.path.dirname(target) if target != os.path.realpath(PROJECT_ROOT) else None,
            "folders": folders,
            "files": files,
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 403
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────
# DATASET CRUD
# ─────────────────────────────────────

@app.route("/datasets/load")
def datasets_load():
    """
    GET /datasets/load?path=<csv_path>

    Reads the CSV at the given path and returns {columns, rows}.
    """
    raw = request.args.get("path", "")
    if not raw:
        return jsonify({"error": "path parameter is required"}), 400
    try:
        filepath = _safe_path(raw)
        if not os.path.isfile(filepath):
            return jsonify({"error": f"File not found: {raw}"}), 404
        data = _read_csv(filepath)
        return jsonify(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 403
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/datasets/save", methods=["POST"])
def datasets_save():
    """
    POST /datasets/save
    Body: {path: str, columns: [str, ...], rows: [{col: val, ...}, ...]}

    Writes all rows back to the CSV, replacing existing content.
    """
    body = request.get_json(force=True)
    raw = (body or {}).get("path", "")
    columns = (body or {}).get("columns", [])
    rows = (body or {}).get("rows", [])
    if not raw:
        return jsonify({"error": "path is required"}), 400
    try:
        filepath = _safe_path(raw)
        _write_csv(filepath, columns, rows)
        return jsonify({"ok": True, "rows_saved": len(rows)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 403
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/datasets/row", methods=["DELETE"])
def datasets_delete_row():
    """
    DELETE /datasets/row
    Body: {path: str, index: int}

    Deletes one row by zero-based index and saves the file.
    """
    body = request.get_json(force=True)
    raw = (body or {}).get("path", "")
    index = (body or {}).get("index")
    if not raw or index is None:
        return jsonify({"error": "path and index are required"}), 400
    try:
        filepath = _safe_path(raw)
        data = _read_csv(filepath)
        rows = data["rows"]
        if index < 0 or index >= len(rows):
            return jsonify({"error": f"Index {index} out of range (0-{len(rows)-1})"}), 400
        rows.pop(index)
        _write_csv(filepath, data["columns"], rows)
        return jsonify({"ok": True, "rows_remaining": len(rows)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 403
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ─────────────────────────────────────
# METRIC METADATA
# ─────────────────────────────────────

@app.route("/metrics")
def metrics():
    """
    GET /metrics

    Returns the available metric keys for each evaluator type.
    """
    return jsonify({
        "llm": GEVAL_METRIC_KEYS,
        "rag": RAG_METRIC_KEYS,
    })


# ─────────────────────────────────────
# RUN EVALUATION
# ─────────────────────────────────────

# Tracks the most recent run status so the UI can poll it
_run_status = {"running": False, "log": "", "error": None, "stopped": False}
_run_lock = threading.Lock()
_stop_event = threading.Event()


@app.route("/run/status")
def run_status():
    """GET /run/status — returns current run state for the UI to poll."""
    with _run_lock:
        return jsonify(dict(_run_status))


@app.route("/run/stop", methods=["POST"])
def run_stop():
    """POST /run/stop — requests cancellation of the running evaluation."""
    with _run_lock:
        if not _run_status["running"]:
            return jsonify({"error": "No evaluation is running."}), 400
        _run_status["log"] += "\n[STOP] Stop requested — will cancel after current step completes...\n"
    _stop_event.set()
    return jsonify({"ok": True, "message": "Stop requested."})


@app.route("/run", methods=["POST"])
def run_evaluation():
    """
    POST /run
    Body: {
        path:         str   — absolute or relative path to CSV,
        type:         str   — "llm" or "rag",
        metrics:      [str] — list of metric keys to evaluate,
        truths_limit: int   — optional, RAG only
    }

    Triggers an evaluation run in a background thread.
    Poll GET /run/status for progress.
    """
    body = request.get_json(force=True) or {}
    csv_path = body.get("path", "")
    eval_type = (body.get("type") or "").lower()
    selected_metrics = body.get("metrics") or []
    truths_limit = body.get("truths_limit")

    if not csv_path:
        return jsonify({"error": "path is required"}), 400
    if eval_type not in ("llm", "rag"):
        return jsonify({"error": "type must be 'llm' or 'rag'"}), 400
    if not selected_metrics:
        return jsonify({"error": "at least one metric must be selected"}), 400

    with _run_lock:
        if _run_status["running"]:
            return jsonify({"error": "An evaluation is already running. Wait for it to finish."}), 409
        _run_status["running"] = True
        _run_status["log"] = "Starting evaluation...\n"
        _run_status["error"] = None
        _run_status["stopped"] = False
    _stop_event.clear()

    def _run():
        # Re-apply UTF-8 on this thread's stdout reference in case the thread
        # inherited a different stream object (happens on some Windows Python builds).
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        try:
            filepath = _safe_path(csv_path)

            from apps.evaluation_center_app import EvaluationCenterFacade
            facade = EvaluationCenterFacade()

            def _append(msg):
                with _run_lock:
                    _run_status["log"] += str(msg) + "\n"

            def _check_stop():
                """Raise if the user has requested cancellation."""
                if _stop_event.is_set():
                    raise InterruptedError("Evaluation stopped by user.")

            # Redirect log_stage output into the browser log for this thread
            import core.logging_utils as _lu
            _orig_log_stage = _lu.log_stage
            _lu.log_stage = lambda msg: (_append(msg), _orig_log_stage(msg))

            _append(f"CSV: {filepath}")
            _append(f"Type: {eval_type.upper()}")
            _append(f"Metrics: {selected_metrics}")

            _check_stop()

            if eval_type == "llm":
                facade.run_geval_evaluation(
                    input_csv=filepath,
                    output_dir=RESULTS_DIR,
                    metrics=selected_metrics,
                )
            else:
                facade.run_rag_evaluation(
                    input_csv=filepath,
                    output_dir=RESULTS_DIR,
                    metrics=selected_metrics,
                    truths_extraction_limit=truths_limit,
                )

            _check_stop()
            _append("Evaluation complete. Dashboard updated.")
        except InterruptedError as exc:
            with _run_lock:
                _run_status["stopped"] = True
                _run_status["log"] += f"\n[STOPPED] {exc}\n"
        except Exception as exc:
            with _run_lock:
                _run_status["error"] = str(exc)
                _run_status["log"] += f"\nERROR: {exc}\n"
        finally:
            # Restore original log_stage
            try:
                import core.logging_utils as _lu
                _lu.log_stage = _orig_log_stage
            except Exception:
                pass
            with _run_lock:
                _run_status["running"] = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Evaluation started. Poll /run/status for progress."})


# ─────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────

if __name__ == "__main__":
    os.chdir(_ROOT)
    print(f"Serving dashboard from: {DASHBOARD_FILE}")
    print(f"Project root (browsing restricted to): {PROJECT_ROOT}")
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
