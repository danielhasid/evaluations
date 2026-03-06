import json
import html
import os
import re
import sys
from pathlib import Path
from datetime import datetime
from statistics import mean, stdev


# ── Markdown → HTML helper ────────────────────────────────────────────────────

def _render_summary_html(text: str) -> str:
    """Convert the GPT markdown analysis report into clean, professional HTML."""
    if not text:
        return ""
    lines = text.splitlines()
    out = []
    in_list = False
    for line in lines:
        stripped = line.strip()
        stripped = re.sub(r'^#{1,6}\s*', '', stripped)
        stripped = re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)
        stripped = re.sub(r'[*_]', '', stripped)
        if not stripped:
            if in_list:
                out.append('</ul>')
                in_list = False
            out.append('<div class="mb-3"></div>')
            continue
        section_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if section_match and len(stripped) < 80:
            if in_list:
                out.append('</ul>')
                in_list = False
            num = section_match.group(1)
            title = html.escape(section_match.group(2))
            out.append(
                f'<div class="flex items-center gap-2 mt-6 mb-3">'
                f'<span class="flex-shrink-0 w-6 h-6 rounded-full bg-violet-500/20 text-violet-400 '
                f'text-xs font-bold flex items-center justify-center border border-violet-500/30">{num}</span>'
                f'<h4 class="text-gray-100 font-semibold text-sm tracking-wide">{title}</h4>'
                f'</div>'
            )
            continue
        bullet_match = re.match(r'^[-*]\s+(.+)$', stripped)
        if bullet_match:
            if not in_list:
                out.append('<ul class="space-y-1.5 ml-4">')
                in_list = True
            content = html.escape(bullet_match.group(1))
            out.append(
                f'<li class="flex items-start gap-2 text-gray-300 text-sm leading-relaxed">'
                f'<span class="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-violet-400/60 mt-2"></span>'
                f'<span>{content}</span></li>'
            )
            continue
        if in_list:
            out.append('</ul>')
            in_list = False
        out.append(f'<p class="text-gray-300 text-sm leading-relaxed">{html.escape(stripped)}</p>')
    if in_list:
        out.append('</ul>')
    return '\n'.join(out)


# ── Small utilities ───────────────────────────────────────────────────────────

def _as_list(data):
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("data", "results", "items", "evaluations"):
            if isinstance(data.get(k), list):
                return data[k]
    return []


def _safe(s: str) -> str:
    return html.escape(s or "", quote=True)


def _to_float(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_bool(v, default=None):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        if v.lower() in ("true", "1", "yes"):
            return True
        if v.lower() in ("false", "0", "no"):
            return False
    return default


def _parse_datetime_value(value: str):
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        pass
    m = re.search(r"(\d{8})_(\d{6})", raw)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y%m%d_%H%M%S")
        except ValueError:
            pass
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}-{m.group(2)}-{m.group(3)}", "%Y-%m-%d")
        except ValueError:
            pass
    m = re.search(r"(\d{2})_(\d{2})_(\d{4})", raw)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)}_{m.group(2)}_{m.group(3)}", "%d_%m_%Y")
        except ValueError:
            pass
    return None


def _format_date_ddmmyyyy(value: str) -> str:
    dt = _parse_datetime_value(value)
    if dt is None:
        return (value or "").strip()
    return dt.strftime("%d-%m-%Y")


def _norm_metric_key(k) -> str:
    if not isinstance(k, str):
        return ""
    return k.strip().lower()


def _normalize_metrics_dict(raw_metrics):
    metrics_by_norm = {}
    display_by_norm = {}
    if isinstance(raw_metrics, dict):
        for k, v in raw_metrics.items():
            if not isinstance(k, str):
                continue
            display = k.strip()
            nk = _norm_metric_key(k)
            if not nk:
                continue
            metrics_by_norm[nk] = v
            display_by_norm.setdefault(nk, display)
    return metrics_by_norm, display_by_norm


def _item_passed(item: dict, metric_names: list, display_to_norm: dict) -> bool:
    direct = _to_bool(item.get("passed"), default=None)
    if direct is not None:
        return direct
    status = (item.get("status") or "").strip().lower()
    if status in ("pass", "passed", "ok", "success"):
        return True
    if status in ("fail", "failed", "error"):
        return False
    raw_metrics = item.get("evaluation_metrics", {}) or {}
    metrics_by_norm, _ = _normalize_metrics_dict(raw_metrics)
    metric_pass_flags = []
    has_any_flag = False
    for m_display in metric_names:
        nk = display_to_norm.get(m_display, _norm_metric_key(m_display))
        md = metrics_by_norm.get(nk, {}) or {}
        p = _to_bool(md.get("passed"), default=None)
        if p is not None:
            has_any_flag = True
            metric_pass_flags.append(p)
    if has_any_flag:
        return all(metric_pass_flags)
    fallback_flags = []
    for m_display in metric_names:
        nk = display_to_norm.get(m_display, _norm_metric_key(m_display))
        md = metrics_by_norm.get(nk, {}) or {}
        score = _to_float(md.get("score", 0.0))
        thr = md.get("threshold", None)
        if thr is None:
            fallback_flags.append(score > 0)
        else:
            fallback_flags.append(score >= _to_float(thr))
    return all(fallback_flags)


def _compute_metric_stdev(scores: list) -> float:
    if len(scores) < 2:
        return 0.0
    return stdev(scores)


# ── Color palettes ────────────────────────────────────────────────────────────

METRIC_COLORS = [
    "#a855f7",  # purple
    "#fde047",  # yellow
    "#3b82f6",  # blue
    "#10b981",  # green
    "#f97316",  # orange
    "#ec4899",  # pink
    "#14b8a6",  # teal
    "#ef4444",  # red
    "#8b5cf6",  # violet
    "#22d3ee",  # cyan
]

CARD_STYLES = [
    ("bg-[#1e1a2e]", "border-[#3d2f6a]/40"),
    ("bg-[#1e1e14]", "border-[#6a5e1a]/40"),
    ("bg-[#141a2e]", "border-[#1e3a6a]/40"),
    ("bg-[#141e1a]", "border-[#1a4a2e]/40"),
    ("bg-[#2e1a14]", "border-[#6a2e1a]/40"),
    ("bg-[#2e141e]", "border-[#6a1a3a]/40"),
    ("bg-[#141e1e]", "border-[#1a4a4a]/40"),
    ("bg-[#2e1414]", "border-[#6a1a1a]/40"),
    ("bg-[#1e1a2e]", "border-[#3d2f6a]/40"),
    ("bg-[#141e2e]", "border-[#1a3a6a]/40"),
]

DASH_PATTERNS = ['[]', '[6,3]', '[2,3]', '[8,3,2,3]', '[12,3]',
                 '[4,2,4,2]', '[1,4]', '[10,2,2,2]', '[6,2,2,2,2,2]', '[3,3,3,3]']
POINT_STYLES = ['circle', 'rect', 'triangle', 'rectRot', 'star',
                'cross', 'crossRot', 'dash', 'line', 'circle']


# ── Input normalisation ───────────────────────────────────────────────────────

def _normalize_dashboard_inputs(json_input) -> list:
    if isinstance(json_input, (str, os.PathLike)):
        candidates = [str(json_input)]
    elif isinstance(json_input, (list, tuple, set)):
        candidates = [str(p) for p in json_input if isinstance(p, (str, os.PathLike))]
    else:
        raise TypeError("json_input must be a path, a directory, or a list/tuple/set of paths.")
    files = []
    for raw_path in candidates:
        p = Path(raw_path)
        if p.is_dir():
            eval_json = sorted(str(fp) for fp in p.glob("evaluation_results*.json"))
            files.extend(eval_json if eval_json else sorted(str(fp) for fp in p.glob("*.json")))
        else:
            files.append(str(p))
    deduped = []
    seen = set()
    for fp in files:
        key = str(Path(fp).resolve())
        if key not in seen:
            deduped.append(fp)
            seen.add(key)
    return deduped


# ── Pre-render test-case rows HTML ────────────────────────────────────────────

def _build_rows_html(table_items, all_metric_names, id_prefix=""):
    """Render the expandable test-case rows for one run."""
    rows_html = ""
    for row in table_items:
        i = row["idx"]
        uid = f"{id_prefix}{i}"
        status_text = "PASSED" if row["passed"] else "FAILED"

        input_text = row["question"] or ""
        output_text = row["generated_answer"] or ""
        expected_text = row["expected_answer"] or ""
        meta_text = row["metadata"] or ""
        ts_text = row["timestamp"] or ""
        top_status = row.get("status", "") or ""
        source_file = row.get("source_file", "") or ""

        input_trunc = (input_text[:50] + "...") if len(input_text) > 50 else input_text
        output_trunc = (output_text[:80] + "...") if len(output_text) > 80 else output_text

        tooltip_text = (
            f"STATUS: {status_text} | JSON status: {top_status} | "
            f"{row['metric_kind']}: {row['metric']} ({row['metric_score']:.3f})"
        )

        detail_row_id = f"details_{uid}"

        metric_cards_html = ""
        for m in all_metric_names:
            md = row["evaluation_metrics"].get(m, {}) or {}
            sc = md.get("score", 0.0)
            rs = md.get("reason", "")
            th = md.get("threshold", None)
            mp = md.get("passed", None)
            th_txt = "n/a" if th is None else f"{th:.3f}"
            mp_txt = "n/a" if mp is None else ("true" if mp else "false")
            metric_cards_html += (
                f'<div class="border border-gray-800/50 rounded-md p-3">'
                f'<div class="flex items-center justify-between">'
                f'<div class="text-gray-200 font-semibold text-xs">{_safe(m)}</div>'
                f'<div class="text-gray-300 text-xs">{sc:.3f}</div>'
                f'</div>'
                f'<div class="flex items-center gap-4 mt-2 text-[10px] uppercase tracking-wider text-gray-500">'
                f'<div>threshold: {th_txt}</div>'
                f'<div>passed: {mp_txt}</div>'
                f'</div>'
                f'<div class="text-gray-400 mt-2 whitespace-pre-wrap">{_safe(rs)}</div>'
                f'</div>'
            )

        passed_cls = ('bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                      if row['passed'] else
                      'bg-rose-500/20 text-rose-400 border-rose-500/30')
        passed_txt = 'Passed' if row['passed'] else 'Failed'

        rows_html += (
            f'<tr class="table-hover-row border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors text-sm cursor-pointer"'
            f' title="{_safe(tooltip_text)}" data-target="{detail_row_id}">'
            f'<td class="py-4 px-4 whitespace-nowrap">'
            f'<span class="px-2 py-1 rounded-md text-[10px] uppercase font-bold tracking-wider border {passed_cls}">'
            f'{passed_txt}</span></td>'
            f'<td class="py-4 px-4 text-gray-300 truncate max-w-[200px]">{_safe(input_trunc)}</td>'
            f'<td class="py-4 px-4 text-gray-400 truncate max-w-[300px]">{_safe(output_trunc)}</td>'
            f'</tr>'
            f'<tr id="{detail_row_id}" class="hidden border-b border-gray-800/50 bg-[#15161b]">'
            f'<td colspan="3" class="py-4 px-4 text-xs text-gray-300">'
            f'<div class="space-y-3">'
            f'<div class="flex flex-wrap gap-6">'
            f'<div><div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Status</div>'
            f'<div class="text-gray-300">{_safe(status_text)}</div></div>'
            f'<div><div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">JSON status</div>'
            f'<div class="text-gray-300">{_safe(top_status)}</div></div>'
            f'<div><div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Source file</div>'
            f'<div class="text-gray-300">{_safe(source_file)}</div></div>'
            f'<div><div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Metadata</div>'
            f'<div class="text-gray-300">{_safe(meta_text)}</div></div>'
            f'<div><div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Timestamp</div>'
            f'<div class="text-gray-300">{_safe(ts_text)}</div></div>'
            f'</div>'
            f'<div><div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Question</div>'
            f'<div class="text-gray-200 whitespace-pre-wrap">{_safe(input_text)}</div></div>'
            f'<div><div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Generated Answer</div>'
            f'<div class="text-gray-200 whitespace-pre-wrap">{_safe(output_text)}</div></div>'
            f'<div><div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Expected Answer</div>'
            f'<div class="text-gray-200 whitespace-pre-wrap">{_safe(expected_text)}</div></div>'
            f'<div><div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-2">Evaluation Metrics</div>'
            f'<div class="space-y-2">{metric_cards_html}</div></div>'
            f'</div></td></tr>'
        )
    return rows_html


# ── Per-run summary builder ───────────────────────────────────────────────────

def _build_per_run_summaries(source_files: list) -> list:
    """
    Load each JSON file, compute per-run stats, pre-render test-case rows HTML.
    Returns list of run summary dicts.
    """
    runs = []
    for filepath in source_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Skipping {filepath}: {e}")
            continue

        rows = _as_list(raw)
        if not rows:
            print(f"Warning: No evaluation rows found in {filepath}; skipping.")
            continue

        source_name = Path(filepath).name
        evaluator_type = ""
        analysis_summary_text = ""
        if isinstance(raw, dict):
            evaluator_type = (raw.get("evaluator_type") or "").strip()
            analysis_summary_text = raw.get("analysis_summary") or ""

        # Discover metrics for this run
        run_metric_norm_set = set()
        run_display_by_norm = {}
        for item in rows:
            rm, dm = _normalize_metrics_dict(item.get("evaluation_metrics", {}) or {})
            run_metric_norm_set.update(rm.keys())
            for nk, disp in dm.items():
                run_display_by_norm.setdefault(nk, disp)

        run_metric_names = sorted([run_display_by_norm.get(nk, nk) for nk in run_metric_norm_set])
        run_display_to_norm = {run_display_by_norm.get(nk, nk): nk for nk in run_metric_norm_set}
        run_metric_color = {m: METRIC_COLORS[i % len(METRIC_COLORS)] for i, m in enumerate(run_metric_names)}

        table_items = []
        passed_count = 0
        metric_scores_all = {m: [] for m in run_metric_names}
        first_ts = ""

        for i, item in enumerate(rows):
            if not isinstance(item, dict):
                continue
            if not first_ts:
                first_ts = item.get("timestamp", "") or ""

            rm, _ = _normalize_metrics_dict(item.get("evaluation_metrics", {}) or {})

            per_metric_score = {}
            per_metric_reason = {}
            per_metric_threshold = {}
            per_metric_passed = {}
            per_metric_verbose_logs = {}
            per_metric_criteria = {}
            per_metric_steps = {}
            per_metric_rubric = {}
            per_metric_model = {}
            lowest_metric = ""
            lowest_score = 1.0
            lowest_reason = ""

            for m_display in run_metric_names:
                nk = run_display_to_norm.get(m_display, _norm_metric_key(m_display))
                md = rm.get(nk, {}) or {}
                score_f = _to_float(md.get("score", 0.0))
                reason = md.get("reason", "No reason provided.")
                thr = md.get("threshold", None)
                thr_f = _to_float(thr) if thr is not None else None
                mp = _to_bool(md.get("passed"), default=None)

                per_metric_score[m_display] = score_f
                per_metric_reason[m_display] = reason
                per_metric_threshold[m_display] = thr_f
                per_metric_passed[m_display] = mp
                per_metric_verbose_logs[m_display] = md.get("verbose_logs", "")
                per_metric_criteria[m_display] = md.get("criteria", "")
                per_metric_steps[m_display] = md.get("evaluation_steps", [])
                per_metric_rubric[m_display] = md.get("rubric", "")
                per_metric_model[m_display] = md.get("evaluation_model", "")
                metric_scores_all[m_display].append(score_f)

                if score_f < lowest_score:
                    lowest_score = score_f
                    lowest_metric = m_display
                    lowest_reason = reason

            item_passed = _item_passed(item, run_metric_names, run_display_to_norm)
            if item_passed:
                passed_count += 1

            failed_metrics = [m for m in run_metric_names if per_metric_passed.get(m) is False]
            if failed_metrics:
                badge_metric = min(failed_metrics, key=lambda m: per_metric_score.get(m, 1.0))
                badge_kind = "FAILED METRIC"
                badge_metric_score = per_metric_score.get(badge_metric, lowest_score)
            elif lowest_metric:
                badge_metric = lowest_metric
                badge_kind = "LOWEST METRIC"
                badge_metric_score = lowest_score
            else:
                badge_metric = "N/A"
                badge_kind = "NO METRICS"
                badge_metric_score = 0.0

            table_items.append({
                "idx": i,
                "passed": item_passed,
                "metric": badge_metric,
                "metric_kind": badge_kind,
                "metric_score": badge_metric_score,
                "lowest_score": lowest_score,
                "question": item.get("question", ""),
                "generated_answer": item.get("generated_answer", ""),
                "expected_answer": item.get("expected_answer", ""),
                "context": item.get("context", ""),
                "retrieval_context": item.get("retrieval_context", ""),
                "metadata": item.get("metadata", ""),
                "timestamp": item.get("timestamp", ""),
                "status": item.get("status", ""),
                "source_file": source_name,
                "evaluation_metrics": {
                    m: {
                        "score": per_metric_score.get(m, 0.0),
                        "reason": per_metric_reason.get(m, ""),
                        "threshold": per_metric_threshold.get(m, None),
                        "passed": per_metric_passed.get(m, None),
                        "verbose_logs": per_metric_verbose_logs.get(m, ""),
                        "criteria": per_metric_criteria.get(m, ""),
                        "evaluation_steps": per_metric_steps.get(m, []),
                        "rubric": per_metric_rubric.get(m, ""),
                        "evaluation_model": per_metric_model.get(m, ""),
                    }
                    for m in run_metric_names
                },
                "lowest_reason": lowest_reason,
            })

        total_count = len(table_items)
        failed_count = total_count - passed_count
        pass_rate = (passed_count / total_count * 100) if total_count > 0 else 0.0
        metric_avgs = {m: (mean(metric_scores_all[m]) if metric_scores_all[m] else 0.0)
                       for m in run_metric_names}
        metric_stdevs = {m: _compute_metric_stdev(metric_scores_all[m]) for m in run_metric_names}

        # Sort: failed first, then passed
        table_items.sort(key=lambda r: r["passed"])
        test_cases = []
        for row in table_items:
            test_cases.append({
                "name": f"test_case_{row['idx']}",
                "status": "Success" if row["passed"] else "Failed",
                "question": row["question"],
                "generated_answer": row["generated_answer"],
                "expected_answer": row["expected_answer"],
                "context": row.get("context", ""),
                "retrieval_context": row.get("retrieval_context", ""),
                "metadata": row["metadata"],
                "timestamp": row["timestamp"],
                "source_file": row["source_file"],
                "metric": row["metric"],
                "metric_kind": row["metric_kind"],
                "metric_score": row["metric_score"],
                "evaluation_metrics": row["evaluation_metrics"],
            })

        run_idx = len(runs)
        rows_html = _build_rows_html(table_items, run_metric_names, id_prefix=f"r{run_idx}_")

        # Build short label for chart x-axis
        label = _format_date_ddmmyyyy(first_ts) if first_ts else source_name
        ts_match = re.search(r'(\d{8}_\d{6})', source_name)
        if ts_match:
            ts_raw = ts_match.group(1)
            try:
                dt = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S")
                label = dt.strftime("%d-%m-%Y")
            except ValueError:
                label = _format_date_ddmmyyyy(ts_raw)
        elif len(label) > 22:
            label = label[:19] + "..."

        runs.append({
            "filename": source_name,
            "filepath": str(filepath),
            "label": label,
            "timestamp": first_ts,
            "evaluator_type": evaluator_type or "Unknown",
            "total": total_count,
            "passed": passed_count,
            "failed": failed_count,
            "pass_rate": round(pass_rate, 1),
            "metric_avgs": metric_avgs,
            "metric_stdevs": metric_stdevs,
            "metric_names": run_metric_names,
            "metric_colors": run_metric_color,
            "analysis_summary": analysis_summary_text,
            "test_cases": test_cases,
            "rows_html": rows_html,
        })

    return runs


# ── Main entry point ──────────────────────────────────────────────────────────

def create_dashboard(json_filepath, html_filepath):
    source_files = _normalize_dashboard_inputs(json_filepath)
    if not source_files:
        print("Error: No JSON input files were provided.")
        return

    runs = _build_per_run_summaries(source_files)
    if not runs:
        print("Error: Could not load valid evaluation items from the provided inputs.")
        return

    # Show newest runs first in all run-based views.
    def _run_sort_key(run_obj: dict) -> datetime:
        ts = (run_obj.get("timestamp") or "").strip()
        if ts:
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                pass
        name = run_obj.get("filename", "")
        ts_match = re.search(r'(\d{8}_\d{6})', name)
        if ts_match:
            try:
                return datetime.strptime(ts_match.group(1), "%Y%m%d_%H%M%S")
            except ValueError:
                pass
        return datetime.min

    runs.sort(key=_run_sort_key, reverse=True)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    all_metric_names_set = set()
    for r in runs:
        all_metric_names_set.update(r["metric_names"])
    all_metric_names = sorted(all_metric_names_set)
    metric_color = {m: METRIC_COLORS[i % len(METRIC_COLORS)] for i, m in enumerate(all_metric_names)}
    metric_card_style = {m: CARD_STYLES[i % len(CARD_STYLES)] for i, m in enumerate(all_metric_names)}

    total_runs = len(runs)
    total_cases = sum(r["total"] for r in runs)
    total_passed = sum(r["passed"] for r in runs)
    total_failed = total_cases - total_passed
    overall_pass_rate = (total_passed / total_cases * 100) if total_cases > 0 else 0.0

    evaluator_types_set = set(r["evaluator_type"] for r in runs
                              if r["evaluator_type"] not in ("", "Unknown"))
    if len(evaluator_types_set) == 1:
        evaluator_type = next(iter(evaluator_types_set))
    elif len(evaluator_types_set) > 1:
        evaluator_type = "Mixed"
    else:
        evaluator_type = "Unknown"

    evaluator_bg = (
        "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" if evaluator_type == "RAG" else
        "bg-violet-500/10 border-violet-500/30 text-violet-400" if evaluator_type == "GEval" else
        "bg-gray-500/10 border-gray-500/30 text-gray-400"
    )
    evaluator_desc = (
        "RAG Metrics (Faithfulness, Precision, Recall, Relevancy)" if evaluator_type == "RAG" else
        "GEval Custom Criteria" if evaluator_type == "GEval" else
        "Mixed evaluator types across runs" if evaluator_type == "Mixed" else
        "Unknown evaluator"
    )

    # ── Runs list chart (x = run, y = metric avg) ─────────────────────────────
    run_labels = [r["label"] for r in runs]

    runs_chart_datasets = []
    for idx, m in enumerate(all_metric_names):
        color = metric_color[m]
        data_points = []
        sd_points = []
        for r in runs:
            avg = r["metric_avgs"].get(m, None)
            sd = r.get("metric_stdevs", {}).get(m, None)
            data_points.append(round(avg, 4) if avg is not None else None)
            sd_points.append(round(sd, 4) if sd is not None else None)

        valid = [p for p in data_points if p is not None]
        delta = ""
        if len(valid) >= 2 and valid[0] != 0:
            d = (valid[-1] - valid[0]) / valid[0] * 100
            delta = f"{'+'if d >= 0 else ''}{d:.1f}%"

        label_with_delta = f"{m}  {delta}" if delta else m
        dash = DASH_PATTERNS[idx % len(DASH_PATTERNS)]
        pstyle = POINT_STYLES[idx % len(POINT_STYLES)]

        parts = [
            '{',
            f'                label: {json.dumps(label_with_delta)},',
            f'                data: {json.dumps(data_points)},',
            f'                errorBarData: {json.dumps(sd_points)},',
            f'                backgroundColor: {json.dumps(color + "33")},',
            f'                borderColor: {json.dumps(color)},',
            '                borderWidth: 2.5,',
            '                tension: 0.4,',
            '                pointRadius: 5,',
            '                pointHoverRadius: 8,',
            f'                pointStyle: {json.dumps(pstyle)},',
            '                fill: false,',
            '                spanGaps: true,',
            '            }',
        ]
        runs_chart_datasets.append('\n'.join(parts))
    runs_chart_datasets_str = ',\n'.join(runs_chart_datasets)

    # ── Runs list table rows HTML ─────────────────────────────────────────────
    runs_table_rows_html = ""
    for i, r in enumerate(runs):
        metric_chips = ""
        for m in r["metric_names"][:3]:
            avg = r["metric_avgs"].get(m, 0.0)
            col = r["metric_colors"].get(m, "#6b7280")
            short = m.split("[")[0].strip()[:12]
            metric_chips += (
                f'<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-mono"'
                f' style="background:{col}22;color:{col};border:1px solid {col}44">'
                f'{_safe(short)}: {avg:.2f}</span> '
            )
        if len(r["metric_names"]) > 3:
            metric_chips += f'<span class="text-gray-500 text-[10px]">+{len(r["metric_names"]) - 3} more</span>'

        ts_display = _format_date_ddmmyyyy(r["timestamp"]) if r["timestamp"] else r["label"]
        fname_short = r["filename"][:32] + ("..." if len(r["filename"]) > 32 else "")

        result_cls = (
            "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" if r["failed"] == 0 else
            "bg-rose-500/20 text-rose-400 border-rose-500/30"
        )
        result_label = f"{r['passed']}/{r['total']} passed"

        runs_table_rows_html += (
            f'<tr class="run-row table-hover-row border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors cursor-pointer"'
            f' data-run-idx="{i}" data-evaluator="{_safe(r["evaluator_type"])}" onclick="showDetail({i})">'
            f'<td class="py-3 px-4 text-gray-400 text-sm whitespace-nowrap">{_safe(ts_display)}</td>'
            f'<td class="py-3 px-4">'
            f'<span class="font-mono text-violet-400 text-xs bg-violet-500/10 px-2 py-1 rounded border border-violet-500/20"'
            f' title="{_safe(r["filepath"])}">{_safe(fname_short)}</span>'
            f'</td>'
            f'<td class="py-3 px-4 text-gray-400 text-xs">{_safe(r["evaluator_type"])}</td>'
            f'<td class="py-3 px-4"><div class="flex flex-wrap gap-1">{metric_chips}</div></td>'
            f'<td class="py-3 px-4 whitespace-nowrap">'
            f'<span class="px-2 py-1 rounded-md text-[10px] uppercase font-bold tracking-wider border {result_cls}">'
            f'{_safe(result_label)}</span></td>'
            f'</tr>'
        )

    # ── Embed per-run data as JS JSON array ───────────────────────────────────
    runs_js_parts = []
    for r in runs:
        obj = {
            "filename": r["filename"],
            "label": r["label"],
            "timestamp": r["timestamp"],
            "evaluator_type": r["evaluator_type"],
            "total": r["total"],
            "passed": r["passed"],
            "failed": r["failed"],
            "pass_rate": r["pass_rate"],
            "metric_avgs": r["metric_avgs"],
            "metric_stdevs": r.get("metric_stdevs", {}),
            "metric_names": r["metric_names"],
            "metric_colors": r["metric_colors"],
            "analysis_summary": r["analysis_summary"],
            "test_cases": r.get("test_cases", []),
            "rows_html": r["rows_html"],
        }
        runs_js_parts.append(json.dumps(obj, ensure_ascii=False))
    runs_js_array = "[\n  " + ",\n  ".join(runs_js_parts) + "\n]"
    metric_color_map_json = json.dumps(metric_color)

    # ── HTML template ─────────────────────────────────────────────────────────
    html_template = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {{ font-family: 'Inter', sans-serif; background-color: #0b0c10; color: #e5e7eb; }}
        .glass-panel {{ background-color: #15161b; border: 1px solid #23252b; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); }}
        .nav-item {{ padding: 0.5rem 1rem; color: #9ca3af; font-size: 0.875rem; border-radius: 0.375rem; display: flex; align-items: center; gap: 0.75rem; transition: all 0.2s; }}
        .nav-item:hover, .nav-item.active {{ background-color: #1f2128; color: #f3f4f6; }}
        .font-mono {{ font-family: 'Courier New', monospace; }}
        .table-hover-row {{ transition: background-color 120ms ease, box-shadow 120ms ease; }}
        .table-hover-row:hover {{ background-color: rgba(71, 85, 105, 0.24) !important; box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.2); }}
        /* Keep native date/time pickers in dark black/white palette */
        .dark-date-input {{ color-scheme: dark; }}
        .dark-date-input::-webkit-calendar-picker-indicator {{ filter: invert(1); opacity: 0.9; }}
        .dark-date-input::-webkit-datetime-edit {{ color: #e5e7eb; }}
        /* Brighter text for test-case modal readability */
        #detail-case-modal .text-gray-500 {{ color: #d1d5db !important; }}
        #detail-case-modal .text-gray-400 {{ color: #e5e7eb !important; }}
        #detail-case-modal .text-gray-300 {{ color: #f3f4f6 !important; }}
        #detail-case-modal .text-gray-200 {{ color: #ffffff !important; }}
        #detail-case-modal .text-gray-100 {{ color: #ffffff !important; }}
        /* Persistent attribution footer for generated dashboards */
        .dashboard-attribution {{
            position: fixed;
            left: 1rem;
            bottom: 0.75rem;
            z-index: 40;
            font-size: 0.75rem;
            letter-spacing: 0.02em;
            color: #9ca3af;
            pointer-events: none;
        }}
        main {{ padding-bottom: 3rem; }}
        @media print {{
            aside {{ display: none !important; }}
            body {{ display: block !important; background: #fff !important; color: #000 !important; overflow: visible !important; height: auto !important; }}
            main {{ padding: 1rem !important; padding-bottom: 2rem !important; overflow: visible !important; height: auto !important; }}
            .glass-panel {{ background: #f9fafb !important; border: 1px solid #e5e7eb !important; box-shadow: none !important; }}
            .text-gray-400, .text-gray-500 {{ color: #374151 !important; }}
            .text-gray-300, .text-gray-200 {{ color: #111827 !important; }}
            .text-white {{ color: #000 !important; }}
            .no-print {{ display: none !important; }}
            tr.hidden {{ display: table-row !important; }}
            .dashboard-attribution {{
                left: 1.5cm;
                bottom: 1cm;
                color: #111827 !important;
            }}
            @page {{ margin: 1.5cm; size: A4; }}
        }}
    </style>
</head>
<body class="flex h-screen overflow-hidden">

    <!-- Sidebar -->
    <aside class="w-64 border-r border-gray-800 flex flex-col bg-[#0b0c10] overflow-y-auto shrink-0">
        <div class="p-5 flex items-center gap-3 border-b border-gray-800">
            <div class="w-7 h-7 bg-white rounded flex items-center justify-center text-black font-extrabold text-sm">C</div>
            <span class="font-bold text-white tracking-wide text-lg">Evaluation Tests</span>
        </div>
        <div class="p-3">
            <p class="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2 px-3 mt-4">Evaluation &amp; Observability</p>
            <nav class="space-y-1">
                <a href="#" onclick="showRuns(); return false;" class="nav-item active" id="nav-evaluation">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>
                    Evaluation
                </a>
                <a href="#" class="nav-item">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"></path></svg>
                    Datasets
                </a>
                <a href="#" class="nav-item">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>
                    Simulations
                </a>
            </nav>
        </div>
    </aside>

    <!-- Main area -->
    <main class="flex-1 overflow-y-auto bg-[#0b0c10]">

        <!-- ══════════════════════════════════════════════════════════════════
             VIEW: RUNS LIST (default)
        ════════════════════════════════════════════════════════════════════ -->
        <div id="view-runs" class="p-8">

            <!-- Top action bar -->
            <div class="flex items-center justify-between mb-6 no-print">
                <div>
                    <h1 class="text-white font-bold text-xl tracking-tight">Test Runs</h1>
                    <p class="text-gray-500 text-xs mt-1">Using evaluation data from <span id="runs-visible-caption-count">{total_runs}</span> displayed test run{"s" if total_runs != 1 else ""} · click a row to inspect</p>
                </div>
                <div class="flex items-center gap-3">
                    <div class="flex items-center gap-2">
                        <span class="text-[11px] uppercase tracking-wider text-gray-500 font-semibold">Evaluator</span>
                        <select id="evaluator-filter-select" class="bg-[#101117] border border-gray-700 rounded px-3 py-2 text-xs text-gray-200 min-w-[130px]"></select>
                    </div>
                    <button onclick="window.print()"
                        class="flex items-center gap-2 px-4 py-2 rounded-md bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium transition-colors border border-violet-500/50 cursor-pointer">
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"/></svg>
                        Export PDF
                    </button>
                </div>
            </div>

            <div class="glass-panel p-4 mb-6 no-print">
                <div class="flex flex-wrap items-center gap-2">
                    <button type="button" data-time-preset="custom" class="time-filter-btn px-3 py-1.5 rounded border border-gray-700 bg-[#101117] text-gray-200 text-xs font-medium">Custom</button>
                    <button type="button" data-time-preset="today" class="time-filter-btn px-3 py-1.5 rounded border border-gray-700 bg-[#101117] text-gray-200 text-xs font-medium">Today</button>
                    <button type="button" data-time-preset="yesterday" class="time-filter-btn px-3 py-1.5 rounded border border-gray-700 bg-[#101117] text-gray-200 text-xs font-medium">Yesterday</button>
                    <button type="button" data-time-preset="7d" class="time-filter-btn px-3 py-1.5 rounded border border-gray-700 bg-[#101117] text-gray-200 text-xs font-medium">7D</button>
                    <button type="button" data-time-preset="30d" class="time-filter-btn px-3 py-1.5 rounded border border-gray-700 bg-[#101117] text-gray-200 text-xs font-medium">30D</button>
                    <button type="button" data-time-preset="3m" class="time-filter-btn px-3 py-1.5 rounded border border-gray-700 bg-[#101117] text-gray-200 text-xs font-medium">3M</button>
                    <button type="button" data-time-preset="12m" class="time-filter-btn px-3 py-1.5 rounded border border-gray-700 bg-[#101117] text-gray-200 text-xs font-medium">12M</button>
                </div>
                <div id="time-filter-custom-panel" class="hidden mt-3 rounded border border-gray-800 bg-[#11131a] p-2">
                    <div class="flex w-full flex-wrap items-end justify-start gap-3">
                        <div class="space-y-2">
                            <label class="text-[11px] uppercase tracking-wider text-gray-500">Start</label>
                            <div class="flex items-center gap-2">
                                <input id="time-custom-start-date" type="text" inputmode="numeric" placeholder="DD/MM/YYYY" maxlength="10" class="bg-[#0b0c10] border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 w-32">
                                <input id="time-custom-start-time" type="text" inputmode="numeric" placeholder="HH:MM" maxlength="5" class="bg-[#0b0c10] border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 w-24">
                            </div>
                        </div>
                        <div class="space-y-2">
                            <label class="text-[11px] uppercase tracking-wider text-gray-500">End</label>
                            <div class="flex items-center gap-2">
                                <input id="time-custom-end-date" type="text" inputmode="numeric" placeholder="DD/MM/YYYY" maxlength="10" class="bg-[#0b0c10] border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 w-32">
                                <input id="time-custom-end-time" type="text" inputmode="numeric" placeholder="HH:MM" maxlength="5" class="bg-[#0b0c10] border border-gray-700 rounded px-2 py-1 text-xs text-gray-200 w-24">
                            </div>
                        </div>
                    </div>
                    <div class="mt-2">
                        <div id="time-filter-error" class="text-xs text-rose-400 min-h-[16px]"></div>
                        <div class="mt-2 flex items-center justify-start gap-2">
                            <button id="time-custom-reset" type="button" class="px-3 py-1.5 rounded border border-gray-700 text-xs text-gray-300 hover:text-white">Reset</button>
                            <button id="time-custom-apply" type="button" class="px-3 py-1.5 rounded border border-violet-500/50 bg-violet-600 text-xs text-white hover:bg-violet-500">Apply</button>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Stat cards -->
            <div class="flex gap-4 mb-6">
                <div class="glass-panel p-5 flex-1 flex items-center justify-between">
                    <div>
                        <div class="flex items-center gap-2">
                            <div class="w-2.5 h-2.5 rounded-full bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)]"></div>
                            <span class="text-gray-400 text-sm font-medium">System Status</span>
                        </div>
                        <div class="text-2xl font-bold mt-1 text-white">Healthy</div>
                    </div>
                    <div class="text-right text-xs text-gray-500 leading-relaxed">based on {total_cases} evaluations</div>
                </div>
                <div class="glass-panel p-5 flex-1 flex flex-col justify-center">
                    <span class="text-gray-400 text-sm font-medium">Total Runs</span>
                    <div id="stat-total-runs" class="text-2xl font-bold mt-1 text-white">{total_runs}</div>
                    <div class="text-xs text-gray-500 mt-1">source JSON files</div>
                </div>
                <div class="glass-panel p-5 flex-1 flex flex-col justify-center">
                    <span class="text-gray-400 text-sm font-medium">Overall Pass Rate</span>
                    <div id="stat-pass-rate" class="text-2xl font-bold mt-1 {'text-emerald-400' if overall_pass_rate >= 80 else 'text-yellow-400' if overall_pass_rate >= 50 else 'text-rose-400'}">{overall_pass_rate:.1f}%</div>
                    <div id="stat-pass-count" class="text-xs text-gray-500 mt-1">{total_passed}/{total_cases} passed</div>
                </div>
                <div class="glass-panel p-5 flex-1 flex flex-col justify-center">
                    <span class="text-gray-400 text-sm font-medium">Evaluation Mode</span>
                    <div class="mt-2 flex items-center gap-2">
                        <span id="stat-evaluator-badge" class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold border {evaluator_bg}">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            <span id="stat-evaluator-text">{evaluator_type}</span>
                        </span>
                    </div>
                    <div id="stat-evaluator-desc" class="text-xs text-gray-500 mt-1.5">{evaluator_desc}</div>
                </div>
            </div>

            <!-- Test Run Performance chart -->
            <div class="glass-panel p-6 mb-6">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h3 class="text-gray-200 font-semibold text-sm">Test Run Performance</h3>
                        <p class="text-gray-500 text-xs mt-0.5">Average metric score per run · one point per evaluation file · error bar = 1 SD</p>
                    </div>
                    <span class="px-2.5 py-1 rounded bg-[#2e2348] text-[#a78bfa] text-[10px] uppercase font-bold tracking-wider border border-[#4c3a7a]">All Metrics</span>
                </div>
                <div class="h-56 w-full">
                    <canvas id="runsChart"></canvas>
                </div>
            </div>

            <!-- Regression compare menu -->
            <div class="glass-panel p-5 mb-6 no-print">
                <div class="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                    <div>
                        <h3 class="text-gray-200 font-semibold text-sm">Compare Regressions</h3>
                        <p class="text-gray-500 text-xs mt-1">Pick two runs and open a side-by-side regression comparison</p>
                    </div>
                    <div class="flex flex-wrap items-center gap-2">
                        <select id="compare-left-select" class="bg-[#101117] border border-gray-700 rounded px-3 py-2 text-xs text-gray-200 min-w-[220px]"></select>
                        <span class="text-gray-500 text-xs font-semibold uppercase tracking-wider">vs</span>
                        <select id="compare-right-select" class="bg-[#101117] border border-gray-700 rounded px-3 py-2 text-xs text-gray-200 min-w-[220px]"></select>
                        <button id="compare-open-btn"
                            class="px-4 py-2 rounded-md bg-violet-600 hover:bg-violet-500 text-white text-xs font-semibold tracking-wide border border-violet-500/50 disabled:opacity-50 disabled:cursor-not-allowed">
                            Compare
                        </button>
                    </div>
                </div>
                <p id="compare-menu-hint" class="text-[11px] text-gray-500 mt-3"></p>
            </div>

            <!-- Runs table -->
            <div class="glass-panel p-0 overflow-hidden">
                <div class="p-5 border-b border-gray-800/50 flex items-center justify-between">
                    <div>
                        <h3 class="text-gray-200 font-medium text-sm">Showing <strong id="runs-showing-range" class="text-white">1 to {total_runs}</strong> of <strong id="runs-showing-total" class="text-white">{total_runs}</strong> Eval regression runs</h3>
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left">
                        <thead class="bg-[#15161b]">
                            <tr class="text-[11px] text-gray-500 uppercase tracking-wider border-b border-gray-800/50">
                                <th class="py-3 px-4 font-semibold w-36">Time</th>
                                <th class="py-3 px-4 font-semibold">Test Run ID</th>
                                <th class="py-3 px-4 font-semibold w-28">Evaluator</th>
                                <th class="py-3 px-4 font-semibold">Metric Scores</th>
                                <th class="py-3 px-4 font-semibold w-32">Test Result</th>
                            </tr>
                        </thead>
                        <tbody id="runs-table-body">
                            {runs_table_rows_html}
                        </tbody>
                    </table>
                </div>
                <div class="px-4 py-3 border-t border-gray-800/60 flex items-center justify-end text-xs text-gray-400">
                    <div class="flex items-center gap-4">
                        <span id="runs-range-text">0-0 of 0</span>
                        <div class="flex items-center gap-1">
                            <button id="runs-page-first" class="h-7 w-7 rounded border border-gray-700">|&lt;</button>
                            <button id="runs-page-prev" class="h-7 w-7 rounded border border-gray-700">&lt;</button>
                            <button id="runs-page-next" class="h-7 w-7 rounded border border-gray-700">&gt;</button>
                            <button id="runs-page-last" class="h-7 w-7 rounded border border-gray-700">&gt;|</button>
                        </div>
                        <span>Page <span id="runs-page-current">1</span> of <span id="runs-page-total">1</span></span>
                    </div>
                </div>
            </div>

        </div><!-- /view-runs -->


        <!-- ══════════════════════════════════════════════════════════════════
             VIEW: RUN DETAIL (hidden until a row is clicked)
        ════════════════════════════════════════════════════════════════════ -->
        <div id="view-detail" class="hidden p-8">

            <!-- Back button -->
            <button onclick="showRuns()"
                class="flex items-center gap-2 text-gray-400 hover:text-white text-sm font-medium mb-6 transition-colors cursor-pointer no-print">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                </svg>
                Back to Test Runs
            </button>

            <!-- Properties + Summary + Hyperparams -->
            <div class="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-6">

                <!-- Test Run Properties -->
                <div class="glass-panel p-6">
                    <h3 class="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-4">Test Run Properties</h3>
                    <div class="space-y-3 text-sm">
                        <div class="flex justify-between gap-4">
                            <span class="text-gray-500 shrink-0">Test Run ID</span>
                            <span id="detail-filename" class="text-violet-400 font-mono text-xs truncate"></span>
                        </div>
                        <div class="flex justify-between gap-4">
                            <span class="text-gray-500 shrink-0">Evaluator</span>
                            <span id="detail-evaluator" class="text-gray-300"></span>
                        </div>
                        <div class="flex justify-between gap-4">
                            <span class="text-gray-500 shrink-0">Executed At</span>
                            <span id="detail-timestamp" class="text-gray-300"></span>
                        </div>
                        <div class="flex justify-between gap-4">
                            <span class="text-gray-500 shrink-0">Total Cases</span>
                            <span id="detail-total" class="text-gray-300"></span>
                        </div>
                    </div>
                </div>

                <!-- Results Summary -->
                <div class="glass-panel p-6 flex flex-col items-center justify-center">
                    <h3 class="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-4 self-start">Results Summary</h3>
                    <div class="relative w-44 h-44">
                        <canvas id="detailDonutChart"></canvas>
                        <div class="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                            <span id="detail-pass-rate" class="text-3xl font-bold text-white"></span>
                            <span id="detail-pass-label" class="text-xs text-gray-400 mt-1"></span>
                        </div>
                    </div>
                    <p id="detail-fail-count" class="text-white font-medium mt-4 text-sm"></p>
                    <p class="text-gray-500 text-xs mt-1">based on JSON passed/status fields</p>
                </div>

                <!-- Metrics Analysis -->
                <div class="glass-panel p-6">
                    <h3 class="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-4">Metrics Analysis</h3>
                    <div id="detail-metric-cards" class="space-y-2"></div>
                </div>

            </div>

            <!-- Test Cases grid -->
            <div class="glass-panel p-0 mb-6 overflow-hidden">
                <div class="px-4 py-3 border-b border-gray-800/60 flex items-center justify-between">
                    <div id="detail-testcases-count" class="text-sm text-gray-300">Showing 0 to 0 of 0 test case(s)</div>
                    <div class="flex items-center gap-2 text-xs">
                        <button id="detail-download-csv" class="h-7 px-3 rounded border border-gray-700 text-gray-200 hover:text-white hover:border-gray-500 flex items-center gap-1">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v12m0 0l4-4m-4 4l-4-4M4 17v2a2 2 0 002 2h12a2 2 0 002-2v-2"></path></svg>
                            Download all as CSV
                        </button>
                        <button id="detail-sort-toggle" class="h-7 w-7 rounded border border-gray-700 text-gray-400 hover:text-white hover:border-gray-500">-</button>
                        <button class="h-7 w-7 rounded border border-gray-700 text-gray-400 hover:text-white hover:border-gray-500">...</button>
                        <div class="relative">
                            <button id="detail-columns-btn" class="h-7 px-2 rounded border border-gray-700 text-gray-300 hover:text-white hover:border-gray-500 flex items-center gap-1">
                                <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path></svg>
                                Columns
                            </button>
                            <div id="detail-columns-menu" class="hidden absolute right-0 mt-2 z-10 w-44 rounded-md border border-gray-700 bg-[#101117] p-2 space-y-1 text-xs">
                                <label class="flex items-center gap-2 text-gray-300 cursor-pointer"><input type="checkbox" data-col="name" checked> Name</label>
                                <label class="flex items-center gap-2 text-gray-300 cursor-pointer"><input type="checkbox" data-col="status" checked> Status</label>
                                <label class="flex items-center gap-2 text-gray-300 cursor-pointer"><input type="checkbox" data-col="input" checked> Input</label>
                                <label class="flex items-center gap-2 text-gray-300 cursor-pointer"><input type="checkbox" data-col="output" checked> Actual Output</label>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-sm">
                        <thead class="bg-[#15161b] border-b border-gray-800/60">
                            <tr class="text-[11px] text-gray-500 uppercase tracking-wider">
                                <th id="col-head-name" class="py-3 px-4 font-semibold w-44 cursor-pointer" data-sort="name">Name</th>
                                <th id="col-head-status" class="py-3 px-4 font-semibold w-44 cursor-pointer" data-sort="status">Status</th>
                                <th id="col-head-input" class="py-3 px-4 font-semibold cursor-pointer" data-sort="question">Input</th>
                                <th id="col-head-output" class="py-3 px-4 font-semibold">Actual Output</th>
                            </tr>
                        </thead>
                        <tbody id="detail-test-cases-grid-body"></tbody>
                    </table>
                </div>
                <div class="px-4 py-3 border-t border-gray-800/60 flex items-center justify-between text-xs text-gray-400">
                    <div class="flex items-center gap-3">
                        <span>Rows per page</span>
                        <select id="detail-rows-per-page" class="bg-[#101117] border border-gray-700 rounded px-2 py-1 text-gray-200">
                            <option value="5" selected>5</option>
                            <option value="10">10</option>
                            <option value="25">25</option>
                        </select>
                    </div>
                    <div class="flex items-center gap-4">
                        <span id="detail-range-text">0-0 of 0</span>
                        <div class="flex items-center gap-1">
                            <button id="detail-page-first" class="h-7 w-7 rounded border border-gray-700">|&lt;</button>
                            <button id="detail-page-prev" class="h-7 w-7 rounded border border-gray-700">&lt;</button>
                            <button id="detail-page-next" class="h-7 w-7 rounded border border-gray-700">&gt;</button>
                            <button id="detail-page-last" class="h-7 w-7 rounded border border-gray-700">&gt;|</button>
                        </div>
                    </div>
                </div>
            </div>

            <div id="detail-case-modal" class="hidden fixed inset-0 z-50">
                <div id="detail-case-backdrop" class="absolute inset-0 bg-black/70"></div>
                <div class="relative h-full w-full flex items-start justify-center p-6 overflow-y-auto">
                    <div class="w-full max-w-[1400px] bg-[#0d0f14] border border-gray-800 rounded-lg shadow-2xl">
                        <div class="px-5 py-3 border-b border-gray-800 flex items-center justify-between">
                            <div>
                                <div class="text-[10px] uppercase tracking-widest text-gray-500">Test Case Details</div>
                                <h4 id="detail-case-panel-title" class="text-gray-200 font-semibold text-sm mt-1">Test Case</h4>
                            </div>
                            <div class="flex items-center gap-2 text-xs">
                                <button id="detail-case-prev" class="px-2 py-1 rounded border border-gray-700 text-gray-300 hover:text-white hover:border-gray-500">Previous</button>
                                <button id="detail-case-next" class="px-2 py-1 rounded border border-gray-700 text-gray-300 hover:text-white hover:border-gray-500">Next</button>
                                <button id="detail-case-close" class="px-2 py-1 rounded border border-gray-700 text-gray-300 hover:text-white hover:border-gray-500">Close</button>
                            </div>
                        </div>
                        <div id="detail-case-panel-body" class="p-5"></div>
                    </div>
                </div>
            </div>

            <!-- Analysis Summary -->
            <div id="detail-analysis-summary" class="glass-panel p-6 mb-6 hidden">
                <div class="flex items-center justify-between mb-4">
                    <h3 class="text-gray-400 text-xs font-semibold uppercase tracking-widest">Analysis Summary</h3>
                    <button id="detail-analysis-toggle" class="text-xs text-gray-400 hover:text-white">Collapse</button>
                </div>
                <div id="detail-analysis-summary-body" class="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap"></div>
            </div>

        </div><!-- /view-detail -->

        <!-- ══════════════════════════════════════════════════════════════════
             VIEW: REGRESSION COMPARE (hidden until compare is selected)
        ════════════════════════════════════════════════════════════════════ -->
        <div id="view-compare" class="hidden p-8">
            <button onclick="showRuns()"
                class="flex items-center gap-2 text-gray-400 hover:text-white text-sm font-medium mb-6 transition-colors cursor-pointer no-print">
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/>
                </svg>
                Back to Test Runs
            </button>

            <div class="flex items-center justify-between mb-5">
                <div>
                    <h1 class="text-white font-bold text-xl tracking-tight">Regression Testing</h1>
                    <p id="compare-title" class="text-gray-500 text-xs mt-1">Compare two selected runs</p>
                </div>
                <div id="compare-summary-chip" class="text-xs px-2.5 py-1 rounded border border-gray-700 text-gray-300 bg-[#15161b]">Waiting for selection</div>
            </div>

            <div class="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
                <div class="glass-panel p-5">
                    <div class="flex items-center justify-between gap-3 mb-3">
                        <h3 id="compare-left-name" class="text-violet-400 text-sm font-semibold font-mono truncate"></h3>
                        <span id="compare-left-evaluator" class="text-[10px] px-2 py-0.5 rounded border border-violet-500/30 bg-violet-500/10 text-violet-300"></span>
                    </div>
                    <div class="grid grid-cols-3 gap-3 mb-4">
                        <div class="bg-[#101117] border border-gray-800 rounded p-3">
                            <div class="text-gray-500 text-[10px] uppercase">Pass rate</div>
                            <div id="compare-left-pass-rate" class="text-white text-lg font-bold mt-1">-</div>
                        </div>
                        <div class="bg-[#101117] border border-gray-800 rounded p-3">
                            <div class="text-gray-500 text-[10px] uppercase">Passed</div>
                            <div id="compare-left-passed" class="text-emerald-400 text-lg font-bold mt-1">-</div>
                        </div>
                        <div class="bg-[#101117] border border-gray-800 rounded p-3">
                            <div class="text-gray-500 text-[10px] uppercase">Failed</div>
                            <div id="compare-left-failed" class="text-rose-400 text-lg font-bold mt-1">-</div>
                        </div>
                    </div>
                    <div id="compare-left-metrics" class="space-y-2"></div>
                </div>

                <div class="glass-panel p-5">
                    <div class="flex items-center justify-between gap-3 mb-3">
                        <h3 id="compare-right-name" class="text-cyan-400 text-sm font-semibold font-mono truncate"></h3>
                        <span id="compare-right-evaluator" class="text-[10px] px-2 py-0.5 rounded border border-cyan-500/30 bg-cyan-500/10 text-cyan-300"></span>
                    </div>
                    <div class="grid grid-cols-3 gap-3 mb-4">
                        <div class="bg-[#101117] border border-gray-800 rounded p-3">
                            <div class="text-gray-500 text-[10px] uppercase">Pass rate</div>
                            <div id="compare-right-pass-rate" class="text-white text-lg font-bold mt-1">-</div>
                        </div>
                        <div class="bg-[#101117] border border-gray-800 rounded p-3">
                            <div class="text-gray-500 text-[10px] uppercase">Passed</div>
                            <div id="compare-right-passed" class="text-emerald-400 text-lg font-bold mt-1">-</div>
                        </div>
                        <div class="bg-[#101117] border border-gray-800 rounded p-3">
                            <div class="text-gray-500 text-[10px] uppercase">Failed</div>
                            <div id="compare-right-failed" class="text-rose-400 text-lg font-bold mt-1">-</div>
                        </div>
                    </div>
                    <div id="compare-right-metrics" class="space-y-2"></div>
                </div>
            </div>

            <div class="glass-panel p-0 overflow-hidden">
                <div class="p-4 border-b border-gray-800/60 flex items-center justify-between">
                    <h3 class="text-gray-200 text-sm font-medium">Matched Test Cases</h3>
                    <div class="flex items-center gap-4">
                        <label class="inline-flex items-center gap-2 text-[11px] text-gray-400 cursor-pointer select-none">
                            <input id="compare-filter-regressions" type="checkbox" class="accent-rose-500">
                            Only Regressions
                        </label>
                        <label class="inline-flex items-center gap-2 text-[11px] text-gray-400 cursor-pointer select-none">
                            <input id="compare-filter-changed" type="checkbox" class="accent-violet-500">
                            Only Changed Cases
                        </label>
                        <span id="compare-cases-count" class="text-[11px] text-gray-500"></span>
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="w-full text-left text-sm">
                        <thead class="bg-[#15161b] border-b border-gray-800/60">
                            <tr class="text-[11px] text-gray-500 uppercase tracking-wider">
                                <th class="py-3 px-4 font-semibold">Test Case</th>
                                <th class="py-3 px-4 font-semibold">Left Status</th>
                                <th class="py-3 px-4 font-semibold">Right Status</th>
                                <th class="py-3 px-4 font-semibold">Primary Metric Δ</th>
                            </tr>
                        </thead>
                        <tbody id="compare-cases-body"></tbody>
                    </table>
                </div>
                <div id="compare-case-panel" class="hidden border-t border-gray-800/60 bg-[#101117]">
                    <div class="px-4 py-3 border-b border-gray-800/60 flex items-center justify-between">
                        <h4 id="compare-case-title" class="text-gray-200 text-sm font-semibold">Test Case Details</h4>
                        <button id="compare-case-close" class="text-xs text-gray-400 hover:text-white">Close</button>
                    </div>
                    <div class="p-4 grid grid-cols-1 xl:grid-cols-2 gap-4">
                        <div>
                            <div class="text-[10px] uppercase tracking-widest text-violet-300 mb-2">Left Run</div>
                            <div id="compare-case-left" class="space-y-3"></div>
                        </div>
                        <div>
                            <div class="text-[10px] uppercase tracking-widest text-cyan-300 mb-2">Right Run</div>
                            <div id="compare-case-right" class="space-y-3"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div><!-- /view-compare -->

    </main>
    <div class="dashboard-attribution">Created by Daniel Hasid<br>Senior QA Auto & Eval Engineer</div>

    <script>
        // ── Embedded run data ─────────────────────────────────────────────────
        const RUNS_DATA = {runs_js_array};
        const METRIC_COLOR_MAP = {metric_color_map_json};
        const METRIC_FALLBACK_COLORS = ['#a855f7', '#fde047', '#3b82f6', '#10b981', '#f97316', '#22d3ee', '#f43f5e', '#14b8a6'];
        const dashboardState = {{
            selectedEvaluator: 'All',
            selectedRunIdx: -1,
            timeRangePreset: 'today',
            customStartDate: '',
            customEndDate: '',
            customStartTime: '',
            customEndTime: '',
        }};

        // ── Formatting helpers ────────────────────────────────────────────────
        function cleanSummaryMarkdown(text) {{
            if (!text) return '';
            return text
                .split(String.fromCharCode(10))
                .map(function(line) {{
                    let s = line.trim();
                    s = s.replace(/^#{{1,6}}\\s*/, '');
                    s = s.replace(/^[-*]\\s+/, '');
                    s = s.replace(/^>\\s+/, '');
                    s = s.replace(/\\*\\*(.*?)\\*\\*/g, '$1');
                    s = s.replace(/__(.*?)__/g, '$1');
                    s = s.replace(/`([^`]+)`/g, '$1');
                    return s;
                }})
                .join(String.fromCharCode(10))
                .trim();
        }}

        // ── Runs list chart ───────────────────────────────────────────────────
        let runsChart = null;
        const runsErrorBarsPlugin = {{
            id: 'runsErrorBars',
            afterDatasetsDraw(chart) {{
                const yScale = chart.scales && chart.scales.y;
                if (!yScale) return;
                const tooltipActive = chart.tooltip && typeof chart.tooltip.getActiveElements === 'function'
                    ? chart.tooltip.getActiveElements()
                    : [];
                const chartActive = typeof chart.getActiveElements === 'function'
                    ? chart.getActiveElements()
                    : [];
                const activeElements = (Array.isArray(tooltipActive) && tooltipActive.length)
                    ? tooltipActive
                    : chartActive;
                if (!Array.isArray(activeElements) || activeElements.length === 0) return;
                const hoveredPoints = new Set();
                activeElements.forEach(function(active) {{
                    const dsIdx = active && active.datasetIndex;
                    const ptIdx = active && active.index;
                    if (Number.isInteger(dsIdx) && Number.isInteger(ptIdx)) {{
                        hoveredPoints.add(dsIdx + ':' + ptIdx);
                    }}
                }});
                if (hoveredPoints.size === 0) return;
                const ctx = chart.ctx;
                const capWidth = 4;
                ctx.save();
                chart.data.datasets.forEach(function(dataset, datasetIndex) {{
                    if (!chart.isDatasetVisible(datasetIndex)) return;
                    const meta = chart.getDatasetMeta(datasetIndex);
                    if (!meta || !Array.isArray(meta.data)) return;
                    const sdValues = Array.isArray(dataset.errorBarData) ? dataset.errorBarData : [];
                    ctx.strokeStyle = dataset.borderColor || '#9ca3af';
                    ctx.lineWidth = 1.5;
                    meta.data.forEach(function(point, index) {{
                        if (!hoveredPoints.has(datasetIndex + ':' + index)) return;
                        const meanValue = dataset.data[index];
                        const sdValue = sdValues[index];
                        if (meanValue == null || sdValue == null) return;
                        const mean = Number(meanValue);
                        const sd = Number(sdValue);
                        if (!Number.isFinite(mean) || !Number.isFinite(sd) || sd <= 0) return;
                        const upper = Math.min(yScale.max, mean + sd);
                        const lower = Math.max(yScale.min, mean - sd);
                        const yTop = yScale.getPixelForValue(upper);
                        const yBottom = yScale.getPixelForValue(lower);
                        const x = point.x;
                        if (!Number.isFinite(x) || !Number.isFinite(yTop) || !Number.isFinite(yBottom)) return;
                        ctx.beginPath();
                        ctx.moveTo(x, yTop);
                        ctx.lineTo(x, yBottom);
                        ctx.moveTo(x - capWidth, yTop);
                        ctx.lineTo(x + capWidth, yTop);
                        ctx.moveTo(x - capWidth, yBottom);
                        ctx.lineTo(x + capWidth, yBottom);
                        ctx.stroke();
                    }});
                }});
                ctx.restore();
            }}
        }};
        try {{
            const runsChartEl = document.getElementById('runsChart');
            if (runsChartEl && typeof Chart !== 'undefined') {{
                const runsCtx = runsChartEl.getContext('2d');
                runsChart = new Chart(runsCtx, {{
                    type: 'line',
                    data: {{
                        labels: {json.dumps(run_labels)},
                        datasets: [
                            {runs_chart_datasets_str}
                        ]
                    }},
                    plugins: [runsErrorBarsPlugin],
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {{ intersect: false, mode: 'index' }},
                        scales: {{
                            x: {{
                                grid: {{ color: 'rgba(31,41,55,0.4)', drawBorder: false }},
                                ticks: {{ color: '#6b7280', font: {{ size: 11 }} }}
                            }},
                            y: {{
                                min: 0.0, max: 1.05,
                                grid: {{ color: 'rgba(31,41,55,0.4)', drawBorder: false }},
                                ticks: {{ color: '#6b7280', font: {{ size: 11 }} }}
                            }}
                        }},
                        plugins: {{
                            legend: {{ position: 'top', align: 'end', labels: {{ color: '#9ca3af', boxWidth: 10, usePointStyle: true, font: {{ size: 11 }} }} }},
                            tooltip: {{
                                backgroundColor: '#1f2028', borderColor: '#2e3040', borderWidth: 1,
                                titleColor: '#e5e7eb', bodyColor: '#9ca3af', padding: 10,
                                callbacks: {{
                                    label: function(context) {{
                                        const ds = context.dataset || {{}};
                                        const metricName = String(ds.label || '').split('  ')[0];
                                        const mean = context.parsed && context.parsed.y;
                                        if (mean == null || !Number.isFinite(Number(mean))) {{
                                            return metricName + ': N/A';
                                        }}
                                        const sdSeries = Array.isArray(ds.errorBarData) ? ds.errorBarData : [];
                                        const sdRaw = sdSeries[context.dataIndex];
                                        if (sdRaw == null || !Number.isFinite(Number(sdRaw))) {{
                                            return metricName + ': ' + Number(mean).toFixed(3);
                                        }}
                                        return metricName + ': ' + Number(mean).toFixed(3) + ' ± ' + Number(sdRaw).toFixed(3) + ' (1 SD)';
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            }} else if (runsChartEl && typeof Chart === 'undefined') {{
                console.warn('Chart.js failed to load (check CDN/network). Open via http:// for best results.');
            }}
        }} catch (e) {{
            console.error('Chart init failed:', e);
        }}

        function withAlpha(color, alpha) {{
            const safeAlpha = Number.isFinite(Number(alpha)) ? Math.max(0, Math.min(1, Number(alpha))) : 1;
            const raw = String(color || '').trim();
            if (!raw) return 'rgba(156,163,175,' + safeAlpha + ')';
            if (/^#([0-9a-f]{{3}}|[0-9a-f]{{6}})$/i.test(raw)) {{
                let hex = raw.slice(1);
                if (hex.length === 3) {{
                    hex = hex.split('').map(function(ch) {{ return ch + ch; }}).join('');
                }}
                const r = parseInt(hex.slice(0, 2), 16);
                const g = parseInt(hex.slice(2, 4), 16);
                const b = parseInt(hex.slice(4, 6), 16);
                return 'rgba(' + r + ',' + g + ',' + b + ',' + safeAlpha + ')';
            }}
            const rgbMatch = raw.match(/^rgba?\\(([^)]+)\\)$/i);
            if (rgbMatch) {{
                const parts = rgbMatch[1].split(',').map(function(part) {{ return part.trim(); }});
                if (parts.length >= 3) {{
                    return 'rgba(' + Number(parts[0]) + ',' + Number(parts[1]) + ',' + Number(parts[2]) + ',' + safeAlpha + ')';
                }}
            }}
            return raw;
        }}

        function getMetricColor(metricName, metricIndex) {{
            if (METRIC_COLOR_MAP && METRIC_COLOR_MAP[metricName]) return METRIC_COLOR_MAP[metricName];
            return METRIC_FALLBACK_COLORS[metricIndex % METRIC_FALLBACK_COLORS.length];
        }}

        function renderRunsChartForEntries(filteredEntries) {{
            if (!runsChart) return;
            const runs = filteredEntries.map(function(item) {{ return item.run; }});
            const labels = runs.map(function(run) {{ return getRunDateDisplay(run); }});
            const metricSet = new Set();
            runs.forEach(function(run) {{
                (run.metric_names || []).forEach(function(name) {{ metricSet.add(name); }});
            }});
            const metricNames = Array.from(metricSet).sort();
            const datasets = metricNames.map(function(metricName, metricIdx) {{
                const color = getMetricColor(metricName, metricIdx);
                const series = runs.map(function(run) {{
                    const val = run.metric_avgs ? run.metric_avgs[metricName] : null;
                    return (val == null || !Number.isFinite(Number(val))) ? null : Number(val);
                }});
                const sdSeries = runs.map(function(run) {{
                    const sd = run.metric_stdevs ? run.metric_stdevs[metricName] : null;
                    return (sd == null || !Number.isFinite(Number(sd))) ? null : Number(sd);
                }});
                return {{
                    label: metricName,
                    data: series,
                    errorBarData: sdSeries,
                    backgroundColor: color + '33',
                    borderColor: color,
                    borderWidth: 2.5,
                    tension: 0.4,
                    pointRadius: 5,
                    pointHoverRadius: 8,
                    fill: false,
                    spanGaps: true,
                }};
            }});

            runsChart.data.labels = labels;
            runsChart.data.datasets = datasets;
            runsChart.update('none');
        }}

        function updateRunsChartHighlightRange(start, endExclusive) {{
            if (!runsChart || !runsChart.data || !Array.isArray(runsChart.data.labels)) return;
            const labels = runsChart.data.labels;
            const total = labels.length;
            const startIdx = Math.max(0, Number(start) || 0);
            const endIdx = Math.max(startIdx, Math.min(total, Number(endExclusive) || 0));
            const isHighlighted = function(idx) {{
                return idx >= startIdx && idx < endIdx;
            }};

            (runsChart.data.datasets || []).forEach(function(ds) {{
                const baseColor = ds.__baseColor || ds.borderColor || '#9ca3af';
                ds.__baseColor = baseColor;
                ds.__basePointRadius = ds.__basePointRadius == null ? 3 : ds.__basePointRadius;
                ds.__baseBorderWidth = ds.__baseBorderWidth == null ? (typeof ds.borderWidth === 'number' ? ds.borderWidth : 2) : ds.__baseBorderWidth;

                const dimColor = withAlpha(baseColor, 0.2);
                ds.pointRadius = labels.map(function(_, idx) {{ return isHighlighted(idx) ? ds.__basePointRadius : 2; }});
                ds.pointHoverRadius = labels.map(function(_, idx) {{ return isHighlighted(idx) ? 6 : 3; }});
                ds.pointBackgroundColor = labels.map(function(_, idx) {{ return isHighlighted(idx) ? baseColor : dimColor; }});
                ds.pointBorderColor = ds.pointBackgroundColor.slice();
                ds.borderWidth = ds.__baseBorderWidth;
                ds.segment = {{
                    borderColor: function(ctx) {{
                        const i0 = Number(ctx && ctx.p0DataIndex);
                        const i1 = Number(ctx && ctx.p1DataIndex);
                        return (isHighlighted(i0) && isHighlighted(i1)) ? baseColor : dimColor;
                    }},
                    borderWidth: function(ctx) {{
                        const i0 = Number(ctx && ctx.p0DataIndex);
                        const i1 = Number(ctx && ctx.p1DataIndex);
                        return (isHighlighted(i0) && isHighlighted(i1)) ? ds.__baseBorderWidth : Math.max(1, ds.__baseBorderWidth - 0.5);
                    }}
                }};
            }});

            runsChart.update('none');
        }}

        // ── Detail donut chart instance ───────────────────────────────────────
        let detailDonut = null;
        const testCaseState = {{
            run: null,
            cases: [],
            sortedCases: [],
            page: 1,
            pageSize: 5,
            sortKey: 'name',
            sortDir: 'asc',
            columns: {{ name: true, status: true, input: true, output: true }},
            selectedCaseName: '',
            selectedCaseIndex: -1,
        }};
        const runsTableState = {{
            page: 1,
            pageSize: 5,
            visibleStart: 0,
            visibleEnd: 0,
        }};
        const compareState = {{
            leftIdx: 0,
            rightIdx: RUNS_DATA.length > 1 ? 1 : 0,
            onlyRegressions: false,
            onlyChanged: false,
            leftCaseMap: new Map(),
            rightCaseMap: new Map(),
            selectedCaseName: '',
        }};

        function escapeHtml(value) {{
            return String(value == null ? '' : value)
                .replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
        }}

        function truncateText(value, maxLen) {{
            const s = String(value || '');
            return s.length > maxLen ? (s.slice(0, maxLen - 3) + '...') : s;
        }}

        function csvEscape(value) {{
            const s = String(value == null ? '' : value);
            if (/[",\\n\\r]/.test(s)) {{
                return '"' + s.replace(/"/g, '""') + '"';
            }}
            return s;
        }}

        function formatVerboseLogs(logValue) {{
            if (Array.isArray(logValue)) {{
                return escapeHtml(logValue.join(String.fromCharCode(10)));
            }}
            if (logValue && typeof logValue === 'object') {{
                try {{
                    return escapeHtml(JSON.stringify(logValue, null, 2));
                }} catch (e) {{
                    return escapeHtml(String(logValue));
                }}
            }}
            return escapeHtml(String(logValue || ''));
        }}

        function statusOrder(status) {{
            return (status || '').toLowerCase() === 'failed' ? 0 : 1;
        }}

        function parseRunTimestamp(run) {{
            if (!run) return null;
            const raw = run.timestamp;
            if (!raw) return null;
            const dt = new Date(raw);
            if (Number.isFinite(dt.getTime())) return dt;
            const dmyMatch = String(raw).trim().match(/^(\d{{2}})_(\d{{2}})_(\d{{4}})$/);
            if (dmyMatch) {{
                const parsed = new Date(Number(dmyMatch[3]), Number(dmyMatch[2]) - 1, Number(dmyMatch[1]));
                return Number.isFinite(parsed.getTime()) ? parsed : null;
            }}
            return null;
        }}

        function pad2(value) {{
            return String(value).padStart(2, '0');
        }}

        function formatDateDdMmYyyy(dateObj) {{
            const dt = new Date(dateObj);
            if (!Number.isFinite(dt.getTime())) return '';
            return pad2(dt.getDate()) + '-' + pad2(dt.getMonth() + 1) + '-' + String(dt.getFullYear());
        }}

        function normalizeTime24h(value) {{
            const raw = String(value || '').trim();
            if (!raw) return '';
            const m = raw.match(/^(\d{{1,2}}):(\d{{2}})$/);
            if (!m) return '';
            const h = Number(m[1]);
            const min = Number(m[2]);
            if (!Number.isInteger(h) || !Number.isInteger(min)) return '';
            if (h < 0 || h > 23 || min < 0 || min > 59) return '';
            return pad2(h) + ':' + pad2(min);
        }}

        function normalizeDateDdMmYyyy(value) {{
            const raw = String(value || '').trim();
            if (!raw) return '';
            const m = raw.match(/^(\d{{1,2}})\/(\d{{1,2}})\/(\d{{4}})$/);
            if (!m) return '';
            const d = Number(m[1]);
            const mon = Number(m[2]);
            const y = Number(m[3]);
            if (!Number.isInteger(d) || !Number.isInteger(mon) || !Number.isInteger(y)) return '';
            if (y < 1000 || y > 9999 || mon < 1 || mon > 12 || d < 1 || d > 31) return '';
            const dt = new Date(y, mon - 1, d);
            if (dt.getFullYear() !== y || (dt.getMonth() + 1) !== mon || dt.getDate() !== d) return '';
            return pad2(d) + '/' + pad2(mon) + '/' + String(y);
        }}

        function formatDateForInput(dateObj) {{
            const dt = new Date(dateObj);
            if (!Number.isFinite(dt.getTime())) return '';
            return pad2(dt.getDate()) + '/' + pad2(dt.getMonth() + 1) + '/' + String(dt.getFullYear());
        }}

        function formatDateValue(value) {{
            const raw = String(value || '').trim();
            if (!raw) return '';
            const fromDateCtor = new Date(raw);
            if (Number.isFinite(fromDateCtor.getTime())) {{
                return formatDateDdMmYyyy(fromDateCtor);
            }}
            const isoMatch = raw.match(/^(\d{{4}})-(\d{{2}})-(\d{{2}})/);
            if (isoMatch) {{
                return isoMatch[3] + '-' + isoMatch[2] + '-' + isoMatch[1];
            }}
            const dmyMatch = raw.match(/^(\d{{2}})-(\d{{2}})-(\d{{4}})$/);
            if (dmyMatch) {{
                return raw;
            }}
            const dmyUnderscoreMatch = raw.match(/^(\d{{2}})_(\d{{2}})_(\d{{4}})$/);
            if (dmyUnderscoreMatch) {{
                return dmyUnderscoreMatch[1] + '-' + dmyUnderscoreMatch[2] + '-' + dmyUnderscoreMatch[3];
            }}
            return raw;
        }}

        function getRunDateDisplay(run) {{
            if (!run) return '';
            const fromTs = formatDateValue(run.timestamp || '');
            if (fromTs) return fromTs;
            return formatDateValue(run.label || '');
        }}

        function normalizeRunRowDateCells() {{
            document.querySelectorAll('tr.run-row').forEach(function(row) {{
                const idx = Number(row.getAttribute('data-run-idx'));
                if (Number.isNaN(idx)) return;
                const run = RUNS_DATA[idx];
                const cell = row.querySelector('td');
                const dateText = getRunDateDisplay(run);
                if (cell && dateText) {{
                    cell.textContent = dateText;
                }}
            }});
        }}

        function startOfDay(dateObj) {{
            const d = new Date(dateObj);
            d.setHours(0, 0, 0, 0);
            return d;
        }}

        function endOfDay(dateObj) {{
            const d = new Date(dateObj);
            d.setHours(23, 59, 59, 999);
            return d;
        }}

        function parseCustomRangeDateTime(dateValue, timeValue, isEnd) {{
            const normalizedDate = normalizeDateDdMmYyyy(dateValue);
            if (!normalizedDate) return null;
            const normalizedTime = normalizeTime24h(timeValue);
            if (timeValue && !normalizedTime) return null;
            const timePart = normalizedTime || '00:00';
            const dateMatch = normalizedDate.match(/^(\d{{2}})\/(\d{{2}})\/(\d{{4}})$/);
            if (!dateMatch) return null;
            const hhmm = timePart.split(':');
            const h = Number(hhmm[0]);
            const min = Number(hhmm[1]);
            const d = Number(dateMatch[1]);
            const mon = Number(dateMatch[2]);
            const y = Number(dateMatch[3]);
            const composed = new Date(y, mon - 1, d, h, min, 0, 0);
            if (!Number.isFinite(composed.getTime())) return null;
            if (isEnd) {{
                composed.setSeconds(59, 999);
            }}
            return composed;
        }}

        function getTimeRangeBounds() {{
            const now = new Date();
            const preset = dashboardState.timeRangePreset;
            if (preset === 'today') {{
                return {{ start: startOfDay(now), end: now }};
            }}
            if (preset === 'yesterday') {{
                const y = new Date(now);
                y.setDate(y.getDate() - 1);
                return {{ start: startOfDay(y), end: endOfDay(y) }};
            }}
            if (preset === '7d') {{
                const start = new Date(now);
                start.setDate(start.getDate() - 7);
                return {{ start: start, end: now }};
            }}
            if (preset === '30d') {{
                const start = new Date(now);
                start.setDate(start.getDate() - 30);
                return {{ start: start, end: now }};
            }}
            if (preset === '3m') {{
                const start = new Date(now);
                start.setMonth(start.getMonth() - 3);
                return {{ start: start, end: now }};
            }}
            if (preset === '12m') {{
                const start = new Date(now);
                start.setMonth(start.getMonth() - 12);
                return {{ start: start, end: now }};
            }}
            if (preset === 'custom') {{
                const start = parseCustomRangeDateTime(dashboardState.customStartDate, dashboardState.customStartTime, false);
                const end = parseCustomRangeDateTime(dashboardState.customEndDate, dashboardState.customEndTime, true);
                return {{ start: start, end: end }};
            }}
            return {{ start: null, end: null }};
        }}

        function normalizeEvaluatorName(name) {{
            const s = String(name || '').trim();
            return s || 'Unknown';
        }}

        function getEvaluatorBadgeClass(evalType) {{
            if (evalType === 'RAG') return 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400';
            if (evalType === 'GEval') return 'bg-violet-500/10 border-violet-500/30 text-violet-400';
            return 'bg-gray-500/10 border-gray-500/30 text-gray-400';
        }}

        function getEvaluatorDescription(evalType) {{
            if (evalType === 'RAG') return 'RAG Metrics (Faithfulness, Precision, Recall, Relevancy)';
            if (evalType === 'GEval') return 'GEval Custom Criteria';
            if (evalType === 'Mixed') return 'Mixed evaluator types across runs';
            return 'Unknown evaluator';
        }}

        function isRunVisibleByEvaluator(run) {{
            if (!run) return false;
            if (dashboardState.selectedEvaluator === 'All') return true;
            return normalizeEvaluatorName(run.evaluator_type) === dashboardState.selectedEvaluator;
        }}

        function isRunVisibleByTimeRange(run) {{
            if (!run) return false;
            const bounds = getTimeRangeBounds();
            if (!bounds.start && !bounds.end) return true;
            const runDt = parseRunTimestamp(run);
            if (!runDt) return false;
            if (bounds.start && runDt < bounds.start) return false;
            if (bounds.end && runDt > bounds.end) return false;
            return true;
        }}

        function isRunVisible(run) {{
            return isRunVisibleByEvaluator(run) && isRunVisibleByTimeRange(run);
        }}

        function getFilteredRunEntries() {{
            const out = [];
            RUNS_DATA.forEach(function(run, idx) {{
                if (isRunVisible(run)) {{
                    out.push({{ run: run, idx: idx }});
                }}
            }});
            return out;
        }}

        function setupEvaluatorFilterOptions() {{
            const selectEl = document.getElementById('evaluator-filter-select');
            if (!selectEl) return;
            const evaluatorSet = new Set();
            RUNS_DATA.forEach(function(run) {{
                evaluatorSet.add(normalizeEvaluatorName(run.evaluator_type));
            }});
            const options = ['All'].concat(Array.from(evaluatorSet).sort());
            selectEl.innerHTML = options.map(function(opt) {{
                return '<option value="' + escapeHtml(opt) + '">' + escapeHtml(opt) + '</option>';
            }}).join('');
            selectEl.value = dashboardState.selectedEvaluator;
        }}

        function updateTimePresetButtons() {{
            document.querySelectorAll('.time-filter-btn').forEach(function(btn) {{
                const preset = btn.getAttribute('data-time-preset');
                const active = preset === dashboardState.timeRangePreset;
                btn.classList.toggle('border-violet-500/50', active);
                btn.classList.toggle('bg-violet-600', active);
                btn.classList.toggle('text-white', active);
                btn.classList.toggle('border-gray-700', !active);
                btn.classList.toggle('bg-[#101117]', !active);
                btn.classList.toggle('text-gray-200', !active);
            }});
            const customPanel = document.getElementById('time-filter-custom-panel');
            if (customPanel) {{
                customPanel.classList.toggle('hidden', dashboardState.timeRangePreset !== 'custom');
            }}
        }}

        function syncTimeInputsFromState() {{
            const startDateEl = document.getElementById('time-custom-start-date');
            const endDateEl = document.getElementById('time-custom-end-date');
            const startTimeEl = document.getElementById('time-custom-start-time');
            const endTimeEl = document.getElementById('time-custom-end-time');
            if (startDateEl) startDateEl.value = dashboardState.customStartDate || '';
            if (endDateEl) endDateEl.value = dashboardState.customEndDate || '';
            if (startTimeEl) startTimeEl.value = dashboardState.customStartTime || '';
            if (endTimeEl) endTimeEl.value = dashboardState.customEndTime || '';
        }}

        function getCasesFromLegacyRows(run) {{
            if (!run.rows_html) return [];
            const parser = new DOMParser();
            const doc = parser.parseFromString('<table><tbody>' + run.rows_html + '</tbody></table>', 'text/html');
            const rows = doc.querySelectorAll('tr[data-target]');
            const out = [];
            rows.forEach(function(tr, idx) {{
                const cells = tr.querySelectorAll('td');
                const status = cells[0] ? cells[0].textContent.trim() : '';
                const input = cells[1] ? cells[1].textContent.trim() : '';
                const output = cells[2] ? cells[2].textContent.trim() : '';
                out.push({{
                    name: 'test_case_' + idx,
                    status: status === 'Passed' ? 'Success' : 'Failed',
                    question: input,
                    generated_answer: output,
                    expected_answer: '',
                    metadata: '',
                    timestamp: run.timestamp || '',
                    source_file: run.filename || '',
                    metric: '',
                    metric_kind: '',
                    metric_score: 0,
                    evaluation_metrics: {{}},
                }});
            }});
            return out;
        }}

        function getRunTestCases(run) {{
            if (Array.isArray(run.test_cases) && run.test_cases.length) {{
                return run.test_cases;
            }}
            return getCasesFromLegacyRows(run);
        }}

        function getRunOptionLabel(run, idx) {{
            const ts = getRunDateDisplay(run);
            return '#' + (idx + 1) + ' · ' + ts + ' · ' + run.filename;
        }}

        function downloadCurrentRunCsv() {{
            if (!testCaseState.run) return;
            const cases = getRunTestCases(testCaseState.run);
            const metricNames = Array.isArray(testCaseState.run.metric_names)
                ? testCaseState.run.metric_names
                : Array.from(new Set(cases.flatMap(function(c) {{
                    return Object.keys(c.evaluation_metrics || {{}});
                }}))).sort();

            const baseHeaders = [
                'Name',
                'Input',
                'Actual Output',
                'Expected Output',
                'Status',
                'Context',
                'Retrieval Context',
                'Tools Called',
                'Expected Tools',
                'Comments',
                'Additional Metadata',
                'Run Duration',
                'Error',
                'Skipped',
            ];

            const metricHeaders = [];
            metricNames.forEach(function(metricName) {{
                metricHeaders.push(metricName + ' Score');
                metricHeaders.push(metricName + ' Reason');
                metricHeaders.push(metricName + ' Threshold');
                metricHeaders.push(metricName + ' Success');
                metricHeaders.push(metricName + ' Error');
                metricHeaders.push(metricName + ' Strict Mode');
                metricHeaders.push(metricName + ' Evaluation Model');
            }});

            const headers = baseHeaders.concat(metricHeaders);
            const rows = [headers];

            cases.forEach(function(caseItem) {{
                const contextText = Array.isArray(caseItem.context)
                    ? caseItem.context.join(' | ')
                    : (caseItem.context || '');
                const retrievalContextText = Array.isArray(caseItem.retrieval_context)
                    ? caseItem.retrieval_context.join(' | ')
                    : (caseItem.retrieval_context || '');
                const baseRow = [
                    caseItem.name || '',
                    caseItem.question || '',
                    caseItem.generated_answer || '',
                    caseItem.expected_answer || '',
                    String(caseItem.status || '').toLowerCase() === 'failed' ? 'Failed' : 'Passed',
                    contextText,
                    retrievalContextText,
                    'N/A',
                    'N/A',
                    '',
                    caseItem.metadata || '',
                    'N/A',
                    '',
                    'false',
                ];

                const metricRow = [];
                metricNames.forEach(function(metricName) {{
                    const metric = (caseItem.evaluation_metrics || {{}})[metricName] || {{}};
                    metricRow.push(metric.score == null ? 'N/A' : metric.score);
                    metricRow.push(metric.reason || '');
                    metricRow.push(metric.threshold == null ? 'N/A' : metric.threshold);
                    metricRow.push(metric.passed == null ? 'N/A' : String(!!metric.passed));
                    metricRow.push(metric.error == null ? 'None' : String(metric.error));
                    metricRow.push(metric.strict_mode == null ? 'false' : String(!!metric.strict_mode));
                    metricRow.push(metric.evaluation_model || 'N/A');
                }});

                rows.push(baseRow.concat(metricRow));
            }});

            const csv = rows.map(function(row) {{
                return row.map(csvEscape).join(',');
            }}).join(String.fromCharCode(13, 10));

            const blob = new Blob([csv], {{ type: 'text/csv;charset=utf-8;' }});
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            const safeName = (testCaseState.run.filename || 'test_run').replace(/\\.json$/i, '');
            link.href = url;
            link.download = 'end_to_end_test_run_' + safeName + '.csv';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }}

        function statusBadgeClass(status) {{
            return String(status || '').toLowerCase() === 'failed'
                ? 'bg-rose-500/20 text-rose-400 border-rose-500/30'
                : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
        }}

        function normalizeStatusKind(status) {{
            const s = String(status || '').toLowerCase();
            if (!s || s === 'missing') return 'missing';
            if (s === 'failed' || s === 'fail' || s === 'error') return 'fail';
            if (s === 'success' || s === 'passed' || s === 'pass' || s === 'ok') return 'pass';
            return 'unknown';
        }}

        function metricCardHtml(metricName, value, color) {{
            const pct = Math.max(0, Math.min(100, Math.round((Number(value) || 0) * 100)));
            return '<div class="rounded border border-gray-800 p-2 bg-[#101117]">' +
                '<div class="flex items-center justify-between text-xs">' +
                '<span class="text-gray-300">' + escapeHtml(metricName) + '</span>' +
                '<span class="font-mono text-white">' + Number(value || 0).toFixed(3) + '</span>' +
                '</div>' +
                '<div class="mt-2 h-1.5 rounded bg-gray-800 overflow-hidden">' +
                '<div class="h-full rounded" style="width:' + pct + '%;background:' + escapeHtml(color || '#6b7280') + '"></div>' +
                '</div>' +
                '</div>';
        }}

        function metricDetailsHtml(metrics) {{
            const entries = Object.entries(metrics || {{}});
            if (!entries.length) {{
                return '<div class="text-xs text-gray-500">No metric details available.</div>';
            }}
            return entries.map(function(entry) {{
                const mName = entry[0];
                const mData = entry[1] || {{}};
                const passed = mData.passed === true ? 'true' : (mData.passed === false ? 'false' : 'n/a');
                return '<div class="rounded border border-gray-800/60 p-2">' +
                    '<div class="flex items-center justify-between text-xs">' +
                    '<span class="text-gray-200">' + escapeHtml(mName) + '</span>' +
                    '<span class="font-mono text-gray-300">' + Number(mData.score || 0).toFixed(3) + '</span>' +
                    '</div>' +
                    '<div class="mt-1 text-[10px] text-gray-500">passed: ' + escapeHtml(passed) + '</div>' +
                    '<div class="mt-1 text-xs text-gray-400 whitespace-pre-wrap">' + escapeHtml(mData.reason || '') + '</div>' +
                    '</div>';
            }}).join('');
        }}

        function renderCompareCaseColumn(containerId, caseItem) {{
            const el = document.getElementById(containerId);
            if (!el) return;
            if (!caseItem) {{
                el.innerHTML = '<div class="rounded border border-gray-800 p-3 text-xs text-gray-500">This test case is missing in this run.</div>';
                return;
            }}

            const metricLabel = caseItem.metric_kind ? (caseItem.metric_kind + ': ' + (caseItem.metric || 'n/a')) : (caseItem.metric || 'n/a');
            const scoreText = caseItem.metric_score != null ? Number(caseItem.metric_score).toFixed(3) : 'n/a';

            el.innerHTML =
                '<div class="rounded border border-gray-800 p-3 bg-[#0f1015]">' +
                    '<div class="flex items-center justify-between mb-2">' +
                        '<span class="px-2 py-0.5 rounded border text-[11px] ' + statusBadgeClass(caseItem.status || 'Unknown') + '">' + escapeHtml(caseItem.status || 'Unknown') + '</span>' +
                        '<span class="text-[10px] text-gray-500 font-mono">' + escapeHtml(caseItem.name || '') + '</span>' +
                    '</div>' +
                    '<div class="text-[11px] text-gray-500">Primary Metric</div>' +
                    '<div class="text-xs text-gray-300">' + escapeHtml(metricLabel) + ' · ' + escapeHtml(scoreText) + '</div>' +
                    '<div class="text-[11px] text-gray-500 mt-2">Input</div>' +
                    '<div class="text-xs text-gray-200 whitespace-pre-wrap">' + escapeHtml(caseItem.question || '') + '</div>' +
                    '<div class="text-[11px] text-gray-500 mt-2">Actual Output</div>' +
                    '<div class="text-xs text-gray-300 whitespace-pre-wrap">' + escapeHtml(caseItem.generated_answer || '') + '</div>' +
                    '<div class="text-[11px] text-gray-500 mt-2">Expected Output</div>' +
                    '<div class="text-xs text-gray-300 whitespace-pre-wrap">' + escapeHtml(caseItem.expected_answer || '') + '</div>' +
                    '<div class="text-[11px] text-gray-500 mt-2">Metadata</div>' +
                    '<div class="text-xs text-gray-400">' + escapeHtml(caseItem.metadata || '') + '</div>' +
                    '<div class="text-[11px] text-gray-500 mt-2">Metrics</div>' +
                    '<div class="space-y-2 mt-1">' + metricDetailsHtml(caseItem.evaluation_metrics || {{}}) + '</div>' +
                '</div>';
        }}

        function openCompareCaseDetails(caseName) {{
            const panel = document.getElementById('compare-case-panel');
            const title = document.getElementById('compare-case-title');
            if (!panel || !title) return;

            if (compareState.selectedCaseName === caseName && !panel.classList.contains('hidden')) {{
                closeCompareCaseDetails();
                return;
            }}

            compareState.selectedCaseName = caseName || '';
            title.textContent = (caseName || 'Test Case') + ' · Details';
            renderCompareCaseColumn('compare-case-left', compareState.leftCaseMap.get(caseName));
            renderCompareCaseColumn('compare-case-right', compareState.rightCaseMap.get(caseName));
            panel.classList.remove('hidden');
        }}

        function closeCompareCaseDetails() {{
            const panel = document.getElementById('compare-case-panel');
            if (panel) panel.classList.add('hidden');
            compareState.selectedCaseName = '';
        }}

        function renderCompareRunSide(prefix, run) {{
            document.getElementById(prefix + '-name').textContent = run.filename;
            document.getElementById(prefix + '-evaluator').textContent = run.evaluator_type || 'Unknown';
            document.getElementById(prefix + '-pass-rate').textContent = Number(run.pass_rate || 0).toFixed(1) + '%';
            document.getElementById(prefix + '-passed').textContent = String(run.passed || 0);
            document.getElementById(prefix + '-failed').textContent = String(run.failed || 0);

            const metricsEl = document.getElementById(prefix + '-metrics');
            metricsEl.innerHTML = '';
            run.metric_names.forEach(function(name) {{
                const avg = run.metric_avgs[name] != null ? run.metric_avgs[name] : 0;
                const color = run.metric_colors[name] || '#6b7280';
                metricsEl.innerHTML += metricCardHtml(name, avg, color);
            }});
        }}

        function renderCompareCases(leftRun, rightRun) {{
            const body = document.getElementById('compare-cases-body');
            const leftCases = getRunTestCases(leftRun);
            const rightCases = getRunTestCases(rightRun);

            const leftMap = new Map();
            leftCases.forEach(function(item) {{ leftMap.set(item.name, item); }});
            const rightMap = new Map();
            rightCases.forEach(function(item) {{ rightMap.set(item.name, item); }});
            compareState.leftCaseMap = leftMap;
            compareState.rightCaseMap = rightMap;

            const names = Array.from(new Set([].concat(leftCases.map(function(c) {{ return c.name; }}), rightCases.map(function(c) {{ return c.name; }})))).sort();
            let changedCount = 0;
            let regressionCount = 0;

            const rowsHtml = names.map(function(name) {{
                const l = leftMap.get(name);
                const r = rightMap.get(name);
                const lStatus = l ? (l.status || 'Unknown') : 'Missing';
                const rStatus = r ? (r.status || 'Unknown') : 'Missing';
                const lScore = l ? Number(l.metric_score || 0) : null;
                const rScore = r ? Number(r.metric_score || 0) : null;
                const lKind = normalizeStatusKind(lStatus);
                const rKind = normalizeStatusKind(rStatus);

                let deltaText = 'n/a';
                let deltaCls = 'text-gray-500';
                if (lScore != null && rScore != null) {{
                    const delta = rScore - lScore;
                    deltaText = (delta >= 0 ? '+' : '') + delta.toFixed(3);
                    deltaCls = delta > 0 ? 'text-emerald-400' : (delta < 0 ? 'text-rose-400' : 'text-gray-300');
                }}

                const changed = lKind !== rKind;
                const isRegression = (lKind === 'pass' && rKind === 'fail');
                if (changed) changedCount += 1;
                if (isRegression) regressionCount += 1;

                if (compareState.onlyChanged && !changed) return '';
                if (compareState.onlyRegressions && !isRegression) return '';

                return '<tr class="table-hover-row border-b border-gray-800/50 hover:bg-gray-800/30 cursor-pointer" data-case-name="' + escapeHtml(name) + '">' +
                    '<td class="py-3 px-4 text-gray-300 font-mono text-xs">' + escapeHtml(name) + '</td>' +
                    '<td class="py-3 px-4"><span class="px-2 py-0.5 rounded border text-[11px] ' + statusBadgeClass(lStatus) + '">' + escapeHtml(lStatus) + '</span></td>' +
                    '<td class="py-3 px-4"><span class="px-2 py-0.5 rounded border text-[11px] ' + statusBadgeClass(rStatus) + '">' + escapeHtml(rStatus) + '</span></td>' +
                    '<td class="py-3 px-4 font-mono text-xs ' + deltaCls + '">' + escapeHtml(deltaText) + '</td>' +
                    '</tr>';
            }}).join('');

            body.innerHTML = rowsHtml || '<tr><td colspan="4" class="py-6 px-4 text-center text-xs text-gray-500">No test cases match the selected filters.</td></tr>';
            body.querySelectorAll('tr[data-case-name]').forEach(function(rowEl) {{
                rowEl.addEventListener('click', function() {{
                    const caseName = rowEl.getAttribute('data-case-name') || '';
                    openCompareCaseDetails(caseName);
                }});
            }});

            const shownCount = body.querySelectorAll('tr').length;
            document.getElementById('compare-cases-count').textContent =
                shownCount + ' shown / ' + names.length + ' total · ' + regressionCount + ' regressions · ' + changedCount + ' status changes';
            return {{ changedCount: changedCount, regressionCount: regressionCount, shownCount: shownCount, totalCount: names.length }};
        }}

        function renderActiveCompare() {{
            const leftRun = RUNS_DATA[compareState.leftIdx];
            const rightRun = RUNS_DATA[compareState.rightIdx];
            if (!leftRun || !rightRun) return;
            if (!isRunVisible(leftRun) || !isRunVisible(rightRun)) return;

            const compareStats = renderCompareCases(leftRun, rightRun);
            const chip = document.getElementById('compare-summary-chip');
            chip.textContent =
                compareStats.regressionCount + ' regressions · ' + compareStats.changedCount + ' changes';
            chip.className = compareStats.regressionCount > 0
                ? 'text-xs px-2.5 py-1 rounded border border-rose-500/40 text-rose-300 bg-rose-500/10'
                : 'text-xs px-2.5 py-1 rounded border border-emerald-500/40 text-emerald-300 bg-emerald-500/10';
        }}

        function showCompare(leftIdx, rightIdx) {{
            const leftRun = RUNS_DATA[leftIdx];
            const rightRun = RUNS_DATA[rightIdx];
            if (!leftRun || !rightRun) return;
            if (!isRunVisible(leftRun) || !isRunVisible(rightRun)) return;

            compareState.leftIdx = leftIdx;
            compareState.rightIdx = rightIdx;

            document.getElementById('view-runs').classList.add('hidden');
            document.getElementById('view-detail').classList.add('hidden');
            document.getElementById('view-compare').classList.remove('hidden');

            document.getElementById('compare-title').textContent =
                'Comparing ' + getRunDateDisplay(leftRun) + ' vs ' + getRunDateDisplay(rightRun);

            renderCompareRunSide('compare-left', leftRun);
            renderCompareRunSide('compare-right', rightRun);
            closeCompareCaseDetails();
            renderActiveCompare();
        }}

        function showCompareFromSelectors() {{
            const leftEl = document.getElementById('compare-left-select');
            const rightEl = document.getElementById('compare-right-select');
            if (!leftEl || !rightEl) return;

            const leftIdx = Number(leftEl.value);
            const rightIdx = Number(rightEl.value);
            if (leftIdx === rightIdx) {{
                const hint = document.getElementById('compare-menu-hint');
                hint.textContent = 'Choose two different runs to compare.';
                return;
            }}
            showCompare(leftIdx, rightIdx);
        }}

        function renderSummaryCards(filteredEntries) {{
            const runs = filteredEntries.map(function(item) {{ return item.run; }});
            const totalRuns = runs.length;
            const totalCases = runs.reduce(function(acc, run) {{ return acc + Number(run.total || 0); }}, 0);
            const totalPassed = runs.reduce(function(acc, run) {{ return acc + Number(run.passed || 0); }}, 0);
            const overallPassRate = totalCases > 0 ? (totalPassed / totalCases * 100) : 0;

            const totalRunsEl = document.getElementById('stat-total-runs');
            const passRateEl = document.getElementById('stat-pass-rate');
            const passCountEl = document.getElementById('stat-pass-count');
            const evalBadgeEl = document.getElementById('stat-evaluator-badge');
            const evalTextEl = document.getElementById('stat-evaluator-text');
            const evalDescEl = document.getElementById('stat-evaluator-desc');
            const captionCountEl = document.getElementById('runs-visible-caption-count');

            if (captionCountEl) captionCountEl.textContent = String(totalRuns);
            if (totalRunsEl) totalRunsEl.textContent = String(totalRuns);
            if (passRateEl) {{
                passRateEl.textContent = overallPassRate.toFixed(1) + '%';
                passRateEl.classList.remove('text-emerald-400', 'text-yellow-400', 'text-rose-400');
                if (overallPassRate >= 80) passRateEl.classList.add('text-emerald-400');
                else if (overallPassRate >= 50) passRateEl.classList.add('text-yellow-400');
                else passRateEl.classList.add('text-rose-400');
            }}
            if (passCountEl) passCountEl.textContent = totalPassed + '/' + totalCases + ' passed';

            let evaluatorType = 'Unknown';
            const evalSet = new Set(runs.map(function(run) {{ return normalizeEvaluatorName(run.evaluator_type); }}));
            if (runs.length === 0) evaluatorType = 'Unknown';
            else if (evalSet.size === 1) evaluatorType = Array.from(evalSet)[0];
            else evaluatorType = 'Mixed';

            if (evalTextEl) evalTextEl.textContent = evaluatorType;
            if (evalDescEl) evalDescEl.textContent = getEvaluatorDescription(evaluatorType);
            if (evalBadgeEl) {{
                evalBadgeEl.classList.remove('bg-emerald-500/10', 'border-emerald-500/30', 'text-emerald-400',
                    'bg-violet-500/10', 'border-violet-500/30', 'text-violet-400',
                    'bg-gray-500/10', 'border-gray-500/30', 'text-gray-400');
                getEvaluatorBadgeClass(evaluatorType).split(' ').forEach(function(cls) {{
                    if (cls) evalBadgeEl.classList.add(cls);
                }});
            }}
        }}

        function setupCompareMenu() {{
            const leftEl = document.getElementById('compare-left-select');
            const rightEl = document.getElementById('compare-right-select');
            const openBtn = document.getElementById('compare-open-btn');
            const hint = document.getElementById('compare-menu-hint');
            if (!leftEl || !rightEl || !openBtn || !hint) return;

            const filteredEntries = getFilteredRunEntries();
            leftEl.innerHTML = '';
            rightEl.innerHTML = '';
            filteredEntries.forEach(function(entry) {{
                const run = entry.run;
                const idx = entry.idx;
                const optionLabel = getRunOptionLabel(run, idx);
                leftEl.innerHTML += '<option value="' + idx + '">' + escapeHtml(optionLabel) + '</option>';
                rightEl.innerHTML += '<option value="' + idx + '">' + escapeHtml(optionLabel) + '</option>';
            }});

            if (!isRunVisible(RUNS_DATA[compareState.leftIdx])) {{
                compareState.leftIdx = filteredEntries.length ? filteredEntries[0].idx : 0;
            }}
            if (!isRunVisible(RUNS_DATA[compareState.rightIdx]) || compareState.rightIdx === compareState.leftIdx) {{
                compareState.rightIdx = filteredEntries.length > 1 ? filteredEntries[1].idx : compareState.leftIdx;
            }}

            leftEl.value = String(compareState.leftIdx);
            rightEl.value = String(compareState.rightIdx);

            if (filteredEntries.length < 2) {{
                openBtn.disabled = true;
                hint.textContent = 'Need at least 2 filtered runs to enable compare mode.';
            }} else {{
                openBtn.disabled = false;
                hint.textContent = 'Tip: compare newest run against a baseline run to catch regressions fast.';
            }}

            if (!openBtn.dataset.bound) {{
                openBtn.onclick = showCompareFromSelectors;

                const onlyRegressions = document.getElementById('compare-filter-regressions');
                const onlyChanged = document.getElementById('compare-filter-changed');
                if (onlyRegressions) {{
                    onlyRegressions.checked = compareState.onlyRegressions;
                    onlyRegressions.addEventListener('change', function(e) {{
                        compareState.onlyRegressions = !!e.target.checked;
                        renderActiveCompare();
                    }});
                }}
                if (onlyChanged) {{
                    onlyChanged.checked = compareState.onlyChanged;
                    onlyChanged.addEventListener('change', function(e) {{
                        compareState.onlyChanged = !!e.target.checked;
                        closeCompareCaseDetails();
                        renderActiveCompare();
                    }});
                }}

                const compareCloseBtn = document.getElementById('compare-case-close');
                if (compareCloseBtn) {{
                    compareCloseBtn.addEventListener('click', closeCompareCaseDetails);
                }}
                openBtn.dataset.bound = '1';
            }}
        }}

        function renderRunsTablePage() {{
            const rows = Array.from(document.querySelectorAll('tr.run-row'));
            const filteredRows = rows.filter(function(row) {{
                const runIdx = Number(row.getAttribute('data-run-idx'));
                return isRunVisible(RUNS_DATA[runIdx]);
            }});
            const filteredEntries = getFilteredRunEntries();
            const total = filteredRows.length;
            const totalPages = Math.max(1, Math.ceil(total / runsTableState.pageSize));
            if (runsTableState.page > totalPages) runsTableState.page = totalPages;
            if (runsTableState.page < 1) runsTableState.page = 1;

            const start = (runsTableState.page - 1) * runsTableState.pageSize;
            const end = Math.min(start + runsTableState.pageSize, total);
            runsTableState.visibleStart = start;
            runsTableState.visibleEnd = end;

            rows.forEach(function(row) {{ row.style.display = 'none'; }});
            filteredRows.forEach(function(row, idx) {{
                row.style.display = (idx >= start && idx < end) ? '' : 'none';
            }});

            const rangeText = document.getElementById('runs-range-text');
            if (rangeText) {{
                rangeText.textContent = (total === 0 ? '0-0' : (start + 1) + '-' + end) + ' of ' + total;
            }}
            const showingRange = document.getElementById('runs-showing-range');
            if (showingRange) {{
                showingRange.textContent = (total === 0 ? '0 to 0' : (start + 1) + ' to ' + end);
            }}
            const showingTotal = document.getElementById('runs-showing-total');
            if (showingTotal) showingTotal.textContent = String(total);
            const pageCur = document.getElementById('runs-page-current');
            if (pageCur) pageCur.textContent = String(runsTableState.page);
            const pageTotal = document.getElementById('runs-page-total');
            if (pageTotal) pageTotal.textContent = String(totalPages);

            renderSummaryCards(filteredEntries);
            renderRunsChartForEntries(filteredEntries);
            updateRunsChartHighlightRange(start, end);
        }}

        function setupRunsTablePagination() {{
            const first = document.getElementById('runs-page-first');
            const prev = document.getElementById('runs-page-prev');
            const next = document.getElementById('runs-page-next');
            const last = document.getElementById('runs-page-last');
            if (!first || !prev || !next || !last) return;

            first.addEventListener('click', function() {{
                runsTableState.page = 1;
                renderRunsTablePage();
            }});
            prev.addEventListener('click', function() {{
                runsTableState.page = Math.max(1, runsTableState.page - 1);
                renderRunsTablePage();
            }});
            next.addEventListener('click', function() {{
                const totalRows = getFilteredRunEntries().length;
                const totalPages = Math.max(1, Math.ceil(totalRows / runsTableState.pageSize));
                runsTableState.page = Math.min(totalPages, runsTableState.page + 1);
                renderRunsTablePage();
            }});
            last.addEventListener('click', function() {{
                const totalRows = getFilteredRunEntries().length;
                runsTableState.page = Math.max(1, Math.ceil(totalRows / runsTableState.pageSize));
                renderRunsTablePage();
            }});

            renderRunsTablePage();
        }}

        function applyColumnVisibility() {{
            const map = {{
                name: 'col-head-name',
                status: 'col-head-status',
                input: 'col-head-input',
                output: 'col-head-output',
            }};
            Object.keys(map).forEach(function(col) {{
                const head = document.getElementById(map[col]);
                if (head) head.classList.toggle('hidden', !testCaseState.columns[col]);
                document.querySelectorAll('[data-col=\"' + col + '\"]').forEach(function(el) {{
                    el.classList.toggle('hidden', !testCaseState.columns[col]);
                }});
            }});
        }}

        function closeCaseModal() {{
            const modal = document.getElementById('detail-case-modal');
            if (modal) modal.classList.add('hidden');
            testCaseState.selectedCaseName = '';
            testCaseState.selectedCaseIndex = -1;
        }}

        function openCaseModalByIndex(caseIdx) {{
            const modal = document.getElementById('detail-case-modal');
            const title = document.getElementById('detail-case-panel-title');
            const body = document.getElementById('detail-case-panel-body');
            const prevBtn = document.getElementById('detail-case-prev');
            const nextBtn = document.getElementById('detail-case-next');
            if (!modal || !title || !body) return;

            const cases = testCaseState.sortedCases || [];
            const idx = Number(caseIdx);
            if (!cases.length || Number.isNaN(idx) || idx < 0 || idx >= cases.length) return;

            const caseItem = cases[idx] || {{}};
            testCaseState.selectedCaseIndex = idx;
            testCaseState.selectedCaseName = caseItem.name || '';
            title.textContent = (caseItem.name || 'Test Case') + ' · Details';

            const statusCls = String(caseItem.status || '').toLowerCase() === 'failed'
                ? 'bg-rose-500/20 text-rose-400 border-rose-500/30'
                : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';

            const metricEntries = Object.entries(caseItem.evaluation_metrics || {{}});
            const metricsHtml = metricEntries.length
                ? metricEntries.map(function(entry) {{
                    const metricName = entry[0];
                    const metric = entry[1] || {{}};
                    const passed = metric.passed === true ? 'success' : (metric.passed === false ? 'failed' : 'n/a');
                    const threshold = metric.threshold == null ? 'n/a' : Number(metric.threshold).toFixed(3);
                    return '<div class=\"rounded border border-gray-800/70 bg-[#0f1117] p-3\">' +
                        '<div class=\"flex items-center justify-between gap-3\">' +
                            '<div class=\"text-sm text-gray-200\">' + escapeHtml(metricName) + '</div>' +
                            '<div class=\"text-sm font-mono text-gray-100\">' + Number(metric.score || 0).toFixed(3) + '</div>' +
                        '</div>' +
                        '<div class=\"mt-2 flex flex-wrap gap-2 text-[10px] uppercase tracking-wider\">' +
                            '<span class=\"px-2 py-0.5 rounded border border-gray-700 text-gray-400\">threshold: ' + escapeHtml(threshold) + '</span>' +
                            '<span class=\"px-2 py-0.5 rounded border border-gray-700 text-gray-400\">status: ' + escapeHtml(passed) + '</span>' +
                        '</div>' +
                        '<div class=\"mt-2 text-xs text-gray-400 whitespace-pre-wrap\">' + escapeHtml(metric.reason || '') + '</div>' +
                    '</div>';
                }}).join('')
                : '<div class=\"text-xs text-gray-500\">No metric details available.</div>';

            const verboseLogsHtml = metricEntries.length
                ? metricEntries.map(function(entry, metricIdx) {{
                    const metricName = entry[0];
                    const metric = entry[1] || {{}};
                    const verboseRaw = metric.verbose_logs;
                    const criteria = metric.criteria || '';
                    const steps = metric.evaluation_steps;
                    const rubric = metric.rubric || '';
                    const model = metric.evaluation_model || '';
                    const verboseId = 'metric-verbose-' + idx + '-' + metricIdx;
                    let details = '';
                    if (criteria) {{
                        details += '<div><div class=\"text-gray-500\">Criteria:</div><div class=\"text-gray-300 mt-1 whitespace-pre-wrap\">' + escapeHtml(criteria) + '</div></div>';
                    }}
                    if (Array.isArray(steps) && steps.length) {{
                        details += '<div><div class=\"text-gray-500\">Evaluation Steps:</div><div class=\"text-gray-300 mt-1 whitespace-pre-wrap\">' + escapeHtml(steps.join(String.fromCharCode(10))) + '</div></div>';
                    }}
                    if (rubric) {{
                        details += '<div><div class=\"text-gray-500\">Rubric:</div><div class=\"text-gray-300 mt-1 whitespace-pre-wrap\">' + escapeHtml(typeof rubric === 'string' ? rubric : JSON.stringify(rubric, null, 2)) + '</div></div>';
                    }}
                    if (model) {{
                        details += '<div><div class=\"text-gray-500\">Evaluation Model:</div><div class=\"text-gray-300 mt-1\">' + escapeHtml(model) + '</div></div>';
                    }}
                    if (verboseRaw) {{
                        details += '<div><div class=\"text-gray-500\">Verbose Logs:</div><pre class=\"mt-1 text-gray-300 whitespace-pre-wrap font-mono text-[11px]\">' + formatVerboseLogs(verboseRaw) + '</pre></div>';
                    }}
                    if (!details) {{
                        details = '<div class=\"text-gray-500\">Verbose logs were not captured for this run. Re-run evaluation to populate this section.</div>';
                    }}
                    return '<div class=\"rounded border border-gray-800/70 bg-[#11141b] p-3\">' +
                        '<div class=\"flex items-center justify-between gap-3\">' +
                            '<div class=\"text-xs text-gray-200\">' + escapeHtml(metricName) + '</div>' +
                            '<button class=\"metric-verbose-toggle text-[11px] text-gray-400 hover:text-white\" data-target=\"' + verboseId + '\">Show verbose logs</button>' +
                        '</div>' +
                        '<div id=\"' + verboseId + '\" class=\"hidden mt-2 rounded border border-gray-800/60 bg-[#0c0e13] p-2 text-[11px] space-y-2\">' + details + '</div>' +
                    '</div>';
                }}).join('')
                : '<div class=\"rounded border border-gray-800/70 bg-[#11141b] p-3 text-xs text-gray-500\">Verbose logs are not available.</div>';

            body.innerHTML =
                '<div class=\"grid grid-cols-1 xl:grid-cols-3 gap-5\">' +
                    '<div class=\"xl:col-span-1 space-y-3\">' +
                        '<div class=\"rounded border border-gray-800/70 bg-[#11141b] p-3\"><div class=\"text-[10px] uppercase tracking-wider text-gray-500 mb-1\">Input Text</div><div class=\"text-xs text-gray-200 whitespace-pre-wrap\">' + escapeHtml(caseItem.question || '') + '</div></div>' +
                        '<div class=\"rounded border border-gray-800/70 bg-[#11141b] p-3\"><div class=\"text-[10px] uppercase tracking-wider text-gray-500 mb-1\">Actual Output</div><div class=\"text-xs text-gray-300 whitespace-pre-wrap\">' + escapeHtml(caseItem.generated_answer || '') + '</div></div>' +
                        '<div class=\"rounded border border-gray-800/70 bg-[#11141b] p-3\"><div class=\"text-[10px] uppercase tracking-wider text-gray-500 mb-1\">Expected Output</div><div class=\"text-xs text-gray-300 whitespace-pre-wrap\">' + escapeHtml(caseItem.expected_answer || '') + '</div></div>' +
                        (caseItem.retrieval_context && (Array.isArray(caseItem.retrieval_context) ? caseItem.retrieval_context.length > 0 : String(caseItem.retrieval_context).trim() !== '')
                            ? '<div class=\"rounded border border-gray-800/70 bg-[#11141b] p-3\"><div class=\"text-[10px] uppercase tracking-wider text-gray-500 mb-1\">Retrieval Context</div><div class=\"text-xs text-gray-300 whitespace-pre-wrap\">' + escapeHtml(Array.isArray(caseItem.retrieval_context) ? caseItem.retrieval_context.join('\\n\\n') : String(caseItem.retrieval_context)) + '</div></div>'
                            : '') +
                        '<div><div class=\"text-[10px] uppercase tracking-wider text-gray-500 mb-2\">Verbose Logs</div><div class=\"space-y-2\">' + verboseLogsHtml + '</div></div>' +
                    '</div>' +
                    '<div class=\"xl:col-span-2 space-y-4\">' +
                        '<div class=\"rounded border border-gray-800/70 bg-[#11141b] p-3\">' +
                            '<div class=\"grid grid-cols-2 lg:grid-cols-3 gap-3 text-xs\">' +
                                '<div><div class=\"text-gray-500 uppercase tracking-wider text-[10px]\">Test Case ID</div><div class=\"text-gray-200 font-mono mt-1\">' + escapeHtml(caseItem.name || '') + '</div></div>' +
                                '<div><div class=\"text-gray-500 uppercase tracking-wider text-[10px]\">Status</div><div class=\"mt-1\"><span class=\"px-2 py-0.5 rounded border text-[11px] ' + statusCls + '\">' + escapeHtml(caseItem.status || 'Unknown') + '</span></div></div>' +
                                '<div><div class=\"text-gray-500 uppercase tracking-wider text-[10px]\">Order</div><div class=\"text-gray-200 mt-1\">' + String(idx + 1) + '</div></div>' +
                                '<div><div class=\"text-gray-500 uppercase tracking-wider text-[10px]\">Timestamp</div><div class=\"text-gray-300 mt-1\">' + escapeHtml(formatDateValue(caseItem.timestamp || '')) + '</div></div>' +
                                '<div><div class=\"text-gray-500 uppercase tracking-wider text-[10px]\">Source File</div><div class=\"text-gray-300 mt-1\">' + escapeHtml(caseItem.source_file || '') + '</div></div>' +
                                '<div><div class=\"text-gray-500 uppercase tracking-wider text-[10px]\">Primary Metric</div><div class=\"text-gray-300 mt-1\">' + escapeHtml(caseItem.metric_kind || '') + ' ' + escapeHtml(caseItem.metric || '') + ' · ' + Number(caseItem.metric_score || 0).toFixed(3) + '</div></div>' +
                            '</div>' +
                            '<div class=\"mt-3 text-[11px] text-gray-500\">Metadata</div>' +
                            '<div class=\"text-xs text-gray-300 mt-1 whitespace-pre-wrap\">' + escapeHtml(caseItem.metadata || '') + '</div>' +
                        '</div>' +
                        '<div class=\"rounded border border-gray-800/70 bg-[#11141b] p-3\">' +
                            '<div class=\"text-[10px] uppercase tracking-wider text-gray-500 mb-2\">Metrics</div>' +
                            '<div class=\"space-y-2\">' + metricsHtml + '</div>' +
                        '</div>' +
                    '</div>' +
                '</div>';

            if (prevBtn) prevBtn.disabled = idx <= 0;
            if (nextBtn) nextBtn.disabled = idx >= (cases.length - 1);
            if (prevBtn) prevBtn.classList.toggle('opacity-40', idx <= 0);
            if (nextBtn) nextBtn.classList.toggle('opacity-40', idx >= (cases.length - 1));

            body.querySelectorAll('.metric-verbose-toggle').forEach(function(btn) {{
                btn.addEventListener('click', function() {{
                    const targetId = btn.getAttribute('data-target');
                    const target = targetId ? document.getElementById(targetId) : null;
                    if (!target) return;
                    const willShow = target.classList.contains('hidden');
                    target.classList.toggle('hidden', !willShow);
                    btn.textContent = willShow ? 'Hide verbose logs' : 'Show verbose logs';
                }});
            }});

            modal.classList.remove('hidden');
        }}

        function renderCaseGrid() {{
            const body = document.getElementById('detail-test-cases-grid-body');
            if (!testCaseState.run) {{
                body.innerHTML = '';
                return;
            }}

            const sorted = [...testCaseState.cases].sort(function(a, b) {{
                if (testCaseState.sortKey === 'status') {{
                    return (statusOrder(a.status) - statusOrder(b.status)) * (testCaseState.sortDir === 'asc' ? 1 : -1);
                }}
                const av = String(a[testCaseState.sortKey] || '').toLowerCase();
                const bv = String(b[testCaseState.sortKey] || '').toLowerCase();
                if (av === bv) return 0;
                return (av > bv ? 1 : -1) * (testCaseState.sortDir === 'asc' ? 1 : -1);
            }});
            testCaseState.sortedCases = sorted;

            const total = sorted.length;
            const maxPage = Math.max(1, Math.ceil(total / testCaseState.pageSize));
            if (testCaseState.page > maxPage) testCaseState.page = maxPage;
            const start = (testCaseState.page - 1) * testCaseState.pageSize;
            const end = Math.min(start + testCaseState.pageSize, total);
            const pageRows = sorted.slice(start, end);

            document.getElementById('detail-testcases-count').textContent =
                'Showing ' + (total === 0 ? 0 : (start + 1)) + ' to ' + end + ' of ' + total + ' test case(s)';
            document.getElementById('detail-range-text').textContent =
                (total === 0 ? '0-0' : (start + 1) + '-' + end) + ' of ' + total;

            body.innerHTML = pageRows.map(function(item, idx) {{
                const statusBadges = (item.status || '').toLowerCase() === 'failed'
                    ? 'bg-rose-500/20 text-rose-400 border-rose-500/30'
                    : 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
                return '<tr class=\"table-hover-row border-b border-gray-800/50 hover:bg-gray-800/40 cursor-pointer\" data-case-index=\"' + (start + idx) + '\">' +
                    '<td data-col=\"name\" class=\"py-3 px-4 text-gray-300\">' + escapeHtml(item.name) + '</td>' +
                    '<td data-col=\"status\" class=\"py-3 px-4\"><span class=\"px-2 py-0.5 rounded-full text-[11px] border ' + statusBadges + '\">' + escapeHtml(item.status) + '</span></td>' +
                    '<td data-col=\"input\" class=\"py-3 px-4 text-gray-300\" title=\"' + escapeHtml(item.question) + '\">' + escapeHtml(truncateText(item.question, 90)) + '</td>' +
                    '<td data-col=\"output\" class=\"py-3 px-4 text-gray-400\" title=\"' + escapeHtml(item.generated_answer) + '\">' + escapeHtml(truncateText(item.generated_answer, 110)) + '</td>' +
                    '</tr>';
            }}).join('');

            document.querySelectorAll('#detail-test-cases-grid-body tr[data-case-index]').forEach(function(rowEl) {{
                rowEl.addEventListener('click', function() {{
                    const idx = Number(rowEl.getAttribute('data-case-index'));
                    openCaseModalByIndex(idx);
                }});
            }});

            applyColumnVisibility();
        }}

        function initCaseGrid(run) {{
            testCaseState.run = run;
            testCaseState.cases = getRunTestCases(run);
            testCaseState.sortedCases = [];
            testCaseState.page = 1;
            testCaseState.selectedCaseName = '';
            testCaseState.selectedCaseIndex = -1;
            closeCaseModal();
            renderCaseGrid();
        }}

        function setupCaseGridEvents() {{
            document.getElementById('detail-rows-per-page').addEventListener('change', function(e) {{
                testCaseState.pageSize = Number(e.target.value) || 10;
                testCaseState.page = 1;
                renderCaseGrid();
            }});
            document.getElementById('detail-page-first').addEventListener('click', function() {{
                testCaseState.page = 1;
                renderCaseGrid();
            }});
            document.getElementById('detail-page-prev').addEventListener('click', function() {{
                testCaseState.page = Math.max(1, testCaseState.page - 1);
                renderCaseGrid();
            }});
            document.getElementById('detail-page-next').addEventListener('click', function() {{
                const totalPages = Math.max(1, Math.ceil(testCaseState.cases.length / testCaseState.pageSize));
                testCaseState.page = Math.min(totalPages, testCaseState.page + 1);
                renderCaseGrid();
            }});
            document.getElementById('detail-page-last').addEventListener('click', function() {{
                testCaseState.page = Math.max(1, Math.ceil(testCaseState.cases.length / testCaseState.pageSize));
                renderCaseGrid();
            }});

            document.querySelectorAll('#view-detail th[data-sort]').forEach(function(th) {{
                th.addEventListener('click', function() {{
                    const key = th.getAttribute('data-sort');
                    if (testCaseState.sortKey === key) {{
                        testCaseState.sortDir = testCaseState.sortDir === 'asc' ? 'desc' : 'asc';
                    }} else {{
                        testCaseState.sortKey = key;
                        testCaseState.sortDir = 'asc';
                    }}
                    renderCaseGrid();
                }});
            }});

            document.getElementById('detail-sort-toggle').addEventListener('click', function() {{
                testCaseState.sortDir = testCaseState.sortDir === 'asc' ? 'desc' : 'asc';
                renderCaseGrid();
            }});

            const columnsBtn = document.getElementById('detail-columns-btn');
            const columnsMenu = document.getElementById('detail-columns-menu');
            columnsBtn.addEventListener('click', function(e) {{
                e.stopPropagation();
                columnsMenu.classList.toggle('hidden');
            }});
            columnsMenu.querySelectorAll('input[type=\"checkbox\"]').forEach(function(cb) {{
                cb.addEventListener('change', function() {{
                    const col = cb.getAttribute('data-col');
                    testCaseState.columns[col] = cb.checked;
                    applyColumnVisibility();
                }});
            }});
            document.addEventListener('click', function(e) {{
                if (!columnsMenu.contains(e.target) && e.target !== columnsBtn) {{
                    columnsMenu.classList.add('hidden');
                }}
            }});

            const caseCloseBtn = document.getElementById('detail-case-close');
            const casePrevBtn = document.getElementById('detail-case-prev');
            const caseNextBtn = document.getElementById('detail-case-next');
            const caseBackdrop = document.getElementById('detail-case-backdrop');

            if (caseCloseBtn) {{
                caseCloseBtn.addEventListener('click', closeCaseModal);
            }}
            const downloadCsvBtn = document.getElementById('detail-download-csv');
            if (downloadCsvBtn) {{
                downloadCsvBtn.addEventListener('click', downloadCurrentRunCsv);
            }}
            if (caseBackdrop) {{
                caseBackdrop.addEventListener('click', closeCaseModal);
            }}
            if (casePrevBtn) {{
                casePrevBtn.addEventListener('click', function() {{
                    if (testCaseState.selectedCaseIndex > 0) {{
                        openCaseModalByIndex(testCaseState.selectedCaseIndex - 1);
                    }}
                }});
            }}
            if (caseNextBtn) {{
                caseNextBtn.addEventListener('click', function() {{
                    const maxIdx = (testCaseState.sortedCases || []).length - 1;
                    if (testCaseState.selectedCaseIndex >= 0 && testCaseState.selectedCaseIndex < maxIdx) {{
                        openCaseModalByIndex(testCaseState.selectedCaseIndex + 1);
                    }}
                }});
            }}

            document.addEventListener('keydown', function(e) {{
                if (e.key === 'Escape') {{
                    closeCaseModal();
                    return;
                }}
                if (e.key === 'ArrowLeft') {{
                    if (testCaseState.selectedCaseIndex > 0) {{
                        openCaseModalByIndex(testCaseState.selectedCaseIndex - 1);
                    }}
                }}
                if (e.key === 'ArrowRight') {{
                    const maxIdx = (testCaseState.sortedCases || []).length - 1;
                    if (testCaseState.selectedCaseIndex >= 0 && testCaseState.selectedCaseIndex < maxIdx) {{
                        openCaseModalByIndex(testCaseState.selectedCaseIndex + 1);
                    }}
                }}
            }});
        }}

        // ── Navigation ────────────────────────────────────────────────────────
        function showRuns() {{
            closeCaseModal();
            document.getElementById('view-runs').classList.remove('hidden');
            document.getElementById('view-detail').classList.add('hidden');
            document.getElementById('view-compare').classList.add('hidden');
        }}

        function applyDashboardFilters(resetPagination) {{
            if (resetPagination) {{
                runsTableState.page = 1;
            }}
            updateTimePresetButtons();
            setupCompareMenu();
            renderRunsTablePage();

            const activeRun = RUNS_DATA[dashboardState.selectedRunIdx];
            if (document.getElementById('view-detail') && !document.getElementById('view-detail').classList.contains('hidden')) {{
                if (!activeRun || !isRunVisible(activeRun)) {{
                    showRuns();
                }}
            }}
            if (document.getElementById('view-compare') && !document.getElementById('view-compare').classList.contains('hidden')) {{
                const leftRun = RUNS_DATA[compareState.leftIdx];
                const rightRun = RUNS_DATA[compareState.rightIdx];
                if (!leftRun || !rightRun || !isRunVisible(leftRun) || !isRunVisible(rightRun)) {{
                    showRuns();
                }}
            }}
        }}

        function setupTimeFilterControls() {{
            function asTimeInputValue(dateObj) {{
                return pad2(dateObj.getHours()) + ':' + pad2(dateObj.getMinutes());
            }}
            function attachDateAutoFormat(inputEl) {{
                if (!inputEl) return;
                inputEl.addEventListener('input', function() {{
                    const digits = String(inputEl.value || '').replace(/\D/g, '').slice(0, 8);
                    if (!digits) {{
                        inputEl.value = '';
                        return;
                    }}
                    if (digits.length <= 2) {{
                        inputEl.value = digits.length === 2 ? (digits + '/') : digits;
                        return;
                    }}
                    if (digits.length <= 4) {{
                        inputEl.value = digits.slice(0, 2) + '/' + digits.slice(2) + (digits.length === 4 ? '/' : '');
                        return;
                    }}
                    inputEl.value = digits.slice(0, 2) + '/' + digits.slice(2, 4) + '/' + digits.slice(4);
                }});
                inputEl.addEventListener('blur', function() {{
                    const normalized = normalizeDateDdMmYyyy(inputEl.value || '');
                    if (normalized) inputEl.value = normalized;
                }});
            }}
            function attachTimeAutoFormat(inputEl) {{
                if (!inputEl) return;
                inputEl.addEventListener('input', function() {{
                    const digits = String(inputEl.value || '').replace(/\\D/g, '').slice(0, 4);
                    if (!digits) {{
                        inputEl.value = '';
                        return;
                    }}
                    if (digits.length <= 2) {{
                        inputEl.value = digits.length === 2 ? (digits + ':') : digits;
                        if (digits.length === 2) {{
                            inputEl.setSelectionRange(3, 3);
                        }}
                        return;
                    }}
                    inputEl.value = digits.slice(0, 2) + ':' + digits.slice(2);
                }});
                inputEl.addEventListener('keydown', function(e) {{
                    if (e.key !== 'Backspace') return;
                    const pos = inputEl.selectionStart;
                    if (pos === 3 && inputEl.selectionEnd === 3 && String(inputEl.value || '').charAt(2) === ':') {{
                        e.preventDefault();
                        inputEl.value = String(inputEl.value || '').slice(0, 2);
                        inputEl.setSelectionRange(2, 2);
                    }}
                }});
                inputEl.addEventListener('blur', function() {{
                    const normalized = normalizeTime24h(inputEl.value || '');
                    if (normalized) inputEl.value = normalized;
                }});
            }}
            const errEl = document.getElementById('time-filter-error');
            document.querySelectorAll('.time-filter-btn').forEach(function(btn) {{
                btn.addEventListener('click', function() {{
                    const preset = String(btn.getAttribute('data-time-preset') || 'today').toLowerCase();
                    dashboardState.timeRangePreset = preset;
                    if (errEl) errEl.textContent = '';
                    if (preset !== 'custom') {{
                        applyDashboardFilters(true);
                    }} else {{
                        updateTimePresetButtons();
                        const customStartDateEl = document.getElementById('time-custom-start-date');
                        if (customStartDateEl) {{
                            customStartDateEl.focus();
                            // Chromium supports showPicker(); fallback focus still works.
                            if (typeof customStartDateEl.showPicker === 'function') {{
                                try {{
                                    customStartDateEl.showPicker();
                                }} catch (e) {{
                                    // Ignore user-gesture or browser restrictions.
                                }}
                            }}
                        }}
                    }}
                }});
            }});

            const startDateEl = document.getElementById('time-custom-start-date');
            const endDateEl = document.getElementById('time-custom-end-date');
            const startTimeEl = document.getElementById('time-custom-start-time');
            const endTimeEl = document.getElementById('time-custom-end-time');
            const applyBtn = document.getElementById('time-custom-apply');
            const resetBtn = document.getElementById('time-custom-reset');
            attachDateAutoFormat(startDateEl);
            attachDateAutoFormat(endDateEl);
            attachTimeAutoFormat(startTimeEl);
            attachTimeAutoFormat(endTimeEl);

            [startDateEl, endDateEl, startTimeEl, endTimeEl].forEach(function(inputEl) {{
                if (!inputEl) return;
                inputEl.addEventListener('focus', function() {{
                    if (typeof inputEl.showPicker === 'function') {{
                        try {{
                            inputEl.showPicker();
                        }} catch (e) {{
                            // Ignore browser-specific restrictions.
                        }}
                    }}
                }});
            }});

            if (applyBtn) {{
                applyBtn.addEventListener('click', function() {{
                    dashboardState.customStartDate = startDateEl ? normalizeDateDdMmYyyy(startDateEl.value || '') : '';
                    dashboardState.customEndDate = endDateEl ? normalizeDateDdMmYyyy(endDateEl.value || '') : '';
                    dashboardState.customStartTime = startTimeEl ? normalizeTime24h(startTimeEl.value || '') : '';
                    dashboardState.customEndTime = endTimeEl ? normalizeTime24h(endTimeEl.value || '') : '';

                    if ((startDateEl && String(startDateEl.value || '').trim() && !dashboardState.customStartDate) ||
                        (endDateEl && String(endDateEl.value || '').trim() && !dashboardState.customEndDate)) {{
                        if (errEl) errEl.textContent = 'Use date format DD/MM/YYYY.';
                        return;
                    }}
                    if ((startTimeEl && String(startTimeEl.value || '').trim() && !dashboardState.customStartTime) ||
                        (endTimeEl && String(endTimeEl.value || '').trim() && !dashboardState.customEndTime)) {{
                        if (errEl) errEl.textContent = 'Use 24H time format HH:MM (00:00-23:59).';
                        return;
                    }}

                    const bounds = getTimeRangeBounds();
                    if (!bounds.start || !bounds.end) {{
                        if (errEl) errEl.textContent = 'Please set both start and end date/time.';
                        return;
                    }}
                    if (bounds.start > bounds.end) {{
                        if (errEl) errEl.textContent = 'Start time must be before end time.';
                        return;
                    }}
                    if (errEl) errEl.textContent = '';
                    applyDashboardFilters(true);
                }});
            }}

            if (resetBtn) {{
                resetBtn.addEventListener('click', function() {{
                    const now = new Date();
                    const ddMmYyyy = formatDateForInput(now);
                    dashboardState.customStartDate = ddMmYyyy;
                    dashboardState.customEndDate = ddMmYyyy;
                    dashboardState.customStartTime = '00:00';
                    dashboardState.customEndTime = asTimeInputValue(now);
                    syncTimeInputsFromState();
                    if (errEl) errEl.textContent = '';
                }});
            }}

            const defaultEnd = new Date();
            const defaultStart = new Date(defaultEnd);
            defaultStart.setDate(defaultStart.getDate() - 30);
            dashboardState.customStartDate = formatDateForInput(defaultStart);
            dashboardState.customEndDate = formatDateForInput(defaultEnd);
            dashboardState.customStartTime = '00:00';
            dashboardState.customEndTime = '23:59';
            syncTimeInputsFromState();
            updateTimePresetButtons();
        }}

        function showDetail(idx) {{
            const run = RUNS_DATA[idx];
            if (!run || !isRunVisible(run)) return;
            dashboardState.selectedRunIdx = idx;
            document.getElementById('view-runs').classList.add('hidden');
            document.getElementById('view-detail').classList.remove('hidden');

            // Properties card
            document.getElementById('detail-filename').textContent = run.filename;
            document.getElementById('detail-evaluator').textContent = run.evaluator_type;
            const ts = getRunDateDisplay(run);
            document.getElementById('detail-timestamp').textContent = ts;
            document.getElementById('detail-total').textContent = run.total + ' test cases';

            // Pass rate donut
            document.getElementById('detail-pass-rate').textContent = run.pass_rate.toFixed(1) + '%';
            document.getElementById('detail-pass-label').textContent = run.passed + '/' + run.total + ' passed';
            document.getElementById('detail-fail-count').textContent = run.failed + ' failing test case' + (run.failed !== 1 ? 's' : '');

            if (detailDonut) {{ detailDonut.destroy(); detailDonut = null; }}
            if (typeof Chart !== 'undefined') {{
                const dCtx = document.getElementById('detailDonutChart').getContext('2d');
                detailDonut = new Chart(dCtx, {{
                    type: 'doughnut',
                    data: {{
                        labels: ['Passed', 'Failed'],
                        datasets: [{{ data: [run.passed, run.failed], backgroundColor: ['#10b981', '#f43f5e'], borderWidth: 0, cutout: '88%' }}]
                    }},
                    options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }} }}
                }});
            }}

            // Metric cards
            const metricCardsEl = document.getElementById('detail-metric-cards');
            metricCardsEl.innerHTML = '';
            run.metric_names.forEach(function(m) {{
                const avg = run.metric_avgs[m] != null ? run.metric_avgs[m] : 0;
                const col = run.metric_colors[m] || '#6b7280';
                const short = m.split('[')[0].trim();
                const pct = Math.round(avg * 100);
                metricCardsEl.innerHTML += (
                    '<div class="flex items-center justify-between p-2 rounded-md" style="background:' + col + '11;border:1px solid ' + col + '33">' +
                    '<div class="flex items-center gap-2">' +
                    '<div class="w-2 h-2 rounded-full" style="background:' + col + '"></div>' +
                    '<span class="text-gray-300 text-xs">' + m + '</span></div>' +
                    '<div class="flex items-center gap-2">' +
                    '<div class="w-20 h-1.5 rounded-full bg-gray-800 overflow-hidden">' +
                    '<div class="h-full rounded-full" style="width:' + pct + '%;background:' + col + '"></div></div>' +
                    '<span class="text-xs font-mono text-white w-10 text-right">' + avg.toFixed(3) + '</span>' +
                    '</div></div>'
                );
            }});

            // Test cases grid
            initCaseGrid(run);

            // Analysis summary
            const analysisSummaryEl = document.getElementById('detail-analysis-summary');
            const analysisSummaryBodyEl = document.getElementById('detail-analysis-summary-body');
            const analysisSummaryToggleEl = document.getElementById('detail-analysis-toggle');
            if (run.analysis_summary && run.analysis_summary.trim()) {{
                const cleanSummary = cleanSummaryMarkdown(run.analysis_summary);
                analysisSummaryEl.classList.remove('hidden');
                analysisSummaryBodyEl.innerHTML =
                    cleanSummary.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\\n/g, '<br>');
                analysisSummaryBodyEl.classList.remove('hidden');
                if (analysisSummaryToggleEl) {{
                    analysisSummaryToggleEl.textContent = 'Collapse';
                    analysisSummaryToggleEl.onclick = function() {{
                        const isHidden = analysisSummaryBodyEl.classList.contains('hidden');
                        analysisSummaryBodyEl.classList.toggle('hidden', !isHidden);
                        analysisSummaryToggleEl.textContent = isHidden ? 'Collapse' : 'Expand';
                    }};
                }}
            }} else {{
                analysisSummaryEl.classList.add('hidden');
                analysisSummaryBodyEl.innerHTML = '';
                if (analysisSummaryToggleEl) {{
                    analysisSummaryToggleEl.onclick = null;
                }}
            }}

        }}

        setupCaseGridEvents();
        setupEvaluatorFilterOptions();
        setupTimeFilterControls();
        const evaluatorSelect = document.getElementById('evaluator-filter-select');
        if (evaluatorSelect) {{
            evaluatorSelect.addEventListener('change', function(e) {{
                dashboardState.selectedEvaluator = String(e.target.value || 'All');
                applyDashboardFilters(true);
            }});
        }}
        setupCompareMenu();
        normalizeRunRowDateCells();
        setupRunsTablePagination();
    </script>
</body>
</html>"""

    Path(html_filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(html_filepath, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"Dashboard generated successfully at: {html_filepath}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        input_source = "results" if Path("results").is_dir() else "evaluation_results.json"
        output_html = "results/confident_ai_dashboard.html"
    elif len(args) == 1:
        input_source = args[0]
        output_html = "results/confident_ai_dashboard.html"
    else:
        if args[-1].lower().endswith(".html"):
            output_html = args[-1]
            input_source = args[:-1] if len(args[:-1]) > 1 else args[0]
        else:
            input_source = args
            output_html = "results/confident_ai_dashboard.html"

    create_dashboard(input_source, output_html)
