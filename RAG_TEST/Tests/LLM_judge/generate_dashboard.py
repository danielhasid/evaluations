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
            f'<tr class="border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors text-sm cursor-pointer"'
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
        label = source_name
        ts_match = re.search(r'(\d{8}_\d{6})', source_name)
        if ts_match:
            ts_raw = ts_match.group(1)
            try:
                dt = datetime.strptime(ts_raw, "%Y%m%d_%H%M%S")
                label = dt.strftime("%b %d %H:%M")
            except ValueError:
                label = ts_raw
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
        for r in runs:
            avg = r["metric_avgs"].get(m, None)
            data_points.append(round(avg, 4) if avg is not None else None)

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

        ts_display = r["timestamp"][:16].replace("T", " ") if r["timestamp"] else r["label"]
        fname_short = r["filename"][:32] + ("..." if len(r["filename"]) > 32 else "")

        result_cls = (
            "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" if r["failed"] == 0 else
            "bg-rose-500/20 text-rose-400 border-rose-500/30"
        )
        result_label = f"{r['passed']}/{r['total']} passed"

        runs_table_rows_html += (
            f'<tr class="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors cursor-pointer"'
            f' onclick="showDetail({i})">'
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
            "metric_names": r["metric_names"],
            "metric_colors": r["metric_colors"],
            "analysis_summary": r["analysis_summary"],
            "test_cases": r.get("test_cases", []),
            "rows_html": r["rows_html"],
        }
        runs_js_parts.append(json.dumps(obj, ensure_ascii=False))
    runs_js_array = "[\n  " + ",\n  ".join(runs_js_parts) + "\n]"

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
        @media print {{
            aside {{ display: none !important; }}
            body {{ display: block !important; background: #fff !important; color: #000 !important; overflow: visible !important; height: auto !important; }}
            main {{ padding: 1rem !important; overflow: visible !important; height: auto !important; }}
            .glass-panel {{ background: #f9fafb !important; border: 1px solid #e5e7eb !important; box-shadow: none !important; }}
            .text-gray-400, .text-gray-500 {{ color: #374151 !important; }}
            .text-gray-300, .text-gray-200 {{ color: #111827 !important; }}
            .text-white {{ color: #000 !important; }}
            .no-print {{ display: none !important; }}
            tr.hidden {{ display: table-row !important; }}
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
                    <p class="text-gray-500 text-xs mt-1">Using evaluation data from {total_runs} displayed test run{"s" if total_runs != 1 else ""} · click a row to inspect</p>
                </div>
                <button onclick="window.print()"
                    class="flex items-center gap-2 px-4 py-2 rounded-md bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium transition-colors border border-violet-500/50 cursor-pointer">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z"/></svg>
                    Export PDF
                </button>
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
                    <div class="text-2xl font-bold mt-1 text-white">{total_runs}</div>
                    <div class="text-xs text-gray-500 mt-1">source JSON files</div>
                </div>
                <div class="glass-panel p-5 flex-1 flex flex-col justify-center">
                    <span class="text-gray-400 text-sm font-medium">Overall Pass Rate</span>
                    <div class="text-2xl font-bold mt-1 {'text-emerald-400' if overall_pass_rate >= 80 else 'text-yellow-400' if overall_pass_rate >= 50 else 'text-rose-400'}">{overall_pass_rate:.1f}%</div>
                    <div class="text-xs text-gray-500 mt-1">{total_passed}/{total_cases} passed</div>
                </div>
                <div class="glass-panel p-5 flex-1 flex flex-col justify-center">
                    <span class="text-gray-400 text-sm font-medium">Evaluation Mode</span>
                    <div class="mt-2 flex items-center gap-2">
                        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold border {evaluator_bg}">
                            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                            {evaluator_type}
                        </span>
                    </div>
                    <div class="text-xs text-gray-500 mt-1.5">{evaluator_desc}</div>
                </div>
            </div>

            <!-- Test Run Performance chart -->
            <div class="glass-panel p-6 mb-6">
                <div class="flex items-center justify-between mb-4">
                    <div>
                        <h3 class="text-gray-200 font-semibold text-sm">Test Run Performance</h3>
                        <p class="text-gray-500 text-xs mt-0.5">Average metric score per run · one point per evaluation file</p>
                    </div>
                    <span class="px-2.5 py-1 rounded bg-[#2e2348] text-[#a78bfa] text-[10px] uppercase font-bold tracking-wider border border-[#4c3a7a]">All Metrics</span>
                </div>
                <div class="h-56 w-full">
                    <canvas id="runsChart"></canvas>
                </div>
            </div>

            <!-- Runs table -->
            <div class="glass-panel p-0 overflow-hidden">
                <div class="p-5 border-b border-gray-800/50 flex items-center justify-between">
                    <div>
                        <h3 class="text-gray-200 font-medium text-sm">Showing <strong class="text-white">1 to {total_runs}</strong> of <strong class="text-white">{total_runs}</strong> test run{"s" if total_runs != 1 else ""}</h3>
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
                        <tbody>
                            {runs_table_rows_html}
                        </tbody>
                    </table>
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
                            <option value="5">5</option>
                            <option value="10" selected>10</option>
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

            <div id="detail-case-panel" class="glass-panel p-4 mb-6 hidden">
                <div class="flex items-center justify-between mb-3">
                    <h4 id="detail-case-panel-title" class="text-gray-200 font-semibold text-sm">Test Case Details</h4>
                    <button id="detail-case-close" class="text-gray-400 hover:text-white text-xs">Close</button>
                </div>
                <div id="detail-case-panel-body" class="text-xs text-gray-300 space-y-3"></div>
            </div>

            <!-- Analysis Summary -->
            <div id="detail-analysis-summary" class="glass-panel p-6 mb-6 hidden">
                <h3 class="text-gray-400 text-xs font-semibold uppercase tracking-widest mb-4">Analysis Summary</h3>
                <div id="detail-analysis-summary-body" class="text-gray-300 text-sm leading-relaxed whitespace-pre-wrap"></div>
            </div>

        </div><!-- /view-detail -->

    </main>

    <script>
        // ── Embedded run data ─────────────────────────────────────────────────
        const RUNS_DATA = {runs_js_array};

        // ── Formatting helpers ────────────────────────────────────────────────
        function cleanSummaryMarkdown(text) {{
            if (!text) return '';
            return text
                .split('\\n')
                .map(function(line) {{
                    let s = line.trim();
                    s = s.replace(/^#{1,6}\\s*/, '');
                    s = s.replace(/^[-*]\\s+/, '');
                    s = s.replace(/^>\\s+/, '');
                    s = s.replace(/\\*\\*(.*?)\\*\\*/g, '$1');
                    s = s.replace(/__(.*?)__/g, '$1');
                    s = s.replace(/`([^`]+)`/g, '$1');
                    return s;
                }})
                .join('\\n')
                .trim();
        }}

        // ── Runs list chart ───────────────────────────────────────────────────
        const runsCtx = document.getElementById('runsChart').getContext('2d');
        const runsChart = new Chart(runsCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(run_labels)},
                datasets: [
                    {runs_chart_datasets_str}
                ]
            }},
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
                        titleColor: '#e5e7eb', bodyColor: '#9ca3af', padding: 10
                    }}
                }}
            }}
        }});

        // ── Detail donut chart instance ───────────────────────────────────────
        let detailDonut = null;
        const testCaseState = {{
            run: null,
            cases: [],
            page: 1,
            pageSize: 10,
            sortKey: 'name',
            sortDir: 'asc',
            columns: {{ name: true, status: true, input: true, output: true }},
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

        function statusOrder(status) {{
            return (status || '').toLowerCase() === 'failed' ? 0 : 1;
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

        function renderCaseDetails(caseItem) {{
            const panel = document.getElementById('detail-case-panel');
            const title = document.getElementById('detail-case-panel-title');
            const body = document.getElementById('detail-case-panel-body');
            title.textContent = caseItem.name + ' Details';

            let metricHtml = '<div class=\"space-y-1\">';
            const metricEntries = Object.entries(caseItem.evaluation_metrics || {{}});
            if (!metricEntries.length) {{
                metricHtml += '<div class=\"text-gray-500\">No metric details available.</div>';
            }} else {{
                metricEntries.forEach(function(entry) {{
                    const metricName = entry[0];
                    const metric = entry[1] || {{}};
                    metricHtml += '<div class=\"rounded border border-gray-800/60 p-2\">' +
                        '<div class=\"flex items-center justify-between\"><span class=\"text-gray-200\">' + escapeHtml(metricName) + '</span>' +
                        '<span class=\"text-gray-300\">' + Number(metric.score || 0).toFixed(3) + '</span></div>' +
                        '<div class=\"text-gray-500 mt-1\">' + escapeHtml(metric.reason || '') + '</div></div>';
                }});
            }}
            metricHtml += '</div>';

            body.innerHTML =
                '<div><span class=\"text-gray-500\">Status:</span> ' + escapeHtml(caseItem.status) + '</div>' +
                '<div><span class=\"text-gray-500\">Input:</span><div class=\"mt-1 text-gray-200 whitespace-pre-wrap\">' + escapeHtml(caseItem.question) + '</div></div>' +
                '<div><span class=\"text-gray-500\">Actual Output:</span><div class=\"mt-1 text-gray-200 whitespace-pre-wrap\">' + escapeHtml(caseItem.generated_answer) + '</div></div>' +
                '<div><span class=\"text-gray-500\">Expected Answer:</span><div class=\"mt-1 text-gray-200 whitespace-pre-wrap\">' + escapeHtml(caseItem.expected_answer) + '</div></div>' +
                '<div><span class=\"text-gray-500\">Metrics:</span>' + metricHtml + '</div>';
            panel.classList.remove('hidden');
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
                return '<tr class=\"border-b border-gray-800/50 hover:bg-gray-800/40 cursor-pointer\" data-case-index=\"' + (start + idx) + '\">' +
                    '<td data-col=\"name\" class=\"py-3 px-4 text-gray-300\">' + escapeHtml(item.name) + '</td>' +
                    '<td data-col=\"status\" class=\"py-3 px-4\"><span class=\"px-2 py-0.5 rounded-full text-[11px] border ' + statusBadges + '\">' + escapeHtml(item.status) + '</span></td>' +
                    '<td data-col=\"input\" class=\"py-3 px-4 text-gray-300\" title=\"' + escapeHtml(item.question) + '\">' + escapeHtml(truncateText(item.question, 90)) + '</td>' +
                    '<td data-col=\"output\" class=\"py-3 px-4 text-gray-400\" title=\"' + escapeHtml(item.generated_answer) + '\">' + escapeHtml(truncateText(item.generated_answer, 110)) + '</td>' +
                    '</tr>';
            }}).join('');

            document.querySelectorAll('#detail-test-cases-grid-body tr[data-case-index]').forEach(function(rowEl) {{
                rowEl.addEventListener('click', function() {{
                    const idx = Number(rowEl.getAttribute('data-case-index'));
                    renderCaseDetails(sorted[idx]);
                }});
            }});

            applyColumnVisibility();
        }}

        function initCaseGrid(run) {{
            testCaseState.run = run;
            testCaseState.cases = getRunTestCases(run);
            testCaseState.page = 1;
            document.getElementById('detail-case-panel').classList.add('hidden');
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

            document.getElementById('detail-case-close').addEventListener('click', function() {{
                document.getElementById('detail-case-panel').classList.add('hidden');
            }});
        }}

        // ── Navigation ────────────────────────────────────────────────────────
        function showRuns() {{
            document.getElementById('view-runs').classList.remove('hidden');
            document.getElementById('view-detail').classList.add('hidden');
        }}

        function showDetail(idx) {{
            const run = RUNS_DATA[idx];
            document.getElementById('view-runs').classList.add('hidden');
            document.getElementById('view-detail').classList.remove('hidden');

            // Properties card
            document.getElementById('detail-filename').textContent = run.filename;
            document.getElementById('detail-evaluator').textContent = run.evaluator_type;
            const ts = run.timestamp ? run.timestamp.slice(0, 16).replace('T', ' ') : run.label;
            document.getElementById('detail-timestamp').textContent = ts;
            document.getElementById('detail-total').textContent = run.total + ' test cases';

            // Pass rate donut
            document.getElementById('detail-pass-rate').textContent = run.pass_rate.toFixed(1) + '%';
            document.getElementById('detail-pass-label').textContent = run.passed + '/' + run.total + ' passed';
            document.getElementById('detail-fail-count').textContent = run.failed + ' failing test case' + (run.failed !== 1 ? 's' : '');

            if (detailDonut) {{ detailDonut.destroy(); }}
            const dCtx = document.getElementById('detailDonutChart').getContext('2d');
            detailDonut = new Chart(dCtx, {{
                type: 'doughnut',
                data: {{
                    labels: ['Passed', 'Failed'],
                    datasets: [{{ data: [run.passed, run.failed], backgroundColor: ['#10b981', '#f43f5e'], borderWidth: 0, cutout: '88%' }}]
                }},
                options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }} }}
            }});

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
            if (run.analysis_summary && run.analysis_summary.trim()) {{
                const cleanSummary = cleanSummaryMarkdown(run.analysis_summary);
                analysisSummaryEl.classList.remove('hidden');
                analysisSummaryBodyEl.innerHTML =
                    cleanSummary.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\\n/g, '<br>');
            }} else {{
                analysisSummaryEl.classList.add('hidden');
                analysisSummaryBodyEl.innerHTML = '';
            }}

        }}

        setupCaseGridEvents();
    </script>
</body>
</html>"""

    with open(html_filepath, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"Dashboard generated successfully at: {html_filepath}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        input_source = "evaluation_results.json"
        output_html = "confident_ai_dashboard.html"
    elif len(args) == 1:
        input_source = args[0]
        output_html = "confident_ai_dashboard.html"
    else:
        if args[-1].lower().endswith(".html"):
            output_html = args[-1]
            input_source = args[:-1] if len(args[:-1]) > 1 else args[0]
        else:
            input_source = args
            output_html = "confident_ai_dashboard.html"

    create_dashboard(input_source, output_html)
