import json
import html
import re
from statistics import mean


def _render_summary_html(text: str) -> str:
    """Convert the GPT-4o markdown analysis report into clean, professional HTML."""
    if not text:
        return ""

    lines = text.splitlines()
    out = []
    in_list = False

    for line in lines:
        stripped = line.strip()

        # Strip leading markdown heading markers (###, ##, #)
        stripped = re.sub(r'^#{1,6}\s*', '', stripped)

        # Strip **bold** markers
        stripped = re.sub(r'\*\*(.+?)\*\*', r'\1', stripped)

        # Strip remaining lone * or _
        stripped = re.sub(r'[*_]', '', stripped)

        if not stripped:
            if in_list:
                out.append('</ul>')
                in_list = False
            out.append('<div class="mb-3"></div>')
            continue

        # Numbered section heading: e.g. "1. Overall Score & Status"
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

        # Bullet point: starts with "- " or "* "
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

        # Plain paragraph line
        if in_list:
            out.append('</ul>')
            in_list = False
        out.append(f'<p class="text-gray-300 text-sm leading-relaxed">{html.escape(stripped)}</p>')

    if in_list:
        out.append('</ul>')

    return '\n'.join(out)


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


def _item_passed(item: dict, metric_names: list[str], display_to_norm: dict[str, str]) -> bool:
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
        score = _to_float(md.get("score", 0.0), default=0.0)
        thr = md.get("threshold", None)
        if thr is None:
            fallback_flags.append(score > 0)
        else:
            fallback_flags.append(score >= _to_float(thr, default=0.0))
    return all(fallback_flags)


# Distinct colors for up to 10 dynamic metrics
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

# Card background/border tints paired with METRIC_COLORS
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


def create_dashboard(json_filepath, html_filepath):
    # 1) Load JSON
    try:
        with open(json_filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find the file {json_filepath}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {json_filepath}: {e}")
        return

    data = _as_list(raw)
    if not data:
        print("Error: JSON did not contain a list of evaluation items.")
        return

    analysis_summary = raw.get("analysis_summary") if isinstance(raw, dict) else None
    evaluator_type = raw.get("evaluator_type", "Unknown") if isinstance(raw, dict) else "Unknown"
    evaluator_color = "#10b981" if evaluator_type == "RAG" else "#a855f7" if evaluator_type == "GEval" else "#6b7280"
    evaluator_bg = "bg-emerald-500/10 border-emerald-500/30 text-emerald-400" if evaluator_type == "RAG" else "bg-violet-500/10 border-violet-500/30 text-violet-400" if evaluator_type == "GEval" else "bg-gray-500/10 border-gray-500/30 text-gray-400"

    # 2) Discover ALL metrics across ALL rows
    all_metric_norm_set = set()
    display_by_norm_global = {}

    for item in data:
        raw_metrics = item.get("evaluation_metrics", {}) or {}
        metrics_by_norm, display_by_norm = _normalize_metrics_dict(raw_metrics)
        all_metric_norm_set.update(metrics_by_norm.keys())
        for nk, disp in display_by_norm.items():
            display_by_norm_global.setdefault(nk, disp)

    # Stable display order: alphabetical by display name
    all_metric_names = sorted(
        [display_by_norm_global.get(nk, nk) for nk in all_metric_norm_set]
    )

    # Map display -> norm
    display_to_norm = {display_by_norm_global.get(nk, nk): nk for nk in all_metric_norm_set}

    # Assign a color to each metric (stable, by sorted order)
    metric_color = {m: METRIC_COLORS[i % len(METRIC_COLORS)] for i, m in enumerate(all_metric_names)}
    metric_card_style = {m: CARD_STYLES[i % len(CARD_STYLES)] for i, m in enumerate(all_metric_names)}

    total_cases = len(data)
    passed_cases = 0

    chart_labels = []
    # chart_data[metric_display] = list of scores per row
    chart_data = {m: [] for m in all_metric_names}

    all_scores = []
    # For summary cards: accumulate scores per metric
    metric_all_scores = {m: [] for m in all_metric_names}

    table_items = []

    for i, item in enumerate(data):
        q = item.get("question", f"Q{i + 1}")
        chart_labels.append((q[:15] + "...") if len(q) > 15 else q)

        raw_metrics = item.get("evaluation_metrics", {}) or {}
        metrics_by_norm, _ = _normalize_metrics_dict(raw_metrics)

        per_metric_score = {}
        per_metric_reason = {}
        per_metric_threshold = {}
        per_metric_passed = {}

        lowest_metric = ""
        lowest_score = 1.0
        lowest_reason = ""

        for m_display in all_metric_names:
            nk = display_to_norm.get(m_display, _norm_metric_key(m_display))
            md = metrics_by_norm.get(nk, {}) or {}

            score_f = _to_float(md.get("score", 0.0), default=0.0)
            reason = md.get("reason", "No reason provided.")
            thr = md.get("threshold", None)
            thr_f = _to_float(thr, default=0.0) if thr is not None else None
            mp = _to_bool(md.get("passed"), default=None)

            per_metric_score[m_display] = score_f
            per_metric_reason[m_display] = reason
            per_metric_threshold[m_display] = thr_f
            per_metric_passed[m_display] = mp

            all_scores.append(score_f)
            metric_all_scores[m_display].append(score_f)
            chart_data[m_display].append(score_f)

            if score_f < lowest_score:
                lowest_score = score_f
                lowest_metric = m_display
                lowest_reason = reason

        item_passed = _item_passed(item, all_metric_names, display_to_norm)
        if item_passed:
            passed_cases += 1

        failed_metrics = [m for m in all_metric_names if per_metric_passed.get(m) is False]

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
            "evaluation_metrics": {
                m: {
                    "score": per_metric_score.get(m, 0.0),
                    "reason": per_metric_reason.get(m, ""),
                    "threshold": per_metric_threshold.get(m, None),
                    "passed": per_metric_passed.get(m, None),
                }
                for m in all_metric_names
            },
            "lowest_reason": lowest_reason,
        })

    failing_cases = total_cases - passed_cases
    pass_rate = (passed_cases / total_cases) * 100 if total_cases > 0 else 0
    avg_model_score = mean(all_scores) if all_scores else 0.0

    # Avg score per metric (for summary cards)
    metric_avg = {m: (mean(metric_all_scores[m]) if metric_all_scores[m] else 0.0) for m in all_metric_names}

    table_items.sort(key=lambda r: (r["passed"],))

    # ── Table rows HTML ──────────────────────────────────────────────────────
    rows_html = ""
    for row in table_items:
        i = row["idx"]
        status_text = "PASSED" if row["passed"] else "FAILED"

        input_text = row["question"] or ""
        output_text = row["generated_answer"] or ""
        expected_text = row["expected_answer"] or ""
        meta_text = row["metadata"] or ""
        ts_text = row["timestamp"] or ""
        top_status = row.get("status", "") or ""

        input_trunc = (input_text[:50] + "...") if len(input_text) > 50 else input_text
        output_trunc = (output_text[:80] + "...") if len(output_text) > 80 else output_text

        tooltip_text = (
            f"STATUS: {status_text} | JSON status: {top_status} | "
            f"{row['metric_kind']}: {row['metric']} ({row['metric_score']:.3f})"
        )

        detail_row_id = f"details_{i}"

        metric_cards_html = ""
        for m in all_metric_names:
            md = row["evaluation_metrics"].get(m, {}) or {}
            sc = md.get("score", 0.0)
            rs = md.get("reason", "")
            th = md.get("threshold", None)
            mp = md.get("passed", None)

            th_txt = "n/a" if th is None else f"{th:.3f}"
            mp_txt = "n/a" if mp is None else ("true" if mp else "false")

            metric_cards_html += f"""
            <div class="border border-gray-800/50 rounded-md p-3">
                <div class="flex items-center justify-between">
                    <div class="text-gray-200 font-semibold text-xs">{_safe(m)}</div>
                    <div class="text-gray-300 text-xs">{sc:.3f}</div>
                </div>
                <div class="flex items-center gap-4 mt-2 text-[10px] uppercase tracking-wider text-gray-500">
                    <div>threshold: {th_txt}</div>
                    <div>passed: {mp_txt}</div>
                </div>
                <div class="text-gray-400 mt-2 whitespace-pre-wrap">{_safe(rs)}</div>
            </div>
            """

        rows_html += f"""
        <tr class="border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors text-sm cursor-pointer"
            title="{_safe(tooltip_text)}" data-target="{detail_row_id}">
          <td class="py-4 px-4 whitespace-nowrap">
                <span class="px-2 py-1 rounded-md text-[10px] uppercase font-bold tracking-wider border
                    {'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' if row['passed'] else 'bg-rose-500/20 text-rose-400 border-rose-500/30'}">
                    {'Passed' if row['passed'] else 'Failed'}
                </span>
          </td>
            <td class="py-4 px-4 text-gray-300 truncate max-w-[200px]">{_safe(input_trunc)}</td>
            <td class="py-4 px-4 text-gray-400 truncate max-w-[300px]">{_safe(output_trunc)}</td>
        </tr>

        <tr id="{detail_row_id}" class="hidden border-b border-gray-800/50 bg-[#15161b]">
            <td colspan="3" class="py-4 px-4 text-xs text-gray-300">
                <div class="space-y-3">
                    <div class="flex flex-wrap gap-6">
                        <div>
                            <div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Status</div>
                            <div class="text-gray-300">{_safe(status_text)}</div>
                        </div>
                        <div>
                            <div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">JSON status</div>
                            <div class="text-gray-300">{_safe(top_status)}</div>
                        </div>
                        <div>
                            <div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Metadata</div>
                            <div class="text-gray-300">{_safe(meta_text)}</div>
                        </div>
                        <div>
                            <div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Timestamp</div>
                            <div class="text-gray-300">{_safe(ts_text)}</div>
                        </div>
                    </div>

                    <div>
                        <div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Question</div>
                        <div class="text-gray-200 whitespace-pre-wrap">{_safe(input_text)}</div>
                    </div>

                    <div>
                        <div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Generated Answer</div>
                        <div class="text-gray-200 whitespace-pre-wrap">{_safe(output_text)}</div>
                    </div>

                    <div>
                        <div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-1">Expected Answer</div>
                        <div class="text-gray-200 whitespace-pre-wrap">{_safe(expected_text)}</div>
                    </div>

                    <div>
                        <div class="text-gray-500 uppercase tracking-wider text-[10px] font-semibold mb-2">Evaluation Metrics</div>
                        <div class="space-y-2">
                            {metric_cards_html}
                        </div>
                    </div>
                </div>
            </td>
        </tr>
        """

    # ── Build JS datasets dynamically ───────────────────────────────────────
    DASH_PATTERNS = ['[]', '[6,3]', '[2,3]', '[8,3,2,3]', '[12,3]',
                     '[4,2,4,2]', '[1,4]', '[10,2,2,2]', '[6,2,2,2,2,2]', '[3,3,3,3]']
    POINT_STYLES = ['circle', 'rect', 'triangle', 'rectRot', 'star',
                    'cross', 'crossRot', 'dash', 'line', 'circle']

    js_datasets = []
    for idx, m in enumerate(all_metric_names):
        color = metric_color[m]
        scores = chart_data[m]
        dash = DASH_PATTERNS[idx % len(DASH_PATTERNS)]
        pstyle = POINT_STYLES[idx % len(POINT_STYLES)]
        lines_parts = [
            '{',
            f'                label: {json.dumps(m)},',
            f'                data: {json.dumps(scores)},',
            f'                backgroundColor: {json.dumps(color + "33")},',
            f'                borderColor: {json.dumps(color)},',
            '                borderWidth: 2.5,',
            '                borderDash: [],',
            '                tension: 0.4,',
            '                pointRadius: 5,',
            '                pointHoverRadius: 8,',
            f'                pointStyle: {json.dumps(pstyle)},',
            '                fill: false',
            '            }',
        ]
        js_datasets.append('\n'.join(lines_parts))
    js_datasets_str = ',\n'.join(js_datasets)

    # ── Build summary cards HTML dynamically ─────────────────────────────────
    summary_cards_html = ""
    for m in all_metric_names:
        color = metric_color[m]
        bg, border = metric_card_style[m]
        avg = metric_avg[m]
        # Short label: strip "Metric" suffix for display if present
        short_label = m.replace("Metric", "").strip()
        summary_cards_html += f"""
        <div class="{bg} border {border} rounded-md p-3 text-center">
            <div class="w-3 h-3 rounded-full mx-auto mb-1" style="background-color:{color}"></div>
            <div class="text-[10px] text-gray-400 uppercase tracking-wider font-mono leading-tight">{_safe(short_label)}</div>
            <div class="text-lg font-bold text-white mt-1">{avg:.2f}</div>
        </div>
        """

    # Variables pre-computed to avoid f-string nesting/literal issues
    gpt4o_label = "GPT-4o Analysis Report"
    analysis_summary_block = ""
    if analysis_summary:
        analysis_summary_block = f"""
        <div class="glass-panel p-6">
            <div class="flex items-center gap-3 pb-4 mb-2 border-b border-gray-800/60">
                <svg class="w-4 h-4 text-violet-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                </svg>
                <h3 class="text-gray-100 font-semibold text-sm tracking-wide">{gpt4o_label}</h3>
                <span class="ml-auto px-2 py-0.5 rounded text-[10px] uppercase font-bold tracking-wider bg-violet-500/10 text-violet-400 border border-violet-500/20">AI Generated</span>
            </div>
            <div class="mt-4">
                {_render_summary_html(analysis_summary)}
            </div>
        </div>"""

    # ── HTML template ────────────────────────────────────────────────────────
    html_template = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Confident AI - Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {{ font-family: 'Inter', sans-serif; background-color: #0b0c10; color: #e5e7eb; }}
        .glass-panel {{ background-color: #15161b; border: 1px solid #23252b; border-radius: 0.5rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5); }}
        .nav-item {{ padding: 0.5rem 1rem; color: #9ca3af; font-size: 0.875rem; border-radius: 0.375rem; display: flex; align-items: center; gap: 0.75rem; transition: all 0.2s; }}
        .nav-item:hover, .nav-item.active {{ background-color: #1f2128; color: #f3f4f6; }}
        .font-mono {{ font-family: 'Courier New', monospace; }}
    </style>
</head>
<body class="flex h-screen overflow-hidden">

    <aside class="w-64 border-r border-gray-800 flex flex-col bg-[#0b0c10] overflow-y-auto shrink-0">
        <div class="p-5 flex items-center gap-3 border-b border-gray-800">
            <div class="w-7 h-7 bg-white rounded flex items-center justify-center text-black font-extrabold text-sm">C</div>
            <span class="font-bold text-white tracking-wide text-lg">Evaluation Tests</span>
        </div>

        <div class="p-3">
            <p class="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2 px-3 mt-4">Evaluation & Observability</p>
            <nav class="space-y-1">
                <a href="#" class="nav-item active"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path></svg>Evaluation</a>
                <a href="#" class="nav-item"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"></path></svg>Datasets</a>
                <a href="#" class="nav-item"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path></svg>Simulations</a>
            </nav>
        </div>
    </aside>

    <main class="flex-1 p-8 overflow-y-auto bg-[#0b0c10]">

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
                <span class="text-gray-400 text-sm font-medium">Avg Model Score</span>
                <div class="text-2xl font-bold mt-1 text-emerald-400 flex items-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg>
                    {avg_model_score:.2f}
                </div>
            </div>
            <div class="glass-panel p-5 flex-1 flex flex-col justify-center">
                <span class="text-gray-400 text-sm font-medium">Evaluation Mode</span>
                <div class="mt-2 flex items-center gap-2">
                    <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-sm font-semibold border {evaluator_bg}">
                        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                        {evaluator_type}
                    </span>
                </div>
                <div class="text-xs text-gray-500 mt-1.5">{"RAG Metrics (Faithfulness, Precision, Recall, Relevancy)" if evaluator_type == "RAG" else "GEval Custom Criteria" if evaluator_type == "GEval" else "Unknown evaluator"}</div>
            </div>
        </div>

        <div class="grid grid-cols-1 xl:grid-cols-3 gap-6 mb-6">
            <div class="glass-panel p-6 flex flex-col h-[350px]">
                <h3 class="text-gray-400 text-sm font-medium mb-6">Latest Deployment</h3>
                <div class="relative flex-1 flex flex-col items-center justify-center">
                    <div class="w-48 h-48 relative">
                        <canvas id="passRateChart"></canvas>
                        <div class="absolute inset-0 flex flex-col items-center justify-center mt-2">
                            <span class="text-4xl font-bold text-white">{pass_rate:.1f}%</span>
                            <span class="text-xs text-gray-400 mt-1">{passed_cases}/{total_cases} passed</span>
                        </div>
                    </div>
                    <div class="text-center mt-6">
                        <p class="text-white font-medium">{failing_cases} failing test cases</p>
                        <p class="text-gray-500 text-xs mt-1">based on JSON passed/status fields</p>
                    </div>
                </div>
            </div>

            <div class="glass-panel p-0 xl:col-span-2 flex flex-col h-[350px]">
                <div class="p-6 pb-4 border-b border-gray-800/50 flex justify-between items-center">
                    <h3 class="text-gray-200 font-medium text-sm">Top failing LLM outputs
                        <span class="text-xs text-gray-500 font-normal ml-2 hover:text-gray-300 transition-colors cursor-help"
                              title="Click a row to expand and see all fields + ALL metrics from JSON.">[Click for details]</span>
                    </h3>
                </div>
                <div class="flex-1 overflow-y-auto px-2">
                    <table class="w-full text-left">
                        <thead class="sticky top-0 bg-[#15161b]">
                            <tr class="text-[11px] text-gray-500 uppercase tracking-wider">
                                <th class="pb-3 pt-4 font-semibold px-4 w-32">Status</th>
                                <th class="pb-3 pt-4 font-semibold px-4 w-1/3">Input</th>
                                <th class="pb-3 pt-4 font-semibold px-4">LLM Output</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Bar Chart + Summary Cards -->
        <div class="glass-panel p-6 mb-6">
            <div class="flex items-center justify-between mb-6">
                <div class="flex items-center gap-4">
                    <h3 class="text-gray-200 font-medium text-sm">Production LLM Output Quality</h3>
                    <span class="px-2.5 py-1 rounded bg-[#2e2348] text-[#a78bfa] text-[10px] uppercase font-bold tracking-wider border border-[#4c3a7a]">All Metrics</span>
                </div>
            </div>
            <div class="h-64 w-full mb-6">
                <canvas id="qualityBarChart"></canvas>
            </div>
            <!-- Per-metric average summary cards -->
            <div class="grid gap-3" style="grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));">
                {summary_cards_html}
            </div>
        </div>

        <!-- Analysis Summary -->
        {analysis_summary_block}

    </main>

    <script>
        // Donut Chart
        const passCtx = document.getElementById('passRateChart').getContext('2d');
        new Chart(passCtx, {{
            type: 'doughnut',
            data: {{
                labels: ['Passed', 'Failed'],
                datasets: [{{
                    data: [{passed_cases}, {failing_cases}],
                    backgroundColor: ['#10b981', '#f43f5e'],
                    borderWidth: 0,
                    cutout: '88%'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }}, tooltip: {{ enabled: false }} }}
            }}
        }});

        // Line Chart — dynamic metrics
        const barCtx = document.getElementById('qualityBarChart').getContext('2d');
        new Chart(barCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [
                    {js_datasets_str}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ intersect: false, mode: 'index' }},
                scales: {{
                    x: {{
                        grid: {{ color: 'rgba(31, 41, 55, 0.4)', drawBorder: false }},
                        ticks: {{ color: '#6b7280', font: {{ size: 11 }} }}
                    }},
                    y: {{
                        min: 0.0,
                        max: 1.05,
                        grid: {{ color: 'rgba(31, 41, 55, 0.4)', drawBorder: false }},
                        ticks: {{ color: '#6b7280', font: {{ size: 11 }} }}
                    }}
                }},
                plugins: {{
                    legend: {{
                        position: 'top',
                        align: 'end',
                        labels: {{ color: '#9ca3af', boxWidth: 10, usePointStyle: true, font: {{ size: 11 }} }}
                    }},
                    tooltip: {{
                        backgroundColor: '#1f2028',
                        borderColor: '#2e3040',
                        borderWidth: 1,
                        titleColor: '#e5e7eb',
                        bodyColor: '#9ca3af',
                        padding: 10
                    }}
                }}
            }}
        }});

        // Expand/collapse details row on click
        document.addEventListener('click', (e) => {{
            const tr = e.target.closest('tr[data-target]');
            if (!tr) return;
            const targetId = tr.getAttribute('data-target');
            const detailsRow = document.getElementById(targetId);
            if (!detailsRow) return;
            detailsRow.classList.toggle('hidden');
        }});
    </script>
</body>
</html>"""

    with open(html_filepath, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"Dashboard generated successfully at: {html_filepath}")


if __name__ == "__main__":
    create_dashboard("evaluation_results.json", "confident_ai_dashboard.html")