import json
import html
from statistics import mean


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


def create_full_dark_dashboard(json_filepath, html_filepath):
    # 1. Load the JSON data
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

    # 2. Process data and calculate Pass/Fail
    threshold = 0.95
    total_cases = len(data)
    passed_cases = 0

    metric_names = ["Fluency", "Correctness", "Coherence", "Relevance"]

    chart_labels = []
    chart_data = {m: [] for m in metric_names}
    all_scores = []

    # ✅ THIS will hold ALL rows for the table (not just failing ones)
    table_items = []

    for i, item in enumerate(data):
        q = item.get("question", f"Q{i + 1}")
        chart_labels.append((q[:15] + "...") if len(q) > 15 else q)

        metrics = item.get("evaluation_metrics", {}) or {}

        passed = True
        lowest_metric = ""
        lowest_score = 1.0
        failure_reason = ""

        # Keep also per-metric reasons for tooltip (no UI changes, just more info)
        per_metric_reason = {}
        per_metric_score = {}

        for m_name in metric_names:
            metric_data = metrics.get(m_name, {}) or {}
            score = metric_data.get("score", 0)
            reason = metric_data.get("reason", "No reason provided.")

            try:
                score_f = float(score)
            except (TypeError, ValueError):
                score_f = 0.0

            chart_data[m_name].append(score_f)
            all_scores.append(score_f)

            per_metric_score[m_name] = score_f
            per_metric_reason[m_name] = reason

            if score_f < lowest_score:
                lowest_score = score_f
                lowest_metric = m_name
                failure_reason = reason

            if score_f < threshold:
                passed = False

        if passed:
            passed_cases += 1

        # ✅ Add EVERY row to the table
        table_items.append(
            {
                "metric": lowest_metric,       # keep badge as "most problematic" metric
                "score": lowest_score,
                "passed": passed,              # only used in tooltip (no UI changes)
                "input": item.get("question", ""),
                "output": item.get("generated_answer", ""),
                "expected": item.get("expected_answer", ""),
                "metadata": item.get("metadata", ""),
                "timestamp": item.get("timestamp", ""),
                "failure_reason": failure_reason,
                "per_metric_score": per_metric_score,
                "per_metric_reason": per_metric_reason,
            }
        )

    pass_rate = (passed_cases / total_cases) * 100 if total_cases > 0 else 0
    failing_cases = total_cases - passed_cases
    avg_model_score = mean(all_scores) if all_scores else 0.0

    # 3. Build the Table HTML (now for ALL rows)
    failing_rows_html = ""
    for row in table_items:
        input_text = row["input"] or ""
        output_text = row["output"] or ""
        expected_text = row["expected"] or ""
        meta_text = row.get("metadata", "") or ""
        ts_text = row.get("timestamp", "") or ""

        input_trunc = (input_text[:50] + "...") if len(input_text) > 50 else input_text
        output_trunc = (output_text[:80] + "...") if len(output_text) > 80 else output_text

        badge_color = "bg-purple-500/20 text-purple-400 border-purple-500/30"
        if row["metric"] == "Correctness":
            badge_color = "bg-yellow-500/20 text-yellow-400 border-yellow-500/30"
        elif row["metric"] == "Fluency":
            badge_color = "bg-blue-500/20 text-blue-400 border-blue-500/30"

        # Tooltip: show pass/fail + expected + all metric scores/reasons (no layout change)
        status_text = "PASSED" if row["passed"] else "FAILED"
        metrics_summary = " | ".join(
            f"{m}={row['per_metric_score'].get(m, 0):.3f}: {row['per_metric_reason'].get(m, '')}"
            for m in metric_names
        )

        tooltip_text = (
            f"STATUS: {status_text} | METADATA: {meta_text} | TIMESTAMP: {ts_text} | "
            f"EXPECTED: {expected_text} | "
            f"LOWEST METRIC: {row['metric']} ({row['score']:.3f}) | "
            f"DETAILS: {metrics_summary}"
        )

        failing_rows_html += f"""
        <tr class="border-b border-gray-800/50 hover:bg-gray-800/50 transition-colors text-sm cursor-help" title="{_safe(tooltip_text)}">
            <td class="py-4 px-4 whitespace-nowrap">
                <span class="px-2 py-1 rounded-md text-[10px] uppercase font-bold tracking-wider border {badge_color}">{_safe(row['metric'])}</span>
            </td>
            <td class="py-4 px-4 text-gray-300 truncate max-w-[200px]">{_safe(input_trunc)}</td>
            <td class="py-4 px-4 text-gray-400 truncate max-w-[300px]">{_safe(output_trunc)}</td>
        </tr>
        """

    # 4. Generate the HTML Template (unchanged UI/colors/layout)
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
    </style>
</head>
<body class="flex h-screen overflow-hidden">

    <aside class="w-64 border-r border-gray-800 flex flex-col bg-[#0b0c10] overflow-y-auto shrink-0">
        <div class="p-5 flex items-center gap-3 border-b border-gray-800">
            <div class="w-7 h-7 bg-white rounded flex items-center justify-center text-black font-extrabold text-sm">C</div>
            <span class="font-bold text-white tracking-wide text-lg">Confident AI</span>
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
                <div class="text-right text-xs text-gray-500 leading-relaxed">based on {total_cases} evaluations<br>from your dataset</div>
            </div>
            <div class="glass-panel p-5 flex-1 flex flex-col justify-center">
                <span class="text-gray-400 text-sm font-medium">Avg Model Score</span>
                <div class="text-2xl font-bold mt-1 text-emerald-400 flex items-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg>
                    {avg_model_score:.2f}
                </div>
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
                        <p class="text-gray-500 text-xs mt-1">Threshold set to 0.95</p>
                    </div>
                </div>
            </div>

            <div class="glass-panel p-0 xl:col-span-2 flex flex-col h-[350px]">
                <div class="p-6 pb-4 border-b border-gray-800/50 flex justify-between items-center">
                    <h3 class="text-gray-200 font-medium text-sm">Top failing LLM outputs <span class="text-xs text-gray-500 font-normal ml-2 hover:text-gray-300 transition-colors cursor-help" title="Hover over rows to see detailed reasoning from your JSON.">[Hover for reason]</span></h3>
                </div>
                <div class="flex-1 overflow-y-auto px-2">
                    <table class="w-full text-left">
                        <thead class="sticky top-0 bg-[#15161b]">
                            <tr class="text-[11px] text-gray-500 uppercase tracking-wider">
                                <th class="pb-3 pt-4 font-semibold px-4 w-32">Metric</th>
                                <th class="pb-3 pt-4 font-semibold px-4 w-1/3">Input</th>
                                <th class="pb-3 pt-4 font-semibold px-4">LLM Output</th>
                            </tr>
                        </thead>
                        <tbody>
                            {failing_rows_html}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="glass-panel p-6">
            <div class="flex items-center justify-between mb-6">
                <div class="flex items-center gap-4">
                    <h3 class="text-gray-200 font-medium text-sm">Production LLM Output Quality</h3>
                    <span class="px-2.5 py-1 rounded bg-[#2e2348] text-[#a78bfa] text-[10px] uppercase font-bold tracking-wider border border-[#4c3a7a]">All Metrics</span>
                </div>
            </div>
            <div class="h-64 w-full">
                <canvas id="qualityLineChart"></canvas>
            </div>
        </div>
    </main>

    <script>
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

        const lineCtx = document.getElementById('qualityLineChart').getContext('2d');
        new Chart(lineCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(chart_labels)},
                datasets: [
                    {{
                        label: 'Fluency',
                        data: {json.dumps(chart_data['Fluency'])},
                        borderColor: '#a855f7',
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 4,
                        pointBackgroundColor: '#0b0c10',
                        pointBorderColor: '#a855f7'
                    }},
                    {{
                        label: 'Correctness',
                        data: {json.dumps(chart_data['Correctness'])},
                        borderColor: '#fde047',
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 4,
                        pointBackgroundColor: '#0b0c10',
                        pointBorderColor: '#fde047'
                    }},
                    {{
                        label: 'Coherence',
                        data: {json.dumps(chart_data['Coherence'])},
                        borderColor: '#3b82f6',
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 4,
                        pointBackgroundColor: '#0b0c10',
                        pointBorderColor: '#3b82f6'
                    }},
                    {{
                        label: 'Relevance',
                        data: {json.dumps(chart_data['Relevance'])},
                        borderColor: '#10b981',
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 4,
                        pointBackgroundColor: '#0b0c10',
                        pointBorderColor: '#10b981'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                interaction: {{ intersect: false, mode: 'index' }},
                scales: {{
                    x: {{ grid: {{ color: 'rgba(31, 41, 55, 0.4)', drawBorder: false }}, ticks: {{ color: '#6b7280', font: {{ size: 11 }} }} }},
                    y: {{ min: 0.7, max: 1.05, grid: {{ color: 'rgba(31, 41, 55, 0.4)', drawBorder: false }}, ticks: {{ color: '#6b7280', font: {{ size: 11 }} }} }}
                }},
                plugins: {{
                    legend: {{ position: 'top', align: 'end', labels: {{ color: '#9ca3af', boxWidth: 10, usePointStyle: true, font: {{ size: 11 }} }} }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(html_filepath, "w", encoding="utf-8") as f:
        f.write(html_template)

    print(f"✅ Dashboard generated successfully at: {html_filepath}")


if __name__ == "__main__":
    create_full_dark_dashboard("evaluation_results.json", "confident_ai_dashboard.html")