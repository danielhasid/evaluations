---
name: devtools-performance-audit
description: >
  Runs a full web performance audit on any URL using Chrome DevTools MCP tools.
  Captures Core Web Vitals (LCP, INP, CLS), runs a Lighthouse audit for
  Accessibility/Best Practices/SEO scores, and analyzes performance trace
  insights (render-blocking resources, third-party scripts, INP breakdown,
  CLS culprits, font display, DOM size). Delivers a structured report with
  a metrics table, Lighthouse scores, key issues, and prioritized recommendations.

  Trigger this skill whenever the user asks to audit, check, analyze, test, or
  profile the performance of a website or URL — even if they use casual phrasing
  like "how fast is X", "is this site slow", "what's the perf like on this page",
  "lighthouse this URL", or "benchmark this site". If a URL is mentioned alongside
  any performance-adjacent question, use this skill.
---

# Web Performance Audit — Chrome DevTools

## Overview

You are auditing a URL's real-world and lab performance using Chrome DevTools MCP tools. The output is a structured performance report covering Core Web Vitals, Lighthouse scores, and actionable recommendations.

Do NOT use playwright or any other browser automation for this task.

---

## Step 1 — Open the page

```
mcp__chrome-devtools__new_page(url=<target_url>)
```

---

## Step 2 — Run performance trace

Start the trace first. It will reload the page automatically:
```
mcp__chrome-devtools__performance_start_trace(reload=true)
```

The trace will return an insight set ID (e.g. `NAVIGATION_0`) and a list of available insights. Note them — you'll need them in the next steps.

## Step 3 — Run Lighthouse audit (after trace completes)

Once the trace is done, run Lighthouse on the already-loaded page using snapshot mode (avoids re-navigating and conflicting with trace data):
```
mcp__chrome-devtools__lighthouse_audit(device="desktop", mode="snapshot")
```

---

## Step 4 — Analyze all available insights

For each insight the trace reports as available, call `mcp__chrome-devtools__performance_analyze_insight`. The common ones (analyze all that appear):

- `INPBreakdown` — phases behind the longest interaction
- `RenderBlocking` — CSS/JS holding up first paint
- `ThirdParties` — third-party transfer sizes and main-thread time
- `CLSCulprits` — elements causing layout shifts
- `FontDisplay` — font loading impact on FCP
- `DOMSize` — DOM node count affecting style recalc

You can fire all of these in the same turn (parallel tool calls).

---

## Step 5 — Deliver the report

Structure the report exactly as shown below. Fill in real numbers from the trace and Lighthouse results. Use the thresholds in the rating column to determine Good / Needs Improvement / Poor.

### Report template

---

## Performance Report: `<url>`

### Core Web Vitals (Field Data — real users, p75)

| Metric | Value | Rating |
|--------|-------|--------|
| LCP (Largest Contentful Paint) | X ms | Good / Needs Improvement / Poor |
| INP (Interaction to Next Paint) | X ms | Good / Needs Improvement / Poor |
| CLS (Cumulative Layout Shift) | X | Good / Needs Improvement / Poor |

**Thresholds:**
- LCP: Good ≤ 2500ms, Needs Improvement ≤ 4000ms, Poor > 4000ms
- INP: Good ≤ 200ms, Needs Improvement ≤ 500ms, Poor > 500ms
- CLS: Good ≤ 0.1, Needs Improvement ≤ 0.25, Poor > 0.25

> Field data comes from CrUX (Chrome User Experience Report) — real Chrome users at p75.
> If field data is unavailable, note that and use lab metrics instead.

---

### Lighthouse Scores

| Category | Score |
|----------|-------|
| Accessibility | X / 100 |
| Best Practices | X / 100 |
| SEO | X / 100 |

*(Note: Lighthouse performance score requires a separate trace — field metrics above are more reliable for performance.)*

---

### Key Issues

For each issue, explain what it is, give the specific numbers from the trace, and explain why it matters to the user experience. Prioritize by impact.

**Structure each issue like:**

#### 1. [Issue Name] — [Metric Affected]
- What: brief description
- Data: specific numbers from the insight
- Impact: why this hurts the user

---

### Recommendations

List recommendations in priority order (biggest impact first). For each:
- What to do
- Why it helps
- The specific data point that motivates it

---

## Tips

- If field data (CrUX) is available, lead with it — it reflects real users, not just this machine.
- If an insight is listed as available but returns no data, note it briefly and move on.
- Keep the report scannable: use tables and headers. Avoid walls of text.
- When LCP TTFB is high (> 800ms), call it out explicitly — slow server response is often the single biggest lever.
- When third-party main-thread time is > 500ms total, flag the offenders by name (e.g. "Google Tag Manager: 624ms").
