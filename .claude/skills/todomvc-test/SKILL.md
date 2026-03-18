---
name: todomvc-test
description: >
  Runs a full suite of TodoMVC test scenarios using playwright-cli in headed mode.
  Covers all happy-path and edge-case flows, verifies each with eval, takes screenshots
  only on failure, and outputs a pass/fail report table.
  Triggers on: "test todomvc", "run todomvc tests", "test this todo app",
  "run playwright todomvc", "todomvc test suite".
allowed-tools: Bash(playwright-cli:*)
---

# TodoMVC Test Suite

## Overview

Runs 11 standard TodoMVC scenarios against any TodoMVC-compatible URL using playwright-cli
in headed mode. Each scenario is verified with `playwright-cli eval`. Screenshots are taken
**only on failure**. A markdown report table is printed at the end.

## When to use

- "test todomvc"
- "run todomvc tests"
- "test this todo app"
- "run playwright todomvc"
- "todomvc test suite"

## Parameters

The user must supply a URL. If not provided, default to `https://demo.playwright.dev/todomvc/`.

---

## Steps

### Step 1 — Open browser

```bash
playwright-cli open <URL> --headed
```

If the browser fails to open, stop and report the error.

Take a snapshot immediately after to confirm the page loaded and get the input ref (`e8` is
typically the "What needs to be done?" textbox — always confirm from the snapshot).

---

### Step 2 — Run each scenario in order

For every scenario below:
1. Perform the actions
2. Run the verification `eval`
3. If verification **passes** → record PASS, continue
4. If verification **fails** → record FAIL, take a screenshot named `<NN>-<scenario-name>-fail.png`, continue

**Never take a screenshot on a passing scenario.**

After any `playwright-cli` command that may change refs (page navigation, re-render),
run `playwright-cli snapshot` and read the new snapshot file to get updated refs before
the next interaction.

---

#### Scenario 01 — Add a todo

```bash
playwright-cli fill e8 "Buy groceries"
playwright-cli press Enter
playwright-cli eval "document.querySelectorAll('.todo-list li').length"
# Expected: 1
```

---

#### Scenario 02 — Complete a todo

Check the "Toggle Todo" checkbox on the first item.

```bash
playwright-cli check <toggle-ref>
playwright-cli eval "document.querySelectorAll('.todo-list li.completed').length"
# Expected: 1
```

---

#### Scenario 03 — Delete a todo

Add a second todo first so the list is not empty after deletion.

```bash
playwright-cli fill e8 "Walk the dog"
playwright-cli press Enter
```

Use `run-code` to hover and click the hidden destroy button on the first item:

```bash
playwright-cli run-code "async page => { await page.locator('.todo-list li').first().hover(); await page.locator('.todo-list li').first().locator('.destroy').click(); }"
playwright-cli eval "document.querySelectorAll('.todo-list li').length"
# Expected: 1
```

---

#### Scenario 04 — Edit a todo (double-click)

Double-click the todo label to enter edit mode. Read the snapshot to find the `Edit` textbox ref.

```bash
playwright-cli dblclick <todo-label-ref>
# read snapshot → find textbox "Edit" ref
playwright-cli fill <edit-ref> "Walk the cat"
playwright-cli press Enter
playwright-cli eval "document.querySelector('.todo-list li label').textContent"
# Expected: "Walk the cat"
```

---

#### Scenario 05 — Filter: Active

Add a second todo. Complete the first one. Then click the Active filter link.

```bash
playwright-cli click <active-link-ref>
playwright-cli eval "document.querySelectorAll('.todo-list li').length"
# Expected: 1 (only the uncompleted item)
```

---

#### Scenario 06 — Filter: Completed

Click the Completed filter link.

```bash
playwright-cli click <completed-link-ref>
playwright-cli eval "document.querySelectorAll('.todo-list li').length"
# Expected: 1 (only the completed item)
```

---

#### Scenario 07 — Clear completed

Switch back to All filter. Click "Clear completed".

```bash
playwright-cli run-code "async page => { await page.getByRole('button', { name: 'Clear completed' }).click(); }"
playwright-cli eval "document.querySelectorAll('.todo-list li.completed').length"
# Expected: 0
```

---

#### Scenario 08 — Mark all complete

Add a second todo if the list has fewer than 2. Check the toggle-all checkbox.

```bash
playwright-cli run-code "async page => { await page.locator('.toggle-all').check(); }"
playwright-cli eval "document.querySelectorAll('.todo-list li.completed').length"
# Expected: equals total item count
```

---

#### Scenario 09 — Empty todo blocked

```bash
playwright-cli fill e8 ""
playwright-cli press Enter
playwright-cli eval "document.querySelectorAll('.todo-list li').length"
# Expected: count unchanged
```

---

#### Scenario 10 — Whitespace-only todo blocked

```bash
playwright-cli fill e8 "     "
playwright-cli press Enter
playwright-cli eval "document.querySelectorAll('.todo-list li').length"
# Expected: count unchanged
```

---

#### Scenario 11 — Edit to empty deletes item

Double-click a todo, clear the edit field, press Enter.

```bash
playwright-cli run-code "async page => { await page.locator('.todo-list li').first().getByTestId('todo-title').dblclick(); }"
# read snapshot → find Edit textbox ref
playwright-cli fill <edit-ref> ""
playwright-cli press Enter
playwright-cli eval "document.querySelectorAll('.todo-list li').length"
# Expected: count reduced by 1 (item deleted)
```

---

### Step 3 — Close browser

```bash
playwright-cli close
```

---

### Step 4 — Output report

Print a markdown table summarising all scenarios:

```
| # | Scenario | Result | Notes |
|---|----------|--------|-------|
| 01 | Add a todo | PASS / FAIL | |
...
```

Include a one-line summary: `X/11 scenarios passed.`

If any scenario failed, list the screenshot filenames so the user can inspect them.

---

## Expected output

- All 11 scenarios reported as PASS
- No screenshots generated (screenshots only appear on failure)
- A markdown table printed in the conversation

## Failure handling

- If `playwright-cli open` fails → stop immediately, report the error, do not continue
- If a browser ref is stale (ref not found error) → run `playwright-cli snapshot`, read the
  new snapshot file, resolve the correct ref, and retry the action once
- If a scenario verification fails → record FAIL, screenshot, move to the next scenario
- Do not abort the run on a single scenario failure — always complete all 11 and report
