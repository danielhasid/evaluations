---
name: skill-to-pytest
description: >
  Reads a SKILL.md file and converts its steps and scenarios into a Python pytest file
  with fixtures, test functions, docstrings, and assertions. Optionally generates a
  conftest.py for shared fixtures.
  Triggers on: "convert skill to pytest", "generate pytest from skill",
  "turn skill into python tests", "make pytest file from skill", "skill to python".
allowed-tools: Read, Write
---

# Skill → Pytest Converter

## Overview

Reads any SKILL.md file, parses its structure, and produces a ready-to-run Python pytest
file. Each scenario or step in the skill becomes a `test_` function. Setup and teardown
logic becomes a pytest fixture. Imports and assertions are inferred from the skill's
tool usage and verification steps.

## When to use

- "convert skill to pytest"
- "generate pytest from skill"
- "turn skill into python tests"
- "make pytest file from skill"
- "skill to python"

---

## Step 1 — Locate the SKILL.md

If the user has provided a file path, read that file.
If not, ask: *"Which SKILL.md do you want to convert? Please provide the path."*

Read the full SKILL.md content before proceeding.

---

## Step 2 — Parse the skill

Extract the following from the SKILL.md:

| Field | Where to find it |
|-------|-----------------|
| **Skill name** | `name:` in frontmatter |
| **Description** | `description:` in frontmatter |
| **Tool type** | `allowed-tools:` — determines which libraries to import |
| **Setup steps** | First step(s) that open a resource (browser, DB, server, etc.) |
| **Teardown steps** | Last step(s) that close/clean up the resource |
| **Scenarios** | Each `#### Scenario` or `### Step` heading that has a verification |
| **Verifications** | Lines commented `# Expected:` or `eval` calls with expected values |

---

## Step 3 — Infer imports and fixtures

Based on `allowed-tools`, map to Python libraries:

| allowed-tools value | Python imports |
|---------------------|---------------|
| `Bash(playwright-cli:*)` | `from playwright.sync_api import sync_playwright, Page` |
| `Bash` (generic shell) | `import subprocess` |
| `Read`, `Write` (file ops) | `import os`, `from pathlib import Path` |
| No tools / reasoning only | `import pytest` only |

Always include `import pytest` at the top.

---

## Step 4 — Generate the pytest file

Write a Python file named `test_<skill-name>.py` (replace hyphens with underscores).

### Structure to follow

```python
"""
<Skill description from frontmatter>
Generated from: .claude/skills/<skill-name>/SKILL.md
"""
import pytest
# <inferred imports>


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def <resource_name>():
    """Set up <resource> before each test, tear down after."""
    # <setup code from skill's setup steps>
    yield <resource>
    # <teardown code from skill's teardown steps>


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_<scenario_01_name>(<fixture_args>):
    """<Docstring: one sentence describing what this scenario tests>"""
    # Arrange
    # <any pre-conditions>

    # Act
    # <actions from the scenario steps>

    # Assert
    # <assertion derived from the # Expected: comment or eval verification>
    assert <condition>, "<failure message>"


def test_<scenario_02_name>(<fixture_args>):
    """..."""
    ...
```

### Rules for generating test functions

- **Name:** `test_` + scenario name lowercased with underscores, no numbers prefix
  (e.g. "Scenario 01 — Add a todo" → `test_add_a_todo`)
- **Docstring:** one sentence — what the test does and what it asserts
- **Arrange / Act / Assert** comments to structure the body
- **Assertions:** derive from `# Expected:` comments in the skill. For playwright-cli
  skills, translate `eval` commands to Playwright `page.locator().count()` or
  `page.locator().text_content()` equivalents
- **Edge cases:** scenarios labelled "blocked", "fail", or "edge case" should assert
  that the count/state did NOT change
- Do not leave `pass` in any test — always write a meaningful assertion

### playwright-cli → Playwright Python translation table

| playwright-cli command | Python Playwright equivalent |
|------------------------|------------------------------|
| `playwright-cli open <url>` | `page.goto("<url>")` |
| `playwright-cli fill e8 "text"` | `page.get_by_role("textbox", name="...").fill("text")` |
| `playwright-cli press Enter` | `page.keyboard.press("Enter")` |
| `playwright-cli check <ref>` | `page.get_by_role("checkbox", name="...").check()` |
| `playwright-cli click <ref>` | `page.get_by_role(...).click()` |
| `playwright-cli dblclick <ref>` | `page.get_by_test_id("todo-title").dblclick()` |
| `playwright-cli eval "document.querySelectorAll('.x').length"` | `page.locator(".x").count()` |
| `playwright-cli eval "document.querySelector('.x').textContent"` | `page.locator(".x").text_content()` |
| `playwright-cli run-code "async page => { ... }"` | inline Python equivalent |
| `playwright-cli close` | handled by fixture teardown |

---

## Step 5 — Generate conftest.py (if needed)

If the skill has setup/teardown that applies to all tests (e.g. browser launch),
also write a `conftest.py` in the same directory:

```python
import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        yield browser
        browser.close()


@pytest.fixture
def page(browser):
    page = browser.new_page()
    yield page
    page.close()
```

Only write conftest.py if there are 2 or more test functions that share the same fixture.

---

## Step 6 — Write the files

Write `test_<skill-name>.py` (and optionally `conftest.py`) to the **same directory**
as the source SKILL.md.

---

## Step 7 — Print a summary

Output a short summary:

```
Generated:
  .claude/skills/<skill-name>/test_<skill-name>.py  — <N> test functions
  .claude/skills/<skill-name>/conftest.py            — <N> fixtures  (if created)

Test functions:
  test_add_a_todo
  test_complete_a_todo
  ...

Run with:
  pytest .claude/skills/<skill-name>/test_<skill-name>.py -v
```

---

## Expected output

- A `test_<skill-name>.py` file with one `test_` function per scenario
- A `conftest.py` if shared fixtures are needed
- A printed summary with the file paths and a `pytest` run command

## Failure handling

- If the SKILL.md has no identifiable scenarios or steps, ask the user to clarify
  which sections should become test functions before writing anything
- If a verification step has no clear expected value, write `assert True  # TODO: add assertion`
  and note it in the summary so the user knows to fill it in
- Never overwrite an existing test file without confirming with the user first
