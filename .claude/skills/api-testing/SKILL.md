---
name: api-testing
description: HTTP API testing for external or local REST APIs using Python (requests, pytest). Automatically scaffolds a tests/ directory with test files, conftest.py, requirements, and runs the suite. Trigger when user says "test API", "run api tests", "test this endpoint", or provides a base URL to test.
allowed-tools: Bash, Read, Edit, Write, Grep, Glob, TodoWrite
---

# API Testing Skill

When invoked, this skill **creates a complete Python test suite** for the given API URL and runs it. Do not just explain how to test — always scaffold the files and execute the tests.

## Step 1: Gather Info

Ask or infer:
- `BASE_URL` — the API base URL (e.g. `https://api.example.com/objects`)
- `OUTPUT_DIR` — where to write test files (default: `./tests/api/`)

## Step 1.5: Check for Existing Output Directory

Before probing the API, check if `OUTPUT_DIR` already exists:

```bash
[ -d "OUTPUT_DIR" ] && echo "EXISTS" || echo "NOT_FOUND"
```

If the directory **exists**, use the `AskUserQuestion` tool to ask:

```
question: "The folder 'OUTPUT_DIR' already exists. What would you like to do?"
header: "Folder conflict"
options:
  - label: "Overwrite it"
    description: "Replace all existing test files in OUTPUT_DIR"
  - label: "Use a new folder"
    description: "I'll tell you the new folder name to use instead"
```

- If the user chooses **"Overwrite it"** — proceed with `OUTPUT_DIR` as-is.
- If the user chooses **"Use a new folder"** — wait for them to provide the new folder name, set `OUTPUT_DIR` to that name, and use it for all subsequent steps.

If the directory **does not exist**, create it and proceed normally:

```bash
mkdir -p OUTPUT_DIR
```

## Step 2: Probe the API

Before writing tests, make a quick GET request to understand the API shape:

```bash
PYTHONIOENCODING=utf-8 python -c "
import requests, json
r = requests.get('BASE_URL', timeout=10)
print('Status:', r.status_code)
print('Headers:', dict(r.headers))
print('Body:', json.dumps(r.json(), indent=2)[:2000])
"
```

Use the response to understand:
- Response shape (list vs object, field names, ID types)
- Status codes returned
- Auth requirements

## Step 3: Scaffold the Test Directory

Create these files in `OUTPUT_DIR`:

### `conftest.py`
```python
import sys, io
import pytest
import requests

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

BASE_URL = "https://your-api.com/endpoint"  # replace with actual URL

@pytest.fixture(scope="session")
def client():
    s = requests.Session()
    s.timeout = 10
    yield s
    s.close()

@pytest.fixture(scope="session")
def base_url():
    return BASE_URL
```

### `test_crud.py`
Cover the full CRUD cycle:
- `test_get_all` — GET collection, assert 200, list response
- `test_get_single` — GET one item by ID, assert fields present
- `test_create` — POST new item, assert ID returned
- `test_update` — PUT/PATCH item, assert change persisted
- `test_delete` — DELETE item, assert 200/204
- `test_not_found` — GET nonexistent ID, assert 404

### `test_validation.py`
Cover error and edge cases:
- `test_invalid_id` — bad ID format
- `test_missing_fields` — POST without required fields
- `test_response_content_type` — assert `application/json` header

### `test_performance.py`
Basic perf checks:
- `test_response_time` — assert response under 3000ms
- `test_concurrent_requests` — fire 5 requests using threads, all succeed

### `requirements.txt`
```
requests
pytest
```

## Step 4: Install Dependencies

```bash
PYTHONIOENCODING=utf-8 pip install requests pytest 2>&1 | tail -3
```

## Step 5: Run the Tests

```bash
PYTHONIOENCODING=utf-8 python -m pytest OUTPUT_DIR -v -s 2>&1
```

## Step 6: Report Results

After running, output a summary table:

```
| Test                        | Result  | Notes                     |
|-----------------------------|---------|---------------------------|
| test_get_all                | PASSED  | 13 objects returned       |
| test_get_single             | PASSED  | id=1, name=Google Pixel   |
| test_create                 | PASSED  | id=abc123                 |
| test_update                 | PASSED  |                           |
| test_delete                 | PASSED  |                           |
| test_not_found              | PASSED  | 404 confirmed             |
| test_response_content_type  | PASSED  | application/json          |
| test_response_time          | PASSED  | 294ms                     |
```

If any test fails, show the assertion error and suggest a fix.

## Important Rules

- Always use `PYTHONIOENCODING=utf-8` when running pytest on Windows to avoid encoding errors with special characters
- Always create `conftest.py` with the shared `client` (`requests.Session`) and `base_url` fixtures
- Use `requests` (not `httpx`) throughout — including inline in `test_performance.py` concurrent test
- Adapt field names and status codes based on the actual API response shape from Step 2
- If the API requires auth, add an `auth_headers` fixture in `conftest.py`
- Default test output directory is `tests/api/` relative to the working directory

## Example Invocation

User: "run api tests to https://api.restful-api.dev/objects"

1. Check if `tests/api/` exists
   - If yes → ask user: overwrite or use a new folder name?
   - If user says new folder → use that name as `OUTPUT_DIR`
   - If no → create it
2. Probe GET https://api.restful-api.dev/objects
3. Create `OUTPUT_DIR/conftest.py`, `OUTPUT_DIR/test_crud.py`, `OUTPUT_DIR/test_validation.py`, `OUTPUT_DIR/test_performance.py`, `OUTPUT_DIR/requirements.txt`
4. Run `PYTHONIOENCODING=utf-8 python -m pytest OUTPUT_DIR/ -v -s`
5. Print results table
