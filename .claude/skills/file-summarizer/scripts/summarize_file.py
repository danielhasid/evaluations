#!/usr/bin/env python3
"""
File Summarizer Script
Reads any file and prints its content for Claude to analyze.
Claude handles the actual summarization and test case suggestion.
"""

import argparse
import os
import sys


SUPPORTED_EXTENSIONS = {
    ".py": "Python",
    ".ts": "TypeScript",
    ".js": "JavaScript",
    ".spec.ts": "Playwright Test (TypeScript)",
    ".spec.js": "Playwright Test (JavaScript)",
    ".md": "Markdown",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".env": "Environment Config",
    ".txt": "Plain Text",
    ".html": "HTML",
    ".xml": "XML",
    ".csv": "CSV",
    ".sql": "SQL",
    ".sh": "Shell Script",
    ".feature": "Gherkin / BDD Feature",
}


def detect_file_type(file_path: str) -> str:
    name = os.path.basename(file_path).lower()
    # Check compound extensions first (e.g. .spec.ts)
    for ext, label in SUPPORTED_EXTENSIONS.items():
        if name.endswith(ext):
            return label
    return "Unknown"


def read_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        print(f"ERROR: Could not read file: {e}", file=sys.stderr)
        sys.exit(1)


def build_prompt(file_path: str, content: str) -> str:
    file_type = detect_file_type(file_path)
    filename = os.path.basename(file_path)
    lines = content.splitlines()
    line_count = len(lines)

    prompt = f"""## File to Summarize

**File:** `{filename}`
**Type:** {file_type}
**Lines:** {line_count}
**Path:** {file_path}

---

```
{content}
```

---

Please provide a structured summary following this format:

## Summary: {filename}

### Overview
<1-3 sentences describing the purpose of this file>

### Key Components
- <component name>: <what it does>

### Dependencies
- <import or dependency>: <its role>

### Suggested Test Cases
- [ ] <test case 1 — what to verify>
- [ ] <test case 2 — what to verify>
- [ ] <add as many as relevant>

---
Would you like me to generate Playwright/pytest test code for any of the above test cases?
"""
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a file for QA review with test case suggestions."
    )
    parser.add_argument(
        "--file", "-f", required=True, help="Path to the file to summarize"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Optional: save output to a markdown file"
    )
    args = parser.parse_args()

    content = read_file(args.file)
    prompt = build_prompt(args.file, content)

    print(prompt)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(prompt)
            print(f"\n[Saved to {args.output}]", file=sys.stderr)
        except Exception as e:
            print(f"WARNING: Could not save output: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
