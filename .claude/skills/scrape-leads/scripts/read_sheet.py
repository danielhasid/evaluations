#!/usr/bin/env python3
"""Read data from a Google Sheet and print as JSON."""

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import gspread
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

load_dotenv(Path(__file__).parent.parent / ".env")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]


def extract_sheet_id(sheet_url: str) -> str:
    parts = sheet_url.split("/")
    try:
        idx = parts.index("d")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return sheet_url


def main():
    if len(sys.argv) < 2:
        print("Usage: read_sheet.py <SHEET_URL_OR_ID> [--output FILE]", file=sys.stderr)
        sys.exit(1)

    sheet_url = sys.argv[1]
    output = None
    if "--output" in sys.argv:
        idx = sys.argv.index("--output")
        output = sys.argv[idx + 1]

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS not set in .env", file=sys.stderr)
        sys.exit(1)

    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    gc = gspread.authorize(creds)

    sheet_id = extract_sheet_id(sheet_url)
    sheet = gc.open_by_key(sheet_id)
    worksheet = sheet.get_worksheet(0)
    records = worksheet.get_all_records()

    print(f"Read {len(records)} rows from '{sheet.title}'", file=sys.stderr)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output}", file=sys.stderr)
    else:
        print(json.dumps(records, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
