#!/usr/bin/env python3
"""Upload leads JSON to a new Google Sheet."""

import argparse
import json
import os
import sys
from pathlib import Path

import gspread
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

load_dotenv(Path(__file__).parent.parent / ".env")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

COLUMNS = [
    "name",
    "website",
    "phone",
    "email",
    "address",
    "city",
    "state",
    "country",
    "categories",
    "description",
    "rating",
    "reviewsCount",
    "classification",
]


def get_cell_value(lead: dict, col: str) -> str:
    val = lead.get(col, "")
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val) if val is not None else ""


def main():
    parser = argparse.ArgumentParser(description="Upload leads to Google Sheets")
    parser.add_argument("input", help="Input JSON file with leads")
    parser.add_argument("--title", default="Leads", help="Google Sheet title")
    args = parser.parse_args()

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS not set in .env", file=sys.stderr)
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        leads = json.load(f)

    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    gc = gspread.authorize(creds)

    print(f"Creating Google Sheet: '{args.title}'")
    sheet = gc.create(args.title)
    worksheet = sheet.get_worksheet(0)

    # Write header
    header = [col.capitalize() for col in COLUMNS]
    rows = [header]

    for lead in leads:
        row = [get_cell_value(lead, col) for col in COLUMNS]
        rows.append(row)

    worksheet.update("A1", rows)
    worksheet.freeze(rows=1)

    # Make the sheet publicly readable
    sheet.share(None, perm_type="anyone", role="reader")

    url = f"https://docs.google.com/spreadsheets/d/{sheet.id}"
    print(f"Sheet created with {len(leads)} leads.")
    print(f"URL: {url}")
    return url


if __name__ == "__main__":
    main()
