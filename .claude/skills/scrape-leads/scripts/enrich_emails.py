#!/usr/bin/env python3
"""Enrich missing emails in a Google Sheet via AnyMailFinder API."""

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import gspread
import requests
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

load_dotenv(Path(__file__).parent.parent / ".env")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

ANYMAILFINDER_API = "https://api.anymailfinder.com/v5.0/search/company.json"


def extract_sheet_id(sheet_url: str) -> str:
    parts = sheet_url.split("/")
    try:
        idx = parts.index("d")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return sheet_url  # assume it's already an ID


def get_domain(website: str) -> str:
    if not website:
        return ""
    if not website.startswith("http"):
        website = "https://" + website
    return urlparse(website).netloc.lstrip("www.")


def find_email(api_key: str, domain: str) -> str:
    if not domain:
        return ""
    try:
        resp = requests.post(
            ANYMAILFINDER_API,
            json={"domain": domain},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        data = resp.json()
        emails = data.get("emails", [])
        if emails:
            return emails[0].get("email", "")
        return ""
    except Exception as e:
        print(f"  Warning: email lookup failed for {domain}: {e}", file=sys.stderr)
        return ""


def main():
    if len(sys.argv) < 2:
        print("Usage: enrich_emails.py <SHEET_URL_OR_ID>", file=sys.stderr)
        sys.exit(1)

    sheet_url = sys.argv[1]
    api_key = os.environ.get("ANYMAILFINDER_API_KEY")
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not api_key:
        print("ERROR: ANYMAILFINDER_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)
    if not creds_path:
        print("ERROR: GOOGLE_APPLICATION_CREDENTIALS not set in .env", file=sys.stderr)
        sys.exit(1)

    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    gc = gspread.authorize(creds)

    sheet_id = extract_sheet_id(sheet_url)
    sheet = gc.open_by_key(sheet_id)
    worksheet = sheet.get_worksheet(0)

    records = worksheet.get_all_records()
    headers = worksheet.row_values(1)

    email_col = next((i + 1 for i, h in enumerate(headers) if h.lower() == "email"), None)
    website_col = next((i + 1 for i, h in enumerate(headers) if h.lower() == "website"), None)

    if not email_col or not website_col:
        print("ERROR: Sheet must have 'Email' and 'Website' columns.", file=sys.stderr)
        sys.exit(1)

    enriched = 0
    for row_idx, record in enumerate(records, start=2):
        email = record.get("Email") or record.get("email", "")
        website = record.get("Website") or record.get("website", "")
        if email or not website:
            continue

        domain = get_domain(website)
        if not domain:
            continue

        found_email = find_email(api_key, domain)
        if found_email:
            worksheet.update_cell(row_idx, email_col, found_email)
            enriched += 1
            print(f"  Enriched row {row_idx}: {domain} → {found_email}")
            time.sleep(0.3)  # rate limit

    print(f"Done. Enriched {enriched} emails.")
    print(f"Sheet: {sheet_url}")


if __name__ == "__main__":
    main()
