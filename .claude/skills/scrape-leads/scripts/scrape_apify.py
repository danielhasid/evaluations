#!/usr/bin/env python3
"""Single scrape using Apify code_crafter/leads-finder actor."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


def format_location(location: str) -> str:
    """Auto-format location: US states get ', us' suffix."""
    us_states = {
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
        "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
        "maine", "maryland", "massachusetts", "michigan", "minnesota",
        "mississippi", "missouri", "montana", "nebraska", "nevada",
        "new hampshire", "new jersey", "new mexico", "new york",
        "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
        "pennsylvania", "rhode island", "south carolina", "south dakota",
        "tennessee", "texas", "utah", "vermont", "virginia", "washington",
        "west virginia", "wisconsin", "wyoming",
    }
    if location.lower() in us_states:
        return f"{location}, us"
    if location.lower() == "united states":
        return "United States"
    return location


def scrape(query: str, location: str, max_items: int, no_email_filter: bool, output: str):
    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        print("ERROR: APIFY_API_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)

    client = ApifyClient(token)
    formatted_location = format_location(location)

    run_input = {
        "searchQuery": query,
        "location": formatted_location,
        "maxItems": max_items,
        "includeEmails": not no_email_filter,
    }

    print(f"Starting Apify scrape: query='{query}', location='{formatted_location}', max_items={max_items}")
    run = client.actor("code_crafter/leads-finder").call(run_input=run_input)

    if not run:
        print("ERROR: Apify run failed or returned no result.", file=sys.stderr)
        sys.exit(1)

    print(f"Run finished. Dataset ID: {run['defaultDatasetId']}")

    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    print(f"Fetched {len(items)} leads.")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Scrape leads from Apify")
    parser.add_argument("--query", required=True, help="Industry/business type to search")
    parser.add_argument("--location", required=True, help="Location to search in")
    parser.add_argument("--max_items", type=int, default=100, help="Maximum number of leads")
    parser.add_argument("--no-email-filter", action="store_true", help="Skip email filtering")
    parser.add_argument("--output", default=".tmp/leads.json", help="Output JSON file path")
    args = parser.parse_args()

    scrape(
        query=args.query,
        location=args.location,
        max_items=args.max_items,
        no_email_filter=args.no_email_filter,
        output=args.output,
    )


if __name__ == "__main__":
    main()
