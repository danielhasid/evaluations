#!/usr/bin/env python3
"""Parallel scraping using Apify code_crafter/leads-finder with geographic partitioning."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from apify_client import ApifyClient
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

REGION_PARTITIONS = {
    "united states": [
        "Northeast United States",
        "Southeast United States",
        "Midwest United States",
        "West United States",
    ],
    "eu": [
        "Western Europe",
        "Southern Europe",
        "Northern Europe",
        "Eastern Europe",
    ],
    "europe": [
        "Western Europe",
        "Southern Europe",
        "Northern Europe",
        "Eastern Europe",
    ],
    "uk": [
        "SE England, UK",
        "North England, UK",
        "Scotland and Wales, UK",
        "SW England, UK",
    ],
    "united kingdom": [
        "SE England, UK",
        "North England, UK",
        "Scotland and Wales, UK",
        "SW England, UK",
    ],
    "canada": [
        "Ontario, Canada",
        "Quebec, Canada",
        "Western Canada",
        "Atlantic Canada",
    ],
    "australia": [
        "New South Wales, Australia",
        "Victoria and Tasmania, Australia",
        "Queensland, Australia",
        "Western and South Australia",
    ],
}


def scrape_region(client: ApifyClient, query: str, location: str, max_items: int, no_email_filter: bool) -> list:
    run_input = {
        "searchQuery": query,
        "location": location,
        "maxItems": max_items,
        "includeEmails": not no_email_filter,
    }
    print(f"  Starting region: {location} ({max_items} items)")
    run = client.actor("code_crafter/leads-finder").call(run_input=run_input)
    if not run:
        print(f"  WARNING: No result for region {location}", file=sys.stderr)
        return []
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    print(f"  Done: {location} → {len(items)} leads")
    return items


def deduplicate(leads: list) -> list:
    seen = set()
    unique = []
    for lead in leads:
        key = lead.get("website") or lead.get("name") or str(lead)
        if key not in seen:
            seen.add(key)
            unique.append(lead)
    return unique


def main():
    parser = argparse.ArgumentParser(description="Parallel Apify scraping with geographic partitioning")
    parser.add_argument("--query", required=True, help="Industry/business type")
    parser.add_argument("--location", required=True, help="Target location")
    parser.add_argument("--total_count", type=int, default=1000, help="Total leads desired")
    parser.add_argument("--strategy", default="regions", choices=["regions"], help="Partitioning strategy")
    parser.add_argument("--no-email-filter", action="store_true", help="Skip email filtering")
    parser.add_argument("--output", default=".tmp/leads.json", help="Output JSON file path")
    args = parser.parse_args()

    token = os.environ.get("APIFY_API_TOKEN")
    if not token:
        print("ERROR: APIFY_API_TOKEN not set in .env", file=sys.stderr)
        sys.exit(1)

    client = ApifyClient(token)
    location_key = args.location.lower()
    regions = REGION_PARTITIONS.get(location_key)

    if not regions:
        print(f"No regional partition for '{args.location}'. Falling back to single scrape.")
        regions = [args.location]

    items_per_region = (args.total_count // len(regions)) + 1
    print(f"Parallel scrape: {len(regions)} regions × {items_per_region} items")

    all_leads = []
    with ThreadPoolExecutor(max_workers=len(regions)) as executor:
        futures = {
            executor.submit(
                scrape_region, client, args.query, region, items_per_region, args.no_email_filter
            ): region
            for region in regions
        }
        for future in as_completed(futures):
            try:
                all_leads.extend(future.result())
            except Exception as e:
                print(f"  ERROR in region {futures[future]}: {e}", file=sys.stderr)

    unique_leads = deduplicate(all_leads)
    final_leads = unique_leads[: args.total_count]
    print(f"\nTotal: {len(all_leads)} scraped → {len(unique_leads)} unique → {len(final_leads)} kept")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_leads, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
