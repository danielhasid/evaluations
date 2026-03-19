#!/usr/bin/env python3
"""LLM-based lead classification using Claude."""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

CLASSIFICATION_PROMPTS = {
    "product_saas": (
        "Classify this business as one of: 'product_saas' (sells software/SaaS products), "
        "'agency' (provides services/consulting), 'unclear' (can't determine). "
        "Respond with ONLY the classification label."
    ),
    "industry_match": (
        "Classify this business as 'match' if it clearly belongs to the target industry, "
        "or 'no_match' if it does not. Respond with ONLY 'match' or 'no_match'."
    ),
}


def classify_lead(client: anthropic.Anthropic, lead: dict, prompt: str) -> str:
    name = lead.get("name", "")
    description = lead.get("description", "") or lead.get("categories", "")
    website = lead.get("website", "")

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    f"Business name: {name}\n"
                    f"Description: {description}\n"
                    f"Website: {website}"
                ),
            }
        ],
    )
    return message.content[0].text.strip().lower()


def main():
    parser = argparse.ArgumentParser(description="Classify leads using Claude LLM")
    parser.add_argument("input", help="Input JSON file with leads")
    parser.add_argument(
        "--classification_type",
        default="product_saas",
        choices=list(CLASSIFICATION_PROMPTS.keys()),
        help="Type of classification to perform",
    )
    parser.add_argument("--output", default=".tmp/classified_leads.json", help="Output JSON file")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set in .env", file=sys.stderr)
        sys.exit(1)

    with open(args.input, encoding="utf-8") as f:
        leads = json.load(f)

    client = anthropic.Anthropic(api_key=api_key)
    prompt = CLASSIFICATION_PROMPTS[args.classification_type]

    print(f"Classifying {len(leads)} leads as '{args.classification_type}'...")
    results = []
    unclear_count = 0

    for i, lead in enumerate(leads, 1):
        label = classify_lead(client, lead, prompt)
        lead["classification"] = label
        results.append(lead)
        if "unclear" in label:
            unclear_count += 1
        if i % 10 == 0:
            print(f"  {i}/{len(leads)} classified...")

    unclear_pct = (unclear_count / len(leads) * 100) if leads else 0
    print(f"Done. Unclear: {unclear_count}/{len(leads)} ({unclear_pct:.1f}%)")
    if unclear_pct > 80:
        print("WARNING: >80% unclear — consider refining your scrape keywords.", file=sys.stderr)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
