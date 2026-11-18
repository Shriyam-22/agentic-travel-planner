"""Command-line interface for the travel planner.

Examples:
    # LLM agent, free-text request
    python cli.py --query "4 days in Jaipur from Mumbai from 2026-12-12 for 2, love forts"

    # LLM agent, structured inputs
    python cli.py --from Delhi --to Goa --start 2026-12-01 --days 4 --travellers 2

    # No API key needed
    python cli.py --from Hyderabad --to Delhi --start 2026-12-01 --days 3 --offline
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

from travel_agent import config
from travel_agent.agent import AgentError, build_query, run_agent
from travel_agent.formatter import to_markdown
from travel_agent.planner import plan_trip


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic AI travel planner (LangChain)")
    parser.add_argument("--query", help="Free-text trip request (agent mode)")
    parser.add_argument("--from", dest="origin", help="Origin city")
    parser.add_argument("--to", dest="destination", help="Destination city")
    parser.add_argument("--start", default=(date.today() + timedelta(days=14)).isoformat(), help="YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=3, choices=range(1, 8), metavar="1-7")
    parser.add_argument("--travellers", type=int, default=1)
    parser.add_argument("--style", choices=["budget", "standard", "luxury"], default="standard")
    parser.add_argument("--budget", type=int, help="Total budget in INR")
    parser.add_argument("--interests", nargs="*", default=[], help="Place types, e.g. beach temple")
    parser.add_argument("--amenities", nargs="*", default=[], help="Hotel amenities, e.g. wifi pool")
    parser.add_argument("--min-stars", type=int, default=0)
    parser.add_argument("--flight-pref", choices=["best_value", "cheapest", "fastest"], default="best_value")
    parser.add_argument("--offline", action="store_true", help="Use the rule-based planner (no LLM)")
    parser.add_argument("--provider", default=config.LLM_PROVIDER, choices=list(config.DEFAULT_MODELS))
    parser.add_argument("--model", help="LLM model name (defaults per provider)")
    parser.add_argument("--save", action="store_true", help="Save JSON + Markdown to outputs/")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show the agent's tool calls")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    steps = []
    try:
        if args.offline:
            if not (args.origin and args.destination):
                print("Offline mode needs --from and --to.", file=sys.stderr)
                return 2
            plan = plan_trip(args.origin, args.destination, args.start, args.days, args.travellers,
                             args.style, args.budget, args.interests, args.min_stars, args.amenities,
                             args.flight_pref)
        else:
            query = args.query
            if not query:
                if not (args.origin and args.destination):
                    print("Give --query, or --from and --to.", file=sys.stderr)
                    return 2
                query = build_query(args.origin, args.destination, args.start, args.days, args.travellers,
                                    args.style, args.budget, args.interests, args.min_stars,
                                    args.amenities, args.flight_pref)
            print(f"🤖 Agent request: {query}\n")
            result = run_agent(query, provider=args.provider, model=args.model)
            plan, steps = result.plan, result.steps
    except (AgentError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.verbose and steps:
        print("Agent trace:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step.tool}({step.arguments})")
        print()

    markdown = to_markdown(plan)
    print(markdown)
    if args.save:
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        stem = config.OUTPUT_DIR / f"trip_{plan.destination.lower()}_{plan.start_date}"
        stem.with_suffix(".json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        stem.with_suffix(".md").write_text(markdown, encoding="utf-8")
        print(f"Saved {stem}.json and {stem}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
