"""Render a TripPlan as human-readable Markdown (CLI output and downloads)."""

from __future__ import annotations

from travel_agent.schemas import FlightChoice, TripPlan


def inr(amount: int | float | None) -> str:
    return "—" if amount is None else f"₹{amount:,.0f}"


def _flight_line(flight: FlightChoice | None) -> str:
    if flight is None:
        return "- No flight found in the dataset"
    route = f"{flight.origin} → {flight.destination}"
    via = f" via {', '.join(flight.via)}" if flight.via else ""
    return (
        f"- **{' + '.join(flight.airlines)}** ({', '.join(flight.flight_ids)}) {route}{via} — "
        f"{flight.travel_date}, departs {flight.departs_at}, arrives {flight.arrives_at} "
        f"({flight.duration}, {flight.stops} stop(s)) — {inr(flight.price_per_person)}/person\n"
        f"  - _Why:_ {flight.reason}"
    )


def to_markdown(plan: TripPlan) -> str:
    """Full itinerary in the format of the project brief's example output."""
    lines = [f"# {plan.trip_title}", "", plan.trip_summary, ""]

    lines += ["## Flights", "**Outbound**", _flight_line(plan.outbound_flight),
              "**Return**", _flight_line(plan.return_flight), ""]

    lines.append("## Hotel")
    if plan.hotel:
        h = plan.hotel
        lines += [
            f"- **{h.name}** ({h.stars}★) — {inr(h.price_per_night)}/night; amenities: {', '.join(h.amenities) or '—'}",
            f"  - _Why:_ {h.reason}",
        ]
    else:
        lines.append("- No hotel found")
    lines.append("")

    lines += ["## Weather", f"_{plan.weather_source}_", ""]
    lines += [f"- Day {d.day} ({d.date}): {d.weather}" for d in plan.daily_plan]
    lines.append("")

    lines.append("## Itinerary")
    for d in plan.daily_plan:
        lines.append(f"### Day {d.day} — {d.date}: {d.title}")
        for a in d.activities:
            extra = f" ({a.place_type}, ★{a.rating})" if a.place_type else ""
            note = f" — {a.note}" if a.note else ""
            lines.append(f"- {a.time_slot}: {a.name}{extra}{note}")
        lines.append("")

    b = plan.budget
    lines += [
        "## Estimated Budget",
        f"- Flights: {inr(b.flights)}",
        f"- Hotel: {inr(b.hotel)}",
        f"- Food & Local Travel: {inr(b.food_and_local_travel)}",
        "-------------------------------------",
        f"**Total Cost: {inr(b.total)}**  ({inr(b.per_person)} per person)",
    ]
    if b.within_budget is not None:
        lines.append(f"- Within your budget: {'Yes ✅' if b.within_budget else 'No ⚠️'}")
    if b.notes:
        lines.append(f"- _{b.notes}_")
    lines.append("")

    if plan.reasoning:
        lines += ["## Why we chose this"] + [f"- {r}" for r in plan.reasoning] + [""]
    if plan.warnings:
        lines += ["## Things to know"] + [f"- {w}" for w in plan.warnings] + [""]
    return "\n".join(lines)
