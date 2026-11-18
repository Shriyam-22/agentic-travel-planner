"""LangChain tools exposed to the agent.

Each tool is a thin wrapper around a service function. Tools validate city
names, catch errors and always return a JSON string, so the LLM gets a useful
message instead of a stack trace.
"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.tools import tool

from travel_agent.database import DatabaseError, list_cities, normalise_city
from travel_agent.services import (
    estimate_budget as _estimate_budget,
)
from travel_agent.services import (
    find_flight_options as _find_flight_options,
)
from travel_agent.services import (
    get_weather_forecast as _get_weather_forecast,
)
from travel_agent.services import (
    search_hotels as _search_hotels,
)
from travel_agent.services import (
    search_places as _search_places,
)
from travel_agent.services.hotel_service import list_amenities
from travel_agent.services.places_service import list_place_types

LONG_JOURNEY_MINUTES = 12 * 60


def _to_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _unknown_city(name: str) -> str:
    return _to_json({"error": f"'{name}' is not in the dataset.", "supported_cities": list_cities()})


def _leg_summary(leg: dict) -> str:
    """One compact line per flight leg, e.g. 'FL0024 SpiceJet Mumbai->Goa 14:38-18:38 Rs3304'."""
    return (f"{leg['flight_id']} {leg['airline']} {leg['origin']}->{leg['destination']} "
            f"{leg['departs_at']}-{leg['arrives_at']} Rs{leg['price']}")


def _compact_option(option: dict) -> dict:
    """Small, LLM-friendly view of a direct flight or connecting itinerary.

    Keeps token usage low (free LLM tiers cap tokens per minute) and drops the
    raw 2025 sample timestamps so the model only sees schedule fields.
    """
    if option.get("type") == "connecting":
        return {
            "flight_ids": option["flight_ids"],
            "airlines": option["airlines"],
            "stops": option["stops"],
            "via": option["via"],
            "departs_at": option["departs_at"],
            "arrives_at": option["arrives_at"],
            "days_in_transit": option["days_in_transit"],
            "duration": option["total_duration"],
            "price": option["total_price"],
            "legs": [_leg_summary(leg) for leg in option["legs"]],
        }
    return {
        "flight_ids": [option["flight_id"]],
        "airlines": [option["airline"]],
        "stops": 0,
        "via": [],
        "departs_at": option["departs_at"],
        "arrives_at": option["arrives_at"],
        "days_in_transit": int(option["arrives_next_day"]),
        "duration": option["duration"],
        "price": option["price"],
    }


@tool
def list_supported_destinations() -> str:
    """List the cities, hotel amenities and attraction types available in the travel database.

    Call this first if the user mentions a city you are unsure about.
    """
    try:
        return _to_json(
            {"cities": list_cities(), "amenities": list_amenities(), "place_types": list_place_types()}
        )
    except DatabaseError as exc:
        return _to_json({"error": str(exc)})


@tool
def search_flights(
    origin: str,
    destination: str,
    sort_by: Literal["price", "duration", "departure"] = "price",
    max_price: int | None = None,
) -> str:
    """Search flights between two cities.

    Returns up to 4 numbered options (direct flights, or connecting itineraries when
    there is no direct flight) and which option number is cheapest, fastest and
    best_value (best_value = fare + ₹200 per hour of journey). Prices are per person in INR. When there is no
    direct flight, returns connecting itineraries (up to 3 stops). Flights are daily
    schedules: use 'departs_at'/'arrives_at' clock times on the traveller's date.

    Args:
        origin: Departure city, e.g. "Delhi".
        destination: Arrival city, e.g. "Goa".
        sort_by: "price" (cheapest first), "duration" (fastest first) or "departure".
        max_price: Optional maximum fare per person in INR.
    """
    src, dst = normalise_city(origin), normalise_city(destination)
    if not src:
        return _unknown_city(origin)
    if not dst:
        return _unknown_city(destination)
    if src == dst:
        return _to_json({"error": "Origin and destination are the same city."})
    try:
        result = _find_flight_options(src, dst, sort_by=sort_by, max_price=max_price)
    except DatabaseError as exc:
        return _to_json({"error": str(exc)})

    candidates = result["direct_flights"] or result["connecting_options"]
    options = [{"option": i + 1, **_compact_option(o)} for i, o in enumerate(candidates[:4])]

    def option_number(pick: dict | None) -> int | None:
        if pick is None:
            return None
        for i, o in enumerate(candidates[:4]):
            if o is pick or o == pick:
                return i + 1
        options.append({"option": len(options) + 1, **_compact_option(pick)})
        return len(options)

    picks = {key: option_number(result[key]) for key in ("cheapest", "fastest", "best_value")}
    output = {
        "origin": src,
        "destination": dst,
        "route_type": "direct" if result["direct_flights"] else ("connecting" if candidates else "none"),
        "options": options,
        "cheapest_option": picks["cheapest"],
        "fastest_option": picks["fastest"],
        "best_value_option": picks["best_value"],
        "note": result["note"],
    }
    fastest = result["fastest"]
    if fastest:
        minutes = fastest.get("total_duration_minutes", fastest.get("duration_minutes", 0))
        if minutes > LONG_JOURNEY_MINUTES:
            output["warning"] = (
                f"Even the fastest option takes {minutes // 60}+ hours. Tell the traveller and "
                "recommend checking train/road alternatives."
            )
    return _to_json(output)


@tool
def search_hotels(
    city: str,
    min_stars: int = 0,
    max_price_per_night: int | None = None,
    required_amenities: list[str] | None = None,
    sort_by: Literal["value", "price", "stars"] = "value",
) -> str:
    """Recommend hotels in a city.

    Args:
        city: City name, e.g. "Jaipur".
        min_stars: Minimum star rating 1-5 (0 = any).
        max_price_per_night: Optional nightly budget in INR.
        required_amenities: Amenities that must all be present, e.g. ["wifi", "pool"].
        sort_by: "value" = most stars per ₹1,000 (default), "price" = cheapest, "stars" = highest rated.
    """
    name = normalise_city(city)
    if not name:
        return _unknown_city(city)
    try:
        hotels = _search_hotels(name, min_stars, max_price_per_night, required_amenities, sort_by)
    except DatabaseError as exc:
        return _to_json({"error": str(exc)})
    if not hotels:
        return _to_json(
            {"city": name, "hotels": [], "note": "No hotel matches all filters. Relax stars, price or amenities."}
        )
    return _to_json({"city": name, "sorted_by": sort_by, "hotels": hotels})


@tool
def search_places(
    city: str,
    preferred_types: list[str] | None = None,
    min_rating: float = 0.0,
    limit: int = 10,
) -> str:
    """Find attractions in a city, best-rated first; preferred types are ranked on top.

    Args:
        city: City name, e.g. "Kolkata".
        preferred_types: Optional types the traveller likes, e.g. ["temple", "museum", "beach"].
        min_rating: Minimum rating 0-5.
        limit: Maximum places returned.
    """
    name = normalise_city(city)
    if not name:
        return _unknown_city(city)
    try:
        places = _search_places(name, preferred_types, min_rating, limit)
    except DatabaseError as exc:
        return _to_json({"error": str(exc)})
    return _to_json({"city": name, "count": len(places), "places": places})


@tool
def get_weather(city: str, start_date: str, num_days: int) -> str:
    """Get day-by-day weather for the trip from the free Open-Meteo API.

    Uses the live forecast for dates within 16 days, otherwise last year's
    observed weather for the same dates as a typical expectation.

    Args:
        city: Destination city.
        start_date: First day of the trip, YYYY-MM-DD.
        num_days: Number of days of the trip.
    """
    name = normalise_city(city)
    if not name:
        return _unknown_city(city)
    try:
        return _to_json(_get_weather_forecast(name, start_date, int(num_days)))
    except ValueError as exc:
        return _to_json({"error": f"Invalid date '{start_date}': {exc}. Use YYYY-MM-DD."})


@tool
def estimate_budget(
    outbound_flight_price: int,
    return_flight_price: int,
    hotel_price_per_night: int,
    nights: int,
    days: int,
    travellers: int = 1,
    travel_style: Literal["budget", "standard", "luxury"] = "standard",
    user_budget: int | None = None,
) -> str:
    """Estimate total trip cost in INR (flights + hotel + food & local travel).

    Args:
        outbound_flight_price: Per-person fare of the chosen outbound flight (sum of legs).
        return_flight_price: Per-person fare of the chosen return flight (0 if none).
        hotel_price_per_night: Nightly rate of the chosen hotel (per room).
        nights: Number of hotel nights (usually days - 1).
        days: Number of trip days.
        travellers: Number of people (2 per room).
        travel_style: "budget", "standard" or "luxury" — sets daily food/local spend.
        user_budget: Optional total budget to check against.
    """
    try:
        return _to_json(
            _estimate_budget(
                outbound_flight_price, return_flight_price, hotel_price_per_night,
                nights, days, travellers, travel_style, user_budget,
            )
        )
    except (TypeError, ValueError) as exc:
        return _to_json({"error": f"Invalid budget inputs: {exc}"})


ALL_TOOLS = [
    list_supported_destinations,
    search_flights,
    search_hotels,
    search_places,
    get_weather,
    estimate_budget,
]

__all__ = ["ALL_TOOLS"]
