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


def _slim_flight(flight: dict) -> dict:
    """Drop the raw 2025 sample timestamps so the LLM only sees schedule fields."""
    return {k: v for k, v in flight.items() if k not in {"departure_time", "arrival_time"}}


def _slim_option(option: dict | None) -> dict | None:
    if option is None:
        return None
    if option.get("type") == "connecting":
        return {**option, "legs": [_slim_flight(leg) for leg in option["legs"]]}
    return {"type": "direct", **_slim_flight(option)}


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

    Returns direct flights plus the cheapest, fastest and best_value options
    (best_value = fare + ₹200 per hour of journey, which avoids very long cheap connections). When there is no
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

    output = {
        "origin": src,
        "destination": dst,
        "direct_flights": [_slim_option(f) for f in result["direct_flights"]],
        "connecting_options": [_slim_option(c) for c in result["connecting_options"]],
        "cheapest": _slim_option(result["cheapest"]),
        "fastest": _slim_option(result["fastest"]),
        "best_value": _slim_option(result["best_value"]),
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
