"""Budget estimation: flights + hotel + per-day local expenses."""

from __future__ import annotations

import math

from travel_agent import config


def rooms_needed(travellers: int) -> int:
    """Rooms required assuming ``TRAVELLERS_PER_ROOM`` guests per room."""
    return max(1, math.ceil(travellers / config.TRAVELLERS_PER_ROOM))


def estimate_budget(
    outbound_flight_price: int,
    return_flight_price: int,
    hotel_price_per_night: int,
    nights: int,
    days: int,
    travellers: int = 1,
    travel_style: str = "standard",
    user_budget: int | None = None,
) -> dict:
    """Total trip cost in INR.

    Flight prices are per person; hotel price is per room per night; local
    expenses (food + local transport + entry tickets) are per person per day
    and depend on ``travel_style`` (budget / standard / luxury).
    """
    travellers = max(1, int(travellers))
    style = travel_style if travel_style in config.DAILY_LOCAL_EXPENSES else "standard"
    rooms = rooms_needed(travellers)

    flights_total = (int(outbound_flight_price) + int(return_flight_price)) * travellers
    hotel_total = int(hotel_price_per_night) * int(nights) * rooms
    daily_rate = config.DAILY_LOCAL_EXPENSES[style]
    local_total = daily_rate * int(days) * travellers
    total = flights_total + hotel_total + local_total

    result = {
        "currency": config.CURRENCY,
        "travellers": travellers,
        "rooms": rooms,
        "nights": int(nights),
        "days": int(days),
        "travel_style": style,
        "flights": flights_total,
        "hotel": hotel_total,
        "food_and_local_travel": local_total,
        "daily_local_rate_per_person": daily_rate,
        "total": total,
        "per_person": round(total / travellers),
    }
    if user_budget:
        result["user_budget"] = int(user_budget)
        result["within_budget"] = total <= int(user_budget)
        result["difference"] = int(user_budget) - total
    return result
