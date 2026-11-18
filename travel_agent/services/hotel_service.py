"""Hotel recommendations: filter by city, stars, price, amenities; rank by value."""

from __future__ import annotations

from typing import Literal

from travel_agent.database import query

HotelSort = Literal["value", "price", "stars"]

_HOTEL_SQL = """
SELECT h.hotel_id, h.hotel_name AS name, c.city_name AS city, h.stars, h.price_per_night,
       COALESCE(GROUP_CONCAT(am.amenity_name, ','), '') AS amenity_list
FROM hotels h
JOIN cities c ON c.city_id = h.city_id
LEFT JOIN hotel_amenities ha ON ha.hotel_id = h.hotel_id
LEFT JOIN amenities am ON am.amenity_id = ha.amenity_id
WHERE c.city_name = ?
GROUP BY h.hotel_id
"""


def value_score(stars: int, price_per_night: int) -> float:
    """Stars delivered per ₹1,000 a night — higher is better value."""
    return round(stars / (price_per_night / 1000), 2) if price_per_night else 0.0


def search_hotels(
    city: str,
    min_stars: int = 0,
    max_price_per_night: int | None = None,
    required_amenities: list[str] | None = None,
    sort_by: HotelSort = "value",
    limit: int = 5,
) -> list[dict]:
    """Hotels in a city matching the filters.

    Args:
        city: Dataset city name.
        min_stars: Minimum star rating (0–5).
        max_price_per_night: Upper bound on nightly rate in INR.
        required_amenities: Every listed amenity must be present (e.g. ["wifi", "pool"]).
        sort_by: "value" (stars per ₹1k), "price" (cheapest) or "stars" (best rated).
        limit: Maximum hotels returned.
    """
    wanted = {a.strip().lower() for a in (required_amenities or []) if a.strip()}
    hotels = []
    for row in query(_HOTEL_SQL, (city,)):
        amenities = sorted(a for a in row.pop("amenity_list").split(",") if a)
        if row["stars"] < min_stars:
            continue
        if max_price_per_night is not None and row["price_per_night"] > max_price_per_night:
            continue
        if wanted and not wanted.issubset(amenities):
            continue
        hotels.append(
            {**row, "amenities": amenities, "value_score": value_score(row["stars"], row["price_per_night"])}
        )

    sorters = {
        "value": lambda h: (-h["value_score"], h["price_per_night"]),
        "price": lambda h: (h["price_per_night"], -h["stars"]),
        "stars": lambda h: (-h["stars"], h["price_per_night"]),
    }
    hotels.sort(key=sorters.get(sort_by, sorters["value"]))
    return hotels[:limit]


def list_amenities() -> list[str]:
    """All amenity names in the dataset."""
    return [r["amenity_name"] for r in query("SELECT amenity_name FROM amenities ORDER BY amenity_name")]
