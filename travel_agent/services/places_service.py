"""Places / points-of-interest discovery by city, type and rating."""

from __future__ import annotations

from travel_agent.database import query

_PLACES_SQL = """
SELECT p.place_id, p.place_name AS name, c.city_name AS city, t.type_name AS type, p.rating
FROM places p
JOIN cities c      ON c.city_id = p.city_id
JOIN place_types t ON t.place_type_id = p.place_type_id
WHERE c.city_name = ? AND p.rating >= ?
ORDER BY p.rating DESC, p.place_name
"""


def search_places(
    city: str,
    preferred_types: list[str] | None = None,
    min_rating: float = 0.0,
    limit: int = 10,
) -> list[dict]:
    """Attractions in a city, best-rated first.

    Places whose type matches ``preferred_types`` are ranked ahead of the rest
    (the rest are still returned so short trips are never left empty).
    """
    preferred = {t.strip().lower() for t in (preferred_types or []) if t.strip()}
    places = query(_PLACES_SQL, (city, min_rating))
    for place in places:
        place["matches_preference"] = bool(preferred) and place["type"].lower() in preferred
    places.sort(key=lambda p: (not p["matches_preference"], -p["rating"]))
    return places[:limit]


def list_place_types() -> list[str]:
    """All attraction types in the dataset."""
    return [r["type_name"] for r in query("SELECT type_name FROM place_types ORDER BY type_name")]
