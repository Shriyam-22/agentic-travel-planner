"""Flight search: direct flights, connecting itineraries, cheapest / fastest picks.

The dataset holds one sample departure per flight (dates in 2025). We treat each
flight as a *recurring daily schedule*: its clock times, duration and fare are
reused on whatever date the traveller picks. Connecting itineraries are built by
chaining flights where the next leg departs at least ``MIN_LAYOVER_MINUTES``
after the previous one lands (rolling to the next day if needed).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Literal

from travel_agent import config
from travel_agent.database import query

SortKey = Literal["price", "duration", "departure"]
Preference = Literal["cheapest", "fastest", "best_value"]

_FLIGHT_SQL = """
SELECT f.flight_id, a.airline_name AS airline,
       o.city_name AS origin, d.city_name AS destination,
       f.departure_time, f.arrival_time, f.duration_minutes, f.price
FROM flights f
JOIN airlines a ON a.airline_id = f.airline_id
JOIN cities o   ON o.city_id = f.origin_city_id
JOIN cities d   ON d.city_id = f.destination_city_id
"""


def _format_duration(minutes: int) -> str:
    hours, mins = divmod(int(minutes), 60)
    return f"{hours}h {mins:02d}m"


def _enrich(flight: dict) -> dict:
    """Add clock-time fields that make sense for a recurring schedule."""
    dep = datetime.fromisoformat(flight["departure_time"])
    arr = datetime.fromisoformat(flight["arrival_time"])
    return {
        **flight,
        "departs_at": dep.strftime("%H:%M"),
        "arrives_at": arr.strftime("%H:%M"),
        "arrives_next_day": arr.date() > dep.date(),
        "duration": _format_duration(flight["duration_minutes"]),
    }


def get_all_flights() -> list[dict]:
    """Every flight in the dataset, enriched."""
    return [_enrich(row) for row in query(_FLIGHT_SQL + " ORDER BY f.flight_id")]


def search_direct_flights(
    origin: str,
    destination: str,
    sort_by: SortKey = "price",
    max_price: int | None = None,
) -> list[dict]:
    """Direct flights on a route, filtered by fare and sorted."""
    sql = _FLIGHT_SQL + " WHERE o.city_name = ? AND d.city_name = ?"
    params: list = [origin, destination]
    if max_price is not None:
        sql += " AND f.price <= ?"
        params.append(max_price)
    order = {"price": "f.price", "duration": "f.duration_minutes", "departure": "time(f.departure_time)"}
    sql += f" ORDER BY {order.get(sort_by, 'f.price')}"
    return [_enrich(row) for row in query(sql, tuple(params))]


def _schedule_on(flight: dict, earliest: datetime) -> tuple[datetime, datetime]:
    """First departure of a daily-scheduled flight at or after ``earliest``."""
    dep_clock = datetime.fromisoformat(flight["departure_time"]).time()
    departure = datetime.combine(earliest.date(), dep_clock)
    if departure < earliest:
        departure += timedelta(days=1)
    return departure, departure + timedelta(minutes=flight["duration_minutes"])


def search_connecting_flights(
    origin: str,
    destination: str,
    max_stops: int = config.DEFAULT_MAX_STOPS,
    limit: int = 5,
) -> list[dict]:
    """Multi-leg itineraries (1..max_stops stops), cheapest first.

    Uses a depth-first search over the route graph that never revisits a city.
    """
    flights = get_all_flights()
    by_origin: dict[str, list[dict]] = {}
    for flight in flights:
        by_origin.setdefault(flight["origin"], []).append(flight)

    itineraries: list[list[dict]] = []

    def walk(city: str, path: list[dict], visited: set[str]) -> None:
        if len(path) > max_stops + 1:
            return
        if city == destination and len(path) > 1:
            itineraries.append(list(path))
            return
        for flight in by_origin.get(city, []):
            nxt = flight["destination"]
            if nxt in visited:
                continue
            path.append(flight)
            visited.add(nxt)
            walk(nxt, path, visited)
            path.pop()
            visited.discard(nxt)

    walk(origin, [], {origin})

    results = []
    for legs in itineraries:
        anchor = datetime.combine(date(2000, 1, 1), time(0, 0))
        dep0, arr = _schedule_on(legs[0], anchor)
        timed_legs = [{**legs[0], "leg_departure_offset_min": 0}]
        for leg in legs[1:]:
            dep, new_arr = _schedule_on(leg, arr + timedelta(minutes=config.MIN_LAYOVER_MINUTES))
            timed_legs.append({**leg, "layover_minutes": int((dep - arr).total_seconds() // 60)})
            arr = new_arr
        total_minutes = int((arr - dep0).total_seconds() // 60)
        results.append(
            {
                "type": "connecting",
                "stops": len(legs) - 1,
                "via": [leg["destination"] for leg in legs[:-1]],
                "flight_ids": [leg["flight_id"] for leg in legs],
                "airlines": sorted({leg["airline"] for leg in legs}),
                "total_price": sum(leg["price"] for leg in legs),
                "total_duration_minutes": total_minutes,
                "total_duration": _format_duration(total_minutes),
                "departs_at": legs[0]["departs_at"],
                "arrives_at": arr.strftime("%H:%M"),
                "days_in_transit": (arr.date() - dep0.date()).days,
                "legs": timed_legs,
            }
        )
    results.sort(key=lambda r: (r["total_price"], r["total_duration_minutes"]))
    return results[:limit]


def option_price(option: dict) -> int:
    return option.get("total_price", option.get("price", 0))


def option_minutes(option: dict) -> int:
    return option.get("total_duration_minutes", option.get("duration_minutes", 0))


def value_score(option: dict) -> float:
    """Fare plus a notional cost per hour of travel — lower is better."""
    return option_price(option) + config.TIME_VALUE_PER_HOUR * option_minutes(option) / 60


def find_flight_options(
    origin: str,
    destination: str,
    sort_by: SortKey = "price",
    max_price: int | None = None,
    max_stops: int = config.DEFAULT_MAX_STOPS,
    listing_limit: int = 5,
) -> dict:
    """One-stop answer for the agent: direct options, cheapest, fastest, best value, or connections.

    Returns a dictionary with ``direct_flights``, ``connecting_options`` (only when
    no direct flight fits), ``cheapest``, ``fastest``, ``best_value`` and a ``note``.
    ``best_value`` minimises fare + ``TIME_VALUE_PER_HOUR`` × journey hours.
    """
    direct = search_direct_flights(origin, destination, sort_by=sort_by, max_price=max_price)
    result: dict = {
        "origin": origin,
        "destination": destination,
        "direct_flights": direct,
        "connecting_options": [],
        "note": "Flights are treated as daily schedules: use the clock times on the traveller's date.",
    }
    candidates = direct
    if not direct:
        connecting = search_connecting_flights(origin, destination, max_stops=max_stops, limit=50)
        if max_price is not None:
            connecting = [c for c in connecting if c["total_price"] <= max_price]
        candidates = connecting
        if connecting:
            result["note"] += " No direct flight on this route; connecting itineraries listed."
        else:
            result["note"] += (
                f" No direct or connecting flight (up to {max_stops} stops) found in the dataset."
                " Suggest train/road travel or a different origin."
            )

    picks = {
        "cheapest": min(candidates, key=lambda o: (option_price(o), option_minutes(o)), default=None),
        "fastest": min(candidates, key=lambda o: (option_minutes(o), option_price(o)), default=None),
        "best_value": min(candidates, key=value_score, default=None),
    }
    result.update(picks)

    if not direct:
        shown = candidates[:listing_limit]
        for pick in picks.values():
            if pick is not None and pick not in shown:
                shown.append(pick)
        result["connecting_options"] = shown
    return result
