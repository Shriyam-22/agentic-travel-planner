"""Deterministic (no-LLM) planner.

Runs the same services the agent's tools use, in a fixed order, and applies
simple ranking rules. It serves three purposes:

* an **offline demo mode** when no LLM API key is available,
* a **baseline** to compare the agent's choices against,
* a fast, reproducible path for **unit tests**.
"""

from __future__ import annotations

from datetime import date, timedelta

from travel_agent.database import normalise_city
from travel_agent.schemas import Activity, BudgetBreakdown, DayPlan, FlightChoice, HotelChoice, TripPlan
from travel_agent.services import (
    estimate_budget,
    find_flight_options,
    get_weather_forecast,
    search_hotels,
    search_places,
)

LONG_JOURNEY_MINUTES = 12 * 60
FREE_TIME_IDEAS = [
    "Local food trail & street-food tasting",
    "Shopping at a local market",
    "Leisure time at the hotel",
    "Explore the old city on foot",
]


def _flight_choice(option: dict | None, travel_date: date, preference: str) -> FlightChoice | None:
    if option is None:
        return None
    if option.get("type") == "connecting":
        arrival = travel_date + timedelta(days=option["days_in_transit"])
        return FlightChoice(
            arrival_date=arrival.isoformat(),
            flight_ids=option["flight_ids"], airlines=option["airlines"],
            origin=option["legs"][0]["origin"], destination=option["legs"][-1]["destination"],
            travel_date=travel_date.isoformat(), departs_at=option["departs_at"],
            arrives_at=option["arrives_at"], duration=option["total_duration"],
            stops=option["stops"], via=option["via"], price_per_person=option["total_price"],
            reason=(f"No direct flight on this route; {preference} connecting itinerary "
                    f"via {', '.join(option['via'])} at ₹{option['total_price']:,}."),
        )
    arrival = travel_date + timedelta(days=int(option["arrives_next_day"]))
    return FlightChoice(
        arrival_date=arrival.isoformat(),
        flight_ids=[option["flight_id"]], airlines=[option["airline"]],
        origin=option["origin"], destination=option["destination"],
        travel_date=travel_date.isoformat(), departs_at=option["departs_at"],
        arrives_at=option["arrives_at"], duration=option["duration"], stops=0,
        price_per_person=option["price"],
        reason=f"{preference.capitalize()} direct flight on the route at ₹{option['price']:,}.",
    )


PREFERENCE_LABELS = {"cheapest": "cheapest", "fastest": "fastest", "best_value": "best-value (fare + time)"}


def _pick_flight(options: dict, preference: str) -> dict | None:
    return options.get(preference) or options["best_value"]


def _price(option: dict) -> int:
    return option.get("total_price", option.get("price", 0))


def _minutes(option: dict | None) -> int:
    if not option:
        return 0
    return option.get("total_duration_minutes", option.get("duration_minutes", 0))


def _weather_label(day: dict) -> str:
    if day.get("temp_max_c") is None:
        return day.get("condition", "Unavailable")
    return f"{day['condition']}, {round(day['temp_max_c'])}°C / {round(day['temp_min_c'])}°C"


def plan_trip(
    origin: str,
    destination: str,
    start_date: str,
    num_days: int = 3,
    travellers: int = 1,
    travel_style: str = "standard",
    budget: int | None = None,
    interests: list[str] | None = None,
    min_hotel_stars: int = 0,
    amenities: list[str] | None = None,
    flight_preference: str = "best_value",
    today: date | None = None,
) -> TripPlan:
    """Build a complete :class:`TripPlan` with rule-based choices."""
    src, dst = normalise_city(origin), normalise_city(destination)
    if not src or not dst:
        raise ValueError(f"Unknown city: {origin if not src else destination}")
    if src == dst:
        raise ValueError("Origin and destination must be different.")
    num_days = max(1, int(num_days))
    start = date.fromisoformat(start_date)
    end = start + timedelta(days=num_days - 1)
    reasoning: list[str] = []
    warnings: list[str] = ["Data comes from a sample dataset — verify availability and prices before booking."]

    # Flights ------------------------------------------------------------- #
    sort = "duration" if flight_preference == "fastest" else "price"
    out_opts = find_flight_options(src, dst, sort_by=sort)
    ret_opts = find_flight_options(dst, src, sort_by=sort)
    out_pick, ret_pick = _pick_flight(out_opts, flight_preference), _pick_flight(ret_opts, flight_preference)
    label = PREFERENCE_LABELS.get(flight_preference, flight_preference)
    outbound = _flight_choice(out_pick, start, label)
    inbound = _flight_choice(ret_pick, end, label)
    for leg_name, opts, pick in (("Outbound", out_opts, out_pick), ("Return", ret_opts, ret_pick)):
        if pick is None:
            warnings.append(f"{leg_name}: no flight found in the dataset — consider train/road travel.")
            continue
        n_direct = len(opts["direct_flights"])
        cheapest = opts["cheapest"]
        line = (f"{leg_name}: picked the {PREFERENCE_LABELS.get(flight_preference, flight_preference)} of "
                f"{n_direct or len(opts['connecting_options'])} {'direct' if n_direct else 'connecting'} option(s)")
        if cheapest is not pick:
            line += (f"; the cheapest costs ₹{_price(pick) - _price(cheapest):,} less but takes "
                     f"{(_minutes(cheapest) - _minutes(pick)) // 60}h longer")
        reasoning.append(line + ".")
        if _minutes(pick) > LONG_JOURNEY_MINUTES:
            warnings.append(
                f"{leg_name} journey takes about {_minutes(pick) // 60} hours with connections — "
                "a train or a different routing may be more practical."
            )

    # Hotel --------------------------------------------------------------- #
    hotel_sort = {"luxury": "stars", "budget": "price"}.get(travel_style, "value")
    hotels = search_hotels(dst, min_hotel_stars, None, amenities, hotel_sort)
    if not hotels and (min_hotel_stars or amenities):
        warnings.append("No hotel matched every filter, so star/amenity filters were relaxed.")
        hotels = search_hotels(dst, 0, None, None, hotel_sort)
    hotel = None
    if hotels:
        best = hotels[0]
        alt = f" Next best: {hotels[1]['name']} (₹{hotels[1]['price_per_night']:,})." if len(hotels) > 1 else ""
        criterion = {"value": "best value (stars per ₹1,000)", "stars": "highest star rating",
                     "price": "lowest nightly rate"}[hotel_sort]
        hotel = HotelChoice(
            hotel_id=best["hotel_id"], name=best["name"], city=best["city"], stars=best["stars"],
            price_per_night=best["price_per_night"], amenities=best["amenities"],
            reason=f"{criterion.capitalize()} among {len(hotels)} matching hotels.{alt}",
        )
        reasoning.append(f"Hotel ranked by {criterion} for a {travel_style} trip.")

    # Weather + places ---------------------------------------------------- #
    weather = get_weather_forecast(dst, start.isoformat(), num_days, today=today)
    weather_by_date = {d["date"]: d for d in weather["days"]}
    if outbound and outbound.arrival_date and outbound.arrival_date > start.isoformat():
        warnings.append(f"The outbound journey lands on {outbound.arrival_date}, so sightseeing starts that day.")
    places = search_places(dst, interests, 0.0, limit=20)
    if interests:
        reasoning.append(f"Attractions matching {', '.join(interests)} were scheduled first, then by rating.")

    daily_plan: list[DayPlan] = []
    queue = list(places)
    free_ideas = iter(FREE_TIME_IDEAS * 3)
    arrival_day = date.fromisoformat(outbound.arrival_date) if outbound and outbound.arrival_date else start
    for i in range(num_days):
        day_date = start + timedelta(days=i)
        is_first, is_last = day_date == arrival_day, i == num_days - 1 and num_days > 1
        if day_date < arrival_day:
            daily_plan.append(DayPlan(
                day=i + 1, date=day_date.isoformat(), title="In transit",
                weather=_weather_label(weather_by_date.get(day_date.isoformat(), {"condition": "Unavailable"})),
                activities=[Activity(time_slot="All day", name=f"Travelling to {dst}",
                                     note=(f"Flights {', '.join(outbound.flight_ids)}"
                                           if i == 0 and outbound else None))],
            ))
            continue
        if is_first:
            lands = outbound.arrives_at if outbound else "12:00"
            if lands >= "20:00":
                slots = []
            elif lands >= "15:00":
                slots = ["Evening"]
            else:  # early-morning or midday arrival: rest, then afternoon + evening
                slots = ["Afternoon", "Evening"]
        elif is_last:
            departs = inbound.departs_at if inbound else "12:00"
            slots = [] if departs < "12:00" else (["Morning", "Afternoon"] if departs >= "18:00" else ["Morning"])
        else:
            slots = ["Morning", "Afternoon", "Evening"]
        if is_first and is_last:
            slots = slots[:1]
        activities = []
        if is_first and outbound:
            activities.append(Activity(time_slot=outbound.arrives_at, name=f"Arrive in {dst}, check in",
                                       note=f"Flight {', '.join(outbound.flight_ids)}"))
        for slot in slots:
            if queue:
                p = queue.pop(0)
                activities.append(Activity(time_slot=slot, name=p["name"], place_id=p["place_id"],
                                           place_type=p["type"], rating=p["rating"]))
            else:
                activities.append(Activity(time_slot=slot, name=next(free_ideas)))
        if is_last and inbound:
            activities.append(Activity(time_slot=inbound.departs_at, name=f"Fly back to {src}",
                                       note=f"Flight {', '.join(inbound.flight_ids)}"))
        title = "Arrival & first look" if is_first else ("Last morning & departure" if is_last else "Explore")
        w = weather_by_date.get(day_date.isoformat(), {"condition": "Unavailable"})
        daily_plan.append(DayPlan(day=i + 1, date=day_date.isoformat(), title=title,
                                  weather=_weather_label(w), activities=activities))

    # Budget -------------------------------------------------------------- #
    b = estimate_budget(
        outbound.price_per_person if outbound else 0, inbound.price_per_person if inbound else 0,
        hotel.price_per_night if hotel else 0, max((end - arrival_day).days, 0), num_days,
        travellers, travel_style, budget,
    )
    if budget and not b["within_budget"]:
        warnings.append(f"Estimated cost exceeds your budget by ₹{-b['difference']:,}.")
    if weather["source"] != "forecast":
        warnings.append(weather["note"])

    return TripPlan(
        trip_title=f"Your {num_days}-Day Trip to {dst} ({start:%b %d}–{end:%b %d})",
        trip_summary=(
            f"{num_days} days in {dst} from {src} for {travellers} traveller(s), {travel_style} style. "
            f"Staying at {hotel.name if hotel else 'TBD'}, visiting "
            f"{sum(1 for d in daily_plan for a in d.activities if a.place_id)} attractions."
        ),
        origin=src, destination=dst, start_date=start.isoformat(), end_date=end.isoformat(),
        num_days=num_days, travellers=travellers, outbound_flight=outbound, return_flight=inbound,
        hotel=hotel, weather_source=f"{weather['source']}: {weather['note']}", daily_plan=daily_plan,
        budget=BudgetBreakdown(
            flights=b["flights"], hotel=b["hotel"], food_and_local_travel=b["food_and_local_travel"],
            total=b["total"], per_person=b["per_person"], within_budget=b.get("within_budget"),
            notes=(f"{b['rooms']} room(s) × {b['nights']} night(s); ₹{b['daily_local_rate_per_person']:,}"
                   f"/person/day for food & local travel ({b['travel_style']})."),
        ),
        reasoning=reasoning, warnings=warnings,
    )
