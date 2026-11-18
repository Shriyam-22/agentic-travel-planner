"""Unit tests for the data layer, services, tools and offline planner."""

from __future__ import annotations

import json
from datetime import date

import pytest
import requests

from travel_agent import tools
from travel_agent.database import list_cities, normalise_city
from travel_agent.formatter import to_markdown
from travel_agent.planner import plan_trip
from travel_agent.services import estimate_budget, find_flight_options, search_hotels, search_places, weather_service
from travel_agent.services.weather_service import get_weather_forecast

FAKE_DAILY = {
    "time": ["a", "b", "c"],
    "weather_code": [0, 3, 63],
    "temperature_2m_max": [31.2, 29.0, 27.5],
    "temperature_2m_min": [22.0, 21.4, 20.9],
    "precipitation_sum": [0.0, 1.0, 12.0],
}


@pytest.fixture(autouse=True)
def offline_weather(monkeypatch):
    """Never hit the network in tests."""
    weather_service._fetch_daily.cache_clear()
    monkeypatch.setattr(weather_service, "_fetch_daily", lambda *a, **k: FAKE_DAILY)


# ---- database --------------------------------------------------------------
def test_all_cities_loaded():
    assert set(list_cities()) == {"Bangalore", "Chennai", "Delhi", "Goa", "Hyderabad", "Jaipur", "Kolkata", "Mumbai"}


@pytest.mark.parametrize("raw,expected", [("bengaluru", "Bangalore"), (" goa ", "Goa"), ("Paris", None)])
def test_normalise_city(raw, expected):
    assert normalise_city(raw) == expected


# ---- flights ---------------------------------------------------------------
def test_direct_flights_cheapest_and_fastest():
    result = find_flight_options("Bangalore", "Delhi")
    assert [f["flight_id"] for f in result["direct_flights"]] == ["FL0012", "FL0009"]  # sorted by price
    assert result["cheapest"]["flight_id"] == "FL0012"
    assert result["fastest"]["flight_id"] == "FL0009"  # 3h vs 4h


def test_connecting_flights_when_no_direct_route():
    result = find_flight_options("Mumbai", "Jaipur")
    assert result["direct_flights"] == []
    assert result["connecting_options"]
    # best value avoids the 31-hour cheapest routing for ₹42 more
    assert result["best_value"]["via"] == ["Goa"]
    assert result["cheapest"]["total_price"] <= result["best_value"]["total_price"]
    legs = result["best_value"]["legs"]
    assert legs[0]["destination"] == legs[1]["origin"]


def test_max_price_filter():
    result = find_flight_options("Bangalore", "Delhi", max_price=7000)
    assert all(f["price"] <= 7000 for f in result["direct_flights"])


# ---- hotels & places -------------------------------------------------------
def test_hotel_filters():
    hotels = search_hotels("Mumbai", min_stars=5, required_amenities=["wifi", "spa"])
    assert hotels and all(h["stars"] == 5 and {"wifi", "spa"} <= set(h["amenities"]) for h in hotels)


def test_hotel_sort_by_price():
    prices = [h["price_per_night"] for h in search_hotels("Goa", sort_by="price")]
    assert prices == sorted(prices)


def test_places_preference_first():
    places = search_places("Kolkata", preferred_types=["museum"])
    assert places[0]["type"] == "museum"
    assert len(places) == 5


# ---- budget ----------------------------------------------------------------
def test_budget_maths():
    b = estimate_budget(4800, 4800, 3200, nights=2, days=3, travellers=3, travel_style="budget", user_budget=50000)
    assert b["rooms"] == 2
    assert b["flights"] == 9600 * 3
    assert b["hotel"] == 3200 * 2 * 2
    assert b["food_and_local_travel"] == 1200 * 3 * 3
    assert b["total"] == b["flights"] + b["hotel"] + b["food_and_local_travel"]
    assert b["within_budget"] is False


# ---- weather ---------------------------------------------------------------
def test_weather_forecast_window():
    w = get_weather_forecast("Goa", "2026-11-20", 3, today=date(2026, 11, 18))
    assert w["source"] == "forecast"
    assert w["days"][2]["condition"] == "Rain"
    assert "umbrella" in w["days"][2]["tip"]


def test_weather_far_future_uses_history():
    w = get_weather_forecast("Goa", "2027-03-01", 3, today=date(2026, 11, 18))
    assert w["source"] == "historical"


def test_weather_failure_is_graceful(monkeypatch):
    def boom(*a, **k):
        raise weather_service.WeatherServiceError("could not connect to Open-Meteo")
    monkeypatch.setattr(weather_service, "_fetch_daily", boom)
    w = get_weather_forecast("Goa", "2026-11-20", 2, today=date(2026, 11, 18))
    assert w["source"] == "unavailable" and len(w["days"]) == 2


def test_real_fetch_wraps_http_errors(monkeypatch):
    monkeypatch.undo()  # restore the real _fetch_daily
    weather_service._fetch_daily.cache_clear()

    def fail(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(weather_service.requests, "get", fail)
    with pytest.raises(weather_service.WeatherServiceError):
        weather_service._fetch_daily("u", 1.0, 2.0, "2026-01-01", "2026-01-02")


# ---- tools -----------------------------------------------------------------
def test_tool_unknown_city_message():
    out = json.loads(tools.search_hotels.invoke({"city": "Paris"}))
    assert "error" in out and "Goa" in out["supported_cities"]


def test_tool_flights_compact_output():
    out = json.loads(tools.search_flights.invoke({"origin": "Delhi", "destination": "Kolkata"}))
    assert out["route_type"] == "direct"
    assert "departure_time" not in out["options"][0]
    cheapest = out["options"][out["cheapest_option"] - 1]
    assert cheapest["departs_at"] == "05:34" and cheapest["price"] == 3779


def test_tool_flights_connecting_best_value():
    raw = tools.search_flights.invoke({"origin": "Mumbai", "destination": "Jaipur"})
    out = json.loads(raw)
    assert out["route_type"] == "connecting" and len(out["options"]) <= 5
    best = out["options"][out["best_value_option"] - 1]
    assert best["via"] == ["Goa"] and best["duration"] == "8h 00m"
    assert len(raw) < 4000, "flight tool output must stay small for free-tier token limits"


# ---- planner ---------------------------------------------------------------
def test_offline_planner_complete_plan():
    plan = plan_trip("Hyderabad", "Delhi", "2026-12-01", 4, travellers=2, interests=["museum"], budget=80000)
    assert plan.num_days == 4 and len(plan.daily_plan) == 4
    assert plan.outbound_flight.flight_ids == ["FL0001"]
    assert plan.hotel.city == "Delhi"
    visited = [a.place_id for d in plan.daily_plan for a in d.activities if a.place_id]
    assert len(visited) == len(set(visited)), "places must not repeat"
    assert plan.budget.total == plan.budget.flights + plan.budget.hotel + plan.budget.food_and_local_travel
    md = to_markdown(plan)
    assert "Total Cost" in md and "Day 4" in md


def test_offline_planner_transit_days():
    plan = plan_trip("Delhi", "Goa", "2026-12-01", 5)
    assert plan.daily_plan[0].title == "In transit"
    assert plan.outbound_flight.arrival_date == "2026-12-02"
    assert plan.budget.hotel == plan.hotel.price_per_night * 3


def test_planner_rejects_bad_city():
    with pytest.raises(ValueError):
        plan_trip("Delhi", "Paris", "2026-12-01", 3)
