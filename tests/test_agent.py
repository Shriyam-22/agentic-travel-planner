"""Agent wiring test with a scripted fake LLM (no API key or network needed).

The fake model emits tool calls exactly like a real tool-calling LLM would; the
test checks that LangChain runs our tools, feeds results back, and returns a
validated TripPlan through the structured-output strategy.
"""

from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from travel_agent.agent import build_query, run_agent
from travel_agent.planner import plan_trip


class ScriptedToolModel(GenericFakeChatModel):
    """Fake chat model that accepts bind_tools() and replays scripted messages."""

    def bind_tools(self, tools, **kwargs):  # noqa: D401 - LangChain API
        return self


def _call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def test_agent_runs_tools_and_returns_structured_plan(monkeypatch):
    monkeypatch.setattr(
        "travel_agent.services.weather_service._fetch_daily",
        lambda *a, **k: {
            "time": ["x", "y", "z"], "weather_code": [0, 2, 61],
            "temperature_2m_max": [30, 31, 29], "temperature_2m_min": [20, 21, 22],
            "precipitation_sum": [0, 0, 6],
        },
    )
    final_plan = plan_trip("Hyderabad", "Delhi", "2026-12-01", 3).model_dump()

    script = iter([
        AIMessage(content="", tool_calls=[
            _call("search_flights", {"origin": "Hyderabad", "destination": "Delhi"}, "c1"),
            _call("search_flights", {"origin": "Delhi", "destination": "Hyderabad"}, "c2"),
        ]),
        AIMessage(content="", tool_calls=[
            _call("search_hotels", {"city": "Delhi", "min_stars": 4}, "c3"),
            _call("search_places", {"city": "Delhi", "preferred_types": ["museum"]}, "c4"),
            _call("get_weather", {"city": "Delhi", "start_date": "2026-12-01", "num_days": 3}, "c5"),
        ]),
        AIMessage(content="", tool_calls=[
            _call("estimate_budget", {"outbound_flight_price": 2907, "return_flight_price": 5000,
                                      "hotel_price_per_night": 3650, "nights": 2, "days": 3}, "c6"),
        ]),
        AIMessage(content="", tool_calls=[_call("TripPlan", final_plan, "c7")]),
    ])
    result = run_agent("3 days in Delhi from Hyderabad", llm=ScriptedToolModel(messages=script))

    assert result.plan.destination == "Delhi"
    tools_used = [s.tool for s in result.steps]
    assert tools_used == ["search_flights", "search_flights", "search_hotels",
                          "search_places", "get_weather", "estimate_budget"]
    flights = json.loads(result.steps[0].output)
    assert flights["cheapest"]["flight_id"] == "FL0001"
    hotels = json.loads(result.steps[2].output)
    assert all(h["stars"] >= 4 for h in hotels["hotels"])


def test_build_query_includes_preferences():
    q = build_query("Delhi", "Goa", "2026-12-01", 4, travellers=2, budget=50000,
                    interests=["beach"], amenities=["pool"], min_hotel_stars=4)
    assert "4-day trip from Delhi to Goa" in q
    assert "₹50,000" in q and "beach" in q and "pool" in q and "4 stars" in q
