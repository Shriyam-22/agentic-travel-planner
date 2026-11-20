"""Agent wiring test with a scripted fake LLM (no API key or network needed).

The fake model emits tool calls exactly like a real tool-calling LLM would; the
test checks that LangChain runs our tools, feeds results back, and returns a
validated TripPlan through the structured-output strategy.
"""

from __future__ import annotations

import json

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from travel_agent.agent import AgentError, build_query, run_agent
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
    assert flights["options"][flights["cheapest_option"] - 1]["flight_ids"] == ["FL0001"]
    hotels = json.loads(result.steps[2].output)
    assert all(h["stars"] >= 4 for h in hotels["hotels"])


def test_build_query_includes_preferences():
    q = build_query("Delhi", "Goa", "2026-12-01", 4, travellers=2, budget=50000,
                    interests=["beach"], amenities=["pool"], min_hotel_stars=4)
    assert "4-day trip from Delhi to Goa" in q
    assert "₹50,000" in q and "beach" in q and "pool" in q and "4 stars" in q


class FailingFinalModel(ScriptedToolModel):
    """Gathers data with tools, then fails like Groq's 'tool_use_failed' on the final answer."""

    final_plan: dict = {}
    fail_always: bool = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.fail_always or sum(1 for m in messages if m.type == "tool") >= 2:
            raise RuntimeError("Error code: 400 - {'error': {'code': 'tool_use_failed', "
                               "'message': 'Failed to parse tool call arguments as JSON'}}")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _prompt: schema.model_validate(self.final_plan))


def test_agent_falls_back_to_finalize_when_final_tool_call_breaks(monkeypatch):
    monkeypatch.setattr("travel_agent.agent.ModelRetryMiddleware.__init__",
                        _no_wait_retry_init, raising=True)
    final_plan = plan_trip("Delhi", "Kolkata", "2026-12-01", 3).model_dump()
    script = iter([
        AIMessage(content="", tool_calls=[
            _call("search_flights", {"origin": "Delhi", "destination": "Kolkata"}, "a1"),
            _call("search_hotels", {"city": "Kolkata"}, "a2"),
        ]),
    ])
    model = FailingFinalModel(messages=script, final_plan=final_plan)
    result = run_agent("3 days in Kolkata from Delhi", llm=model)
    assert result.plan.destination == "Kolkata"
    assert [s.tool for s in result.steps] == ["search_flights", "search_hotels"]


def test_agent_error_is_friendly_when_nothing_gathered(monkeypatch):
    monkeypatch.setattr("travel_agent.agent.ModelRetryMiddleware.__init__",
                        _no_wait_retry_init, raising=True)
    model = FailingFinalModel(messages=iter([]), fail_always=True)
    try:
        run_agent("3 days in Goa", llm=model)
    except AgentError as exc:
        assert "malformed answer" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected AgentError")


_ORIGINAL_RETRY_INIT = __import__("langchain.agents.middleware", fromlist=["x"]).ModelRetryMiddleware.__init__


def _no_wait_retry_init(self, **kwargs):
    kwargs.update(initial_delay=0.0, backoff_factor=0.0, jitter=False)
    _ORIGINAL_RETRY_INIT(self, **kwargs)
