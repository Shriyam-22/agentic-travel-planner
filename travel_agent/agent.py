"""The LangChain tool-calling travel agent.

Built with ``langchain.agents.create_agent`` (LangChain 1.x). The model decides
which tools to call and in what order (ReAct-style reason → act → observe loop),
and returns its final answer as a validated :class:`TripPlan` via
``ToolStrategy`` structured output.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from travel_agent import config
from travel_agent.schemas import TripPlan
from travel_agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)


class AgentError(RuntimeError):
    """Raised when the LLM agent cannot produce an itinerary."""


SYSTEM_PROMPT = """You are an expert Indian travel planner agent. Today is {today}.
You build realistic, optimised trip itineraries using ONLY the data your tools return.

Work step by step:
1. Understand the request: origin, destination, start date, number of days (3-7), travellers,
   travel style (budget/standard/luxury), budget, interests (place types), hotel needs.
   If something is missing, assume sensible defaults (1 traveller, standard style, 3 days,
   start date 14 days from today) and record each assumption in `warnings`.
2. Call `search_flights` for the outbound journey AND for the return journey (destination -> origin).
   Use the user's flight preference: "cheapest", "fastest" or "best_value" (default) — best_value
   balances fare and journey time so a slightly dearer but much shorter route wins.
3. Call `search_hotels` for the destination with the user's star/amenity/price needs.
   Prefer the best value (stars per rupee) unless the user asks for luxury or cheapest.
   Relax filters and say so if nothing matches.
4. Call `search_places` with the user's preferred attraction types.
5. Call `get_weather` for the destination and trip dates.
6. Call `estimate_budget` with the chosen fares (per person; sum connecting legs), hotel rate,
   nights = days - 1, days, travellers, travel style and the user's budget if given.
7. Build the day-wise plan:
   - Arrival day: set `arrival_date` on each flight (connecting options report `days_in_transit`).
     Days before the outbound flight lands are "In transit" days. On arrival day schedule only light
     activities after landing.
   - Last day: departure — one morning activity only if the return flight leaves after 12:00.
   - Middle days: 2-3 places per day, highest rated / preferred types first, never repeat a place.
     If you run out of places, add free-time items (local food, markets, shopping) with place_id null.
   - Put outdoor places on better-weather days; add weather to every day.
8. Fill `reasoning` with why each flight, hotel and the activity order was chosen (mention
   alternatives and prices). Fill `warnings` with caveats: long connections, weather source,
   over-budget, assumptions, and that the dataset is a sample (verify before booking).

Rules: never invent flights, hotels, places, prices or weather. Use INR. Dates are YYYY-MM-DD.

Final answer: call the `TripPlan` tool exactly once with the complete plan (no other tool name, no
plain-text JSON). Keep it compact: at most 4 activities per day, notes under 15 words, reasoning and
warnings as short sentences, plain ASCII punctuation.
"""

FINALIZE_PROMPT = """You are an expert Indian travel planner. Today is {today}.
The research for this trip is already done: the user's request and every tool result are below.
Using ONLY these results (never invent flights, hotels, places, prices or weather), write the final
itinerary following these rules:
- Respect the user's flight preference (best_value by default). Days before the outbound flight lands
  are "In transit". On arrival day only light activities after landing. On the last day one morning
  activity only if the return flight leaves after 12:00.
- 2-3 places per full day, best rated / preferred types first, never repeat a place; free-time items
  have place_id null. Add the weather to every day.
- Use the estimate_budget result for the budget. Explain choices in `reasoning`; list caveats in `warnings`.
- Keep it compact: at most 4 activities per day, notes under 15 words. Use INR and YYYY-MM-DD dates.
"""


@dataclass
class AgentStep:
    """One tool call made by the agent, for the 'agent trace' view."""

    tool: str
    arguments: dict
    output: str


@dataclass
class AgentResult:
    """Outcome of a single agent run."""

    plan: TripPlan
    steps: list[AgentStep] = field(default_factory=list)
    model: str = ""


def build_llm(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Create a chat model for Groq, OpenAI or Google Gemini.

    The API key may be passed directly (e.g. from the Streamlit sidebar) or read
    from the provider's environment variable (``GROQ_API_KEY`` etc.).
    """
    provider = (provider or config.LLM_PROVIDER).lower()
    if provider not in config.DEFAULT_MODELS:
        raise AgentError(f"Unsupported LLM provider '{provider}'. Use one of {list(config.DEFAULT_MODELS)}.")
    model = model or config.DEFAULT_MODELS[provider]
    env_var = config.API_KEY_ENV_VARS[provider]
    if api_key:
        os.environ[env_var] = api_key
    if not os.getenv(env_var):
        raise AgentError(f"No API key found. Set {env_var} in your .env file or enter it in the app.")
    try:
        return init_chat_model(
            model,
            model_provider=provider,
            temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        )
    except Exception as exc:  # provider packages raise assorted errors
        raise AgentError(f"Could not initialise {provider}:{model} — {exc}") from exc


def _is_retryable(exc: Exception) -> bool:
    """Retry malformed tool calls and transient server errors, not auth or quota problems."""
    text = str(exc)
    return any(marker in text for marker in ("tool_use_failed", "Failed to parse tool call",
                                              "Error code: 500", "Error code: 502", "Error code: 503"))


def build_agent(llm: BaseChatModel, checkpointer=None):
    """Assemble the tool-calling agent with structured TripPlan output.

    ``ModelRetryMiddleware`` re-asks the model when it emits a malformed tool call
    (some open models occasionally garble long JSON arguments).
    """
    return create_agent(
        model=llm,
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT.format(today=date.today().isoformat()),
        response_format=ToolStrategy(TripPlan),
        middleware=[ModelRetryMiddleware(max_retries=2, retry_on=_is_retryable, on_failure="error",
                                         initial_delay=1.0)],
        checkpointer=checkpointer,
    )


def _extract_steps(messages: list) -> list[AgentStep]:
    """Pair each tool call with its tool output for display."""
    outputs = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    steps = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if call["name"] == "TripPlan":  # the structured-output "tool"
                    continue
                steps.append(AgentStep(call["name"], call.get("args", {}), str(outputs.get(call["id"], ""))))
    return steps


def _fallback_parse(messages: list) -> TripPlan | None:
    """If the model answered in plain JSON instead of the structured tool, parse it."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and "{" in message.content:
            text = message.content[message.content.find("{"): message.content.rfind("}") + 1]
            try:
                return TripPlan.model_validate(json.loads(text))
            except (ValueError, TypeError):
                return None
    return None


def _friendly_error(exc: Exception) -> str:
    """Turn provider errors into messages a traveller can act on."""
    text = str(exc)
    if "rate_limit" in text or "Error code: 429" in text or "Request too large" in text or "Error code: 413" in text:
        return ("The AI provider's free-tier limit was reached. Wait a minute and try again, "
                "or shorten the trip / pick a route with direct flights.")
    if "model_not_found" in text or "Error code: 404" in text:
        return ("The selected model is not available for your API key. "
                f"Pick another model in the sidebar. ({text[:160]})")
    if "invalid_api_key" in text or "Error code: 401" in text:
        return "The API key was rejected. Check the key in .env or the app's Secrets."
    if "tool_use_failed" in text or "Failed to parse tool call" in text:
        return "The AI model returned a malformed answer several times. Please try again."
    return f"The agent failed while planning: {text[:300]}"


def _research_digest(query: str, messages: list) -> str:
    """The user request plus every tool call and its result, as plain text."""
    outputs = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    lines = [f"USER REQUEST: {query}", "", "TOOL RESULTS:"]
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                if call["name"] == "TripPlan" or call["id"] not in outputs:
                    continue
                lines.append(f"- {call['name']}({json.dumps(call.get('args', {}), ensure_ascii=False)})")
                lines.append(f"  -> {outputs[call['id']]}")
    return "\n".join(lines)


def finalize_plan(llm: BaseChatModel, query: str, messages: list) -> TripPlan:
    """Write the TripPlan from already-gathered tool results in one structured-output call.

    Used as a fallback when the agent's own final tool call fails. JSON-schema mode
    constrains the output, which is far more reliable for long structured answers.
    """
    prompt = [
        SystemMessage(FINALIZE_PROMPT.format(today=date.today().isoformat())),
        HumanMessage(_research_digest(query, messages)),
    ]
    try:
        structured = llm.with_structured_output(TripPlan, method="json_schema")
    except (TypeError, ValueError, NotImplementedError):
        structured = llm.with_structured_output(TripPlan)
    plan = structured.invoke(prompt)
    if isinstance(plan, dict):
        plan = TripPlan.model_validate(plan)
    return plan


def run_agent(query: str, llm: BaseChatModel | None = None, **llm_kwargs) -> AgentResult:
    """Plan a trip from a natural-language request.

    Args:
        query: e.g. "4 days in Jaipur from Mumbai starting 2026-12-12, 2 people, love forts".
        llm: A ready chat model; if omitted one is built from ``llm_kwargs`` / environment.

    If the agent gathers its data but fails to return the final structured plan,
    :func:`finalize_plan` writes it from the tool results instead of starting over.

    Raises:
        AgentError: if the model fails or does not return a valid itinerary.
    """
    llm = llm or build_llm(**llm_kwargs)
    agent = build_agent(llm, checkpointer=InMemorySaver())
    run_config = {"recursion_limit": config.AGENT_RECURSION_LIMIT,
                  "configurable": {"thread_id": str(uuid.uuid4())}}
    model_name = str(getattr(llm, "model_name", None) or getattr(llm, "model", "") or "")

    try:
        state = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=run_config)
    except Exception as exc:
        logger.warning("Agent run failed (%s); trying to finalise from gathered results", exc)
        try:
            messages = agent.get_state(run_config).values.get("messages", [])
        except Exception:  # noqa: BLE001 - state may be unavailable
            messages = []
        steps = _extract_steps(messages)
        if not steps:
            raise AgentError(_friendly_error(exc)) from exc
        try:
            plan = finalize_plan(llm, query, messages)
        except Exception as final_exc:
            logger.exception("Fallback finalisation failed")
            raise AgentError(_friendly_error(final_exc)) from exc
        return AgentResult(plan=plan, steps=steps, model=model_name)

    messages = state.get("messages", [])
    plan = state.get("structured_response") or _fallback_parse(messages)
    if plan is None:
        steps = _extract_steps(messages)
        if not steps:
            raise AgentError("The agent finished without returning a structured itinerary.")
        try:
            plan = finalize_plan(llm, query, messages)
        except Exception as exc:
            raise AgentError(_friendly_error(exc)) from exc
    return AgentResult(plan=plan, steps=_extract_steps(messages), model=model_name)


def build_query(
    origin: str,
    destination: str,
    start_date: str,
    num_days: int,
    travellers: int = 1,
    travel_style: str = "standard",
    budget: int | None = None,
    interests: list[str] | None = None,
    min_hotel_stars: int = 0,
    amenities: list[str] | None = None,
    flight_preference: str = "best_value",
    extra_notes: str = "",
) -> str:
    """Turn the Streamlit form fields into a natural-language request for the agent."""
    parts = [
        f"Plan a {num_days}-day trip from {origin} to {destination} starting {start_date}",
        f"for {travellers} traveller(s), {travel_style} travel style.",
        f"Flight preference: {flight_preference}.",
    ]
    if budget:
        parts.append(f"Total budget: ₹{budget:,}.")
    if interests:
        parts.append(f"Interested in: {', '.join(interests)}.")
    if min_hotel_stars:
        parts.append(f"Hotel at least {min_hotel_stars} stars.")
    if amenities:
        parts.append(f"Hotel must have: {', '.join(amenities)}.")
    if extra_notes.strip():
        parts.append(f"Notes: {extra_notes.strip()}")
    return " ".join(parts)
