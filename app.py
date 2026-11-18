"""Streamlit UI for the Agentic AI Travel Planner.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from travel_agent import config
from travel_agent.agent import AgentError, build_query, run_agent
from travel_agent.database import DatabaseError, build_database, list_cities
from travel_agent.formatter import inr, to_markdown
from travel_agent.planner import plan_trip
from travel_agent.schemas import FlightChoice, TripPlan
from travel_agent.services.flight_service import get_all_flights
from travel_agent.services.hotel_service import list_amenities, search_hotels
from travel_agent.services.places_service import list_place_types, search_places

st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="wide")


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
@st.cache_resource
def init_database() -> list[str]:
    build_database()
    return list_cities()


try:
    CITIES = init_database()
except DatabaseError as exc:
    st.error(f"Could not load the travel datasets: {exc}")
    st.stop()

PROVIDER_LABELS = {"groq": "Groq (free tier)", "openai": "OpenAI", "google_genai": "Google Gemini"}
FLIGHT_PREFS = {"Best value (fare + time)": "best_value", "Cheapest": "cheapest", "Fastest": "fastest"}

# --------------------------------------------------------------------------- #
# Sidebar — engine settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Planner engine")
    engine = st.radio(
        "How should the trip be planned?",
        ["LangChain agent (LLM)", "Offline rule-based"],
        help="The agent reasons with an LLM and decides which tools to call. "
             "Offline mode runs the same tools with fixed rules and needs no API key.",
    )
    provider = model = api_key = None
    if engine.startswith("LangChain"):
        provider = st.selectbox("LLM provider", list(PROVIDER_LABELS), format_func=PROVIDER_LABELS.get,
                                index=list(PROVIDER_LABELS).index(config.LLM_PROVIDER)
                                if config.LLM_PROVIDER in PROVIDER_LABELS else 0)
        model = st.text_input("Model", value=config.DEFAULT_MODELS[provider])
        env_var = config.API_KEY_ENV_VARS[provider]
        api_key = st.text_input(f"API key ({env_var})", value=os.getenv(env_var, ""), type="password")
        if not api_key:
            st.info(f"Add `{env_var}` to `.env` or paste it above. Groq keys are free at console.groq.com.")
    st.divider()
    st.caption("**Agent tools**")
    st.caption("✈️ search_flights · 🏨 search_hotels · 📍 search_places · 🌦️ get_weather "
               "(Open-Meteo) · 💰 estimate_budget · 🗺️ list_supported_destinations")


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_flight(title: str, flight: FlightChoice | None) -> None:
    st.markdown(f"**{title}**")
    if flight is None:
        st.warning("No flight found in the dataset for this route.")
        return
    stops = "Direct" if flight.stops == 0 else f"{flight.stops} stop(s) via {', '.join(flight.via)}"
    arrival = f" ({flight.arrival_date})" if flight.arrival_date and flight.arrival_date != flight.travel_date else ""
    st.markdown(
        f"{' + '.join(flight.airlines)} · `{', '.join(flight.flight_ids)}`  \n"
        f"{flight.origin} → {flight.destination} · {flight.travel_date}  \n"
        f"🛫 {flight.departs_at} → 🛬 {flight.arrives_at}{arrival} · {flight.duration} · {stops}  \n"
        f"**{inr(flight.price_per_person)}** per person"
    )
    st.caption(f"Why: {flight.reason}")


def render_plan(plan: TripPlan, steps: list | None = None, engine_label: str = "") -> None:
    st.subheader(plan.trip_title)
    st.write(plan.trip_summary)
    if engine_label:
        st.caption(f"Planned by: {engine_label}")

    b = plan.budget
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total cost", inr(b.total))
    c2.metric("Per person", inr(b.per_person))
    c3.metric("Flights", inr(b.flights))
    c4.metric("Hotel", inr(b.hotel))
    if b.within_budget is False:
        st.warning("This plan is over your budget — try budget style, fewer days or a cheaper hotel.")

    tab_plan, tab_budget, tab_why, tab_json = st.tabs(["🗓️ Itinerary", "💰 Budget", "🧠 Reasoning", "🧾 JSON"])

    with tab_plan:
        left, right = st.columns(2)
        with left:
            render_flight("✈️ Outbound flight", plan.outbound_flight)
        with right:
            render_flight("✈️ Return flight", plan.return_flight)
        st.divider()
        if plan.hotel:
            h = plan.hotel
            st.markdown(f"**🏨 {h.name}** · {'★' * h.stars} · {inr(h.price_per_night)}/night · "
                        f"{', '.join(h.amenities) or 'no listed amenities'}")
            st.caption(f"Why: {h.reason}")
        st.divider()
        st.caption(f"🌦️ Weather — {plan.weather_source}")
        for day in plan.daily_plan:
            with st.container(border=True):
                st.markdown(f"**Day {day.day} · {day.date} — {day.title}**  ·  🌤️ {day.weather}")
                for a in day.activities:
                    detail = f" · {a.place_type} · ★{a.rating}" if a.place_type else ""
                    note = f" — _{a.note}_" if a.note else ""
                    st.markdown(f"- **{a.time_slot}:** {a.name}{detail}{note}")

    with tab_budget:
        breakdown = pd.DataFrame(
            {"Item": ["Flights", "Hotel", "Food & local travel"],
             "Amount (INR)": [b.flights, b.hotel, b.food_and_local_travel]}
        )
        st.bar_chart(breakdown, x="Item", y="Amount (INR)", horizontal=True)
        st.dataframe(breakdown, hide_index=True, width="stretch")
        st.markdown(f"**Total: {inr(b.total)}** ({inr(b.per_person)} per person)")
        if b.notes:
            st.caption(b.notes)

    with tab_why:
        st.markdown("#### Why we chose this")
        for reason in plan.reasoning:
            st.markdown(f"- {reason}")
        if plan.warnings:
            st.markdown("#### Things to know")
            for warning in plan.warnings:
                st.markdown(f"- ⚠️ {warning}")
        if steps:
            st.markdown("#### Agent trace — tools the agent decided to call")
            for i, step in enumerate(steps, 1):
                with st.expander(f"Step {i}: `{step.tool}`({json.dumps(step.arguments, ensure_ascii=False)})"):
                    try:
                        st.json(json.loads(step.output), expanded=False)
                    except json.JSONDecodeError:
                        st.code(step.output)

    with tab_json:
        st.json(plan.model_dump(), expanded=False)

    slug = f"trip_{plan.destination.lower()}_{plan.start_date}"
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Download JSON", plan.model_dump_json(indent=2), f"{slug}.json", "application/json")
    d2.download_button("⬇️ Download itinerary (Markdown)", to_markdown(plan), f"{slug}.md", "text/markdown")


def run_with_agent(query: str) -> None:
    error = None
    with st.status("🤖 Agent is planning your trip…", expanded=True) as status:
        st.write("Reasoning about your request and calling tools…")
        try:
            result = run_agent(query, provider=provider, model=model, api_key=api_key or None)
        except AgentError as exc:
            error = exc
            status.update(label="Planning failed — see the error below", state="error", expanded=False)
        else:
            for step in result.steps:
                st.write(f"🔧 `{step.tool}` {json.dumps(step.arguments, ensure_ascii=False)}")
            status.update(label=f"Done — {len(result.steps)} tool calls", state="complete", expanded=False)
    if error is not None:
        st.error(f"{error}\n\nTip: switch to **Offline rule-based** in the sidebar to try without an LLM.")
        return
    st.session_state["result"] = (result.plan, result.steps, f"LangChain agent · {provider}:{model}")


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.title("✈️ Agentic AI Travel Planner")
st.caption("LangChain tool-calling agent over flight, hotel and attraction data, with live Open-Meteo weather.")

tab_form, tab_chat, tab_data = st.tabs(["🧭 Plan a trip", "💬 Ask in your own words", "📊 Explore the data"])

with tab_form:
    with st.form("trip_form"):
        c1, c2, c3 = st.columns(3)
        origin = c1.selectbox("From", CITIES, index=CITIES.index("Delhi"))
        destination = c2.selectbox("To", CITIES, index=CITIES.index("Goa"))
        start = c3.date_input("Start date", value=date.today() + timedelta(days=7), min_value=date.today())

        c4, c5, c6, c7 = st.columns(4)
        num_days = c4.slider("Days", 3, 7, 4)
        travellers = c5.number_input("Travellers", 1, 10, 2)
        style = c6.selectbox("Travel style", ["standard", "budget", "luxury"])
        budget = c7.number_input("Total budget (₹, 0 = none)", 0, 1_000_000, 0, step=5000)

        c8, c9, c10 = st.columns(3)
        interests = c8.multiselect("Interests", list_place_types(), default=[])
        amenities = c9.multiselect("Must-have hotel amenities", list_amenities())
        min_stars = c10.slider("Minimum hotel stars", 0, 5, 0)
        flight_pref = st.radio("Flight preference", list(FLIGHT_PREFS), horizontal=True)
        notes = st.text_input("Anything else? (agent mode only)", placeholder="e.g. vegetarian food, slow mornings")
        submitted = st.form_submit_button("Plan my trip", type="primary", width="stretch")

    if submitted:
        if origin == destination:
            st.error("Pick different origin and destination cities.")
        elif engine.startswith("LangChain"):
            run_with_agent(build_query(
                origin, destination, start.isoformat(), num_days, travellers, style, budget or None,
                interests, min_stars, amenities, FLIGHT_PREFS[flight_pref], notes,
            ))
        else:
            with st.spinner("Planning with rule-based engine…"):
                plan = plan_trip(origin, destination, start.isoformat(), num_days, travellers, style,
                                 budget or None, interests, min_stars, amenities, FLIGHT_PREFS[flight_pref])
            st.session_state["result"] = (plan, None, "Offline rule-based planner")

with tab_chat:
    st.write("Describe your trip naturally — the agent works out what to search for.")
    examples = [
        "Plan a 4-day trip from Mumbai to Jaipur starting 2026-12-12 for 2 people. "
        "We love forts and parks; budget ₹60,000.",
        "Cheapest 3-day Kolkata trip from Hyderabad next weekend, 5-star hotel with spa, temples please.",
        "Luxury 5 days in Goa from Bangalore in January with a pool hotel.",
    ]
    example = st.selectbox("Try an example", ["—"] + examples)
    query = st.text_area("Your request", value="" if example == "—" else example, height=100)
    if st.button("Ask the agent", type="primary", disabled=not query.strip()):
        if engine.startswith("LangChain"):
            run_with_agent(query)
        else:
            st.info("Free-text requests need the LLM agent. Switch the engine in the sidebar, "
                    "or use the **Plan a trip** form for offline mode.")

with tab_data:
    dataset = st.radio("Dataset", ["Flights", "Hotels", "Places"], horizontal=True)
    cities = st.multiselect("Cities", CITIES, default=CITIES)
    if dataset == "Flights":
        df = pd.DataFrame(get_all_flights())[
            ["flight_id", "airline", "origin", "destination", "departs_at", "arrives_at", "duration", "price"]]
        max_price = st.slider("Max fare (₹)", 0, int(df.price.max()), int(df.price.max()))
        df = df[(df.origin.isin(cities) | df.destination.isin(cities)) & (df.price <= max_price)]
    elif dataset == "Hotels":
        stars = st.slider("Minimum stars", 1, 5, 1)
        df = pd.DataFrame([h for c in cities for h in search_hotels(c, min_stars=stars, limit=100)])
        if not df.empty:
            df["amenities"] = df["amenities"].apply(", ".join)
    else:
        rating = st.slider("Minimum rating", 0.0, 5.0, 0.0, 0.1)
        df = pd.DataFrame([p for c in cities for p in search_places(c, min_rating=rating, limit=100)])
        if not df.empty:
            df = df.drop(columns=["matches_preference"])
    st.dataframe(df, hide_index=True, width="stretch")
    st.caption(f"{len(df)} rows")

# --------------------------------------------------------------------------- #
# Result (persists across reruns)
# --------------------------------------------------------------------------- #
if "result" in st.session_state:
    st.divider()
    plan, steps, label = st.session_state["result"]
    render_plan(plan, steps, label)
