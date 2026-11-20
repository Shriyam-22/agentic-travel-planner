# ✈️ Agentic AI Travel Planner (LangChain)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://agentic-travel-planner-ne7j5dmse87tvy7hfaag87.streamlit.app/)

**▶️ Live demo:** [agentic-travel-planner-ne7j5dmse87tvy7hfaag87.streamlit.app](https://agentic-travel-planner-ne7j5dmse87tvy7hfaag87.streamlit.app/)

An AI travel agent that plans complete 3–7 day trips in India: flights, hotel, day-by-day itinerary, weather and a budget. It explains every choice it makes.

A **LangChain tool-calling agent** reads the traveller's request and decides which tools to call and in what order. It searches the flight, hotel and attraction datasets, checks live weather from **Open-Meteo**, estimates cost, and returns a **validated JSON itinerary** together with a human-readable version, through a **Streamlit** app or a **CLI**.

---

## 1. Problem & business use case

Planning a trip means comparing flights, hotels, attractions, weather and budget across many websites. Doing that by hand is slow and often produces itineraries that are unrealistic or over budget.

Travel platforms (MakeMyTrip, Booking.com, Cleartrip, Ixigo) are moving to **conversational, agentic self-service planning**. It cuts support workload, personalises recommendations, automates itinerary design and saves travellers time and money. This project is a working prototype of such an agent.

## 2. Objectives covered

| Objective | Where |
|---|---|
| Agentic system with LangChain that autonomously builds itineraries | `travel_agent/agent.py` (`create_agent` + tool calling) |
| Flight, hotel and places tools over the JSON datasets | `travel_agent/tools.py` → `travel_agent/services/` |
| Real-time weather (Open-Meteo, no key) | `services/weather_service.py` |
| Multi-step reasoning & decision making (ReAct-style tool loop) | agent system prompt + LangGraph loop |
| Day-wise plan, accommodation, weather, budget | `schemas.TripPlan` |
| Filtering, ranking, optimisation (cheapest / fastest / best-value flight, best-value hotel) | `flight_service.py`, `hotel_service.py` |
| Decision justification ("Why we chose this") | `reasoning` + per-flight/hotel `reason` fields |
| Clean JSON + human-readable output | Pydantic structured output + `formatter.py` |
| Simple interface | `app.py` (Streamlit) and `cli.py` |
| SQL best practices (normalised tables, indexes) | `travel_agent/database.py` |

## 3. Architecture

```
            ┌──────────────────────────── Streamlit app / CLI ────────────────────────────┐
 user ───▶  │  form fields or free text ──▶ build_query() ──▶ run_agent()                │
            └──────────────────────────────────────────┬──────────────────────────────────┘
                                                       ▼
                         LangChain agent  (create_agent, tool calling, ReAct loop)
                         LLM: Groq gpt-oss-120b / OpenAI / Gemini      response_format = TripPlan
                                                       │  decides which tools to call
     ┌──────────────┬──────────────┬───────────────────┼──────────────┬────────────────┐
     ▼              ▼              ▼                   ▼              ▼                ▼
 search_flights  search_hotels  search_places     get_weather    estimate_budget  list_supported_
 (direct + 1–3   (stars, price, (type, rating)    (Open-Meteo    (flights + hotel   destinations
  stop routing)   amenities)                       forecast/      + daily spend)
                                                   archive)
     └──────────────┴──────────────┴──── SQLite (normalised, indexed) ◀── flights/hotels/places.json
```

**Agent loop:** the LLM reasons about the request, calls a tool, reads the JSON result and decides on the next step. Typically it searches outbound and return flights, then hotels, places, weather and budget. It then composes the itinerary and returns it via LangChain's `ToolStrategy` structured output, validated against the `TripPlan` Pydantic model.

**Offline mode:** `travel_agent/planner.py` runs the same services in a fixed order with rule-based choices. Use it as a demo without an API key, as a baseline to compare against the agent, and for fast unit tests.

## 4. Project structure

```
agentic-travel-planner/
├── app.py                    # Streamlit UI (form, free-text chat, data explorer)
├── cli.py                    # Command-line interface
├── data/
│   ├── flights.json  hotels.json  places.json   # provided datasets
│   └── travel.db             # SQLite, auto-built on first run (git-ignored)
├── travel_agent/
│   ├── config.py             # paths, city coordinates, cost assumptions, LLM settings
│   ├── database.py           # JSON → normalised SQLite with indexes
│   ├── services/             # pure business logic (no LLM)
│   │   ├── flight_service.py   # direct + connecting search, cheapest/fastest/best value
│   │   ├── hotel_service.py    # filters + value ranking
│   │   ├── places_service.py   # attractions by type & rating
│   │   ├── weather_service.py  # Open-Meteo forecast / historical / graceful fallback
│   │   └── budget_service.py   # cost estimation
│   ├── tools.py              # LangChain @tool wrappers (validation + JSON output)
│   ├── agent.py              # LLM factory, system prompt, agent, run_agent()
│   ├── schemas.py            # Pydantic TripPlan (structured output)
│   ├── planner.py            # offline rule-based planner
│   └── formatter.py          # TripPlan → Markdown
├── tests/                    # pytest: services, tools, planner, agent loop (fake LLM)
├── outputs/                  # saved itineraries (CLI --save)
├── requirements.txt
└── .env.example
```

### Database schema (3NF)

`cities`, `airlines`, `amenities` and `place_types` are lookup tables. `flights`, `hotels` and `places` reference them by foreign key, and `hotel_amenities` is the many-to-many join. Indexes are on `flights(origin, destination)`, `flights(price)`, `hotels(city, price)`, `hotels(stars)` and `places(city, rating)`, which are the columns the tools filter and sort on.

## 5. Setup

```bash
git clone https://github.com/Shriyam-22/agentic-travel-planner.git && cd agentic-travel-planner
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then add your key
```

**Get a free LLM key:** [Groq](https://console.groq.com/keys) (default, free tier) · or OpenAI / Google Gemini. Set `LLM_PROVIDER` and the matching key in `.env`. You can also paste the key into the Streamlit sidebar.

### Run the app

Use the hosted version at **[the live demo](https://agentic-travel-planner-ne7j5dmse87tvy7hfaag87.streamlit.app/)**, or run it locally:

```bash
streamlit run app.py
```

- **Plan a trip:** structured form (cities, dates, days, travellers, style, budget, interests, amenities, flight preference).
- **Ask in your own words:** free-text request handled entirely by the agent.
- **Explore the data:** filterable tables of flights, hotels and places.
- The results show metrics, flights, hotel, a day-by-day plan with weather, a budget chart, reasoning, the **agent trace** (every tool call with its arguments and output), the JSON, and downloads.

### Deploy on Streamlit Community Cloud

1. On [share.streamlit.io](https://share.streamlit.io) choose **Create app → Deploy a public app from GitHub**.
2. Repository `Shriyam-22/agentic-travel-planner`, branch `main`, main file `app.py`.
3. **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
4. Deploy. The key stays on the server and is never shown in the page.

### CLI

```bash
python cli.py --query "4 days in Jaipur from Mumbai starting 2026-12-12 for 2 people, we love forts, budget 60000" -v
python cli.py --from Delhi --to Goa --start 2026-12-20 --days 5 --travellers 2 --interests beach museum --save
python cli.py --from Hyderabad --to Delhi --start 2026-12-01 --days 3 --offline     # no API key needed
```

### Tests

```bash
pytest -q
```

There are 22 tests. They cover the database, flight routing, hotel/place filters, the budget maths, weather (forecast / historical / failure, with the network mocked), tool error handling, the offline planner, and the **full LangChain agent loop**, driven by a scripted fake LLM so it runs without an API key.

## 6. Key design decisions

- **Flights as daily schedules.** The dataset has one sample date per flight (2025). Each flight is treated as a recurring daily service: its clock times, duration and fare apply to the traveller's chosen date. The raw sample dates are hidden from the LLM so it can't be confused by them.
- **Connecting flights.** Many routes have no direct flight (e.g. Delhi → Goa). The flight tool chains up to 3 legs with a minimum 60-minute layover and rolls to the next day when needed. Multi-day journeys produce "In transit" days, and hotel nights start from the arrival date.
- **Best-value flight (default).** The score is fare + ₹200 per hour of journey. For example, Mumbai → Jaipur: the cheapest option takes about 31 h, while the best-value option costs only ₹42 more and takes 8 h. Cheapest and fastest are also offered.
- **Best-value hotel.** Hotels are ranked by stars per ₹1,000 a night. Luxury trips rank by stars and budget trips by price.
- **Weather.** Dates within 16 days use the live forecast. Dates further out use the same dates last year from the Open-Meteo archive, labelled "typical". If the API fails, the plan still completes and the failure is flagged as a warning.
- **Budget.** Flights are per person and round trip. The hotel is charged per room (2 travellers/room) for each night. Food and local travel cost ₹1,200 / ₹2,000 / ₹4,000 per person per day for the budget / standard / luxury styles (set in `config.py`).
- **Robustness.** Tools validate city names (with aliases like Bengaluru → Bangalore) and always return JSON errors instead of raising. Database and API errors are wrapped in typed exceptions, and the agent has a recursion limit.

## 7. Sample output

```
# Your 4-Day Trip to Jaipur (Dec 12–Dec 15)

Flights
- Outbound: IndiGo + SpiceJet (FL0024, FL0008) Mumbai → Jaipur via Goa — departs 14:38,
  arrives 22:38 (8h 00m, 1 stop) — ₹10,059/person
  Why: best-value; the cheapest costs ₹42 less but takes 22h longer.
- Return: Vistara (FL0027) — departs 07:49, 3h 00m — ₹8,981/person

Hotel: Sea View Resort (5★) — ₹3,673/night; breakfast, parking, pool

Day 1: Arrive 22:38, check in
Day 2: Scenic Park · Beautiful Fort · Beautiful Park
Day 3: Famous Park · Scenic Temple · Local food trail
Day 4: Fly back 07:49

Estimated Budget (2 travellers)
- Flights: ₹38,080   - Hotel: ₹11,019   - Food & Local Travel: ₹16,000
Total Cost: ₹65,099 (₹32,550 per person)
```

## 8. Limitations & next steps

- The datasets are small samples (30 flights, 5 hotels and 5 places per city), and some place names and types are synthetic. Plug in a live flight/hotel API for production.
- Add distance-aware ordering of attractions (geocoding + nearest-neighbour routing).
- Add conversational memory (LangGraph checkpointer) so travellers can refine a plan ("make day 2 more relaxed").
- Add a train/road fallback for routes with long flight connections.

## Tech stack

Python 3.10+ · LangChain 1.x (`create_agent`, tools, structured output) · LangGraph · Groq / OpenAI / Gemini · Open-Meteo API · SQLite · Pydantic · Streamlit · pytest
