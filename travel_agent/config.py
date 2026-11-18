"""Central configuration: paths, city coordinates, cost assumptions and LLM settings."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DB_PATH = Path(os.getenv("TRAVEL_DB_PATH", DATA_DIR / "travel.db"))

FLIGHTS_FILE = DATA_DIR / "flights.json"
HOTELS_FILE = DATA_DIR / "hotels.json"
PLACES_FILE = DATA_DIR / "places.json"

# --------------------------------------------------------------------------- #
# Geography — used by the weather tool (Open-Meteo needs lat/lon)
# --------------------------------------------------------------------------- #
CITY_COORDINATES: dict[str, tuple[float, float]] = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Goa": (15.2993, 74.1240),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639),
    "Jaipur": (26.9124, 75.7873),
}

# Common alternative spellings users type -> canonical dataset name
CITY_ALIASES: dict[str, str] = {
    "new delhi": "Delhi",
    "bengaluru": "Bangalore",
    "bombay": "Mumbai",
    "madras": "Chennai",
    "calcutta": "Kolkata",
    "panaji": "Goa",
    "pink city": "Jaipur",
}

# --------------------------------------------------------------------------- #
# Budget assumptions (INR, per person per day) for food + local transport
# --------------------------------------------------------------------------- #
DAILY_LOCAL_EXPENSES: dict[str, int] = {
    "budget": 1200,
    "standard": 2000,
    "luxury": 4000,
}
CURRENCY = "INR"
CURRENCY_SYMBOL = "₹"
TRAVELLERS_PER_ROOM = 2

# Flight routing
DEFAULT_MAX_STOPS = 3
MIN_LAYOVER_MINUTES = 60
# 'Best value' flight score = fare + this many rupees per hour of journey time
TIME_VALUE_PER_HOUR = 200

# --------------------------------------------------------------------------- #
# Weather API (Open-Meteo, no key needed)
# --------------------------------------------------------------------------- #
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 16
HTTP_TIMEOUT_SECONDS = 10

# --------------------------------------------------------------------------- #
# LLM settings (override with environment variables or the Streamlit sidebar)
# --------------------------------------------------------------------------- #
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # groq | openai | google_genai
DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "google_genai": "gemini-2.0-flash",
}
API_KEY_ENV_VARS: dict[str, str] = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
}
LLM_MODEL = os.getenv("LLM_MODEL", DEFAULT_MODELS.get(LLM_PROVIDER, "llama-3.3-70b-versatile"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "30"))
