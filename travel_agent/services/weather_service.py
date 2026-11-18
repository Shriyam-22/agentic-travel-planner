"""Weather lookup via the free Open-Meteo API (no key required).

* Trip dates within the 16-day forecast window -> live forecast.
* Dates further out -> the same calendar dates last year from the Open-Meteo
  archive, labelled as "typical" weather.
* Network/API failure -> an explicit "unavailable" result so the agent can
  carry on and warn the traveller instead of crashing.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from functools import lru_cache

import requests

from travel_agent import config

logger = logging.getLogger(__name__)

# WMO weather interpretation codes -> plain English
WMO_CODES: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Freezing drizzle", 57: "Freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}

_DAILY_VARS = "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum"


class WeatherServiceError(RuntimeError):
    """Raised when Open-Meteo cannot be reached or returns bad data."""


def describe_weather_code(code: int | None) -> str:
    return WMO_CODES.get(int(code), "Unknown") if code is not None else "Unknown"


def _shift_year_back(day: date) -> date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:  # 29 Feb
        return day.replace(year=day.year - 1, day=28)


@lru_cache(maxsize=128)
def _fetch_daily(url: str, latitude: float, longitude: float, start: str, end: str) -> dict:
    """Call Open-Meteo and return its ``daily`` block (cached per request)."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": _DAILY_VARS,
        "timezone": "auto",
        "start_date": start,
        "end_date": end,
    }
    try:
        response = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except requests.Timeout as exc:
        raise WeatherServiceError("Open-Meteo timed out") from exc
    except requests.HTTPError as exc:
        raise WeatherServiceError(f"Open-Meteo returned HTTP {exc.response.status_code}") from exc
    except requests.RequestException as exc:
        logger.debug("Open-Meteo connection error: %s", exc)
        raise WeatherServiceError("could not connect to Open-Meteo") from exc
    except ValueError as exc:
        raise WeatherServiceError("Open-Meteo returned invalid JSON") from exc
    if "daily" not in payload:
        raise WeatherServiceError(payload.get("reason", "Open-Meteo response had no daily data"))
    return payload["daily"]


def _packing_tip(temp_max: float | None, precipitation: float | None) -> str:
    if precipitation is not None and precipitation >= 5:
        return "Carry an umbrella; plan indoor options."
    if temp_max is not None and temp_max >= 33:
        return "Hot — keep outdoor visits to mornings/evenings and stay hydrated."
    if temp_max is not None and temp_max <= 18:
        return "Cool — pack a light jacket."
    return "Pleasant — good for sightseeing."


def get_weather_forecast(city: str, start_date: str, num_days: int, today: date | None = None) -> dict:
    """Day-by-day weather for a trip.

    Args:
        city: Dataset city name (must be in ``config.CITY_COORDINATES``).
        start_date: First day of the trip, ``YYYY-MM-DD``.
        num_days: Number of days to cover.
        today: Override for "today" (used by tests).

    Returns:
        ``{"city", "source", "note", "days": [{date, condition, temp_max_c, ...}]}``
    """
    if city not in config.CITY_COORDINATES:
        return {"city": city, "source": "unavailable", "note": f"No coordinates for {city}.", "days": []}

    latitude, longitude = config.CITY_COORDINATES[city]
    today = today or date.today()
    start = date.fromisoformat(start_date)
    end = start + timedelta(days=max(num_days, 1) - 1)

    horizon_end = today + timedelta(days=config.FORECAST_HORIZON_DAYS - 1)
    if today <= start and end <= horizon_end:
        url, query_start, query_end = config.OPEN_METEO_FORECAST_URL, start, end
        source, note = "forecast", "Live Open-Meteo forecast."
    else:
        url = config.OPEN_METEO_ARCHIVE_URL
        query_start, query_end = _shift_year_back(start), _shift_year_back(end)
        source = "historical"
        note = (
            "Trip dates are outside the 16-day forecast window, so this shows the weather observed "
            f"on the same dates in {query_start.year} as a typical expectation."
        )

    try:
        daily = _fetch_daily(url, latitude, longitude, query_start.isoformat(), query_end.isoformat())
    except WeatherServiceError as exc:
        logger.warning("Weather lookup failed for %s: %s", city, exc)
        return {
            "city": city,
            "source": "unavailable",
            "note": f"Weather service unavailable ({exc}). Check a forecast closer to travel.",
            "days": [
                {"date": (start + timedelta(days=i)).isoformat(), "condition": "Unavailable",
                 "temp_max_c": None, "temp_min_c": None, "precipitation_mm": None, "tip": ""}
                for i in range(num_days)
            ],
        }

    days = []
    for i in range(len(daily.get("time", []))):
        temp_max = daily["temperature_2m_max"][i]
        precipitation = daily["precipitation_sum"][i]
        days.append(
            {
                "date": (start + timedelta(days=i)).isoformat(),
                "condition": describe_weather_code(daily["weather_code"][i]),
                "temp_max_c": temp_max,
                "temp_min_c": daily["temperature_2m_min"][i],
                "precipitation_mm": precipitation,
                "tip": _packing_tip(temp_max, precipitation),
            }
        )
    return {"city": city, "source": source, "note": note, "days": days}
