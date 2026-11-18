"""Pydantic models for the agent's structured output (the final itinerary JSON)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FlightChoice(BaseModel):
    """The flight (or connecting itinerary) the agent selected."""

    flight_ids: list[str] = Field(description="Flight id(s) from the dataset; more than one for connections.")
    airlines: list[str] = Field(description="Airline name(s).")
    origin: str
    destination: str
    travel_date: str = Field(description="Date of travel, YYYY-MM-DD.")
    departs_at: str = Field(description="Departure clock time HH:MM.")
    arrives_at: str = Field(description="Arrival clock time HH:MM.")
    arrival_date: str | None = Field(
        default=None, description="YYYY-MM-DD; later than travel_date for overnight/multi-day journeys."
    )
    duration: str = Field(description="Total journey time, e.g. '4h 00m'.")
    stops: int = 0
    via: list[str] = Field(default_factory=list, description="Connection cities, if any.")
    price_per_person: int
    reason: str = Field(description="Why this flight was chosen over the alternatives.")


class HotelChoice(BaseModel):
    """The hotel the agent recommended."""

    hotel_id: str
    name: str
    city: str
    stars: int
    price_per_night: int
    amenities: list[str] = Field(default_factory=list)
    reason: str = Field(description="Why this hotel was chosen over the alternatives.")


class Activity(BaseModel):
    """One stop in a day's plan."""

    time_slot: str = Field(description="Morning / Afternoon / Evening, or a clock time.")
    name: str
    place_id: str | None = Field(default=None, description="Dataset place id, or null for free-time items.")
    place_type: str | None = None
    rating: float | None = None
    note: str | None = None


class DayPlan(BaseModel):
    """Itinerary and weather for one day of the trip."""

    day: int
    date: str = Field(description="YYYY-MM-DD")
    title: str = Field(description="Short theme for the day, e.g. 'Arrival & lakeside evening'.")
    weather: str = Field(description="Condition and temperature, e.g. 'Partly cloudy, 31°C / 22°C'.")
    activities: list[Activity]


class BudgetBreakdown(BaseModel):
    """Estimated cost of the whole trip in INR."""

    flights: int
    hotel: int
    food_and_local_travel: int
    total: int
    per_person: int
    currency: str = "INR"
    within_budget: bool | None = None
    notes: str | None = None


class TripPlan(BaseModel):
    """Complete itinerary returned by the travel agent."""

    trip_title: str = Field(description="e.g. 'Your 4-Day Trip to Jaipur (Dec 12–15)'.")
    trip_summary: str = Field(description="2–3 sentence overview of the trip.")
    origin: str
    destination: str
    start_date: str
    end_date: str
    num_days: int
    travellers: int
    outbound_flight: FlightChoice | None = None
    return_flight: FlightChoice | None = None
    hotel: HotelChoice | None = None
    weather_source: str = Field(description="'forecast', 'historical' or 'unavailable', with a short note.")
    daily_plan: list[DayPlan]
    budget: BudgetBreakdown
    reasoning: list[str] = Field(description="Key decisions and why they were made.")
    warnings: list[str] = Field(default_factory=list, description="Caveats the traveller should know.")
