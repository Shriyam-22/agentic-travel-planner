"""SQLite data layer.

The three JSON datasets are loaded once into a normalised SQLite database
(cities, airlines, amenities and place types live in lookup tables) with
indexes on the columns the tools filter and sort by. Every service queries
this database instead of re-reading the JSON files.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from travel_agent import config

logger = logging.getLogger(__name__)


class DatabaseError(RuntimeError):
    """Raised when the travel database cannot be built or queried."""


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cities (
    city_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    city_name   TEXT NOT NULL UNIQUE,
    latitude    REAL,
    longitude   REAL
);

CREATE TABLE IF NOT EXISTS airlines (
    airline_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    airline_name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS flights (
    flight_id            TEXT PRIMARY KEY,
    airline_id           INTEGER NOT NULL REFERENCES airlines(airline_id),
    origin_city_id       INTEGER NOT NULL REFERENCES cities(city_id),
    destination_city_id  INTEGER NOT NULL REFERENCES cities(city_id),
    departure_time       TEXT NOT NULL,
    arrival_time         TEXT NOT NULL,
    duration_minutes     INTEGER NOT NULL,
    price                INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS hotels (
    hotel_id         TEXT PRIMARY KEY,
    hotel_name       TEXT NOT NULL,
    city_id          INTEGER NOT NULL REFERENCES cities(city_id),
    stars            INTEGER NOT NULL,
    price_per_night  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS amenities (
    amenity_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    amenity_name  TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS hotel_amenities (
    hotel_id    TEXT NOT NULL REFERENCES hotels(hotel_id),
    amenity_id  INTEGER NOT NULL REFERENCES amenities(amenity_id),
    PRIMARY KEY (hotel_id, amenity_id)
);

CREATE TABLE IF NOT EXISTS place_types (
    place_type_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    type_name      TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS places (
    place_id       TEXT PRIMARY KEY,
    place_name     TEXT NOT NULL,
    city_id        INTEGER NOT NULL REFERENCES cities(city_id),
    place_type_id  INTEGER NOT NULL REFERENCES place_types(place_type_id),
    rating         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_flights_route   ON flights(origin_city_id, destination_city_id);
CREATE INDEX IF NOT EXISTS idx_flights_price   ON flights(price);
CREATE INDEX IF NOT EXISTS idx_hotels_city     ON hotels(city_id, price_per_night);
CREATE INDEX IF NOT EXISTS idx_hotels_stars    ON hotels(stars);
CREATE INDEX IF NOT EXISTS idx_places_city     ON places(city_id, rating);
"""


def _read_json(path: Path) -> list[dict]:
    """Read a JSON array from disk with a clear error if it is missing or malformed."""
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise DatabaseError(f"Dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DatabaseError(f"Dataset is not valid JSON: {path} ({exc})") from exc


def _get_or_create(cursor: sqlite3.Cursor, table: str, id_col: str, name_col: str, value: str) -> int:
    """Return the id of a lookup-table row, inserting it first if needed."""
    cursor.execute(f"SELECT {id_col} FROM {table} WHERE {name_col} = ?", (value,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute(f"INSERT INTO {table} ({name_col}) VALUES (?)", (value,))
    return cursor.lastrowid


def _city_id(cursor: sqlite3.Cursor, city_name: str) -> int:
    """Return a city id, creating the city (with coordinates if known)."""
    cursor.execute("SELECT city_id FROM cities WHERE city_name = ?", (city_name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    lat, lon = config.CITY_COORDINATES.get(city_name, (None, None))
    cursor.execute(
        "INSERT INTO cities (city_name, latitude, longitude) VALUES (?, ?, ?)",
        (city_name, lat, lon),
    )
    return cursor.lastrowid


def _flight_duration_minutes(departure: str, arrival: str) -> int:
    """Duration between two ISO timestamps, in minutes."""
    delta = datetime.fromisoformat(arrival) - datetime.fromisoformat(departure)
    return int(delta.total_seconds() // 60)


def build_database(db_path: Path | None = None, force: bool = False) -> Path:
    """Create the SQLite database from the JSON datasets.

    Args:
        db_path: Where to write the database (defaults to ``config.DB_PATH``).
        force: Rebuild even if the database already exists.

    Returns:
        The path of the database file.
    """
    db_path = Path(db_path or config.DB_PATH)
    if db_path.exists() and not force:
        return db_path
    if db_path.exists():
        db_path.unlink()

    flights = _read_json(config.FLIGHTS_FILE)
    hotels = _read_json(config.HOTELS_FILE)
    places = _read_json(config.PLACES_FILE)

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.executescript(SCHEMA_SQL)

            for city in config.CITY_COORDINATES:
                _city_id(cursor, city)

            for flight in flights:
                cursor.execute(
                    "INSERT INTO flights VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        flight["flight_id"],
                        _get_or_create(cursor, "airlines", "airline_id", "airline_name", flight["airline"]),
                        _city_id(cursor, flight["from"]),
                        _city_id(cursor, flight["to"]),
                        flight["departure_time"],
                        flight["arrival_time"],
                        _flight_duration_minutes(flight["departure_time"], flight["arrival_time"]),
                        int(flight["price"]),
                    ),
                )

            for hotel in hotels:
                cursor.execute(
                    "INSERT INTO hotels VALUES (?, ?, ?, ?, ?)",
                    (
                        hotel["hotel_id"],
                        hotel["name"],
                        _city_id(cursor, hotel["city"]),
                        int(hotel["stars"]),
                        int(hotel["price_per_night"]),
                    ),
                )
                for amenity in hotel.get("amenities", []):
                    amenity_id = _get_or_create(cursor, "amenities", "amenity_id", "amenity_name", amenity)
                    cursor.execute(
                        "INSERT OR IGNORE INTO hotel_amenities VALUES (?, ?)",
                        (hotel["hotel_id"], amenity_id),
                    )

            for place in places:
                cursor.execute(
                    "INSERT INTO places VALUES (?, ?, ?, ?, ?)",
                    (
                        place["place_id"],
                        place["name"],
                        _city_id(cursor, place["city"]),
                        _get_or_create(cursor, "place_types", "place_type_id", "type_name", place["type"]),
                        float(place["rating"]),
                    ),
                )
            conn.commit()
    except (sqlite3.Error, KeyError) as exc:
        if db_path.exists():
            db_path.unlink()
        raise DatabaseError(f"Failed to build travel database: {exc}") from exc

    logger.info("Built travel database at %s", db_path)
    return db_path


@contextmanager
def get_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a read connection (rows behave like dicts). Builds the DB on first use."""
    path = build_database(db_path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    except sqlite3.Error as exc:
        raise DatabaseError(f"Database query failed: {exc}") from exc
    finally:
        conn.close()


def query(sql: str, params: tuple = ()) -> list[dict]:
    """Run a SELECT and return the rows as plain dictionaries."""
    with get_connection() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def normalise_city(name: str) -> str | None:
    """Map user input ('bengaluru', ' goa ') to the dataset's city name, or None."""
    if not name:
        return None
    cleaned = name.strip().lower()
    if cleaned in config.CITY_ALIASES:
        return config.CITY_ALIASES[cleaned]
    for city in list_cities():
        if city.lower() == cleaned:
            return city
    return None


def list_cities() -> list[str]:
    """All cities present in the database."""
    return [row["city_name"] for row in query("SELECT city_name FROM cities ORDER BY city_name")]
