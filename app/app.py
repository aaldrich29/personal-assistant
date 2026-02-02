import os

# Fix ONNX runtime thread affinity errors in Docker - must be set before importing onnxruntime
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["ORT_DISABLE_CPU_AFFINITY"] = "1"

# Suppress ONNX runtime thread affinity errors (they're harmless but noisy in Docker)
import onnxruntime as ort
ort.set_default_logger_severity(4)  # 4 = fatal only, suppresses the pthread_setaffinity_np errors

import json
import asyncio
import base64
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Union
import urllib.request
import ssl
from zoneinfo import ZoneInfo

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from openai import AsyncOpenAI
import chromadb
from sqlalchemy import create_engine, text

# ============================================================================
# Authentication
# ============================================================================
AUTH_USERNAME = os.environ.get("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("AUTH_PASSWORD", "personal_assistant_vault")

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authenticate the user."""
    if (username, password) == (AUTH_USERNAME, AUTH_PASSWORD):
        return cl.User(identifier=username, metadata={"role": "admin", "provider": "credentials"})
    return None

# ============================================================================
# Configuration
# ============================================================================

VAULT_PATH = Path(os.environ.get("VAULT_PATH", "/vault"))
CHROMA_HOST = os.environ.get("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemini-3-flash-preview")
SEARCH_MODEL = os.environ.get("SEARCH_MODEL", "google/gemini-3-flash-preview:online")
EXTRACTION_MODEL = os.environ.get("EXTRACTION_MODEL", "google/gemini-3-flash-preview")
ICAL_URL = os.environ.get("ICAL_URL", "")

# SQLite Database Configuration
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
SQLITE_DB_PATH = DATA_DIR / "chainlit.db"

def init_sqlite_db():
    """Initialize SQLite database with required schema for Chainlit data layer."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Use synchronous engine for schema creation
    engine = create_engine(f"sqlite:///{SQLITE_DB_PATH}")

    with engine.connect() as conn:
        # Create users table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                "id" TEXT PRIMARY KEY,
                "identifier" TEXT NOT NULL UNIQUE,
                "metadata" TEXT NOT NULL DEFAULT '{}',
                "createdAt" TEXT
            )
        """))

        # Create threads table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS threads (
                "id" TEXT PRIMARY KEY,
                "createdAt" TEXT,
                "name" TEXT,
                "userId" TEXT,
                "userIdentifier" TEXT,
                "tags" TEXT,
                "metadata" TEXT,
                FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
            )
        """))

        # Create steps table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS steps (
                "id" TEXT PRIMARY KEY,
                "name" TEXT NOT NULL,
                "type" TEXT NOT NULL,
                "threadId" TEXT NOT NULL,
                "parentId" TEXT,
                "streaming" INTEGER NOT NULL,
                "waitForAnswer" INTEGER,
                "isError" INTEGER,
                "metadata" TEXT,
                "tags" TEXT,
                "input" TEXT,
                "output" TEXT,
                "createdAt" TEXT,
                "command" TEXT,
                "start" TEXT,
                "end" TEXT,
                "generation" TEXT,
                "showInput" TEXT,
                "language" TEXT,
                "indent" INTEGER,
                "defaultOpen" INTEGER,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            )
        """))

        # Create elements table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS elements (
                "id" TEXT PRIMARY KEY,
                "threadId" TEXT,
                "type" TEXT,
                "url" TEXT,
                "chainlitKey" TEXT,
                "name" TEXT NOT NULL,
                "display" TEXT,
                "objectKey" TEXT,
                "size" TEXT,
                "page" INTEGER,
                "language" TEXT,
                "forId" TEXT,
                "mime" TEXT,
                "props" TEXT,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            )
        """))

        # Create feedbacks table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feedbacks (
                "id" TEXT PRIMARY KEY,
                "forId" TEXT NOT NULL,
                "threadId" TEXT NOT NULL,
                "value" INTEGER NOT NULL,
                "comment" TEXT,
                FOREIGN KEY ("threadId") REFERENCES threads("id") ON DELETE CASCADE
            )
        """))

        conn.commit()

    print(f"[DB] SQLite database initialized at {SQLITE_DB_PATH}")

# Initialize the database on module load
init_sqlite_db()

@cl.data_layer
def get_data_layer():
    """Return the SQLAlchemy data layer for Chainlit persistence."""
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{SQLITE_DB_PATH}")


def get_local_timezone() -> ZoneInfo:
    """Get the configured timezone or fall back to system local. Uses TZ env var."""
    tz = os.environ.get("TZ", "")
    if tz:
        try:
            return ZoneInfo(tz)
        except Exception as e:
            print(f"Invalid TZ '{tz}': {e}, falling back to local")
    # Fall back to system local timezone
    try:
        import time
        local_tz_name = time.tzname[0]
        # Try to get a proper ZoneInfo from localtime
        return datetime.now().astimezone().tzinfo
    except Exception:
        return timezone.utc


# Initialize OpenRouter client (OpenAI-compatible)
client = AsyncOpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Initialize ChromaDB client
chroma_client = None

def get_chroma_client():
    global chroma_client
    if chroma_client is None:
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return chroma_client

# ============================================================================
# Vault File Operations
# ============================================================================

TOPIC_REGISTRY_PATH = "meta/topic_registry.json"

def read_vault_file(relative_path: str) -> str:
    """Read a file from the vault, return empty string if not found."""
    file_path = VAULT_PATH / relative_path
    if file_path.exists():
        return file_path.read_text()
    return ""

def write_vault_file(relative_path: str, content: str) -> None:
    """Write content to a vault file, creating directories as needed."""
    file_path = VAULT_PATH / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content)

def append_vault_file(relative_path: str, content: str) -> None:
    """Append content to a vault file."""
    file_path = VAULT_PATH / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a") as f:
        f.write(content)

def get_existing_vault_files() -> List[str]:
    """Get list of all markdown files in the vault (excluding system dirs)."""
    exclude_dirs = {'conversations', 'daily', 'meta'}
    files = []
    for md_file in VAULT_PATH.rglob("*.md"):
        rel_path = md_file.relative_to(VAULT_PATH)
        # Skip excluded directories
        if rel_path.parts[0] not in exclude_dirs:
            files.append(str(rel_path))
    return sorted(files)

# ============================================================================
# Topic Registry Operations
# ============================================================================

def read_topic_registry() -> Dict:
    """Read the topic registry, return dict with topics key if not found."""
    content = read_vault_file(TOPIC_REGISTRY_PATH)
    if content:
        try:
            registry = json.loads(content)
            # Ensure topics key exists
            if "topics" not in registry:
                registry["topics"] = {}
            return registry
        except json.JSONDecodeError:
            print("[REGISTRY] Failed to parse topic registry, returning empty")
            return {"topics": {}}
    return {"topics": {}}

def write_topic_registry(registry: Dict) -> None:
    """Write the topic registry to the vault."""
    write_vault_file(TOPIC_REGISTRY_PATH, json.dumps(registry, indent=2))

def register_topic(topic_id: str, description: str, vault_path: str, keywords: List[str]) -> bool:
    """Register a topic in the registry if not already present. Returns True if registered."""
    registry = read_topic_registry()
    if topic_id in registry.get("topics", {}):
        return False  # Already registered
    registry["topics"][topic_id] = {
        "description": description,
        "vault_path": vault_path,
        "keywords": keywords,
        "created": datetime.now().isoformat()
    }
    write_topic_registry(registry)
    print(f"[REGISTRY] Registered new topic: {topic_id} -> {vault_path}")
    return True

def initialize_topic_registry() -> int:
    """Scan existing vault files and add them to the topic registry. Returns count of new topics."""
    registry = read_topic_registry()
    existing_topics = set(registry.get("topics", {}).keys())
    existing_paths = {info.get("vault_path") for info in registry.get("topics", {}).values()}

    vault_files = get_existing_vault_files()
    added = 0

    for file_path in vault_files:
        # Skip if this path is already registered
        if file_path in existing_paths:
            continue

        # Generate topic_id from path (e.g., "family/members/john.md" -> "family_members_john")
        topic_id = file_path.replace("/", "_").replace(".md", "").replace(" ", "_").lower()

        # Skip if topic_id already exists
        if topic_id in existing_topics:
            continue

        # Generate description from path
        parts = file_path.replace(".md", "").split("/")
        if len(parts) > 1:
            description = f"{parts[-1].replace('_', ' ').title()} ({parts[0]})"
        else:
            description = parts[0].replace("_", " ").title()

        # Extract keywords from path
        keywords = [p.replace("_", " ").lower() for p in parts]

        registry["topics"][topic_id] = {
            "description": description,
            "vault_path": file_path,
            "keywords": keywords,
            "created": datetime.now().isoformat(),
            "auto_indexed": True
        }
        existing_topics.add(topic_id)
        existing_paths.add(file_path)
        added += 1
        print(f"[REGISTRY] Auto-indexed: {topic_id} -> {file_path}")

    if added > 0:
        write_topic_registry(registry)
        print(f"[REGISTRY] Added {added} existing files to topic registry")

    return added

def get_topic_summary() -> str:
    """Get a summary of registered topics for the LLM."""
    registry = read_topic_registry()
    if not registry.get("topics"):
        return "No topics registered yet."

    lines = []
    for topic_id, info in registry["topics"].items():
        keywords = ", ".join(info.get("keywords", [])[:5])
        lines.append(f"- {topic_id}: {info.get('description', 'No description')} (path: {info.get('vault_path')}, keywords: {keywords})")
    return "\n".join(lines)

# ============================================================================
# Tool Implementations
# ============================================================================

from tool_definitions import tools_schema

async def list_files_tool(directory: str = None) -> str:
    """List files in the vault."""
    files = get_existing_vault_files()
    if directory:
        files = [f for f in files if f.startswith(directory)]
    if not files:
        return "No files found."
    return "\n".join(f"- {f}" for f in files[:100])

async def read_file_tool(file_path: str) -> str:
    """Read a vault file."""
    content = read_vault_file(file_path)
    if not content:
        return f"File not found: {file_path}"
    return content

async def update_file_tool(file_path: str, content: str, instructions: str = "") -> str:
    """Update a vault file using smart merge."""
    existing = read_vault_file(file_path)
    if not existing:
        write_vault_file(file_path, f"# {file_path}\n\n{content}")
        return f"Created new file: {file_path}"
    
    # Merge with instructions context
    merge_info = f"{content}\n\nInstructions: {instructions}"
    new_content = await merge_and_update_file(file_path, existing, merge_info)
    write_vault_file(file_path, new_content)
    return f"Updated file: {file_path}"

async def search_files_tool(query: str) -> str:
    """Search files using chroma or simple keyword search."""
    # Using existing chroma search function
    context = await get_semantic_context(query)
    return context if context else "No matching information found."

async def web_search_tool(query: str) -> str:
    """Perform a web search using the online model variant."""
    try:
        response = await client.chat.completions.create(
            model=SEARCH_MODEL,
            messages=[{"role": "user", "content": query}],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Web search failed: {str(e)}"

async def get_weather_tool(location: str, days: int = 3, hourly: bool = False) -> str:
    """Get weather from Open-Meteo API (free, no API key required)."""
    import urllib.parse

    days = max(1, min(7, days))  # Clamp to 1-7

    def fetch_weather():
        # Check if location is coordinates (lat,lon format)
        if ',' in location and all(part.replace('.', '').replace('-', '').isdigit()
                                    for part in location.split(',')):
            parts = location.split(',')
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            name = f"{lat}, {lon}"
        else:
            # Geocode the city name using Open-Meteo geocoding API
            ctx = ssl.create_default_context()
            
            # First attempt: search for the exact string
            encoded_location = urllib.parse.quote(location)
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_location}&count=10"
            
            with urllib.request.urlopen(geo_url, timeout=10, context=ctx) as resp:
                geo_data = json.loads(resp.read().decode())

            results = geo_data.get("results", [])

            # Second attempt: If no results and comma exists, split and search for city, then filter
            if not results and "," in location:
                parts = [p.strip() for p in location.split(",")]
                city_name = parts[0]
                context_parts = parts[1:]
                
                encoded_city = urllib.parse.quote(city_name)
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={encoded_city}&count=10"
                with urllib.request.urlopen(geo_url, timeout=10, context=ctx) as resp:
                    geo_data = json.loads(resp.read().decode())
                
                candidates = geo_data.get("results", [])
                
                # Filter candidates based on context (state/country)
                for candidate in candidates:
                    candidate_values = [
                        candidate.get("admin1", "").lower(),
                        candidate.get("country", "").lower(),
                        candidate.get("admin2", "").lower()
                    ]
                    # Check if any context part matches any candidate value
                    for ctx_part in context_parts:
                        if any(ctx_part.lower() in val for val in candidate_values if val):
                            results = [candidate]
                            break
                    if results:
                        break
                
                # If still no results after filtering, but we found candidates for the city, use the first one
                if not results and candidates:
                    results = [candidates[0]]

            if not results:
                return f"Location not found: {location}"

            result = results[0]
            lat = result["latitude"]
            lon = result["longitude"]
            name = result.get("name", location)
            country = result.get("country", "")
            state = result.get("admin1", "")
            
            display_parts = [name]
            if state: display_parts.append(state)
            if country: display_parts.append(country)
            name = ", ".join(display_parts)

        # Fetch weather data
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
            f"precipitation,weather_code,wind_speed_10m,wind_direction_10m"
            f"&daily=weather_code,temperature_2m_max,temperature_2m_min,"
            f"precipitation_probability_max,precipitation_sum"
            f"&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
            f"&timezone=auto&forecast_days={days}"
        )

        if hourly:
            weather_url += "&hourly=temperature_2m,weather_code,precipitation_probability"

        with urllib.request.urlopen(weather_url, timeout=10, context=ctx) as resp:
            weather = json.loads(resp.read().decode())

        # Weather code descriptions
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }

        current = weather.get("current", {})
        daily = weather.get("daily", {})
        hourly_data = weather.get("hourly", {})
        units = weather.get("current_units", {})

        # Format current weather
        code = current.get("weather_code", 0)
        condition = weather_codes.get(code, "Unknown")

        output = [f"Weather for {name}:", ""]
        output.append("CURRENT CONDITIONS:")
        output.append(f"  {condition}")
        output.append(f"  Temperature: {current.get('temperature_2m')}°{units.get('temperature_2m', 'F')}")
        output.append(f"  Feels like: {current.get('apparent_temperature')}°{units.get('apparent_temperature', 'F')}")
        output.append(f"  Humidity: {current.get('relative_humidity_2m')}%")
        output.append(f"  Wind: {current.get('wind_speed_10m')} {units.get('wind_speed_10m', 'mph')}")

        if current.get('precipitation', 0) > 0:
            output.append(f"  Precipitation: {current.get('precipitation')} {units.get('precipitation', 'inch')}")

        # Format forecast
        if daily.get("time"):
            output.append("")
            output.append(f"FORECAST ({days} days):")
            for i, date in enumerate(daily["time"]):
                code = daily["weather_code"][i] if daily.get("weather_code") else 0
                condition = weather_codes.get(code, "Unknown")
                high = daily.get("temperature_2m_max", [None])[i]
                low = daily.get("temperature_2m_min", [None])[i]
                precip_prob = daily.get("precipitation_probability_max", [None])[i]

                line = f"  {date}: {condition}, High {high}°, Low {low}°"
                if precip_prob is not None and precip_prob > 0:
                    line += f", {precip_prob}% chance of precipitation"
                output.append(line)

        # Format hourly if requested
        if hourly and hourly_data.get("time"):
            output.append("")
            output.append(f"HOURLY FORECAST ({days} days):")
            
            current_date = None
            
            # Find closest start time or default to beginning
            current_time_str = current.get("time") 
            try:
                start_idx = hourly_data["time"].index(current_time_str)
            except (ValueError, KeyError):
                start_idx = 0
            
            # Iterate through all available data from start_idx to the end
            # Step by 3 to keep output concise (every 3 hours)
            for i in range(start_idx, len(hourly_data["time"]), 3):
                full_time_str = hourly_data["time"][i]
                date_part, time_part = full_time_str.split("T")
                
                # Print date header when date changes
                if date_part != current_date:
                    output.append(f"  [{date_part}]")
                    current_date = date_part
                
                temp = hourly_data["temperature_2m"][i]
                code = hourly_data["weather_code"][i]
                prob = hourly_data["precipitation_probability"][i]
                cond = weather_codes.get(code, "Unknown")
                
                output.append(f"    {time_part}: {temp}°, {cond} ({prob}% precip)")

        return "\n".join(output)

    try:
        # Run blocking HTTP calls in thread pool
        result = await asyncio.get_event_loop().run_in_executor(None, fetch_weather)
        return result
    except Exception as e:
        return f"Weather lookup failed: {str(e)}"

async def log_extraction_to_context(extraction: Dict):
    """Log a summary of what was learned to recent_context.md."""
    if not extraction:
        return

    subject = extraction.get("subject", "Info")
    topic_id = extraction.get("topic_id", "general")
    timestamp = datetime.now().strftime('%H:%M')
    
    log_entry = f"\n- [{timestamp}] Learned new info about {subject} ({topic_id})"
    
    # Append to recent_context.md
    context_path = "meta/recent_context.md"
    existing = read_vault_file(context_path)
    
    if "## Today's Activity" not in existing:
         existing += f"\n\n## Today's Activity{log_entry}"
    else:
        # Insert into the section
        parts = existing.split("## Today's Activity")
        existing = parts[0] + "## Today's Activity" + parts[1] + log_entry
        
    write_vault_file(context_path, existing)

# ============================================================================
# iCal Calendar Integration (with caching)
# ============================================================================

# Calendar cache to avoid repeated fetches
_calendar_cache = {
    "events": [],
    "last_fetch": None,
    "calendar_text_7day": "",
    "calendar_md_content": ""
}
CALENDAR_CACHE_MINUTES = 10  # Minimum time between iCal fetches

# Track which conversations have already been greeted (by thread ID)
_greeted_threads: set = set()


def _calendar_needs_refresh() -> bool:
    """Check if calendar cache is stale without fetching."""
    if not ICAL_URL:
        return False
    if not _calendar_cache["last_fetch"]:
        return True
    age = datetime.now() - _calendar_cache["last_fetch"]
    return age.total_seconds() >= CALENDAR_CACHE_MINUTES * 60


def _parse_ical_data(ical_data: str) -> List[Dict]:
    """Parse iCal data string and return list of events."""
    events = []
    current_event = {}
    in_event = False

    for line in ical_data.split('\n'):
        line = line.strip()
        if line == 'BEGIN:VEVENT':
            in_event = True
            current_event = {}
        elif line == 'END:VEVENT':
            in_event = False
            if current_event.get('start'):
                events.append(current_event)
            current_event = {}
        elif in_event:
            if line.startswith('SUMMARY:'):
                current_event['summary'] = line[8:]
            elif line.startswith('DTSTART'):
                # Handle various date/time formats with timezone support
                try:
                    target_tz = get_local_timezone()

                    # Extract TZID if present
                    event_tz = None
                    if ';TZID=' in line:
                        tzid_part = line.split(';TZID=')[1].split(':')[0]
                        try:
                            event_tz = ZoneInfo(tzid_part)
                        except Exception:
                            pass

                    date_str = line.split(':')[-1]

                    if 'T' in date_str:
                        is_utc = date_str.endswith('Z')
                        dt_str = date_str[:15]
                        dt = datetime.strptime(dt_str, '%Y%m%dT%H%M%S')

                        if is_utc:
                            dt = dt.replace(tzinfo=timezone.utc)
                            dt = dt.astimezone(target_tz)
                        elif event_tz:
                            dt = dt.replace(tzinfo=event_tz)
                            dt = dt.astimezone(target_tz)
                        else:
                            dt = dt.replace(tzinfo=target_tz)

                        current_event['start'] = dt
                    else:
                        dt = datetime.strptime(date_str[:8], '%Y%m%d')
                        dt = dt.replace(tzinfo=target_tz)
                        current_event['start'] = dt
                        current_event['all_day'] = True
                except Exception as e:
                    print(f"[CALENDAR] Failed to parse date: {line}, error: {e}")
            elif line.startswith('LOCATION:'):
                current_event['location'] = line[9:]

    return events


def _fetch_ical_if_needed(force: bool = False) -> bool:
    """Fetch iCal data if cache is stale. Returns True if fetch occurred."""
    global _calendar_cache

    if not ICAL_URL:
        return False

    # Check if cache is still valid
    if not force and _calendar_cache["last_fetch"]:
        age = datetime.now() - _calendar_cache["last_fetch"]
        if age.total_seconds() < CALENDAR_CACHE_MINUTES * 60:
            return False  # Cache still valid

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(ICAL_URL, timeout=10, context=ctx) as response:
            ical_data = response.read().decode('utf-8')

        events = _parse_ical_data(ical_data)
        today = datetime.now().date()

        # Filter to future events
        future_events = [
            e for e in events
            if e.get('start') and e['start'].date() >= today
        ]
        future_events.sort(key=lambda x: x['start'])

        # Build 7-day summary text
        week_ahead = today + timedelta(days=7)
        upcoming = [e for e in future_events if e['start'].date() <= week_ahead]

        calendar_text = ""
        if upcoming:
            calendar_text = "## Calendar (Next 7 Days)\n"
            current_date = None
            for event in upcoming:
                event_date = event['start'].date()
                if event_date != current_date:
                    current_date = event_date
                    day_name = event['start'].strftime('%A, %B %d')
                    calendar_text += f"\n**{day_name}**\n"
                time_str = "All day" if event.get('all_day') else event['start'].strftime('%I:%M %p')
                summary = event.get('summary', 'Untitled')
                location = f" @ {event.get('location')}" if event.get('location') else ""
                calendar_text += f"- {time_str}: {summary}{location}\n"

        # Build full calendar.md content
        md_content = ""
        if future_events:
            md_content = f"# Calendar Events\n\nLast updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            current_month = None
            current_date = None
            for event in future_events:
                event_date = event['start'].date()
                month_key = event['start'].strftime('%B %Y')
                if month_key != current_month:
                    current_month = month_key
                    md_content += f"\n## {month_key}\n"
                    current_date = None
                if event_date != current_date:
                    current_date = event_date
                    day_name = event['start'].strftime('%A, %B %d')
                    md_content += f"\n### {day_name}\n"
                time_str = "All day" if event.get('all_day') else event['start'].strftime('%I:%M %p')
                summary = event.get('summary', 'Untitled')
                location = f" @ {event.get('location')}" if event.get('location') else ""
                md_content += f"- {time_str}: {summary}{location}\n"

        # Update cache
        _calendar_cache = {
            "events": future_events,
            "last_fetch": datetime.now(),
            "calendar_text_7day": calendar_text,
            "calendar_md_content": md_content
        }

        print(f"[CALENDAR] Fetched and cached {len(future_events)} events")
        return True

    except Exception as e:
        print(f"[CALENDAR] Fetch error: {e}")
        return False


def fetch_calendar_events() -> str:
    """Get 7-day calendar summary from cache (fetches if needed)."""
    _fetch_ical_if_needed()
    return _calendar_cache["calendar_text_7day"]


def save_calendar_to_vault():
    """Save calendar.md to vault from cache (does not trigger fetch)."""
    if _calendar_cache["calendar_md_content"]:
        write_vault_file("schedules/calendar.md", _calendar_cache["calendar_md_content"])
        print(f"[CALENDAR] Saved {len(_calendar_cache['events'])} events to schedules/calendar.md")


async def index_calendar_async():
    """Index calendar to ChromaDB asynchronously (non-blocking)."""
    try:
        # Delete old calendar entries first
        collection = get_or_create_collection()
        try:
            # Get all calendar event IDs and delete them
            results = collection.get(where={"category": "calendar"})
            if results and results.get("ids"):
                collection.delete(ids=results["ids"])
                print(f"[CALENDAR] Removed {len(results['ids'])} old calendar entries from ChromaDB")
        except Exception as e:
            print(f"[CALENDAR] Error cleaning old entries: {e}")

        # Index calendar.md as chunks (not individual events)
        content = _calendar_cache["calendar_md_content"]
        if not content:
            return

        # Chunk the calendar content (1000 chars, 200 overlap)
        chunk_size = 1000
        overlap = 200
        chunks = []
        for i in range(0, len(content), chunk_size - overlap):
            chunk = content[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)

        # Index each chunk
        for i, chunk in enumerate(chunks):
            doc_id = f"schedules/calendar.md_chunk_{i}"
            index_to_chroma(
                doc_id=doc_id,
                content=chunk,
                metadata={
                    "source": "schedules/calendar.md",
                    "category": "calendar",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )

        print(f"[CALENDAR] Indexed calendar.md as {len(chunks)} chunks to ChromaDB")

    except Exception as e:
        print(f"[CALENDAR] Async indexing error: {e}")


def refresh_calendar_if_needed():
    """Check if calendar needs refresh and trigger async indexing if so."""
    fetched = _fetch_ical_if_needed()
    if fetched:
        # Save file synchronously (fast), index async (slow)
        save_calendar_to_vault()
        # Return the coroutine for async execution
        return index_calendar_async()
    return None

# ============================================================================
# ChromaDB Operations
# ============================================================================

def get_or_create_collection(name: str = "vault_documents"):
    """Get or create a ChromaDB collection."""
    return get_chroma_client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

def search_chroma(query: str, n_results: int = 5) -> List[Dict]:
    """Search ChromaDB for relevant documents."""
    try:
        collection = get_or_create_collection()
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                retrieved.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None
                })
        return retrieved
    except Exception as e:
        print(f"ChromaDB search error: {e}")
        return []

def index_to_chroma(doc_id: str, content: str, metadata: Dict = None):
    """Index a document to ChromaDB."""
    try:
        collection = get_or_create_collection()
        collection.upsert(
            ids=[doc_id],
            documents=[content],
            metadatas=[metadata or {}]
        )
    except Exception as e:
        print(f"ChromaDB indexing error: {e}")

# ============================================================================
# File Upload Processing
# ============================================================================

SUPPORTED_IMAGE_TYPES = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
SUPPORTED_DOC_TYPES = {'.pdf'}

def get_mime_type(file_path: str) -> str:
    """Get MIME type based on file extension."""
    ext = Path(file_path).suffix.lower()
    mime_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.pdf': 'application/pdf',
    }
    return mime_map.get(ext, 'application/octet-stream')

async def process_uploaded_files(elements: List) -> tuple[List[Dict], List[str], List[str]]:
    """
    Process uploaded file elements and convert to base64 for the API.
    Returns a tuple of (content_parts for API, file_descriptions for history, unsupported_files).
    """
    content_parts = []
    file_descriptions = []
    unsupported_files = []

    if not elements:
        return content_parts, file_descriptions, unsupported_files

    for element in elements:
        try:
            file_path = element.path if hasattr(element, 'path') else None
            file_name = element.name if hasattr(element, 'name') else 'unknown'

            if not file_path:
                continue

            ext = Path(file_name).suffix.lower()

            # Check if it's a supported file type
            if ext in SUPPORTED_IMAGE_TYPES:
                # Read and encode image
                with open(file_path, 'rb') as f:
                    file_data = base64.b64encode(f.read()).decode('utf-8')

                mime_type = get_mime_type(file_name)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{file_data}"
                    }
                })
                file_descriptions.append(f"[Uploaded image: {file_name}]")
                print(f"[FILE] Processed image: {file_name}")

            elif ext in SUPPORTED_DOC_TYPES:
                # Read and encode PDF
                with open(file_path, 'rb') as f:
                    file_data = base64.b64encode(f.read()).decode('utf-8')

                mime_type = get_mime_type(file_name)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{file_data}"
                    }
                })
                file_descriptions.append(f"[Uploaded PDF: {file_name}]")
                print(f"[FILE] Processed PDF: {file_name}")
            else:
                print(f"[FILE] Skipping unsupported file type: {file_name}")
                file_descriptions.append(f"[Skipped unsupported file: {file_name}]")
                unsupported_files.append(file_name)

        except Exception as e:
            print(f"[FILE] Error processing {element}: {e}")
            unsupported_files.append(f"{file_name} (Error)")

    return content_parts, file_descriptions, unsupported_files

def build_multimodal_content(text: str, file_parts: List[Dict]) -> Union[str, List[Dict]]:
    """
    Build content for API message, either as plain text or multimodal array.
    """
    if not file_parts:
        return text

    # Build multimodal content array
    content = []
    if text:
        content.append({"type": "text", "text": text})
    content.extend(file_parts)
    return content

# ============================================================================
# Context Building
# ============================================================================

def build_system_context(learned_behaviors: str = None, daily_prep: str = None) -> str:
    """Build the system prompt from vault files.

    Args:
        learned_behaviors: Pre-loaded learned behaviors content (avoids duplicate read)
        daily_prep: Pre-loaded daily prep content (avoids duplicate read)
    """

    # Load core instructions
    instructions = read_vault_file("meta/instructions.md")
    if not instructions:
        instructions = """You are a helpful personal AI assistant.
You help manage family information, schedules, notes, and daily tasks.
Be warm, helpful, and remember important details about the family."""

    # Load learned behaviors (use passed value or read from file)
    if learned_behaviors is None:
        learned_behaviors = read_vault_file("meta/learned_behaviors.md")

    # Load recent context
    recent_context = read_vault_file("meta/recent_context.md")

    # Load daily prep (use passed value or read from file)
    if daily_prep is None:
        daily_prep = read_vault_file("meta/daily_prep.md")

    # Load preferences
    preferences = read_vault_file("meta/preferences.md")

    # Get calendar events from cache (already fetched if needed during on_chat_start)
    calendar_events = _calendar_cache["calendar_text_7day"]

    # Build full system prompt
    system_prompt = f"""# Your Core Instructions
{instructions}

# Your Learned Behaviors (You can update this!)
{learned_behaviors if learned_behaviors else "No learned behaviors yet."}

# Recent Context
{recent_context if recent_context else "No recent context available."}

# Today's Preparation
{daily_prep if daily_prep else "No specific preparation for today."}

# User Preferences
{preferences if preferences else "No specific preferences recorded yet."}

{calendar_events}

# Current Date/Time
{datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")}

# Your Capabilities
You can:
- Search your knowledge vault for relevant information
- Save important information to appropriate vault files
- **Correct or delete incorrect information** when the user identifies it as a mistake
- Remember family members, schedules, preferences
- Track health, school, and general notes
- Help with meal planning, workouts, and daily tasks
- View the family calendar
- **Update your own learned behaviors** when you discover patterns or preferences

# Proactive Behavior
Be proactively helpful:
- If you notice something relevant from your context, bring it up naturally
- If you have follow-up items or questions from previous conversations, ask about them
- If today's calendar shows relevant events, mention them when appropriate
- If you learn a new preference or pattern, mention that you'll remember it

# Self-Improvement
When you notice patterns in how the user likes things done, or learn new rules to follow:
- You can update your "learned_behaviors" to remember these patterns
- Include follow-up items you want to ask about later
- Track communication preferences
- Note routine patterns you observe

When you learn something important, naturally acknowledge it. Be warm and conversational.
"""

    # Add topic summary (knowledge index)
    topic_summary = get_topic_summary()
    system_prompt += f"\n# Knowledge Index (Topics you know about)\n{topic_summary}\n"

    return system_prompt

async def get_semantic_context(user_message: str) -> str:
    """Retrieve semantically relevant context from ChromaDB."""
    results = search_chroma(user_message, n_results=5)

    if not results:
        return ""

    context_parts = ["# Relevant Information from Your Knowledge Vault:"]
    for result in results:
        # Only include if reasonably relevant (distance < 1.0 for cosine)
        if result["distance"] is not None and result["distance"] < 1.0:
            source = result["metadata"].get("source", "notes")
            context_parts.append(f"\n## From {source}:\n{result['content']}")

    return "\n".join(context_parts) if len(context_parts) > 1 else ""

# ============================================================================
# Topic Resolution
# ============================================================================

TOPIC_RESOLUTION_PROMPT = """Analyze this conversation and determine where information should be stored or modified.

## Registered Topics (use these paths if the conversation matches):
{topic_summary}

## Existing Vault Files (for reference):
{existing_files}

## Conversation:
User: {user_message}
Assistant: {assistant_response}

## Your Task:
1. Does this conversation contain information worth saving or MODIFYING? If the user provides new facts, CORRECTS existing information, or asks to DELETE something, respond with {{"should_save": true}}. If it's just chitchat or simple Q&A with no new or corrective info, respond with {{"should_save": false}}.

2. If worth saving/modifying, does it relate to an EXISTING registered topic above or an EXISTING vault file?
   - If yes: use that topic's vault_path or the existing file path.
   - If no: create a NEW topic with a sensible path.

3. For NEW topics, follow these conventions:
   - Family member info → family/members/[name].md
   - Family events/trips → family/events/[event_name].md
   - Health/medical → notes/health/[topic].md
   - School info → notes/school/[child_name].md
   - Travel/vacations → notes/travel/[destination_year].md
   - Food/dining preferences → notes/food/[topic].md
   - Recipes → recipes/[recipe_name].md
   - Workouts → workouts/[name].md
   - General topics → notes/[category]/[topic].md

Respond with JSON only:
{{
    "should_save": true/false,
    "is_new_topic": true/false,
    "topic_id": "snake_case_identifier",
    "topic_description": "Brief description of what this topic covers",
    "vault_path": "path/to/file.md",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "reason": "Brief explanation of your decision"
}}

If should_save is false, only include: {{"should_save": false, "reason": "why not saving"}}
"""

async def resolve_topic(messages: List[Dict]) -> Optional[Dict]:
    """Resolve which topic/path this conversation belongs to, based on recent context."""
    try:
        # Get context for the resolution
        topic_summary = get_topic_summary()
        existing_files = get_existing_vault_files()
        existing_files_str = "\n".join(f"- {f}" for f in existing_files[:50])  # Limit to 50 files

        if not existing_files_str:
            existing_files_str = "No existing files yet."

        # Format last few messages for context (last 6 messages)
        recent_msgs = messages[-6:] if len(messages) > 6 else messages
        conversation_str = ""
        for msg in recent_msgs:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg.get("content") or ""
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "... (truncated)"
            conversation_str += f"{role}: {content}\n"

        print(f"[TOPIC] Resolving topic for recent context...")

        response = await client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "You are a knowledge organization assistant. Respond with valid JSON only. No markdown, no code blocks, just raw JSON."},
                {"role": "user", "content": TOPIC_RESOLUTION_PROMPT.format(
                    topic_summary=topic_summary,
                    existing_files=existing_files_str,
                    user_message="[See Conversation Context]",
                    assistant_response="[See Conversation Context]"
                ).replace("Conversation:\nUser: {user_message}\nAssistant: {assistant_response}", f"Conversation (Last 6 messages):\n{conversation_str}")}
            ],
            temperature=0.1,
            max_tokens=500,
            extra_body={"reasoning_effort": "low"}
        )

        result_text = response.choices[0].message.content
        print(f"[TOPIC] Raw response: {result_text[:200]}")

        # Clean up potential markdown formatting
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result_text = result_text.strip()

        # Find JSON object if not at start
        if not result_text.startswith("{"):
            start_idx = result_text.find("{")
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(result_text[start_idx:], start_idx):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                result_text = result_text[start_idx:end_idx]

        parsed = json.loads(result_text)
        print(f"[TOPIC] Resolved: should_save={parsed.get('should_save')}, topic={parsed.get('topic_id')}, path={parsed.get('vault_path')}")

        # Register topic if not already in registry (regardless of is_new_topic flag)
        if parsed.get("should_save") and parsed.get("topic_id"):
            register_topic(
                topic_id=parsed.get("topic_id", "unknown"),
                description=parsed.get("topic_description", ""),
                vault_path=parsed.get("vault_path", "notes/general/misc.md"),
                keywords=parsed.get("keywords", [])
            )

        return parsed

    except Exception as e:
        print(f"[TOPIC] Error resolving topic: {e}")
        import traceback
        print(f"[TOPIC] Traceback: {traceback.format_exc()}")
        return None

# ============================================================================
# Learning and Information Extraction
# ============================================================================

MERGE_PROMPT = """You are a knowledge curator. Your goal is to merge new information into an existing markdown file, ensuring it remains concise, organized, and free of duplicates.

## Task
Merge the "New Information" into the "Current File Content".
- **DELETION/REMOVAL:** If the new information indicates that a fact is incorrect, should be removed, or is a mistake, DELETE it from the file. Do not just add a note saying it was removed; actually remove the text.
- **UPDATE:** Update existing facts if the new info is more recent or corrective.
- **ADDITION:** Add new facts naturally into relevant sections.
- **DEDUPLICATION:** Remove duplicate information.
- **STRUCTURE:** Maintain the existing markdown structure (headers, lists).
- If the file is empty, just format the new information nicely.

## Current File Content:
{current_content}

## New Information:
{new_info}

## Output:
Return ONLY the updated markdown content for the file. Do not add ```markdown blocks or conversational text.
"""

async def merge_and_update_file(vault_path: str, current_content: str, new_info: str) -> str:
    """Merge new info into existing content using the LLM to deduplicate and organize."""
    try:
        response = await client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful knowledge assistant. Output only the updated file content."},
                {"role": "user", "content": MERGE_PROMPT.format(
                    current_content=current_content,
                    new_info=new_info
                )}
            ],
            temperature=0.1,
            max_tokens=2000,
            extra_body={"reasoning_effort": "low"}
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[MERGE] Error merging content: {e}")
        # Fallback to append if merge fails
        return f"{current_content}\n\n## Update\n{new_info}"

EXTRACTION_PROMPT = """Extract the key information from this conversation to save to the vault.

## Target File: {vault_path}
## Topic: {topic_description}

## What to Extract:
- **NEW FACTS:** Key facts, dates, details, preferences mentioned.
- **CORRECTIONS/DELETIONS:** If the user identifies information as incorrect or asks to delete something, clearly state what should be REMOVED or CORRECTED.
- **DETAILS:** Names, places, times, costs if relevant.
- **DECISIONS:** Decisions made or plans confirmed.
- **ACTION ITEMS:** Any actionable items or things to remember.

## Self-Improvement (optional):
If you notice patterns in how the user communicates or preferences for how things should be done, note them.

Respond with JSON:
{{
    "information": "The key information to save, correct, or delete (formatted as markdown). For deletions, clearly state 'Remove the information about [X]'.",
    "subject": "Brief subject line for this entry",
    "behavior_updates": {{
        "should_update": true/false,
        "section": "communication_preferences|routine_patterns|follow_up_items|custom_rules",
        "content": "what to add to learned behaviors (only if there's a new pattern/preference)"
    }}
}}

Conversation:
User: {user_message}
Assistant: {assistant_response}
"""

async def extract_learnings(messages: List[Dict], topic_info: Dict) -> Optional[Dict]:
    """Extract important information from a conversation exchange using resolved topic and context."""
    result_text = None
    try:
        vault_path = topic_info.get("vault_path", "notes/general/misc.md")
        topic_description = topic_info.get("topic_description", "General notes")

        # Format context (last 10 messages)
        recent_msgs = messages[-10:] if len(messages) > 10 else messages
        conversation_str = ""
        for msg in recent_msgs:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg.get("content") or ""
            # Truncate very long messages
            if len(content) > 1000:
                content = content[:1000] + "... (truncated)"
            conversation_str += f"{role}: {content}\n"

        print(f"[EXTRACT] Extracting for topic: {topic_info.get('topic_id')} -> {vault_path}")

        response = await client.chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[
                {"role": "system", "content": "You are an information extraction assistant. Respond with valid JSON only. No markdown, no code blocks, just raw JSON."},
                {"role": "user", "content": EXTRACTION_PROMPT.format(
                    vault_path=vault_path,
                    topic_description=topic_description,
                    user_message="[See Context]",
                    assistant_response="[See Context]"
                ).replace("Conversation:\nUser: {user_message}\nAssistant: {assistant_response}", f"Conversation (Last 10 messages):\n{conversation_str}")}
            ],
            temperature=0.1,
            max_tokens=1500,
            extra_body={"reasoning_effort": "low"}
        )

        result_text = response.choices[0].message.content
        print(f"[EXTRACT] Raw response: {repr(result_text[:200])}")

        # Clean up potential markdown formatting
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        result_text = result_text.strip()

        # Find JSON object if not at start
        if not result_text.startswith("{"):
            start_idx = result_text.find("{")
            if start_idx != -1:
                brace_count = 0
                end_idx = start_idx
                for i, char in enumerate(result_text[start_idx:], start_idx):
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                result_text = result_text[start_idx:end_idx]
            else:
                print("[EXTRACT] No JSON object found in response!")
                return None

        parsed = json.loads(result_text)

        # Add the vault_path from topic resolution
        parsed["vault_path"] = vault_path
        parsed["topic_id"] = topic_info.get("topic_id")

        print(f"[EXTRACT] Extracted info for {vault_path}: {parsed.get('subject', 'no subject')}")
        return parsed

    except json.JSONDecodeError as e:
        print(f"[EXTRACT] JSON decode error: {e}")
        if result_text:
            print(f"[EXTRACT] Failed text: {repr(result_text[:500])}")
        return None
    except Exception as e:
        print(f"[EXTRACT] Exception: {type(e).__name__}: {e}")
        import traceback
        print(f"[EXTRACT] Traceback: {traceback.format_exc()}")
        return None

async def save_learnings(extraction: Dict):
    """Save extracted learnings to the vault and index to ChromaDB."""
    if not extraction:
        return

    vault_path = extraction.get("vault_path")
    subject = extraction.get("subject", "Note")
    info = extraction.get("information", "")
    topic_id = extraction.get("topic_id", "unknown")

    if not info or not vault_path:
        print("[SAVE] No information or vault_path to save")
        return

    # Check if file exists
    existing = read_vault_file(vault_path)

    if existing:
        print(f"[SAVE] Merging new info into existing file: {vault_path}")
        # Smart merge: Read + Update + Write
        new_content = await merge_and_update_file(vault_path, existing, info)
        write_vault_file(vault_path, new_content)
    else:
        print(f"[SAVE] Creating new file: {vault_path}")
        new_content = f"# {subject}\n\n{info}\n"
        write_vault_file(vault_path, new_content)

    # Index to ChromaDB
    doc_id = f"{vault_path}_{datetime.now().timestamp()}"
    index_to_chroma(
        doc_id=doc_id,
        content=info,
        metadata={
            "source": vault_path,
            "topic_id": topic_id,
            "subject": subject,
            "timestamp": datetime.now().isoformat()
        }
    )
    print(f"[SAVE] Saved to {vault_path}: {info[:50]}...")

    # Update learned behaviors if needed
    behavior_updates = extraction.get("behavior_updates", {})
    if behavior_updates.get("should_update"):
        await update_learned_behaviors(
            behavior_updates.get("section", "custom_rules"),
            behavior_updates.get("content", "")
        )

async def update_learned_behaviors(section: str, content: str):
    """Update the learned behaviors file by merging new insights."""
    if not content:
        return

    behaviors_path = "meta/learned_behaviors.md"
    current = read_vault_file(behaviors_path)

    # Use the same smart merge logic
    print(f"[BEHAVIOR] Merging new behavior into {section}...")
    new_info = f"Update section '{section}' with: {content}"
    
    # If file is empty/missing, initialize it
    if not current:
        current = "# Learned Behaviors\n\n## Communication Preferences\n\n## Routine Patterns\n\n## Follow-up Items\n\n## Custom Rules\n"

    new_content = await merge_and_update_file(behaviors_path, current, new_info)
    write_vault_file(behaviors_path, new_content)

# ============================================================================
# Conversation Saving
# ============================================================================

async def save_conversation(messages: List[Dict], session_id: str):
    """Save the conversation to the vault."""
    today = datetime.now().strftime("%Y-%m-%d")
    conv_path = f"conversations/{today}/{session_id}.md"

    content = f"# Conversation - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
    for msg in messages:
        role = msg.get("role", "unknown").title()
        msg_content = msg.get("content", "")
        content += f"**{role}:** {msg_content}\n\n"

    write_vault_file(conv_path, content)

# ============================================================================
# Chainlit Event Handlers
# ============================================================================

def ensure_vault_setup():
    """Ensure essential vault files exist with default content."""
    defaults = {
        "meta/instructions.md": """# Personal Assistant Instructions

You are a personal AI assistant for a busy family. Your role is to be genuinely helpful, remember important details, and make daily life easier.

## Your Personality
- Warm and friendly, like a trusted friend
- Concise but thorough - respect their time
- Proactive when helpful, but not pushy
- Remember you're talking to a real person managing a busy life

## What You Help With

### Family Management
- Remember each family member's name, age, activities, and preferences
- Track kids' school activities, homework, accomplishments
- Note important milestones and events

### Health & Wellness
- Track health information, allergies, medications
- Remember doctor appointments and notes
- Monitor sleep patterns, symptoms, or concerns mentioned
- Support workout routines and fitness goals

### Daily Life
- Help with meal planning based on preferences and dietary needs
- Remember favorite recipes and food preferences
- Track schedules and routines
- Generate to-do lists based on what you know

### Learning & Adapting
- Notice patterns in how the family operates
- Learn preferences without being told explicitly
- Adapt your suggestions based on past conversations
- Remember how they like things done

## How to Learn

When you hear important information:
1. Naturally acknowledge it in conversation
2. It will be automatically saved to your knowledge vault
3. Use it in future conversations when relevant

## Important Guidelines
- Never share family information externally
- Treat health information with extra care
- When unsure, ask clarifying questions
- If something seems urgent or concerning, acknowledge it appropriately
- Be helpful with curriculum-based questions using stored school information

## Daily Rhythm
- In the morning, be ready with relevant reminders
- During the day, help with whatever comes up
- Remember that context from earlier conversations carries forward
""",
        "meta/learned_behaviors.md": """# Learned Behaviors

This file is automatically updated by the assistant as it learns patterns and preferences.

## Communication Preferences
*How the family prefers to receive information*

## Routine Patterns
*Regular schedules and habits the assistant has noticed*

## Follow-up Items
*Things to check back on or ask about*

## Custom Rules
*Specific rules the assistant has learned to follow*
""",
        "meta/recent_context.md": "# Recent Context\n\n*Summary of recent conversations and events*\n",
        "meta/daily_prep.md": "# Daily Prep\n\n*Reminders and context for the upcoming day*\n",
        "meta/preferences.md": "# User Preferences\n\n*Preferences will be learned automatically from conversations.*\n",
    }
    for path, content in defaults.items():
        file_path = VAULT_PATH / path
        if not file_path.exists():
            print(f"[INIT] Creating default vault file: {path}")
            write_vault_file(path, content)

    # Initialize topic registry from existing vault files
    initialize_topic_registry()

@cl.on_chat_start
async def on_chat_start():
    """Initialize chat session with context from vault."""
    global _greeted_threads

    # Ensure vault is ready
    ensure_vault_setup()

    # Get thread ID to track if we've already greeted this conversation
    thread_id = cl.context.session.thread_id

    # Generate session ID
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("messages", [])

    # Refresh calendar in background if cache is stale
    # We do NOT await this, so the UI loads instantly.
    if _calendar_needs_refresh():
        # Fire and forget - runs in background
        asyncio.create_task(asyncio.to_thread(refresh_calendar_if_needed))

    # Fire off ChromaDB indexing in background as well
    # (Note: refresh_calendar_if_needed handles its own re-indexing, but if we didn't call it, we might need to check vault index)
    # The original code had logic for `index_task`, but simplistic fire-and-forget is better for UX here.

    # Read files once for both system context and welcome message
    learned = read_vault_file("meta/learned_behaviors.md")
    daily_prep = read_vault_file("meta/daily_prep.md")

    # Build and store system context (pass pre-loaded files to avoid duplicate reads)
    system_context = build_system_context(learned_behaviors=learned, daily_prep=daily_prep)
    cl.user_session.set("system_context", system_context)

    # Skip greeting if this conversation was already greeted (e.g., tab switch reconnection)
    if thread_id in _greeted_threads:
        return

    # Mark this conversation as greeted
    _greeted_threads.add(thread_id)

    # Use cached calendar for welcome message
    calendar = _calendar_cache["calendar_text_7day"]

    # Build a proactive welcome message
    welcome_parts = []

    # Time-based greeting
    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good morning"
    elif hour < 17:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    welcome_parts.append(f"{greeting}! I'm ready to help.")

    # Check for follow-up items
    if learned and "## Follow-up Items" in learned:
        follow_section = learned.split("## Follow-up Items")[1]
        next_section = follow_section.find("\n## ")
        if next_section != -1:
            follow_section = follow_section[:next_section]
        if follow_section.strip() and "-" in follow_section:
            welcome_parts.append("\n\nI have some follow-up items from before - feel free to ask me about them!")

    # Mention calendar if there are events today
    if calendar and datetime.now().strftime('%A, %B %d') in calendar:
        welcome_parts.append("\n\nI can see you have events on the calendar today if you'd like to review them.")

    welcome = " ".join(welcome_parts)
    await cl.Message(content=welcome).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Process incoming messages with tool support."""

    # Available tools mapping
    available_tools = {
        "list_files": list_files_tool,
        "read_file": read_file_tool,
        "update_file": update_file_tool,
        "search_files": search_files_tool,
        "web_search": web_search_tool,
        "get_weather": get_weather_tool
    }

    # Get session data
    messages = cl.user_session.get("messages", [])
    system_context = cl.user_session.get("system_context", "")
    session_id = cl.user_session.get("session_id", "unknown")

    # Process any uploaded files
    file_parts, file_descriptions, unsupported_files = await process_uploaded_files(message.elements)

    # Notify user about unsupported files
    if unsupported_files:
        warning_msg = f"⚠️ I cannot process the following files (unsupported type): {', '.join(unsupported_files)}"
        await cl.Message(content=warning_msg).send()
        return

    # Get semantic context from ChromaDB
    semantic_context = await get_semantic_context(message.content)

    # Build full system prompt with semantic context
    full_system_prompt = system_context
    if semantic_context:
        full_system_prompt += f"\n\n{semantic_context}"

    # Build the user message content (multimodal if files attached)
    user_content = build_multimodal_content(message.content, file_parts)

    # For conversation history, store text version with file descriptions
    history_content = message.content
    if file_descriptions:
        history_content = f"{message.content}\n{' '.join(file_descriptions)}"

    # Add user message to history
    messages.append({"role": "user", "content": user_content})

    # Prepare for response
    response_message = cl.Message(content="")
    await response_message.send()

    # Tool Loop (Max 3 turns)
    for turn in range(3):
        api_messages = [
            {"role": "system", "content": full_system_prompt}
        ] + messages[-100:]

        full_response = ""
        tool_calls_data = [] # To accumulate chunks
        
        try:
            stream = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=api_messages,
                tools=tools_schema,
                tool_choice="auto",
                stream=True,
                temperature=0.7,
                max_tokens=2000,
                extra_body={"reasoning_effort": "medium"}
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                
                # Handle Content
                if delta.content:
                    full_response += delta.content
                    await response_message.stream_token(delta.content)

                # Handle Tool Calls (Accumulate)
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        if len(tool_calls_data) <= tc.index:
                            tool_calls_data.append({"id": "", "function": {"name": "", "arguments": ""}, "type": "function"})
                        
                        # Append pieces
                        if tc.id: tool_calls_data[tc.index]["id"] += tc.id
                        if tc.function.name: tool_calls_data[tc.index]["function"]["name"] += tc.function.name
                        if tc.function.arguments: tool_calls_data[tc.index]["function"]["arguments"] += tc.function.arguments

            await response_message.update()

            # If we have tool calls, execute them
            if tool_calls_data:
                # Append the assistant's "thinking" (or tool call request) to history
                # Note: OpenRouter/OpenAI expects the assistant message to include the tool_calls field
                assistant_msg = {
                    "role": "assistant", 
                    "content": full_response if full_response else None,
                    "tool_calls": tool_calls_data
                }
                messages.append(assistant_msg)
                
                for tool in tool_calls_data:
                    func_name = tool["function"]["name"]
                    func_args_str = tool["function"]["arguments"]
                    call_id = tool["id"]
                    
                    try:
                        args = json.loads(func_args_str)
                        print(f"[TOOL] Calling {func_name} with {args}")
                        
                        if func_name in available_tools:
                            # Show status update
                            async with cl.Step(name=func_name) as step:
                                step.input = args
                                result = await available_tools[func_name](**args)
                                step.output = result
                        else:
                            result = f"Error: Tool {func_name} not found."
                            
                    except Exception as e:
                        result = f"Error executing tool: {str(e)}"
                        print(f"[TOOL] Error: {e}")

                    # Append Tool Result
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": result,
                        "name": func_name
                    })

                # Loop continues to next turn to generate response based on tool results
                continue
            
            # No tool calls, we are done
            break

        except Exception as e:
            err_msg = f"I encountered an error: {str(e)}"
            response_message.content = err_msg
            await response_message.update()
            full_response = err_msg
            
            # Remove the last message (the one that caused the error) from history
            # so the user can retry without getting stuck.
            if messages and messages[-1]["role"] == "user":
                messages.pop()
                cl.user_session.set("messages", messages)
            
            break

    # Add final response to history
    if not tool_calls_data:
         messages.append({"role": "assistant", "content": full_response})

    cl.user_session.set("messages", messages)

    # Background tasks
    asyncio.create_task(extract_and_save(messages))
    asyncio.create_task(save_conversation(messages, session_id))

async def extract_and_save(messages: List[Dict]):
    """Background task to resolve topic, extract, and save learnings using recent context."""
    if not messages:
        return
        
    # Phase 1: Resolve which topic this belongs to, using the last few messages for context
    topic_info = await resolve_topic(messages)

    if not topic_info or not topic_info.get("should_save"):
        print(f"[EXTRACT] Nothing to save: {topic_info.get('reason', 'no reason given') if topic_info else 'topic resolution failed'}")
        return

    # Phase 2: Extract the actual information using the resolved topic and context
    extraction = await extract_learnings(messages, topic_info)

    if extraction:
        await save_learnings(extraction)
        await log_extraction_to_context(extraction)

@cl.on_chat_end
async def on_chat_end():
    """Handle chat session end."""
    messages = cl.user_session.get("messages", [])
    session_id = cl.user_session.get("session_id", "unknown")

    # Final save of conversation
    if messages:
        await save_conversation(messages, session_id)

@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """Handle resuming a previous chat session."""
    global _greeted_threads

    # Ensure vault is ready
    ensure_vault_setup()

    # Mark this thread as already greeted (don't send welcome message again)
    thread_id = thread.get("id")
    if thread_id:
        _greeted_threads.add(thread_id)

    # Restore messages from thread history
    messages = []
    for step in thread.get("steps", []):
        step_type = step.get("type")
        output = step.get("output", "")

        if step_type == "user_message":
            messages.append({"role": "user", "content": output})
        elif step_type == "assistant_message":
            messages.append({"role": "assistant", "content": output})

    # Generate session ID for this resumed session
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_resumed"
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("messages", messages)

    # Build and store system context
    system_context = build_system_context()
    cl.user_session.set("system_context", system_context)

    print(f"[RESUME] Resumed thread {thread_id} with {len(messages)} messages")
