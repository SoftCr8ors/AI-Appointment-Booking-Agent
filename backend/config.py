"""Central configuration loaded from environment variables."""
import json
import os
from datetime import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _parse_clock(value: str) -> time:
    hour, minute = value.strip().split(":")
    return time(int(hour), int(minute))


# --- Secrets and connections -------------------------------------------------
GEMINI_API_KEY = _require("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CALENDAR_ID = _require("CALENDAR_ID")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Karachi")

# --- Business rules (not secret, all optional) -------------------------------
BUSINESS_START = _parse_clock(os.getenv("BUSINESS_START", "10:00"))
BUSINESS_END = _parse_clock(os.getenv("BUSINESS_END", "18:00"))
SLOT_MINUTES = int(os.getenv("SLOT_MINUTES", "60"))      # duration of each appointment slot in minutes
# 0 = Monday ... 6 = Sunday
WORKING_DAYS = frozenset(
    int(d) for d in os.getenv("WORKING_DAYS", "0,1,2,3,4").split(",") if d.strip()
)
MIN_NOTICE_MINUTES = int(os.getenv("MIN_NOTICE_MINUTES", "60"))
MAX_DAYS_AHEAD = int(os.getenv("MAX_DAYS_AHEAD", "60"))
SERVICES = tuple(
    s.strip().lower()
    for s in os.getenv("SERVICES", "consultation,follow-up,demo").split(",")
    if s.strip()
)

if BUSINESS_END <= BUSINESS_START:
    raise RuntimeError("BUSINESS_END must be after BUSINESS_START")
if SLOT_MINUTES <= 0:
    raise RuntimeError("SLOT_MINUTES must be positive")


def load_service_account_info() -> dict:
    """Production: JSON string in env var. Local: path to a JSON file."""
    raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if raw:
        return json.loads(raw)

    file_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    if file_path:
        path = Path(file_path)
        if not path.is_absolute():
            path = ROOT / path
        return json.loads(path.read_text())

    raise RuntimeError(
        "Set GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_FILE"
    )