"""Thin wrapper around the Google Calendar API.

No AI and no business rules (business hours, slot length) live here.
This module only does: free/busy, create, get, update, delete, find.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import config

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar"]
TZ = ZoneInfo(config.TIMEZONE)
RETRIES = 3  # automatic retry with backoff for 5xx / 429 / network errors


# --------------------------------------------------------------------------
# Errors and data model
# --------------------------------------------------------------------------
class CalendarError(Exception):
    """Any failure while talking to Google Calendar."""


class EventNotFoundError(CalendarError):
    """The event does not exist (or was already deleted)."""


@dataclass(frozen=True)
class CalendarEvent:
    id: str
    title: str
    start: datetime
    end: datetime
    description: str = ""
    private_props: dict[str, str] = field(default_factory=dict)
    link: str = ""


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
_local = threading.local()


def _service():
    """One Google client per thread (the client is not thread-safe)."""
    if not hasattr(_local, "service"):
        creds = service_account.Credentials.from_service_account_info(
            config.load_service_account_info(), scopes=SCOPES
        )
        _local.service = build(
            "calendar", "v3", credentials=creds, cache_discovery=False
        )
    return _local.service


def _require_aware(value: datetime, name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _validate_range(start: datetime, end: datetime) -> None:
    _require_aware(start, "start")
    _require_aware(end, "end")
    if end <= start:
        raise ValueError("end must be after start")


def _parse_dt(obj: dict) -> datetime:
    """Parse Google's start/end object (timed or all-day) into an aware datetime."""
    if "dateTime" in obj:
        return datetime.fromisoformat(obj["dateTime"]).astimezone(TZ)
    return datetime.combine(date.fromisoformat(obj["date"]), time.min, tzinfo=TZ)


def _to_event(raw: dict) -> CalendarEvent:
    return CalendarEvent(
        id=raw["id"],
        title=raw.get("summary", ""),
        start=_parse_dt(raw["start"]),
        end=_parse_dt(raw["end"]),
        description=raw.get("description", ""),
        private_props=raw.get("extendedProperties", {}).get("private", {}),
        link=raw.get("htmlLink", ""),
    )


def _raise_from(exc: HttpError, action: str) -> None:
    status = getattr(exc.resp, "status", None)
    logger.error("Calendar API error during %s: %s", action, exc)
    if status in (404, 410):
        raise EventNotFoundError(f"Event not found ({action})") from exc
    raise CalendarError(f"Calendar API error during {action} (HTTP {status})") from exc


# --------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------
def owner_props(name: str | None = None, email: str | None = None) -> dict[str, str]:
    """Normalized ownership properties stored on (and searched in) events.

    Use the same function when creating and when searching, so matching is
    case-insensitive and ignores extra spaces.
    """
    props: dict[str, str] = {}
    if name:
        props["client_name"] = " ".join(name.split()).lower()
    if email:
        props["client_email"] = email.strip().lower()
    return props


# --------------------------------------------------------------------------
# Free / busy
# --------------------------------------------------------------------------
def get_busy_between(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """Busy periods overlapping [start, end), sorted by start time."""
    _validate_range(start, end)
    body = {
        "timeMin": start.isoformat(),
        "timeMax": end.isoformat(),
        "timeZone": config.TIMEZONE,
        "items": [{"id": config.CALENDAR_ID}],
    }
    try:
        resp = _service().freebusy().query(body=body).execute(num_retries=RETRIES)
    except HttpError as exc:
        _raise_from(exc, "free/busy query")

    calendar = resp["calendars"][config.CALENDAR_ID]
    if calendar.get("errors"):
        raise CalendarError(f"Free/busy returned errors: {calendar['errors']}")

    busy = [
        (
            datetime.fromisoformat(b["start"]).astimezone(TZ),
            datetime.fromisoformat(b["end"]).astimezone(TZ),
        )
        for b in calendar.get("busy", [])
    ]
    return sorted(busy)


def get_busy_slots(day: date) -> list[tuple[datetime, datetime]]:
    """Busy periods for one calendar day in the business timezone."""
    day_start = datetime.combine(day, time.min, tzinfo=TZ)
    day_end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=TZ)
    return get_busy_between(day_start, day_end)


# --------------------------------------------------------------------------
# CRUD
# --------------------------------------------------------------------------
def create_event(
    start: datetime,
    end: datetime,
    title: str,
    description: str = "",
    private_props: dict[str, str] | None = None,
) -> CalendarEvent:
    _validate_range(start, end)
    body: dict = {
        "summary": title,
        "description": description,
        "start": {"dateTime": start.isoformat(), "timeZone": config.TIMEZONE},
        "end": {"dateTime": end.isoformat(), "timeZone": config.TIMEZONE},
    }
    if private_props:
        body["extendedProperties"] = {
            "private": {k: str(v) for k, v in private_props.items()}
        }
    try:
        raw = (
            _service()
            .events()
            .insert(calendarId=config.CALENDAR_ID, body=body)
            .execute(num_retries=RETRIES)
        )
    except HttpError as exc:
        _raise_from(exc, "create_event")
    logger.info("Created event %s", raw["id"])
    return _to_event(raw)


def get_event(event_id: str) -> CalendarEvent:
    try:
        raw = (
            _service()
            .events()
            .get(calendarId=config.CALENDAR_ID, eventId=event_id)
            .execute(num_retries=RETRIES)
        )
    except HttpError as exc:
        _raise_from(exc, "get_event")
    if raw.get("status") == "cancelled":
        raise EventNotFoundError("Event was cancelled")
    return _to_event(raw)


def update_event(
    event_id: str, new_start: datetime, new_end: datetime
) -> CalendarEvent:
    """Move an existing event to a new time (title and properties stay)."""
    _validate_range(new_start, new_end)
    body = {
        "start": {"dateTime": new_start.isoformat(), "timeZone": config.TIMEZONE},
        "end": {"dateTime": new_end.isoformat(), "timeZone": config.TIMEZONE},
    }
    try:
        raw = (
            _service()
            .events()
            .patch(calendarId=config.CALENDAR_ID, eventId=event_id, body=body)
            .execute(num_retries=RETRIES)
        )
    except HttpError as exc:
        _raise_from(exc, "update_event")
    logger.info("Updated event %s", event_id)
    return _to_event(raw)


def delete_event(event_id: str) -> None:
    try:
        (
            _service()
            .events()
            .delete(calendarId=config.CALENDAR_ID, eventId=event_id)
            .execute(num_retries=RETRIES)
        )
    except HttpError as exc:
        _raise_from(exc, "delete_event")
    logger.info("Deleted event %s", event_id)


def find_events(
    name: str | None = None,
    email: str | None = None,
    include_past: bool = False,
    max_results: int = 50,
) -> list[CalendarEvent]:
    """Find events owned by this name and/or email (both must match if given)."""
    props = owner_props(name, email)
    if not props:
        raise ValueError("Provide at least a name or an email")

    params: dict = {
        "calendarId": config.CALENDAR_ID,
        "privateExtendedProperty": [f"{k}={v}" for k, v in props.items()],
        "singleEvents": True,
        "orderBy": "startTime",
        "maxResults": max_results,
    }
    if not include_past:
        params["timeMin"] = datetime.now(TZ).isoformat()

    try:
        raw = _service().events().list(**params).execute(num_retries=RETRIES)
    except HttpError as exc:
        _raise_from(exc, "find_events")
    return [_to_event(item) for item in raw.get("items", [])]