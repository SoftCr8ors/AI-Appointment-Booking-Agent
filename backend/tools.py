"""Business rules and the tools the AI agent can call.

Every tool returns a dict:
  success -> {"status": "...", ...}
  failure -> {"error": "<code>", "message": "<human text>", ...extra}
Tools never raise to the caller; unexpected calendar failures become
{"error": "calendar_unavailable"}.

The docstrings of the public tools are what Gemini reads in Phase 3.
"""
from __future__ import annotations

import logging
import re
import threading
from datetime import date, datetime, time, timedelta
from functools import wraps
from typing import Any

import config
import gcal

logger = logging.getLogger(__name__)

TZ = gcal.TZ
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_ALTERNATIVES = 3
_booking_lock = threading.Lock()


def _now() -> datetime:
    """Current time in the business timezone (patched in tests)."""
    return datetime.now(TZ)


# ==========================================================================
# Pure slot logic (no Google, easy to unit test)
# ==========================================================================
def generate_slots(day: date) -> list[datetime]:
    """All bookable slot start times for a day, ignoring existing bookings."""
    if day.weekday() not in config.WORKING_DAYS:
        return []
    step = timedelta(minutes=config.SLOT_MINUTES)
    cursor = datetime.combine(day, config.BUSINESS_START, tzinfo=TZ)
    closing = datetime.combine(day, config.BUSINESS_END, tzinfo=TZ)
    slots = []
    while cursor + step <= closing:
        slots.append(cursor)
        cursor += step
    return slots


def _overlaps(start: datetime, end: datetime, busy: list[tuple[datetime, datetime]]) -> bool:
    return any(start < b_end and b_start < end for b_start, b_end in busy)


def compute_free_slots(
    day: date, busy: list[tuple[datetime, datetime]], now: datetime
) -> list[datetime]:
    """Slots that are in the future (with minimum notice) and not busy."""
    earliest = now + timedelta(minutes=config.MIN_NOTICE_MINUTES)
    step = timedelta(minutes=config.SLOT_MINUTES)
    return [
        s
        for s in generate_slots(day)
        if s >= earliest and not _overlaps(s, s + step, busy)
    ]


def nearest_slots(
    slots: list[datetime], target: datetime, limit: int = MAX_ALTERNATIVES
) -> list[datetime]:
    """The `limit` slots closest to `target`, returned in time order."""
    closest = sorted(slots, key=lambda s: (abs(s - target), s))[:limit]
    return sorted(closest)


# ==========================================================================
# Small helpers
# ==========================================================================
def _error(code: str, message: str, **extra: Any) -> dict:
    return {"error": code, "message": message, **extra}


def _label(dt: datetime) -> str:
    """'3:00 PM' (works on Windows, unlike %-I)."""
    return f"{dt.hour % 12 or 12}:{dt:%M} {'AM' if dt.hour < 12 else 'PM'}"


def _slot_dict(dt: datetime) -> dict:
    return {"time": f"{dt:%H:%M}", "label": _label(dt), "start_iso": dt.isoformat()}


def _event_dict(event: gcal.CalendarEvent) -> dict:
    return {
        "event_id": event.id,
        "service": event.private_props.get("service", ""),
        "date": event.start.date().isoformat(),
        "weekday": f"{event.start:%A}",
        "time": _label(event.start),
        "start_iso": event.start.isoformat(),
    }


def _parse_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _parse_clock(value: Any) -> time | None:
    try:
        return time.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _parse_start(value: Any) -> datetime | None:
    """ISO datetime -> aware datetime in the business timezone.

    A value without an offset is assumed to already be business-local time.
    """
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=TZ)
    return parsed.astimezone(TZ)


def _check_fields(**fields: Any) -> dict | None:
    missing = [k for k, v in fields.items() if v is None or not str(v).strip()]
    if missing:
        return _error(
            "missing_fields",
            f"Missing required information: {', '.join(missing)}. Ask the user for it.",
            missing=missing,
        )
    if "email" in fields and not EMAIL_RE.match(str(fields["email"]).strip()):
        return _error("invalid_email", "That email address does not look valid.")
    return None


def _validate_start(start: datetime, now: datetime) -> dict | None:
    """Rules every appointment time must satisfy (before checking the calendar)."""
    if start <= now:
        return _error("in_past", "That time has already passed.")
    if start < now + timedelta(minutes=config.MIN_NOTICE_MINUTES):
        return _error(
            "too_soon",
            f"Appointments need at least {config.MIN_NOTICE_MINUTES} minutes' notice.",
        )
    if start.date() > now.date() + timedelta(days=config.MAX_DAYS_AHEAD):
        return _error(
            "too_far_ahead",
            f"We only book up to {config.MAX_DAYS_AHEAD} days ahead.",
        )
    if start.weekday() not in config.WORKING_DAYS:
        return _error("non_working_day", f"We are closed on {start:%A}.")
    if start not in generate_slots(start.date()):
        return _error(
            "invalid_slot",
            f"Appointments run {config.BUSINESS_START:%H:%M}-{config.BUSINESS_END:%H:%M} "
            f"in {config.SLOT_MINUTES}-minute slots, starting on the slot boundary.",
        )
    return None


def _next_available_dates(
    after: date, now: datetime, count: int = 3, horizon: int = 14
) -> list[dict]:
    """Next days that still have free slots (one Calendar call for the range)."""
    first = after + timedelta(days=1)
    last = min(after + timedelta(days=horizon), now.date() + timedelta(days=config.MAX_DAYS_AHEAD))
    if first > last:
        return []
    busy = gcal.get_busy_between(
        datetime.combine(first, time.min, tzinfo=TZ),
        datetime.combine(last + timedelta(days=1), time.min, tzinfo=TZ),
    )
    found: list[dict] = []
    day = first
    while day <= last and len(found) < count:
        free = compute_free_slots(day, busy, now)
        if free:
            found.append(
                {
                    "date": day.isoformat(),
                    "weekday": f"{day:%A}",
                    "first_free_time": _label(free[0]),
                }
            )
        day += timedelta(days=1)
    return found


def _alternatives_payload(target: datetime, now: datetime) -> dict:
    day = target.date()
    free = compute_free_slots(day, gcal.get_busy_slots(day), now)
    payload: dict = {"alternatives": [_slot_dict(s) for s in nearest_slots(free, target)]}
    if not free:
        payload["next_available_dates"] = _next_available_dates(day, now)
    return payload


def _select_event(
    events: list[gcal.CalendarEvent], event_id: str | None
) -> tuple[gcal.CalendarEvent | None, dict | None]:
    """Pick which of the user's own events an action applies to."""
    if event_id:
        for event in events:
            if event.id == event_id:
                return event, None
        # Same message as "no appointments": do not reveal other people's events.
        return None, _error("appointment_not_found", "No matching appointment was found.")
    if len(events) == 1:
        return events[0], None
    return None, _error(
        "multiple_appointments",
        "The user has several appointments. Ask which one, then retry with its event_id.",
        appointments=[_event_dict(e) for e in events],
    )


def _safe(fn):
    """Turn calendar outages into a structured error instead of an exception."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except gcal.CalendarError:
            logger.exception("Calendar failure in %s", fn.__name__)
            return _error(
                "calendar_unavailable",
                "The calendar is temporarily unavailable. Please try again in a moment.",
            )

    return wrapper


# ==========================================================================
# Tools
# ==========================================================================
@_safe
def check_availability(day: str, preferred_time: str | None = None) -> dict:
    """Check free appointment slots for one day.

    Args:
        day: Date in YYYY-MM-DD format.
        preferred_time: Optional time the user asked for, in 24-hour HH:MM format.
            If it is not free, the nearest free alternatives are returned.
    """
    requested_day = _parse_date(day)
    if requested_day is None:
        return _error("invalid_date", "Use the YYYY-MM-DD date format.")

    now = _now()
    if requested_day < now.date():
        return _error("date_in_past", "That date has already passed.")
    if requested_day > now.date() + timedelta(days=config.MAX_DAYS_AHEAD):
        return _error("too_far_ahead", f"We only book up to {config.MAX_DAYS_AHEAD} days ahead.")

    preferred = None
    if preferred_time:
        clock = _parse_clock(preferred_time)
        if clock is None:
            return _error("invalid_time", "Use the 24-hour HH:MM time format.")
        preferred = datetime.combine(requested_day, clock, tzinfo=TZ)

    open_day = requested_day.weekday() in config.WORKING_DAYS
    busy = gcal.get_busy_slots(requested_day) if open_day else []
    free = compute_free_slots(requested_day, busy, now)

    result: dict = {
        "status": "ok",
        "date": requested_day.isoformat(),
        "weekday": f"{requested_day:%A}",
        "free_slots": [_slot_dict(s) for s in free],
    }
    if not open_day:
        result["note"] = "Closed on this day."
    if preferred is not None:
        result["requested_time"] = f"{preferred:%H:%M}"
        result["requested_available"] = preferred in free
        if preferred not in free:
            result["alternatives"] = [_slot_dict(s) for s in nearest_slots(free, preferred)]
    if not free:
        result["next_available_dates"] = _next_available_dates(requested_day, now)
    return result


@_safe
def book_appointment(
    name: str | None = None,
    email: str | None = None,
    service: str | None = None,
    start_iso: str | None = None,
    confirmed: bool = False,
) -> dict:
    """Book an appointment in the calendar.

    Only call this after the user has explicitly confirmed this exact slot.

    Args:
        name: Client's full name.
        email: Client's email address.
        service: Service type, for example "consultation".
        start_iso: Slot start, ISO format like 2026-10-06T14:00:00.
        confirmed: Set by the system, never by you.
    """
    err = _check_fields(name=name, email=email, service=service, start_iso=start_iso)
    if err:
        return err

    clean_service = " ".join(str(service).lower().split())
    if clean_service not in config.SERVICES:
        return _error(
            "invalid_service",
            "That service is not offered.",
            available_services=list(config.SERVICES),
        )

    start = _parse_start(start_iso)
    if start is None:
        return _error("invalid_datetime", "Use ISO format, for example 2026-10-06T14:00:00.")
    now = _now()
    err = _validate_start(start, now)
    if err:
        return err

    end = start + timedelta(minutes=config.SLOT_MINUTES)
    if not confirmed:
        return _error(
            "not_confirmed",
            "Ask the user to confirm this exact slot before booking.",
            proposed=_slot_dict(start),
        )

    clean_name = " ".join(str(name).split())
    clean_email = str(email).strip().lower()
    with _booking_lock:  # re-check right before creating, no gap for a second user
        taken = _overlaps(start, end, gcal.get_busy_between(start, end))
        event = None
        if not taken:
            event = gcal.create_event(
                start,
                end,
                title=f"{clean_service.title()} - {clean_name}",
                description=(
                    f"Client: {clean_name}\nEmail: {clean_email}\n"
                    f"Service: {clean_service}\nBooked via AI Appointment Assistant"
                ),
                private_props=gcal.owner_props(clean_name, clean_email)
                | {"service": clean_service},
            )
    if taken:
        return _error(
            "slot_taken",
            "That slot was just taken.",
            **_alternatives_payload(start, now),
        )
    return {
        "status": "booked",
        "message": f"{clean_service.title()} booked for {start:%A, %Y-%m-%d} at {_label(start)}.",
        **_event_dict(event),
    }


@_safe
def list_appointments(name: str | None = None, email: str | None = None) -> dict:
    """List the upcoming appointments of one client.

    Args:
        name: Client's full name.
        email: Client's email address.
    """
    err = _check_fields(name=name, email=email)
    if err:
        return err
    events = gcal.find_events(name=name, email=email)
    return {"status": "ok", "appointments": [_event_dict(e) for e in events]}


@_safe
def cancel_appointment(
    name: str | None = None,
    email: str | None = None,
    event_id: str | None = None,
    confirmed: bool = False,
) -> dict:
    """Cancel one of the client's own appointments.

    Only call this after the user has confirmed which appointment to cancel.

    Args:
        name: Client's full name (must match the booking).
        email: Client's email address (must match the booking).
        event_id: Which appointment, needed only if the client has several.
        confirmed: Set by the system, never by you.
    """
    err = _check_fields(name=name, email=email)
    if err:
        return err
    events = gcal.find_events(name=name, email=email)
    if not events:
        return _error(
            "no_appointments_found",
            "No upcoming appointment matches that name and email.",
        )
    event, err = _select_event(events, event_id)
    if err:
        return err
    if not confirmed:
        return _error(
            "not_confirmed",
            "Ask the user to confirm cancelling this appointment.",
            appointment=_event_dict(event),
        )
    try:
        gcal.delete_event(event.id)
    except gcal.EventNotFoundError:
        return _error("appointment_not_found", "That appointment no longer exists.")
    return {"status": "cancelled", **_event_dict(event)}


@_safe
def reschedule_appointment(
    name: str | None = None,
    email: str | None = None,
    new_start_iso: str | None = None,
    event_id: str | None = None,
    confirmed: bool = False,
) -> dict:
    """Move one of the client's own appointments to a new time.

    Only call this after the user has confirmed the new slot.

    Args:
        name: Client's full name (must match the booking).
        email: Client's email address (must match the booking).
        new_start_iso: New slot start, ISO format like 2026-10-06T16:00:00.
        event_id: Which appointment, needed only if the client has several.
        confirmed: Set by the system, never by you.
    """
    err = _check_fields(name=name, email=email, new_start_iso=new_start_iso)
    if err:
        return err
    events = gcal.find_events(name=name, email=email)
    if not events:
        return _error(
            "no_appointments_found",
            "No upcoming appointment matches that name and email.",
        )
    event, err = _select_event(events, event_id)
    if err:
        return err

    new_start = _parse_start(new_start_iso)
    if new_start is None:
        return _error("invalid_datetime", "Use ISO format, for example 2026-10-06T16:00:00.")
    now = _now()
    err = _validate_start(new_start, now)
    if err:
        return err
    if new_start == event.start:
        return _error("same_time", "The appointment is already at that time.")
    if not confirmed:
        return _error(
            "not_confirmed",
            "Ask the user to confirm moving the appointment to this slot.",
            appointment=_event_dict(event),
            proposed=_slot_dict(new_start),
        )

    new_end = new_start + timedelta(minutes=config.SLOT_MINUTES)
    with _booking_lock:
        busy = [
            b
            for b in gcal.get_busy_between(new_start, new_end)
            if b != (event.start, event.end)  # ignore the appointment's own slot
        ]
        taken = _overlaps(new_start, new_end, busy)
        updated = None
        if not taken:
            try:
                updated = gcal.update_event(event.id, new_start, new_end)
            except gcal.EventNotFoundError:
                return _error("appointment_not_found", "That appointment no longer exists.")
    if taken:
        return _error(
            "slot_taken",
            "That slot is not available.",
            **_alternatives_payload(new_start, now),
        )
    return {
        "status": "rescheduled",
        "old_start_iso": event.start.isoformat(),
        **_event_dict(updated),
    }


# Used by the LangGraph agent in Phase 3
TOOLS = [
    check_availability,
    book_appointment,
    list_appointments,
    cancel_appointment,
    reschedule_appointment,
]