"""Manual test for Phase 2: run every tool against the real calendar (no LLM)."""
import json
import sys
from datetime import timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import config
from backend import gcal
from backend import tools

NAME = "Walid Khan"
EMAIL = "walid.test@example.com"
OTHER_NAME = "Other Person"
OTHER_EMAIL = "someone.else@example.com"


def show(title: str, result: dict) -> None:
    print(f"\n--- {title}\n{json.dumps(result, indent=2)}")


def first_working_day():
    day = tools._now().date() + timedelta(days=1)
    while day.weekday() not in config.WORKING_DAYS:
        day += timedelta(days=1)
    return day


def main() -> None:
    day = first_working_day().isoformat()
    try:
        # 1) Availability
        availability = tools.check_availability(day)
        show(f"Availability on {day}", availability)
        slots = availability["free_slots"]
        assert len(slots) >= 2, "need at least 2 free slots to run this test"
        first, last = slots[0], slots[-1]

        # 2) Booking is refused without confirmation
        r = tools.book_appointment(NAME, EMAIL, "consultation", first["start_iso"])
        show("Book WITHOUT confirmation", r)
        assert r["error"] == "not_confirmed"

        # 3) Booking with confirmation
        r = tools.book_appointment(NAME, EMAIL, "consultation", first["start_iso"], confirmed=True)
        show("Book WITH confirmation", r)
        assert r["status"] == "booked"

        # 4) Same slot now unavailable, alternatives offered
        r = tools.check_availability(day, preferred_time=first["time"])
        show("Preferred time now taken", r)
        assert r["requested_available"] is False and r["alternatives"]

        # 5) A second person cannot take the same slot
        r = tools.book_appointment(OTHER_NAME, OTHER_EMAIL, "consultation", first["start_iso"], confirmed=True)
        show("Double booking attempt", r)
        assert r["error"] == "slot_taken"

        # 6) Wrong owner cannot cancel
        r = tools.cancel_appointment(NAME, OTHER_EMAIL, confirmed=True)
        show("Cancel with wrong email", r)
        assert r["error"] == "no_appointments_found"

        # 7) Reschedule
        r = tools.reschedule_appointment(NAME, EMAIL, last["start_iso"], confirmed=True)
        show("Reschedule", r)
        assert r["status"] == "rescheduled"

        # 8) List
        r = tools.list_appointments(NAME, EMAIL)
        show("List appointments", r)
        assert len(r["appointments"]) == 1

        # 9) Cancel
        r = tools.cancel_appointment(NAME, EMAIL, confirmed=True)
        show("Cancel", r)
        assert r["status"] == "cancelled"

        print("\nPhase 2 complete.")
    finally:
        for event in gcal.find_events(email=EMAIL):
            gcal.delete_event(event.id)  # always clean up


if __name__ == "__main__":
    main()