"""Phase 0 check: verifies Calendar access (read + write) and the Gemini key."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from google.oauth2 import service_account
from googleapiclient.discovery import build

import config

SCOPES = ["https://www.googleapis.com/auth/calendar"]


def check_calendar() -> None:
    creds = service_account.Credentials.from_service_account_info(
        config.load_service_account_info(), scopes=SCOPES
    )
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    tz = ZoneInfo(config.TIMEZONE)
    now = datetime.now(tz)

    # 1) Read
    events = (
        service.events()
        .list(
            calendarId=config.CALENDAR_ID,
            timeMin=now.isoformat(),
            maxResults=5,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
        .get("items", [])
    )
    print(f"[OK] Calendar read works. Upcoming events: {len(events)}")
    for e in events:
        print("    -", e.get("summary"), e["start"].get("dateTime", e["start"].get("date")))

    # 2) Write + delete (proves "Make changes to events" permission)
    start = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
    body = {
        "summary": "SETUP TEST - safe to delete",
        "start": {"dateTime": start.isoformat(), "timeZone": config.TIMEZONE},
        "end": {"dateTime": (start + timedelta(minutes=30)).isoformat(), "timeZone": config.TIMEZONE},
    }
    created = service.events().insert(calendarId=config.CALENDAR_ID, body=body).execute()
    print("[OK] Calendar write works. Created event:", created["id"])
    service.events().delete(calendarId=config.CALENDAR_ID, eventId=created["id"]).execute()
    print("[OK] Calendar delete works. Test event removed.")


def check_gemini() -> None:
    from google import genai

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL, contents="Reply with exactly: OK"
    )
    print("[OK] Gemini works. Response:", response.text.strip())


if __name__ == "__main__":
    check_calendar()
    check_gemini()
    print("\nPhase 0 complete.")