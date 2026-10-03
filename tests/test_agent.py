"""Agent tests: confirmation guard, graph flow, helpers. No network, no real keys."""
import inspect
import os
import sys
from datetime import datetime
from pathlib import Path

# --- Make `backend/` importable and use fixed, fake settings -----------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("CALENDAR_ID", "test@group.calendar.google.com")
os.environ["TIMEZONE"] = "Asia/Karachi"
os.environ["BUSINESS_START"] = "10:00"
os.environ["BUSINESS_END"] = "18:00"
os.environ["SLOT_MINUTES"] = "60"
os.environ["WORKING_DAYS"] = "0,1,2,3,4"
os.environ["MIN_NOTICE_MINUTES"] = "60"
os.environ["MAX_DAYS_AHEAD"] = "60"
os.environ["SERVICES"] = "consultation,follow-up,demo"

import pytest  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from backend import agent  # noqa: E402

BOOK = dict(
    name="Walid Khan",
    email="walid@example.com",
    service="consultation",
    start_iso="2026-10-06T16:00:00",
)
WHO = dict(name="Walid Khan", email="walid@example.com")


# ==========================================================================
# Fixtures: fake tool functions that record how they were called
# ==========================================================================
@pytest.fixture
def book_calls(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("confirmed"):
            return {"error": "not_confirmed", "message": "confirm", "proposed": {"time": "16:00"}}
        return {"status": "booked", "event_id": "evt1"}

    monkeypatch.setattr(agent._REGISTRY["book_appointment"], "fn", fake)
    return calls


@pytest.fixture
def cancel_calls(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("confirmed"):
            return {"error": "not_confirmed", "appointment": {"event_id": "abc"}}
        return {"status": "cancelled", "event_id": kwargs.get("event_id")}

    monkeypatch.setattr(agent._REGISTRY["cancel_appointment"], "fn", fake)
    return calls


@pytest.fixture
def reschedule_calls(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        if not kwargs.get("confirmed"):
            return {
                "error": "not_confirmed",
                "appointment": {"event_id": "abc"},
                "proposed": {"time": "11:00"},
            }
        return {"status": "rescheduled", "event_id": kwargs.get("event_id")}

    monkeypatch.setattr(agent._REGISTRY["reschedule_appointment"], "fn", fake)
    return calls


# ==========================================================================
# 1. Confirmation guard
# ==========================================================================
def test_model_cannot_confirm_itself(book_calls):
    result, pending, done = agent._run_tool("book_appointment", {**BOOK, "confirmed": True}, None, 1)
    assert book_calls[0]["confirmed"] is False
    assert result["error"] == "not_confirmed"
    assert pending["turn"] == 1 and done is False


def test_same_turn_repeat_is_not_confirmation(book_calls):
    _, pending, _ = agent._run_tool("book_appointment", BOOK, None, 1)
    result, _, _ = agent._run_tool("book_appointment", BOOK, pending, 1)
    assert book_calls[-1]["confirmed"] is False
    assert result["error"] == "not_confirmed"


def test_later_turn_with_same_args_confirms(book_calls):
    _, pending, _ = agent._run_tool("book_appointment", BOOK, None, 1)
    result, pending, done = agent._run_tool("book_appointment", BOOK, pending, 2)
    assert book_calls[-1]["confirmed"] is True
    assert result["status"] == "booked" and pending is None and done is True


def test_changed_slot_needs_a_new_confirmation(book_calls):
    _, pending, _ = agent._run_tool("book_appointment", BOOK, None, 1)
    other = {**BOOK, "start_iso": "2026-10-06T17:00:00"}
    result, new_pending, _ = agent._run_tool("book_appointment", other, pending, 2)
    assert book_calls[-1]["confirmed"] is False
    assert result["error"] == "not_confirmed"
    assert new_pending["args"]["start_iso"].startswith("2026-10-06T17:00")
    assert new_pending["turn"] == 2


def test_equivalent_time_and_name_formats_match(book_calls):
    _, pending, _ = agent._run_tool("book_appointment", BOOK, None, 1)
    same = {**BOOK, "start_iso": "2026-10-06T16:00:00+05:00", "name": "  walid   KHAN "}
    result, _, _ = agent._run_tool("book_appointment", same, pending, 2)
    assert result["status"] == "booked"


def test_cancel_acts_on_the_confirmed_event(cancel_calls):
    _, pending, _ = agent._run_tool("cancel_appointment", WHO, None, 1)
    assert pending["event_id"] == "abc"
    result, _, _ = agent._run_tool("cancel_appointment", WHO, pending, 2)
    assert cancel_calls[-1]["event_id"] == "abc"
    assert cancel_calls[-1]["confirmed"] is True
    assert result["status"] == "cancelled"


def test_reschedule_needs_same_new_time_and_pins_event(reschedule_calls):
    args = {**WHO, "new_start_iso": "2026-10-07T11:00:00"}
    _, pending, _ = agent._run_tool("reschedule_appointment", args, None, 1)

    other = {**args, "new_start_iso": "2026-10-07T12:00:00"}
    agent._run_tool("reschedule_appointment", other, pending, 2)
    assert reschedule_calls[-1]["confirmed"] is False

    result, _, _ = agent._run_tool("reschedule_appointment", args, pending, 2)
    assert reschedule_calls[-1]["confirmed"] is True
    assert reschedule_calls[-1]["event_id"] == "abc"
    assert result["status"] == "rescheduled"


def test_failed_action_clears_the_proposal(monkeypatch, book_calls):
    _, pending, _ = agent._run_tool("book_appointment", BOOK, None, 1)
    monkeypatch.setattr(
        agent._REGISTRY["book_appointment"], "fn", lambda **kw: {"error": "slot_taken"}
    )
    result, pending, done = agent._run_tool("book_appointment", BOOK, pending, 2)
    assert result["error"] == "slot_taken" and pending is None and done is False


def test_checking_availability_keeps_the_pending_proposal(monkeypatch, book_calls):
    monkeypatch.setattr(
        agent._REGISTRY["check_availability"], "fn", lambda **kw: {"status": "ok"}
    )
    _, pending, _ = agent._run_tool("book_appointment", BOOK, None, 1)
    _, still_pending, _ = agent._run_tool("check_availability", {"day": "2026-10-06"}, pending, 2)
    assert still_pending == pending


def test_extra_arguments_are_stripped(book_calls):
    agent._run_tool("book_appointment", {**BOOK, "bogus": 1}, None, 1)
    assert "bogus" not in book_calls[0]


def test_unknown_tool_is_reported():
    result, _, _ = agent._run_tool("delete_everything", {}, None, 1)
    assert result["error"] == "unknown_tool"


# ==========================================================================
# 2. Tool schemas shown to Gemini
# ==========================================================================
def test_gemini_never_sees_the_confirmed_field():
    for spec in agent._SPECS:
        assert "confirmed" not in spec.args_model.model_fields


def test_schemas_match_the_real_tool_signatures():
    for spec in agent._SPECS:
        real = set(inspect.signature(spec.fn).parameters) - {"confirmed"}
        assert set(spec.args_model.model_fields) == real, spec.name


# ==========================================================================
# 3. Collected info
# ==========================================================================
def test_collected_info_is_kept_then_cleared_after_booking():
    info = {}
    agent._remember(info, "book_appointment", BOOK, {"error": "not_confirmed"})
    assert info["name"] == "Walid Khan"
    assert info["date"] == "2026-10-06" and info["time"] == "16:00"
    agent._remember(info, "book_appointment", BOOK, {"status": "booked"})
    assert "date" not in info and "service" not in info
    assert info["email"] == "walid@example.com"


# ==========================================================================
# 4. Prompt and message helpers
# ==========================================================================
def test_system_prompt_contains_dates_and_rules():
    now = datetime(2026, 10, 3, 12, 0, tzinfo=agent.tools.TZ)
    prompt = agent.build_system_prompt(now, {"name": "Walid"})
    assert "Today is Saturday, 2026-10-03" in prompt
    assert "Sunday 2026-10-04 (closed, tomorrow)" in prompt
    assert "Monday 2026-10-05 (open)" in prompt
    assert "Monday, Tuesday, Wednesday, Thursday, Friday" in prompt
    assert "10:00-18:00" in prompt
    assert "consultation, follow-up, demo" in prompt
    assert "- name: Walid" in prompt


def test_text_of_handles_block_content():
    message = AIMessage(content=[{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}])
    assert agent._text_of(message) == "Hello there"
    assert agent._text_of(AIMessage(content="  plain  ")) == "plain"


def test_window_starts_on_a_user_message():
    messages = [HumanMessage(content="u"), AIMessage(content="a")] * 30 + [HumanMessage(content="u")]
    window = agent._window(messages)
    assert isinstance(window[0], HumanMessage)
    assert len(window) <= agent.MAX_HISTORY_MESSAGES


def test_dangling_tool_calls_are_removed():
    def call(call_id):
        return AIMessage(
            content="",
            tool_calls=[{"name": "check_availability", "args": {}, "id": call_id, "type": "tool_call"}],
        )

    dangling, answered = call("x1"), call("x2")
    messages = [
        HumanMessage(content="hi"),
        dangling,
        HumanMessage(content="again"),
        answered,
        ToolMessage(content="{}", tool_call_id="x2", name="check_availability"),
        AIMessage(content="ok"),
    ]
    cleaned = agent._drop_dangling_tool_calls(messages)
    assert len(cleaned) == 5
    assert dangling not in cleaned and answered in cleaned


def test_route_after_llm():
    with_call = AIMessage(
        content="",
        tool_calls=[{"name": "check_availability", "args": {}, "id": "c", "type": "tool_call"}],
    )
    assert agent.route_after_llm({"messages": [with_call]}) == "tools"
    assert agent.route_after_llm({"messages": [AIMessage(content="hi")]}) == agent.END


def test_chat_rejects_empty_and_too_long_messages():
    assert "type a message" in agent.chat("s1", "   ").lower()
    assert "too long" in agent.chat("s1", "x" * (agent.MAX_MESSAGE_CHARS + 1)).lower()


# ==========================================================================
# 5. Whole graph with a scripted LLM
# ==========================================================================
class ScriptedLLM:
    def __init__(self, replies):
        self._replies = iter(replies)

    def invoke(self, messages):
        return next(self._replies)


def book_call(call_id):
    return AIMessage(
        content="",
        tool_calls=[{"name": "book_appointment", "args": BOOK, "id": call_id, "type": "tool_call"}],
    )


def install(monkeypatch, replies):
    confirmed_flags = []

    def fake_book(**kwargs):
        confirmed_flags.append(kwargs["confirmed"])
        if not kwargs["confirmed"]:
            return {"error": "not_confirmed", "proposed": {"time": "16:00"}}
        return {"status": "booked", "event_id": "evt1"}

    script = ScriptedLLM(replies)  # ONE script shared by every LLM call in the test
    monkeypatch.setattr(agent._REGISTRY["book_appointment"], "fn", fake_book)
    monkeypatch.setattr(agent, "_llm", lambda: script)
    return confirmed_flags, agent.build_graph(InMemorySaver())


def test_booking_needs_confirmation_across_turns(monkeypatch):
    flags, graph = install(
        monkeypatch,
        [book_call("c1"), AIMessage(content="Please confirm. Yes?"), book_call("c2"),
         AIMessage(content="Booked!")],
    )
    assert agent.chat("s1", "Book Monday 4pm", graph=graph) == "Please confirm. Yes?"
    assert flags == [False]
    assert agent.chat("s1", "yes", graph=graph) == "Booked!"
    assert flags == [False, True]
    state = agent.debug_state("s1", graph=graph)
    assert state["pending_slot"] is None
    assert state["collected_info"]["name"] == "Walid Khan"


def test_two_calls_in_one_turn_never_book(monkeypatch):
    flags, graph = install(
        monkeypatch, [book_call("c1"), book_call("c2"), AIMessage(content="Do you confirm?")]
    )
    assert agent.chat("s1", "Book Monday 4pm", graph=graph) == "Do you confirm?"
    assert flags == [False, False]


def test_confirmation_does_not_leak_between_sessions(monkeypatch):
    flags, graph = install(
        monkeypatch,
        [book_call("c1"), AIMessage(content="Confirm?"), book_call("c2"),
         AIMessage(content="Confirm?")],
    )
    assert agent.chat("alice", "Book Monday 4pm", graph=graph) == "Confirm?"
    assert agent.chat("bob", "Book Monday 4pm", graph=graph) == "Confirm?"
    assert flags == [False, False]


def test_gemini_failure_returns_friendly_message(monkeypatch):
    class Broken:
        def invoke(self, messages):
            raise RuntimeError("quota exceeded")

    monkeypatch.setattr(agent, "_llm", lambda: Broken())
    reply = agent.chat("s1", "hello", graph=agent.build_graph(InMemorySaver()))
    assert reply == agent.ERROR_REPLY


def test_empty_gemini_reply_returns_fallback(monkeypatch):
    monkeypatch.setattr(
        agent, "_llm", lambda: ScriptedLLM([AIMessage(content=""), AIMessage(content="")])
    )
    reply = agent.chat("s1", "hello", graph=agent.build_graph(InMemorySaver()))
    assert reply == agent.EMPTY_REPLY


def test_runaway_tool_loop_is_stopped(monkeypatch):
    def call(i):
        return AIMessage(
            content="",
            tool_calls=[{"name": "check_availability", "args": {"day": "2026-10-06"},
                         "id": f"c{i}", "type": "tool_call"}],
        )

    monkeypatch.setattr(
        agent._REGISTRY["check_availability"], "fn", lambda **kw: {"status": "ok"}
    )
    monkeypatch.setattr(
        agent, "_llm", lambda: ScriptedLLM([call(i) for i in range(agent.MAX_TOOL_ROUNDS + 3)])
    )
    reply = agent.chat("s1", "hello", graph=agent.build_graph(InMemorySaver()))
    assert reply == agent.LOOP_REPLY