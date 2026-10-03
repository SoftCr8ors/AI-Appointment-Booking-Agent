"""Terminal chat for testing the agent. Run:  python chat_cli.py [--debug]"""
import json
import logging
import sys
import uuid
from pathlib import Path

# Add parent directory to path so we can import backend modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend import agent                                                         


def main() -> None:
    debug = "--debug" in sys.argv
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING)
    session_id = uuid.uuid4().hex
    print("AI Appointment Assistant  (type 'quit' to exit)\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        print(f"\nAI: {agent.chat(session_id, text)}\n")
        if debug:
            print("[state]", json.dumps(agent.debug_state(session_id), default=str), "\n")


if __name__ == "__main__":
    main()