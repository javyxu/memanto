"""
Run the full research → writer pipeline in a single session.

This combines run_research.py and run_writer.py into one end-to-end demo,
useful for a quick demonstration or CI test.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from langgraph_memanto.graph import run_research

load_dotenv()

TOPIC = os.getenv("RESEARCH_TOPIC", "AI agent framework market size and trends 2024")
AGENT_ID = os.getenv("MEMANTO_AGENT_ID", "langgraph-research-team")


def main():
    print(f"Running full LangGraph + Memanto pipeline...")
    print(f"Topic: {TOPIC}")
    print(f"Memanto Agent ID: {AGENT_ID}")
    print("---")

    result = run_research(topic=TOPIC, memanto_agent_id=AGENT_ID)

    print("\n=== Pipeline Complete ===")
    print(f"Total messages: {len(result.get('messages', []))}")
    print(f"Findings stored: {len(result.get('findings', []))}")

    print("\n--- Messages ---")
    for msg in result.get("messages", []):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        print(f"\n[{role.upper()}]\n{content[:500]}")


if __name__ == "__main__":
    main()
