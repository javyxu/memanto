"""
Run 2: Writer Agent retrieves memories from Memanto.

This script proves cross-session persistence: the memories stored
by run_research.py are retrieved here even in a completely new session.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langgraph_memanto.memory_tools import memanto_recall, memanto_answer

load_dotenv()

MEMANTO_API_KEY = os.getenv("MOORCHEH_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
AGENT_ID = os.getenv("MEMANTO_AGENT_ID", "langgraph-research-team")
TOPIC = os.getenv("RESEARCH_TOPIC", "AI agent framework market size and trends 2024")


def main():
    if not MEMANTO_API_KEY or not OPENROUTER_API_KEY:
        print("ERROR: Missing API keys.")
        print("Copy .env.example to .env and fill in MOORCHEH_API_KEY and OPENROUTER_API_KEY")
        sys.exit(1)

    llm = ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-haiku",
        temperature=0.7,
    )

    state = {"memanto_agent_id": AGENT_ID, "research_topic": TOPIC}

    print(f"Writer Agent retrieving memories for: {TOPIC}")
    print("---")

    # Recall memories
    print("\n[Step 1: Recalling memories from Memanto...]")
    recall_result = memanto_recall(
        state=state,
        api_key=MEMANTO_API_KEY,
        query=f"key findings about {TOPIC}",
        limit=10,
    )
    recall_content = recall_result.get("messages", [{}])[0].get("content", "")
    print(f"Recall result:\n{recall_content}\n")

    # RAG answer
    print("[Step 2: Synthesizing via RAG...]")
    answer_result = memanto_answer(
        state=state,
        api_key=MEMANTO_API_KEY,
        question=f"Summarize all research findings about {TOPIC} in a clear executive briefing",
    )
    answer_content = answer_result.get("messages", [{}])[0].get("content", "")
    print(f"Answer:\n{answer_content}\n")

    # LLM write briefing
    print("[Step 3: Writing executive briefing...]")
    synthesis_prompt = (
        f"You are a Technical Briefing Writer.\n"
        f"Topic: {TOPIC}\n\n"
        f"Retrieved memories:\n{recall_content}\n\n"
        f"RAG synthesis:\n{answer_content}\n\n"
        f"Write a clear executive briefing on '{TOPIC}' using ONLY the "
        f"information above. Do not fabricate. Cite sources."
    )
    response = llm.invoke(synthesis_prompt)
    content = response.content if hasattr(response, "content") else str(response)

    print("\n[Executive Briefing]")
    print(content)


if __name__ == "__main__":
    main()
