"""
LangGraph pipeline for the Research + Writer team with Memanto memory.
"""

from __future__ import annotations

from typing import Dict, Any

from langgraph.graph import StateGraph, END

from langgraph_memanto.state import ResearchState
from langgraph_memanto.nodes import research_agent, writer_agent, should_continue


def build_research_graph() -> StateGraph:
    """
    Build and compile the research pipeline graph.

    Flow:
        START → research_agent → writer_agent → END

    The Memanto agent ID is passed in via the initial state so both
    agents share the same persistent memory namespace.
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("research", research_agent)
    graph.add_node("writer", writer_agent)

    # Set entry point
    graph.set_entry_point("research")

    # Conditional routing after research
    graph.add_conditional_edges(
        "research",
        should_continue,
        {
            "writer": "writer",
            "end": END,
        },
    )

    # writer always ends
    graph.add_edge("writer", END)

    return graph


def compile_graph():
    """Build and return a compiled graph ready to invoke."""
    builder = build_research_graph()
    return builder.compile()


# ---------------------------------------------------------------------------
# Convenience runners
# ---------------------------------------------------------------------------


def run_research(topic: str, memanto_agent_id: str = "langgraph-research-team") -> Dict[str, Any]:
    """
    Run the full research → writer pipeline.

    Args:
        topic: The research topic to investigate.
        memanto_agent_id: Memanto agent namespace for shared memory.

    Returns:
        The final state after the graph completes.
    """
    compiled = compile_graph()

    initial_state = {
        "messages": [],
        "memanto_agent_id": memanto_agent_id,
        "research_topic": topic,
        "findings": [],
    }

    return compiled.invoke(initial_state)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    topic = os.getenv("RESEARCH_TOPIC", "AI agent framework market size and trends")
    agent_id = os.getenv("MEMANTO_AGENT_ID", "langgraph-research-team")

    print(f"Running LangGraph + Memanto research pipeline...")
    print(f"Topic: {topic}")
    print(f"Memanto Agent ID: {agent_id}")
    print("---")

    result = run_research(topic, agent_id)

    print("\n=== Final Output ===")
    for msg in result.get("messages", []):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        print(f"\n[{role.upper()}]\n{content}")
