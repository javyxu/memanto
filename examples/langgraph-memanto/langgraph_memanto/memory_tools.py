"""
Memanto memory tools as pure LangGraph-compatible functions.

These tools wrap the Memanto SDK's SdkClient so they can be used
directly as LangGraph node functions — no class inheritance needed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from memanto.cli.client.sdk_client import SdkClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid Memanto memory types
# ---------------------------------------------------------------------------

VALID_MEMORY_TYPES = (
    "fact, preference, goal, decision, artifact, learning, "
    "event, instruction, relationship, context, observation, "
    "commitment, error"
)

# ---------------------------------------------------------------------------
# Client factory (singleton per agent_id)
# ---------------------------------------------------------------------------

_CLIENTS: Dict[str, SdkClient] = {}


def _get_client(api_key: str, agent_id: str) -> SdkClient:
    """Return a cached SdkClient for the given agent_id, creating if needed."""
    cache_key = f"{agent_id}"
    if cache_key not in _CLIENTS:
        client = SdkClient(api_key=api_key)
        # Ensure the agent exists in Memanto
        try:
            client.create_agent(
                agent_id=agent_id,
                pattern="tool",
                description="LangGraph research team shared memory",
            )
            logger.info("Created Memanto agent '%s'", agent_id)
        except Exception:
            logger.info("Memanto agent '%s' already exists, reusing", agent_id)
        # Activate session (6 hours)
        client.activate_agent(agent_id, duration_hours=6)
        _CLIENTS[cache_key] = client
    return _CLIENTS[cache_key]


# ---------------------------------------------------------------------------
# Tool functions  (stateful — read agent_id from state)
# ---------------------------------------------------------------------------


def memanto_remember(
    state: Dict[str, Any],
    *,
    api_key: str,
    memory_type: str,
    title: str,
    content: str,
    confidence: float = 0.85,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Store a structured memory in Memanto.

    LangGraph node function — reads ``memanto_agent_id`` from state
    and appends a human-readable result to ``messages``.

    Args:
        state: The current LangGraph state dict.
        api_key: Moorcheh API key.
        memory_type: Semantic type (see VALID_MEMORY_TYPES).
        title: Short title (max 100 chars).
        content: Memory content (max 500 chars).
        confidence: Confidence score 0.0–1.0.
        tags: Optional list of string tags.

    Returns:
        Dict to merge back into state (appends a tool result message).
    """
    agent_id = state.get("memanto_agent_id", "langgraph-default")
    client = _get_client(api_key, agent_id)

    tag_list = tags or []

    result = client.remember(
        agent_id=agent_id,
        memory_type=memory_type,
        title=title,
        content=content,
        confidence=confidence,
        tags=tag_list,
        source="langgraph-agent",
        provenance="explicit_statement",
    )

    memory_id = result.get("memory_id", "unknown")
    msg = (
        f"[memanto_remember] Stored memory.\n"
        f"  ID: {memory_id}\n"
        f"  Type: {memory_type}\n"
        f"  Title: {title}\n"
        f"  Confidence: {confidence}"
    )
    logger.info("Memanto remember: %s → %s", title, memory_id)

    return {
        "messages": [
            {
                "role": "tool",
                "content": msg,
                "tool_call_id": f"remember_{memory_id}",
            }
        ]
    }


def memanto_recall(
    state: Dict[str, Any],
    *,
    api_key: str,
    query: str,
    limit: int = 5,
    memory_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Search and retrieve memories from Memanto.

    LangGraph node function — reads ``memanto_agent_id`` from state.

    Args:
        state: The current LangGraph state dict.
        api_key: Moorcheh API key.
        query: Natural language search query.
        limit: Max memories to retrieve (1–20, default 5).
        memory_types: Optional list of memory types to filter by.

    Returns:
        Dict to merge back into state (appends a tool result message).
    """
    agent_id = state.get("memanto_agent_id", "langgraph-default")
    client = _get_client(api_key, agent_id)

    result = client.recall(
        agent_id=agent_id,
        query=query,
        limit=min(limit, 20),
        type=memory_types,
    )

    memories = result.get("memories", [])
    if not memories:
        content = f"No memories found for query: '{query}'"
    else:
        lines = [f"Found {len(memories)} memories for '{query}':\n"]
        for i, mem in enumerate(memories, 1):
            lines.append(
                f"  {i}. [{mem.get('type', '?')}] {mem.get('title', 'Untitled')}"
                f" (confidence: {mem.get('confidence', 'N/A')})\n"
                f"     {mem.get('content', '')}\n"
            )
        content = "\n".join(lines)

    logger.info("Memanto recall: %s → %d results", query, len(memories))

    return {
        "messages": [
            {
                "role": "tool",
                "content": content,
                "tool_call_id": f"recall_{hash(query) % 10000}",
            }
        ]
    }


def memanto_answer(
    state: Dict[str, Any],
    *,
    api_key: str,
    question: str,
) -> Dict[str, Any]:
    """
    Get an AI-generated answer grounded in stored memories (RAG).

    LangGraph node function — reads ``memanto_agent_id`` from state.

    Args:
        state: The current LangGraph state dict.
        api_key: Moorcheh API key.
        question: A question to answer using RAG over stored memories.

    Returns:
        Dict to merge back into state (appends a tool result message).
    """
    agent_id = state.get("memanto_agent_id", "langgraph-default")
    client = _get_client(api_key, agent_id)

    result = client.answer(agent_id=agent_id, question=question)

    answer = result.get("answer", "No answer could be generated.")
    sources = result.get("sources", [])
    content = f"Answer: {answer}"
    if sources:
        content += f"\n\nBased on {len(sources)} memory source(s)."

    logger.info("Memanto answer: %s", question[:50])

    return {
        "messages": [
            {
                "role": "tool",
                "content": content,
                "tool_call_id": f"answer_{hash(question) % 10000}",
            }
        ]
    }
