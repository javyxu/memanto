"""
LangGraph node definitions for the Research + Writer pipeline.

Each node is a function (state_in, state_out) that gets compiled into
the LangGraph pipeline via langgraph_memanto.graph.build_graph().
"""

from __future__ import annotations

import os
from typing import Any, Dict, Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from dotenv import load_dotenv

from langgraph_memanto.memory_tools import (
    memanto_remember as _memanto_remember,
    memanto_recall,
    memanto_answer,
)

load_dotenv()

MOORCHEH_API_KEY = os.getenv("MOORCHEH_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _get_llm():
    """Build an OpenRouter-backed chat model."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-haiku",
        temperature=0.7,
    )


# ---------------------------------------------------------------------------
# LangChain Tool wrapper factory (enables actual tool calling by the LLM)
# ---------------------------------------------------------------------------

def _build_memanto_remember_tool(agent_id: str):
    """Build a LangChain-compatible tool bound to the current Memanto namespace."""

    @tool("memanto_remember")
    def memanto_remember_tool(
        memory_type: str,
        title: str,
        content: str,
        confidence: float = 0.85,
        tags: list[str] | None = None,
    ) -> str:
        """Store a structured memory in Memanto for later cross-session recall."""
        result = _memanto_remember(
            state={"memanto_agent_id": agent_id},
            api_key=MOORCHEH_API_KEY,
            memory_type=memory_type or "observation",
            title=(title or "")[:100],
            content=(content or "")[:500],
            confidence=float(confidence),
            tags=tags or [],
        )
        return result.get("messages", [{}])[0].get("content", "")

    return memanto_remember_tool


# ---------------------------------------------------------------------------
# Research Agent nodes
# ---------------------------------------------------------------------------

def research_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The Research Agent analyzes a topic and stores key findings as Memanto memories.

    Uses tool calling: the LLM invokes memanto_remember as a real bound tool,
    demonstrating actual cross-session memory persistence.
    """
    topic = state.get("research_topic", "")
    agent_id = state.get("memanto_agent_id", "langgraph-default")
    llm = _get_llm()

    # Bind memanto_remember as an actual LangChain tool the LLM can call.
    tools = [_build_memanto_remember_tool(agent_id)]
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = (
        f"You are a Senior Market Research Analyst specialized in '{topic}'.\n"
        f"Your job is to research this topic thoroughly and store every significant "
        f"finding as a structured Memanto memory using the memanto_remember tool.\n\n"
        f"Research approach:\n"
        f"1. Think about the key aspects of '{topic}'\n"
        f"2. Use the memanto_remember tool to store at least 3 findings\n"
        f"3. Each memory should be atomic: one fact/observation per memory\n\n"
        f"Memory types: fact, observation, decision, learning, event\n"
        f"Confidence: 1.0 for verified facts, 0.7-0.9 for observations\n"
        f"Tags: relevant keywords like ['AI', 'market', 'trends']\n\n"
        f"After storing memories, summarize what you found in 2-3 sentences."
    )

    messages = [{"role": "system", "content": system_prompt}]
    response = llm_with_tools.invoke(messages)
    content = response.content if hasattr(response, "content") else str(response)

    # Execute any tool calls the model made
    tool_calls = getattr(response, "tool_calls", []) or []
    tool_results = []
    findings = []
    for tc in tool_calls:
        if tc.get("name") == "memanto_remember":
            args = tc.get("args", {})
            result = tools[0].invoke(args)
            tool_results.append({
                "role": "tool",
                "content": result,
                "name": "memanto_remember",
                "tool_call_id": tc.get("id", f"remember_{len(tool_results)}"),
            })
            title = args.get("title", "unknown")
            findings.append(title)

    all_messages = [{"role": "assistant", "content": content}]
    all_messages.extend(tool_results)

    return {
        "messages": all_messages,
        "findings": findings,
    }


def writer_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    The Writer Agent retrieves stored memories and writes an executive briefing.

    It calls memanto_recall to fetch all research memories, then uses the LLM
    to synthesize them into a written briefing.
    """
    topic = state.get("research_topic", "")
    agent_id = state.get("memanto_agent_id", "langgraph-default")
    llm = _get_llm()

    # First, recall memories
    recall_result = memanto_recall(
        state={**state, "memanto_agent_id": agent_id},
        api_key=MOORCHEH_API_KEY,
        query=f"research findings about {topic}",
        limit=10,
    )

    # Then answer a synthesis question
    answer_result = memanto_answer(
        state={**state, "memanto_agent_id": agent_id},
        api_key=MOORCHEH_API_KEY,
        question=f"Provide a detailed summary of all research findings about {topic}",
    )

    recall_content = recall_result.get("messages", [{}])[0].get("content", "")
    answer_content = answer_result.get("messages", [{}])[0].get("content", "")

    synthesis_prompt = (
        f"You are a Technical Briefing Writer.\n"
        f"Topic: {topic}\n\n"
        f"Retrieved memories:\n{recall_content}\n\n"
        f"RAG Answer:\n{answer_content}\n\n"
        f"Write a clear, data-driven executive briefing on '{topic}' "
        f"using ONLY the information from the retrieved memories above. "
        f"Do not fabricate data. Cite sources."
    )

    response = llm.invoke(synthesis_prompt)
    content = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [
            {"role": "assistant", "content": content},
        ],
    }


def should_continue(state: Dict[str, Any]) -> Literal["research", "writer", "end"]:
    """
    Routing logic: research → writer → end.
    """
    messages = state.get("messages", [])
    if not messages:
        return "research"
    last = messages[-1]
    role = last.get("role", "")
    if role == "assistant":
        return "writer"
    return "end"
