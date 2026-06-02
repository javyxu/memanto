"""
Offline structural validation for the LangGraph + Memanto example.

This script intentionally uses only the Python standard library. It gives
reviewers a fast smoke test without requiring Memanto, LangGraph, OpenRouter,
or real API keys.
"""

from __future__ import annotations

import ast
import fnmatch
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / "langgraph_memanto"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse(path: Path) -> ast.Module:
    return ast.parse(_read(path), filename=str(path))


def _function_names(tree: ast.Module) -> set[str]:
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _called_attribute_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def _add_node_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_add_node = (
            isinstance(func, ast.Attribute) and func.attr == 'add_node'
        ) or (
            isinstance(func, ast.Name) and func.id == 'add_node'
        )
        if not is_add_node:
            continue
        if (
            node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            names.add(node.args[0].value)
        for keyword in node.keywords:
            if (
                keyword.arg == 'name'
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                names.add(keyword.value.value)
    return names


def _agent_id_default(path: Path) -> str | None:
    tree = _parse(path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == 'AGENT_ID'
            for target in node.targets
        ):
            continue
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == 'getenv'
        ):
            if (
                len(value.args) >= 2
                and isinstance(value.args[1], ast.Constant)
                and isinstance(value.args[1].value, str)
            ):
                return value.args[1].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_files() -> None:
    required = [
        ROOT / "README.md",
        ROOT / "requirements.txt",
        ROOT / ".env.example",
        ROOT / "demo.gif",
        ROOT / "run_research.py",
        ROOT / "run_writer.py",
        ROOT / "run_full_pipeline.py",
        PACKAGE / "__init__.py",
        PACKAGE / "state.py",
        PACKAGE / "memory_tools.py",
        PACKAGE / "nodes.py",
        PACKAGE / "graph.py",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    _assert(not missing, f"Missing expected files: {missing}")


def validate_syntax() -> None:
    for path in [
        ROOT / "run_research.py",
        ROOT / "run_writer.py",
        ROOT / "run_full_pipeline.py",
        PACKAGE / "state.py",
        PACKAGE / "memory_tools.py",
        PACKAGE / "nodes.py",
        PACKAGE / "graph.py",
    ]:
        _parse(path)


def validate_memanto_tools() -> None:
    tree = _parse(PACKAGE / "memory_tools.py")
    functions = _function_names(tree)
    calls = _called_attribute_names(tree)

    _assert("memanto_remember" in functions, "memanto_remember() is missing")
    _assert("memanto_recall" in functions, "memanto_recall() is missing")
    _assert("memanto_answer" in functions, "memanto_answer() is missing")
    _assert("remember" in calls, "SdkClient.remember() is not called")
    _assert("recall" in calls, "SdkClient.recall() is not called")
    _assert("answer" in calls, "SdkClient.answer() is not called")
    _assert("activate_agent" in calls, "Memanto agent activation is missing")


def validate_langgraph_flow() -> None:
    graph_tree = _parse(PACKAGE / "graph.py")
    node_names = _add_node_names(graph_tree)
    calls = _called_attribute_names(graph_tree)

    _assert("research" in node_names, "research node is not registered")
    _assert("writer" in node_names, "writer node is not registered")
    _assert("add_node" in calls, "StateGraph.add_node() is not used")
    _assert("add_conditional_edges" in calls, "conditional routing is not configured")
    _assert("compile" in calls, "graph.compile() is not called")


def validate_cross_session_demo() -> None:
    research = _read(ROOT / "run_research.py")
    writer = _read(ROOT / "run_writer.py")
    readme = _read(ROOT / "README.md").lower()

    _assert("MEMANTO_AGENT_ID" in research, "run_research.py must share MEMANTO_AGENT_ID")
    _assert("MEMANTO_AGENT_ID" in writer, "run_writer.py must share MEMANTO_AGENT_ID")
    _assert("cross-session" in readme, "README should describe cross-session recall")
    _assert("demo.gif" in readme, "README should embed the 30-second GIF")
    _assert("python run_research.py" in readme, "README should show research session command")
    _assert("python run_writer.py" in readme, "README should show writer session command")


def validate_env_safety() -> None:
    env_example = _read(ROOT / ".env.example")
    gitignore = _read(ROOT / ".gitignore") if (ROOT / ".gitignore").exists() else ""

    _assert("MOORCHEH_API_KEY" in env_example, ".env.example missing MOORCHEH_API_KEY")
    _assert("OPENROUTER_API_KEY" in env_example, ".env.example missing OPENROUTER_API_KEY")
    _assert(".env" in gitignore, ".gitignore must exclude local .env files")


def main() -> None:
    os.chdir(ROOT)
    checks = [
        validate_files,
        validate_syntax,
        validate_memanto_tools,
        validate_langgraph_flow,
        validate_cross_session_demo,
        validate_env_safety,
    ]
    for check in checks:
        check()
        print(f"ok - {check.__name__}")
    print("\nOffline validation passed for examples/langgraph-memanto.")


if __name__ == "__main__":
    main()
