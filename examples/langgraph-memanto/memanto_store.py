"""MemantoStore - a LangGraph BaseStore backed by Memanto.

This is what makes the demo work. Compile the graph with
``store=MemantoStore(client, agent_id)`` and nodes get cross-thread,
cross-session memory through the official LangGraph store API
(``store.aput`` / ``store.asearch``), just like ``InMemoryStore``,
``PostgresStore``, or ``RedisStore``.

Mapping between abstractions
----------------------------

LangGraph's BaseStore is a namespaced key-value store with semantic
search. Memanto is a typed semantic memory database addressed by
``agent_id`` and ``memory_id``. The mapping:

    BaseStore                       ->  Memanto
    --------------------------------------------------------------
    namespace (tuple[str, ...])     ->  reserved tags  ``lg:ns:0:<p0>``,
                                                       ``lg:ns:1:<p1>``, ...
    key (str)                       ->  reserved tag   ``lg:key:<key>``
    value["kind"] / value["type"]   ->  memory_type    (default "fact")
    value["title"]                  ->  title          (auto-derived if absent)
    value["content"]                ->  content        (auto-stringified if absent)
    value["confidence"]             ->  confidence     (default 0.8)
    value["tags"]                   ->  user tags      (joined with reserved)
    SearchOp.query                  ->  recall query   ("*" if empty)
    SearchOp.filter["type"]         ->  type filter
    SearchOp.filter["tags"]         ->  extra tag filter
    SearchOp.filter["min_confidence"] -> min_confidence

Documented limitations
----------------------

* **Delete** (``PutOp`` with ``value=None``) raises ``NotImplementedError``.
  Memanto deletions go through its conflict-resolution flow, not free-form
  removal. Use ``memanto conflicts resolve`` instead.
* **TTL** on put is ignored - Memanto doesn't expire memories on a timer.
* **Pagination offset** in search is ignored - Memanto recall doesn't
  paginate. Raise the ``limit`` instead.
* **list_namespaces** is best-effort: samples up to ``limit`` recent
  memories and derives unique namespaces from their tags.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    PutOp,
    SearchItem,
    SearchOp,
)

from memanto.cli.client.sdk_client import SdkClient

logger = logging.getLogger(__name__)

_NS_TAG_PREFIX = "lg:ns:"
_KEY_TAG_PREFIX = "lg:key:"
_RESERVED_PREFIX = "lg:"

_VALID_MEMORY_TYPES = {
    "fact",
    "preference",
    "goal",
    "decision",
    "artifact",
    "learning",
    "event",
    "instruction",
    "relationship",
    "context",
    "observation",
    "commitment",
    "error",
}


class MemantoStore(BaseStore):
    """LangGraph ``BaseStore`` backed by Memanto's typed semantic memory.

    Drop-in replacement for ``InMemoryStore`` / ``PostgresStore`` /
    ``RedisStore``. Memories persist across threads and sessions because
    Memanto persists them server-side, scoped by ``agent_id``.

    Example::

        from memanto_setup import MemantoSetup
        from memanto_store import MemantoStore

        client = MemantoSetup(api_key).setup(agent_id="my-app")
        store = MemantoStore(client, agent_id="my-app")
        graph = builder.compile(store=store, checkpointer=InMemorySaver())
    """

    def __init__(self, client: SdkClient, agent_id: str) -> None:
        self._client = client
        self._agent_id = agent_id
        # (namespace, query, limit) -> (timestamp, list[SearchItem])
        self._search_cache: dict[tuple, tuple[float, list[SearchItem]]] = {}
        # Last good result per namespace, used to ride out 429s without
        # showing an empty memory panel.
        self._last_good: dict[tuple[str, ...], list[SearchItem]] = {}

    # ------------------------------------------------------------------ #
    # Required abstract methods                                          #
    # ------------------------------------------------------------------ #

    def batch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute a batch of store operations synchronously."""
        return [self._dispatch_one(op) for op in ops]

    async def abatch(self, ops: Iterable[Any]) -> list[Any]:
        """Execute a batch of store operations asynchronously.

        The Memanto SDK is synchronous, so we offload to a worker thread
        to keep the event loop free.
        """
        op_list = list(ops)
        return await asyncio.to_thread(self.batch, op_list)

    # ------------------------------------------------------------------ #
    # Per-op dispatch                                                    #
    # ------------------------------------------------------------------ #

    def _dispatch_one(self, op: Any) -> Any:
        if isinstance(op, GetOp):
            return self._do_get(op)
        if isinstance(op, PutOp):
            return self._do_put(op)
        if isinstance(op, SearchOp):
            return self._do_search(op)
        if isinstance(op, ListNamespacesOp):
            return self._do_list_namespaces(op)
        raise NotImplementedError(f"Unsupported store op: {type(op).__name__}")

    # ------------------------------------------------------------------ #
    # GET                                                                #
    # ------------------------------------------------------------------ #

    def _do_get(self, op: GetOp) -> Item | None:
        """Lookup-by-key, implemented via tag-filtered recall."""
        ns_tags = self._namespace_to_tags(op.namespace)
        key_tag = self._key_to_tag(op.key)

        result = self._client.recall(
            agent_id=self._agent_id,
            query=op.key or "*",
            limit=10,
            tags=ns_tags + [key_tag],
        )

        # Memanto's tag filter may be permissive - enforce match here.
        for mem in result.get("memories", []):
            tags = mem.get("tags", []) or []
            if key_tag in tags and all(t in tags for t in ns_tags):
                return self._memory_to_item(mem, op.namespace, op.key)
        return None

    # ------------------------------------------------------------------ #
    # PUT (and delete-via-put-None)                                      #
    # ------------------------------------------------------------------ #

    def _do_put(self, op: PutOp) -> None:
        if op.value is None:
            raise NotImplementedError(
                "MemantoStore does not support delete via PutOp(value=None). "
                "Memanto removals go through the conflict-resolution flow; "
                "use `memanto conflicts resolve` or the SdkClient's resolve API."
            )

        value: dict[str, Any] = dict(op.value)

        memory_type = str(value.pop("kind", value.pop("type", "fact"))).lower()
        if memory_type not in _VALID_MEMORY_TYPES:
            memory_type = "fact"

        raw_content = value.pop("content", None)
        if raw_content is None:
            raw_content = self._stringify(value)

        title = value.pop("title", None)
        if not title:
            title = raw_content if len(raw_content) <= 80 else raw_content[:77] + "..."
        title = title[:100]

        confidence = float(value.pop("confidence", 0.8))
        confidence = max(0.0, min(1.0, confidence))

        user_tags = list(value.pop("tags", []) or [])
        user_tags = [t for t in user_tags if not str(t).startswith(_RESERVED_PREFIX)]

        all_tags = (
            user_tags
            + self._namespace_to_tags(op.namespace)
            + [self._key_to_tag(op.key)]
        )

        self._client.remember(
            agent_id=self._agent_id,
            memory_type=memory_type,
            title=title,
            content=str(raw_content),
            confidence=confidence,
            tags=all_tags,
            source="langgraph-store",
            provenance="explicit_statement",
        )

        # Invalidate cached searches for this namespace so the next asearch
        # re-fetches and surfaces the new memory. Keep _last_good around -
        # it's still useful as a 429 fallback until the next successful
        # asearch overwrites it.
        prefix = op.namespace
        self._search_cache = {
            k: v for k, v in self._search_cache.items() if k[0] != prefix
        }

    # ------------------------------------------------------------------ #
    # SEARCH                                                             #
    # ------------------------------------------------------------------ #

    # Memanto's recall caps `limit` at 100 server-side AND ranks by semantic
    # similarity to a single query string. That makes "list everything in
    # this namespace" non-trivial: when the agent contains many memories
    # across other namespaces, a single recall call (even at the 100 cap)
    # rarely surfaces every memory belonging to one specific namespace.
    #
    # Diagnostic example (real numbers from this codebase): 4 memories
    # were stored under user_id 'bob-X'. With query='*' only 2 came back;
    # with query='email' only 2 (different overlap); with query='peanut'
    # only 1. The UNION of multiple semantic anchors finally returned all 4.
    #
    # So when a namespace filter is in play we fan out across a handful of
    # diverse anchor queries, dedupe by memory_id, and apply strict AND
    # namespace matching client-side.
    _MEMANTO_RECALL_CAP = 100
    # Diverse semantic anchors that, taken as a union, broadly cover the
    # categories of facts a typical agent stores about a user. Each anchor
    # is one network round-trip to Memanto, so we keep the set TIGHT (4
    # queries) to stay under the Community-plan rate limit. The set was
    # empirically tuned to surface all 4 facts of the demo (name, email,
    # allergy, phone instruction) - dropping any anchor regresses one fact.
    _NS_ANCHOR_QUERIES = (
        "user identity name profile",
        "user contact email phone address",
        "user allergies health food restrictions",
        "user instructions rules preferences goals",
    )

    # In-memory cache of asearch results, keyed by (namespace, query, limit).
    # Memanto's Community-plan rate limit (~60-100 recall calls/min) is easy
    # to exceed without caching because Streamlit's UI panel polls memories
    # every rerun, and each poll fans out across all anchors. Cache TTL is
    # short (30 s) so newly-stored memories appear quickly via the same path
    # _poll_for_memories uses to wait for indexing.
    _CACHE_TTL_S = 30.0

    def _do_search(self, op: SearchOp) -> list[SearchItem]:
        """Retrieve memories matching the namespace.

        When a namespace filter is present we always fan out across diverse
        semantic anchors and union the matching results, because a single
        recall (even at Memanto's 100-result cap) is biased by semantic
        similarity to one query and frequently misses memories that exist
        in the namespace. The caller's query runs first so its top results
        lead the output; anchors fill in the rest of the namespace.

        Cost: ~8-10 s per asearch with a namespace filter. Acceptable for
        a demo where the LLM call itself takes 20+ s per turn. If you need
        sub-second recall in production, narrow the anchor set or scope
        the agent_id per-user so there's nothing else to compete with.
        """
        query = op.query or "*"
        filter_dict = op.filter or {}
        ns_tags = self._namespace_to_tags(op.namespace_prefix)

        type_filter = filter_dict.get("type") or filter_dict.get("kind")
        if isinstance(type_filter, str):
            type_filter = [type_filter]

        extra_tags = list(filter_dict.get("tags", []) or [])
        min_conf = filter_dict.get("min_confidence")

        # Cache hit? Returning a recent result avoids hammering Memanto's
        # rate limit when the UI polls or multiple graph nodes search the
        # same namespace in quick succession.
        cache_key = (op.namespace_prefix, query, op.limit, tuple(extra_tags))
        cached = self._search_cache.get(cache_key)
        if cached is not None:
            ts, items = cached
            if time.time() - ts < self._CACHE_TTL_S:
                return items

        # When namespace-filtering, fan out across anchors so the union
        # surfaces every memory in the namespace. The caller's query
        # runs first so its semantic ranking is honoured for whatever
        # memories share a strong dimension with it; anchors then fill
        # in the rest.
        if ns_tags:
            queries = [query] + [q for q in self._NS_ANCHOR_QUERIES if q != query]
            fetch_limit = self._MEMANTO_RECALL_CAP
        else:
            queries = [query]
            fetch_limit = max(1, min(op.limit, self._MEMANTO_RECALL_CAP))

        seen_ids: set[str] = set()
        out: list[SearchItem] = []
        rate_limited = False
        for q in queries:
            if len(out) >= op.limit:
                break
            try:
                result = self._client.recall(
                    agent_id=self._agent_id,
                    query=q,
                    limit=fetch_limit,
                    type=type_filter,
                    tags=ns_tags + extra_tags if (ns_tags or extra_tags) else None,
                    min_confidence=min_conf,
                )
            except Exception as e:  # pragma: no cover - resilience
                logger.warning("MemantoStore: recall(%r) failed: %s", q, e)
                # Stop fanning out as soon as we hit a non-retryable error
                # (rate limit, auth failure, session expired). More anchor
                # calls will hit the same error and just flood the logs;
                # the last-good fallback below preserves the UI state.
                err = str(e)
                if any(
                    marker in err
                    for marker in (
                        "429",
                        "Limit Exceeded",
                        "Forbidden",
                        "Unauthorized",
                        "401",
                        "403",
                    )
                ):
                    rate_limited = True
                    break
                continue

            for mem in result.get("memories", []):
                mem_id = mem.get("id")
                if mem_id and mem_id in seen_ids:
                    continue
                tags = mem.get("tags", []) or []
                # Strict AND-match on namespace - Memanto's tag filter is
                # OR-matching server-side, so this is where isolation is
                # actually enforced.
                if ns_tags and not all(t in tags for t in ns_tags):
                    continue
                if extra_tags and not all(t in tags for t in extra_tags):
                    continue
                if mem_id:
                    seen_ids.add(mem_id)
                namespace = self._tags_to_namespace(tags) or op.namespace_prefix
                key = self._tags_to_key(tags) or mem.get("id", "")
                out.append(self._memory_to_search_item(mem, namespace, key))
                if len(out) >= op.limit:
                    break

        # If we got nothing this round AND the last call hit the rate
        # limit AND we have a stored result from a previous successful
        # call: return that. This is what makes the UI memory panel
        # survive 429s without flashing to zero.
        if not out and rate_limited and op.namespace_prefix in self._last_good:
            logger.info(
                "MemantoStore: rate-limited, returning last-good result for %r",
                op.namespace_prefix,
            )
            return self._last_good[op.namespace_prefix]

        # Cache successful (non-empty) results so subsequent rapid-fire
        # asearches don't burn more rate-limit budget.
        if out and not rate_limited:
            self._search_cache[cache_key] = (time.time(), out)
            self._last_good[op.namespace_prefix] = out

        return out

    # ------------------------------------------------------------------ #
    # LIST NAMESPACES (best-effort)                                      #
    # ------------------------------------------------------------------ #

    def _do_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        # Sample generously (at least 200) so we cover all namespaces, then
        # truncate the *output* to op.limit. op.limit controls what the caller
        # sees, not how many memories we query.
        sample_limit = max(op.limit or 0, 200)
        sample = self._client.recall(
            agent_id=self._agent_id,
            query="*",
            limit=sample_limit,
        )
        seen: set[tuple[str, ...]] = set()
        for mem in sample.get("memories", []):
            tags = mem.get("tags", []) or []
            ns = self._tags_to_namespace(tags)
            if ns:
                seen.add(ns)

        result = sorted(seen)
        if op.max_depth is not None:
            result = [ns[: op.max_depth] for ns in result]
            result = sorted(set(result))
        if op.limit:
            result = result[: op.limit]
        return result

    # ------------------------------------------------------------------ #
    # Encoding helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _namespace_to_tags(namespace: tuple[str, ...]) -> list[str]:
        """Encode a namespace tuple as a list of reserved tags.

        ``("user-bob", "memories")`` -> ``["lg:ns:0:user-bob", "lg:ns:1:memories"]``
        """
        return [f"{_NS_TAG_PREFIX}{i}:{part}" for i, part in enumerate(namespace)]

    @staticmethod
    def _key_to_tag(key: str) -> str:
        return f"{_KEY_TAG_PREFIX}{key}"

    @staticmethod
    def _tags_to_namespace(tags: list[str]) -> tuple[str, ...]:
        """Reverse ``_namespace_to_tags``. Returns () if none present."""
        positioned: dict[int, str] = {}
        for t in tags:
            if not t.startswith(_NS_TAG_PREFIX):
                continue
            rest = t[len(_NS_TAG_PREFIX) :]
            idx_str, _, value = rest.partition(":")
            try:
                positioned[int(idx_str)] = value
            except ValueError:
                continue
        if not positioned:
            return ()
        return tuple(positioned[i] for i in sorted(positioned))

    @staticmethod
    def _tags_to_key(tags: list[str]) -> str | None:
        for t in tags:
            if t.startswith(_KEY_TAG_PREFIX):
                return t[len(_KEY_TAG_PREFIX) :]
        return None

    @staticmethod
    def _stringify(value: dict[str, Any]) -> str:
        if not value:
            return "(empty)"
        return ", ".join(f"{k}={v}" for k, v in value.items())

    # ------------------------------------------------------------------ #
    # Item construction                                                  #
    # ------------------------------------------------------------------ #

    def _memory_to_item(
        self, mem: dict[str, Any], namespace: tuple[str, ...], key: str
    ) -> Item:
        return Item(
            value=self._memory_to_value(mem),
            key=key,
            namespace=namespace,
            created_at=self._parse_dt(mem.get("created_at")),
            updated_at=self._parse_dt(mem.get("updated_at") or mem.get("created_at")),
        )

    def _memory_to_search_item(
        self, mem: dict[str, Any], namespace: tuple[str, ...], key: str
    ) -> SearchItem:
        score = mem.get("score")
        if score is None:
            score = mem.get("similarity")
        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None

        return SearchItem(
            value=self._memory_to_value(mem),
            key=key,
            namespace=namespace,
            created_at=self._parse_dt(mem.get("created_at")),
            updated_at=self._parse_dt(mem.get("updated_at") or mem.get("created_at")),
            score=score_f,
        )

    @staticmethod
    def _memory_to_value(mem: dict[str, Any]) -> dict[str, Any]:
        tags = mem.get("tags", []) or []
        user_tags = [t for t in tags if not t.startswith(_RESERVED_PREFIX)]
        return {
            "kind": mem.get("type", "fact"),
            "title": mem.get("title", ""),
            "content": mem.get("content", ""),
            "confidence": mem.get("confidence"),
            "tags": user_tags,
            "memory_id": mem.get("id"),
        }

    @staticmethod
    def _parse_dt(raw: Any) -> datetime:
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                pass
        return datetime.now(tz=timezone.utc)
