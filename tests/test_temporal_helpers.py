from datetime import timezone

from memanto.app.routes.memory import RecallRequest
from memanto.app.services.memory_read_service import MemoryReadService
from memanto.app.utils.temporal_helpers import (
    build_temporal_query,
    parse_iso_timestamp,
)


def test_parse_iso_timestamp_normalizes_offset_to_utc():
    parsed = parse_iso_timestamp("2026-01-15T08:30:00-05:00")

    assert parsed.tzinfo == timezone.utc
    assert parsed.isoformat() == "2026-01-15T13:30:00+00:00"


def test_parse_iso_timestamp_assumes_naive_values_are_utc():
    parsed = parse_iso_timestamp("2026-01-15T13:30:00")

    assert parsed.tzinfo == timezone.utc
    assert parsed.isoformat() == "2026-01-15T13:30:00+00:00"


def test_temporal_filter_skips_malformed_memory_timestamps_per_record():
    service = MemoryReadService(object())

    results = [
        {"id": "old", "created_at": "2025-01-01T00:00:00Z"},
        {"id": "new", "created_at": "2025-02-01T00:00:00Z"},
        {"id": "bad", "created_at": "not-a-timestamp"},
    ]

    filtered = service._apply_temporal_filter(
        results, created_after="2025-01-15T00:00:00Z"
    )

    assert [memory["id"] for memory in filtered] == ["new"]


def test_temporal_query_payload_is_accepted_by_recall_request_model():
    payload = build_temporal_query(
        "http://localhost:8000",
        "agent-1",
        "deployment notes",
        relative_time="last 7 days",
    )["json"]

    request = RecallRequest.model_validate(payload)

    assert payload["created_after"] is not None
    assert request.created_after is not None
