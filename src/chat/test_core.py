"""Run with: python -m pytest delta_agent/tests -q  (from /home/claude)"""

from pathlib import Path

from src.chat.artifacts import build_store_from_artifacts
from src.chat.demo import build_example_store
from src.chat.ingest import register_entity
from src.chat.models import ChangeRecord, ChangeType
from src.chat.tools import (
    dispatch_tool,
    find_entities,
    get_entity_changes,
    get_entity_neighborhood,
    get_nearby_changes,
    get_overall_changes,
)

ARTIFACT_ROOT = Path(__file__).resolve().parents[2] / "data" / "artifacts"


def test_find_entities_matches_valve_alias():
    store, pair_id = build_example_store()
    results = find_entities(store, pair_id, "valve")
    assert results
    assert results[0]["entity_name"] == "Valve V-101"
    assert results[0]["confidence"] == 1.0


def test_find_entities_no_match_returns_empty():
    store, pair_id = build_example_store()
    results = find_entities(store, pair_id, "compressor")
    assert results == []


def test_nearby_changes_includes_all_three_evidence_types():
    store, pair_id = build_example_store()
    valve = find_entities(store, pair_id, "V-101")[0]
    nearby = get_nearby_changes(store, valve["entity_id"], radius=100)
    types = {c["type"] for c in nearby["nearby_changes"]}
    assert types == {"text", "geometry", "vlm"}


def test_nearby_changes_radius_shrinks_result_set():
    store, pair_id = build_example_store()
    valve = find_entities(store, pair_id, "V-101")[0]
    wide = get_nearby_changes(store, valve["entity_id"], radius=100)
    narrow = get_nearby_changes(store, valve["entity_id"], radius=0)
    assert len(narrow["nearby_changes"]) <= len(wide["nearby_changes"])


def test_get_entity_changes_filters_by_type():
    store, pair_id = build_example_store()
    valve = find_entities(store, pair_id, "V-101")[0]
    geo_only = get_entity_changes(store, valve["entity_id"], change_type="geometry")
    assert all(c["type"] == "geometry" for c in geo_only["changes"])
    assert geo_only["changes"]


def test_unknown_entity_id_returns_error():
    store, _ = build_example_store()
    result = get_nearby_changes(store, "entity_9999")
    assert "error" in result


def test_find_entities_respects_document_version():
    store, pair_id = build_example_store()
    register_entity(
        store,
        pair_id,
        name="Pump P-201",
        type_="pump",
        page=4,
        bbox=(300, 300, 340, 340),
        document_version="revision",
    )
    assert find_entities(
        store, pair_id, "pump", document_version="baseline"
    ) == []
    revision_results = find_entities(
        store, pair_id, "pump", document_version="revision"
    )
    assert revision_results[0]["entity_name"] == "Pump P-201"


def test_short_glyph_does_not_match_long_entity_query():
    store, pair_id = build_example_store()
    register_entity(
        store,
        pair_id,
        name="P",
        type_="text",
        page=4,
        bbox=(300, 300, 305, 305),
        document_version="revision",
    )
    assert find_entities(
        store, pair_id, "pump", document_version="revision"
    ) == []


def test_entity_neighborhood_returns_entities_and_changes():
    store, pair_id = build_example_store()
    valve = find_entities(store, pair_id, "valve")[0]
    result = get_entity_neighborhood(
        store, valve["entity_id"], radius=100, entity_limit=10, change_limit=10
    )
    assert result["nearby_entities"]
    assert result["nearby_changes"]


def test_unlocated_unlinked_change_does_not_leak_into_neighborhood():
    store, pair_id = build_example_store()
    valve = find_entities(store, pair_id, "valve")[0]
    orphan = store.add_change(
        ChangeRecord(
            document_pair_id=pair_id,
            change_type=ChangeType.TEXT,
            page=4,
            bbox=None,
            text_delta="Unrelated unlocated text",
        )
    )
    result = get_nearby_changes(store, valve["entity_id"], radius=100)
    ids = {change["change_id"] for change in result["nearby_changes"]}
    assert orphan.change_id not in ids


def test_invalid_tool_change_type_returns_error():
    store, _ = build_example_store()
    result = dispatch_tool(
        store,
        "get_nearby_changes",
        {"entity_id": "entity_0001", "change_type": "not-a-type"},
    )
    assert "error" in result


def test_get_overall_changes_reads_report_summaries():
    if not ARTIFACT_ROOT.is_dir():
        return
    store, pair_id = build_store_from_artifacts(ARTIFACT_ROOT)
    result = get_overall_changes(store, pair_id)
    assert "error" not in result
    kinds = {source["kind"] for source in result["sources"]}
    # Text + geometry are always present; VLM is optional for CAD pairs.
    assert {"text", "geometry"} <= kinds
    if "vlm_comparison" in kinds:
        vlm = next(s for s in result["sources"] if s["kind"] == "vlm_comparison")
        assert vlm["summary"]["overall_summary"]
        assert vlm["changes"]


def test_get_overall_changes_requires_artifact_root():
    store, pair_id = build_example_store()
    result = get_overall_changes(store, pair_id)
    assert "error" in result
