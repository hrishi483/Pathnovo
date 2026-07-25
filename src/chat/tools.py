from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
from langsmith import traceable
from .models import ChangeType
from .store import DeltaStore
 
 
def _parse_change_type(value: Optional[str]) -> Optional[ChangeType]:
    if value is None:
        return None
    try:
        return ChangeType(value)
    except ValueError as exc:
        valid = ", ".join(change_type.value for change_type in ChangeType)
        raise ValueError(
            f"Invalid change_type {value!r}; expected one of: {valid}"
        ) from exc

@traceable(
    name="Find Entities",
    run_type="tool",
)
def find_entities(
    store: DeltaStore,
    document_pair_id: str,
    query: str,
    min_confidence: float = 0.5,
    document_version: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """Find entities in a document pair matching a natural-language query
    such as 'valve' or 'V-101'."""
    matches = store.find_entities(
        document_pair_id,
        query,
        min_confidence=min_confidence,
        document_version=document_version,
        limit=limit,
    )
    return [
        {
            "entity_id": e.entity_id,
            "entity_name": e.name,
            "entity_type": e.type,
            "page": e.page,
            "bbox": list(e.bbox),
            "confidence": round(score, 2),
            "document_version": e.document_version,
            "source_kind": e.source_kind,
            "source_element_id": e.source_element_id,
        }
        for e, score in matches
    ]

@traceable(
    name="Get Nearby Entities",
    run_type="tool",
)
def get_nearby_entities(
    store: DeltaStore,
    entity_id: str,
    radius: float = 100.0,
    limit: int = 50,
    source_kind: Optional[str] = None,
) -> dict:
    """Return baseline/revision entities spatially near the selected entity."""
    entity = store.get_entity(entity_id)
    if entity is None:
        return {"error": f"No entity with id {entity_id}"}
    nearby = store.nearby_entities(
        entity, radius=radius, limit=limit, source_kind=source_kind
    )
    return {
        "entity": _serialize_entity(entity),
        "document_version": entity.document_version,
        "search_radius": radius,
        "nearby_entities": [_serialize_entity(candidate) for candidate in nearby],
    }

@traceable(
    name="Get Entity Neighborhood",
    run_type="tool",
)
def get_entity_neighborhood(
    store: DeltaStore,
    entity_id: str,
    radius: float = 100.0,
    entity_limit: int = 30,
    change_limit: int = 100,
) -> dict:
    """Return nearby document entities and nearby delta evidence together."""
    entity = store.get_entity(entity_id)
    if entity is None:
        return {"error": f"No entity with id {entity_id}"}
    nearby_entities = store.nearby_entities(
        entity, radius=radius, limit=entity_limit
    )
    nearby_changes = store.nearby_changes(
        entity, radius=radius, limit=change_limit
    )
    return {
        "entity": _serialize_entity(entity),
        "document_version": entity.document_version,
        "search_radius": radius,
        "nearby_entities": [_serialize_entity(candidate) for candidate in nearby_entities],
        "nearby_changes": [_serialize_change(change) for change in nearby_changes],
    }
 
@traceable(
    name="Get Nearby Changes",
    run_type="tool",
)
def get_nearby_changes(
    store: DeltaStore,
    entity_id: str,
    radius: float = 100.0,
    change_type: Optional[str] = None,
    limit: int = 100,
) -> dict:
    """Get all text/geometry/vlm changes spatially near an entity (the
    entity's bounding box expanded by `radius`). This is the primary tool
    for 'what changed around X?' questions."""
    entity = store.get_entity(entity_id)
    if entity is None:
        return {"error": f"No entity with id {entity_id}"}
 
    ct = _parse_change_type(change_type)
    changes = store.nearby_changes(
        entity, radius=radius, change_type=ct, limit=limit
    )
 
    return {
        "entity": {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "type": entity.type,
            "page": entity.page,
            "bbox": list(entity.bbox),
        },
        "search_radius": radius,
        "nearby_changes": [_serialize_change(c) for c in changes],
    }
 
@traceable(
    name="Get Entity Changes",
    run_type="tool",
)
def get_entity_changes(store: DeltaStore, entity_id: str, change_type: Optional[str] = None) -> dict:
    """Get all changes directly tied to a specific entity (its own moves,
    text mentions, vlm descriptions), independent of spatial radius."""
    entity = store.get_entity(entity_id)
    if entity is None:
        return {"error": f"No entity with id {entity_id}"}
 
    ct = _parse_change_type(change_type)
    changes = store.changes_for_entity(entity_id, change_type=ct)
 
    return {
        "entity": {
            "entity_id": entity.entity_id,
            "name": entity.name,
            "type": entity.type,
        },
        "changes": [_serialize_change(c) for c in changes],
    }
 
 
def _serialize_change(c) -> dict:
    return {
        "change_id": c.change_id,
        "type": c.change_type.value,
        "entity_name": c.entity_name,
        "page": c.page,
        "bbox": list(c.bbox) if c.bbox else None,
        "summary": c.summary(),
        "text_delta": c.text_delta,
        "geometry_delta": c.geometry_delta,
        "vlm_delta": c.vlm_delta,
        "event": c.event,
        "confidence": c.confidence,
    }


def _serialize_entity(entity) -> dict:
    return {
        "entity_id": entity.entity_id,
        "entity_name": entity.name,
        "entity_type": entity.type,
        "page": entity.page,
        "bbox": list(entity.bbox),
        "document_version": entity.document_version,
        "source_kind": entity.source_kind,
        "source_element_id": entity.source_element_id,
    }


def _read_report_summary(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report.get("summary")
    if summary is None and isinstance(report.get("vlm_analysis"), dict):
        summary = report["vlm_analysis"].get("summary")
    payload: dict[str, Any] = {
        "source_file": path.name,
        "summary": summary,
    }
    # VLM reports carry a short narrative change list that is useful for
    # global "what changed overall?" answers without dumping full deltas.
    if path.name.endswith("_vlm_comparison.json"):
        payload["changes"] = [
            {
                "change_id": raw.get("change_id"),
                "change_type": raw.get("change_type"),
                "description": raw.get("description"),
                "location_description": raw.get("location_description"),
                "confidence": raw.get("confidence"),
            }
            for raw in report.get("changes", [])
        ]
    return payload

@traceable(
    name="Get Overall Changes",
    run_type="tool",
)
def get_overall_changes(store: DeltaStore, document_pair_id: str) -> dict:
    """Read text, geometry, and VLM report summaries for a document pair.

    Walks ``artifacts/delta/results-{pair_id}/`` and returns each report's
    summary so the agent can synthesize a global change answer.
    """
    if not store.artifact_root:
        return {
            "error": (
                "Overall change summaries require a store loaded from on-disk "
                "artifacts (artifact_root is unset)."
            )
        }

    result_dir = (
        Path(store.artifact_root) / "delta" / f"results-{document_pair_id}"
    )
    if not result_dir.is_dir():
        return {
            "error": f"Delta results directory not found: {result_dir}",
            "document_pair_id": document_pair_id,
        }

    report_kinds = (
        "text",
        "geometry",
        "vlm_comparison",
        "vlm_with_difference_analysis",
    )
    sources: list[dict[str, Any]] = []
    missing: list[str] = []
    for kind in report_kinds:
        path = result_dir / f"{document_pair_id}_{kind}.json"
        if not path.is_file():
            missing.append(path.name)
            continue
        try:
            payload = _read_report_summary(path)
        except (OSError, json.JSONDecodeError) as exc:
            sources.append(
                {
                    "kind": kind,
                    "source_file": path.name,
                    "error": str(exc),
                }
            )
            continue
        payload["kind"] = kind
        sources.append(payload)

    if not sources:
        return {
            "error": "No readable delta report summaries found",
            "document_pair_id": document_pair_id,
            "missing": missing,
        }

    return {
        "document_pair_id": document_pair_id,
        "result_dir": str(result_dir),
        "sources": sources,
        "missing": missing,
    }


# ---- Tool schemas for LLM tool-calling: TOOL_SCHEMAS (generic/Anthropic-style) and
# GEMINI_TOOL_DECLARATIONS (derived below, for google-genai) ----

TOOL_SCHEMAS = [
    {
        "name": "find_entities",
        "description": (
            "Find entities (e.g. valves, lines, pumps) in a document pair that "
            "match a natural-language query. Use this first to resolve which "
            "specific object the user is asking about. Returns a list of "
            "candidates with a confidence score; if more than one plausible "
            "match is returned, disambiguate using the rest of the question "
            "or ask the user."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "document_pair_id": {"type": "string"},
                "query": {"type": "string", "description": "e.g. 'valve', 'V-101', 'pump near the valve'"},
                "min_confidence": {"type": "number", "default": 0.5},
                "document_version": {
                    "type": "string",
                    "enum": ["baseline", "revision"],
                    "description": (
                        "Restrict search to the requested diagram version. "
                        "Use revision for revised/new/current diagram questions."
                    ),
                },
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["document_pair_id", "query"],
        },
    },
    {
        "name": "get_nearby_entities",
        "description": (
            "Return entities spatially near an already resolved entity in the "
            "same baseline or revision diagram. Use for questions such as "
            "'what is near Pump in the revised diagram?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "radius": {"type": "number", "default": 100},
                "limit": {"type": "integer", "default": 50},
                "source_kind": {
                    "type": "string",
                    "enum": ["text", "geometry"],
                },
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "get_entity_neighborhood",
        "description": (
            "Return both nearby baseline/revision entities and nearby text, "
            "geometry, and VLM delta evidence. Prefer this for broad questions "
            "about what is near or changed around a resolved entity."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "radius": {"type": "number", "default": 100},
                "entity_limit": {"type": "integer", "default": 30},
                "change_limit": {"type": "integer", "default": 100},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "get_nearby_changes",
        "description": (
            "Given an entity_id, return all text, geometry, and VLM changes "
            "found spatially near that entity (within `radius`). This is the "
            "primary tool for questions like 'what changed around X?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "radius": {"type": "number", "default": 100},
                "change_type": {
                    "type": "string",
                    "enum": ["text", "geometry", "vlm"],
                    "description": "Optional filter to only one evidence source.",
                },
                "limit": {"type": "integer", "default": 100},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "get_entity_changes",
        "description": (
            "Given an entity_id, return all changes directly associated with "
            "that entity (not spatially expanded). Use this when the question "
            "is about one specific, already-identified object rather than its "
            "surroundings, e.g. 'what changed about V-101 itself?'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string"},
                "change_type": {"type": "string", "enum": ["text", "geometry", "vlm"]},
            },
            "required": ["entity_id"],
        },
    },
    {
        "name": "get_overall_changes",
        "description": (
            "Return document-level change summaries by reading the text, "
            "geometry, and VLM delta reports for a document pair. Use this for "
            "global questions such as 'what changed overall?', 'what are the "
            "major differences?', or 'where are the biggest changes?'. Does "
            "not require an entity_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "document_pair_id": {
                    "type": "string",
                    "description": "Baseline-revision pair id, e.g. '<baseline>-<revision>'.",
                },
            },
            "required": ["document_pair_id"],
        },
    },
]
 
 
def _strip_unsupported_schema_keys(schema):
    """Gemini's function-declaration schema only supports a subset of
    OpenAPI (see notes_and_limitations in the Gemini function-calling docs).
    `default` is not part of that subset, so strip it recursively when
    building Gemini declarations from the Anthropic-style TOOL_SCHEMAS."""
    if isinstance(schema, dict):
        return {
            k: _strip_unsupported_schema_keys(v)
            for k, v in schema.items()
            if k != "default"
        }
    if isinstance(schema, list):
        return [_strip_unsupported_schema_keys(v) for v in schema]
    return schema
 
 
# Gemini (google-genai) function declarations: same shape as TOOL_SCHEMAS but
# with `input_schema` renamed to `parameters` and unsupported keys stripped,
# per https://ai.google.dev/gemini-api/docs/function-calling#function-declarations
GEMINI_TOOL_DECLARATIONS = [
    {
        "name": t["name"],
        "description": t["description"],
        "parameters": _strip_unsupported_schema_keys(t["input_schema"]),
    }
    for t in TOOL_SCHEMAS
]
 
 
def dispatch_tool(store: DeltaStore, name: str, tool_input: dict):
    """Route a tool_use block from the LLM to the matching Python function."""
    try:
        if name == "find_entities":
            return find_entities(store, **tool_input)
        if name == "get_nearby_changes":
            return get_nearby_changes(store, **tool_input)
        if name == "get_nearby_entities":
            return get_nearby_entities(store, **tool_input)
        if name == "get_entity_neighborhood":
            return get_entity_neighborhood(store, **tool_input)
        if name == "get_entity_changes":
            return get_entity_changes(store, **tool_input)
        if name == "get_overall_changes":
            return get_overall_changes(store, **tool_input)
        return {"error": f"Unknown tool: {name}"}
    except (TypeError, ValueError) as exc:
        return {"error": str(exc), "tool": name}
 