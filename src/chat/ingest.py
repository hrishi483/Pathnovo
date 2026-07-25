"""
Step 1 (inspect the existing delta outputs) + Step 2 (create a common
ChangeRecord) from the design doc, wired together: these functions take the
raw output of each of the three existing pipelines and normalize it into
ChangeRecords in the DeltaStore, linked to Entities.

Adjust the field names in `raw` dicts to match your actual pipeline output;
these mirror the example payloads from the design doc.
"""

from __future__ import annotations

from typing import Optional

from src.chat.models import BBox, ChangeRecord, ChangeType, Entity
from src.chat.store import DeltaStore


def register_entity(
    store: DeltaStore,
    document_pair_id: str,
    name: str,
    type_: str,
    page: int,
    bbox: BBox,
    aliases: Optional[list[str]] = None,
    document_version: str = "revision",
    source_kind: str = "text",
    source_element_id: Optional[str] = None,
) -> Entity:
    """Create (or reuse) an Entity. In a real pipeline this would come from
    the entity/object-detection step that already tags things like
    'Valve V-101' with a page and bounding box."""
    existing = [
        e for e in store.entities_for_pair(document_pair_id)
        if e.name.lower() == name.lower()
        and e.page == page
        and e.document_version == document_version
        and e.source_element_id == source_element_id
    ]
    if existing:
        return existing[0]
    entity = Entity(
        document_pair_id=document_pair_id,
        name=name,
        type=type_,
        page=page,
        bbox=bbox,
        aliases=aliases or [],
        document_version=document_version,
        source_kind=source_kind,
        source_element_id=source_element_id,
    )
    return store.add_entity(entity)


def ingest_text_delta(
    store: DeltaStore,
    document_pair_id: str,
    page: int,
    text_delta: str,
    entity: Optional[Entity] = None,
    related_entities: Optional[list[Entity]] = None,
    bbox: Optional[BBox] = None,
) -> ChangeRecord:
    """Normalize one delta_text record, e.g.:
    'Connection of Valve V-101 changed from Line L-20 to Line L-25.'
    """
    change = ChangeRecord(
        document_pair_id=document_pair_id,
        change_type=ChangeType.TEXT,
        page=page,
        entity_id=entity.entity_id if entity else None,
        entity_name=entity.name if entity else None,
        bbox=bbox or (entity.bbox if entity else None),
        text_delta=text_delta,
        related_entity_ids=[e.entity_id for e in (related_entities or [])],
    )
    return store.add_change(change)


def ingest_geometry_delta(
    store: DeltaStore,
    document_pair_id: str,
    page: int,
    entity: Entity,
    old_position: Optional[tuple[float, float]] = None,
    new_position: Optional[tuple[float, float]] = None,
    old_bbox: Optional[BBox] = None,
    new_bbox: Optional[BBox] = None,
    event: str = "moved",  # "moved" | "added" | "removed"
) -> ChangeRecord:
    """Normalize one delta_geometry record, e.g.:
    {"object": "Valve V-101", "old_position": [120, 340],
     "new_position": [180, 340], "movement": {"dx": 60, "dy": 0}}
    """
    geometry_delta: dict = {"event": event}
    if old_position and new_position:
        dx = new_position[0] - old_position[0]
        dy = new_position[1] - old_position[1]
        geometry_delta.update({"dx": dx, "dy": dy})

    change = ChangeRecord(
        document_pair_id=document_pair_id,
        change_type=ChangeType.GEOMETRY,
        page=page,
        entity_id=entity.entity_id,
        entity_name=entity.name,
        bbox=new_bbox or entity.bbox,
        old_state=str(old_bbox) if old_bbox else None,
        new_state=str(new_bbox) if new_bbox else None,
        geometry_delta=geometry_delta,
    )
    return store.add_change(change)


def ingest_vlm_delta(
    store: DeltaStore,
    document_pair_id: str,
    page: int,
    vlm_text: str,
    bbox: BBox,
    entity: Optional[Entity] = None,
    related_entities: Optional[list[Entity]] = None,
) -> ChangeRecord:
    """Normalize one delta_vlm_comparison record, e.g.:
    'The valve appears to have been relocated to the right side of the
    pipeline. A new bypass connection is visible near the valve.'
    """
    change = ChangeRecord(
        document_pair_id=document_pair_id,
        change_type=ChangeType.VLM,
        page=page,
        entity_id=entity.entity_id if entity else None,
        entity_name=entity.name if entity else None,
        bbox=bbox,
        vlm_delta=vlm_text,
        related_entity_ids=[e.entity_id for e in (related_entities or [])],
    )
    return store.add_change(change)
