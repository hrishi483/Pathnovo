"""
In-memory index over Entities and ChangeRecords for one or more document
pairs. This is the "DocumentPair -> Entity -> {Text, Geometry, VLM} Delta"
structure from section 14 of the design doc, plus the spatial neighbourhood
query from section 15 / Step 4.

Swap this out for a real vector/spatial index (e.g. Postgres + PostGIS,
or an embedding store) once the data volume grows; the query surface
(`find_entities`, `get_nearby_changes`, `get_entity_changes`) should stay
the same so the agent tools built on top of it don't need to change.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

from src.chat.models import ChangeRecord, ChangeType, Entity, bbox_distance, bbox_intersects, expand_bbox


class DeltaStore:
    def __init__(self) -> None:
        self._entities: dict[str, Entity] = {}
        self._changes: dict[str, ChangeRecord] = {}
        # Set when the store is built from on-disk compare artifacts.
        self.artifact_root: Optional[str] = None

        # indices
        self._entities_by_pair: dict[str, list[str]] = defaultdict(list)
        self._changes_by_entity: dict[str, list[str]] = defaultdict(list)
        self._changes_by_pair_page: dict[tuple[str, int], list[str]] = defaultdict(list)
        self._entities_by_source: dict[tuple[str, str, str, str], str] = {}

    # ---- ingestion -----------------------------------------------------

    def add_entity(self, entity: Entity) -> Entity:
        self._entities[entity.entity_id] = entity
        self._entities_by_pair[entity.document_pair_id].append(entity.entity_id)
        if entity.source_element_id:
            key = (
                entity.document_pair_id,
                entity.document_version,
                entity.source_kind,
                entity.source_element_id,
            )
            self._entities_by_source[key] = entity.entity_id
        return entity

    def add_change(self, change: ChangeRecord) -> ChangeRecord:
        self._changes[change.change_id] = change
        if change.entity_id:
            self._changes_by_entity[change.entity_id].append(change.change_id)
        for rel_id in change.related_entity_ids:
            self._changes_by_entity[rel_id].append(change.change_id)
        self._changes_by_pair_page[(change.document_pair_id, change.page)].append(change.change_id)
        return change

    # ---- lookups ---------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def entities_for_pair(
        self, document_pair_id: str, document_version: Optional[str] = None
    ) -> list[Entity]:
        entities = [
            self._entities[eid]
            for eid in self._entities_by_pair.get(document_pair_id, [])
        ]
        if document_version:
            entities = [
                entity
                for entity in entities
                if entity.document_version == document_version
            ]
        return entities

    def find_entities(
        self,
        document_pair_id: str,
        query: str,
        min_confidence: float = 0.5,
        document_version: Optional[str] = None,
        limit: int = 20,
    ) -> list[tuple[Entity, float]]:
        """Fuzzy entity search scoped to a document pair. Returns (entity, confidence),
        sorted by confidence descending."""
        results = []
        for entity in self.entities_for_pair(document_pair_id, document_version):
            score = entity.matches(query)
            if score >= min_confidence:
                results.append((entity, score))
        results.sort(key=lambda t: t[1], reverse=True)
        return results[:limit]

    def get_entity_by_source(
        self,
        document_pair_id: str,
        document_version: str,
        source_kind: str,
        source_element_id: str,
    ) -> Optional[Entity]:
        entity_id = self._entities_by_source.get(
            (
                document_pair_id,
                document_version,
                source_kind,
                source_element_id,
            )
        )
        return self._entities.get(entity_id) if entity_id else None

    def nearby_entities(
        self,
        entity: Entity,
        radius: float = 100.0,
        limit: int = 50,
        source_kind: Optional[str] = None,
    ) -> list[Entity]:
        expanded = expand_bbox(entity.bbox, radius)
        candidates = self.entities_for_pair(
            entity.document_pair_id, entity.document_version
        )
        results = [
            candidate
            for candidate in candidates
            if candidate.entity_id != entity.entity_id
            and (source_kind is None or candidate.source_kind == source_kind)
            and bbox_intersects(candidate.bbox, expanded)
        ]
        results.sort(key=lambda candidate: bbox_distance(candidate.bbox, entity.bbox))
        return results[:limit]

    def changes_for_entity(self, entity_id: str, change_type: Optional[ChangeType] = None) -> list[ChangeRecord]:
        changes = [self._changes[cid] for cid in self._changes_by_entity.get(entity_id, [])]
        if change_type:
            changes = [c for c in changes if c.change_type == change_type]
        return changes

    def nearby_changes(
        self,
        entity: Entity,
        radius: float = 100.0,
        change_type: Optional[ChangeType] = None,
        exclude_entity_own_changes: bool = False,
        limit: int = 100,
    ) -> list[ChangeRecord]:
        """Spatial neighbourhood query: all changes on the same page whose
        bbox intersects the expanded bounding box around `entity`, plus any
        change directly tied to the entity's own id even if it has no bbox
        (e.g. a pure text delta)."""
        expanded = expand_bbox(entity.bbox, radius)
        page_key = (entity.document_pair_id, entity.page)
        direct_ids = set(self._changes_by_entity.get(entity.entity_id, []))
        candidate_ids = set(self._changes_by_pair_page.get(page_key, []))
        candidate_ids |= direct_ids

        results: list[ChangeRecord] = []
        for cid in candidate_ids:
            change = self._changes[cid]
            if exclude_entity_own_changes and change.entity_id == entity.entity_id:
                continue
            if change.change_type == change_type or change_type is None:
                if (
                    change.bbox is not None
                    and bbox_intersects(change.bbox, expanded)
                ) or cid in direct_ids:
                    results.append(change)

        def sort_key(c: ChangeRecord):
            if c.bbox is None:
                return 0.0
            return bbox_distance(c.bbox, entity.bbox)

        results.sort(key=sort_key)
        return results[:limit]
