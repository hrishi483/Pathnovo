"""
Core data models for the document-delta agent.

These implement the "unified delta data model" described in the design doc:
every change, regardless of whether it came from the text-diff pipeline,
the geometry pipeline, or the VLM comparison pipeline, is represented as a
single ChangeRecord that is linked to a document_pair / page / entity /
location. This is what lets the agent combine evidence from all three
sources instead of searching them independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import itertools

_id_counter = itertools.count(1)


def _next_id(prefix: str) -> str:
    return f"{prefix}_{next(_id_counter):04d}"


BBox = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def expand_bbox(bbox: BBox, radius: float) -> BBox:
    """Expand a bounding box by `radius` on every side."""
    x1, y1, x2, y2 = bbox
    return (x1 - radius, y1 - radius, x2 + radius, y2 + radius)


def bbox_intersects(a: BBox, b: BBox) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def bbox_distance(a: BBox, b: BBox) -> float:
    """Approximate center-to-center distance between two boxes."""
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class ChangeType(str, Enum):
    TEXT = "text"
    GEOMETRY = "geometry"
    VLM = "vlm"


@dataclass
class Entity:
    """A named object located on a page of a document pair (e.g. a valve)."""

    document_pair_id: str
    name: str
    type: str
    page: int
    bbox: BBox
    entity_id: str = field(default_factory=lambda: _next_id("entity"))
    aliases: list[str] = field(default_factory=list)
    document_version: str = "revision"
    source_kind: str = "text"
    source_element_id: Optional[str] = None

    def matches(self, query: str) -> float:
        """Very small fuzzy-match scorer used by find_entities.

        Returns a confidence score in [0, 1]. Real implementations should
        swap this for embedding similarity / a proper search index.
        """
        q = query.strip().casefold()
        if not q:
            return 0.0
        candidates = [
            self.name.casefold(),
            self.type.casefold(),
            *[alias.casefold() for alias in self.aliases],
        ]
        for c in candidates:
            if q == c:
                return 1.0
        for c in candidates:
            if len(q) >= 3 and len(c) >= 3 and (q in c or c in q):
                return 0.85
        # token overlap fallback
        q_tokens = {token for token in q.split() if len(token) >= 2}
        for c in candidates:
            c_tokens = {token for token in c.split() if len(token) >= 2}
            if q_tokens & c_tokens:
                return 0.6
        return 0.0


@dataclass
class ChangeRecord:
    """Unified representation of a single detected change.

    Mirrors the schema in section 4 of the design doc. `entity_id` and
    `related_entity_ids` are what allow text/geometry/vlm changes about the
    same object to be joined together and retrieved as one bundle of
    evidence.
    """

    document_pair_id: str
    change_type: ChangeType
    page: int
    change_id: str = field(default_factory=lambda: _next_id("change"))

    entity_id: Optional[str] = None
    entity_name: Optional[str] = None

    bbox: Optional[BBox] = None  # location this change applies to

    old_state: Optional[str] = None
    new_state: Optional[str] = None

    text_delta: Optional[str] = None
    geometry_delta: Optional[dict] = None
    vlm_delta: Optional[str] = None

    related_entity_ids: list[str] = field(default_factory=list)
    event: Optional[str] = None
    confidence: Optional[float] = None

    def summary(self) -> str:
        if self.change_type == ChangeType.TEXT:
            prefix = f"{self.event}: " if self.event else ""
            return prefix + (self.text_delta or "")
        if self.change_type == ChangeType.GEOMETRY:
            g = self.geometry_delta or {}
            if g.get("event") == "added":
                return f"{self.entity_name or 'Object'} was added."
            if g.get("event") == "removed":
                return f"{self.entity_name or 'Object'} was removed."
            dx, dy = g.get("dx"), g.get("dy")
            if dx is not None:
                direction = "right" if dx > 0 else "left"
                return f"{self.entity_name or 'Object'} moved {abs(dx)}px {direction}" + (
                    f", {abs(dy)}px {'down' if dy > 0 else 'up'}" if dy else ""
                )
            return f"{self.entity_name or 'Object'} geometry changed."
        if self.change_type == ChangeType.VLM:
            prefix = f"{self.event}: " if self.event else ""
            return prefix + (self.vlm_delta or "")
        return ""
