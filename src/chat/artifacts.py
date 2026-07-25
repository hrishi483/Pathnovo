"""Load real baseline, revision, and delta artifacts into the chat index."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from src.chat.ingest import register_entity
from src.chat.models import BBox, ChangeRecord, ChangeType
from src.chat.store import DeltaStore

PAIR_PATTERN = re.compile(r"^results-(?P<baseline>[a-f0-9]{24})-(?P<revision>[a-f0-9]{24})$")

ENTITY_KEYWORDS = (
    "pump",
    "valve",
    "compressor",
    "vessel",
    "turbine",
    "instrument",
    "pipeline",
    "pipe",
    "line",
    "motor",
    "filter",
    "separator",
    "cooler",
    "heater",
    "tank",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _bbox(raw: Any) -> Optional[BBox]:
    if isinstance(raw, dict):
        values = (raw.get("x0"), raw.get("y0"), raw.get("x1"), raw.get("y1"))
    elif isinstance(raw, (list, tuple)) and len(raw) >= 4:
        values = tuple(raw[:4])
    else:
        return None
    if not all(isinstance(value, (int, float)) for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _infer_entity_type(text: str) -> str:
    normalized = text.casefold()
    for keyword in ENTITY_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}s?\b", normalized):
            return keyword
    return "text"


def _aliases(text: str, element_id: str, entity_type: str) -> list[str]:
    aliases = [element_id]
    if entity_type != "text":
        aliases.append(entity_type)
    aliases.extend(
        token
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", text)
        if any(character.isdigit() for character in token)
    )
    return list(dict.fromkeys(aliases))


def _group_text_elements(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstruct searchable labels when PDF extraction emits single glyphs."""
    positioned = []
    for element in elements:
        box = _bbox(element.get("bbox"))
        text = str(element.get("text") or "").strip()
        if box and text:
            positioned.append((element, box, text))
    positioned.sort(key=lambda item: ((item[1][1] + item[1][3]) / 2, item[1][0]))

    lines: list[list[tuple[dict[str, Any], BBox, str]]] = []
    line_centers: list[float] = []
    for item in positioned:
        _, box, _ = item
        center_y = (box[1] + box[3]) / 2
        height = max(1.0, box[3] - box[1])
        best_index = None
        best_distance = float("inf")
        for index, line_center in enumerate(line_centers):
            distance = abs(center_y - line_center)
            if distance <= max(1.5, height * 0.35) and distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is None:
            lines.append([item])
            line_centers.append(center_y)
        else:
            lines[best_index].append(item)
            count = len(lines[best_index])
            line_centers[best_index] = (
                line_centers[best_index] * (count - 1) + center_y
            ) / count

    groups: list[dict[str, Any]] = []
    group_index = 0
    for line in lines:
        line.sort(key=lambda item: item[1][0])
        current: list[tuple[dict[str, Any], BBox, str]] = []
        previous_box: Optional[BBox] = None
        for item in line:
            _, box, _ = item
            gap = box[0] - previous_box[2] if previous_box else 0.0
            typical_height = max(1.0, box[3] - box[1])
            if current and gap > max(8.0, typical_height * 1.8):
                groups.append(_make_text_group(current, group_index))
                group_index += 1
                current = []
            current.append(item)
            previous_box = box
        if current:
            groups.append(_make_text_group(current, group_index))
            group_index += 1
    return [group for group in groups if group["element_count"] >= 2]


def _make_text_group(
    items: list[tuple[dict[str, Any], BBox, str]], group_index: int
) -> dict[str, Any]:
    parts: list[str] = []
    previous_box: Optional[BBox] = None
    for _, box, text in items:
        gap = box[0] - previous_box[2] if previous_box else 0.0
        separator = " " if previous_box and gap > 1.5 else ""
        parts.append(separator + text)
        previous_box = box
    boxes = [item[1] for item in items]
    return {
        "id": f"text_group_{group_index:06d}",
        "text": "".join(parts),
        "bbox": (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ),
        "element_count": len(items),
    }


def discover_pairs(artifact_root: Path) -> list[dict[str, Any]]:
    """List available baseline/revision pairs under ``delta/results-*``."""
    artifact_root = Path(artifact_root).resolve()
    pairs: list[dict[str, Any]] = []
    delta_root = artifact_root / "delta"
    if not delta_root.is_dir():
        return pairs
    for directory in delta_root.glob("results-*"):
        if not directory.is_dir():
            continue
        match = PAIR_PATTERN.match(directory.name)
        if not match:
            continue
        baseline_id = match.group("baseline")
        revision_id = match.group("revision")
        pair_id = f"{baseline_id}-{revision_id}"
        pairs.append(
            {
                "document_pair_id": pair_id,
                "baseline_document_id": baseline_id,
                "revision_document_id": revision_id,
                "updated_at": directory.stat().st_mtime,
            }
        )
    pairs.sort(key=lambda item: item["updated_at"], reverse=True)
    return pairs


def _discover_pair(artifact_root: Path) -> tuple[str, str, Path]:
    candidates: list[tuple[float, str, str, Path]] = []
    delta_root = artifact_root / "delta"
    for directory in delta_root.glob("results-*"):
        if not directory.is_dir():
            continue
        match = PAIR_PATTERN.match(directory.name)
        if match:
            candidates.append(
                (
                    directory.stat().st_mtime,
                    match.group("baseline"),
                    match.group("revision"),
                    directory,
                )
            )
    if not candidates:
        raise FileNotFoundError(f"No delta results directory found under {delta_root}")
    _, baseline_id, revision_id, result_dir = max(candidates)
    return baseline_id, revision_id, result_dir


def _resolve_pair(
    artifact_root: Path, document_pair_id: Optional[str]
) -> tuple[str, str, str, Path]:
    if document_pair_id:
        try:
            baseline_id, revision_id = document_pair_id.split("-", 1)
        except ValueError as exc:
            raise ValueError(
                "document_pair_id must be '<baseline_id>-<revision_id>'"
            ) from exc
        result_dir = artifact_root / "delta" / f"results-{document_pair_id}"
        if not result_dir.is_dir():
            raise FileNotFoundError(f"Delta directory does not exist: {result_dir}")
    else:
        baseline_id, revision_id, result_dir = _discover_pair(artifact_root)
        document_pair_id = f"{baseline_id}-{revision_id}"
    return document_pair_id, baseline_id, revision_id, result_dir


def _load_document_entities(
    store: DeltaStore,
    artifact_root: Path,
    document_pair_id: str,
    document_id: str,
    version: str,
) -> None:
    document_dir = artifact_root / version / document_id
    text_path = document_dir / "text_elements.json"
    geometry_path = document_dir / "geometry_elements.json"

    text_elements = _read_json(text_path)
    for element in text_elements:
        box = _bbox(element.get("bbox"))
        text = str(element.get("text") or "").strip()
        element_id = str(element.get("id") or "")
        if not box or not text or not element_id:
            continue
        entity_type = _infer_entity_type(text)
        register_entity(
            store,
            document_pair_id,
            name=text,
            type_=entity_type,
            page=1,
            bbox=box,
            aliases=_aliases(text, element_id, entity_type),
            document_version=version,
            source_kind="text",
            source_element_id=element_id,
        )

    for group in _group_text_elements(text_elements):
        text = group["text"].strip()
        if len(text) < 3:
            continue
        entity_type = _infer_entity_type(text)
        register_entity(
            store,
            document_pair_id,
            name=text,
            type_=entity_type,
            page=1,
            bbox=group["bbox"],
            aliases=_aliases(text, group["id"], entity_type),
            document_version=version,
            source_kind="text_group",
            source_element_id=group["id"],
        )

    for element in _read_json(geometry_path):
        box = _bbox(element.get("bbox"))
        element_id = str(element.get("id") or "")
        if not box or not element_id:
            continue
        geometry_type = str(element.get("geometry_type") or "geometry")
        layer = str(element.get("layer") or "unlayered")
        register_entity(
            store,
            document_pair_id,
            name=f"{layer} {geometry_type} {element_id}",
            type_=geometry_type.removeprefix("GeometryType.").casefold(),
            page=1,
            bbox=box,
            aliases=[element_id, layer, geometry_type],
            document_version=version,
            source_kind="geometry",
            source_element_id=element_id,
        )


def _linked_entity(
    store: DeltaStore,
    document_pair_id: str,
    source_kind: str,
    change: dict[str, Any],
) -> tuple[Optional[str], Optional[str]]:
    if change.get("revision_b"):
        version, side = "revision", change["revision_b"]
    elif change.get("revision_a"):
        version, side = "baseline", change["revision_a"]
    else:
        return None, None
    source_id = side.get("id")
    if not source_id:
        return None, None
    entity = store.get_entity_by_source(
        document_pair_id, version, source_kind, str(source_id)
    )
    if not entity:
        return None, None
    return entity.entity_id, entity.name


def _load_text_changes(
    store: DeltaStore, document_pair_id: str, report: dict[str, Any]
) -> None:
    for raw in report.get("changes", []):
        event = str(raw.get("change_type") or "unknown")
        if event == "unchanged":
            continue
        side_a = raw.get("revision_a") or {}
        side_b = raw.get("revision_b") or {}
        before = side_a.get("text")
        after = side_b.get("text")
        entity_id, entity_name = _linked_entity(
            store, document_pair_id, "text", raw
        )
        store.add_change(
            ChangeRecord(
                document_pair_id=document_pair_id,
                change_type=ChangeType.TEXT,
                page=1,
                entity_id=entity_id,
                entity_name=entity_name,
                bbox=_bbox(side_b.get("bbox")) or _bbox(side_a.get("bbox")),
                old_state=before,
                new_state=after,
                text_delta=f"{before or '(none)'} -> {after or '(none)'}",
                event=event,
                confidence=raw.get("confidence"),
            )
        )


def _load_geometry_changes(
    store: DeltaStore, document_pair_id: str, report: dict[str, Any]
) -> None:
    for raw in report.get("changes", []):
        event = str(raw.get("change_type") or "unknown")
        if event == "unchanged":
            continue
        side_a = raw.get("revision_a") or {}
        side_b = raw.get("revision_b") or {}
        entity_id, entity_name = _linked_entity(
            store, document_pair_id, "geometry", raw
        )
        store.add_change(
            ChangeRecord(
                document_pair_id=document_pair_id,
                change_type=ChangeType.GEOMETRY,
                page=1,
                entity_id=entity_id,
                entity_name=entity_name,
                bbox=_bbox(side_b.get("bbox")) or _bbox(side_a.get("bbox")),
                old_state=side_a.get("id"),
                new_state=side_b.get("id"),
                geometry_delta={
                    "event": event,
                    "baseline_type": side_a.get("geometry_type"),
                    "revision_type": side_b.get("geometry_type"),
                    "match": raw.get("match"),
                },
                event=event,
                confidence=raw.get("confidence"),
            )
        )


def _load_vlm_changes(
    store: DeltaStore,
    document_pair_id: str,
    report: dict[str, Any],
    page_width: float,
    page_height: float,
) -> None:
    for raw in report.get("changes", []):
        normalized = _bbox(raw.get("bounding_box"))
        box = (
            (
                normalized[0] * page_width,
                normalized[1] * page_height,
                normalized[2] * page_width,
                normalized[3] * page_height,
            )
            if normalized
            else None
        )
        description = str(raw.get("description") or "")
        location = str(raw.get("location_description") or "")
        event = str(raw.get("change_type") or "unknown")
        source_id = str(raw.get("change_id") or "vlm_change")
        semantic_entities = []
        if box:
            normalized_description = description.casefold()
            version = "baseline" if event == "removed" else "revision"
            for keyword in ENTITY_KEYWORDS:
                if re.search(
                    rf"\b{re.escape(keyword)}s?\b", normalized_description
                ):
                    semantic_entities.append(
                        register_entity(
                            store,
                            document_pair_id,
                            name=f"{keyword.title()} (visual region: {location})",
                            type_=keyword,
                            page=1,
                            bbox=box,
                            aliases=[keyword, source_id],
                            document_version=version,
                            source_kind="vlm",
                            source_element_id=f"{source_id}:{keyword}",
                        )
                    )
        store.add_change(
            ChangeRecord(
                document_pair_id=document_pair_id,
                change_type=ChangeType.VLM,
                page=1,
                entity_id=(
                    semantic_entities[0].entity_id if semantic_entities else None
                ),
                entity_name=semantic_entities[0].name if semantic_entities else None,
                bbox=box,
                vlm_delta=" ".join(part for part in (description, location) if part),
                related_entity_ids=[
                    entity.entity_id for entity in semantic_entities[1:]
                ],
                event=event,
                confidence=raw.get("confidence"),
            )
        )


def build_store_from_artifacts(
    artifact_root: Path,
    document_pair_id: Optional[str] = None,
) -> tuple[DeltaStore, str]:
    """Build an in-memory, version-aware index from existing JSON artifacts."""
    artifact_root = Path(artifact_root).resolve()
    pair_id, baseline_id, revision_id, result_dir = _resolve_pair(
        artifact_root, document_pair_id
    )
    store = DeltaStore()
    store.artifact_root = str(artifact_root)
    _load_document_entities(
        store, artifact_root, pair_id, baseline_id, "baseline"
    )
    _load_document_entities(
        store, artifact_root, pair_id, revision_id, "revision"
    )

    text_report = _read_json(result_dir / f"{pair_id}_text.json")
    geometry_report = _read_json(result_dir / f"{pair_id}_geometry.json")
    vlm_report = _read_json(result_dir / f"{pair_id}_vlm_comparison.json")
    _load_text_changes(store, pair_id, text_report)
    _load_geometry_changes(store, pair_id, geometry_report)
    _load_vlm_changes(
        store,
        pair_id,
        vlm_report,
        float(text_report["revision_b"]["page_width"]),
        float(text_report["revision_b"]["page_height"]),
    )
    return store, pair_id
