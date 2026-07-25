"""Utilities for discovering and safely loading generated report artifacts."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPORT_PATTERN = re.compile(
    r"^(?P<run_id>[a-f0-9]{24}-[a-f0-9]{24})_(?P<kind>text|geometry|vlm(?:_comparison|_with_difference_analysis)?)\.json$"
)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class ReportEntry:
    id: str
    name: str
    kind: str
    run_id: str
    relative_path: str
    mtime: float

    def to_api_dict(self) -> dict[str, Any]:
        baseline_id, revision_id = self.run_id.split("-", 1)
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "run_id": self.run_id,
            "baseline_document_id": baseline_id,
            "revision_document_id": revision_id,
            "updated_at": self.mtime,
        }


def _safe_resolve(root: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("Path must not be empty")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("Absolute paths are not allowed")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise ValueError("Path escapes the artifact directory")
    return resolved


def _report_id(relative_path: str) -> str:
    digest = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:10]
    stem = Path(relative_path).stem
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", stem).strip("-")
    return f"{safe_stem}-{digest}"


def _normalize_kind(raw_kind: str) -> str:
    if raw_kind == "vlm_comparison":
        return "vlm_comparison"
    if raw_kind.startswith("vlm"):
        return "vlm"
    return raw_kind


def _candidate_report_files(artifact_root: Path) -> list[Path]:
    results_root = artifact_root / "/delta/results"
    files = list(results_root.glob("*.json"))
    for directory in artifact_root.glob("delta/results-*"):
        if directory.is_dir():
            files.extend(directory.glob("*.json"))
    return files


def discover_reports(artifact_root: Path) -> dict[str, Any]:
    entries = discover_report_entries(artifact_root)

    entries.sort(key=lambda entry: entry.mtime, reverse=True)
    reports_payload = [entry.to_api_dict() for entry in entries]

    runs_map: dict[str, dict[str, Any]] = {}
    for entry in entries:
        run = runs_map.setdefault(
            entry.run_id,
            {
                "id": entry.run_id,
                "baseline_document_id": os.path.join("baseline", entry.run_id.split("-", 1)[0]),
                "revision_document_id": os.path.join("revision", entry.run_id.split("-", 1)[1]),
                "updated_at": entry.mtime,
                "reports": {
                    "text": None,
                    "geometry": None,
                    "vlm": None,
                    "vlm_comparison": None,
                },
            },
        )
        run["updated_at"] = max(run["updated_at"], entry.mtime)
        if entry.kind == "vlm_comparison":
            run["reports"]["vlm_comparison"] = entry.id
        elif entry.kind == "vlm":
            run["reports"]["vlm"] = entry.id
        elif entry.kind in run["reports"]:
            run["reports"][entry.kind] = entry.id

    runs = sorted(runs_map.values(), key=lambda run: run["updated_at"], reverse=True)
    return {"reports": reports_payload, "runs": runs}


def resolve_report_file(artifact_root: Path, report_id: str) -> Path:
    for entry in discover_report_entries(artifact_root):
        if entry.id == report_id:
            return _safe_resolve(artifact_root, entry.relative_path)
    raise FileNotFoundError(f"Unknown report_id: {report_id}")


def discover_report_entries(artifact_root: Path) -> list[ReportEntry]:
    entries: list[ReportEntry] = []
    report_files = _candidate_report_files(artifact_root)
    artifact_resolved = artifact_root.resolve()

    for report_path in report_files:
        match = REPORT_PATTERN.match(report_path.name)
        if not match:
            continue
        run_id = match.group("run_id")
        kind = _normalize_kind(match.group("kind"))
        relative_path = str(report_path.resolve().relative_to(artifact_resolved))
        stat = report_path.stat()
        entries.append(
            ReportEntry(
                id=_report_id(relative_path),
                name=report_path.name,
                kind=kind,
                run_id=run_id,
                relative_path=relative_path,
                mtime=stat.st_mtime,
            )
        )

    entries.sort(key=lambda entry: entry.mtime, reverse=True)
    return entries


def resolve_image_file(artifact_root: Path, image_path: str) -> Path:
    resolved = _safe_resolve(artifact_root, image_path)
    if resolved.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Only image files are supported")
    return resolved
