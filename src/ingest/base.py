"""Shared ingestion contracts and coordinate conversion helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.canonical.model import BoundingBox, DocumentCanonicalRepresentation


def pdf_to_pixel_bbox(bbox: BoundingBox, zoom: float) -> tuple[int, int, int, int]:
    """Convert a page box to an enclosing rendered-pixel box."""
    if zoom <= 0:
        raise ValueError("zoom must be positive")
    return (
        round(bbox.x0 * zoom),
        round(bbox.y0 * zoom),
        round(bbox.x1 * zoom),
        round(bbox.y1 * zoom),
    )


def pixel_to_pdf_bbox(
    bbox: tuple[int, int, int, int], zoom: float
) -> BoundingBox:
    """Convert a rendered-pixel box back to canonical page coordinates."""
    if zoom <= 0:
        raise ValueError("zoom must be positive")
    x0, y0, x1, y1 = bbox
    return BoundingBox(x0=x0 / zoom, y0=y0 / zoom, x1=x1 / zoom, y1=y1 / zoom)


class FormatAdapter(ABC):
    @property
    @abstractmethod
    def extensions(self) -> frozenset[str]:
        """Lowercase file extensions this adapter handles (including the dot)."""

    def can_handle(self, file_path: Path) -> bool:
        """Return whether this adapter can ingest the path."""
        path = Path(file_path)
        return path.is_file() and path.suffix.lower() in self.extensions

    @abstractmethod
    def extract_canonical(
        self, file_path: Path, output_dir: Path
    ) -> DocumentCanonicalRepresentation:
        """Extract a deterministic canonical representation."""


class AdapterRegistry:
    """Resolve an ingest adapter from a file path by extension / can_handle."""

    def __init__(self, adapters: Sequence[FormatAdapter] | None = None) -> None:
        self._adapters: list[FormatAdapter] = list(adapters or [])

    def register(self, adapter: FormatAdapter) -> None:
        self._adapters.append(adapter)

    @property
    def adapters(self) -> tuple[FormatAdapter, ...]:
        return tuple(self._adapters)

    def supported_extensions(self) -> frozenset[str]:
        extensions: set[str] = set()
        for adapter in self._adapters:
            extensions.update(adapter.extensions)
        return frozenset(extensions)

    def resolve(self, file_path: Path) -> FormatAdapter:
        path = Path(file_path)
        for adapter in self._adapters:
            if adapter.can_handle(path):
                return adapter
        supported = ", ".join(sorted(self.supported_extensions())) or "(none)"
        suffix = path.suffix.lower() or "(none)"
        raise ValueError(
            f"No ingest adapter for extension {suffix!r}. Supported: {supported}"
        )

    def resolve_for_filename(self, filename: str) -> FormatAdapter:
        """Resolve using only the filename suffix (file need not exist yet)."""
        suffix = Path(filename or "").suffix.lower()
        for adapter in self._adapters:
            if suffix in adapter.extensions:
                return adapter
        supported = ", ".join(sorted(self.supported_extensions())) or "(none)"
        raise ValueError(
            f"No ingest adapter for extension {suffix or '(none)'!r}. "
            f"Supported: {supported}"
        )


def write_element_json(path: Path, elements: Iterable[object]) -> None:
    """Serialize Pydantic elements to a JSON array file."""
    import json

    path.write_text(
        json.dumps(
            [
                element.model_dump(mode="json")  # type: ignore[attr-defined]
                for element in elements
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
