"""Shared ingestion contracts and coordinate conversion helpers."""

from abc import ABC, abstractmethod
from pathlib import Path

from src.canonical.model import BoundingBox, DocumentCanonicalRepresentation


def pdf_to_pixel_bbox(bbox: BoundingBox, zoom: float) -> tuple[int, int, int, int]:
    """Convert a PDF box to an enclosing rendered-pixel box."""
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
    """Convert a rendered-pixel box back to canonical PDF coordinates."""
    if zoom <= 0:
        raise ValueError("zoom must be positive")
    x0, y0, x1, y1 = bbox
    return BoundingBox(x0=x0 / zoom, y0=y0 / zoom, x1=x1 / zoom, y1=y1 / zoom)


class FormatAdapter(ABC):
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Return whether this adapter can ingest the path."""

    @abstractmethod
    def extract_canonical(
        self, file_path: Path, output_dir: Path
    ) -> DocumentCanonicalRepresentation:
        """Extract a deterministic canonical representation."""
