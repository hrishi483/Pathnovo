"""Contract placeholder for a future scanned-PDF/OCR adapter."""

from pathlib import Path

from src.canonical.model import DocumentCanonicalRepresentation
from src.ingest.base import FormatAdapter


class ScannedPdfAdapter(FormatAdapter):
    @property
    def extensions(self) -> frozenset[str]:
        # Reserved for a future OCR path; never matches while unimplemented.
        return frozenset()

    def can_handle(self, file_path: Path) -> bool:
        return False

    def extract_canonical(
        self, file_path: Path, output_dir: Path
    ) -> DocumentCanonicalRepresentation:
        raise NotImplementedError("Scanned PDF ingestion is not implemented")
