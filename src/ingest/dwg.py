"""Contract placeholder for a future DWG adapter."""

from pathlib import Path

from src.canonical.model import DocumentCanonicalRepresentation
from src.ingest.base import FormatAdapter


class DwgAdapter(FormatAdapter):
    def can_handle(self, file_path: Path) -> bool:
        return False

    def extract_canonical(
        self, file_path: Path, output_dir: Path
    ) -> DocumentCanonicalRepresentation:
        raise NotImplementedError("DWG ingestion is not implemented")
