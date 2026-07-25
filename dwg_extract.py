"""CLI entrypoint for CAD text/geometry extraction.

Prefer: python -m src.ingest.cad_extract <file.dwg|.dxf>
"""

from __future__ import annotations

from src.ingest.cad_extract import extract, load_document, main, open

__all__ = ["extract", "load_document", "main", "open"]


if __name__ == "__main__":
    raise SystemExit(main())
