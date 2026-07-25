from pathlib import Path

import ezdxf
import numpy as np
from PIL import Image

from src.ingest.base import AdapterRegistry
from src.ingest.dwg import DwgAdapter, DwgConfig
from src.ingest.pdf_native import NativePdfAdapter


def _write_sample_dxf(path: Path) -> Path:
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_line((0, 0), (100, 0))
    msp.add_line((100, 0), (100, 50))
    msp.add_circle((50, 25), radius=10)
    msp.add_lwpolyline([(10, 10), (40, 10), (40, 30), (10, 30)], close=True)
    msp.add_text("ROOM-A", dxfattribs={"height": 5}).set_placement((15, 40))
    msp.add_text("DOOR-1", dxfattribs={"height": 3}).set_placement((60, 5))
    doc.saveas(path)
    return path


def test_adapter_registry_resolves_by_extension(tmp_path: Path) -> None:
    registry = AdapterRegistry([NativePdfAdapter(), DwgAdapter()])
    dxf = tmp_path / "sample.dxf"
    dxf.write_text("0\nEOF\n")
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"%PDF-1.4")

    assert isinstance(registry.resolve(dxf), DwgAdapter)
    assert isinstance(registry.resolve(pdf), NativePdfAdapter)
    assert registry.resolve_for_filename("rev.DWG").extensions == frozenset(
        {".dwg", ".dxf"}
    )


def test_dwg_adapter_extracts_canonical_and_pngs(tmp_path: Path) -> None:
    dxf_path = _write_sample_dxf(tmp_path / "sample.dxf")
    adapter = DwgAdapter(DwgConfig(zoom=2, max_render_edge=2048))
    canonical = adapter.extract_canonical(dxf_path, tmp_path / "artifacts")

    assert canonical.source_format == "dxf"
    assert canonical.page_width > 0
    assert canonical.page_height > 0
    assert any(element.text == "ROOM-A" for element in canonical.text_elements)
    assert any(element.text == "DOOR-1" for element in canonical.text_elements)
    assert canonical.geometry_elements
    assert {"line", "polyline"}.issubset(
        {element.geometry_type.value for element in canonical.geometry_elements}
    )

    for element in [*canonical.text_elements, *canonical.geometry_elements]:
        assert 0 <= element.bbox.x0 <= element.bbox.x1 <= canonical.page_width
        assert 0 <= element.bbox.y0 <= element.bbox.y1 <= canonical.page_height
        assert element.source == "dwg"

    artifacts = canonical.metadata["artifacts"]
    assert {"original_render", "geometry_mask", "overlay"} <= set(artifacts)
    for path in artifacts.values():
        assert Path(path).is_file()

    artifact_dir = tmp_path / "artifacts" / canonical.document_id
    assert (artifact_dir / "text_elements.json").is_file()
    assert (artifact_dir / "geometry_elements.json").is_file()

    geometry_mask = np.array(Image.open(artifacts["geometry_mask"]))
    assert geometry_mask.ndim == 2
    assert np.count_nonzero(geometry_mask) > 0
