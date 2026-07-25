from pathlib import Path

import fitz
import numpy as np

from src.ingest.pdf_native import NativePdfAdapter, NativePdfConfig


def test_text_and_geometry_extraction(sample_pdf: Path) -> None:
    adapter = NativePdfAdapter()
    with fitz.open(sample_pdf) as document:
        text = adapter.extract_text(document[0])
        geometry = adapter.extract_geometry(document[0])

    assert any(element.text == "26-PIT-9055" for element in text)
    assert all(element.bbox.width >= 0 and element.bbox.height >= 0 for element in text)
    assert {"line", "rectangle"}.issubset(
        {element.geometry_type.value for element in geometry}
    )
    assert all(
        element.bbox.width >= 0 and element.bbox.height >= 0
        for element in geometry
    )


def test_mask_creation(sample_pdf: Path) -> None:
    adapter = NativePdfAdapter(NativePdfConfig(zoom=2, cleanup_kernel_size=0))
    with fitz.open(sample_pdf) as document:
        page = document[0]
        image = adapter.render_page(page)
        text = adapter.extract_text(page)
        geometry = adapter.extract_geometry(page)

    text_mask = adapter.create_text_mask(image.shape[:2], text)
    geometry_mask = adapter.create_geometry_mask(image.shape[:2], geometry)
    assert text_mask.shape == image.shape[:2]
    assert geometry_mask.shape == image.shape[:2]
    assert np.count_nonzero(text_mask) > 0
    assert np.count_nonzero(geometry_mask) > 0


def test_residual_generation_subtracts_known_content() -> None:
    adapter = NativePdfAdapter(NativePdfConfig(cleanup_kernel_size=0))
    original = np.full((40, 40, 3), 255, dtype=np.uint8)
    original[5:15, 5:15] = 0
    original[25:35, 25:35] = 0
    text_mask = np.zeros((40, 40), dtype=np.uint8)
    text_mask[5:15, 5:15] = 255
    geometry_mask = np.zeros((40, 40), dtype=np.uint8)

    residual_mask, residual_image = adapter.create_residual(
        original, text_mask, geometry_mask
    )
    assert np.count_nonzero(residual_mask[5:15, 5:15]) == 0
    assert np.all(residual_image[5:15, 5:15] == 255)
    assert np.count_nonzero(residual_mask[25:35, 25:35]) == 100
    assert np.all(residual_image[25:35, 25:35] == 0)


def test_native_pdf_adapter(sample_pdf: Path, tmp_path: Path) -> None:
    adapter = NativePdfAdapter(
        NativePdfConfig(zoom=2, min_residual_area=20, cleanup_kernel_size=2)
    )
    canonical = adapter.extract_canonical(sample_pdf, tmp_path / "artifacts")

    assert canonical.page_width == 300
    assert canonical.page_height == 200
    assert canonical.text_elements
    assert canonical.geometry_elements

    all_elements = [
        *canonical.text_elements,
        *canonical.geometry_elements,
    ]
    for element in all_elements:
        assert 0 <= element.bbox.x0 <= element.bbox.x1 <= canonical.page_width
        assert 0 <= element.bbox.y0 <= element.bbox.y1 <= canonical.page_height

    artifacts = canonical.metadata["artifacts"]
    expected = {
        "original_render",
        "geometry_mask",
        "overlay",
    }
    assert expected == set(artifacts)
    assert all(Path(path).is_file() for path in artifacts.values())
