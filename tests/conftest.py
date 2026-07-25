from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    raster_path = tmp_path / "raster.png"
    raster = Image.new("RGB", (40, 40), "white")
    ImageDraw.Draw(raster).ellipse((4, 4, 35, 35), fill="black")
    raster.save(raster_path)

    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text((30, 35), "26-PIT-9055", fontsize=10)
    page.draw_line((25, 70), (180, 70), width=1.5, color=(0, 0, 0))
    page.draw_rect((25, 90, 120, 130), width=1, color=(0, 0, 0))
    page.insert_image((210, 120, 250, 160), filename=str(raster_path))
    document.save(pdf_path)
    document.close()
    return pdf_path
