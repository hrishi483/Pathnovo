import pytest
from pydantic import ValidationError

from src.canonical.model import BoundingBox
from src.ingest.base import pdf_to_pixel_bbox, pixel_to_pdf_bbox


def test_bounding_box_serialization() -> None:
    box = BoundingBox(x0=10.25, y0=20.5, x1=30.75, y1=40.0)
    assert box.model_dump() == {
        "x0": 10.25,
        "y0": 20.5,
        "x1": 30.75,
        "y1": 40.0,
    }


def test_invalid_bounding_box_is_rejected() -> None:
    with pytest.raises(ValidationError):
        BoundingBox(x0=20, y0=0, x1=10, y1=10)


def test_coordinate_conversion_round_trip() -> None:
    original = BoundingBox(x0=10.0, y0=15.0, x1=30.0, y1=40.0)
    pixels = pdf_to_pixel_bbox(original, zoom=2.0)
    assert pixels == (20, 30, 60, 80)
    assert pixel_to_pdf_bbox(pixels, zoom=2.0) == original


def test_coordinate_conversion_rejects_invalid_zoom() -> None:
    with pytest.raises(ValueError):
        pdf_to_pixel_bbox(BoundingBox(x0=0, y0=0, x1=1, y1=1), 0)
