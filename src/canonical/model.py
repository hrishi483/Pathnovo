"""Format-agnostic, loss-aware representation of one document page."""

from __future__ import annotations

import math
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BoundingBox(BaseModel):
    """Axis-aligned box in top-left-origin PDF page coordinates."""

    model_config = ConfigDict(extra="forbid")

    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_bounds(self) -> BoundingBox:
        values = (self.x0, self.y0, self.x1, self.y1)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Bounding-box coordinates must be finite")
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("Bounding-box lower-right must follow its upper-left")
        return self

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


class TextSource(str, Enum):
    PDF_NATIVE = "pdf_native"
    OCR = "ocr"


class GeometrySource(str, Enum):
    PDF_NATIVE = "pdf_native"


class ResidualSource(str, Enum):
    PDF_NATIVE_RESIDUAL = "pdf_native_residual"


class GeometryType(str, Enum):
    LINE = "line"
    RECTANGLE = "rectangle"
    CURVE = "curve"
    POLYLINE = "polyline"
    PATH = "path"


class TextElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    bbox: BoundingBox
    font_name: str | None = None
    font_size: float | None = None
    font_flags: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: TextSource | str


class GeometryElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    geometry_type: GeometryType
    bbox: BoundingBox
    coordinates: dict[str, Any] | list[dict[str, Any]]
    stroke_width: float | None = Field(default=None, ge=0.0)
    stroke_color: tuple[float, ...] | None = None
    fill_color: tuple[float, ...] | None = None
    layer: str | None = None
    source: GeometrySource | str


class ResidualImageRegion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    bbox: BoundingBox
    image_path: str
    pixel_width: int = Field(gt=0)
    pixel_height: int = Field(gt=0)
    source: ResidualSource | str


class DocumentCanonicalRepresentation(BaseModel):
    """Canonical representation for a single page.

    The three arrays are the only content layers. Metadata contains provenance
    and extraction settings, never page content.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_format: str
    page_width: float = Field(gt=0.0)
    page_height: float = Field(gt=0.0)
    text_elements: list[TextElement] = Field(default_factory=list)
    geometry_elements: list[GeometryElement] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ensure_elements_are_on_page(self) -> DocumentCanonicalRepresentation:
        tolerance = 1e-3
        elements = [
            *self.text_elements,
            *self.geometry_elements,
        ]
        for element in elements:
            box = element.bbox
            if (
                box.x0 < -tolerance
                or box.y0 < -tolerance
                or box.x1 > self.page_width + tolerance
                or box.y1 > self.page_height + tolerance
            ):
                raise ValueError(f"{element.id} bounding box lies outside the PDF page")
        return self
