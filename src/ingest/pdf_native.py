"""Deterministic native-PDF adapter built on PyMuPDF."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
from PIL import Image

from src.canonical.model import (
    BoundingBox,
    DocumentCanonicalRepresentation,
    GeometryElement,
    GeometryType,
    ResidualImageRegion,
    TextElement,
)
from src.ingest.base import FormatAdapter, pdf_to_pixel_bbox, pixel_to_pdf_bbox


@dataclass(frozen=True)
class NativePdfConfig:
    zoom: float = 2.0
    min_residual_area: int = 64
    region_padding: int = 4
    text_mask_padding: int = 1
    geometry_mask_padding: int = 1
    cleanup_kernel_size: int = 2
    foreground_threshold: int = 245
    save_overlay: bool = True

    def __post_init__(self) -> None:
        if self.zoom <= 0:
            raise ValueError("zoom must be positive")
        if self.min_residual_area < 1:
            raise ValueError("min_residual_area must be at least 1")
        if self.region_padding < 0:
            raise ValueError("region_padding cannot be negative")
        if self.cleanup_kernel_size < 0:
            raise ValueError("cleanup_kernel_size cannot be negative")
        if not 0 <= self.foreground_threshold <= 255:
            raise ValueError("foreground_threshold must be between 0 and 255")


class NativePdfAdapter(FormatAdapter):
    def __init__(self, config: NativePdfConfig | None = None) -> None:
        self.config = config or NativePdfConfig()

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".pdf"})

    def extract_canonical(
        self, file_path: Path, output_dir: Path
    ) -> DocumentCanonicalRepresentation:
        file_path = Path(file_path)
        if not self.can_handle(file_path):
            raise ValueError(f"Not a readable PDF: {file_path}")

        document_id = hashlib.sha256(file_path.read_bytes()).hexdigest()[:24]
        artifact_dir = Path(output_dir) / document_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(file_path) as document:
            if document.page_count != 1:
                raise ValueError(
                    f"NativePdfAdapter currently supports exactly one page; got {document.page_count}"
                )
            page = document[0]
            page_width = float(page.rect.width)
            page_height = float(page.rect.height)
            text_elements = self.extract_text(page)
            geometry_elements = self.extract_geometry(page)
            original = self.render_page(page)

        text_mask = self.create_text_mask(original.shape[:2], text_elements)
        text_bboxes = self.create_text_bboxes(original.shape[:2], text_elements)
        geometry_mask = self.create_geometry_mask(
            original.shape[:2], geometry_elements
        )
                
        artifacts = self.save_debug_artifacts(
            artifact_dir,
            original,
            text_bboxes,
            geometry_mask,
        )
        text_path = (artifact_dir/ "text_elements.json")
        geometry_path = (artifact_dir/ "geometry_elements.json")

        # -----------------------------------
        # Text elements
        # -----------------------------------

        text_path.write_text(
            json.dumps(
                [element.model_dump(mode="json")
                    for element in text_elements
                ],
                indent=2,
            ),encoding="utf-8",
        )

        # -----------------------------------
        # Geometry elements
        # -----------------------------------

        geometry_path.write_text(
            json.dumps([element.model_dump(mode="json")
                    for element in geometry_elements
                ],
                indent=2,
            ),encoding="utf-8",
        )

        return DocumentCanonicalRepresentation(
            document_id=document_id,
            source_format="pdf",
            page_width=page_width,
            page_height=page_height,
            text_elements=text_elements,
            geometry_elements=geometry_elements,
            metadata={
                "source_name": file_path.name,
                "page_number": 0,
                "page_count": 1,
                "coordinate_system": "pdf_top_left",
                "render_zoom": self.config.zoom,
                "artifacts": artifacts,
                "extractor": "pymupdf",
            },
        )
        

    def render_page(self, page: fitz.Page) -> np.ndarray:
        pixmap = page.get_pixmap(
            matrix=fitz.Matrix(self.config.zoom, self.config.zoom),
            colorspace=fitz.csRGB,
            alpha=False,
        )
        return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height, pixmap.width, pixmap.n
        )[:, :, :3].copy()

    def extract_text(self, page: fitz.Page) -> list[TextElement]:
        elements: list[TextElement] = []
        raw = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = _bbox_from_values(span["bbox"])
                    elements.append(
                        TextElement(
                            id=f"text_{len(elements) + 1:06d}",
                            text=span.get("text", ""),
                            bbox=bbox,
                            font_name=span.get("font"),
                            font_size=span.get("size"),
                            font_flags=span.get("flags"),
                            confidence=1.0,
                            source="pdf_native",
                        )
                    )
        return elements

    def extract_geometry(self, page: fitz.Page) -> list[GeometryElement]:
        elements: list[GeometryElement] = []
        for drawing in page.get_drawings(extended=True):
            style = {
                "stroke_width": drawing.get("width"),
                "stroke_color": _color_tuple(drawing.get("color")),
                "fill_color": _color_tuple(drawing.get("fill")),
                "layer": drawing.get("layer"),
            }
            for item in drawing.get("items", []):
                parsed = _parse_drawing_item(item)
                if parsed is None:
                    continue
                geometry_type, bbox, coordinates = parsed
                elements.append(
                    GeometryElement(
                        id=f"geometry_{len(elements) + 1:06d}",
                        geometry_type=geometry_type,
                        bbox=bbox,
                        coordinates=coordinates,
                        source="pdf_native",
                        **style,
                    )
                )
        return elements

    def create_text_bboxes(
        self, image_shape: tuple[int, int], elements: Iterable[TextElement],
    ) -> list[list[int]]:
        """
        Convert text element bounding boxes from PDF coordinates to pixel
        coordinates and return them as bounding boxes.

        Returns:
            List of bounding boxes in the format:
            [[x0, y0, x1, y1], ...]
        """

        bboxes = []

        pad = self.config.text_mask_padding
        height, width = image_shape

        for element in elements:
            x0, y0, x1, y1 = pdf_to_pixel_bbox(
                element.bbox,
                self.config.zoom
            )

            bbox = [
                max(0, x0 - pad),
                max(0, y0 - pad),
                min(width - 1, x1 + pad),
                min(height - 1, y1 + pad),
            ]

            bboxes.append(bbox)

        return bboxes

    def create_text_mask(
        self, image_shape: tuple[int, int], elements: Iterable[TextElement]
    ) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        pad = self.config.text_mask_padding
        height, width = image_shape
        for element in elements:
            x0, y0, x1, y1 = pdf_to_pixel_bbox(element.bbox, self.config.zoom)
            cv2.rectangle(
                mask,
                (max(0, x0 - pad), max(0, y0 - pad)),
                (min(width - 1, x1 + pad), min(height - 1, y1 + pad)),
                255,
                thickness=-1,
            )
        return mask

    def create_geometry_mask(
        self, image_shape: tuple[int, int], elements: Iterable[GeometryElement]
    ) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        for element in elements:
            self._draw_geometry(mask, element)
        pad = self.config.geometry_mask_padding
        if pad:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)
            )
            mask = cv2.dilate(mask, kernel)
        return mask

    def _draw_geometry(self, mask: np.ndarray, element: GeometryElement) -> None:
        coordinates = element.coordinates
        thickness = max(
            1, round((element.stroke_width or 1.0) * self.config.zoom)
        )
        if element.geometry_type == GeometryType.LINE:
            assert isinstance(coordinates, dict)
            cv2.line(
                mask,
                self._point(coordinates["x1"], coordinates["y1"]),
                self._point(coordinates["x2"], coordinates["y2"]),
                255,
                thickness,
                lineType=cv2.LINE_AA,
            )
        elif element.geometry_type == GeometryType.RECTANGLE:
            assert isinstance(coordinates, dict)
            box = BoundingBox(**coordinates)
            x0, y0, x1, y1 = pdf_to_pixel_bbox(box, self.config.zoom)
            cv2.rectangle(
                mask,
                (x0, y0),
                (x1, y1),
                255,
                -1 if element.fill_color is not None else thickness,
                lineType=cv2.LINE_AA,
            )
        elif element.geometry_type == GeometryType.CURVE:
            assert isinstance(coordinates, dict)
            points = _sample_bezier(coordinates)
            cv2.polylines(
                mask,
                [np.asarray([self._point(x, y) for x, y in points], np.int32)],
                False,
                255,
                thickness,
                lineType=cv2.LINE_AA,
            )
        elif element.geometry_type in (GeometryType.POLYLINE, GeometryType.PATH):
            commands = coordinates if isinstance(coordinates, list) else [coordinates]
            for command in commands:
                points = command.get("points", [])
                if len(points) >= 2:
                    cv2.polylines(
                        mask,
                        [
                            np.asarray(
                                [self._point(point["x"], point["y"]) for point in points],
                                np.int32,
                            )
                        ],
                        bool(command.get("closed")),
                        255,
                        thickness,
                        lineType=cv2.LINE_AA,
                    )

    def _point(self, x: float, y: float) -> tuple[int, int]:
        return round(x * self.config.zoom), round(y * self.config.zoom)

    def create_residual(
        self,
        original: np.ndarray,
        text_mask: np.ndarray,
        geometry_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if text_mask.shape != original.shape[:2] or geometry_mask.shape != original.shape[:2]:
            raise ValueError("Masks and rendered image must have identical dimensions")
        foreground = (np.min(original, axis=2) < self.config.foreground_threshold).astype(
            np.uint8
        ) * 255
        known = cv2.bitwise_or(text_mask, geometry_mask)
        residual_mask = cv2.bitwise_and(foreground, cv2.bitwise_not(known))

        size = self.config.cleanup_kernel_size
        if size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            residual_mask = cv2.morphologyEx(
                residual_mask, cv2.MORPH_OPEN, kernel
            )
            residual_mask = cv2.morphologyEx(
                residual_mask, cv2.MORPH_CLOSE, kernel
            )

        residual_image = np.full_like(original, 255)
        residual_image[residual_mask > 0] = original[residual_mask > 0]
        return residual_mask, residual_image

    def extract_residual_regions(
        self,
        mask: np.ndarray,
        residual_image: np.ndarray,
        artifact_dir: Path,
        page_width: float,
        page_height: float,
    ) -> list[ResidualImageRegion]:
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidates = [
            tuple(int(value) for value in stats[index])
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= self.config.min_residual_area
        ]
        candidates.sort(key=lambda stat: (stat[1], stat[0]))

        regions: list[ResidualImageRegion] = []
        height, width = mask.shape
        for index, (x, y, component_width, component_height, _) in enumerate(
            candidates, start=1
        ):
            pad = self.config.region_padding
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1 = min(width, x + component_width + pad)
            y1 = min(height, y + component_height + pad)
            region_id = f"residual_{index:06d}"
            region_path = artifact_dir / f"{region_id}.png"
            _save_rgb(region_path, residual_image[y0:y1, x0:x1])
            pdf_bbox = pixel_to_pdf_bbox((x0, y0, x1, y1), self.config.zoom)
            regions.append(
                ResidualImageRegion(
                    id=region_id,
                    bbox=BoundingBox(
                        x0=max(0.0, pdf_bbox.x0),
                        y0=max(0.0, pdf_bbox.y0),
                        x1=min(page_width, pdf_bbox.x1),
                        y1=min(page_height, pdf_bbox.y1),
                    ),
                    image_path=str(region_path),
                    pixel_width=x1 - x0,
                    pixel_height=y1 - y0,
                    source="pdf_native_residual",
                )
            )
        return regions

    def save_debug_artifacts(
        self,
        artifact_dir: Path,
        original: np.ndarray,
        text_bboxes: list[list[int]],
        geometry_mask: np.ndarray,
    ) -> dict[str, str]:

        artifacts = {
            "original_render": artifact_dir / "original_render.png",
            "geometry_mask": artifact_dir / "geometry_mask.png",
        }

        _save_rgb(artifacts["original_render"], original)
        _save_gray(artifacts["geometry_mask"], geometry_mask)

        if self.config.save_overlay:
            overlay = original.copy()

            for bbox in text_bboxes:
                x0, y0, x1, y1 = bbox

                cv2.rectangle(
                    overlay,
                    (x0, y0),
                    (x1, y1),
                    (0, 255, 0),
                    thickness=2,
                )

            overlay_path = artifact_dir / "text_bboxes.png"

            _save_rgb(
                overlay_path,
                overlay,
            )

            artifacts["overlay"] = overlay_path

        return {
            name: str(path)
            for name, path in artifacts.items()
        }

def _bbox_from_values(values: Iterable[float]) -> BoundingBox:
    x0, y0, x1, y1 = (float(value) for value in values)
    return BoundingBox(
        x0=min(x0, x1), y0=min(y0, y1), x1=max(x0, x1), y1=max(y0, y1)
    )


def _point_dict(point: fitz.Point) -> dict[str, float]:
    return {"x": float(point.x), "y": float(point.y)}


def _color_tuple(color: Any) -> tuple[float, ...] | None:
    if color is None:
        return None
    return tuple(float(channel) for channel in color)


def _parse_drawing_item(
    item: tuple[Any, ...],
) -> tuple[GeometryType, BoundingBox, dict[str, Any] | list[dict[str, Any]]] | None:
    operation = item[0]
    if operation == "l":
        start, end = item[1], item[2]
        return (
            GeometryType.LINE,
            _bbox_from_points([start, end]),
            {
                "x1": float(start.x),
                "y1": float(start.y),
                "x2": float(end.x),
                "y2": float(end.y),
            },
        )
    if operation == "re":
        rectangle = fitz.Rect(item[1])
        bbox = _bbox_from_values(rectangle)
        return GeometryType.RECTANGLE, bbox, bbox.model_dump()
    if operation == "c":
        points = list(item[1:5])
        coordinates = {
            name: _point_dict(point)
            for name, point in zip(("start", "control1", "control2", "end"), points)
        }
        return GeometryType.CURVE, _bbox_from_points(points), coordinates
    if operation == "qu":
        quad = fitz.Quad(item[1])
        points = [quad.ul, quad.ur, quad.lr, quad.ll]
        return (
            GeometryType.POLYLINE,
            _bbox_from_points(points),
            [{"points": [_point_dict(point) for point in points], "closed": True}],
        )

    points = [value for value in item[1:] if isinstance(value, fitz.Point)]
    if points:
        return (
            GeometryType.PATH,
            _bbox_from_points(points),
            [
                {
                    "operation": str(operation),
                    "points": [_point_dict(point) for point in points],
                    "closed": False,
                }
            ],
        )
    return None


def _bbox_from_points(points: Iterable[fitz.Point]) -> BoundingBox:
    point_list = list(points)
    return BoundingBox(
        x0=min(float(point.x) for point in point_list),
        y0=min(float(point.y) for point in point_list),
        x1=max(float(point.x) for point in point_list),
        y1=max(float(point.y) for point in point_list),
    )


def _sample_bezier(coordinates: dict[str, Any], samples: int = 24) -> list[tuple[float, float]]:
    points = [
        coordinates[name] for name in ("start", "control1", "control2", "end")
    ]
    result: list[tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, samples):
        one_minus_t = 1.0 - t
        weights = (
            one_minus_t**3,
            3 * one_minus_t**2 * t,
            3 * one_minus_t * t**2,
            t**3,
        )
        result.append(
            (
                math.fsum(weight * point["x"] for weight, point in zip(weights, points)),
                math.fsum(weight * point["y"] for weight, point in zip(weights, points)),
            )
        )
    return result


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path)


def _save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="L").save(path)
