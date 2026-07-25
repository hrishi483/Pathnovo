"""CAD (.dwg / .dxf) adapter producing canonical text/geometry + PNG artifacts."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.canonical.model import (
    BoundingBox,
    DocumentCanonicalRepresentation,
    GeometryElement,
    GeometryType,
    TextElement,
)
from src.ingest.base import FormatAdapter, pdf_to_pixel_bbox, write_element_json
from src.ingest.cad_extract import extract as extract_cad


@dataclass(frozen=True)
class DwgConfig:
    zoom: float = 2.0
    layouts: str = "modelspace"
    expand_blocks: bool = True
    padding_ratio: float = 0.02
    min_page_size: float = 1.0
    max_render_edge: int = 8192
    text_mask_padding: int = 1
    geometry_mask_padding: int = 1
    save_overlay: bool = True
    circle_samples: int = 64
    arc_samples: int = 48
    ellipse_samples: int = 64

    def __post_init__(self) -> None:
        if self.zoom <= 0:
            raise ValueError("zoom must be positive")
        if self.padding_ratio < 0:
            raise ValueError("padding_ratio cannot be negative")
        if self.min_page_size <= 0:
            raise ValueError("min_page_size must be positive")
        if self.max_render_edge < 64:
            raise ValueError("max_render_edge must be at least 64")


class DwgAdapter(FormatAdapter):
    def __init__(self, config: DwgConfig | None = None) -> None:
        self.config = config or DwgConfig()

    @property
    def extensions(self) -> frozenset[str]:
        return frozenset({".dwg", ".dxf"})

    def extract_canonical(
        self, file_path: Path, output_dir: Path
    ) -> DocumentCanonicalRepresentation:
        file_path = Path(file_path)
        if not self.can_handle(file_path):
            raise ValueError(f"Not a readable CAD file: {file_path}")

        document_id = hashlib.sha256(file_path.read_bytes()).hexdigest()[:24]
        artifact_dir = Path(output_dir) / document_id
        artifact_dir.mkdir(parents=True, exist_ok=True)

        raw = extract_cad(
            file_path,
            layouts=self.config.layouts,
            expand_blocks=self.config.expand_blocks,
        )
        extents = _compute_extents(raw["text"], raw["geometry"])
        transform = _build_transform(extents, self.config)

        text_elements = self._map_text(raw["text"], transform)
        geometry_elements = self._map_geometry(raw["geometry"], transform)

        render_zoom = _effective_zoom(
            transform.page_width,
            transform.page_height,
            self.config.zoom,
            self.config.max_render_edge,
        )
        image_shape = (
            max(1, round(transform.page_height * render_zoom)),
            max(1, round(transform.page_width * render_zoom)),
        )
        original = self._render_original(
            image_shape, text_elements, geometry_elements, render_zoom
        )
        text_bboxes = self._text_pixel_bboxes(image_shape, text_elements, render_zoom)
        geometry_mask = self._create_geometry_mask(
            image_shape, geometry_elements, render_zoom
        )
        artifacts = self._save_artifacts(
            artifact_dir, original, text_bboxes, geometry_mask
        )

        write_element_json(artifact_dir / "text_elements.json", text_elements)
        write_element_json(artifact_dir / "geometry_elements.json", geometry_elements)

        return DocumentCanonicalRepresentation(
            document_id=document_id,
            source_format=file_path.suffix.lower().lstrip(".") or "dwg",
            page_width=transform.page_width,
            page_height=transform.page_height,
            text_elements=text_elements,
            geometry_elements=geometry_elements,
            metadata={
                "source_name": file_path.name,
                "page_number": 0,
                "page_count": 1,
                "coordinate_system": "normalized_top_left",
                "cad_extents": {
                    "min_x": extents[0],
                    "min_y": extents[1],
                    "max_x": extents[2],
                    "max_y": extents[3],
                },
                "layouts": self.config.layouts,
                "render_zoom": render_zoom,
                "artifacts": artifacts,
                "extractor": "ezdxf",
            },
        )

    def _map_text(
        self, records: list[dict[str, Any]], transform: "_CoordTransform"
    ) -> list[TextElement]:
        elements: list[TextElement] = []
        for record in records:
            text = str(record.get("text") or "")
            insert = record.get("insert")
            if insert is None:
                continue
            height = float(record.get("height") or 1.0)
            if height <= 0:
                height = 1.0
            rotation = float(record.get("rotation") or 0.0)
            width = max(height * 0.6 * max(len(text.strip()), 1), height * 0.5)
            corners = _rotated_rect_corners(
                float(insert[0]), float(insert[1]), width, height, rotation
            )
            mapped = [transform.map_point(x, y) for x, y in corners]
            bbox = _bbox_from_xy(mapped)
            elements.append(
                TextElement(
                    id=f"text_{len(elements) + 1:06d}",
                    text=text,
                    bbox=bbox,
                    font_name=record.get("style"),
                    font_size=height,
                    font_flags=None,
                    confidence=1.0,
                    source="dwg",
                )
            )
        return elements

    def _map_geometry(
        self, records: list[dict[str, Any]], transform: "_CoordTransform"
    ) -> list[GeometryElement]:
        elements: list[GeometryElement] = []
        for record in records:
            mapped = _map_cad_geometry(record, transform, self.config)
            if mapped is None:
                continue
            geometry_type, bbox, coordinates = mapped
            elements.append(
                GeometryElement(
                    id=f"geometry_{len(elements) + 1:06d}",
                    geometry_type=geometry_type,
                    bbox=bbox,
                    coordinates=coordinates,
                    stroke_width=1.0,
                    stroke_color=None,
                    fill_color=None,
                    layer=record.get("layer"),
                    source="dwg",
                )
            )
        return elements

    def _render_original(
        self,
        image_shape: tuple[int, int],
        text_elements: Iterable[TextElement],
        geometry_elements: Iterable[GeometryElement],
        zoom: float,
    ) -> np.ndarray:
        height, width = image_shape
        image = np.full((height, width, 3), 255, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        for element in geometry_elements:
            _draw_geometry(mask, element, zoom)
        image[mask > 0] = (20, 20, 20)

        for element in text_elements:
            if not element.text.strip():
                continue
            x0, y0, _, y1 = pdf_to_pixel_bbox(element.bbox, zoom)
            font_scale = max(0.3, min(2.0, (y1 - y0) / 18.0))
            cv2.putText(
                image,
                element.text[:80],
                (max(0, x0), max(12, y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                max(1, round(font_scale)),
                lineType=cv2.LINE_AA,
            )
        return image

    def _text_pixel_bboxes(
        self,
        image_shape: tuple[int, int],
        elements: Iterable[TextElement],
        zoom: float,
    ) -> list[list[int]]:
        pad = self.config.text_mask_padding
        height, width = image_shape
        bboxes: list[list[int]] = []
        for element in elements:
            x0, y0, x1, y1 = pdf_to_pixel_bbox(element.bbox, zoom)
            bboxes.append(
                [
                    max(0, x0 - pad),
                    max(0, y0 - pad),
                    min(width - 1, x1 + pad),
                    min(height - 1, y1 + pad),
                ]
            )
        return bboxes

    def _create_geometry_mask(
        self,
        image_shape: tuple[int, int],
        elements: Iterable[GeometryElement],
        zoom: float,
    ) -> np.ndarray:
        mask = np.zeros(image_shape, dtype=np.uint8)
        for element in elements:
            _draw_geometry(mask, element, zoom)
        pad = self.config.geometry_mask_padding
        if pad:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1)
            )
            mask = cv2.dilate(mask, kernel)
        return mask

    def _save_artifacts(
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
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), thickness=2)
            overlay_path = artifact_dir / "text_bboxes.png"
            _save_rgb(overlay_path, overlay)
            artifacts["overlay"] = overlay_path

        return {name: str(path) for name, path in artifacts.items()}


@dataclass(frozen=True)
class _CoordTransform:
    min_x: float
    max_y: float
    page_width: float
    page_height: float
    pad: float

    def map_point(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.min_x + self.pad, self.max_y - y + self.pad)


def _effective_zoom(
    page_width: float, page_height: float, zoom: float, max_edge: int
) -> float:
    width_px = page_width * zoom
    height_px = page_height * zoom
    longest = max(width_px, height_px)
    if longest <= max_edge:
        return zoom
    return zoom * (max_edge / longest)


def _compute_extents(
    texts: list[dict[str, Any]], geoms: list[dict[str, Any]]
) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []

    def add_point(point: Any) -> None:
        if point is None:
            return
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(float(point[0]))
            ys.append(float(point[1]))

    for record in texts:
        insert = record.get("insert")
        if insert is None:
            continue
        height = float(record.get("height") or 1.0)
        width = max(height * 0.6 * max(len(str(record.get("text") or "").strip()), 1), height)
        rotation = float(record.get("rotation") or 0.0)
        for x, y in _rotated_rect_corners(
            float(insert[0]), float(insert[1]), width, height, rotation
        ):
            xs.append(x)
            ys.append(y)

    for record in geoms:
        for point in _iter_geometry_points(record):
            add_point(point)

    if not xs or not ys:
        return (0.0, 0.0, 1.0, 1.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _build_transform(
    extents: tuple[float, float, float, float], config: DwgConfig
) -> _CoordTransform:
    min_x, min_y, max_x, max_y = extents
    width = max(max_x - min_x, config.min_page_size)
    height = max(max_y - min_y, config.min_page_size)
    pad = max(width, height) * config.padding_ratio
    return _CoordTransform(
        min_x=min_x,
        max_y=max_y,
        page_width=width + 2 * pad,
        page_height=height + 2 * pad,
        pad=pad,
    )


def _iter_geometry_points(record: dict[str, Any]) -> Iterable[Any]:
    geometry = record.get("geometry") or {}
    entity_type = record.get("entity_type")

    if entity_type == "CIRCLE":
        center = geometry.get("center")
        radius = float(geometry.get("radius") or 0.0)
        if center is not None:
            cx, cy = float(center[0]), float(center[1])
            yield (cx - radius, cy - radius)
            yield (cx + radius, cy + radius)
        return

    if entity_type == "ARC":
        for point in _sample_arc_points(geometry, samples=16):
            yield point
        return

    if entity_type == "ELLIPSE":
        for point in _sample_ellipse_points(geometry, samples=16):
            yield point
        return

    for key in ("start", "end", "center", "location", "point", "insert"):
        if key in geometry:
            yield geometry[key]
    for key in ("points", "control_points", "fit_points", "vertices"):
        for point in geometry.get(key) or []:
            yield point
    for path in geometry.get("paths") or []:
        for point in path:
            yield point


def _map_cad_geometry(
    record: dict[str, Any],
    transform: _CoordTransform,
    config: DwgConfig,
) -> tuple[GeometryType, BoundingBox, dict[str, Any] | list[dict[str, Any]]] | None:
    entity_type = record.get("entity_type")
    geometry = record.get("geometry") or {}

    if entity_type == "LINE":
        start = geometry.get("start")
        end = geometry.get("end")
        if start is None or end is None:
            return None
        x1, y1 = transform.map_point(float(start[0]), float(start[1]))
        x2, y2 = transform.map_point(float(end[0]), float(end[1]))
        bbox = _bbox_from_xy([(x1, y1), (x2, y2)])
        return (
            GeometryType.LINE,
            bbox,
            {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        )

    if entity_type == "CIRCLE":
        points = [
            transform.map_point(x, y)
            for x, y in _sample_circle_points(geometry, config.circle_samples)
        ]
        return _polyline_result(points, closed=True)

    if entity_type == "ARC":
        points = [
            transform.map_point(x, y)
            for x, y in _sample_arc_points(geometry, config.arc_samples)
        ]
        return _polyline_result(points, closed=False)

    if entity_type == "ELLIPSE":
        points = [
            transform.map_point(x, y)
            for x, y in _sample_ellipse_points(geometry, config.ellipse_samples)
        ]
        start = float(geometry.get("start_param") or 0.0)
        end = float(geometry.get("end_param") or (2 * math.pi))
        closed = abs((end - start) % (2 * math.pi)) < 1e-6 or abs(end - start) >= 2 * math.pi - 1e-6
        return _polyline_result(points, closed=closed)

    if entity_type in ("LWPOLYLINE", "POLYLINE", "SOLID", "TRACE", "3DFACE"):
        raw_points = geometry.get("points") or []
        points = [
            transform.map_point(float(point[0]), float(point[1]))
            for point in raw_points
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        closed = bool(geometry.get("closed")) or entity_type in ("SOLID", "TRACE", "3DFACE")
        return _polyline_result(points, closed=closed)

    if entity_type == "SPLINE":
        raw_points = geometry.get("fit_points") or geometry.get("control_points") or []
        points = [
            transform.map_point(float(point[0]), float(point[1]))
            for point in raw_points
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        return _path_result(points, closed=False)

    if entity_type == "HATCH":
        commands: list[dict[str, Any]] = []
        all_points: list[tuple[float, float]] = []
        for path in geometry.get("paths") or []:
            points = [
                transform.map_point(float(point[0]), float(point[1]))
                for point in path
                if isinstance(point, (list, tuple)) and len(point) >= 2
            ]
            if len(points) < 2:
                continue
            all_points.extend(points)
            commands.append(
                {
                    "points": [{"x": x, "y": y} for x, y in points],
                    "closed": True,
                }
            )
        if not commands:
            return None
        return GeometryType.PATH, _bbox_from_xy(all_points), commands

    if entity_type == "POINT":
        location = geometry.get("location")
        if location is None:
            return None
        x, y = transform.map_point(float(location[0]), float(location[1]))
        return _polyline_result([(x, y), (x, y)], closed=False)

    return None


def _polyline_result(
    points: list[tuple[float, float]], *, closed: bool
) -> tuple[GeometryType, BoundingBox, list[dict[str, Any]]] | None:
    if len(points) < 2:
        return None
    return (
        GeometryType.POLYLINE,
        _bbox_from_xy(points),
        [{"points": [{"x": x, "y": y} for x, y in points], "closed": closed}],
    )


def _path_result(
    points: list[tuple[float, float]], *, closed: bool
) -> tuple[GeometryType, BoundingBox, list[dict[str, Any]]] | None:
    if len(points) < 2:
        return None
    return (
        GeometryType.PATH,
        _bbox_from_xy(points),
        [{"points": [{"x": x, "y": y} for x, y in points], "closed": closed}],
    )


def _sample_circle_points(geometry: dict[str, Any], samples: int) -> list[tuple[float, float]]:
    center = geometry.get("center")
    radius = float(geometry.get("radius") or 0.0)
    if center is None or radius <= 0:
        return []
    cx, cy = float(center[0]), float(center[1])
    return [
        (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        for angle in np.linspace(0.0, 2 * math.pi, samples, endpoint=False)
    ]


def _sample_arc_points(geometry: dict[str, Any], samples: int) -> list[tuple[float, float]]:
    center = geometry.get("center")
    radius = float(geometry.get("radius") or 0.0)
    if center is None or radius <= 0:
        return []
    cx, cy = float(center[0]), float(center[1])
    start = math.radians(float(geometry.get("start_angle") or 0.0))
    end = math.radians(float(geometry.get("end_angle") or 0.0))
    if end < start:
        end += 2 * math.pi
    return [
        (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
        for angle in np.linspace(start, end, max(2, samples))
    ]


def _sample_ellipse_points(geometry: dict[str, Any], samples: int) -> list[tuple[float, float]]:
    center = geometry.get("center")
    major = geometry.get("major_axis")
    if center is None or major is None:
        return []
    cx, cy = float(center[0]), float(center[1])
    mx, my = float(major[0]), float(major[1])
    ratio = float(geometry.get("ratio") or 1.0)
    start = float(geometry.get("start_param") or 0.0)
    end = float(geometry.get("end_param") or (2 * math.pi))
    major_len = math.hypot(mx, my)
    if major_len == 0:
        return []
    ux, uy = mx / major_len, my / major_len
    vx, vy = -uy, ux
    minor_len = major_len * ratio
    points: list[tuple[float, float]] = []
    for param in np.linspace(start, end, max(2, samples)):
        points.append(
            (
                cx + major_len * math.cos(param) * ux + minor_len * math.sin(param) * vx,
                cy + major_len * math.cos(param) * uy + minor_len * math.sin(param) * vy,
            )
        )
    return points


def _rotated_rect_corners(
    x: float, y: float, width: float, height: float, rotation_deg: float
) -> list[tuple[float, float]]:
    angle = math.radians(rotation_deg)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    local = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
    return [
        (x + lx * cos_a - ly * sin_a, y + lx * sin_a + ly * cos_a)
        for lx, ly in local
    ]


def _bbox_from_xy(points: Iterable[tuple[float, float]]) -> BoundingBox:
    point_list = list(points)
    if not point_list:
        return BoundingBox(x0=0.0, y0=0.0, x1=0.0, y1=0.0)
    xs = [point[0] for point in point_list]
    ys = [point[1] for point in point_list]
    return BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))


def _draw_geometry(mask: np.ndarray, element: GeometryElement, zoom: float) -> None:
    coordinates = element.coordinates
    thickness = max(1, round((element.stroke_width or 1.0) * zoom))

    def point(x: float, y: float) -> tuple[int, int]:
        return round(x * zoom), round(y * zoom)

    if element.geometry_type == GeometryType.LINE:
        assert isinstance(coordinates, dict)
        cv2.line(
            mask,
            point(coordinates["x1"], coordinates["y1"]),
            point(coordinates["x2"], coordinates["y2"]),
            255,
            thickness,
            lineType=cv2.LINE_AA,
        )
        return

    if element.geometry_type == GeometryType.RECTANGLE:
        assert isinstance(coordinates, dict)
        box = BoundingBox(**coordinates)
        x0, y0, x1, y1 = pdf_to_pixel_bbox(box, zoom)
        cv2.rectangle(
            mask,
            (x0, y0),
            (x1, y1),
            255,
            -1 if element.fill_color is not None else thickness,
            lineType=cv2.LINE_AA,
        )
        return

    commands = coordinates if isinstance(coordinates, list) else [coordinates]
    for command in commands:
        points = command.get("points", [])
        if len(points) >= 2:
            cv2.polylines(
                mask,
                [np.asarray([point(p["x"], p["y"]) for p in points], np.int32)],
                bool(command.get("closed")),
                255,
                thickness,
                lineType=cv2.LINE_AA,
            )


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="RGB").save(path)


def _save_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image, mode="L").save(path)
