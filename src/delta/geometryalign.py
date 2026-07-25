from __future__ import annotations

import logging
import time
from math import sqrt
from typing import Any

from src.canonical.model import (
    DocumentCanonicalRepresentation,
    GeometryElement,
)

logger = logging.getLogger(__name__)


class GeometryAligner:
    """
    Align geometry elements between two revisions.

    Matching strategy:

    1. Geometry type compatibility
    2. Spatial proximity
    3. Bounding-box similarity
    4. Coordinate similarity
    5. One-to-one assignment

    The output is suitable for:
    - JSON delta reports
    - visual overlays
    - future geometry-aware VLM analysis
    """

    def __init__(
        self,
        max_distance: float = 0.025,
        min_bbox_similarity: float = 0.30,
        min_geometry_similarity: float = 0.40,
    ):

        self.max_distance = max_distance

        self.min_bbox_similarity = (
            min_bbox_similarity
        )

        self.min_geometry_similarity = (
            min_geometry_similarity
        )

    # =========================================================
    # BOUNDING BOX CENTER
    # =========================================================

    @staticmethod
    def bbox_center(
        element: GeometryElement,
    ) -> tuple[float, float]:

        bbox = element.bbox

        return (
            (bbox.x0 + bbox.x1) / 2.0,
            (bbox.y0 + bbox.y1) / 2.0,
        )

    # =========================================================
    # NORMALIZED SPATIAL DISTANCE
    # =========================================================

    @staticmethod
    def normalized_distance(
        element_a: GeometryElement,
        element_b: GeometryElement,
        page_width: float,
        page_height: float,
    ) -> float:

        ax, ay = GeometryAligner.bbox_center(
            element_a
        )

        bx, by = GeometryAligner.bbox_center(
            element_b
        )

        dx = (
            ax - bx
        ) / page_width

        dy = (
            ay - by
        ) / page_height

        return sqrt(
            dx * dx
            +
            dy * dy
        )

    # =========================================================
    # BOUNDING BOX AREA
    # =========================================================

    @staticmethod
    def bbox_area(
        element: GeometryElement,
    ) -> float:

        return max(
            0.0,
            element.bbox.width
        ) * max(
            0.0,
            element.bbox.height
        )

    # =========================================================
    # INTERSECTION OVER UNION
    # =========================================================

    @staticmethod
    def bbox_iou(
        element_a: GeometryElement,
        element_b: GeometryElement,
    ) -> float:

        a = element_a.bbox
        b = element_b.bbox

        intersection_x0 = max(
            a.x0,
            b.x0
        )

        intersection_y0 = max(
            a.y0,
            b.y0
        )

        intersection_x1 = min(
            a.x1,
            b.x1
        )

        intersection_y1 = min(
            a.y1,
            b.y1
        )

        intersection_width = max(
            0.0,
            intersection_x1
            -
            intersection_x0
        )

        intersection_height = max(
            0.0,
            intersection_y1
            -
            intersection_y0
        )

        intersection_area = (
            intersection_width
            *
            intersection_height
        )

        area_a = (
            (a.x1 - a.x0)
            *
            (a.y1 - a.y0)
        )

        area_b = (
            (b.x1 - b.x0)
            *
            (b.y1 - b.y0)
        )

        union_area = (
            area_a
            +
            area_b
            -
            intersection_area
        )

        if union_area <= 0:
            return 0.0

        return (
            intersection_area
            /
            union_area
        )

    # =========================================================
    # BOUNDING BOX SIMILARITY
    # =========================================================

    @staticmethod
    def bbox_similarity(
        element_a: GeometryElement,
        element_b: GeometryElement,
    ) -> float:

        a = element_a.bbox
        b = element_b.bbox

        width_similarity = (
            min(a.width, b.width)
            /
            max(a.width, b.width)
            if max(a.width, b.width) > 0
            else 1.0
        )

        height_similarity = (
            min(a.height, b.height)
            /
            max(a.height, b.height)
            if max(a.height, b.height) > 0
            else 1.0
        )

        iou = GeometryAligner.bbox_iou(
            element_a,
            element_b,
        )

        # IoU is very useful for large objects.
        # Width/height ratio helps for lines
        # where IoU is often close to zero.

        return (
            0.5 * iou
            +
            0.25 * width_similarity
            +
            0.25 * height_similarity
        )

    # =========================================================
    # EXTRACT NUMERIC COORDINATES
    # =========================================================

    @staticmethod
    def extract_numeric_values(
        coordinates: Any,
    ) -> list[float]:

        values = []

        if isinstance(
            coordinates,
            dict,
        ):

            for value in coordinates.values():

                values.extend(
                    GeometryAligner.extract_numeric_values(
                        value
                    )
                )

        elif isinstance(
            coordinates,
            list,
        ):

            for value in coordinates:

                values.extend(
                    GeometryAligner.extract_numeric_values(
                        value
                    )
                )

        elif isinstance(
            coordinates,
            tuple,
        ):

            for value in coordinates:

                values.extend(
                    GeometryAligner.extract_numeric_values(
                        value
                    )
                )

        elif isinstance(
            coordinates,
            (int, float),
        ):

            values.append(
                float(coordinates)
            )

        return values

    # =========================================================
    # COORDINATE SIMILARITY
    # =========================================================

    def coordinate_similarity(
        self,
        element_a: GeometryElement,
        element_b: GeometryElement,
        page_width: float,
        page_height: float,
    ) -> float:

        values_a = (
            GeometryAligner.extract_numeric_values(
                element_a.coordinates
            )
        )

        values_b = (
            GeometryAligner.extract_numeric_values(
                element_b.coordinates
            )
        )

        if not values_a or not values_b:

            return 0.0

        # Different coordinate structures
        # cannot be directly compared.

        if len(values_a) != len(values_b):

            return 0.0

        total_error = 0.0

        for value_a, value_b in zip(
            values_a,
            values_b,
        ):

            # Coordinates are either x or y.
            # A conservative normalization is used
            # against the largest page dimension.

            error = abs(
                value_a
                -
                value_b
            ) / max(
                page_width,
                page_height,
            )

            total_error += error

        mean_error = (
            total_error
            /
            len(values_a)
        )

        return max(
            0.0,
            1.0
            -
            mean_error
            /
            self.self_coordinate_tolerance(
                page_width,
                page_height,
            )
        )

    def self_coordinate_tolerance(
        self,
        page_width: float,
        page_height: float,
    ) -> float:

        # Approximately 1% of the page's
        # largest dimension.

        return 0.01

    # =========================================================
    # GEOMETRY SIMILARITY
    # =========================================================

    def geometry_similarity(
        self,
        element_a: GeometryElement,
        element_b: GeometryElement,
        page_width: float,
        page_height: float,
    ) -> dict[str, float]:

        distance = (
            self.normalized_distance(
                element_a,
                element_b,
                page_width,
                page_height,
            )
        )

        spatial_similarity = max(
            0.0,
            1.0
            -
            distance
            /
            self.max_distance,
        )

        bbox_score = (
            self.bbox_similarity(
                element_a,
                element_b,
            )
        )

        coordinate_score = (
            self.coordinate_similarity(
                element_a,
                element_b,
                page_width,
                page_height,
            )
        )

        # Geometry similarity score.

        score = (
            0.35
            *
            spatial_similarity
            +
            0.35
            *
            bbox_score
            +
            0.30
            *
            coordinate_score
        )

        return {
            "score": score,
            "spatial_similarity": spatial_similarity,
            "bbox_similarity": bbox_score,
            "coordinate_similarity": coordinate_score,
            "normalized_distance": distance,
        }

    # =========================================================
    # CANDIDATE GENERATION
    # =========================================================

    def generate_candidates(
        self,
        geometry_a: list[GeometryElement],
        geometry_b: list[GeometryElement],
        page_width: float,
        page_height: float,
    ) -> list[dict[str, Any]]:

        started = time.perf_counter()
        pair_budget = len(geometry_a) * len(geometry_b)
        logger.info(
            "GeometryAligner.generate_candidates start a=%s b=%s pair_budget=%s",
            len(geometry_a),
            len(geometry_b),
            pair_budget,
        )

        candidates = []
        compared = 0
        skipped_type = 0
        skipped_distance = 0
        skipped_score = 0

        for a_index, element_a in enumerate(
            geometry_a
        ):

            if a_index > 0 and a_index % 200 == 0:
                logger.info(
                    "GeometryAligner.generate_candidates progress "
                    "a_index=%s/%s compared=%s candidates=%s "
                    "skipped_type=%s skipped_distance=%s "
                    "skipped_score=%s elapsed=%.3fs",
                    a_index,
                    len(geometry_a),
                    compared,
                    len(candidates),
                    skipped_type,
                    skipped_distance,
                    skipped_score,
                    time.perf_counter() - started,
                )

            for b_index, element_b in enumerate(
                geometry_b
            ):

                # Geometry type must match.

                if (
                    element_a.geometry_type
                    !=
                    element_b.geometry_type
                ):
                    skipped_type += 1
                    continue

                # Cheap spatial prune before expensive similarity.
                distance = self.normalized_distance(
                    element_a,
                    element_b,
                    page_width,
                    page_height,
                )
                if distance > self.max_distance:
                    skipped_distance += 1
                    continue

                compared += 1
                similarity = (
                    self.geometry_similarity(
                        element_a,
                        element_b,
                        page_width,
                        page_height,
                    )
                )

                if (
                    similarity[
                        "bbox_similarity"
                    ]
                    <
                    self.min_bbox_similarity
                ):
                    skipped_score += 1
                    continue

                if (
                    similarity[
                        "score"
                    ]
                    <
                    self.min_geometry_similarity
                ):
                    skipped_score += 1
                    continue

                candidates.append(
                    {
                        "a_index": a_index,

                        "b_index": b_index,

                        **similarity,
                    }
                )

        logger.info(
            "GeometryAligner.generate_candidates done candidates=%s "
            "compared=%s skipped_type=%s skipped_distance=%s "
            "skipped_score=%s elapsed=%.3fs",
            len(candidates),
            compared,
            skipped_type,
            skipped_distance,
            skipped_score,
            time.perf_counter() - started,
        )
        return candidates

    # =========================================================
    # ONE-TO-ONE MATCHING
    # =========================================================

    @staticmethod
    def select_matches(
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        candidates = sorted(
            candidates,
            key=lambda candidate: candidate[
                "score"
            ],
            reverse=True,
        )

        matched_a = set()

        matched_b = set()

        selected = []

        for candidate in candidates:

            a_index = candidate[
                "a_index"
            ]

            b_index = candidate[
                "b_index"
            ]

            if a_index in matched_a:

                continue

            if b_index in matched_b:

                continue

            matched_a.add(
                a_index
            )

            matched_b.add(
                b_index
            )

            selected.append(
                candidate
            )

        return selected

    # =========================================================
    # SERIALIZE GEOMETRY ELEMENT
    # =========================================================

    @staticmethod
    def serialize_element(
        element: GeometryElement,
    ) -> dict[str, Any]:

        return {

            "id": element.id,

            "geometry_type": (
                str(
                    element.geometry_type
                )
            ),

            "bbox": {

                "x0": element.bbox.x0,

                "y0": element.bbox.y0,

                "x1": element.bbox.x1,

                "y1": element.bbox.y1,

            },

            "coordinates": (
                element.coordinates
            ),

            "stroke_width": (
                element.stroke_width
            ),

            "stroke_color": (
                element.stroke_color
            ),

            "fill_color": (
                element.fill_color
            ),

            "layer": element.layer,

            "source": (
                str(
                    element.source
                )
            ),

        }

    # =========================================================
    # ALIGN
    # =========================================================

    def align(
        self,
        document_a: DocumentCanonicalRepresentation,
        document_b: DocumentCanonicalRepresentation,
    ) -> dict[str, Any]:

        started = time.perf_counter()
        geometry_a = (
            document_a.geometry_elements
        )

        geometry_b = (
            document_b.geometry_elements
        )

        page_width = (
            document_a.page_width
        )

        page_height = (
            document_a.page_height
        )

        logger.info(
            "GeometryAligner.align start a=%s b=%s pair_budget=%s",
            len(geometry_a),
            len(geometry_b),
            len(geometry_a) * len(geometry_b),
        )

        candidates = (
            self.generate_candidates(
                geometry_a=geometry_a,
                geometry_b=geometry_b,
                page_width=page_width,
                page_height=page_height,
            )
        )

        match_started = time.perf_counter()
        matches = (
            self.select_matches(
                candidates
            )
        )
        logger.info(
            "GeometryAligner.select_matches done matches=%s elapsed=%.3fs",
            len(matches),
            time.perf_counter() - match_started,
        )

        matched_a = {
            match["a_index"]
            for match in matches
        }

        matched_b = {
            match["b_index"]
            for match in matches
        }

        changes = []

        # =====================================================
        # MATCHED GEOMETRY
        # =====================================================

        for match in matches:

            element_a = geometry_a[
                match["a_index"]
            ]

            element_b = geometry_b[
                match["b_index"]
            ]

            if (
                match["score"]
                >=
                0.85
            ):

                change_type = (
                    "unchanged"
                )

            else:

                change_type = (
                    "modified"
                )

            changes.append(
                {

                    "change_type": (
                        change_type
                    ),

                    "confidence": round(
                        match["score"],
                        4,
                    ),

                    "match": {

                        "geometry_similarity": round(
                            match["score"],
                            4,
                        ),

                        "spatial_similarity": round(
                            match[
                                "spatial_similarity"
                            ],
                            4,
                        ),

                        "bbox_similarity": round(
                            match[
                                "bbox_similarity"
                            ],
                            4,
                        ),

                        "coordinate_similarity": round(
                            match[
                                "coordinate_similarity"
                            ],
                            4,
                        ),

                        "normalized_distance": round(
                            match[
                                "normalized_distance"
                            ],
                            6,
                        ),

                    },

                    "revision_a": (
                        self.serialize_element(
                            element_a
                        )
                    ),

                    "revision_b": (
                        self.serialize_element(
                            element_b
                        )
                    ),

                }
            )

        # =====================================================
        # REMOVED GEOMETRY
        # =====================================================

        for index, element_a in enumerate(
            geometry_a
        ):

            if index in matched_a:

                continue

            changes.append(
                {

                    "change_type": (
                        "removed"
                    ),

                    "confidence": 1.0,

                    "match": None,

                    "revision_a": (
                        self.serialize_element(
                            element_a
                        )
                    ),

                    "revision_b": None,

                }
            )

        # =====================================================
        # ADDED GEOMETRY
        # =====================================================

        for index, element_b in enumerate(
            geometry_b
        ):

            if index in matched_b:

                continue

            changes.append(
                {

                    "change_type": (
                        "added"
                    ),

                    "confidence": 1.0,

                    "match": None,

                    "revision_a": None,

                    "revision_b": (
                        self.serialize_element(
                            element_b
                        )
                    ),

                }
            )

        # =====================================================
        # SUMMARY
        # =====================================================

        summary = {

            "total_revision_a": len(
                geometry_a
            ),

            "total_revision_b": len(
                geometry_b
            ),

            "unchanged": sum(

                1

                for change in changes

                if change[
                    "change_type"
                ]
                ==
                "unchanged"

            ),

            "modified": sum(

                1

                for change in changes

                if change[
                    "change_type"
                ]
                ==
                "modified"

            ),

            "added": sum(

                1

                for change in changes

                if change[
                    "change_type"
                ]
                ==
                "added"

            ),

            "removed": sum(

                1

                for change in changes

                if change[
                    "change_type"
                ]
                ==
                "removed"

            ),

        }

        logger.info(
            "GeometryAligner.align complete changes=%s elapsed=%.3fs",
            len(changes),
            time.perf_counter() - started,
        )
        return {

            "comparison_type": (
                "geometry"
            ),

            "revision_a": {

                "document_id": (
                    document_a.document_id
                ),

                "page_width": (
                    document_a.page_width
                ),

                "page_height": (
                    document_a.page_height
                ),

            },

            "revision_b": {

                "document_id": (
                    document_b.document_id
                ),

                "page_width": (
                    document_b.page_width
                ),

                "page_height": (
                    document_b.page_height
                ),

            },

            "summary": summary,

            "changes": changes,

        }