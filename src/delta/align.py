from __future__ import annotations

import logging
import time
from difflib import SequenceMatcher
from math import sqrt
from typing import Any

from src.canonical.model import (
    DocumentCanonicalRepresentation,
    TextElement,
)

logger = logging.getLogger(__name__)


class TextAligner:
    """
    Align text elements between two revisions of the same document.

    Matching strategy:

    1. Exact normalized text + spatial proximity
    2. Fuzzy text similarity + spatial proximity
    3. One-to-one assignment
    4. Unmatched A -> removed
    5. Unmatched B -> added
    """

    def __init__(
        self,
        max_distance: float = 0.025,
        min_text_similarity: float = 0.45,
        exact_match_max_distance: float = 0.05,
    ):

        self.max_distance = max_distance

        self.min_text_similarity = (
            min_text_similarity
        )

        self.exact_match_max_distance = (
            exact_match_max_distance
        )

    # =========================================================
    # TEXT NORMALIZATION
    # =========================================================

    @staticmethod
    def normalize_text(
        text: str,
    ) -> str:
        """
        Normalize text only for comparison.

        The original text is always preserved
        in the comparison JSON.
        """

        if not text:
            return ""

        return " ".join(
            text.strip().split()
        ).casefold()

    # =========================================================
    # BOUNDING BOX
    # =========================================================

    @staticmethod
    def bbox_center(
        element: TextElement,
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
        element_a: TextElement,
        element_b: TextElement,
        page_width: float,
        page_height: float,
    ) -> float:

        ax, ay = TextAligner.bbox_center(
            element_a
        )

        bx, by = TextAligner.bbox_center(
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
    # TEXT SIMILARITY
    # =========================================================

    def text_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:

        normalized_a = self.normalize_text(
            text_a
        )

        normalized_b = self.normalize_text(
            text_b
        )

        if normalized_a == normalized_b:
            return 1.0

        if not normalized_a or not normalized_b:
            return 0.0

        return SequenceMatcher(
            None,
            normalized_a,
            normalized_b,
        ).ratio()

    # =========================================================
    # SPATIAL SIMILARITY
    # =========================================================

    def spatial_similarity(
        self,
        distance: float,
    ) -> float:

        if distance >= self.max_distance:
            return 0.0

        return 1.0 - (
            distance
            /
            self.max_distance
        )

    # =========================================================
    # CANDIDATE GENERATION
    # =========================================================

    def generate_candidates(
        self,
        text_a: list[TextElement],
        text_b: list[TextElement],
        page_width: float,
        page_height: float,
    ) -> list[dict[str, Any]]:

        started = time.perf_counter()
        pair_budget = len(text_a) * len(text_b)
        logger.info(
            "TextAligner.generate_candidates start a=%s b=%s pair_budget=%s",
            len(text_a),
            len(text_b),
            pair_budget,
        )

        candidates = []

        for a_index, element_a in enumerate(
            text_a
        ):

            if a_index > 0 and a_index % 500 == 0:
                logger.info(
                    "TextAligner.generate_candidates progress a_index=%s/%s "
                    "candidates=%s elapsed=%.3fs",
                    a_index,
                    len(text_a),
                    len(candidates),
                    time.perf_counter() - started,
                )

            normalized_a = (
                self.normalize_text(
                    element_a.text
                )
            )

            for b_index, element_b in enumerate(
                text_b
            ):

                normalized_b = (
                    self.normalize_text(
                        element_b.text
                    )
                )

                distance = (
                    self.normalized_distance(
                        element_a,
                        element_b,
                        page_width,
                        page_height,
                    )
                )

                # =============================================
                # EXACT TEXT MATCH
                # =============================================

                if normalized_a == normalized_b:

                    if (
                        distance
                        >
                        self.exact_match_max_distance
                    ):
                        continue

                    spatial_score = (
                        self.spatial_similarity(
                            min(
                                distance,
                                self.max_distance,
                            )
                        )
                    )

                    score = (
                        0.80
                        +
                        0.20
                        *
                        spatial_score
                    )

                    candidates.append(
                        {
                            "a_index": a_index,
                            "b_index": b_index,
                            "text_similarity": 1.0,
                            "spatial_similarity": spatial_score,
                            "distance": distance,
                            "score": score,
                        }
                    )

                    continue

                # =============================================
                # DIFFERENT TEXT
                # =============================================

                if (
                    distance
                    >
                    self.max_distance
                ):
                    continue

                text_score = (
                    self.text_similarity(
                        element_a.text,
                        element_b.text,
                    )
                )

                if (
                    text_score
                    <
                    self.min_text_similarity
                ):
                    continue

                spatial_score = (
                    self.spatial_similarity(
                        distance
                    )
                )

                score = (
                    0.65
                    *
                    text_score
                    +
                    0.35
                    *
                    spatial_score
                )

                candidates.append(
                    {
                        "a_index": a_index,
                        "b_index": b_index,
                        "text_similarity": text_score,
                        "spatial_similarity": spatial_score,
                        "distance": distance,
                        "score": score,
                    }
                )

        logger.info(
            "TextAligner.generate_candidates done candidates=%s elapsed=%.3fs",
            len(candidates),
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
            key=lambda candidate: candidate["score"],
            reverse=True,
        )

        matched_a = set()
        matched_b = set()

        selected = []

        for candidate in candidates:

            a_index = candidate["a_index"]
            b_index = candidate["b_index"]

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
    # SERIALIZE TEXT ELEMENT
    # =========================================================

    @staticmethod
    def serialize_element(
        element: TextElement,
    ) -> dict[str, Any]:

        return {
            "id": element.id,

            "text": element.text,

            "normalized_text": (
                TextAligner.normalize_text(
                    element.text
                )
            ),

            "bbox": {
                "x0": element.bbox.x0,
                "y0": element.bbox.y0,
                "x1": element.bbox.x1,
                "y1": element.bbox.y1,
            },

            "font_name": element.font_name,

            "font_size": element.font_size,

            "font_flags": element.font_flags,

            "confidence": element.confidence,

            "source": str(
                element.source
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
        text_a = document_a.text_elements

        text_b = document_b.text_elements

        page_width = document_a.page_width

        page_height = document_a.page_height

        logger.info(
            "TextAligner.align start a=%s b=%s",
            len(text_a),
            len(text_b),
        )

        candidates = (
            self.generate_candidates(
                text_a=text_a,
                text_b=text_b,
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
            "TextAligner.select_matches done matches=%s elapsed=%.3fs",
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
        # MATCHED ELEMENTS
        # =====================================================

        for match in matches:

            element_a = text_a[
                match["a_index"]
            ]

            element_b = text_b[
                match["b_index"]
            ]

            normalized_a = (
                self.normalize_text(
                    element_a.text
                )
            )

            normalized_b = (
                self.normalize_text(
                    element_b.text
                )
            )

            if (
                normalized_a
                ==
                normalized_b
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
                    "change_type": change_type,

                    "confidence": round(
                        match["score"],
                        4,
                    ),

                    "match": {
                        "text_similarity": round(
                            match[
                                "text_similarity"
                            ],
                            4,
                        ),

                        "spatial_similarity": round(
                            match[
                                "spatial_similarity"
                            ],
                            4,
                        ),

                        "normalized_distance": round(
                            match[
                                "distance"
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
        # REMOVED ELEMENTS
        # =====================================================

        for index, element_a in enumerate(
            text_a
        ):

            if index in matched_a:
                continue

            changes.append(
                {
                    "change_type": "removed",

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
        # ADDED ELEMENTS
        # =====================================================

        for index, element_b in enumerate(
            text_b
        ):

            if index in matched_b:
                continue

            changes.append(
                {
                    "change_type": "added",

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
                text_a
            ),

            "total_revision_b": len(
                text_b
            ),

            "unchanged": sum(
                1
                for change in changes
                if change["change_type"]
                ==
                "unchanged"
            ),

            "modified": sum(
                1
                for change in changes
                if change["change_type"]
                ==
                "modified"
            ),

            "added": sum(
                1
                for change in changes
                if change["change_type"]
                ==
                "added"
            ),

            "removed": sum(
                1
                for change in changes
                if change["change_type"]
                ==
                "removed"
            ),
        }
        logger.info(
            "TextAligner.align complete changes=%s elapsed=%.3fs",
            len(changes),
            time.perf_counter() - started,
        )
        return {
            "comparison_type": "text",

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
