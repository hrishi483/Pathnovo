"""Document comparison orchestration for canonical representations."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.canonical.model import DocumentCanonicalRepresentation
from src.delta.align import TextAligner
from src.delta.engine import DeltaEngine
from src.delta.vlm import VLMDeltaAnalyzer
from src.delta.geometryalign import GeometryAligner

logger = logging.getLogger(__name__)


class DocumentComparer(DeltaEngine):
    """Compare baseline and revision canonical documents.

    Text alignment is implemented today. Geometry and residual-mask comparison
    can be added alongside ``text`` in ``compare`` when those aligners exist.
    """

    def __init__(
        self,
        text_aligner: TextAligner | None = None,
        geometry_aligner: GeometryAligner | None = None,
        vlm_aligner: VLMDeltaAnalyzer | None = None,
    ) -> None:
        self._text_aligner = text_aligner or TextAligner()
        self._geometry_aligner = geometry_aligner or GeometryAligner()
        self._vlm_aligner = vlm_aligner or VLMDeltaAnalyzer()
    def compare(
        self,
        baseline: DocumentCanonicalRepresentation,
        revision: DocumentCanonicalRepresentation,
        output_dir: Path,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        logger.info(
            "DocumentComparer.compare start baseline=%s revision=%s",
            baseline.document_id,
            revision.document_id,
        )
        # Text comparison
        text_started = time.perf_counter()
        text_comparison = self._text_aligner.align(
            document_a=baseline,
            document_b=revision,
        )
        logger.info(
            "text align done elapsed=%.3fs summary=%s",
            time.perf_counter() - text_started,
            text_comparison.get("summary"),
        )

        results_dir = output_dir / f"results-{baseline.document_id}-{revision.document_id}"
        results_dir.mkdir(parents=True, exist_ok=True)
        text_comparison_path = (
            results_dir / f"{baseline.document_id}-{revision.document_id}_text.json"
        )
        write_started = time.perf_counter()
        text_comparison_path.write_text(
            json.dumps(text_comparison, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "text json written path=%s elapsed=%.3fs",
            text_comparison_path,
            time.perf_counter() - write_started,
        )

        # Geometry comparison
        geometry_started = time.perf_counter()
        logger.info(
            "geometry align start count_a=%s count_b=%s pair_budget=%s",
            len(baseline.geometry_elements),
            len(revision.geometry_elements),
            len(baseline.geometry_elements) * len(revision.geometry_elements),
        )
        geometry_comparison = self._geometry_aligner.align(
            document_a=baseline,
            document_b=revision,
        )
        logger.info(
            "geometry align done elapsed=%.3fs summary=%s",
            time.perf_counter() - geometry_started,
            geometry_comparison.get("summary"),
        )

        geometry_comparison_path = (
            results_dir
            / f"{baseline.document_id}-{revision.document_id}_geometry.json"
        )
        write_started = time.perf_counter()
        geometry_comparison_path.write_text(
            json.dumps(geometry_comparison, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "geometry json written path=%s elapsed=%.3fs",
            geometry_comparison_path,
            time.perf_counter() - write_started,
        )


        # VLM comparison with difference analysis
        vlm_started = time.perf_counter()
        vlm_comparison_with_difference_analysis = self._vlm_aligner.compare(
            baseline_image_path=baseline.metadata["artifacts"]["geometry_mask"],
            revision_image_path=revision.metadata["artifacts"]["geometry_mask"],
            output_dir=results_dir
        )
        logger.info(
            "vlm align done elapsed=%.3fs summary=%s",
            time.perf_counter() - vlm_started,
            vlm_comparison_with_difference_analysis.get("summary"),
        )

        vlm_comparison_path = (
            results_dir
            / f"{baseline.document_id}-{revision.document_id}_vlm_with_difference_analysis.json"
        )
        write_started = time.perf_counter()
        vlm_comparison_path.write_text(
            json.dumps(vlm_comparison_with_difference_analysis, indent=2),
            encoding="utf-8",
        )

        logger.info(
            "DocumentComparer.compare complete total_elapsed=%.3fs",
            time.perf_counter() - started,
        )

        # VLM comparison with Gemini
        vlm_comparison = self._vlm_aligner.compare_with_gemini(
            baseline_image_path=baseline.metadata["artifacts"]["geometry_mask"],
            revision_image_path=revision.metadata["artifacts"]["geometry_mask"],
        )
        vlm_comparison_path = (
            results_dir
            / f"{baseline.document_id}-{revision.document_id}_vlm_comparison.json"
        )
        write_started = time.perf_counter()
        vlm_comparison_path.write_text(
            json.dumps(vlm_comparison, indent=2),
            encoding="utf-8",
        )
        return {
            "baseline_document_id": baseline.document_id,
            "revision_document_id": revision.document_id,
            "statistics": {
                "text": text_comparison["summary"],
                "geometry": geometry_comparison["summary"],
                "vlm_with_difference_analysis": vlm_comparison_with_difference_analysis,
                "vlm": vlm_comparison,
            },
        }
