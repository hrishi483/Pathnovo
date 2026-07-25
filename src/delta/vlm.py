from __future__ import annotations

import json
import os
from pathlib import Path
from dotenv import load_dotenv
import cv2
import numpy as np

from google import genai
from google.genai import types

load_dotenv()


class VLMDeltaAnalyzer:

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
    ):
        self.client = genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.model = model

    def compare(
        self,
        baseline_image_path: str,
        revision_image_path: str,
        output_dir: str = "visual_diff",
        threshold: int = 30,
        min_contour_area: int = 10,
        padding: int = 10,
    ) -> dict:

        """
        Compare two geometry/residual images.

        Pipeline:

            baseline image
                    +
            revision image
                    │
                    ▼
              pixel difference
                    │
                    ▼
                threshold
                    │
                    ▼
                contours
                    │
                    ▼
             filter small changes
                    │
                    ▼
             annotate revision image
                    │
                    ▼
                  Gemini

        Returns:
            {
                "visual_diff_image": "...",
                "change_regions": [...],
                "vlm_analysis": {...}
            }
        """

        output_dir = Path(
            output_dir
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------
        # 1. Load images
        # --------------------------------------------------

        baseline = cv2.imread(
            baseline_image_path
        )

        revision = cv2.imread(
            revision_image_path
        )

        if baseline is None:

            raise ValueError(
                f"Could not read baseline image: "
                f"{baseline_image_path}"
            )

        if revision is None:

            raise ValueError(
                f"Could not read revision image: "
                f"{revision_image_path}"
            )

        # --------------------------------------------------
        # 2. Validate dimensions
        # --------------------------------------------------

        if baseline.shape != revision.shape:

            raise ValueError(
                "Baseline and revision images must "
                "have identical dimensions."
            )

        # --------------------------------------------------
        # 3. Convert to grayscale
        # --------------------------------------------------

        gray_baseline = cv2.cvtColor(
            baseline,
            cv2.COLOR_BGR2GRAY
        )

        gray_revision = cv2.cvtColor(
            revision,
            cv2.COLOR_BGR2GRAY
        )

        # --------------------------------------------------
        # 4. Absolute pixel difference
        # --------------------------------------------------

        diff = cv2.absdiff(
            gray_baseline,
            gray_revision
        )

        # --------------------------------------------------
        # 5. Threshold difference
        # --------------------------------------------------

        _, thresholded = cv2.threshold(
            diff,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        # --------------------------------------------------
        # 6. Find contours
        # --------------------------------------------------

        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        print(
            f"Detected {len(contours)} raw contours."
        )

        # --------------------------------------------------
        # 7. Create annotated revision image
        # --------------------------------------------------

        annotated_revision = revision.copy()

        change_regions = []

        change_index = 0

        for contour in contours:

            contour_area = cv2.contourArea(
                contour
            )

            # Ignore tiny changes
            if contour_area < min_contour_area:

                continue

            x, y, w, h = cv2.boundingRect(
                contour
            )

            # Add context around the detected region
            x0 = max(
                0,
                x - padding
            )

            y0 = max(
                0,
                y - padding
            )

            x1 = min(
                revision.shape[1],
                x + w + padding
            )

            y1 = min(
                revision.shape[0],
                y + h + padding
            )

            change_id = (
                f"change_{change_index:04d}"
            )

            # Draw bounding box
            cv2.rectangle(
                annotated_revision,
                (x0, y0),
                (x1, y1),
                (0, 0, 255),
                2
            )

            # Add change label
            cv2.putText(
                annotated_revision,
                change_id,
                (x0, max(20, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

            change_regions.append(
                {
                    "id": change_id,
                    "bbox": {
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                    },
                    "width": x1 - x0,
                    "height": y1 - y0,
                    "contour_area": float(
                        contour_area
                    ),
                }
            )

            change_index += 1

        # --------------------------------------------------
        # 8. Save intermediate artifacts
        # --------------------------------------------------

        diff_path = (
            output_dir
            / "pixel_difference.png"
        )

        threshold_path = (
            output_dir
            / "threshold_difference.png"
        )

        annotated_path = (
            output_dir
            / "revision_with_changes.png"
        )

        cv2.imwrite(
            str(diff_path),
            diff
        )

        cv2.imwrite(
            str(threshold_path),
            thresholded
        )

        cv2.imwrite(
            str(annotated_path),
            annotated_revision
        )

        print(
            f"Meaningful change regions: "
            f"{len(change_regions)}"
        )

        print(
            f"Saved annotated image to: "
            f"{annotated_path}"
        )

        # --------------------------------------------------
        # 9. Send ONE annotated image to Gemini
        # --------------------------------------------------

        vlm_analysis = self._analyze_with_gemini(
            annotated_image_path=str(
                annotated_path
            ),
            number_of_regions=len(
                change_regions
            ),
        )

        return {

            "visual_diff_image": str(
                annotated_path
            ),

            "pixel_difference_image": str(
                diff_path
            ),

            "threshold_difference_image": str(
                threshold_path
            ),

            "change_regions": change_regions,

            "vlm_analysis": vlm_analysis,

        }
    def _analyze_with_gemini(
            self, annotated_image_path: str, number_of_regions: int) -> dict:

            prompt = f"""
    You are analyzing visual changes in an engineering drawing.

    The supplied image is the REVISION drawing.

    Red bounding boxes indicate regions where the revision differs
    from the baseline drawing according to a deterministic pixel-level
    comparison.

    There are approximately
    {number_of_regions}
    detected change regions.

    Your task is to interpret the visual changes inside the red boxes.

    Focus on meaningful engineering changes such as:

    - added equipment
    - removed equipment
    - moved equipment
    - modified equipment
    - changed process symbols
    - changed vessels
    - changed compressors
    - changed pumps
    - changed valves
    - changed pipe routing
    - changed connections
    - changed instrumentation
    - other meaningful drawing changes

    Ignore:

    - anti-aliasing
    - tiny pixel shifts
    - rendering artifacts
    - insignificant line thickness differences
    - minor pixel-level noise

    For each meaningful change region, provide:

    - the region ID shown in the red label
    - change type
    - description
    - confidence

    Return ONLY valid JSON in this format:

    {{
        "summary": {{
            "has_meaningful_changes": true,
            "number_of_meaningful_changes": 0,
            "overall_summary": "..."
        }},
        "changes": [
            {{
                "region_id": "change_0000",
                "change_type": "added",
                "description": "...",
                "confidence": 0.0
            }}
        ]
    }}

    Valid change types are:

    - added
    - removed
    - modified
    - moved
    - rerouted
    - unknown

    Do not describe text changes.
    Do not describe changes that are only minor rendering differences.
    """

            with open(
                annotated_image_path,
                "rb"
            ) as image_file:

                image_bytes = image_file.read()

            response = self.client.models.generate_content(

                model=self.model,

                contents=[

                    types.Part.from_text(
                        text=prompt
                    ),

                    types.Part.from_bytes(

                        data=image_bytes,

                        mime_type="image/png"

                    ),

                ],

                config=types.GenerateContentConfig(

                    temperature=0.0,

                    response_mime_type=(
                        "application/json"
                    ),

                ),

            )

            return json.loads(
                response.text
            )
    
    def compare_with_gemini(self,baseline_image_path: str, revision_image_path: str) -> dict:

        prompt = """
        You are comparing two versions of the same engineering drawing.

        The first image is the BASELINE revision.

        The second image is the REVISED revision.

        These images contain the geometry of the engineering drawing after
        document-level elements such as extracted text and some document
        layout elements have been removed.

        Compare the two images carefully.

        Your task is to identify meaningful visual and geometric changes.

        Focus on:

        - added equipment
        - removed equipment
        - moved equipment
        - modified equipment
        - changed equipment shapes
        - changed process symbols
        - changed vessels
        - changed compressors
        - changed pumps
        - changed valves
        - changed pipe routing
        - changed connections
        - changed instrumentation symbols
        - other meaningful engineering drawing changes

        Ignore:

        - tiny pixel shifts
        - anti-aliasing differences
        - insignificant line thickness differences
        - rendering artifacts
        - compression artifacts
        - minor differences caused by image rasterization

        For every meaningful change, provide:

        1. The type of change
        2. A concise description
        3. The approximate location of the change
        4. A normalized bounding box from 0 to 1
        5. A confidence score

        The first image is BASELINE.
        The second image is REVISION.

        Return ONLY valid JSON using exactly this structure:

        {
        "summary": {
            "has_meaningful_changes": true,
            "number_of_changes": 0,
            "overall_summary": "..."
        },
        "changes": [
            {
            "change_id": "visual_change_0000",
            "change_type": "added",
            "description": "...",
            "location_description": "...",
            "confidence": 0.0,
            }
        ]
        }

        Valid change_type values are:

        - added
        - removed
        - modified
        - moved
        - rerouted
        - unknown

        Important:

        - Do not report text changes.
        - Do not report tiny visual rendering differences.
        - Do not report differences caused only by slight positional shifts of the entire drawing.
        - Group multiple small geometric differences together if they represent one logical engineering change.
        - If the same equipment or structure has moved, classify it as "moved" rather than "removed" and "added".
        - If there are no meaningful visual changes, return an empty changes array.
        - Be conservative. Do not invent changes that are not clearly visible.
        """

        with open(
            baseline_image_path,
            "rb",
        ) as baseline_file:

            baseline_bytes = baseline_file.read()

        with open(
            revision_image_path,
            "rb",
        ) as revision_file:

            revision_bytes = revision_file.read()

        response = self.client.models.generate_content(

            model=self.model,

            contents=[

                types.Part.from_text(
                    text=prompt
                ),

                types.Part.from_bytes(
                    data=baseline_bytes,
                    mime_type="image/png",
                ),

                types.Part.from_bytes(
                    data=revision_bytes,
                    mime_type="image/png",
                ),

            ],

            config=types.GenerateContentConfig(

                temperature=0.0,

                response_mime_type=(
                    "application/json"
                ),
            ),
        )
        try:
            return json.loads(
                response.text
            )
        except json.JSONDecodeError as exc:
            raise ValueError(
                "Gemini returned invalid JSON "
                "for visual comparison"
            ) from exc