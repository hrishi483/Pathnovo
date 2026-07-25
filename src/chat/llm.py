"""Provider-neutral visual interpretation boundary."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.canonical.model import BoundingBox


class VisualInterpreter(ABC):
    @abstractmethod
    def interpret(self, image_path: Path, bbox: BoundingBox) -> dict[str, Any]:
        """Interpret one residual region; never called during ingestion."""
