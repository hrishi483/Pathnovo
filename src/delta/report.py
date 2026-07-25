"""Report-rendering contract for a future structured delta."""

from abc import ABC, abstractmethod


class DeltaReportRenderer(ABC):
    @abstractmethod
    def render(self, delta: dict) -> dict:
        """Convert an engine result into a serializable report."""
